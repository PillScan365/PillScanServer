import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from aiolimiter import AsyncLimiter
from fastapi import UploadFile

from pillscan_server.config import Settings
from pillscan_server.errors import RateLimitExceeded
from pillscan_server.imaging import prepare_upload
from pillscan_server.models import (
    IDENTIFICATION_DISCLAIMER,
    DrugResolution,
    PillAnalysisResponse,
    PipelineTimings,
    ResolutionSource,
    ResolutionStatus,
)
from pillscan_server.protocols import DrugCatalogResolver, PillVisionAnalyzer


class PillAnalysisService:
    def __init__(
        self,
        analyzer: PillVisionAnalyzer,
        settings: Settings,
        limiter: AsyncLimiter,
        concurrency: asyncio.Semaphore,
        catalog: DrugCatalogResolver | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._settings = settings
        self._catalog = catalog
        self._gate = AnalysisGate(
            limiter=limiter,
            concurrency=concurrency,
            wait_seconds=settings.rate_limit_wait_seconds,
        )

    async def analyze(
        self,
        upload: UploadFile,
        *,
        market: str,
        context: str | None,
        request_id: str,
    ) -> PillAnalysisResponse:
        pipeline_started_at = perf_counter()
        prepared = await prepare_upload(upload, self._settings)
        async with self._gate.acquire() as gate_wait:
            vision_started_at = perf_counter()
            vision_result = await self._analyzer.analyze(
                prepared.image,
                market=market,
                context=context,
            )
            vision_analysis_ms = _elapsed_ms(vision_started_at)

        analysis = vision_result.analysis

        catalog_started_at = perf_counter()
        resolution = (
            await self._catalog.resolve(analysis, market=market)
            if self._catalog is not None
            else DrugResolution(
                status=_pre_catalog_status(analysis.state),
                source=ResolutionSource.NOT_QUERIED,
                product=None,
                candidates=[],
                catalog_version=None,
            )
        )
        catalog_resolution_ms = _elapsed_ms(catalog_started_at)
        timings = PipelineTimings(
            upload_read_ms=prepared.upload_read_ms,
            image_normalization_ms=prepared.image_normalization_ms,
            rate_limit_wait_ms=gate_wait.rate_limit_wait_ms,
            concurrency_wait_ms=gate_wait.concurrency_wait_ms,
            vision_analysis_ms=vision_analysis_ms,
            catalog_resolution_ms=catalog_resolution_ms,
            pipeline_total_ms=_elapsed_ms(pipeline_started_at),
        )
        return PillAnalysisResponse(
            schema_version="1.2",
            analysis_id=uuid4(),
            request_id=request_id,
            provider=self._analyzer.provider_name,
            model=self._analyzer.model_name,
            timings=timings,
            usage=vision_result.usage,
            analysis=analysis,
            resolution=resolution,
            disclaimer=IDENTIFICATION_DISCLAIMER,
        )


def _pre_catalog_status(
    analysis_state: str,
) -> ResolutionStatus:
    if analysis_state == "needs_better_image":
        return ResolutionStatus.NEEDS_BETTER_IMAGE
    if analysis_state == "no_visual_match":
        return ResolutionStatus.NOT_MEDICATION_IMAGE
    return ResolutionStatus.EVIDENCE_EXTRACTED


class AnalysisGate:
    """Bound request rate and concurrent expensive VLM calls per process."""

    def __init__(
        self,
        *,
        limiter: AsyncLimiter,
        concurrency: asyncio.Semaphore,
        wait_seconds: float,
    ) -> None:
        self._limiter = limiter
        self._concurrency = concurrency
        self._wait_seconds = wait_seconds

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator["GateWaitTimings"]:
        acquired_concurrency = False
        try:
            async with asyncio.timeout(self._wait_seconds):
                rate_limit_started_at = perf_counter()
                await self._limiter.acquire()
                rate_limit_wait_ms = _elapsed_ms(rate_limit_started_at)

                concurrency_started_at = perf_counter()
                await self._concurrency.acquire()
                concurrency_wait_ms = _elapsed_ms(concurrency_started_at)
                acquired_concurrency = True
        except TimeoutError:
            raise RateLimitExceeded from None

        try:
            yield GateWaitTimings(
                rate_limit_wait_ms=rate_limit_wait_ms,
                concurrency_wait_ms=concurrency_wait_ms,
            )
        finally:
            if acquired_concurrency:
                self._concurrency.release()


@dataclass(frozen=True, slots=True)
class GateWaitTimings:
    rate_limit_wait_ms: float
    concurrency_wait_ms: float


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)
