import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast

from aiolimiter import AsyncLimiter
from fastapi import APIRouter, Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from openai import AsyncOpenAI
from starlette.middleware.trustedhost import TrustedHostMiddleware

from pillscan_server import __version__
from pillscan_server.auth import require_api_token
from pillscan_server.catalog import TfdaCatalog
from pillscan_server.config import Settings, get_settings
from pillscan_server.errors import PillScanError
from pillscan_server.logging_config import configure_logging
from pillscan_server.middleware import RequestIDMiddleware
from pillscan_server.models import (
    ErrorResponse,
    HealthResponse,
    MedicationAnalysisResponse,
    PillAnalysisResponse,
)
from pillscan_server.openai_provider import OpenAIPillVisionAnalyzer
from pillscan_server.protocols import DrugCatalogResolver, PillVisionAnalyzer
from pillscan_server.service import PillAnalysisService


def create_app(
    settings: Settings | None = None,
    analyzer_override: PillVisionAnalyzer | None = None,
    catalog_override: DrugCatalogResolver | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[dict[str, object]]:
        limiter = AsyncLimiter(resolved_settings.analyses_per_minute, time_period=60)
        concurrency = asyncio.Semaphore(resolved_settings.max_concurrent_analyses)
        logger.bind(
            environment=resolved_settings.environment,
            model=resolved_settings.openai_model,
        ).info("service_starting")
        catalog: DrugCatalogResolver | None = catalog_override
        owned_catalog: TfdaCatalog | None = None
        if catalog is None and resolved_settings.tfda_catalog_path.is_file():
            owned_catalog = await TfdaCatalog.open(resolved_settings.tfda_catalog_path)
            catalog = owned_catalog
            logger.bind(
                catalog_version=owned_catalog.catalog_version,
                product_count=owned_catalog.record_count,
            ).info("catalog_loaded")
        elif catalog is None and resolved_settings.tfda_catalog_required:
            raise RuntimeError(
                "TFDA catalog is required but missing. Run pillscan-catalog-sync first: "
                f"{resolved_settings.tfda_catalog_path}"
            )
        if analyzer_override is not None:
            try:
                yield {
                    "analysis_service": PillAnalysisService(
                        analyzer_override,
                        resolved_settings,
                        limiter,
                        concurrency,
                        catalog,
                    )
                }
            finally:
                if owned_catalog is not None:
                    await owned_catalog.close()
                logger.info("service_stopped")
            return

        client = AsyncOpenAI(
            api_key=resolved_settings.openai_api_key.get_secret_value(),
            timeout=resolved_settings.openai_timeout_seconds,
            max_retries=resolved_settings.openai_max_retries,
        )
        analyzer = OpenAIPillVisionAnalyzer(
            client,
            model=resolved_settings.openai_model,
            image_detail=resolved_settings.openai_image_detail,
        )
        try:
            yield {
                "analysis_service": PillAnalysisService(
                    analyzer,
                    resolved_settings,
                    limiter,
                    concurrency,
                    catalog,
                )
            }
        finally:
            await client.close()
            if owned_catalog is not None:
                await owned_catalog.close()
            logger.info("service_stopped")

    app = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        description=(
            "Extracts structured visual evidence from medication images and resolves it against "
            "a local authoritative TFDA/NHIA catalog."
        ),
        lifespan=lifespan,
        docs_url="/docs" if resolved_settings.docs_enabled else None,
        redoc_url=None,
    )
    app.state.settings = resolved_settings
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=resolved_settings.trusted_hosts)
    if resolved_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )

    app.add_exception_handler(PillScanError, _pillscan_error_handler)
    app.include_router(_router())
    return app


async def _pillscan_error_handler(request: Request, exc: Exception) -> JSONResponse:
    error = cast(PillScanError, exc)
    request_id = getattr(request.state, "request_id", "unknown")
    payload = ErrorResponse(
        code=error.code,
        message=error.public_message,
        request_id=request_id,
    )
    return JSONResponse(status_code=error.status_code, content=payload.model_dump(mode="json"))


def _router() -> APIRouter:
    router = APIRouter()

    @router.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def live(request: Request) -> HealthResponse:
        settings = cast(Settings, request.app.state.settings)
        return HealthResponse(status="ok", service=settings.app_name, version=__version__)

    @router.get("/health/ready", response_model=HealthResponse, tags=["health"])
    async def ready(request: Request) -> HealthResponse:
        settings = cast(Settings, request.app.state.settings)
        if not hasattr(request.state, "analysis_service"):
            raise PillScanError
        return HealthResponse(status="ready", service=settings.app_name, version=__version__)

    @router.post(
        "/v1/pills/analyze",
        response_model=PillAnalysisResponse,
        responses={
            400: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
        },
        tags=["pills"],
    )
    async def analyze_pill(
        request: Request,
        image: Annotated[
            UploadFile,
            File(description="Single photo of a loose pill or medication package"),
        ],
        _auth: Annotated[None, Depends(require_api_token)],
        market: Annotated[str, Form(min_length=2, max_length=32)] = "TW",
        context: Annotated[str | None, Form(max_length=500)] = None,
    ) -> PillAnalysisResponse:
        service = cast(PillAnalysisService, request.state.analysis_service)
        response = await service.analyze(
            image,
            market=market,
            context=context,
            request_id=request.state.request_id,
        )
        request.state.pipeline_timings = response.timings
        return response

    @router.post(
        "/v2/medications/analyze",
        response_model=MedicationAnalysisResponse,
        responses={
            400: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
        },
        tags=["medications"],
    )
    async def analyze_medications(
        request: Request,
        image: Annotated[
            UploadFile,
            File(
                description=(
                    "Single photo of a pill, medication package, prescription, medication list, "
                    "medication bag, or dispensing label"
                )
            ),
        ],
        _auth: Annotated[None, Depends(require_api_token)],
        market: Annotated[str, Form(min_length=2, max_length=32)] = "TW",
        context: Annotated[str | None, Form(max_length=500)] = None,
    ) -> MedicationAnalysisResponse:
        service = cast(PillAnalysisService, request.state.analysis_service)
        response = await service.analyze_medications(
            image,
            market=market,
            context=context,
            request_id=request.state.request_id,
        )
        request.state.pipeline_timings = response.timings
        return response

    return router
