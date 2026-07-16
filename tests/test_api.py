from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pillscan_server import app as app_module
from pillscan_server.app import create_app
from pillscan_server.config import Settings
from pillscan_server.models import ModelUsage, SubjectType
from pillscan_server.protocols import PreparedImage, VisionAnalysisResult
from tests.conftest import FakeAnalyzer
from tests.test_catalog import analysis, make_catalog


class PackageAnalyzer(FakeAnalyzer):
    async def analyze(
        self,
        image: PreparedImage,
        *,
        market: str,
        context: str | None,
    ) -> VisionAnalysisResult:
        self.received_image = image
        return VisionAnalysisResult(
            analysis=analysis(
                subject_type=SubjectType.PACKAGE,
                product_name="百樂行膜衣錠20毫克",
                strength="20mg",
                manufacturer="瑞士藥廠股份有限公司新市廠",
            ),
            usage=ModelUsage(
                input_tokens=1200,
                cached_input_tokens=100,
                output_tokens=300,
                reasoning_tokens=0,
                total_tokens=1500,
            ),
        )


@asynccontextmanager
async def client_for(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.mark.asyncio
async def test_health_endpoints(settings: Settings, fake_analyzer: FakeAnalyzer) -> None:
    app = create_app(settings, fake_analyzer)
    async with client_for(app) as client:
        live = await client.get("/health/live")
        ready = await client.get("/health/ready")

    assert live.status_code == 200
    assert live.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert live.headers["x-request-id"]


@pytest.mark.asyncio
async def test_analyze_normalizes_single_image(
    settings: Settings,
    fake_analyzer: FakeAnalyzer,
    jpeg_bytes: bytes,
) -> None:
    app = create_app(settings, fake_analyzer)
    async with client_for(app) as client:
        response = await client.post(
            "/v1/pills/analyze",
            files={"image": ("capture.jpg", jpeg_bytes, "image/jpeg")},
            data={"market": "TW"},
            headers={"X-Request-ID": "test-request-1"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "test-request-1"
    assert payload["provider"] == "fake"
    assert payload["schema_version"] == "1.2"
    assert set(payload["timings"]) == {
        "upload_read_ms",
        "image_normalization_ms",
        "rate_limit_wait_ms",
        "concurrency_wait_ms",
        "vision_analysis_ms",
        "catalog_resolution_ms",
        "pipeline_total_ms",
    }
    assert all(value >= 0 for value in payload["timings"].values())
    assert payload["usage"]["total_tokens"] == 0
    assert "vision;dur=" in response.headers["server-timing"]
    assert "catalog;dur=" in response.headers["server-timing"]
    assert payload["analysis"]["subject_type"] == "pill"
    assert payload["analysis"]["state"] == "visual_evidence_only"
    assert payload["resolution"] == {
        "status": "evidence_extracted",
        "source": "not_queried",
        "product": None,
        "candidates": [],
        "catalog_version": None,
    }
    assert payload["disclaimer"]
    assert fake_analyzer.received_image is not None
    assert fake_analyzer.received_image.media_type == "image/jpeg"


@pytest.mark.asyncio
async def test_rejects_unsupported_upload(
    settings: Settings,
    fake_analyzer: FakeAnalyzer,
) -> None:
    app = create_app(settings, fake_analyzer)
    async with client_for(app) as client:
        response = await client.post(
            "/v1/pills/analyze",
            files={"image": ("capture.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_image"


def test_openapi_marks_the_response_contract_as_fixed(
    settings: Settings,
    fake_analyzer: FakeAnalyzer,
) -> None:
    schemas = create_app(settings, fake_analyzer).openapi()["components"]["schemas"]

    for schema_name in (
        "PillAnalysisResponse",
        "DrugResolution",
        "DrugProduct",
        "DrugIngredient",
        "ProductIdentifiers",
        "CatalogCandidate",
        "PipelineTimings",
        "ModelUsage",
    ):
        schema = schemas[schema_name]
        assert set(schema["required"]) == set(schema["properties"])


@pytest.mark.asyncio
async def test_api_loads_real_catalog_and_returns_official_id(
    settings: Settings,
    jpeg_bytes: bytes,
    tmp_path: Path,
) -> None:
    configured = settings.model_copy(
        update={
            "tfda_catalog_path": make_catalog(tmp_path),
            "tfda_catalog_required": True,
        }
    )
    app = create_app(configured, PackageAnalyzer())
    async with client_for(app) as client:
        response = await client.post(
            "/v1/pills/analyze",
            files={"image": ("package.jpg", jpeg_bytes, "image/jpeg")},
            data={"market": "TW"},
        )

    assert response.status_code == 200
    resolution = response.json()["resolution"]
    assert resolution["status"] == "catalog_exact"
    assert resolution["source"] == "tfda_nhi"
    assert resolution["product"]["identifiers"]["tfda_permit_number"] == ("衛部藥製字第058256號")
    assert resolution["product"]["identifiers"]["nhi_code"] == "AC58256100"


@pytest.mark.asyncio
async def test_app_refuses_to_start_when_required_catalog_is_missing(
    settings: Settings,
    fake_analyzer: FakeAnalyzer,
    tmp_path: Path,
) -> None:
    configured = settings.model_copy(
        update={
            "tfda_catalog_path": tmp_path / "missing.sqlite3",
            "tfda_catalog_required": True,
        }
    )
    app = create_app(configured, fake_analyzer)

    with pytest.raises(RuntimeError, match="TFDA catalog is required but missing"):
        async with LifespanManager(app):
            pass


@pytest.mark.asyncio
async def test_lifespan_owns_and_closes_default_openai_client(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenAIClient:
        def __init__(self, **_: object) -> None:
            self.closed = False
            clients.append(self)

        async def close(self) -> None:
            self.closed = True

    clients: list[FakeOpenAIClient] = []
    analyzer = PackageAnalyzer()

    monkeypatch.setattr(app_module, "AsyncOpenAI", FakeOpenAIClient)
    monkeypatch.setattr(
        app_module,
        "OpenAIPillVisionAnalyzer",
        lambda *_args, **_kwargs: analyzer,
    )
    app = create_app(settings)

    async with client_for(app) as client:
        response = await client.get("/health/ready")

    assert response.status_code == 200
    assert len(clients) == 1
    assert clients[0].closed is True


def test_configured_cors_origin_adds_cors_middleware(
    settings: Settings,
    fake_analyzer: FakeAnalyzer,
) -> None:
    configured = settings.model_copy(update={"cors_origins": ["https://app.example.test"]})
    app = create_app(configured, fake_analyzer)

    assert any(
        getattr(middleware.cls, "__name__", None) == CORSMiddleware.__name__
        for middleware in app.user_middleware
    )
