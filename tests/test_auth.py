import httpx
import pytest
from asgi_lifespan import LifespanManager
from pydantic import SecretStr

from pillscan_server.app import create_app
from pillscan_server.config import Settings
from tests.conftest import FakeAnalyzer


@pytest.mark.asyncio
async def test_configured_token_protects_analysis_endpoint(
    settings: Settings,
    fake_analyzer: FakeAnalyzer,
) -> None:
    protected = settings.model_copy(update={"api_token": SecretStr("server-token")})
    app = create_app(protected, fake_analyzer)

    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/pills/analyze")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
