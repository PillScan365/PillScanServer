import pytest
from pydantic import SecretStr, ValidationError

from pillscan_server.config import Settings


def test_production_requires_api_token_and_explicit_hosts() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            environment="production",
            openai_api_key=SecretStr("test-openai-key"),
        )


def test_production_accepts_hardened_configuration() -> None:
    settings = Settings(
        _env_file=None,
        environment="production",
        openai_api_key=SecretStr("test-openai-key"),
        api_token=SecretStr("server-token"),
        trusted_hosts=["rpi.local"],
    )

    assert settings.docs_enabled is False
