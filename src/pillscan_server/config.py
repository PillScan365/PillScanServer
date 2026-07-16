from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated process configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        env_prefix="PILLSCAN_",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_name: str = "PillScan Server"
    environment: Literal["development", "test", "production"] = "development"
    host: str = "0.0.0.0"  # noqa: S104 - configurable network service bind address
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["critical", "error", "warning", "info", "debug"] = "info"
    log_format: Literal["console", "json"] = "console"

    openai_api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")
    openai_model: str = "gpt-5.6"
    openai_timeout_seconds: float = Field(default=90.0, gt=0, le=300)
    openai_max_retries: int = Field(default=2, ge=0, le=5)
    openai_image_detail: Literal["auto", "high"] = "high"

    api_token: SecretStr | None = None
    trusted_hosts: list[str] = Field(default_factory=lambda: ["*"])
    cors_origins: list[str] = Field(default_factory=list)

    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(default=40_000_000, ge=1_000_000)
    max_image_dimension: int = Field(default=4096, ge=512, le=8192)
    analyses_per_minute: int = Field(default=20, ge=1, le=10_000)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=100)
    rate_limit_wait_seconds: float = Field(default=5.0, gt=0, le=60)

    tfda_catalog_path: Path = Path(".data/tfda/catalog.sqlite3")
    tfda_raw_dir: Path = Path(".data/tfda/raw")
    nhia_drug_csv_path: Path = Path(".data/nhia/drugs.csv")
    tfda_catalog_required: bool = True

    @property
    def docs_enabled(self) -> bool:
        return self.environment != "production"

    @model_validator(mode="after")
    def validate_production_security(self) -> Self:
        if self.environment != "production":
            return self
        if self.api_token is None:
            raise ValueError("PILLSCAN_API_TOKEN is required in production")
        if "*" in self.trusted_hosts:
            raise ValueError("PILLSCAN_TRUSTED_HOSTS must be explicit in production")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
