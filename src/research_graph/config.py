"""Project configuration loaded from environment variables and .env files."""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


class Settings(BaseSettings):
    """Runtime settings for CLI commands and future data pipelines."""

    model_config = SettingsConfigDict(
        env_prefix="RG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    github_token: str | None = Field(default=None, repr=False)
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    request_timeout: float = Field(default=10.0, ge=0.1, le=300.0, allow_inf_nan=False)
    max_concurrency: int = Field(default=4, ge=1, le=64)

    @field_validator("github_token")
    @classmethod
    def normalize_token(cls, value: str | None) -> str | None:
        """Treat an empty optional token from .env.example as absent."""
        return value.strip() or None if value is not None else None

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize and validate stdlib logging levels."""

        normalized = value.upper()
        if normalized not in VALID_LOG_LEVELS:
            allowed = ", ".join(sorted(VALID_LOG_LEVELS))
            msg = f"log_level must be one of: {allowed}"
            raise ValueError(msg)
        return normalized

    def ensure_data_dir(self) -> Self:
        """Create the configured data directory if it does not exist."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached project settings."""

    return Settings()
