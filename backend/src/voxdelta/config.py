"""Load application settings through the protected backend environment file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from voxdelta.credentials import (
    DEFAULT_ENV_FILE,
    UnsafeEnvFilePermissions,
    validate_env_file_permissions,
)

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


class Settings(BaseSettings):
    """Runtime paths and audio limits for the local backend."""

    data_root: Path = _DATA_ROOT
    database_path: Path = _DATA_ROOT / "voxdelta.sqlite3"
    max_audio_seconds: int = Field(default=3600, gt=0)
    min_audio_seconds: int = Field(default=60, gt=0)
    max_upload_bytes: int = Field(default=1024 * 1024 * 1024, gt=0)

    model_config = SettingsConfigDict(env_prefix="VOXDELTA_", extra="ignore")

    @model_validator(mode="after")
    def valid_audio_limits(self) -> Settings:
        if self.min_audio_seconds > self.max_audio_seconds:
            raise ValueError("min_audio_seconds must not exceed max_audio_seconds")
        return self


def load_settings(env_file: Path | None = None) -> Settings:
    """Validate and load settings from the backend env path and process environment."""

    selected_env_file = DEFAULT_ENV_FILE if env_file is None else env_file
    validate_env_file_permissions(selected_env_file)
    return Settings(_env_file=selected_env_file, _env_file_encoding="utf-8")


__all__ = ["Settings", "UnsafeEnvFilePermissions", "load_settings"]
