"""Load application settings through the protected backend environment file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from voxdelta.credentials import (
    DEFAULT_ENV_FILE,
    UnsafeEnvFilePermissions,
    validate_env_file_permissions,
)


class Settings(BaseSettings):
    """Runtime paths and audio limits for the local backend."""

    data_root: Path = Path("../data")
    database_path: Path = Path("../data/voxdelta.sqlite3")
    max_audio_seconds: int = 3600
    min_audio_seconds: int = 60

    model_config = SettingsConfigDict(env_prefix="VOXDELTA_", extra="ignore")


def load_settings(env_file: Path | None = None) -> Settings:
    """Validate and load settings from the backend env path and process environment."""

    selected_env_file = DEFAULT_ENV_FILE if env_file is None else env_file
    validate_env_file_permissions(selected_env_file)
    return Settings(_env_file=selected_env_file, _env_file_encoding="utf-8")


__all__ = ["Settings", "UnsafeEnvFilePermissions", "load_settings"]
