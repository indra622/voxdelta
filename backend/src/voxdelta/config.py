"""Load application settings through the protected backend environment file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
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
    admission_reconciliation_lease_seconds: int = Field(default=300, gt=0)
    max_active_jobs: int = Field(default=8, gt=0)
    api_capability_token: SecretStr | None = None
    diarization_provider: Literal["fake", "pyannote-community"] = "fake"
    asr_provider: Literal["fake", "faster-whisper", "qwen3"] = "fake"
    emotion_provider: Literal["fake", "wav2vec", "emotion2vec"] = "fake"
    pyannote_checkpoint_path: Path | None = None
    emotion_checkpoint_path: Path | None = None
    asr_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    qwen_profile: Literal["default", "low-memory"] = "default"
    emotion_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    xlsr_release_enabled: bool = False
    xlsr_release_path: Path | None = None
    xlsr_calibration_enabled: bool = False
    xlsr_calibration_path: Path | None = None

    model_config = SettingsConfigDict(env_prefix="VOXDELTA_", extra="ignore")

    @model_validator(mode="after")
    def valid_audio_limits(self) -> Settings:
        if self.min_audio_seconds > self.max_audio_seconds:
            raise ValueError("min_audio_seconds must not exceed max_audio_seconds")
        return self

    @model_validator(mode="after")
    def valid_promoted_release_selection(self) -> Settings:
        """Reject release combinations that would leave the promoted bundle ambiguous.

        Messages name settings only; a configured bundle location is never echoed.
        """

        if self.xlsr_release_path is not None and not self.xlsr_release_path.is_absolute():
            raise ValueError("xlsr_release_path must be absolute")
        if self.xlsr_calibration_path is not None and not self.xlsr_calibration_path.is_absolute():
            raise ValueError("xlsr_calibration_path must be absolute")
        if self.xlsr_calibration_enabled:
            if not self.xlsr_release_enabled:
                raise ValueError("xlsr_calibration_enabled requires xlsr_release_enabled")
            if self.xlsr_calibration_path is None:
                raise ValueError("xlsr_calibration_enabled requires xlsr_calibration_path")
        if not self.xlsr_release_enabled:
            return self
        if self.xlsr_release_path is None:
            raise ValueError("xlsr_release_enabled requires xlsr_release_path")
        if self.emotion_provider != "wav2vec":
            raise ValueError("xlsr_release_enabled requires emotion_provider=wav2vec")
        if self.emotion_checkpoint_path is not None:
            raise ValueError("xlsr_release_enabled forbids emotion_checkpoint_path")
        return self


def load_settings(env_file: Path | None = None) -> Settings:
    """Validate and load settings from the backend env path and process environment."""

    selected_env_file = DEFAULT_ENV_FILE if env_file is None else env_file
    validate_env_file_permissions(selected_env_file)
    return Settings(_env_file=selected_env_file, _env_file_encoding="utf-8")


__all__ = ["Settings", "UnsafeEnvFilePermissions", "load_settings"]
