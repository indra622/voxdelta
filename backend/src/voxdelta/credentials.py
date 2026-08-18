"""Load and verify local credentials without revealing their values."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

CredentialStatus = Literal["configured", "missing"]


class Credentials(BaseSettings):
    """Credential values loaded from the process environment and local env file."""

    model_config = SettingsConfigDict(case_sensitive=True, extra="ignore")

    huggingface_token: SecretStr | None = Field(default=None, validation_alias="HUGGINGFACE_TOKEN")
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    pyannoteai_api_key: SecretStr | None = Field(
        default=None, validation_alias="PYANNOTEAI_API_KEY"
    )

    @field_validator("*", mode="before")
    @classmethod
    def blank_is_missing(cls, value: object) -> object:
        raw_value = value.get_secret_value() if isinstance(value, SecretStr) else value
        if isinstance(raw_value, str) and not raw_value.strip():
            return None
        return value


class UnsafeEnvFilePermissions(Exception):
    """Raised before loading a POSIX env file that is not private."""

    def __init__(self, env_file: Path) -> None:
        self.env_file = env_file
        super().__init__(
            f"Unsafe permissions on {env_file}; expected 0600. Run: chmod 600 backend/.env"
        )


@dataclass(frozen=True, slots=True)
class CredentialCheck:
    """Safe readiness result for the configured credentials."""

    credentials: Credentials

    @property
    def statuses(self) -> dict[str, CredentialStatus]:
        return {
            "HUGGINGFACE_TOKEN": _status(self.credentials.huggingface_token),
            "GEMINI_API_KEY": _status(self.credentials.gemini_api_key),
            "PYANNOTEAI_API_KEY": _status(self.credentials.pyannoteai_api_key),
        }

    @property
    def exit_code(self) -> Literal[0, 1]:
        required = (
            self.credentials.huggingface_token,
            self.credentials.gemini_api_key,
        )
        return 0 if all(value is not None for value in required) else 1


def _status(value: SecretStr | None) -> CredentialStatus:
    return "configured" if value is not None else "missing"


def validate_env_file_permissions(env_file: Path) -> None:
    """Require an existing POSIX env file to have mode exactly 0600."""

    if os.name != "posix" or not env_file.exists():
        return
    if stat.S_IMODE(env_file.stat().st_mode) != 0o600:
        raise UnsafeEnvFilePermissions(env_file)


def load_credentials(env_file: Path | None = None) -> Credentials:
    """Validate and load secrets from the backend env path and process environment."""

    selected_env_file = DEFAULT_ENV_FILE if env_file is None else env_file
    validate_env_file_permissions(selected_env_file)
    return Credentials(_env_file=selected_env_file, _env_file_encoding="utf-8")


def check_credentials(env_file: Path | None = None) -> CredentialCheck:
    """Validate the env file boundary and return a safe readiness result."""

    return CredentialCheck(credentials=load_credentials(env_file))
