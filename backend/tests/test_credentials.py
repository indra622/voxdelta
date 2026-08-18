from __future__ import annotations

import importlib
import os
import stat
from pathlib import Path
from types import ModuleType

import pytest

KEYS = ("HUGGINGFACE_TOKEN", "GEMINI_API_KEY", "PYANNOTEAI_API_KEY")


def credentials_module() -> ModuleType:
    return importlib.import_module("voxdelta.credentials")


def clear_process_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def write_env(path: Path, content: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)


def test_empty_and_missing_values_are_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_process_credentials(monkeypatch)
    env_file = tmp_path / ".env"
    write_env(
        env_file,
        "HUGGINGFACE_TOKEN=\nGEMINI_API_KEY=   \n",
    )

    check = credentials_module().check_credentials(env_file)

    assert check.statuses == {
        "HUGGINGFACE_TOKEN": "missing",
        "GEMINI_API_KEY": "missing",
        "PYANNOTEAI_API_KEY": "missing",
    }
    assert check.exit_code == 1


def test_required_credentials_are_ready_without_optional_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_process_credentials(monkeypatch)
    env_file = tmp_path / ".env"
    write_env(
        env_file,
        "HUGGINGFACE_TOKEN=hf-file-secret\nGEMINI_API_KEY=gemini-file-secret\n",
    )

    check = credentials_module().check_credentials(env_file)

    assert check.statuses == {
        "HUGGINGFACE_TOKEN": "configured",
        "GEMINI_API_KEY": "configured",
        "PYANNOTEAI_API_KEY": "missing",
    }
    assert check.exit_code == 0


def test_process_environment_overrides_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_process_credentials(monkeypatch)
    env_file = tmp_path / ".env"
    write_env(
        env_file,
        "HUGGINGFACE_TOKEN=hf-file-secret\nGEMINI_API_KEY=gemini-file-secret\n",
    )
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf-process-secret")

    credentials = credentials_module().load_credentials(env_file)

    assert credentials.huggingface_token.get_secret_value() == "hf-process-secret"


def test_secret_values_are_redacted_from_representations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_process_credentials(monkeypatch)
    env_file = tmp_path / ".env"
    raw_secrets = ("hf-do-not-leak", "gemini-do-not-leak", "pyannote-do-not-leak")
    write_env(
        env_file,
        "\n".join(
            (
                f"HUGGINGFACE_TOKEN={raw_secrets[0]}",
                f"GEMINI_API_KEY={raw_secrets[1]}",
                f"PYANNOTEAI_API_KEY={raw_secrets[2]}",
            )
        ),
    )

    check = credentials_module().check_credentials(env_file)
    combined_repr = f"{check.credentials!r} {check!r}"

    assert all(secret not in combined_repr for secret in raw_secrets)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission modes are required")
def test_public_loader_rejects_unsafe_env_file_permissions(tmp_path: Path) -> None:
    env_file = tmp_path / "backend" / ".env"
    secret = "must-not-appear-in-error"
    write_env(env_file, f"HUGGINGFACE_TOKEN={secret}\n", mode=0o644)
    module = credentials_module()

    with pytest.raises(module.UnsafeEnvFilePermissions) as error:
        module.load_credentials(env_file)

    message = str(error.value)
    assert str(env_file) in message
    assert "chmod 600 backend/.env" in message
    assert secret not in message


def test_exact_0600_env_file_permissions_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_process_credentials(monkeypatch)
    env_file = tmp_path / ".env"
    write_env(
        env_file,
        "HUGGINGFACE_TOKEN=hf-safe\nGEMINI_API_KEY=gemini-safe\n",
        mode=0o600,
    )

    check = credentials_module().check_credentials(env_file)

    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert check.exit_code == 0


def test_default_env_path_is_independent_of_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_process_credentials(monkeypatch)
    module = credentials_module()
    env_file = tmp_path / "backend" / ".env"
    write_env(
        env_file,
        "HUGGINGFACE_TOKEN=hf-cwd\nGEMINI_API_KEY=gemini-cwd\n",
    )
    monkeypatch.setattr(module, "DEFAULT_ENV_FILE", env_file)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    check = module.check_credentials()

    assert check.exit_code == 0
