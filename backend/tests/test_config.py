from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

import pytest


def config_module() -> ModuleType:
    return importlib.import_module("voxdelta.config")


def write_env(path: Path, content: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)


def clear_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "VOXDELTA_DATA_ROOT",
        "VOXDELTA_DATABASE_PATH",
        "VOXDELTA_MAX_AUDIO_SECONDS",
        "VOXDELTA_MIN_AUDIO_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_public_loader_reads_application_settings_and_ignores_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    env_file = tmp_path / "backend" / ".env"
    write_env(
        env_file,
        "\n".join(
            (
                "VOXDELTA_DATA_ROOT=/safe/data",
                "VOXDELTA_MAX_AUDIO_SECONDS=900",
                "HUGGINGFACE_TOKEN=must-remain-a-secret",
            )
        ),
    )

    settings = config_module().load_settings(env_file)

    assert settings.data_root == Path("/safe/data")
    assert settings.max_audio_seconds == 900


def test_process_environment_overrides_settings_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    env_file = tmp_path / "backend" / ".env"
    write_env(env_file, "VOXDELTA_MAX_AUDIO_SECONDS=900\n")
    monkeypatch.setenv("VOXDELTA_MAX_AUDIO_SECONDS", "1200")

    settings = config_module().load_settings(env_file)

    assert settings.max_audio_seconds == 1200


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission modes are required")
def test_public_loader_rejects_unsafe_env_file_before_loading(tmp_path: Path) -> None:
    env_file = tmp_path / "backend" / ".env"
    secret = "settings-loader-must-not-leak"
    write_env(
        env_file,
        f"HUGGINGFACE_TOKEN={secret}\nVOXDELTA_MAX_AUDIO_SECONDS=900\n",
        mode=0o644,
    )
    module = config_module()

    with pytest.raises(module.UnsafeEnvFilePermissions) as error:
        module.load_settings(env_file)

    assert secret not in str(error.value)


def test_default_settings_env_path_is_independent_of_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    module = config_module()
    env_file = tmp_path / "backend" / ".env"
    write_env(env_file, "VOXDELTA_MIN_AUDIO_SECONDS=42\n")
    monkeypatch.setattr(module, "DEFAULT_ENV_FILE", env_file)
    elsewhere = tmp_path / "caller"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    settings = module.load_settings()

    assert settings.min_audio_seconds == 42
