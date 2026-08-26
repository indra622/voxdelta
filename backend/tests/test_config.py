from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

REPOSITORY = Path(__file__).resolve().parents[2]


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
        "VOXDELTA_MAX_UPLOAD_BYTES",
        "VOXDELTA_ADMISSION_RECONCILIATION_LEASE_SECONDS",
        "VOXDELTA_EMOTION_PROVIDER",
        "VOXDELTA_EMOTION_CHECKPOINT_PATH",
        "VOXDELTA_XLSR_RELEASE_ENABLED",
        "VOXDELTA_XLSR_RELEASE_PATH",
        "VOXDELTA_XLSR_CALIBRATION_ENABLED",
        "VOXDELTA_XLSR_CALIBRATION_PATH",
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
    assert settings.max_upload_bytes > 0
    assert settings.admission_reconciliation_lease_seconds > 0


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


def test_default_data_paths_are_absolute_and_independent_of_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    env_file = tmp_path / "backend" / ".env"
    write_env(env_file, "")
    elsewhere = tmp_path / "caller"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    settings = config_module().load_settings(env_file)

    assert settings.data_root == REPOSITORY / "data"
    assert settings.database_path == REPOSITORY / "data" / "voxdelta.sqlite3"
    assert settings.data_root.is_absolute()
    assert settings.database_path.is_absolute()


@pytest.mark.parametrize(
    "content",
    [
        "VOXDELTA_MIN_AUDIO_SECONDS=0\n",
        "VOXDELTA_MAX_AUDIO_SECONDS=0\n",
    ],
)
def test_public_loader_rejects_non_positive_audio_limits(
    content: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    env_file = tmp_path / "backend" / ".env"
    write_env(env_file, content)

    with pytest.raises(ValidationError):
        config_module().load_settings(env_file)


def test_public_loader_rejects_minimum_audio_limit_above_maximum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    env_file = tmp_path / "backend" / ".env"
    write_env(
        env_file,
        "VOXDELTA_MIN_AUDIO_SECONDS=61\nVOXDELTA_MAX_AUDIO_SECONDS=60\n",
    )

    with pytest.raises(ValidationError):
        config_module().load_settings(env_file)


def test_promoted_release_flag_defaults_to_disabled_with_no_bundle_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    monkeypatch.delenv("VOXDELTA_XLSR_RELEASE_ENABLED", raising=False)
    monkeypatch.delenv("VOXDELTA_XLSR_RELEASE_PATH", raising=False)
    env_file = tmp_path / "backend" / ".env"
    write_env(env_file, "")

    settings = config_module().load_settings(env_file)

    assert settings.xlsr_release_enabled is False
    assert settings.xlsr_release_path is None


def test_promoted_release_settings_parse_from_the_environment_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    monkeypatch.delenv("VOXDELTA_XLSR_RELEASE_ENABLED", raising=False)
    monkeypatch.delenv("VOXDELTA_XLSR_RELEASE_PATH", raising=False)
    env_file = tmp_path / "backend" / ".env"
    write_env(
        env_file,
        "\n".join(
            (
                "VOXDELTA_EMOTION_PROVIDER=wav2vec",
                "VOXDELTA_XLSR_RELEASE_ENABLED=true",
                "VOXDELTA_XLSR_RELEASE_PATH=/srv/releases/xls-r-emotion-7class-v1",
            )
        ),
    )

    settings = config_module().load_settings(env_file)

    assert settings.xlsr_release_enabled is True
    assert settings.xlsr_release_path == Path("/srv/releases/xls-r-emotion-7class-v1")
    assert settings.emotion_provider == "wav2vec"


def test_promoted_release_flag_requires_a_bundle_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    env_file = tmp_path / "backend" / ".env"
    write_env(
        env_file,
        "VOXDELTA_EMOTION_PROVIDER=wav2vec\nVOXDELTA_XLSR_RELEASE_ENABLED=true\n",
    )

    with pytest.raises(ValidationError):
        config_module().load_settings(env_file)


@pytest.mark.parametrize("provider", ["fake", "emotion2vec"])
def test_promoted_release_flag_requires_the_wav2vec_emotion_provider(
    provider: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    env_file = tmp_path / "backend" / ".env"
    write_env(
        env_file,
        "\n".join(
            (
                f"VOXDELTA_EMOTION_PROVIDER={provider}",
                "VOXDELTA_XLSR_RELEASE_ENABLED=true",
                "VOXDELTA_XLSR_RELEASE_PATH=/srv/releases/xls-r-emotion-7class-v1",
            )
        ),
    )

    with pytest.raises(ValidationError):
        config_module().load_settings(env_file)


def test_promoted_release_flag_forbids_a_separate_emotion_checkpoint_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    env_file = tmp_path / "backend" / ".env"
    write_env(
        env_file,
        "\n".join(
            (
                "VOXDELTA_EMOTION_PROVIDER=wav2vec",
                "VOXDELTA_EMOTION_CHECKPOINT_PATH=/srv/other/checkpoint",
                "VOXDELTA_XLSR_RELEASE_ENABLED=true",
                "VOXDELTA_XLSR_RELEASE_PATH=/srv/releases/xls-r-emotion-7class-v1",
            )
        ),
    )

    with pytest.raises(ValidationError):
        config_module().load_settings(env_file)


def test_promoted_release_bundle_path_must_be_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    env_file = tmp_path / "backend" / ".env"
    write_env(
        env_file,
        "\n".join(
            (
                "VOXDELTA_EMOTION_PROVIDER=wav2vec",
                "VOXDELTA_XLSR_RELEASE_ENABLED=true",
                "VOXDELTA_XLSR_RELEASE_PATH=releases/xls-r-emotion-7class-v1",
            )
        ),
    )

    with pytest.raises(ValidationError):
        config_module().load_settings(env_file)


def test_settings_validation_errors_never_disclose_the_configured_bundle_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    secret = "/srv/private-release-sentinel"
    env_file = tmp_path / "backend" / ".env"
    write_env(
        env_file,
        "\n".join(
            (
                "VOXDELTA_EMOTION_PROVIDER=fake",
                "VOXDELTA_XLSR_RELEASE_ENABLED=true",
                f"VOXDELTA_XLSR_RELEASE_PATH={secret}",
            )
        ),
    )

    with pytest.raises(ValidationError) as raised:
        config_module().load_settings(env_file)

    assert secret not in str(raised.value)


def test_xlsr_calibration_defaults_off_with_no_artifact_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    env_file = tmp_path / "backend" / ".env"
    write_env(env_file, "")

    settings = config_module().load_settings(env_file)

    assert settings.xlsr_calibration_enabled is False
    assert settings.xlsr_calibration_path is None


def test_xlsr_calibration_parses_only_with_the_promoted_release_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    env_file = tmp_path / "backend" / ".env"
    write_env(
        env_file,
        "\n".join(
            (
                "VOXDELTA_EMOTION_PROVIDER=wav2vec",
                "VOXDELTA_XLSR_RELEASE_ENABLED=true",
                "VOXDELTA_XLSR_RELEASE_PATH=/srv/releases/xls-r-emotion-7class-v1",
                "VOXDELTA_XLSR_CALIBRATION_ENABLED=true",
                "VOXDELTA_XLSR_CALIBRATION_PATH=/srv/calibration/xls-r-emotion-7class-v1-calibration-v1",
            )
        ),
    )

    settings = config_module().load_settings(env_file)

    assert settings.xlsr_calibration_enabled is True
    assert settings.xlsr_calibration_path == Path(
        "/srv/calibration/xls-r-emotion-7class-v1-calibration-v1"
    )


@pytest.mark.parametrize(
    "content",
    (
        "VOXDELTA_XLSR_CALIBRATION_ENABLED=true\n"
        "VOXDELTA_XLSR_CALIBRATION_PATH=/srv/calibration/v1\n",
        "VOXDELTA_EMOTION_PROVIDER=wav2vec\n"
        "VOXDELTA_XLSR_RELEASE_ENABLED=true\n"
        "VOXDELTA_XLSR_RELEASE_PATH=/srv/releases/v1\n"
        "VOXDELTA_XLSR_CALIBRATION_ENABLED=true\n",
        "VOXDELTA_EMOTION_PROVIDER=wav2vec\n"
        "VOXDELTA_XLSR_RELEASE_ENABLED=true\n"
        "VOXDELTA_XLSR_RELEASE_PATH=/srv/releases/v1\n"
        "VOXDELTA_XLSR_CALIBRATION_ENABLED=true\n"
        "VOXDELTA_XLSR_CALIBRATION_PATH=calibration/v1\n",
    ),
)
def test_xlsr_calibration_rejects_incomplete_or_relative_configuration(
    content: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_settings_environment(monkeypatch)
    env_file = tmp_path / "backend" / ".env"
    write_env(env_file, content)

    with pytest.raises(ValidationError):
        config_module().load_settings(env_file)
