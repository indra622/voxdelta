from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from voxdelta.api.dependencies import (
    ProviderConfigurationError,
    ProviderFactories,
    build_dependencies,
)
from voxdelta.config import Settings, load_settings
from voxdelta.credentials import Credentials
from voxdelta.providers.fake import FakeDiarizationProvider
from voxdelta.providers.nemotron_diarization import CommandResult, NemotronDiarizationProvider
from voxdelta.providers.pyannote_diarization import PyannoteDiarizationProvider
from voxdelta.providers.pyannote_precision import PyannotePrecisionProvider


class _VersionOnlyRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self, argv: Sequence[str], *, env: Mapping[str, str], timeout_seconds: float
    ) -> CommandResult:
        del env, timeout_seconds
        self.calls.append(list(argv))
        return CommandResult(0, b"nemo-speech 0.1.0\n")


@pytest.fixture()
def runtime(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "runtime" / "nemo-speech"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 1\n")
    executable.chmod(0o755)
    model = tmp_path / "models" / "Nemotron-3-Diarization.q8_0.gguf"
    model.parent.mkdir()
    model.write_bytes(b"synthetic-gguf")
    return executable, model


def _settings(tmp_path: Path, **updates: object) -> Settings:
    base: dict[str, object] = {
        "data_root": tmp_path / "data",
        "database_path": tmp_path / "data" / "db.sqlite3",
    }
    base.update(updates)
    return Settings(**base)


def test_default_diarization_provider_is_unchanged(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.diarization_provider == "fake"
    assert settings.nemotron_executable_path is None
    assert settings.nemotron_model_path is None
    dependencies = build_dependencies(settings, credentials=Credentials())
    assert isinstance(dependencies.runner._diarization, FakeDiarizationProvider)


def test_nemotron_is_built_only_when_selected_and_is_local(
    tmp_path: Path, runtime: tuple[Path, Path]
) -> None:
    runner = _VersionOnlyRunner()
    settings = _settings(
        tmp_path,
        diarization_provider="nemotron-3-local",
        nemotron_executable_path=runtime[0],
        nemotron_model_path=runtime[1],
        nemotron_device="metal",
        nemotron_timeout_seconds=30,
    )

    dependencies = build_dependencies(
        settings,
        credentials=Credentials(),
        provider_factories=ProviderFactories(nemotron_runner=runner),
    )

    provider = dependencies.runner._diarization
    assert isinstance(provider, NemotronDiarizationProvider)
    assert provider.provenance.remote is False
    assert provider.configuration["device"] == "metal"
    assert provider.configuration["timeout_seconds"] == 30
    assert runner.calls == [[str(runtime[0].resolve()), "--version"]]


def test_nemotron_needs_no_credentials(tmp_path: Path, runtime: tuple[Path, Path]) -> None:
    settings = _settings(
        tmp_path,
        diarization_provider="nemotron-3-local",
        nemotron_executable_path=runtime[0],
        nemotron_model_path=runtime[1],
    )
    dependencies = build_dependencies(
        settings,
        credentials=Credentials(),
        provider_factories=ProviderFactories(nemotron_runner=_VersionOnlyRunner()),
    )
    assert isinstance(dependencies.runner._diarization, NemotronDiarizationProvider)


@pytest.mark.parametrize(
    ("missing", "code"),
    [("runtime", "local_runtime_missing"), ("model", "local_model_missing")],
)
def test_missing_runtime_or_model_fails_startup_without_fallback(
    tmp_path: Path, runtime: tuple[Path, Path], missing: str, code: str
) -> None:
    executable, model = runtime
    if missing == "runtime":
        executable.unlink()
    else:
        model.unlink()
    settings = _settings(
        tmp_path,
        diarization_provider="nemotron-3-local",
        nemotron_executable_path=executable,
        nemotron_model_path=model,
    )

    with pytest.raises(ProviderConfigurationError) as failure:
        build_dependencies(
            settings,
            credentials=Credentials(),
            provider_factories=ProviderFactories(nemotron_runner=_VersionOnlyRunner()),
        )

    assert failure.value.provider_code == code
    assert str(failure.value) == "provider_configuration_invalid"
    assert str(tmp_path) not in str(failure.value)


@pytest.mark.parametrize("unset", ["nemotron_executable_path", "nemotron_model_path"])
def test_selection_requires_both_locations(
    tmp_path: Path, runtime: tuple[Path, Path], unset: str
) -> None:
    values: dict[str, object] = {
        "diarization_provider": "nemotron-3-local",
        "nemotron_executable_path": runtime[0],
        "nemotron_model_path": runtime[1],
    }
    values.pop(unset)
    with pytest.raises(ValidationError) as failure:
        _settings(tmp_path, **values)
    assert unset in str(failure.value)


def test_relative_locations_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _settings(tmp_path, nemotron_executable_path=Path("bin/nemo-speech"))
    with pytest.raises(ValidationError):
        _settings(tmp_path, nemotron_model_path=Path("model.gguf"))


def test_timeout_is_bounded(tmp_path: Path) -> None:
    for timeout in (0, -1, 3601):
        with pytest.raises(ValidationError):
            _settings(tmp_path, nemotron_timeout_seconds=timeout)


def test_settings_parse_from_the_environment_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in (
        "VOXDELTA_DIARIZATION_PROVIDER",
        "VOXDELTA_NEMOTRON_EXECUTABLE_PATH",
        "VOXDELTA_NEMOTRON_MODEL_PATH",
        "VOXDELTA_NEMOTRON_DEVICE",
    ):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "VOXDELTA_DIARIZATION_PROVIDER=nemotron-3-local",
                "VOXDELTA_NEMOTRON_EXECUTABLE_PATH=/opt/nemo-speech/bin/nemo-speech",
                "VOXDELTA_NEMOTRON_MODEL_PATH=/opt/models/Nemotron-3-Diarization.q8_0.gguf",
                "VOXDELTA_NEMOTRON_DEVICE=metal",
            )
        ),
        encoding="utf-8",
    )
    env_file.chmod(0o600)

    settings = load_settings(env_file)

    assert settings.diarization_provider == "nemotron-3-local"
    assert settings.nemotron_executable_path == Path("/opt/nemo-speech/bin/nemo-speech")
    assert settings.nemotron_device == "metal"


def test_settings_errors_do_not_disclose_a_configured_location(tmp_path: Path) -> None:
    secret = "private-runtime-sentinel/nemo-speech"
    with pytest.raises(ValidationError) as failure:
        _settings(
            tmp_path,
            diarization_provider="nemotron-3-local",
            nemotron_executable_path=Path(secret),
            nemotron_model_path=Path("/m.gguf"),
        )
    assert "private-runtime-sentinel" not in str(failure.value)


def test_existing_pyannote_selections_are_unaffected(tmp_path: Path) -> None:
    local = build_dependencies(
        _settings(tmp_path, diarization_provider="pyannote-community"),
        credentials=Credentials(HUGGINGFACE_TOKEN="hf-test"),
    )
    assert isinstance(local.runner._diarization, PyannoteDiarizationProvider)
    remote = build_dependencies(
        _settings(tmp_path, diarization_provider="pyannoteai-precision"),
        credentials=Credentials(PYANNOTEAI_API_KEY="pa-test"),
    )
    assert isinstance(remote.runner._diarization, PyannotePrecisionProvider)
    assert remote.runner._diarization.provenance.remote is True
