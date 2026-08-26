from __future__ import annotations

import ast
import json
import os
import socket
from pathlib import Path

import pytest
from conftest import (
    PINNED_BASE_WEIGHTS_SHA256,
    RELEASE_LABELS,
    RELEASE_MODEL_ID,
    RELEASE_MODEL_REVISION,
    ReleaseBundleBuilder,
)

from voxdelta.api import dependencies as dependencies_module
from voxdelta.api.dependencies import (
    ProviderConfigurationError,
    ProviderFactories,
    build_dependencies,
)
from voxdelta.config import Settings
from voxdelta.credentials import Credentials
from voxdelta.providers.calibrated_emotion import CalibratedEmotionProvider
from voxdelta.providers.fake import FakeEmotionProvider
from voxdelta.providers.release_bundle import PROMOTED_RELEASE_ID
from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

REPOSITORY = Path(__file__).resolve().parents[3]
REAL_BUNDLE = (
    REPOSITORY / "runpod" / "dist" / "xlsr-622-20260825-v7" / "release" / PROMOTED_RELEASE_ID
)
REAL_CHECKPOINT_DIGEST = "6325effc75d27b3a7dcf512ad4a528d1b5f87cb7a2e4c15f23cd942d16afac5d"
REAL_CALIBRATION = (
    REPOSITORY
    / "runpod"
    / "dist"
    / "xlsr-622-20260825-v7"
    / "calibration"
    / "xls-r-emotion-7class-v1-calibration-v2"
)
REORDERED_LABELS = ("anger", "happiness", "disgust", "fear", "neutral", "sadness", "surprise")


class _StubPredictor:
    def predict(self, samples: tuple[float, ...], sample_rate: int) -> list[float]:
        del samples, sample_rate
        return [0.0] * 7


def _stub_factory(checkpoint: Path, *, base_model_path: Path | None, device: str) -> _StubPredictor:
    del checkpoint, base_model_path, device
    return _StubPredictor()


def _standalone_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "schema_version": "4",
                "architecture": "wav2vec-xls-r",
                "model_id": RELEASE_MODEL_ID,
                "labels": list(RELEASE_LABELS),
                "model_revision": RELEASE_MODEL_REVISION,
                "base_model_sha256": PINNED_BASE_WEIGHTS_SHA256,
                "class_weighting": "none",
                "class_weights": [1.0] * 7,
            }
        ),
        encoding="utf-8",
    )
    (path / "label_mapping.json").write_text(
        json.dumps({str(index): label for index, label in enumerate(RELEASE_LABELS)}),
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps({"macro_f1": 0.7, "validation_hash": "a" * 64}), encoding="utf-8"
    )
    (path / "model.safetensors").write_bytes(b"standalone-weights")
    return path


def _settings(tmp_path: Path, **updates: object) -> Settings:
    base: dict[str, object] = {
        "data_root": tmp_path / "data",
        "database_path": tmp_path / "data" / "db.sqlite3",
    }
    base.update(updates)
    return Settings(**base)


def _forbid_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dependencies_module,
        "verify_release_bundle",
        lambda _path: pytest.fail("a disabled flag must never verify a release bundle"),
    )


def test_flag_defaults_off_and_leaves_the_fake_emotion_provider_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_verification(monkeypatch)
    settings = _settings(tmp_path)

    dependencies = build_dependencies(settings, credentials=Credentials())

    assert settings.xlsr_release_enabled is False
    assert settings.xlsr_release_path is None
    assert isinstance(dependencies.runner._emotion, FakeEmotionProvider)


def test_flag_off_still_builds_the_standalone_wav2vec_checkpoint_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_verification(monkeypatch)
    checkpoint = _standalone_checkpoint(tmp_path / "checkpoint")
    settings = _settings(
        tmp_path,
        emotion_provider="wav2vec",
        emotion_checkpoint_path=checkpoint,
    )

    dependencies = build_dependencies(
        settings,
        credentials=Credentials(),
        provider_factories=ProviderFactories(wav2vec_model=_stub_factory),
    )

    emotion = dependencies.runner._emotion
    assert isinstance(emotion, Wav2VecEmotionProvider)
    assert emotion._checkpoint.path == checkpoint
    assert emotion._base_model_path is None


def test_flag_off_ignores_a_configured_release_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    _forbid_verification(monkeypatch)
    settings = _settings(tmp_path, xlsr_release_path=root)

    dependencies = build_dependencies(settings, credentials=Credentials())

    assert isinstance(dependencies.runner._emotion, FakeEmotionProvider)


def test_enabled_flag_threads_the_verified_bundle_paths_into_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    recorded: list[tuple[object, object, object]] = []

    class Recorder:
        def __init__(
            self,
            checkpoint_path: Path,
            *,
            base_model_path: Path | None = None,
            device: str = "auto",
            model_factory: object | None = None,
        ) -> None:
            recorded.append((checkpoint_path, base_model_path, device))
            self.model_factory = model_factory

    monkeypatch.setattr(dependencies_module, "Wav2VecEmotionProvider", Recorder)
    settings = _settings(
        tmp_path,
        emotion_provider="wav2vec",
        emotion_device="cpu",
        xlsr_release_enabled=True,
        xlsr_release_path=root,
    )

    dependencies = build_dependencies(settings, credentials=Credentials())

    assert recorded == [(root / "checkpoint", root / "base-model", "cpu")]
    assert isinstance(dependencies.runner._emotion, Recorder)


def test_enabled_calibration_verifies_against_the_release_and_wraps_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    calibration_path = (tmp_path / "calibration").resolve()
    calibration_path.mkdir()
    seen: dict[str, object] = {}
    verified = object()

    class RawProvider:
        provenance = object()

        def __init__(
            self,
            checkpoint_path: Path,
            *,
            base_model_path: Path | None = None,
            device: str = "auto",
            model_factory: object | None = None,
        ) -> None:
            del model_factory
            seen["checkpoint_path"] = checkpoint_path
            seen["base_model_path"] = base_model_path
            seen["device"] = device

    def verify_calibration(path: Path, *, release: object) -> object:
        seen["path"] = path
        seen["release"] = release
        return verified

    class Wrapper:
        def __init__(self, inner: object, calibration: object) -> None:
            seen["inner"] = inner
            seen["calibration"] = calibration
            self.provenance = inner.provenance  # type: ignore[union-attr]

    monkeypatch.setattr(
        dependencies_module, "verify_calibration_artifact", verify_calibration, raising=False
    )
    monkeypatch.setattr(dependencies_module, "CalibratedEmotionProvider", Wrapper, raising=False)
    monkeypatch.setattr(dependencies_module, "Wav2VecEmotionProvider", RawProvider)
    settings = _settings(
        tmp_path,
        emotion_provider="wav2vec",
        xlsr_release_enabled=True,
        xlsr_release_path=root,
        xlsr_calibration_enabled=True,
        xlsr_calibration_path=calibration_path,
    )

    dependencies = build_dependencies(
        settings,
        credentials=Credentials(),
        provider_factories=ProviderFactories(wav2vec_model=_stub_factory),
    )

    assert isinstance(dependencies.runner._emotion, Wrapper)
    assert seen["path"] == calibration_path
    assert seen["release"].path == root  # type: ignore[union-attr]
    assert isinstance(seen["inner"], RawProvider)
    assert seen["calibration"] is verified


def test_enabled_calibration_fails_closed_without_constructing_a_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    calibration_path = (tmp_path / "private-calibration-sentinel").resolve()

    def reject(path: Path, *, release: object) -> object:
        del path, release
        raise ValueError("invalid_calibration_artifact")

    monkeypatch.setattr(dependencies_module, "verify_calibration_artifact", reject, raising=False)
    monkeypatch.setattr(
        dependencies_module,
        "FakeEmotionProvider",
        lambda: pytest.fail("invalid calibration must not fall back"),
    )
    settings = _settings(
        tmp_path,
        emotion_provider="wav2vec",
        xlsr_release_enabled=True,
        xlsr_release_path=root,
        xlsr_calibration_enabled=True,
        xlsr_calibration_path=calibration_path,
    )

    with pytest.raises(ProviderConfigurationError) as raised:
        build_dependencies(
            settings,
            credentials=Credentials(),
            provider_factories=ProviderFactories(wav2vec_model=_stub_factory),
        )

    assert str(raised.value) == "provider_configuration_invalid"
    assert "private-calibration-sentinel" not in str(raised.value)


def test_enabled_flag_rejects_a_missing_bundle_with_the_path_free_public_error(
    tmp_path: Path,
) -> None:
    absent = tmp_path / "private-release-sentinel"
    settings = _settings(
        tmp_path,
        emotion_provider="wav2vec",
        xlsr_release_enabled=True,
        xlsr_release_path=absent,
    )

    with pytest.raises(ProviderConfigurationError) as raised:
        build_dependencies(settings, credentials=Credentials())

    assert str(raised.value) == "provider_configuration_invalid"
    assert "private-release-sentinel" not in str(raised.value)
    assert str(tmp_path) not in str(raised.value)


def test_enabled_flag_rejects_a_tampered_bundle_without_falling_back_to_another_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    (root / "checkpoint" / "model.safetensors").write_bytes(b"tampered-candidate-weights")
    monkeypatch.setattr(
        dependencies_module,
        "Wav2VecEmotionProvider",
        lambda *args, **kwargs: pytest.fail("a tampered bundle must never construct a provider"),
    )
    monkeypatch.setattr(
        dependencies_module,
        "FakeEmotionProvider",
        lambda *args, **kwargs: pytest.fail("a tampered bundle must never fall back"),
    )
    settings = _settings(
        tmp_path,
        emotion_provider="wav2vec",
        xlsr_release_enabled=True,
        xlsr_release_path=root,
    )

    with pytest.raises(ProviderConfigurationError) as raised:
        build_dependencies(settings, credentials=Credentials())

    assert str(raised.value) == "provider_configuration_invalid"


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks are required")
def test_enabled_flag_rejects_a_symlinked_bundle(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    link = tmp_path / "linked-release"
    link.symlink_to(root, target_is_directory=True)
    settings = _settings(
        tmp_path,
        emotion_provider="wav2vec",
        xlsr_release_enabled=True,
        xlsr_release_path=link,
    )

    with pytest.raises(ProviderConfigurationError):
        build_dependencies(settings, credentials=Credentials())


def test_enabled_flag_rejects_a_wrong_label_bundle(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(
        tmp_path / "release",
        label_mapping={str(index): label for index, label in enumerate(REORDERED_LABELS)},
    )
    settings = _settings(
        tmp_path,
        emotion_provider="wav2vec",
        xlsr_release_enabled=True,
        xlsr_release_path=root,
    )

    with pytest.raises(ProviderConfigurationError):
        build_dependencies(settings, credentials=Credentials())


def test_enabled_flag_rejects_a_malformed_bundle_manifest(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    (root / "RELEASE.json").write_bytes(b"{not-json")
    settings = _settings(
        tmp_path,
        emotion_provider="wav2vec",
        xlsr_release_enabled=True,
        xlsr_release_path=root,
    )

    with pytest.raises(ProviderConfigurationError):
        build_dependencies(settings, credentials=Credentials())


def test_production_backend_package_never_imports_the_experiment_only_runpod_package() -> None:
    package = Path(dependencies_module.__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            if any(
                name == "voxdelta_runpod" or name.startswith("voxdelta_runpod.")
                for name in imported
            ):
                offenders.append(path.relative_to(package).as_posix())

    assert offenders == []


@pytest.mark.skipif(not REAL_BUNDLE.is_dir(), reason="the promoted release bundle is not present")
def test_promoted_release_bundle_builds_an_offline_provider_from_the_real_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: pytest.fail("release verification must not open a socket"),
    )
    settings = _settings(
        tmp_path,
        emotion_provider="wav2vec",
        emotion_device="cpu",
        xlsr_release_enabled=True,
        xlsr_release_path=REAL_BUNDLE,
    )

    dependencies = build_dependencies(
        settings,
        credentials=Credentials(),
        provider_factories=ProviderFactories(wav2vec_model=_stub_factory),
    )

    emotion = dependencies.runner._emotion
    assert isinstance(emotion, Wav2VecEmotionProvider)
    assert emotion._checkpoint.path == REAL_BUNDLE / "checkpoint"
    assert emotion._base_model_path == REAL_BUNDLE / "base-model"
    assert emotion.provenance.remote is False
    assert emotion.provenance.revision == REAL_CHECKPOINT_DIGEST


@pytest.mark.skipif(
    not REAL_BUNDLE.is_dir() or not REAL_CALIBRATION.is_dir(),
    reason="the promoted release and calibration artifacts are not present",
)
def test_real_calibration_artifact_builds_the_fail_closed_offline_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: pytest.fail("calibration verification must not open a socket"),
    )
    settings = _settings(
        tmp_path,
        emotion_provider="wav2vec",
        emotion_device="cpu",
        xlsr_release_enabled=True,
        xlsr_release_path=REAL_BUNDLE,
        xlsr_calibration_enabled=True,
        xlsr_calibration_path=REAL_CALIBRATION,
    )

    dependencies = build_dependencies(
        settings,
        credentials=Credentials(),
        provider_factories=ProviderFactories(wav2vec_model=_stub_factory),
    )

    emotion = dependencies.runner._emotion
    assert isinstance(emotion, CalibratedEmotionProvider)
    assert emotion.calibration.calibration_id == "xls-r-emotion-7class-v1-calibration-v2"
    assert emotion.calibration.validation_item_count == 3569
    assert emotion.calibration.temperature == pytest.approx(1.5058076119236363)
    assert emotion.calibration.abstain_threshold == pytest.approx(0.49488208562011043)
