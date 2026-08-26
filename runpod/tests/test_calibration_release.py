from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_release import _fake_base_validator, _Sources
from voxdelta.evaluation.calibration import CalibrationSummary
from voxdelta.providers.release_bundle import VerifiedRelease

from voxdelta_runpod.calibration_release import (
    CalibrationReleaseError,
    CalibrationValidationSource,
    build_calibration_artifact,
    verify_calibration_release,
)
from voxdelta_runpod.release import verify_release_bundle as verify_runpod_release


def _fit(*, item_count: int = 3569) -> CalibrationSummary:
    return CalibrationSummary(
        fitted_item_count=item_count,
        ece_bin_count=10,
        temperature=1.5058076119236363,
        target_coverage=0.9,
        achieved_coverage=0.9002521714766041,
        abstain_threshold=0.49488208562011043,
        accuracy_at_coverage=0.8123249299719888,
        pre_temperature_ece=0.08974033133605751,
        post_temperature_ece=0.01881642949260545,
    )


def _validation(*, item_count: int = 3569) -> CalibrationValidationSource:
    return CalibrationValidationSource(
        archive_sha256="1" * 64,
        sidecar_sha256="2" * 64,
        packaged_manifest_sha256="3" * 64,
        validation_items_sha256="4" * 64,
        validation_item_count=item_count,
    )


def _verify_synthetic_release(path: Path) -> VerifiedRelease:
    manifest = verify_runpod_release(path, base_validator=_fake_base_validator)
    return VerifiedRelease(
        path=path,
        release_id=manifest.release_id,
        checkpoint_path=path / "checkpoint",
        base_model_path=path / "base-model",
        labels=tuple(manifest.labels),
        candidate_checkpoint_sha256=manifest.candidate_checkpoint_sha256,
        bundle_tree_sha256=manifest.bundle_tree_sha256,
    )


def test_separate_calibration_artifact_is_private_release_bound_and_reverifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _Sources(tmp_path)
    release = sources.build((tmp_path / "release").resolve())
    monkeypatch.setattr(
        "voxdelta_runpod.calibration_release.verify_release_bundle", _verify_synthetic_release
    )
    calibration = build_calibration_artifact(
        release_dir=release,
        output_dir=(tmp_path / "calibration").resolve(),
        calibration_id="xls-r-emotion-7class-v1-calibration-v1",
        fit=_fit(),
        validation=_validation(),
    )

    verified = verify_calibration_release(calibration, release)

    assert verified.release_id == "xls-r-emotion-7class-v1"
    assert verified.validation_item_count == 3569
    assert verified.summary == _fit()
    assert {path.name for path in calibration.iterdir()} == {
        "CALIBRATION.json",
        "CALIBRATION_CARD.md",
        "SHA256SUMS",
    }
    assert all(path.stat().st_mode & 0o077 == 0 for path in (calibration, *calibration.rglob("*")))
    assert "final" not in json.loads((calibration / "CALIBRATION.json").read_text())["validation"]


def test_calibration_artifact_refuses_overwrite_count_mismatch_and_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _Sources(tmp_path)
    release = sources.build((tmp_path / "release").resolve())

    monkeypatch.setattr(
        "voxdelta_runpod.calibration_release.verify_release_bundle", _verify_synthetic_release
    )
    output = (tmp_path / "calibration").resolve()
    calibration = build_calibration_artifact(
        release_dir=release,
        output_dir=output,
        calibration_id="xls-r-emotion-7class-v1-calibration-v1",
        fit=_fit(),
        validation=_validation(),
    )

    with pytest.raises(CalibrationReleaseError, match="^calibration_exists$"):
        build_calibration_artifact(
            release_dir=release,
            output_dir=output,
            calibration_id="xls-r-emotion-7class-v1-calibration-v1",
            fit=_fit(),
            validation=_validation(),
        )
    with pytest.raises(CalibrationReleaseError, match="^calibration_item_count_mismatch$"):
        build_calibration_artifact(
            release_dir=release,
            output_dir=(tmp_path / "wrong-count").resolve(),
            calibration_id="xls-r-emotion-7class-v1-calibration-v1",
            fit=_fit(item_count=3569),
            validation=_validation(item_count=350),
        )

    card = calibration / "CALIBRATION_CARD.md"
    card.write_text(card.read_text() + "tampered\n")
    with pytest.raises(CalibrationReleaseError, match="^invalid_calibration_artifact$"):
        verify_calibration_release(calibration, release)
