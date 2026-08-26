"""Regression cover for the local readiness/canary runner and its privacy contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import wave
from pathlib import Path

import pytest
from conftest import ReleaseBundleBuilder

from voxdelta.api.dependencies import (
    ProviderConfigurationError,
    ProviderFactories,
    build_dependencies,
)
from voxdelta.config import Settings
from voxdelta.credentials import Credentials
from voxdelta.domain.models import EmotionLabel, EmotionResult, ProviderProvenance, ProviderUsage
from voxdelta.evaluation.calibration import CalibrationSummary, apply_temperature
from voxdelta.evaluation.emotion_training import CANONICAL_LABELS
from voxdelta.evaluation.readiness import (
    MEMORY_CEILING_MB,
    CanaryInput,
    LatencySummary,
    OutcomeSummary,
    ReadinessError,
    ReadinessGates,
    ReadinessReport,
    apply_offline_environment,
    canary_items_digest,
    offline_environment_enforced,
    select_canary_items,
    write_readiness_report,
    write_synthetic_canary,
)
from voxdelta.providers.calibrated_emotion import CalibratedEmotionProvider, calibrate_result
from voxdelta.providers.calibration_artifact import VerifiedCalibration
from voxdelta.providers.fake import FakeEmotionProvider

BACKEND = Path(__file__).resolve().parents[2]


def _summary(*, temperature: float = 1.5, threshold: float = 0.5) -> CalibrationSummary:
    return CalibrationSummary(
        fitted_item_count=3569,
        ece_bin_count=10,
        temperature=temperature,
        target_coverage=0.9,
        achieved_coverage=0.9002521714766041,
        abstain_threshold=threshold,
        accuracy_at_coverage=0.8123249299719888,
        pre_temperature_ece=0.08974033133605751,
        post_temperature_ece=0.01881642949260545,
    )


def _verified(*, temperature: float = 1.5, threshold: float = 0.5) -> VerifiedCalibration:
    return VerifiedCalibration(
        path=Path("/calibration"),
        calibration_id="xls-r-emotion-7class-v1-calibration-v2",
        binding_sha256="a" * 64,
        release_id="xls-r-emotion-7class-v1",
        bundle_tree_sha256="b" * 64,
        candidate_checkpoint_sha256="c" * 64,
        validation_items_sha256="d" * 64,
        validation_item_count=3569,
        summary=_summary(temperature=temperature, threshold=threshold),
    )


def _result(peak: float, *, label: EmotionLabel = "neutral") -> EmotionResult:
    remaining = (1.0 - peak) / 6
    probabilities = {name: remaining for name in CANONICAL_LABELS}
    probabilities[label] = peak
    return EmotionResult(
        utterance_id="sample-1",
        probabilities=probabilities,
        dominant_emotion=label,
        negative_intensity=0.0,
        operational_state="stable",
        confidence=peak,
        provider=ProviderProvenance(name="wav2vec-xls-r", model="m", remote=False, revision="r"),
        usage=ProviderUsage(latency_ms=12.0, peak_rss_mb=100.0),
    )


def _manifest_row(index: int, label: str, audio: Path, *, split: str = "validation") -> str:
    return json.dumps(
        {
            "id": f"item-{index:04d}",
            "call_id": f"call-{index:04d}",
            "speaker_id": f"speaker-{index:04d}",
            "audio_path": str(audio),
            "transcript": "",
            "split": split,
            "source": "emotion",
            "emotion": label,
            "start": None,
            "end": None,
            "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        }
    )


def _canary_manifest(tmp_path: Path, *, split: str = "validation", labels: object = None) -> Path:
    audio_root = tmp_path / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    clips = write_synthetic_canary(audio_root, 7)
    chosen = list(CANONICAL_LABELS) if labels is None else list(labels)  # type: ignore[arg-type]
    rows = [
        _manifest_row(index, chosen[index], clip, split=split)
        for index, clip in enumerate(clips[: len(chosen)])
    ]
    manifest = tmp_path / "canary.jsonl"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


# --- feature-flag regression: OFF preserves behaviour, ON wires the calibrated path ---


def _settings(tmp_path: Path, **updates: object) -> Settings:
    base: dict[str, object] = {
        "data_root": tmp_path / "data",
        "database_path": tmp_path / "data" / "voxdelta.sqlite3",
        "diarization_provider": "fake",
        "asr_provider": "fake",
        "emotion_provider": "fake",
    }
    base.update(updates)
    return Settings(**base)  # type: ignore[arg-type]


def test_both_flags_off_preserve_the_existing_fake_emotion_behaviour(tmp_path: Path) -> None:
    dependencies = build_dependencies(_settings(tmp_path), credentials=Credentials())

    provider = dependencies.runner.emotion_provider
    assert isinstance(provider, FakeEmotionProvider)
    assert not isinstance(provider, CalibratedEmotionProvider)


def test_rollback_by_disabling_both_flags_never_falls_back_to_the_release(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    calibration = (tmp_path / "calibration").resolve()
    calibration.mkdir()

    # Both flags off, but the paths remain configured: rollback must ignore them
    # entirely rather than quietly keep serving the release.
    rolled_back = build_dependencies(
        _settings(
            tmp_path,
            xlsr_release_enabled=False,
            xlsr_release_path=root,
            xlsr_calibration_enabled=False,
            xlsr_calibration_path=calibration,
        ),
        credentials=Credentials(),
    )

    provider = rolled_back.runner.emotion_provider
    assert isinstance(provider, FakeEmotionProvider)


def test_calibration_flag_without_the_release_flag_is_refused_at_settings_time(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="xlsr_calibration_enabled requires xlsr_release_enabled"):
        _settings(
            tmp_path,
            emotion_provider="wav2vec",
            xlsr_calibration_enabled=True,
            xlsr_calibration_path=(tmp_path / "calibration").resolve(),
        )


def test_enabled_release_flag_with_a_missing_calibration_fails_closed(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")

    with pytest.raises(ProviderConfigurationError):
        build_dependencies(
            _settings(
                tmp_path,
                emotion_provider="wav2vec",
                xlsr_release_enabled=True,
                xlsr_release_path=root,
                xlsr_calibration_enabled=True,
                xlsr_calibration_path=(tmp_path / "absent").resolve(),
            ),
            credentials=Credentials(),
            provider_factories=ProviderFactories(wav2vec_model=lambda *a, **k: object()),
        )


def test_release_is_verified_before_the_model_is_ever_allocated(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    """A tampered bundle must be rejected without the factory being called once."""

    root = release_bundles.build(tmp_path / "release")
    (root / "checkpoint" / "model.safetensors").write_bytes(b"tampered")
    allocations: list[object] = []

    def factory(checkpoint: Path, *, base_model_path: Path | None, device: str) -> object:
        allocations.append(checkpoint)
        return object()

    with pytest.raises(ProviderConfigurationError):
        build_dependencies(
            _settings(
                tmp_path,
                emotion_provider="wav2vec",
                xlsr_release_enabled=True,
                xlsr_release_path=root,
            ),
            credentials=Credentials(),
            provider_factories=ProviderFactories(wav2vec_model=factory),
        )

    assert allocations == []


# --- calibration semantics the canary depends on ---


@pytest.mark.parametrize("temperature", [0.25, 0.9, 1.0, 1.5058076119236363, 4.0, 9.5])
def test_temperature_scaling_never_changes_the_raw_top_label(temperature: float) -> None:
    generator = random.Random(20260826)
    for _ in range(500):
        weights = [generator.random() + 1e-9 for _ in CANONICAL_LABELS]
        total = math.fsum(weights)
        raw: dict[EmotionLabel, float] = {
            label: weight / total for label, weight in zip(CANONICAL_LABELS, weights, strict=True)
        }
        scaled = apply_temperature(raw, temperature)
        assert max(raw, key=lambda k: (raw[k], k)) == max(scaled, key=lambda k: (scaled[k], k))


def test_abstention_below_the_threshold_maps_to_the_uncertain_state() -> None:
    verified = _verified(temperature=1.0, threshold=0.9)

    abstained = calibrate_result(_result(0.5), verified)

    assert abstained.calibration is not None
    assert abstained.calibration.abstained is True
    assert abstained.operational_state == "uncertain"
    assert abstained.calibration.raw_confidence == pytest.approx(0.5)


def test_answering_above_the_threshold_keeps_a_real_operational_state() -> None:
    verified = _verified(temperature=1.0, threshold=0.2)

    answered = calibrate_result(_result(0.95), verified)

    assert answered.calibration is not None
    assert answered.calibration.abstained is False
    assert answered.operational_state != "uncertain"


# --- canary input contract: validation-only, authenticated, holdout unreachable ---


def test_canary_selection_takes_one_authenticated_item_per_label(tmp_path: Path) -> None:
    manifest = _canary_manifest(tmp_path)

    selected = select_canary_items(manifest)

    assert len(selected) == 7
    assert {item.emotion for item in selected} == set(CANONICAL_LABELS)
    assert all(item.split == "validation" for item in selected)


def test_canary_selection_is_deterministic(tmp_path: Path) -> None:
    manifest = _canary_manifest(tmp_path)

    first = canary_items_digest(select_canary_items(manifest))
    second = canary_items_digest(select_canary_items(manifest))

    assert first == second
    assert len(first) == 64


def test_canary_selection_refuses_a_test_split_manifest_before_any_inference(
    tmp_path: Path,
) -> None:
    manifest = _canary_manifest(tmp_path, split="test")

    with pytest.raises(ReadinessError) as error:
        select_canary_items(manifest)

    assert error.value.code == "readiness_input_not_validation_only"


def test_canary_selection_refuses_a_manifest_hiding_one_test_row(tmp_path: Path) -> None:
    manifest = _canary_manifest(tmp_path)
    rows = manifest.read_text(encoding="utf-8").splitlines()
    smuggled = json.loads(rows[3])
    smuggled["split"] = "test"
    rows[3] = json.dumps(smuggled)
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ReadinessError) as error:
        select_canary_items(manifest)

    assert error.value.code == "readiness_input_not_validation_only"


def test_canary_selection_refuses_audio_whose_bytes_do_not_match_the_manifest(
    tmp_path: Path,
) -> None:
    manifest = _canary_manifest(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    Path(row["audio_path"]).write_bytes(b"replaced")

    with pytest.raises(ReadinessError) as error:
        select_canary_items(manifest)

    assert error.value.code == "readiness_audio_mismatch"


def test_the_readiness_cli_exposes_no_final_holdout_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(BACKEND))
    from scripts.check_release_readiness import _parser

    options = {option for action in _parser()._actions for option in action.option_strings}

    assert not any(
        marker in option for option in options for marker in ("holdout", "package", "test-split")
    )
    assert "--validation-manifest" in options


def test_synthetic_canary_audio_is_deterministic_and_private(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = write_synthetic_canary(tmp_path / "a", 3)
    second = write_synthetic_canary(tmp_path / "b", 3)

    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes()
        assert os.stat(left).st_mode & 0o077 == 0
    with wave.open(str(first[0]), "rb") as handle:
        assert handle.getframerate() == 16_000
        assert handle.getnchannels() == 1


# --- report contract: aggregate-only, private, atomic, refuses overwrite ---


def _report(**gate_updates: object) -> ReadinessReport:
    gates: dict[str, object] = {
        "identity_exact": True,
        "offline_load": True,
        "verified_before_model_allocation": True,
        "completion_complete": True,
        "finite_valid_outputs": True,
        "peak_rss_within_ceiling": True,
        "abstention_maps_to_uncertain": True,
        "top_label_preserved": True,
        "overall_ok": True,
    }
    gates.update(gate_updates)
    return ReadinessReport(
        release={  # type: ignore[arg-type]
            "release_id": "xls-r-emotion-7class-v1",
            "bundle_tree_sha256": "b" * 64,
            "candidate_checkpoint_sha256": "c" * 64,
            "labels": tuple(CANONICAL_LABELS),
        },
        calibration={  # type: ignore[arg-type]
            "calibration_id": "xls-r-emotion-7class-v1-calibration-v2",
            "binding_sha256": "a" * 64,
            "method": "temperature-scaling",
            "temperature": 1.5,
            "abstain_threshold": 0.49,
            "target_coverage": 0.9,
            "fitted_item_count": 3569,
            "fitted_achieved_coverage": 0.9,
        },
        runtime={  # type: ignore[arg-type]
            "requested_device": "cpu",
            "selected_device": "cpu",
            "offline_enforced": True,
            "verified_before_model_allocation": True,
            "startup_verify_ms": 10.0,
            "cold_first_inference_ms": 900.0,
            "elapsed_seconds": 5.0,
            "peak_rss_mb": 2000.0,
            "memory_ceiling_mb": MEMORY_CEILING_MB,
        },
        canary={  # type: ignore[arg-type]
            "kind": "validation",
            "split": "validation",
            "item_count": 7,
            "repeats": 3,
            "label_coverage": 7,
            "audio_authenticated": True,
            "quality_evidence": False,
            "canary_items_sha256": "e" * 64,
        },
        latency=LatencySummary(
            count=20, min_ms=1.0, median_ms=2.0, p95_ms=3.0, max_ms=4.0, mean_ms=2.0
        ),
        outcome=OutcomeSummary(
            attempted_count=21,
            completed_count=21,
            completion_rate=1.0,
            finite_valid_outputs=True,
            abstained_count=2,
            abstention_rate=2 / 21,
            abstention_mapping_exercised=True,
            uncertain_state_count=2,
            mean_calibrated_confidence=0.7,
            mean_raw_confidence=0.8,
            min_calibrated_confidence=0.4,
            max_calibrated_confidence=0.9,
            top_label_agreement_rate=1.0,
            agreement_sample_count=7,
            top_label_agreement_exercised=True,
        ),
        gates=ReadinessGates(**gates),  # type: ignore[arg-type]
    )


def test_report_is_private_atomic_and_carries_its_own_checksum(tmp_path: Path) -> None:
    published = write_readiness_report(tmp_path / "run", _report())

    payload = published.read_bytes()
    assert os.stat(published).st_mode & 0o077 == 0
    checksums = (published.parent / "SHA256SUMS").read_text(encoding="utf-8")
    assert checksums == f"{hashlib.sha256(payload).hexdigest()}  READINESS.json\n"


def test_report_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    write_readiness_report(tmp_path / "run", _report())

    with pytest.raises(ReadinessError) as error:
        write_readiness_report(tmp_path / "run", _report())

    assert error.value.code == "readiness_report_publication_failed"


def test_report_refuses_an_existing_empty_run_directory(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()

    with pytest.raises(ReadinessError) as error:
        write_readiness_report(tmp_path / "run", _report())

    assert error.value.code == "readiness_report_publication_failed"
    assert list((tmp_path / "run").iterdir()) == []


def test_report_creation_race_never_removes_another_writers_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    real_mkdir = Path.mkdir

    def lose_creation_race(path: Path, *args: object, **kwargs: object) -> None:
        if path == run:
            real_mkdir(path)
            (path / "other-writer").write_text("keep", encoding="utf-8")
            raise FileExistsError
        real_mkdir(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", lose_creation_race)

    with pytest.raises(ReadinessError) as error:
        write_readiness_report(run, _report())

    assert error.value.code == "readiness_report_publication_failed"
    assert (run / "other-writer").read_text(encoding="utf-8") == "keep"


def test_report_publication_failure_leaves_no_partial_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voxdelta.evaluation.run_publication as run_publication

    real_publish = run_publication.publish_private_file
    calls = 0

    def fail_report(path: Path, payload: bytes, error_code: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError(error_code)
        real_publish(path, payload, error_code)

    monkeypatch.setattr(run_publication, "publish_private_file", fail_report)

    with pytest.raises(ReadinessError) as error:
        write_readiness_report(tmp_path / "run", _report())

    assert error.value.code == "readiness_report_publication_failed"
    assert not (tmp_path / "run").exists()


def test_report_contains_no_transcript_item_id_path_or_raw_probability(tmp_path: Path) -> None:
    published = write_readiness_report(tmp_path / "run", _report())

    text = published.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert "probabilities" not in text
    assert "transcript" not in text
    assert "audio_path" not in text
    assert "item-" not in text
    assert "/" not in text.replace("\\/", "")
    for label in CANONICAL_LABELS:
        # Labels appear only as the fixed release label order, never as a score map.
        assert f'"{label}":' not in text
    assert set(payload) == {
        "schema_version",
        "artifact_kind",
        "release",
        "calibration",
        "runtime",
        "canary",
        "latency",
        "outcome",
        "gates",
    }


def test_gate_summary_cannot_disagree_with_its_own_blocking_gates() -> None:
    with pytest.raises(ValueError, match="invalid readiness gates"):
        _report(peak_rss_within_ceiling=False)


def test_an_advisory_latency_threshold_never_changes_the_overall_verdict() -> None:
    report = _report()
    breached = report.gates.model_copy(
        update={"advisory_p95_latency_ms": 1.0, "advisory_p95_within_threshold": False}
    )

    assert breached.overall_ok is True


def test_offline_environment_is_enforced_before_any_load() -> None:
    environment: dict[str, str] = {}
    assert offline_environment_enforced(environment) is False

    apply_offline_environment(environment)

    assert offline_environment_enforced(environment) is True
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"


def test_canary_input_cannot_declare_the_holdout_reachable() -> None:
    with pytest.raises(ValueError):
        CanaryInput(
            kind="validation",
            split="validation",
            item_count=7,
            repeats=1,
            label_coverage=7,
            audio_authenticated=True,
            quality_evidence=False,
            holdout_reachable=True,  # type: ignore[arg-type]
        )


def test_quality_evidence_can_never_be_asserted_by_a_runtime_canary() -> None:
    """Reference labels are never scored here, so no run may claim quality evidence."""

    with pytest.raises(ValueError):
        CanaryInput(
            kind="validation",
            split="validation",
            item_count=7,
            repeats=1,
            label_coverage=7,
            audio_authenticated=True,
            quality_evidence=True,  # type: ignore[arg-type]
        )

    honest = CanaryInput(
        kind="validation",
        split="validation",
        item_count=7,
        repeats=1,
        label_coverage=7,
        audio_authenticated=True,
    )
    assert honest.quality_evidence is False


def test_a_canary_without_an_abstention_reports_the_mapping_as_not_exercised() -> None:
    report = _report()
    unexercised = report.outcome.model_copy(
        update={"abstained_count": 0, "abstention_rate": 0.0, "abstention_mapping_exercised": False}
    )

    assert unexercised.abstention_mapping_exercised is False
    # The blocking verdict must not be dragged down by an unexercised observation.
    assert report.gates.overall_ok is True


def test_a_canary_without_an_agreement_sample_reports_it_as_not_exercised() -> None:
    outcome = _report().outcome.model_copy(
        update={
            "agreement_sample_count": 0,
            "top_label_agreement_rate": 0.0,
            "top_label_agreement_exercised": False,
        }
    )

    assert outcome.top_label_agreement_exercised is False
    assert outcome.agreement_sample_count == 0
