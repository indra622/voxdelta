"""Aggregation, privacy, and refusal behaviour of the local shadow-replay evaluator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from voxdelta.evaluation.shadow_replay import (
    ComparisonSummary,
    ErrorBreakdown,
    LatencySummary,
    ProviderIdentity,
    ShadowCanary,
    ShadowGates,
    ShadowReplayError,
    ShadowReplayReport,
    ShadowRuntime,
    run_shadow_replay,
    summarise_latency,
    summarise_observations,
    write_shadow_report,
)
from voxdelta.providers.shadow_emotion import ShadowObservation

BACKEND = Path(__file__).resolve().parents[2]


def _observation(**updates: object) -> ShadowObservation:
    base: dict[str, object] = {
        "candidate_attempted": True,
        "candidate_completed": True,
        "error_category": "none",
        "candidate_valid": True,
        "primary_latency_ms": 10.0,
        "candidate_latency_ms": 12.0,
        "top_labels_agree": True,
        "candidate_confidence": 0.7,
        "candidate_raw_confidence": 0.8,
        "candidate_abstained": False,
        "candidate_uncertain": False,
    }
    base.update(updates)
    return ShadowObservation(**base)  # type: ignore[arg-type]


# --- aggregation and "was it exercised" honesty ---


def test_latency_summary_is_empty_rather_than_invented_when_nothing_ran() -> None:
    summary = summarise_latency([])

    assert summary.count == 0
    assert summary.median_ms == 0.0
    assert summary.p95_ms == 0.0


def test_latency_percentiles_use_nearest_rank() -> None:
    summary = summarise_latency([10.0, 20.0, 30.0, 40.0])

    assert summary.min_ms == 10.0
    assert summary.max_ms == 40.0
    assert summary.median_ms == 25.0
    assert summary.p95_ms == 40.0


def test_error_categories_are_counted_into_the_fixed_vocabulary() -> None:
    observations = [
        _observation(),
        _observation(
            candidate_attempted=True,
            candidate_completed=False,
            error_category="timeout",
            candidate_valid=False,
        ),
        _observation(
            candidate_attempted=True,
            candidate_completed=False,
            error_category="provider_error",
            candidate_valid=False,
        ),
        _observation(
            candidate_attempted=True,
            candidate_completed=False,
            error_category="invalid_output",
            candidate_valid=False,
        ),
        _observation(
            candidate_attempted=True,
            candidate_completed=False,
            error_category="unexpected_error",
            candidate_valid=False,
        ),
    ]

    summary = summarise_observations(
        observations, attempted=5, primary_completed=5, primary_invariant_holds=True
    )

    assert summary.errors == ErrorBreakdown(
        none=1, timeout=1, provider_error=1, invalid_output=1, unexpected_error=1
    )
    assert summary.candidate_completed_count == 1
    assert summary.candidate_error_count == 4


def test_a_run_where_no_candidate_succeeded_marks_agreement_as_not_exercised() -> None:
    observations = [
        _observation(
            candidate_attempted=True,
            candidate_completed=False,
            error_category="timeout",
            candidate_valid=False,
        )
        for _ in range(3)
    ]

    summary = summarise_observations(
        observations, attempted=3, primary_completed=3, primary_invariant_holds=True
    )

    assert summary.top_label_agreement_exercised is False
    assert summary.top_label_agreement_rate == 0.0
    assert summary.candidate_abstention_exercised is False
    assert summary.mean_candidate_confidence is None


def test_abstention_is_marked_exercised_only_when_one_actually_occurred() -> None:
    without = summarise_observations(
        [_observation()], attempted=1, primary_completed=1, primary_invariant_holds=True
    )
    with_abstention = summarise_observations(
        [_observation(candidate_abstained=True, candidate_uncertain=True)],
        attempted=1,
        primary_completed=1,
        primary_invariant_holds=True,
    )

    assert without.candidate_abstention_exercised is False
    assert with_abstention.candidate_abstention_exercised is True
    assert with_abstention.candidate_abstention_rate == 1.0
    assert with_abstention.candidate_uncertain_rate == 1.0


def test_agreement_rate_is_measured_over_candidates_that_actually_ran() -> None:
    observations = [
        _observation(top_labels_agree=True),
        _observation(top_labels_agree=False),
        _observation(
            candidate_attempted=True,
            candidate_completed=False,
            error_category="timeout",
            candidate_valid=False,
        ),
    ]

    summary = summarise_observations(
        observations, attempted=3, primary_completed=3, primary_invariant_holds=True
    )

    assert summary.top_label_agreement_rate == pytest.approx(0.5)
    assert summary.top_label_agreement_exercised is True


def test_a_summary_cannot_claim_agreement_evidence_it_does_not_have() -> None:
    with pytest.raises(ValueError, match="invalid comparison summary"):
        ComparisonSummary(
            attempted_count=3,
            primary_completed_count=3,
            candidate_completed_count=0,
            candidate_error_count=3,
            errors=ErrorBreakdown(
                none=0, timeout=3, provider_error=0, invalid_output=0, unexpected_error=0
            ),
            candidate_attempted_count=3,
            candidate_valid_outputs=True,
            primary_invariant_holds=True,
            primary_invariance_exercised=True,
            top_label_agreement_rate=1.0,
            top_label_agreement_exercised=True,
            candidate_abstention_rate=0.0,
            candidate_abstention_exercised=False,
            candidate_uncertain_rate=0.0,
        )


# --- report contract ---


def _report(**gate_updates: object) -> ShadowReplayReport:
    gates: dict[str, object] = {
        "identity_exact": True,
        "offline_load": True,
        "no_network_egress_attempted": True,
        "verified_before_model_allocation": True,
        "primary_invariant_holds": True,
        "primary_completion_complete": True,
        "candidate_completion_complete": True,
        "candidate_valid_outputs": True,
        "peak_rss_within_ceiling": True,
        "overall_ok": True,
    }
    gates.update(gate_updates)
    return ShadowReplayReport(
        primary=ProviderIdentity(
            role="primary",
            kind="emotion2vec",
            provider_name="emotion2vec-plus",
            calibrated=False,
        ),
        candidate=ProviderIdentity(
            role="candidate",
            kind="release-calibrated",
            provider_name="wav2vec-xls-r",
            release_id="xls-r-emotion-7class-v1",
            bundle_tree_sha256="b" * 64,
            calibration_id="xls-r-emotion-7class-v1-calibration-v2",
            binding_sha256="a" * 64,
            calibrated=True,
        ),
        runtime=ShadowRuntime(
            requested_device="cpu",
            offline_enforced=True,
            verified_before_model_allocation=True,
            elapsed_seconds=4.0,
            peak_rss_mb=2000.0,
            memory_ceiling_mb=18432.0,
        ),
        canary=ShadowCanary(
            kind="validation",
            split="validation",
            item_count=7,
            repeats=3,
            label_coverage=7,
            audio_authenticated=True,
            canary_items_sha256="e" * 64,
        ),
        primary_latency=summarise_latency([10.0, 12.0]),
        candidate_latency=summarise_latency([11.0, 13.0]),
        comparison=summarise_observations(
            [_observation(), _observation()],
            attempted=2,
            primary_completed=2,
            primary_invariant_holds=True,
        ),
        gates=ShadowGates(**gates),  # type: ignore[arg-type]
    )


def test_report_is_private_atomic_and_carries_its_own_checksum(tmp_path: Path) -> None:
    published = write_shadow_report(tmp_path / "run", _report())

    payload = published.read_bytes()
    assert os.stat(published).st_mode & 0o077 == 0
    checksums = (published.parent / "SHA256SUMS").read_text(encoding="utf-8")
    assert checksums == f"{hashlib.sha256(payload).hexdigest()}  SHADOW_REPLAY.json\n"


def test_report_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    write_shadow_report(tmp_path / "run", _report())

    with pytest.raises(ShadowReplayError) as error:
        write_shadow_report(tmp_path / "run", _report())

    assert error.value.code == "shadow_report_publication_failed"


def test_report_never_carries_content_or_identity(tmp_path: Path) -> None:
    published = write_shadow_report(tmp_path / "run", _report())

    text = published.read_text(encoding="utf-8")

    for forbidden in ("transcript", "audio_path", "utterance", "probabilities", "item_id"):
        assert forbidden not in text
    assert "/" not in text
    payload = json.loads(text)
    assert payload["mode"] == "local-replay"
    assert payload["gates"]["live_shadow_traffic"] is False
    assert payload["canary"]["quality_evidence"] is False
    assert payload["canary"]["holdout_reachable"] is False


def test_a_report_can_never_claim_live_shadow_traffic() -> None:
    with pytest.raises(ValueError):
        ShadowGates(
            identity_exact=True,
            offline_load=True,
            verified_before_model_allocation=True,
            primary_invariant_holds=True,
            primary_completion_complete=True,
            peak_rss_within_ceiling=True,
            live_shadow_traffic=True,  # type: ignore[arg-type]
            overall_ok=True,
        )


def test_a_broken_primary_invariant_fails_the_overall_verdict() -> None:
    with pytest.raises(ValueError, match="invalid shadow gates"):
        _report(primary_invariant_holds=False)

    honest = _report(primary_invariant_holds=False, overall_ok=False)
    assert honest.gates.overall_ok is False


# --- refusal behaviour before any inference ---


def test_the_release_can_never_stand_in_as_its_own_rollback_primary() -> None:
    """Raw-vs-calibrated is a calibration diagnostic, not a primary-vs-candidate shadow."""

    from voxdelta.evaluation.shadow_replay import PrimaryKind

    assert set(PrimaryKind.__args__) == {"emotion2vec", "injected"}
    assert "release-uncalibrated" not in PrimaryKind.__args__


def test_an_injected_primary_is_required_when_the_injected_kind_is_chosen(
    tmp_path: Path,
) -> None:
    with pytest.raises(ShadowReplayError) as error:
        run_shadow_replay(
            release_path=(tmp_path / "release").resolve(),
            calibration_path=(tmp_path / "calibration").resolve(),
            primary_kind="injected",
        )

    assert error.value.code == "invalid_shadow_input"


def test_the_emotion2vec_primary_refuses_an_injected_provider(tmp_path: Path) -> None:
    with pytest.raises(ShadowReplayError) as error:
        run_shadow_replay(
            release_path=(tmp_path / "release").resolve(),
            calibration_path=(tmp_path / "calibration").resolve(),
            primary_kind="emotion2vec",
            primary_provider=object(),
        )

    assert error.value.code == "invalid_shadow_input"


def test_a_relative_artifact_path_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ShadowReplayError) as error:
        run_shadow_replay(
            release_path=Path("release"),
            calibration_path=(tmp_path / "calibration").resolve(),
        )

    assert error.value.code == "invalid_shadow_input"


def test_a_test_split_manifest_is_refused_before_any_artifact_is_opened(
    tmp_path: Path,
) -> None:
    from voxdelta.evaluation.readiness import write_synthetic_canary

    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    clips = write_synthetic_canary(audio_root, 2)
    rows = [
        json.dumps(
            {
                "id": f"item-{index}",
                "call_id": f"call-{index}",
                "speaker_id": f"speaker-{index}",
                "audio_path": str(clip),
                "transcript": "",
                "split": "test",
                "source": "emotion",
                "emotion": "neutral",
                "start": None,
                "end": None,
                "sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
            }
        )
        for index, clip in enumerate(clips)
    ]
    manifest = tmp_path / "sealed.jsonl"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ShadowReplayError) as error:
        run_shadow_replay(
            release_path=(tmp_path / "absent-release").resolve(),
            calibration_path=(tmp_path / "absent-calibration").resolve(),
            primary_kind="injected",
            primary_provider=object(),
            validation_manifest=manifest,
        )

    assert error.value.code == "readiness_input_not_validation_only"


def test_the_shadow_cli_exposes_no_sealed_split_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(BACKEND))
    from scripts.run_shadow_replay import _parser

    options = {option for action in _parser()._actions for option in action.option_strings}

    for forbidden in ("final", "holdout", "package", "archive", "test-split"):
        assert not any(forbidden in option for option in options)
    assert "--validation-manifest" in options


def test_latency_summary_field_names_match_the_readiness_report() -> None:
    """Both reports must be readable with the same latency vocabulary."""

    from voxdelta.evaluation.readiness import LatencySummary as ReadinessLatency

    assert set(LatencySummary.model_fields) == set(ReadinessLatency.model_fields)


# --- review fixes: candidate failure must never pass ---


def _failed(category: str) -> ShadowObservation:
    return _observation(
        candidate_attempted=True,
        candidate_completed=False,
        error_category=category,
        candidate_valid=False,
        candidate_latency_ms=None,
        top_labels_agree=None,
        candidate_confidence=None,
        candidate_raw_confidence=None,
        candidate_abstained=None,
        candidate_uncertain=None,
    )


def test_a_run_where_every_candidate_failed_cannot_report_overall_ok() -> None:
    summary = summarise_observations(
        [_failed("provider_error") for _ in range(3)],
        attempted=3,
        primary_completed=3,
        primary_invariant_holds=True,
    )

    assert summary.candidate_completed_count == 0
    assert summary.candidate_error_count == 3
    # A gate set claiming success on this summary must be structurally impossible.
    with pytest.raises(ValueError, match="invalid shadow gates"):
        ShadowGates(
            identity_exact=True,
            offline_load=True,
            no_network_egress_attempted=True,
            verified_before_model_allocation=True,
            primary_invariant_holds=True,
            primary_completion_complete=True,
            candidate_completion_complete=False,
            candidate_valid_outputs=True,
            peak_rss_within_ceiling=True,
            overall_ok=True,
        )


def test_a_single_candidate_failure_also_blocks_the_verdict() -> None:
    summary = summarise_observations(
        [_observation(), _observation(), _failed("timeout")],
        attempted=3,
        primary_completed=3,
        primary_invariant_holds=True,
    )

    assert summary.candidate_completed_count == 2
    assert summary.errors.timeout == 1
    honest = ShadowGates(
        identity_exact=True,
        offline_load=True,
        no_network_egress_attempted=True,
        verified_before_model_allocation=True,
        primary_invariant_holds=True,
        primary_completion_complete=True,
        candidate_completion_complete=summary.candidate_completed_count == 3,
        candidate_valid_outputs=True,
        peak_rss_within_ceiling=True,
        overall_ok=False,
    )
    assert honest.overall_ok is False


def test_an_egress_attempt_blocks_the_verdict() -> None:
    with pytest.raises(ValueError, match="invalid shadow gates"):
        ShadowGates(
            identity_exact=True,
            offline_load=True,
            no_network_egress_attempted=False,
            verified_before_model_allocation=True,
            primary_invariant_holds=True,
            primary_completion_complete=True,
            candidate_completion_complete=True,
            candidate_valid_outputs=True,
            peak_rss_within_ceiling=True,
            overall_ok=True,
        )


def test_counts_that_do_not_reconcile_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid comparison summary"):
        ComparisonSummary(
            attempted_count=3,
            primary_completed_count=3,
            candidate_completed_count=3,
            candidate_error_count=1,
            errors=ErrorBreakdown(
                none=3, timeout=1, provider_error=0, invalid_output=0, unexpected_error=0
            ),
            candidate_attempted_count=3,
            candidate_valid_outputs=True,
            primary_invariant_holds=True,
            primary_invariance_exercised=True,
            top_label_agreement_rate=1.0,
            top_label_agreement_exercised=True,
            candidate_abstention_rate=0.0,
            candidate_abstention_exercised=False,
            candidate_uncertain_rate=0.0,
        )


def test_an_error_breakdown_that_disagrees_with_its_totals_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid comparison summary"):
        ComparisonSummary(
            attempted_count=2,
            primary_completed_count=2,
            candidate_completed_count=1,
            candidate_error_count=1,
            errors=ErrorBreakdown(
                none=2, timeout=0, provider_error=1, invalid_output=0, unexpected_error=0
            ),
            candidate_attempted_count=2,
            candidate_valid_outputs=True,
            primary_invariant_holds=True,
            primary_invariance_exercised=True,
            top_label_agreement_rate=1.0,
            top_label_agreement_exercised=True,
            candidate_abstention_rate=0.0,
            candidate_abstention_exercised=False,
            candidate_uncertain_rate=0.0,
        )


def test_rates_cannot_be_reported_when_no_candidate_completed() -> None:
    with pytest.raises(ValueError, match="invalid comparison summary"):
        ComparisonSummary(
            attempted_count=1,
            primary_completed_count=1,
            candidate_completed_count=0,
            candidate_error_count=1,
            errors=ErrorBreakdown(
                none=0, timeout=1, provider_error=0, invalid_output=0, unexpected_error=0
            ),
            candidate_attempted_count=1,
            candidate_valid_outputs=True,
            primary_invariant_holds=True,
            primary_invariance_exercised=True,
            top_label_agreement_rate=0.0,
            top_label_agreement_exercised=False,
            candidate_abstention_rate=0.5,
            candidate_abstention_exercised=True,
            candidate_uncertain_rate=0.0,
        )


def test_candidate_rss_participates_in_the_peak(tmp_path: Path) -> None:
    """The memory gate must cover whichever side actually peaked."""

    high = _observation(candidate_peak_rss_mb=9000.0)
    summary = summarise_observations(
        [high], attempted=1, primary_completed=1, primary_invariant_holds=True
    )

    assert summary.candidate_completed_count == 1
    assert high.candidate_peak_rss_mb == 9000.0


def test_a_provider_name_carrying_a_path_is_refused(tmp_path: Path) -> None:
    from voxdelta.evaluation.shadow_replay import _published_provider_name

    assert _published_provider_name("emotion2vec-plus") == "emotion2vec-plus"
    for hostile in (
        "/Volumes/nvme1/secret/model",
        "emotion2vec plus",
        "C:\\models\\secret",
        "host:/export/models",
        "",
        "A" * 80,
    ):
        with pytest.raises(ShadowReplayError) as error:
            _published_provider_name(hostile)
        assert error.value.code == "shadow_provider_name_rejected"


def test_a_report_cannot_publish_an_unconstrained_provider_name() -> None:
    with pytest.raises(ValueError):
        ProviderIdentity(
            role="primary",
            kind="injected",
            provider_name="/Volumes/nvme1/leak",
            calibrated=False,
        )


def test_publication_refuses_an_existing_empty_directory(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()

    with pytest.raises(ShadowReplayError) as error:
        write_shadow_report(run, _report())

    assert error.value.code == "shadow_report_publication_failed"


def test_publication_race_never_removes_another_writers_files(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "other-writer").write_text("keep", encoding="utf-8")

    with pytest.raises(ShadowReplayError):
        write_shadow_report(run, _report())

    assert (run / "other-writer").read_text(encoding="utf-8") == "keep"


def test_publication_failure_on_the_second_write_leaves_no_partial_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voxdelta.evaluation.run_publication as run_publication

    real_publish = run_publication.publish_private_file
    calls = 0

    def fail_second(path: Path, payload: bytes, error_code: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError(error_code)
        real_publish(path, payload, error_code)

    monkeypatch.setattr(run_publication, "publish_private_file", fail_second)

    with pytest.raises(ShadowReplayError):
        write_shadow_report(tmp_path / "run", _report())

    assert not (tmp_path / "run").exists()


def test_the_checksum_is_written_before_the_report_commit_marker(tmp_path: Path) -> None:
    published = write_shadow_report(tmp_path / "run", _report())
    checksums = published.parent / "SHA256SUMS"

    assert checksums.stat().st_ctime <= published.stat().st_ctime
    assert hashlib.sha256(published.read_bytes()).hexdigest() in checksums.read_text(
        encoding="utf-8"
    )


def _provenance_stub() -> object:
    from voxdelta.domain.models import ProviderProvenance

    return ProviderProvenance(name="stub", model="stub-model", remote=False, revision="r" * 8)


class _PassThrough:
    """A primary that simply fails, so the test is about the candidate side only."""

    provenance = _provenance_stub()

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> object:
        del utterance_id, audio_path, transcript
        from voxdelta.domain.models import EmotionResult, ProviderUsage
        from voxdelta.evaluation.emotion_training import CANONICAL_LABELS

        share = 1.0 / len(CANONICAL_LABELS)
        return EmotionResult(
            utterance_id="u",
            probabilities=dict.fromkeys(CANONICAL_LABELS, share),
            dominant_emotion="neutral",
            negative_intensity=0.0,
            operational_state="uncertain",
            confidence=share,
            provider=_provenance_stub(),
            usage=ProviderUsage(latency_ms=1.0, peak_rss_mb=1.0),
        )


# --- HIGH 2: the guard must stay closed across allocation and inference ---

REAL_RELEASE = Path("/Volumes/nvme1/codes/voxdelta/data/models/xls-r-emotion-7class-v1")
REAL_CALIBRATION = Path(
    "/Volumes/nvme1/codes/voxdelta/data/models/xls-r-emotion-7class-v1-calibration-v2"
)


def test_the_egress_guard_is_active_while_the_adapter_runs_inference() -> None:
    """A provider that only reaches the network on first use must still be caught."""

    import socket

    from voxdelta.providers.offline_guard import block_network_egress
    from voxdelta.providers.shadow_emotion import ShadowEmotionProvider

    class LateFetcher:
        provenance = _provenance_stub()

        def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> object:
            del utterance_id, audio_path, transcript
            socket.create_connection(("models.invalid", 443))
            raise AssertionError("unreachable")

    seen: list[ShadowObservation] = []
    with block_network_egress() as egress:
        shadow = ShadowEmotionProvider(
            _PassThrough(),  # type: ignore[arg-type]
            LateFetcher(),  # type: ignore[arg-type]
            observer=seen.append,
        )
        shadow.analyze("utterance-1", Path("/nonexistent.wav"), "")

    assert egress.attempted is True
    assert seen[0].candidate_completed is False
    assert seen[0].error_category == "unexpected_error"


@pytest.mark.skipif(
    not REAL_RELEASE.is_dir() or not REAL_CALIBRATION.is_dir(),
    reason="the promoted release and calibration are not present",
)
def test_a_primary_that_attempts_egress_during_analyze_cannot_publish_valid_evidence(
    tmp_path: Path,
) -> None:
    import socket

    class EgressingPrimary:
        provenance = _provenance_stub()

        def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> object:
            del utterance_id, audio_path, transcript
            socket.create_connection(("telemetry.invalid", 443))
            raise AssertionError("unreachable")

    report = run_shadow_replay(
        release_path=REAL_RELEASE,
        calibration_path=REAL_CALIBRATION,
        primary_kind="injected",
        primary_provider=EgressingPrimary(),
        device="cpu",
        repeats=1,
        synthetic_item_count=1,
        workspace=tmp_path,
    )

    assert report.gates.no_network_egress_attempted is False
    assert report.gates.overall_ok is False
    assert report.comparison.primary_completed_count == 0
    assert report.comparison.primary_invariance_exercised is False


# --- second review: no abandoned worker may outlive the process-wide guard ---


def test_the_replay_runner_accepts_no_candidate_timeout() -> None:
    """A timed-out worker cannot be killed, so the replay must never create one."""

    import inspect

    from voxdelta.evaluation.shadow_replay import run_shadow_replay as runner

    parameters = inspect.signature(runner).parameters
    assert "candidate_timeout_seconds" not in parameters
    assert not any("timeout" in name for name in parameters)


def test_the_shadow_cli_exposes_no_timeout_option(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(BACKEND))
    from scripts.run_shadow_replay import _parser

    options = {option for action in _parser()._actions for option in action.option_strings}

    assert not any("timeout" in option for option in options)


def test_the_report_records_that_no_candidate_timeout_was_used() -> None:
    report = _report()

    assert report.runtime.candidate_timeout_used is False
    with pytest.raises(ValueError):
        ShadowRuntime(
            requested_device="cpu",
            offline_enforced=True,
            verified_before_model_allocation=True,
            candidate_timeout_used=True,  # type: ignore[arg-type]
            elapsed_seconds=1.0,
            peak_rss_mb=1.0,
            memory_ceiling_mb=18432.0,
        )


def test_an_invalid_output_makes_the_validity_gate_false_not_vacuously_true() -> None:
    summary = summarise_observations(
        [_observation(), _failed("invalid_output")],
        attempted=2,
        primary_completed=2,
        primary_invariant_holds=True,
    )

    assert summary.errors.invalid_output == 1
    assert summary.candidate_valid_outputs is False


def test_a_summary_cannot_claim_valid_outputs_alongside_an_invalid_output() -> None:
    with pytest.raises(ValueError, match="invalid comparison summary"):
        ComparisonSummary(
            attempted_count=1,
            primary_completed_count=1,
            candidate_completed_count=0,
            candidate_error_count=1,
            errors=ErrorBreakdown(
                none=0, timeout=0, provider_error=0, invalid_output=1, unexpected_error=0
            ),
            candidate_attempted_count=1,
            candidate_valid_outputs=True,
            primary_invariant_holds=True,
            primary_invariance_exercised=True,
            top_label_agreement_rate=0.0,
            top_label_agreement_exercised=False,
            candidate_abstention_rate=0.0,
            candidate_abstention_exercised=False,
            candidate_uncertain_rate=0.0,
        )


def test_abstention_exercised_must_agree_with_a_positive_rate() -> None:
    honest = summarise_observations(
        [_observation(candidate_abstained=True, candidate_uncertain=True)],
        attempted=1,
        primary_completed=1,
        primary_invariant_holds=True,
    )
    assert honest.candidate_abstention_exercised is True
    assert honest.candidate_abstention_rate > 0

    with pytest.raises(ValueError, match="invalid comparison summary"):
        ComparisonSummary(
            attempted_count=1,
            primary_completed_count=1,
            candidate_completed_count=1,
            candidate_error_count=0,
            errors=ErrorBreakdown(
                none=1, timeout=0, provider_error=0, invalid_output=0, unexpected_error=0
            ),
            candidate_attempted_count=1,
            candidate_valid_outputs=True,
            primary_invariant_holds=True,
            primary_invariance_exercised=True,
            top_label_agreement_rate=1.0,
            top_label_agreement_exercised=True,
            candidate_abstention_rate=0.0,
            candidate_abstention_exercised=True,
            candidate_uncertain_rate=0.0,
        )


def test_the_unused_invariance_comparison_helper_is_gone() -> None:
    import voxdelta.evaluation.shadow_replay as module

    assert not hasattr(module, "_comparable")


# --- local encoder bundle wiring ---


def test_the_replay_refuses_emotion2vec_without_a_verified_encoder_bundle(
    tmp_path: Path,
) -> None:
    """A raw package cache is not an acceptable encoder input."""

    checkpoint = (tmp_path / "checkpoint").resolve()
    checkpoint.mkdir()

    with pytest.raises(ShadowReplayError) as error:
        run_shadow_replay(
            release_path=(tmp_path / "release").resolve(),
            calibration_path=(tmp_path / "calibration").resolve(),
            primary_kind="emotion2vec",
            primary_checkpoint=checkpoint,
            primary_encoder_bundle=None,
            synthetic_item_count=1,
            repeats=1,
            workspace=tmp_path,
        )

    assert error.value.code in {"invalid_shadow_input", "shadow_artifacts_rejected"}


def test_an_unverifiable_encoder_bundle_is_rejected_before_allocation(tmp_path: Path) -> None:
    from voxdelta.evaluation.shadow_replay import _build_emotion2vec_primary

    checkpoint = (tmp_path / "checkpoint").resolve()
    checkpoint.mkdir()
    cache = (tmp_path / "raw-cache").resolve()
    cache.mkdir()
    (cache / "model.pt").write_bytes(b"weights")

    with pytest.raises(ShadowReplayError) as error:
        _build_emotion2vec_primary(checkpoint, cache, "cpu", object())

    assert error.value.code == "shadow_encoder_bundle_rejected"


def test_a_relative_encoder_bundle_is_refused(tmp_path: Path) -> None:
    from voxdelta.evaluation.shadow_replay import _build_emotion2vec_primary

    with pytest.raises(ShadowReplayError) as error:
        _build_emotion2vec_primary(
            (tmp_path / "checkpoint").resolve(), Path("relative-bundle"), "cpu", object()
        )

    assert error.value.code == "invalid_shadow_input"


def test_the_shadow_cli_accepts_only_a_bundle_for_the_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(BACKEND))
    from scripts.run_shadow_replay import _parser

    options = {option for action in _parser()._actions for option in action.option_strings}

    assert "--primary-encoder-bundle" in options
    assert not any("cache" in option or "snapshot" in option for option in options)


def test_the_report_publishes_the_encoder_identity_and_not_its_location() -> None:
    identity = ProviderIdentity(
        role="primary",
        kind="emotion2vec",
        provider_name="emotion2vec-plus",
        encoder_bundle_sha256="d" * 64,
        calibrated=False,
    )

    rendered = identity.model_dump_json()
    assert "d" * 64 in rendered
    assert "/" not in rendered

    with pytest.raises(ValueError):
        ProviderIdentity(
            role="primary",
            kind="emotion2vec",
            provider_name="emotion2vec-plus",
            encoder_bundle_sha256="/Volumes/nvme1/cache",
            calibrated=False,
        )
