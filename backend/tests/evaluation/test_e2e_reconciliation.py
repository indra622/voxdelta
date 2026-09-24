"""Tests for the E2E status taxonomy and the additive reconciliation artifact.

The defect this covers was a reporting one: a post-run scorer failure made a pipeline that
had completed look like a pipeline that had failed. So the tests pin both directions of
that confusion — a scoring gap must not read as a failed run, and a failed run must not be
softened into a scoring gap — plus the property that makes the fix trustworthy: the
original record is referenced by digest and never rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from voxdelta.evaluation.e2e_reconciliation import (
    OVERALL_COMPLETED,
    OVERALL_COMPLETED_SCORING_UNAVAILABLE,
    OVERALL_FAILED,
    PIPELINE_COMPLETED,
    PIPELINE_FAILED,
    SCORING_COMPLETE,
    SCORING_FAILURE_PREFIX,
    SCORING_UNAVAILABLE,
    classify,
    file_sha256,
    reconcile,
    write_reconciliation,
)

SCORING_FAILURE = f"{SCORING_FAILURE_PREFIX}: KcscBenchmarkError: entry has no speakers"
PIPELINE_FAILURE = "transcription provenance is not the local Qwen provider"


# ------------------------------------------------------------------------------- taxonomy


def test_a_scoring_failure_alone_leaves_the_pipeline_completed() -> None:
    """The exact confusion that produced this module: a measurement is not the run."""

    outcome = classify([SCORING_FAILURE])

    assert outcome.pipeline_status == PIPELINE_COMPLETED
    assert outcome.scoring_status == SCORING_UNAVAILABLE
    assert outcome.overall_status == OVERALL_COMPLETED_SCORING_UNAVAILABLE
    assert outcome.pipeline_failures == ()


def test_an_unscored_run_is_never_called_simply_completed() -> None:
    """There must be no status that presents an unscored run as a scored one."""

    outcome = classify([SCORING_FAILURE])

    assert outcome.overall_status != OVERALL_COMPLETED
    assert "scoring_unavailable" in outcome.overall_status


def test_a_clean_run_is_completed_and_scored() -> None:
    outcome = classify([])

    assert outcome.pipeline_status == PIPELINE_COMPLETED
    assert outcome.scoring_status == SCORING_COMPLETE
    assert outcome.overall_status == OVERALL_COMPLETED


def test_a_real_verification_failure_fails_the_pipeline() -> None:
    outcome = classify([PIPELINE_FAILURE])

    assert outcome.pipeline_status == PIPELINE_FAILED
    assert outcome.overall_status == OVERALL_FAILED
    assert outcome.scoring_status == SCORING_COMPLETE


def test_a_pipeline_failure_is_not_softened_by_an_accompanying_scoring_failure() -> None:
    outcome = classify([SCORING_FAILURE, PIPELINE_FAILURE])

    assert outcome.overall_status == OVERALL_FAILED
    assert outcome.pipeline_failures == (PIPELINE_FAILURE,)
    assert outcome.scoring_failures == (SCORING_FAILURE,)


def test_an_unrecognised_failure_counts_against_the_pipeline() -> None:
    """Unknown messages must not be able to demote themselves into a footnote."""

    outcome = classify(["something nobody has classified yet"])

    assert outcome.pipeline_status == PIPELINE_FAILED
    assert outcome.overall_status == OVERALL_FAILED


# -------------------------------------------------------------------------- reconciliation


def _record(tmp_path: Path, failures: list[str]) -> Path:
    path = tmp_path / "EVALUATION.json"
    path.write_text(
        json.dumps(
            {
                "kind": "kcsc-precision2-qwen-xlsr-e2e",
                "started_at": "2026-09-01T11:59:19+00:00",
                "transcript_free": True,
                "rights": {"scope": "programme scope naming three conversations"},
                "metrics": {
                    "utterance_count": 82,
                    "emotion_result_count": 29,
                    "emotion_abstained_count": 6,
                    "transition_count": 19,
                    "hypothesis_segment_count": 181,
                    "hypothesis_speaker_count": 2,
                    "reference_turn_count": 0,
                    "reference_speakers": 0,
                    "elapsed_seconds": 809.526,
                    "real_time_factor": 1.386389,
                },
                "remote": {
                    "diarization_jobs_submitted": 1,
                    "uploads": 1,
                    "retries": 0,
                    "call_counts_by_kind": {"diarize": 1, "upload": 1},
                    "hosts_contacted": ["api.pyannote.ai"],
                },
                "timestamp_coverage": {"omitted_words": 14, "uncertain": True},
                "verification_failures": failures,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _reconcile(path: Path, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "corrected_scope": {"conversation_ids": ["A6000_S0005_0"], "conversation_count": 1},
        "wrapper_state": {"state": "failed", "exit_code": 2, "reason": "exit code 2"},
        "scoring_note": "the caller passed an incomplete manifest entry",
        "recoverable": False,
        "recovery_note": "segments existed only in the removed scratch root",
    }
    arguments.update(overrides)
    return reconcile(path, **arguments)


def test_the_reconciliation_references_the_original_by_digest(tmp_path: Path) -> None:
    path = _record(tmp_path, [SCORING_FAILURE])
    before = file_sha256(path)

    record = _reconcile(path)

    assert record["reconciles"]["sha256"] == before
    # Additive: reading the original must not have changed it.
    assert file_sha256(path) == before


def test_the_reconciliation_reports_completion_with_scoring_unavailable(tmp_path: Path) -> None:
    record = _reconcile(_record(tmp_path, [SCORING_FAILURE]))

    assert record["outcome"]["pipeline_status"] == PIPELINE_COMPLETED
    assert record["outcome"]["overall_status"] == OVERALL_COMPLETED_SCORING_UNAVAILABLE
    assert record["scoring"]["status"] == SCORING_UNAVAILABLE


def test_it_never_claims_der_or_jer(tmp_path: Path) -> None:
    record = _reconcile(_record(tmp_path, [SCORING_FAILURE]))

    assert set(record["scoring"]["what_was_not_produced"]) == {
        "der",
        "jer",
        "diarization_accuracy",
    }
    flattened = json.dumps(record, ensure_ascii=False)
    assert '"der":' not in flattened
    assert '"jer":' not in flattened


def test_metrics_that_were_never_measured_are_named_rather_than_read_as_zero(
    tmp_path: Path,
) -> None:
    """The original reports 0 turns only because the scorer raised before counting any."""

    record = _reconcile(_record(tmp_path, [SCORING_FAILURE]))

    assert set(record["scoring"]["unmeasured_metrics_reported_as_zero"]) == {
        "reference_turn_count",
        "reference_speakers",
    }


def test_the_wrapper_history_is_preserved_verbatim(tmp_path: Path) -> None:
    """The audit trail keeps the failed/exit-2 record rather than correcting it."""

    record = _reconcile(_record(tmp_path, [SCORING_FAILURE]))

    assert record["wrapper_history"]["state"] == "failed"
    assert record["wrapper_history"]["exit_code"] == 2


def test_the_corrected_scope_names_one_conversation_and_keeps_rights_unconfirmed(
    tmp_path: Path,
) -> None:
    record = _reconcile(_record(tmp_path, [SCORING_FAILURE]))

    assert record["corrected_scope"]["conversation_ids"] == ["A6000_S0005_0"]
    assert record["corrected_scope"]["conversation_count"] == 1
    assert record["rights_confirmed"] is False
    assert record["user_limited_override"] is True
    assert record["superseded"]["field"] == "rights.scope"
    assert "three" in str(record["superseded"]["original_value"])


def test_the_reconciliation_carries_no_transcript(tmp_path: Path) -> None:
    record = _reconcile(_record(tmp_path, [SCORING_FAILURE]))

    def keys(node: object) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {key for value in node.values() for key in keys(value)}
        if isinstance(node, list):
            return {key for value in node for key in keys(value)}
        return set()

    present = keys(record)
    assert "transcript" not in present
    assert "utterances" not in present
    assert "text" not in present
    assert record["transcript_free"] is True


def test_a_missing_record_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing evaluation record"):
        _reconcile(tmp_path / "absent.json")


def test_writing_the_reconciliation_is_deterministic(tmp_path: Path) -> None:
    record = _reconcile(_record(tmp_path, [SCORING_FAILURE]))
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    assert write_reconciliation(record, first) == write_reconciliation(record, second)


# ------------------------------------------------------------- the shipped default


def test_the_shipped_env_selects_the_canonical_recogniser_with_a_fallback() -> None:
    """Qwen is the promoted default; faster-whisper remains reachable as the safety net."""

    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.is_file():
        pytest.skip("no local env file")
    text = env.read_text(encoding="utf-8")

    assert "VOXDELTA_ASR_PROVIDER=qwen3" in text
    assert "VOXDELTA_QWEN_PROFILE=default" in text
    assert "VOXDELTA_ASR_FALLBACK_PROVIDER=faster-whisper" in text


# ------------------------------------------------------------------- withdrawn claims


WITHDRAWN = {
    "field": "local.no_model_downloads",
    "original_value": True,
    "status": "withdrawn_unverified",
    "why": "the run had no mechanism that could have established the claim",
    "reverified": False,
}


def test_a_withdrawn_claim_is_carried_as_a_first_class_entry(tmp_path: Path) -> None:
    """A withdrawal buried in prose is barely a withdrawal."""

    record = _reconcile(_record(tmp_path, [SCORING_FAILURE]), withdrawn_claims=(WITHDRAWN,))

    assert record["withdrawn_claims"] == [WITHDRAWN]
    assert record["withdrawn_claims"][0]["status"] == "withdrawn_unverified"
    assert record["withdrawn_claims"][0]["reverified"] is False


def test_the_withdrawal_does_not_re_assert_the_claim_it_withdraws(tmp_path: Path) -> None:
    record = _reconcile(_record(tmp_path, [SCORING_FAILURE]), withdrawn_claims=(WITHDRAWN,))

    claim = record["withdrawn_claims"][0]
    # It records what was originally asserted, and marks it unverified rather than false:
    # the run established neither, so claiming the negative would be the same mistake.
    assert claim["original_value"] is True
    assert claim["status"] != "disproved"


def test_no_withdrawals_yields_an_empty_list_not_a_missing_key(tmp_path: Path) -> None:
    record = _reconcile(_record(tmp_path, [SCORING_FAILURE]))

    assert record["withdrawn_claims"] == []


def test_withdrawing_a_claim_leaves_the_taxonomy_and_scope_intact(tmp_path: Path) -> None:
    """The earlier corrections must survive this one."""

    record = _reconcile(_record(tmp_path, [SCORING_FAILURE]), withdrawn_claims=(WITHDRAWN,))

    assert record["outcome"]["overall_status"] == OVERALL_COMPLETED_SCORING_UNAVAILABLE
    assert record["outcome"]["pipeline_status"] == PIPELINE_COMPLETED
    assert record["corrected_scope"]["conversation_count"] == 1
    assert record["rights_confirmed"] is False
