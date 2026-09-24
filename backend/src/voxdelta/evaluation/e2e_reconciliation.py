"""Status taxonomy and additive reconciliation for the one-conversation KCSC E2E record.

The first run of that evaluation exposed a reporting flaw rather than a pipeline flaw.
The pipeline itself ran to completion — remote diarization, local ASR, local emotion, a
finished report — but a *post-run scoring* step failed, and because every check shared one
``verification_failures`` list the runner exited non-zero. Read from the outside, that made
a completed run look like a failed one.

This module separates the two questions that were being conflated:

* **Did the pipeline complete?** A property of the run itself.
* **Was the run scored?** A property of the measurement taken afterwards.

A scorer that cannot run says nothing about whether the pipeline worked, so a failure in
one must never be reported as a failure of the other. Equally, a run that completed
without being scored is not a scored run, and this taxonomy refuses to let it be presented
as one: there is no status that means "complete" while scoring is unavailable.

Reconciliation is deliberately **additive**. The original artifact is left byte-for-byte
as written and referenced by digest, and the wrapper's own ``failed``/exit-2 history is
carried forward verbatim rather than corrected. A record that quietly rewrote its own past
would be worth less than one that shows what happened and what was later understood.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "1"

#: Pipeline outcomes: whether the run itself reached a finished report.
PIPELINE_COMPLETED = "completed"
PIPELINE_FAILED = "failed"

#: Scoring outcomes: whether the post-run measurement could be taken at all.
SCORING_COMPLETE = "complete"
SCORING_UNAVAILABLE = "unavailable"

#: Overall outcomes. Note there is no value meaning "completed and scored" unless scoring
#: actually ran: an unscored run always carries the longer, less flattering name.
OVERALL_COMPLETED = "completed"
OVERALL_COMPLETED_SCORING_UNAVAILABLE = "completed_with_scoring_unavailable"
OVERALL_FAILED = "failed"

#: Failures raised by the post-run scorer rather than by the pipeline. Matched on prefix
#: because the runner formats the exception type into the rest of the message.
SCORING_FAILURE_PREFIX = "diarization scoring unavailable"


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a run's verification failures actually say about it."""

    pipeline_status: str
    scoring_status: str
    overall_status: str
    pipeline_failures: tuple[str, ...]
    scoring_failures: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "pipeline_status": self.pipeline_status,
            "scoring_status": self.scoring_status,
            "overall_status": self.overall_status,
            "pipeline_failures": list(self.pipeline_failures),
            "scoring_failures": list(self.scoring_failures),
        }


def classify(verification_failures: Sequence[str]) -> Outcome:
    """Split a flat failure list into what failed the pipeline and what failed the scoring.

    Anything not recognised as a scorer failure counts against the pipeline. Unknown
    failures are treated as the more serious kind on purpose: a message this module does
    not understand must not be able to downgrade itself into a footnote.
    """

    scoring = tuple(
        item for item in verification_failures if item.startswith(SCORING_FAILURE_PREFIX)
    )
    pipeline = tuple(
        item for item in verification_failures if not item.startswith(SCORING_FAILURE_PREFIX)
    )
    pipeline_status = PIPELINE_FAILED if pipeline else PIPELINE_COMPLETED
    scoring_status = SCORING_UNAVAILABLE if scoring else SCORING_COMPLETE
    if pipeline:
        overall = OVERALL_FAILED
    elif scoring:
        overall = OVERALL_COMPLETED_SCORING_UNAVAILABLE
    else:
        overall = OVERALL_COMPLETED
    return Outcome(
        pipeline_status=pipeline_status,
        scoring_status=scoring_status,
        overall_status=overall,
        pipeline_failures=pipeline,
        scoring_failures=scoring,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


#: Metrics the original record reports as zero only because the scorer never ran. Zero is
#: a measurement; these were not measured, and the reconciliation says so by name.
UNMEASURED_ON_SCORING_FAILURE = ("reference_turn_count", "reference_speakers")


def reconcile(
    evaluation_path: Path,
    *,
    corrected_scope: Mapping[str, object],
    wrapper_state: Mapping[str, object],
    scoring_note: str,
    recoverable: bool,
    recovery_note: str,
    withdrawn_claims: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Build an additive reconciliation for one evaluation record.

    Reads the original, never writes it. Every figure carried forward is metadata that was
    already in it; no audio, transcript, or segment data is read or reconstructed.
    """

    if not evaluation_path.is_file():
        raise FileNotFoundError(f"missing evaluation record: {evaluation_path}")
    original = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not isinstance(original, dict):
        raise ValueError(f"evaluation record has an unexpected shape: {evaluation_path}")

    failures = original.get("verification_failures", [])
    if not isinstance(failures, list):
        raise ValueError("verification_failures is not a list")
    outcome = classify([str(item) for item in failures])

    metrics = original.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    unmeasured = [name for name in UNMEASURED_ON_SCORING_FAILURE if name in metrics]

    remote = original.get("remote")
    remote = remote if isinstance(remote, dict) else {}
    rights = original.get("rights")
    rights = rights if isinstance(rights, dict) else {}

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "kcsc-precision2-qwen-xlsr-e2e-reconciliation",
        "transcript_free": True,
        "reconciles": {
            "path": str(evaluation_path),
            "sha256": file_sha256(evaluation_path),
            "started_at": original.get("started_at"),
            "note": "the referenced record is unmodified; this artifact only adds to it",
        },
        "outcome": outcome.as_dict(),
        "scoring": {
            "status": outcome.scoring_status,
            "what_was_not_produced": ["der", "jer", "diarization_accuracy"],
            "reason": scoring_note,
            "recoverable": recoverable,
            "recovery_note": recovery_note,
            "unmeasured_metrics_reported_as_zero": unmeasured,
        },
        "wrapper_history": {
            "state": wrapper_state.get("state"),
            "exit_code": wrapper_state.get("exit_code"),
            "reason": wrapper_state.get("reason"),
            "elapsed_seconds": wrapper_state.get("elapsed_seconds"),
            "note": (
                "preserved verbatim. The wrapper saw a non-zero exit and recorded failure, "
                "which was correct for what it could observe; the taxonomy above explains "
                "what that exit actually meant."
            ),
        },
        "corrected_scope": dict(corrected_scope),
        # Assertions the original record made that its own run did not establish. Kept as
        # a first-class list rather than a footnote: a withdrawn claim that is hard to
        # find is barely withdrawn.
        "withdrawn_claims": [dict(claim) for claim in withdrawn_claims],
        "superseded": {
            "field": "rights.scope",
            "original_value": rights.get("scope"),
            "why": (
                "the original text described the three-conversation programme scope, not "
                "this run, which submitted exactly one conversation"
            ),
        },
        "carried_metrics": {
            "utterance_count": metrics.get("utterance_count"),
            "emotion_result_count": metrics.get("emotion_result_count"),
            "emotion_abstained_count": metrics.get("emotion_abstained_count"),
            "transition_count": metrics.get("transition_count"),
            "hypothesis_segment_count": metrics.get("hypothesis_segment_count"),
            "hypothesis_speaker_count": metrics.get("hypothesis_speaker_count"),
            "elapsed_seconds": metrics.get("elapsed_seconds"),
            "real_time_factor": metrics.get("real_time_factor"),
        },
        "timestamp_coverage": original.get("timestamp_coverage"),
        "remote_calls": {
            "diarization_jobs_submitted": remote.get("diarization_jobs_submitted"),
            "uploads": remote.get("uploads"),
            "retries": remote.get("retries"),
            "call_counts_by_kind": remote.get("call_counts_by_kind"),
            "hosts_contacted": remote.get("hosts_contacted"),
        },
        "rights_confirmed": False,
        "user_limited_override": True,
    }


def write_reconciliation(record: Mapping[str, object], path: Path) -> str:
    """Write the reconciliation and return its digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(f"{raw}\n", encoding="utf-8")
    return file_sha256(path)


__all__ = [
    "OVERALL_COMPLETED",
    "OVERALL_COMPLETED_SCORING_UNAVAILABLE",
    "OVERALL_FAILED",
    "PIPELINE_COMPLETED",
    "PIPELINE_FAILED",
    "SCHEMA_VERSION",
    "SCORING_COMPLETE",
    "SCORING_FAILURE_PREFIX",
    "SCORING_UNAVAILABLE",
    "UNMEASURED_ON_SCORING_FAILURE",
    "Outcome",
    "classify",
    "file_sha256",
    "reconcile",
    "write_reconciliation",
]
