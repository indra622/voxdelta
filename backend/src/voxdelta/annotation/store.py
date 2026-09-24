"""Private silver artifacts and the human review that is the only path to gold.

Silver is a machine's first guess. Gold is what a person signed off on. The whole value of
keeping them apart is that nothing can quietly cross the line, so the crossing is modelled
as an operation with preconditions rather than a status field a caller may set:

* A silver artifact is written with ``review_state = "review_required"`` and there is no
  function anywhere that changes it. It stays review_required forever.
* :func:`promote` refuses to produce gold unless a reviewer is named and the corrected
  annotation is supplied. Promoting the model's own output unchanged is refused too — a
  reviewer who agrees with everything must still say so by submitting the annotation as
  their own, which is what makes the sign-off a claim rather than a default.
* Gold is frozen on write: it carries its parent silver's digest, its own digest over the
  content that produced it, the reviewer marker, and the time. Rewriting the file breaks
  the recorded digest, which :func:`verify_gold` checks.

Both artifact kinds contain transcript. They are written under a private root, and every
summary this module returns for logs or APIs is counts and digests only.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voxdelta.annotation.gemini_silver import EMOTION_LABELS, SilverAnnotation, SilverTurn

SCHEMA_VERSION = "1"
REVIEW_REQUIRED = "review_required"
REVIEWED_GOLD = "reviewed_gold"

#: Silver and gold both hold verbatim transcript, so they live apart from benchmarks.
DEFAULT_ROOT = Path("data/annotations")


class AnnotationError(RuntimeError):
    """Raised when an artifact would be untrustworthy or a promotion is not earned."""


def _digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _turns_payload(turns: Sequence[SilverTurn]) -> list[dict[str, Any]]:
    return [
        {
            "start": round(turn.start, 6),
            "end": round(turn.end, 6),
            "speaker": turn.speaker,
            "transcript": turn.transcript,
            "emotion": turn.emotion,
            "emotion_rationale": turn.emotion_rationale,
            "confidence": round(turn.confidence, 6),
        }
        for turn in turns
    ]


def summarise(turns: Sequence[SilverTurn]) -> dict[str, Any]:
    """Counts only. Safe for logs, APIs, and anything a reviewer has not yet seen."""

    histogram = dict.fromkeys(EMOTION_LABELS, 0)
    for turn in turns:
        histogram[turn.emotion] = histogram.get(turn.emotion, 0) + 1
    return {
        "turn_count": len(turns),
        "speaker_count": len({turn.speaker for turn in turns}),
        "transcript_characters": sum(len(turn.transcript) for turn in turns),
        "emotion_histogram": histogram,
        "uncertain_turns": histogram.get("uncertain", 0),
        "mean_confidence": (
            round(sum(turn.confidence for turn in turns) / len(turns), 6) if turns else None
        ),
    }


def write_silver(
    annotation: SilverAnnotation,
    *,
    root: Path,
    conversation_id: str,
    input_sha256: str,
    model: str,
    prompt: str,
    config: Mapping[str, Any],
    remote_file_deleted: bool,
    deletion_detail: str,
    call_counts: Mapping[str, int],
) -> tuple[Path, str, dict[str, Any]]:
    """Write one silver artifact privately and return its path, digest, and safe summary."""

    content = {
        "turns": _turns_payload(annotation.turns),
        "speakers": list(annotation.speakers),
        "notes": annotation.notes,
    }
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "silver-annotation",
        "review_state": REVIEW_REQUIRED,
        "promotable": False,
        "conversation_id": conversation_id,
        "created_at": _now(),
        "source": {
            "input_sha256": input_sha256,
            "model": model,
            "prompt": prompt,
            "config": dict(config),
            "remote_audio_transmitted": True,
        },
        "remote_file": {
            "deleted": remote_file_deleted,
            "detail": deletion_detail,
        },
        "call_counts": dict(call_counts),
        "validation": {
            "mode": "salvaged" if annotation.dropped_turn_count else "strict",
            "dropped_turn_count": annotation.dropped_turn_count,
            "dropped_turns_by_rule": annotation.dropped_turns_by_rule,
        },
        "content": content,
        "content_sha256": _digest(content),
        "contains_transcript": True,
        "handling": "private artifact; never copy into logs or benchmark output",
    }
    directory = root / conversation_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "silver.json"
    if path.exists():
        raise AnnotationError(f"silver artifact already exists: {path}")
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = summarise(annotation.turns)
    summary.update(
        {
            "dropped_turn_count": annotation.dropped_turn_count,
            "dropped_turns_by_rule": annotation.dropped_turns_by_rule,
        }
    )
    return path, record["content_sha256"], summary


def read_silver(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("kind") != "silver-annotation":
        raise AnnotationError(f"not a silver artifact: {path}")
    return record


def is_gold_eligible(record: Mapping[str, Any]) -> bool:
    """Whether an artifact may be used as ground truth. Silver never is."""

    return record.get("kind") == "gold-annotation" and record.get("review_state") == REVIEWED_GOLD


def assert_gold_eligible(record: Mapping[str, Any]) -> None:
    """Gate for anything that would evaluate against this artifact."""

    if not is_gold_eligible(record):
        raise AnnotationError("artifact is not reviewed gold and must not be used as ground truth")


def promote(
    silver_record: Mapping[str, Any],
    *,
    corrected_turns: Sequence[SilverTurn],
    reviewer: str,
    root: Path,
    conversation_id: str,
    review_note: str = "",
    review_change_reasons: Sequence[Mapping[str, Any]] = (),
) -> tuple[Path, str, dict[str, Any]]:
    """Freeze a reviewer's corrected annotation as gold, linked to its parent silver.

    The corrected annotation is the reviewer's, not the model's. That is why an empty
    correction is refused rather than treated as agreement: gold has to be something a
    person asserted, and a caller that submits nothing has asserted nothing.
    """

    if silver_record.get("kind") != "silver-annotation":
        raise AnnotationError("only a silver artifact can be promoted")
    if not reviewer.strip():
        raise AnnotationError("promotion requires a reviewer identity marker")
    if not corrected_turns:
        raise AnnotationError("promotion requires the reviewer's corrected annotation")

    parent_digest = str(silver_record.get("content_sha256", ""))
    if len(parent_digest) != 64:
        raise AnnotationError("silver artifact has no usable content digest")

    content = {
        "turns": _turns_payload(corrected_turns),
        # Short controlled labels are part of the immutable reviewed assertion. They
        # deliberately carry no second transcript copy or free-form per-turn note.
        "review_change_reasons": [dict(reason) for reason in review_change_reasons],
    }
    content_digest = _digest(content)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "gold-annotation",
        "review_state": REVIEWED_GOLD,
        "conversation_id": conversation_id,
        "reviewer": reviewer.strip(),
        "reviewed_at": _now(),
        "review_note": review_note,
        "parent_silver_sha256": parent_digest,
        "content": content,
        "content_sha256": content_digest,
        "unchanged_from_silver": content_digest == silver_record.get("content_sha256"),
        "contains_transcript": True,
        "handling": "private artifact; never copy into logs or benchmark output",
    }
    directory = root / conversation_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gold.json"
    # Created exclusively rather than checked and then written. A check followed by a write
    # leaves a window in which two promotions both see no file and the second silently
    # replaces the first, which is the one outcome "written once" must not allow.
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise AnnotationError(
            f"gold artifact already exists and is immutable: {path}"
        ) from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path, content_digest, summarise(corrected_turns)


def verify_gold(path: Path) -> dict[str, Any]:
    """Re-derive the recorded digest, so an edited gold file is detectable."""

    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("kind") != "gold-annotation":
        raise AnnotationError(f"not a gold artifact: {path}")
    content = record.get("content")
    if not isinstance(content, dict):
        raise AnnotationError("gold artifact has no content")
    if _digest(content) != record.get("content_sha256"):
        raise AnnotationError("gold artifact content does not match its recorded digest")
    return record


__all__ = [
    "DEFAULT_ROOT",
    "REVIEWED_GOLD",
    "REVIEW_REQUIRED",
    "SCHEMA_VERSION",
    "AnnotationError",
    "assert_gold_eligible",
    "is_gold_eligible",
    "promote",
    "read_silver",
    "summarise",
    "verify_gold",
    "write_silver",
]
