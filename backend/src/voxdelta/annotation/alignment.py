"""Private, immutable timestamp-only proposals for reviewing a Silver draft.

An alignment proposal is deliberately not another Silver draft: it has no transcript,
never changes Silver, and cannot become Gold.  It is just a reviewer-controlled
suggestion for the timing fields of a fixed suffix of one immutable Silver artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voxdelta.annotation.store import AnnotationError, read_silver
from voxdelta.providers.base import ProviderError

TARGET_START_POSITION = 10
PREFIX_START_POSITION = 1
PREFIX_END_POSITION = 9
KIND = "alignment-proposal"


class AlignmentError(AnnotationError):
    """The proposal is malformed, stale, or would mutate its source draft."""


def _digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _source_turns(
    record: Mapping[str, Any], *, target_start_position: int = TARGET_START_POSITION
) -> list[Mapping[str, Any]]:
    content = record.get("content")
    turns = content.get("turns") if isinstance(content, Mapping) else None
    if not isinstance(turns, list) or len(turns) < target_start_position:
        raise AlignmentError("source silver has no target range")
    if not all(isinstance(turn, Mapping) for turn in turns):
        raise AlignmentError("source silver has malformed turns")
    return turns


def _target_end(*, source_turn_count: int, target_end_position: int | None) -> int:
    end = source_turn_count if target_end_position is None else target_end_position
    if end < 1 or end > source_turn_count:
        raise AlignmentError("target range is outside source turns")
    return end


def prompt_for_range(
    record: Mapping[str, Any],
    *,
    duration_seconds: float,
    target_start_position: int = TARGET_START_POSITION,
    target_end_position: int | None = None,
) -> str:
    """Build one timestamp-only request. Text is sent to Gemini but never logged."""

    turns = _source_turns(record, target_start_position=target_start_position)
    end_position = _target_end(
        source_turn_count=len(turns), target_end_position=target_end_position
    )
    if target_start_position > end_position:
        raise AlignmentError("target range is empty")
    labels = [
        {
            "position": index,
            "speaker": turn["speaker"],
            "transcript": turn["transcript"],
            "old_start": turn["start"],
            "old_end": turn["end"],
        }
        for index, turn in enumerate(turns, start=1)
        if target_start_position <= index <= end_position
    ]
    return (
        "You are correcting TIMESTAMPS ONLY for a Korean two-party conversation. "
        f"The audio duration is {duration_seconds:.3f} seconds.  Re-align every supplied "
        f"label from position {target_start_position} through {end_position} against the audio. "
        "Preserve the "
        "position mapping exactly; do not return transcript, speaker, emotion, rationale, or "
        "any new labels. Return JSON with `rows` and optional short `notes`. Each row must be "
        "{position:int,start:number,end:number,confidence:number}; 0 <= start < end <= audio "
        "duration and confidence is in [0,1]. If a label cannot be aligned, omit that row "
        "rather than guessing. Source labels (private, for correspondence only):\n"
        + json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
    )


def prompt_for_suffix(record: Mapping[str, Any], *, duration_seconds: float) -> str:
    """Compatibility wrapper for the original 10-through-final proposal."""

    return prompt_for_range(record, duration_seconds=duration_seconds)


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position": {"type": "integer"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "confidence": {"type": "number"},
                },
                "required": ["position", "start", "end", "confidence"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["rows"],
}


def validate_rows(
    payload: object,
    *,
    source_turn_count: int,
    duration_seconds: float,
    target_start_position: int = TARGET_START_POSITION,
    target_end_position: int | None = None,
) -> tuple[tuple[dict[str, float | int], ...], dict[str, int]]:
    """Keep valid timestamp rows only; never persist model text or bad timestamps."""

    if not isinstance(payload, Mapping) or not isinstance(payload.get("rows"), list):
        raise ProviderError("invalid_provider_output")
    target_end = _target_end(
        source_turn_count=source_turn_count, target_end_position=target_end_position
    )
    if target_start_position > target_end:
        raise ProviderError("invalid_provider_output")
    accepted: list[dict[str, float | int]] = []
    dropped: dict[str, int] = {}
    seen: set[int] = set()
    for row in payload["rows"]:
        if not isinstance(row, Mapping):
            rule = "row.not_object"
        else:
            try:
                position = int(row["position"])
                start = float(row["start"])
                end = float(row["end"])
                confidence = float(row["confidence"])
            except (KeyError, TypeError, ValueError):
                rule = "row.invalid_fields"
            else:
                if position < target_start_position or position > target_end:
                    rule = "row.position_out_of_target"
                elif position in seen:
                    rule = "row.duplicate_position"
                elif not all(math.isfinite(value) for value in (start, end, confidence)):
                    rule = "row.non_finite"
                elif start < 0 or end <= start or end > duration_seconds:
                    rule = "row.invalid_interval"
                elif not 0 <= confidence <= 1:
                    rule = "row.confidence_out_of_range"
                else:
                    seen.add(position)
                    accepted.append(
                        {
                            "position": position,
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "confidence": round(confidence, 3),
                        }
                    )
                    continue
        dropped[rule] = dropped.get(rule, 0) + 1
    if not accepted:
        raise ProviderError("invalid_provider_output")
    accepted.sort(key=lambda row: int(row["position"]))
    return tuple(accepted), dict(sorted(dropped.items()))


def write_proposal(
    *,
    root: Path,
    conversation_id: str,
    source_record: Mapping[str, Any],
    rows: Sequence[Mapping[str, float | int]],
    dropped_by_rule: Mapping[str, int],
    model: str,
    input_sha256: str,
    remote_file_deleted: bool,
    deletion_detail: str,
    call_counts: Mapping[str, int],
    target_start_position: int = TARGET_START_POSITION,
    target_end_position: int | None = None,
) -> tuple[Path, str]:
    """Write once beside Silver; a later attempt must use a distinct version intentionally."""

    source_digest = source_record.get("content_sha256")
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        raise AlignmentError("source silver has no digest")
    source_turn_count = len(
        _source_turns(source_record, target_start_position=target_start_position)
    )
    end_position = _target_end(
        source_turn_count=source_turn_count, target_end_position=target_end_position
    )
    if target_start_position > end_position:
        raise AlignmentError("target range is empty")
    content = {
        "target_start_position": target_start_position,
        "target_end_position": end_position,
        "rows": list(rows),
    }
    record = {
        "schema_version": "1",
        "kind": KIND,
        "review_state": "review_required",
        "conversation_id": conversation_id,
        "created_at": _now(),
        "source_silver_content_sha256": source_digest,
        "source_input_sha256": input_sha256,
        "source_turn_count": source_turn_count,
        "model": model,
        "remote_file": {"deleted": remote_file_deleted, "detail": deletion_detail},
        "call_counts": dict(call_counts),
        "validation": {
            "proposed_row_count": len(rows),
            "dropped_row_count": sum(dropped_by_rule.values()),
            "dropped_rows_by_rule": dict(dropped_by_rule),
        },
        "content": content,
        "content_sha256": _digest(content),
        "contains_transcript": False,
        "handling": "private review proposal; never auto-applied to silver or gold",
    }
    directory = root / conversation_id
    directory.mkdir(parents=True, exist_ok=True)
    # Keep the original suffix filename stable so the existing proposal remains usable.
    filename = (
        f"alignment-proposal-{TARGET_START_POSITION}.json"
        if target_start_position == TARGET_START_POSITION and target_end_position is None
        else f"alignment-proposal-{target_start_position}-{end_position}.json"
    )
    path = directory / filename
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise AlignmentError("alignment proposal already exists") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path, str(record["content_sha256"])


def load_source(root: Path, conversation_id: str) -> dict[str, Any]:
    return read_silver(root / conversation_id / "silver.json")


def load_proposal(
    root: Path,
    conversation_id: str,
    *,
    source_digest: str,
    target_start_position: int = TARGET_START_POSITION,
    target_end_position: int | None = None,
) -> dict[str, Any] | None:
    """Read one range only when it is bound to the exact Silver the reviewer opened."""

    filename = (
        f"alignment-proposal-{TARGET_START_POSITION}.json"
        if target_start_position == TARGET_START_POSITION and target_end_position is None
        else f"alignment-proposal-{target_start_position}-{target_end_position}.json"
    )
    path = root / conversation_id / filename
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AlignmentError("alignment proposal is unreadable") from error
    if not isinstance(record, dict) or record.get("kind") != KIND:
        raise AlignmentError("alignment proposal has invalid kind")
    if record.get("source_silver_content_sha256") != source_digest:
        raise AlignmentError("alignment proposal belongs to a different silver")
    content = record.get("content")
    if not isinstance(content, Mapping):
        raise AlignmentError("alignment proposal has invalid content")
    if content.get("target_start_position") != target_start_position:
        raise AlignmentError("alignment proposal has wrong target range")
    if (
        target_end_position is not None
        and content.get("target_end_position") != target_end_position
    ):
        raise AlignmentError("alignment proposal has wrong target range")
    return record


def load_proposals(
    root: Path, conversation_id: str, *, source_digest: str
) -> list[dict[str, Any]]:
    """Return the known prefix and suffix suggestions in review order."""

    proposals: list[dict[str, Any]] = []
    for start, end in ((PREFIX_START_POSITION, PREFIX_END_POSITION), (TARGET_START_POSITION, None)):
        proposal = load_proposal(
            root,
            conversation_id,
            source_digest=source_digest,
            target_start_position=start,
            target_end_position=end,
        )
        if proposal is not None:
            proposals.append(proposal)
    return proposals
