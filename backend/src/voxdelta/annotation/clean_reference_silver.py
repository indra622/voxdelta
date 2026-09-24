"""Replace a drifted Gemini Silver draft with one coherent KCSC-reference draft.

This is deliberately a one-way artifact replacement, not another review-time patch:
the original Gemini artifact is preserved byte-for-byte, the replacement uses the
complete local human KCSC timeline, and all emotions remain ``uncertain``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voxdelta.annotation.reference_candidate import ReferenceCandidateError, load_candidate
from voxdelta.annotation.review import CONVERSATION_ID
from voxdelta.annotation.store import AnnotationError, read_silver

ARCHIVE_FILENAME = "silver.gemini-original.json"
SILVER_FILENAME = "silver.json"
MODEL_NAME = "kcsc-human-reference"


class CleanReferenceSilverError(RuntimeError):
    """Raised before any existing Silver artifact is changed."""


@dataclass(frozen=True, slots=True)
class CleanReferenceSilverReceipt:
    conversation_id: str
    archived_path: Path
    silver_path: Path
    archived_content_sha256: str
    content_sha256: str
    turn_count: int


def _digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _reject(message: str) -> CleanReferenceSilverError:
    return CleanReferenceSilverError(message)


def _input_sha256(record: Mapping[str, Any]) -> str:
    source = record.get("source")
    value = source.get("input_sha256") if isinstance(source, Mapping) else None
    if not isinstance(value, str) or len(value) != 64:
        raise _reject("the existing Silver draft has no usable source digest")
    return value


def regenerate_from_reference(
    root: Path,
    reference_root: Path,
    *,
    conversation_id: str,
) -> CleanReferenceSilverReceipt:
    """Archive one Gemini Silver draft and atomically publish a reference replacement."""

    if not CONVERSATION_ID.fullmatch(conversation_id):
        raise _reject("invalid conversation identifier")
    directory = root / conversation_id
    silver_path = directory / SILVER_FILENAME
    archive_path = directory / ARCHIVE_FILENAME
    if (directory / "gold.json").exists():
        raise _reject("a reviewed Gold artifact exists and cannot be replaced")
    if archive_path.exists():
        raise _reject("the original Gemini Silver artifact is already archived")
    try:
        original = read_silver(silver_path)
    except (AnnotationError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise _reject("the existing Silver draft is unreadable") from None
    source = original.get("source")
    if isinstance(source, Mapping) and source.get("model") == MODEL_NAME:
        raise _reject("the current Silver draft is already reference-based")
    original_digest = original.get("content_sha256")
    if not isinstance(original_digest, str) or len(original_digest) != 64:
        raise _reject("the existing Silver draft has no usable content digest")

    try:
        candidate = load_candidate(
            reference_root,
            conversation_id=conversation_id,
            source_silver_content_sha256=original_digest,
        )
    except ReferenceCandidateError:
        raise _reject("the local KCSC reference cannot be used for this conversation") from None
    if candidate is None:
        raise _reject("no local KCSC reference is available for this conversation")

    turns = [
        {
            "start": turn.start,
            "end": turn.end,
            "speaker": turn.speaker,
            "transcript": turn.transcript,
            "emotion": turn.emotion,
            "emotion_rationale": turn.emotion_rationale,
            "confidence": turn.confidence,
        }
        for turn in candidate.turns
    ]
    content = {
        "turns": turns,
        "speakers": list(candidate.speakers),
        "notes": (
            "Clean Silver draft rebuilt from the complete KCSC human reference timeline. "
            "Timing, speaker labels, and transcripts are reference-based; emotion requires "
            "reviewer judgment."
        ),
    }
    record: dict[str, Any] = {
        "schema_version": "1",
        "kind": "silver-annotation",
        "review_state": "review_required",
        "promotable": False,
        "conversation_id": conversation_id,
        "created_at": _now(),
        "source": {
            "input_sha256": _input_sha256(original),
            "model": MODEL_NAME,
            "prompt": "not_applicable_local_human_reference",
            "config": {
                "reference_sha256": candidate.source_reference_sha256,
                "replaces_silver_content_sha256": original_digest,
                "timeline": "complete_human_reference",
            },
            "remote_audio_transmitted": False,
        },
        "remote_file": {"deleted": False, "detail": "not_applicable_local_reference"},
        "call_counts": {"uploads": 0, "interactions": 0, "deletes": 0},
        "validation": {
            "mode": "reference_normalized",
            "dropped_turn_count": 0,
            "dropped_turns_by_rule": {},
        },
        "content": content,
        "content_sha256": _digest(content),
        "contains_transcript": True,
        "handling": (
            "private local reference artifact; emotion must be reviewed before Gold promotion"
        ),
    }

    temporary_path = directory / ".silver.reference.tmp"
    try:
        with temporary_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise _reject(
            "a previous clean Silver replacement was interrupted; resolve it first"
        ) from None
    except OSError:
        raise _reject("the clean Silver replacement could not be staged") from None

    try:
        os.replace(silver_path, archive_path)
        os.replace(temporary_path, silver_path)
    except OSError:
        if not silver_path.exists() and archive_path.exists():
            os.replace(archive_path, silver_path)
        raise _reject("the clean Silver replacement could not be published") from None

    return CleanReferenceSilverReceipt(
        conversation_id=conversation_id,
        archived_path=archive_path,
        silver_path=silver_path,
        archived_content_sha256=original_digest,
        content_sha256=str(record["content_sha256"]),
        turn_count=len(candidate.turns),
    )


__all__ = [
    "ARCHIVE_FILENAME",
    "CleanReferenceSilverError",
    "CleanReferenceSilverReceipt",
    "MODEL_NAME",
    "regenerate_from_reference",
]
