"""Remote Gemini emotion-only overlay candidates for one immutable Silver draft.

This is the Gemini sibling of :mod:`voxdelta.annotation.emotion_candidates`, and the
difference that matters is the one in its name: producing this candidate sends the
recording to Google, so it carries ``remote_audio_transmitted = true`` and is generated
only behind the same explicit ``gemini_audio_transfer`` consent, the same fixed
one-upload/one-interaction/one-delete ledger, and the same no-retry rule as
:mod:`voxdelta.annotation.gemini_silver`.

What it may change is deliberately narrow. Timing, speaker, and transcript are copied
out of the clean Silver draft field for field; only ``emotion``, ``confidence``, and
``emotion_rationale`` come from the model. Nothing here writes ``silver.json``, gold, or
the local XLS-R candidate, and the artifact is written ``review_required`` and
``promotable = false``.

Validation is all-or-nothing, unlike the salvaging path Silver drafting uses. A Silver
draft can afford to drop a malformed turn because each turn carries its own timing. An
overlay cannot: it is nothing but a positional mapping onto turns it does not restate,
so a response missing position 42 is indistinguishable from one that silently shifted
everything after it. A single bad row therefore rejects the whole response.

Transcript text is sent to Gemini so the model can key its answer to the right turn, the
same way the alignment proposal does. It is never logged, never put in an error message,
and never written anywhere but the private candidate artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voxdelta.annotation.gemini_silver import EMOTION_LABELS, GEMINI_MODEL, SilverTurn
from voxdelta.annotation.store import AnnotationError, _turns_payload, summarise
from voxdelta.domain.models import ProviderProvenance
from voxdelta.providers.base import ProviderError

KIND = "gemini-emotion-overlay-candidate"
MODEL = GEMINI_MODEL
FILENAME = "emotion-candidate.gemini-remote.json"

#: A rationale is meant to be a sentence a reviewer can check against the audio, not a
#: transcript restatement. Bounded so one runaway field cannot bloat a private artifact.
MAX_RATIONALE_CHARACTERS = 400


class GeminiEmotionOverlayError(ProviderError):
    """A rejected overlay response plus a text-free description of the broken rule."""

    def __init__(self, diagnostic: Mapping[str, object]) -> None:
        self.diagnostic = dict(diagnostic)
        super().__init__("invalid_provider_output")


class OverlayCandidateError(RuntimeError):
    """A stored Gemini overlay candidate could not be used safely."""


@dataclass(frozen=True, slots=True)
class OverlayRow:
    """One model opinion about one existing turn, keyed by its 1-based position."""

    position: int
    emotion: str
    confidence: float
    rationale: str


def provenance() -> ProviderProvenance:
    """Declares remote transmission of both the audio and the reference transcript."""

    return ProviderProvenance(
        name="gemini-emotion-overlay",
        model=MODEL,
        remote=True,
        transmits=("audio", "text"),
        retention_policy_url="https://ai.google.dev/gemini-api/terms",
        revision=MODEL,
    )


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "integer",
                        "description": "The 1-based position of the supplied turn.",
                    },
                    "emotion": {"type": "string", "enum": list(EMOTION_LABELS)},
                    "confidence": {"type": "number"},
                    "rationale": {
                        "type": "string",
                        "description": "One short reason a reviewer can check against the audio.",
                    },
                },
                "required": ["position", "emotion", "confidence", "rationale"],
            },
        },
    },
    "required": ["rows"],
}


def _digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def source_turns(record: Mapping[str, Any]) -> tuple[SilverTurn, ...]:
    """Read the clean Silver turns this overlay must preserve exactly."""

    content = record.get("content")
    rows = content.get("turns") if isinstance(content, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise AnnotationError("silver artifact has no reviewable turns")
    try:
        return tuple(
            SilverTurn(
                start=float(row["start"]),
                end=float(row["end"]),
                speaker=str(row["speaker"]),
                transcript=str(row["transcript"]),
                emotion=str(row["emotion"]),
                emotion_rationale=str(row["emotion_rationale"]),
                confidence=float(row["confidence"]),
            )
            for row in rows
            if isinstance(row, Mapping)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AnnotationError("silver artifact has invalid turns") from error


def build_prompt(turns: Sequence[SilverTurn], *, duration_seconds: float) -> str:
    """Ask for emotion only, for every turn, keyed by a position mapping it must not move.

    The turns are supplied verbatim because the model has to know which stretch of audio
    each position refers to. Nothing in the reply is allowed to restate them: rows carry
    a position and a judgement, so a shifted or partial answer is detectable rather than
    silently plausible.
    """

    labels = [
        {
            "position": position,
            "speaker": turn.speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "transcript": turn.transcript,
        }
        for position, turn in enumerate(turns, start=1)
    ]
    return (
        "You are labelling EMOTION ONLY for a Korean two-party conversation that has "
        "already been segmented and transcribed by a human. The audio duration is "
        f"{duration_seconds:.3f} seconds. Do not re-segment, re-transcribe, or re-assign "
        "speakers: the supplied timing, speaker, and transcript are authoritative and "
        "your reply must not contain them.\n"
        f"Return JSON with `rows` holding EXACTLY {len(labels)} entries, one for every "
        f"supplied position from 1 through {len(labels)}, each exactly once, in order. "
        "Each row must be {position:int,emotion:string,confidence:number,rationale:string}. "
        f"`emotion` must be one of: {', '.join(EMOTION_LABELS)}. `confidence` is in [0,1]. "
        "`rationale` is one short sentence about the audio a reviewer can verify; do not "
        "quote the transcript back.\n"
        "Use 'uncertain' with a low confidence whenever you cannot tell, rather than "
        "guessing a specific emotion or omitting the row. A reply that omits, duplicates, "
        "or renumbers any position is unusable and will be discarded in full.\n"
        "Turns to label (private, for positional correspondence only):\n"
        + json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
    )


def _reject(
    rule: str,
    *,
    position: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> GeminiEmotionOverlayError:
    violation: dict[str, object] = {"rule": rule}
    if position is not None:
        violation["position"] = position
    violation.update(metadata or {})
    return GeminiEmotionOverlayError({"first_violation": violation})


def validate_rows(payload: object, *, source_turn_count: int) -> tuple[OverlayRow, ...]:
    """Accept only a complete, exactly-keyed set of rows; there is no partial success.

    The strictness is the point. Every rule here guards a way a positional overlay can be
    wrong without looking wrong: a missing row shifts nothing but leaves a turn silently
    unlabelled, a duplicate row makes "which one won" an implementation detail, and an
    out-of-range position means the model was not answering about this draft.

    There is deliberately no row-count check. In-range, non-duplicate, and complete
    together already force the count to match — too many rows must collide, too few must
    leave a gap — and each of those three reports which position went wrong, which a
    count comparison cannot.
    """

    if source_turn_count < 1:
        raise _reject("source.no_turns")
    if not isinstance(payload, Mapping):
        raise _reject("payload.not_object", metadata={"payload_type": type(payload).__name__})
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise _reject("rows.not_list", metadata={"rows_type": type(raw_rows).__name__})

    by_position: dict[int, OverlayRow] = {}
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise _reject("row.not_object", position=index + 1)
        try:
            position = int(row["position"])
            emotion = row["emotion"]
            confidence = float(row["confidence"])
            rationale = row["rationale"]
        except (KeyError, TypeError, ValueError):
            raise _reject("row.missing_or_invalid_field", position=index + 1) from None
        if not isinstance(emotion, str) or not isinstance(rationale, str):
            raise _reject("row.required_text_not_string", position=position)
        if position < 1 or position > source_turn_count:
            raise _reject(
                "row.position_out_of_range",
                position=position,
                metadata={"source_turn_count": source_turn_count},
            )
        if position in by_position:
            raise _reject("row.duplicate_position", position=position)
        if emotion not in EMOTION_LABELS:
            raise _reject("row.emotion_unknown", position=position)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise _reject("row.confidence_out_of_range", position=position)
        stripped = rationale.strip()
        if not stripped:
            raise _reject("row.rationale_empty", position=position)
        if len(stripped) > MAX_RATIONALE_CHARACTERS:
            raise _reject(
                "row.rationale_too_long",
                position=position,
                metadata={"limit": MAX_RATIONALE_CHARACTERS},
            )
        by_position[position] = OverlayRow(
            position=position,
            emotion=emotion,
            confidence=confidence,
            rationale=stripped,
        )

    missing = [
        position for position in range(1, source_turn_count + 1) if position not in by_position
    ]
    if missing:
        # Report how many, and only the first, so the diagnostic stays bounded. This is
        # also what a truncated reply looks like: the model answered about the first N
        # turns and stopped.
        raise _reject(
            "rows.missing_positions",
            position=missing[0],
            metadata={
                "missing_count": len(missing),
                "expected": source_turn_count,
                "received": len(raw_rows),
            },
        )
    return tuple(by_position[position] for position in range(1, source_turn_count + 1))


def apply_rows(turns: Sequence[SilverTurn], rows: Sequence[OverlayRow]) -> tuple[SilverTurn, ...]:
    """Overlay emotion fields onto the Silver turns, copying everything else verbatim."""

    if len(rows) != len(turns):
        raise _reject(
            "rows.count_mismatch",
            metadata={"expected": len(turns), "received": len(rows)},
        )
    return tuple(
        SilverTurn(
            start=turn.start,
            end=turn.end,
            speaker=turn.speaker,
            transcript=turn.transcript,
            emotion=row.emotion,
            emotion_rationale=row.rationale,
            confidence=row.confidence,
        )
        for turn, row in zip(turns, rows, strict=True)
    )


def write_candidate(
    *,
    annotation_root: Path,
    conversation_id: str,
    silver_record: Mapping[str, Any],
    rows: Sequence[OverlayRow],
    input_sha256: str,
    model: str,
    remote_file_deleted: bool,
    deletion_detail: str,
    call_counts: Mapping[str, int],
) -> tuple[Path, dict[str, Any]]:
    """Write the overlay beside Silver without touching Silver, gold, or the local candidate.

    The write is ``O_EXCL``: a second attempt is a second transmission, so it has to be a
    deliberate act by someone who first moved the existing artifact out of the way.
    """

    silver_digest = silver_record.get("content_sha256")
    if not isinstance(silver_digest, str) or len(silver_digest) != 64:
        raise AnnotationError("silver artifact has no usable content digest")
    turns = source_turns(silver_record)
    overlay_turns = apply_rows(turns, rows)
    content = {"turns": _turns_payload(overlay_turns)}
    candidate: dict[str, Any] = {
        "schema_version": "1",
        "kind": KIND,
        "review_state": "review_required",
        "promotable": False,
        "conversation_id": conversation_id,
        "created_at": _now(),
        "source": {
            "silver_content_sha256": silver_digest,
            "input_sha256": input_sha256,
            "model": model,
            "provider": provenance().name,
            "remote_audio_transmitted": True,
            "consent": "gemini_audio_transfer",
            "alignment": "timing, speaker, and transcript copied exactly from clean silver",
            "overlaid_fields": ["emotion", "emotion_rationale", "confidence"],
        },
        "remote_file": {"deleted": remote_file_deleted, "detail": deletion_detail},
        "call_counts": dict(call_counts),
        "validation": {
            "mode": "strict_positional",
            "salvage": "not_permitted",
            "row_count": len(rows),
            "dropped_row_count": 0,
        },
        "content": content,
        "content_sha256": _digest(content),
        "summary": summarise(overlay_turns),
        "contains_transcript": True,
        "handling": "private remote review candidate; never promote without human review",
    }
    directory = annotation_root / conversation_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILENAME
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise AnnotationError("gemini emotion overlay candidate already exists") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path, dict(candidate["summary"])


def load_candidate(
    *, annotation_root: Path, conversation_id: str, source_silver_content_sha256: str
) -> dict[str, Any] | None:
    """Load only the overlay belonging to this exact immutable Silver draft."""

    path = annotation_root / conversation_id / FILENAME
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OverlayCandidateError("candidate_unreadable") from error
    if not isinstance(record, Mapping) or record.get("kind") != KIND:
        raise OverlayCandidateError("candidate_unreadable")
    source = record.get("source")
    content = record.get("content")
    rows = content.get("turns") if isinstance(content, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or not isinstance(content, Mapping)
        or not isinstance(rows, list)
        or not rows
    ):
        raise OverlayCandidateError("candidate_unreadable")
    if source.get("silver_content_sha256") != source_silver_content_sha256:
        # A stale overlay is not an error, it is simply not this draft's material.
        return None
    if _digest(content) != record.get("content_sha256"):
        raise OverlayCandidateError("candidate_unreadable")
    if record.get("review_state") != "review_required" or record.get("promotable") is not False:
        raise OverlayCandidateError("candidate_unreadable")
    return dict(record)


__all__ = [
    "FILENAME",
    "KIND",
    "MAX_RATIONALE_CHARACTERS",
    "MODEL",
    "RESPONSE_SCHEMA",
    "GeminiEmotionOverlayError",
    "OverlayCandidateError",
    "OverlayRow",
    "apply_rows",
    "build_prompt",
    "load_candidate",
    "provenance",
    "source_turns",
    "validate_rows",
    "write_candidate",
]
