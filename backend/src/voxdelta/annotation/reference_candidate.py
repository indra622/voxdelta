"""Read-only KCSC reference candidates for repairing a drifted Silver draft.

This module deliberately creates no new Silver or Gold artifact.  KCSC's human
reference gives a reviewer a trustworthy *candidate* for speaker, transcript, and
boundaries when a model draft has coupled the wrong text to the wrong audio.  Applying
it remains a browser-local review action; emotion is explicitly ``uncertain`` because
the source reference contains no emotion labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from voxdelta.annotation.gemini_silver import SilverTurn

MAX_TURNS = 5_000
MAX_SPEAKER_CHARACTERS = 64
MAX_TRANSCRIPT_CHARACTERS = 4_000


class ReferenceCandidateError(RuntimeError):
    """The local KCSC reference cannot safely become a review candidate."""


@dataclass(frozen=True, slots=True)
class ReferenceResegmentationCandidate:
    source_silver_content_sha256: str
    source_reference_sha256: str
    reference_turn_count: int
    speakers: tuple[str, ...]
    notes: str
    turns: tuple[SilverTurn, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject(message: str) -> ReferenceCandidateError:
    return ReferenceCandidateError(message)


def load_candidate(
    reference_root: Path,
    *,
    conversation_id: str,
    source_silver_content_sha256: str,
) -> ReferenceResegmentationCandidate | None:
    """Return a local KCSC reference candidate, or ``None`` when none exists.

    Reference text never enters a summary/logging surface.  This is called only by the
    capability-fenced review route, which already serves the reviewer's transcript.
    """

    path = reference_root / f"{conversation_id}.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise _reject("local reference is unreadable") from None
    if not isinstance(raw, Mapping) or raw.get("conversation_id") != conversation_id:
        raise _reject("local reference does not match the requested conversation")
    duration = raw.get("duration_seconds")
    rows = raw.get("turns")
    speakers = raw.get("speakers")
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0
        or not isinstance(rows, list)
        or not rows
        or len(rows) > MAX_TURNS
        or not isinstance(speakers, list)
        or not speakers
        or not all(isinstance(speaker, str) and speaker.strip() for speaker in speakers)
    ):
        raise _reject("local reference has an invalid schema")

    allowed_speakers = tuple(dict.fromkeys(speakers))
    if any(len(speaker) > MAX_SPEAKER_CHARACTERS for speaker in allowed_speakers):
        raise _reject("local reference has an invalid speaker")
    turns: list[SilverTurn] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise _reject("local reference has an invalid turn")
        try:
            start = float(row["start"])
            end = float(row["end"])
            speaker = row["speaker"]
            transcript = row["transcript"]
        except (KeyError, TypeError, ValueError):
            raise _reject("local reference has an invalid turn") from None
        if (
            not isinstance(speaker, str)
            or speaker not in allowed_speakers
            or not isinstance(transcript, str)
            or not transcript.strip()
            or len(transcript) > MAX_TRANSCRIPT_CHARACTERS
            or not all(math.isfinite(value) for value in (start, end))
            or start < 0
            or end <= start
            or end > float(duration)
        ):
            raise _reject("local reference has an invalid turn")
        turns.append(
            SilverTurn(
                start=round(start, 6),
                end=round(end, 6),
                speaker=speaker,
                transcript=transcript,
                emotion="uncertain",
                emotion_rationale=(
                    "KCSC reference resegmentation candidate; emotion requires reviewer judgment."
                ),
                confidence=0.0,
            )
        )
    return ReferenceResegmentationCandidate(
        source_silver_content_sha256=source_silver_content_sha256,
        source_reference_sha256=_sha256(path),
        reference_turn_count=len(turns),
        speakers=allowed_speakers,
        notes=(
            "KCSC source reference candidate: human speaker, transcript, and timing are supplied; "
            "emotion is intentionally uncertain and must be reviewed."
        ),
        turns=tuple(turns),
    )


__all__ = ["ReferenceCandidateError", "ReferenceResegmentationCandidate", "load_candidate"]
