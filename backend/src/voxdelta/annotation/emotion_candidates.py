"""Reference-aligned, local emotion overlay candidates for silver review.

Candidates are deliberately separate from ``silver.json``: their only job is to offer a
reviewer another emotion hypothesis while preserving the human-reference timing,
speaker, and transcript.  They can never be promoted directly to gold.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voxdelta.annotation import audio
from voxdelta.annotation.gemini_silver import SilverTurn
from voxdelta.annotation.store import AnnotationError, _turns_payload, read_silver, summarise
from voxdelta.providers.base import EmotionProvider, ProviderError

KIND = "emotion-overlay-candidate"
MODEL = "calibrated-xls-r-local"
FILENAME = "emotion-candidate.calibrated-xlsr.json"


class EmotionCandidateError(RuntimeError):
    """A private candidate could not be used safely."""


def _digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _turns(record: Mapping[str, Any]) -> tuple[SilverTurn, ...]:
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


def _uncertain(turn: SilverTurn) -> SilverTurn:
    return SilverTurn(
        start=turn.start,
        end=turn.end,
        speaker=turn.speaker,
        transcript=turn.transcript,
        emotion="uncertain",
        emotion_rationale="Local emotion model abstained or could not analyze this short segment.",
        confidence=0.0,
    )


def _overlay(
    turn: SilverTurn,
    *,
    position: int,
    provider: EmotionProvider,
    source: audio.SourceAudio,
) -> SilverTurn:
    try:
        plan = audio.plan_clip(source, start=turn.start, end=turn.end)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            clip_path = Path(handle.name)
            for chunk in audio.clip_bytes(plan):
                handle.write(chunk)
        try:
            result = provider.analyze(f"emotion-overlay-{position}", clip_path, turn.transcript)
        finally:
            clip_path.unlink(missing_ok=True)
    except (ProviderError, OSError, ValueError):
        return _uncertain(turn)
    if result.calibration is not None and result.calibration.abstained:
        return _uncertain(turn)
    label = max(result.probabilities.items(), key=lambda item: item[1])[0]
    return SilverTurn(
        start=turn.start,
        end=turn.end,
        speaker=turn.speaker,
        transcript=turn.transcript,
        emotion=label,
        emotion_rationale="Local calibrated XLS-R emotion overlay; reviewer confirmation required.",
        confidence=result.confidence,
    )


def generate_local_candidate(
    *,
    annotation_root: Path,
    audio_root: Path,
    conversation_id: str,
    provider: EmotionProvider,
) -> tuple[Path, dict[str, Any]]:
    """Generate a one-time local candidate without changing silver or gold."""

    directory = annotation_root / conversation_id
    silver_path = directory / "silver.json"
    record = read_silver(silver_path)
    turns = _turns(record)
    source = record.get("source")
    expected_sha = source.get("input_sha256") if isinstance(source, Mapping) else ""
    if not isinstance(expected_sha, str):
        raise AnnotationError("silver artifact has no source digest")
    recording = audio.resolve_source(audio_root, conversation_id, expected_sha256=expected_sha)
    candidate_turns = tuple(
        _overlay(turn, position=position, provider=provider, source=recording)
        for position, turn in enumerate(turns, start=1)
    )
    content = {"turns": _turns_payload(candidate_turns)}
    candidate: dict[str, Any] = {
        "schema_version": "1",
        "kind": KIND,
        "review_state": "review_required",
        "promotable": False,
        "conversation_id": conversation_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {
            "silver_content_sha256": record.get("content_sha256", ""),
            "model": MODEL,
            "remote_audio_transmitted": False,
            "alignment": "copied exactly from clean silver",
        },
        "content": content,
        "content_sha256": _digest(content),
        "summary": summarise(candidate_turns),
        "contains_transcript": True,
        "handling": "private review candidate; never promote without human review",
    }
    path = directory / FILENAME
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise AnnotationError("local emotion candidate already exists") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path, candidate["summary"]


def load_local_candidate(
    *, annotation_root: Path, conversation_id: str, source_silver_content_sha256: str
) -> dict[str, Any] | None:
    """Load only the candidate belonging to this exact immutable silver draft."""

    path = annotation_root / conversation_id / FILENAME
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EmotionCandidateError("candidate_unreadable") from error
    if not isinstance(record, Mapping) or record.get("kind") != KIND:
        raise EmotionCandidateError("candidate_unreadable")
    source = record.get("source")
    content = record.get("content")
    rows = content.get("turns") if isinstance(content, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or not isinstance(content, Mapping)
        or source.get("silver_content_sha256") != source_silver_content_sha256
        or not isinstance(rows, list)
        or not rows
    ):
        return None
    if _digest(content) != record.get("content_sha256"):
        raise EmotionCandidateError("candidate_unreadable")
    return dict(record)


__all__ = [
    "FILENAME",
    "KIND",
    "MODEL",
    "EmotionCandidateError",
    "generate_local_candidate",
    "load_local_candidate",
]
