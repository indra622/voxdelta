"""Derive immutable evaluation cases from a reviewer-directed split point.

The source recording and its reviewed Gold remain untouched.  This script cuts a PCM WAV
at an exact frame boundary, re-bases only the corresponding reviewed turns, and writes a
separate manifest with hashes.  It intentionally contains no model invocation or network
access.
"""

from __future__ import annotations

import hashlib
import json
import os
import wave
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voxdelta.annotation.store import verify_gold

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ID = "USER_EVAL_20260911_02"
SPLIT_SECONDS = 34.0
SOURCE_AUDIO = (
    ROOT
    / "data/derived/user-provided/2026-09-11-korean-evaluation-audio"
    / f"{SOURCE_ID}.wav"
)
SOURCE_GOLD = ROOT / "data/annotations" / SOURCE_ID / "gold.json"
OUTPUT_ROOT = (
    ROOT / "data/derived/user-provided/2026-09-11-korean-evaluation-audio/gold-splits-v1"
)
ANNOTATIONS = ROOT / "data/annotations"


class SplitError(RuntimeError):
    """Raised when the requested split could alter reviewed Gold semantics."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_digest(content: dict[str, Any]) -> str:
    raw = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise SplitError(f"refusing to overwrite derived artifact: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _copy_wav_interval(source: Path, destination: Path, start: float, end: float) -> float:
    with wave.open(str(source), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise SplitError("source must be mono PCM16 WAV")
        frame_rate = reader.getframerate()
        total_frames = reader.getnframes()
        start_frame = round(start * frame_rate)
        end_frame = round(end * frame_rate)
        if not 0 <= start_frame < end_frame <= total_frames:
            raise SplitError("split interval is outside source WAV")
        reader.setpos(start_frame)
        frames = reader.readframes(end_frame - start_frame)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise SplitError(f"refusing to overwrite derived WAV: {destination}") from error
        with os.fdopen(descriptor, "wb") as handle, wave.open(handle, "wb") as writer:
            writer.setparams(reader.getparams())
            writer.writeframes(frames)
    return (end_frame - start_frame) / frame_rate


def _derived_gold(
    original: dict[str, Any], *, conversation_id: str, start: float, end: float
) -> dict[str, Any]:
    raw_turns = original.get("content", {}).get("turns")
    if not isinstance(raw_turns, list):
        raise SplitError("source Gold has no turns")
    if any(
        not isinstance(turn, dict)
        or not isinstance(turn.get("start"), int | float)
        or not isinstance(turn.get("end"), int | float)
        for turn in raw_turns
    ):
        raise SplitError("source Gold has malformed turns")
    crossing = [turn for turn in raw_turns if float(turn["start"]) < start < float(turn["end"])]
    if crossing:
        raise SplitError("requested split crosses a reviewed Gold turn")
    turns = []
    for raw_turn in raw_turns:
        if float(raw_turn["start"]) < start or float(raw_turn["end"]) > end:
            continue
        turn = deepcopy(raw_turn)
        turn["start"] = round(float(turn["start"]) - start, 6)
        turn["end"] = round(float(turn["end"]) - start, 6)
        turns.append(turn)
    if not turns:
        raise SplitError("split produced no reviewed Gold turns")
    content = {"turns": turns}
    source_digest = str(original.get("content_sha256", ""))
    record = {
        **{
            key: value
            for key, value in original.items()
            if key
            not in {"content", "content_sha256", "conversation_id", "review_note", "reviewed_at"}
        },
        "conversation_id": conversation_id,
        "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "review_note": (
            f"Reviewer-directed derived case from {SOURCE_ID} [{start:.3f}, {end:.3f}) "
            f"seconds; source reviewed Gold SHA-256 {source_digest}."
        ),
        "content": content,
        "content_sha256": _content_digest(content),
        "derived_from": {
            "conversation_id": SOURCE_ID,
            "gold_content_sha256": source_digest,
            "start_seconds": start,
            "end_seconds": end,
        },
        "handling": "private artifact; never copy into logs or benchmark output",
    }
    return record


def main() -> int:
    source_gold = verify_gold(SOURCE_GOLD)
    if not SOURCE_AUDIO.is_file():
        raise SplitError("source WAV is missing")
    with wave.open(str(SOURCE_AUDIO), "rb") as reader:
        duration = reader.getnframes() / reader.getframerate()
    cases = (
        ("USER_EVAL_20260911_02A", 0.0, SPLIT_SECONDS),
        ("USER_EVAL_20260911_02B", SPLIT_SECONDS, duration),
    )
    entries: list[dict[str, Any]] = []
    for conversation_id, start, end in cases:
        wav_path = OUTPUT_ROOT / f"{conversation_id}.wav"
        derived_duration = _copy_wav_interval(SOURCE_AUDIO, wav_path, start, end)
        gold = _derived_gold(source_gold, conversation_id=conversation_id, start=start, end=end)
        _write_exclusive(ANNOTATIONS / conversation_id / "gold.json", gold)
        entries.append(
            {
                "id": conversation_id,
                "file": wav_path.name,
                "sha256": _sha256(wav_path),
                "duration_seconds": round(derived_duration, 6),
                "source": {
                    "conversation_id": SOURCE_ID,
                    "audio_sha256": _sha256(SOURCE_AUDIO),
                    "start_seconds": start,
                    "end_seconds": end,
                    "gold_content_sha256": source_gold["content_sha256"],
                },
            }
        )
    _write_exclusive(
        OUTPUT_ROOT / "manifest.json",
        {
            "schema_version": "1",
            "dataset_id": "user-provided-gold-splits-v1",
            "derivation": (
                "reviewer-directed exact-frame WAV split; original recording and Gold preserved"
            ),
            "items": entries,
        },
    )
    print(f"manifest: {OUTPUT_ROOT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
