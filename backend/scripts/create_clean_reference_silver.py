"""Create a fresh, review-required Clean Silver from one local KCSC reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from voxdelta.annotation.reference_candidate import ReferenceCandidateError, load_candidate
from voxdelta.annotation.review import CONVERSATION_ID


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(content: object) -> str:
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--conversation-id", required=True)
    args = parser.parse_args()

    conversation_id = args.conversation_id
    if not CONVERSATION_ID.fullmatch(conversation_id):
        print(json.dumps({"status": "rejected", "reason": "invalid_conversation_id"}))
        return 2
    audio_path = args.audio_root / f"{conversation_id}.wav"
    target = args.annotation_root / conversation_id / "silver.json"
    if not audio_path.is_file():
        print(json.dumps({"status": "rejected", "reason": "audio_missing"}))
        return 2
    if target.exists():
        print(json.dumps({"status": "rejected", "reason": "silver_already_exists"}))
        return 2
    try:
        candidate = load_candidate(
            args.reference_root,
            conversation_id=conversation_id,
            source_silver_content_sha256="",
        )
    except ReferenceCandidateError:
        print(json.dumps({"status": "rejected", "reason": "reference_unreadable"}))
        return 2
    if candidate is None:
        print(json.dumps({"status": "rejected", "reason": "reference_missing"}))
        return 2

    content = {
        "turns": [
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
        ],
        "speakers": list(candidate.speakers),
        "notes": (
            "Clean Silver created from the complete KCSC human reference timeline. "
            "Emotion requires reviewer judgment."
        ),
    }
    record = {
        "schema_version": "1",
        "kind": "silver-annotation",
        "review_state": "review_required",
        "promotable": False,
        "conversation_id": conversation_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {
            "input_sha256": _sha256(audio_path),
            "model": "kcsc-human-reference",
            "prompt": "not_applicable_local_human_reference",
            "config": {
                "reference_sha256": candidate.source_reference_sha256,
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
        "content_sha256": _content_digest(content),
        "contains_transcript": True,
        "handling": "private local reference artifact; emotion requires review before Gold",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print(json.dumps({"status": "rejected", "reason": "silver_already_exists"}))
        return 2
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "created",
                "conversation_id": conversation_id,
                "turn_count": len(candidate.turns),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
