from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxdelta.annotation.reference_candidate import ReferenceCandidateError, load_candidate

CONVERSATION = "A6000_S0005_0"


def write_reference(root: Path, *, turns: list[dict[str, object]] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{CONVERSATION}.json"
    path.write_text(
        json.dumps(
            {
                "conversation_id": CONVERSATION,
                "duration_seconds": 20.0,
                "speakers": ["G5999", "G6000"],
                "turns": turns
                if turns is not None
                else [
                    {"start": 1.0, "end": 2.0, "speaker": "G5999", "transcript": "private"},
                    {"start": 2.0, "end": 3.0, "speaker": "G6000", "transcript": "also private"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_reference_candidate_is_read_only_and_marks_emotion_uncertain(tmp_path: Path) -> None:
    root = tmp_path / "reference"
    source = write_reference(root)
    before = source.read_bytes()

    candidate = load_candidate(
        root,
        conversation_id=CONVERSATION,
        source_silver_content_sha256="a" * 64,
    )

    assert candidate is not None
    assert candidate.reference_turn_count == 2
    assert candidate.speakers == ("G5999", "G6000")
    assert [turn.emotion for turn in candidate.turns] == ["uncertain", "uncertain"]
    assert [turn.confidence for turn in candidate.turns] == [0.0, 0.0]
    assert source.read_bytes() == before


def test_reference_candidate_is_absent_when_this_conversation_has_no_local_reference(
    tmp_path: Path,
) -> None:
    assert (
        load_candidate(
            tmp_path,
            conversation_id=CONVERSATION,
            source_silver_content_sha256="a" * 64,
        )
        is None
    )


def test_reference_candidate_refuses_a_bad_interval(tmp_path: Path) -> None:
    root = tmp_path / "reference"
    write_reference(
        root,
        turns=[{"start": 5.0, "end": 5.0, "speaker": "G5999", "transcript": "private"}],
    )

    with pytest.raises(ReferenceCandidateError):
        load_candidate(root, conversation_id=CONVERSATION, source_silver_content_sha256="a" * 64)
