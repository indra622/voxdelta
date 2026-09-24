from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxdelta.annotation.alignment import (
    PREFIX_END_POSITION,
    PREFIX_START_POSITION,
    TARGET_START_POSITION,
    AlignmentError,
    validate_rows,
    write_proposal,
)
from voxdelta.providers.base import ProviderError


def _silver() -> dict[str, object]:
    turns = [
        {
            "start": float(index), "end": float(index) + 0.5,
            "speaker": "SPEAKER_00", "transcript": "private", "emotion": "neutral",
            "emotion_rationale": "private", "confidence": 0.5,
        }
        for index in range(12)
    ]
    return {"content_sha256": "a" * 64, "content": {"turns": turns}}


def test_alignment_keeps_only_target_suffix_and_salvages_bad_rows() -> None:
    rows, dropped = validate_rows(
        {"rows": [
            {"position": 9, "start": 1, "end": 2, "confidence": 0.8},
            {"position": 10, "start": 2, "end": 3, "confidence": 0.8},
            {"position": 11, "start": 4, "end": 4, "confidence": 0.8},
            {"position": 12, "start": 5, "end": 6, "confidence": 0.9},
        ]},
        source_turn_count=12,
        duration_seconds=20,
    )
    assert [row["position"] for row in rows] == [10, 12]
    assert dropped == {"row.invalid_interval": 1, "row.position_out_of_target": 1}


def test_alignment_refuses_when_no_row_is_usable() -> None:
    with pytest.raises(ProviderError):
        validate_rows(
            {"rows": [{"position": 9, "start": 1, "end": 2, "confidence": 0.5}]},
            source_turn_count=12,
            duration_seconds=20,
        )


def test_prefix_alignment_accepts_only_positions_one_through_nine() -> None:
    rows, dropped = validate_rows(
        {"rows": [
            {"position": 1, "start": 0, "end": 1, "confidence": 0.8},
            {"position": 9, "start": 2, "end": 3, "confidence": 0.9},
            {"position": 10, "start": 4, "end": 5, "confidence": 0.7},
        ]},
        source_turn_count=12,
        duration_seconds=20,
        target_start_position=PREFIX_START_POSITION,
        target_end_position=PREFIX_END_POSITION,
    )
    assert [row["position"] for row in rows] == [1, 9]
    assert dropped == {"row.position_out_of_target": 1}


def test_proposal_is_isolated_and_never_overwrites_silver(tmp_path: Path) -> None:
    source = _silver()
    root = tmp_path / "annotations"
    directory = root / "A6000_S0005_0"
    directory.mkdir(parents=True)
    silver = directory / "silver.json"
    silver.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    before = silver.read_bytes()
    path, digest = write_proposal(
        root=root, conversation_id="A6000_S0005_0", source_record=source,
        rows=({"position": TARGET_START_POSITION, "start": 2.0, "end": 3.0, "confidence": 0.8},),
        dropped_by_rule={}, model="gemini", input_sha256="b" * 64,
        remote_file_deleted=True, deletion_detail="deleted",
        call_counts={"uploads": 1, "interactions": 1, "deletes": 1},
    )
    assert path.name == "alignment-proposal-10.json"
    assert len(digest) == 64
    assert silver.read_bytes() == before
    proposal = json.loads(path.read_text(encoding="utf-8"))
    assert proposal["source_silver_content_sha256"] == "a" * 64
    assert proposal["content"]["target_end_position"] == 12
    assert proposal["review_state"] == "review_required"
    assert proposal["contains_transcript"] is False
    with pytest.raises(AlignmentError):
        write_proposal(
            root=root, conversation_id="A6000_S0005_0", source_record=source,
            rows=({"position": 10, "start": 2.0, "end": 3.0, "confidence": 0.8},),
            dropped_by_rule={}, model="gemini", input_sha256="b" * 64,
            remote_file_deleted=True, deletion_detail="deleted",
            call_counts={"uploads": 1, "interactions": 1, "deletes": 1},
        )


def test_prefix_proposal_uses_a_separate_immutable_filename(tmp_path: Path) -> None:
    source = _silver()
    root = tmp_path / "annotations"
    path, _ = write_proposal(
        root=root,
        conversation_id="A6000_S0005_0",
        source_record=source,
        rows=({"position": 1, "start": 0.0, "end": 1.0, "confidence": 0.8},),
        dropped_by_rule={},
        model="gemini",
        input_sha256="b" * 64,
        remote_file_deleted=True,
        deletion_detail="deleted",
        call_counts={"uploads": 1, "interactions": 1, "deletes": 1},
        target_start_position=PREFIX_START_POSITION,
        target_end_position=PREFIX_END_POSITION,
    )
    assert path.name == "alignment-proposal-1-9.json"
    proposal = json.loads(path.read_text(encoding="utf-8"))
    assert proposal["content"]["target_start_position"] == 1
    assert proposal["content"]["target_end_position"] == 9
