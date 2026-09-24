from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxdelta.annotation.clean_reference_silver import (
    ARCHIVE_FILENAME,
    MODEL_NAME,
    CleanReferenceSilverError,
    regenerate_from_reference,
)
from voxdelta.annotation.gemini_silver import SilverAnnotation, SilverTurn
from voxdelta.annotation.store import read_silver, write_silver

CONVERSATION = "A6000_S0005_0"


def write_original(root: Path) -> Path:
    annotation = SilverAnnotation(
        speakers=("SPEAKER_00", "SPEAKER_01"),
        turns=(
            SilverTurn(
                start=0.0,
                end=10.0,
                speaker="SPEAKER_00",
                transcript="private original",
                emotion="neutral",
                emotion_rationale="private original",
                confidence=0.5,
            ),
        ),
        notes="private original",
    )
    path, _digest, _summary = write_silver(
        annotation,
        root=root,
        conversation_id=CONVERSATION,
        input_sha256="a" * 64,
        model="gemini-test",
        prompt="private",
        config={},
        remote_file_deleted=True,
        deletion_detail="deleted",
        call_counts={"uploads": 1, "interactions": 1, "deletes": 1},
    )
    return path


def write_reference(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{CONVERSATION}.json"
    path.write_text(
        json.dumps(
            {
                "conversation_id": CONVERSATION,
                "duration_seconds": 20.0,
                "speakers": ["G5999", "G6000"],
                "turns": [
                    {"start": 0.0, "end": 2.0, "speaker": "G5999", "transcript": "one"},
                    {"start": 1.5, "end": 3.0, "speaker": "G6000", "transcript": "two"},
                    {"start": 3.1, "end": 4.0, "speaker": "G5999", "transcript": "three"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_regenerate_archives_original_and_publishes_complete_reference(tmp_path: Path) -> None:
    annotation_root = tmp_path / "annotations"
    reference_root = tmp_path / "reference"
    source = write_original(annotation_root)
    before = source.read_bytes()
    reference = write_reference(reference_root)
    reference_before = reference.read_bytes()

    receipt = regenerate_from_reference(
        annotation_root,
        reference_root,
        conversation_id=CONVERSATION,
    )

    archived = annotation_root / CONVERSATION / ARCHIVE_FILENAME
    replacement = read_silver(annotation_root / CONVERSATION / "silver.json")
    assert receipt.archived_path == archived
    assert archived.read_bytes() == before
    assert reference.read_bytes() == reference_before
    assert replacement["source"]["model"] == MODEL_NAME
    assert replacement["source"]["remote_audio_transmitted"] is False
    assert replacement["source"]["input_sha256"] == "a" * 64
    assert replacement["validation"]["mode"] == "reference_normalized"
    assert replacement["validation"]["dropped_turn_count"] == 0
    assert [row["transcript"] for row in replacement["content"]["turns"]] == [
        "one",
        "two",
        "three",
    ]
    assert {row["emotion"] for row in replacement["content"]["turns"]} == {"uncertain"}


def test_regenerate_refuses_to_replace_gold_or_overwrite_archive(tmp_path: Path) -> None:
    annotation_root = tmp_path / "annotations"
    reference_root = tmp_path / "reference"
    source = write_original(annotation_root)
    write_reference(reference_root)
    (source.parent / "gold.json").write_text("{}", encoding="utf-8")

    with pytest.raises(CleanReferenceSilverError, match="Gold"):
        regenerate_from_reference(annotation_root, reference_root, conversation_id=CONVERSATION)

    (source.parent / "gold.json").unlink()
    (source.parent / ARCHIVE_FILENAME).write_text("keep", encoding="utf-8")
    with pytest.raises(CleanReferenceSilverError, match="already archived"):
        regenerate_from_reference(annotation_root, reference_root, conversation_id=CONVERSATION)
