from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from voxdelta.evaluation.aihub_fields import resolve_canonical_field
from voxdelta.evaluation.manifest import DatasetItem, load_manifest, validate_disjoint_splits

BACKEND = Path(__file__).resolve().parents[2]


def _item(
    item_id: str,
    *,
    call_id: str = "call-1",
    speaker_id: str = "speaker-1",
    split: str = "train",
) -> DatasetItem:
    return DatasetItem(
        id=item_id,
        call_id=call_id,
        speaker_id=speaker_id,
        audio_path=f"/{item_id}.wav",
        transcript="fixture transcript",
        split=split,
        sha256="0" * 64,
    )


def _write_pair(
    root: Path,
    stem: str,
    metadata: dict[str, object],
    *,
    audio: bytes = b"fixture-audio",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    audio_path = root / f"{stem}.wav"
    audio_path.write_bytes(audio)
    (root / f"{stem}.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    return audio_path


def _run_builder(
    consultation: Path, emotion: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/build_aihub_manifests.py",
            "--consultation-root",
            str(consultation),
            "--emotion-root",
            str(emotion),
            "--output-root",
            str(output),
        ],
        cwd=BACKEND,
        text=True,
        capture_output=True,
        check=False,
    )


def test_split_validator_rejects_same_speaker_in_train_and_test() -> None:
    train = [_item("a", call_id="c1", speaker_id="s1", split="train")]
    test = [_item("b", call_id="c2", speaker_id="s1", split="test")]

    with pytest.raises(ValueError, match="speaker leakage: s1"):
        validate_disjoint_splits(train + test)


def test_split_validator_rejects_same_call_in_different_splits() -> None:
    train = [_item("a", call_id="c1", speaker_id="s1", split="train")]
    validation = [_item("b", call_id="c1", speaker_id="s2", split="validation")]

    with pytest.raises(ValueError, match="call leakage: c1"):
        validate_disjoint_splits(train + validation)


def test_load_manifest_parses_jsonl_and_sorts_by_item_id(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    lines = [_item("z").model_dump_json(), _item("a").model_dump_json()]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert [item.id for item in load_manifest(manifest)] == ["a", "z"]


def test_load_manifest_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real.jsonl"
    target.write_text(_item("a").model_dump_json() + "\n", encoding="utf-8")
    link = tmp_path / "linked.jsonl"
    try:
        link.symlink_to(target)
    except OSError as error:  # pragma: no cover - platform capability guard
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="symlink"):
        load_manifest(link)


def test_field_resolution_is_depth_first_and_rejects_differing_matches() -> None:
    metadata = {"wrapper": {"id": "first"}, "wav_id": "second"}

    with pytest.raises(ValueError, match="multiple differing matches for id"):
        resolve_canonical_field(metadata, "id")


def test_missing_field_diagnostic_lists_available_key_paths() -> None:
    metadata = {"outer": {"known": "value"}, "other": 1}

    with pytest.raises(ValueError, match=r"available key paths: outer, outer\.known, other"):
        resolve_canonical_field(metadata, "transcript")


def test_builder_writes_absolute_hashed_sorted_jsonl_manifests(tmp_path: Path) -> None:
    consultation = tmp_path / "consultation"
    emotion = tmp_path / "emotion"
    output = tmp_path / "manifests"
    consultation_audio = _write_pair(
        consultation / "nested",
        "z-item",
        {
            "payload": {
                "wav_id": "z",
                "conversation_id": "consultation-call",
                "speaker": "consultation-speaker",
                "sentence": "consultation text",
            }
        },
        audio=b"consultation-audio",
    )
    emotion_audio = _write_pair(
        emotion,
        "a-item",
        {
            "audio_id": "a",
            "dialogue_id": "emotion-call",
            "talker_id": "emotion-speaker",
            "발화문": "emotion text",
            "감정": "anger",
        },
        audio=b"emotion-audio",
    )

    result = _run_builder(consultation, emotion, output)

    assert result.returncode == 0, result.stderr

    output_files = [output / "train.jsonl", output / "validation.jsonl", output / "test.jsonl"]
    assert all(path.is_file() for path in output_files)
    items = [item for path in output_files for item in load_manifest(path)]
    assert sorted(item.id for item in items) == ["a", "z"]
    assert all(
        [item.id for item in load_manifest(path)] == sorted(item.id for item in load_manifest(path))
        for path in output_files
    )
    by_id = {item.id: item for item in items}
    assert by_id["a"].audio_path == str(emotion_audio.resolve())
    assert by_id["a"].sha256 == hashlib.sha256(b"emotion-audio").hexdigest()
    assert by_id["a"].emotion == "anger"
    assert by_id["z"].audio_path == str(consultation_audio.resolve())
    assert by_id["z"].sha256 == hashlib.sha256(b"consultation-audio").hexdigest()
    assert by_id["z"].emotion is None
    validate_disjoint_splits(items)


@pytest.mark.parametrize(
    ("missing_suffix", "match"),
    [(".json", "missing audio for metadata"), (".wav", "missing metadata for audio")],
)
def test_builder_rejects_unpaired_audio_or_metadata(
    tmp_path: Path, missing_suffix: str, match: str
) -> None:
    consultation = tmp_path / "consultation"
    emotion = tmp_path / "emotion"
    output = tmp_path / "manifests"
    consultation.mkdir()
    emotion.mkdir()
    (consultation / f"orphan{missing_suffix}").write_text("{}", encoding="utf-8")

    result = _run_builder(consultation, emotion, output)

    assert result.returncode == 2
    assert match in result.stderr


def test_builder_rejects_duplicate_ids(tmp_path: Path) -> None:
    consultation = tmp_path / "consultation"
    emotion = tmp_path / "emotion"
    output = tmp_path / "manifests"
    base = {
        "id": "duplicate",
        "call_id": "call-1",
        "speaker_id": "speaker-1",
        "transcript": "text",
    }
    _write_pair(consultation / "one", "first", base)
    _write_pair(consultation / "two", "second", base | {"call_id": "call-2"})
    emotion.mkdir()

    result = _run_builder(consultation, emotion, output)

    assert result.returncode == 2
    assert "duplicate item id: duplicate" in result.stderr


def test_builder_rejects_emotion_outside_seven_labels(tmp_path: Path) -> None:
    consultation = tmp_path / "consultation"
    emotion = tmp_path / "emotion"
    output = tmp_path / "manifests"
    consultation.mkdir()
    _write_pair(
        emotion,
        "bad-emotion",
        {
            "id": "bad",
            "call_id": "call-1",
            "speaker_id": "speaker-1",
            "transcript": "text",
            "emotion": "annoyed",
        },
    )

    result = _run_builder(consultation, emotion, output)

    assert result.returncode == 2
    assert "unsupported emotion label" in result.stderr


def test_builder_rejects_symlinked_input_file(tmp_path: Path) -> None:
    consultation = tmp_path / "consultation"
    emotion = tmp_path / "emotion"
    output = tmp_path / "manifests"
    consultation.mkdir()
    emotion.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    linked = consultation / "linked.wav"
    try:
        linked.symlink_to(outside)
    except OSError as error:  # pragma: no cover - platform capability guard
        pytest.skip(f"symlinks unavailable: {error}")
    (consultation / "linked.json").write_text("{}", encoding="utf-8")

    result = _run_builder(consultation, emotion, output)

    assert result.returncode == 2
    assert "symlink" in result.stderr
