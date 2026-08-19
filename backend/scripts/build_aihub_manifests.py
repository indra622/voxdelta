"""Build deterministic, local-only AI Hub evaluation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Never, cast

from voxdelta.domain.models import EmotionLabel
from voxdelta.evaluation.aihub_fields import resolve_canonical_field
from voxdelta.evaluation.manifest import (
    DatasetItem,
    DatasetSource,
    DatasetSplit,
    read_trusted_regular_file,
    validate_disjoint_splits,
)

_AUDIO_SUFFIXES = frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"})
_SPLITS: tuple[DatasetSplit, ...] = ("train", "validation", "test")
_EMOTION_MAP: dict[str, EmotionLabel] = {
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happiness",
    "happy": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "공포": "fear",
    "기쁨": "happiness",
    "놀람": "surprise",
    "두려움": "fear",
    "무감정": "neutral",
    "분노": "anger",
    "슬픔": "sadness",
    "중립": "neutral",
    "행복": "happiness",
    "혐오": "disgust",
    "화남": "anger",
}


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked path is not allowed: {path}")


def _input_root(path: Path) -> Path:
    absolute = _absolute_path(path)
    _reject_symlink_components(absolute)
    if not absolute.is_dir():
        raise ValueError(f"input root is not a directory: {absolute}")
    return absolute


def _output_root(path: Path) -> Path:
    absolute = _absolute_path(path)
    _reject_symlink_components(absolute)
    absolute.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(absolute)
    if not absolute.is_dir():
        raise ValueError(f"output root is not a directory: {absolute}")
    return absolute


def _raise_walk_error(error: OSError) -> Never:
    raise error


def _discover_pairs(root: Path) -> list[tuple[Path, Path]]:
    audio_by_key: dict[tuple[Path, str], Path] = {}
    metadata_by_key: dict[tuple[Path, str], Path] = {}
    for directory_name, directory_names, file_names in os.walk(
        root, onerror=_raise_walk_error, followlinks=False
    ):
        directory = Path(directory_name)
        for name in directory_names:
            candidate = directory / name
            if candidate.is_symlink():
                raise ValueError(f"symlinked input is not allowed: {candidate}")
        for name in file_names:
            candidate = directory / name
            if candidate.is_symlink():
                raise ValueError(f"symlinked input is not allowed: {candidate}")
            suffix = candidate.suffix.lower()
            key = (directory, candidate.stem)
            if suffix in _AUDIO_SUFFIXES:
                if key in audio_by_key:
                    raise ValueError(f"multiple audio files share metadata stem: {candidate.stem}")
                audio_by_key[key] = candidate
            elif suffix == ".json":
                if key in metadata_by_key:
                    raise ValueError(f"multiple metadata files share audio stem: {candidate.stem}")
                metadata_by_key[key] = candidate

    for key, audio in audio_by_key.items():
        if key not in metadata_by_key:
            raise ValueError(f"missing metadata for audio: {audio.name}")
    for key, metadata in metadata_by_key.items():
        if key not in audio_by_key:
            raise ValueError(f"missing audio for metadata: {metadata.name}")

    return [
        (audio_by_key[key], metadata_by_key[key])
        for key in sorted(audio_by_key, key=lambda item: (str(item[0]), item[1]))
    ]


def _load_metadata(path: Path) -> dict[str, object]:
    raw = read_trusted_regular_file(path)
    try:
        payload: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON metadata: {path.name}") from error
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError(f"metadata must be a JSON object: {path.name}")
    return cast(dict[str, object], payload)


def _normalize_emotion(raw: str) -> EmotionLabel:
    normalized = raw.strip().casefold()
    try:
        return _EMOTION_MAP[normalized]
    except KeyError as error:
        raise ValueError("unsupported emotion label") from error


def _split_for_call(call_id: str) -> DatasetSplit:
    bucket = int(hashlib.sha256(call_id.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _build_item(audio: Path, metadata: Path, *, source: DatasetSource) -> DatasetItem:
    payload = _load_metadata(metadata)
    item_id = resolve_canonical_field(payload, "id")
    call_id = resolve_canonical_field(payload, "call_id")
    speaker_id = resolve_canonical_field(payload, "speaker_id")
    transcript = resolve_canonical_field(payload, "transcript")
    raw_emotion = (
        resolve_canonical_field(payload, "emotion")
        if source == "emotion"
        else resolve_canonical_field(payload, "emotion", required=False)
    )
    audio_bytes = read_trusted_regular_file(audio)
    return DatasetItem(
        id=item_id,
        call_id=call_id,
        speaker_id=speaker_id,
        audio_path=str(audio),
        transcript=transcript,
        split=_split_for_call(call_id),
        source=source,
        emotion=_normalize_emotion(raw_emotion) if raw_emotion is not None else None,
        sha256=hashlib.sha256(audio_bytes).hexdigest(),
    )


def build_manifests(consultation_root: Path, emotion_root: Path, output_root: Path) -> None:
    consultation = _input_root(consultation_root)
    emotion = _input_root(emotion_root)
    output = _output_root(output_root)
    source_items: dict[DatasetSource, list[DatasetItem]] = {
        "consultation": [
            _build_item(audio, metadata, source="consultation")
            for audio, metadata in _discover_pairs(consultation)
        ],
        "emotion": [
            _build_item(audio, metadata, source="emotion")
            for audio, metadata in _discover_pairs(emotion)
        ],
    }
    items = [
        *source_items["consultation"],
        *source_items["emotion"],
    ]
    items.sort(key=lambda item: item.id)

    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise ValueError(f"duplicate item id: {item.id}")
        seen.add(item.id)
    validate_disjoint_splits(items)

    grouped: dict[DatasetSplit, list[DatasetItem]] = {split: [] for split in _SPLITS}
    for item in items:
        grouped[item.split].append(item)
    for split in _SPLITS:
        _write_manifest(output / f"{split}.jsonl", grouped[split])
    for source, source_records in source_items.items():
        _write_manifest(
            output / f"{source}.jsonl", sorted(source_records, key=lambda item: item.id)
        )


def _write_manifest(path: Path, items: list[DatasetItem]) -> None:
    if path.is_symlink():
        raise ValueError(f"symlinked output is not allowed: {path}")
    payload = "".join(f"{item.model_dump_json()}\n" for item in items)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        if path.is_symlink():
            raise ValueError(f"symlinked output is not allowed: {path}")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consultation-root", required=True, type=Path)
    parser.add_argument("--emotion-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        build_manifests(arguments.consultation_root, arguments.emotion_root, arguments.output_root)
    except OSError:
        print("manifest build failed: filesystem operation failed", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"manifest build failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
