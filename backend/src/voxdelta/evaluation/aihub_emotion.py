"""Prepare the licensed AI Hub 263 emotion dataset for local training."""

from __future__ import annotations

import csv
import os
import stat
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import BinaryIO, cast
from zipfile import ZipFile

from voxdelta.domain.models import EmotionLabel
from voxdelta.evaluation.manifest import read_trusted_regular_file

OFFICIAL_HEADERS = (
    "wav_id",
    "발화문",
    "상황",
    "1번 감정",
    "1번 감정세기",
    "2번 감정",
    "2번 감정세기",
    "3번 감정",
    "3번 감정세기",
    "4번 감정",
    "4번감정세기",
    "5번 감정",
    "5번 감정세기",
    "나이",
    "성별",
)

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
}


@dataclass(frozen=True)
class EmotionImportRow:
    """One high-consensus item selected for local normalization."""

    item_id: str
    emotion: EmotionLabel
    archive_member: str
    transcript: str = ""


@dataclass(frozen=True)
class EmotionImportPlan:
    """Transcript-free classification of one annual AI Hub release."""

    csv_path: Path
    zip_path: Path
    accepted: tuple[EmotionImportRow, ...]
    ambiguous_ids: tuple[str, ...]
    missing_audio_ids: tuple[str, ...]
    orphan_audio_ids: tuple[str, ...]
    csv_rows: int
    zip_entries: int


def _normalize_emotion(raw: str) -> EmotionLabel:
    try:
        return _EMOTION_MAP[raw.strip().casefold()]
    except KeyError as error:
        raise ValueError("unsupported emotion label") from error


def _consensus(votes: tuple[str, str, str, str, str]) -> EmotionLabel | None:
    counts = Counter(_normalize_emotion(vote) for vote in votes)
    label, count = counts.most_common(1)[0]
    return label if count >= 3 else None


def _load_rows(path: Path) -> list[dict[str, str]]:
    raw = read_trusted_regular_file(path)
    try:
        text = raw.decode("cp949")
    except UnicodeDecodeError as error:
        raise ValueError("invalid CP949 emotion metadata") from error
    reader = csv.DictReader(StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != OFFICIAL_HEADERS:
        raise ValueError("invalid AI Hub 263 metadata headers")
    rows = list(reader)
    seen: set[str] = set()
    for row in rows:
        item_id = row["wav_id"].strip()
        if not item_id:
            raise ValueError("empty metadata ID")
        if item_id in seen:
            raise ValueError("duplicate metadata ID")
        seen.add(item_id)
    return rows


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("symlinked input is not allowed")


@contextmanager
def _trusted_regular_file(path: Path) -> Iterator[BinaryIO]:
    absolute = _absolute_path(path)
    _reject_symlink_components(absolute)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError("unable to open trusted input") from error
    stream: BinaryIO | None = None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("input is not a regular file")
        stream = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
        yield stream
        after = os.fstat(stream.fileno())
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise ValueError("input changed while being read")
    finally:
        if stream is not None:
            stream.close()
        elif descriptor >= 0:
            os.close(descriptor)


def _archive_members(path: Path) -> dict[str, str]:
    audio_members: dict[str, str] = {}
    with _trusted_regular_file(path) as stream, ZipFile(stream) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if member.suffix.casefold() != ".wav":
                raise ValueError("unexpected archive member")
            item_id = member.stem
            if not item_id:
                raise ValueError("empty archive audio ID")
            if item_id in audio_members:
                raise ValueError("duplicate archive audio ID")
            audio_members[item_id] = info.filename
    return audio_members


def plan_emotion_import(
    csv_path: Path,
    zip_path: Path,
    *,
    max_missing_audio: int,
    max_orphan_audio: int,
) -> EmotionImportPlan:
    """Classify one annual CSV/ZIP pair without extracting or copying transcripts."""

    if max_missing_audio < 0 or max_orphan_audio < 0:
        raise ValueError("mismatch allowances must be non-negative")
    rows = _load_rows(csv_path)
    audio_members = _archive_members(zip_path)

    row_ids = {row["wav_id"].strip() for row in rows}
    missing = tuple(sorted(row_ids - audio_members.keys()))
    orphan = tuple(sorted(audio_members.keys() - row_ids))
    if len(missing) > max_missing_audio or len(orphan) > max_orphan_audio:
        raise ValueError("AI Hub 263 audio/metadata mismatch exceeds allowance")

    accepted: list[EmotionImportRow] = []
    ambiguous: list[str] = []
    for row in rows:
        item_id = row["wav_id"].strip()
        if item_id in missing:
            continue
        votes = cast(
            tuple[str, str, str, str, str],
            tuple(row[f"{index}번 감정"] for index in range(1, 6)),
        )
        emotion = _consensus(votes)
        if emotion is None:
            ambiguous.append(item_id)
        else:
            accepted.append(
                EmotionImportRow(
                    item_id=item_id,
                    emotion=emotion,
                    archive_member=audio_members[item_id],
                )
            )

    return EmotionImportPlan(
        csv_path=csv_path,
        zip_path=zip_path,
        accepted=tuple(sorted(accepted, key=lambda item: item.item_id)),
        ambiguous_ids=tuple(sorted(ambiguous)),
        missing_audio_ids=missing,
        orphan_audio_ids=orphan,
        csv_rows=len(rows),
        zip_entries=len(audio_members),
    )
