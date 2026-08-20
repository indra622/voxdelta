"""Prepare the licensed AI Hub 263 emotion dataset for local training."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import BinaryIO, cast
from zipfile import BadZipFile, ZipFile

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
_RELEASE_NAMES = ("4차년도", "5차년도", "5차년도_2차")
_SAFE_ITEM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


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


@dataclass(frozen=True)
class EmotionReleaseReport:
    """Transcript-free outcome for one official annual release."""

    release: str
    csv_rows: int
    zip_entries: int
    accepted: int
    ambiguous_ids: tuple[str, ...]
    missing_audio_ids: tuple[str, ...]
    orphan_audio_ids: tuple[str, ...]


@dataclass(frozen=True)
class EmotionImportReport:
    """Machine-readable summary of a completed local normalization."""

    dataset: str
    accepted: int
    ambiguous: int
    missing_audio: int
    orphan_audio: int
    label_counts: dict[str, int]
    source_file_sizes: dict[str, int]
    releases: tuple[EmotionReleaseReport, ...]


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


def _input_directory(path: Path) -> Path:
    absolute = _absolute_path(path)
    _reject_symlink_components(absolute)
    if not absolute.is_dir():
        raise ValueError("source root is not a directory")
    return absolute


def _output_path(path: Path) -> Path:
    absolute = _absolute_path(path)
    _reject_symlink_components(absolute)
    if absolute.exists():
        raise ValueError("output already exists")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(absolute.parent)
    if not absolute.parent.is_dir():
        raise ValueError("output parent is not a directory")
    return absolute


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary_path = Path(f.name)
            os.fchmod(f.fileno(), 0o600)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_audio_member(archive: ZipFile, member: str, path: Path) -> str:
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    try:
        with (
            archive.open(member, "r") as source,
            tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as target,
        ):
            temporary_path = Path(target.name)
            os.fchmod(target.fileno(), 0o600)
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        return digest.hexdigest()
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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


def import_emotion_dataset(
    source_root: Path,
    output_root: Path,
    *,
    max_missing_audio: int,
    max_orphan_audio: int,
) -> EmotionImportReport:
    """Normalize all three official releases through a same-filesystem staging directory."""

    if max_missing_audio < 0 or max_orphan_audio < 0:
        raise ValueError("mismatch allowances must be non-negative")
    source = _input_directory(source_root)
    output = _output_path(output_root)
    expected = {f"{release}{suffix}" for release in _RELEASE_NAMES for suffix in (".csv", ".zip")}
    actual = {path.name for path in source.iterdir()}
    if actual != expected:
        raise ValueError("invalid official release set")

    plans = tuple(
        plan_emotion_import(
            source / f"{release}.csv",
            source / f"{release}.zip",
            max_missing_audio=max_missing_audio,
            max_orphan_audio=max_orphan_audio,
        )
        for release in _RELEASE_NAMES
    )
    total_missing = sum(len(plan.missing_audio_ids) for plan in plans)
    total_orphan = sum(len(plan.orphan_audio_ids) for plan in plans)
    if total_missing > max_missing_audio or total_orphan > max_orphan_audio:
        raise ValueError("AI Hub 263 audio/metadata mismatch exceeds allowance")

    accepted_ids = [row.item_id for plan in plans for row in plan.accepted]
    if len(accepted_ids) != len(set(accepted_ids)):
        raise ValueError("duplicate accepted ID across releases")
    if any(_SAFE_ITEM_ID.fullmatch(item_id) is None for item_id in accepted_ids):
        raise ValueError("unsafe metadata ID")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(staging, 0o700)
    try:
        pairs_root = staging / "pairs"
        pairs_root.mkdir(mode=0o700)
        label_counts: Counter[str] = Counter()
        release_reports: list[EmotionReleaseReport] = []
        for release, plan in zip(_RELEASE_NAMES, plans, strict=True):
            try:
                with _trusted_regular_file(plan.zip_path) as stream, ZipFile(stream) as archive:
                    for row in plan.accepted:
                        shard = pairs_root / row.item_id[:2]
                        shard.mkdir(mode=0o700, exist_ok=True)
                        audio_path = shard / f"{row.item_id}.wav"
                        metadata_path = shard / f"{row.item_id}.json"
                        _write_audio_member(archive, row.archive_member, audio_path)
                        synthetic_group = f"aihub-263-item:{row.item_id}"
                        metadata = {
                            "id": row.item_id,
                            "call_id": synthetic_group,
                            "speaker_id": synthetic_group,
                            "transcript": "",
                            "emotion": row.emotion,
                        }
                        _write_bytes_atomic(
                            metadata_path,
                            (
                                json.dumps(metadata, sort_keys=True, ensure_ascii=False) + "\n"
                            ).encode(),
                        )
                        label_counts[row.emotion] += 1
            except BadZipFile:
                raise ValueError("archive read failed") from None
            release_reports.append(
                EmotionReleaseReport(
                    release=release,
                    csv_rows=plan.csv_rows,
                    zip_entries=plan.zip_entries,
                    accepted=len(plan.accepted),
                    ambiguous_ids=plan.ambiguous_ids,
                    missing_audio_ids=plan.missing_audio_ids,
                    orphan_audio_ids=plan.orphan_audio_ids,
                )
            )

        report = EmotionImportReport(
            dataset="AI Hub 263",
            accepted=sum(item.accepted for item in release_reports),
            ambiguous=sum(len(item.ambiguous_ids) for item in release_reports),
            missing_audio=sum(len(item.missing_audio_ids) for item in release_reports),
            orphan_audio=sum(len(item.orphan_audio_ids) for item in release_reports),
            label_counts=dict(sorted(label_counts.items())),
            source_file_sizes={path.name: path.stat().st_size for path in sorted(source.iterdir())},
            releases=tuple(release_reports),
        )
        _write_bytes_atomic(
            staging / "import-report.json",
            (json.dumps(asdict(report), sort_keys=True, ensure_ascii=False) + "\n").encode(),
        )
        os.replace(staging, output)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
