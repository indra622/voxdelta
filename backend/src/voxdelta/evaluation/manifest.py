"""Typed dataset manifest contracts and split-integrity checks."""

from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, ValidationError, model_validator

from voxdelta.domain.models import EmotionLabel

DatasetSplit = Literal["train", "validation", "test"]
DatasetSource = Literal["consultation", "emotion"]


def _reject_json_constant(_value: str) -> None:
    raise ValueError


class DatasetItem(BaseModel):
    """One immutable utterance record in an evaluation manifest."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    id: str
    call_id: str
    speaker_id: str
    audio_path: str
    transcript: str
    split: DatasetSplit
    source: DatasetSource
    emotion: EmotionLabel | None = None
    start: StrictFloat | None = Field(default=None, ge=0)
    end: StrictFloat | None = Field(default=None, ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_interval(self) -> DatasetItem:
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must both be present or absent")
        if self.start is not None and self.end is not None:
            if not math.isfinite(self.start) or not math.isfinite(self.end):
                raise ValueError("start and end must be finite")
            if self.end <= self.start:
                raise ValueError("end must be greater than start")
        return self


def _absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked path is not allowed: {path}")


def read_trusted_regular_file(path: str | Path) -> bytes:
    """Read a regular file without accepting lexical escapes or symlink traversal."""

    absolute = _absolute_path(path)
    _reject_symlink_components(absolute)
    if not absolute.exists():
        raise ValueError(f"file does not exist: {absolute}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError(f"unable to open trusted file: {absolute}") from error

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"path is not a regular file: {absolute}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"file changed while being read: {absolute}")
    return b"".join(chunks)


def load_manifest(path: str | Path) -> list[DatasetItem]:
    """Load a trusted JSONL manifest and return records sorted by item ID."""

    raw = read_trusted_regular_file(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("manifest must be UTF-8") from error

    items: list[DatasetItem] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid JSON on manifest line {line_number}") from error
        try:
            items.append(DatasetItem.model_validate(payload))
        except ValidationError as error:
            details: list[str] = []
            for detail in error.errors(
                include_url=False, include_context=False, include_input=False
            ):
                location = ".".join(
                    str(part)
                    if isinstance(part, int) or part in DatasetItem.model_fields
                    else "<field>"
                    for part in detail["loc"]
                )
                details.append(f"{location or '<model>'} ({detail['type']})")
            diagnostics = ", ".join(details)
            raise ValueError(
                f"invalid dataset item on manifest line {line_number}: {diagnostics}"
            ) from None

    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise ValueError(f"duplicate item id: {item.id}")
        seen.add(item.id)
    return sorted(items, key=lambda item: item.id)


def validate_disjoint_splits(items: list[DatasetItem]) -> None:
    """Reject call or speaker identities that occur in more than one split."""

    calls: dict[str, DatasetSplit] = {}
    speakers: dict[str, DatasetSplit] = {}
    for item in items:
        call_split = calls.setdefault(item.call_id, item.split)
        if call_split != item.split:
            raise ValueError(f"call leakage: {item.call_id}")
        speaker_split = speakers.setdefault(item.speaker_id, item.split)
        if speaker_split != item.split:
            raise ValueError(f"speaker leakage: {item.speaker_id}")
