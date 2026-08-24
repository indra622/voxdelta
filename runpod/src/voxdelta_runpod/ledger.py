"""Append-only stage ledger with immutable experiment-identity enforcement."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from voxdelta.evaluation.manifest import read_trusted_regular_file

Stage = Literal[
    "initialized",
    "preflight",
    "pilot-a",
    "pilot-b",
    "pilot-selected",
    "full",
    "candidate-frozen",
    "final",
    "retrieved",
    "deletion-ready",
    "failed",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRANSITIONS: dict[Stage, frozenset[Stage]] = {
    "initialized": frozenset({"preflight", "failed"}),
    "preflight": frozenset({"pilot-a", "failed"}),
    "pilot-a": frozenset({"pilot-b", "failed"}),
    "pilot-b": frozenset({"pilot-selected", "failed"}),
    "pilot-selected": frozenset({"full", "failed"}),
    "full": frozenset({"candidate-frozen", "failed"}),
    "candidate-frozen": frozenset({"final", "failed"}),
    "final": frozenset({"retrieved", "failed"}),
    "retrieved": frozenset({"deletion-ready", "failed"}),
    "deletion-ready": frozenset(),
    "failed": frozenset(),
}


class LedgerError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class RunIdentity(_FrozenModel):
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    container_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sampler_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class LedgerRecord(_FrozenModel):
    schema_version: Literal["1"] = "1"
    sequence: int = Field(ge=0)
    previous_record_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    from_stage: Stage | None
    to_stage: Stage
    identity: RunIdentity
    report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    recorded_at: str

    @model_validator(mode="after")
    def legal_shape(self) -> LedgerRecord:
        if self.sequence == 0:
            if (
                self.previous_record_sha256 is not None
                or self.from_stage is not None
                or self.to_stage != "initialized"
            ):
                raise ValueError("invalid initial ledger record")
        elif self.previous_record_sha256 is None or self.from_stage is None:
            raise ValueError("invalid chained ledger record")
        try:
            timestamp = datetime.fromisoformat(self.recorded_at.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("invalid ledger timestamp") from None
        if timestamp.tzinfo is None or not math.isfinite(timestamp.timestamp()):
            raise ValueError("invalid ledger timestamp")
        return self

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _private_publish(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.staging")
    if path.exists() or temporary.exists():
        raise LedgerError("ledger_record_exists")
    try:
        with temporary.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_ledger(root: Path) -> tuple[LedgerRecord, ...]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise LedgerError("invalid_ledger")
    paths = sorted(root.glob("record-*.json"))
    if any(path.is_symlink() for path in paths):
        raise LedgerError("invalid_ledger")
    records: list[LedgerRecord] = []
    try:
        for sequence, path in enumerate(paths):
            if path.name != f"record-{sequence:06d}.json":
                raise LedgerError("invalid_ledger")
            record = LedgerRecord.model_validate_json(read_trusted_regular_file(path))
            if record.sequence != sequence:
                raise LedgerError("invalid_ledger")
            if sequence == 0:
                if record.from_stage is not None or record.to_stage != "initialized":
                    raise LedgerError("invalid_ledger")
            else:
                previous = records[-1]
                if (
                    record.previous_record_sha256 != previous.digest()
                    or record.from_stage != previous.to_stage
                    or record.to_stage not in _TRANSITIONS[previous.to_stage]
                    or record.identity != previous.identity
                ):
                    raise LedgerError("invalid_ledger")
            records.append(record)
    except LedgerError:
        raise
    except Exception:
        raise LedgerError("invalid_ledger") from None
    return tuple(records)


def initialize_ledger(
    root: Path, identity: RunIdentity, *, recorded_at: datetime | None = None
) -> LedgerRecord:
    if not root.is_absolute() or root.is_symlink() or root.exists():
        raise LedgerError("ledger_exists")
    staging = root.with_name(f".{root.name}.staging")
    if staging.exists():
        raise LedgerError("ledger_exists")
    staging.mkdir(mode=0o700, parents=True)
    timestamp = recorded_at or datetime.now(UTC)
    record = LedgerRecord(
        sequence=0,
        previous_record_sha256=None,
        from_stage=None,
        to_stage="initialized",
        identity=identity,
        report_sha256=None,
        recorded_at=timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    )
    try:
        _private_publish(staging / "record-000000.json", record.canonical_bytes())
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return record


def append_transition(
    root: Path,
    to_stage: Stage,
    identity: RunIdentity,
    *,
    report_sha256: str | None = None,
    recorded_at: datetime | None = None,
) -> LedgerRecord:
    records = load_ledger(root)
    if not records:
        raise LedgerError("invalid_ledger")
    previous = records[-1]
    if identity != previous.identity:
        raise LedgerError("ledger_identity_mismatch")
    if to_stage not in _TRANSITIONS[previous.to_stage]:
        raise LedgerError("illegal_stage_transition")
    if report_sha256 is not None and not _SHA256.fullmatch(report_sha256):
        raise LedgerError("invalid_report_digest")
    timestamp = recorded_at or datetime.now(UTC)
    record = LedgerRecord(
        sequence=len(records),
        previous_record_sha256=previous.digest(),
        from_stage=previous.to_stage,
        to_stage=to_stage,
        identity=identity,
        report_sha256=report_sha256,
        recorded_at=timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    )
    _private_publish(root / f"record-{record.sequence:06d}.json", record.canonical_bytes())
    return record


__all__ = [
    "LedgerError",
    "LedgerRecord",
    "RunIdentity",
    "Stage",
    "append_transition",
    "initialize_ledger",
    "load_ledger",
]
