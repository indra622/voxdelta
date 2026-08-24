"""Atomic epoch recovery checkpoints with exact resume-identity validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from voxdelta.evaluation.manifest import read_trusted_regular_file

from voxdelta_runpod.ledger import RunIdentity

CheckpointStage = Literal["pilot-a", "pilot-b", "full"]
_EPOCH = re.compile(r"^epoch-([0-9]{4})$")
_STAGING_EPOCH = re.compile(r"^\.epoch-([0-9]{4})\.staging$")
_REQUIRED_FILES = frozenset(
    {
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng-python.json",
        "rng-torch-cpu.pt",
        "rng-torch-cuda.pt",
    }
)
_OPTIONAL_FILES = frozenset({"scaler.pt", "best-model.safetensors"})


class CheckpointError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CheckpointState(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    stage: CheckpointStage
    epoch: int = Field(ge=0)
    optimizer_step: int = Field(ge=0)
    best_metric: float = Field(ge=0, le=1)
    best_epoch: int = Field(ge=0)
    patience_used: int = Field(ge=0)
    micro_batch_size: Literal[8, 4, 2, 1]
    gradient_accumulation_steps: Literal[2, 4, 8, 16]
    identity: RunIdentity
    files: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_state(self) -> CheckpointState:
        if (
            not math.isfinite(self.best_metric)
            or self.best_epoch > self.epoch
            or self.micro_batch_size * self.gradient_accumulation_steps != 16
            or not _REQUIRED_FILES <= set(self.files)
            or not set(self.files) <= _REQUIRED_FILES | _OPTIONAL_FILES
            or any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in self.files.values())
        ):
            raise ValueError("invalid checkpoint state")
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


class ResumeSelection(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    checkpoint: Path | None
    state: CheckpointState | None
    incomplete_checkpoint: Path | None

    @property
    def incomplete_epoch_replay(self) -> bool:
        return self.incomplete_checkpoint is not None


def publish_epoch_checkpoint(
    root: Path,
    state: CheckpointState,
    payloads: dict[str, bytes],
) -> Path:
    if not root.is_absolute() or root.is_symlink():
        raise CheckpointError("invalid_checkpoint_root")
    if set(payloads) != set(state.files):
        raise CheckpointError("checkpoint_payload_mismatch")
    for name, payload in payloads.items():
        if "/" in name or "\\" in name or hashlib.sha256(payload).hexdigest() != state.files[name]:
            raise CheckpointError("checkpoint_payload_mismatch")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = root / f"epoch-{state.epoch:04d}"
    staging = root / f".epoch-{state.epoch:04d}.staging"
    if target.exists() or staging.exists():
        raise CheckpointError("checkpoint_exists")
    staging.mkdir(mode=0o700)
    try:
        for name, payload in sorted(payloads.items()):
            path = staging / name
            with path.open("xb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        with (staging / "state.json").open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(state.canonical_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, target)
        parent = os.open(root, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _load_checkpoint(path: Path, expected_stage: CheckpointStage) -> CheckpointState:
    try:
        state = CheckpointState.model_validate_json(read_trusted_regular_file(path / "state.json"))
        if state.stage != expected_stage:
            raise CheckpointError("checkpoint_stage_mismatch")
        expected_names = set(state.files) | {"state.json"}
        actual_names = {item.name for item in path.iterdir()}
        if expected_names != actual_names or any(item.is_symlink() for item in path.iterdir()):
            raise CheckpointError("invalid_checkpoint")
        for name, expected_digest in state.files.items():
            digest = hashlib.sha256(read_trusted_regular_file(path / name)).hexdigest()
            if digest != expected_digest:
                raise CheckpointError("checkpoint_digest_mismatch")
        return state
    except CheckpointError:
        raise
    except Exception:
        raise CheckpointError("invalid_checkpoint") from None


def select_resume_checkpoint(
    root: Path,
    *,
    expected_stage: CheckpointStage,
    expected_identity: RunIdentity,
) -> ResumeSelection:
    if not root.is_absolute() or root.is_symlink():
        raise CheckpointError("invalid_checkpoint_root")
    if not root.exists():
        return ResumeSelection(checkpoint=None, state=None, incomplete_checkpoint=None)
    complete: list[tuple[int, Path]] = []
    incomplete: list[int] = []
    for path in root.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise CheckpointError("invalid_checkpoint")
        complete_match = _EPOCH.fullmatch(path.name)
        incomplete_match = _STAGING_EPOCH.fullmatch(path.name)
        if complete_match is not None:
            complete.append((int(complete_match.group(1)), path))
        elif incomplete_match is not None:
            incomplete.append(int(incomplete_match.group(1)))
        else:
            raise CheckpointError("invalid_checkpoint")
    if len(incomplete) > 1:
        raise CheckpointError("too_many_incomplete_epochs")
    states: list[tuple[int, Path, CheckpointState]] = []
    for epoch, path in sorted(complete):
        state = _load_checkpoint(path, expected_stage)
        if state.epoch != epoch or state.identity != expected_identity:
            raise CheckpointError("checkpoint_identity_mismatch")
        states.append((epoch, path, state))
    if incomplete and states and incomplete[0] > states[-1][0] + 1:
        raise CheckpointError("invalid_incomplete_epoch")
    if not states:
        return ResumeSelection(
            checkpoint=None,
            state=None,
            incomplete_checkpoint=(
                root / f".epoch-{incomplete[0]:04d}.staging" if incomplete else None
            ),
        )
    _, path, state = states[-1]
    return ResumeSelection(
        checkpoint=path,
        state=state,
        incomplete_checkpoint=(
            root / f".epoch-{incomplete[0]:04d}.staging" if incomplete else None
        ),
    )


__all__ = [
    "CheckpointError",
    "CheckpointStage",
    "CheckpointState",
    "ResumeSelection",
    "publish_epoch_checkpoint",
    "select_resume_checkpoint",
]
