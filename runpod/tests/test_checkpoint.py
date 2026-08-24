from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from voxdelta_runpod.checkpoint import (
    CheckpointError,
    CheckpointState,
    publish_epoch_checkpoint,
    select_resume_checkpoint,
)
from voxdelta_runpod.ledger import RunIdentity


def _identity(character: str = "a") -> RunIdentity:
    return RunIdentity(
        config_sha256=character * 64,
        code_sha256="b" * 64,
        container_sha256="c" * 64,
        base_model_sha256="d" * 64,
        archive_sha256="e" * 64,
        manifest_sha256="f" * 64,
        sampler_sha256="0" * 64,
    )


def _payloads() -> dict[str, bytes]:
    return {
        "model.safetensors": b"model",
        "optimizer.pt": b"optimizer",
        "scheduler.pt": b"scheduler",
        "rng-python.json": b"python-rng",
        "rng-torch-cpu.pt": b"cpu-rng",
        "rng-torch-cuda.pt": b"cuda-rng",
    }


def _state(epoch: int, *, identity: RunIdentity | None = None) -> CheckpointState:
    payloads = _payloads()
    return CheckpointState(
        stage="full",
        epoch=epoch,
        optimizer_step=epoch * 100,
        best_metric=0.25,
        patience_used=0,
        micro_batch_size=4,
        gradient_accumulation_steps=4,
        identity=identity or _identity(),
        files={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
    )


def test_checkpoint_publication_and_highest_complete_resume(tmp_path: Path) -> None:
    root = (tmp_path / "checkpoints" / "full").resolve()
    first = publish_epoch_checkpoint(root, _state(1), _payloads())
    second = publish_epoch_checkpoint(root, _state(2), _payloads())

    selection = select_resume_checkpoint(root, expected_stage="full", expected_identity=_identity())

    assert first.name == "epoch-0001"
    assert second.name == "epoch-0002"
    assert selection.checkpoint == second
    assert selection.state is not None and selection.state.epoch == 2
    assert selection.incomplete_epoch_replay is False
    assert second.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o077 == 0 for path in second.iterdir())


def test_resume_allows_at_most_one_next_incomplete_epoch(tmp_path: Path) -> None:
    root = (tmp_path / "checkpoints" / "full").resolve()
    publish_epoch_checkpoint(root, _state(1), _payloads())
    (root / ".epoch-0002.staging").mkdir()

    selection = select_resume_checkpoint(root, expected_stage="full", expected_identity=_identity())
    assert selection.state is not None and selection.state.epoch == 1
    assert selection.incomplete_epoch_replay is True

    (root / ".epoch-0003.staging").mkdir()
    with pytest.raises(CheckpointError, match="^too_many_incomplete_epochs$"):
        select_resume_checkpoint(root, expected_stage="full", expected_identity=_identity())


def test_checkpoint_rejects_digest_identity_stage_and_existing_target(tmp_path: Path) -> None:
    root = (tmp_path / "checkpoints" / "full").resolve()
    target = publish_epoch_checkpoint(root, _state(1), _payloads())

    with pytest.raises(CheckpointError, match="^checkpoint_exists$"):
        publish_epoch_checkpoint(root, _state(1), _payloads())

    (target / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(CheckpointError, match="^checkpoint_digest_mismatch$"):
        select_resume_checkpoint(root, expected_stage="full", expected_identity=_identity())

    root_two = (tmp_path / "checkpoints" / "identity").resolve()
    publish_epoch_checkpoint(root_two, _state(1), _payloads())
    with pytest.raises(CheckpointError, match="^checkpoint_identity_mismatch$"):
        select_resume_checkpoint(root_two, expected_stage="full", expected_identity=_identity("1"))
    with pytest.raises(CheckpointError, match="^checkpoint_stage_mismatch$"):
        select_resume_checkpoint(root_two, expected_stage="pilot-a", expected_identity=_identity())


def test_checkpoint_payload_digest_is_checked_before_publication(tmp_path: Path) -> None:
    payloads = _payloads()
    payloads["model.safetensors"] = b"different"
    with pytest.raises(CheckpointError, match="^checkpoint_payload_mismatch$"):
        publish_epoch_checkpoint((tmp_path / "root").resolve(), _state(1), payloads)
