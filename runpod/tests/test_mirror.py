from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from voxdelta_runpod.checkpoint import CheckpointState, publish_epoch_checkpoint
from voxdelta_runpod.ledger import RunIdentity
from voxdelta_runpod.mirror import (
    MirrorError,
    completed_epochs,
    latest_verified_epoch,
    publish_mirror,
    verify_mirrored_checkpoint,
)

MIRROR_CLI = Path(__file__).parents[1] / "scripts" / "mirror_checkpoints.py"


def _cli() -> Any:
    spec = importlib.util.spec_from_file_location("mirror_checkpoints", MIRROR_CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity() -> RunIdentity:
    return RunIdentity(
        config_sha256="a" * 64,
        code_sha256="b" * 64,
        container_sha256="c" * 64,
        base_model_sha256="d" * 64,
        archive_sha256="e" * 64,
        manifest_sha256="f" * 64,
        sampler_sha256="0" * 64,
    )


def _payloads() -> dict[str, bytes]:
    return {
        "model.safetensors": b"model-weights",
        "optimizer.pt": b"optimizer-state",
        "scheduler.pt": b"scheduler-state",
        "rng-python.json": b"python-rng",
        "rng-torch-cpu.pt": b"cpu-rng",
        "rng-torch-cuda.pt": b"cuda-rng",
    }


def _write_epoch(root: Path, epoch: int, *, stage: str = "full") -> Path:
    payloads = _payloads()
    state = CheckpointState(
        stage=stage,
        epoch=epoch,
        optimizer_step=epoch * 100,
        best_metric=0.25,
        best_epoch=epoch,
        patience_used=0,
        micro_batch_size=4,
        gradient_accumulation_steps=4,
        identity=_identity(),
        files={name: hashlib.sha256(body).hexdigest() for name, body in payloads.items()},
    )
    return publish_epoch_checkpoint(root, state, payloads)


def test_completed_epochs_ignores_staging_and_sorts(tmp_path: Path) -> None:
    root = (tmp_path / "checkpoints" / "full").resolve()
    _write_epoch(root, 2)
    _write_epoch(root, 1)
    (root / ".epoch-0003.staging").mkdir(mode=0o700)

    found = completed_epochs(root)

    assert [path.name for path in found] == ["epoch-0001", "epoch-0002"]


def test_verify_mirrored_checkpoint_rejects_partial_and_corrupt(tmp_path: Path) -> None:
    root = (tmp_path / "checkpoints" / "full").resolve()
    good = _write_epoch(root, 1)
    assert verify_mirrored_checkpoint(good, expected_stage="full").epoch == 1

    truncated = _write_epoch(root, 2)
    (truncated / "optimizer.pt").unlink()
    with pytest.raises(MirrorError):
        verify_mirrored_checkpoint(truncated, expected_stage="full")

    corrupt = _write_epoch(root, 3)
    (corrupt / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(MirrorError):
        verify_mirrored_checkpoint(corrupt, expected_stage="full")


def test_publish_mirror_is_atomic_private_and_non_overwriting(tmp_path: Path) -> None:
    source = (tmp_path / "src" / "full").resolve()
    epoch = _write_epoch(source, 1)
    target_root = (tmp_path / "mirror" / "full").resolve()
    target_root.mkdir(mode=0o700, parents=True)

    published = publish_mirror(epoch, target_root / "epoch-0001")

    assert published.is_dir()
    assert published.stat().st_mode & 0o077 == 0
    assert all(item.stat().st_mode & 0o077 == 0 for item in published.iterdir())
    with pytest.raises(MirrorError):
        publish_mirror(epoch, published)


def test_latest_verified_epoch_skips_unverifiable(tmp_path: Path) -> None:
    mirror = (tmp_path / "mirror" / "full").resolve()
    _write_epoch(mirror, 1)
    _write_epoch(mirror, 2)
    broken = _write_epoch(mirror, 3)
    (broken / "rng.json").write_bytes(b"tampered")

    selected = latest_verified_epoch(mirror, expected_stage="full")

    assert selected is not None and selected[0] == 2


def _fake_bin(directory: Path, remote_root: Path) -> Path:
    """ssh/rsync stand-ins backed by a local directory acting as the Pod."""
    directory.mkdir(parents=True, exist_ok=True)
    ssh = directory / "ssh"
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cmd="${@: -1}"\n'
        f'export REMOTE_ROOT="{remote_root}"\n'
        'eval "${cmd//\\$REMOTE/$REMOTE_ROOT}"\n'
    )
    ssh.chmod(0o700)
    rsync = directory / "rsync"
    rsync.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'REMOTE_ROOT="{remote_root}"\n'
        'args=("$@")\n'
        'destination="${args[${#args[@]}-1]}"\n'
        'source="${args[${#args[@]}-2]}"\n'
        'source="${source#*:}"\n'
        'destination="${destination#*:}"\n'
        'source="${source//\\$REMOTE/$REMOTE_ROOT}"\n'
        'destination="${destination//\\$REMOTE/$REMOTE_ROOT}"\n'
        '[[ -e "${source%/}" ]] || exit 23\n'
        'if [[ -d "${source%/}" ]]; then\n'
        '  mkdir -p "$destination"\n'
        '  cp -R "${source%/}/." "$destination"\n'
        "else\n"
        '  mkdir -p "$(dirname "$destination")"\n'
        '  cp "$source" "$destination"\n'
        "fi\n"
    )
    rsync.chmod(0o700)
    return directory


def test_mirror_cli_copies_only_complete_epochs_and_keeps_them_private(tmp_path: Path) -> None:
    module = _cli()
    remote = (tmp_path / "remote").resolve()
    remote_full = remote / "state" / "checkpoints" / "full"
    _write_epoch(remote_full, 1)
    _write_epoch(remote_full, 2)
    (remote_full / ".epoch-0003.staging").mkdir(mode=0o700, parents=True)
    (remote / "ledger").mkdir(mode=0o700, parents=True)
    (remote / "ledger" / "ledger.jsonl").write_text("{}\n")
    (remote / "data").mkdir(mode=0o700, parents=True)
    (remote / "data" / "secret.wav").write_bytes(b"licensed-audio")

    local = (tmp_path / "mirror").resolve()
    environment = dict(os.environ)
    environment["PATH"] = f"{_fake_bin(tmp_path / 'bin', remote)}:{environment['PATH']}"

    completed = subprocess.run(
        [
            "python3",
            str(MIRROR_CLI),
            "sync-once",
            "--remote-root",
            "$REMOTE",
            "--stage",
            "full",
            "--local",
            str(local),
            "--ssh-user",
            "root",
            "--ssh-host",
            "example.invalid",
            "--ssh-port",
            "22",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    mirrored = sorted(p.name for p in (local / "checkpoints" / "full").iterdir())
    assert mirrored == ["epoch-0001", "epoch-0002"]
    assert not (local / "data").exists()
    for epoch in (local / "checkpoints" / "full").iterdir():
        assert epoch.stat().st_mode & 0o077 == 0
    del module


def test_mirror_cli_reports_restore_point(tmp_path: Path) -> None:
    module = _cli()
    mirror = (tmp_path / "mirror").resolve()
    full = mirror / "checkpoints" / "full"
    _write_epoch(full, 1)
    _write_epoch(full, 4)

    payload = module.restore_point(mirror, stage="full")

    assert payload["epoch"] == 4
    assert json.loads(json.dumps(payload))["stage"] == "full"
