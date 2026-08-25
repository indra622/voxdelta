"""Mirror completed remote epoch checkpoints to private local storage, and restore them.

The active run lives on the Pod's container disk, which is lost when the Pod stops.
``watch`` polls a detached training stage and pulls each epoch directory down once it
has been atomically published, verifying it before it is kept. Transient SSH failures
are tolerated: observation must never take training down. ``restore`` pushes the newest
verified epoch back onto a fresh Pod so ``full-or-resume`` can continue from it.

Raw audio is never mirrored back out.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voxdelta_runpod.checkpoint import CheckpointStage  # noqa: E402
from voxdelta_runpod.mirror import (  # noqa: E402
    MIRROR_EVIDENCE,
    MirrorError,
    completed_epochs,
    discard_staging,
    latest_verified_epoch,
    publish_mirror,
    verify_mirrored_checkpoint,
)

_SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new")
_RSYNC_FLAGS = (
    "--archive",
    "--no-owner",
    "--no-group",
    "--partial",
    "--chmod=F600,D700",
)


def _ssh_command(arguments: argparse.Namespace) -> list[str]:
    return ["ssh", "-p", str(arguments.ssh_port), *_SSH_OPTIONS]


def _rsync_shell(arguments: argparse.Namespace) -> str:
    return f"ssh -p {arguments.ssh_port} {' '.join(_SSH_OPTIONS)}"


def _target(arguments: argparse.Namespace) -> str:
    return f"{arguments.ssh_user}@{arguments.ssh_host}"


def _remote_listing(arguments: argparse.Namespace, remote_root: str, stage: str) -> list[str]:
    """Names of published epoch directories on the Pod, newest last."""
    script = (
        f"ls -1 {remote_root}/state/checkpoints/{stage} 2>/dev/null "
        "| grep -E '^epoch-[0-9]{4}$' || true"
    )
    completed = subprocess.run(
        [*_ssh_command(arguments), _target(arguments), script],
        capture_output=True,
        text=True,
        check=True,
        timeout=arguments.ssh_timeout,
    )
    return sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})


def _pull(arguments: argparse.Namespace, source: str, destination: Path) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            *_RSYNC_FLAGS,
            "-e",
            _rsync_shell(arguments),
            f"{_target(arguments)}:{source}",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=arguments.rsync_timeout,
    )


def _mirror_evidence(arguments: argparse.Namespace, remote_root: str, local: Path) -> None:
    for relative in MIRROR_EVIDENCE:
        destination = local / "evidence" / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            _pull(arguments, f"{remote_root}/{relative}", destination)
        except subprocess.SubprocessError:
            continue
    for directory in ("ledger", "logs"):
        destination = local / "evidence" / directory
        destination.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            _pull(arguments, f"{remote_root}/{directory}/", destination)
        except subprocess.SubprocessError:
            continue


def sync_once(arguments: argparse.Namespace) -> int:
    """Pull every published epoch not already held locally. Returns the number kept."""
    local: Path = arguments.local.resolve()
    stage: str = arguments.stage
    mirror_root = local / "checkpoints" / stage
    mirror_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    held = {path.name for path in completed_epochs(mirror_root)}
    kept = 0
    for name in _remote_listing(arguments, arguments.remote_root, stage):
        if name in held:
            continue
        staging = mirror_root / f".{name}.staging"
        discard_staging(staging)
        staging.mkdir(mode=0o700, parents=True)
        try:
            _pull(arguments, f"{arguments.remote_root}/state/checkpoints/{stage}/{name}/", staging)
            verify_mirrored_checkpoint(staging, expected_stage=_stage(stage))
            publish_mirror(staging, mirror_root / name)
        except (MirrorError, subprocess.SubprocessError, OSError):
            # An epoch still being written simply arrives on a later pass.
            discard_staging(staging)
            continue
        kept += 1
    _mirror_evidence(arguments, arguments.remote_root, local)
    return kept


def _stage(value: str) -> CheckpointStage:
    if value not in ("pilot-a", "pilot-b", "full"):
        raise MirrorError("invalid_stage")
    return value  # type: ignore[return-value]


def _stage_outcome(arguments: argparse.Namespace) -> str | None:
    script = (
        f"{arguments.remote_python} {arguments.remote_scripts}/run_remote_stage.py "
        f"status {arguments.remote_stage} --root {arguments.remote_root} 2>/dev/null || true"
    )
    try:
        completed = subprocess.run(
            [*_ssh_command(arguments), _target(arguments), script],
            capture_output=True,
            text=True,
            check=False,
            timeout=arguments.ssh_timeout,
        )
    except subprocess.SubprocessError:
        return None
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return str(json.loads(line)["outcome"])
            except (ValueError, KeyError):
                return None
    return None


def watch(arguments: argparse.Namespace) -> int:
    deadline = time.monotonic() + arguments.max_minutes * 60
    consecutive_failures = 0
    total = 0
    while time.monotonic() < deadline:
        try:
            total += sync_once(arguments)
            consecutive_failures = 0
        except (subprocess.SubprocessError, OSError):
            consecutive_failures += 1
            print(f"mirror_observation_failed {consecutive_failures}", flush=True)
            if consecutive_failures > arguments.tolerate_failures:
                print("mirror_giving_up", flush=True)
                return 3
        outcome = _stage_outcome(arguments)
        if outcome in ("succeeded", "failed"):
            sync_once(arguments)
            print(f"mirror_complete outcome={outcome} epochs_mirrored={total}", flush=True)
            return 0 if outcome == "succeeded" else 4
        print(f"mirror_tick epochs_mirrored={total}", flush=True)
        time.sleep(arguments.interval)
    print(f"mirror_deadline epochs_mirrored={total}", flush=True)
    return 5


def restore_point(local: Path, *, stage: str) -> dict[str, Any]:
    selected = latest_verified_epoch(
        local.resolve() / "checkpoints" / stage, expected_stage=_stage(stage)
    )
    if selected is None:
        raise MirrorError("no_verified_checkpoint")
    epoch, path, state = selected
    return {
        "stage": stage,
        "epoch": epoch,
        "path": str(path),
        "optimizer_step": state.optimizer_step,
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
    }


def restore(arguments: argparse.Namespace) -> int:
    point = restore_point(arguments.local, stage=arguments.stage)
    source = Path(point["path"])
    remote_dir = f"{arguments.remote_root}/state/checkpoints/{arguments.stage}/{source.name}"
    subprocess.run(
        [
            *_ssh_command(arguments),
            _target(arguments),
            f"install -d -m 700 {arguments.remote_root}/state/checkpoints/{arguments.stage}",
        ],
        check=True,
        timeout=arguments.ssh_timeout,
    )
    subprocess.run(
        [
            "rsync",
            *_RSYNC_FLAGS,
            "-e",
            _rsync_shell(arguments),
            f"{source}/",
            f"{_target(arguments)}:{remote_dir}/",
        ],
        check=True,
        timeout=arguments.rsync_timeout,
    )
    for relative in MIRROR_EVIDENCE:
        candidate = arguments.local.resolve() / "evidence" / relative
        if not candidate.is_file():
            continue
        subprocess.run(
            [
                *_ssh_command(arguments),
                _target(arguments),
                f"install -d -m 700 {arguments.remote_root}/{Path(relative).parent}",
            ],
            check=True,
            timeout=arguments.ssh_timeout,
        )
        subprocess.run(
            [
                "rsync",
                *_RSYNC_FLAGS,
                "-e",
                _rsync_shell(arguments),
                str(candidate),
                f"{_target(arguments)}:{arguments.remote_root}/{relative}",
            ],
            check=True,
            timeout=arguments.rsync_timeout,
        )
    print(json.dumps(point, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("sync-once", "watch", "restore", "restore-point"):
        command = subparsers.add_parser(action)
        command.add_argument("--local", required=True, type=Path)
        command.add_argument("--stage", default="full", choices=("pilot-a", "pilot-b", "full"))
        if action != "restore-point":
            command.add_argument("--remote-root", required=True)
            command.add_argument("--ssh-user", required=True)
            command.add_argument("--ssh-host", required=True)
            command.add_argument("--ssh-port", required=True)
            command.add_argument("--ssh-timeout", type=float, default=120.0)
            command.add_argument("--rsync-timeout", type=float, default=3600.0)
        if action == "watch":
            command.add_argument("--interval", type=float, default=120.0)
            command.add_argument("--max-minutes", type=float, default=1440.0)
            command.add_argument("--tolerate-failures", type=int, default=10)
            command.add_argument("--remote-stage", default="full-or-resume")
            command.add_argument("--remote-python", default="/opt/voxdelta/runpod/.venv/bin/python")
            command.add_argument("--remote-scripts", default="/opt/voxdelta/runpod/scripts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.action == "sync-once":
            print(f"mirror_synced epochs={sync_once(arguments)}")
            return 0
        if arguments.action == "watch":
            return watch(arguments)
        if arguments.action == "restore-point":
            print(json.dumps(restore_point(arguments.local, stage=arguments.stage), sort_keys=True))
            return 0
        return restore(arguments)
    except (MirrorError, OSError, ValueError, subprocess.SubprocessError):
        print("mirror_error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
