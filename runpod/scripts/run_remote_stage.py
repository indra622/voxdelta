"""Launch, resume, inspect, and wait for a detached remote training stage."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Literal

from pydantic import BaseModel, ConfigDict, Field
from voxdelta.evaluation.manifest import read_trusted_regular_file

Stage = Literal["pilot", "full-or-resume"]
StageOutcome = Literal["running", "succeeded", "failed"]


class StageStatus(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    stage: Literal["pilot", "full-or-resume"]
    attempt: int = Field(gt=0)
    outcome: Literal["running", "succeeded", "failed"]
    pid: int = Field(gt=0)
    started_at: str = Field(min_length=20, max_length=40)
    finished_at: str | None = Field(default=None, min_length=20, max_length=40)
    exit_code: int | None = None


StageStatus.model_rebuild(_types_namespace={"Literal": Literal})


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _name(stage: Stage) -> str:
    return "pilot" if stage == "pilot" else "full"


def _paths(root: Path, stage: Stage) -> tuple[Path, Path, Path, Path]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError
    state_root = root / "state" / "stages"
    log_root = root / "logs"
    state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = _name(stage)
    return (
        state_root / f"{name}.json",
        state_root / f"{name}.lock",
        state_root / f".{name}.ready",
        log_root / f"{name}.log",
    )


def _write_status(path: Path, status: StageStatus) -> None:
    temporary = path.with_name(f".{path.name}.staging")
    temporary.unlink(missing_ok=True)
    with temporary.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write((status.model_dump_json() + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_status(path: Path) -> StageStatus:
    return StageStatus.model_validate_json(read_trusted_regular_file(path))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@contextmanager
def _locked(path: Path) -> Iterator[IO[bytes]]:
    with path.open("a+b") as stream:
        os.fchmod(stream.fileno(), 0o600)
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield stream


def _launch(arguments: argparse.Namespace) -> int:
    root = arguments.root.resolve()
    config = arguments.config.resolve()
    runner = arguments.runner.resolve()
    if not config.is_file() or config.is_symlink() or not runner.is_file() or runner.is_symlink():
        raise ValueError
    status_path, lock_path, ready_path, log_path = _paths(root, arguments.stage)
    with _locked(lock_path):
        previous = _load_status(status_path) if status_path.exists() else None
        if previous is not None and previous.outcome == "succeeded":
            print("stage_succeeded")
            return 0
        if previous is not None and previous.outcome == "running" and _alive(previous.pid):
            print("stage_running")
            return 0
        attempt = 1 if previous is None else previous.attempt + 1
        ready_path.unlink(missing_ok=True)
        with log_path.open("ab") as log:
            os.fchmod(log.fileno(), 0o600)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_worker",
                    arguments.stage,
                    "--config",
                    str(config),
                    "--root",
                    str(root),
                    "--runner",
                    str(runner),
                    "--attempt",
                    str(attempt),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        _write_status(
            status_path,
            StageStatus(
                stage=arguments.stage,
                attempt=attempt,
                outcome="running",
                pid=process.pid,
                started_at=_now(),
            ),
        )
        with ready_path.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(b"ready\n")
    print("stage_launched")
    return 0


def _worker(arguments: argparse.Namespace) -> int:
    root = arguments.root.resolve()
    status_path, _, ready_path, _ = _paths(root, arguments.stage)
    deadline = time.monotonic() + 30
    while not ready_path.exists():
        if time.monotonic() >= deadline:
            return 2
        time.sleep(0.05)
    ready_path.unlink(missing_ok=True)
    started = _load_status(status_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(arguments.runner.resolve()),
            arguments.stage,
            "--config",
            str(arguments.config.resolve()),
            "--root",
            str(root),
        ],
        check=False,
    )
    _write_status(
        status_path,
        StageStatus(
            stage=arguments.stage,
            attempt=arguments.attempt,
            outcome="succeeded" if completed.returncode == 0 else "failed",
            pid=os.getpid(),
            started_at=started.started_at,
            finished_at=_now(),
            exit_code=completed.returncode,
        ),
    )
    return completed.returncode


def _wait(arguments: argparse.Namespace) -> int:
    status_path, _, _, _ = _paths(arguments.root.resolve(), arguments.stage)
    while True:
        status = _load_status(status_path)
        if status.outcome == "succeeded":
            print("stage_succeeded")
            return 0
        if status.outcome == "failed":
            print("stage_failed")
            return 3
        if not _alive(status.pid):
            print("stage_worker_missing")
            return 4
        time.sleep(arguments.poll_interval)


def _status(arguments: argparse.Namespace) -> int:
    status_path, _, _, _ = _paths(arguments.root.resolve(), arguments.stage)
    print(_load_status(status_path).model_dump_json())
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    runner = Path(__file__).with_name("run_experiment.py")
    for action in ("launch", "wait", "status"):
        command = subparsers.add_parser(action)
        command.add_argument("stage", choices=("pilot", "full-or-resume"))
        command.add_argument("--root", required=True, type=Path)
        if action == "launch":
            command.add_argument("--config", required=True, type=Path)
            command.add_argument("--runner", type=Path, default=runner)
        if action == "wait":
            command.add_argument("--poll-interval", type=float, default=5.0)
    worker = subparsers.add_parser("_worker")
    worker.add_argument("stage", choices=("pilot", "full-or-resume"))
    worker.add_argument("--config", required=True, type=Path)
    worker.add_argument("--root", required=True, type=Path)
    worker.add_argument("--runner", required=True, type=Path)
    worker.add_argument("--attempt", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.action == "launch":
            return _launch(arguments)
        if arguments.action == "wait":
            if arguments.poll_interval <= 0:
                raise ValueError
            return _wait(arguments)
        if arguments.action == "status":
            return _status(arguments)
        return _worker(arguments)
    except Exception:
        print("remote_stage_error")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
