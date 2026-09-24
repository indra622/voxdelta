"""Run one long local inference as a supervised job with a durable state record.

A long benchmark that is launched and forgotten leaves no way to tell "still running"
from "died twenty minutes ago", and a child that detaches itself can make the launching
session look finished while the real work is still going. This wrapper closes both:

* The child is started in **its own process group** and the group id is recorded, so the
  whole tree can be found and signalled later even if the wrapper's own parent is gone.
* The wrapper **waits for the child** and only then writes the terminal state. It never
  detaches and never exits while the job is alive, so "wrapper exited" and "job reached a
  terminal state" are the same event.
* The state file moves ``running`` -> ``completed`` | ``failed`` exactly once, and a
  ``failed`` record always names why: a non-zero exit, a signal, or a declared output
  artifact that the command did not produce.

Inputs and outputs are hashed into the record, so a state file identifies which manifest
the run consumed and which artifact it produced, rather than merely asserting success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Never

DEFAULT_STATE_DIR = Path("data/jobs/benchmarks")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_state(path: Path, payload: dict[str, object]) -> None:
    """Write the state record atomically, so a reader never sees a half-written file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    temporary.write_text(f"{raw}\n", encoding="utf-8")
    temporary.replace(path)


def pid_alive(pid: int) -> bool:
    """Signal 0 asks the kernel whether the pid exists without disturbing it."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reconcile(state_dir: Path) -> list[str]:
    """Close out records left at ``running`` by a wrapper that was killed outright.

    A SIGKILL leaves no chance to write anything, so the state file would otherwise claim
    a job is running long after its process is gone. Reconciliation is deliberately
    conservative: it only touches records whose recorded pid no longer exists, and it
    marks them failed rather than guessing at an outcome the wrapper never observed.
    """

    closed: list[str] = []
    for path in sorted(state_dir.glob("*.state.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(record, dict) or record.get("state") != "running":
            continue
        pid = record.get("pid")
        if isinstance(pid, int) and pid_alive(pid):
            continue
        record["state"] = "failed"
        record["reason"] = "process disappeared without recording a terminal state"
        record["finished_at"] = _now()
        record["reconciled"] = True
        write_state(path, record)
        closed.append(str(record.get("job", path.stem)))
    return closed


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="job name; also the state file stem")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="input file to hash into the record; repeatable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="artifact the command must produce; its absence fails the job",
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="close out stale running records in the state dir, then exit",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("job failed: invalid arguments", file=sys.stderr)
        return 2

    if arguments.reconcile:
        closed = reconcile(arguments.state_dir)
        print(f"reconciled {len(closed)} stale record(s): {', '.join(closed) or 'none'}")
        return 0

    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("job failed: no command given", file=sys.stderr)
        return 2

    state_dir: Path = arguments.state_dir
    state_path = state_dir / f"{arguments.job}.state.json"
    log_path = state_dir / f"{arguments.job}.log"
    state_dir.mkdir(parents=True, exist_ok=True)

    inputs: list[dict[str, object]] = []
    for path in arguments.input:
        if not path.is_file():
            print(f"job failed: missing declared input {path}", file=sys.stderr)
            write_state(
                state_path,
                {
                    "job": arguments.job,
                    "state": "failed",
                    "reason": f"missing declared input: {path}",
                    "command": command,
                    "finished_at": _now(),
                },
            )
            return 2
        inputs.append({"path": str(path), "sha256": file_sha256(path)})

    output: Path = arguments.output
    record: dict[str, object] = {
        "job": arguments.job,
        "state": "running",
        "command": command,
        "cwd": str(Path.cwd()),
        "wrapper_pid": os.getpid(),
        "log": str(log_path),
        "inputs": inputs,
        "output": {"path": str(output), "sha256": None},
        "started_at": _now(),
        "finished_at": None,
        "exit_code": None,
        "reason": None,
    }

    started = time.monotonic()
    with log_path.open("wb") as log:
        # start_new_session puts the child in its own process group, so the whole tree is
        # addressable by -pgid even after this wrapper is gone.
        process = subprocess.Popen(  # noqa: S603
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record["pid"] = process.pid
        try:
            record["pgid"] = os.getpgid(process.pid)
        except ProcessLookupError:
            record["pgid"] = None
        write_state(state_path, record)
        print(f"job {arguments.job}: pid={process.pid} pgid={record['pgid']} log={log_path}")

        # A wrapper that is asked to stop must take the job down with it and still write a
        # terminal state; otherwise the record claims "running" for a process that is gone.
        received: list[str] = []

        def _stop(number: int, _frame: object) -> None:
            received.append(signal.Signals(number).name)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        previous = {
            number: signal.signal(number, _stop)
            for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        }
        try:
            exit_code = process.wait()
        except KeyboardInterrupt:
            received.append("SIGINT")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            exit_code = process.wait()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)

    elapsed = time.monotonic() - started
    record["finished_at"] = _now()
    record["elapsed_seconds"] = round(elapsed, 3)
    record["exit_code"] = exit_code

    if exit_code != 0:
        record["state"] = "failed"
        if received:
            record["reason"] = f"wrapper received {received[0]}; job group terminated"
        elif exit_code < 0:
            record["reason"] = f"terminated by signal {-exit_code}"
        else:
            record["reason"] = f"exit code {exit_code}"
    elif not output.is_file():
        record["state"] = "failed"
        record["reason"] = f"command succeeded but produced no artifact at {output}"
    else:
        record["state"] = "completed"
        record["output"] = {"path": str(output), "sha256": file_sha256(output)}

    write_state(state_path, record)
    print(
        f"job {arguments.job}: {record['state']} in {elapsed:.1f}s "
        f"(exit={exit_code}) state={state_path}"
    )
    return 0 if record["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
