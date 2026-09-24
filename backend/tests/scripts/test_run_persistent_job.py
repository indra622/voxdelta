"""Tests for the persistent job wrapper.

The wrapper's whole value is that its state file can be trusted after the fact, so what is
pinned here is exactly that: a terminal state is always written, it is written only after
the child is really gone, and success is not claimed for a run that produced no artifact.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_persistent_job",
    Path(__file__).resolve().parents[2] / "scripts" / "run_persistent_job.py",
)
assert _SPEC and _SPEC.loader
run_persistent_job = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_persistent_job)


def _state(state_dir: Path, job: str) -> dict[str, Any]:
    return json.loads((state_dir / f"{job}.state.json").read_text(encoding="utf-8"))


def test_a_successful_run_records_completed_with_the_output_digest(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    code = run_persistent_job.main(
        [
            "--job",
            "ok",
            "--state-dir",
            str(tmp_path / "state"),
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            f"open({str(output)!r}, 'w').write('{{}}')",
        ]
    )

    assert code == 0
    record = _state(tmp_path / "state", "ok")
    assert record["state"] == "completed"
    assert record["exit_code"] == 0
    assert record["output"]["sha256"] == run_persistent_job.file_sha256(output)
    assert record["finished_at"] is not None


def test_a_nonzero_exit_records_failed_and_never_claims_an_artifact(tmp_path: Path) -> None:
    code = run_persistent_job.main(
        [
            "--job",
            "boom",
            "--state-dir",
            str(tmp_path / "state"),
            "--output",
            str(tmp_path / "artifact.json"),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(3)",
        ]
    )

    assert code == 1
    record = _state(tmp_path / "state", "boom")
    assert record["state"] == "failed"
    assert record["exit_code"] == 3
    assert record["reason"] == "exit code 3"
    assert record["output"]["sha256"] is None


def test_a_command_that_exits_clean_but_writes_nothing_is_a_failure(tmp_path: Path) -> None:
    """Exit zero is not evidence; the artifact is."""

    code = run_persistent_job.main(
        [
            "--job",
            "empty",
            "--state-dir",
            str(tmp_path / "state"),
            "--output",
            str(tmp_path / "artifact.json"),
            "--",
            sys.executable,
            "-c",
            "pass",
        ]
    )

    assert code == 1
    record = _state(tmp_path / "state", "empty")
    assert record["state"] == "failed"
    assert "produced no artifact" in str(record["reason"])


def test_the_record_carries_the_process_group_and_the_input_digests(tmp_path: Path) -> None:
    source = tmp_path / "manifest.json"
    source.write_text('{"tracks": []}', encoding="utf-8")
    output = tmp_path / "artifact.json"

    run_persistent_job.main(
        [
            "--job",
            "ids",
            "--state-dir",
            str(tmp_path / "state"),
            "--input",
            str(source),
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            f"open({str(output)!r}, 'w').write('x')",
        ]
    )

    record = _state(tmp_path / "state", "ids")
    assert record["pid"] > 0
    # Its own group, so the whole tree stays addressable after the wrapper is gone.
    assert record["pgid"] == record["pid"] != os.getpgid(0)
    assert record["inputs"] == [
        {"path": str(source), "sha256": run_persistent_job.file_sha256(source)}
    ]
    assert record["log"].endswith("ids.log")


def test_a_missing_declared_input_fails_before_the_command_runs(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"

    code = run_persistent_job.main(
        [
            "--job",
            "noinput",
            "--state-dir",
            str(tmp_path / "state"),
            "--input",
            str(tmp_path / "absent.json"),
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            f"open({str(output)!r}, 'w').write('x')",
        ]
    )

    assert code == 2
    assert _state(tmp_path / "state", "noinput")["state"] == "failed"
    assert not output.exists()


def test_the_child_output_lands_in_the_log(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"

    run_persistent_job.main(
        [
            "--job",
            "logged",
            "--state-dir",
            str(tmp_path / "state"),
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            f"print('hello from the job'); open({str(output)!r}, 'w').write('x')",
        ]
    )

    assert "hello from the job" in (tmp_path / "state" / "logged.log").read_text(encoding="utf-8")


def test_the_wrapper_waits_so_a_terminal_state_means_the_child_is_gone(tmp_path: Path) -> None:
    """A wrapper that returned early would leave 'completed' next to a live process."""

    output = tmp_path / "artifact.json"

    run_persistent_job.main(
        [
            "--job",
            "slow",
            "--state-dir",
            str(tmp_path / "state"),
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            f"import time; time.sleep(0.4); open({str(output)!r}, 'w').write('x')",
        ]
    )

    record = _state(tmp_path / "state", "slow")
    assert record["state"] == "completed"
    assert record["elapsed_seconds"] >= 0.4
    with pytest.raises(ProcessLookupError):
        os.kill(int(record["pid"]), 0)


def test_a_wrapper_killed_by_sigterm_still_records_a_terminal_state(tmp_path: Path) -> None:
    """The gap this closes: a killed wrapper used to leave the record stuck at 'running'."""

    state_dir = tmp_path / "state"
    output = tmp_path / "artifact.json"
    wrapper = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            str(Path(run_persistent_job.__file__)),
            "--job",
            "signalled",
            "--state-dir",
            str(state_dir),
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    state_path = state_dir / "signalled.state.json"
    for _ in range(200):
        if state_path.is_file() and _state(state_dir, "signalled")["state"] == "running":
            break
        time.sleep(0.05)

    child_pid = int(_state(state_dir, "signalled")["pid"])
    wrapper.terminate()
    wrapper.wait(timeout=30)

    record = _state(state_dir, "signalled")
    assert record["state"] == "failed"
    assert "SIGTERM" in str(record["reason"])
    assert record["finished_at"] is not None
    # The job must not outlive the wrapper that was supervising it.
    for _ in range(100):
        if not run_persistent_job.pid_alive(child_pid):
            break
        time.sleep(0.05)
    assert not run_persistent_job.pid_alive(child_pid)


def test_reconcile_closes_a_record_whose_process_is_gone(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    run_persistent_job.write_state(
        state_dir / "orphan.state.json",
        {"job": "orphan", "state": "running", "pid": 2**22 - 1, "finished_at": None},
    )

    closed = run_persistent_job.reconcile(state_dir)

    assert closed == ["orphan"]
    record = _state(state_dir, "orphan")
    assert record["state"] == "failed"
    assert record["reconciled"] is True
    assert "without recording a terminal state" in str(record["reason"])


def test_reconcile_leaves_a_live_job_alone(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    run_persistent_job.write_state(
        state_dir / "live.state.json",
        {"job": "live", "state": "running", "pid": os.getpid(), "finished_at": None},
    )

    assert run_persistent_job.reconcile(state_dir) == []
    assert _state(state_dir, "live")["state"] == "running"


def test_reconcile_does_not_touch_terminal_records(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    run_persistent_job.write_state(
        state_dir / "done.state.json",
        {"job": "done", "state": "completed", "pid": 2**22 - 1, "reason": None},
    )

    assert run_persistent_job.reconcile(state_dir) == []
    assert _state(state_dir, "done")["state"] == "completed"
