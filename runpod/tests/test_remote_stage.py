from __future__ import annotations

import importlib.util
import time
from pathlib import Path

REMOTE_STAGE = Path(__file__).parents[1] / "scripts" / "run_remote_stage.py"


def _module() -> object:
    spec = importlib.util.spec_from_file_location("run_remote_stage", REMOTE_STAGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detached_stage_persists_log_status_and_is_idempotent(tmp_path: Path) -> None:
    module = _module()
    root = (tmp_path / "remote").resolve()
    root.mkdir(mode=0o700)
    config = (tmp_path / "experiment.toml").resolve()
    config.write_text("test = true\n")
    runner = (tmp_path / "fake_runner.py").resolve()
    runner.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "print('fake training completed', flush=True)\n"
        "Path(sys.argv[sys.argv.index('--root') + 1], 'runner-ok').write_text('ok')\n"
    )

    launch = [
        "launch",
        "pilot",
        "--config",
        str(config),
        "--root",
        str(root),
        "--runner",
        str(runner),
    ]
    assert module.main(launch) == 0
    assert module.main(["wait", "pilot", "--root", str(root), "--poll-interval", "0.01"]) == 0
    assert module.main(launch) == 0

    status = module.StageStatus.model_validate_json(
        (root / "state" / "stages" / "pilot.json").read_bytes()
    )
    assert status.outcome == "succeeded" and status.attempt == 1 and status.exit_code == 0
    assert "fake training completed" in (root / "logs" / "pilot.log").read_text()
    assert (root / "runner-ok").read_text() == "ok"
    assert all(path.stat().st_mode & 0o077 == 0 for path in (root / "state" / "stages").iterdir())


def test_wait_reports_failed_detached_stage(tmp_path: Path) -> None:
    module = _module()
    root = (tmp_path / "remote").resolve()
    root.mkdir(mode=0o700)
    config = (tmp_path / "experiment.toml").resolve()
    config.write_text("test = true\n")
    runner = (tmp_path / "failed_runner.py").resolve()
    runner.write_text("raise SystemExit(7)\n")
    assert (
        module.main(
            [
                "launch",
                "full-or-resume",
                "--config",
                str(config),
                "--root",
                str(root),
                "--runner",
                str(runner),
            ]
        )
        == 0
    )
    deadline = time.monotonic() + 5
    while module._load_status(root / "state" / "stages" / "full.json").outcome == "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert (
        module.main(["wait", "full-or-resume", "--root", str(root), "--poll-interval", "0.01"]) == 3
    )
