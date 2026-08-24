from __future__ import annotations

import os
import subprocess
from pathlib import Path

from voxdelta_runpod.commands import render_operator_commands, update_checksums


def _executable(path: Path) -> Path:
    path.write_text("#!/usr/bin/env bash\nexit 0\n")
    path.chmod(0o700)
    return path


def test_command_packets_are_stage_separated_syntax_valid_and_fake_rehearsed(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "handoff").resolve()
    root.mkdir(mode=0o700)
    (root / "train-validation.tar.zst").write_bytes(b"archive")
    (root / "train-validation.sidecar.json").write_text("{}")
    (root / "image-digest.txt").write_text(f"sha256:{'a' * 64}\n")
    scripts = render_operator_commands(root, run_id="run-622")
    update_checksums(
        root,
        (
            *scripts,
            root / "train-validation.tar.zst",
            root / "train-validation.sidecar.json",
            root / "image-digest.txt",
        ),
    )

    assert {path.name for path in scripts} == {
        "01-preflight-and-pilot.sh",
        "02-full-or-resume.sh",
        "03-download-results.sh",
    }
    assert all(
        subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0 for path in scripts
    )
    combined = "\n".join(path.read_text() for path in scripts)
    assert "<temporary-host>" not in combined and "VOXDELTA_RUN_ID" not in combined
    assert "/workspace/voxdelta/run-622" in combined
    assert "final-holdout" not in combined
    assert "pilot" in (root / "01-preflight-and-pilot.sh").read_text()
    assert "full-or-resume" not in (root / "01-preflight-and-pilot.sh").read_text()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "ssh")
    _executable(fake_bin / "rsync")
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    completed = subprocess.run(
        [str(root / "01-preflight-and-pilot.sh"), "root", "example.invalid", "22"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    final_root = (root / "final").resolve()
    final_scripts = render_operator_commands(final_root, run_id="run-622", final=True)
    assert {path.name for path in final_scripts} == {"04-final-once.sh", "05-download-final.sh"}
    assert all(
        "pilot" not in path.read_text() and "full-or-resume" not in path.read_text()
        for path in final_scripts
    )
