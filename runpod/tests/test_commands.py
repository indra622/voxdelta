from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from voxdelta_runpod.commands import (
    CommandPacketError,
    render_operator_commands,
    update_checksums,
)


def _executable(path: Path) -> Path:
    path.write_text("#!/usr/bin/env bash\nexit 0\n")
    path.chmod(0o700)
    return path


def _fake_rsync(path: Path) -> Path:
    """Stand in for rsync, including its refusal to create a missing parent."""
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'destination="${@: -1}"\n'
        'if [[ "$destination" == *:* ]]; then exit 0; fi\n'
        'parent="$(dirname "$destination")"\n'
        '[[ -d "$parent" ]] || { echo "rsync: mkdir failed: $destination" >&2; exit 1; }\n'
        'if [[ "$destination" == */ ]]; then mkdir -p "$destination"; fi\n'
    )
    path.chmod(0o700)
    return path


def test_command_packets_are_stage_separated_syntax_valid_and_fake_rehearsed(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "handoff").resolve()
    root.mkdir(mode=0o700)
    (root / "train-validation.tar.zst").write_bytes(b"archive")
    (root / "train-validation.sidecar.json").write_text("{}")
    (root / "xls-r-base.tar.zst").write_bytes(b"base")
    (root / "xls-r-base.sidecar.json").write_text("{}")
    (root / "image-digest.txt").write_text(f"sha256:{'a' * 64}\n")
    scripts = render_operator_commands(root, run_id="run-622")
    update_checksums(
        root,
        (
            *scripts,
            root / "train-validation.tar.zst",
            root / "train-validation.sidecar.json",
            root / "xls-r-base.tar.zst",
            root / "xls-r-base.sidecar.json",
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
    assert "/opt/voxdelta-run/runs/run-622" in combined
    assert "final-holdout" not in combined
    assert "pilot" in (root / "01-preflight-and-pilot.sh").read_text()
    assert "full-or-resume" not in (root / "01-preflight-and-pilot.sh").read_text()
    assert "extract_model_bundle.py" in (root / "01-preflight-and-pilot.sh").read_text()
    assert "state/checkpoints" in (root / "03-download-results.sh").read_text()
    assert "full-retrieval.tar.zst" in (root / "03-download-results.sh").read_text()
    assert "run_remote_stage.py" in combined

    # rsync --archive implies -o/-g; the RunPod receiver cannot chown on a
    # network mount, which aborted the transfer with code 23.
    rsync_lines = [line for line in combined.splitlines() if line.startswith("rsync ")]
    assert rsync_lines
    assert all(
        "--no-owner" in line and "--no-group" in line and "--chmod=F600,D700" in line
        for line in rsync_lines
    )

    # the capability probe must gate the first licensed-audio transfer
    stage01 = (root / "01-preflight-and-pilot.sh").read_text()
    assert "probe_filesystem.py" in stage01
    assert stage01.index("probe_filesystem.py") < stage01.index("train-validation.tar.zst")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "ssh")
    _fake_rsync(fake_bin / "rsync")
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
    assert (root / "results" / "pilots").is_dir()
    assert (root / "results" / "pilots").stat().st_mode & 0o077 == 0

    final_root = (root / "final").resolve()
    final_root.mkdir(mode=0o700)
    (final_root / "final-holdout.tar.zst").write_bytes(b"final")
    (final_root / "final-holdout.sidecar.json").write_text("{}")
    (final_root / "frozen-candidate.json").write_text("{}")
    (final_root / "emotion2vec-baseline.tar.zst").write_bytes(b"baseline")
    (final_root / "emotion2vec-baseline.sidecar.json").write_text("{}")
    final_scripts = render_operator_commands(final_root, run_id="run-622", final=True)
    assert {path.name for path in final_scripts} == {"04-final-once.sh", "05-download-final.sh"}
    assert all(
        "pilot" not in path.read_text() and "full-or-resume" not in path.read_text()
        for path in final_scripts
    )
    final_combined = "\n".join(path.read_text() for path in final_scripts)
    assert "emotion2vec-baseline" in final_combined
    assert "final-consumed.json" in final_combined
    assert "final-retrieval.tar.zst" in final_combined


def test_command_packets_honour_configured_remote_roots(tmp_path: Path) -> None:
    root = (tmp_path / "handoff").resolve()
    root.mkdir(mode=0o700)
    scripts = render_operator_commands(
        root,
        run_id="run-622",
        volume_root="/mnt/secure/voxdelta",
        model_root="/mnt/secure/models",
    )
    combined = "\n".join(path.read_text() for path in scripts)

    assert "/mnt/secure/voxdelta/run-622" in combined
    assert "/mnt/secure/models/xls-r-300m" in combined
    assert "/workspace" not in combined
    assert all(
        subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0 for path in scripts
    )


def test_command_packets_reject_unsafe_remote_roots(tmp_path: Path) -> None:
    root = (tmp_path / "handoff").resolve()
    root.mkdir(mode=0o700)
    for bad in ("relative/path", "/has space", "/quote'injection", "/back`tick", "/dots/../up"):
        with pytest.raises(CommandPacketError, match="^invalid_command_target$"):
            render_operator_commands(root, run_id="run-622", volume_root=bad)
        with pytest.raises(CommandPacketError, match="^invalid_command_target$"):
            render_operator_commands(root, run_id="run-622", model_root=bad)
