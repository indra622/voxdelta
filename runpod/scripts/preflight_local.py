"""Run no-audio local identity, authorization-presence, and clean-code checks."""

from __future__ import annotations

import argparse
import stat
import subprocess
from collections.abc import Sequence
from pathlib import Path

from voxdelta_runpod.config import load_experiment_config
from voxdelta_runpod.package import derive_holdout_identity


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--exposed-manifest", required=True, type=Path)
    parser.add_argument("--authorization-marker", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        marker = arguments.authorization_marker.resolve(strict=True)
        mode = marker.stat(follow_symlinks=False).st_mode
        if marker.is_symlink() or not stat.S_ISREG(mode) or mode & 0o077:
            raise ValueError
        config = load_experiment_config(arguments.config.resolve())
        derive_holdout_identity(
            arguments.manifest.resolve(), arguments.exposed_manifest.resolve(), config
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=arguments.repository.resolve(),
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode != 0 or status.stdout or status.stderr:
            raise ValueError
    except Exception:
        print("local_preflight_error")
        return 2
    print("local_preflight_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
