"""Prove a target filesystem honours the private 0700/0600 contract before any transfer.

RunPod backs some regional volumes with a network filesystem that silently ignores
requested permission bits. Licensed audio, the base model, and the final holdout must
never land on such a mount, so this runs first and fails closed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import tempfile
from collections.abc import Sequence
from pathlib import Path


class FilesystemProbeError(ValueError):
    """Stable failure for a filesystem that cannot hold private modes."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)


def probe_path(target: Path, *, min_free_gb: float = 0.0) -> None:
    if not target.is_absolute() or target.is_symlink():
        raise FilesystemProbeError("invalid_probe_target")
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    probe = Path(tempfile.mkdtemp(prefix=".voxdelta-probe-", dir=target))
    try:
        os.chmod(probe, 0o700)
        if _mode(probe) != 0o700:
            raise FilesystemProbeError("directory_mode_not_honored")
        sample = probe / "sample"
        with sample.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(b"voxdelta-probe\n")
        os.chmod(sample, 0o600)
        if _mode(sample) != 0o600:
            raise FilesystemProbeError("file_mode_not_honored")
        if shutil.disk_usage(probe).free < min_free_gb * 1024**3:
            raise FilesystemProbeError("insufficient_free_space")
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, action="append", type=Path)
    parser.add_argument("--min-free-gb", type=float, default=0.0)
    arguments = parser.parse_args(argv)
    try:
        for target in arguments.path:
            probe_path(target, min_free_gb=arguments.min_free_gb)
    except Exception:
        print("filesystem_probe_error")
        return 2
    print("filesystem_probe_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
