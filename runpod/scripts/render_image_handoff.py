"""Render the verified, credential-free Docker image handoff packet."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from voxdelta.evaluation.manifest import read_trusted_regular_file

from voxdelta_runpod.image import ImageHandoffError, publish_image_handoff


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oci-archive", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--lock", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        lock_sha256 = hashlib.sha256(
            read_trusted_regular_file(arguments.lock.resolve())
        ).hexdigest()
        publish_image_handoff(
            arguments.oci_archive.resolve(),
            arguments.output_root.resolve(),
            run_id=arguments.run_id,
            git_commit=arguments.git_commit,
            lock_sha256=lock_sha256,
        )
    except (ImageHandoffError, OSError, ValueError):
        print("image_handoff_error", flush=True)
        return 2
    print("image_handoff_ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
