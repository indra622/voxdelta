"""Normalize the licensed AI Hub 263 emotion release for local VoxDelta training."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from zipfile import BadZipFile

from voxdelta.evaluation.aihub_emotion import import_emotion_dataset


def _non_negative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-missing-audio", type=_non_negative_int, default=0)
    parser.add_argument("--max-orphan-audio", type=_non_negative_int, default=0)
    arguments = parser.parse_args(argv)
    try:
        import_emotion_dataset(
            arguments.source_root,
            arguments.output_root,
            max_missing_audio=arguments.max_missing_audio,
            max_orphan_audio=arguments.max_orphan_audio,
        )
    except (BadZipFile, OSError, ValueError):
        print("emotion import failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
