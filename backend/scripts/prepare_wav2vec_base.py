"""Prepare the exact local XLS-R 300M base without exposing local details."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.evaluation.wav2vec_base import prepare_wav2vec_base


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("wav2vec_base_preparation_failed")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        prepare_wav2vec_base(arguments.output)
    except Exception:
        print(
            "wav2vec_base_error: wav2vec_base_preparation_failed",
            file=sys.stderr,
        )
        return 2
    print("wav2vec base prepared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
