"""Build a deterministic balanced real-data emotion smoke manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.evaluation.emotion_experiment import build_stratified_smoke_manifest


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("positive integer required") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError("positive integer required")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-per-label", type=_positive_int, default=20)
    parser.add_argument("--validation-per-label", type=_positive_int, default=5)
    parser.add_argument("--test-per-label", type=_positive_int, default=5)
    parser.add_argument("--seed", type=_positive_int, default=622)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        summary = build_stratified_smoke_manifest(
            arguments.manifest,
            arguments.output,
            train_per_label=arguments.train_per_label,
            validation_per_label=arguments.validation_per_label,
            test_per_label=arguments.test_per_label,
            seed=arguments.seed,
        )
    except Exception:
        print("smoke manifest failed", file=sys.stderr)
        return 2
    print(f"smoke manifest: {summary.total_count} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
