"""Evaluate a local seven-emotion checkpoint without item-level output."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Never, cast

from voxdelta.evaluation.emotion_experiment import (
    evaluate_emotion_checkpoint,
    write_experiment_report,
)
from voxdelta.providers._emotion_runtime import Device

Architecture = Literal["emotion2vec-plus", "wav2vec-xls-r"]
EvaluationSplit = Literal["validation", "test"]


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--architecture",
        required=True,
        choices=("emotion2vec-plus", "wav2vec-xls-r"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        report = evaluate_emotion_checkpoint(
            arguments.manifest,
            arguments.checkpoint,
            architecture=cast(Architecture, arguments.architecture),
            device=cast(Device, arguments.device),
            split=cast(EvaluationSplit, arguments.split),
        )
        write_experiment_report(arguments.output, report)
    except Exception:
        print("emotion evaluation failed", file=sys.stderr)
        return 2
    print(f"emotion evaluation: {report.completed_count}/{report.item_count} completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
