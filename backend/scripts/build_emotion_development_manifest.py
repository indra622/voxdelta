"""Build the fixed train/validation-only partial-last-four development manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.evaluation.emotion_experiment import (
    build_partial_last4_development_manifest,
)


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    try:
        arguments = parser.parse_args(argv)
        summary = build_partial_last4_development_manifest(arguments.manifest, arguments.output)
    except Exception:
        print("development manifest failed", file=sys.stderr)
        return 2
    print(f"development manifest: {summary.total_count} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
