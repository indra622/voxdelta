"""Package one already-validated local model input for private transfer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from voxdelta_runpod.model_bundle import ModelBundleError, build_model_bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("xls-r-base", "emotion2vec-baseline"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        build_model_bundle(arguments.source.resolve(), arguments.output.resolve(), arguments.kind)
    except (OSError, ModelBundleError, ValueError):
        print("model_bundle_error")
        return 2
    print("model_bundle_ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
