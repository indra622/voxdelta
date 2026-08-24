"""Verify and install one transferred model input beneath its pinned remote path."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from voxdelta_runpod.model_bundle import ModelBundleError, extract_model_bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("xls-r-base", "emotion2vec-baseline"))
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        extract_model_bundle(
            arguments.archive.resolve(),
            arguments.sidecar.resolve(),
            arguments.target.resolve(),
            expected_kind=arguments.kind,
        )
    except (OSError, ModelBundleError, ValueError):
        print("model_bundle_extract_error")
        return 2
    print("model_bundle_extract_ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
