"""Create metadata-only holdout seals or the privacy-minimized training package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from voxdelta_runpod.config import load_experiment_config
from voxdelta_runpod.package import (
    PackageError,
    build_train_validation_package,
    derive_holdout_identity,
    write_holdout_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    identity = subparsers.add_parser("holdout-identity")
    identity.add_argument("--config", required=True, type=Path)
    identity.add_argument("--manifest", required=True, type=Path)
    identity.add_argument("--exposed-manifest", required=True, type=Path)
    identity.add_argument("--output", required=True, type=Path)
    training = subparsers.add_parser("training")
    training.add_argument("--config", required=True, type=Path)
    training.add_argument("--manifest", required=True, type=Path)
    training.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = load_experiment_config(arguments.config.resolve())
        if arguments.stage == "holdout-identity":
            identity = derive_holdout_identity(
                arguments.manifest.resolve(),
                arguments.exposed_manifest.resolve(),
                config,
            )
            write_holdout_identity(arguments.output.resolve(), identity)
            print("holdout_identity_ready")
        else:
            build_train_validation_package(
                arguments.manifest.resolve(), arguments.output.resolve(), config
            )
            print("training_package_ready")
    except (OSError, PackageError, ValueError):
        print("package_error")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
