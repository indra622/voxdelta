"""Publish an immutable, verified local encoder bundle from a package cache snapshot.

The cache is read only: nothing is written to it and nothing is deleted from it. Only the
four files the local loader actually reads are copied out, weights are dereferenced into
real files, and the result is staged and renamed so a reader never sees a partial bundle.

No evaluation data of any kind enters a bundle: no holdout, validation audio, transcript,
item identity, or probability.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from voxdelta.providers.encoder_publication import (
    EncoderPublicationError,
    publish_encoder_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path, help="absolute cache snapshot")
    parser.add_argument("--output", required=True, type=Path, help="new absolute bundle path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        # Passed through unchanged: the contract says absolute, so a relative path
        # must fail closed with a stable code, not be resolved against the cwd.
        bundle = publish_encoder_bundle(arguments.snapshot, arguments.output)
    except EncoderPublicationError as error:
        print(error.code)
        return 2
    except Exception:
        print("encoder_bundle_error")
        return 2
    print("encoder_bundle_published")
    print(f"encoder_id {bundle.encoder_id}")
    print(f"encoder_revision {bundle.encoder_revision}")
    print(f"bundle_tree_sha256 {bundle.bundle_tree_sha256}")
    print(f"init_param_sha256 {bundle.init_param_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
