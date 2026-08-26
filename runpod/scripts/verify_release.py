"""Re-verify an immutable XLS-R release bundle's integrity and reload prerequisites."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from voxdelta_runpod.release import verify_release_bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        manifest = verify_release_bundle(arguments.bundle.absolute())
    except Exception:
        print("release_verification_failed")
        return 2
    print("release_verified")
    print(f"release_id {manifest.release_id}")
    print(f"candidate_checkpoint_sha256 {manifest.candidate_checkpoint_sha256}")
    print(f"bundle_tree_sha256 {manifest.bundle_tree_sha256}")
    print(f"payload_count {len(manifest.payloads)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
