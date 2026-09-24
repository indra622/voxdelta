"""Derive the local-only KCSC per-speaker track set from the pinned raw corpus.

The tracks are cut onto the same trimmed timeline as the mixed evaluation set, so a
per-track second and a mixed second are the same second and the two benchmarks measure
one thing apart: the diarization in the path. Runs under the network egress guard so a
completed run is positive evidence that nothing left the host, and prints identifiers,
counts, and checksums only — never transcript text.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.evaluation.kcsc_derivation import KcscDerivationError
from voxdelta.evaluation.kcsc_track_derivation import BENCHMARK_CONVERSATIONS, derive_track_set
from voxdelta.providers.offline_guard import block_network_egress

DEFAULT_SOURCE = Path("data/raw/hf/kcsc")
DEFAULT_MIXED = Path("data/derived/kcsc")
DEFAULT_OUTPUT = Path("data/derived/kcsc-tracks")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--mixed",
        type=Path,
        default=DEFAULT_MIXED,
        help="the derived mixed evaluation set whose timeline and reference rows are matched",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing derived track set instead of refusing to run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("track derivation failed: invalid arguments", file=sys.stderr)
        return 2

    output: Path = arguments.output
    if (output / "manifest.json").exists() and not arguments.force:
        print(
            f"track derivation refused: {output / 'manifest.json'} already exists (pass --force)",
            file=sys.stderr,
        )
        return 2

    print("preflight: KCSC per-speaker track derivation")
    print(f"  conversations   : {', '.join(BENCHMARK_CONVERSATIONS)}")
    print(f"  mixed set       : {arguments.mixed}")
    print("  egress          : guarded")

    try:
        with block_network_egress() as egress:
            summary = derive_track_set(arguments.source, arguments.mixed, output)
    except KcscDerivationError as error:
        print(f"track derivation failed: {error}", file=sys.stderr)
        return 2
    except Exception:
        print("track derivation failed: unexpected error", file=sys.stderr)
        return 2

    if egress.attempted:
        print(
            f"track derivation failed: {egress.attempts} network egress attempts",
            file=sys.stderr,
        )
        return 2

    for track in summary.tracks:
        print(
            f"{track.conversation_id}_{track.speaker} "
            f"duration={track.derived_duration_seconds:.3f}s "
            f"trim={track.trim_offset_seconds:.3f}s "
            f"gain={track.gain:.6f} "
            f"turns={track.turn_count} "
            f"events={track.unattributed_event_count} "
            f"silenced={len(track.silenced_intervals)} "
            f"audio_sha256={track.audio_sha256}"
        )
    print(
        f"derived {summary.track_count} tracks over "
        f"{len(BENCHMARK_CONVERSATIONS)} conversations, "
        f"{summary.turn_count} turns, "
        f"{summary.total_duration_seconds:.1f}s of audio"
    )
    print(f"source revision: {summary.provenance.revision}")
    print(f"manifest: {summary.manifest_path} sha256={summary.manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
