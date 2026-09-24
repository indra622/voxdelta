"""Derive the local-only KCSC evaluation set from the pinned raw corpus.

Runs under the network egress guard so a completed run is positive evidence that
nothing left the host, and prints identifiers, counts, and checksums only — never
transcript text.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.evaluation.kcsc_derivation import KcscDerivationError, derive_evaluation_set
from voxdelta.providers.offline_guard import block_network_egress

DEFAULT_SOURCE = Path("data/raw/hf/kcsc")
DEFAULT_OUTPUT = Path("data/derived/kcsc")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing derived set instead of refusing to run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("derivation failed: invalid arguments", file=sys.stderr)
        return 2

    output: Path = arguments.output
    if (output / "manifest.json").exists() and not arguments.force:
        print(
            f"derivation refused: {output / 'manifest.json'} already exists (pass --force)",
            file=sys.stderr,
        )
        return 2

    try:
        with block_network_egress() as egress:
            summary = derive_evaluation_set(arguments.source, output)
    except KcscDerivationError as error:
        print(f"derivation failed: {error}", file=sys.stderr)
        return 2
    except Exception:
        print("derivation failed: unexpected error", file=sys.stderr)
        return 2

    if egress.attempted:
        print(f"derivation failed: {egress.attempts} network egress attempts", file=sys.stderr)
        return 2

    for conversation in summary.conversations:
        print(
            f"{conversation.conversation_id} "
            f"speakers={'+'.join(conversation.speakers)} "
            f"duration={conversation.derived_duration_seconds:.3f}s "
            f"trim={conversation.trim_offset_seconds:.3f}s "
            f"gain={conversation.mix_gain:.6f} "
            f"turns={conversation.turn_count} "
            f"events={conversation.unattributed_event_count} "
            f"silenced={len(conversation.silenced_intervals)} "
            f"audio_sha256={conversation.audio_sha256}"
        )
    print(
        f"derived {summary.conversation_count} conversations, "
        f"{summary.turn_count} turns, "
        f"{summary.unattributed_event_count} unattributed events, "
        f"{summary.vendor_marker_count} vendor markers, "
        f"{summary.silenced_interval_count} intervals silenced"
    )
    print(f"source revision: {summary.provenance.revision}")
    print(f"manifest: {summary.manifest_path} sha256={summary.manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
