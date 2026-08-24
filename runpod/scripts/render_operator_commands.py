"""Render checksum-covered manual RunPod command packets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from voxdelta_runpod.commands import CommandPacketError, render_operator_commands, update_checksums


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--final", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        output = arguments.output.resolve()
        paths = render_operator_commands(output, run_id=arguments.run_id, final=arguments.final)
        update_checksums(output, paths)
    except (CommandPacketError, OSError, ValueError):
        print("command_packet_error")
        return 2
    print("command_packet_ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
