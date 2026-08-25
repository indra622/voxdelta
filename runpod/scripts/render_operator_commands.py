"""Render checksum-covered manual RunPod command packets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from voxdelta_runpod.commands import (
    CommandPacketError,
    render_operator_commands,
    update_checksums,
)
from voxdelta_runpod.config import load_experiment_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "experiment.toml"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args(argv)
    try:
        output = arguments.output.resolve()
        runtime = load_experiment_config(arguments.config.resolve()).runtime
        paths = render_operator_commands(
            output,
            run_id=arguments.run_id,
            final=arguments.final,
            volume_root=runtime.volume_root,
            model_root=runtime.model_root,
        )
        update_checksums(output, paths)
    except (CommandPacketError, OSError, ValueError):
        print("command_packet_error")
        return 2
    print("command_packet_ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
