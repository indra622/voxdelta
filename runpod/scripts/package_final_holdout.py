"""Build the authorized final-only package and checksum-covered operator packet."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from voxdelta.evaluation.manifest import read_trusted_regular_file

from voxdelta_runpod.commands import render_operator_commands, update_checksums
from voxdelta_runpod.config import load_experiment_config
from voxdelta_runpod.gates import FrozenCandidate, build_authorized_final_package


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--exposed-manifest", required=True, type=Path)
    parser.add_argument("--authorization-marker", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args(argv)
    try:
        config = load_experiment_config(arguments.config.resolve())
        candidate_path = arguments.candidate.resolve()
        candidate = FrozenCandidate.model_validate_json(read_trusted_regular_file(candidate_path))
        output = arguments.output.resolve()
        build_authorized_final_package(
            arguments.manifest.resolve(),
            arguments.exposed_manifest.resolve(),
            output,
            arguments.authorization_marker.resolve(),
            candidate,
            candidate.capability_token,
            config,
        )
        frozen = output / "frozen-candidate.json"
        with frozen.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(read_trusted_regular_file(candidate_path))
        scripts = render_operator_commands(output, run_id=arguments.run_id, final=True)
        update_checksums(
            output,
            (
                output / "final-holdout.tar.zst",
                output / "final-holdout.sidecar.json",
                frozen,
                *scripts,
            ),
        )
    except Exception:
        print("final_package_error")
        return 2
    print("final_package_ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
