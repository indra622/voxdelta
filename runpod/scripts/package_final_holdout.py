"""Build the authorized final-only package and checksum-covered operator packet."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from voxdelta.evaluation.manifest import read_trusted_regular_file

from voxdelta_runpod.commands import render_operator_commands, update_checksums
from voxdelta_runpod.config import load_experiment_config
from voxdelta_runpod.final_restore import build_final_restore_bundle
from voxdelta_runpod.gates import FrozenCandidate, build_authorized_final_package

BASELINE_FILES = frozenset({"emotion2vec-baseline.tar.zst", "emotion2vec-baseline.sidecar.json"})
BASE_MODEL_FILES = frozenset({"xls-r-base.tar.zst", "xls-r-base.sidecar.json"})


def _copy_bundle(root: Path, output: Path, expected: frozenset[str]) -> tuple[Path, ...]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError
    files = {
        path.name: path
        for path in root.iterdir()
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
    }
    if set(files) != expected:
        raise ValueError
    copied: list[Path] = []
    for name, source in sorted(files.items()):
        read_trusted_regular_file(source)
        destination = output / name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
            destination.chmod(0o600)
        copied.append(destination)
    return tuple(copied)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--exposed-manifest", required=True, type=Path)
    parser.add_argument("--authorization-marker", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-bundle-root", required=True, type=Path)
    parser.add_argument("--base-model-bundle-root", required=True, type=Path)
    parser.add_argument("--full-result-root", required=True, type=Path)
    parser.add_argument("--full-ledger-root", required=True, type=Path)
    parser.add_argument("--run-identity", required=True, type=Path)
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
        baseline = _copy_bundle(arguments.baseline_bundle_root.resolve(), output, BASELINE_FILES)
        base_model = _copy_bundle(
            arguments.base_model_bundle_root.resolve(), output, BASE_MODEL_FILES
        )
        restore = build_final_restore_bundle(
            arguments.full_result_root.resolve(),
            arguments.full_ledger_root.resolve(),
            arguments.run_identity.resolve(),
            output,
        )
        scripts = render_operator_commands(
            output,
            run_id=arguments.run_id,
            final=True,
            volume_root=config.runtime.volume_root,
            model_root=config.runtime.model_root,
        )
        update_checksums(
            output,
            (
                output / "final-holdout.tar.zst",
                output / "final-holdout.sidecar.json",
                frozen,
                *baseline,
                *base_model,
                *restore,
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
