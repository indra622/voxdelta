"""Assemble the immutable XLS-R release bundle from already-verified local artifacts."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from voxdelta_runpod.model_bundle import extract_model_bundle
from voxdelta_runpod.release import build_release_bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--base-archive", required=True, type=Path)
    parser.add_argument("--base-sidecar", required=True, type=Path)
    parser.add_argument("--frozen-candidate", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--final-report", required=True, type=Path)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-id", required=True)
    arguments = parser.parse_args(argv)

    output = arguments.output.resolve()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".release-base-", dir=output.parent))
    try:
        base_dir = extract_model_bundle(
            arguments.base_archive.resolve(),
            arguments.base_sidecar.resolve(),
            scratch / "base-model",
            expected_kind="xls-r-base",
        )
        build_release_bundle(
            checkpoint_dir=arguments.checkpoint_dir.absolute(),
            base_model_dir=base_dir,
            frozen_candidate=arguments.frozen_candidate.resolve(),
            validation_report=arguments.validation_report.resolve(),
            final_report=arguments.final_report.resolve(),
            baseline_report=arguments.baseline_report.resolve(),
            decision=arguments.decision.resolve(),
            experiment_config=arguments.config.resolve(),
            output_dir=output,
            release_id=arguments.release_id,
        )
    except Exception:
        print("release_build_error")
        return 2
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("release_built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
