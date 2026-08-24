"""Freeze one passing full-validation candidate and publish its final capability."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from voxdelta.evaluation.manifest import read_trusted_regular_file

from voxdelta_runpod.config import load_experiment_config
from voxdelta_runpod.gates import freeze_candidate, publish_frozen_candidate
from voxdelta_runpod.package import HoldoutIdentity
from voxdelta_runpod.workflow import digest_file, load_aggregate_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--holdout-identity", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint-sha256", required=True)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--metric-schema", required=True, type=Path)
    parser.add_argument("--decision-rule", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        config = load_experiment_config(arguments.config.resolve())
        report = load_aggregate_report(arguments.validation_report.resolve())
        identity = HoldoutIdentity.model_validate_json(
            read_trusted_regular_file(arguments.holdout_identity.resolve())
        )
        candidate = freeze_candidate(
            report,
            config,
            baseline_checkpoint_sha256=arguments.baseline_checkpoint_sha256,
            baseline_config_sha256=digest_file(arguments.baseline_config.resolve()),
            metric_schema_sha256=digest_file(arguments.metric_schema.resolve()),
            decision_rule_sha256=digest_file(arguments.decision_rule.resolve()),
            holdout_identity_sha256=identity.identity_sha256,
        )
        publish_frozen_candidate(arguments.output.resolve(), candidate)
    except Exception:
        print("candidate_freeze_error")
        return 2
    print("candidate_frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
