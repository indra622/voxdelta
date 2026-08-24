"""Verify retrieved aggregate packets, immutable ledger, privacy, and deletion readiness."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from voxdelta.evaluation.manifest import read_trusted_regular_file
from voxdelta.providers._emotion_runtime import validate_checkpoint

from voxdelta_runpod.config import load_experiment_config
from voxdelta_runpod.gates import FinalDecision, decide_final_comparison, full_validation_eligible
from voxdelta_runpod.ledger import append_transition, load_ledger
from voxdelta_runpod.workflow import (
    load_aggregate_report,
    load_run_identity,
    no_sensitive_result_json,
    verify_result_checksums,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("full", "final"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        config = load_experiment_config(arguments.config.resolve())
        result_root = arguments.result_root.resolve()
        ledger_root = arguments.ledger.resolve()
        identity = load_run_identity(arguments.identity.resolve())
        records = load_ledger(ledger_root)
        if identity != records[-1].identity or identity.config_sha256 != config.digest():
            raise ValueError
        verify_result_checksums(result_root)
        if arguments.stage == "full":
            if records[-1].to_stage != "full":
                raise ValueError
            report_path = result_root / "report.json"
            report = load_aggregate_report(report_path)
            if not full_validation_eligible(report, config):
                raise ValueError
            checkpoint = validate_checkpoint(
                result_root / "checkpoint",
                architecture="wav2vec-xls-r",
                model_id=config.model.model_id,
            )
            if checkpoint.digest != report.checkpoint_sha256 or not no_sensitive_result_json(
                (report_path,)
            ):
                raise ValueError
            outcome = "retrieved_results_verified"
        else:
            if records[-1].to_stage != "final":
                raise ValueError
            xls_path = result_root / "xls-r-report.json"
            baseline_path = result_root / "baseline-report.json"
            decision_path = result_root / "decision.json"
            xls = load_aggregate_report(xls_path)
            baseline = load_aggregate_report(baseline_path)
            decision = FinalDecision.model_validate_json(read_trusted_regular_file(decision_path))
            expected_decision = decide_final_comparison(xls, baseline, config)
            result_privacy = no_sensitive_result_json((xls_path, baseline_path, decision_path))
            if decision != expected_decision or not result_privacy:
                raise ValueError
            append_transition(ledger_root, "retrieved", identity)
            append_transition(ledger_root, "deletion-ready", identity)
            outcome = "deletion_ready"
    except Exception:
        print("result_verification_error")
        return 2
    print(outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
