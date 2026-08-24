"""Consume the frozen capability and evaluate both providers on final holdout once."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from voxdelta.evaluation.emotion_experiment import evaluate_emotion_checkpoint
from voxdelta.evaluation.manifest import read_trusted_regular_file

from voxdelta_runpod.config import load_experiment_config
from voxdelta_runpod.gates import (
    FrozenCandidate,
    consume_final_capability,
    decide_final_comparison,
)
from voxdelta_runpod.ledger import append_transition, load_ledger
from voxdelta_runpod.package import extract_verified_package
from voxdelta_runpod.workflow import (
    aggregate_from_backend,
    digest_file,
    load_aggregate_report,
    load_packaged_manifest,
    load_run_identity,
    publish_model,
    write_evaluation_manifest,
    write_result_checksums,
)


def _cuda_peaks() -> tuple[float, float]:
    import torch

    return (
        float(torch.cuda.max_memory_allocated()) / 1024**2,
        float(torch.cuda.max_memory_reserved()) / 1024**2,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--base-model", type=Path, default=Path("/workspace/models/xls-r-300m"))
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path("/workspace/models/emotion2vec-plus"),
    )
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    try:
        import torch

        config = load_experiment_config(arguments.config.resolve())
        state = root / "state"
        results = root / "results"
        incoming = root / "incoming" / "final"
        ledger = root / "ledger"
        identity = load_run_identity(state / "run-identity.json")
        if load_ledger(ledger)[-1].to_stage != "full":
            raise ValueError
        candidate_path = incoming / "frozen-candidate.json"
        candidate = FrozenCandidate.model_validate_json(read_trusted_regular_file(candidate_path))
        full_report = load_aggregate_report(results / "full" / "report.json")
        if (
            not candidate.verify_token()
            or candidate.config_sha256 != config.digest()
            or candidate.validation_report_sha256 != full_report.digest()
            or candidate.candidate_checkpoint_sha256 != full_report.checkpoint_sha256
        ):
            raise ValueError
        append_transition(
            ledger,
            "candidate-frozen",
            identity,
            report_sha256=digest_file(candidate_path),
        )
        consume_final_capability(
            candidate_path,
            state / "final-consumed.json",
            candidate.capability_token,
        )
        final_data = extract_verified_package(
            incoming / "final-holdout.tar.zst",
            incoming / "final-holdout.sidecar.json",
            state / "final-data",
        )
        records = load_packaged_manifest(final_data / "manifest.jsonl")
        if len(records) != config.data.final_holdout_count or any(
            record.split != "test" for record in records
        ):
            raise ValueError
        manifest = write_evaluation_manifest(
            records, final_data, state / "evaluation" / "final.jsonl"
        )
        result_root = results / "final"
        result_root.mkdir(mode=0o700, parents=True, exist_ok=False)

        torch.cuda.reset_peak_memory_stats()
        xls_backend = evaluate_emotion_checkpoint(
            manifest,
            results / "full" / "checkpoint",
            architecture="wav2vec-xls-r",
            device="cuda",
            split="test",
            base_model_path=arguments.base_model.resolve(),
        )
        allocated, reserved = _cuda_peaks()
        xls_report = aggregate_from_backend(
            xls_backend,
            checkpoint_root=results / "full" / "checkpoint",
            opened_test_count=config.data.final_holdout_count,
            peak_cuda_allocated_mb=allocated,
            peak_cuda_reserved_mb=reserved,
        )
        baseline_backend = evaluate_emotion_checkpoint(
            manifest,
            arguments.baseline_checkpoint.resolve(),
            architecture="emotion2vec-plus",
            device="cuda",
            split="test",
        )
        baseline_report = aggregate_from_backend(
            baseline_backend,
            checkpoint_root=arguments.baseline_checkpoint.resolve(),
            opened_test_count=config.data.final_holdout_count,
        )
        if (
            xls_report.checkpoint_sha256 != candidate.candidate_checkpoint_sha256
            or baseline_report.checkpoint_sha256 != candidate.baseline_checkpoint_sha256
        ):
            raise ValueError
        decision = decide_final_comparison(xls_report, baseline_report, config)
        publish_model(result_root / "xls-r-report.json", xls_report)
        publish_model(result_root / "baseline-report.json", baseline_report)
        publish_model(result_root / "decision.json", decision)
        write_result_checksums(result_root)
        append_transition(
            ledger,
            "final",
            identity,
            report_sha256=digest_file(result_root / "decision.json"),
        )
    except Exception:
        try:
            ledger_records = load_ledger(root / "ledger")
            identity = load_run_identity(root / "state" / "run-identity.json")
            if ledger_records[-1].to_stage == "candidate-frozen":
                append_transition(root / "ledger", "failed", identity)
        except Exception:
            pass
        print("final_evaluation_error")
        return 2
    print("final_evaluation_ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
