"""Run restartable pilot/full XLS-R stages or one isolated memory-probe attempt."""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from voxdelta.evaluation.emotion_experiment import evaluate_emotion_checkpoint
from voxdelta.evaluation.manifest import read_trusted_regular_file

from voxdelta_runpod.config import ExperimentConfig, load_experiment_config
from voxdelta_runpod.gates import (
    AggregateReport,
    PilotDecision,
    full_validation_eligible,
    select_pilot_winner,
)
from voxdelta_runpod.ledger import append_transition, load_ledger
from voxdelta_runpod.recipes import (
    RecipePlan,
    build_pilot_plans_from_package,
    expand_full_recipe_from_package,
)
from voxdelta_runpod.training import (
    BatchProfile,
    TrainingRuntimeError,
    publish_provider_checkpoint,
    run_cuda_training,
    run_isolated_memory_probe,
    run_memory_probe_attempt,
)
from voxdelta_runpod.workflow import (
    SelectedBatchProfile,
    WorkflowError,
    aggregate_from_backend,
    load_aggregate_report,
    load_run_identity,
    publish_model,
    verify_result_checksums,
    write_evaluation_manifest,
    write_result_checksums,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("pilot", "full-or-resume"):
        stage = subparsers.add_parser(name)
        stage.add_argument("--config", required=True, type=Path)
        stage.add_argument("--root", required=True, type=Path)
        stage.add_argument("--base-model", type=Path, default=Path("/workspace/models/xls-r-300m"))
    probe = subparsers.add_parser("memory-probe-attempt")
    probe.add_argument("--config", required=True, type=Path)
    probe.add_argument("--root", required=True, type=Path)
    probe.add_argument("--base-model", required=True, type=Path)
    probe.add_argument("--micro-batch-size", required=True, type=int)
    probe.add_argument("--gradient-accumulation-steps", required=True, type=int)
    return parser


def _paths(root: Path) -> tuple[Path, Path, Path]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise WorkflowError("invalid_run_root")
    return root / "data", root / "state", root / "results"


def _selected_profile(
    config_path: Path,
    root: Path,
    base_model: Path,
    config: ExperimentConfig,
) -> BatchProfile:
    del config
    _, state, _ = _paths(root)
    path = state / "batch-profile.json"
    if path.exists():
        try:
            return SelectedBatchProfile.model_validate_json(read_trusted_regular_file(path)).profile
        except Exception:
            raise WorkflowError("invalid_batch_profile") from None
    result = run_isolated_memory_probe(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "memory-probe-attempt",
            "--config",
            str(config_path),
            "--root",
            str(root),
            "--base-model",
            str(base_model),
        ]
    )
    publish_model(path, SelectedBatchProfile(profile=result.selected))
    return result.selected


def _cuda_peaks() -> tuple[float, float]:
    import torch

    return (
        float(torch.cuda.max_memory_allocated()) / 1024**2,
        float(torch.cuda.max_memory_reserved()) / 1024**2,
    )


def _train_and_evaluate(
    plan: RecipePlan,
    *,
    config: ExperimentConfig,
    identity_path: Path,
    audio_root: Path,
    state_root: Path,
    result_root: Path,
    base_model: Path,
    profile: BatchProfile,
) -> AggregateReport:
    import torch

    identity = load_run_identity(identity_path)
    result_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    report_path = result_root / "report.json"
    if report_path.exists():
        verify_result_checksums(result_root)
        return load_aggregate_report(report_path)
    torch.cuda.reset_peak_memory_stats()
    provider_checkpoint = result_root / "checkpoint"
    if not provider_checkpoint.exists():
        recovery = state_root / "checkpoints" / (plan.name if plan.scope == "pilot" else "full")
        trained = run_cuda_training(
            plan,
            audio_root,
            base_model,
            recovery,
            identity,
            config,
            profile,
            resume=recovery.exists(),
        )
        best_weights = trained.checkpoint_path / "best-model.safetensors"
        publish_provider_checkpoint(
            provider_checkpoint,
            read_trusted_regular_file(best_weights),
            plan,
            trained.metrics,
            config,
        )
    manifest = write_evaluation_manifest(
        plan.validation,
        audio_root,
        state_root / "evaluation" / f"{plan.scope}-{plan.name}.jsonl",
    )
    backend_report = evaluate_emotion_checkpoint(
        manifest,
        provider_checkpoint,
        architecture="wav2vec-xls-r",
        device="cuda",
        split="validation",
        base_model_path=base_model,
    )
    allocated, reserved = _cuda_peaks()
    aggregate = aggregate_from_backend(
        backend_report,
        checkpoint_root=provider_checkpoint,
        opened_test_count=0,
        peak_cuda_allocated_mb=allocated,
        peak_cuda_reserved_mb=reserved,
    )
    publish_model(report_path, aggregate)
    write_result_checksums(result_root)
    return aggregate


def _pilot(config_path: Path, root: Path, base_model: Path) -> str:
    config = load_experiment_config(config_path)
    data, state, results = _paths(root)
    identity = load_run_identity(state / "run-identity.json")
    ledger = root / "ledger"
    current = load_ledger(ledger)[-1].to_stage
    plans = build_pilot_plans_from_package(data / "manifest.jsonl", seed=config.seed)
    profile = _selected_profile(config_path, root, base_model, config)
    pilots_root = results / "pilots"
    pilots_root.mkdir(mode=0o700, parents=True, exist_ok=True)

    if current == "preflight":
        report_a = _train_and_evaluate(
            plans.pilot_a,
            config=config,
            identity_path=state / "run-identity.json",
            audio_root=data,
            state_root=state,
            result_root=pilots_root / "pilot-a",
            base_model=base_model,
            profile=profile,
        )
        append_transition(ledger, "pilot-a", identity, report_sha256=report_a.digest())
        current = "pilot-a"
    if current == "pilot-a":
        report_b = _train_and_evaluate(
            plans.pilot_b,
            config=config,
            identity_path=state / "run-identity.json",
            audio_root=data,
            state_root=state,
            result_root=pilots_root / "pilot-b",
            base_model=base_model,
            profile=profile,
        )
        append_transition(ledger, "pilot-b", identity, report_sha256=report_b.digest())
        current = "pilot-b"
    if current == "pilot-b":
        report_a = load_aggregate_report(pilots_root / "pilot-a" / "report.json")
        report_b = load_aggregate_report(pilots_root / "pilot-b" / "report.json")
        decision = select_pilot_winner(report_a, report_b, config)
        publish_model(pilots_root / "decision.json", decision)
        write_result_checksums(pilots_root)
        if decision.winner is None:
            append_transition(ledger, "failed", identity, report_sha256=report_b.digest())
            return "pilot_gate_failed"
        append_transition(ledger, "pilot-selected", identity, report_sha256=report_b.digest())
        return "pilot_ready"
    if current == "pilot-selected":
        return "pilot_ready"
    raise WorkflowError("illegal_pilot_stage")


def _load_pilot_decision(path: Path) -> PilotDecision:
    try:
        return PilotDecision.model_validate_json(read_trusted_regular_file(path))
    except Exception:
        raise WorkflowError("invalid_pilot_decision") from None


def _full(config_path: Path, root: Path, base_model: Path) -> str:
    config = load_experiment_config(config_path)
    data, state, results = _paths(root)
    identity = load_run_identity(state / "run-identity.json")
    ledger = root / "ledger"
    current = load_ledger(ledger)[-1].to_stage
    if current == "full":
        verify_result_checksums(results / "full")
        return "full_ready"
    if current != "pilot-selected":
        raise WorkflowError("illegal_full_stage")
    decision = _load_pilot_decision(results / "pilots" / "decision.json")
    if decision.winner is None:
        raise WorkflowError("invalid_pilot_decision")
    plan = expand_full_recipe_from_package(
        data / "manifest.jsonl", decision.winner, seed=config.seed
    )
    profile = _selected_profile(config_path, root, base_model, config)
    report = _train_and_evaluate(
        plan,
        config=config,
        identity_path=state / "run-identity.json",
        audio_root=data,
        state_root=state,
        result_root=results / "full",
        base_model=base_model,
        profile=profile,
    )
    if not full_validation_eligible(report, config):
        append_transition(ledger, "failed", identity, report_sha256=report.digest())
        return "full_gate_failed"
    append_transition(ledger, "full", identity, report_sha256=report.digest())
    return "full_ready"


def _memory_probe(arguments: argparse.Namespace) -> int:
    attempt_root = (
        arguments.root.resolve()
        / "state"
        / "memory-probe"
        / (f"mb{arguments.micro_batch_size}-ga{arguments.gradient_accumulation_steps}")
    )
    try:
        import torch

        shutil.rmtree(attempt_root, ignore_errors=True)
        config = load_experiment_config(arguments.config.resolve())
        data, state, _ = _paths(arguments.root.resolve())
        plans = build_pilot_plans_from_package(data / "manifest.jsonl", seed=config.seed)
        profile = BatchProfile(
            micro_batch_size=arguments.micro_batch_size,
            gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_memory_probe_attempt(
                plans.pilot_a,
                data,
                arguments.base_model.resolve(),
                attempt_root,
                load_run_identity(state / "run-identity.json"),
                config,
                profile,
            )
    except Exception as error:
        shutil.rmtree(attempt_root, ignore_errors=True)
        if "torch" in locals() and isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            print("probe_oom", file=sys.stderr)
            return 75
        print("probe_error", file=sys.stderr)
        return 2
    print("probe_ok")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.stage == "memory-probe-attempt":
        return _memory_probe(arguments)
    try:
        config_path = arguments.config.resolve()
        root = arguments.root.resolve()
        base_model = arguments.base_model.resolve()
        outcome = (
            _pilot(config_path, root, base_model)
            if arguments.stage == "pilot"
            else _full(config_path, root, base_model)
        )
    except (OSError, ValueError, TrainingRuntimeError, WorkflowError):
        print("experiment_error")
        return 2
    print(outcome)
    return 0 if outcome.endswith("_ready") else 3


if __name__ == "__main__":
    raise SystemExit(main())
