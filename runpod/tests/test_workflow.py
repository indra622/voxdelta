from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from pytest import MonkeyPatch

from voxdelta_runpod.config import CANONICAL_LABELS, load_experiment_config
from voxdelta_runpod.gates import AggregateReport
from voxdelta_runpod.ledger import append_transition, initialize_ledger, load_ledger
from voxdelta_runpod.package import PackageSidecar, SanitizedRecord
from voxdelta_runpod.recipes import (
    PILOT_B_COUNTS,
    RecipePlan,
    build_pilot_plans_from_package,
    expand_full_recipe_from_package,
)
from voxdelta_runpod.workflow import (
    RuntimeEnvironment,
    build_run_identity,
    load_run_identity,
    load_runtime_environment,
    publish_model,
    verify_result_checksums,
    write_result_checksums,
)

CONFIG = Path(__file__).parents[1] / "config" / "experiment.toml"
ASSEMBLER = Path(__file__).parents[1] / "scripts" / "assemble_handoff.py"
RUNNER = Path(__file__).parents[1] / "scripts" / "run_experiment.py"


def _package_manifest(path: Path) -> Path:
    train_counts = {label: max(500, PILOT_B_COUNTS[label]) for label in CANONICAL_LABELS}
    train_counts["sadness"] += 29_476 - sum(train_counts.values())
    validation_counts = {label: 510 for label in CANONICAL_LABELS}
    validation_counts[CANONICAL_LABELS[-1]] -= sum(validation_counts.values()) - 3_569
    records: list[SanitizedRecord] = []
    index = 1
    for split, counts in (("train", train_counts), ("validation", validation_counts)):
        for label in CANONICAL_LABELS:
            for _ in range(counts[label]):
                key = f"{index:064x}"
                records.append(
                    SanitizedRecord(
                        item_key=key,
                        audio_path=f"audio/{key}.wav",
                        split=split,
                        emotion=label,
                        audio_sha256=f"{index + 100_000:064x}",
                    )
                )
                index += 1
    path.write_text("".join(record.model_dump_json() + "\n" for record in records))
    path.chmod(0o600)
    return path


def test_remote_package_recipes_and_identity_are_deterministic(tmp_path: Path) -> None:
    manifest = _package_manifest((tmp_path / "manifest.jsonl").resolve())
    config = load_experiment_config(CONFIG.resolve())

    first = build_pilot_plans_from_package(manifest, seed=config.seed)
    second = build_pilot_plans_from_package(manifest, seed=config.seed)
    full = expand_full_recipe_from_package(manifest, "pilot-b", seed=config.seed)

    assert first == second
    assert len(first.pilot_a.train) == 3_500
    assert len(first.pilot_b.train) == 3_500
    assert len(first.pilot_a.validation) == 350
    assert len(full.train) == 29_476 and len(full.validation) == 3_569
    sidecar = PackageSidecar(
        package_kind="train-validation",
        archive_sha256="a" * 64,
        archive_bytes=1,
        audio_file_count=33_045,
        archive_member_count=33_046,
        split_counts={"train": 29_476, "validation": 3_569},
        label_counts={
            label: sum(record.emotion == label for record in (*full.train, *full.validation))
            for label in CANONICAL_LABELS
        },
        manifest_sha256="b" * 64,
    )
    identity = build_run_identity(
        config,
        sidecar,
        first,
        code_sha256="c" * 64,
        container_sha256="d" * 64,
    )
    identity_path = (tmp_path / "state" / "run-identity.json").resolve()
    publish_model(identity_path, identity)
    assert load_run_identity(identity_path) == identity

    environment = RuntimeEnvironment(
        python_version="3.12.11",
        torch_version="2.8.0+cu128",
        transformers_version="4.55.0",
        cuda_version="12.8",
        cudnn_version=91002,
        driver_version="570.169",
        gpu_name="NVIDIA A40",
        gpu_count=1,
        gpu_total_memory_bytes=48 * 1024**3,
        gpu_capability=(8, 6),
        bf16_supported=True,
        disk_total_bytes=200 * 1024**3,
        disk_free_bytes=120 * 1024**3,
        run_identity_sha256="e" * 64,
    )
    environment_path = (tmp_path / "state" / "environment.json").resolve()
    publish_model(environment_path, environment)
    assert load_runtime_environment(environment_path) == environment


def test_result_checksums_and_handoff_assembly_are_private(tmp_path: Path) -> None:
    result = (tmp_path / "result").resolve()
    result.mkdir(mode=0o700)
    payload = result / "report.json"
    payload.write_text("{}\n")
    payload.chmod(0o600)
    write_result_checksums(result)
    verify_result_checksums(result)

    image = (tmp_path / "image").resolve()
    training = (tmp_path / "training").resolve()
    base_model = (tmp_path / "base-model").resolve()
    image.mkdir(mode=0o700)
    training.mkdir(mode=0o700)
    base_model.mkdir(mode=0o700)
    image_files = {
        "voxdelta-runpod.oci.tar.zst": b"image",
        "image-digest.txt": f"sha256:{'a' * 64}\n".encode(),
        "image-build.json": b"{}\n",
        "image-sbom.spdx.json": b"{}\n",
        "push-image.sh": b"#!/usr/bin/env bash\nexit 0\n",
    }
    for name, content in image_files.items():
        path = image / name
        path.write_bytes(content)
        path.chmod(0o700 if name.endswith(".sh") else 0o600)
    for name, content in {
        "train-validation.tar.zst": b"training",
        "train-validation.sidecar.json": b"{}\n",
    }.items():
        path = training / name
        path.write_bytes(content)
        path.chmod(0o600)
    for name, content in {
        "xls-r-base.tar.zst": b"base-model",
        "xls-r-base.sidecar.json": b"{}\n",
    }.items():
        path = base_model / name
        path.write_bytes(content)
        path.chmod(0o600)

    spec = importlib.util.spec_from_file_location("assemble_handoff", ASSEMBLER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = (tmp_path / "handoff").resolve()
    assert (
        module.main(
            [
                "--image-root",
                str(image),
                "--training-root",
                str(training),
                "--base-model-root",
                str(base_model),
                "--output",
                str(output),
                "--run-id",
                "synthetic-622",
            ]
        )
        == 0
    )
    assert (output / "01-preflight-and-pilot.sh").exists()
    assert (output / "02-full-or-resume.sh").exists()
    assert (output / "03-download-results.sh").exists()
    assert os.stat(output).st_mode & 0o077 == 0


def _aggregate(count: int, macro_f1: float) -> AggregateReport:
    matrix = tuple(
        (count, *([0] * 6)) if index == 0 else tuple(0 for _ in CANONICAL_LABELS)
        for index in range(len(CANONICAL_LABELS))
    )
    return AggregateReport(
        provider="wav2vec-xls-r",
        item_count=count,
        completed_count=count,
        macro_f1=macro_f1,
        per_label_f1={label: macro_f1 for label in CANONICAL_LABELS},
        confusion_matrix=matrix,
        expected_calibration_error=0.05,
        predicted_class_count=7,
        latency_ms=1.0,
        elapsed_seconds=2.0,
        peak_cpu_rss_mb=3.0,
        peak_cuda_allocated_mb=4.0,
        peak_cuda_reserved_mb=5.0,
        checkpoint_sha256="a" * 64,
        report_input_sha256="b" * 64,
        finite_training_state=True,
        provider_reload_verified=True,
        provenance_verified=True,
        permissions_private=True,
        privacy_verified=True,
        opened_test_count=0,
    )


def test_synthetic_orchestration_reaches_passing_full_gate(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    root = (tmp_path / "remote").resolve()
    data = root / "data"
    state = root / "state"
    results = root / "results"
    for path in (root, data, state, results):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest = _package_manifest(data / "manifest.jsonl")
    config = load_experiment_config(CONFIG.resolve())
    plans = build_pilot_plans_from_package(manifest, seed=config.seed)
    sidecar = PackageSidecar(
        package_kind="train-validation",
        archive_sha256="1" * 64,
        archive_bytes=1,
        audio_file_count=33_045,
        archive_member_count=33_046,
        split_counts={"train": 29_476, "validation": 3_569},
        label_counts={label: 1 for label in CANONICAL_LABELS[:-1]} | {CANONICAL_LABELS[-1]: 33_039},
        manifest_sha256="2" * 64,
    )
    identity = build_run_identity(
        config,
        sidecar,
        plans,
        code_sha256="3" * 64,
        container_sha256="4" * 64,
    )
    publish_model(state / "run-identity.json", identity)
    initialize_ledger(root / "ledger", identity)
    append_transition(root / "ledger", "preflight", identity)

    spec = importlib.util.spec_from_file_location("run_experiment", RUNNER)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def fake_profile(*_args: object, **_kwargs: object) -> object:
        from voxdelta_runpod.training import BATCH_PROFILES

        return BATCH_PROFILES[1]

    def fake_train(plan: RecipePlan, **kwargs: object) -> AggregateReport:
        scope = plan.scope
        name = plan.name
        count = 3_569 if scope == "full" else 350
        score = 0.31 if scope == "full" else (0.20 if name == "pilot-a" else 0.22)
        report = _aggregate(count, score)
        result_root = Path(str(kwargs["result_root"]))
        result_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        publish_model(result_root / "report.json", report)
        write_result_checksums(result_root)
        return report

    monkeypatch.setattr(runner, "_selected_profile", fake_profile)
    monkeypatch.setattr(runner, "_train_and_evaluate", fake_train)
    base = (tmp_path / "base").resolve()
    base.mkdir(mode=0o700)

    assert runner._pilot(CONFIG.resolve(), root, base) == "pilot_ready"
    assert runner._full(CONFIG.resolve(), root, base) == "full_ready"
    assert [record.to_stage for record in load_ledger(root / "ledger")] == [
        "initialized",
        "preflight",
        "pilot-a",
        "pilot-b",
        "pilot-selected",
        "full",
    ]
