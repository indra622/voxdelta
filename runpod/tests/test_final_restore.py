from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from voxdelta.evaluation.emotion_training import CANONICAL_LABELS
from voxdelta.evaluation.wav2vec_base import (
    WAV2VEC_MODEL_ID,
    WAV2VEC_MODEL_REVISION,
    WAV2VEC_WEIGHTS_SHA256,
)
from voxdelta.providers._emotion_runtime import validate_checkpoint

from voxdelta_runpod.final_restore import build_final_restore_bundle
from voxdelta_runpod.gates import AggregateReport
from voxdelta_runpod.ledger import RunIdentity, append_transition, initialize_ledger
from voxdelta_runpod.workflow import publish_model, write_result_checksums


def _private_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    path.chmod(0o600)


def _sources(root: Path) -> tuple[Path, Path, Path]:
    result = root / "results" / "full"
    checkpoint = result / "checkpoint"
    ledger = root / "ledger"
    state = root / "state"
    for directory in (root, result.parent, result, checkpoint, state):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    _private_json(
        checkpoint / "config.json",
        {
            "architecture": "wav2vec-xls-r",
            "base_model_sha256": WAV2VEC_WEIGHTS_SHA256,
            "class_weighting": "none",
            "class_weights": [1.0] * len(CANONICAL_LABELS),
            "labels": list(CANONICAL_LABELS),
            "model_id": WAV2VEC_MODEL_ID,
            "model_revision": WAV2VEC_MODEL_REVISION,
            "schema_version": "4",
        },
    )
    _private_json(
        checkpoint / "label_mapping.json",
        {str(index): label for index, label in enumerate(CANONICAL_LABELS)},
    )
    _private_json(
        checkpoint / "metrics.json",
        {"macro_f1": 0.3, "validation_hash": "a" * 64},
    )
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"synthetic-weights")
    weights.chmod(0o600)
    checkpoint_digest = validate_checkpoint(
        checkpoint, architecture="wav2vec-xls-r", model_id=WAV2VEC_MODEL_ID
    ).digest
    report = AggregateReport(
        provider="wav2vec-xls-r",
        item_count=7,
        completed_count=7,
        macro_f1=0.3,
        per_label_f1={label: 0.3 for label in CANONICAL_LABELS},
        confusion_matrix=tuple(
            tuple(1 if row == column else 0 for column in range(len(CANONICAL_LABELS)))
            for row in range(len(CANONICAL_LABELS))
        ),
        expected_calibration_error=0.1,
        predicted_class_count=7,
        latency_ms=1.0,
        elapsed_seconds=2.0,
        peak_cpu_rss_mb=3.0,
        peak_cuda_allocated_mb=4.0,
        peak_cuda_reserved_mb=5.0,
        checkpoint_sha256=checkpoint_digest,
        report_input_sha256="b" * 64,
        finite_training_state=True,
        provider_reload_verified=True,
        provenance_verified=True,
        permissions_private=True,
        privacy_verified=True,
        opened_test_count=0,
    )
    publish_model(result / "report.json", report)
    write_result_checksums(result)

    identity = RunIdentity(
        config_sha256="1" * 64,
        code_sha256="2" * 64,
        container_sha256="3" * 64,
        base_model_sha256="4" * 64,
        archive_sha256="5" * 64,
        manifest_sha256="6" * 64,
        sampler_sha256="7" * 64,
    )
    initialize_ledger(ledger, identity)
    for stage in ("preflight", "pilot-a", "pilot-b", "pilot-selected", "full"):
        append_transition(ledger, stage, identity)
    identity_path = state / "run-identity.json"
    publish_model(identity_path, identity)
    return result, ledger, identity_path


def test_final_restore_bundle_is_private_deterministic_and_minimal(tmp_path: Path) -> None:
    result, ledger, identity = _sources((tmp_path / "source").resolve())
    outputs = tuple((tmp_path / name).resolve() for name in ("first", "second"))
    for output in outputs:
        output.mkdir(mode=0o700)
        build_final_restore_bundle(result, ledger, identity, output)

    first = outputs[0] / "full-state.tar.zst"
    second = outputs[1] / "full-state.tar.zst"
    assert first.read_bytes() == second.read_bytes()
    assert all(
        path.stat(follow_symlinks=False).st_mode & 0o077 == 0 for path in outputs[0].iterdir()
    )
    members = subprocess.check_output(["tar", "--zstd", "-tf", first], text=True).splitlines()
    assert members == [
        "ledger/record-000000.json",
        "ledger/record-000001.json",
        "ledger/record-000002.json",
        "ledger/record-000003.json",
        "ledger/record-000004.json",
        "ledger/record-000005.json",
        "results/full/SHA256SUMS",
        "results/full/checkpoint/config.json",
        "results/full/checkpoint/label_mapping.json",
        "results/full/checkpoint/metrics.json",
        "results/full/checkpoint/model.safetensors",
        "results/full/report.json",
        "state/run-identity.json",
    ]
    assert os.stat(first).st_mode & 0o077 == 0
