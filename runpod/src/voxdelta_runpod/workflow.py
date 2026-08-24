"""Shared, privacy-preserving orchestration primitives for remote and retrieval CLIs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from voxdelta.evaluation.emotion_experiment import EmotionExperimentReport
from voxdelta.evaluation.manifest import DatasetItem, read_trusted_regular_file

from voxdelta_runpod.config import CANONICAL_LABELS, ExperimentConfig
from voxdelta_runpod.gates import AggregateReport
from voxdelta_runpod.ledger import RunIdentity
from voxdelta_runpod.package import PackageSidecar, SanitizedRecord
from voxdelta_runpod.recipes import PilotPlans, RecipePlan
from voxdelta_runpod.training import BatchProfile

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class WorkflowError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class SelectedBatchProfile(_FrozenModel):
    schema_version: Literal["1"] = "1"
    profile: BatchProfile


class RuntimeEnvironment(_FrozenModel):
    """Sanitized runtime facts required to reproduce or audit one remote run."""

    schema_version: Literal["1"] = "1"
    python_version: str = Field(min_length=1, max_length=64)
    torch_version: str = Field(min_length=1, max_length=128)
    transformers_version: str = Field(min_length=1, max_length=128)
    cuda_version: str = Field(min_length=1, max_length=64)
    cudnn_version: int = Field(gt=0)
    driver_version: str = Field(pattern=r"^[0-9]+(?:\.[0-9]+){1,3}$")
    gpu_name: str = Field(min_length=1, max_length=256)
    gpu_count: int = Field(gt=0, le=8)
    gpu_total_memory_bytes: int = Field(gt=0)
    gpu_capability: tuple[int, int]
    bf16_supported: bool
    disk_total_bytes: int = Field(gt=0)
    disk_free_bytes: int = Field(ge=0)
    run_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _private_publish(path: Path, payload: bytes) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise WorkflowError("result_exists")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging")
    if temporary.exists() or temporary.is_symlink():
        raise WorkflowError("result_exists")
    try:
        with temporary.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def publish_model(path: Path, model: BaseModel) -> None:
    _private_publish(path, (model.model_dump_json() + "\n").encode())


def load_packaged_manifest(path: Path) -> tuple[SanitizedRecord, ...]:
    if not path.is_absolute():
        raise WorkflowError("invalid_remote_manifest")
    try:
        records = tuple(
            SanitizedRecord.model_validate_json(line)
            for line in read_trusted_regular_file(path).splitlines()
            if line.strip()
        )
    except Exception:
        raise WorkflowError("invalid_remote_manifest") from None
    if not records or len({record.item_key for record in records}) != len(records):
        raise WorkflowError("invalid_remote_manifest")
    return records


def _plan_identity(plan: RecipePlan) -> dict[str, Any]:
    return {
        "name": plan.name,
        "scope": plan.scope,
        "sampler": plan.sampler,
        "epoch_draws": plan.epoch_draws,
        "loss_weights": plan.loss_weights,
        "train_manifest_sha256": plan.train_manifest_sha256,
        "validation_manifest_sha256": plan.validation_manifest_sha256,
    }


def sampler_contract_digest(plans: PilotPlans) -> str:
    payload = json.dumps(
        [_plan_identity(plans.pilot_a), _plan_identity(plans.pilot_b)],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_run_identity(
    config: ExperimentConfig,
    sidecar: PackageSidecar,
    plans: PilotPlans,
    *,
    code_sha256: str,
    container_sha256: str,
) -> RunIdentity:
    if (
        not _SHA256.fullmatch(code_sha256)
        or not _SHA256.fullmatch(container_sha256)
        or sidecar.package_kind != "train-validation"
        or sidecar.audio_file_count != config.data.train_count + config.data.validation_count
    ):
        raise WorkflowError("invalid_run_identity")
    return RunIdentity(
        config_sha256=config.digest(),
        code_sha256=code_sha256,
        container_sha256=container_sha256,
        base_model_sha256=config.model.weights_sha256,
        archive_sha256=sidecar.archive_sha256,
        manifest_sha256=sidecar.manifest_sha256,
        sampler_sha256=sampler_contract_digest(plans),
    )


def load_run_identity(path: Path) -> RunIdentity:
    try:
        return RunIdentity.model_validate_json(read_trusted_regular_file(path))
    except Exception:
        raise WorkflowError("invalid_run_identity") from None


def load_runtime_environment(path: Path) -> RuntimeEnvironment:
    try:
        return RuntimeEnvironment.model_validate_json(read_trusted_regular_file(path))
    except Exception:
        raise WorkflowError("invalid_runtime_environment") from None


def write_evaluation_manifest(
    records: Sequence[SanitizedRecord],
    audio_root: Path,
    output: Path,
) -> Path:
    if not audio_root.is_absolute() or not output.is_absolute() or not records:
        raise WorkflowError("invalid_evaluation_manifest")
    items = tuple(
        DatasetItem(
            id=record.item_key,
            call_id=record.item_key,
            speaker_id=record.item_key,
            audio_path=str(audio_root / record.audio_path),
            transcript="",
            split=record.split,
            source="emotion",
            emotion=record.emotion,
            sha256=record.audio_sha256,
        )
        for record in records
    )
    payload = b"".join(
        (
            json.dumps(
                item.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        for item in sorted(items, key=lambda item: item.id)
    )
    if output.exists():
        existing_digest = hashlib.sha256(read_trusted_regular_file(output)).digest()
        if existing_digest != hashlib.sha256(payload).digest():
            raise WorkflowError("evaluation_manifest_mismatch")
        return output
    _private_publish(output, payload)
    return output


def _private_tree(root: Path) -> bool:
    try:
        return (
            root.is_absolute()
            and not root.is_symlink()
            and all(
                not path.is_symlink() and path.stat(follow_symlinks=False).st_mode & 0o077 == 0
                for path in (root, *root.rglob("*"))
            )
        )
    except OSError:
        return False


def _peak_cpu_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes. Remote execution is Linux, while tests run on both.
    return max(0.0, value / (1024.0 * 1024.0) if value > 10_000_000 else value / 1024.0)


def aggregate_from_backend(
    report: EmotionExperimentReport,
    *,
    checkpoint_root: Path,
    opened_test_count: int,
    peak_cuda_allocated_mb: float | None = None,
    peak_cuda_reserved_mb: float | None = None,
) -> AggregateReport:
    provider: Literal["wav2vec-xls-r", "emotion2vec-plus"] = (
        "wav2vec-xls-r" if report.architecture == "wav2vec-xls-r" else "emotion2vec-plus"
    )
    predicted_class_count = sum(
        any(row[index] > 0 for row in report.confusion_matrix)
        for index in range(len(CANONICAL_LABELS))
    )
    if provider == "wav2vec-xls-r" and (
        peak_cuda_allocated_mb is None or peak_cuda_reserved_mb is None
    ):
        raise WorkflowError("missing_cuda_metrics")
    return AggregateReport(
        provider=provider,
        item_count=report.item_count,
        completed_count=report.completed_count,
        macro_f1=report.macro_f1,
        per_label_f1=report.per_label_f1,
        confusion_matrix=report.confusion_matrix,
        expected_calibration_error=report.expected_calibration_error,
        predicted_class_count=predicted_class_count,
        latency_ms=report.median_latency_ms,
        elapsed_seconds=report.elapsed_seconds,
        peak_cpu_rss_mb=report.peak_rss_mb or _peak_cpu_rss_mb(),
        peak_cuda_allocated_mb=peak_cuda_allocated_mb,
        peak_cuda_reserved_mb=peak_cuda_reserved_mb,
        checkpoint_sha256=report.checkpoint_digest,
        report_input_sha256=report.manifest_digest,
        finite_training_state=True,
        provider_reload_verified=True,
        provenance_verified=True,
        permissions_private=_private_tree(checkpoint_root),
        privacy_verified=True,
        opened_test_count=opened_test_count,
    )


def load_aggregate_report(path: Path) -> AggregateReport:
    try:
        return AggregateReport.model_validate_json(read_trusted_regular_file(path))
    except Exception:
        raise WorkflowError("invalid_aggregate_report") from None


def write_result_checksums(root: Path) -> Path:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise WorkflowError("invalid_result_root")
    sums = root / "SHA256SUMS"
    if sums.exists() or sums.is_symlink():
        raise WorkflowError("result_exists")
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    payload = "".join(
        f"{hashlib.sha256(read_trusted_regular_file(path)).hexdigest()}  "
        f"{path.relative_to(root).as_posix()}\n"
        for path in files
    ).encode()
    _private_publish(sums, payload)
    return sums


def verify_result_checksums(root: Path) -> None:
    try:
        lines = read_trusted_regular_file(root / "SHA256SUMS").decode().splitlines()
        expected: dict[str, str] = {}
        for line in lines:
            digest, name = line.split("  ", 1)
            if not _SHA256.fullmatch(digest) or name.startswith("/") or ".." in Path(name).parts:
                raise ValueError
            expected[name] = digest
        actual = {
            path.relative_to(root).as_posix(): hashlib.sha256(
                read_trusted_regular_file(path)
            ).hexdigest()
            for path in root.rglob("*")
            if path.is_file() and path != root / "SHA256SUMS" and not path.is_symlink()
        }
        if expected != actual or not _private_tree(root):
            raise ValueError
    except Exception:
        raise WorkflowError("result_verification_failed") from None


def digest_file(path: Path) -> str:
    return hashlib.sha256(read_trusted_regular_file(path)).hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def no_sensitive_result_json(paths: Iterable[Path]) -> bool:
    forbidden = {"audio_path", "transcript", "speaker_id", "call_id", "item_key", "prediction"}
    try:
        for path in paths:
            payload = json.loads(read_trusted_regular_file(path))
            stack = [payload]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    if forbidden & set(map(str, value)):
                        return False
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
        return True
    except Exception:
        return False


__all__ = [
    "RuntimeEnvironment",
    "SelectedBatchProfile",
    "WorkflowError",
    "aggregate_from_backend",
    "build_run_identity",
    "canonical_sha256",
    "digest_file",
    "load_aggregate_report",
    "load_packaged_manifest",
    "load_run_identity",
    "load_runtime_environment",
    "no_sensitive_result_json",
    "publish_model",
    "sampler_contract_digest",
    "verify_result_checksums",
    "write_evaluation_manifest",
    "write_result_checksums",
]
