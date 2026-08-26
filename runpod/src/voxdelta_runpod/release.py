"""Freeze the promoted XLS-R candidate into an immutable, reproducible release bundle.

The bundle is self-contained for offline inference: it carries the fine-tuned
checkpoint, the pinned base model (whose feature extractor the runtime loads), the
fixed 7-class label mapping, the training recipe / provenance, a deterministic
release manifest with SHA-256 for every payload, and a model card. Every file is
written with private modes and no host paths or secrets are recorded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from voxdelta.evaluation.manifest import read_trusted_regular_file
from voxdelta.evaluation.wav2vec_base import (
    WAV2VEC_MODEL_ID,
    WAV2VEC_MODEL_REVISION,
    WAV2VEC_WEIGHTS_SHA256,
    PreparedWav2VecBase,
    validate_wav2vec_base,
)
from voxdelta.providers._emotion_runtime import CheckpointInfo, validate_checkpoint

from voxdelta_runpod.config import CANONICAL_LABELS, ExperimentConfig, load_experiment_config
from voxdelta_runpod.gates import (
    AggregateReport,
    FinalDecision,
    FrozenCandidate,
    decide_final_comparison,
    full_validation_eligible,
)
from voxdelta_runpod.workflow import (
    canonical_sha256,
    load_aggregate_report,
    verify_result_checksums,
    write_result_checksums,
)

RELEASE_SCHEMA_VERSION = "1"
_ARCHITECTURE: Literal["wav2vec-xls-r"] = "wav2vec-xls-r"

_CHECKPOINT_FILES = ("config.json", "label_mapping.json", "metrics.json", "model.safetensors")
_BASE_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")
_RESERVED_NAMES = frozenset({"RELEASE.json", "SHA256SUMS"})
_SHA256_HEX = r"^[0-9a-f]{64}$"

CheckpointValidator = Callable[[Path], CheckpointInfo]
BaseValidator = Callable[[Path], PreparedWav2VecBase]


def validate_checkpoint_default(path: Path) -> CheckpointInfo:
    return validate_checkpoint(Path(path), architecture=_ARCHITECTURE, model_id=WAV2VEC_MODEL_ID)


class ReleaseError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ReleasePayload(_FrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_HEX)
    bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def safe_relative_path(self) -> ReleasePayload:
        parts = self.path.split("/")
        if (
            self.path.startswith("/")
            or self.path != Path(self.path).as_posix()
            or "" in parts
            or ".." in parts
            or "." in parts
            or self.path in _RESERVED_NAMES
        ):
            raise ValueError("invalid release payload path")
        return self


class ReleaseMetrics(_FrozenModel):
    validation_macro_f1: float = Field(ge=0, le=1)
    validation_expected_calibration_error: float = Field(ge=0, le=1)
    final_macro_f1: float = Field(ge=0, le=1)
    final_expected_calibration_error: float = Field(ge=0, le=1)
    baseline_final_macro_f1: float = Field(ge=0, le=1)


class ReleaseProvenance(_FrozenModel):
    frozen_candidate_sha256: str = Field(pattern=_SHA256_HEX)
    validation_report_sha256: str = Field(pattern=_SHA256_HEX)
    final_report_sha256: str = Field(pattern=_SHA256_HEX)
    baseline_report_sha256: str = Field(pattern=_SHA256_HEX)
    decision_sha256: str = Field(pattern=_SHA256_HEX)
    experiment_config_digest: str = Field(pattern=_SHA256_HEX)
    capability_token: str = Field(pattern=_SHA256_HEX)


class ReleaseManifest(_FrozenModel):
    schema_version: Literal["1"] = "1"
    release_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    model_id: Literal["facebook/wav2vec2-xls-r-300m"] = "facebook/wav2vec2-xls-r-300m"
    model_revision: Literal["1a640f32ac3e39899438a2931f9924c02f080a54"] = WAV2VEC_MODEL_REVISION
    base_model_sha256: Literal[
        "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"
    ] = WAV2VEC_WEIGHTS_SHA256
    candidate_checkpoint_sha256: str = Field(pattern=_SHA256_HEX)
    labels: tuple[str, ...]
    final_sealed: Literal[True] = True
    final_holdout_count: int = Field(gt=0)
    decision: Literal["xls-r-wins"] = "xls-r-wins"
    metrics: ReleaseMetrics
    provenance: ReleaseProvenance
    payloads: tuple[ReleasePayload, ...]
    bundle_tree_sha256: str = Field(pattern=_SHA256_HEX)

    @model_validator(mode="after")
    def consistent_manifest(self) -> ReleaseManifest:
        paths = [payload.path for payload in self.payloads]
        if (
            tuple(self.labels) != CANONICAL_LABELS
            or not self.payloads
            or paths != sorted(paths)
            or len(set(paths)) != len(paths)
            or _tree_digest(self.payloads) != self.bundle_tree_sha256
        ):
            raise ValueError("invalid release manifest")
        return self


def _tree_digest(payloads: tuple[ReleasePayload, ...]) -> str:
    return canonical_sha256(
        [
            {"path": payload.path, "sha256": payload.sha256, "bytes": payload.bytes}
            for payload in sorted(payloads, key=lambda payload: payload.path)
        ]
    )


def _resolved_dir(path: Path, code: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ReleaseError(code)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ReleaseError(code)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise ReleaseError(code) from None
    if not resolved.is_dir():
        raise ReleaseError(code)
    return resolved


def _copy_private(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("xb") as writer:
        os.fchmod(writer.fileno(), 0o600)
        shutil.copyfileobj(reader, writer)


def _read_report(path: Path) -> AggregateReport:
    try:
        return load_aggregate_report(path)
    except Exception:
        raise ReleaseError("invalid_release_report") from None


def _payloads(root: Path) -> tuple[ReleasePayload, ...]:
    entries = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.relative_to(root).as_posix() not in _RESERVED_NAMES
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    payloads: list[ReleasePayload] = []
    for path in entries:
        data = read_trusted_regular_file(path)
        payloads.append(
            ReleasePayload(
                path=path.relative_to(root).as_posix(),
                sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data),
            )
        )
    return tuple(payloads)


def build_release_bundle(
    *,
    checkpoint_dir: Path,
    base_model_dir: Path,
    frozen_candidate: Path,
    validation_report: Path,
    final_report: Path,
    baseline_report: Path,
    decision: Path,
    experiment_config: Path,
    output_dir: Path,
    release_id: str,
    checkpoint_validator: CheckpointValidator = validate_checkpoint_default,
    base_validator: BaseValidator = validate_wav2vec_base,
) -> Path:
    """Assemble one immutable release bundle from already-verified local artifacts."""

    output = Path(output_dir).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or output.is_symlink() or staging.exists() or staging.is_symlink():
        raise ReleaseError("release_exists")

    checkpoint_source = _resolved_dir(checkpoint_dir, "invalid_checkpoint_source")
    base_source = _resolved_dir(base_model_dir, "invalid_base_source")

    try:
        info = checkpoint_validator(checkpoint_source)
        base = base_validator(base_source)
    except ReleaseError:
        raise
    except Exception:
        raise ReleaseError("invalid_release_source") from None
    if (
        info.architecture != _ARCHITECTURE
        or info.model_id != WAV2VEC_MODEL_ID
        or info.model_revision != WAV2VEC_MODEL_REVISION
        or info.base_model_sha256 != WAV2VEC_WEIGHTS_SHA256
        or base.weights_sha256 != WAV2VEC_WEIGHTS_SHA256
        or base.revision != WAV2VEC_MODEL_REVISION
    ):
        raise ReleaseError("invalid_release_source")

    config = _load_config(experiment_config)
    candidate = _load_candidate(frozen_candidate, info.digest)
    validation = _read_report(validation_report)
    final = _read_report(final_report)
    baseline = _read_report(baseline_report)
    final_decision = _load_decision(decision)
    _validate_release_contract(
        config,
        candidate,
        validation,
        final,
        baseline,
        final_decision,
        info.digest,
    )

    metrics = ReleaseMetrics(
        validation_macro_f1=validation.macro_f1,
        validation_expected_calibration_error=validation.expected_calibration_error,
        final_macro_f1=final.macro_f1,
        final_expected_calibration_error=final.expected_calibration_error,
        baseline_final_macro_f1=baseline.macro_f1,
    )

    try:
        staging.mkdir(mode=0o700, parents=True)
        _stage_tree(staging / "checkpoint", checkpoint_source, _CHECKPOINT_FILES)
        _stage_tree(staging / "base-model", base_source, _BASE_FILES)
        provenance = staging / "provenance"
        provenance.mkdir(mode=0o700)
        _copy_private(frozen_candidate.resolve(), provenance / "frozen-candidate.json")
        _copy_private(validation_report.resolve(), provenance / "validation-report.json")
        _copy_private(final_report.resolve(), provenance / "final-report.json")
        _copy_private(baseline_report.resolve(), provenance / "baseline-report.json")
        _copy_private(decision.resolve(), provenance / "decision.json")
        _write_private(
            provenance / "experiment-config.json",
            json.dumps(
                config.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            + b"\n",
        )
        _write_private(
            staging / "MODEL_CARD.md",
            _render_model_card(
                release_id=release_id,
                candidate_checkpoint_sha256=info.digest,
                validation=validation,
                final=final,
                baseline=baseline,
            ).encode(),
        )

        payloads = _payloads(staging)
        manifest = ReleaseManifest(
            release_id=release_id,
            candidate_checkpoint_sha256=info.digest,
            labels=CANONICAL_LABELS,
            final_holdout_count=final.item_count,
            metrics=metrics,
            provenance=ReleaseProvenance(
                frozen_candidate_sha256=_digest_bytes(frozen_candidate),
                validation_report_sha256=_digest_bytes(validation_report),
                final_report_sha256=_digest_bytes(final_report),
                baseline_report_sha256=_digest_bytes(baseline_report),
                decision_sha256=_digest_bytes(decision),
                experiment_config_digest=config.digest(),
                capability_token=candidate.capability_token,
            ),
            payloads=payloads,
            bundle_tree_sha256=_tree_digest(payloads),
        )
        _write_private(
            staging / "RELEASE.json",
            json.dumps(
                manifest.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            + b"\n",
        )
        write_result_checksums(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def load_release_manifest(root: Path) -> ReleaseManifest:
    try:
        return ReleaseManifest.model_validate_json(
            read_trusted_regular_file(Path(root).resolve() / "RELEASE.json")
        )
    except Exception:
        raise ReleaseError("invalid_release_manifest") from None


def verify_release_bundle(
    root: Path,
    *,
    checkpoint_validator: CheckpointValidator = validate_checkpoint_default,
    base_validator: BaseValidator = validate_wav2vec_base,
) -> ReleaseManifest:
    """Re-verify integrity, private modes, and reload prerequisites of a release bundle."""

    bundle = _resolved_dir(root, "invalid_release_bundle")
    manifest = load_release_manifest(bundle)
    try:
        verify_result_checksums(bundle)
    except Exception:
        raise ReleaseError("release_integrity_failed") from None

    if _payloads(bundle) != manifest.payloads:
        raise ReleaseError("release_integrity_failed")

    try:
        info = checkpoint_validator(bundle / "checkpoint")
        base = base_validator(bundle / "base-model")
    except Exception:
        raise ReleaseError("release_reload_prerequisites_failed") from None
    if (
        info.digest != manifest.candidate_checkpoint_sha256
        or info.architecture != _ARCHITECTURE
        or info.model_id != manifest.model_id
        or info.model_revision != manifest.model_revision
        or info.base_model_sha256 != manifest.base_model_sha256
        or base.weights_sha256 != manifest.base_model_sha256
    ):
        raise ReleaseError("release_reload_prerequisites_failed")

    provenance = bundle / "provenance"
    try:
        config = _load_bundled_config(provenance / "experiment-config.json")
        candidate = _load_candidate(provenance / "frozen-candidate.json", info.digest)
        validation = _read_report(provenance / "validation-report.json")
        final = _read_report(provenance / "final-report.json")
        baseline = _read_report(provenance / "baseline-report.json")
        final_decision = _load_decision(provenance / "decision.json")
        _validate_release_contract(
            config,
            candidate,
            validation,
            final,
            baseline,
            final_decision,
            info.digest,
        )
        actual_provenance = ReleaseProvenance(
            frozen_candidate_sha256=_digest_bytes(provenance / "frozen-candidate.json"),
            validation_report_sha256=_digest_bytes(provenance / "validation-report.json"),
            final_report_sha256=_digest_bytes(provenance / "final-report.json"),
            baseline_report_sha256=_digest_bytes(provenance / "baseline-report.json"),
            decision_sha256=_digest_bytes(provenance / "decision.json"),
            experiment_config_digest=config.digest(),
            capability_token=candidate.capability_token,
        )
        expected_metrics = ReleaseMetrics(
            validation_macro_f1=validation.macro_f1,
            validation_expected_calibration_error=validation.expected_calibration_error,
            final_macro_f1=final.macro_f1,
            final_expected_calibration_error=final.expected_calibration_error,
            baseline_final_macro_f1=baseline.macro_f1,
        )
    except Exception:
        raise ReleaseError("release_provenance_failed") from None
    if (
        actual_provenance != manifest.provenance
        or expected_metrics != manifest.metrics
        or final.item_count != manifest.final_holdout_count
    ):
        raise ReleaseError("release_provenance_failed")

    return manifest


def _stage_tree(destination: Path, source: Path, names: tuple[str, ...]) -> None:
    destination.mkdir(mode=0o700)
    for name in names:
        _copy_private(source / name, destination / name)


def _write_private(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(payload)


def _digest_bytes(path: Path) -> str:
    return hashlib.sha256(read_trusted_regular_file(Path(path).resolve())).hexdigest()


def _load_config(path: Path) -> ExperimentConfig:
    try:
        return load_experiment_config(Path(path).resolve())
    except Exception:
        raise ReleaseError("invalid_experiment_config") from None


def _load_bundled_config(path: Path) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate_json(read_trusted_regular_file(path))
    except Exception:
        raise ReleaseError("invalid_experiment_config") from None


def _load_candidate(path: Path, checkpoint_digest: str) -> FrozenCandidate:
    try:
        candidate = FrozenCandidate.model_validate_json(
            read_trusted_regular_file(Path(path).resolve())
        )
    except Exception:
        raise ReleaseError("invalid_frozen_candidate") from None
    if not candidate.verify_token():
        raise ReleaseError("invalid_frozen_candidate")
    if candidate.candidate_checkpoint_sha256 != checkpoint_digest:
        raise ReleaseError("candidate_identity_mismatch")
    return candidate


def _validate_release_contract(
    config: ExperimentConfig,
    candidate: FrozenCandidate,
    validation: AggregateReport,
    final: AggregateReport,
    baseline: AggregateReport,
    decision: FinalDecision,
    checkpoint_digest: str,
) -> None:
    if candidate.candidate_checkpoint_sha256 != checkpoint_digest:
        raise ReleaseError("candidate_identity_mismatch")
    if (
        candidate.config_sha256 != config.digest()
        or candidate.validation_report_sha256 != validation.digest()
        or candidate.baseline_checkpoint_sha256 != baseline.checkpoint_sha256
    ):
        raise ReleaseError("candidate_provenance_mismatch")
    if not full_validation_eligible(validation, config):
        raise ReleaseError("invalid_validation_report")
    if (
        final.provider != _ARCHITECTURE
        or final.checkpoint_sha256 != checkpoint_digest
        or final.opened_test_count != final.item_count
    ):
        raise ReleaseError("final_not_sealed")
    if baseline.provider != "emotion2vec-plus":
        raise ReleaseError("invalid_baseline_report")
    reports = (final, baseline)
    if final.report_input_sha256 != baseline.report_input_sha256 or any(
        report.item_count != config.data.final_holdout_count
        or report.completed_count != config.data.final_holdout_count
        or report.opened_test_count != config.data.final_holdout_count
        or not report.finite_training_state
        or not report.provider_reload_verified
        or not report.provenance_verified
        or not report.permissions_private
        or not report.privacy_verified
        for report in reports
    ):
        raise ReleaseError("invalid_final_comparison")
    expected = decide_final_comparison(final, baseline, config)
    if (
        decision != expected
        or not decision.promote_xls_r
        or decision.selected_provider != _ARCHITECTURE
        or decision.reason != "xls-r-wins"
    ):
        raise ReleaseError("invalid_final_decision")


def _load_decision(path: Path) -> FinalDecision:
    try:
        decision = FinalDecision.model_validate_json(
            read_trusted_regular_file(Path(path).resolve())
        )
    except Exception:
        raise ReleaseError("invalid_final_decision") from None
    if not decision.promote_xls_r or decision.selected_provider != _ARCHITECTURE:
        raise ReleaseError("invalid_final_decision")
    return decision


def _per_label_lines(report: AggregateReport) -> str:
    return "\n".join(
        f"| {label} | {report.per_label_f1[label]:.4f} |" for label in CANONICAL_LABELS
    )


def _render_model_card(
    *,
    release_id: str,
    candidate_checkpoint_sha256: str,
    validation: AggregateReport,
    final: AggregateReport,
    baseline: AggregateReport,
) -> str:
    labels = ", ".join(f"{index}={label}" for index, label in enumerate(CANONICAL_LABELS))
    return f"""# VoxDelta XLS-R Speech Emotion Recognition — Release `{release_id}`

Immutable release of the promoted XLS-R 300M categorical speech-emotion classifier.
Integrity and identity are fixed by `RELEASE.json` and `SHA256SUMS`; re-verify with
`scripts/verify_release.py`.

## Model

- Architecture: `wav2vec-xls-r` (Wav2Vec2 audio classification head).
- Base model: `{WAV2VEC_MODEL_ID}` @ revision `{WAV2VEC_MODEL_REVISION}`
  (base weights SHA-256 `{WAV2VEC_WEIGHTS_SHA256}`).
- Fine-tuned checkpoint identity (tree SHA-256): `{candidate_checkpoint_sha256}`.
- Fixed 7-class label mapping (id=label): {labels}.

## Intended use

Offline categorical speech-emotion recognition over single-speaker 16 kHz mono audio,
producing one of the 7 fixed labels per utterance. Inference loads entirely from local
files (`local_files_only=True`): the fine-tuned weights from `checkpoint/` and the
feature extractor / architecture from `base-model/`. Both directories are required.

## Training recipe

Full fine-tuning of the XLS-R 300M encoder plus classification head. Winning recipe
`pilot-a`: class-balanced sampling with replacement, uniform loss weights. Effective
batch size 16, encoder LR 1e-5, head LR 1e-4, weight decay 0.01, warmup ratio 0.1,
gradient clip 1.0, up to 10 epochs with early-stopping patience 2. 20 s window at
16 kHz, seed 622. Train 29,476 / validation 3,569 utterances. See
`provenance/experiment-config.json` for the exact, hashed configuration.

## Validation metrics (development split)

- Macro-F1: {validation.macro_f1:.6f}
- Expected calibration error (ECE): {validation.expected_calibration_error:.6f}

| label | validation F1 |
| --- | --- |
{_per_label_lines(validation)}

## Final holdout aggregate (sealed)

The final holdout was consumed exactly once and is sealed. It MUST NOT be re-evaluated,
opened for tuning, or rerun. These aggregates are reproduced from the sealed report in
`provenance/final-report.json` (no holdout audio is included in this bundle).

- XLS-R final macro-F1: {final.macro_f1:.16f}
- XLS-R final ECE: {final.expected_calibration_error:.16f}
- emotion2vec baseline final macro-F1: {baseline.macro_f1:.16f}
- Holdout item count: {final.item_count}
- Decision: xls-r-wins (XLS-R promoted).

## Calibration warning

The XLS-R final ECE is {final.expected_calibration_error}. Predicted softmax
probabilities are **not** well calibrated and must not be treated as true confidence.
Apply downstream calibration (e.g. temperature scaling) or thresholding before using the
scores for gating or ranking decisions.

## Known limitations

- Class imbalance: minority labels (e.g. `surprise`, `neutral`) have weaker per-label F1.
- Domain shift: performance degrades on audio unlike the training distribution
  (language, channel, recording conditions), and on multi-speaker or noisy inputs.
- Audio longer than the 20 s window is truncated at inference.
- Not suitable for high-stakes or safety-critical decisions without human review.
- The sealed final metrics reflect a single, non-repeatable evaluation.
"""


__all__ = [
    "RELEASE_SCHEMA_VERSION",
    "ReleaseError",
    "ReleaseManifest",
    "ReleaseMetrics",
    "ReleasePayload",
    "ReleaseProvenance",
    "build_release_bundle",
    "load_release_manifest",
    "validate_checkpoint_default",
    "verify_release_bundle",
]
