"""Mechanical pilot, validation, and one-time final-comparison gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from voxdelta.domain.models import EmotionLabel
from voxdelta.evaluation.manifest import read_trusted_regular_file

from voxdelta_runpod.config import CANONICAL_LABELS, ExperimentConfig
from voxdelta_runpod.package import (
    _emotion_items,
    _identity_payload,
    _publish_package,
    derive_holdout_identity,
    semantic_fingerprint,
)
from voxdelta_runpod.recipes import RecipeName


class GateError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class AggregateReport(_FrozenModel):
    schema_version: Literal["1"] = "1"
    provider: Literal["wav2vec-xls-r", "emotion2vec-plus"]
    item_count: int = Field(gt=0)
    completed_count: int = Field(ge=0)
    macro_f1: float = Field(ge=0, le=1)
    per_label_f1: dict[EmotionLabel, float]
    confusion_matrix: tuple[tuple[int, ...], ...]
    expected_calibration_error: float = Field(ge=0, le=1)
    predicted_class_count: int = Field(ge=0, le=7)
    latency_ms: float = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    peak_cpu_rss_mb: float = Field(ge=0)
    peak_cuda_allocated_mb: float | None = Field(default=None, ge=0)
    peak_cuda_reserved_mb: float | None = Field(default=None, ge=0)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finite_training_state: bool
    provider_reload_verified: bool
    provenance_verified: bool
    permissions_private: bool
    privacy_verified: bool
    opened_test_count: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_aggregate(self) -> AggregateReport:
        numeric = (
            self.macro_f1,
            self.expected_calibration_error,
            self.latency_ms,
            self.elapsed_seconds,
            self.peak_cpu_rss_mb,
        )
        optional = (self.peak_cuda_allocated_mb, self.peak_cuda_reserved_mb)
        if (
            self.completed_count > self.item_count
            or set(self.per_label_f1) != set(CANONICAL_LABELS)
            or any(
                not math.isfinite(value) or not 0 <= value <= 1
                for value in self.per_label_f1.values()
            )
            or any(not math.isfinite(value) for value in numeric)
            or any(value is not None and not math.isfinite(value) for value in optional)
            or len(self.confusion_matrix) != len(CANONICAL_LABELS)
            or any(len(row) != len(CANONICAL_LABELS) for row in self.confusion_matrix)
            or any(value < 0 for row in self.confusion_matrix for value in row)
            or sum(value for row in self.confusion_matrix for value in row) != self.completed_count
        ):
            raise ValueError("invalid aggregate report")
        if self.provider == "wav2vec-xls-r" and any(value is None for value in optional):
            raise ValueError("missing CUDA metrics")
        return self

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class PilotDecision(_FrozenModel):
    eligible_a: bool
    eligible_b: bool
    winner: RecipeName | None
    reason: Literal[
        "no-eligible-pilot",
        "only-a-eligible",
        "only-b-eligible",
        "macro-f1",
        "ece",
        "pilot-b-tiebreak",
    ]


def pilot_eligible(report: AggregateReport, config: ExperimentConfig) -> bool:
    return (
        report.provider == "wav2vec-xls-r"
        and report.item_count == config.data.pilot_validation_count
        and report.completed_count == config.data.pilot_validation_count
        and report.macro_f1 > config.gates.pilot_macro_f1_min_exclusive
        and report.predicted_class_count >= config.gates.pilot_min_predicted_classes
        and sum(value > 0 for value in report.per_label_f1.values())
        >= config.gates.pilot_min_positive_label_f1
        and report.finite_training_state
        and report.provider_reload_verified
        and report.provenance_verified
        and report.permissions_private
        and report.privacy_verified
        and report.opened_test_count == 0
    )


def select_pilot_winner(
    pilot_a: AggregateReport,
    pilot_b: AggregateReport,
    config: ExperimentConfig,
) -> PilotDecision:
    eligible_a = pilot_eligible(pilot_a, config)
    eligible_b = pilot_eligible(pilot_b, config)
    if not eligible_a and not eligible_b:
        return PilotDecision(
            eligible_a=False,
            eligible_b=False,
            winner=None,
            reason="no-eligible-pilot",
        )
    if eligible_a and not eligible_b:
        return PilotDecision(
            eligible_a=True, eligible_b=False, winner="pilot-a", reason="only-a-eligible"
        )
    if eligible_b and not eligible_a:
        return PilotDecision(
            eligible_a=False, eligible_b=True, winner="pilot-b", reason="only-b-eligible"
        )
    macro_difference = pilot_a.macro_f1 - pilot_b.macro_f1
    if abs(macro_difference) >= 0.01:
        return PilotDecision(
            eligible_a=True,
            eligible_b=True,
            winner="pilot-a" if macro_difference > 0 else "pilot-b",
            reason="macro-f1",
        )
    ece_difference = pilot_a.expected_calibration_error - pilot_b.expected_calibration_error
    if abs(ece_difference) >= 0.01:
        return PilotDecision(
            eligible_a=True,
            eligible_b=True,
            winner="pilot-a" if ece_difference < 0 else "pilot-b",
            reason="ece",
        )
    return PilotDecision(
        eligible_a=True,
        eligible_b=True,
        winner="pilot-b",
        reason="pilot-b-tiebreak",
    )


def full_validation_eligible(report: AggregateReport, config: ExperimentConfig) -> bool:
    return (
        report.provider == "wav2vec-xls-r"
        and report.item_count == config.data.validation_count
        and report.completed_count == config.data.validation_count
        and report.macro_f1 > config.gates.full_macro_f1_min_exclusive
        and report.predicted_class_count == config.gates.full_required_predicted_classes
        and sum(value > 0 for value in report.per_label_f1.values())
        == config.gates.full_required_positive_label_f1
        and report.finite_training_state
        and report.provider_reload_verified
        and report.provenance_verified
        and report.permissions_private
        and report.privacy_verified
        and report.opened_test_count == 0
    )


class FrozenCandidate(_FrozenModel):
    schema_version: Literal["1"] = "1"
    candidate_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_rule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    holdout_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_token: str = Field(pattern=r"^[0-9a-f]{64}$")

    def token_payload(self) -> bytes:
        payload = self.model_dump(mode="json", exclude={"capability_token"})
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    def verify_token(self) -> bool:
        return hashlib.sha256(
            b"voxdelta-final-capability-v1\0" + self.token_payload()
        ).hexdigest() == (self.capability_token)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def freeze_candidate(
    report: AggregateReport,
    config: ExperimentConfig,
    *,
    baseline_checkpoint_sha256: str,
    baseline_config_sha256: str,
    metric_schema_sha256: str,
    decision_rule_sha256: str,
    holdout_identity_sha256: str,
) -> FrozenCandidate:
    if not full_validation_eligible(report, config):
        raise GateError("full_validation_gate_failed")
    payload = {
        "schema_version": "1",
        "candidate_checkpoint_sha256": report.checkpoint_sha256,
        "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
        "config_sha256": config.digest(),
        "validation_report_sha256": report.digest(),
        "baseline_config_sha256": baseline_config_sha256,
        "metric_schema_sha256": metric_schema_sha256,
        "decision_rule_sha256": decision_rule_sha256,
        "holdout_identity_sha256": holdout_identity_sha256,
    }
    token_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["capability_token"] = hashlib.sha256(
        b"voxdelta-final-capability-v1\0" + token_payload
    ).hexdigest()
    try:
        return FrozenCandidate.model_validate(payload)
    except Exception:
        raise GateError("invalid_freeze_identity") from None


def publish_frozen_candidate(path: Path, candidate: FrozenCandidate) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink() or not candidate.verify_token():
        raise GateError("candidate_freeze_failed")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write((candidate.model_dump_json() + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def consume_final_capability(
    candidate_path: Path,
    consumption_marker: Path,
    presented_token: str,
) -> FrozenCandidate:
    if (
        not candidate_path.is_absolute()
        or not consumption_marker.is_absolute()
        or consumption_marker.exists()
        or consumption_marker.is_symlink()
    ):
        raise GateError("final_capability_consumed")
    try:
        candidate = FrozenCandidate.model_validate_json(read_trusted_regular_file(candidate_path))
    except Exception:
        raise GateError("invalid_final_capability") from None
    if not candidate.verify_token() or presented_token != candidate.capability_token:
        raise GateError("invalid_final_capability")
    consumption_marker.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        with consumption_marker.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(
                (
                    json.dumps(
                        {
                            "schema_version": "1",
                            "capability_sha256": hashlib.sha256(
                                candidate.canonical_json().encode()
                            ).hexdigest(),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode()
            )
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise GateError("final_capability_consumed") from None
    return candidate


def build_authorized_final_package(
    source_manifest: Path,
    exposed_manifest: Path,
    output_root: Path,
    authorization_marker: Path,
    candidate: FrozenCandidate,
    presented_token: str,
    config: ExperimentConfig,
) -> Path:
    """Open/package final audio only after freeze-token and local authorization presence."""

    try:
        marker_mode = authorization_marker.stat(follow_symlinks=False).st_mode
    except OSError:
        raise GateError("final_package_not_authorized") from None
    if (
        not authorization_marker.is_absolute()
        or authorization_marker.is_symlink()
        or not authorization_marker.is_file()
        or marker_mode & 0o077
        or not candidate.verify_token()
        or presented_token != candidate.capability_token
    ):
        raise GateError("final_package_not_authorized")
    identity = derive_holdout_identity(source_manifest, exposed_manifest, config)
    if identity.identity_sha256 != candidate.holdout_identity_sha256:
        raise GateError("final_holdout_identity_mismatch")
    source = _emotion_items(source_manifest)
    exposed = _emotion_items(exposed_manifest)
    exposed_fingerprints = {semantic_fingerprint(item) for item in exposed if item.split == "test"}
    holdout = [
        item
        for item in source
        if item.split == "test" and semantic_fingerprint(item) not in exposed_fingerprints
    ]
    if (
        len(holdout) != config.data.final_holdout_count
        or hashlib.sha256(_identity_payload(holdout)).hexdigest()
        != candidate.holdout_identity_sha256
    ):
        raise GateError("final_holdout_identity_mismatch")
    return _publish_package(
        holdout,
        output_root,
        package_kind="final-holdout",
        archive_name="final-holdout.tar.zst",
        sidecar_name="final-holdout.sidecar.json",
    )


class FinalDecision(_FrozenModel):
    promote_xls_r: bool
    selected_provider: Literal["wav2vec-xls-r", "emotion2vec-plus"]
    reason: Literal["xls-r-wins", "baseline-retained"]


def decide_final_comparison(
    xls_r: AggregateReport,
    baseline: AggregateReport,
    config: ExperimentConfig,
) -> FinalDecision:
    exact_count = config.data.final_holdout_count
    integrity = all(
        (
            report.item_count == exact_count
            and report.completed_count == exact_count
            and report.provenance_verified
            and report.permissions_private
            and report.privacy_verified
            and report.opened_test_count == exact_count
        )
        for report in (xls_r, baseline)
    )
    promote = (
        integrity
        and xls_r.provider == "wav2vec-xls-r"
        and baseline.provider == "emotion2vec-plus"
        and xls_r.macro_f1 > baseline.macro_f1
        and all(value > 0 for value in xls_r.per_label_f1.values())
    )
    return FinalDecision(
        promote_xls_r=promote,
        selected_provider="wav2vec-xls-r" if promote else "emotion2vec-plus",
        reason="xls-r-wins" if promote else "baseline-retained",
    )


__all__ = [
    "AggregateReport",
    "FinalDecision",
    "FrozenCandidate",
    "GateError",
    "PilotDecision",
    "consume_final_capability",
    "build_authorized_final_package",
    "decide_final_comparison",
    "freeze_candidate",
    "full_validation_eligible",
    "pilot_eligible",
    "publish_frozen_candidate",
    "select_pilot_winner",
]
