"""Local production-readiness and canary measurement for the promoted XLS-R release.

This runner answers one question before any rollout: does the product's own startup path,
with both opt-in XLS-R flags enabled, bring up a verified release and its separately
published calibration, and does that provider then behave inside the limits the project
already enforces? It is a measurement tool, not a deployment tool: it contacts nothing,
mutates no artifact, and performs no traffic rollout.

Three properties matter and are enforced here rather than assumed.

The release and the calibration are verified through the production dependency builder
*before* the heavy model is allocated, so a rejected artifact never reaches a model load.
The Hugging Face runtime is pinned offline before that builder runs, so a stale cache
cannot silently substitute a remote download. And the published report is aggregate-only:
no transcript, item identity, source path, per-item prediction, or raw probability is
recorded or printed, because a readiness artifact is meant to be shareable evidence.

Inputs are validation-only or deterministic synthetic audio. There is no argument by which
the sealed final holdout can be named, and a manifest carrying any non-validation item is
refused before a single inference runs.

No latency service level is invented here. Only gates the project already enforces are
blocking; any latency threshold is supplied by the operator and is advisory by contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import struct
import tempfile
import time
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voxdelta.domain.models import EmotionLabel, EmotionResult
from voxdelta.evaluation.emotion_training import CANONICAL_LABELS
from voxdelta.evaluation.manifest import DatasetItem, load_manifest, read_trusted_regular_file
from voxdelta.evaluation.run_publication import canonical_payload, publish_run_directory
from voxdelta.providers._emotion_runtime import Device

READINESS_SCHEMA_VERSION = "1"
ARTIFACT_KIND = "release-readiness"
MANIFEST_NAME = "READINESS.json"
CHECKSUMS_NAME = "SHA256SUMS"

# The existing project-wide candidate ceiling, reused rather than reinvented.
MEMORY_CEILING_MB = 18_432.0
MINIMUM_COMPLETION_RATE = 1.0

# Reloading must never reach the network, whatever a stale cache or environment would
# otherwise permit. These are set before the dependency builder allocates anything.
OFFLINE_ENVIRONMENT: tuple[tuple[str, str], ...] = (
    ("HF_HUB_OFFLINE", "1"),
    ("TRANSFORMERS_OFFLINE", "1"),
    ("HF_DATASETS_OFFLINE", "1"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
)

_CANARY_SPLIT = "validation"
_SYNTHETIC_SAMPLE_RATE = 16_000
_SYNTHETIC_DURATION_SECONDS = 2
_CANARY_PREFIX = "readiness-canary"

InputKind = Literal["validation", "synthetic"]


class ReadinessError(ValueError):
    """Fixed, path-free readiness failure carrying one stable machine-readable code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _Frozen(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ReleaseIdentity(_Frozen):
    """Exactly which immutable release was verified and loaded."""

    release_id: str = Field(min_length=1)
    bundle_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels: tuple[str, ...]


class CalibrationIdentity(_Frozen):
    """Exactly which separately published calibration was bound to that release."""

    calibration_id: str = Field(min_length=1)
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    method: str = Field(min_length=1)
    temperature: float = Field(gt=0)
    abstain_threshold: float = Field(ge=0, le=1)
    target_coverage: float = Field(gt=0, le=1)
    fitted_item_count: int = Field(gt=0)
    fitted_achieved_coverage: float = Field(gt=0, le=1)


class RuntimeProfile(_Frozen):
    """What the run cost, on which device, under which ceiling."""

    requested_device: Device
    selected_device: str = Field(min_length=1)
    offline_enforced: bool
    verified_before_model_allocation: bool
    startup_verify_ms: float = Field(ge=0)
    cold_first_inference_ms: float = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    peak_rss_mb: float | None = Field(default=None, ge=0)
    memory_ceiling_mb: float = Field(gt=0)


class CanaryInput(_Frozen):
    """What was fed in, described without naming any item or path."""

    kind: InputKind
    split: Literal["validation", "synthetic"]
    item_count: int = Field(gt=0)
    repeats: int = Field(gt=0)
    label_coverage: int = Field(ge=0)
    audio_authenticated: bool
    # Validation audio is replayed but never scored against its reference label, so this
    # run is runtime evidence only. Setting it true would require an agreed quality
    # contract that scores references; label agreement is not accuracy.
    quality_evidence: Literal[False] = False
    holdout_reachable: Literal[False] = False
    canary_items_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class LatencySummary(_Frozen):
    """Warm per-item latency, excluding the cold first inference."""

    count: int = Field(gt=0)
    min_ms: float = Field(ge=0)
    median_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    max_ms: float = Field(ge=0)
    mean_ms: float = Field(ge=0)


class OutcomeSummary(_Frozen):
    """Aggregate calibrated behaviour. No per-item value appears here."""

    attempted_count: int = Field(gt=0)
    completed_count: int = Field(ge=0)
    completion_rate: float = Field(ge=0, le=1)
    finite_valid_outputs: bool
    abstained_count: int = Field(ge=0)
    abstention_rate: float = Field(ge=0, le=1)
    abstention_mapping_exercised: bool
    uncertain_state_count: int = Field(ge=0)
    mean_calibrated_confidence: float = Field(ge=0, le=1)
    mean_raw_confidence: float = Field(ge=0, le=1)
    min_calibrated_confidence: float = Field(ge=0, le=1)
    max_calibrated_confidence: float = Field(ge=0, le=1)
    top_label_agreement_rate: float = Field(ge=0, le=1)
    agreement_sample_count: int = Field(ge=0)
    # False means this canary produced no agreement sample at all, so that gate would
    # rest on regression tests rather than on anything observed in this run.
    top_label_agreement_exercised: bool


class ReadinessGates(_Frozen):
    """Only pre-existing safety gates block. The latency threshold is advisory."""

    identity_exact: bool
    offline_load: bool
    verified_before_model_allocation: bool
    completion_complete: bool
    finite_valid_outputs: bool
    peak_rss_within_ceiling: bool
    abstention_maps_to_uncertain: bool
    top_label_preserved: bool
    overall_ok: bool
    advisory_p95_latency_ms: float | None = Field(default=None, gt=0)
    advisory_p95_within_threshold: bool | None = None

    @model_validator(mode="after")
    def advisory_never_blocks(self) -> ReadinessGates:
        blocking = (
            self.identity_exact,
            self.offline_load,
            self.verified_before_model_allocation,
            self.completion_complete,
            self.finite_valid_outputs,
            self.peak_rss_within_ceiling,
            self.abstention_maps_to_uncertain,
            self.top_label_preserved,
        )
        if self.overall_ok != all(blocking):
            raise ValueError("invalid readiness gates")
        if (self.advisory_p95_latency_ms is None) != (self.advisory_p95_within_threshold is None):
            raise ValueError("invalid readiness gates")
        return self


class ReadinessReport(_Frozen):
    """One aggregate-only, versioned readiness artifact."""

    schema_version: Literal["1"] = "1"
    artifact_kind: Literal["release-readiness"] = "release-readiness"
    release: ReleaseIdentity
    calibration: CalibrationIdentity
    runtime: RuntimeProfile
    canary: CanaryInput
    latency: LatencySummary
    outcome: OutcomeSummary
    gates: ReadinessGates


def apply_offline_environment(environment: dict[str, str] | None = None) -> None:
    """Pin the Hugging Face runtime to local files before any dependency is built."""

    target = os.environ if environment is None else environment
    for name, value in OFFLINE_ENVIRONMENT:
        target[name] = value


def offline_environment_enforced(environment: dict[str, str] | None = None) -> bool:
    source = os.environ if environment is None else environment
    return all(source.get(name) == value for name, value in OFFLINE_ENVIRONMENT)


def canary_items_digest(items: Sequence[DatasetItem]) -> str:
    """A reproducible identity for exactly which canary items ran, revealing no id."""

    if not items:
        raise ReadinessError("invalid_readiness_input")
    rows = sorted(
        ({"item_key": item.id, "emotion": item.emotion} for item in items),
        key=lambda row: str(row["item_key"]),
    )
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def select_canary_items(manifest: Path) -> list[DatasetItem]:
    """Take one authenticated validation item per label, deterministically.

    Every record in the manifest must be a labelled validation item. A single `test`
    row makes the whole manifest unusable, so the sealed split cannot be reached by
    hiding one line among many. Audio bytes are re-hashed against the manifest before
    any of them is handed to a model.
    """

    try:
        items = load_manifest(manifest)
    except Exception:
        raise ReadinessError("invalid_readiness_input") from None
    if not items:
        raise ReadinessError("invalid_readiness_input")
    for item in items:
        if item.split != _CANARY_SPLIT or item.emotion is None:
            raise ReadinessError("readiness_input_not_validation_only")

    chosen: dict[EmotionLabel, DatasetItem] = {}
    for item in sorted(items, key=lambda record: record.id):
        label = item.emotion
        if label is not None and label not in chosen:
            chosen[label] = item
    selected = [chosen[label] for label in CANONICAL_LABELS if label in chosen]
    if not selected:
        raise ReadinessError("invalid_readiness_input")

    for item in selected:
        try:
            payload = read_trusted_regular_file(item.audio_path)
        except Exception:
            raise ReadinessError("readiness_audio_unreadable") from None
        if hashlib.sha256(payload).hexdigest() != item.sha256:
            raise ReadinessError("readiness_audio_mismatch")
    return selected


def _synthetic_samples(index: int) -> tuple[int, ...]:
    """A fixed, reproducible voiced-like waveform. No RNG and no recorded audio."""

    count = _SYNTHETIC_SAMPLE_RATE * _SYNTHETIC_DURATION_SECONDS
    base = 180.0 + 20.0 * index
    values: list[int] = []
    for position in range(count):
        seconds = position / _SYNTHETIC_SAMPLE_RATE
        amplitude = (
            0.40 * math.sin(2 * math.pi * base * seconds)
            + 0.25 * math.sin(2 * math.pi * (base * 2) * seconds + 0.5)
            + 0.10 * math.sin(2 * math.pi * 55.0 * seconds)
        )
        values.append(max(-32768, min(32767, int(amplitude * 12000.0))))
    return tuple(values)


def write_synthetic_canary(directory: Path, count: int) -> list[Path]:
    """Publish deterministic private clips. Runtime evidence only, never quality evidence."""

    if count < 1:
        raise ReadinessError("invalid_readiness_input")
    written: list[Path] = []
    for index in range(count):
        path = directory / f"{_CANARY_PREFIX}-{index:04d}.wav"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with open(descriptor, "wb") as stream:
            with wave.open(stream, "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(_SYNTHETIC_SAMPLE_RATE)
                samples = _synthetic_samples(index)
                writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        written.append(path)
    return written


@dataclass(frozen=True, slots=True)
class _Measurement:
    latency_ms: float
    rss_mb: float | None
    calibrated_confidence: float
    raw_confidence: float
    abstained: bool
    uncertain: bool
    finite: bool


def _valid_distribution(result: EmotionResult) -> bool:
    values = result.probabilities
    if set(values) != set(CANONICAL_LABELS):
        return False
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        return False
    return abs(math.fsum(values.values()) - 1.0) <= 1e-6


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile; deterministic and defined for a single sample."""

    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _top_label(probabilities: dict[EmotionLabel, float]) -> str:
    return max(probabilities.items(), key=lambda pair: (pair[1], pair[0]))[0]


def write_readiness_report(directory: Path, report: ReadinessReport) -> Path:
    """Publish a private report as the final commit marker for a new run directory.

    The safe ordering and cleanup live in
    :func:`voxdelta.evaluation.run_publication.publish_run_directory`; this wrapper only
    binds them to the readiness manifest and error code.
    """

    try:
        return publish_run_directory(
            Path(directory),
            manifest_name=MANIFEST_NAME,
            checksums_name=CHECKSUMS_NAME,
            payload=canonical_payload(report.model_dump(mode="json")),
        )
    except Exception:
        raise ReadinessError("readiness_report_publication_failed") from None


def _selected_device_name(requested: Device) -> str:
    """Report the device actually chosen, falling back to the request when opaque."""

    from voxdelta.providers._emotion_runtime import default_hardware_probe, select_device

    try:
        return select_device(requested, default_hardware_probe)
    except Exception:
        return requested


def run_readiness_check(
    *,
    release_path: Path,
    calibration_path: Path,
    device: Device = "auto",
    repeats: int = 3,
    validation_manifest: Path | None = None,
    synthetic_item_count: int = 7,
    advisory_p95_latency_ms: float | None = None,
    workspace: Path | None = None,
) -> ReadinessReport:
    """Bring the product startup path up on a real release and measure what it does.

    Verification of both artifacts happens inside the dependency builder, before the
    model is allocated; that ordering is asserted rather than assumed. Raises
    ``ReadinessError`` with a stable, path-free code for every rejection.
    """

    # Deferred: the API dependency graph imports providers, which import this package.
    from voxdelta.api.dependencies import build_dependencies
    from voxdelta.config import Settings
    from voxdelta.providers.calibrated_emotion import CalibratedEmotionProvider, calibrate_result
    from voxdelta.providers.release_bundle import verify_release_bundle

    if repeats < 1 or synthetic_item_count < 1:
        raise ReadinessError("invalid_readiness_input")
    if advisory_p95_latency_ms is not None and (
        not math.isfinite(advisory_p95_latency_ms) or advisory_p95_latency_ms <= 0
    ):
        raise ReadinessError("invalid_readiness_input")
    release = Path(release_path)
    calibration = Path(calibration_path)
    if not release.is_absolute() or not calibration.is_absolute():
        raise ReadinessError("invalid_readiness_input")

    apply_offline_environment()
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix=".readiness-", dir=workspace) as scratch:
        root = Path(scratch)

        if validation_manifest is not None:
            items = select_canary_items(Path(validation_manifest))
            clips = [Path(item.audio_path) for item in items]
            kind: InputKind = "validation"
            label_coverage = len({item.emotion for item in items})
            items_digest: str | None = canary_items_digest(items)
            authenticated = True
        else:
            clips = write_synthetic_canary(root, synthetic_item_count)
            kind = "synthetic"
            label_coverage = 0
            items_digest = None
            authenticated = False

        try:
            settings = Settings(
                data_root=root,
                database_path=root / "readiness.sqlite3",
                diarization_provider="fake",
                asr_provider="fake",
                emotion_provider="wav2vec",
                emotion_checkpoint_path=None,
                emotion_device=device,
                xlsr_release_enabled=True,
                xlsr_release_path=release,
                xlsr_calibration_enabled=True,
                xlsr_calibration_path=calibration,
            )
        except Exception:
            raise ReadinessError("invalid_readiness_settings") from None

        # Verify the release independently first, so the identity gate is a real
        # comparison against a second verification rather than an assumption that
        # the dependency builder must have checked something.
        try:
            attested = verify_release_bundle(release)
        except Exception:
            raise ReadinessError("readiness_release_rejected") from None

        verify_started = time.perf_counter()
        try:
            dependencies = build_dependencies(settings)
        except Exception:
            raise ReadinessError("readiness_startup_rejected") from None
        startup_verify_ms = (time.perf_counter() - verify_started) * 1000

        provider = dependencies.runner.emotion_provider
        if not isinstance(provider, CalibratedEmotionProvider):
            raise ReadinessError("readiness_provider_not_calibrated")
        inner = provider.inner
        verified_before_allocation = getattr(inner, "model_loaded", True) is False
        verified = provider.calibration

        measurements: list[_Measurement] = []
        cold_first_inference_ms = 0.0
        agreements = 0
        agreement_samples = 0
        try:
            for repeat in range(repeats):
                for index, clip in enumerate(clips):
                    utterance = f"{_CANARY_PREFIX}-{repeat:02d}-{index:04d}"
                    cold = not measurements
                    began = time.perf_counter()
                    try:
                        result = provider.analyze(utterance, clip, "")
                    except Exception:
                        raise ReadinessError("readiness_inference_failed") from None
                    observed = (time.perf_counter() - began) * 1000
                    if cold:
                        cold_first_inference_ms = observed
                    record = result.calibration
                    if record is None:
                        raise ReadinessError("readiness_provider_not_calibrated")
                    usage = result.usage
                    measurements.append(
                        _Measurement(
                            latency_ms=usage.latency_ms if usage is not None else observed,
                            rss_mb=usage.peak_rss_mb if usage is not None else None,
                            calibrated_confidence=result.confidence,
                            raw_confidence=record.raw_confidence,
                            abstained=record.abstained,
                            uncertain=result.operational_state == "uncertain",
                            finite=_valid_distribution(result),
                        )
                    )

            # One extra uncalibrated pass per distinct clip, so the claim that
            # temperature scaling preserves the top label is measured on this audio
            # and not merely asserted in tests. Nothing per-item is retained.
            for index, clip in enumerate(clips):
                try:
                    raw = inner.analyze(f"{_CANARY_PREFIX}-raw-{index:04d}", clip, "")
                    scaled = calibrate_result(raw, verified)
                except Exception:
                    continue
                agreement_samples += 1
                agreements += int(_top_label(raw.probabilities) == _top_label(scaled.probabilities))
        finally:
            unload = getattr(provider, "unload", None)
            if callable(unload):
                unload()

        if not measurements:
            raise ReadinessError("readiness_inference_failed")

        attempted = len(clips) * repeats
        completed = len(measurements)
        warm = [record.latency_ms for record in measurements[1:]] or [measurements[0].latency_ms]
        rss_values = [record.rss_mb for record in measurements if record.rss_mb is not None]
        peak_rss = max(rss_values) if rss_values else None
        abstained = sum(record.abstained for record in measurements)
        calibrated = [record.calibrated_confidence for record in measurements]
        finite_ok = all(record.finite for record in measurements)
        completion_rate = completed / attempted
        # With no sample there is no evidence, so the gate must not pass by default.
        agreement_rate = agreements / agreement_samples if agreement_samples else 0.0
        preserved = agreement_samples > 0 and agreements == agreement_samples

        latency = LatencySummary(
            count=len(warm),
            min_ms=min(warm),
            median_ms=statistics.median(warm),
            p95_ms=_percentile(warm, 0.95),
            max_ms=max(warm),
            mean_ms=math.fsum(warm) / len(warm),
        )
        outcome = OutcomeSummary(
            attempted_count=attempted,
            completed_count=completed,
            completion_rate=completion_rate,
            finite_valid_outputs=finite_ok,
            abstained_count=abstained,
            abstention_rate=abstained / completed,
            abstention_mapping_exercised=abstained > 0,
            uncertain_state_count=sum(record.uncertain for record in measurements),
            mean_calibrated_confidence=math.fsum(calibrated) / len(calibrated),
            mean_raw_confidence=math.fsum(record.raw_confidence for record in measurements)
            / completed,
            min_calibrated_confidence=min(calibrated),
            max_calibrated_confidence=max(calibrated),
            top_label_agreement_rate=agreement_rate,
            agreement_sample_count=agreement_samples,
            top_label_agreement_exercised=agreement_samples > 0,
        )

        offline_ok = offline_environment_enforced()
        peak_within = peak_rss is not None and peak_rss <= MEMORY_CEILING_MB
        abstention_consistent = all(record.uncertain for record in measurements if record.abstained)
        complete = completion_rate >= MINIMUM_COMPLETION_RATE
        identity_exact = (
            verified.release_id == attested.release_id
            and verified.bundle_tree_sha256 == attested.bundle_tree_sha256
            and verified.candidate_checkpoint_sha256 == attested.candidate_checkpoint_sha256
            and tuple(attested.labels) == tuple(CANONICAL_LABELS)
        )
        gates = ReadinessGates(
            identity_exact=identity_exact,
            offline_load=offline_ok,
            verified_before_model_allocation=verified_before_allocation,
            completion_complete=complete,
            finite_valid_outputs=finite_ok,
            peak_rss_within_ceiling=peak_within,
            abstention_maps_to_uncertain=abstention_consistent,
            top_label_preserved=preserved,
            advisory_p95_latency_ms=advisory_p95_latency_ms,
            advisory_p95_within_threshold=(
                None
                if advisory_p95_latency_ms is None
                else latency.p95_ms <= advisory_p95_latency_ms
            ),
            overall_ok=(
                identity_exact
                and offline_ok
                and verified_before_allocation
                and complete
                and finite_ok
                and peak_within
                and abstention_consistent
                and preserved
            ),
        )

        return ReadinessReport(
            release=ReleaseIdentity(
                release_id=attested.release_id,
                bundle_tree_sha256=attested.bundle_tree_sha256,
                candidate_checkpoint_sha256=attested.candidate_checkpoint_sha256,
                labels=tuple(attested.labels),
            ),
            calibration=CalibrationIdentity(
                calibration_id=verified.calibration_id,
                binding_sha256=verified.binding_sha256,
                method=verified.summary.method,
                temperature=verified.temperature,
                abstain_threshold=verified.abstain_threshold,
                target_coverage=verified.summary.target_coverage,
                fitted_item_count=verified.summary.fitted_item_count,
                fitted_achieved_coverage=verified.summary.achieved_coverage,
            ),
            runtime=RuntimeProfile(
                requested_device=device,
                selected_device=_selected_device_name(device),
                offline_enforced=offline_ok,
                verified_before_model_allocation=verified_before_allocation,
                startup_verify_ms=startup_verify_ms,
                cold_first_inference_ms=cold_first_inference_ms,
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                peak_rss_mb=peak_rss,
                memory_ceiling_mb=MEMORY_CEILING_MB,
            ),
            canary=CanaryInput(
                kind=kind,
                split="validation" if kind == "validation" else "synthetic",
                item_count=len(clips),
                repeats=repeats,
                label_coverage=label_coverage,
                audio_authenticated=authenticated,
                # Labels select a representative runtime canary, but this runner does
                # not score predictions against them.  Accuracy therefore remains out
                # of scope even when the audio came from validation.
                quality_evidence=False,
                canary_items_sha256=items_digest,
            ),
            latency=latency,
            outcome=outcome,
            gates=gates,
        )


__all__ = [
    "ARTIFACT_KIND",
    "CHECKSUMS_NAME",
    "MANIFEST_NAME",
    "MEMORY_CEILING_MB",
    "MINIMUM_COMPLETION_RATE",
    "OFFLINE_ENVIRONMENT",
    "READINESS_SCHEMA_VERSION",
    "CalibrationIdentity",
    "CanaryInput",
    "LatencySummary",
    "OutcomeSummary",
    "ReadinessError",
    "ReadinessGates",
    "ReadinessReport",
    "ReleaseIdentity",
    "RuntimeProfile",
    "apply_offline_environment",
    "canary_items_digest",
    "offline_environment_enforced",
    "run_readiness_check",
    "select_canary_items",
    "write_readiness_report",
    "write_synthetic_canary",
]
