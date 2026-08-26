"""Replay a local canary through a primary and a shadow candidate, and report aggregates.

This is a **local replay evaluator, not a live shadow**. Nothing here observes production
traffic; it re-runs a fixed local canary through two providers and records how they
compared. The distinction matters because the two answer different questions: a live
shadow tells you how the candidate behaves on real inputs, whereas this tells you only
that the wiring, the verification, and the comparison plumbing behave on inputs you chose.

The pipeline runner calls the emotion provider synchronously and persists its result
(`runner.py`, emotion stage), so today there is no non-blocking observation boundary a
production shadow could attach to. Adding a second synchronous inference there would put
candidate latency and candidate faults on the request path, which the shadow contract
exists to prevent. So the contract lives in
``voxdelta.providers.shadow_emotion`` and is exercised here, off the request path, until
such a boundary exists.

The primary is the **incumbent rollback baseline** — the emotion2vec checkpoint the final
decision compared against — and never the XLS-R release itself. Running the release
uncalibrated against the release calibrated measures only what temperature scaling does;
it is a calibration diagnostic and calling it a primary-versus-candidate shadow would
misrepresent what was compared. This evaluator therefore refuses to stand the release in
for the baseline.

Both sides are built under :func:`block_network_egress`, so "offline" is enforced and any
fetch attempt is detected even when the loader swallows its own connection error. A
primary that cannot be built under that guard is refused with a stable code; it is never
replaced by a fake, because a comparison against invented output is worse than none.
"""

from __future__ import annotations

import math
import re
import statistics
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voxdelta.evaluation.emotion_training import CANONICAL_LABELS
from voxdelta.evaluation.readiness import (
    MEMORY_CEILING_MB,
    ReadinessError,
    apply_offline_environment,
    canary_items_digest,
    offline_environment_enforced,
    select_canary_items,
    write_synthetic_canary,
)
from voxdelta.evaluation.run_publication import canonical_payload, publish_run_directory
from voxdelta.providers.encoder_bundle import VerifiedEncoderBundle, verify_encoder_bundle
from voxdelta.providers.offline_guard import NetworkEgressAttempted, block_network_egress
from voxdelta.providers.shadow_emotion import (
    ERROR_CATEGORIES,
    ShadowEmotionProvider,
    ShadowObservation,
    valid_emotion_result,
)

SHADOW_SCHEMA_VERSION = "1"
ARTIFACT_KIND = "shadow-replay"
MANIFEST_NAME = "SHADOW_REPLAY.json"
CHECKSUMS_NAME = "SHA256SUMS"

PrimaryKind = Literal["emotion2vec", "injected"]
# A short lowercase slug only: no separator, whitespace, or path character can survive.
PROVIDER_NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
_CANARY_PREFIX = "shadow-canary"


class ShadowReplayError(ValueError):
    """Fixed, path-free replay failure carrying one stable machine-readable code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _Frozen(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ProviderIdentity(_Frozen):
    """Which provider stood on each side, named without a path or a host detail.

    ``provider_name`` is constrained rather than trusted: an injected primary supplies
    its own provenance, and a free-form string written straight into a shared report is
    how a host path or a credential fragment escapes.
    """

    role: Literal["primary", "candidate"]
    kind: str = Field(min_length=1)
    provider_name: str = Field(min_length=1, max_length=64, pattern=PROVIDER_NAME_PATTERN)
    release_id: str | None = None
    bundle_tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    calibration_id: str | None = None
    binding_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Identity of the local encoder bundle, never its location.
    encoder_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    calibrated: bool


class LatencySummary(_Frozen):
    count: int = Field(ge=0)
    min_ms: float = Field(ge=0)
    median_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    max_ms: float = Field(ge=0)
    mean_ms: float = Field(ge=0)


class ErrorBreakdown(_Frozen):
    """Stable candidate error categories. Every key is a fixed vocabulary member."""

    none: int = Field(ge=0)
    timeout: int = Field(ge=0)
    provider_error: int = Field(ge=0)
    invalid_output: int = Field(ge=0)
    unexpected_error: int = Field(ge=0)


class ComparisonSummary(_Frozen):
    """Aggregate comparison. Carries no per-item value of any kind."""

    attempted_count: int = Field(gt=0)
    primary_completed_count: int = Field(ge=0)
    candidate_completed_count: int = Field(ge=0)
    candidate_error_count: int = Field(ge=0)
    errors: ErrorBreakdown
    candidate_attempted_count: int = Field(ge=0)
    candidate_valid_outputs: bool
    primary_invariant_holds: bool
    primary_invariance_exercised: bool
    top_label_agreement_rate: float = Field(ge=0, le=1)
    top_label_agreement_exercised: bool
    candidate_abstention_rate: float = Field(ge=0, le=1)
    candidate_abstention_exercised: bool
    candidate_uncertain_rate: float = Field(ge=0, le=1)
    mean_candidate_confidence: float | None = Field(default=None, ge=0, le=1)
    mean_candidate_raw_confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def counts_and_flags_must_reconcile(self) -> ComparisonSummary:
        errors = self.errors
        # One observation exists per primary result, so the two sides must add up.
        if self.candidate_completed_count + self.candidate_error_count != (
            self.primary_completed_count
        ):
            raise ValueError("invalid comparison summary")
        if self.primary_completed_count > self.attempted_count:
            raise ValueError("invalid comparison summary")
        if self.candidate_attempted_count > self.primary_completed_count:
            raise ValueError("invalid comparison summary")
        if self.candidate_completed_count > self.candidate_attempted_count:
            raise ValueError("invalid comparison summary")
        # A success is exactly a `none`; every other category is exactly a failure.
        if errors.none != self.candidate_completed_count:
            raise ValueError("invalid comparison summary")
        failures = (
            errors.timeout + errors.provider_error + errors.invalid_output + errors.unexpected_error
        )
        if failures != self.candidate_error_count:
            raise ValueError("invalid comparison summary")
        # Exercised flags must match their real denominators, never a hopeful default.
        if self.top_label_agreement_exercised != (self.candidate_completed_count > 0):
            raise ValueError("invalid comparison summary")
        # An output the candidate produced but that was unusable must fail this gate,
        # not slip past it because the observation never counted as a completion.
        if errors.invalid_output and self.candidate_valid_outputs:
            raise ValueError("invalid comparison summary")
        if self.candidate_abstention_exercised != (self.candidate_abstention_rate > 0):
            raise ValueError("invalid comparison summary")
        if self.primary_invariance_exercised != (self.primary_completed_count > 0):
            raise ValueError("invalid comparison summary")
        if self.candidate_completed_count == 0 and (
            self.top_label_agreement_rate
            or self.candidate_abstention_rate
            or self.candidate_uncertain_rate
            or self.candidate_abstention_exercised
            or self.mean_candidate_confidence is not None
        ):
            raise ValueError("invalid comparison summary")
        return self


class ShadowRuntime(_Frozen):
    requested_device: str = Field(min_length=1)
    offline_enforced: bool
    verified_before_model_allocation: bool
    # Structurally false: the replay never sets a candidate timeout. A timed-out
    # worker cannot be killed, and this runner's egress guard is process-wide and
    # temporary, so an abandoned worker could reach the network after the guard is
    # restored. Not creating one is the only safe answer here.
    candidate_timeout_used: Literal[False] = False
    elapsed_seconds: float = Field(ge=0)
    peak_rss_mb: float | None = Field(default=None, ge=0)
    memory_ceiling_mb: float = Field(gt=0)


class ShadowCanary(_Frozen):
    kind: Literal["validation", "synthetic"]
    split: Literal["validation", "synthetic"]
    item_count: int = Field(gt=0)
    repeats: int = Field(gt=0)
    label_coverage: int = Field(ge=0)
    audio_authenticated: bool
    quality_evidence: Literal[False] = False
    holdout_reachable: Literal[False] = False
    canary_items_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ShadowGates(_Frozen):
    """Blocking gates reuse existing product safety limits only."""

    identity_exact: bool
    offline_load: bool
    no_network_egress_attempted: bool
    verified_before_model_allocation: bool
    primary_invariant_holds: bool
    primary_completion_complete: bool
    candidate_completion_complete: bool
    candidate_valid_outputs: bool
    peak_rss_within_ceiling: bool
    live_shadow_traffic: Literal[False] = False
    overall_ok: bool = True

    @model_validator(mode="after")
    def overall_is_the_conjunction(self) -> ShadowGates:
        blocking = (
            self.identity_exact,
            self.offline_load,
            self.no_network_egress_attempted,
            self.verified_before_model_allocation,
            self.primary_invariant_holds,
            self.primary_completion_complete,
            self.candidate_completion_complete,
            self.candidate_valid_outputs,
            self.peak_rss_within_ceiling,
        )
        if self.overall_ok != all(blocking):
            raise ValueError("invalid shadow gates")
        return self


class ShadowReplayReport(_Frozen):
    """One aggregate-only, versioned local shadow-replay artifact."""

    schema_version: Literal["1"] = "1"
    artifact_kind: Literal["shadow-replay"] = "shadow-replay"
    mode: Literal["local-replay"] = "local-replay"
    primary: ProviderIdentity
    candidate: ProviderIdentity
    runtime: ShadowRuntime
    canary: ShadowCanary
    primary_latency: LatencySummary
    candidate_latency: LatencySummary
    comparison: ComparisonSummary
    gates: ShadowGates


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarise_latency(values: Sequence[float]) -> LatencySummary:
    if not values:
        return LatencySummary(
            count=0, min_ms=0.0, median_ms=0.0, p95_ms=0.0, max_ms=0.0, mean_ms=0.0
        )
    return LatencySummary(
        count=len(values),
        min_ms=min(values),
        median_ms=statistics.median(values),
        p95_ms=_percentile(values, 0.95),
        max_ms=max(values),
        mean_ms=math.fsum(values) / len(values),
    )


def summarise_observations(
    observations: Sequence[ShadowObservation],
    *,
    attempted: int,
    primary_completed: int,
    primary_invariant_holds: bool,
) -> ComparisonSummary:
    """Reduce raw observations to aggregates, marking anything unexercised as such."""

    counts = dict.fromkeys(ERROR_CATEGORIES, 0)
    for observation in observations:
        counts[observation.error_category] += 1
    ran = [observation for observation in observations if observation.candidate_completed]
    tried = [observation for observation in observations if observation.candidate_attempted]
    agreed = [observation for observation in ran if observation.top_labels_agree]
    abstained = [observation for observation in ran if observation.candidate_abstained]
    uncertain = [observation for observation in ran if observation.candidate_uncertain]
    confidences = [
        observation.candidate_confidence
        for observation in ran
        if observation.candidate_confidence is not None
    ]
    raw_confidences = [
        observation.candidate_raw_confidence
        for observation in ran
        if observation.candidate_raw_confidence is not None
    ]
    return ComparisonSummary(
        attempted_count=attempted,
        primary_completed_count=primary_completed,
        candidate_completed_count=len(ran),
        candidate_error_count=len(observations) - len(ran),
        errors=ErrorBreakdown(**counts),
        candidate_attempted_count=len(tried),
        candidate_valid_outputs=(
            counts["invalid_output"] == 0
            and all(observation.candidate_valid for observation in ran)
        ),
        primary_invariant_holds=primary_invariant_holds,
        primary_invariance_exercised=primary_completed > 0,
        top_label_agreement_rate=len(agreed) / len(ran) if ran else 0.0,
        top_label_agreement_exercised=bool(ran),
        candidate_abstention_rate=len(abstained) / len(ran) if ran else 0.0,
        candidate_abstention_exercised=bool(abstained),
        candidate_uncertain_rate=len(uncertain) / len(ran) if ran else 0.0,
        mean_candidate_confidence=(
            math.fsum(confidences) / len(confidences) if confidences else None
        ),
        mean_candidate_raw_confidence=(
            math.fsum(raw_confidences) / len(raw_confidences) if raw_confidences else None
        ),
    )


def write_shadow_report(directory: Path, report: ShadowReplayReport) -> Path:
    """Publish a private report as the final commit marker for a new run directory.

    The safe ordering and cleanup live in
    :func:`voxdelta.evaluation.run_publication.publish_run_directory`; this wrapper only
    binds them to the shadow manifest and error code.
    """

    try:
        return publish_run_directory(
            Path(directory),
            manifest_name=MANIFEST_NAME,
            checksums_name=CHECKSUMS_NAME,
            payload=canonical_payload(report.model_dump(mode="json")),
        )
    except Exception:
        raise ShadowReplayError("shadow_report_publication_failed") from None


def run_shadow_replay(
    *,
    release_path: Path,
    calibration_path: Path,
    primary_kind: PrimaryKind = "emotion2vec",
    primary_checkpoint: Path | None = None,
    primary_encoder_bundle: Path | None = None,
    primary_provider: object | None = None,
    device: str = "auto",
    repeats: int = 3,
    validation_manifest: Path | None = None,
    synthetic_item_count: int = 7,
    workspace: Path | None = None,
) -> ShadowReplayReport:
    """Replay a local canary through the rollback primary and the shadow candidate.

    The primary is the incumbent emotion2vec baseline, built from an explicit local
    checkpoint under an enforced network-egress guard, or an injected provider supplied by
    a caller that has already established its provenance. The XLS-R release is never
    accepted as its own primary. Raises ``ShadowReplayError`` with a stable, path-free
    code.
    """

    from voxdelta.providers.calibrated_emotion import CalibratedEmotionProvider
    from voxdelta.providers.calibration_artifact import verify_calibration_artifact
    from voxdelta.providers.release_bundle import verify_release_bundle
    from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

    if repeats < 1 or synthetic_item_count < 1:
        raise ShadowReplayError("invalid_shadow_input")
    release = Path(release_path)
    calibration = Path(calibration_path)
    if not release.is_absolute() or not calibration.is_absolute():
        raise ShadowReplayError("invalid_shadow_input")
    if primary_kind == "injected" and primary_provider is None:
        raise ShadowReplayError("invalid_shadow_input")
    if primary_kind == "emotion2vec" and primary_provider is not None:
        raise ShadowReplayError("invalid_shadow_input")

    apply_offline_environment()
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix=".shadow-", dir=workspace) as scratch:
        root = Path(scratch)
        if validation_manifest is not None:
            try:
                items = select_canary_items(Path(validation_manifest))
            except ReadinessError as error:
                raise ShadowReplayError(error.code) from None
            clips = [Path(item.audio_path) for item in items]
            kind: Literal["validation", "synthetic"] = "validation"
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
            attested = verify_release_bundle(release)
            verified = verify_calibration_artifact(calibration, release=attested)
        except Exception:
            raise ShadowReplayError("shadow_artifacts_rejected") from None

        # The guard stays open across construction, lazy allocation, and every
        # inference: a provider that only reaches the network on first use would
        # otherwise escape a guard that closed after construction.
        with block_network_egress() as egress:
            try:
                candidate_inner = Wav2VecEmotionProvider(
                    attested.checkpoint_path,
                    base_model_path=attested.base_model_path,
                    device=device,  # type: ignore[arg-type]
                )
            except Exception:
                raise ShadowReplayError("shadow_provider_construction_failed") from None

            # Observed here, before anything forces a load: this is a statement about
            # the candidate's own artifacts, so a primary that is already resident
            # must not be able to falsify it.
            verified_before_allocation = getattr(candidate_inner, "model_loaded", True) is False

            resolved_primary = primary_provider
            if resolved_primary is None:
                resolved_primary, encoder_bundle = _build_emotion2vec_primary(
                    primary_checkpoint, primary_encoder_bundle, device, egress
                )
            else:
                encoder_bundle = None

            candidate = CalibratedEmotionProvider(candidate_inner, verified)
            recorder = _PrimaryRecorder(resolved_primary)
            observations: list[ShadowObservation] = []
            # No timeout, deliberately: see ``candidate_timeout_used``. Without one the
            # candidate runs inline, so no worker can outlive this guarded block.
            shadow = ShadowEmotionProvider(
                recorder,  # type: ignore[arg-type]
                candidate,
                observer=observations.append,
            )

            primary_latencies: list[float] = []
            candidate_latencies: list[float] = []
            primary_completed = 0
            invariant_holds = True
            peak_rss: float | None = None
            try:
                for repeat in range(repeats):
                    for index, clip in enumerate(clips):
                        utterance = f"{_CANARY_PREFIX}-{repeat:02d}-{index:04d}"
                        before = len(observations)
                        try:
                            # Exactly one primary inference per item: the adapter must
                            # hand back that call's own object, never a second run.
                            returned = shadow.analyze(utterance, clip, "")
                        except Exception:
                            # A shadow that can fail the primary has already broken its
                            # contract; record it rather than hiding it.
                            invariant_holds = False
                            continue
                        primary_completed += 1
                        if (
                            returned is not recorder.last
                            or len(observations) != before + 1
                            or not valid_emotion_result(returned)
                        ):
                            invariant_holds = False
                        usage = returned.usage
                        if usage is not None:
                            primary_latencies.append(usage.latency_ms)
                            if usage.peak_rss_mb is not None:
                                peak_rss = max(peak_rss or 0.0, usage.peak_rss_mb)
                        latest = observations[-1] if observations else None
                        if latest is not None:
                            if latest.candidate_latency_ms is not None:
                                candidate_latencies.append(latest.candidate_latency_ms)
                            if latest.candidate_peak_rss_mb is not None:
                                peak_rss = max(peak_rss or 0.0, latest.candidate_peak_rss_mb)
                        if shadow.candidate_timed_out:
                            # Unreachable while this runner sets no timeout, and kept as
                            # a backstop: an abandoned worker must never outlive the
                            # guarded block.
                            raise ShadowReplayError("shadow_candidate_timed_out")
            finally:
                shadow.unload()

            egress_attempted = egress.attempted

        attempted = len(clips) * repeats
        comparison = summarise_observations(
            observations,
            attempted=attempted,
            primary_completed=primary_completed,
            primary_invariant_holds=invariant_holds,
        )
        offline_ok = offline_environment_enforced()
        peak_within = peak_rss is not None and peak_rss <= MEMORY_CEILING_MB
        candidate_complete = comparison.candidate_completed_count == attempted
        identity_exact = (
            verified.release_id == attested.release_id
            and verified.bundle_tree_sha256 == attested.bundle_tree_sha256
            and verified.candidate_checkpoint_sha256 == attested.candidate_checkpoint_sha256
            and tuple(attested.labels) == tuple(CANONICAL_LABELS)
        )
        complete = primary_completed == attempted

        return ShadowReplayReport(
            primary=ProviderIdentity(
                role="primary",
                kind=primary_kind,
                provider_name=_published_provider_name(
                    resolved_primary.provenance.name  # type: ignore[attr-defined]
                ),
                encoder_bundle_sha256=(
                    None if encoder_bundle is None else encoder_bundle.bundle_tree_sha256
                ),
                calibrated=False,
            ),
            candidate=ProviderIdentity(
                role="candidate",
                kind="release-calibrated",
                provider_name=_published_provider_name(candidate.provenance.name),
                release_id=attested.release_id,
                bundle_tree_sha256=attested.bundle_tree_sha256,
                calibration_id=verified.calibration_id,
                binding_sha256=verified.binding_sha256,
                calibrated=True,
            ),
            runtime=ShadowRuntime(
                requested_device=device,
                offline_enforced=offline_ok,
                verified_before_model_allocation=verified_before_allocation,
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                peak_rss_mb=peak_rss,
                memory_ceiling_mb=MEMORY_CEILING_MB,
            ),
            canary=ShadowCanary(
                kind=kind,
                split=kind,
                item_count=len(clips),
                repeats=repeats,
                label_coverage=label_coverage,
                audio_authenticated=authenticated,
                canary_items_sha256=items_digest,
            ),
            primary_latency=summarise_latency(primary_latencies),
            candidate_latency=summarise_latency(candidate_latencies),
            comparison=comparison,
            gates=ShadowGates(
                identity_exact=identity_exact,
                offline_load=offline_ok,
                no_network_egress_attempted=not egress_attempted,
                verified_before_model_allocation=verified_before_allocation,
                primary_invariant_holds=invariant_holds,
                primary_completion_complete=complete,
                candidate_completion_complete=candidate_complete,
                candidate_valid_outputs=comparison.candidate_valid_outputs,
                peak_rss_within_ceiling=peak_within,
                overall_ok=(
                    identity_exact
                    and offline_ok
                    and not egress_attempted
                    and verified_before_allocation
                    and invariant_holds
                    and complete
                    and candidate_complete
                    and comparison.candidate_valid_outputs
                    and peak_within
                ),
            ),
        )


def _published_provider_name(name: object) -> str:
    """Refuse a provider name that could carry a path, a host, or whitespace."""

    if not isinstance(name, str) or not re.fullmatch(PROVIDER_NAME_PATTERN, name):
        raise ShadowReplayError("shadow_provider_name_rejected")
    return name


class _PrimaryRecorder:
    """Pass-through that remembers the exact object the primary returned.

    The invariance check is object identity, not a second inference: re-running a
    nondeterministic primary and diffing the two results would report a model property
    as an adapter defect.
    """

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.last: object | None = None
        self.provenance = inner.provenance  # type: ignore[attr-defined]

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> object:
        result = self._inner.analyze(utterance_id, audio_path, transcript)  # type: ignore[attr-defined]
        self.last = result
        return result

    def unload(self) -> None:
        unload = getattr(self._inner, "unload", None)
        if callable(unload):
            unload()


def _build_emotion2vec_primary(
    checkpoint: Path | None,
    encoder_bundle: Path | None,
    device: str,
    egress: object,
) -> tuple[object, VerifiedEncoderBundle]:
    """Build the incumbent rollback baseline from explicit, verified local assets.

    The encoder bundle is verified *before* the provider is constructed and long before
    any weight is allocated, so an edited, truncated, symlinked, or half-published bundle
    is refused while it is still just a path. A raw package cache is not accepted: only a
    bundle that has passed the production verifier.
    """

    from voxdelta.providers.emotion2vec_emotion import Emotion2VecEmotionProvider

    if checkpoint is None or not Path(checkpoint).is_absolute():
        raise ShadowReplayError("invalid_shadow_input")
    if encoder_bundle is None or not Path(encoder_bundle).is_absolute():
        raise ShadowReplayError("invalid_shadow_input")
    try:
        verified_encoder = verify_encoder_bundle(Path(encoder_bundle))
    except Exception:
        raise ShadowReplayError("shadow_encoder_bundle_rejected") from None

    try:
        provider = Emotion2VecEmotionProvider(
            Path(checkpoint),
            device=device,  # type: ignore[arg-type]
            encoder_source=verified_encoder.path,
        )
    except NetworkEgressAttempted:
        raise ShadowReplayError("shadow_primary_not_available_offline") from None
    except Exception:
        if getattr(egress, "attempted", False):
            raise ShadowReplayError("shadow_primary_not_available_offline") from None
        raise ShadowReplayError("shadow_primary_checkpoint_rejected") from None
    # Constructing only validates metadata; force allocation here so a fetch cannot hide
    # behind lazy loading and surface later as a mid-run failure.
    try:
        warm = getattr(provider, "_load", None)
        if callable(warm):
            warm()
    except Exception:
        if getattr(egress, "attempted", False):
            raise ShadowReplayError("shadow_primary_not_available_offline") from None
        raise ShadowReplayError("shadow_primary_allocation_failed") from None
    if getattr(egress, "attempted", False):
        raise ShadowReplayError("shadow_primary_not_available_offline")
    return provider, verified_encoder


__all__ = [
    "ARTIFACT_KIND",
    "CHECKSUMS_NAME",
    "MANIFEST_NAME",
    "SHADOW_SCHEMA_VERSION",
    "ComparisonSummary",
    "ErrorBreakdown",
    "LatencySummary",
    "PrimaryKind",
    "ProviderIdentity",
    "ShadowCanary",
    "ShadowGates",
    "ShadowReplayError",
    "ShadowReplayReport",
    "ShadowRuntime",
    "run_shadow_replay",
    "summarise_latency",
    "summarise_observations",
    "write_shadow_report",
]
