"""Run a candidate emotion provider beside a primary one without ever risking the primary.

A shadow comparison is only worth having if it cannot hurt the thing it observes. This
adapter therefore has one hard rule: whatever the candidate does — succeed, fail, hang,
or return nonsense — the caller receives exactly the primary's ``EmotionResult``, and a
primary success is never converted into a product failure.

The candidate runs only after the primary has already produced its result, so a candidate
fault cannot pre-empt the primary. Every candidate outcome is reduced to one stable
category and handed to an observer as an aggregate-shaped record: the observation carries
no utterance id, no path, no transcript, and no probability map, because those are exactly
the fields a shadow deployment must not start persisting.

An observer that raises is itself contained. Observation is diagnostics; it may not become
a new failure mode.

**Latency boundary.** The candidate is a second, synchronous inference. On a request path
that cost is real, so this adapter is deliberately *not* wired into
``build_dependencies``: it exists for the offline replay evaluator, where the extra cost
is paid by an operator rather than by a caller.

**Timeouts are terminal for the candidate, and the replay declines them entirely.**
``timeout_seconds`` bounds a hung candidate, but the abandoned worker cannot be killed —
Python has no way to interrupt it — so it may still be inside the shared model. The
adapter therefore latches: after one timeout the candidate is disabled for the rest of
this adapter's life, later calls skip it rather than queue behind it, and
``candidate_timed_out`` lets a caller stop the run.

That contract is kept and tested here, but the offline replay evaluator sets no timeout at
all. Its network-egress guard is process-wide and temporary, and an abandoned worker could
outlive it and then reach the network unguarded. Only a caller that can contain a stray
worker for its whole lifetime — a live path with a bounded per-candidate pool — should
pass ``timeout_seconds``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from voxdelta.domain.models import EmotionLabel, EmotionResult, ProviderProvenance
from voxdelta.evaluation.emotion_training import CANONICAL_LABELS
from voxdelta.providers.base import EmotionProvider, ProviderError

ShadowErrorCategory = Literal[
    "none",
    "timeout",
    "provider_error",
    "invalid_output",
    "unexpected_error",
]

ERROR_CATEGORIES: tuple[ShadowErrorCategory, ...] = (
    "none",
    "timeout",
    "provider_error",
    "invalid_output",
    "unexpected_error",
)


@dataclass(frozen=True, slots=True)
class ShadowObservation:
    """One aggregate-shaped comparison record.

    Deliberately carries no utterance id, audio path, transcript, or probability map:
    a shadow rollout must be able to log every field here without leaking content.
    """

    candidate_attempted: bool
    candidate_completed: bool
    error_category: ShadowErrorCategory
    candidate_valid: bool
    primary_latency_ms: float | None = None
    candidate_latency_ms: float | None = None
    candidate_peak_rss_mb: float | None = None
    top_labels_agree: bool | None = None
    candidate_confidence: float | None = None
    candidate_raw_confidence: float | None = None
    candidate_abstained: bool | None = None
    candidate_uncertain: bool | None = None

    def __post_init__(self) -> None:
        completed = self.error_category == "none"
        if self.candidate_completed != completed:
            raise ValueError("candidate_completed must agree with the error category")
        if self.candidate_completed and not self.candidate_valid:
            raise ValueError("a completed candidate must have produced a valid result")
        if self.candidate_completed and not self.candidate_attempted:
            raise ValueError("a completed candidate must have been attempted")


Observer = Callable[[ShadowObservation], None]


def _top_label(probabilities: dict[EmotionLabel, float]) -> EmotionLabel:
    return max(probabilities.items(), key=lambda pair: (pair[1], pair[0]))[0]


def valid_emotion_result(candidate: object) -> bool:
    """Whether a candidate returned a structurally usable seven-label result."""

    if not isinstance(candidate, EmotionResult):
        return False
    probabilities = candidate.probabilities
    if set(probabilities) != set(CANONICAL_LABELS):
        return False
    values = tuple(probabilities.values())
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        return False
    if abs(math.fsum(values) - 1.0) > 1e-6:
        return False
    scalars = (candidate.confidence, candidate.negative_intensity)
    if any(not math.isfinite(scalar) or not 0 <= scalar <= 1 for scalar in scalars):
        return False
    calibration = candidate.calibration
    if calibration is not None and not (
        math.isfinite(calibration.temperature)
        and calibration.temperature > 0
        and math.isfinite(calibration.raw_confidence)
    ):
        return False
    return True


class ShadowEmotionProvider:
    """An emotion provider that answers as the primary and merely watches the candidate."""

    def __init__(
        self,
        primary: EmotionProvider,
        candidate: EmotionProvider,
        *,
        observer: Observer,
        timeout_seconds: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive, finite number of seconds")
        self._primary = primary
        self._candidate = candidate
        self._observer = observer
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        self._timed_out = False
        self.provenance: ProviderProvenance = primary.provenance

    @property
    def candidate_timed_out(self) -> bool:
        """Whether a candidate call has ever timed out; the candidate is then disabled."""

        return self._timed_out

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        """Return the primary's result. The candidate can only be observed, never obeyed."""

        started = self._clock()
        primary = self._primary.analyze(utterance_id, audio_path, transcript)
        primary_latency_ms = (self._clock() - started) * 1000

        # From here on, nothing may escape: the primary result is already decided.
        try:
            observation = self._observe(utterance_id, audio_path, transcript, primary)
        except Exception:
            # Contained on purpose. KeyboardInterrupt and SystemExit are BaseException
            # and deliberately still propagate: process control is not ours to swallow.
            observation = ShadowObservation(
                candidate_attempted=True,
                candidate_completed=False,
                error_category="unexpected_error",
                candidate_valid=False,
                primary_latency_ms=primary_latency_ms,
            )
        else:
            observation = _with_primary_latency(observation, primary_latency_ms)

        try:
            self._observer(observation)
        except Exception:
            # Diagnostics may not become a new failure mode.
            pass
        return primary

    def _observe(
        self,
        utterance_id: str,
        audio_path: Path,
        transcript: str,
        primary: EmotionResult,
    ) -> ShadowObservation:
        if self._timed_out:
            # An abandoned worker may still be inside the shared model; do not start
            # overlapping work on it.
            return ShadowObservation(
                candidate_attempted=False,
                candidate_completed=False,
                error_category="timeout",
                candidate_valid=False,
            )
        began = self._clock()
        try:
            candidate = self._run_candidate(utterance_id, audio_path, transcript)
        except FutureTimeoutError:
            self._timed_out = True
            return ShadowObservation(
                candidate_attempted=True,
                candidate_completed=False,
                error_category="timeout",
                candidate_valid=False,
            )
        except ProviderError:
            return ShadowObservation(
                candidate_attempted=True,
                candidate_completed=False,
                error_category="provider_error",
                candidate_valid=False,
            )
        except Exception:
            return ShadowObservation(
                candidate_attempted=True,
                candidate_completed=False,
                error_category="unexpected_error",
                candidate_valid=False,
            )
        candidate_latency_ms = (self._clock() - began) * 1000

        if not valid_emotion_result(candidate):
            return ShadowObservation(
                candidate_attempted=True,
                candidate_completed=False,
                error_category="invalid_output",
                candidate_valid=False,
                candidate_latency_ms=candidate_latency_ms,
            )

        assert isinstance(candidate, EmotionResult)
        calibration = candidate.calibration
        usage = candidate.usage
        return ShadowObservation(
            candidate_attempted=True,
            candidate_completed=True,
            error_category="none",
            candidate_valid=True,
            candidate_latency_ms=candidate_latency_ms,
            candidate_peak_rss_mb=None if usage is None else usage.peak_rss_mb,
            top_labels_agree=_top_label(primary.probabilities)
            == _top_label(candidate.probabilities),
            candidate_confidence=candidate.confidence,
            candidate_raw_confidence=None if calibration is None else calibration.raw_confidence,
            candidate_abstained=None if calibration is None else calibration.abstained,
            candidate_uncertain=candidate.operational_state == "uncertain",
        )

    def _run_candidate(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        if self._timeout_seconds is None:
            return self._candidate.analyze(utterance_id, audio_path, transcript)
        # A timed-out worker is abandoned, not cancelled; see the module docstring.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voxdelta-shadow")
        try:
            future = executor.submit(self._candidate.analyze, utterance_id, audio_path, transcript)
            return future.result(timeout=self._timeout_seconds)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def unload(self) -> None:
        for provider in (self._primary, self._candidate):
            unload = getattr(provider, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception:
                    continue


def _with_primary_latency(
    observation: ShadowObservation, primary_latency_ms: float
) -> ShadowObservation:
    return ShadowObservation(
        candidate_attempted=observation.candidate_attempted,
        candidate_completed=observation.candidate_completed,
        error_category=observation.error_category,
        candidate_valid=observation.candidate_valid,
        primary_latency_ms=primary_latency_ms,
        candidate_latency_ms=observation.candidate_latency_ms,
        candidate_peak_rss_mb=observation.candidate_peak_rss_mb,
        top_labels_agree=observation.top_labels_agree,
        candidate_confidence=observation.candidate_confidence,
        candidate_raw_confidence=observation.candidate_raw_confidence,
        candidate_abstained=observation.candidate_abstained,
        candidate_uncertain=observation.candidate_uncertain,
    )


__all__ = [
    "ERROR_CATEGORIES",
    "Observer",
    "ShadowEmotionProvider",
    "ShadowErrorCategory",
    "ShadowObservation",
    "valid_emotion_result",
]
