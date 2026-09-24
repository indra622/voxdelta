"""A bounded local fallback from the canonical ASR provider to a second local one.

Promoting Qwen3-ASR to the default makes one question load-bearing: what should happen
when it cannot run on a given machine? Falling back to faster-whisper keeps the PoC
usable when a checkpoint is missing or a device rejects the model. Falling back on
*anything* would be worse than not falling back at all, because the second provider would
quietly paper over the first one's refusals.

So the boundary is drawn at the reason, not at the fact of failure:

* **Qwen could not run** — it failed to initialise, the runtime rejected it, or execution
  timed out. Nothing has been learned about the audio, so trying the other local model is
  a reasonable second attempt. These are :data:`FALLBACK_ELIGIBLE_CODES`.
* **Qwen ran and the result was not trustworthy** — malformed decoder output, a word
  timeline that failed the timestamp sanitation contract, an alignment that would have
  crossed a speaker boundary, an unusable asset, an unsupported speaker count. These are
  fail-closed by design. Retrying them on another model would convert a refusal into a
  silently different answer, and the refusal is the point.

The distinction is exhaustive by construction: eligibility is an allowlist, so a code this
module has never seen does **not** fall back. A new failure mode is treated as the serious
kind until someone decides otherwise.

Whichever provider ran is the one whose provenance and timestamp coverage this wrapper
reports, so a report can never attribute one model's output to the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from voxdelta.domain.models import AudioAsset, ProviderProvenance, SpeakerSegment, Utterance
from voxdelta.providers.base import ProviderError, ProviderErrorCode

#: Failures that say "this provider could not run here", and nothing about the audio.
FALLBACK_ELIGIBLE_CODES: frozenset[ProviderErrorCode] = frozenset(
    {"provider_unavailable", "provider_runtime_unsupported", "provider_timeout"}
)


class _Transcriber(Protocol):
    provenance: ProviderProvenance

    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]: ...


@dataclass(frozen=True, slots=True)
class FallbackEvent:
    """Why the primary provider was abandoned, and what ran instead."""

    primary: str
    fallback: str
    code: str

    def as_dict(self) -> dict[str, str]:
        return {"primary": self.primary, "fallback": self.fallback, "reason_code": self.code}


class FallbackTranscriptionProvider:
    """Run the primary transcription provider, falling back only when it could not run."""

    def __init__(self, primary: _Transcriber, fallback: _Transcriber) -> None:
        self._primary = primary
        self._fallback = fallback
        self._active: _Transcriber = primary
        #: The provider that produced the most recent result. Read after a transcription
        #: this names the model that actually ran; read before one it names the primary,
        #: which is what will be attempted. A plain attribute, not a property, because the
        #: TranscriptionProvider protocol requires a settable one.
        self.provenance: ProviderProvenance = primary.provenance
        #: Set for the most recent transcription that fell back; None when the primary ran.
        self.last_fallback: FallbackEvent | None = None

    @property
    def last_warning_count(self) -> int:
        return int(getattr(self._active, "last_warning_count", 0))

    @property
    def last_timestamp_coverage(self) -> object | None:
        """Coverage from the provider that ran, so a fallback cannot inherit Qwen's.

        faster-whisper reports none, so a fallen-back run correctly reports none rather
        than carrying a stale figure from the attempt that failed.
        """

        return getattr(self._active, "last_timestamp_coverage", None)

    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]:
        self.last_fallback = None
        self._active = self._primary
        self.provenance = self._primary.provenance
        try:
            return self._primary.transcribe(asset, segments)
        except ProviderError as error:
            if error.code not in FALLBACK_ELIGIBLE_CODES:
                # A refusal about the data or the contract. Re-raised unchanged: the whole
                # value of failing closed is that a second opinion cannot overrule it.
                raise
            primary_name = self._primary.provenance.name
            self._release(self._primary)
            self._active = self._fallback
            self.provenance = self._fallback.provenance
            self.last_fallback = FallbackEvent(
                primary=primary_name,
                fallback=self._fallback.provenance.name,
                code=error.code,
            )
        return self._fallback.transcribe(asset, segments)

    @staticmethod
    def _release(provider: _Transcriber) -> None:
        """Let a failed provider drop its weights; a stuck one must not hold the device."""

        unload = getattr(provider, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception:
                pass

    def unload(self) -> None:
        for provider in (self._primary, self._fallback):
            self._release(provider)


__all__ = [
    "FALLBACK_ELIGIBLE_CODES",
    "FallbackEvent",
    "FallbackTranscriptionProvider",
]
