"""Provider boundaries for model-backed pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from voxdelta.domain.models import (
    AnalysisReport,
    AudioAsset,
    EmotionResult,
    ProviderProvenance,
    ResponseStrategyResult,
    SpeakerSegment,
    Utterance,
)

ProviderErrorCode = Literal[
    "audio_too_short",
    "invalid_audio_asset",
    "invalid_local_checkpoint",
    "invalid_provider_output",
    "missing_huggingface_token",
    "missing_pyannote_api_key",
    "provider_timeout",
    "provider_unavailable",
    "provider_runtime_unsupported",
    "unsupported_speaker_count",
]

_PROVIDER_ERROR_MESSAGES: dict[ProviderErrorCode, str] = {
    "audio_too_short": "The audio segment is too short to analyze.",
    "invalid_audio_asset": "The normalized audio asset is invalid.",
    "invalid_local_checkpoint": "The local model checkpoint is invalid.",
    "invalid_provider_output": "The provider returned invalid output.",
    "missing_huggingface_token": "A Hugging Face token is required for this model.",
    "missing_pyannote_api_key": "A pyannoteAI API key is required for this provider.",
    "provider_timeout": "The provider timed out.",
    "provider_unavailable": "The provider is unavailable.",
    "provider_runtime_unsupported": "The selected provider is unsupported on this runtime.",
    "unsupported_speaker_count": "Exactly two observed speakers are required.",
}


# Where in a remote provider exchange a failure happened. A typed code such as
# provider_unavailable is the same whether the key was refused, the upload host was
# unreachable, or the remote job itself failed, which is exactly the distinction an
# operator needs. These labels are fixed vocabulary, never provider-supplied text.
ProviderBoundary = Literal[
    "media_input",
    "media_upload",
    "diarize_submit",
    "job_poll",
]

ProviderFailureKind = Literal[
    "http_status",
    "transport_error",
    "transport_timeout",
    "response_decode",
    "job_failed",
    "job_canceled",
    "job_poll_timeout",
]

# An exception class name is the one detail that makes a transport failure actionable
# (name resolution vs. refused connection vs. certificate). Rather than sanitize an
# arbitrary string, only these known transport and decoding names are reportable, so a
# name can never become a channel for a message, body or credential. Anything else is
# reported as "OtherError"; an operator investigating one still has the live exception.
_REPORTABLE_ERROR_TYPES = frozenset(
    {
        # Standard library transport and TLS failures.
        "BrokenPipeError",
        "ConnectionAbortedError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "OSError",
        "SSLCertVerificationError",
        "SSLEOFError",
        "SSLError",
        "SSLZeroReturnError",
        "TimeoutError",
        "ValueError",
        "gaierror",
        "herror",
        "timeout",
        # httpx / httpcore transport failures.
        "CloseError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionNotAvailable",
        "DecodingError",
        "HTTPError",
        "InvalidURL",
        "LocalProtocolError",
        "NetworkError",
        "PoolTimeout",
        "ProtocolError",
        "ProxyError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "StreamError",
        "TooManyRedirects",
        "TransportError",
        "UnsupportedProtocol",
        "WriteError",
        "WriteTimeout",
    }
)
_OTHER_ERROR_TYPE = "OtherError"


def sanitized_error_type(error: BaseException) -> str:
    """Return the exception class name only when it is a known transport failure name."""

    name = type(error).__name__
    return name if name in _REPORTABLE_ERROR_TYPES else _OTHER_ERROR_TYPE


@dataclass(frozen=True, slots=True)
class ProviderDiagnostic:
    """Which boundary failed and how, in enumerated values an operator log may hold.

    Deliberately has no field for a response body, header, URL or credential: it is a
    classification, not a capture.
    """

    boundary: ProviderBoundary
    failure: ProviderFailureKind
    status: int | None = None
    error_type: str | None = None

    def as_metadata(self) -> dict[str, object]:
        """Render the classification for logging, omitting what does not apply."""

        metadata: dict[str, object] = {"boundary": self.boundary, "failure": self.failure}
        if self.status is not None:
            metadata["status"] = self.status
        if self.error_type is not None:
            metadata["error_type"] = self.error_type
        return metadata


class ProviderError(RuntimeError):
    """Typed provider failure whose public text never includes provider payloads."""

    # Declared on the class, and written by this constructor only when one is supplied, so
    # a subclass that attaches its own text-free diagnostic before calling super() keeps it.
    diagnostic: Mapping[str, object] | None = None

    def __init__(
        self, code: ProviderErrorCode, *, diagnostic: ProviderDiagnostic | None = None
    ) -> None:
        self.code = code
        if diagnostic is not None:
            # Held as an attribute rather than an argument: str(error) and args stay exactly
            # the public message, so nothing new appears in a traceback or a response.
            self.diagnostic = diagnostic.as_metadata()
        super().__init__(_PROVIDER_ERROR_MESSAGES[code])


@dataclass(frozen=True, slots=True)
class DiarizationTimelines:
    """Overlap-aware evidence and a non-overlapping alignment timeline."""

    evidence: list[SpeakerSegment]
    exclusive: list[SpeakerSegment]


@runtime_checkable
class DiarizationProvider(Protocol):
    provenance: ProviderProvenance

    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]: ...


@runtime_checkable
class DiarizationTimelineProvider(Protocol):
    provenance: ProviderProvenance

    def diarize_timelines(self, asset: AudioAsset) -> DiarizationTimelines: ...


@runtime_checkable
class TranscriptionProvider(Protocol):
    provenance: ProviderProvenance

    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]: ...


@runtime_checkable
class EmotionProvider(Protocol):
    provenance: ProviderProvenance

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult: ...


@runtime_checkable
class ResponseStrategyProvider(Protocol):
    provenance: ProviderProvenance

    def classify(
        self, utterance: Utterance, context: list[Utterance]
    ) -> ResponseStrategyResult: ...


@runtime_checkable
class ReportSummaryProvider(Protocol):
    provenance: ProviderProvenance

    def summarize(self, report: AnalysisReport) -> str: ...
