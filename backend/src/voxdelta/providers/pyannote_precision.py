"""Explicit opt-in remote diarization through the managed pyannoteAI API.

This is the only provider in the pipeline that sends audio off this machine. It uploads the
normalized call to pyannoteAI temporary storage, submits a pure diarization job with
``transcription`` disabled, polls until the job settles, and maps the returned speaker turns
onto the same two-timeline contract the local Community-1 provider produces.

The API key is read from the injected credentials at the moment a request is built and is
never stored in an attribute, echoed into an exception, or written to a log. Provider
failures are reported as typed codes only, so no response body or header ever reaches a
caller.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from importlib import import_module
from math import isfinite
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from voxdelta.credentials import Credentials
from voxdelta.domain.models import AudioAsset, ProviderProvenance, SpeakerSegment
from voxdelta.providers.base import (
    DiarizationTimelines,
    ProviderBoundary,
    ProviderDiagnostic,
    ProviderError,
    sanitized_error_type,
)
from voxdelta.providers.pyannote_diarization import (
    _exclusive_segments,
    _segments,
    _speaker_map,
    _Turn,
)

PYANNOTEAI_API_BASE = "https://api.pyannote.ai"
PRECISION_MODEL_ID = "precision-2"
PYANNOTE_DATA_RETENTION_URL = "https://docs.pyannote.ai/data-retention"
# pyannoteAI states that Media API input may be held in temporary storage for up to this
# long, and the public API exposes no deletion endpoint. Declared so the consent prompt
# can name the window instead of linking to a policy nobody opens.
PYANNOTE_MEDIA_RETENTION_HOURS = 48

_SETTLED_STATUSES = frozenset({"succeeded", "failed", "canceled"})
_TIMEOUT_MARKERS = ("timeout", "timedout")


class HttpResponse(Protocol):
    status_code: int

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse: ...

    def close(self) -> None: ...


class ClientFactory(Protocol):
    def __call__(self, *, timeout_seconds: float) -> HttpClient: ...


def _default_client_factory(*, timeout_seconds: float) -> HttpClient:
    """Import httpx only when this remote provider is actually used.

    The pyannoteAI API is reachable directly from the host.  Inheriting ambient proxy
    variables made the first media-input hop intermittently fail with ``ProxyError``
    before any audio was uploaded.  Keep this provider's explicit remote boundary
    direct; presigned upload URLs use the same client later in the exchange.
    """

    httpx = import_module("httpx")
    return cast(HttpClient, httpx.Client(timeout=timeout_seconds, trust_env=False))


def _is_timeout(error: BaseException) -> bool:
    name = type(error).__name__.lower()
    return isinstance(error, TimeoutError) or any(mark in name for mark in _TIMEOUT_MARKERS)


def _validate_asset(asset: AudioAsset) -> tuple[float, tuple[str, ...]]:
    duration = asset.duration_seconds
    if duration is None or not isfinite(duration) or duration <= 0:
        raise ProviderError("invalid_audio_asset")
    paths = asset.normalized_paths
    if asset.channel_mode == "mixed":
        expected = 1
    elif asset.channel_mode == "separate":
        expected = 2
    else:
        expected = 0
    if len(paths) != expected or any(not path for path in paths):
        raise ProviderError("invalid_audio_asset")
    return duration, paths


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderError("invalid_provider_output")
    number = float(value)
    if not isfinite(number):
        raise ProviderError("invalid_provider_output")
    return number


def _turns_from_payload(items: object, duration: float) -> list[_Turn]:
    """Convert one API diarization array into the shared turn representation.

    Bounds are clamped to the decoded media the way the local provider clamps pyannote's
    own output, so a remote segment that runs past the end of the file cannot produce an
    utterance the pipeline could not slice later.
    """

    if not isinstance(items, list) or not items:
        raise ProviderError("invalid_provider_output")
    turns: set[_Turn] = set()
    for item in cast(Iterable[object], items):
        if not isinstance(item, dict):
            raise ProviderError("invalid_provider_output")
        label = item.get("speaker")
        if not isinstance(label, str) or not label:
            raise ProviderError("invalid_provider_output")
        start = _finite(item.get("start"))
        end = _finite(item.get("end"))
        if end <= start:
            raise ProviderError("invalid_provider_output")
        bounded_start = max(0.0, min(duration, start))
        bounded_end = max(0.0, min(duration, end))
        if bounded_end <= bounded_start:
            raise ProviderError("invalid_provider_output")
        turns.add(_Turn(bounded_start, bounded_end, label))
    if not turns:
        raise ProviderError("invalid_provider_output")
    return sorted(turns)


class PyannotePrecisionProvider:
    """Diarize remotely through pyannoteAI when this provider is selected explicitly."""

    def __init__(
        self,
        credentials: Credentials,
        *,
        model: str = PRECISION_MODEL_ID,
        api_base: str = PYANNOTEAI_API_BASE,
        retention_policy_url: str = PYANNOTE_DATA_RETENTION_URL,
        retention_window_hours: int = PYANNOTE_MEDIA_RETENTION_HOURS,
        request_timeout_seconds: float = 60.0,
        job_timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 2.0,
        client_factory: ClientFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provenance = ProviderProvenance(
            name="pyannoteai",
            model=model,
            remote=True,
            transmits=("audio",),
            retention_policy_url=retention_policy_url,
            retention_window_hours=retention_window_hours,
            revision=model,
        )
        self._credentials = credentials
        self._model = model
        self._api_base = api_base.rstrip("/")
        self._request_timeout_seconds = request_timeout_seconds
        self._job_timeout_seconds = job_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._client_factory = client_factory or _default_client_factory
        self._clock = clock
        self._sleep = sleep

    def _authorization(self) -> str:
        """Build the bearer header at call time; the raw key is never held on the instance."""

        secret = self._credentials.pyannoteai_api_key
        if secret is None:
            raise ProviderError("missing_pyannote_api_key")
        return f"Bearer {secret.get_secret_value()}"

    def _request(
        self,
        client: HttpClient,
        method: str,
        url: str,
        *,
        boundary: ProviderBoundary,
        json: Any = None,
        content: bytes | None = None,
        authorize: bool = True,
    ) -> HttpResponse:
        headers = {"Authorization": self._authorization()} if authorize else {}
        try:
            response = client.request(method, url, json=json, content=content, headers=headers)
            status = response.status_code
        except ProviderError:
            raise
        except Exception as error:
            # The response body and request headers are deliberately dropped: either could
            # carry the key or provider payload into a caller-visible message. Only ordinary
            # exceptions are remapped, so KeyboardInterrupt and SystemExit still propagate.
            # The classification keeps which hop failed and the exception class name, both
            # of which are fixed vocabulary rather than anything the provider sent back.
            timed_out = _is_timeout(error)
            raise ProviderError(
                "provider_timeout" if timed_out else "provider_unavailable",
                diagnostic=ProviderDiagnostic(
                    boundary=boundary,
                    failure="transport_timeout" if timed_out else "transport_error",
                    error_type=sanitized_error_type(error),
                ),
            ) from None
        if not isinstance(status, int):
            raise ProviderError(
                "invalid_provider_output",
                diagnostic=ProviderDiagnostic(boundary=boundary, failure="response_decode"),
            )
        if status in (408, 504):
            raise ProviderError(
                "provider_timeout",
                diagnostic=ProviderDiagnostic(
                    boundary=boundary, failure="http_status", status=status
                ),
            )
        if status < 200 or status >= 300:
            raise ProviderError(
                "provider_unavailable",
                diagnostic=ProviderDiagnostic(
                    boundary=boundary, failure="http_status", status=status
                ),
            )
        return response

    @staticmethod
    def _payload(response: HttpResponse, *, boundary: ProviderBoundary) -> Mapping[str, object]:
        # The status accompanies a decode failure so a 2xx with an unexpected body is
        # distinguishable from a boundary that answered with something else entirely.
        status = response.status_code
        failed = ProviderDiagnostic(
            boundary=boundary,
            failure="response_decode",
            status=status if isinstance(status, int) else None,
        )
        try:
            body = response.json()
        except ProviderError:
            raise
        except Exception:
            # Ordinary decode failures only; interpreter exits must not become a provider code.
            raise ProviderError("invalid_provider_output", diagnostic=failed) from None
        if not isinstance(body, dict):
            raise ProviderError("invalid_provider_output", diagnostic=failed)
        return cast(Mapping[str, object], body)

    def _upload(self, client: HttpClient, path: str) -> str:
        """Place one normalized file in pyannoteAI temporary storage and return its handle."""

        # Read and validate the local bytes first. A missing, unreadable or empty file must
        # not reserve a remote media slot that would then be abandoned.
        try:
            content = Path(path).read_bytes()
        except OSError:
            raise ProviderError("invalid_audio_asset") from None
        if not content:
            raise ProviderError("invalid_audio_asset")
        media_url = f"media://voxdelta/{uuid4().hex}.wav"
        created = self._payload(
            self._request(
                client,
                "POST",
                f"{self._api_base}/v1/media/input",
                boundary="media_input",
                json={"url": media_url},
            ),
            boundary="media_input",
        )
        destination = created.get("url")
        if not isinstance(destination, str) or not destination:
            raise ProviderError(
                "invalid_provider_output",
                diagnostic=ProviderDiagnostic(boundary="media_input", failure="response_decode"),
            )
        # The presigned destination already carries its own credentials.
        self._request(
            client, "PUT", destination, boundary="media_upload", content=content, authorize=False
        )
        return media_url

    def _submit(self, client: HttpClient, media_url: str, *, num_speakers: int | None) -> str:
        request: dict[str, object] = {
            "url": media_url,
            "model": self._model,
            # Pure diarization: no transcript is requested, so none is produced or returned.
            "transcription": False,
            "exclusive": True,
        }
        if num_speakers is not None:
            request["numSpeakers"] = num_speakers
        body = self._payload(
            self._request(
                client,
                "POST",
                f"{self._api_base}/v1/diarize",
                boundary="diarize_submit",
                json=request,
            ),
            boundary="diarize_submit",
        )
        job_id = body.get("jobId")
        if not isinstance(job_id, str) or not job_id:
            raise ProviderError(
                "invalid_provider_output",
                diagnostic=ProviderDiagnostic(boundary="diarize_submit", failure="response_decode"),
            )
        return job_id

    def _await_output(self, client: HttpClient, job_id: str) -> Mapping[str, object]:
        """Poll until the job settles, bounded by an overall deadline."""

        deadline = self._clock() + self._job_timeout_seconds
        undecodable = ProviderDiagnostic(boundary="job_poll", failure="response_decode")
        while True:
            body = self._payload(
                self._request(
                    client, "GET", f"{self._api_base}/v1/jobs/{job_id}", boundary="job_poll"
                ),
                boundary="job_poll",
            )
            status = body.get("status")
            if not isinstance(status, str):
                raise ProviderError("invalid_provider_output", diagnostic=undecodable)
            if status == "succeeded":
                output = body.get("output")
                if not isinstance(output, dict):
                    raise ProviderError("invalid_provider_output", diagnostic=undecodable)
                return cast(Mapping[str, object], output)
            if status in _SETTLED_STATUSES:
                # "failed" and "canceled" are provider-side outcomes, never a local fault.
                # The label is chosen from this module's own vocabulary after matching the
                # settled set, so no provider-supplied string is carried through.
                raise ProviderError(
                    "provider_unavailable",
                    diagnostic=ProviderDiagnostic(
                        boundary="job_poll",
                        failure="job_failed" if status == "failed" else "job_canceled",
                    ),
                )
            if self._clock() >= deadline:
                raise ProviderError(
                    "provider_timeout",
                    diagnostic=ProviderDiagnostic(boundary="job_poll", failure="job_poll_timeout"),
                )
            self._sleep(self._poll_interval_seconds)

    def _diarize_one(
        self, client: HttpClient, path: str, duration: float, *, num_speakers: int | None
    ) -> Mapping[str, object]:
        return self._await_output(
            client, self._submit(client, self._upload(client, path), num_speakers=num_speakers)
        )

    def diarize_timelines(self, asset: AudioAsset) -> DiarizationTimelines:
        duration, paths = _validate_asset(asset)
        # Fail before any upload when the key is absent, so nothing is transmitted.
        self._authorization()
        client = self._client_factory(timeout_seconds=self._request_timeout_seconds)
        try:
            if asset.channel_mode == "separate":
                channel_results: list[SpeakerSegment] = []
                for index, path in enumerate(paths):
                    output = self._diarize_one(client, path, duration, num_speakers=1)
                    turns = _turns_from_payload(output.get("diarization"), duration)
                    if len({turn.label for turn in turns}) != 1:
                        raise ProviderError("invalid_provider_output")
                    channel_results.extend(
                        SpeakerSegment(
                            start=turn.start,
                            end=turn.end,
                            speaker_id=f"SPEAKER_{index:02d}",
                            overlap=False,
                            confidence=1.0,
                        )
                        for turn in turns
                    )
                ordered = sorted(
                    channel_results, key=lambda item: (item.start, item.end, item.speaker_id)
                )
                return DiarizationTimelines(evidence=ordered, exclusive=list(ordered))

            output = self._diarize_one(client, paths[0], duration, num_speakers=None)
            evidence_turns = _turns_from_payload(output.get("diarization"), duration)
            speakers = _speaker_map(evidence_turns)
            exclusive_turns = _turns_from_payload(output.get("exclusiveDiarization"), duration)
            return DiarizationTimelines(
                evidence=_segments(evidence_turns, speakers, mark_overlaps=True),
                exclusive=_exclusive_segments(exclusive_turns, speakers),
            )
        finally:
            try:
                client.close()
            except Exception:
                pass

    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]:
        return self.diarize_timelines(asset).evidence

    def diarize_for_alignment(self, asset: AudioAsset) -> list[SpeakerSegment]:
        """Return the non-overlapping timeline used to reconcile the transcript."""

        return self.diarize_timelines(asset).exclusive


__all__ = [
    "PRECISION_MODEL_ID",
    "PYANNOTEAI_API_BASE",
    "PYANNOTE_DATA_RETENTION_URL",
    "PYANNOTE_MEDIA_RETENTION_HOURS",
    "ClientFactory",
    "HttpClient",
    "HttpResponse",
    "PyannotePrecisionProvider",
]
