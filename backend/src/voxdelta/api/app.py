"""Dependency-injected FastAPI application for the local VoxDelta pipeline."""

from __future__ import annotations

import asyncio
import hmac
import io
import json
import os
import wave
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractContextManager, asynccontextmanager
from pathlib import Path
from threading import Lock
from time import time
from typing import Annotated, Any, BinaryIO, Literal, cast
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Security,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import SecretStr
from starlette.background import BackgroundTask
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from voxdelta.annotation import alignment as annotation_alignment
from voxdelta.annotation import audio as annotation_audio
from voxdelta.annotation import emotion_candidates as annotation_emotion_candidates
from voxdelta.annotation import (
    gemini_emotion_overlay as annotation_gemini_emotion_overlay,
)
from voxdelta.annotation import reference_candidate as annotation_reference_candidate
from voxdelta.annotation import review as annotation_review
from voxdelta.annotation.gemini_silver import EMOTION_LABELS
from voxdelta.api.dependencies import (
    build_dependencies,
    default_annotation_audio_root,
    default_annotation_root,
)
from voxdelta.api.schemas import (
    AlignmentProposal,
    AlignmentProposalCollection,
    AlignmentProposalRow,
    AnnotationAudioOverview,
    AnnotationReviewIndex,
    AnnotationReviewState,
    EmotionOverlayCandidate,
    ExpertGuidanceRequest,
    ExpertGuidanceStatus,
    ExpertGuidanceSubmission,
    GeminiEmotionOverlayCandidate,
    GoldPromotionRequest,
    GoldPromotionResult,
    JobCreated,
    ProviderConfiguration,
    PublicError,
    PublicErrorEnvelope,
    PublicJob,
    PublicStage,
    ReferenceResegmentationCandidate,
    RetryRequest,
    ReviewGap,
    ReviewTurn,
    ReviewWarning,
    RoleCandidate,
    RoleConfirmation,
    RoleSample,
    SilverReviewDraft,
    TurnClip,
)
from voxdelta.audio.service import AudioRejected, PreparedAudio
from voxdelta.config import Settings
from voxdelta.domain.models import AnalysisReport, StageName
from voxdelta.expert_mode import ExpertHandoffRecord, ExpertHandoffStore, build_expert_request
from voxdelta.jobs._ids import JobAbsenceProof
from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.repository import JobCapacityExceeded, JobRepository
from voxdelta.pipeline.runner import (
    PipelineRunner,
    PipelineStateError,
    PipelineValidationError,
)

UPLOAD_CHUNK_BYTES = 1024 * 1024
AUDIO_CHUNK_BYTES = 64 * 1024
# Multipart boundaries and headers are bounded separately from the exact stored-file limit.
MULTIPART_OVERHEAD_BYTES = 64 * 1024
_SUPPORTED_SUFFIXES = frozenset({".wav", ".mp3", ".m4a", ".mp4"})
_ROLE_SAMPLE_SECONDS = 8.0
_ROLE_SAMPLES_PER_SPEAKER = 2
_PUBLIC_STAGE_ERRORS = {
    "pipeline_failed": "The pipeline stage failed.",
    "audio_rejected": "The uploaded audio was rejected.",
    "insufficient_emotion_coverage": (
        "There is not enough customer emotion coverage to generate a report."
    ),
    "invalid_stage_output": "The pipeline stage produced invalid data.",
    "unsupported_speaker_count": "Exactly two observed speakers are required.",
}
_CAPABILITY_HEADER = b"x-voxdelta-token"
_CAPABILITY_SECURITY = APIKeyHeader(
    name="X-VoxDelta-Token",
    scheme_name="VoxDeltaCapability",
    auto_error=False,
)
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# Job data and annotation drafts both hold verbatim speech, so both sit behind the
# per-launch capability. The provider disclosure does not: it is the one thing a client
# has to be able to read before it decides whether to hand anything over.
_CAPABILITY_PROTECTED_PREFIXES = ("/api/jobs", "/api/annotations")
# Every refusal the annotation review path can produce, mapped to the status a client
# should act on. An unmapped code would be a new refusal nobody chose a status for.
_REVIEW_ERROR_STATUS = {
    "invalid_conversation_id": 422,
    "annotation_not_found": 404,
    "annotation_unreadable": 409,
    "reviewer_required": 422,
    "review_acknowledgement_required": 422,
    "review_note_too_long": 422,
    "corrected_turns_required": 422,
    "too_many_turns": 422,
    "invalid_turn_speaker": 422,
    "invalid_turn_transcript": 422,
    "invalid_turn_rationale": 422,
    "invalid_turn_emotion": 422,
    "invalid_turn_interval": 422,
    "invalid_turn_confidence": 422,
    # Listening is a read of audio the draft already points at, so a missing or swapped
    # recording is the draft's state rather than the caller's mistake: 409, not 404.
    "audio_source_unavailable": 409,
    "audio_source_mismatch": 409,
    "audio_source_unreadable": 409,
    "invalid_clip_range": 422,
    "clip_too_long": 422,
    "gold_already_exists": 409,
    "promotion_refused": 409,
    "silver_modified": 500,
}


def _error_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    return {
        status: {
            "model": PublicErrorEnvelope,
            "description": "Sanitized public error",
        }
        for status in statuses
    }


_WAV_CONTENT = {
    "audio/wav": {
        "schema": {"type": "string", "format": "binary"},
    }
}


class _UploadAdmissionError(ValueError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


class _RequestBodyLimitExceeded(Exception):
    pass


def _security_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": {"code": code, "message": message}},
    )


def _host_name(raw: str) -> str | None:
    try:
        parsed = urlsplit(f"//{raw}")
        if parsed.username is not None or parsed.password is not None:
            return None
        return parsed.hostname.casefold() if parsed.hostname is not None else None
    except ValueError:
        return None


def _origin_is_local(raw: str) -> bool:
    try:
        parsed = urlsplit(raw)
        return (
            parsed.scheme in {"http", "https"}
            and parsed.username is None
            and parsed.password is None
            and parsed.hostname is not None
            and parsed.hostname.casefold() in _LOCAL_HOSTS
            and not parsed.path.strip("/")
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


class _LocalCapabilityMiddleware:
    """Fence local job data from DNS rebinding, drive-by browsers, and other users."""

    def __init__(self, app: ASGIApp, *, token: SecretStr) -> None:
        self._app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers = scope.get("headers", [])
        hosts = [value.decode("latin-1") for name, value in headers if name.lower() == b"host"]
        if len(hosts) != 1 or _host_name(hosts[0]) not in _LOCAL_HOSTS:
            await _security_error(400, "invalid_host", "The request host is not allowed.")(
                scope, receive, send
            )
            return
        origins = [value.decode("latin-1") for name, value in headers if name.lower() == b"origin"]
        if len(origins) > 1 or (origins and not _origin_is_local(origins[0])):
            await _security_error(403, "origin_not_allowed", "The request origin is not allowed.")(
                scope, receive, send
            )
            return
        path = str(scope.get("path", ""))
        if path.startswith(_CAPABILITY_PROTECTED_PREFIXES):
            supplied = [value for name, value in headers if name.lower() == _CAPABILITY_HEADER]
            expected = self._token.get_secret_value().encode("utf-8")
            if len(supplied) != 1 or not hmac.compare_digest(supplied[0], expected):
                await _security_error(
                    401,
                    "capability_required",
                    "A valid local API capability is required.",
                )(scope, receive, send)
                return
        await self._app(scope, receive, send)


def _upload_too_large_response() -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={
            "detail": {
                "code": "upload_too_large",
                "message": "The uploaded file exceeds the configured size limit.",
            }
        },
    )


def _declared_length_exceeds(raw: bytes, maximum: int) -> bool | None:
    candidate = raw.strip()
    if not candidate or any(byte < ord("0") or byte > ord("9") for byte in candidate):
        return None
    canonical = candidate.lstrip(b"0") or b"0"
    limit = str(maximum).encode("ascii")
    if len(canonical) != len(limit):
        return len(canonical) > len(limit)
    if canonical != limit:
        return canonical > limit
    try:
        return int(canonical) > maximum
    except ValueError:
        return None


class _RequestBodyLimitMiddleware:
    """Stop oversized upload bodies while the multipart parser is still reading ASGI messages."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/api/jobs"
        ):
            await self._app(scope, receive, send)
            return

        content_lengths = [
            value.strip()
            for name, value in scope.get("headers", [])
            if name.lower() == b"content-length"
        ]
        if len(content_lengths) == 1 and _declared_length_exceeds(
            content_lengths[0], self._max_bytes
        ):
            await _upload_too_large_response()(scope, receive, send)
            return

        total = 0
        limit_exceeded = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal limit_exceeded, total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self._max_bytes:
                    limit_exceeded = True
                    raise _RequestBodyLimitExceeded
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if limit_exceeded and not response_started:
                raise _RequestBodyLimitExceeded
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self._app(scope, limited_receive, tracking_send)
        except _RequestBodyLimitExceeded:
            if response_started:
                raise
            await _upload_too_large_response()(scope, receive, send)


class _ContextCloser:
    """Idempotently release a streamed descriptor from body or response cleanup."""

    def __init__(self, context: AbstractContextManager[BinaryIO]) -> None:
        self._context = context
        self._lock = Lock()
        self._closed = False

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._context.__exit__(None, None, None)
        except Exception:
            pass


class _ManagedStreamingResponse(StreamingResponse):
    """Own a streamed descriptor for the full ASGI response lifecycle."""

    def __init__(self, *args: object, closer: _ContextCloser, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._closer = closer

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._closer.close()


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"code": "job_not_found", "message": "The requested job was not found."},
    )


def _pipeline_error(status_code: int, error: PipelineValidationError) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": error.code, "message": error.message},
    )


def _safe_background_run(runner: PipelineRunner, job_id: str) -> None:
    try:
        runner.run_until_pause(job_id)
    except Exception:
        try:
            runner.mark_unhandled_failure(job_id)
        except Exception:
            pass
        return


def _public_error(raw: object) -> PublicError | None:
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return PublicError(code="pipeline_failed", message="The pipeline stage failed.")
    code = payload.get("code") if isinstance(payload, dict) else None
    if isinstance(code, str) and code in _PUBLIC_STAGE_ERRORS:
        return PublicError(code=code, message=_PUBLIC_STAGE_ERRORS[code])
    return PublicError(code="pipeline_failed", message="The pipeline stage failed.")


def _public_job(repository: JobRepository, runner: PipelineRunner, job_id: str) -> PublicJob:
    try:
        raw = repository.get_job(job_id)
    except (KeyError, ValueError):
        raise _not_found() from None
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, dict):
        raise HTTPException(status_code=500, detail="Invalid persisted job state")
    stages: dict[str, PublicStage] = {}
    for stage in StageName:
        row = raw_stages.get(stage.value)
        if not isinstance(row, dict) or not isinstance(row.get("status"), str):
            raise HTTPException(status_code=500, detail="Invalid persisted job state")
        stages[stage.value] = PublicStage(
            status=cast(str, row["status"]),
            error=_public_error(row.get("error_json")),
        )

    candidate = None
    role_row = raw_stages.get(StageName.CONFIRM_ROLES.value)
    if isinstance(role_row, dict) and role_row.get("status") == "paused":
        try:
            artifact = runner.role_candidate(job_id)
        except PipelineValidationError:
            artifact = None
        if artifact is not None:
            candidate = RoleCandidate(
                speakers=sorted({item.speaker_id for item in artifact.utterances}),
                suggested_mapping=artifact.suggestion,
                samples=_role_samples(artifact),
            )

    required_strings = ("status", "created_at", "updated_at")
    if any(not isinstance(raw.get(key), str) for key in required_strings):
        raise HTTPException(status_code=500, detail="Invalid persisted job state")
    return PublicJob(
        job_id=job_id,
        status=cast(str, raw["status"]),
        diagnostic_capture=raw.get("diagnostic_capture") == 1,
        created_at=cast(str, raw["created_at"]),
        updated_at=cast(str, raw["updated_at"]),
        stages=stages,
        role_candidate=candidate,
    )


def _role_samples(artifact: Any) -> dict[str, list[RoleSample]]:
    """Choose a small, deterministic listening set without inferring a role.

    Role confirmation happens *after* ASR in this pipeline. These excerpts let a reviewer
    listen to the diarized voice while reading already-attributed transcript, but never
    label SPEAKER_00/01 as customer or agent automatically.
    """

    by_speaker: dict[str, list[Any]] = {}
    for utterance in sorted(artifact.utterances, key=lambda item: (item.start, item.end)):
        if isinstance(utterance.transcript, str) and utterance.transcript.strip():
            by_speaker.setdefault(utterance.speaker_id, []).append(utterance)

    samples: dict[str, list[RoleSample]] = {}
    for speaker, utterances in by_speaker.items():
        rows: list[RoleSample] = []
        for index, utterance in enumerate(utterances[:_ROLE_SAMPLES_PER_SPEAKER]):
            clip_end = min(float(utterance.end), float(utterance.start) + _ROLE_SAMPLE_SECONDS)
            rows.append(
                RoleSample(
                    speaker_id=speaker,
                    index=index,
                    start=round(float(utterance.start), 3),
                    end=round(clip_end, 3),
                    transcript=utterance.transcript.strip(),
                    clip_truncated=clip_end < float(utterance.end),
                )
            )
        samples[speaker] = rows
    return samples


def _review_error(error: annotation_review.ReviewRejected) -> HTTPException:
    """Answer with the refusal's own code, never with the store's path-bearing text."""

    status_code = _REVIEW_ERROR_STATUS.get(error.code, 422)
    return HTTPException(
        status_code=status_code,
        detail={"code": error.code, "message": error.message},
    )


def _review_state(state: annotation_review.ReviewState) -> AnnotationReviewState:
    return AnnotationReviewState(
        conversation_id=state.conversation_id,
        review_state=state.review_state,
        promotable=state.promotable,
        created_at=state.created_at,
        model=state.model,
        input_sha256=state.input_sha256,
        content_sha256=state.content_sha256,
        remote_file_deleted=state.remote_file_deleted,
        turn_count=state.turn_count,
        speaker_count=state.speaker_count,
        uncertain_turns=state.uncertain_turns,
        mean_confidence=state.mean_confidence,
        gold_present=state.gold_present,
        emotion_candidate_count=state.emotion_candidate_count,
    )


def _review_warning(warning: annotation_review.ReviewWarning) -> ReviewWarning:
    return ReviewWarning(
        code=warning.code,
        detail=warning.detail,
        count=warning.count,
        by_rule=dict(warning.by_rule) if warning.by_rule is not None else None,
    )


def _cleanup_incoming_upload(
    artifacts: ArtifactStore,
    descriptor: int | None,
    incoming: Path | None,
) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass
    if incoming is not None:
        try:
            artifacts.discard_incoming_upload(incoming)
        except Exception:
            pass


async def _await_preflight_task(
    task: asyncio.Task[PreparedAudio],
) -> PreparedAudio:
    """Drain a bounded worker through repeated cancellation and never lose its result."""

    cancelled = False
    result: PreparedAudio | None = None
    worker_error: BaseException | None = None
    while result is None and worker_error is None:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            current = asyncio.current_task()
            if current is not None:
                while current.cancelling():
                    current.uncancel()
        except BaseException as error:
            worker_error = error
    if cancelled:
        if result is not None:
            result.discard()
        raise asyncio.CancelledError
    if worker_error is not None:
        raise worker_error
    if result is None:  # pragma: no cover - loop invariants guarantee a result or error
        raise RuntimeError("preflight worker ended without a result")
    return result


def _parse_range(raw: str | None, size: int) -> tuple[int, int] | None:
    if raw is None:
        return None
    if "=" not in raw:
        if raw.strip().casefold().startswith("bytes"):
            raise ValueError("invalid range")
        return None
    unit, value = raw.split("=", 1)
    if unit.strip().casefold() != "bytes":
        return None
    if size == 0:
        raise ValueError("invalid range")
    value = value.strip()
    if not value or "," in value or value.count("-") != 1:
        raise ValueError("invalid range")
    start_text, end_text = value.split("-", 1)
    if start_text:
        if not start_text.isascii() or not start_text.isdecimal():
            raise ValueError("invalid range")
        start = int(start_text)
        if start >= size:
            raise ValueError("invalid range")
        if end_text:
            if not end_text.isascii() or not end_text.isdecimal():
                raise ValueError("invalid range")
            requested_end = int(end_text)
            if requested_end < start:
                raise ValueError("invalid range")
            end = min(requested_end, size - 1)
        else:
            end = size - 1
        return start, end
    if not end_text.isascii() or not end_text.isdecimal():
        raise ValueError("invalid range")
    suffix = int(end_text)
    if suffix <= 0:
        raise ValueError("invalid range")
    return max(size - suffix, 0), size - 1


async def _audio_chunks(
    opened: BinaryIO,
    closer: _ContextCloser,
    start: int,
    length: int,
) -> AsyncIterator[bytes]:
    try:
        opened.seek(start)
        remaining = length
        while remaining:
            chunk = opened.read(min(AUDIO_CHUNK_BYTES, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
    finally:
        closer.close()


def _role_sample_wav(opened: BinaryIO, *, start: float, end: float) -> bytes:
    """Copy a bounded PCM excerpt from a verified job preview into memory.

    ``open_mixed_preview`` deliberately yields a descriptor, not a path. Keeping the
    slice descriptor-based preserves that containment and avoids a new path-resolution
    surface just for role confirmation.
    """

    opened.seek(0)
    with wave.open(opened, "rb") as source:
        if (
            source.getcomptype() != "NONE"
            or source.getframerate() < 1
            or source.getnchannels() < 1
            or source.getsampwidth() < 1
        ):
            raise ValueError("normalized preview is not PCM WAV")
        frame_rate = source.getframerate()
        start_frame = int(start * frame_rate)
        end_frame = min(int(round(end * frame_rate)), source.getnframes())
        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError("role sample bounds are invalid")
        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)
        expected_bytes = (end_frame - start_frame) * source.getnchannels() * source.getsampwidth()
        if len(frames) != expected_bytes:
            raise ValueError("normalized preview ended before role sample")
        output = io.BytesIO()
        with wave.open(output, "wb") as clipped:
            clipped.setparams(
                (
                    source.getnchannels(),
                    source.getsampwidth(),
                    frame_rate,
                    end_frame - start_frame,
                    "NONE",
                    "not compressed",
                )
            )
            clipped.writeframes(frames)
        return output.getvalue()


def reconcile_local_state(
    repository: JobRepository,
    artifacts: ArtifactStore,
    runner: PipelineRunner,
    *,
    lease_seconds: float,
    now: float | None = None,
    schedule: Callable[[str], None] | None = None,
) -> tuple[str, ...]:
    """Reconcile stale admission state and recover pristine jobs in this local process."""

    current_time = time() if now is None else now
    artifacts.remove_stale_ingest_workspaces(
        lease_seconds=lease_seconds,
        now=current_time,
    )
    artifacts.remove_stale_incoming_uploads(
        lease_seconds=lease_seconds,
        now=current_time,
    )
    for job_id in artifacts.stale_unregistered_job_candidates(
        lease_seconds=lease_seconds,
        now=current_time,
    ):

        def discard_absent(proof: JobAbsenceProof, selected: str = job_id) -> None:
            artifacts._discard_unregistered_job_under_absence_proof(  # noqa: SLF001
                selected, proof
            )

        repository.discard_if_absent(job_id, discard_absent)

    recovered: list[str] = []
    candidates = (
        *repository.list_pristine_pending_jobs(),
        *repository.list_expired_running_jobs(),
    )
    for job_id, source_name in candidates:
        if not artifacts.canonical_source_exists(job_id, source_name):
            continue
        recovered.append(job_id)
        if schedule is None:
            _safe_background_run(runner, job_id)
        else:
            schedule(job_id)
    return tuple(recovered)


def create_app(
    *,
    repository: JobRepository | None = None,
    artifacts: ArtifactStore | None = None,
    runner: PipelineRunner | None = None,
    max_upload_bytes: int | None = None,
    reconciliation_lease_seconds: float | None = None,
    api_capability_token: SecretStr | None = None,
    max_active_jobs: int | None = None,
    annotation_root: Path | None = None,
    annotation_audio_root: Path | None = None,
) -> FastAPI:
    """Create an isolated application, or construct safe local production dependencies."""

    if repository is None or artifacts is None or runner is None:
        if any(value is not None for value in (repository, artifacts, runner)):
            raise ValueError("repository, artifacts, and runner must be injected together")
        dependencies = build_dependencies()
        repository = dependencies.repository
        artifacts = dependencies.artifacts
        runner = dependencies.runner
        if max_upload_bytes is None:
            max_upload_bytes = dependencies.max_upload_bytes
        if reconciliation_lease_seconds is None:
            reconciliation_lease_seconds = dependencies.admission_reconciliation_lease_seconds
        if api_capability_token is None:
            api_capability_token = dependencies.api_capability_token
        if max_active_jobs is None:
            max_active_jobs = dependencies.max_active_jobs
        if annotation_root is None:
            annotation_root = dependencies.annotation_root
        if annotation_audio_root is None:
            annotation_audio_root = dependencies.annotation_audio_root
    if max_upload_bytes is None:
        max_upload_bytes = Settings().max_upload_bytes
    if reconciliation_lease_seconds is None:
        reconciliation_lease_seconds = Settings().admission_reconciliation_lease_seconds
    if max_active_jobs is None:
        max_active_jobs = Settings().max_active_jobs
    if annotation_root is None:
        annotation_root = default_annotation_root(Settings())
    if annotation_audio_root is None:
        annotation_audio_root = default_annotation_audio_root(Settings())
    selected_annotation_root = annotation_root
    selected_annotation_audio_root = annotation_audio_root
    if api_capability_token is None:
        raise ValueError(
            "VOXDELTA_API_CAPABILITY_TOKEN must configure a per-launch capability token"
        )
    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be positive")
    if reconciliation_lease_seconds <= 0:
        raise ValueError("reconciliation_lease_seconds must be positive")
    if max_active_jobs <= 0:
        raise ValueError("max_active_jobs must be positive")
    raw_capability_token = api_capability_token.get_secret_value()
    if len(raw_capability_token) < 32 or any(
        not 0x21 <= ord(character) <= 0x7E for character in raw_capability_token
    ):
        raise ValueError(
            "api capability token must contain at least 32 ASCII characters, all in the visible "
            "HTTP-header ASCII range 0x21-0x7E"
        )

    recovery_tasks: set[asyncio.Task[None]] = set()

    def schedule_recovery(job_id: str) -> None:
        task = asyncio.create_task(asyncio.to_thread(_safe_background_run, runner, job_id))
        recovery_tasks.add(task)
        task.add_done_callback(recovery_tasks.discard)

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        reconcile_local_state(
            repository,
            artifacts,
            runner,
            lease_seconds=reconciliation_lease_seconds,
            schedule=schedule_recovery,
        )
        yield
        if recovery_tasks:
            await asyncio.gather(*tuple(recovery_tasks), return_exceptions=True)

    app = FastAPI(title="VoxDelta", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        _RequestBodyLimitMiddleware,
        max_bytes=max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )
    app.add_middleware(_LocalCapabilityMiddleware, token=api_capability_token)
    expert_handoffs = ExpertHandoffStore(artifacts.root)

    def expert_status(record: ExpertHandoffRecord | None) -> ExpertGuidanceStatus:
        if record is None:
            return ExpertGuidanceStatus(status="not_requested", evidence_turn_count=0)
        return ExpertGuidanceStatus(
            status=record.status,
            target=record.request.target,
            transport=record.request.transport,
            request_sha256=record.request_sha256,
            transcription_uncertain=record.request.transcription_uncertain,
            evidence_turn_count=len(record.request.evidence),
            guidance=record.guidance,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        del request, error
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "invalid_request",
                    "message": "The request is invalid.",
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def public_http_error(request: Request, error: HTTPException) -> JSONResponse:
        del request
        detail = error.detail
        if (
            isinstance(detail, dict)
            and isinstance(detail.get("code"), str)
            and isinstance(detail.get("message"), str)
        ):
            public_detail = {
                "code": detail["code"],
                "message": detail["message"],
            }
        else:
            public_detail = {
                "code": "request_failed" if error.status_code < 500 else "internal_error",
                "message": (
                    "The request could not be completed."
                    if error.status_code < 500
                    else "The server could not complete the request."
                ),
            }
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": public_detail},
            headers=error.headers,
        )

    @app.get(
        "/api/config/providers",
        response_model=ProviderConfiguration,
        responses=_error_responses(400, 403),
    )
    def provider_configuration() -> ProviderConfiguration:
        return ProviderConfiguration(stages=runner.provider_disclosures())

    @app.post(
        "/api/jobs",
        status_code=202,
        response_model=JobCreated,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses={
            **_error_responses(400, 401, 403, 413, 422, 429, 500),
        },
    )
    async def create_job(
        background_tasks: BackgroundTasks,
        uploaded_file: Annotated[UploadFile, File(alias="file")],
        diagnostic_capture: Annotated[bool, Form()] = False,
    ) -> JobCreated:
        suffix = Path(uploaded_file.filename or "").suffix.lower()
        if suffix not in _SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unsupported_audio_type",
                    "message": "Upload a WAV, MP3, M4A, or MP4 audio file.",
                },
            )
        job_id = uuid4().hex
        incoming: Path | None = None
        descriptor: int | None = None
        failure: BaseException | None = None
        try:
            descriptor, incoming = artifacts.open_incoming_upload(job_id, suffix)
            total = 0
            while chunk := await uploaded_file.read(UPLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > max_upload_bytes:
                    raise _UploadAdmissionError(
                        413,
                        "upload_too_large",
                        "The uploaded file exceeds the configured size limit.",
                    )
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("upload write made no progress")
                    remaining = remaining[written:]
            if total == 0:
                raise _UploadAdmissionError(
                    422,
                    "empty_upload",
                    "The uploaded audio file is empty.",
                )
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o600)
            os.close(descriptor)
            descriptor = None
        except BaseException as error:
            failure = error
        try:
            await uploaded_file.close()
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            _cleanup_incoming_upload(artifacts, descriptor, incoming)
            if isinstance(failure, _UploadAdmissionError):
                raise HTTPException(
                    status_code=failure.status_code,
                    detail={"code": failure.code, "message": failure.message},
                ) from None
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "upload_failed",
                    "message": "The upload could not be stored safely.",
                },
            ) from None
        assert incoming is not None
        prepared_audio: PreparedAudio | None = None
        preflight_task = asyncio.create_task(asyncio.to_thread(runner.preflight_audio, incoming))
        try:
            prepared_audio = await _await_preflight_task(preflight_task)
        except asyncio.CancelledError:
            _cleanup_incoming_upload(artifacts, None, incoming)
            raise
        except AudioRejected:
            _cleanup_incoming_upload(artifacts, None, incoming)
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "audio_rejected",
                    "message": "The uploaded audio was rejected.",
                },
            ) from None
        except BaseException:
            _cleanup_incoming_upload(artifacts, None, incoming)
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "upload_failed",
                    "message": "The upload could not be validated safely.",
                },
            ) from None
        try:
            destination = artifacts.adopt_incoming_upload(job_id, incoming, suffix)
        except BaseException:
            prepared_audio.discard()
            _cleanup_incoming_upload(artifacts, None, incoming)
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "upload_failed",
                    "message": "The upload could not be stored safely.",
                },
            ) from None
        try:
            repository.create_job(
                str(destination),
                diagnostic_capture=diagnostic_capture,
                job_id=job_id,
                max_active_jobs=max_active_jobs,
            )
        except JobCapacityExceeded:
            prepared_audio.discard()

            def discard_capacity_rejected(proof: JobAbsenceProof) -> None:
                artifacts._discard_unregistered_job_under_absence_proof(  # noqa: SLF001
                    job_id, proof
                )

            repository.discard_if_absent(job_id, discard_capacity_rejected)
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "job_capacity_reached",
                    "message": "The local incomplete-job limit has been reached.",
                },
            ) from None
        except BaseException:

            def discard_adopted(proof: JobAbsenceProof) -> None:
                artifacts._discard_unregistered_job_under_absence_proof(  # noqa: SLF001
                    job_id, proof
                )

            try:
                committed = repository.resolve_create_after_error(
                    job_id,
                    source_name=str(destination),
                    diagnostic_capture=diagnostic_capture,
                    discard_if_absent=discard_adopted,
                )
            except BaseException:
                committed = False
            if not committed:
                prepared_audio.discard()
                raise HTTPException(
                    status_code=500,
                    detail={
                        "code": "upload_failed",
                        "message": "The upload could not be stored safely.",
                    },
                ) from None
        try:
            if not runner.admit_prepared_audio(job_id, prepared_audio, destination):
                raise RuntimeError("fresh normalization claim was unavailable")
        except BaseException:
            prepared_audio.discard()
            try:
                runner.mark_unhandled_failure(job_id)
            except Exception:
                pass
        else:
            prepared_audio.discard()
        try:
            background_tasks.add_task(_safe_background_run, runner, job_id)
        except BaseException:
            _safe_background_run(runner, job_id)
            try:
                runner.mark_unhandled_failure(job_id)
            except Exception:
                pass
        return JobCreated(job_id=job_id, status_url=f"/api/jobs/{job_id}")

    @app.get(
        "/api/jobs/{job_id}",
        response_model=PublicJob,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 422, 500),
    )
    def get_job(job_id: str) -> PublicJob:
        return _public_job(repository, runner, job_id)

    @app.post(
        "/api/jobs/{job_id}/roles",
        response_model=PublicJob,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def confirm_roles(job_id: str, confirmation: RoleConfirmation) -> PublicJob:
        try:
            runner.confirm_roles(job_id, confirmation.mapping)
        except KeyError:
            raise _not_found() from None
        except PipelineStateError as error:
            raise _pipeline_error(409, error) from None
        except PipelineValidationError as error:
            raise _pipeline_error(422, error) from None
        except ValueError:
            raise _not_found() from None
        return _public_job(repository, runner, job_id)

    @app.post(
        "/api/jobs/{job_id}/retry",
        response_model=PublicJob,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def retry(job_id: str, request: RetryRequest) -> PublicJob:
        try:
            runner.retry(job_id, request.stage)
        except KeyError:
            raise _not_found() from None
        except PipelineStateError as error:
            raise _pipeline_error(409, error) from None
        except PipelineValidationError as error:
            raise _pipeline_error(422, error) from None
        except ValueError:
            raise _not_found() from None
        return _public_job(repository, runner, job_id)

    @app.get(
        "/api/jobs/{job_id}/report",
        response_model=AnalysisReport,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def report(job_id: str) -> AnalysisReport:
        try:
            canonical = runner.report(job_id)
        except KeyError:
            raise _not_found() from None
        except PipelineStateError as error:
            raise _pipeline_error(409, error) from None
        except PipelineValidationError as error:
            raise _pipeline_error(409, error) from None
        except ValueError:
            raise _not_found() from None
        return canonical

    @app.get(
        "/api/jobs/{job_id}/expert-guidance",
        response_model=ExpertGuidanceStatus,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def get_expert_guidance(job_id: str) -> ExpertGuidanceStatus:
        try:
            # Fence arbitrary paths and deleted jobs before exposing a status.
            runner.report(job_id)
            return expert_status(expert_handoffs.read(job_id))
        except KeyError:
            raise _not_found() from None
        except PipelineStateError as error:
            raise _pipeline_error(409, error) from None
        except PipelineValidationError as error:
            raise _pipeline_error(409, error) from None
        except ValueError:
            raise _not_found() from None

    @app.post(
        "/api/jobs/{job_id}/expert-guidance/request",
        response_model=ExpertGuidanceStatus,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def request_expert_guidance(
        job_id: str, request: ExpertGuidanceRequest
    ) -> ExpertGuidanceStatus:
        """Queue a text-only ACP handoff; the API never calls a model SDK itself."""

        try:
            report = runner.report(job_id)
            target = cast(Literal["claude", "codex"], request.target)
            record = expert_handoffs.queue(build_expert_request(report, target=target))
            return expert_status(record)
        except KeyError:
            raise _not_found() from None
        except PipelineStateError as error:
            raise _pipeline_error(409, error) from None
        except PipelineValidationError as error:
            raise _pipeline_error(409, error) from None
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "expert_request_rejected",
                    "message": "The expert-guidance request could not be created.",
                },
            ) from None

    @app.post(
        "/api/jobs/{job_id}/expert-guidance/response",
        response_model=ExpertGuidanceStatus,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def submit_expert_guidance(
        job_id: str, submission: ExpertGuidanceSubmission
    ) -> ExpertGuidanceStatus:
        """Accept only ACP output matching the exact queued evidence packet."""

        try:
            runner.report(job_id)
            record = expert_handoffs.submit(
                job_id,
                request_sha256=submission.request_sha256,
                guidance=submission.guidance,
            )
            return expert_status(record)
        except KeyError:
            raise _not_found() from None
        except PipelineStateError as error:
            raise _pipeline_error(409, error) from None
        except PipelineValidationError as error:
            raise _pipeline_error(409, error) from None
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "expert_guidance_rejected",
                    "message": "The expert-guidance response did not match the queued evidence.",
                },
            ) from None

    @app.get(
        "/api/jobs/{job_id}/audio",
        response_class=StreamingResponse,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses={
            200: {"description": "Full normalized preview", "content": _WAV_CONTENT},
            206: {"description": "Partial normalized preview", "content": _WAV_CONTENT},
            **_error_responses(400, 401, 403, 404, 409, 416, 422, 500),
        },
    )
    def audio_preview(job_id: str, request: Request) -> Response:
        context = runner.open_mixed_preview(job_id)
        try:
            opened = context.__enter__()
        except KeyError:
            raise _not_found() from None
        except PipelineStateError as error:
            raise _pipeline_error(409, error) from None
        except PipelineValidationError as error:
            raise _pipeline_error(409, error) from None
        except (OSError, ValueError):
            raise _not_found() from None
        closer = _ContextCloser(context)
        try:
            size = os.fstat(opened.fileno()).st_size
        except OSError:
            closer.close()
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "invalid_normalized_audio",
                    "message": "The normalized audio preview is invalid.",
                },
            ) from None
        try:
            selected = _parse_range(request.headers.get("range"), size)
        except ValueError:
            closer.close()
            return JSONResponse(
                status_code=416,
                content={
                    "detail": {
                        "code": "range_not_satisfiable",
                        "message": "The requested audio range is not satisfiable.",
                    }
                },
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes */{size}",
                },
            )
        if selected is None:
            start, end, status_code = 0, size - 1, 200
        else:
            start, end = selected
            status_code = 206
        length = end - start + 1 if size else 0
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        }
        if status_code == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return _ManagedStreamingResponse(
            _audio_chunks(opened, closer, start, length),
            status_code=status_code,
            headers=headers,
            media_type="audio/wav",
            background=BackgroundTask(closer.close),
            closer=closer,
        )

    @app.get(
        "/api/jobs/{job_id}/role-samples/{speaker_id}/{sample_index}/audio",
        response_class=Response,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses={
            200: {"description": "Short role-confirmation WAV clip", "content": _WAV_CONTENT},
            **_error_responses(400, 401, 403, 404, 409, 422, 500),
        },
    )
    def role_sample_audio(job_id: str, speaker_id: str, sample_index: int) -> Response:
        """Serve one preselected diarization excerpt for human role confirmation."""

        context: AbstractContextManager[BinaryIO] | None = None
        try:
            candidate = runner.role_candidate(job_id)
            options = _role_samples(candidate).get(speaker_id, [])
            if sample_index < 0 or sample_index >= len(options):
                raise PipelineValidationError(
                    "invalid_role_sample",
                    "The requested role sample is not available.",
                )
            sample = options[sample_index]
            context = runner.open_mixed_preview(job_id)
            opened = context.__enter__()
            # Eight-second clips are small. Materializing them avoids a second descriptor
            # and never writes an excerpt to disk.
            return Response(
                content=_role_sample_wav(opened, start=sample.start, end=sample.end),
                media_type="audio/wav",
            )
        except KeyError:
            raise _not_found() from None
        except PipelineStateError as error:
            raise _pipeline_error(409, error) from None
        except PipelineValidationError as error:
            raise _pipeline_error(409, error) from None
        except annotation_review.ReviewRejected as error:
            raise HTTPException(
                status_code=422,
                detail={"code": error.code, "message": error.message},
            ) from None
        except (OSError, ValueError, wave.Error):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "invalid_normalized_audio",
                    "message": "The normalized audio preview is invalid.",
                },
            ) from None
        finally:
            if context is not None:
                context.__exit__(None, None, None)

    @app.get(
        "/api/annotations",
        response_model=AnnotationReviewIndex,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 500),
    )
    def list_annotations() -> AnnotationReviewIndex:
        """Counts and digests only. Choosing a draft must not reveal any speech."""

        index = annotation_review.available(selected_annotation_root)
        return AnnotationReviewIndex(
            annotations=[_review_state(state) for state in index.annotations],
            unreadable_count=index.unreadable_count,
        )

    @app.get(
        "/api/annotations/{conversation_id}",
        response_model=SilverReviewDraft,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def silver_draft(conversation_id: str) -> SilverReviewDraft:
        """Serve one draft in full. Transcript leaves this process only through here."""

        try:
            draft = annotation_review.load_draft(selected_annotation_root, conversation_id)
        except annotation_review.ReviewRejected as error:
            raise _review_error(error) from None
        return SilverReviewDraft(
            state=_review_state(draft.state),
            speakers=list(draft.speakers),
            notes=draft.notes,
            turns=[
                ReviewTurn(
                    start=turn.start,
                    end=turn.end,
                    speaker=turn.speaker,
                    transcript=turn.transcript,
                    emotion=turn.emotion,
                    emotion_rationale=turn.emotion_rationale,
                    confidence=turn.confidence,
                )
                for turn in draft.turns
            ],
            emotion_labels=list(EMOTION_LABELS),
            warnings=[_review_warning(warning) for warning in draft.warnings],
        )

    @app.get(
        "/api/annotations/{conversation_id}/alignment-proposal",
        response_model=AlignmentProposal | None,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def alignment_proposal(conversation_id: str) -> AlignmentProposal | None:
        """Timestamp-only Gemini suggestion, never applied to Silver automatically."""

        try:
            draft = annotation_review.load_draft(selected_annotation_root, conversation_id)
            record = annotation_alignment.load_proposal(
                selected_annotation_root,
                conversation_id,
                source_digest=draft.state.content_sha256,
            )
        except annotation_review.ReviewRejected as error:
            raise _review_error(error) from None
        except annotation_alignment.AlignmentError:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "alignment_proposal_unreadable",
                    "message": "The timestamp alignment proposal cannot be used for this draft.",
                },
            ) from None
        if record is None:
            return None
        content = record.get("content")
        raw_rows = content.get("rows") if isinstance(content, dict) else []
        rows = raw_rows if isinstance(raw_rows, list) else []
        validation = record.get("validation")
        raw_dropped_rules = (
            validation.get("dropped_rows_by_rule", {}) if isinstance(validation, dict) else {}
        )
        dropped_rules = raw_dropped_rules if isinstance(raw_dropped_rules, dict) else {}
        target_start = (
            int(content.get("target_start_position", 0)) if isinstance(content, dict) else 0
        )
        dropped_count = (
            int(validation.get("dropped_row_count", 0)) if isinstance(validation, dict) else 0
        )
        safe_rules = {
            str(key): int(value) for key, value in dropped_rules.items() if isinstance(value, int)
        }
        return AlignmentProposal(
            source_silver_content_sha256=draft.state.content_sha256,
            target_start_position=target_start,
            target_end_position=int(content.get("target_end_position", draft.state.turn_count))
            if isinstance(content, dict)
            else draft.state.turn_count,
            rows=[AlignmentProposalRow(**row) for row in rows if isinstance(row, dict)],
            dropped_row_count=dropped_count,
            dropped_rows_by_rule=safe_rules,
        )

    @app.get(
        "/api/annotations/{conversation_id}/alignment-proposals",
        response_model=AlignmentProposalCollection,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def alignment_proposals(conversation_id: str) -> AlignmentProposalCollection:
        """All known range-bound timing suggestions, never applied automatically."""

        try:
            draft = annotation_review.load_draft(selected_annotation_root, conversation_id)
            records = annotation_alignment.load_proposals(
                selected_annotation_root,
                conversation_id,
                source_digest=draft.state.content_sha256,
            )
        except annotation_review.ReviewRejected as error:
            raise _review_error(error) from None
        except annotation_alignment.AlignmentError:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "alignment_proposal_unreadable",
                    "message": "A timestamp alignment proposal cannot be used for this draft.",
                },
            ) from None

        proposals: list[AlignmentProposal] = []
        for record in records:
            content = record.get("content")
            validation = record.get("validation")
            if not isinstance(content, dict) or not isinstance(validation, dict):
                continue
            raw_rows = content.get("rows")
            raw_dropped_rules = validation.get("dropped_rows_by_rule", {})
            if not isinstance(raw_rows, list) or not isinstance(raw_dropped_rules, dict):
                continue
            proposals.append(
                AlignmentProposal(
                    source_silver_content_sha256=draft.state.content_sha256,
                    target_start_position=int(content["target_start_position"]),
                    target_end_position=int(
                        content.get("target_end_position", draft.state.turn_count)
                    ),
                    rows=[AlignmentProposalRow(**row) for row in raw_rows if isinstance(row, dict)],
                    dropped_row_count=int(validation.get("dropped_row_count", 0)),
                    dropped_rows_by_rule={
                        str(key): int(value)
                        for key, value in raw_dropped_rules.items()
                        if isinstance(value, int)
                    },
                )
            )
        return AlignmentProposalCollection(proposals=proposals)

    @app.get(
        "/api/annotations/{conversation_id}/reference-resegmentation-candidate",
        response_model=ReferenceResegmentationCandidate | None,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def reference_resegmentation_candidate(
        conversation_id: str,
    ) -> ReferenceResegmentationCandidate | None:
        """A KCSC human-reference repair candidate, never applied automatically."""

        try:
            draft = annotation_review.load_draft(selected_annotation_root, conversation_id)
            if draft.state.model == "kcsc-human-reference":
                return None
            candidate = annotation_reference_candidate.load_candidate(
                selected_annotation_audio_root.parent / "reference",
                conversation_id=conversation_id,
                source_silver_content_sha256=draft.state.content_sha256,
            )
        except annotation_review.ReviewRejected as error:
            raise _review_error(error) from None
        except annotation_reference_candidate.ReferenceCandidateError:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "reference_candidate_unreadable",
                    "message": "The local reference candidate cannot be used for this draft.",
                },
            ) from None
        if candidate is None:
            return None
        return ReferenceResegmentationCandidate(
            source_silver_content_sha256=candidate.source_silver_content_sha256,
            source_reference_sha256=candidate.source_reference_sha256,
            reference_turn_count=candidate.reference_turn_count,
            speakers=list(candidate.speakers),
            notes=candidate.notes,
            turns=[
                ReviewTurn(
                    start=turn.start,
                    end=turn.end,
                    speaker=turn.speaker,
                    transcript=turn.transcript,
                    emotion=turn.emotion,
                    emotion_rationale=turn.emotion_rationale,
                    confidence=turn.confidence,
                )
                for turn in candidate.turns
            ],
        )

    @app.get(
        "/api/annotations/{conversation_id}/emotion-overlay-candidate",
        response_model=EmotionOverlayCandidate | None,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def emotion_overlay_candidate(conversation_id: str) -> EmotionOverlayCandidate | None:
        """Local XLS-R emotion overlay, never applied to Silver or Gold automatically."""

        try:
            draft = annotation_review.load_draft(selected_annotation_root, conversation_id)
            candidate = annotation_emotion_candidates.load_local_candidate(
                annotation_root=selected_annotation_root,
                conversation_id=conversation_id,
                source_silver_content_sha256=draft.state.content_sha256,
            )
        except annotation_review.ReviewRejected as error:
            raise _review_error(error) from None
        except annotation_emotion_candidates.EmotionCandidateError:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "emotion_candidate_unreadable",
                    "message": "The local emotion candidate cannot be used for this draft.",
                },
            ) from None
        if candidate is None:
            return None
        content = candidate["content"]
        rows = content["turns"]
        summary = candidate.get("summary")
        source = candidate["source"]
        histogram = summary.get("emotion_histogram", {}) if isinstance(summary, dict) else {}
        return EmotionOverlayCandidate(
            source_silver_content_sha256=str(source["silver_content_sha256"]),
            model=str(source.get("model", "local emotion overlay")),
            remote_audio_transmitted=bool(source.get("remote_audio_transmitted", False)),
            emotion_histogram={str(key): int(value) for key, value in histogram.items()},
            uncertain_turns=(
                int(summary.get("uncertain_turns", 0)) if isinstance(summary, dict) else 0
            ),
            turns=[ReviewTurn(**row) for row in rows if isinstance(row, dict)],
        )

    @app.get(
        "/api/annotations/{conversation_id}/gemini-emotion-overlay-candidate",
        response_model=GeminiEmotionOverlayCandidate | None,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def gemini_emotion_overlay_candidate(
        conversation_id: str,
    ) -> GeminiEmotionOverlayCandidate | None:
        """A remote Gemini emotion overlay; read-only here and never promoted by the API."""

        try:
            draft = annotation_review.load_draft(selected_annotation_root, conversation_id)
            candidate = annotation_gemini_emotion_overlay.load_candidate(
                annotation_root=selected_annotation_root,
                conversation_id=conversation_id,
                source_silver_content_sha256=draft.state.content_sha256,
            )
        except annotation_review.ReviewRejected as error:
            raise _review_error(error) from None
        except annotation_gemini_emotion_overlay.OverlayCandidateError:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "gemini_emotion_candidate_unreadable",
                    "message": "The Gemini emotion overlay cannot be used for this draft.",
                },
            ) from None
        if candidate is None:
            return None
        content = candidate["content"]
        rows = content["turns"]
        summary = candidate.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        source = candidate["source"]
        histogram = summary.get("emotion_histogram", {})
        mean_confidence = summary.get("mean_confidence")
        return GeminiEmotionOverlayCandidate(
            source_silver_content_sha256=str(source["silver_content_sha256"]),
            model=str(source.get("model", "gemini emotion overlay")),
            # Read from the artifact rather than hardcoded: an overlay that somehow
            # recorded otherwise must not be shown to a reviewer as remote anyway.
            remote_audio_transmitted=bool(source.get("remote_audio_transmitted", True)),
            review_required=candidate.get("review_state") == "review_required",
            promotable=bool(candidate.get("promotable", False)),
            emotion_histogram={
                str(key): int(value) for key, value in histogram.items() if isinstance(value, int)
            },
            uncertain_turns=int(summary.get("uncertain_turns", 0)),
            mean_confidence=(
                float(mean_confidence) if isinstance(mean_confidence, (int, float)) else None
            ),
            turns=[ReviewTurn(**row) for row in rows if isinstance(row, dict)],
        )

    @app.get(
        "/api/annotations/{conversation_id}/audio",
        response_model=AnnotationAudioOverview,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def annotation_audio_overview(conversation_id: str) -> AnnotationAudioOverview:
        """What can be listened to for one draft: its turns, and the gaps between them.

        Separate from the draft so a machine that holds the annotation but not the
        recording still serves a reviewable draft; only listening is unavailable.
        """

        try:
            overview = annotation_review.audio_overview(
                selected_annotation_root,
                conversation_id,
                audio_root=selected_annotation_audio_root,
            )
        except annotation_review.ReviewRejected as error:
            raise _review_error(error) from None
        return AnnotationAudioOverview(
            conversation_id=overview.conversation_id,
            duration_seconds=overview.duration_seconds,
            sample_rate=overview.sample_rate,
            max_clip_seconds=overview.max_clip_seconds,
            min_gap_seconds=overview.min_gap_seconds,
            turn_clips=[
                TurnClip(position=clip.position, start=clip.start, end=clip.end)
                for clip in overview.turn_clips
            ],
            gaps=[ReviewGap(start=gap.start, end=gap.end) for gap in overview.gaps],
        )

    @app.get(
        "/api/annotations/{conversation_id}/audio/clip",
        response_class=StreamingResponse,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses={
            200: {"description": "One clipped range of the source audio", "content": _WAV_CONTENT},
            **_error_responses(400, 401, 403, 404, 409, 422, 500),
        },
    )
    def annotation_audio_clip(
        conversation_id: str,
        start: float,
        end: float,
    ) -> Response:
        """Stream one range of the draft's own recording, synthesized rather than stored.

        The range is explicit in seconds rather than expressed as a byte range: a reviewer
        asks for a turn, not for an offset, and translating time into frames here is what
        keeps the caller from ever addressing the file's bytes. The response is a complete
        small WAV, so a browser can play it without knowing anything about the source.
        """

        try:
            plan = annotation_review.plan_audio_clip(
                selected_annotation_root,
                conversation_id,
                audio_root=selected_annotation_audio_root,
                start=start,
                end=end,
            )
        except annotation_review.ReviewRejected as error:
            raise _review_error(error) from None

        def stream() -> Iterator[bytes]:
            try:
                yield from annotation_audio.clip_bytes(plan)
            except OSError:
                # The header has already been sent, so the only honest end is a short
                # body. Raising here would otherwise surface as an unhandled error.
                return

        return StreamingResponse(
            stream(),
            status_code=200,
            media_type="audio/wav",
            headers={
                "Content-Length": str(plan.total_bytes),
                "Cache-Control": "no-store",
            },
        )

    @app.post(
        "/api/annotations/{conversation_id}/gold",
        status_code=201,
        response_model=GoldPromotionResult,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def promote_annotation(
        conversation_id: str,
        submission: GoldPromotionRequest,
    ) -> GoldPromotionResult:
        """The only route to gold, and it refuses everything a person did not assert."""

        try:
            promotion = annotation_review.promote_reviewed(
                selected_annotation_root,
                conversation_id,
                reviewer=submission.reviewer,
                acknowledged=submission.acknowledged,
                corrected_turns=[turn.model_dump() for turn in submission.turns],
                review_note=submission.review_note,
                change_reasons=[reason.model_dump() for reason in submission.change_reasons],
            )
        except annotation_review.ReviewRejected as error:
            raise _review_error(error) from None
        return GoldPromotionResult(
            conversation_id=promotion.conversation_id,
            reviewer=promotion.reviewer,
            reviewed_at=promotion.reviewed_at,
            review_note=promotion.review_note,
            content_sha256=promotion.content_sha256,
            parent_silver_sha256=promotion.parent_silver_sha256,
            unchanged_from_silver=promotion.unchanged_from_silver,
            turn_count=promotion.turn_count,
            speaker_count=promotion.speaker_count,
            uncertain_turns=promotion.uncertain_turns,
            mean_confidence=promotion.mean_confidence,
            change_reason_count=promotion.change_reason_count,
            silver_unmodified=promotion.silver_unmodified,
        )

    @app.delete(
        "/api/jobs/{job_id}",
        status_code=204,
        dependencies=[Security(_CAPABILITY_SECURITY)],
        responses=_error_responses(400, 401, 403, 404, 409, 422, 500),
    )
    def delete_job(job_id: str) -> Response:
        try:
            runner.delete_job(job_id)
        except PipelineStateError as error:
            raise _pipeline_error(409, error) from None
        except (KeyError, ValueError):
            raise _not_found() from None
        return Response(status_code=204)

    return app


class _LazyProductionApplication:
    """Initialize production storage only when the ASGI server starts serving."""

    def __init__(self) -> None:
        self._application: FastAPI | None = None
        self._lock = asyncio.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._application is None:
            async with self._lock:
                if self._application is None:
                    self._application = create_app()
        await self._application(scope, receive, send)


app = _LazyProductionApplication()


__all__ = [
    "AUDIO_CHUNK_BYTES",
    "MULTIPART_OVERHEAD_BYTES",
    "UPLOAD_CHUNK_BYTES",
    "app",
    "create_app",
    "reconcile_local_state",
]
