"""Dependency-injected FastAPI application for the local VoxDelta pipeline."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager, asynccontextmanager
from pathlib import Path
from threading import Lock
from time import time
from typing import Annotated, Any, BinaryIO, cast
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

from voxdelta.api.dependencies import build_dependencies
from voxdelta.api.schemas import (
    JobCreated,
    ProviderConfiguration,
    PublicError,
    PublicErrorEnvelope,
    PublicJob,
    PublicStage,
    RetryRequest,
    RoleCandidate,
    RoleConfirmation,
)
from voxdelta.audio.service import AudioRejected, PreparedAudio
from voxdelta.config import Settings
from voxdelta.domain.models import AnalysisReport, StageName
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
_SUPPORTED_SUFFIXES = frozenset({".wav", ".mp3", ".m4a"})
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
        if path.startswith("/api/jobs"):
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
    if max_upload_bytes is None:
        max_upload_bytes = Settings().max_upload_bytes
    if reconciliation_lease_seconds is None:
        reconciliation_lease_seconds = Settings().admission_reconciliation_lease_seconds
    if max_active_jobs is None:
        max_active_jobs = Settings().max_active_jobs
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
    if (
        len(raw_capability_token) < 32
        or not raw_capability_token.isascii()
        or any(character.isspace() for character in raw_capability_token)
    ):
        raise ValueError(
            "api capability token must contain at least 32 ASCII characters without whitespace"
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
                    "message": "Upload a WAV, MP3, or M4A audio file.",
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
            prepared_audio = await asyncio.shield(preflight_task)
        except asyncio.CancelledError:
            try:
                late_prepared = await preflight_task
            except BaseException:
                pass
            else:
                late_prepared.discard()
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
