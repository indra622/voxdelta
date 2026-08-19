"""Dependency-injected FastAPI application for the local VoxDelta pipeline."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager
from pathlib import Path
from threading import Lock
from typing import Annotated, BinaryIO, cast
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

from voxdelta.api.dependencies import build_dependencies
from voxdelta.api.schemas import (
    JobCreated,
    ProviderConfiguration,
    PublicError,
    PublicJob,
    PublicStage,
    RetryRequest,
    RoleCandidate,
    RoleConfirmation,
)
from voxdelta.config import Settings
from voxdelta.domain.models import StageName
from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.repository import JobRepository
from voxdelta.pipeline.runner import (
    PipelineRunner,
    PipelineStateError,
    PipelineValidationError,
)

UPLOAD_CHUNK_BYTES = 1024 * 1024
AUDIO_CHUNK_BYTES = 64 * 1024
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


class _UploadAdmissionError(ValueError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


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


def _cleanup_failed_upload(
    repository: JobRepository,
    artifacts: ArtifactStore,
    job_id: str,
    descriptor: int | None,
    destination: Path | None,
) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass
    if destination is not None:
        try:
            destination.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        artifacts.delete_job(job_id)
    except Exception:
        pass
    try:
        repository.delete_job(job_id)
    except Exception:
        try:
            repository.fail_unhandled_job(job_id)
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


def create_app(
    *,
    repository: JobRepository | None = None,
    artifacts: ArtifactStore | None = None,
    runner: PipelineRunner | None = None,
    max_upload_bytes: int | None = None,
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
    if max_upload_bytes is None:
        max_upload_bytes = Settings().max_upload_bytes
    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be positive")

    app = FastAPI(title="VoxDelta", version="0.1.0")

    @app.get("/api/config/providers", response_model=ProviderConfiguration)
    def provider_configuration() -> ProviderConfiguration:
        return ProviderConfiguration(stages=runner.provider_disclosures())

    @app.post("/api/jobs", status_code=202, response_model=JobCreated)
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
        job_id = repository.create_job("", diagnostic_capture=diagnostic_capture)
        destination: Path | None = None
        descriptor: int | None = None
        failure: BaseException | None = None
        try:
            directory = artifacts.job_dir(job_id)
            destination = directory / f"source-upload-{uuid4().hex}{suffix}"
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(destination, flags, 0o600)
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
            _cleanup_failed_upload(
                repository,
                artifacts,
                job_id,
                descriptor,
                destination,
            )
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
        try:
            repository.update_source_name(job_id, str(destination))
            background_tasks.add_task(_safe_background_run, runner, job_id)
        except BaseException:
            _cleanup_failed_upload(
                repository,
                artifacts,
                job_id,
                descriptor,
                destination,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "upload_failed",
                    "message": "The upload could not be stored safely.",
                },
            ) from None
        return JobCreated(job_id=job_id, status_url=f"/api/jobs/{job_id}")

    @app.get("/api/jobs/{job_id}", response_model=PublicJob)
    def get_job(job_id: str) -> PublicJob:
        return _public_job(repository, runner, job_id)

    @app.post("/api/jobs/{job_id}/roles", response_model=PublicJob)
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

    @app.post("/api/jobs/{job_id}/retry", response_model=PublicJob)
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

    @app.get("/api/jobs/{job_id}/report")
    def report(job_id: str) -> dict[str, object]:
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
        return canonical.model_dump(mode="json")

    @app.get("/api/jobs/{job_id}/audio")
    def audio_preview(job_id: str, request: Request) -> StreamingResponse:
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
            return StreamingResponse(
                iter(()),
                status_code=416,
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes */{size}",
                    "Content-Length": "0",
                },
                media_type="audio/wav",
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
        return StreamingResponse(
            _audio_chunks(opened, closer, start, length),
            status_code=status_code,
            headers=headers,
            media_type="audio/wav",
            background=BackgroundTask(closer.close),
        )

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> Response:
        try:
            runner.delete_job(job_id)
        except PipelineStateError as error:
            raise _pipeline_error(409, error) from None
        except (KeyError, ValueError):
            raise _not_found() from None
        return Response(status_code=204)

    return app


app = create_app()


__all__ = ["AUDIO_CHUNK_BYTES", "UPLOAD_CHUNK_BYTES", "app", "create_app"]
