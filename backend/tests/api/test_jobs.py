from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import threading
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi import BackgroundTasks, FastAPI, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import SecretStr
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.requests import ClientDisconnect

import voxdelta.jobs.artifacts as artifacts_module
from voxdelta.api.app import create_app, reconcile_local_state
from voxdelta.audio.service import AudioService
from voxdelta.domain.models import AudioAsset, StageName, StageStatus
from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.repository import JobRepository
from voxdelta.pipeline.runner import PipelineRunner
from voxdelta.pipeline.stages import MediaReference, NormalizeArtifact, cache_key_for_stage
from voxdelta.providers.fake import FakeDiarizationProvider

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_65s.wav"
TEST_CAPABILITY_TOKEN = "test-capability-token-with-enough-entropy"


class UnexpectedDiarizer(FakeDiarizationProvider):
    def diarize(self, asset: AudioAsset):  # type: ignore[no-untyped-def]
        raise RuntimeError("private /absolute/path transcript secret-token")


class BlockingDiarizer(FakeDiarizationProvider):
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def diarize(self, asset: AudioAsset):  # type: ignore[no-untyped-def]
        self.entered.set()
        assert self.release.wait(timeout=5)
        return super().diarize(asset)


class PreClaimFailRunner(PipelineRunner):
    def run_until_pause(self, job_id: str) -> dict[str, object]:
        del job_id
        raise RuntimeError("private pre-claim failure /absolute/path")


class FailingCreateRepository(JobRepository):
    def create_job(
        self,
        source_name: str,
        diagnostic_capture: bool = False,
        *,
        job_id: str | None = None,
        max_active_jobs: int | None = None,
    ) -> str:
        del source_name, diagnostic_capture, job_id, max_active_jobs
        raise RuntimeError("private database create failure")


class CommitThenRaiseRepository(JobRepository):
    def create_job(
        self,
        source_name: str,
        diagnostic_capture: bool = False,
        *,
        job_id: str | None = None,
        max_active_jobs: int | None = None,
    ) -> str:
        created = super().create_job(
            source_name,
            diagnostic_capture=diagnostic_capture,
            job_id=job_id,
            max_active_jobs=max_active_jobs,
        )
        raise RuntimeError(f"ambiguous commit for {created}")


class FailOnceFinalizeRepository(JobRepository):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.finalize_attempts = 0

    def finalize_delete(self, job_id: str) -> None:
        self.finalize_attempts += 1
        if self.finalize_attempts == 1:
            raise RuntimeError("private database finalization row")
        super().finalize_delete(job_id)


def build_harness(
    tmp_path: Path,
    *,
    diarizer: FakeDiarizationProvider | None = None,
    max_upload_bytes: int | None = None,
    artifacts_override: ArtifactStore | None = None,
    repository_override: JobRepository | None = None,
    runner_type: type[PipelineRunner] = PipelineRunner,
) -> tuple[FastAPI, JobRepository, ArtifactStore, PipelineRunner]:
    jobs_root = tmp_path / "jobs"
    repository = repository_override or JobRepository(tmp_path / "voxdelta.sqlite3")
    artifacts = artifacts_override or ArtifactStore(jobs_root)
    runner = runner_type(
        repository,
        artifacts,
        AudioService(jobs_root, 60, 3600),
        diarization_provider=diarizer,
    )
    return (
        create_app(
            repository=repository,
            artifacts=artifacts,
            runner=runner,
            max_upload_bytes=max_upload_bytes,
            api_capability_token=SecretStr(TEST_CAPABILITY_TOKEN),
        ),
        repository,
        artifacts,
        runner,
    )


@asynccontextmanager
async def client_for(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
        headers={"X-VoxDelta-Token": TEST_CAPABILITY_TOKEN},
    ) as client:
        yield client


def assert_no_private_paths(value: object, root: Path) -> None:
    serialized = json.dumps(value, ensure_ascii=False)
    assert str(root) not in serialized
    assert "artifact_path" not in serialized
    assert "cache_key" not in serialized
    assert "artifact_hash" not in serialized
    assert "claim_token" not in serialized


async def upload(client: httpx.AsyncClient, *, diagnostic_capture: str = "false") -> httpx.Response:
    return await client.post(
        "/api/jobs",
        files={"file": ("../../private-call.WAV", FIXTURE.read_bytes(), "audio/wav")},
        data={"diagnostic_capture": diagnostic_capture},
    )


async def raw_chunked_request(
    app: FastAPI,
    body: bytes,
    *,
    include_content_length: bool,
    content_length_values: list[bytes] | None = None,
    chunk_size: int = 4096,
) -> tuple[list[dict[str, object]], int, int]:
    """Drive the ASGI app without allowing an HTTP client to pre-buffer the request."""

    offset = 0
    received = 0
    receive_calls = 0
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        nonlocal offset, receive_calls, received
        receive_calls += 1
        chunk = body[offset : offset + chunk_size]
        offset += len(chunk)
        received += len(chunk)
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": offset < len(body),
        }

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    boundary = b"voxdelta-boundary"
    headers = [
        (b"host", b"localhost"),
        (b"x-voxdelta-token", TEST_CAPABILITY_TOKEN.encode("ascii")),
        (b"content-type", b"multipart/form-data; boundary=" + boundary),
    ]
    if content_length_values is not None:
        headers.extend((b"content-length", value) for value in content_length_values)
    elif include_content_length:
        headers.append((b"content-length", str(len(body)).encode("ascii")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/jobs",
        "raw_path": b"/api/jobs",
        "query_string": b"",
        "headers": headers,
        "client": ("test", 1),
        "server": ("localhost", 80),
    }
    await app(scope, receive, send)  # type: ignore[arg-type]
    return sent, received, receive_calls


def job_count(repository: JobRepository) -> int:
    with sqlite3.connect(repository.path) as database:
        return int(database.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])


@pytest.mark.asyncio
async def test_docs_and_provider_disclosures_are_public_and_secret_free(tmp_path: Path) -> None:
    app, _, _, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        docs = await client.get("/docs")
        response = await client.get("/api/config/providers")

    assert docs.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    assert [item["stage"] for item in payload["stages"]] == [stage.value for stage in StageName]
    assert all(
        set(item) == {"stage", "provenance", "transmits", "retention_policy_url"}
        for item in payload["stages"]
    )
    assert all("api_key" not in json.dumps(item).casefold() for item in payload["stages"])
    assert_no_private_paths(payload, tmp_path)


def test_importing_production_api_module_creates_no_runtime_database(tmp_path: Path) -> None:
    data_root = tmp_path / "isolated-data"
    environment = {
        **os.environ,
        "VOXDELTA_DATA_ROOT": str(data_root),
        "VOXDELTA_DATABASE_PATH": str(data_root / "voxdelta.sqlite3"),
    }

    subprocess.run(
        [sys.executable, "-c", "import voxdelta.api.app"],
        check=True,
        capture_output=True,
        env=environment,
        timeout=15,
    )

    assert not data_root.exists()


@pytest.mark.asyncio
async def test_job_routes_require_capability_and_reject_untrusted_host_or_origin(
    tmp_path: Path,
) -> None:
    jobs_root = tmp_path / "jobs"
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")
    artifacts = ArtifactStore(jobs_root)
    runner = PipelineRunner(repository, artifacts, AudioService(jobs_root, 60, 3600))
    with pytest.raises(ValueError, match="capability token"):
        create_app(repository=repository, artifacts=artifacts, runner=runner)
    app = create_app(
        repository=repository,
        artifacts=artifacts,
        runner=runner,
        api_capability_token=SecretStr(TEST_CAPABILITY_TOKEN),
    )
    job_id = repository.create_job("private-source.wav")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
    ) as client:
        missing = await client.get(f"/api/jobs/{job_id}")
        wrong = await client.get(
            f"/api/jobs/{job_id}",
            headers={"X-VoxDelta-Token": "wrong"},
        )
        hostile_origin = await client.get(
            f"/api/jobs/{job_id}",
            headers={
                "X-VoxDelta-Token": TEST_CAPABILITY_TOKEN,
                "Origin": "https://attacker.example",
            },
        )
        hostile_host = await client.get(
            f"/api/jobs/{job_id}",
            headers={
                "X-VoxDelta-Token": TEST_CAPABILITY_TOKEN,
                "Host": "attacker.example",
            },
        )
        accepted = await client.get(
            f"/api/jobs/{job_id}",
            headers={"X-VoxDelta-Token": TEST_CAPABILITY_TOKEN},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert hostile_origin.status_code == 403
    assert hostile_host.status_code == 400
    assert accepted.status_code == 200
    assert "private-source" not in "".join(
        response.text for response in (missing, wrong, hostile_origin, hostile_host)
    )


@pytest.mark.asyncio
async def test_incomplete_job_quota_rejects_second_admission_without_residue(
    tmp_path: Path,
) -> None:
    jobs_root = tmp_path / "jobs"
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")
    artifacts = ArtifactStore(jobs_root)
    runner = PipelineRunner(repository, artifacts, AudioService(jobs_root, 60, 3600))
    app = create_app(
        repository=repository,
        artifacts=artifacts,
        runner=runner,
        api_capability_token=SecretStr(TEST_CAPABILITY_TOKEN),
        max_active_jobs=1,
    )
    headers = {"X-VoxDelta-Token": TEST_CAPABILITY_TOKEN}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
        headers=headers,
    ) as client:
        first = await upload(client)
        second = await upload(client)

    assert first.status_code == 202
    assert second.status_code == 429
    assert second.json()["detail"]["code"] == "job_capacity_reached"
    assert job_count(repository) == 1
    assert len(list(artifacts.root.glob("[0-9a-f]" * 32))) == 1


@pytest.mark.asyncio
async def test_request_validation_errors_are_uniform_and_never_echo_input(
    tmp_path: Path,
) -> None:
    app, _, _, _ = build_harness(tmp_path)
    transcript = "PRIVATE_TRANSCRIPT_SENTINEL"
    provider_payload = "PRIVATE_PROVIDER_PAYLOAD_SENTINEL"
    expected = {
        "detail": {
            "code": "invalid_request",
            "message": "The request is invalid.",
        }
    }

    async with client_for(app) as client:
        invalid_role = await client.post(
            "/api/jobs/validation-sentinel/roles",
            json={
                "mapping": {
                    "SPEAKER_00": "customer",
                    "SPEAKER_01": "PRIVATE_INVALID_ROLE_SENTINEL",
                },
                "transcript": transcript,
                "provider_payload": {"raw": provider_payload},
            },
        )
        malformed_retry = await client.post(
            "/api/jobs/validation-sentinel/retry",
            json={
                "stage": {"transcript": transcript},
                "provider_payload": provider_payload,
            },
        )
        missing_multipart_file = await client.post(
            "/api/jobs",
            data={
                "transcript": transcript,
                "provider_payload": provider_payload,
            },
        )

    for response in (invalid_role, malformed_retry, missing_multipart_file):
        assert response.status_code == 422
        assert response.json() == expected
        assert transcript not in response.text
        assert provider_payload not in response.text
        assert "PRIVATE_INVALID_ROLE_SENTINEL" not in response.text
        assert str(tmp_path) not in response.text


@pytest.mark.asyncio
async def test_validation_handler_preserves_openapi_and_intentional_http_errors(
    tmp_path: Path,
) -> None:
    app, _, _, _ = build_harness(tmp_path)

    async with client_for(app) as client:
        openapi = await client.get("/openapi.json")
        missing = await client.get("/api/jobs/missing")

    assert openapi.status_code == 200
    assert missing.status_code == 404
    assert missing.json() == {
        "detail": {
            "code": "job_not_found",
            "message": "The requested job was not found.",
        }
    }


@pytest.mark.asyncio
async def test_openapi_documents_uniform_request_validation_envelope(tmp_path: Path) -> None:
    app, _, _, _ = build_harness(tmp_path)

    async with client_for(app) as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    error_envelope = schema["components"]["schemas"]["PublicErrorEnvelope"]
    assert error_envelope["additionalProperties"] is False
    assert error_envelope["required"] == ["detail"]
    assert error_envelope["properties"]["detail"] == {"$ref": "#/components/schemas/PublicError"}

    for path in (
        "/api/jobs",
        "/api/jobs/{job_id}/roles",
        "/api/jobs/{job_id}/retry",
    ):
        validation_response = schema["paths"][path]["post"]["responses"]["422"]
        assert validation_response["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/PublicErrorEnvelope"
        }
        assert "HTTPValidationError" not in json.dumps(validation_response)


@pytest.mark.asyncio
async def test_openapi_exactly_documents_report_audio_and_public_error_contracts(
    tmp_path: Path,
) -> None:
    app, _, _, _ = build_harness(tmp_path)

    async with client_for(app) as client:
        schema = (await client.get("/openapi.json")).json()

    paths = schema["paths"]
    report = paths["/api/jobs/{job_id}/report"]["get"]["responses"]
    assert report["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AnalysisReport"
    }
    assert {"400", "401", "403", "404", "409"}.issubset(report)

    audio = paths["/api/jobs/{job_id}/audio"]["get"]["responses"]
    assert {"200", "206", "400", "401", "403", "404", "409", "416"}.issubset(audio)
    for status in ("200", "206"):
        assert audio[status]["content"]["audio/wav"]["schema"] == {
            "type": "string",
            "format": "binary",
        }
    for status in ("400", "401", "403", "404", "409", "416"):
        assert audio[status]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/PublicErrorEnvelope"
        }

    create = paths["/api/jobs"]["post"]["responses"]
    assert {"202", "400", "401", "403", "413", "422", "429", "500"}.issubset(create)
    assert "HTTPValidationError" not in json.dumps(paths)


@pytest.mark.asyncio
async def test_upload_pauses_persists_diagnostic_false_and_never_trusts_filename(
    tmp_path: Path,
) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        response = await upload(client)
        status = await client.get(response.json()["status_url"])

    assert response.status_code == 202
    assert set(response.json()) == {"job_id", "status_url"}
    job_id = response.json()["job_id"]
    assert response.json()["status_url"] == f"/api/jobs/{job_id}"
    assert status.status_code == 200
    assert status.json()["status"] == "paused"
    assert status.json()["diagnostic_capture"] is False
    assert status.json()["stages"]["confirm_roles"]["status"] == "paused"
    assert set(status.json()["role_candidate"]["speakers"]) == {"SPEAKER_00", "SPEAKER_01"}
    assert_no_private_paths(status.json(), tmp_path)
    row = repository.get_job(job_id)
    assert row["diagnostic_capture"] == 0
    source = Path(str(row["source_name"]))
    assert source.parent == artifacts.job_dir(job_id)
    assert source.name == "source-upload.wav"
    assert source.suffix == ".wav"
    assert source.stat().st_mode & 0o777 == 0o600
    assert "private-call" not in source.name


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_suffix_without_creating_job(tmp_path: Path) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        response = await client.post(
            "/api/jobs",
            files={"file": ("../../call.exe", b"not audio", "application/octet-stream")},
        )

    assert response.status_code == 422
    with sqlite3.connect(repository.path) as database:
        assert database.execute("SELECT COUNT(*) FROM jobs").fetchone() == (0,)
    assert not artifacts.root.exists()
    assert "call.exe" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "payload", "expected_code"),
    [
        ("corrupt.wav", b"not actually audio", "audio_rejected"),
        ("short.wav", b"RIFF-invalid", "audio_rejected"),
    ],
)
async def test_upload_preflight_rejects_invalid_audio_before_durable_admission(
    tmp_path: Path,
    name: str,
    payload: bytes,
    expected_code: str,
) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path)

    async with client_for(app) as client:
        response = await client.post(
            "/api/jobs",
            files={"file": (name, payload, "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": expected_code,
            "message": "The uploaded audio was rejected.",
        }
    }
    assert job_count(repository) == 0
    assert not list(artifacts.root.glob("[0-9a-f]" * 32)) if artifacts.root.exists() else True
    assert not list((artifacts.root / ".incoming").glob("*"))


@pytest.mark.asyncio
async def test_upload_preflight_rejects_decoded_duration_before_durable_admission(
    tmp_path: Path,
) -> None:
    short = tmp_path / "short.wav"
    with wave.open(str(short), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 16_000)
    app, repository, artifacts, _ = build_harness(tmp_path)

    async with client_for(app) as client:
        response = await client.post(
            "/api/jobs",
            files={"file": ("short.wav", short.read_bytes(), "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "audio_rejected",
        "message": "The uploaded audio was rejected.",
    }
    assert job_count(repository) == 0
    assert not list(artifacts.root.glob("[0-9a-f]" * 32)) if artifacts.root.exists() else True


@pytest.mark.asyncio
async def test_upload_reuses_preflight_normalization_without_decoding_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import voxdelta.audio.service as audio_module

    calls = 0
    original = audio_module._normalize_mixed

    def count_normalization(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        original(source, target)

    monkeypatch.setattr(audio_module, "_normalize_mixed", count_normalization)
    app, _, _, _ = build_harness(tmp_path)

    async with client_for(app) as client:
        response = await upload(client)

    assert response.status_code == 202
    assert calls == 1


@pytest.mark.asyncio
async def test_upload_accepts_exact_byte_cap_and_reads_only_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_sizes: list[int] = []
    original_read = StarletteUploadFile.read

    async def recording_read(self: UploadFile, size: int = -1) -> bytes:
        read_sizes.append(size)
        return await original_read(self, size)

    monkeypatch.setattr(StarletteUploadFile, "read", recording_read)
    cap = FIXTURE.stat().st_size
    app, repository, _, _ = build_harness(tmp_path, max_upload_bytes=cap)

    async with client_for(app) as client:
        response = await upload(client)

    assert response.status_code == 202
    assert job_count(repository) == 1
    assert read_sizes
    assert set(read_sizes) == {1024 * 1024}


@pytest.mark.asyncio
async def test_upload_rejects_cap_plus_one_and_empty_and_cleans_partial_state(
    tmp_path: Path,
) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path, max_upload_bytes=8)
    async with client_for(app) as client:
        oversized = [
            await client.post(
                "/api/jobs",
                files={"file": ("call.wav", b"123456789", "audio/wav")},
            )
            for _ in range(2)
        ]
        empty = [
            await client.post(
                "/api/jobs",
                files={"file": ("call.wav", b"", "audio/wav")},
            )
            for _ in range(2)
        ]

    assert all(response.status_code == 413 for response in oversized)
    assert all(response.json()["detail"]["code"] == "upload_too_large" for response in oversized)
    assert all(response.status_code == 422 for response in empty)
    assert all(response.json()["detail"]["code"] == "empty_upload" for response in empty)
    assert job_count(repository) == 0
    assert not artifacts.root.exists() or not list(artifacts.root.glob("[!.]*"))
    assert not (artifacts.root / ".deleted").exists()
    assert not (artifacts.root / ".locks").exists()
    assert not [
        path
        for path in artifacts_module._OPERATION_LOCKS  # noqa: SLF001
        if artifacts.root in path.parents
    ]
    assert not [
        path
        for path in artifacts_module._INCOMING_OWNERS  # noqa: SLF001
        if artifacts.root in path.parents
    ]
    assert not list((artifacts.root / ".incoming").glob("*"))
    assert str(tmp_path) not in "".join(response.text for response in oversized + empty)


@pytest.mark.asyncio
async def test_invalid_admissions_preserve_claimed_job_and_sentinel(tmp_path: Path) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path, max_upload_bytes=8)
    claimed = repository.create_job("existing.wav")
    assert repository.claim_stage(claimed, StageName.NORMALIZE) is not None
    sentinel = artifacts.job_dir(claimed) / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    async with client_for(app) as client:
        responses = [
            await client.post(
                "/api/jobs",
                files={"file": ("call.wav", b"123456789", "audio/wav")},
            )
            for _ in range(3)
        ]

    assert all(response.status_code == 413 for response in responses)
    assert repository.get_job(claimed)["stages"]["normalize"]["status"] == "running"
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (artifacts.root / ".deleted").exists()
    assert not (artifacts.root / ".locks").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("include_content_length", [False, True])
async def test_raw_multipart_ingress_stops_near_request_cap_before_parsing(
    tmp_path: Path,
    include_content_length: bool,
) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path, max_upload_bytes=8)
    boundary = b"voxdelta-boundary"
    body = b"".join(
        (
            b"--" + boundary + b"\r\n",
            b'Content-Disposition: form-data; name="file"; filename="call.wav"\r\n',
            b"Content-Type: audio/wav\r\n\r\n",
            b"x" * (3 * 1024 * 1024),
            b"\r\n--" + boundary + b"--\r\n",
        )
    )

    sent, received, receive_calls = await raw_chunked_request(
        app,
        body,
        include_content_length=include_content_length,
    )

    start = next(message for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")  # type: ignore[arg-type]
        for message in sent
        if message["type"] == "http.response.body"
    )
    assert start["status"] == 413
    assert json.loads(response_body)["detail"] == {
        "code": "upload_too_large",
        "message": "The uploaded file exceeds the configured size limit.",
    }
    assert received <= 8 + (64 * 1024) + 4096
    assert received < len(body)
    if include_content_length:
        assert receive_calls == 0
    else:
        assert receive_calls > 0
    assert job_count(repository) == 0
    assert not artifacts.root.exists()


@pytest.mark.asyncio
async def test_obviously_oversized_content_length_rejects_without_receiving_body(
    tmp_path: Path,
) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path, max_upload_bytes=8)
    body = b"x" * (3 * 1024 * 1024)

    sent, received, receive_calls = await raw_chunked_request(
        app,
        body,
        include_content_length=False,
        content_length_values=[str(8 + (64 * 1024) + 1).encode("ascii")],
    )

    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == 413
    assert received == 0
    assert receive_calls == 0
    assert job_count(repository) == 0
    assert not artifacts.root.exists()


@pytest.mark.asyncio
async def test_extremely_long_content_length_is_bounded_and_never_received(tmp_path: Path) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path, max_upload_bytes=8)

    sent, received, receive_calls = await raw_chunked_request(
        app,
        b"unread body",
        include_content_length=False,
        content_length_values=[b"9" * 5000],
    )

    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == 413
    assert received == 0
    assert receive_calls == 0
    assert job_count(repository) == 0
    assert not artifacts.root.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_length_values",
    [[b"invalid"], [b"-1"], [b"1", b"9999999"]],
)
async def test_malformed_or_conflicting_content_length_uses_streaming_cap(
    tmp_path: Path,
    content_length_values: list[bytes],
) -> None:
    app, repository, _, _ = build_harness(tmp_path, max_upload_bytes=8)
    boundary = b"voxdelta-boundary"
    body = b"".join(
        (
            b"--" + boundary + b"\r\n",
            b'Content-Disposition: form-data; name="file"; filename="call.wav"\r\n',
            b"Content-Type: audio/wav\r\n\r\n",
            b"x" * (3 * 1024 * 1024),
            b"\r\n--" + boundary + b"--\r\n",
        )
    )

    sent, received, receive_calls = await raw_chunked_request(
        app,
        body,
        include_content_length=False,
        content_length_values=content_length_values,
    )

    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == 413
    assert 0 < received <= 8 + (64 * 1024) + 4096
    assert receive_calls > 0
    assert job_count(repository) == 0


@pytest.mark.asyncio
async def test_upload_close_failure_is_sanitized_and_cleans_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_close = StarletteUploadFile.close
    close_calls = 0

    async def failing_close(self: UploadFile) -> None:
        nonlocal close_calls
        close_calls += 1
        await original_close(self)
        if close_calls == 1:
            raise OSError("private close /absolute/path")

    monkeypatch.setattr(StarletteUploadFile, "close", failing_close)
    app, repository, artifacts, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        response = await upload(client)

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "upload_failed",
        "message": "The upload could not be stored safely.",
    }
    assert job_count(repository) == 0
    assert not artifacts.root.exists() or not list(artifacts.root.glob("[!.]*"))
    assert not (artifacts.root / ".deleted").exists()
    assert not (artifacts.root / ".locks").exists()
    assert not list((artifacts.root / ".incoming").glob("*"))
    assert "private" not in response.text
    assert str(tmp_path) not in response.text


@pytest.mark.asyncio
async def test_upload_ownership_spans_file_close_until_adoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path)
    original_close = StarletteUploadFile.close
    observed: list[Path] = []

    async def reconcile_during_close(self: UploadFile) -> None:
        candidates = list((artifacts.root / ".incoming").glob(".upload-*"))
        if not candidates:
            await original_close(self)
            return
        assert len(candidates) == 1 and not observed
        incoming = candidates[0]
        observed.append(incoming)
        os.utime(incoming, (0, 0))
        assert (
            ArtifactStore(artifacts.root).remove_stale_incoming_uploads(
                lease_seconds=10,
                now=1000,
            )
            == 0
        )
        await original_close(self)

    monkeypatch.setattr(StarletteUploadFile, "close", reconcile_during_close)
    async with client_for(app) as client:
        response = await upload(client)

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert observed
    assert not observed[0].exists()
    assert observed[0].absolute() not in artifacts_module._INCOMING_OWNERS  # noqa: SLF001
    assert repository.get_job(job_id)["status"] == "paused"


@pytest.mark.asyncio
async def test_upload_cleanup_continues_after_source_unlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_unlink = Path.unlink

    def failing_source_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.startswith(".upload-"):
            raise OSError("private unlink path")
        original_unlink(self, *args, **kwargs)  # type: ignore[arg-type]

    original_close = StarletteUploadFile.close
    close_calls = 0

    async def failing_close(self: UploadFile) -> None:
        nonlocal close_calls
        close_calls += 1
        await original_close(self)
        if close_calls == 1:
            raise OSError("trigger cleanup")

    monkeypatch.setattr(Path, "unlink", failing_source_unlink)
    monkeypatch.setattr(StarletteUploadFile, "close", failing_close)
    app, repository, artifacts, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        response = await upload(client)

    assert response.status_code == 500
    assert job_count(repository) == 0
    assert not artifacts.root.exists() or not list(artifacts.root.glob("[!.]*"))


@pytest.mark.asyncio
async def test_database_create_failure_after_adopt_discards_unregistered_directory(
    tmp_path: Path,
) -> None:
    repository = FailingCreateRepository(tmp_path / "voxdelta.sqlite3")
    app, _, artifacts, _ = build_harness(
        tmp_path,
        repository_override=repository,
    )
    async with client_for(app) as client:
        response = await upload(client)

    assert response.status_code == 500
    assert job_count(repository) == 0
    assert not [
        path for path in artifacts.root.iterdir() if path.is_dir() and not path.name.startswith(".")
    ]
    assert not list((artifacts.root / ".incoming").glob("*"))
    assert not (artifacts.root / ".deleted").exists()
    assert not (artifacts.root / ".locks").exists()
    assert "private" not in response.text


@pytest.mark.asyncio
async def test_database_commit_then_raise_is_verified_and_continues(tmp_path: Path) -> None:
    repository = CommitThenRaiseRepository(tmp_path / "voxdelta.sqlite3")
    app, _, artifacts, _ = build_harness(tmp_path, repository_override=repository)

    async with client_for(app) as client:
        response = await upload(client)
        status = await client.get(response.json()["status_url"])

    assert response.status_code == 202
    assert status.status_code == 200
    assert status.json()["status"] == "paused"
    job_id = response.json()["job_id"]
    assert repository.get_job(job_id)["source_name"] == str(
        artifacts.root / job_id / "source-upload.wav"
    )


@pytest.mark.asyncio
async def test_scheduling_claim_then_raise_keeps_job_and_terminalizes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path)

    def claim_then_raise(
        self: BackgroundTasks,
        function: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        del self, function, kwargs
        job_id = str(args[-1])
        assert repository.claim_stage(job_id, StageName.DIARIZE) is not None
        raise RuntimeError("scheduler failed after claim")

    monkeypatch.setattr(BackgroundTasks, "add_task", claim_then_raise)
    async with client_for(app) as client:
        response = await upload(client)
        status = await client.get(response.json()["status_url"])

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert status.status_code == 200
    assert status.json()["status"] == "failed"
    assert repository.get_job(job_id)["status"] == "failed"
    assert (artifacts.root / job_id / "source-upload.wav").is_file()


def test_reconciliation_cleans_only_stale_absent_state_and_recovers_pending_job(
    tmp_path: Path,
) -> None:
    _, repository, artifacts, runner = build_harness(tmp_path)
    incoming_id = "a" * 32
    orphan_id = "b" * 32
    pending_id = "c" * 32
    tombstone_id = "d" * 32
    lock_id = "e" * 32
    claimed_id = "f" * 32

    incoming_directory = artifacts.root / ".incoming"
    incoming_directory.mkdir(mode=0o700, parents=True)
    stale_incoming = incoming_directory / f".upload-{incoming_id}-crash.wav"
    stale_incoming.write_bytes(b"stale")
    stale_incoming.chmod(0o600)
    os.utime(stale_incoming, (0, 0))

    descriptor, orphan_incoming = artifacts.open_incoming_upload(orphan_id, ".wav")
    os.write(descriptor, b"orphan")
    os.close(descriptor)
    artifacts.adopt_incoming_upload(orphan_id, orphan_incoming, ".wav")
    os.utime(artifacts.root / orphan_id, (0, 0))

    descriptor, pending_incoming = artifacts.open_incoming_upload(pending_id, ".wav")
    os.write(descriptor, FIXTURE.read_bytes())
    os.fsync(descriptor)
    os.close(descriptor)
    pending_source = artifacts.adopt_incoming_upload(pending_id, pending_incoming, ".wav")
    repository.create_job(str(pending_source), job_id=pending_id)

    claimed_source = artifacts.job_dir(claimed_id) / "source-upload.wav"
    claimed_source.write_bytes(b"claimed sentinel")
    repository.create_job(str(claimed_source), job_id=claimed_id)
    assert repository.claim_stage(claimed_id, StageName.NORMALIZE) is not None
    os.utime(artifacts.root / claimed_id, (0, 0))

    artifacts.mark_deletion_tombstone(tombstone_id)
    with artifacts.operation_lock(lock_id):
        pass

    recovered = reconcile_local_state(
        repository,
        artifacts,
        runner,
        lease_seconds=10,
        now=1000,
    )

    assert recovered == (pending_id,)
    assert not stale_incoming.exists()
    assert not (artifacts.root / orphan_id).exists()
    assert repository.get_job(pending_id)["status"] == "paused"
    assert claimed_source.read_bytes() == b"claimed sentinel"
    assert repository.get_job(claimed_id)["status"] == "running"
    assert (artifacts.root / ".deleted" / f"{tombstone_id}.tombstone").is_file()
    assert (artifacts.root / ".locks" / f"{lock_id}.lock").is_file()


@pytest.mark.parametrize(
    "stage",
    [StageName.NORMALIZE, StageName.DIARIZE, StageName.REPORT],
)
def test_reconciliation_schedules_expired_running_claims_but_never_live_claims(
    stage: StageName,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    current = [now]
    repository = JobRepository(
        tmp_path / "voxdelta.sqlite3",
        clock=lambda: current[0],
        claim_lease_seconds=30,
    )
    artifacts = ArtifactStore(tmp_path / "jobs")
    runner = PipelineRunner(repository, artifacts, AudioService(artifacts.root, 60, 3600))

    def create_claimed(job_id: str) -> None:
        source = artifacts.job_dir(job_id) / "source-upload.wav"
        source.write_bytes(FIXTURE.read_bytes())
        repository.create_job(str(source), job_id=job_id)
        assert repository.claim_stage(job_id, stage) is not None

    expired = "1" * 32
    live = "2" * 32
    create_claimed(expired)
    current[0] += timedelta(seconds=31)
    create_claimed(live)
    scheduled: list[str] = []

    recovered = reconcile_local_state(
        repository,
        artifacts,
        runner,
        lease_seconds=10,
        now=1000,
        schedule=scheduled.append,
    )

    assert recovered == (expired,)
    assert scheduled == [expired]
    assert repository.get_job(expired)["stages"][stage.value]["status"] == "running"
    assert repository.get_job(live)["stages"][stage.value]["status"] == "running"


def test_symlink_root_is_canonical_for_persisted_source_and_pending_recovery(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-jobs"
    real_root.mkdir()
    alias_root = tmp_path / "jobs-alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    artifacts = ArtifactStore(alias_root)
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")
    runner = PipelineRunner(
        repository,
        artifacts,
        AudioService(artifacts.root, 60, 3600),
    )
    job_id = "9" * 32
    descriptor, incoming = artifacts.open_incoming_upload(job_id, ".wav")
    os.write(descriptor, FIXTURE.read_bytes())
    os.fsync(descriptor)
    os.close(descriptor)
    source = artifacts.adopt_incoming_upload(job_id, incoming, ".wav")
    repository.create_job(str(source), job_id=job_id)

    recovered = reconcile_local_state(
        repository,
        artifacts,
        runner,
        lease_seconds=10,
        now=1000,
    )

    assert artifacts.root == real_root.resolve()
    assert source.parent == real_root.resolve() / job_id
    assert recovered == (job_id,)
    assert repository.get_job(job_id)["status"] == "paused"


@pytest.mark.asyncio
async def test_background_preclaim_failure_becomes_terminal_public_failure(tmp_path: Path) -> None:
    app, repository, _, _ = build_harness(tmp_path, runner_type=PreClaimFailRunner)
    async with client_for(app) as client:
        created = await upload(client)
        status = await client.get(created.json()["status_url"])

    assert created.status_code == 202
    assert status.status_code == 200
    assert status.json()["status"] == "failed"
    failed = [stage for stage in status.json()["stages"].values() if stage["status"] == "failed"]
    assert failed == [
        {
            "status": "failed",
            "error": {"code": "pipeline_failed", "message": "The pipeline stage failed."},
        }
    ]
    assert all(stage["status"] != "running" for stage in status.json()["stages"].values())
    assert "private" not in status.text
    assert str(tmp_path) not in status.text
    assert repository.get_job(created.json()["job_id"])["status"] == "failed"


@pytest.mark.asyncio
async def test_role_schema_and_runner_validation_then_completion_report_and_retry(
    tmp_path: Path,
) -> None:
    app, _, _, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        created = await upload(client)
        job_id = created.json()["job_id"]

        extra = await client.post(
            f"/api/jobs/{job_id}/roles",
            json={
                "mapping": {"SPEAKER_00": "customer", "SPEAKER_01": "agent"},
                "unexpected": True,
            },
        )
        duplicate = await client.post(
            f"/api/jobs/{job_id}/roles",
            json={"mapping": {"SPEAKER_00": "customer", "SPEAKER_01": "customer"}},
        )
        wrong_speakers = await client.post(
            f"/api/jobs/{job_id}/roles",
            json={"mapping": {"OTHER_00": "customer", "OTHER_01": "agent"}},
        )
        completed = await client.post(
            f"/api/jobs/{job_id}/roles",
            json={"mapping": {"SPEAKER_00": "customer", "SPEAKER_01": "agent"}},
        )
        report = await client.get(f"/api/jobs/{job_id}/report")
        bad_retry = await client.post(f"/api/jobs/{job_id}/retry", json={"stage": "private-stage"})
        retried = await client.post(f"/api/jobs/{job_id}/retry", json={"stage": "report"})

    assert extra.status_code == 422
    assert duplicate.status_code == 422
    assert wrong_speakers.status_code == 422
    assert wrong_speakers.json()["detail"]["code"] == "invalid_role_mapping"
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert_no_private_paths(completed.json(), tmp_path)
    assert report.status_code == 200
    assert report.json()["job_id"] == job_id
    assert len(report.json()["emotions"]) == 3
    assert bad_retry.status_code == 422
    assert retried.status_code == 200
    assert retried.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_role_state_conflict_and_incomplete_report_are_409(tmp_path: Path) -> None:
    app, repository, artifacts, runner = build_harness(tmp_path)
    job_id = repository.create_job(str(FIXTURE))
    artifacts.job_dir(job_id)
    async with client_for(app) as client:
        role = await client.post(
            f"/api/jobs/{job_id}/roles",
            json={"mapping": {"SPEAKER_00": "customer", "SPEAKER_01": "agent"}},
        )
        report = await client.get(f"/api/jobs/{job_id}/report")

    assert runner is not None
    assert role.status_code == 409
    assert report.status_code == 409
    assert_no_private_paths(role.json(), tmp_path)


@pytest.mark.asyncio
async def test_audio_full_and_all_single_range_forms_without_path_disclosure(
    tmp_path: Path,
) -> None:
    app, _, _, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        created = await upload(client)
        job_id = created.json()["job_id"]
        full = await client.get(f"/api/jobs/{job_id}/audio")
        bounded = await client.get(f"/api/jobs/{job_id}/audio", headers={"Range": "bytes=0-1023"})
        open_ended = await client.get(f"/api/jobs/{job_id}/audio", headers={"Range": "bytes=1024-"})
        suffix = await client.get(f"/api/jobs/{job_id}/audio", headers={"Range": "bytes=-128"})

    size = len(full.content)
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-length"] == str(size)
    assert "content-range" not in full.headers
    assert bounded.status_code == 206
    assert bounded.content == full.content[:1024]
    assert bounded.headers["content-range"] == f"bytes 0-1023/{size}"
    assert bounded.headers["content-length"] == "1024"
    assert open_ended.status_code == 206
    assert open_ended.content == full.content[1024:]
    assert suffix.status_code == 206
    assert suffix.content == full.content[-128:]
    for response in (full, bounded, open_ended, suffix):
        assert str(tmp_path) not in json.dumps(dict(response.headers))
        assert "content-disposition" not in response.headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("range_header", "expected_status", "expected_content"),
    [
        ("Bytes=0-1", 206, None),
        ("bytes= 0-1", 206, None),
        ("items=0-1", 200, "full"),
    ],
)
async def test_audio_range_unit_is_case_insensitive_with_ows_and_unknown_units_ignored(
    range_header: str,
    expected_status: int,
    expected_content: str | None,
    tmp_path: Path,
) -> None:
    app, _, _, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        created = await upload(client)
        job_id = created.json()["job_id"]
        full = await client.get(f"/api/jobs/{job_id}/audio")
        response = await client.get(
            f"/api/jobs/{job_id}/audio",
            headers={"Range": range_header},
        )

    assert response.status_code == expected_status
    assert response.content == (full.content if expected_content == "full" else full.content[:2])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "range_header",
    [
        "bytes=999999999-",
        "bytes=100-99",
        "bytes=0-1,2-3",
        "bytes=",
        "bytes=-0",
        "bytes=a-b",
        "bytes=--",
    ],
)
async def test_audio_invalid_or_unsatisfiable_ranges_return_416(
    range_header: str, tmp_path: Path
) -> None:
    app, _, _, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        created = await upload(client)
        job_id = created.json()["job_id"]
        full = await client.get(f"/api/jobs/{job_id}/audio")
        response = await client.get(f"/api/jobs/{job_id}/audio", headers={"Range": range_header})

    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{len(full.content)}"
    assert response.headers["accept-ranges"] == "bytes"
    assert str(tmp_path) not in response.text


def seed_empty_normalize(
    repository: JobRepository, artifacts: ArtifactStore, runner: PipelineRunner
) -> str:
    job_id = repository.create_job("empty.wav")
    generation = artifacts.job_dir(job_id) / "audio-empty"
    generation.mkdir()
    preview = generation / "mixed.wav"
    preview.write_bytes(b"")
    digest = hashlib.sha256(b"").hexdigest()
    cache_key = cache_key_for_stage(StageName.NORMALIZE, (), None, {"channel_preference": "auto"})
    reference = MediaReference(path=str(preview), sha256=digest)
    artifact = NormalizeArtifact(
        cache_key=cache_key,
        asset=AudioAsset(
            source_name="empty.wav",
            source_path="empty.wav",
            normalized_paths=(str(preview),),
            channel_mode="mixed",
            duration_seconds=0,
            channels=1,
            sha256=digest,
        ),
        normalized_media=(reference,),
        mixed_preview=reference,
    )
    path = artifacts.write_model(job_id, StageName.NORMALIZE, artifact)
    repository.set_stage(
        job_id,
        StageName.NORMALIZE,
        StageStatus.COMPLETED,
        path.name,
        cache_key=cache_key,
        artifact_hash=artifacts.content_hash(job_id, StageName.NORMALIZE),
    )
    assert runner is not None
    return job_id


@pytest.mark.asyncio
async def test_empty_valid_audio_is_streamed_and_any_range_is_416(tmp_path: Path) -> None:
    app, repository, artifacts, runner = build_harness(tmp_path)
    job_id = seed_empty_normalize(repository, artifacts, runner)
    async with client_for(app) as client:
        full = await client.get(f"/api/jobs/{job_id}/audio")
        ranged = await client.get(f"/api/jobs/{job_id}/audio", headers={"Range": "bytes=0-"})

    assert full.status_code == 200
    assert full.content == b""
    assert full.headers["content-length"] == "0"
    assert ranged.status_code == 416
    assert ranged.headers["content-range"] == "bytes */0"


@pytest.mark.asyncio
async def test_audio_rejects_symlink_or_hash_tampering(tmp_path: Path) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        created = await upload(client)
        job_id = created.json()["job_id"]
        normalized = artifacts.read_model(job_id, StageName.NORMALIZE, NormalizeArtifact)
        preview = Path(normalized.mixed_preview.path)
        preview.write_bytes(b"tampered")
        response = await client.get(f"/api/jobs/{job_id}/audio")

    assert repository.get_job(job_id)["stages"]["normalize"]["status"] == "completed"
    assert response.status_code == 409
    assert str(preview) not in response.text


def audio_endpoint(app: FastAPI):  # type: ignore[no-untyped-def]
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/api/jobs/{job_id}/audio"
    )


def audio_request(job_id: str) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": f"/api/jobs/{job_id}/audio",
            "raw_path": f"/api/jobs/{job_id}/audio".encode(),
            "query_string": b"",
            "headers": [],
            "client": ("test", 1),
            "server": ("test", 80),
        }
    )


@pytest.mark.asyncio
async def test_audio_response_background_closes_unstarted_body_once(tmp_path: Path) -> None:
    app, _, _, runner = build_harness(tmp_path)
    async with client_for(app) as client:
        created = await upload(client)
    job_id = created.json()["job_id"]
    original = runner.open_mixed_preview
    closes = 0

    @contextmanager
    def tracking_open(selected_job_id: str):  # type: ignore[no-untyped-def]
        nonlocal closes
        with original(selected_job_id) as opened:
            try:
                yield opened
            finally:
                closes += 1

    runner.open_mixed_preview = tracking_open  # type: ignore[method-assign]
    response = audio_endpoint(app)(job_id, audio_request(job_id))

    assert isinstance(response, StreamingResponse)
    assert closes == 0
    assert response.background is not None
    await response.background()
    await response.background()
    assert closes == 1


@pytest.mark.asyncio
async def test_audio_cancelled_body_and_background_do_not_double_close(tmp_path: Path) -> None:
    app, _, _, runner = build_harness(tmp_path)
    async with client_for(app) as client:
        created = await upload(client)
    job_id = created.json()["job_id"]
    original = runner.open_mixed_preview
    closes = 0

    @contextmanager
    def tracking_open(selected_job_id: str):  # type: ignore[no-untyped-def]
        nonlocal closes
        with original(selected_job_id) as opened:
            try:
                yield opened
            finally:
                closes += 1

    runner.open_mixed_preview = tracking_open  # type: ignore[method-assign]
    response = audio_endpoint(app)(job_id, audio_request(job_id))
    first = await anext(response.body_iterator)
    assert first
    await response.body_iterator.aclose()
    assert response.background is not None
    await response.background()

    assert closes == 1


@pytest.mark.asyncio
async def test_audio_send_disconnect_closes_descriptor_without_manual_cleanup(
    tmp_path: Path,
) -> None:
    app, _, _, runner = build_harness(tmp_path)
    async with client_for(app) as client:
        created = await upload(client)
    job_id = created.json()["job_id"]
    original = runner.open_mixed_preview
    closes = 0

    @contextmanager
    def tracking_open(selected_job_id: str):  # type: ignore[no-untyped-def]
        nonlocal closes
        with original(selected_job_id) as opened:
            try:
                yield opened
            finally:
                closes += 1

    runner.open_mixed_preview = tracking_open  # type: ignore[method-assign]
    response = audio_endpoint(app)(job_id, audio_request(job_id))

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def disconnected_send(message: dict[str, object]) -> None:
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": f"/api/jobs/{job_id}/audio",
        "raw_path": f"/api/jobs/{job_id}/audio".encode("ascii"),
        "query_string": b"",
        "headers": [],
        "client": ("test", 1),
        "server": ("test", 80),
    }
    with pytest.raises(ClientDisconnect):
        await response(scope, receive, disconnected_send)  # type: ignore[arg-type]

    assert closes == 1


@pytest.mark.asyncio
async def test_audio_accessor_hashes_only_the_opened_mixed_preview_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, artifacts, runner = build_harness(tmp_path)
    async with client_for(app) as client:
        created = await upload(client)
    job_id = created.json()["job_id"]
    normalized = artifacts.read_model(job_id, StageName.NORMALIZE, NormalizeArtifact)
    original = runner._open_trusted_media  # noqa: SLF001
    opened_paths: list[str] = []

    @contextmanager
    def counting_open(selected_job_id: str, raw_path: str):  # type: ignore[no-untyped-def]
        opened_paths.append(raw_path)
        with original(selected_job_id, raw_path) as opened:
            yield opened

    monkeypatch.setattr(runner, "_open_trusted_media", counting_open)
    with runner.open_mixed_preview(job_id) as opened:
        assert opened.read(1)

    assert opened_paths == [normalized.mixed_preview.path]


@pytest.mark.asyncio
async def test_missing_and_traversal_jobs_are_safe_404s(tmp_path: Path) -> None:
    app, _, _, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        responses = [
            await client.get("/api/jobs/missing"),
            await client.get("/api/jobs/%2E%2E"),
            await client.get("/api/jobs/C%3A"),
            await client.delete("/api/jobs/missing"),
        ]

    assert all(response.status_code == 404 for response in responses)
    for response in responses:
        assert "traceback" not in response.text.casefold()
        assert str(tmp_path) not in response.text


@pytest.mark.asyncio
async def test_delete_removes_exact_job_database_and_artifacts(tmp_path: Path) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        first = await upload(client)
        second = await upload(client)
        first_id = first.json()["job_id"]
        second_id = second.json()["job_id"]
        deleted = await client.delete(f"/api/jobs/{first_id}")
        missing = await client.get(f"/api/jobs/{first_id}")

    assert deleted.status_code == 204
    assert deleted.content == b""
    assert missing.status_code == 404
    with pytest.raises(KeyError):
        repository.get_job(first_id)
    assert not (artifacts.root / first_id).exists()
    assert repository.get_job(second_id)["id"] == second_id
    assert (artifacts.root / second_id).is_dir()


@pytest.mark.asyncio
async def test_delete_rmtree_failure_is_retryable_without_recreation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path)
    real_rmtree = artifacts_module.shutil.rmtree
    attempts = 0

    def fail_once(path: str | os.PathLike[str], *args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("private rmtree path")
        real_rmtree(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(artifacts_module.shutil, "rmtree", fail_once)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://localhost",
        headers={"X-VoxDelta-Token": TEST_CAPABILITY_TOKEN},
    ) as client:
        created = await upload(client)
        job_id = created.json()["job_id"]
        first = await client.delete(f"/api/jobs/{job_id}")
        persisted = repository.get_job(job_id)
        second = await client.delete(f"/api/jobs/{job_id}")

    assert first.status_code == 409
    assert first.json()["detail"]["code"] == "deletion_incomplete"
    assert str(tmp_path) not in first.text
    assert persisted["status"] == "deleting"
    assert all(row["status"] == "deleting" for row in persisted["stages"].values())
    assert all(int(row["generation"]) >= 1 for row in persisted["stages"].values())
    assert all(row["claim_token"] is None for row in persisted["stages"].values())
    assert second.status_code == 204
    with pytest.raises(KeyError):
        repository.get_job(job_id)
    assert not (artifacts.root / job_id).exists()
    assert (artifacts.root / ".deleted" / f"{job_id}.tombstone").is_file()


@pytest.mark.asyncio
async def test_delete_database_finalization_failure_is_retryable(
    tmp_path: Path,
) -> None:
    repository = FailOnceFinalizeRepository(tmp_path / "voxdelta.sqlite3")
    app, _, artifacts, _ = build_harness(tmp_path, repository_override=repository)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://localhost",
        headers={"X-VoxDelta-Token": TEST_CAPABILITY_TOKEN},
    ) as client:
        created = await upload(client)
        job_id = created.json()["job_id"]
        first = await client.delete(f"/api/jobs/{job_id}")
        persisted = repository.get_job(job_id)
        second = await client.delete(f"/api/jobs/{job_id}")

    assert first.status_code == 409
    assert first.json()["detail"]["code"] == "deletion_incomplete"
    assert persisted["status"] == "deleting"
    assert not (artifacts.root / job_id).exists()
    assert second.status_code == 204
    with pytest.raises(KeyError):
        repository.get_job(job_id)


@pytest.mark.asyncio
async def test_retry_of_deleting_job_is_a_conflict_not_not_found(tmp_path: Path) -> None:
    app, repository, _, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        created = await upload(client)
        job_id = created.json()["job_id"]
        repository.begin_delete(job_id)
        response = await client.post(
            f"/api/jobs/{job_id}/retry",
            json={"stage": "diarize"},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "job_deleting",
        "message": "The job is being deleted and cannot be retried.",
    }


@pytest.mark.asyncio
async def test_retry_delete_race_from_repository_value_error_remains_http_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, repository, _, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        created = await upload(client)
        job_id = created.json()["job_id"]
        original_invalidate = repository.invalidate_stages

        def deleting_invalidate(selected_job_id: str, stages: tuple[StageName, ...]) -> None:
            repository.begin_delete(selected_job_id)
            original_invalidate(selected_job_id, stages)

        monkeypatch.setattr(repository, "invalidate_stages", deleting_invalidate)
        response = await client.post(
            f"/api/jobs/{job_id}/retry",
            json={"stage": "diarize"},
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "job_deleting"


@pytest.mark.asyncio
async def test_retry_completed_delete_after_locked_recheck_remains_http_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path)
    async with client_for(app) as client:
        created = await upload(client)
        job_id = created.json()["job_id"]

        def completing_delete(
            selected_job_id: str,
            stages: tuple[StageName, ...],
        ) -> None:
            del stages
            repository.begin_delete(selected_job_id)
            artifacts.delete_job(selected_job_id)
            repository.finalize_delete(selected_job_id)
            raise KeyError(selected_job_id)

        monkeypatch.setattr(repository, "invalidate_stages", completing_delete)
        response = await client.post(
            f"/api/jobs/{job_id}/retry",
            json={"stage": "diarize"},
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "job_deleting"
    with pytest.raises(KeyError):
        repository.get_job(job_id)
    assert (artifacts.root / ".deleted" / f"{job_id}.tombstone").is_file()


@pytest.mark.asyncio
async def test_delete_syncs_artifact_root_before_database_finalization_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, repository, artifacts, _ = build_harness(tmp_path)
    real_fsync_directory = artifacts_module._fsync_directory
    real_finalize = repository.finalize_delete
    events: list[str] = []
    root_failures = 0
    selected_job_id = ""

    def fail_first_post_rmtree_root_sync(directory: Path) -> None:
        nonlocal root_failures
        if directory == artifacts.root and not (artifacts.root / selected_job_id).exists():
            events.append("root_fsync")
            root_failures += 1
            if root_failures == 1:
                raise OSError("private artifact root fsync")
        real_fsync_directory(directory)

    def tracking_finalize(job_id: str) -> None:
        events.append("finalize")
        real_finalize(job_id)

    monkeypatch.setattr(artifacts_module, "_fsync_directory", fail_first_post_rmtree_root_sync)
    monkeypatch.setattr(repository, "finalize_delete", tracking_finalize)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://localhost",
        headers={"X-VoxDelta-Token": TEST_CAPABILITY_TOKEN},
    ) as client:
        created = await upload(client)
        selected_job_id = created.json()["job_id"]
        first = await client.delete(f"/api/jobs/{selected_job_id}")
        after_first = repository.get_job(selected_job_id)
        second = await client.delete(f"/api/jobs/{selected_job_id}")

    assert first.status_code == 409
    assert first.json()["detail"]["code"] == "deletion_incomplete"
    assert after_first["status"] == "deleting"
    assert events == ["root_fsync", "root_fsync", "finalize"]
    assert second.status_code == 204
    with pytest.raises(KeyError):
        repository.get_job(selected_job_id)


@pytest.mark.asyncio
async def test_background_unexpected_exception_is_contained_and_sanitized(tmp_path: Path) -> None:
    app, _, _, _ = build_harness(tmp_path, diarizer=UnexpectedDiarizer())
    async with client_for(app) as client:
        created = await upload(client)
        status = await client.get(created.json()["status_url"])

    assert created.status_code == 202
    assert status.status_code == 200
    assert status.json()["status"] == "failed"
    assert status.json()["stages"]["diarize"]["error"] == {
        "code": "pipeline_failed",
        "message": "The pipeline stage failed.",
    }
    assert "private" not in status.text
    assert str(tmp_path) not in status.text


@pytest.mark.asyncio
async def test_delete_fences_inflight_worker_from_a_separate_runner(
    tmp_path: Path,
) -> None:
    diarizer = BlockingDiarizer()
    app, repository, artifacts, _ = build_harness(tmp_path, diarizer=diarizer)
    deleting_runner = PipelineRunner(
        repository,
        artifacts,
        AudioService(artifacts.root, 60, 3600),
    )
    deleting_app = create_app(
        repository=repository,
        artifacts=artifacts,
        runner=deleting_runner,
        api_capability_token=SecretStr(TEST_CAPABILITY_TOKEN),
    )
    async with client_for(app) as client, client_for(deleting_app) as deleting_client:
        upload_task = __import__("asyncio").create_task(upload(client))
        assert await __import__("asyncio").to_thread(diarizer.entered.wait, 5)
        with sqlite3.connect(repository.path) as database:
            job_id = str(database.execute("SELECT id FROM jobs").fetchone()[0])
        deleted = await deleting_client.delete(f"/api/jobs/{job_id}")
        diarizer.release.set()
        created = await upload_task

    assert created.status_code == 202
    assert deleted.status_code == 204
    with pytest.raises(KeyError):
        repository.get_job(job_id)
    assert not (artifacts.root / job_id).exists()
    assert not any(path.name.startswith(".confirm_roles") for path in artifacts.root.rglob("*"))


def test_streamed_upload_chunk_constant_is_bounded() -> None:
    from voxdelta.api.app import UPLOAD_CHUNK_BYTES

    assert 0 < UPLOAD_CHUNK_BYTES <= 1024 * 1024


def test_audio_stream_descriptor_is_regular_non_link_and_mode_is_private(tmp_path: Path) -> None:
    app, repository, artifacts, runner = build_harness(tmp_path)
    del app
    job_id = repository.create_job(str(FIXTURE))
    directory = artifacts.job_dir(job_id)
    source = directory / "source-upload-test.wav"
    source.write_bytes(FIXTURE.read_bytes())
    source.chmod(0o600)
    repository.update_source_name(job_id, str(source))
    runner.run_until_pause(job_id)

    with runner.open_mixed_preview(job_id) as opened:
        metadata = os.fstat(opened.fileno())
        assert stat.S_ISREG(metadata.st_mode)
        assert metadata.st_mode & 0o777 == 0o600
        assert opened.read(32)
