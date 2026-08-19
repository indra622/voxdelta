from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from voxdelta.api.app import create_app
from voxdelta.audio.service import AudioService
from voxdelta.domain.models import AudioAsset, StageName, StageStatus
from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.repository import JobRepository
from voxdelta.pipeline.runner import PipelineRunner
from voxdelta.pipeline.stages import MediaReference, NormalizeArtifact, cache_key_for_stage
from voxdelta.providers.fake import FakeDiarizationProvider

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_65s.wav"


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


class FailingDeleteStore(ArtifactStore):
    def delete_job(self, job_id: str) -> None:
        del job_id
        raise OSError("private artifact cleanup path")


class FailingDeleteRepository(JobRepository):
    def delete_job(self, job_id: str) -> None:
        del job_id
        raise RuntimeError("private database cleanup row")


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
        ),
        repository,
        artifacts,
        runner,
    )


@asynccontextmanager
async def client_for(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
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
    assert source.name.startswith("source-upload-")
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
        oversized = await client.post(
            "/api/jobs",
            files={"file": ("call.wav", b"123456789", "audio/wav")},
        )
        empty = await client.post(
            "/api/jobs",
            files={"file": ("call.wav", b"", "audio/wav")},
        )

    assert oversized.status_code == 413
    assert oversized.json()["detail"]["code"] == "upload_too_large"
    assert empty.status_code == 422
    assert empty.json()["detail"]["code"] == "empty_upload"
    assert job_count(repository) == 0
    assert not artifacts.root.exists() or not list(artifacts.root.glob("[!.]*"))
    assert str(tmp_path) not in oversized.text + empty.text


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
    assert "private" not in response.text
    assert str(tmp_path) not in response.text


@pytest.mark.asyncio
async def test_upload_cleanup_continues_after_source_unlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_unlink = Path.unlink

    def failing_source_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.startswith("source-upload-"):
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
async def test_upload_cleanup_continues_after_artifact_delete_failure(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    artifacts = FailingDeleteStore(jobs_root)
    app, repository, _, _ = build_harness(
        tmp_path,
        max_upload_bytes=1,
        artifacts_override=artifacts,
    )
    async with client_for(app) as client:
        response = await client.post(
            "/api/jobs",
            files={"file": ("call.wav", b"12", "audio/wav")},
        )

    assert response.status_code == 413
    assert job_count(repository) == 0
    assert not list(jobs_root.rglob("source-upload-*"))


@pytest.mark.asyncio
async def test_upload_cleanup_repository_delete_failure_falls_back_to_failed_state(
    tmp_path: Path,
) -> None:
    repository = FailingDeleteRepository(tmp_path / "voxdelta.sqlite3")
    app, _, artifacts, _ = build_harness(
        tmp_path,
        max_upload_bytes=1,
        repository_override=repository,
    )
    async with client_for(app) as client:
        response = await client.post(
            "/api/jobs",
            files={"file": ("call.wav", b"12", "audio/wav")},
        )

    assert response.status_code == 413
    with sqlite3.connect(repository.path) as database:
        assert database.execute("SELECT status FROM jobs").fetchone() == ("failed",)
    assert not list(artifacts.root.rglob("source-upload-*"))
    assert "private" not in response.text


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
