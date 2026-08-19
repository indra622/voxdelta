from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest

from voxdelta.audio.service import AudioService
from voxdelta.domain.models import AudioAsset, Role, StageName, StageStatus
from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.repository import JobRepository
from voxdelta.pipeline.runner import PipelineRunner, PipelineValidationError
from voxdelta.pipeline.stages import (
    EmotionArtifact,
    NormalizeArtifact,
    ReportArtifact,
    RoleArtifact,
    StrategyArtifact,
    TransitionsArtifact,
    cache_key_for_stage,
)
from voxdelta.providers.fake import (
    FakeDiarizationProvider,
    FakeEmotionProvider,
    FakeReportSummaryProvider,
    FakeResponseStrategyProvider,
    FakeTranscriptionProvider,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_65s.wav"


class CountingDiarizer(FakeDiarizationProvider):
    def __init__(self) -> None:
        self.calls = 0

    def diarize(self, asset: AudioAsset):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().diarize(asset)


class CountingEmotion(FakeEmotionProvider):
    def __init__(self) -> None:
        self.utterance_ids: list[str] = []

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str):  # type: ignore[no-untyped-def]
        self.utterance_ids.append(utterance_id)
        return super().analyze(utterance_id, audio_path, transcript)


class PrivateValidationDiarizer(FakeDiarizationProvider):
    def diarize(self, asset: AudioAsset):  # type: ignore[no-untyped-def]
        raise ValueError("private transcript and secret-token")


class PrivateUnexpectedDiarizer(FakeDiarizationProvider):
    def diarize(self, asset: AudioAsset):  # type: ignore[no-untyped-def]
        raise RuntimeError("private transcript and secret-token")


def _harness(
    tmp_path: Path,
    *,
    diarizer: FakeDiarizationProvider | None = None,
    emotion: FakeEmotionProvider | None = None,
) -> tuple[PipelineRunner, JobRepository, ArtifactStore, str]:
    jobs_root = tmp_path / "jobs"
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")
    store = ArtifactStore(jobs_root)
    job_id = repository.create_job(str(FIXTURE))
    asset = AudioAsset(
        source_name=FIXTURE.name,
        source_path=str(FIXTURE),
        normalized_paths=(str(FIXTURE),),
        channel_mode="mixed",
        duration_seconds=65.0,
        channels=1,
        sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
    )
    cache_key = cache_key_for_stage(
        StageName.NORMALIZE,
        (),
        None,
        {"channel_preference": "auto"},
    )
    artifact = NormalizeArtifact(cache_key=cache_key, asset=asset)
    path = store.write_model(job_id, StageName.NORMALIZE, artifact)
    repository.set_stage(
        job_id,
        StageName.NORMALIZE,
        StageStatus.COMPLETED,
        path.name,
        cache_key=cache_key,
    )
    runner = PipelineRunner(
        repository,
        store,
        AudioService(jobs_root, 60, 3600),
        diarization_provider=diarizer or FakeDiarizationProvider(),
        transcription_provider=FakeTranscriptionProvider(),
        emotion_provider=emotion or FakeEmotionProvider(),
        strategy_provider=FakeResponseStrategyProvider(),
        report_provider=FakeReportSummaryProvider(),
    )
    return runner, repository, store, job_id


def _confirm(runner: PipelineRunner, job_id: str) -> dict[str, object]:
    return runner.confirm_roles(
        job_id,
        {"SPEAKER_00": Role.CUSTOMER, "SPEAKER_01": Role.AGENT},
    )


def test_seeded_fake_pipeline_pauses_then_finishes_with_aligned_outputs(tmp_path: Path) -> None:
    emotion = CountingEmotion()
    runner, repository, store, job_id = _harness(tmp_path, emotion=emotion)

    paused = runner.run_until_pause(job_id)

    assert paused["status"] == "paused"
    assert paused["stages"]["confirm_roles"]["status"] == "paused"
    role_candidate = store.read_model(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
    assert role_candidate.confirmed is False
    assert {utterance.speaker_id for utterance in role_candidate.utterances} == {
        "SPEAKER_00",
        "SPEAKER_01",
    }
    assert not store.artifact_path(job_id, StageName.EMOTION).exists()
    assert emotion.utterance_ids == []

    completed = _confirm(runner, job_id)

    assert completed["status"] == "completed"
    assert completed["stages"]["report"]["status"] == "completed"
    role = store.read_model(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
    emotions = store.read_model(job_id, StageName.EMOTION, EmotionArtifact)
    strategies = store.read_model(job_id, StageName.RESPONSE_STRATEGY, StrategyArtifact)
    transitions = store.read_model(job_id, StageName.TRANSITIONS, TransitionsArtifact)
    report = store.read_model(job_id, StageName.REPORT, ReportArtifact)
    customer_ids = [u.id for u in role.utterances if u.role == Role.CUSTOMER]
    assert emotion.utterance_ids == customer_ids
    assert [result.utterance_id for result in emotions.results] == customer_ids
    assert all(result.smoothed_negative_intensity is not None for result in emotions.results)
    assert {result.utterance_id for result in strategies.results} == {
        transition.agent_id for transition in transitions.results
    }
    assert report.report.utterances == role.utterances
    assert report.report.emotions == emotions.results
    assert report.report.strategies == strategies.results
    assert report.report.transitions == transitions.results
    for stage in StageName:
        raw = json.loads(store.artifact_path(job_id, stage).read_bytes())
        assert raw["schema_version"] == "1"
        assert len(raw["cache_key"]) == 64
        assert "provider" in raw


@pytest.mark.parametrize(
    "mapping",
    [
        {"SPEAKER_00": Role.CUSTOMER},
        {
            "SPEAKER_00": Role.CUSTOMER,
            "SPEAKER_01": Role.AGENT,
            "SPEAKER_02": Role.AGENT,
        },
        {"SPEAKER_00": Role.CUSTOMER, "UNKNOWN": Role.AGENT},
        {"SPEAKER_00": Role.CUSTOMER, "SPEAKER_01": Role.CUSTOMER},
        {"SPEAKER_00": Role.CUSTOMER, "SPEAKER_01": Role.UNKNOWN},
    ],
)
def test_role_confirmation_requires_exact_observed_bijection(
    mapping: dict[str, Role], tmp_path: Path
) -> None:
    runner, repository, store, job_id = _harness(tmp_path)
    runner.run_until_pause(job_id)

    with pytest.raises(PipelineValidationError) as raised:
        runner.confirm_roles(job_id, mapping)

    assert raised.value.code == "invalid_role_mapping"
    assert repository.get_job(job_id)["stages"]["confirm_roles"]["status"] == "paused"
    assert store.read_model(job_id, StageName.CONFIRM_ROLES, RoleArtifact).confirmed is False
    assert not store.artifact_path(job_id, StageName.EMOTION).exists()


def test_completed_candidate_cannot_bypass_explicit_role_confirmation(tmp_path: Path) -> None:
    runner, repository, store, job_id = _harness(tmp_path)
    runner.run_until_pause(job_id)
    candidate = store.read_model(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
    repository.set_stage(
        job_id,
        StageName.CONFIRM_ROLES,
        StageStatus.COMPLETED,
        store.artifact_path(job_id, StageName.CONFIRM_ROLES).name,
        cache_key=candidate.cache_key,
    )

    result = runner.run_until_pause(job_id)

    assert result["stages"]["confirm_roles"]["status"] == "paused"
    assert store.read_model(job_id, StageName.CONFIRM_ROLES, RoleArtifact).confirmed is False
    assert not store.artifact_path(job_id, StageName.EMOTION).exists()


def test_completed_pipeline_is_idempotent_and_concurrent_runs_execute_provider_once(
    tmp_path: Path,
) -> None:
    diarizer = CountingDiarizer()
    runner, _, _, job_id = _harness(tmp_path, diarizer=diarizer)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: runner.run_until_pause(job_id), range(2)))

    assert [result["status"] for result in results] == ["paused", "paused"]
    assert diarizer.calls == 1
    _confirm(runner, job_id)
    calls_after_completion = diarizer.calls

    assert runner.run_until_pause(job_id)["status"] == "completed"
    assert diarizer.calls == calls_after_completion


def test_corrupt_completed_artifact_invalidates_it_and_all_downstream(tmp_path: Path) -> None:
    runner, repository, store, job_id = _harness(tmp_path)
    runner.run_until_pause(job_id)
    _confirm(runner, job_id)
    store.artifact_path(job_id, StageName.TRANSCRIBE).write_bytes(b"{corrupt")

    result = runner.run_until_pause(job_id)

    assert result["stages"]["confirm_roles"]["status"] == "paused"
    assert store.read_model(job_id, StageName.CONFIRM_ROLES, RoleArtifact).confirmed is False
    for stage in (
        StageName.EMOTION,
        StageName.RESPONSE_STRATEGY,
        StageName.TRANSITIONS,
        StageName.REPORT,
    ):
        assert repository.get_job(job_id)["stages"][stage.value]["status"] == "pending"
        assert not store.artifact_path(job_id, stage).exists()


def test_retry_transcribe_preserves_upstream_audio_and_reaches_role_pause(tmp_path: Path) -> None:
    runner, repository, store, job_id = _harness(tmp_path)
    runner.run_until_pause(job_id)
    _confirm(runner, job_id)
    normalized_generation = store.job_dir(job_id) / "audio-existing" / "mixed.wav"
    normalized_generation.parent.mkdir()
    normalized_generation.write_bytes(b"keep normalized generation")
    source_upload = store.job_dir(job_id) / "source-upload.wav"
    source_upload.write_bytes(b"keep source upload")

    result = runner.retry(job_id, StageName.TRANSCRIBE)

    assert result["stages"]["confirm_roles"]["status"] == "paused"
    assert store.artifact_path(job_id, StageName.NORMALIZE).exists()
    assert store.artifact_path(job_id, StageName.DIARIZE).exists()
    assert store.artifact_path(job_id, StageName.TRANSCRIBE).exists()
    assert normalized_generation.read_bytes() == b"keep normalized generation"
    assert source_upload.read_bytes() == b"keep source upload"
    for stage in (
        StageName.EMOTION,
        StageName.RESPONSE_STRATEGY,
        StageName.TRANSITIONS,
        StageName.REPORT,
    ):
        assert not store.artifact_path(job_id, stage).exists()
        assert repository.get_job(job_id)["stages"][stage.value]["status"] == "pending"


def test_retry_normalize_never_deletes_original_source(tmp_path: Path) -> None:
    runner, repository, store, job_id = _harness(tmp_path)
    source = store.job_dir(job_id) / "source-upload.wav"
    shutil.copyfile(FIXTURE, source)
    normalize = store.read_model(job_id, StageName.NORMALIZE, NormalizeArtifact)
    replacement = normalize.model_copy(
        update={"asset": normalize.asset.model_copy(update={"source_path": str(source)})}
    )
    store.write_model(job_id, StageName.NORMALIZE, replacement)
    repository.set_stage(
        job_id,
        StageName.NORMALIZE,
        StageStatus.COMPLETED,
        store.artifact_path(job_id, StageName.NORMALIZE).name,
        cache_key=replacement.cache_key,
    )

    result = runner.retry(job_id, StageName.NORMALIZE)

    assert result["stages"]["confirm_roles"]["status"] == "paused"
    assert source.exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == normalize.asset.sha256


def test_retry_rejects_invalid_stage_and_unknown_job_without_mutation(tmp_path: Path) -> None:
    runner, repository, store, job_id = _harness(tmp_path)
    before = repository.get_job(job_id)

    with pytest.raises(PipelineValidationError) as raised:
        runner.retry(job_id, cast(StageName, "unknown"))
    assert raised.value.code == "invalid_stage"
    with pytest.raises(KeyError, match="missing"):
        runner.retry("missing", StageName.REPORT)

    assert repository.get_job(job_id) == before
    assert store.artifact_path(job_id, StageName.NORMALIZE).exists()


def test_known_validation_failure_is_public_and_leaves_no_running_state(tmp_path: Path) -> None:
    runner, repository, _, job_id = _harness(tmp_path, diarizer=PrivateValidationDiarizer())

    result = runner.run_until_pause(job_id)

    row = result["stages"]["diarize"]
    error = json.loads(row["error_json"])
    assert row["status"] == "failed"
    assert error == {
        "code": "invalid_stage_output",
        "message": "The diarize stage produced invalid data.",
    }
    assert "private" not in row["error_json"]
    assert repository.get_job(job_id)["status"] == "failed"


def test_unexpected_failure_records_only_exception_class_and_reraises(tmp_path: Path) -> None:
    runner, repository, _, job_id = _harness(tmp_path, diarizer=PrivateUnexpectedDiarizer())

    with pytest.raises(RuntimeError, match="private transcript"):
        runner.run_until_pause(job_id)

    row = repository.get_job(job_id)["stages"]["diarize"]
    assert row["status"] == "failed"
    assert json.loads(row["error_json"]) == {
        "code": "RuntimeError",
        "message": "An unexpected pipeline error occurred.",
    }
    assert "private" not in row["error_json"]
