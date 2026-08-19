from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from voxdelta.audio.service import AudioService
from voxdelta.domain.models import (
    AnalysisReport,
    AudioAsset,
    EmotionResult,
    ResponseStrategyResult,
    Role,
    SpeakerSegment,
    StageName,
    StageStatus,
    Utterance,
)
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


class BlockingDiarizer(FakeDiarizationProvider):
    provenance = FakeDiarizationProvider.provenance.model_copy(update={"name": "blocked-old"})

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def diarize(self, asset: AudioAsset):  # type: ignore[no-untyped-def]
        self.entered.set()
        assert self.release.wait(timeout=5)
        return super().diarize(asset)


class ReversedDiarizer(FakeDiarizationProvider):
    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]:
        return list(reversed(super().diarize(asset)))


class ReversedTranscription(FakeTranscriptionProvider):
    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]:
        return list(reversed(super().transcribe(asset, segments)))


class WrongIdEmotion(FakeEmotionProvider):
    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        result = super().analyze(utterance_id, audio_path, transcript)
        return result.model_copy(update={"utterance_id": f"wrong-{utterance_id}"})


class WrongIdStrategy(FakeResponseStrategyProvider):
    def classify(self, utterance: Utterance, context: list[Utterance]) -> ResponseStrategyResult:
        result = super().classify(utterance, context)
        return result.model_copy(update={"utterance_id": f"wrong-{utterance.id}"})


class IntegerSummary(FakeReportSummaryProvider):
    def summarize(self, report: AnalysisReport) -> str:
        return cast(str, 7)


class ClipSpyEmotion(FakeEmotionProvider):
    def __init__(self) -> None:
        self.clips: list[tuple[Path, int, int, str]] = []

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        with wave.open(str(audio_path), "rb") as clip:
            frames = clip.readframes(clip.getnframes())
            self.clips.append(
                (
                    audio_path,
                    clip.getframerate(),
                    clip.getnframes(),
                    hashlib.sha256(frames).hexdigest(),
                )
            )
        return super().analyze(utterance_id, audio_path, transcript)


def _harness(
    tmp_path: Path,
    *,
    diarizer: FakeDiarizationProvider | None = None,
    transcription: FakeTranscriptionProvider | None = None,
    emotion: FakeEmotionProvider | None = None,
    strategy: FakeResponseStrategyProvider | None = None,
    report: FakeReportSummaryProvider | None = None,
    diagnostic_capture: bool = False,
    repository_override: JobRepository | None = None,
) -> tuple[PipelineRunner, JobRepository, ArtifactStore, str]:
    jobs_root = tmp_path / "jobs"
    repository = repository_override or JobRepository(tmp_path / "voxdelta.sqlite3")
    store = ArtifactStore(jobs_root)
    job_id = repository.create_job(str(FIXTURE), diagnostic_capture=diagnostic_capture)
    seeded_generation = store.job_dir(job_id) / "audio-seeded"
    seeded_generation.mkdir()
    seeded_mixed = seeded_generation / "mixed.wav"
    shutil.copyfile(FIXTURE, seeded_mixed)
    asset = AudioAsset(
        source_name=FIXTURE.name,
        source_path=str(FIXTURE),
        normalized_paths=(str(seeded_mixed),),
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
        artifact_hash=store.content_hash(job_id, StageName.NORMALIZE),
    )
    runner = PipelineRunner(
        repository,
        store,
        AudioService(jobs_root, 60, 3600),
        diarization_provider=diarizer or FakeDiarizationProvider(),
        transcription_provider=transcription or FakeTranscriptionProvider(),
        emotion_provider=emotion or FakeEmotionProvider(),
        strategy_provider=strategy or FakeResponseStrategyProvider(),
        report_provider=report or FakeReportSummaryProvider(),
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
    assert completed["stages"]["confirm_roles"]["role_confirmed"] == 1
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


def test_emotion_provider_receives_distinct_aligned_temporary_customer_clips(
    tmp_path: Path,
) -> None:
    emotion = ClipSpyEmotion()
    runner, _, store, job_id = _harness(tmp_path, emotion=emotion)
    runner.run_until_pause(job_id)

    _confirm(runner, job_id)

    assert len(emotion.clips) == 3
    clip_paths = [item[0] for item in emotion.clips]
    assert len(set(clip_paths)) == 3
    assert all(path.parent == store.job_dir(job_id) for path in clip_paths)
    assert all(path.name.startswith(".emotion-") for path in clip_paths)
    assert all(not path.exists() for path in clip_paths)
    assert {sample_rate for _, sample_rate, _, _ in emotion.clips} == {16_000}
    assert all(
        frame_count == pytest.approx(16_000 * 65 / 6, abs=1)
        for _, _, frame_count, _ in emotion.clips
    )
    assert len({digest for *_, digest in emotion.clips}) == 3


@pytest.mark.parametrize("enabled", [False, True])
def test_runner_diagnostics_follow_persisted_job_opt_in_and_remain_bounded(
    enabled: bool, tmp_path: Path
) -> None:
    runner, _, store, job_id = _harness(tmp_path, diagnostic_capture=enabled)
    runner.run_until_pause(job_id)
    _confirm(runner, job_id)

    directory = store.job_dir(job_id) / "diagnostics"
    if not enabled:
        assert not directory.exists()
        return

    files = list(directory.glob("*.json"))
    assert files
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in files)
    for private in (
        "안녕하세요",
        "문의 내용을",
        "source-upload",
        "api_key",
        "authorization",
        "payload",
    ):
        assert private not in serialized.casefold()


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
    mapping = {"SPEAKER_00": Role.CUSTOMER, "SPEAKER_01": Role.AGENT}
    forged = candidate.model_copy(
        update={
            "confirmed": True,
            "mapping": mapping,
            "utterances": [
                utterance.model_copy(update={"role": mapping[utterance.speaker_id]})
                for utterance in candidate.utterances
            ],
            "cache_key": cache_key_for_stage(
                StageName.CONFIRM_ROLES,
                (store.content_hash(job_id, StageName.TRANSCRIBE),),
                None,
                {
                    "confirmed": True,
                    "mapping": {key: value.value for key, value in sorted(mapping.items())},
                },
            ),
        }
    )
    store.write_model(job_id, StageName.CONFIRM_ROLES, forged)
    forged_hash = store.content_hash(job_id, StageName.CONFIRM_ROLES)
    with sqlite3.connect(repository.path) as database:
        database.execute(
            """
            UPDATE stages
            SET status = 'completed', artifact_path = 'confirm_roles.v1.json',
                cache_key = ?, artifact_hash = ?, role_confirmed = 0
            WHERE job_id = ? AND stage = 'confirm_roles'
            """,
            (forged.cache_key, forged_hash, job_id),
        )

    result = runner.run_until_pause(job_id)

    assert result["stages"]["confirm_roles"]["status"] == "paused"
    assert store.read_model(job_id, StageName.CONFIRM_ROLES, RoleArtifact).confirmed is False
    assert not store.artifact_path(job_id, StageName.EMOTION).exists()


def test_set_stage_cannot_mark_role_confirmation_paused_or_completed(tmp_path: Path) -> None:
    _, repository, _, job_id = _harness(tmp_path)

    for status in (StageStatus.PAUSED, StageStatus.COMPLETED):
        with pytest.raises(ValueError, match="fenced publication"):
            repository.set_stage(job_id, StageName.CONFIRM_ROLES, status)


def test_tampered_paused_candidate_cannot_supply_confirmed_utterance_body(
    tmp_path: Path,
) -> None:
    runner, repository, store, job_id = _harness(tmp_path)
    runner.run_until_pause(job_id)
    candidate = store.read_model(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
    tampered_utterances = list(candidate.utterances)
    tampered_utterances[0] = tampered_utterances[0].model_copy(
        update={"transcript": "attacker supplied transcript"}
    )
    tampered = candidate.model_copy(update={"utterances": tampered_utterances})
    store.write_model(job_id, StageName.CONFIRM_ROLES, tampered)
    with sqlite3.connect(repository.path) as database:
        database.execute(
            """
            UPDATE stages SET artifact_hash = ?
            WHERE job_id = ? AND stage = 'confirm_roles'
            """,
            (store.content_hash(job_id, StageName.CONFIRM_ROLES), job_id),
        )

    with pytest.raises(PipelineValidationError) as raised:
        _confirm(runner, job_id)

    assert raised.value.code == "invalid_role_candidate"
    assert repository.get_job(job_id)["stages"]["confirm_roles"]["role_confirmed"] == 0
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


def test_retry_fences_a_blocked_worker_before_it_can_overwrite_new_output(
    tmp_path: Path,
) -> None:
    blocked = BlockingDiarizer()
    old_runner, repository, store, job_id = _harness(tmp_path, diarizer=blocked)
    new_runner = PipelineRunner(
        repository,
        store,
        AudioService(tmp_path / "jobs", 60, 3600),
        diarization_provider=FakeDiarizationProvider(),
        transcription_provider=FakeTranscriptionProvider(),
        emotion_provider=FakeEmotionProvider(),
        strategy_provider=FakeResponseStrategyProvider(),
        report_provider=FakeReportSummaryProvider(),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        stale_future = executor.submit(old_runner.run_until_pause, job_id)
        assert blocked.entered.wait(timeout=5)

        current = new_runner.retry(job_id, StageName.DIARIZE)
        assert current["stages"]["confirm_roles"]["status"] == "paused"
        expected_bytes = store.artifact_path(job_id, StageName.DIARIZE).read_bytes()

        blocked.release.set()
        stale_future.result(timeout=5)

    assert store.artifact_path(job_id, StageName.DIARIZE).read_bytes() == expected_bytes
    assert json.loads(expected_bytes)["provider"]["name"] == "fake-diarization"
    assert not list(store.job_dir(job_id).glob(".diarize.v1.json.*.tmp"))


def test_runner_recovers_an_expired_claim_but_not_before_its_lease(tmp_path: Path) -> None:
    now = datetime(2026, 8, 19, tzinfo=UTC)

    def clock() -> datetime:
        return now

    repository = JobRepository(
        tmp_path / "voxdelta.sqlite3",
        clock=clock,
        claim_lease_seconds=30,
    )
    runner, _, _, job_id = _harness(tmp_path, repository_override=repository)
    original = repository.claim_stage(job_id, StageName.DIARIZE)
    assert original is not None

    active = runner.run_until_pause(job_id)
    assert active["stages"]["diarize"]["status"] == "running"
    assert active["stages"]["diarize"]["generation"] == original.generation

    now += timedelta(seconds=31)
    recovered = runner.run_until_pause(job_id)

    assert recovered["stages"]["confirm_roles"]["status"] == "paused"
    assert recovered["stages"]["diarize"]["generation"] == original.generation + 1


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
        artifact_hash=store.content_hash(job_id, StageName.NORMALIZE),
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


@pytest.mark.parametrize(
    "config",
    [
        {"emotion": {"median_window": 5}},
        {"transitions": {"delta_threshold": 0.3}},
        {"report": {"minimum_coverage": 0.8}},
        {"report": {"minimum_results": 5}},
        {"normalize": {"unsupported": True}},
    ],
)
def test_runner_rejects_configuration_that_is_not_applied(
    config: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    jobs_root = tmp_path / "jobs"

    with pytest.raises(ValueError, match="runner configuration"):
        PipelineRunner(
            JobRepository(tmp_path / "voxdelta.sqlite3"),
            ArtifactStore(jobs_root),
            AudioService(jobs_root, 60, 3600),
            config=config,
        )


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


@pytest.mark.parametrize(
    ("provider_kwargs", "failed_stage"),
    [
        ({"diarizer": ReversedDiarizer()}, StageName.DIARIZE),
        ({"transcription": ReversedTranscription()}, StageName.TRANSCRIBE),
    ],
)
def test_misaligned_timeline_provider_output_fails_its_own_stage(
    provider_kwargs: dict[str, object], failed_stage: StageName, tmp_path: Path
) -> None:
    runner, repository, store, job_id = _harness(
        tmp_path,
        **provider_kwargs,  # type: ignore[arg-type]
    )

    result = runner.run_until_pause(job_id)

    assert result["stages"][failed_stage.value]["status"] == "failed"
    assert not store.artifact_path(job_id, failed_stage).exists()
    assert (
        json.loads(repository.get_job(job_id)["stages"][failed_stage.value]["error_json"])["code"]
        == "invalid_stage_output"
    )


@pytest.mark.parametrize(
    ("provider_kwargs", "failed_stage"),
    [
        ({"emotion": WrongIdEmotion()}, StageName.EMOTION),
        ({"strategy": WrongIdStrategy()}, StageName.RESPONSE_STRATEGY),
        ({"report": IntegerSummary()}, StageName.REPORT),
    ],
)
def test_cross_artifact_or_wrong_type_provider_output_fails_safely(
    provider_kwargs: dict[str, object], failed_stage: StageName, tmp_path: Path
) -> None:
    runner, repository, store, job_id = _harness(
        tmp_path,
        **provider_kwargs,  # type: ignore[arg-type]
    )
    runner.run_until_pause(job_id)

    result = _confirm(runner, job_id)

    assert result["stages"][failed_stage.value]["status"] == "failed"
    assert not store.artifact_path(job_id, failed_stage).exists()
    error = json.loads(repository.get_job(job_id)["stages"][failed_stage.value]["error_json"])
    assert error["code"] == "invalid_stage_output"
    assert "7" not in error["message"]
