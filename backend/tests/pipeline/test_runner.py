from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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
from voxdelta.pipeline.runner import PipelineRunner, PipelineStateError, PipelineValidationError
from voxdelta.pipeline.stages import (
    DiarizeArtifact,
    EmotionArtifact,
    MediaReference,
    NormalizeArtifact,
    ReportArtifact,
    RoleArtifact,
    StrategyArtifact,
    TranscribeArtifact,
    TransitionsArtifact,
    cache_key_for_stage,
)
from voxdelta.providers.base import DiarizationTimelines, ProviderDiagnostic, ProviderError
from voxdelta.providers.fake import (
    FakeDiarizationProvider,
    FakeEmotionProvider,
    FakeReportSummaryProvider,
    FakeResponseStrategyProvider,
    FakeTranscriptionProvider,
)
from voxdelta.providers.fallback_asr import FallbackEvent
from voxdelta.providers.qwen_timestamps import SANITATION_POLICY, TimestampCoverage

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


class BlockingDeleteStore(ArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.delete_entered = threading.Event()
        self.release_delete = threading.Event()
        self._blocked_once = False

    def delete_stage(self, job_id: str, stage: StageName) -> None:
        if not self._blocked_once:
            self._blocked_once = True
            self.delete_entered.set()
            assert self.release_delete.wait(timeout=5)
        super().delete_stage(job_id, stage)


class DeletingOnOperationLockStore(ArtifactStore):
    def __init__(self, root: Path, repository: JobRepository) -> None:
        super().__init__(root)
        self.repository = repository
        self.armed_job_id: str | None = None

    @contextmanager
    def operation_lock(self, job_id: str):  # type: ignore[no-untyped-def]
        with super().operation_lock(job_id):
            if self.armed_job_id == job_id:
                self.armed_job_id = None
                self.repository.begin_delete(job_id)
            yield


class ReversedDiarizer(FakeDiarizationProvider):
    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]:
        return list(reversed(super().diarize(asset)))


class ReversedTranscription(FakeTranscriptionProvider):
    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]:
        return list(reversed(super().transcribe(asset, segments)))


class EvidenceDiarizer(FakeDiarizationProvider):
    def __init__(self) -> None:
        self.timeline_calls = 0
        self.legacy_calls = 0

    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]:
        self.legacy_calls += 1
        return super().diarize(asset)

    def diarize_timelines(self, asset: AudioAsset) -> DiarizationTimelines:
        self.timeline_calls += 1
        exclusive = FakeDiarizationProvider.diarize(self, asset)
        evidence = [
            segment.model_copy(update={"overlap": index in {0, 1}})
            for index, segment in enumerate(exclusive)
        ]
        return DiarizationTimelines(evidence=evidence, exclusive=exclusive)


class CapturingTranscription(FakeTranscriptionProvider):
    def __init__(self) -> None:
        self.segments: list[SpeakerSegment] = []

    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]:
        self.segments = list(segments)
        return super().transcribe(asset, segments)


class ConsecutiveDiarizer(FakeDiarizationProvider):
    def diarize_timelines(self, asset: AudioAsset) -> DiarizationTimelines:
        del asset
        segments = [
            SpeakerSegment(start=0, end=10, speaker_id="SPEAKER_00", confidence=1),
            SpeakerSegment(start=10, end=20, speaker_id="SPEAKER_00", confidence=1),
            SpeakerSegment(start=20, end=30, speaker_id="SPEAKER_01", confidence=1),
            SpeakerSegment(start=30, end=40, speaker_id="SPEAKER_00", confidence=1),
            SpeakerSegment(start=40, end=50, speaker_id="SPEAKER_01", confidence=1),
            SpeakerSegment(start=50, end=60, speaker_id="SPEAKER_01", confidence=1),
        ]
        return DiarizationTimelines(evidence=segments, exclusive=list(segments))


class WordDerivedTranscription(FakeTranscriptionProvider):
    def __init__(self, corruption: str | None = None) -> None:
        self.corruption = corruption

    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]:
        del asset, segments
        utterances = [
            Utterance(
                id="word-1",
                start=1,
                end=2,
                speaker_id="SPEAKER_00",
                confidence=1,
                transcript="첫 번째",
            ),
            Utterance(
                id="word-2",
                start=2,
                end=3,
                speaker_id="SPEAKER_00",
                confidence=1,
                transcript="두 번째",
            ),
            Utterance(
                id="span-same-speaker",
                start=5,
                end=15,
                speaker_id="SPEAKER_00",
                confidence=1,
                transcript="연속 구간",
            ),
            Utterance(
                id="word-3",
                start=21,
                end=22,
                speaker_id="SPEAKER_01",
                confidence=1,
                transcript="세 번째",
            ),
            Utterance(
                id="word-4",
                start=41,
                end=42,
                speaker_id="SPEAKER_01",
                confidence=1,
                transcript="네 번째",
            ),
            Utterance(
                id="partial-coverage",
                start=59,
                end=62,
                speaker_id="SPEAKER_01",
                confidence=1,
                transcript="부분 겹침",
            ),
        ]
        if self.corruption == "mismatch":
            utterances[0] = utterances[0].model_copy(update={"speaker_id": "SPEAKER_01"})
        elif self.corruption == "zero-overlap":
            utterances[-1] = utterances[-1].model_copy(update={"start": 62.0, "end": 63.0})
        elif self.corruption == "duplicate":
            utterances[1] = utterances[1].model_copy(update={"id": utterances[0].id})
        elif self.corruption == "semantic-duplicate":
            utterances[1] = utterances[0].model_copy(update={"id": "word-1-copy"})
        elif self.corruption == "unsorted":
            utterances = list(reversed(utterances))
        elif self.corruption == "cross-speaker":
            utterances[2] = utterances[2].model_copy(update={"start": 19.0, "end": 21.0})
        elif self.corruption == "outside":
            utterances[-1] = utterances[-1].model_copy(update={"end": 66.0})
        elif self.corruption == "empty":
            utterances[0] = utterances[0].model_copy(update={"transcript": "   "})
        return utterances


class WrongIdEmotion(FakeEmotionProvider):
    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        result = super().analyze(utterance_id, audio_path, transcript)
        return result.model_copy(update={"utterance_id": f"wrong-{utterance_id}"})


class ShortClipEmotion(FakeEmotionProvider):
    """Refuse exactly one customer turn the way the local model refuses a sub-0.5 s clip."""

    def __init__(self, refuse_index: int = 1) -> None:
        self.refuse_index = refuse_index
        self.seen: list[str] = []

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        self.seen.append(utterance_id)
        if len(self.seen) - 1 == self.refuse_index:
            raise ProviderError("audio_too_short")
        return super().analyze(utterance_id, audio_path, transcript)


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


class LateNormalizeAudioService(AudioService):
    def __init__(self, jobs_root: Path) -> None:
        super().__init__(jobs_root, 60, 3600)
        self.published = threading.Event()
        self.release = threading.Event()

    def ingest(  # type: ignore[no-untyped-def]
        self,
        upload_path: Path,
        job_id: str,
        channel_preference="auto",
        *,
        hold_generation_lease=False,
    ):
        asset = super().ingest(
            upload_path,
            job_id,
            channel_preference=channel_preference,
            hold_generation_lease=hold_generation_lease,
        )
        self.published.set()
        assert self.release.wait(timeout=5)
        return asset


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
    separate_media: bool = False,
) -> tuple[PipelineRunner, JobRepository, ArtifactStore, str]:
    jobs_root = tmp_path / "jobs"
    repository = repository_override or JobRepository(tmp_path / "voxdelta.sqlite3")
    store = ArtifactStore(jobs_root)
    job_id = repository.create_job(str(FIXTURE), diagnostic_capture=diagnostic_capture)
    seeded_generation = store.job_dir(job_id) / "audio-seeded"
    seeded_generation.mkdir()
    seeded_mixed = seeded_generation / "mixed.wav"
    shutil.copyfile(FIXTURE, seeded_mixed)
    if separate_media:
        seeded_left = seeded_generation / "left.wav"
        seeded_right = seeded_generation / "right.wav"
        shutil.copyfile(FIXTURE, seeded_left)
        shutil.copyfile(FIXTURE, seeded_right)
        normalized_paths = (str(seeded_left), str(seeded_right))
        channel_mode = "separate"
        channels = 2
    else:
        normalized_paths = (str(seeded_mixed),)
        channel_mode = "mixed"
        channels = 1
    asset = AudioAsset(
        source_name=FIXTURE.name,
        source_path=str(FIXTURE),
        normalized_paths=normalized_paths,
        channel_mode=channel_mode,
        duration_seconds=65.0,
        channels=channels,
        sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
    )
    cache_key = cache_key_for_stage(
        StageName.NORMALIZE,
        (),
        None,
        {"channel_preference": "auto"},
    )
    normalized_media = tuple(
        MediaReference(
            path=path,
            sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        )
        for path in normalized_paths
    )
    artifact = NormalizeArtifact(
        cache_key=cache_key,
        asset=asset,
        normalized_media=normalized_media,
        mixed_preview=(
            MediaReference(
                path=str(seeded_mixed),
                sha256=hashlib.sha256(seeded_mixed.read_bytes()).hexdigest(),
            )
            if separate_media
            else normalized_media[0]
        ),
    )
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
        assert raw["schema_version"] == ("2" if stage == StageName.DIARIZE else "1")
        assert len(raw["cache_key"]) == 64
        assert "provider" in raw


def test_pipeline_persists_overlap_evidence_but_transcribes_exclusive_timeline(
    tmp_path: Path,
) -> None:
    diarizer = EvidenceDiarizer()
    transcription = CapturingTranscription()
    runner, _repository, store, job_id = _harness(
        tmp_path, diarizer=diarizer, transcription=transcription
    )

    paused = runner.run_until_pause(job_id)

    assert paused["status"] == "paused"
    artifact = store.read_model(job_id, StageName.DIARIZE, DiarizeArtifact)
    assert diarizer.timeline_calls == 1
    assert diarizer.legacy_calls == 0
    assert any(segment.overlap for segment in artifact.segments)
    assert all(not segment.overlap for segment in artifact.alignment_segments)
    assert transcription.segments == artifact.alignment_segments


def test_transcription_semantics_accept_word_intervals_and_adjacent_same_speaker_turns(
    tmp_path: Path,
) -> None:
    runner, _repository, store, job_id = _harness(
        tmp_path,
        diarizer=ConsecutiveDiarizer(),
        transcription=WordDerivedTranscription(),
    )

    result = runner.run_until_pause(job_id)

    assert result["status"] == "paused"
    transcribed = store.read_model(job_id, StageName.TRANSCRIBE, TranscribeArtifact)
    assert len(transcribed.utterances) == 6
    assert transcribed.utterances[2].start == 5
    assert transcribed.utterances[2].end == 15
    assert transcribed.utterances[-1].end == 62


@pytest.mark.parametrize(
    "corruption",
    [
        "mismatch",
        "zero-overlap",
        "duplicate",
        "semantic-duplicate",
        "unsorted",
        "cross-speaker",
        "outside",
        "empty",
    ],
)
def test_transcription_semantics_reject_hostile_word_alignment(
    corruption: str,
    tmp_path: Path,
) -> None:
    runner, repository, store, job_id = _harness(
        tmp_path,
        diarizer=ConsecutiveDiarizer(),
        transcription=WordDerivedTranscription(corruption),
    )

    result = runner.run_until_pause(job_id)

    assert result["stages"][StageName.TRANSCRIBE.value]["status"] == "failed"
    assert not store.artifact_path(job_id, StageName.TRANSCRIBE).exists()
    error = json.loads(
        repository.get_job(job_id)["stages"][StageName.TRANSCRIBE.value]["error_json"]
    )
    assert error["code"] == "invalid_stage_output"


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


def test_missing_confirmation_marker_prevents_downstream_role_use(tmp_path: Path) -> None:
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


def test_candidate_body_mismatch_with_transcription_is_rejected(
    tmp_path: Path,
) -> None:
    runner, repository, store, job_id = _harness(tmp_path)
    runner.run_until_pause(job_id)
    candidate = store.read_model(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
    tampered_utterances = list(candidate.utterances)
    tampered_utterances[0] = tampered_utterances[0].model_copy(
        update={"transcript": "candidate transcript mismatch"}
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
    assert not any(
        thread.name.startswith(f"voxdelta-heartbeat-{job_id}") for thread in threading.enumerate()
    )


def test_late_stale_normalize_publication_garbage_collects_its_generation(
    tmp_path: Path,
) -> None:
    _, repository, store, job_id = _harness(tmp_path)
    late_audio = LateNormalizeAudioService(store.root)
    stale_runner = PipelineRunner(repository, store, late_audio)
    current_runner = PipelineRunner(repository, store, AudioService(store.root, 60, 3600))

    with ThreadPoolExecutor(max_workers=2) as executor:
        stale_future = executor.submit(stale_runner.retry, job_id, StageName.NORMALIZE)
        assert late_audio.published.wait(timeout=5)
        current = current_runner.retry(job_id, StageName.NORMALIZE)
        assert current["stages"]["confirm_roles"]["status"] == "paused"
        expected = store.read_model(job_id, StageName.NORMALIZE, NormalizeArtifact)
        expected_generation = Path(expected.asset.normalized_paths[0]).parent
        late_audio.release.set()
        stale_future.result(timeout=5)

    generations = [
        path
        for path in store.job_dir(job_id).iterdir()
        if path.is_dir() and path.name.startswith("audio-")
    ]
    assert generations == [expected_generation]


def test_reset_holds_job_operation_lock_through_exact_artifact_deletion(
    tmp_path: Path,
) -> None:
    runner, repository, store, job_id = _harness(tmp_path)
    runner.run_until_pause(job_id)
    _confirm(runner, job_id)
    blocking_store = BlockingDeleteStore(store.root)

    def make_runner() -> PipelineRunner:
        return PipelineRunner(
            repository,
            blocking_store,
            AudioService(store.root, 60, 3600),
            diarization_provider=FakeDiarizationProvider(),
            transcription_provider=FakeTranscriptionProvider(),
            emotion_provider=FakeEmotionProvider(),
            strategy_provider=FakeResponseStrategyProvider(),
            report_provider=FakeReportSummaryProvider(),
        )

    retry_runner = make_runner()
    competing_runner = make_runner()
    with ThreadPoolExecutor(max_workers=2) as executor:
        retry_future = executor.submit(retry_runner.retry, job_id, StageName.DIARIZE)
        assert blocking_store.delete_entered.wait(timeout=5)
        assert repository.get_job(job_id)["stages"]["diarize"]["status"] == "pending"

        competing_future = executor.submit(competing_runner.run_until_pause, job_id)
        time.sleep(0.2)
        assert not competing_future.done()
        assert repository.get_job(job_id)["stages"]["diarize"]["claim_token"] is None

        blocking_store.release_delete.set()
        retry_future.result(timeout=5)
        competing_future.result(timeout=5)
        assert repository.get_job(job_id)["stages"]["confirm_roles"]["status"] == "paused"


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


def test_heartbeat_prevents_live_blocked_provider_claim_from_being_stolen(
    tmp_path: Path,
) -> None:
    repository = JobRepository(
        tmp_path / "voxdelta.sqlite3",
        claim_lease_seconds=0.15,
    )
    blocked = BlockingDiarizer()
    _, _, store, job_id = _harness(
        tmp_path,
        diarizer=blocked,
        repository_override=repository,
    )
    live_runner = PipelineRunner(
        repository,
        store,
        AudioService(store.root, 60, 3600),
        diarization_provider=blocked,
        heartbeat_interval_seconds=0.03,
    )
    competing_runner = PipelineRunner(
        repository,
        store,
        AudioService(store.root, 60, 3600),
        heartbeat_interval_seconds=0.03,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        live_future = executor.submit(live_runner.run_until_pause, job_id)
        assert blocked.entered.wait(timeout=5)
        original = repository.get_job(job_id)["stages"]["diarize"]
        time.sleep(0.3)

        competing = competing_runner.run_until_pause(job_id)

        assert competing["stages"]["diarize"]["status"] == StageStatus.RUNNING.value
        current = repository.get_job(job_id)["stages"]["diarize"]
        assert current["generation"] == original["generation"]
        assert current["claim_token"] == original["claim_token"]
        blocked.release.set()
        assert live_future.result(timeout=5)["stages"]["confirm_roles"]["status"] == "paused"

    assert not any(
        thread.name.startswith(f"voxdelta-heartbeat-{job_id}") for thread in threading.enumerate()
    )


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


@pytest.mark.parametrize("media_name", ["mixed.wav", "left.wav", "right.wav"])
def test_changed_normalized_media_invalidates_cached_report(
    media_name: str,
    tmp_path: Path,
) -> None:
    runner, repository, store, job_id = _harness(tmp_path, separate_media=True)
    runner.run_until_pause(job_id)
    _confirm(runner, job_id)
    before = repository.get_job(job_id)["stages"]["normalize"]["generation"]
    normalize = store.read_model(job_id, StageName.NORMALIZE, NormalizeArtifact)
    references = (*normalize.normalized_media, normalize.mixed_preview)
    target = next(Path(item.path) for item in references if Path(item.path).name == media_name)
    target.write_bytes(b"changed normalized media")

    result = runner.run_until_pause(job_id)

    assert result["status"] != StageStatus.COMPLETED.value
    assert result["stages"]["normalize"]["generation"] == before + 1
    assert result["stages"]["confirm_roles"]["status"] == StageStatus.PAUSED.value
    assert result["stages"]["report"]["status"] == StageStatus.PENDING.value
    assert not store.artifact_path(job_id, StageName.REPORT).exists()


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
    prior_generation = Path(normalize.asset.normalized_paths[0]).parent
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
    current = store.read_model(job_id, StageName.NORMALIZE, NormalizeArtifact)
    current_generation = Path(current.asset.normalized_paths[0]).parent
    assert current_generation != prior_generation
    assert current_generation.is_dir()
    assert not prior_generation.exists()


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


def test_retry_translates_delete_race_inside_operation_lock_to_conflict(tmp_path: Path) -> None:
    runner, repository, store, job_id = _harness(tmp_path)
    runner.run_until_pause(job_id)
    racing_store = DeletingOnOperationLockStore(store.root, repository)
    racing_runner = PipelineRunner(
        repository,
        racing_store,
        AudioService(store.root, 60, 3600),
    )
    racing_store.armed_job_id = job_id

    with pytest.raises(PipelineStateError) as raised:
        racing_runner.retry(job_id, StageName.DIARIZE)

    assert raised.value.code == "job_deleting"
    assert repository.get_job(job_id)["status"] == "deleting"


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
    assert not any(
        thread.name.startswith(f"voxdelta-heartbeat-{job_id}") for thread in threading.enumerate()
    )


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


def test_a_turn_too_short_to_score_is_omitted_and_named_in_the_report_warnings(
    tmp_path: Path,
) -> None:
    """No fabricated distribution, and no silent drop either.

    The pipeline models "we could not score this turn" already: the utterance simply has
    no emotion result, which makes its transition triples ineligible and lowers
    `valid_coverage`. The only thing missing was saying so out loud.
    """

    emotion = ShortClipEmotion(refuse_index=1)
    runner, _, store, job_id = _harness(tmp_path, emotion=emotion)
    runner.run_until_pause(job_id)

    completed = _confirm(runner, job_id)

    assert completed["stages"]["emotion"]["status"] == "completed"
    role = store.read_model(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
    customer_ids = [u.id for u in role.utterances if u.role == Role.CUSTOMER]
    skipped = customer_ids[1]
    emotions = store.read_model(job_id, StageName.EMOTION, EmotionArtifact)

    # Every other turn is scored exactly as before, and the refused one is not invented.
    assert [result.utterance_id for result in emotions.results] == [
        item for item in customer_ids if item != skipped
    ]
    assert any(skipped in warning for warning in emotions.warnings)
    assert any("0.5초" in warning for warning in emotions.warnings)


def test_a_turn_too_short_to_score_never_becomes_a_fabricated_distribution(
    tmp_path: Path,
) -> None:
    emotion = ShortClipEmotion(refuse_index=0)
    runner, _, store, job_id = _harness(tmp_path, emotion=emotion)
    runner.run_until_pause(job_id)
    _confirm(runner, job_id)

    role = store.read_model(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
    refused = [u.id for u in role.utterances if u.role == Role.CUSTOMER][0]
    emotions = store.read_model(job_id, StageName.EMOTION, EmotionArtifact)

    assert refused not in {result.utterance_id for result in emotions.results}


def test_an_emotion_provider_failure_that_is_not_length_still_fails_the_stage(
    tmp_path: Path,
) -> None:
    """Only the length case is a policy; every other provider failure still fails."""

    class UnavailableEmotion(FakeEmotionProvider):
        def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
            raise ProviderError("provider_unavailable")

    runner, _, _, job_id = _harness(tmp_path, emotion=UnavailableEmotion())
    runner.run_until_pause(job_id)

    completed = _confirm(runner, job_id)

    assert completed["stages"]["emotion"]["status"] == "failed"
    assert completed["status"] == "failed"


def test_retrying_a_failed_emotion_stage_never_re_runs_diarization_or_transcription(
    tmp_path: Path,
) -> None:
    """Recovery after an emotion failure must stay local to emotion and below.

    Diarization is the one stage that can leave this machine, so a retry that re-ran it
    would turn a local recovery into a second remote call on the same audio.
    """

    class FailThenSucceedEmotion(FakeEmotionProvider):
        def __init__(self) -> None:
            self.fail = True

        def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
            if self.fail:
                raise ProviderError("provider_unavailable")
            return super().analyze(utterance_id, audio_path, transcript)

    diarizer = CountingDiarizer()
    emotion = FailThenSucceedEmotion()
    runner, _, store, job_id = _harness(tmp_path, diarizer=diarizer, emotion=emotion)
    runner.run_until_pause(job_id)
    failed = _confirm(runner, job_id)
    assert failed["stages"]["emotion"]["status"] == "failed"
    diarize_calls_after_first_pass = diarizer.calls
    transcribe_before = store.content_hash(job_id, StageName.TRANSCRIBE)

    emotion.fail = False
    recovered = runner.retry(job_id, StageName.EMOTION)

    assert recovered["status"] == "completed"
    assert diarizer.calls == diarize_calls_after_first_pass
    assert recovered["stages"]["diarize"]["status"] == "completed"
    assert store.content_hash(job_id, StageName.TRANSCRIBE) == transcribe_before
    # The role gate stays decided: a retry from emotion must not re-open it.
    assert recovered["stages"]["confirm_roles"]["role_confirmed"] == 1


class CoverageReportingTranscription(FakeTranscriptionProvider):
    """A recognizer that reports timestamp-sanitation coverage, as Qwen's does."""

    def __init__(self, coverage: TimestampCoverage | None) -> None:
        self.last_timestamp_coverage = coverage


def test_transcription_coverage_reaches_the_report_and_names_the_loss(tmp_path: Path) -> None:
    coverage = TimestampCoverage(
        total_words=100, positive_spans=90, zero_spans=10, zero_spans_unplaceable=4, omitted_words=4
    ).with_alignment_omissions(0)
    runner, _repository, store, job_id = _harness(
        tmp_path, transcription=CoverageReportingTranscription(coverage)
    )

    runner.run_until_pause(job_id)
    _confirm(runner, job_id)
    runner.run_until_pause(job_id)

    transcribed = store.read_model(job_id, StageName.TRANSCRIBE, TranscribeArtifact)
    assert transcribed.timestamp_coverage is not None
    assert transcribed.timestamp_coverage.omitted_words == 4
    report = store.read_model(job_id, StageName.REPORT, ReportArtifact).report
    assert report.transcription_coverage is not None
    assert report.transcription_coverage.policy == SANITATION_POLICY
    assert report.transcription_coverage.attributed_words == 96
    assert report.transcription_coverage.uncertain is True
    assert any("4개 단어" in warning for warning in report.warnings)


def test_a_recognizer_that_lost_nothing_adds_no_warning(tmp_path: Path) -> None:
    coverage = TimestampCoverage(
        total_words=50, positive_spans=48, zero_spans=2, zero_spans_unplaceable=0, omitted_words=0
    ).with_alignment_omissions(0)
    runner, _repository, store, job_id = _harness(
        tmp_path, transcription=CoverageReportingTranscription(coverage)
    )

    runner.run_until_pause(job_id)
    _confirm(runner, job_id)
    runner.run_until_pause(job_id)

    report = store.read_model(job_id, StageName.REPORT, ReportArtifact).report
    assert report.transcription_coverage is not None
    assert report.transcription_coverage.uncertain is False
    assert not any("전사에서" in warning or "단어" in warning for warning in report.warnings)


def test_a_recognizer_without_coverage_leaves_the_report_field_absent(tmp_path: Path) -> None:
    """faster-whisper reports none, and must keep producing a report that claims none."""

    runner, _repository, store, job_id = _harness(tmp_path)

    runner.run_until_pause(job_id)
    _confirm(runner, job_id)
    runner.run_until_pause(job_id)

    transcribed = store.read_model(job_id, StageName.TRANSCRIBE, TranscribeArtifact)
    assert transcribed.timestamp_coverage is None
    report = store.read_model(job_id, StageName.REPORT, ReportArtifact).report
    assert report.transcription_coverage is None
    assert report.warnings == []


class FallingBackTranscription(FakeTranscriptionProvider):
    """A transcription provider that reports having fallen back, as the wrapper does."""

    def __init__(self, code: str = "provider_unavailable") -> None:
        self.last_fallback = FallbackEvent(
            primary="qwen3-asr", fallback="faster-whisper", code=code
        )
        self.last_timestamp_coverage = None


def test_a_fallback_is_named_in_the_report_warnings(tmp_path: Path) -> None:
    """Provenance alone is easy to miss; the substitution is said out loud."""

    runner, _repository, store, job_id = _harness(
        tmp_path, transcription=FallingBackTranscription()
    )

    runner.run_until_pause(job_id)
    _confirm(runner, job_id)
    runner.run_until_pause(job_id)

    report = store.read_model(job_id, StageName.REPORT, ReportArtifact).report
    assert any(
        "qwen3-asr" in warning and "faster-whisper" in warning for warning in report.warnings
    )
    assert any("provider_unavailable" in warning for warning in report.warnings)


def test_a_run_without_a_fallback_adds_no_fallback_warning(tmp_path: Path) -> None:
    runner, _repository, store, job_id = _harness(tmp_path)

    runner.run_until_pause(job_id)
    _confirm(runner, job_id)
    runner.run_until_pause(job_id)

    report = store.read_model(job_id, StageName.REPORT, ReportArtifact).report
    assert not any("대체 모델" in warning for warning in report.warnings)


def test_a_provider_diagnostic_reaches_the_failed_stage_event_sanitized(tmp_path: Path) -> None:
    """The stored public code stays a bare typed code; the operator log names the hop."""

    class ClassifyingDiarizer(FakeDiarizationProvider):
        def diarize_timelines(self, asset: AudioAsset) -> DiarizationTimelines:
            raise ProviderError(
                "provider_unavailable",
                diagnostic=ProviderDiagnostic(
                    boundary="media_input",
                    failure="http_status",
                    status=403,
                ),
            )

    runner, repository, store, job_id = _harness(tmp_path, diarizer=ClassifyingDiarizer())

    runner.run_until_pause(job_id)

    stage = cast(dict[str, object], repository.get_job(job_id)["stages"])["diarize"]
    assert stage["status"] == "failed"
    assert json.loads(cast(str, stage["error_json"])) == {
        "code": "provider_unavailable",
        "message": "The provider is unavailable.",
    }
    events = [
        json.loads(line)
        for line in (store.job_dir(job_id) / "pipeline.jsonl").read_text().splitlines()
    ]
    failed = [
        event for event in events if event["stage"] == "diarize" and event["event"] == "failed"
    ]
    assert len(failed) == 1
    assert failed[0]["error_code"] == "provider_unavailable"
    assert failed[0]["metadata"] == {
        "boundary": "media_input",
        "failure": "http_status",
        "status": 403,
    }
