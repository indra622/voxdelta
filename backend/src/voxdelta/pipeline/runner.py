"""Resumable, cache-validating orchestration for the local VoxDelta pipeline."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import tempfile
import time
import wave
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from typing import TypeVar, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from voxdelta.analysis.emotions import median_smooth
from voxdelta.analysis.roles import suggest_roles
from voxdelta.analysis.summary import InsufficientEmotionCoverage, build_call_summary
from voxdelta.analysis.transitions import build_transitions
from voxdelta.audio.service import AudioRejected, AudioService, ChannelPreference
from voxdelta.domain.models import (
    AnalysisReport,
    CallSummary,
    ProviderProvenance,
    Role,
    StageName,
    StageStatus,
    Utterance,
)
from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.logging import PipelineLogger
from voxdelta.jobs.repository import JobRepository, StageClaim
from voxdelta.pipeline.stages import (
    ARTIFACT_MODELS,
    STAGE_ORDER,
    UPSTREAM_STAGES,
    DiarizeArtifact,
    EmotionArtifact,
    MediaReference,
    NormalizeArtifact,
    ReportArtifact,
    RoleArtifact,
    StageArtifact,
    StrategyArtifact,
    TranscribeArtifact,
    TransitionsArtifact,
    cache_key_for_stage,
    downstream_stages,
)
from voxdelta.providers.base import (
    DiarizationProvider,
    EmotionProvider,
    ReportSummaryProvider,
    ResponseStrategyProvider,
    TranscriptionProvider,
)
from voxdelta.providers.fake import (
    FakeDiarizationProvider,
    FakeEmotionProvider,
    FakeReportSummaryProvider,
    FakeResponseStrategyProvider,
    FakeTranscriptionProvider,
)

TArtifact = TypeVar("TArtifact", bound=StageArtifact)

_DEFAULT_STAGE_CONFIG: dict[StageName, dict[str, object]] = {
    StageName.NORMALIZE: {"channel_preference": "auto"},
    StageName.DIARIZE: {},
    StageName.TRANSCRIBE: {},
    StageName.CONFIRM_ROLES: {"confirmed": False},
    StageName.EMOTION: {"customer_only": True, "median_window": 3},
    StageName.RESPONSE_STRATEGY: {"relevant_agent_only": True},
    StageName.TRANSITIONS: {"adjacent_only": True, "delta_threshold": 0.2},
    StageName.REPORT: {"minimum_coverage": 0.5, "minimum_results": 3},
}


class NormalizeRunnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel_preference: ChannelPreference = "auto"


class RunnerConfig(BaseModel):
    """The complete set of behavior-changing options currently implemented."""

    model_config = ConfigDict(extra="forbid")

    normalize: NormalizeRunnerConfig = NormalizeRunnerConfig()


class PipelineValidationError(ValueError):
    """A stable public pipeline error without provider or user payload text."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class PipelineStateError(PipelineValidationError):
    """A safe conflict between the requested operation and persisted stage state."""


def _stage_rows(job: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = job.get("stages")
    if not isinstance(rows, dict):
        raise RuntimeError("repository returned invalid stage data")
    return cast(dict[str, dict[str, object]], rows)


def _ordered_utterances(artifact: RoleArtifact) -> list[Utterance]:
    return sorted(
        artifact.utterances,
        key=lambda item: (item.start, item.end, item.id, item.speaker_id),
    )


def _relevant_agent_ids(ordered: list[Utterance]) -> list[str]:
    return [
        item.id
        for index, item in enumerate(ordered)
        if 0 < index < len(ordered) - 1
        and item.role == Role.AGENT
        and ordered[index - 1].role == Role.CUSTOMER
        and ordered[index + 1].role == Role.CUSTOMER
    ]


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)


class PipelineRunner:
    """Compose storage, providers, and deterministic analysis into resumable stages."""

    def __init__(
        self,
        repository: JobRepository,
        artifacts: ArtifactStore,
        audio_service: AudioService,
        *,
        diarization_provider: DiarizationProvider | None = None,
        transcription_provider: TranscriptionProvider | None = None,
        emotion_provider: EmotionProvider | None = None,
        strategy_provider: ResponseStrategyProvider | None = None,
        report_provider: ReportSummaryProvider | None = None,
        config: RunnerConfig | Mapping[str, Mapping[str, object]] | None = None,
        logger: PipelineLogger | None = None,
        heartbeat_interval_seconds: float | None = None,
    ) -> None:
        self._repository = repository
        self._artifacts = artifacts
        self._audio = audio_service
        self._diarization = diarization_provider or FakeDiarizationProvider()
        self._transcription = transcription_provider or FakeTranscriptionProvider()
        self._emotion = emotion_provider or FakeEmotionProvider()
        self._strategy = strategy_provider or FakeResponseStrategyProvider()
        self._report = report_provider or FakeReportSummaryProvider()
        try:
            self._config = (
                config
                if isinstance(config, RunnerConfig)
                else RunnerConfig.model_validate(config or {})
            )
        except ValidationError:
            raise ValueError("runner configuration contains unsupported fields or values") from None
        heartbeat_interval = (
            repository.claim_lease_seconds / 3
            if heartbeat_interval_seconds is None
            else heartbeat_interval_seconds
        )
        if not 0 < heartbeat_interval < repository.claim_lease_seconds:
            raise ValueError("heartbeat interval must be positive and shorter than the claim lease")
        self._heartbeat_interval = heartbeat_interval
        self._logger = logger or PipelineLogger(artifacts)
        self._locks: dict[str, RLock] = {}
        self._locks_guard = Lock()

    def _job_lock(self, job_id: str) -> RLock:
        with self._locks_guard:
            return self._locks.setdefault(job_id, RLock())

    def _start_claim_heartbeat(self, claim: StageClaim) -> tuple[Event, Thread]:
        """Start renewing one claim until its returned stop event is set."""

        stopped = Event()

        def heartbeat() -> None:
            while not stopped.wait(self._heartbeat_interval):
                try:
                    if not self._repository.renew_claim(claim):
                        return
                except Exception:
                    # Publication remains generation-fenced; retry transient renewal errors.
                    continue

        thread = Thread(
            target=heartbeat,
            name=f"voxdelta-heartbeat-{claim.job_id}-{claim.stage.value}",
            daemon=True,
        )
        thread.start()
        return stopped, thread

    def _provider(self, stage: StageName) -> ProviderProvenance | None:
        providers: dict[StageName, ProviderProvenance] = {
            StageName.DIARIZE: self._diarization.provenance,
            StageName.TRANSCRIBE: self._transcription.provenance,
            StageName.EMOTION: self._emotion.provenance,
            StageName.RESPONSE_STRATEGY: self._strategy.provenance,
            StageName.REPORT: self._report.provenance,
        }
        return providers.get(stage)

    def _stage_config(
        self,
        stage: StageName,
        role_artifact: RoleArtifact | None = None,
    ) -> dict[str, object]:
        config = dict(_DEFAULT_STAGE_CONFIG[stage])
        if stage == StageName.NORMALIZE:
            config.update(self._config.normalize.model_dump(mode="python"))
        if stage == StageName.CONFIRM_ROLES and role_artifact is not None:
            config["confirmed"] = role_artifact.confirmed
            if role_artifact.mapping is not None:
                config["mapping"] = {
                    speaker_id: role.value
                    for speaker_id, role in sorted(role_artifact.mapping.items())
                }
            else:
                config.pop("mapping", None)
        return config

    def _upstream_hashes(self, job_id: str, stage: StageName) -> tuple[str, ...]:
        return tuple(
            self._artifacts.content_hash(job_id, upstream) for upstream in UPSTREAM_STAGES[stage]
        )

    def _cache_key(
        self,
        job_id: str,
        stage: StageName,
        role_artifact: RoleArtifact | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        upstream_hashes = self._upstream_hashes(job_id, stage)
        return (
            cache_key_for_stage(
                stage,
                upstream_hashes,
                self._provider(stage),
                self._stage_config(stage, role_artifact),
            ),
            upstream_hashes,
        )

    def _read(self, job_id: str, stage: StageName, model: type[TArtifact]) -> TArtifact:
        return self._artifacts.read_model(job_id, stage, model)

    def _hash_trusted_media(self, job_id: str, raw_path: str) -> str:
        """Hash one regular, non-link file in a direct job audio generation."""

        job_directory = self._artifacts.job_dir(job_id)
        path = Path(raw_path)
        if not path.is_absolute():
            raise ValueError("normalized media path must be absolute")
        generation = path.parent
        if (
            not generation.name.startswith("audio-")
            or generation.parent != job_directory
            or generation.is_symlink()
            or not generation.is_dir()
            or generation.resolve() != generation
        ):
            raise ValueError("normalized media must remain in one validated audio generation")
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise ValueError("normalized media must be a regular non-link file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or _file_identity(opened) != _file_identity(before):
                raise ValueError("normalized media changed while being opened")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            after = path.lstat()
            if path.is_symlink() or _file_identity(after) != _file_identity(opened):
                raise ValueError("normalized media changed while being hashed")
            return digest.hexdigest()
        finally:
            os.close(descriptor)

    def _normalize_media_are_valid(self, job_id: str, artifact: NormalizeArtifact) -> bool:
        references = (*artifact.normalized_media, artifact.mixed_preview)
        generations = {Path(item.path).parent for item in references}
        if len(generations) != 1:
            return False
        return all(
            self._hash_trusted_media(job_id, item.path) == item.sha256 for item in references
        )

    @contextmanager
    def _emotion_clip(
        self,
        job_id: str,
        asset: NormalizeArtifact,
        utterance: Utterance,
    ) -> Iterator[Path]:
        audio = asset.asset
        if not audio.normalized_paths:
            raise ValueError("normalized audio path is missing")
        preview = Path(asset.mixed_preview.path)
        job_directory = self._artifacts.job_dir(job_id)
        if self._hash_trusted_media(job_id, asset.mixed_preview.path) != asset.mixed_preview.sha256:
            raise ValueError("normalized mixed preview is unavailable")
        resolved_preview = preview.resolve()
        if (
            not math.isfinite(utterance.start)
            or not math.isfinite(utterance.end)
            or utterance.start < 0
            or utterance.end <= utterance.start
        ):
            raise ValueError("utterance bounds are invalid for emotion slicing")

        descriptor, temporary_name = tempfile.mkstemp(
            dir=job_directory,
            prefix=".emotion-",
            suffix=".wav",
        )
        clip = Path(temporary_name)
        try:
            try:
                os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
            try:
                with wave.open(str(resolved_preview), "rb") as source:
                    if (
                        source.getnchannels() != 1
                        or source.getsampwidth() != 2
                        or source.getcomptype() != "NONE"
                        or source.getframerate() <= 0
                    ):
                        raise ValueError("normalized mixed preview format is invalid")
                    sample_rate = source.getframerate()
                    start_frame = round(utterance.start * sample_rate)
                    end_frame = round(utterance.end * sample_rate)
                    if (
                        start_frame < 0
                        or end_frame <= start_frame
                        or end_frame > source.getnframes()
                    ):
                        raise ValueError("utterance bounds exceed normalized audio")
                    source.setpos(start_frame)
                    frames = source.readframes(end_frame - start_frame)
                    if len(frames) != (end_frame - start_frame) * source.getsampwidth():
                        raise ValueError("normalized audio ended before the utterance bound")
                    with wave.open(str(clip), "wb") as output:
                        output.setparams(
                            (
                                1,
                                source.getsampwidth(),
                                sample_rate,
                                end_frame - start_frame,
                                "NONE",
                                "not compressed",
                            )
                        )
                        output.writeframes(frames)
                os.chmod(clip, 0o600)
                with clip.open("rb") as output:
                    os.fsync(output.fileno())
            except (EOFError, OSError, wave.Error):
                raise ValueError("normalized mixed preview is invalid") from None
            yield clip
        finally:
            clip.unlink(missing_ok=True)

    def _artifact_semantics_are_valid(self, job_id: str, artifact: StageArtifact) -> bool:
        if isinstance(artifact, NormalizeArtifact):
            return self._normalize_media_are_valid(job_id, artifact)
        if isinstance(artifact, DiarizeArtifact):
            ordered_segments = sorted(
                artifact.segments,
                key=lambda item: (item.start, item.end, item.speaker_id),
            )
            return (
                bool(artifact.segments)
                and artifact.segments == ordered_segments
                and len({item.speaker_id for item in artifact.segments}) == 2
            )
        if isinstance(artifact, TranscribeArtifact):
            diarized = self._read(job_id, StageName.DIARIZE, DiarizeArtifact)
            if len(artifact.utterances) != len(diarized.segments):
                return False
            if len({item.id for item in artifact.utterances}) != len(artifact.utterances):
                return False
            return all(
                utterance.role == Role.UNKNOWN
                and (
                    utterance.start,
                    utterance.end,
                    utterance.speaker_id,
                    utterance.overlap,
                    utterance.confidence,
                )
                == (
                    segment.start,
                    segment.end,
                    segment.speaker_id,
                    segment.overlap,
                    segment.confidence,
                )
                for utterance, segment in zip(artifact.utterances, diarized.segments, strict=True)
            )
        if isinstance(artifact, RoleArtifact):
            transcribed = self._read(job_id, StageName.TRANSCRIBE, TranscribeArtifact)
            speakers = {utterance.speaker_id for utterance in artifact.utterances}
            if len(speakers) != 2:
                return False
            if artifact.suggestion != suggest_roles(transcribed.utterances):
                return False
            if not artifact.confirmed:
                expected = [
                    Utterance.model_validate(
                        {**utterance.model_dump(mode="python"), "role": Role.UNKNOWN}
                    )
                    for utterance in transcribed.utterances
                ]
                return artifact.mapping is None and artifact.utterances == expected
            if artifact.mapping is None or set(artifact.mapping) != speakers:
                return False
            if sorted(artifact.mapping.values()) != [Role.AGENT, Role.CUSTOMER]:
                return False
            expected = [
                Utterance.model_validate(
                    {
                        **utterance.model_dump(mode="python"),
                        "role": artifact.mapping[utterance.speaker_id],
                    }
                )
                for utterance in transcribed.utterances
            ]
            return artifact.utterances == expected
        if isinstance(artifact, EmotionArtifact):
            roles = self._read(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
            customer_ids = [
                item.id for item in _ordered_utterances(roles) if item.role == Role.CUSTOMER
            ]
            return [item.utterance_id for item in artifact.results] == customer_ids and all(
                item.provider == self._emotion.provenance for item in artifact.results
            )
        if isinstance(artifact, StrategyArtifact):
            roles = self._read(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
            expected_ids = _relevant_agent_ids(_ordered_utterances(roles))
            return [item.utterance_id for item in artifact.results] == expected_ids and all(
                item.provider == self._strategy.provenance for item in artifact.results
            )
        if isinstance(artifact, TransitionsArtifact):
            roles = self._read(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
            emotions = self._read(job_id, StageName.EMOTION, EmotionArtifact)
            return artifact.results == build_transitions(
                _ordered_utterances(roles), emotions.results
            )
        if isinstance(artifact, ReportArtifact):
            roles = self._read(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
            emotions = self._read(job_id, StageName.EMOTION, EmotionArtifact)
            strategies = self._read(job_id, StageName.RESPONSE_STRATEGY, StrategyArtifact)
            transitions = self._read(job_id, StageName.TRANSITIONS, TransitionsArtifact)
            report_utterances = _ordered_utterances(roles)
            customers = [item for item in report_utterances if item.role == Role.CUSTOMER]
            expected_summary = build_call_summary(customers, emotions.results, transitions.results)
            summary_without_narrative = CallSummary.model_validate(
                {
                    **artifact.report.summary.model_dump(mode="python"),
                    "narrative": None,
                }
            )
            return (
                artifact.report.job_id == job_id
                and artifact.report.utterances == report_utterances
                and artifact.report.emotions == emotions.results
                and artifact.report.strategies == strategies.results
                and artifact.report.transitions == transitions.results
                and summary_without_narrative == expected_summary
            )
        return False

    def _completed_artifact_is_valid(
        self,
        job_id: str,
        stage: StageName,
        row: dict[str, object],
        *,
        require_confirmed_role: bool,
    ) -> bool:
        try:
            model = ARTIFACT_MODELS[stage]
            artifact = self._artifacts.read_model(job_id, stage, model)
            if not isinstance(artifact, StageArtifact):
                return False
            if row.get("artifact_path") != self._artifacts.artifact_path(job_id, stage).name:
                return False
            if row.get("cache_key") != artifact.cache_key:
                return False
            if row.get("artifact_hash") != self._artifacts.content_hash(job_id, stage):
                return False
            if artifact.provider != self._provider(stage):
                return False
            role_artifact = artifact if isinstance(artifact, RoleArtifact) else None
            if require_confirmed_role and (role_artifact is None or not role_artifact.confirmed):
                return False
            if role_artifact is not None:
                expected_marker = 1 if role_artifact.confirmed else 0
                if row.get("role_confirmed") != expected_marker:
                    return False
            expected_key, upstream_hashes = self._cache_key(job_id, stage, role_artifact)
            return (
                artifact.cache_key == expected_key
                and artifact.upstream_hashes == upstream_hashes
                and self._artifact_semantics_are_valid(job_id, artifact)
            )
        except (OSError, TypeError, ValueError, ValidationError):
            return False

    def _invalidate_from(self, job_id: str, stage: StageName) -> None:
        selected = downstream_stages(stage)
        with self._artifacts.operation_lock(job_id):
            self._repository.invalidate_stages(job_id, selected)
            for selected_stage in selected:
                self._artifacts.delete_stage(job_id, selected_stage)

    def _public_failure(self, stage: StageName, error: BaseException) -> PipelineValidationError:
        if isinstance(error, PipelineValidationError):
            return error
        if isinstance(error, AudioRejected):
            return PipelineValidationError("audio_rejected", str(error))
        if isinstance(error, InsufficientEmotionCoverage):
            return PipelineValidationError(
                "insufficient_emotion_coverage",
                "There is not enough customer emotion coverage to generate a report.",
            )
        return PipelineValidationError(
            "invalid_stage_output",
            f"The {stage.value} stage produced invalid data.",
        )

    def _log(
        self,
        job_id: str,
        stage: StageName,
        event: str,
        started: float,
        error_code: str | None = None,
    ) -> None:
        provider = self._provider(stage)
        self._logger.event(
            job_id=job_id,
            stage=stage.value,
            event=event,
            duration_ms=(time.monotonic() - started) * 1000,
            provider=provider.name if provider is not None else None,
            model=provider.model if provider is not None else None,
            error_code=error_code,
        )

    def _diagnostic(
        self,
        job_id: str,
        stage: StageName,
        phase: str,
        metadata: Mapping[str, object],
    ) -> None:
        job = self._repository.get_job(job_id)
        enabled = job.get("diagnostic_capture") == 1
        provider = self._provider(stage)
        self._logger.diagnostic(
            job_id,
            enabled,
            f"{stage.value}-{phase}",
            {
                "stage": stage.value,
                "phase": phase,
                "provider": provider.name if provider is not None else None,
                "model": provider.model if provider is not None else None,
                **metadata,
            },
        )

    def _build_artifact(self, job_id: str, stage: StageName) -> StageArtifact:
        cache_key, upstream_hashes = self._cache_key(job_id, stage)
        provider = self._provider(stage)
        if stage == StageName.NORMALIZE:
            job = self._repository.get_job(job_id)
            source_name = job.get("source_name")
            if not isinstance(source_name, str):
                raise ValueError("job source name is invalid")
            preference = self._stage_config(stage).get("channel_preference")
            if preference not in {"auto", "mixed", "separate"}:
                raise ValueError("channel preference is invalid")
            asset = self._audio.ingest(
                Path(source_name),
                job_id,
                channel_preference=cast(ChannelPreference, preference),
            )
            normalized_media = tuple(
                MediaReference(path=path, sha256=self._hash_trusted_media(job_id, path))
                for path in asset.normalized_paths
            )
            mixed_path = (
                str(Path(asset.normalized_paths[0]).parent / "mixed.wav")
                if asset.channel_mode == "separate"
                else asset.normalized_paths[0]
            )
            mixed_preview = (
                normalized_media[0]
                if asset.channel_mode == "mixed"
                else MediaReference(
                    path=mixed_path,
                    sha256=self._hash_trusted_media(job_id, mixed_path),
                )
            )
            return NormalizeArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                asset=asset,
                normalized_media=normalized_media,
                mixed_preview=mixed_preview,
            )
        if stage == StageName.DIARIZE:
            normalized = self._read(job_id, StageName.NORMALIZE, NormalizeArtifact)
            self._diagnostic(
                job_id,
                stage,
                "request",
                {"duration_seconds": normalized.asset.duration_seconds},
            )
            segments = self._diarization.diarize(normalized.asset)
            self._diagnostic(
                job_id,
                stage,
                "response",
                {"result_type": "speaker_segments", "item_count": len(segments)},
            )
            return DiarizeArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                segments=segments,
            )
        if stage == StageName.TRANSCRIBE:
            normalized = self._read(job_id, StageName.NORMALIZE, NormalizeArtifact)
            diarized = self._read(job_id, StageName.DIARIZE, DiarizeArtifact)
            self._diagnostic(
                job_id,
                stage,
                "request",
                {"segment_count": len(diarized.segments)},
            )
            utterances = self._transcription.transcribe(normalized.asset, diarized.segments)
            self._diagnostic(
                job_id,
                stage,
                "response",
                {"result_type": "utterances", "item_count": len(utterances)},
            )
            return TranscribeArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                utterances=utterances,
            )
        if stage == StageName.CONFIRM_ROLES:
            transcribed = self._read(job_id, StageName.TRANSCRIBE, TranscribeArtifact)
            candidate = RoleArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                utterances=[
                    utterance.model_copy(update={"role": Role.UNKNOWN}, deep=True)
                    for utterance in transcribed.utterances
                ],
                suggestion=suggest_roles(transcribed.utterances),
            )
            if len({utterance.speaker_id for utterance in candidate.utterances}) != 2:
                raise PipelineValidationError(
                    "unsupported_speaker_count",
                    "Exactly two observed speakers are required.",
                )
            return candidate
        roles = self._read(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
        if not roles.confirmed:
            raise PipelineStateError(
                "role_confirmation_required",
                "Roles must be explicitly confirmed before analysis.",
            )
        ordered = _ordered_utterances(roles)
        if stage == StageName.EMOTION:
            normalized = self._read(job_id, StageName.NORMALIZE, NormalizeArtifact)
            customer_utterances = [item for item in ordered if item.role == Role.CUSTOMER]
            raw_results = []
            for item in customer_utterances:
                with self._emotion_clip(job_id, normalized, item) as audio_path:
                    self._diagnostic(
                        job_id,
                        stage,
                        "request",
                        {
                            "utterance_id": item.id,
                            "start_seconds": item.start,
                            "end_seconds": item.end,
                        },
                    )
                    emotion_result = self._emotion.analyze(item.id, audio_path, item.transcript)
                    self._diagnostic(
                        job_id,
                        stage,
                        "response",
                        {"utterance_id": item.id, "result_type": "emotion_result"},
                    )
                    raw_results.append(emotion_result)
            return EmotionArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                results=median_smooth(raw_results),
            )
        if stage == StageName.RESPONSE_STRATEGY:
            relevant_ids = set(_relevant_agent_ids(ordered))
            relevant_agents = [item for item in ordered if item.id in relevant_ids]
            strategy_results = []
            for item in relevant_agents:
                self._diagnostic(
                    job_id,
                    stage,
                    "request",
                    {"utterance_id": item.id, "context_count": len(ordered)},
                )
                strategy_result = self._strategy.classify(item, ordered)
                self._diagnostic(
                    job_id,
                    stage,
                    "response",
                    {"utterance_id": item.id, "result_type": "strategy_result"},
                )
                strategy_results.append(strategy_result)
            return StrategyArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                results=strategy_results,
            )
        emotions = self._read(job_id, StageName.EMOTION, EmotionArtifact)
        if stage == StageName.TRANSITIONS:
            return TransitionsArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                results=build_transitions(ordered, emotions.results),
            )
        if stage == StageName.REPORT:
            strategies = self._read(job_id, StageName.RESPONSE_STRATEGY, StrategyArtifact)
            transitions = self._read(job_id, StageName.TRANSITIONS, TransitionsArtifact)
            customers = [item for item in ordered if item.role == Role.CUSTOMER]
            summary = build_call_summary(customers, emotions.results, transitions.results)
            report = AnalysisReport(
                job_id=job_id,
                summary=summary,
                utterances=ordered,
                emotions=emotions.results,
                strategies=strategies.results,
                transitions=transitions.results,
            )
            self._diagnostic(
                job_id,
                stage,
                "request",
                {
                    "utterance_count": len(ordered),
                    "transition_count": len(transitions.results),
                },
            )
            narrative = self._report.summarize(report)
            if not isinstance(narrative, str):
                raise ValueError("report summary provider returned an invalid type")
            self._diagnostic(
                job_id,
                stage,
                "response",
                {"result_type": "summary", "character_count": len(narrative)},
            )
            summary_with_narrative = CallSummary.model_validate(
                {**report.summary.model_dump(mode="python"), "narrative": narrative}
            )
            return ReportArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                report=AnalysisReport.model_validate(
                    {
                        **report.model_dump(mode="python"),
                        "summary": summary_with_narrative,
                    }
                ),
            )
        raise KeyError(stage)

    def _execute_stage(self, job_id: str, stage: StageName) -> bool:
        with self._artifacts.operation_lock(job_id):
            claim = self._repository.claim_stage(job_id, stage)
        if claim is None:
            return False
        started = time.monotonic()
        heartbeat_stopped, heartbeat_thread = self._start_claim_heartbeat(claim)
        try:
            self._log(job_id, stage, "started", started)
            artifact = self._build_artifact(job_id, stage)
            artifact = type(artifact).model_validate(artifact.model_dump(mode="python"))
            if not self._artifact_semantics_are_valid(job_id, artifact):
                raise ValueError("stage artifact does not match its upstream contracts")
            prepared = self._artifacts.prepare_model(job_id, stage, artifact)
            status = (
                StageStatus.PAUSED if stage == StageName.CONFIRM_ROLES else StageStatus.COMPLETED
            )
            try:
                published = self._repository.publish_claimed_stage(
                    claim,
                    status=status,
                    artifact_path=prepared.target.name,
                    cache_key=artifact.cache_key,
                    artifact_hash=prepared.content_hash,
                    role_confirmed=False,
                    publish=prepared.publish,
                )
            finally:
                prepared.discard()
            if not published:
                return False
            self._log(job_id, stage, status.value, started)
            return True
        except (
            AudioRejected,
            InsufficientEmotionCoverage,
            PipelineValidationError,
            ValidationError,
            ValueError,
        ) as error:
            public = self._public_failure(stage, error)
            self._repository.fail_claimed_stage(
                claim,
                {"code": public.code, "message": public.message},
            )
            self._log(job_id, stage, "failed", started, public.code)
            return False
        except Exception as error:
            error_class = type(error).__name__
            self._repository.fail_claimed_stage(
                claim,
                {
                    "code": error_class,
                    "message": "An unexpected pipeline error occurred.",
                },
            )
            self._log(job_id, stage, "failed", started, error_class)
            raise
        finally:
            heartbeat_stopped.set()
            heartbeat_thread.join()

    def _run_locked(self, job_id: str) -> dict[str, object]:
        for stage in STAGE_ORDER:
            job = self._repository.get_job(job_id)
            row = _stage_rows(job)[stage.value]
            status = row.get("status")
            if status == StageStatus.COMPLETED.value:
                if self._completed_artifact_is_valid(
                    job_id,
                    stage,
                    row,
                    require_confirmed_role=stage == StageName.CONFIRM_ROLES,
                ):
                    started = time.monotonic()
                    self._log(job_id, stage, "cache_hit", started)
                    continue
                self._invalidate_from(job_id, stage)
                status = StageStatus.PENDING.value
            elif status == StageStatus.PAUSED.value:
                if stage == StageName.CONFIRM_ROLES and self._completed_artifact_is_valid(
                    job_id,
                    stage,
                    row,
                    require_confirmed_role=False,
                ):
                    return job
                self._invalidate_from(job_id, stage)
                status = StageStatus.PENDING.value
            if status == StageStatus.FAILED.value:
                return self._repository.get_job(job_id)
            if status not in {
                StageStatus.PENDING.value,
                StageStatus.RUNNING.value,
            }:
                self._invalidate_from(job_id, stage)
            if not self._execute_stage(job_id, stage):
                return self._repository.get_job(job_id)
            if stage == StageName.CONFIRM_ROLES:
                return self._repository.get_job(job_id)
        return self._repository.get_job(job_id)

    def run_until_pause(self, job_id: str) -> dict[str, object]:
        """Run pending work, validating every completed cache before skipping it."""

        self._repository.get_job(job_id)
        with self._job_lock(job_id):
            return self._run_locked(job_id)

    def confirm_roles(
        self,
        job_id: str,
        mapping: Mapping[str, Role | str],
    ) -> dict[str, object]:
        """Atomically publish exactly one customer/agent assignment, then resume."""

        self._repository.get_job(job_id)
        with self._job_lock(job_id):
            job = self._repository.get_job(job_id)
            row = _stage_rows(job)[StageName.CONFIRM_ROLES.value]
            if row.get(
                "status"
            ) != StageStatus.PAUSED.value or not self._completed_artifact_is_valid(
                job_id,
                StageName.CONFIRM_ROLES,
                row,
                require_confirmed_role=False,
            ):
                if row.get("status") == StageStatus.PAUSED.value:
                    raise PipelineValidationError(
                        "invalid_role_candidate",
                        "The paused role candidate no longer matches transcription.",
                    )
                raise PipelineStateError(
                    "role_confirmation_not_ready",
                    "Role confirmation is available only while the role stage is paused.",
                )
            candidate = self._read(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
            transcribe_row = _stage_rows(job)[StageName.TRANSCRIBE.value]
            if not self._completed_artifact_is_valid(
                job_id,
                StageName.TRANSCRIBE,
                transcribe_row,
                require_confirmed_role=False,
            ):
                raise PipelineValidationError(
                    "invalid_role_candidate",
                    "The paused role candidate no longer matches transcription.",
                )
            transcribed = self._read(job_id, StageName.TRANSCRIBE, TranscribeArtifact)
            observed = {utterance.speaker_id for utterance in transcribed.utterances}
            try:
                normalized_mapping = {
                    speaker_id: value if isinstance(value, Role) else Role(value)
                    for speaker_id, value in mapping.items()
                }
            except (TypeError, ValueError):
                normalized_mapping = {}
            if (
                len(observed) != 2
                or set(normalized_mapping) != observed
                or sorted(normalized_mapping.values()) != [Role.AGENT, Role.CUSTOMER]
            ):
                raise PipelineValidationError(
                    "invalid_role_mapping",
                    "Map exactly the two observed speakers to one customer and one agent.",
                )
            confirmed_utterances = [
                Utterance.model_validate(
                    {
                        **utterance.model_dump(mode="python"),
                        "role": normalized_mapping[utterance.speaker_id],
                    }
                )
                for utterance in transcribed.utterances
            ]
            confirmed = RoleArtifact(
                cache_key=candidate.cache_key,
                upstream_hashes=candidate.upstream_hashes,
                provider=None,
                utterances=confirmed_utterances,
                suggestion=suggest_roles(transcribed.utterances),
                confirmed=True,
                mapping=normalized_mapping,
            )
            cache_key, upstream_hashes = self._cache_key(job_id, StageName.CONFIRM_ROLES, confirmed)
            confirmed = RoleArtifact.model_validate(
                {
                    **confirmed.model_dump(mode="python"),
                    "cache_key": cache_key,
                    "upstream_hashes": upstream_hashes,
                }
            )
            prepared = self._artifacts.prepare_model(job_id, StageName.CONFIRM_ROLES, confirmed)
            generation = row.get("generation")
            candidate_hash = row.get("artifact_hash")
            if (
                isinstance(generation, bool)
                or not isinstance(generation, int)
                or not isinstance(candidate_hash, str)
            ):
                prepared.discard()
                raise PipelineValidationError(
                    "invalid_role_candidate",
                    "The paused role candidate metadata is invalid.",
                )
            try:
                with self._artifacts.operation_lock(job_id):
                    published = self._repository.publish_role_confirmation(
                        job_id,
                        expected_generation=generation,
                        expected_candidate_hash=candidate_hash,
                        artifact_path=prepared.target.name,
                        cache_key=cache_key,
                        artifact_hash=prepared.content_hash,
                        publish=prepared.publish,
                    )
            finally:
                prepared.discard()
            if not published:
                raise PipelineStateError(
                    "stale_role_candidate",
                    "Role confirmation raced with newer pipeline work.",
                )
            return self._run_locked(job_id)

    def retry(self, job_id: str, stage: StageName) -> dict[str, object]:
        """Invalidate the selected stage and downstream JSONs, then resume safely."""

        if not isinstance(stage, StageName):
            raise PipelineValidationError("invalid_stage", "Select a valid pipeline stage.")
        self._repository.get_job(job_id)
        with self._job_lock(job_id):
            self._invalidate_from(job_id, stage)
            return self._run_locked(job_id)


__all__ = ["PipelineRunner", "PipelineStateError", "PipelineValidationError"]
