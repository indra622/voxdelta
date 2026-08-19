"""Resumable, cache-validating orchestration for the local VoxDelta pipeline."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from threading import Lock, RLock
from typing import TypeVar, cast

from pydantic import ValidationError

from voxdelta.analysis.emotions import median_smooth
from voxdelta.analysis.roles import suggest_roles
from voxdelta.analysis.summary import InsufficientEmotionCoverage, build_call_summary
from voxdelta.analysis.transitions import build_transitions
from voxdelta.audio.service import AudioRejected, AudioService, ChannelPreference
from voxdelta.domain.models import (
    AnalysisReport,
    ProviderProvenance,
    Role,
    StageName,
    StageStatus,
    Utterance,
)
from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.logging import PipelineLogger
from voxdelta.jobs.repository import JobRepository
from voxdelta.pipeline.stages import (
    ARTIFACT_MODELS,
    STAGE_ORDER,
    UPSTREAM_STAGES,
    DiarizeArtifact,
    EmotionArtifact,
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
        config: Mapping[str, Mapping[str, object]] | None = None,
        logger: PipelineLogger | None = None,
    ) -> None:
        self._repository = repository
        self._artifacts = artifacts
        self._audio = audio_service
        self._diarization = diarization_provider or FakeDiarizationProvider()
        self._transcription = transcription_provider or FakeTranscriptionProvider()
        self._emotion = emotion_provider or FakeEmotionProvider()
        self._strategy = strategy_provider or FakeResponseStrategyProvider()
        self._report = report_provider or FakeReportSummaryProvider()
        self._config = {stage: dict((config or {}).get(stage.value, {})) for stage in STAGE_ORDER}
        self._logger = logger or PipelineLogger(artifacts)
        self._locks: dict[str, RLock] = {}
        self._locks_guard = Lock()

    def _job_lock(self, job_id: str) -> RLock:
        with self._locks_guard:
            return self._locks.setdefault(job_id, RLock())

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
        config.update(self._config[stage])
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

    def _artifact_semantics_are_valid(self, job_id: str, artifact: StageArtifact) -> bool:
        if isinstance(artifact, NormalizeArtifact):
            return bool(artifact.asset.normalized_paths) and all(
                Path(path).is_file() for path in artifact.asset.normalized_paths
            )
        if isinstance(artifact, RoleArtifact):
            speakers = {utterance.speaker_id for utterance in artifact.utterances}
            if len(speakers) != 2:
                return False
            if not artifact.confirmed:
                return all(utterance.role == Role.UNKNOWN for utterance in artifact.utterances)
            if artifact.mapping is None or set(artifact.mapping) != speakers:
                return False
            if sorted(artifact.mapping.values()) != [Role.AGENT, Role.CUSTOMER]:
                return False
            return all(
                utterance.role == artifact.mapping[utterance.speaker_id]
                for utterance in artifact.utterances
            )
        if isinstance(artifact, ReportArtifact):
            return artifact.report.job_id == job_id
        return True

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
            if artifact.provider != self._provider(stage):
                return False
            role_artifact = artifact if isinstance(artifact, RoleArtifact) else None
            if require_confirmed_role and (role_artifact is None or not role_artifact.confirmed):
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
            return NormalizeArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                asset=asset,
            )
        if stage == StageName.DIARIZE:
            normalized = self._read(job_id, StageName.NORMALIZE, NormalizeArtifact)
            return DiarizeArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                segments=self._diarization.diarize(normalized.asset),
            )
        if stage == StageName.TRANSCRIBE:
            normalized = self._read(job_id, StageName.NORMALIZE, NormalizeArtifact)
            diarized = self._read(job_id, StageName.DIARIZE, DiarizeArtifact)
            return TranscribeArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                utterances=self._transcription.transcribe(normalized.asset, diarized.segments),
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
            audio_path = Path(normalized.asset.normalized_paths[0])
            customer_utterances = [item for item in ordered if item.role == Role.CUSTOMER]
            raw_results = [
                self._emotion.analyze(item.id, audio_path, item.transcript)
                for item in customer_utterances
            ]
            return EmotionArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                results=median_smooth(raw_results),
            )
        if stage == StageName.RESPONSE_STRATEGY:
            relevant_agents = [
                item
                for index, item in enumerate(ordered)
                if 0 < index < len(ordered) - 1
                and item.role == Role.AGENT
                and ordered[index - 1].role == Role.CUSTOMER
                and ordered[index + 1].role == Role.CUSTOMER
            ]
            return StrategyArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                results=[self._strategy.classify(item, ordered) for item in relevant_agents],
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
            narrative = self._report.summarize(report)
            return ReportArtifact(
                cache_key=cache_key,
                upstream_hashes=upstream_hashes,
                provider=provider,
                report=report.model_copy(
                    update={"summary": report.summary.model_copy(update={"narrative": narrative})}
                ),
            )
        raise KeyError(stage)

    def _execute_stage(self, job_id: str, stage: StageName) -> bool:
        if not self._repository.try_start_stage(job_id, stage):
            return False
        started = time.monotonic()
        try:
            self._log(job_id, stage, "started", started)
            artifact = self._build_artifact(job_id, stage)
            path = self._artifacts.write_model(job_id, stage, artifact)
            status = (
                StageStatus.PAUSED if stage == StageName.CONFIRM_ROLES else StageStatus.COMPLETED
            )
            self._repository.set_stage(
                job_id,
                stage,
                status,
                path.name,
                cache_key=artifact.cache_key,
            )
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
            self._repository.set_stage(
                job_id,
                stage,
                StageStatus.FAILED,
                error={"code": public.code, "message": public.message},
            )
            self._log(job_id, stage, "failed", started, public.code)
            return False
        except Exception as error:
            error_class = type(error).__name__
            self._repository.set_stage(
                job_id,
                stage,
                StageStatus.FAILED,
                error={
                    "code": error_class,
                    "message": "An unexpected pipeline error occurred.",
                },
            )
            self._log(job_id, stage, "failed", started, error_class)
            raise

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
            if status in {StageStatus.RUNNING.value, StageStatus.FAILED.value}:
                return self._repository.get_job(job_id)
            if status != StageStatus.PENDING.value:
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
                raise PipelineStateError(
                    "role_confirmation_not_ready",
                    "Role confirmation is available only while the role stage is paused.",
                )
            candidate = self._read(job_id, StageName.CONFIRM_ROLES, RoleArtifact)
            observed = {utterance.speaker_id for utterance in candidate.utterances}
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
            confirmed = candidate.model_copy(
                update={
                    "utterances": [
                        utterance.model_copy(
                            update={"role": normalized_mapping[utterance.speaker_id]},
                            deep=True,
                        )
                        for utterance in candidate.utterances
                    ],
                    "confirmed": True,
                    "mapping": normalized_mapping,
                },
                deep=True,
            )
            cache_key, upstream_hashes = self._cache_key(job_id, StageName.CONFIRM_ROLES, confirmed)
            confirmed = confirmed.model_copy(
                update={"cache_key": cache_key, "upstream_hashes": upstream_hashes},
                deep=True,
            )
            path = self._artifacts.write_model(job_id, StageName.CONFIRM_ROLES, confirmed)
            self._repository.set_stage(
                job_id,
                StageName.CONFIRM_ROLES,
                StageStatus.COMPLETED,
                path.name,
                cache_key=cache_key,
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
