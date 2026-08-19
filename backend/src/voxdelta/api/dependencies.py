"""Construct production dependencies without loading credentials or remote models."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import SecretStr

from voxdelta.audio.service import AudioService
from voxdelta.config import Settings
from voxdelta.credentials import Credentials, load_credentials
from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.repository import JobRepository
from voxdelta.pipeline.runner import PipelineRunner
from voxdelta.providers.base import (
    DiarizationProvider,
    EmotionProvider,
    TranscriptionProvider,
)
from voxdelta.providers.emotion2vec_emotion import (
    Emotion2VecEmotionProvider,
)
from voxdelta.providers.emotion2vec_emotion import (
    ModelFactory as Emotion2VecFactory,
)
from voxdelta.providers.fake import (
    FakeDiarizationProvider,
    FakeEmotionProvider,
    FakeTranscriptionProvider,
)
from voxdelta.providers.faster_whisper_asr import (
    FasterWhisperProvider,
)
from voxdelta.providers.faster_whisper_asr import (
    ModelFactory as FasterWhisperFactory,
)
from voxdelta.providers.pyannote_diarization import (
    PipelineFactory,
    PyannoteDiarizationProvider,
)
from voxdelta.providers.qwen3_asr import (
    AlignerFactory,
    Qwen3AsrProvider,
)
from voxdelta.providers.qwen3_asr import (
    ModelFactory as QwenModelFactory,
)
from voxdelta.providers.wav2vec_emotion import (
    ModelFactory as Wav2VecFactory,
)
from voxdelta.providers.wav2vec_emotion import (
    Wav2VecEmotionProvider,
)


class ProviderConfigurationError(RuntimeError):
    """Fixed, path-free startup failure for invalid provider selection."""

    def __init__(self) -> None:
        super().__init__("provider_configuration_invalid")


@dataclass(frozen=True, slots=True)
class ProviderFactories:
    """Test-only lazy model construction seams; production leaves every field unset."""

    pyannote_pipeline: PipelineFactory | None = None
    faster_whisper_model: FasterWhisperFactory | None = None
    qwen_model: QwenModelFactory | None = None
    qwen_aligner: AlignerFactory | None = None
    wav2vec_model: Wav2VecFactory | None = None
    emotion2vec_model: Emotion2VecFactory | None = None


@dataclass(frozen=True, slots=True)
class ApiDependencies:
    repository: JobRepository
    artifacts: ArtifactStore
    runner: PipelineRunner
    max_upload_bytes: int
    admission_reconciliation_lease_seconds: int
    max_active_jobs: int
    api_capability_token: SecretStr | None


def build_dependencies(
    settings: Settings | None = None,
    *,
    credentials: Credentials | None = None,
    provider_factories: ProviderFactories | None = None,
) -> ApiDependencies:
    selected = settings or Settings()
    factories = provider_factories or ProviderFactories()
    try:
        diarization: DiarizationProvider
        transcription: TranscriptionProvider
        emotion: EmotionProvider
        if selected.diarization_provider == "fake":
            diarization = FakeDiarizationProvider()
        else:
            selected_credentials = credentials
            if selected_credentials is None:
                selected_credentials = load_credentials()
            if (
                selected.pyannote_checkpoint_path is None
                and selected_credentials.huggingface_token is None
            ):
                raise ProviderConfigurationError()
            diarization = PyannoteDiarizationProvider(
                selected_credentials,
                model_path=selected.pyannote_checkpoint_path,
                pipeline_factory=factories.pyannote_pipeline,
            )

        if selected.asr_provider == "fake":
            transcription = FakeTranscriptionProvider()
        elif selected.asr_provider == "faster-whisper":
            if selected.asr_device == "mps":
                raise ProviderConfigurationError()
            transcription = FasterWhisperProvider(
                device=selected.asr_device if selected.asr_device != "auto" else "cpu",
                model_factory=factories.faster_whisper_model,
            )
        else:
            transcription = Qwen3AsrProvider(
                profile=selected.qwen_profile,
                device=selected.asr_device,
                model_factory=factories.qwen_model,
                aligner_factory=factories.qwen_aligner,
            )

        if selected.emotion_provider == "fake":
            emotion = FakeEmotionProvider()
        else:
            if selected.emotion_checkpoint_path is None:
                raise ProviderConfigurationError()
            if selected.emotion_provider == "wav2vec":
                emotion = Wav2VecEmotionProvider(
                    selected.emotion_checkpoint_path,
                    device=selected.emotion_device,
                    model_factory=factories.wav2vec_model,
                )
            else:
                emotion = Emotion2VecEmotionProvider(
                    selected.emotion_checkpoint_path,
                    device=selected.emotion_device,
                    model_factory=factories.emotion2vec_model,
                )
    except ProviderConfigurationError:
        raise
    except Exception:
        raise ProviderConfigurationError() from None

    jobs_root = selected.data_root / "jobs"
    repository = JobRepository(selected.database_path)
    artifacts = ArtifactStore(jobs_root)
    audio = AudioService(
        jobs_root,
        min_seconds=selected.min_audio_seconds,
        max_seconds=selected.max_audio_seconds,
    )
    return ApiDependencies(
        repository=repository,
        artifacts=artifacts,
        runner=PipelineRunner(
            repository,
            artifacts,
            audio,
            diarization_provider=diarization,
            transcription_provider=transcription,
            emotion_provider=emotion,
        ),
        max_upload_bytes=selected.max_upload_bytes,
        admission_reconciliation_lease_seconds=selected.admission_reconciliation_lease_seconds,
        max_active_jobs=selected.max_active_jobs,
        api_capability_token=selected.api_capability_token,
    )


__all__ = [
    "ApiDependencies",
    "ProviderConfigurationError",
    "ProviderFactories",
    "build_dependencies",
]
