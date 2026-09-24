"""Construct production dependencies without loading credentials or remote models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    ProviderError,
    ProviderErrorCode,
    TranscriptionProvider,
)
from voxdelta.providers.calibrated_emotion import CalibratedEmotionProvider
from voxdelta.providers.calibration_artifact import verify_calibration_artifact
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
from voxdelta.providers.fallback_asr import FallbackTranscriptionProvider
from voxdelta.providers.faster_whisper_asr import (
    FasterWhisperProvider,
)
from voxdelta.providers.faster_whisper_asr import (
    ModelFactory as FasterWhisperFactory,
)
from voxdelta.providers.nemotron_diarization import (
    NemotronDiarizationProvider,
)
from voxdelta.providers.nemotron_diarization import (
    Runner as NemotronRunner,
)
from voxdelta.providers.pyannote_diarization import (
    PipelineFactory,
    PyannoteDiarizationProvider,
)
from voxdelta.providers.pyannote_precision import (
    ClientFactory as PyannoteAiClientFactory,
)
from voxdelta.providers.pyannote_precision import (
    PyannotePrecisionProvider,
)
from voxdelta.providers.qwen3_asr import (
    AlignerFactory,
    Qwen3AsrProvider,
)
from voxdelta.providers.qwen3_asr import (
    ModelFactory as QwenModelFactory,
)
from voxdelta.providers.release_bundle import verify_release_bundle
from voxdelta.providers.wav2vec_emotion import (
    ModelFactory as Wav2VecFactory,
)
from voxdelta.providers.wav2vec_emotion import (
    Wav2VecEmotionProvider,
)


class ProviderConfigurationError(RuntimeError):
    """Fixed, path-free startup failure for invalid provider selection.

    ``provider_code`` optionally carries the typed provider code (for example
    ``local_runtime_missing``) so an operator can act on it; the message stays fixed.
    """

    def __init__(self, provider_code: ProviderErrorCode | None = None) -> None:
        self.provider_code = provider_code
        super().__init__("provider_configuration_invalid")


@dataclass(frozen=True, slots=True)
class ProviderFactories:
    """Test-only lazy model construction seams; production leaves every field unset."""

    pyannote_pipeline: PipelineFactory | None = None
    pyannoteai_client: PyannoteAiClientFactory | None = None
    faster_whisper_model: FasterWhisperFactory | None = None
    qwen_model: QwenModelFactory | None = None
    qwen_aligner: AlignerFactory | None = None
    wav2vec_model: Wav2VecFactory | None = None
    emotion2vec_model: Emotion2VecFactory | None = None
    nemotron_runner: NemotronRunner | None = None


@dataclass(frozen=True, slots=True)
class ApiDependencies:
    repository: JobRepository
    artifacts: ArtifactStore
    runner: PipelineRunner
    max_upload_bytes: int
    admission_reconciliation_lease_seconds: int
    max_active_jobs: int
    api_capability_token: SecretStr | None
    annotation_root: Path
    annotation_audio_root: Path


def default_annotation_root(settings: Settings) -> Path:
    """Where private annotation artifacts live when no location was configured.

    Silver and gold both hold verbatim transcript, so the default is a named directory
    under the data root rather than anything that could collide with job artifacts.
    """

    return settings.annotation_root or settings.data_root / "annotations"


def default_annotation_audio_root(settings: Settings) -> Path:
    """Where the recordings behind silver drafts are read from, in place.

    Both supported, locally-derived review collections live under this root. Resolution
    itself is still an explicit allowlist (KCSC, 022 finance, and the dated
    user-provided evaluation sets), rather
    than a recursive search or a path supplied by the browser.
    """

    return settings.annotation_audio_root or settings.data_root / "derived"


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
        elif selected.diarization_provider == "pyannoteai-precision":
            # The only provider that leaves this machine, so it is never a fallback: it is
            # built solely because it was named, and only when its own key is present.
            selected_credentials = credentials
            if selected_credentials is None:
                selected_credentials = load_credentials()
            if selected_credentials.pyannoteai_api_key is None:
                raise ProviderConfigurationError()
            diarization = PyannotePrecisionProvider(
                selected_credentials,
                client_factory=factories.pyannoteai_client,
            )
        elif selected.diarization_provider == "nemotron-3-local":
            # Local-only and opt-in: built solely because it was named, never as a
            # fallback, and a missing runtime or model is a startup error, not a switch.
            if selected.nemotron_executable_path is None or selected.nemotron_model_path is None:
                raise ProviderConfigurationError()
            try:
                diarization = NemotronDiarizationProvider(
                    executable_path=selected.nemotron_executable_path,
                    model_path=selected.nemotron_model_path,
                    device=selected.nemotron_device,
                    timeout_seconds=selected.nemotron_timeout_seconds,
                    runner=factories.nemotron_runner,
                )
            except ProviderError as error:
                raise ProviderConfigurationError(error.code) from None
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
            qwen = Qwen3AsrProvider(
                profile=selected.qwen_profile,
                device=selected.asr_device,
                model_factory=factories.qwen_model,
                aligner_factory=factories.qwen_aligner,
            )
            if selected.asr_fallback_provider == "none":
                transcription = qwen
            else:
                # The fallback is pinned to CPU rather than inheriting asr_device: it
                # exists for the case where the requested device rejected Qwen, so
                # requesting that same device again would defeat the point. faster-whisper
                # also refuses mps outright.
                transcription = FallbackTranscriptionProvider(
                    qwen,
                    FasterWhisperProvider(
                        device="cpu",
                        model_factory=factories.faster_whisper_model,
                    ),
                )

        if selected.emotion_provider == "fake":
            emotion = FakeEmotionProvider()
        elif selected.xlsr_release_enabled:
            if selected.emotion_provider != "wav2vec" or selected.xlsr_release_path is None:
                raise ProviderConfigurationError()
            release = verify_release_bundle(selected.xlsr_release_path)
            emotion = Wav2VecEmotionProvider(
                release.checkpoint_path,
                base_model_path=release.base_model_path,
                device=selected.emotion_device,
                model_factory=factories.wav2vec_model,
            )
            if selected.xlsr_calibration_enabled:
                if selected.xlsr_calibration_path is None:
                    raise ProviderConfigurationError()
                calibration = verify_calibration_artifact(
                    selected.xlsr_calibration_path,
                    release=release,
                )
                emotion = CalibratedEmotionProvider(emotion, calibration)
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
        annotation_root=default_annotation_root(selected),
        annotation_audio_root=default_annotation_audio_root(selected),
    )


__all__ = [
    "ApiDependencies",
    "ProviderConfigurationError",
    "ProviderFactories",
    "build_dependencies",
    "default_annotation_audio_root",
    "default_annotation_root",
]
