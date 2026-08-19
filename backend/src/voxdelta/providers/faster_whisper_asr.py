"""Lazy local faster-whisper adapter for stable Korean transcription."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import Literal, Protocol, cast

from voxdelta.domain.models import AudioAsset, ProviderProvenance, SpeakerSegment, Utterance
from voxdelta.providers.asr_alignment import (
    LOCAL_ASR_INFERENCE_LOCK,
    AlignedWord,
    align_mixed,
    align_separate,
    validate_asset,
    validated_words,
)
from voxdelta.providers.base import ProviderError

MODEL_ID = "large-v3-turbo"


class _WhisperWord(Protocol):
    start: object
    end: object
    word: object


class _WhisperSegment(Protocol):
    words: object


class _WhisperModel(Protocol):
    def transcribe(self, path: str, **kwargs: object) -> tuple[object, object]: ...


class ModelFactory(Protocol):
    def __call__(self, model_id: str, *, device: str, compute_type: str) -> _WhisperModel: ...


def _default_factory(model_id: str, *, device: str, compute_type: str) -> _WhisperModel:
    module = import_module("faster_whisper")
    model_class = module.WhisperModel
    return cast(_WhisperModel, model_class(model_id, device=device, compute_type=compute_type))


def _words(output: object, duration: float) -> list[AlignedWord]:
    try:
        segments, _info = cast(tuple[object, object], output)

        def records() -> Iterable[tuple[object, object, object]]:
            for raw_segment in cast(Iterable[object], segments):
                segment = cast(_WhisperSegment, raw_segment)
                raw_words = segment.words
                if raw_words is None:
                    raise ProviderError("invalid_provider_output")
                for raw_word in cast(Iterable[object], raw_words):
                    word = cast(_WhisperWord, raw_word)
                    yield word.start, word.end, word.word

        return validated_words(records(), duration)
    except ProviderError:
        raise
    except Exception:
        raise ProviderError("invalid_provider_output") from None


class FasterWhisperProvider:
    """Run the pinned stable ASR candidate with authoritative pyannote timing."""

    def __init__(
        self,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        model_id: str = MODEL_ID,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if model_id != MODEL_ID or device not in {"cpu", "cuda"}:
            raise ProviderError("provider_unavailable")
        self.provenance = ProviderProvenance(name="faster-whisper", model=MODEL_ID, remote=False)
        self._device = device
        self._compute_type = "float16" if device == "cuda" else "int8"
        self._factory = model_factory or _default_factory
        self._model: _WhisperModel | None = None
        self.last_warning_count = 0

    def _load(self) -> _WhisperModel:
        if self._model is not None:
            return self._model
        try:
            self._model = self._factory(
                MODEL_ID, device=self._device, compute_type=self._compute_type
            )
        except ProviderError:
            raise
        except Exception:
            raise ProviderError("provider_unavailable") from None
        return self._model

    def _transcribe_path(self, path: str, duration: float) -> list[AlignedWord]:
        try:
            output = self._load().transcribe(
                path,
                language="ko",
                word_timestamps=True,
                vad_filter=False,
                beam_size=5,
            )
            return _words(output, duration)
        except ProviderError:
            raise
        except TimeoutError:
            raise ProviderError("provider_timeout") from None
        except Exception:
            raise ProviderError("provider_unavailable") from None

    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]:
        duration, paths = validate_asset(asset)
        with LOCAL_ASR_INFERENCE_LOCK:
            if asset.channel_mode == "separate":
                result, omitted = align_separate(
                    [self._transcribe_path(path, duration) for path in paths]
                )
            else:
                result, omitted = align_mixed(
                    self._transcribe_path(paths[0], duration), segments, duration
                )
            self.last_warning_count = omitted
        return result
