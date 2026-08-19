"""Lazy local Wav2Vec2 XLS-R seven-emotion adapter."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from voxdelta.domain.models import EmotionResult, ProviderProvenance
from voxdelta.providers._emotion_runtime import (
    LOCAL_EMOTION_INFERENCE_LOCK,
    Device,
    Predictor,
    activate_candidate,
    analyze_local_emotion,
    default_hardware_probe,
    default_inference_context,
    default_rss_probe,
    prepare_candidate_load,
    release_candidate,
    select_device,
    validate_checkpoint,
)
from voxdelta.providers.base import ProviderError

MODEL_ID = "facebook/wav2vec2-xls-r-300m"


class ModelFactory(Protocol):
    def __call__(self, checkpoint: Path, *, model_id: str, device: str) -> Predictor: ...


class _TransformersPredictor:
    def __init__(self, checkpoint: Path, model_id: str, device: str) -> None:
        torch = import_module("torch")
        transformers = import_module("transformers")
        safetensors = import_module("safetensors.torch")
        self._torch = torch
        self._device = device
        self._extractor = transformers.AutoFeatureExtractor.from_pretrained(model_id)
        self._model = transformers.AutoModelForAudioClassification.from_pretrained(
            model_id,
            num_labels=7,
            id2label={
                0: "happiness",
                1: "anger",
                2: "disgust",
                3: "fear",
                4: "neutral",
                5: "sadness",
                6: "surprise",
            },
            label2id={
                "happiness": 0,
                "anger": 1,
                "disgust": 2,
                "fear": 3,
                "neutral": 4,
                "sadness": 5,
                "surprise": 6,
            },
            ignore_mismatched_sizes=True,
        )
        self._model.load_state_dict(safetensors.load_file(str(checkpoint / "model.safetensors")))
        self._model.to(device)
        self._model.eval()

    def predict(self, samples: tuple[float, ...], sample_rate: int) -> list[float]:
        inputs = self._extractor(list(samples), sampling_rate=sample_rate, return_tensors="pt")
        prepared = {name: value.to(self._device) for name, value in inputs.items()}
        output = self._model(**prepared)
        return cast(list[float], output.logits[0].detach().float().cpu().tolist())


def _default_factory(checkpoint: Path, *, model_id: str, device: str) -> Predictor:
    return _TransformersPredictor(checkpoint, model_id, device)


class Wav2VecEmotionProvider:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: Device = "auto",
        model_factory: ModelFactory | None = None,
        hardware_probe: Callable[[], tuple[bool, bool]] | None = None,
        inference_context: Callable[[], AbstractContextManager[object]] | None = None,
        clock: Callable[[], float] | None = None,
        rss_probe: Callable[[], float | None] | None = None,
    ) -> None:
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ProviderError("provider_runtime_unsupported")
        self._checkpoint = validate_checkpoint(
            checkpoint_path, architecture="wav2vec-xls-r", model_id=MODEL_ID
        )
        self.provenance = ProviderProvenance(
            name="wav2vec-xls-r",
            model="wav2vec2-xls-r-300m-seven-emotion",
            remote=False,
            revision=self._checkpoint.digest,
        )
        self._requested_device = device
        self._factory = model_factory or _default_factory
        self._hardware_probe = hardware_probe or default_hardware_probe
        self._inference_context = inference_context or default_inference_context
        self._clock = clock
        self._rss_probe = rss_probe or default_rss_probe
        self._predictor: Predictor | None = None

    def _load(self) -> Predictor:
        if self._predictor is not None:
            return self._predictor
        device = select_device(self._requested_device, self._hardware_probe)
        prepare_candidate_load(self)
        try:
            self._predictor = self._factory(self._checkpoint.path, model_id=MODEL_ID, device=device)
        except ProviderError:
            raise
        except Exception as error:
            if isinstance(error, TimeoutError):
                raise ProviderError("provider_timeout") from None
            raise ProviderError("provider_unavailable") from None
        activate_candidate(self, self._release)
        return self._predictor

    def _release(self) -> None:
        predictor = self._predictor
        self._predictor = None
        close = getattr(predictor, "close", None)
        if callable(close):
            close()

    def unload(self) -> None:
        with LOCAL_EMOTION_INFERENCE_LOCK:
            self._release()
            release_candidate(self)

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        kwargs: dict[str, Any] = {
            "utterance_id": utterance_id,
            "audio_path": audio_path,
            "transcript": transcript,
            "provenance": self.provenance,
            "load_predictor": self._load,
            "inference_context": self._inference_context,
            "rss_probe": self._rss_probe,
        }
        if self._clock is not None:
            kwargs["clock"] = self._clock
        return analyze_local_emotion(**kwargs)
