"""Lazy local emotion2vec+ encoder with a learned seven-emotion head."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from voxdelta.domain.models import EmotionResult, ProviderProvenance
from voxdelta.evaluation.emotion_training import emotion2vec_input, encoder_state_hash
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

ENCODER_ID = "iic/emotion2vec_plus_large"


class ModelFactory(Protocol):
    def __call__(
        self,
        checkpoint: Path,
        *,
        encoder_id: str,
        encoder_revision: str,
        encoder_hash: str,
        device: str,
        freeze_encoder: bool,
    ) -> Predictor: ...


class _Encoder(Protocol):
    def generate(self, **kwargs: object) -> object: ...


class _Emotion2VecPredictor:
    def __init__(
        self,
        checkpoint: Path,
        encoder_id: str,
        encoder_revision: str,
        expected_encoder_hash: str,
        device: str,
    ) -> None:
        try:
            torch = import_module("torch")
            safetensors = import_module("safetensors.torch")
            config = __import__("json").loads((checkpoint / "config.json").read_text())
            raw_embedding_size = config.get("embedding_size")
            if (
                isinstance(raw_embedding_size, bool)
                or not isinstance(raw_embedding_size, (int, float))
                or not float(raw_embedding_size).is_integer()
            ):
                raise ValueError
            embedding_size = int(raw_embedding_size)
            self._encoder = _default_encoder_factory(
                encoder_id, revision=encoder_revision, device=device
            )
            _verify_encoder_identity(
                self._encoder,
                expected_hash=expected_encoder_hash,
                hasher=lambda encoder: encoder_state_hash(encoder, torch),
            )
            self._device = device
            self._torch = torch
            self._head = torch.nn.Sequential(
                torch.nn.LayerNorm(embedding_size),
                torch.nn.Linear(embedding_size, 256),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(256, 7),
            )
            self._head.load_state_dict(safetensors.load_file(str(checkpoint / "model.safetensors")))
            self._head.to(device)
            self._head.eval()
        except ProviderError:
            raise
        except Exception:
            raise ProviderError("provider_runtime_unsupported") from None

    def predict(self, samples: tuple[float, ...], sample_rate: int) -> list[float]:
        del sample_rate
        try:
            output = self._encoder.generate(
                input=emotion2vec_input(samples),
                granularity="utterance",
                extract_embedding=True,
            )
            if not isinstance(output, list) or len(output) != 1 or not isinstance(output[0], dict):
                raise ValueError
            embedding = output[0].get("feats")
            tensor = self._torch.as_tensor(
                embedding, dtype=self._torch.float32, device=self._device
            )
            logits = self._head(tensor).reshape(-1)
            return cast(list[float], logits.detach().float().cpu().tolist())
        except Exception:
            raise ProviderError("invalid_provider_output") from None


def _default_factory(
    checkpoint: Path,
    *,
    encoder_id: str,
    encoder_revision: str,
    encoder_hash: str,
    device: str,
    freeze_encoder: bool,
) -> Predictor:
    if not freeze_encoder:
        raise ProviderError("invalid_local_checkpoint")
    return _Emotion2VecPredictor(checkpoint, encoder_id, encoder_revision, encoder_hash, device)


def _default_encoder_factory(encoder_id: str, *, revision: str, device: str) -> _Encoder:
    funasr = import_module("funasr")
    return cast(
        _Encoder,
        funasr.AutoModel(
            model=encoder_id,
            model_revision=revision,
            device=device,
            disable_update=True,
        ),
    )


def _verify_encoder_identity(
    encoder: object,
    *,
    expected_hash: str,
    hasher: Callable[[object], str],
) -> None:
    try:
        observed_hash = hasher(encoder)
    except Exception:
        raise ProviderError("invalid_local_checkpoint") from None
    if observed_hash != expected_hash:
        raise ProviderError("invalid_local_checkpoint")


class Emotion2VecEmotionProvider:
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
            checkpoint_path, architecture="emotion2vec-plus", model_id=ENCODER_ID
        )
        self.provenance = ProviderProvenance(
            name="emotion2vec-plus",
            model="emotion2vec-plus-large-seven-emotion@v2.0.5",
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
            self._predictor = self._factory(
                self._checkpoint.path,
                encoder_id=ENCODER_ID,
                encoder_revision=cast(str, self._checkpoint.encoder_revision),
                encoder_hash=cast(str, self._checkpoint.encoder_hash),
                device=device,
                freeze_encoder=True,
            )
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
