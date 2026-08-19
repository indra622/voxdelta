from __future__ import annotations

import importlib
import json
import struct
import sys
import wave
from pathlib import Path

import pytest

from voxdelta.providers.base import EmotionProvider, ProviderError

LABELS = ("happiness", "anger", "disgust", "fear", "neutral", "sadness", "surprise")


def _wav(path: Path, seconds: float) -> Path:
    frames = max(1, round(seconds * 16_000))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(struct.pack(f"<{frames}h", *([100] * frames)))
    return path


class FakePredictor:
    def __init__(self, outputs: list[list[float]] | BaseException) -> None:
        self.outputs = outputs
        self.calls: list[tuple[int, int]] = []

    def predict(self, samples: tuple[float, ...], sample_rate: int) -> list[float]:
        self.calls.append((len(samples), sample_rate))
        if isinstance(self.outputs, BaseException):
            raise self.outputs
        return self.outputs[len(self.calls) - 1]


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "architecture": "emotion2vec-plus",
                "model_id": "iic/emotion2vec_plus_large",
                "encoder_revision": "v2.0.4",
                "encoder_hash": "b" * 64,
                "embedding_size": 4,
                "freeze_encoder": True,
                "labels": list(LABELS),
            }
        ),
        encoding="utf-8",
    )
    (path / "label_mapping.json").write_text(
        json.dumps({str(index): label for index, label in enumerate(LABELS)}), encoding="utf-8"
    )
    (path / "metrics.json").write_text(
        json.dumps({"macro_f1": 0.75, "validation_hash": "c" * 64}), encoding="utf-8"
    )
    (path / "model.safetensors").write_bytes(b"seven-class-head")
    return path


class FakeFactory:
    def __init__(self, predictor: FakePredictor) -> None:
        self.predictor = predictor
        self.calls: list[tuple[Path, str, str, str, str, bool]] = []

    def __call__(
        self,
        checkpoint: Path,
        *,
        encoder_id: str,
        encoder_revision: str,
        encoder_hash: str,
        device: str,
        freeze_encoder: bool,
    ) -> FakePredictor:
        self.calls.append(
            (checkpoint, encoder_id, encoder_revision, encoder_hash, device, freeze_encoder)
        )
        return self.predictor


def test_import_is_lazy_and_default_encoder_is_exact(tmp_path: Path) -> None:
    before = set(sys.modules)
    module = importlib.import_module("voxdelta.providers.emotion2vec_emotion")

    added = set(sys.modules) - before
    assert "torch" not in added
    assert "funasr" not in added
    factory = FakeFactory(FakePredictor([[0.0] * 7]))
    provider = module.Emotion2VecEmotionProvider(
        _checkpoint(tmp_path / "checkpoint"),
        model_factory=factory,
        hardware_probe=lambda: (False, False),
    )
    assert isinstance(provider, EmotionProvider)
    assert factory.calls == []
    provider.analyze("utt", _wav(tmp_path / "audio.wav", 1), "unused")
    assert factory.calls == [
        (
            (tmp_path / "checkpoint").resolve(),
            "iic/emotion2vec_plus_large",
            "v2.0.4",
            "b" * 64,
            "cpu",
            True,
        )
    ]
    assert provider.provenance.name == "emotion2vec-plus"


def test_pretrained_nine_class_logits_are_never_remapped(tmp_path: Path) -> None:
    from voxdelta.providers.emotion2vec_emotion import Emotion2VecEmotionProvider

    provider = Emotion2VecEmotionProvider(
        _checkpoint(tmp_path / "checkpoint"),
        model_factory=FakeFactory(FakePredictor([[0.0] * 9])),
        hardware_probe=lambda: (False, False),
    )
    with pytest.raises(ProviderError) as raised:
        provider.analyze("utt", _wav(tmp_path / "audio.wav", 1), "")
    assert raised.value.code == "invalid_provider_output"


def test_emotion2vec_uses_shared_seven_label_result_mapping(tmp_path: Path) -> None:
    from voxdelta.providers.emotion2vec_emotion import Emotion2VecEmotionProvider

    provider = Emotion2VecEmotionProvider(
        _checkpoint(tmp_path / "checkpoint"),
        model_factory=FakeFactory(FakePredictor([[0, 3, 0, 0, 0, 0, 0]])),
        hardware_probe=lambda: (False, False),
        clock=lambda: 2.0,
        rss_probe=lambda: 12.0,
    )
    result = provider.analyze("utterance", _wav(tmp_path / "audio.wav", 1), "SECRET")

    assert tuple(result.probabilities) == LABELS
    assert result.confidence == result.probabilities["anger"]
    assert result.operational_state == "escalated"
    assert result.negative_intensity == pytest.approx(
        sum(result.probabilities[name] for name in ("anger", "disgust", "fear", "sadness"))
    )
    assert "SECRET" not in result.model_dump_json()


def test_malformed_checkpoint_metadata_is_safe(tmp_path: Path) -> None:
    from voxdelta.providers.emotion2vec_emotion import Emotion2VecEmotionProvider

    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "architecture": "emotion2vec-plus",
                "model_id": "wrong",
                "encoder_revision": "v2.0.4",
                "encoder_hash": "not-sha",
                "embedding_size": 4,
                "freeze_encoder": False,
                "labels": list(LABELS),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProviderError) as raised:
        Emotion2VecEmotionProvider(
            checkpoint, model_factory=FakeFactory(FakePredictor([[0.0] * 7]))
        )
    assert raised.value.code == "invalid_local_checkpoint"


def test_factory_failure_is_sanitized(tmp_path: Path) -> None:
    from voxdelta.providers.emotion2vec_emotion import Emotion2VecEmotionProvider

    provider = Emotion2VecEmotionProvider(
        _checkpoint(tmp_path / "checkpoint"),
        model_factory=FakeFactory(FakePredictor(RuntimeError("/private transcript token"))),
        hardware_probe=lambda: (False, False),
    )
    with pytest.raises(ProviderError) as raised:
        provider.analyze("utt", _wav(tmp_path / "audio.wav", 1), "secret")
    assert raised.value.code == "provider_unavailable"
    assert "private" not in str(raised.value).lower()


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_default_encoder_factory_passes_device_and_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    from voxdelta.providers import emotion2vec_emotion as module

    calls: list[dict[str, object]] = []

    class Funasr:
        @staticmethod
        def AutoModel(**kwargs: object) -> object:
            calls.append(kwargs)
            return object()

    monkeypatch.setattr(module, "import_module", lambda name: Funasr())
    module._default_encoder_factory("iic/emotion2vec_plus_large", revision="v2.0.4", device=device)

    assert calls == [
        {
            "model": "iic/emotion2vec_plus_large",
            "model_revision": "v2.0.4",
            "device": device,
            "disable_update": True,
        }
    ]


def test_encoder_hash_mismatch_fails_safely() -> None:
    from voxdelta.providers.emotion2vec_emotion import _verify_encoder_identity

    with pytest.raises(ProviderError) as raised:
        _verify_encoder_identity(object(), expected_hash="a" * 64, hasher=lambda _encoder: "b" * 64)

    assert raised.value.code == "invalid_local_checkpoint"
