from __future__ import annotations

import importlib
import json
import math
import struct
import sys
import wave
from contextlib import contextmanager
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


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "architecture": "wav2vec-xls-r",
                "model_id": "facebook/wav2vec2-xls-r-300m",
                "labels": list(LABELS),
            }
        ),
        encoding="utf-8",
    )
    (path / "label_mapping.json").write_text(
        json.dumps({str(index): label for index, label in enumerate(LABELS)}),
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps({"macro_f1": 0.7, "validation_hash": "a" * 64}), encoding="utf-8"
    )
    (path / "model.safetensors").write_bytes(b"fake-weights")
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


class FakeFactory:
    def __init__(self, predictor: FakePredictor) -> None:
        self.predictor = predictor
        self.calls: list[tuple[Path, str, str]] = []

    def __call__(self, checkpoint: Path, *, model_id: str, device: str) -> FakePredictor:
        self.calls.append((checkpoint, model_id, device))
        return self.predictor


def test_import_is_lazy_and_provider_implements_protocol(tmp_path: Path) -> None:
    before = set(sys.modules)
    module = importlib.import_module("voxdelta.providers.wav2vec_emotion")

    assert "torch" not in set(sys.modules) - before
    assert "transformers" not in set(sys.modules) - before
    provider = module.Wav2VecEmotionProvider(
        _checkpoint(tmp_path / "checkpoint"),
        model_factory=FakeFactory(FakePredictor([[1.0] * 7])),
        hardware_probe=lambda: (False, False),
    )
    assert isinstance(provider, EmotionProvider)
    assert provider.provenance.name == "wav2vec-xls-r"
    assert provider.provenance.remote is False


def test_logits_are_meaned_before_softmax_and_use_canonical_mapping(tmp_path: Path) -> None:
    from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

    checkpoint = _checkpoint(tmp_path / "checkpoint")
    audio = _wav(tmp_path / "long.wav", 20.5)
    predictor = FakePredictor(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        ]
    )
    factory = FakeFactory(predictor)
    entered: list[bool] = []

    @contextmanager
    def inference_mode():
        entered.append(True)
        yield

    provider = Wav2VecEmotionProvider(
        checkpoint,
        model_factory=factory,
        hardware_probe=lambda: (False, True),
        inference_context=inference_mode,
        clock=lambda: 1.0,
        rss_probe=lambda: 321.0,
    )
    result = provider.analyze("utt-1", audio, "transcript-secret must stay unused")

    assert factory.calls == [(checkpoint.resolve(), "facebook/wav2vec2-xls-r-300m", "mps")]
    assert predictor.calls == [(320_000, 16_000), (8_000, 16_000)]
    assert entered == [True]
    assert tuple(result.probabilities) == LABELS
    expected = math.exp(2.0) / (math.exp(2.0) + math.exp(1.0) + 5)
    assert result.probabilities["happiness"] == pytest.approx(expected)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.confidence == max(result.probabilities.values())
    assert result.negative_intensity == pytest.approx(
        sum(result.probabilities[label] for label in ("anger", "disgust", "fear", "sadness"))
    )
    assert result.operational_state == "uncertain"
    assert result.usage is not None
    assert result.usage.peak_rss_mb == 321.0
    assert "transcript-secret" not in result.model_dump_json()


@pytest.mark.parametrize(
    "outputs",
    [
        [[0.0] * 6],
        [[0.0] * 8],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.nan]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.inf]],
    ],
)
def test_malformed_logits_are_safely_rejected(tmp_path: Path, outputs: list[list[float]]) -> None:
    from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

    provider = Wav2VecEmotionProvider(
        _checkpoint(tmp_path / "checkpoint"),
        model_factory=FakeFactory(FakePredictor(outputs)),
        hardware_probe=lambda: (False, False),
    )

    with pytest.raises(ProviderError) as raised:
        provider.analyze("utt", _wav(tmp_path / "call.wav", 1), "private transcript")

    assert raised.value.code == "invalid_provider_output"
    assert "private" not in str(raised.value).lower()


@pytest.mark.parametrize("seconds", [0.0, 0.4999375])
def test_short_audio_is_rejected_before_model_loading(tmp_path: Path, seconds: float) -> None:
    from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

    factory = FakeFactory(FakePredictor([[0.0] * 7]))
    provider = Wav2VecEmotionProvider(
        _checkpoint(tmp_path / "checkpoint"),
        model_factory=factory,
        hardware_probe=lambda: (False, False),
    )

    with pytest.raises(ProviderError) as raised:
        provider.analyze("utt", _wav(tmp_path / "short.wav", seconds), "")

    assert raised.value.code == "invalid_audio_asset"
    assert factory.calls == []


def test_invalid_audio_and_factory_exceptions_do_not_leak_paths_or_secrets(tmp_path: Path) -> None:
    from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

    sentinel = tmp_path / "private-transcript-secret.wav"
    sentinel.write_text("not-wave", encoding="utf-8")
    provider = Wav2VecEmotionProvider(
        _checkpoint(tmp_path / "checkpoint"),
        model_factory=FakeFactory(FakePredictor(RuntimeError(f"{sentinel} hf_token"))),
        hardware_probe=lambda: (False, False),
    )
    for path in (sentinel, _wav(tmp_path / "valid.wav", 1)):
        with pytest.raises(ProviderError) as raised:
            provider.analyze("utt", path, "transcript-secret")
        serialized = str(raised.value).lower()
        assert "private" not in serialized
        assert "secret" not in serialized


def test_checkpoint_symlinks_and_bad_label_mapping_fail_closed(tmp_path: Path) -> None:
    from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

    target = _checkpoint(tmp_path / "real")
    link = tmp_path / "checkpoint-secret"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ProviderError) as raised:
        Wav2VecEmotionProvider(link, model_factory=FakeFactory(FakePredictor([[0.0] * 7])))
    assert raised.value.code == "invalid_local_checkpoint"
    assert "secret" not in str(raised.value).lower()

    bad = _checkpoint(tmp_path / "bad")
    (bad / "label_mapping.json").write_text(json.dumps({"0": "joy"}), encoding="utf-8")
    with pytest.raises(ProviderError, match="checkpoint"):
        Wav2VecEmotionProvider(bad, model_factory=FakeFactory(FakePredictor([[0.0] * 7])))


def test_checkpoint_traversal_and_extra_metadata_are_rejected(tmp_path: Path) -> None:
    from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

    real = _checkpoint(tmp_path / "real")
    (tmp_path / "container").mkdir()
    traversing = Path(str(tmp_path / "container" / ".." / "real"))
    with pytest.raises(ProviderError) as raised:
        Wav2VecEmotionProvider(traversing, model_factory=FakeFactory(FakePredictor([[0.0] * 7])))
    assert raised.value.code == "invalid_local_checkpoint"

    config_path = real / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["provider_payload"] = "transcript-secret"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ProviderError) as raised:
        Wav2VecEmotionProvider(real, model_factory=FakeFactory(FakePredictor([[0.0] * 7])))
    assert raised.value.code == "invalid_local_checkpoint"
    assert "secret" not in str(raised.value).lower()


def test_auto_device_prefers_cuda_then_mps_and_explicit_unavailable_is_safe(
    tmp_path: Path,
) -> None:
    from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

    checkpoint = _checkpoint(tmp_path / "checkpoint")
    hardware_profiles = [
        ((True, True), "cuda"),
        ((False, True), "mps"),
        ((False, False), "cpu"),
    ]
    for hardware, expected in hardware_profiles:
        factory = FakeFactory(FakePredictor([[0.0] * 7]))
        provider = Wav2VecEmotionProvider(
            checkpoint, model_factory=factory, hardware_probe=lambda value=hardware: value
        )
        provider.analyze("utt", _wav(tmp_path / f"{expected}.wav", 1), "")
        assert factory.calls[0][2] == expected

    provider = Wav2VecEmotionProvider(
        checkpoint,
        device="cuda",
        model_factory=FakeFactory(FakePredictor([[0.0] * 7])),
        hardware_probe=lambda: (False, False),
    )
    with pytest.raises(ProviderError) as raised:
        provider.analyze("utt", _wav(tmp_path / "no-cuda.wav", 1), "")
    assert raised.value.code == "provider_runtime_unsupported"


def test_loading_next_candidate_unloads_previous_resident_model(tmp_path: Path) -> None:
    from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

    class ClosablePredictor(FakePredictor):
        def __init__(self) -> None:
            super().__init__([[0.0] * 7])
            self.closed = False

        def close(self) -> None:
            self.closed = True

    checkpoint = _checkpoint(tmp_path / "checkpoint")
    first_predictor = ClosablePredictor()
    second_predictor = ClosablePredictor()
    first = Wav2VecEmotionProvider(
        checkpoint,
        model_factory=FakeFactory(first_predictor),
        hardware_probe=lambda: (False, False),
    )
    second = Wav2VecEmotionProvider(
        checkpoint,
        model_factory=FakeFactory(second_predictor),
        hardware_probe=lambda: (False, False),
    )
    audio = _wav(tmp_path / "audio.wav", 1)

    first.analyze("first", audio, "")
    assert first_predictor.closed is False
    second.analyze("second", audio, "")

    assert first_predictor.closed is True
    assert second_predictor.closed is False
