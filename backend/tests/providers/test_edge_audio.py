"""Edge-audio regression through the production load/window/aggregate path.

Every fixture is synthesised deterministically inside `tmp_path`; no recorded audio and
no private sample ever enters the repository. The model itself is stubbed so the real
path under test is the one that actually differs at the edges — WAV decoding, the minimum
length rule, and 20-second windowing — rather than the encoder weights.

No label expectation is asserted. Each case must produce either a finite, valid
seven-label distribution or one of the provider's existing, unchanged error codes.
"""

from __future__ import annotations

import math
import struct
import wave
from collections.abc import Sequence
from pathlib import Path

import pytest

from voxdelta.evaluation.emotion_training import (
    MIN_SAMPLES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    evaluation_windows,
    load_audio,
)
from voxdelta.evaluation.wav2vec_base import WAV2VEC_MODEL_REVISION, WAV2VEC_WEIGHTS_SHA256
from voxdelta.providers.base import ProviderError
from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

# A 45-second clip is the largest fixture here: 45 * 16000 * 2 bytes ~= 1.4 MB, and it is
# the smallest length that still forces three windows through the aggregation path.
LONG_SECONDS = 45
MAX_FIXTURE_BYTES = 2 * 1024 * 1024


def _write_wav(path: Path, samples: Sequence[int]) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    assert path.stat().st_size <= MAX_FIXTURE_BYTES
    return path


def _silence(count: int) -> list[int]:
    return [0] * count


def _low_amplitude_noise(count: int) -> list[int]:
    """A deterministic dither-level signal: loud enough to exist, quiet enough to be nothing."""

    value = 20260826
    samples: list[int] = []
    for _ in range(count):
        value = (1103515245 * value + 12345) % (2**31)
        samples.append((value % 5) - 2)
    return samples


def _tone(count: int) -> list[int]:
    return [
        int(12000 * math.sin(2 * math.pi * 220.0 * index / SAMPLE_RATE)) for index in range(count)
    ]


class _StubPredictor:
    """Returns a fixed logit row per window, so aggregation is exercised, not the encoder."""

    def __init__(self) -> None:
        self.window_lengths: list[int] = []

    def predict(self, samples: tuple[float, ...], sample_rate: int) -> list[float]:
        assert sample_rate == SAMPLE_RATE
        self.window_lengths.append(len(samples))
        return [0.5, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1]


def _provider(tmp_path: Path, predictor: _StubPredictor) -> Wav2VecEmotionProvider:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    import json

    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "schema_version": "4",
                "architecture": "wav2vec-xls-r",
                "model_id": "facebook/wav2vec2-xls-r-300m",
                "labels": [
                    "happiness",
                    "anger",
                    "disgust",
                    "fear",
                    "neutral",
                    "sadness",
                    "surprise",
                ],
                "model_revision": WAV2VEC_MODEL_REVISION,
                "base_model_sha256": WAV2VEC_WEIGHTS_SHA256,
                "class_weighting": "none",
                "class_weights": [1.0] * 7,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "label_mapping.json").write_text(
        json.dumps(
            {
                "0": "happiness",
                "1": "anger",
                "2": "disgust",
                "3": "fear",
                "4": "neutral",
                "5": "sadness",
                "6": "surprise",
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "metrics.json").write_text(
        json.dumps({"macro_f1": 0.7, "validation_hash": "a" * 64}), encoding="utf-8"
    )
    (checkpoint / "model.safetensors").write_bytes(b"stub")
    return Wav2VecEmotionProvider(
        checkpoint,
        device="cpu",
        model_factory=lambda *args, **kwargs: predictor,
        hardware_probe=lambda: (False, False),
        inference_context=lambda: __import__("contextlib").nullcontext(),
    )


def _assert_valid_result(result: object) -> None:
    from voxdelta.domain.models import EmotionResult

    assert isinstance(result, EmotionResult)
    values = tuple(result.probabilities.values())
    assert len(values) == 7
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in values)
    assert abs(math.fsum(values) - 1.0) <= 1e-6
    assert math.isfinite(result.confidence)


@pytest.mark.parametrize(
    ("name", "builder", "seconds"),
    [
        ("silence", _silence, 1.0),
        ("low_amplitude_noise", _low_amplitude_noise, 1.0),
        ("tone", _tone, 1.0),
        ("exactly_minimum_length", _tone, MIN_SAMPLES / SAMPLE_RATE),
        ("long_multi_window", _tone, float(LONG_SECONDS)),
    ],
)
def test_edge_audio_yields_a_finite_valid_result_or_a_known_error_code(
    tmp_path: Path, name: str, builder: object, seconds: float
) -> None:
    count = int(SAMPLE_RATE * seconds)
    clip = _write_wav(tmp_path / f"{name}.wav", builder(count))  # type: ignore[operator]
    predictor = _StubPredictor()
    provider = _provider(tmp_path, predictor)

    try:
        result = provider.analyze(f"edge-{name}", clip, "")
    except ProviderError as error:
        assert error.code in {
            "invalid_audio_asset",
            "invalid_local_checkpoint",
            "invalid_provider_output",
            "provider_runtime_unsupported",
            "provider_unavailable",
            "provider_timeout",
        }
        return
    _assert_valid_result(result)


def test_audio_shorter_than_the_minimum_is_refused_with_the_existing_code(
    tmp_path: Path,
) -> None:
    clip = _write_wav(tmp_path / "too-short.wav", _tone(MIN_SAMPLES - 1))
    provider = _provider(tmp_path, _StubPredictor())

    with pytest.raises(ProviderError) as error:
        provider.analyze("edge-too-short", clip, "")

    assert error.value.code == "invalid_audio_asset"


def test_an_empty_clip_is_refused_rather_than_producing_a_distribution(tmp_path: Path) -> None:
    clip = _write_wav(tmp_path / "empty.wav", [])
    provider = _provider(tmp_path, _StubPredictor())

    with pytest.raises(ProviderError) as error:
        provider.analyze("edge-empty", clip, "")

    assert error.value.code == "invalid_audio_asset"


def test_long_audio_really_traverses_multiple_production_windows(tmp_path: Path) -> None:
    count = SAMPLE_RATE * LONG_SECONDS
    clip = _write_wav(tmp_path / "long.wav", _tone(count))
    predictor = _StubPredictor()
    provider = _provider(tmp_path, predictor)

    result = provider.analyze("edge-long", clip, "")

    _assert_valid_result(result)
    # 45 s at a 20 s window is three windows: two full and one remainder.
    assert len(predictor.window_lengths) == 3
    assert predictor.window_lengths[:2] == [WINDOW_SAMPLES, WINDOW_SAMPLES]
    assert predictor.window_lengths[2] == count - 2 * WINDOW_SAMPLES


def test_short_audio_uses_exactly_one_window(tmp_path: Path) -> None:
    clip = _write_wav(tmp_path / "short.wav", _tone(SAMPLE_RATE))
    predictor = _StubPredictor()
    provider = _provider(tmp_path, predictor)

    provider.analyze("edge-short", clip, "")

    assert len(predictor.window_lengths) == 1


def test_silence_is_decoded_rather_than_rejected_as_a_malformed_asset(tmp_path: Path) -> None:
    clip = _write_wav(tmp_path / "silence.wav", _silence(SAMPLE_RATE))

    decoded = load_audio(clip)

    assert len(decoded.samples) == SAMPLE_RATE
    assert set(decoded.samples) == {0.0}
    assert len(evaluation_windows(decoded)) == 1


def test_a_non_wav_payload_is_refused_by_the_production_loader(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"RIFF\x00\x00\x00\x00WAVEnope")

    with pytest.raises(ValueError, match="invalid_audio"):
        load_audio(corrupt)


def test_a_stereo_or_wrong_rate_clip_is_refused_by_the_production_loader(tmp_path: Path) -> None:
    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(struct.pack("<2000h", *([0] * 2000)))

    with pytest.raises(ValueError, match="invalid_audio"):
        load_audio(stereo)

    resampled = tmp_path / "8k.wav"
    with wave.open(str(resampled), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8_000)
        writer.writeframes(struct.pack("<16000h", *([0] * 16000)))

    with pytest.raises(ValueError, match="invalid_audio"):
        load_audio(resampled)


def test_every_fixture_stays_within_the_declared_size_bound(tmp_path: Path) -> None:
    clip = _write_wav(tmp_path / "bound.wav", _tone(SAMPLE_RATE * LONG_SECONDS))

    assert clip.stat().st_size <= MAX_FIXTURE_BYTES
