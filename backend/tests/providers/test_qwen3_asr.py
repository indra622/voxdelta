from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

import pytest

from voxdelta.domain.models import AudioAsset, SpeakerSegment
from voxdelta.providers.base import ProviderError, TranscriptionProvider


@dataclass(frozen=True)
class FakeUnit:
    text: object
    start: object
    end: object


class FakeAligner:
    def __init__(self, units: object) -> None:
        self.units = units
        self.calls: list[tuple[str, str, str]] = []

    def align(self, audio: str, text: str, *, language: str) -> object:
        self.calls.append((audio, text, language))
        return self.units


class FakeModel:
    def __init__(self, text: object = "안녕하세요", time_stamps: object | None = None) -> None:
        self.text = text
        self.time_stamps = time_stamps
        self.calls: list[tuple[str, str, bool]] = []

    def transcribe(self, audio: str, *, language: str, return_time_stamps: bool) -> object:
        self.calls.append((audio, language, return_time_stamps))
        return [type("Result", (), {"text": self.text, "time_stamps": self.time_stamps})()]


class ModelFactory:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.calls: list[tuple[str, str, str, str]] = []

    def __call__(self, model_id: str, *, aligner_id: str, device: str, dtype: str) -> FakeModel:
        self.calls.append((model_id, aligner_id, device, dtype))
        return self.model


class AlignerFactory:
    def __init__(self, aligner: FakeAligner) -> None:
        self.aligner = aligner
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, model_id: str, *, device: str, dtype: str) -> FakeAligner:
        self.calls.append((model_id, device, dtype))
        return self.aligner


def _asset() -> AudioAsset:
    return AudioAsset(
        source_name="call.wav",
        source_path="private/call.wav",
        normalized_paths=("normalized/mixed.wav",),
        channel_mode="mixed",
        duration_seconds=3.0,
        channels=1,
        sha256="0" * 64,
    )


def _segments() -> list[SpeakerSegment]:
    return [
        SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00", confidence=1.0),
        SpeakerSegment(start=1.0, end=3.0, speaker_id="SPEAKER_01", confidence=1.0),
    ]


def test_import_is_lazy_and_default_ids_and_request_contract_are_exact() -> None:
    before = set(sys.modules)
    module = importlib.import_module("voxdelta.providers.qwen3_asr")
    assert "qwen_asr" not in set(sys.modules) - before

    units = [FakeUnit("안녕", 0.0, 0.8), FakeUnit("하세요", 1.0, 1.5)]
    model = FakeModel(time_stamps=units)
    aligner = FakeAligner(units)
    model_factory = ModelFactory(model)
    aligner_factory = AlignerFactory(aligner)
    provider = module.Qwen3AsrProvider(
        model_factory=model_factory,
        aligner_factory=aligner_factory,
        hardware_probe=lambda: (False, False),
    )

    utterances = provider.transcribe(_asset(), _segments())

    assert isinstance(provider, TranscriptionProvider)
    assert provider.provenance.model == "Qwen3-ASR-1.7B"
    assert model_factory.calls == [
        (
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3-ForcedAligner-0.6B",
            "cpu",
            "float32",
        )
    ]
    assert aligner_factory.calls == [("Qwen/Qwen3-ForcedAligner-0.6B", "cpu", "float32")]
    assert model.calls == [("normalized/mixed.wav", "Korean", False)]
    assert aligner.calls == [("normalized/mixed.wav", "안녕하세요", "Korean")]
    assert [(u.speaker_id, u.transcript) for u in utterances] == [
        ("SPEAKER_00", "안녕"),
        ("SPEAKER_01", "하세요"),
    ]


def test_low_memory_profile_is_only_explicit() -> None:
    from voxdelta.providers.qwen3_asr import Qwen3AsrProvider

    model_factory = ModelFactory(FakeModel(time_stamps=[FakeUnit("안녕", 0.0, 0.5)]))
    aligner_factory = AlignerFactory(FakeAligner([FakeUnit("안녕", 0.0, 0.5)]))
    provider = Qwen3AsrProvider(
        profile="low-memory",
        model_factory=model_factory,
        aligner_factory=aligner_factory,
        hardware_probe=lambda: (False, False),
    )
    provider.transcribe(_asset(), _segments())

    assert model_factory.calls[0][0] == "Qwen/Qwen3-ASR-0.6B"
    assert provider.provenance.model == "Qwen3-ASR-0.6B"


def test_single_speaker_transcription_is_explicitly_unattributed() -> None:
    from voxdelta.providers.qwen3_asr import Qwen3AsrProvider

    units = [FakeUnit("안녕", 0.0, 0.8), FakeUnit("하세요", 1.0, 1.5)]
    provider = Qwen3AsrProvider(
        model_factory=ModelFactory(FakeModel(time_stamps=units)),
        aligner_factory=AlignerFactory(FakeAligner(units)),
        hardware_probe=lambda: (False, False),
    )

    utterances = provider.transcribe_single_speaker(_asset())

    assert [(item.speaker_id, item.transcript, item.start, item.end) for item in utterances] == [
        ("SPEAKER_00", "안녕 하세요", 0.0, 1.5)
    ]
    assert provider.last_timestamp_coverage is not None


@pytest.mark.parametrize(
    ("hardware", "expected"),
    [
        ((True, True), ("cuda", "bfloat16")),
        ((False, True), ("mps", "float16")),
        ((False, False), ("cpu", "float32")),
    ],
)
def test_auto_device_precedence(hardware: tuple[bool, bool], expected: tuple[str, str]) -> None:
    from voxdelta.providers.qwen3_asr import Qwen3AsrProvider

    model_factory = ModelFactory(FakeModel(time_stamps=[FakeUnit("x", 0.0, 0.5)]))
    provider = Qwen3AsrProvider(
        model_factory=model_factory,
        aligner_factory=AlignerFactory(FakeAligner([FakeUnit("x", 0.0, 0.5)])),
        hardware_probe=lambda: hardware,
    )
    provider.transcribe(_asset(), _segments())

    assert model_factory.calls[0][2:] == expected


def test_unsupported_model_device_and_profile_fail_before_factories() -> None:
    from voxdelta.providers.qwen3_asr import Qwen3AsrProvider

    factory = ModelFactory(FakeModel())
    aligner_factory = AlignerFactory(FakeAligner([]))
    with pytest.raises(ProviderError):
        Qwen3AsrProvider(model_id="private/model", model_factory=factory)
    with pytest.raises(ProviderError):
        Qwen3AsrProvider(device="tpu", model_factory=factory)  # type: ignore[arg-type]
    with pytest.raises(ProviderError):
        Qwen3AsrProvider(profile="magic", model_factory=factory)  # type: ignore[arg-type]
    assert factory.calls == []
    assert aligner_factory.calls == []


def test_runtime_memory_or_mps_failure_is_typed_and_never_falls_back() -> None:
    from voxdelta.providers.qwen3_asr import Qwen3AsrProvider

    class ExplodingModel(FakeModel):
        def transcribe(self, audio: str, *, language: str, return_time_stamps: bool) -> object:
            del audio, language, return_time_stamps
            raise RuntimeError("MPS out of memory /private/audio.wav transcript-secret")

    model_factory = ModelFactory(ExplodingModel())
    provider = Qwen3AsrProvider(
        model_factory=model_factory,
        aligner_factory=AlignerFactory(FakeAligner([])),
        hardware_probe=lambda: (False, True),
    )

    with pytest.raises(ProviderError) as raised:
        provider.transcribe(_asset(), _segments())

    assert raised.value.code == "provider_runtime_unsupported"
    assert "private" not in str(raised.value).lower()
    assert len(model_factory.calls) == 1


def test_aligner_iterator_failure_and_malformed_text_are_sanitized() -> None:
    from voxdelta.providers.qwen3_asr import Qwen3AsrProvider

    class ExplodingUnits:
        def __iter__(self) -> ExplodingUnits:
            return self

        def __next__(self) -> object:
            raise RuntimeError("provider payload /private/x.wav secret transcript")

    provider = Qwen3AsrProvider(
        model_factory=ModelFactory(FakeModel()),
        aligner_factory=AlignerFactory(FakeAligner(ExplodingUnits())),
        hardware_probe=lambda: (False, False),
    )

    with pytest.raises(ProviderError) as raised:
        provider.transcribe(_asset(), _segments())

    assert raised.value.code == "invalid_provider_output"
    assert "private" not in str(raised.value).lower()


def test_no_environment_or_global_random_state_is_modified(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import random

    from voxdelta.providers.qwen3_asr import Qwen3AsrProvider

    monkeypatch.setenv("VOXDELTA_SENTINEL", "unchanged")
    state = random.getstate()
    provider = Qwen3AsrProvider(
        model_factory=ModelFactory(FakeModel(time_stamps=[FakeUnit("x", 0.0, 0.5)])),
        aligner_factory=AlignerFactory(FakeAligner([FakeUnit("x", 0.0, 0.5)])),
        hardware_probe=lambda: (False, False),
    )
    provider.transcribe(_asset(), _segments())

    assert os.environ["VOXDELTA_SENTINEL"] == "unchanged"
    assert random.getstate() == state
