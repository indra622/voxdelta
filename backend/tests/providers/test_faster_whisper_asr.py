from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass

import pytest

from voxdelta.domain.models import AudioAsset, SpeakerSegment
from voxdelta.providers.base import ProviderError, TranscriptionProvider


@dataclass(frozen=True)
class FakeWord:
    start: object
    end: object
    word: object


@dataclass(frozen=True)
class FakeSegment:
    words: object


class FakeModel:
    def __init__(self, outputs: list[object] | BaseException) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, dict[str, object]]] = []

    def transcribe(self, path: str, **kwargs: object) -> tuple[object, object]:
        self.calls.append((path, kwargs))
        if isinstance(self.outputs, BaseException):
            raise self.outputs
        return iter(self.outputs), object()


class FakeFactory:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, model_id: str, *, device: str, compute_type: str) -> FakeModel:
        self.calls.append((model_id, device, compute_type))
        return self.model


def _asset(
    *,
    mode: str = "mixed",
    paths: tuple[str, ...] = ("normalized/mixed.wav",),
    duration: float = 4.0,
) -> AudioAsset:
    return AudioAsset(
        source_name="call.wav",
        source_path="private/call.wav",
        normalized_paths=paths,
        channel_mode=mode,  # type: ignore[arg-type]
        duration_seconds=duration,
        channels=len(paths),
        sha256="0" * 64,
    )


def _segments() -> list[SpeakerSegment]:
    return [
        SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00", confidence=1.0),
        SpeakerSegment(start=1.0, end=2.0, speaker_id="SPEAKER_01", confidence=1.0),
        SpeakerSegment(start=2.0, end=3.0, speaker_id="SPEAKER_00", confidence=1.0),
    ]


def test_import_is_lazy_and_provider_conforms_to_protocol() -> None:
    before = set(sys.modules)
    module = importlib.import_module("voxdelta.providers.faster_whisper_asr")

    assert "faster_whisper" not in set(sys.modules) - before
    provider = module.FasterWhisperProvider(model_factory=FakeFactory(FakeModel([])))
    assert isinstance(provider, TranscriptionProvider)
    assert provider.provenance.model == "large-v3-turbo"
    assert provider.provenance.remote is False


def test_mixed_words_use_maximum_overlap_and_safe_grouping() -> None:
    from voxdelta.providers.faster_whisper_asr import FasterWhisperProvider

    model = FakeModel(
        [
            FakeSegment(
                [
                    FakeWord(0.0, 0.8, " 안녕 "),
                    FakeWord(1.0, 1.5, " 하세요"),
                    FakeWord(2.0, 2.8, " 반갑습니다 "),
                    FakeWord(3.1, 3.5, "outside"),
                ]
            )
        ]
    )
    factory = FakeFactory(model)
    provider = FasterWhisperProvider(model_factory=factory)

    utterances = provider.transcribe(_asset(), _segments())

    assert factory.calls == [("large-v3-turbo", "cpu", "int8")]
    assert model.calls == [
        (
            "normalized/mixed.wav",
            {
                "language": "ko",
                "word_timestamps": True,
                "vad_filter": False,
                "beam_size": 5,
            },
        )
    ]
    assert [(u.id, u.speaker_id, u.transcript, u.start, u.end) for u in utterances] == [
        ("utt-0001", "SPEAKER_00", "안녕", 0.0, 0.8),
        ("utt-0002", "SPEAKER_01", "하세요", 1.0, 1.5),
        ("utt-0003", "SPEAKER_00", "반갑습니다", 2.0, 2.8),
    ]
    assert provider.last_warning_count == 1


def test_exact_overlap_tie_is_deterministic() -> None:
    from voxdelta.providers.faster_whisper_asr import FasterWhisperProvider

    model = FakeModel([FakeSegment([FakeWord(0.5, 1.5, "경계")])])
    provider = FasterWhisperProvider(model_factory=FakeFactory(model))

    utterances = provider.transcribe(_asset(), _segments()[:2])

    assert [utterance.speaker_id for utterance in utterances] == ["SPEAKER_00"]


@pytest.mark.parametrize(
    "segments",
    [
        [SpeakerSegment(start=0, end=4, speaker_id="only", confidence=1)],
        [
            SpeakerSegment(start=0, end=1, speaker_id="a", confidence=1),
            SpeakerSegment(start=1, end=2, speaker_id="b", confidence=1),
            SpeakerSegment(start=2, end=4, speaker_id="c", confidence=1),
        ],
    ],
)
def test_mixed_alignment_requires_exactly_two_distinct_speakers(
    segments: list[SpeakerSegment],
) -> None:
    from voxdelta.providers.faster_whisper_asr import FasterWhisperProvider

    factory = FakeFactory(FakeModel([FakeSegment([FakeWord(0.0, 0.5, "word")])]))
    provider = FasterWhisperProvider(model_factory=factory)

    with pytest.raises(ProviderError) as raised:
        provider.transcribe(_asset(), segments)

    assert raised.value.code == "unsupported_speaker_count"
    assert factory.calls == []


def test_grouped_overlapping_words_cover_the_complete_interval() -> None:
    from voxdelta.providers.faster_whisper_asr import FasterWhisperProvider

    model = FakeModel([FakeSegment([FakeWord(0.0, 2.0, "긴"), FakeWord(1.0, 1.5, "겹침")])])
    provider = FasterWhisperProvider(model_factory=FakeFactory(model))
    segments = [
        SpeakerSegment(start=0, end=3, speaker_id="SPEAKER_00", confidence=1),
        SpeakerSegment(start=3, end=4, speaker_id="SPEAKER_01", confidence=1),
    ]

    utterances = provider.transcribe(_asset(), segments)

    assert [(item.start, item.end, item.transcript) for item in utterances] == [
        (0.0, 2.0, "긴 겹침")
    ]


def test_separate_channels_are_fixed_and_merged_chronologically() -> None:
    from voxdelta.providers.faster_whisper_asr import FasterWhisperProvider

    class ChannelModel(FakeModel):
        def transcribe(self, path: str, **kwargs: object) -> tuple[object, object]:
            self.calls.append((path, kwargs))
            words = (
                [FakeWord(2.0, 2.4, "왼쪽")]
                if path.endswith("left.wav")
                else [FakeWord(1.0, 1.4, "오른쪽")]
            )
            return iter([FakeSegment(words)]), object()

    model = ChannelModel([])
    provider = FasterWhisperProvider(model_factory=FakeFactory(model))
    asset = _asset(mode="separate", paths=("normalized/left.wav", "normalized/right.wav"))

    utterances = provider.transcribe(asset, _segments())

    assert [(u.speaker_id, u.transcript) for u in utterances] == [
        ("SPEAKER_01", "오른쪽"),
        ("SPEAKER_00", "왼쪽"),
    ]


def test_separate_grouped_overlap_covers_all_channel_words() -> None:
    from voxdelta.providers.faster_whisper_asr import FasterWhisperProvider

    class ChannelModel(FakeModel):
        def transcribe(self, path: str, **kwargs: object) -> tuple[object, object]:
            del kwargs
            words = (
                [FakeWord(0.0, 2.0, "긴"), FakeWord(1.0, 1.5, "겹침")]
                if path.endswith("left.wav")
                else []
            )
            return iter([FakeSegment(words)]), object()

    provider = FasterWhisperProvider(model_factory=FakeFactory(ChannelModel([])))
    asset = _asset(mode="separate", paths=("normalized/left.wav", "normalized/right.wav"))

    utterances = provider.transcribe(asset, _segments())

    assert [(item.start, item.end, item.transcript) for item in utterances] == [
        (0.0, 2.0, "긴 겹침")
    ]


@pytest.mark.parametrize(
    "words",
    [
        [FakeWord(math.nan, 1.0, "bad")],
        [FakeWord(0.0, math.inf, "bad")],
        [FakeWord(-0.1, 0.5, "bad")],
        [FakeWord(0.5, 5.0, "bad")],
        [FakeWord(1.0, 0.5, "bad")],
        [FakeWord(1.0, 1.5, "later"), FakeWord(0.0, 0.5, "earlier")],
        [FakeWord(0.0, 0.5, "same"), FakeWord(0.0, 0.5, "same")],
        [FakeWord(0.0, 0.5, "")],
    ],
)
def test_malformed_provider_words_are_rejected(words: list[FakeWord]) -> None:
    from voxdelta.providers.faster_whisper_asr import FasterWhisperProvider

    provider = FasterWhisperProvider(model_factory=FakeFactory(FakeModel([FakeSegment(words)])))

    with pytest.raises(ProviderError) as raised:
        provider.transcribe(_asset(), _segments())

    assert raised.value.code == "invalid_provider_output"


def test_iterator_and_runtime_exceptions_are_sanitized() -> None:
    from voxdelta.providers.faster_whisper_asr import FasterWhisperProvider

    class ExplodingIterator:
        def __iter__(self) -> ExplodingIterator:
            return self

        def __next__(self) -> object:
            raise RuntimeError("/private/call.wav transcript-secret hf_secret")

    class ExplodingModel(FakeModel):
        def transcribe(self, path: str, **kwargs: object) -> tuple[object, object]:
            del path, kwargs
            return ExplodingIterator(), object()

    provider = FasterWhisperProvider(model_factory=FakeFactory(ExplodingModel([])))

    with pytest.raises(ProviderError) as raised:
        provider.transcribe(_asset(), _segments())

    assert raised.value.code == "invalid_provider_output"
    assert "private" not in str(raised.value).lower()
    assert "secret" not in str(raised.value).lower()


@pytest.mark.parametrize("failure_point", ["factory", "model", "segments", "words"])
def test_timeout_is_preserved_across_lazy_faster_whisper_boundaries(
    failure_point: str,
) -> None:
    from voxdelta.providers.faster_whisper_asr import FasterWhisperProvider

    class TimeoutIterator:
        def __iter__(self) -> TimeoutIterator:
            return self

        def __next__(self) -> object:
            raise TimeoutError("/private/call.wav transcript-secret provider-payload")

    class TimeoutFactory:
        def __call__(self, model_id: str, *, device: str, compute_type: str) -> FakeModel:
            del model_id, device, compute_type
            raise TimeoutError("/private/model token-secret")

    if failure_point == "factory":
        provider = FasterWhisperProvider(model_factory=TimeoutFactory())
    elif failure_point == "model":
        provider = FasterWhisperProvider(model_factory=FakeFactory(FakeModel(TimeoutError())))
    elif failure_point == "segments":

        class SegmentTimeoutModel(FakeModel):
            def transcribe(self, path: str, **kwargs: object) -> tuple[object, object]:
                del path, kwargs
                return TimeoutIterator(), object()

        provider = FasterWhisperProvider(model_factory=FakeFactory(SegmentTimeoutModel([])))
    else:
        provider = FasterWhisperProvider(
            model_factory=FakeFactory(FakeModel([FakeSegment(TimeoutIterator())]))
        )

    with pytest.raises(ProviderError) as raised:
        provider.transcribe(_asset(), _segments())

    assert raised.value.code == "provider_timeout"
    serialized = str(raised.value).lower()
    assert "private" not in serialized
    assert "secret" not in serialized
    assert "payload" not in serialized


def test_device_and_model_are_validated_before_factory() -> None:
    from voxdelta.providers.faster_whisper_asr import FasterWhisperProvider

    factory = FakeFactory(FakeModel([]))
    with pytest.raises(ProviderError):
        FasterWhisperProvider(model_id="private/model", model_factory=factory)
    with pytest.raises(ProviderError):
        FasterWhisperProvider(device="mps", model_factory=factory)  # type: ignore[arg-type]
    assert factory.calls == []


def test_cuda_uses_float16_and_factory_is_called_once() -> None:
    from voxdelta.providers.faster_whisper_asr import FasterWhisperProvider

    factory = FakeFactory(FakeModel([]))
    provider = FasterWhisperProvider(device="cuda", model_factory=factory)
    provider.transcribe(_asset(), _segments())
    provider.transcribe(_asset(), _segments())

    assert factory.calls == [("large-v3-turbo", "cuda", "float16")]
