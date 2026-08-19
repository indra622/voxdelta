from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from voxdelta.credentials import Credentials
from voxdelta.domain.models import AudioAsset
from voxdelta.providers.base import DiarizationProvider, ProviderError
from voxdelta.providers.pyannote_diarization import PyannoteDiarizationProvider


@dataclass(frozen=True)
class FakeTurn:
    start: object
    end: object


class FakeAnnotation:
    def __init__(self, turns: list[tuple[object, object, object]]) -> None:
        self._turns = turns

    def itertracks(self, *, yield_label: bool) -> object:
        assert yield_label is True
        return iter(
            (FakeTurn(start, end), index, label)
            for index, (start, end, label) in enumerate(self._turns)
        )


class FakeOutput:
    def __init__(
        self,
        turns: list[tuple[object, object, object]],
        exclusive: list[tuple[object, object, object]] | None = None,
    ) -> None:
        self.speaker_diarization = FakeAnnotation(turns)
        self.exclusive_speaker_diarization = FakeAnnotation(
            exclusive if exclusive is not None else turns
        )


class ExplodingAnnotation:
    def itertracks(self, *, yield_label: bool) -> object:
        del yield_label
        raise RuntimeError("private-provider-payload")


class FakePipeline:
    def __init__(self, outputs: list[object] | object) -> None:
        self._outputs = list(outputs) if isinstance(outputs, list) else [outputs]
        self.calls: list[tuple[str, dict[str, int]]] = []

    def __call__(self, audio_path: str, **kwargs: int) -> object:
        self.calls.append((audio_path, kwargs))
        output = self._outputs[min(len(self.calls) - 1, len(self._outputs) - 1)]
        if isinstance(output, BaseException):
            raise output
        return output


class FakeFactory:
    def __init__(self, pipeline: FakePipeline | BaseException) -> None:
        self.pipeline = pipeline
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, model: str, *, token: str | None) -> FakePipeline:
        self.calls.append((model, token))
        if isinstance(self.pipeline, BaseException):
            raise self.pipeline
        return self.pipeline


def _asset(
    *,
    mode: str = "mixed",
    paths: tuple[str, ...] = ("normalized/mixed.wav",),
    duration: float = 10.0,
) -> AudioAsset:
    return AudioAsset(
        source_name="private-call.wav",
        source_path="private/input/private-call.wav",
        normalized_paths=paths,
        channel_mode=mode,  # type: ignore[arg-type]
        duration_seconds=duration,
        channels=max(1, len(paths)),
        sha256="0" * 64,
    )


def _provider(
    output: object,
    *,
    token: str = "hf_private_sentinel",
) -> tuple[PyannoteDiarizationProvider, FakeFactory, FakePipeline]:
    pipeline = FakePipeline(output)
    factory = FakeFactory(pipeline)
    provider = PyannoteDiarizationProvider(
        Credentials(HUGGINGFACE_TOKEN=token),
        pipeline_factory=factory,
    )
    return provider, factory, pipeline


def test_mixed_audio_preserves_overlap_evidence_and_exposes_exclusive_alignment() -> None:
    output = FakeOutput(
        [
            (2.0, 6.0, "agent-original"),
            (0.0, 3.0, "customer-original"),
            (0.0, 3.0, "customer-original"),
            (8.0, 12.0, "agent-original"),
        ],
        exclusive=[
            (0.0, 2.0, "customer-original"),
            (2.0, 5.0, "agent-original"),
            (5.0, 10.0, "customer-original"),
        ],
    )
    provider, factory, pipeline = _provider(output)

    timelines = provider.diarize_timelines(_asset())

    assert factory.calls == [("pyannote/speaker-diarization-community-1", "hf_private_sentinel")]
    assert pipeline.calls == [("normalized/mixed.wav", {"min_speakers": 1, "max_speakers": 4})]
    assert [
        (item.start, item.end, item.speaker_id, item.overlap) for item in timelines.evidence
    ] == [
        (0.0, 3.0, "SPEAKER_00", True),
        (2.0, 6.0, "SPEAKER_01", True),
        (8.0, 10.0, "SPEAKER_01", False),
    ]
    assert [(item.start, item.end, item.speaker_id) for item in timelines.exclusive] == [
        (0.0, 2.0, "SPEAKER_00"),
        (2.0, 5.0, "SPEAKER_01"),
        (5.0, 10.0, "SPEAKER_00"),
    ]
    assert all(not item.overlap for item in timelines.exclusive)
    assert provider.diarize(_asset()) == timelines.evidence


@pytest.mark.parametrize(
    "turns",
    [
        [(0.0, 10.0, "only")],
        [(0.0, 2.0, "a"), (2.0, 4.0, "b"), (4.0, 10.0, "c")],
    ],
)
def test_mixed_audio_requires_exactly_two_detected_speakers(
    turns: list[tuple[object, object, object]],
) -> None:
    provider, _, _ = _provider(FakeOutput(turns))

    with pytest.raises(ProviderError) as raised:
        provider.diarize(_asset())

    assert raised.value.code == "unsupported_speaker_count"
    assert str(raised.value) == "Exactly two observed speakers are required."


def test_separate_audio_runs_one_speaker_per_channel_and_merges_chronologically() -> None:
    pipeline = FakePipeline(
        [
            FakeOutput([(3.0, 5.0, "ignored"), (0.0, 1.0, "ignored")]),
            FakeOutput([(1.0, 4.0, "also-ignored")]),
        ]
    )
    factory = FakeFactory(pipeline)
    provider = PyannoteDiarizationProvider(
        Credentials(HUGGINGFACE_TOKEN="token"), pipeline_factory=factory
    )

    segments = provider.diarize(
        _asset(mode="separate", paths=("normalized/left.wav", "normalized/right.wav"))
    )

    assert pipeline.calls == [
        ("normalized/left.wav", {"num_speakers": 1}),
        ("normalized/right.wav", {"num_speakers": 1}),
    ]
    assert [(item.start, item.end, item.speaker_id) for item in segments] == [
        (0.0, 1.0, "SPEAKER_00"),
        (1.0, 4.0, "SPEAKER_01"),
        (3.0, 5.0, "SPEAKER_00"),
    ]


@pytest.mark.parametrize(
    ("asset", "code"),
    [
        (_asset(paths=()), "invalid_audio_asset"),
        (_asset(paths=("a.wav", "b.wav")), "invalid_audio_asset"),
        (_asset(mode="separate", paths=("left.wav",)), "invalid_audio_asset"),
        (_asset(duration=math.inf), "invalid_audio_asset"),
        (_asset(duration=0.0), "invalid_audio_asset"),
    ],
)
def test_invalid_asset_contract_is_rejected_before_loading(asset: AudioAsset, code: str) -> None:
    factory = FakeFactory(AssertionError("factory must not be called"))
    provider = PyannoteDiarizationProvider(
        Credentials(HUGGINGFACE_TOKEN="token"), pipeline_factory=factory
    )

    with pytest.raises(ProviderError) as raised:
        provider.diarize(asset)

    assert raised.value.code == code
    assert factory.calls == []


@pytest.mark.parametrize(
    "output",
    [
        object(),
        FakeOutput([]),
        FakeOutput([(math.nan, 1.0, "a"), (1.0, 2.0, "b")]),
        FakeOutput([(0.0, math.inf, "a"), (1.0, 2.0, "b")]),
        FakeOutput([(2.0, 1.0, "a"), (1.0, 2.0, "b")]),
        FakeOutput([("private/path.wav", 1.0, "a"), (1.0, 2.0, "b")]),
        FakeOutput([(0.0, 1.0, ""), (1.0, 2.0, "b")]),
    ],
)
def test_hostile_or_malformed_pipeline_output_has_a_safe_typed_error(output: object) -> None:
    provider, _, _ = _provider(output)

    with pytest.raises(ProviderError) as raised:
        provider.diarize(_asset())

    assert raised.value.code == "invalid_provider_output"
    assert str(raised.value) == "The diarization provider returned invalid output."
    assert "private" not in str(raised.value)


def test_hostile_iterator_failure_is_sanitized() -> None:
    output = FakeOutput([(0.0, 5.0, "a"), (5.0, 10.0, "b")])
    output.speaker_diarization = ExplodingAnnotation()
    provider, _, _ = _provider(output)

    with pytest.raises(ProviderError) as raised:
        provider.diarize(_asset())

    assert raised.value.code == "invalid_provider_output"
    assert "private-provider-payload" not in str(raised.value)


@pytest.mark.parametrize(
    "exclusive",
    [
        [(0.0, 6.0, "a"), (5.0, 10.0, "b")],
        [(0.0, 10.0, "a")],
    ],
)
def test_exclusive_alignment_requires_a_nonoverlapping_two_speaker_timeline(
    exclusive: list[tuple[object, object, object]],
) -> None:
    output = FakeOutput(
        [(0.0, 5.0, "a"), (5.0, 10.0, "b")],
        exclusive=exclusive,
    )
    provider, _, _ = _provider(output)

    with pytest.raises(ProviderError) as raised:
        provider.diarize_timelines(_asset())

    assert raised.value.code == "invalid_provider_output"


def test_missing_huggingface_token_fails_before_factory_or_audio_access() -> None:
    factory = FakeFactory(AssertionError("factory must not be called"))
    provider = PyannoteDiarizationProvider(
        Credentials(HUGGINGFACE_TOKEN=None), pipeline_factory=factory
    )

    with pytest.raises(ProviderError) as raised:
        provider.diarize(_asset(paths=("does-not-exist/private.wav",)))

    assert raised.value.code == "missing_huggingface_token"
    assert factory.calls == []


def test_local_checkpoint_bypasses_token_and_uses_safe_directory_name(tmp_path: Path) -> None:
    checkpoint = tmp_path / "community-checkpoint"
    checkpoint.mkdir()
    pipeline = FakePipeline(FakeOutput([(0.0, 5.0, "a"), (5.0, 10.0, "b")]))
    factory = FakeFactory(pipeline)
    provider = PyannoteDiarizationProvider(
        Credentials(HUGGINGFACE_TOKEN=None),
        model_path=checkpoint,
        pipeline_factory=factory,
    )

    segments = provider.diarize(_asset())

    assert len(segments) == 2
    assert factory.calls == [(str(checkpoint.resolve()), None)]
    assert provider.provenance.model == "community-checkpoint"
    assert str(tmp_path) not in provider.provenance.model_dump_json()


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (TimeoutError("hf_private_sentinel /private/audio.wav"), "provider_timeout"),
        (RuntimeError("hf_private_sentinel /private/audio.wav"), "provider_unavailable"),
    ],
)
def test_sdk_failures_are_sanitized_and_do_not_disclose_secrets(
    failure: BaseException, expected_code: str
) -> None:
    provider, _, _ = _provider(failure)

    with pytest.raises(ProviderError) as raised:
        provider.diarize(_asset())

    assert raised.value.code == expected_code
    public_text = f"{raised.value!s} {raised.value!r} {provider!r} {provider.provenance!r}"
    assert "hf_private_sentinel" not in public_text
    assert "/private/audio.wav" not in public_text


def test_local_provider_conforms_to_protocol_and_declares_provenance() -> None:
    provider, _, _ = _provider(FakeOutput([(0.0, 5.0, "a"), (5.0, 10.0, "b")]))

    assert isinstance(provider, DiarizationProvider)
    assert provider.provenance.name == "pyannote"
    assert provider.provenance.model == "speaker-diarization-community-1"
    assert provider.provenance.remote is False
    assert provider.provenance.transmits == ()
    assert provider.provenance.retention_policy_url is None
