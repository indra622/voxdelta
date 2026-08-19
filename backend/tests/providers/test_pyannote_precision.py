from __future__ import annotations

import pytest

from voxdelta.credentials import Credentials
from voxdelta.domain.models import AudioAsset
from voxdelta.providers.base import DiarizationProvider, ProviderError
from voxdelta.providers.pyannote_precision import (
    PYANNOTE_DATA_RETENTION_URL,
    PyannotePrecisionProvider,
)


class FakeTurn:
    def __init__(self, start: object, end: object) -> None:
        self.start = start
        self.end = end


class FakeAnnotation:
    def __init__(self, turns: list[tuple[object, object, object]]) -> None:
        self.turns = turns

    def itertracks(self, *, yield_label: bool) -> object:
        assert yield_label is True
        return iter(
            (FakeTurn(start, end), index, label)
            for index, (start, end, label) in enumerate(self.turns)
        )


class FakeOutput:
    def __init__(self, turns: list[tuple[object, object, object]]) -> None:
        self.speaker_diarization = FakeAnnotation(turns)
        self.exclusive_speaker_diarization = FakeAnnotation(turns)


class FakePipeline:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[tuple[str, dict[str, int]]] = []

    def __call__(self, audio_path: str, **kwargs: int) -> object:
        self.calls.append((audio_path, kwargs))
        if isinstance(self.output, BaseException):
            raise self.output
        return self.output


class FakeFactory:
    def __init__(self, pipeline: FakePipeline) -> None:
        self.pipeline = pipeline
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, model: str, *, token: str | None) -> FakePipeline:
        self.calls.append((model, token))
        return self.pipeline


def _asset() -> AudioAsset:
    return AudioAsset(
        source_name="call.wav",
        source_path="private/source.wav",
        normalized_paths=("private/normalized.wav",),
        channel_mode="mixed",
        duration_seconds=10.0,
        channels=1,
        sha256="0" * 64,
    )


def _provider(
    output: object,
    *,
    credentials: Credentials | None = None,
) -> tuple[PyannotePrecisionProvider, FakeFactory, FakePipeline]:
    pipeline = FakePipeline(output)
    factory = FakeFactory(pipeline)
    provider = PyannotePrecisionProvider(
        credentials
        if credentials is not None
        else Credentials(PYANNOTEAI_API_KEY="pyannote_private_sentinel"),
        pipeline_factory=factory,
    )
    return provider, factory, pipeline


def test_precision_selects_remote_model_and_declares_audio_retention_contract() -> None:
    provider, factory, pipeline = _provider(
        FakeOutput([(5.0, 10.0, "second"), (0.0, 5.0, "first")])
    )

    segments = provider.diarize(_asset())

    assert factory.calls == [
        ("pyannote/speaker-diarization-precision-2", "pyannote_private_sentinel")
    ]
    assert pipeline.calls == [("private/normalized.wav", {"min_speakers": 1, "max_speakers": 4})]
    assert [item.speaker_id for item in segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert isinstance(provider, DiarizationProvider)
    assert provider.provenance.name == "pyannote"
    assert provider.provenance.model == "speaker-diarization-precision-2"
    assert provider.provenance.remote is True
    assert provider.provenance.transmits == ("audio",)
    assert provider.provenance.retention_policy_url == PYANNOTE_DATA_RETENTION_URL
    assert PYANNOTE_DATA_RETENTION_URL == "https://docs.pyannote.ai/data-retention"


@pytest.mark.parametrize(
    "turns",
    [
        [(0.0, 10.0, "only")],
        [(0.0, 2.0, "a"), (2.0, 4.0, "b"), (4.0, 10.0, "c")],
    ],
)
def test_precision_requires_exactly_two_speakers(
    turns: list[tuple[object, object, object]],
) -> None:
    provider, _, _ = _provider(FakeOutput(turns))

    with pytest.raises(ProviderError) as raised:
        provider.diarize(_asset())

    assert raised.value.code == "unsupported_speaker_count"


def test_missing_api_key_fails_before_pipeline_loading_or_audio_reading() -> None:
    provider, factory, pipeline = _provider(
        FakeOutput([]), credentials=Credentials(PYANNOTEAI_API_KEY=None)
    )

    with pytest.raises(ProviderError) as raised:
        provider.diarize(_asset())

    assert raised.value.code == "missing_pyannote_api_key"
    assert factory.calls == []
    assert pipeline.calls == []


def test_precision_timeout_is_safe_and_never_falls_back_to_community() -> None:
    provider, factory, _ = _provider(
        TimeoutError("pyannote_private_sentinel private/normalized.wav")
    )

    with pytest.raises(ProviderError) as raised:
        provider.diarize(_asset())

    assert raised.value.code == "provider_timeout"
    assert [model for model, _ in factory.calls] == ["pyannote/speaker-diarization-precision-2"]
    assert "pyannote_private_sentinel" not in str(raised.value)
    assert "private/normalized.wav" not in str(raised.value)


def test_precision_malformed_output_is_sanitized() -> None:
    provider, _, _ = _provider(FakeOutput([(float("nan"), 2.0, "private-label")]))

    with pytest.raises(ProviderError) as raised:
        provider.diarize(_asset())

    assert raised.value.code == "invalid_provider_output"
    assert "private-label" not in str(raised.value)


def test_precision_constructor_repr_and_provenance_do_not_reveal_secret() -> None:
    provider, _, _ = _provider(FakeOutput([(0.0, 5.0, "a"), (5.0, 10.0, "b")]))

    rendered = f"{provider!r} {provider.provenance!r}"

    assert "pyannote_private_sentinel" not in rendered
