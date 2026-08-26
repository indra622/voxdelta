from __future__ import annotations

import importlib.util
import math
import wave
from pathlib import Path
from typing import Any

import pytest

# The release bundle fixtures live with the release tests.
from test_release import _fake_base_validator, _Sources
from voxdelta.domain.models import EmotionResult, ProviderProvenance, ProviderUsage
from voxdelta.evaluation.emotion_training import load_audio
from voxdelta.providers._emotion_runtime import Device

from voxdelta_runpod.config import CANONICAL_LABELS
from voxdelta_runpod.release import ReleaseManifest, verify_release_bundle
from voxdelta_runpod.smoke import (
    OFFLINE_ENVIRONMENT,
    SMOKE_SAMPLE_RATE,
    SmokeError,
    run_release_smoke,
    write_smoke_audio,
)

SMOKE_SCRIPT = Path(__file__).parents[1] / "scripts" / "smoke_release_inference.py"


def _verifier(bundle: Path) -> ReleaseManifest:
    """The synthetic fixture weights cannot satisfy the pinned base hashes."""

    return verify_release_bundle(bundle, base_validator=_fake_base_validator)


_DISTRIBUTION = {
    "happiness": 0.05,
    "anger": 0.10,
    "disgust": 0.05,
    "fear": 0.05,
    "neutral": 0.60,
    "sadness": 0.10,
    "surprise": 0.05,
}


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("smoke_release_inference", SMOKE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeProvider:
    def __init__(self, checkpoint: Path, base_model: Path, revision: str) -> None:
        self.checkpoint = checkpoint
        self.base_model = base_model
        self.revision = revision
        self.audio: Path | None = None
        self.unloaded = False

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        # A real reload reads both directories; assert the bundle is self-contained.
        assert (self.checkpoint / "model.safetensors").is_file()
        assert (self.base_model / "preprocessor_config.json").is_file()
        clip = load_audio(audio_path)
        assert len(clip.samples) == SMOKE_SAMPLE_RATE * 2
        self.audio = audio_path
        return EmotionResult(
            utterance_id=utterance_id,
            probabilities=dict(_DISTRIBUTION),
            operational_state="stable",
            negative_intensity=0.30,
            confidence=0.60,
            provider=ProviderProvenance(
                name="wav2vec-xls-r",
                model="wav2vec2-xls-r-300m-seven-emotion",
                remote=False,
                revision=self.revision,
            ),
            usage=ProviderUsage(latency_ms=12.5),
        )

    def unload(self) -> None:
        self.unloaded = True


def _factory(revision: str, sink: list[_FakeProvider]) -> Any:
    def build(checkpoint: Path, base_model: Path, device: Device) -> _FakeProvider:
        assert device == "cpu"
        provider = _FakeProvider(checkpoint, base_model, revision)
        sink.append(provider)
        return provider

    return build


def test_smoke_audio_is_a_deterministic_private_sixteen_kilohertz_pcm16_clip(
    tmp_path: Path,
) -> None:
    first = write_smoke_audio(tmp_path / "a.wav")
    second = write_smoke_audio(tmp_path / "b.wav")

    assert first.stat().st_mode & 0o077 == 0
    assert first.read_bytes() == second.read_bytes()
    with wave.open(str(first), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getframerate() == SMOKE_SAMPLE_RATE
        assert reader.getsampwidth() == 2
        assert reader.getcomptype() == "NONE"
    assert len(load_audio(first).samples) == SMOKE_SAMPLE_RATE * 2

    with pytest.raises(FileExistsError):
        write_smoke_audio(first)


def test_release_smoke_reloads_offline_from_the_bundle_and_reports_aggregates(
    tmp_path: Path,
) -> None:
    sources = _Sources(tmp_path)
    bundle = sources.build((tmp_path / "release").resolve())
    providers: list[_FakeProvider] = []
    environment: dict[str, str] = {}

    result = run_release_smoke(
        bundle,
        verifier=_verifier,
        provider_factory=_factory(sources.digest, providers),
        environment=environment,
    )

    assert environment == dict(OFFLINE_ENVIRONMENT)
    assert result.release_id == "xls-r-emotion-7class-v1"
    assert result.candidate_checkpoint_sha256 == sources.digest
    assert result.device == "cpu"
    assert result.top_label == "neutral"
    assert result.top_label in CANONICAL_LABELS
    assert math.isclose(result.confidence, 0.60)
    assert math.isclose(result.probability_sum, 1.0, abs_tol=1e-9)
    assert math.isclose(result.negative_intensity, 0.30)
    assert math.isclose(result.latency_ms, 12.5)

    provider = providers[0]
    assert provider.checkpoint == bundle / "checkpoint"
    assert provider.base_model == bundle / "base-model"
    assert provider.unloaded
    assert provider.audio is not None and not provider.audio.exists()
    assert not provider.audio.parent.exists()


def test_release_smoke_refuses_a_tampered_bundle(tmp_path: Path) -> None:
    sources = _Sources(tmp_path)
    bundle = sources.build((tmp_path / "release").resolve())
    card = bundle / "MODEL_CARD.md"
    card.chmod(0o600)
    card.write_text(card.read_text() + "tampered\n")
    providers: list[_FakeProvider] = []

    with pytest.raises(SmokeError) as error:
        run_release_smoke(
            bundle,
            verifier=_verifier,
            provider_factory=_factory(sources.digest, providers),
            environment={},
        )

    assert error.value.code == "release_verification_failed"
    assert not providers


def test_release_smoke_reports_reload_inference_and_identity_failures_separately(
    tmp_path: Path,
) -> None:
    sources = _Sources(tmp_path)
    bundle = sources.build((tmp_path / "release").resolve())

    def broken_reload(checkpoint: Path, base_model: Path, device: Device) -> _FakeProvider:
        raise RuntimeError("no weights")

    with pytest.raises(SmokeError) as reload_error:
        run_release_smoke(
            bundle, verifier=_verifier, provider_factory=broken_reload, environment={}
        )
    assert reload_error.value.code == "release_reload_failed"

    class _Failing(_FakeProvider):
        def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
            raise RuntimeError("forward failed")

    def failing(checkpoint: Path, base_model: Path, device: Device) -> _FakeProvider:
        return _Failing(checkpoint, base_model, sources.digest)

    with pytest.raises(SmokeError) as inference_error:
        run_release_smoke(bundle, verifier=_verifier, provider_factory=failing, environment={})
    assert inference_error.value.code == "smoke_inference_failed"

    providers: list[_FakeProvider] = []
    with pytest.raises(SmokeError) as identity_error:
        run_release_smoke(
            bundle,
            verifier=_verifier,
            provider_factory=_factory("9" * 64, providers),
            environment={},
        )
    assert identity_error.value.code == "smoke_checkpoint_identity_mismatch"


def test_release_smoke_rejects_an_output_that_is_not_the_requested_utterance(
    tmp_path: Path,
) -> None:
    sources = _Sources(tmp_path)
    bundle = sources.build((tmp_path / "release").resolve())

    class _Mislabelled(_FakeProvider):
        def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
            return super().analyze("other-utterance", audio_path, transcript)

    def mislabelled(checkpoint: Path, base_model: Path, device: Device) -> _FakeProvider:
        return _Mislabelled(checkpoint, base_model, sources.digest)

    with pytest.raises(SmokeError) as error:
        run_release_smoke(bundle, verifier=_verifier, provider_factory=mislabelled, environment={})
    assert error.value.code == "smoke_invalid_output"


def test_smoke_cli_prints_stable_non_sensitive_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sources = _Sources(tmp_path)
    bundle = sources.build((tmp_path / "release").resolve())
    module = _script()
    providers: list[_FakeProvider] = []
    build = _factory(sources.digest, providers)

    def ok(target: Path, **kwargs: Any) -> Any:
        return run_release_smoke(
            target, verifier=_verifier, provider_factory=build, environment={}, **kwargs
        )

    monkeypatch.setattr(module, "run_release_smoke", ok)
    assert module.main(["--bundle", str(bundle)]) == 0
    output = capsys.readouterr().out
    assert output.startswith("release_smoke_ok\n")
    assert f"candidate_checkpoint_sha256 {sources.digest}" in output
    assert "top_label neutral" in output
    assert "device cpu" in output
    assert str(tmp_path) not in output

    def broken(target: Path, **kwargs: Any) -> Any:
        raise SmokeError("release_reload_failed")

    monkeypatch.setattr(module, "run_release_smoke", broken)
    assert module.main(["--bundle", str(bundle)]) == 2
    assert capsys.readouterr().out == "release_reload_failed\n"

    def exploded(target: Path, **kwargs: Any) -> Any:
        raise OSError("boom")

    monkeypatch.setattr(module, "run_release_smoke", exploded)
    assert module.main(["--bundle", str(bundle)]) == 2
    assert capsys.readouterr().out == "release_smoke_error\n"
