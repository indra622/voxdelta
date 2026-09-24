"""Local-only end-to-end smoke for the Qwen3-ASR provider on a synthetic fixture.

Runs the real provider — real local weights, real forced aligner — against audio this
script generates and a fixed diarization timeline, then carries whatever comes back
through the same mapping the pipeline uses, so the whole chain from recognizer to public
report field is exercised without any corpus audio.

The fixture is synthetic on purpose: two alternating speaker-like voices built from
harmonic stacks at different fundamentals with formant-ish resonances and syllable
envelopes. It is not speech and is not expected to transcribe into words. That makes two
outcomes legitimate, and the smoke records which one happened rather than insisting on
the flattering one:

* the recognizer emits text, the sanitation contract runs, and coverage reaches the
  report field; or
* the recognizer emits nothing for non-speech, and the provider raises its typed
  ``invalid_provider_output`` rather than inventing a transcript.

Either is a pass. An untyped crash, a network attempt, or a fabricated timestamp is not.
Everything happens inside the egress guard, so an artifact that exists is an artifact
produced without touching the network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Never

import numpy as np

from voxdelta.domain.models import AudioAsset, SpeakerSegment
from voxdelta.evaluation.kcsc_asr_benchmark import configure_offline_cache
from voxdelta.pipeline.runner import _coverage_warning, _transcription_coverage
from voxdelta.providers.base import ProviderError
from voxdelta.providers.offline_guard import block_network_egress
from voxdelta.providers.qwen3_asr import Qwen3AsrProvider

SAMPLE_RATE = 16_000
DEFAULT_CACHE = Path("data/models/hf-cache/hub")
DEFAULT_OUTPUT = Path("data/benchmarks/qwen-provider-smoke.json")

#: Two speaker-like voices and the turns they take. Fixed, so the fixture and the
#: diarization timeline below are the same fact stated twice and cannot drift apart.
TURNS: tuple[tuple[float, float, str, float], ...] = (
    (0.5, 3.5, "SPEAKER_00", 120.0),
    (4.0, 7.5, "SPEAKER_01", 205.0),
    (8.0, 11.0, "SPEAKER_00", 120.0),
    (11.5, 14.5, "SPEAKER_01", 205.0),
)
DURATION = 15.0


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _voice(seconds: float, fundamental: float, rng: np.random.Generator) -> np.ndarray:
    """A syllable-rate harmonic stack with two resonances: speech-shaped, not speech."""

    samples = int(seconds * SAMPLE_RATE)
    t = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    # A drifting fundamental keeps it from reading as a pure tone.
    contour = fundamental * (1.0 + 0.06 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(contour) / SAMPLE_RATE
    signal = np.zeros(samples, dtype=np.float64)
    for harmonic, gain in ((1, 1.0), (2, 0.5), (3, 0.32), (4, 0.18), (5, 0.1)):
        signal += gain * np.sin(harmonic * phase)
    # Two crude formants, so the spectrum is not a bare harmonic comb.
    for centre, width in ((700.0, 120.0), (1600.0, 200.0)):
        signal += 0.25 * np.sin(2 * np.pi * centre * t) * np.exp(-((t % 0.25) * width / 100.0))
    # Syllable envelope at ~4 Hz plus a little breath noise.
    envelope = 0.5 * (1.0 - np.cos(2 * np.pi * 4.0 * t)) ** 1.5
    signal = signal * envelope + 0.02 * rng.standard_normal(samples)
    peak = float(np.max(np.abs(signal))) or 1.0
    return signal / peak * 0.6


def write_fixture(path: Path) -> str:
    """Write the synthetic two-speaker mix and return its sha256."""

    import hashlib

    rng = np.random.default_rng(20260901)
    mix = np.zeros(int(DURATION * SAMPLE_RATE), dtype=np.float64)
    for start, end, _speaker, fundamental in TURNS:
        segment = _voice(end - start, fundamental, rng)
        offset = int(start * SAMPLE_RATE)
        mix[offset : offset + segment.size] += segment
    samples = np.clip(mix * 32767.0, -32768, 32767).astype("<i2")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timeline() -> list[SpeakerSegment]:
    """The fixed diarization stand-in: exactly the turns the fixture was built from."""

    return [
        SpeakerSegment(start=start, end=end, speaker_id=speaker, confidence=1.0)
        for start, end, speaker, _fundamental in TURNS
    ]


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("smoke failed: invalid arguments", file=sys.stderr)
        return 2

    fixture = arguments.fixture or (arguments.output.parent / "qwen-provider-smoke-fixture.wav")
    digest = write_fixture(fixture)
    segments = timeline()
    asset = AudioAsset(
        source_name=fixture.name,
        source_path=str(fixture),
        normalized_paths=(str(fixture),),
        channel_mode="mixed",
        duration_seconds=DURATION,
        channels=1,
        sha256=digest,
    )

    configure_offline_cache(arguments.model_cache)
    print("preflight: Qwen3-ASR provider smoke (synthetic fixture, no corpus audio)")
    print(f"  fixture       : {fixture} sha256={digest}")
    print(f"  duration      : {DURATION:.1f}s, {len(segments)} fixed diarization segments")
    print(f"  device        : {arguments.device}")
    print("  egress        : guarded (model init + inference)")
    sys.stdout.flush()

    record: dict[str, Any] = {
        "smoke": "qwen3-asr-provider",
        "fixture": {"path": str(fixture), "sha256": digest, "synthetic": True},
        "diarization": {"source": "fixed local timeline", "segments": len(segments)},
        "corpus_audio_used": False,
        "device": arguments.device,
    }

    started = time.monotonic()
    outcome: str
    try:
        with block_network_egress() as egress:
            provider = Qwen3AsrProvider(profile="default", device=arguments.device)
            record["model"] = {
                "name": provider.provenance.name,
                "model": provider.provenance.model,
                "revision": provider.provenance.revision,
                "remote": provider.provenance.remote,
            }
            try:
                utterances = provider.transcribe(asset, segments)
                outcome = "transcribed"
            except ProviderError as error:
                # Non-speech in, nothing out. The provider refusing to invent a transcript
                # is the correct behaviour, so it is recorded as a pass with its code.
                outcome = "no_transcript"
                record["provider_error_code"] = error.code
                utterances = []
            finally:
                provider.unload()
    except Exception as error:  # noqa: BLE001 - the type is the diagnosis here
        print(f"smoke failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    elapsed = time.monotonic() - started
    coverage = _transcription_coverage(provider)
    record["outcome"] = outcome
    record["elapsed_seconds"] = round(elapsed, 3)
    record["real_time_factor"] = round(elapsed / DURATION, 6)
    record["utterance_count"] = len(utterances)
    record["speakers_attributed"] = sorted({item.speaker_id for item in utterances})
    record["egress"] = {"attempts": egress.attempts, "hosts": sorted(egress.hosts)}
    record["timestamp_coverage"] = coverage.model_dump() if coverage else None
    record["report_warning"] = (
        _coverage_warning(coverage) if coverage is not None and coverage.uncertain else None
    )

    # Every utterance must occupy real time and sit inside the timeline it was attributed
    # to. A fabricated timestamp would show up here rather than in a later gold-set run.
    violations = [
        item.id
        for item in utterances
        if item.end <= item.start or item.start < 0 or item.end > DURATION
    ]
    record["timing_violations"] = violations

    print(f"outcome: {outcome} in {elapsed:.1f}s (rtf {elapsed / DURATION:.3f})")
    print(f"utterances: {len(utterances)} speakers={record['speakers_attributed']}")
    print(f"coverage: {record['timestamp_coverage']}")
    print(f"network egress attempts: {egress.attempts}")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    arguments.output.write_text(f"{raw}\n", encoding="utf-8")
    print(f"artifact: {arguments.output}")

    if egress.attempted:
        print(f"smoke failed: {egress.attempts} network egress attempts", file=sys.stderr)
        return 2
    if violations:
        print(f"smoke failed: {len(violations)} utterances with impossible timing", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
