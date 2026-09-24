"""Lazy local Qwen3-ASR adapter with explicit hardware and memory profiles."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Literal, Protocol, cast

from voxdelta.domain.models import AudioAsset, ProviderProvenance, SpeakerSegment, Utterance
from voxdelta.providers._asr_runtime import (
    activate_candidate,
    prepare_candidate_load,
    release_candidate,
)
from voxdelta.providers.asr_alignment import (
    LOCAL_ASR_INFERENCE_LOCK,
    AlignedWord,
    align_mixed,
    align_separate,
    validate_asset,
    validate_mixed_segments,
)
from voxdelta.providers.base import ProviderError
from voxdelta.providers.qwen_timestamps import TimestampCoverage, sanitize_words

DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
LOW_MEMORY_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"


class _QwenResult(Protocol):
    text: object
    time_stamps: object


class _AlignUnit(Protocol):
    text: object


class _QwenModel(Protocol):
    def transcribe(self, audio: str, *, language: str, return_time_stamps: bool) -> object: ...


class _Aligner(Protocol):
    def align(self, audio: str, text: str, *, language: str) -> object: ...


class ModelFactory(Protocol):
    def __call__(
        self,
        model_id: str,
        *,
        aligner_id: str,
        device: str,
        dtype: str,
    ) -> _QwenModel: ...


class AlignerFactory(Protocol):
    def __call__(self, model_id: str, *, device: str, dtype: str) -> _Aligner: ...


def _torch_dtype(torch: object, dtype: str) -> object:
    return getattr(torch, dtype)


def _default_model_factory(
    model_id: str, *, aligner_id: str, device: str, dtype: str
) -> _QwenModel:
    qwen = import_module("qwen_asr")
    torch = import_module("torch")
    model_class = qwen.Qwen3ASRModel
    return cast(
        _QwenModel,
        model_class.from_pretrained(
            model_id,
            forced_aligner=aligner_id,
            forced_aligner_kwargs={
                "device_map": device,
                "dtype": _torch_dtype(torch, dtype),
            },
            device_map=device,
            dtype=_torch_dtype(torch, dtype),
            max_inference_batch_size=1,
        ),
    )


def _default_aligner_factory(model_id: str, *, device: str, dtype: str) -> _Aligner:
    qwen = import_module("qwen_asr")
    torch = import_module("torch")
    return cast(
        _Aligner,
        qwen.Qwen3ForcedAligner.from_pretrained(
            model_id, device_map=device, dtype=_torch_dtype(torch, dtype)
        ),
    )


def _hardware() -> tuple[bool, bool]:
    torch = import_module("torch")
    cuda = bool(torch.cuda.is_available())
    mps = bool(torch.backends.mps.is_available())
    return cuda, mps


def _runtime_code(
    error: Exception,
) -> Literal["provider_timeout", "provider_runtime_unsupported", "provider_unavailable"]:
    if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
        return "provider_timeout"
    message = str(error).lower()
    unsupported = ("out of memory", "mps", "not implemented", "unsupported")
    if any(marker in message for marker in unsupported):
        return "provider_runtime_unsupported"
    return "provider_unavailable"


def _result(output: object) -> _QwenResult:
    try:
        results = list(cast(Iterable[object], output))
        if len(results) != 1:
            raise ProviderError("invalid_provider_output")
        result = cast(_QwenResult, results[0])
        if not isinstance(result.text, str):
            raise ProviderError("invalid_provider_output")
        return result
    except ProviderError:
        raise
    except Exception:
        raise ProviderError("invalid_provider_output") from None


def _time_records(raw: object) -> Iterable[tuple[object, object, object]]:
    try:
        for raw_unit in cast(Iterable[object], raw):
            unit = cast(_AlignUnit, raw_unit)
            text = unit.text
            start = getattr(unit, "start_time", getattr(unit, "start", None))
            end = getattr(unit, "end_time", getattr(unit, "end", None))
            yield start, end, text
    except ProviderError:
        raise
    except Exception:
        raise ProviderError("invalid_provider_output") from None


class Qwen3AsrProvider:
    """Run Qwen3-ASR without fallback to another provider."""

    def __init__(
        self,
        *,
        profile: Literal["default", "low-memory"] = "default",
        device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        model_id: str | None = None,
        model_factory: ModelFactory | None = None,
        aligner_factory: AlignerFactory | None = None,
        hardware_probe: Callable[[], tuple[bool, bool]] | None = None,
    ) -> None:
        if profile not in {"default", "low-memory"} or device not in {
            "auto",
            "cpu",
            "cuda",
            "mps",
        }:
            raise ProviderError("provider_unavailable")
        selected_id = DEFAULT_MODEL_ID if profile == "default" else LOW_MEMORY_MODEL_ID
        if model_id is not None and model_id != selected_id:
            raise ProviderError("provider_unavailable")
        self.provenance = ProviderProvenance(
            name="qwen3-asr",
            model=selected_id.removeprefix("Qwen/"),
            remote=False,
            revision=f"{selected_id}+{ALIGNER_MODEL_ID}",
        )
        self._model_id = selected_id
        self._requested_device = device
        self._model_factory = model_factory or _default_model_factory
        self._aligner_factory = aligner_factory or _default_aligner_factory
        self._uses_embedded_aligner = aligner_factory is None
        self._hardware_probe = hardware_probe or _hardware
        self._model: _QwenModel | None = None
        self._aligner: _Aligner | None = None
        self._runtime: tuple[str, str] | None = None
        self.last_warning_count = 0
        #: Coverage of the most recent transcription under the timestamp sanitation
        #: contract. Read it to tell a complete transcript from one with holes.
        self.last_timestamp_coverage: TimestampCoverage | None = None

    def _select_runtime(self) -> tuple[str, str]:
        if self._runtime is not None:
            return self._runtime
        cuda, mps = self._hardware_probe()
        if self._requested_device == "auto":
            device = "cuda" if cuda else "mps" if mps else "cpu"
        else:
            device = self._requested_device
            if (device == "cuda" and not cuda) or (device == "mps" and not mps):
                raise ProviderError("provider_runtime_unsupported")
        dtype = "bfloat16" if device == "cuda" else "float16" if device == "mps" else "float32"
        self._runtime = device, dtype
        return self._runtime

    def _load(self) -> _QwenModel:
        if self._model is not None:
            return self._model
        device, dtype = self._select_runtime()
        prepare_candidate_load(self)
        try:
            model = self._model_factory(
                self._model_id,
                aligner_id=ALIGNER_MODEL_ID,
                device=device,
                dtype=dtype,
            )
        except ProviderError:
            self._release()
            release_candidate(self)
            raise
        except Exception as error:
            self._release()
            release_candidate(self)
            raise ProviderError(_runtime_code(error)) from None
        self._model = model
        activate_candidate(self, self._release)
        return self._model

    def _load_aligner(self) -> _Aligner:
        if self._aligner is not None:
            return self._aligner
        device, dtype = self._select_runtime()
        try:
            self._aligner = self._aligner_factory(ALIGNER_MODEL_ID, device=device, dtype=dtype)
        except ProviderError:
            self._release()
            release_candidate(self)
            raise
        except Exception as error:
            self._release()
            release_candidate(self)
            raise ProviderError(_runtime_code(error)) from None
        return self._aligner

    def _release(self) -> None:
        model = self._model
        aligner = self._aligner
        self._model = None
        self._aligner = None
        for candidate in (aligner, model):
            close = getattr(candidate, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def unload(self) -> None:
        with LOCAL_ASR_INFERENCE_LOCK:
            self._release()
            release_candidate(self)

    def _transcribe_path(
        self, path: str, duration: float, segments: list[SpeakerSegment] | None
    ) -> tuple[list[AlignedWord], TimestampCoverage]:
        try:
            result = _result(
                self._load().transcribe(
                    path, language="Korean", return_time_stamps=self._uses_embedded_aligner
                )
            )
            raw_times = result.time_stamps
            if not self._uses_embedded_aligner:
                aligned = self._load_aligner().align(
                    path, cast(str, result.text), language="Korean"
                )
                try:
                    aligned_results = list(cast(Iterable[object], aligned))
                except Exception:
                    raise ProviderError("invalid_provider_output") from None
                if (
                    len(aligned_results) == 1
                    and not hasattr(aligned_results[0], "start")
                    and not hasattr(aligned_results[0], "start_time")
                ):
                    raw_times = aligned_results[0]
                else:
                    raw_times = aligned_results
            if raw_times is None:
                raise ProviderError("invalid_provider_output")
            # The Qwen aligner reports zero-duration instants; the sanitation contract
            # decides which of them may be attributed, and never invents a span.
            return sanitize_words(_time_records(raw_times), duration, segments)
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(_runtime_code(error)) from None

    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]:
        duration, paths = validate_asset(asset)
        if asset.channel_mode == "mixed":
            validate_mixed_segments(segments, duration)
        with LOCAL_ASR_INFERENCE_LOCK:
            if asset.channel_mode == "separate":
                channels = [self._transcribe_path(path, duration, None) for path in paths]
                result, omitted = align_separate([words for words, _ in channels])
                coverage = channels[0][1]
                for _words, channel_coverage in channels[1:]:
                    coverage = coverage.merged(channel_coverage)
            else:
                words, coverage = self._transcribe_path(paths[0], duration, segments)
                result, omitted = align_mixed(words, segments, duration)
            self.last_warning_count = omitted
            self.last_timestamp_coverage = coverage.with_alignment_omissions(omitted)
        return result

    def transcribe_single_speaker(self, asset: AudioAsset) -> list[Utterance]:
        """Transcribe a known single-speaker recording for offline evaluation only.

        This deliberately bypasses diarization attribution.  The public pipeline still
        requires two observed speakers before it offers role confirmation; callers of
        this method must therefore report diarization as unmeasured rather than treating
        its one synthetic speaker as a diarization result.
        """

        duration, paths = validate_asset(asset)
        if asset.channel_mode != "mixed":
            raise ProviderError("invalid_audio_asset")
        with LOCAL_ASR_INFERENCE_LOCK:
            words, coverage = self._transcribe_path(paths[0], duration, None)
            if not words:
                raise ProviderError("invalid_provider_output")
            self.last_warning_count = 0
            self.last_timestamp_coverage = coverage
        return [
            Utterance(
                id="utt-0001",
                start=min(word.start for word in words),
                end=max(word.end for word in words),
                speaker_id="SPEAKER_00",
                confidence=1.0,
                transcript=" ".join(word.text for word in words),
            )
        ]
