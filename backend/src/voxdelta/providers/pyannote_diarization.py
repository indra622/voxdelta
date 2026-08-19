"""Lazy, local pyannote Community-1 speaker diarization adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import SecretStr

from voxdelta.credentials import Credentials
from voxdelta.domain.models import AudioAsset, ProviderProvenance, SpeakerSegment
from voxdelta.providers.base import DiarizationTimelines, ProviderError, ProviderErrorCode
from voxdelta.providers.checkpoints import checkpoint_tree_digest

COMMUNITY_MODEL_ID = "pyannote/speaker-diarization-community-1"


class _Annotation(Protocol):
    def itertracks(self, *, yield_label: bool) -> Iterable[tuple[object, object, object]]: ...


class _Pipeline(Protocol):
    def __call__(self, audio_path: str, **kwargs: int) -> object: ...


class PipelineFactory(Protocol):
    def __call__(self, model: str, *, token: str | None) -> _Pipeline: ...


class _Segment(Protocol):
    start: object
    end: object


class _TelemetryDisabledPipeline:
    def __init__(
        self,
        pipeline: _Pipeline,
        set_telemetry_metrics: Callable[[bool], None],
    ) -> None:
        self._pipeline = pipeline
        self._set_telemetry_metrics = set_telemetry_metrics

    def __call__(self, audio_path: str, **kwargs: int) -> object:
        self._set_telemetry_metrics(False)
        return self._pipeline(audio_path, **kwargs)


@dataclass(frozen=True, slots=True, order=True)
class _Turn:
    start: float
    end: float
    label: str


def _default_pipeline_factory(model: str, *, token: str | None) -> _Pipeline:
    """Import pyannote only at the exact point the selected provider is first used."""

    pyannote_audio = import_module("pyannote.audio")
    telemetry = import_module("pyannote.audio.telemetry")
    set_telemetry_metrics = cast(Callable[[bool], None], telemetry.set_telemetry_metrics)
    set_telemetry_metrics(False)
    pipeline_class = pyannote_audio.Pipeline
    pipeline = cast(_Pipeline, pipeline_class.from_pretrained(model, token=token))
    return _TelemetryDisabledPipeline(pipeline, set_telemetry_metrics)


def _local_checkpoint(model_path: Path) -> tuple[str, str, str]:
    try:
        if model_path.is_symlink():
            raise ProviderError("invalid_local_checkpoint")
        resolved = model_path.resolve(strict=True)
        if resolved.is_dir():
            config = resolved / "config.yaml"
            if config.is_symlink() or not config.is_file():
                raise ProviderError("invalid_local_checkpoint")
            model_name = resolved.name
        elif resolved.is_file() and resolved.name == "config.yaml":
            model_name = resolved.parent.name
        else:
            raise ProviderError("invalid_local_checkpoint")
    except ProviderError:
        raise
    except (OSError, RuntimeError):
        raise ProviderError("invalid_local_checkpoint") from None
    if not model_name:
        raise ProviderError("invalid_local_checkpoint")
    digest_root = resolved if resolved.is_dir() else resolved.parent
    try:
        digest = checkpoint_tree_digest(digest_root)
    except ValueError:
        raise ProviderError("invalid_local_checkpoint") from None
    return str(resolved), model_name, digest


def _safe_provider_error(error: Exception) -> ProviderError:
    code: ProviderErrorCode = (
        "provider_timeout"
        if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower()
        else "provider_unavailable"
    )
    return ProviderError(code)


def _annotation(output: object, attribute: str) -> _Annotation:
    try:
        annotation = getattr(output, attribute)
        iterator = annotation.itertracks
    except Exception:
        raise ProviderError("invalid_provider_output") from None
    if not callable(iterator):
        raise ProviderError("invalid_provider_output")
    return cast(_Annotation, annotation)


def _finite_number(value: object) -> float:
    if isinstance(value, bool):
        raise ProviderError("invalid_provider_output")
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ProviderError("invalid_provider_output") from None
    if not isfinite(converted):
        raise ProviderError("invalid_provider_output")
    return converted


def _turns(annotation: _Annotation, duration: float) -> list[_Turn]:
    turns: set[_Turn] = set()
    try:
        records = annotation.itertracks(yield_label=True)
        for record in records:
            segment, _track, raw_label = record
            if not isinstance(raw_label, str) or not raw_label:
                raise ProviderError("invalid_provider_output")
            typed_segment = cast(_Segment, segment)
            start = _finite_number(typed_segment.start)
            end = _finite_number(typed_segment.end)
            if end <= start:
                raise ProviderError("invalid_provider_output")
            bounded_start = max(0.0, min(duration, start))
            bounded_end = max(0.0, min(duration, end))
            if bounded_end <= bounded_start:
                raise ProviderError("invalid_provider_output")
            turns.add(_Turn(bounded_start, bounded_end, raw_label))
    except ProviderError:
        raise
    except Exception:
        raise ProviderError("invalid_provider_output") from None
    if not turns:
        raise ProviderError("invalid_provider_output")
    return sorted(turns)


def _speaker_map(turns: list[_Turn]) -> dict[str, str]:
    labels = list(dict.fromkeys(turn.label for turn in turns))
    if len(labels) != 2:
        raise ProviderError("unsupported_speaker_count")
    return {label: f"SPEAKER_{index:02d}" for index, label in enumerate(labels)}


def _segments(
    turns: list[_Turn],
    speakers: dict[str, str],
    *,
    mark_overlaps: bool,
) -> list[SpeakerSegment]:
    if any(turn.label not in speakers for turn in turns):
        raise ProviderError("invalid_provider_output")
    overlaps = [False] * len(turns)
    if mark_overlaps:
        for index, current in enumerate(turns):
            overlaps[index] = any(
                current.label != other.label
                and current.start < other.end
                and other.start < current.end
                for other in turns
            )
    return [
        SpeakerSegment(
            start=turn.start,
            end=turn.end,
            speaker_id=speakers[turn.label],
            overlap=overlaps[index],
            confidence=1.0,
        )
        for index, turn in enumerate(turns)
    ]


def _exclusive_segments(turns: list[_Turn], speakers: dict[str, str]) -> list[SpeakerSegment]:
    if {turn.label for turn in turns} != set(speakers):
        raise ProviderError("invalid_provider_output")
    if any(
        current.end > following.start for current, following in zip(turns, turns[1:], strict=False)
    ):
        raise ProviderError("invalid_provider_output")
    return _segments(turns, speakers, mark_overlaps=False)


def _channel_segments(
    annotation: _Annotation, duration: float, speaker_id: str
) -> list[SpeakerSegment]:
    turns = _turns(annotation, duration)
    if len({turn.label for turn in turns}) != 1:
        raise ProviderError("invalid_provider_output")
    return [
        SpeakerSegment(
            start=turn.start,
            end=turn.end,
            speaker_id=speaker_id,
            overlap=False,
            confidence=1.0,
        )
        for turn in turns
    ]


class _PyannoteAdapter:
    def __init__(
        self,
        credentials: Credentials,
        *,
        model_reference: str,
        provenance: ProviderProvenance,
        credential_name: Literal["huggingface", "pyannoteai"],
        local_checkpoint: bool,
        pipeline_factory: PipelineFactory | None,
    ) -> None:
        self.provenance = provenance
        self._credentials = credentials
        self._model_reference = model_reference
        self._credential_name = credential_name
        self._local_checkpoint = local_checkpoint
        self._pipeline_factory = pipeline_factory or _default_pipeline_factory
        self._pipeline: _Pipeline | None = None

    def _secret(self) -> SecretStr | None:
        if self._credential_name == "huggingface":
            return self._credentials.huggingface_token
        return self._credentials.pyannoteai_api_key

    def _missing_code(self) -> ProviderErrorCode:
        if self._credential_name == "huggingface":
            return "missing_huggingface_token"
        return "missing_pyannote_api_key"

    def _load_pipeline(self) -> _Pipeline:
        if self._pipeline is not None:
            return self._pipeline
        secret = None if self._local_checkpoint else self._secret()
        if secret is None and not self._local_checkpoint:
            raise ProviderError(self._missing_code())
        try:
            self._pipeline = self._pipeline_factory(
                self._model_reference,
                token=None if secret is None else secret.get_secret_value(),
            )
        except ProviderError:
            raise
        except Exception as error:
            raise _safe_provider_error(error) from None
        return self._pipeline

    @staticmethod
    def _validate_asset(asset: AudioAsset) -> tuple[float, tuple[str, ...]]:
        duration = asset.duration_seconds
        if duration is None or not isfinite(duration) or duration <= 0:
            raise ProviderError("invalid_audio_asset")
        paths = asset.normalized_paths
        if asset.channel_mode == "mixed":
            expected_paths = 1
        elif asset.channel_mode == "separate":
            expected_paths = 2
        else:
            expected_paths = 0
        if len(paths) != expected_paths or any(not path for path in paths):
            raise ProviderError("invalid_audio_asset")
        return duration, paths

    def _run(self, path: str, **kwargs: int) -> object:
        pipeline = self._load_pipeline()
        try:
            return pipeline(path, **kwargs)
        except ProviderError:
            raise
        except Exception as error:
            raise _safe_provider_error(error) from None

    def diarize_timelines(self, asset: AudioAsset) -> DiarizationTimelines:
        duration, paths = self._validate_asset(asset)
        if asset.channel_mode == "separate":
            channel_results: list[SpeakerSegment] = []
            for index, path in enumerate(paths):
                output = self._run(path, num_speakers=1)
                channel_results.extend(
                    _channel_segments(
                        _annotation(output, "speaker_diarization"),
                        duration,
                        f"SPEAKER_{index:02d}",
                    )
                )
            ordered = sorted(
                channel_results,
                key=lambda item: (item.start, item.end, item.speaker_id),
            )
            return DiarizationTimelines(evidence=ordered, exclusive=list(ordered))

        output = self._run(paths[0], min_speakers=1, max_speakers=4)
        evidence_turns = _turns(_annotation(output, "speaker_diarization"), duration)
        speakers = _speaker_map(evidence_turns)
        exclusive_turns = _turns(_annotation(output, "exclusive_speaker_diarization"), duration)
        return DiarizationTimelines(
            evidence=_segments(evidence_turns, speakers, mark_overlaps=True),
            exclusive=_exclusive_segments(exclusive_turns, speakers),
        )

    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]:
        return self.diarize_timelines(asset).evidence

    def diarize_for_alignment(self, asset: AudioAsset) -> list[SpeakerSegment]:
        """Return Community-1's exclusive timeline for transcript reconciliation."""

        return self.diarize_timelines(asset).exclusive


class PyannoteDiarizationProvider(_PyannoteAdapter):
    """Run Community-1 locally using an injected credential or checkpoint."""

    def __init__(
        self,
        credentials: Credentials,
        *,
        model_path: Path | None = None,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        if model_path is None:
            model_reference = COMMUNITY_MODEL_ID
            model_name = "speaker-diarization-community-1"
            revision = "community-1"
            local_checkpoint = False
        else:
            model_reference, model_name, revision = _local_checkpoint(model_path)
            local_checkpoint = True
        super().__init__(
            credentials,
            model_reference=model_reference,
            provenance=ProviderProvenance(
                name="pyannote",
                model=model_name,
                remote=False,
                revision=revision,
            ),
            credential_name="huggingface",
            local_checkpoint=local_checkpoint,
            pipeline_factory=pipeline_factory,
        )
