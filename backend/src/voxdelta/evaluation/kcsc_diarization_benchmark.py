"""Score local diarization against the derived KCSC evaluation set.

The benchmark exists to answer one question honestly: how well does the diarization
provider VoxDelta actually ships do on real two-party Korean conversations? That is only
worth anything if the run cannot quietly degrade into something else, so four conditions
abort it rather than being reported as a caveat:

* **A missing local model or artifact.** No silent fallback to a hub download, and no
  substitute checkpoint. The pipeline is loaded from a named local directory whose tree
  digest is recorded in the report.
* **Any network attempt.** The run happens under an egress guard; a single blocked
  attempt invalidates the result, because a model that reached out is not the local
  model the report claims was measured.
* **A checksum disagreement.** Every audio and reference file is re-hashed against the
  derivation manifest before it is used. Scoring against a file that has drifted from
  the manifest would produce a number attributed to the wrong input.
* **Malformed input.** Reference turns must parse, stay inside the audio, and name only
  the two speakers the manifest declares.

The report is aggregate and transcript-free by construction: reference transcript is
read only to be discarded, and nothing downstream of :func:`_reference_annotation`
carries text. A test pins this.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from voxdelta.domain.models import AudioAsset, SpeakerSegment

SCHEMA_VERSION = "1"
REFERENCE_SCHEMA_VERSION = "1"
SPEAKERS_PER_CONVERSATION = 2

#: Conventional NIST-style scoring tolerance, and a strict variant with no tolerance at
#: all. Both are reported: the collar figure is what the literature quotes, the strict
#: figure is what a downstream consumer of exact turn boundaries would actually feel.
COLLAR_SECONDS = 0.25
STRICT_COLLAR_SECONDS = 0.0


class _Annotation(Protocol):
    """The slice of ``pyannote.core.Annotation`` this module builds and scores."""

    def __setitem__(self, segment: object, label: str) -> None: ...
    def labels(self) -> list[object]: ...


class _Metric(Protocol):
    def __call__(self, reference: object, hypothesis: object, **kwargs: object) -> object: ...


def _pyannote_core() -> tuple[Callable[..., _Annotation], Callable[[float, float], object]]:
    """Import pyannote only when a benchmark actually runs, and give it a typed face."""

    module = import_module("pyannote.core")
    return (
        cast(Callable[..., _Annotation], module.Annotation),
        cast(Callable[[float, float], object], module.Segment),
    )


def _pyannote_scope(duration: float) -> object:
    module = import_module("pyannote.core")
    segment = cast(Callable[[float, float], object], module.Segment)
    timeline = cast(Callable[[object], object], module.Timeline)
    return timeline([segment(0.0, duration)])


def _pyannote_metrics() -> tuple[Callable[..., _Metric], Callable[..., _Metric]]:
    module = import_module("pyannote.metrics.diarization")
    return (
        cast(Callable[..., _Metric], module.DiarizationErrorRate),
        cast(Callable[..., _Metric], module.JaccardErrorRate),
    )


class KcscBenchmarkError(RuntimeError):
    """Raised for any condition that would make a reported number untrustworthy."""


class Diarizer(Protocol):
    """The narrow slice of the diarization provider this benchmark depends on."""

    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]: ...


@dataclass(frozen=True, slots=True)
class ConversationScore:
    """Per-conversation diarization result. Carries timings and rates, never text."""

    conversation_id: str
    duration_seconds: float
    reference_speakers: tuple[str, ...]
    reference_turn_count: int
    reference_speech_seconds: float
    hypothesis_speaker_count: int
    hypothesis_segment_count: int
    hypothesis_speech_seconds: float
    elapsed_seconds: float
    metrics: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregate benchmark result. Carries no transcript and no file contents."""

    model_name: str
    #: ``None`` for a remote model, which has no local tree to hash. The report keeps the
    #: key either way so local and remote reports stay mechanically comparable.
    model_tree_sha256: str | None
    source_revision: str
    manifest_sha256: str
    scores: tuple[ConversationScore, ...]
    aggregate: Mapping[str, Mapping[str, float]]
    egress_attempts: int
    remote: bool = False

    @property
    def conversation_count(self) -> int:
        return len(self.scores)


def file_sha256(path: Path) -> str:
    """Stream a file into SHA-256; used for every checksum this benchmark verifies."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(derived_root: Path) -> Mapping[str, object]:
    """Read the derivation manifest, failing closed when it is absent or malformed."""

    path = derived_root / "manifest.json"
    if not path.is_file():
        raise KcscBenchmarkError(f"missing derivation manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise KcscBenchmarkError(f"derivation manifest is not valid JSON: {path}") from None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("conversations"), list):
        raise KcscBenchmarkError(f"derivation manifest has an unexpected shape: {path}")
    return manifest


def select_conversations(
    manifest: Mapping[str, object], conversation_ids: Sequence[str]
) -> tuple[Mapping[str, object], ...]:
    """Pick the requested conversations, failing closed on an unknown or repeated id."""

    if not conversation_ids:
        raise KcscBenchmarkError("no conversations selected")
    if len(set(conversation_ids)) != len(conversation_ids):
        raise KcscBenchmarkError("conversation selection contains duplicates")
    entries = manifest["conversations"]
    assert isinstance(entries, list)
    by_id = {
        entry["conversation_id"]: entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("conversation_id"), str)
    }
    missing = [identifier for identifier in conversation_ids if identifier not in by_id]
    if missing:
        raise KcscBenchmarkError(f"unknown conversation ids: {', '.join(sorted(missing))}")
    return tuple(by_id[identifier] for identifier in conversation_ids)


def verify_inputs(entry: Mapping[str, object], *, derived_root: Path) -> tuple[Path, Path]:
    """Re-hash one conversation's audio and reference against the manifest."""

    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        raise KcscBenchmarkError(f"{entry.get('conversation_id')}: manifest entry has no outputs")
    resolved: list[Path] = []
    for kind in ("audio", "reference"):
        relative = outputs.get(kind)
        expected = outputs.get(f"{kind}_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise KcscBenchmarkError(
                f"{entry.get('conversation_id')}: manifest {kind} is malformed"
            )
        path = derived_root / relative
        if not path.is_file():
            raise KcscBenchmarkError(f"missing derived {kind}: {path}")
        found = file_sha256(path)
        if found != expected:
            raise KcscBenchmarkError(
                f"{entry.get('conversation_id')}: {kind} checksum mismatch "
                f"(manifest {expected[:12]}…, file {found[:12]}…)"
            )
        resolved.append(path)
    return resolved[0], resolved[1]


def _reference_annotation(
    reference_path: Path, entry: Mapping[str, object]
) -> tuple[object, tuple[str, ...], float, float, int]:
    """Build a pyannote ``Annotation`` from the reference turns, discarding transcript.

    Transcript text is read off disk here and deliberately never bound to a name: the
    annotation carries only intervals and speaker labels, so no caller downstream of
    this function can place text into the report even by accident.
    """

    annotation_class, segment_class = _pyannote_core()

    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise KcscBenchmarkError(f"reference is not valid JSON: {reference_path}") from None
    if not isinstance(reference, dict):
        raise KcscBenchmarkError(f"reference has an unexpected shape: {reference_path}")
    if reference.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise KcscBenchmarkError(
            f"{reference_path.name}: unsupported reference schema "
            f"{reference.get('schema_version')!r}"
        )
    if reference.get("conversation_id") != entry.get("conversation_id"):
        raise KcscBenchmarkError(f"{reference_path.name}: conversation id does not match manifest")

    duration = reference.get("duration_seconds")
    if not isinstance(duration, int | float) or duration <= 0:
        raise KcscBenchmarkError(f"{reference_path.name}: duration is missing or not positive")
    declared = entry.get("derived_duration_seconds")
    if isinstance(declared, int | float) and abs(float(declared) - float(duration)) > 1e-6:
        raise KcscBenchmarkError(f"{reference_path.name}: duration disagrees with manifest")

    turns = reference.get("turns")
    if not isinstance(turns, list) or not turns:
        raise KcscBenchmarkError(f"{reference_path.name}: reference has no turns")
    expected_speakers = entry.get("speakers")
    if not isinstance(expected_speakers, list) or len(expected_speakers) != (
        SPEAKERS_PER_CONVERSATION
    ):
        raise KcscBenchmarkError(f"{reference_path.name}: manifest speaker list is malformed")

    annotation = annotation_class(uri=str(entry.get("conversation_id")))
    speech = 0.0
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise KcscBenchmarkError(f"{reference_path.name}: turn {index} is not an object")
        start, end, speaker = turn.get("start"), turn.get("end"), turn.get("speaker")
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise KcscBenchmarkError(f"{reference_path.name}: turn {index} has non-numeric timing")
        if not isinstance(speaker, str) or speaker not in expected_speakers:
            raise KcscBenchmarkError(
                f"{reference_path.name}: turn {index} names an unknown speaker"
            )
        if start < 0 or end <= start or end > float(duration) + 1e-6:
            raise KcscBenchmarkError(f"{reference_path.name}: turn {index} falls outside the audio")
        annotation[segment_class(float(start), float(end))] = speaker
        speech += float(end) - float(start)

    labels = tuple(sorted(str(label) for label in annotation.labels()))
    if len(labels) != SPEAKERS_PER_CONVERSATION:
        raise KcscBenchmarkError(
            f"{reference_path.name}: reference names {len(labels)} speakers, "
            f"expected {SPEAKERS_PER_CONVERSATION}"
        )
    return annotation, labels, float(duration), speech, len(turns)


def _hypothesis_annotation(segments: Sequence[SpeakerSegment], uri: str) -> object:
    if not segments:
        raise KcscBenchmarkError(f"{uri}: diarization returned no segments")
    annotation_class, segment_class = _pyannote_core()
    annotation = annotation_class(uri=uri)
    for segment in segments:
        annotation[segment_class(segment.start, segment.end)] = segment.speaker_id
    return annotation


def _score(reference: object, hypothesis: object, duration: float) -> dict[str, dict[str, float]]:
    """Compute DER components and JER under a collar and under no tolerance at all."""

    error_rate_class, jaccard_class = _pyannote_metrics()
    scope = _pyannote_scope(duration)
    results: dict[str, dict[str, float]] = {}
    for name, collar in (("collar_250ms", COLLAR_SECONDS), ("strict", STRICT_COLLAR_SECONDS)):
        metric = error_rate_class(collar=collar, skip_overlap=False)
        components = cast(
            Mapping[str, float], metric(reference, hypothesis, uem=scope, detailed=True)
        )
        total = float(components["total"])
        if total <= 0:
            raise KcscBenchmarkError("reference contains no scorable speech")
        jaccard = jaccard_class(collar=collar, skip_overlap=False)
        results[name] = {
            "der": round(float(components["diarization error rate"]), 6),
            "miss": round(float(components["missed detection"]) / total, 6),
            "false_alarm": round(float(components["false alarm"]) / total, 6),
            "confusion": round(float(components["confusion"]) / total, 6),
            "jer": round(float(cast(float, jaccard(reference, hypothesis, uem=scope))), 6),
            "miss_seconds": round(float(components["missed detection"]), 3),
            "false_alarm_seconds": round(float(components["false alarm"]), 3),
            "confusion_seconds": round(float(components["confusion"]), 3),
            "scored_speech_seconds": round(total, 3),
        }
    return results


def score_conversation(
    entry: Mapping[str, object],
    *,
    derived_root: Path,
    diarizer: Diarizer,
) -> ConversationScore:
    """Verify, diarize, and score one conversation."""

    audio_path, reference_path = verify_inputs(entry, derived_root=derived_root)
    reference, speakers, duration, speech, turn_count = _reference_annotation(reference_path, entry)
    conversation_id = str(entry["conversation_id"])

    outputs = entry["outputs"]
    assert isinstance(outputs, dict)
    asset = AudioAsset(
        source_name=f"{conversation_id}.wav",
        source_path=str(audio_path),
        normalized_paths=(str(audio_path),),
        channel_mode="mixed",
        duration_seconds=duration,
        channels=1,
        sha256=str(outputs["audio_sha256"]),
    )

    started = time.monotonic()
    segments = diarizer.diarize(asset)
    elapsed = time.monotonic() - started

    hypothesis = _hypothesis_annotation(segments, conversation_id)
    return ConversationScore(
        conversation_id=conversation_id,
        duration_seconds=round(duration, 3),
        reference_speakers=speakers,
        reference_turn_count=turn_count,
        reference_speech_seconds=round(speech, 3),
        hypothesis_speaker_count=len({segment.speaker_id for segment in segments}),
        hypothesis_segment_count=len(segments),
        hypothesis_speech_seconds=round(
            sum(segment.end - segment.start for segment in segments), 3
        ),
        elapsed_seconds=round(elapsed, 3),
        metrics=_score(reference, hypothesis, duration),
    )


def _aggregate(scores: Sequence[ConversationScore]) -> dict[str, dict[str, float]]:
    """Pool error seconds across conversations before dividing.

    Averaging per-conversation rates would weight a ten-minute call the same as a
    fifteen-minute one; pooling reports the error a listener would actually hear across
    the whole benchmark set.
    """

    aggregate: dict[str, dict[str, float]] = {}
    for variant in ("collar_250ms", "strict"):
        total = sum(score.metrics[variant]["scored_speech_seconds"] for score in scores)
        if total <= 0:
            raise KcscBenchmarkError("benchmark set contains no scorable speech")
        pooled = {
            component: sum(score.metrics[variant][f"{component}_seconds"] for score in scores)
            for component in ("miss", "false_alarm", "confusion")
        }
        aggregate[variant] = {
            "der": round(sum(pooled.values()) / total, 6),
            "miss": round(pooled["miss"] / total, 6),
            "false_alarm": round(pooled["false_alarm"] / total, 6),
            "confusion": round(pooled["confusion"] / total, 6),
            "jer_macro": round(
                sum(score.metrics[variant]["jer"] for score in scores) / len(scores), 6
            ),
            "scored_speech_seconds": round(total, 3),
        }
    return aggregate


def run_benchmark(
    *,
    derived_root: Path,
    conversation_ids: Sequence[str],
    diarizer: Diarizer,
    model_name: str,
    model_tree_sha256: str | None,
    egress_attempts: int = 0,
    remote: bool = False,
) -> BenchmarkReport:
    """Score the selected conversations and assemble the aggregate report."""

    manifest = load_manifest(derived_root)
    entries = select_conversations(manifest, conversation_ids)
    scores = tuple(
        score_conversation(entry, derived_root=derived_root, diarizer=diarizer) for entry in entries
    )
    source = manifest.get("source")
    revision = source.get("revision") if isinstance(source, dict) else None
    return BenchmarkReport(
        model_name=model_name,
        model_tree_sha256=model_tree_sha256,
        source_revision=str(revision),
        manifest_sha256=file_sha256(derived_root / "manifest.json"),
        scores=scores,
        aggregate=_aggregate(scores),
        egress_attempts=egress_attempts,
        remote=remote,
    )


def write_report(
    report: BenchmarkReport,
    path: Path,
    *,
    disclosure: Mapping[str, object] | None = None,
) -> str:
    """Write the transcript-free aggregate report and return its digest.

    ``disclosure`` adds a ``processing`` block naming where the audio was handled. It is
    the only shape difference between a local and a remote report, so the two remain
    directly comparable field by field.
    """

    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "kcsc-diarization",
        "transcript_free": True,
        "task": "speaker-diarization",
        "model": {
            "name": report.model_name,
            "checkpoint_tree_sha256": report.model_tree_sha256,
            "remote": report.remote,
        },
        "dataset": {
            "name": "kcsc-derived-evaluation-set",
            "source_revision": report.source_revision,
            "derivation_manifest_sha256": report.manifest_sha256,
            "conversation_count": report.conversation_count,
        },
        "scoring": {
            "implementation": "pyannote.metrics",
            "collar_seconds": COLLAR_SECONDS,
            "skip_overlap": False,
            "aggregation": "pooled error seconds over pooled reference speech",
        },
        "offline": {"network_egress_attempts": report.egress_attempts},
        "processing": dict(disclosure) if disclosure is not None else {"location": "local"},
        "aggregate": {variant: dict(values) for variant, values in report.aggregate.items()},
        "conversations": [
            {
                "conversation_id": score.conversation_id,
                "duration_seconds": score.duration_seconds,
                "reference_speakers": list(score.reference_speakers),
                "reference_turn_count": score.reference_turn_count,
                "reference_speech_seconds": score.reference_speech_seconds,
                "hypothesis_speaker_count": score.hypothesis_speaker_count,
                "hypothesis_segment_count": score.hypothesis_segment_count,
                "hypothesis_speech_seconds": score.hypothesis_speech_seconds,
                "elapsed_seconds": score.elapsed_seconds,
                "metrics": {variant: dict(values) for variant, values in score.metrics.items()},
            }
            for score in report.scores
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(f"{raw}\n", encoding="utf-8")
    return file_sha256(path)


__all__ = [
    "COLLAR_SECONDS",
    "REFERENCE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SPEAKERS_PER_CONVERSATION",
    "STRICT_COLLAR_SECONDS",
    "BenchmarkReport",
    "ConversationScore",
    "Diarizer",
    "KcscBenchmarkError",
    "file_sha256",
    "load_manifest",
    "run_benchmark",
    "score_conversation",
    "select_conversations",
    "verify_inputs",
    "write_report",
]
