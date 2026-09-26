"""Local-only A/B of Nemotron 3 Diarization on the derived KCSC two-speaker set.

Follow-up to the v01 user-Gold pilot. Nemotron runs through the ``nemotron-3-local``
provider with the same model-card 30.4 s offline geometry, its ``nemo-speech`` subprocess
confined by the ``sandbox-exec`` deny-network profile of the user-Gold script. The optional
Community-1 baseline is re-run from its local checkpoint inside the egress guard.
Precision-2 is *not* called: the existing KCSC benchmark artifacts are cited only when
their derivation-manifest digest (which pins every audio and reference digest) and source
revision equal the inputs scored here.

Sessions of the same speaker pair are correlated, so aggregates are computed over speaker
pairs (pooled within a pair first), never over sessions. The report is transcript-free and
path-free.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import wave
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_nemotron_gold_ab import (  # noqa: E402
    DENY_NETWORK_PROFILE,
    SandboxedRunner,
    _community_turns,
    _same_timelines,
    sha256_file,
    timing,
)

from voxdelta.credentials import Credentials  # noqa: E402
from voxdelta.domain.models import AudioAsset, SpeakerSegment  # noqa: E402
from voxdelta.evaluation.kcsc_diarization_benchmark import (  # noqa: E402
    KcscBenchmarkError,
    _hypothesis_annotation,
    _reference_annotation,
    _score,
    load_manifest,
    verify_inputs,
)
from voxdelta.providers.base import DiarizationTimelines, ProviderError  # noqa: E402
from voxdelta.providers.checkpoints import checkpoint_tree_digest  # noqa: E402
from voxdelta.providers.nemotron_diarization import NemotronDiarizationProvider  # noqa: E402
from voxdelta.providers.offline_guard import block_network_egress  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"
KCSC_DERIVED = Path("derived/kcsc")
HISTORICAL_ARTIFACTS = {
    "pyannote_community_1_historical": "kcsc-diarization-community-1.json",
    "pyannoteai_precision_2_historical": "kcsc-diarization-precision-2.json",
}
SCHEMA_VERSION = "1"
REPORT_KIND = "kcsc-nemotron-3-local-diarization-ab"
VARIANTS = ("strict", "collar_250ms")
TIMELINES = ("exclusive", "overlap_aware_evidence")
#: Historical artifacts round to 6 decimals; a reproduction must agree to that precision.
REPRODUCTION_TOLERANCE = 1e-6


class EvaluationError(RuntimeError):
    """A deterministic precondition or integrity failure."""


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def speaker_pair(entry: Mapping[str, Any]) -> str:
    speakers = entry.get("speakers")
    if not isinstance(speakers, list) or not all(isinstance(item, str) for item in speakers):
        raise EvaluationError(f"{entry.get('conversation_id')}: malformed speaker list")
    return "+".join(sorted(speakers))


def audio_check(path: Path, duration: float) -> str | None:
    """Return why the audio is not a valid 16 kHz mono PCM16 input, or ``None``."""

    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getframerate() != 16_000
                or source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getcomptype() != "NONE"
            ):
                return "audio_not_16khz_mono_pcm16"
            measured = source.getnframes() / source.getframerate()
    except (OSError, wave.Error):
        return "audio_unreadable"
    if abs(measured - duration) > 0.01:
        return "audio_duration_disagrees_with_manifest"
    return None


def score_timeline(
    segments: Sequence[SpeakerSegment], reference: object, uri: str, duration: float
) -> dict[str, Any]:
    speakers = len({segment.speaker_id for segment in segments})
    return {
        "metrics": _score(reference, _hypothesis_annotation(segments, uri), duration),
        "hypothesis_speakers": speakers,
        "hypothesis_segments": len(segments),
        "speaker_count_error": speakers - 2,
    }


def score_timelines(
    timelines: DiarizationTimelines, reference: object, uri: str, duration: float
) -> dict[str, Any]:
    return {
        "exclusive": score_timeline(timelines.exclusive, reference, uri, duration),
        "overlap_aware_evidence": score_timeline(timelines.evidence, reference, uri, duration),
    }


def load_historical(path: Path, *, manifest_sha256: str, source_revision: str) -> dict[str, Any]:
    """Index an existing KCSC artifact by conversation, only if it scored these inputs."""

    if not path.is_file():
        return {"status": "not_available_artifact_missing", "artifact": path.name}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "not_available_artifact_unreadable", "artifact": path.name}
    dataset = payload.get("dataset") if isinstance(payload, dict) else None
    if (
        not isinstance(dataset, dict)
        or dataset.get("derivation_manifest_sha256") != manifest_sha256
        or dataset.get("source_revision") != source_revision
    ):
        return {"status": "not_comparable_manifest_or_revision_changed", "artifact": path.name}
    conversations = payload.get("conversations")
    by_id = {
        item["conversation_id"]: item
        for item in (conversations if isinstance(conversations, list) else [])
        if isinstance(item, dict) and isinstance(item.get("conversation_id"), str)
    }
    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    return {
        "status": "verified_historical_artifact_no_new_run",
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "model": model.get("name"),
        "checkpoint_tree_sha256": model.get("checkpoint_tree_sha256"),
        "remote": model.get("remote"),
        "timeline": "provider diarize() output at artifact time; not recorded in the artifact",
        "conversations": by_id,
    }


def historical_case(
    index: Mapping[str, Any], conversation_id: str, duration: float, turn_count: int
) -> dict[str, Any]:
    if index.get("status") != "verified_historical_artifact_no_new_run":
        return {"status": index.get("status")}
    record = index["conversations"].get(conversation_id)
    if record is None:
        return {"status": "not_available_case_absent_from_artifact"}
    if (
        record.get("reference_turn_count") != turn_count
        or not isinstance(record.get("duration_seconds"), int | float)
        or abs(float(record["duration_seconds"]) - duration) > 0.01
    ):
        return {"status": "not_comparable_reference_shape_changed"}
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or not all(variant in metrics for variant in VARIANTS):
        return {"status": "no_metrics_in_artifact"}
    speakers = record.get("hypothesis_speaker_count")
    return {
        "status": "verified_historical_artifact_no_new_run",
        "metrics": {variant: metrics[variant] for variant in VARIANTS},
        "hypothesis_speakers": speakers,
        "hypothesis_segments": record.get("hypothesis_segment_count"),
        "speaker_count_error": speakers - 2 if isinstance(speakers, int) else None,
        "historical_elapsed_seconds": record.get("elapsed_seconds"),
    }


def reproduced_timeline(historical: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    """Name the current Community-1 timeline whose metrics equal the historical ones."""

    if "metrics" not in historical:
        return "not_checked"
    matches = [
        timeline
        for timeline in TIMELINES
        if all(
            abs(
                float(historical["metrics"][variant][key])
                - float(current[timeline]["metrics"][variant][key])
            )
            <= REPRODUCTION_TOLERANCE
            for variant in VARIANTS
            for key in ("der", "jer")
        )
    ]
    return matches[0] if len(matches) == 1 else "not_reproduced"


def sign_test_p(wins: int, losses: int) -> float | None:
    """Exact two-sided binomial sign test, ties dropped."""

    trials = wins + losses
    if trials == 0:
        return None
    tail: float = sum(math.comb(trials, k) for k in range(min(wins, losses) + 1)) / 2**trials
    return round(min(1.0, 2.0 * tail), 6)


def _pooled(cases: Sequence[Mapping[str, Any]], variant: str) -> dict[str, float]:
    total = sum(float(case["metrics"][variant]["scored_speech_seconds"]) for case in cases)
    errors = sum(
        float(case["metrics"][variant][f"{component}_seconds"])
        for case in cases
        for component in ("miss", "false_alarm", "confusion")
    )
    return {
        "der": errors / total,
        "jer": statistics.fmean(float(case["metrics"][variant]["jer"]) for case in cases),
    }


def _summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def pair_aggregate(
    evaluations: Sequence[Mapping[str, Any]],
    systems: Mapping[str, str | None],
    reference_system: str,
) -> dict[str, Any] | None:
    """Aggregate over speaker pairs that every named system scored.

    ``systems`` maps a system key to a timeline name (``None`` for a historical artifact,
    whose record is already a single timeline). Within a pair, error seconds are pooled
    across its sessions and JER is averaged; across pairs, only unweighted summaries and
    paired differences against ``reference_system`` are reported.
    """

    def record(item: Mapping[str, Any], system: str) -> Mapping[str, Any] | None:
        entry = item["systems"].get(system)
        if not isinstance(entry, dict):
            return None
        timeline = systems[system]
        scored = entry.get(timeline) if timeline else entry
        return scored if isinstance(scored, dict) and "metrics" in scored else None

    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for item in evaluations:
        if all(record(item, system) is not None for system in systems):
            pairs.setdefault(item["speaker_pair"], []).append(item)
    if not pairs:
        return None

    per_pair: dict[str, dict[str, Any]] = {}
    for pair, items in sorted(pairs.items()):
        per_pair[pair] = {
            "sessions": [item["conversation_id"] for item in items],
            **{
                system: {
                    variant: {
                        key: round(value, 6)
                        for key, value in _pooled(
                            [record(item, system) for item in items],  # type: ignore[misc]
                            variant,
                        ).items()
                    }
                    for variant in VARIANTS
                }
                for system in systems
            },
        }

    across: dict[str, Any] = {}
    for variant in VARIANTS:
        across[variant] = {}
        for system in systems:
            across[variant][system] = {
                metric: _summary([pair[system][variant][metric] for pair in per_pair.values()])
                for metric in ("der", "jer")
            }
        for system in systems:
            if system == reference_system:
                continue
            for metric in ("der", "jer"):
                deltas = [
                    pair[reference_system][variant][metric] - pair[system][variant][metric]
                    for pair in per_pair.values()
                ]
                wins = sum(delta < 0 for delta in deltas)
                losses = sum(delta > 0 for delta in deltas)
                across[variant][f"{reference_system}_minus_{system}_{metric}"] = {
                    **_summary(deltas),
                    "reference_better_pairs": wins,
                    "reference_worse_pairs": losses,
                    "sign_test_two_sided_p": sign_test_p(wins, losses),
                }
    return {
        "unit": "speaker pair (sessions of one pair pooled first)",
        "systems": dict(systems),
        "independent_units": len(per_pair),
        "sessions": sum(len(pair["sessions"]) for pair in per_pair.values()),
        "per_pair": per_pair,
        "across_pairs": across,
    }


def evaluate(
    *,
    data_root: Path,
    provider: NemotronDiarizationProvider,
    repeats: int,
    community_checkpoint: Path | None,
    community_repeats: int,
) -> dict[str, Any]:
    derived = data_root / KCSC_DERIVED
    manifest = load_manifest(derived)
    manifest_sha256 = sha256_file(derived / "manifest.json")
    source = manifest.get("source")
    source_revision = str(source.get("revision")) if isinstance(source, dict) else ""
    historical = {
        key: load_historical(
            data_root / "benchmarks" / name,
            manifest_sha256=manifest_sha256,
            source_revision=source_revision,
        )
        for key, name in HISTORICAL_ARTIFACTS.items()
    }

    community: Any = None
    community_record: dict[str, Any] | None = None
    if community_checkpoint is not None:
        from voxdelta.providers.pyannote_diarization import PyannoteDiarizationProvider

        community = PyannoteDiarizationProvider(Credentials(), model_path=community_checkpoint)
        started = time.monotonic()
        community._load_pipeline()
        community_record = {
            "model": community.provenance.model,
            "tree_sha256": checkpoint_tree_digest(community_checkpoint.resolve()),
            "device": "cpu (provider default)",
            "input": "in-memory 16 kHz mono waveform (venv torchcodec lacks FFmpeg libraries)",
            "load_seconds": round(time.monotonic() - started, 3),
            "speaker_bounds": {"min_speakers": 1, "max_speakers": 4},
            "timing_excludes_model_load": True,
            "repeats": community_repeats,
        }

    entries = manifest["conversations"]
    assert isinstance(entries, list)
    evaluations: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for entry in sorted(entries, key=lambda item: str(item.get("conversation_id"))):
        conversation_id = str(entry.get("conversation_id"))
        speakers = entry.get("speakers")
        if not isinstance(speakers, list) or len(speakers) != 2:
            excluded.append({"conversation_id": conversation_id, "reason": "not_two_speaker"})
            continue
        try:
            audio, reference_path = verify_inputs(entry, derived_root=derived)
            reference, labels, duration, speech, turn_count = _reference_annotation(
                reference_path, entry
            )
        except KcscBenchmarkError:
            excluded.append(
                {"conversation_id": conversation_id, "reason": "input_verification_failed"}
            )
            continue
        problem = audio_check(audio, duration)
        if problem is not None:
            excluded.append({"conversation_id": conversation_id, "reason": problem})
            continue
        outputs = entry["outputs"]
        asset = AudioAsset(
            source_name=conversation_id,
            source_path=str(audio),
            normalized_paths=(str(audio),),
            channel_mode="mixed",
            duration_seconds=duration,
            channels=1,
            sha256=str(outputs["audio_sha256"]),
        )

        walls: list[float] = []
        first: DiarizationTimelines | None = None
        deterministic = True
        for _ in range(repeats):
            started = time.monotonic()
            timelines = provider.diarize_unconstrained(asset)
            walls.append(time.monotonic() - started)
            if first is None:
                first = timelines
            elif not _same_timelines(first, timelines):
                deterministic = False
        assert first is not None
        systems: dict[str, Any] = {
            "nemotron_3_local": {
                "run": "current",
                **score_timelines(first, reference, conversation_id, duration),
                "overlap_segments": sum(segment.overlap for segment in first.evidence),
                "timing": timing(walls, duration),
                "timing_includes_process_start_and_model_load": True,
                "repeat_outputs_identical": deterministic,
            }
        }
        if community is not None:
            community_walls: list[float] = []
            community_first: DiarizationTimelines | None = None
            community_deterministic = True
            for _ in range(community_repeats):
                started = time.monotonic()
                current = _community_turns(community, audio, duration)
                community_walls.append(time.monotonic() - started)
                if community_first is None:
                    community_first = current
                elif not _same_timelines(community_first, current):
                    community_deterministic = False
            assert community_first is not None
            systems["pyannote_community_1_local"] = {
                "run": "current",
                **score_timelines(community_first, reference, conversation_id, duration),
                "timing": timing(community_walls, duration),
                "repeat_outputs_identical": community_deterministic
                if community_repeats > 1
                else None,
            }
        for key, index in historical.items():
            systems[key] = historical_case(index, conversation_id, duration, turn_count)
        if "pyannote_community_1_local" in systems:
            systems["pyannote_community_1_historical"]["current_run_reproduces_timeline"] = (
                reproduced_timeline(
                    systems["pyannote_community_1_historical"],
                    systems["pyannote_community_1_local"],
                )
            )
        evaluations.append(
            {
                "conversation_id": conversation_id,
                "speaker_pair": "+".join(sorted(labels)),
                "audio_sha256": outputs["audio_sha256"],
                "reference_sha256": outputs["reference_sha256"],
                "duration_seconds": round(duration, 3),
                "reference_speakers": 2,
                "reference_turns": turn_count,
                "reference_speech_seconds": round(speech, 3),
                "systems": systems,
            }
        )

    aggregates: dict[str, Any] = {}
    if community is not None:
        for timeline in TIMELINES:
            aggregates[f"all_pairs_nemotron_vs_community_current_{timeline}"] = pair_aggregate(
                evaluations,
                {"nemotron_3_local": timeline, "pyannote_community_1_local": timeline},
                "nemotron_3_local",
            )
    else:
        for timeline in TIMELINES:
            aggregates[f"all_pairs_nemotron_{timeline}"] = pair_aggregate(
                evaluations, {"nemotron_3_local": timeline}, "nemotron_3_local"
            )
    for timeline in TIMELINES:
        aggregates[f"historical_subset_nemotron_{timeline}_vs_historical"] = pair_aggregate(
            evaluations,
            {
                "nemotron_3_local": timeline,
                "pyannote_community_1_historical": None,
                "pyannoteai_precision_2_historical": None,
            },
            "nemotron_3_local",
        )

    return {
        "dataset": {
            "name": "kcsc-derived-evaluation-set",
            "source_dataset": manifest.get("dataset")
            if not isinstance(source, dict)
            else source.get("dataset"),
            "source_revision": source_revision,
            "derivation_manifest_sha256": manifest_sha256,
            "manifest_conversations": len(entries),
            "evaluated_conversations": len(evaluations),
            "speaker_pairs": len({item["speaker_pair"] for item in evaluations}),
            "excluded": excluded,
        },
        "historical_artifacts": {
            key: {name: value for name, value in index.items() if name != "conversations"}
            for key, index in historical.items()
        },
        "community_1": community_record,
        "evaluations": evaluations,
        "aggregates": aggregates,
    }


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "metal", "cpu"), default="metal")
    parser.add_argument("--runtime-source-commit", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--community-checkpoint", type=Path)
    parser.add_argument("--community-repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.output.exists():
            raise EvaluationError("refusing to overwrite an existing report")
        for count in (arguments.repeats, arguments.community_repeats):
            if not 1 <= count <= 10:
                raise EvaluationError("repeats must be between 1 and 10")
        commit = str(arguments.runtime_source_commit)
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise EvaluationError("runtime source commit must be a full lowercase git SHA")
        runner = SandboxedRunner()
        with block_network_egress() as egress:
            provider = NemotronDiarizationProvider(
                executable_path=arguments.executable,
                model_path=arguments.model,
                device=arguments.device,
                runner=runner,
            )
            result = evaluate(
                data_root=arguments.data_root,
                provider=provider,
                repeats=arguments.repeats,
                community_checkpoint=arguments.community_checkpoint,
                community_repeats=arguments.community_repeats,
            )
        if egress.attempted:
            raise EvaluationError("network egress was attempted during a local-only run")
        if not result["evaluations"]:
            raise EvaluationError("no valid KCSC two-speaker input was evaluated")
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "transcript_free": True,
            "path_free": True,
            "scoring": {
                "metric": "pyannote.metrics DiarizationErrorRate / JaccardErrorRate",
                "collars_seconds": {"strict": 0.0, "collar_250ms": 0.25},
                "skip_overlap": False,
                "uem": "whole recording",
                "timelines": "exclusive and overlap_aware_evidence both scored for current runs",
                "aggregation": (
                    "per speaker pair: pooled error seconds / pooled reference speech and "
                    "mean JER; across pairs: unweighted summaries and paired differences"
                ),
            },
            "network_isolation": {
                "remote_calls": 0,
                "nemotron_subprocess": f"sandbox-exec profile: {DENY_NETWORK_PROFILE}",
                "nemotron_subprocess_invocations": runner.invocations,
                "python_egress_guard_attempts": egress.attempts,
                "precision_2": "not called; KCSC artifact cited only on matching manifest digest",
            },
            "nemotron": {
                **provider.configuration,
                "provenance_revision": provider.provenance.revision,
                "remote": provider.provenance.remote,
                "runtime_source_commit": commit,
                "runtime_build": "NeMo-Speech.cpp source build, CMake preset metal-diar",
                "executable_sha256": sha256_file(arguments.executable.resolve()),
            },
            **result,
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for item in result["evaluations"]:
            nemotron = item["systems"]["nemotron_3_local"]
            line = [item["conversation_id"], item["speaker_pair"]]
            for timeline in TIMELINES:
                strict = nemotron[timeline]["metrics"]["strict"]
                collar = nemotron[timeline]["metrics"]["collar_250ms"]
                line.append(
                    f"{timeline}: strict {strict['der']:.4f}/{strict['jer']:.4f} "
                    f"collar {collar['der']:.4f}/{collar['jer']:.4f} "
                    f"spk_err {nemotron[timeline]['speaker_count_error']}"
                )
            line.append(f"RTFx {nemotron['timing']['median_rtfx']}")
            print(" | ".join(line))
        for item in result["dataset"]["excluded"]:
            print(f"excluded {item['conversation_id']}: {item['reason']}")
        print(f"report: {arguments.output.name}")
        print(f"sha256: {sha256_file(arguments.output)}")
        return 0
    except ProviderError as error:
        print(f"nemotron KCSC A/B failed: provider error {error.code}", file=sys.stderr)
        return 2
    except (EvaluationError, KcscBenchmarkError, KeyError, ValueError) as error:
        print(f"nemotron KCSC A/B failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        # An OS error message can carry a local path; the class name is enough here.
        print(f"nemotron KCSC A/B failed: {type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
