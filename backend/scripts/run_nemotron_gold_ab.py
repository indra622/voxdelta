"""Local-only A/B of Nemotron 3 Diarization against the reviewer-confirmed user Gold.

Nothing leaves the machine. Nemotron runs through the ``nemotron-3-local`` provider, with
its ``nemo-speech`` subprocess additionally confined by a macOS ``sandbox-exec`` profile
that denies all networking. The optional Community-1 baseline runs from a local
checkpoint inside the process-wide egress guard. Precision-2 is *not* called: its numbers
are read from the existing transcript-free artifacts, and only when the Gold and audio
digests recorded there still match the inputs scored here.

Scoring reuses the KCSC/user-Gold protocol (pyannote.metrics DER/JER, strict and 250 ms
collar, whole-recording UEM, overlap scored). The report is transcript-free and path-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never

from voxdelta.annotation.store import AnnotationError, verify_gold
from voxdelta.credentials import Credentials
from voxdelta.domain.models import AudioAsset, SpeakerSegment
from voxdelta.evaluation.kcsc_diarization_benchmark import _hypothesis_annotation, _score
from voxdelta.providers.base import DiarizationTimelines, ProviderError
from voxdelta.providers.checkpoints import checkpoint_tree_digest
from voxdelta.providers.nemotron_diarization import (
    CommandResult,
    CommandRunner,
    NemotronDiarizationProvider,
)
from voxdelta.providers.offline_guard import block_network_egress

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"
USER_AUDIO = Path("derived/user-provided/2026-09-11-korean-evaluation-audio")
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
DENY_NETWORK_PROFILE = "(version 1)(allow default)(deny network*)"
SCHEMA_VERSION = "1"
REPORT_KIND = "user-gold-nemotron-3-local-diarization-ab"


@dataclass(frozen=True, slots=True)
class Case:
    conversation_id: str
    manifest: Path
    role: str
    precision_artifact: str | None


CASES = (
    Case(
        "USER_EVAL_20260911_01",
        USER_AUDIO / "manifest.json",
        "headline_two_speaker",
        "user-gold-evaluation-01.json",
    ),
    Case(
        "USER_EVAL_20260911_02A",
        USER_AUDIO / "gold-splits-v1/manifest.json",
        "headline_two_speaker",
        "user-gold-evaluation-02A.json",
    ),
    Case(
        "USER_EVAL_20260911_02",
        USER_AUDIO / "manifest.json",
        "auxiliary_full_recording_three_gold_speakers_overlaps_02A_02B",
        "user-gold-evaluation-02.json",
    ),
    Case(
        "USER_EVAL_20260911_02B",
        USER_AUDIO / "gold-splits-v1/manifest.json",
        "auxiliary_single_speaker_split",
        None,
    ),
)


class EvaluationError(RuntimeError):
    """A deterministic precondition or integrity failure."""


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


class SandboxedRunner:
    """Run the local runtime under a macOS profile that denies every network operation."""

    def __init__(self, inner: CommandRunner | None = None) -> None:
        if not SANDBOX_EXEC.is_file():
            raise EvaluationError("sandbox-exec is unavailable; refusing an unconfined run")
        self._inner = inner or CommandRunner()
        self.invocations = 0

    def __call__(
        self, argv: Sequence[str], *, env: Mapping[str, str], timeout_seconds: float
    ) -> CommandResult:
        self.invocations += 1
        return self._inner(
            [str(SANDBOX_EXEC), "-p", DENY_NETWORK_PROFILE, *argv],
            env=env,
            timeout_seconds=timeout_seconds,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"unreadable artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"invalid artifact: {path.name}")
    return value


def manifest_entry(manifest_path: Path, conversation_id: str) -> tuple[Path, str, float]:
    items = _read_json(manifest_path).get("items")
    if not isinstance(items, list):
        raise EvaluationError("manifest has no items")
    for item in items:
        if isinstance(item, dict) and item.get("id") == conversation_id:
            filename, digest, duration = (
                item.get("file"),
                item.get("sha256"),
                item.get("duration_seconds"),
            )
            if (
                isinstance(filename, str)
                and "/" not in filename
                and isinstance(digest, str)
                and isinstance(duration, int | float)
            ):
                return manifest_path.parent / filename, digest, float(duration)
            raise EvaluationError(f"{conversation_id}: malformed manifest entry")
    raise EvaluationError(f"{conversation_id}: absent from manifest")


def gold_turns(gold: Mapping[str, Any], duration: float) -> list[tuple[float, float, str]]:
    """Timing and speaker only; the transcript is never read past validation."""

    content = gold.get("content")
    turns = content.get("turns") if isinstance(content, dict) else None
    if not isinstance(turns, list) or not turns:
        raise EvaluationError("gold has no turns")
    parsed: list[tuple[float, float, str]] = []
    for raw in turns:
        if not isinstance(raw, dict):
            raise EvaluationError("gold has malformed turn")
        start, end, speaker = raw.get("start"), raw.get("end"), raw.get("speaker")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int | float)
            or not isinstance(end, int | float)
            or not isinstance(speaker, str)
            or not speaker
        ):
            raise EvaluationError("gold has malformed timing or speaker")
        if start < 0 or end <= start or end > duration + 1e-6:
            raise EvaluationError("gold timing is outside the audio")
        parsed.append((float(start), float(end), speaker))
    return sorted(parsed)


def _gold_annotation(turns: Sequence[tuple[float, float, str]], uri: str) -> object:
    from pyannote.core import Annotation, Segment  # type: ignore[import-untyped]

    annotation = Annotation(uri=uri)
    for start, end, speaker in turns:
        annotation[Segment(start, end)] = speaker
    return annotation


def _require_normalized_wav(path: Path, duration: float) -> None:
    with wave.open(str(path), "rb") as source:
        if (
            source.getframerate() != 16_000
            or source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getcomptype() != "NONE"
        ):
            raise EvaluationError("evaluation audio is not 16 kHz mono PCM16")
        measured = source.getnframes() / source.getframerate()
    if abs(measured - duration) > 0.01:
        raise EvaluationError("evaluation audio duration disagrees with its manifest")


def score_system(
    segments: Sequence[SpeakerSegment],
    reference: object,
    uri: str,
    duration: float,
    gold_speakers: int,
) -> dict[str, Any]:
    hypothesis_speakers = len({segment.speaker_id for segment in segments})
    return {
        "metrics": _score(reference, _hypothesis_annotation(segments, uri), duration),
        "hypothesis_speakers": hypothesis_speakers,
        "hypothesis_segments": len(segments),
        "speaker_count_error": hypothesis_speakers - gold_speakers,
    }


def timing(wall_seconds: Sequence[float], duration: float) -> dict[str, Any]:
    median = statistics.median(wall_seconds)
    return {
        "runs": len(wall_seconds),
        "wall_seconds": [round(value, 3) for value in wall_seconds],
        "median_wall_seconds": round(median, 3),
        "median_rtfx": round(duration / median, 2) if median > 0 else None,
    }


def precision_baseline(
    artifact: Path | None,
    conversation_id: str,
    gold_sha256: str,
    audio_sha256: str,
    gold_speakers: int,
) -> dict[str, Any]:
    """Copy an existing Precision-2 result only if it scored exactly these inputs."""

    if artifact is None:
        return {"status": "not_available_no_prior_precision_run_for_this_case"}
    if not artifact.is_file():
        return {"status": "not_available_artifact_missing", "artifact": artifact.name}
    evaluations = _read_json(artifact).get("evaluations")
    matches = [
        item
        for item in (evaluations if isinstance(evaluations, list) else [])
        if isinstance(item, dict) and item.get("conversation_id") == conversation_id
    ]
    if len(matches) != 1:
        return {"status": "not_available_case_absent_from_artifact", "artifact": artifact.name}
    record = matches[0]
    if record.get("gold_sha256") != gold_sha256 or record.get("audio_sha256") != audio_sha256:
        return {"status": "not_comparable_gold_or_audio_digest_changed", "artifact": artifact.name}
    scored = record.get("precision_qwen")
    diarization = scored.get("diarization") if isinstance(scored, dict) else None
    raw_pipeline = record.get("pipeline")
    pipeline: dict[str, Any] = raw_pipeline if isinstance(raw_pipeline, dict) else {}
    if not isinstance(diarization, dict) or "strict" not in diarization:
        stages = pipeline.get("public_state", {}).get("stages", {}) if pipeline else {}
        diarize = stages.get("diarize", {}) if isinstance(stages, dict) else {}
        return {
            "status": "no_metrics_in_prior_run",
            "artifact": artifact.name,
            "prior_evaluation_status": record.get("evaluation_status"),
            "prior_diarize_error_code": diarize.get("error_code")
            if isinstance(diarize, dict)
            else None,
        }
    speakers = pipeline.get("hypothesis_speakers")
    return {
        "status": "copied_from_existing_artifact_no_new_remote_call",
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "timeline": "exclusive (alignment_segments)",
        "metrics": diarization,
        "hypothesis_speakers": speakers,
        "hypothesis_segments": pipeline.get("hypothesis_segments"),
        "speaker_count_error": speakers - gold_speakers if isinstance(speakers, int) else None,
        "timing": "not_comparable_prior_run_timed_whole_pipeline_including_upload_and_asr",
    }


def _community_turns(provider: Any, path: Path, duration: float) -> DiarizationTimelines:
    """Run Community-1 with the product's speaker bounds but without its two-speaker gate."""

    import numpy as np
    import torch

    from voxdelta.providers import pyannote_diarization as community

    # pyannote's documented in-memory input: this venv's torchcodec cannot load the FFmpeg
    # libraries pyannote 4 needs to decode a file path, and the samples are identical.
    with wave.open(str(path), "rb") as source:
        frames = source.readframes(source.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples.copy()).unsqueeze(0)
    output = provider._load_pipeline()(
        {"waveform": waveform, "sample_rate": 16_000},
        min_speakers=1,
        max_speakers=4,
    )
    timelines: dict[str, list[SpeakerSegment]] = {}
    for name, attribute in (
        ("evidence", "speaker_diarization"),
        ("exclusive", "exclusive_speaker_diarization"),
    ):
        turns = community._turns(community._annotation(output, attribute), duration)
        labels = {
            label: f"SPEAKER_{index:02d}"
            for index, label in enumerate(dict.fromkeys(turn.label for turn in turns))
        }
        timelines[name] = [
            SpeakerSegment(
                start=turn.start, end=turn.end, speaker_id=labels[turn.label], confidence=1.0
            )
            for turn in turns
        ]
    return DiarizationTimelines(evidence=timelines["evidence"], exclusive=timelines["exclusive"])


def _same_timelines(left: DiarizationTimelines, right: DiarizationTimelines) -> bool:
    return left.evidence == right.evidence and left.exclusive == right.exclusive


def evaluate(
    *,
    data_root: Path,
    provider: NemotronDiarizationProvider,
    repeats: int,
    community_checkpoint: Path | None,
    cases: Sequence[Case] = CASES,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
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
        }

    results: list[dict[str, Any]] = []
    for case in cases:
        manifest = data_root / case.manifest
        audio, audio_sha256, duration = manifest_entry(manifest, case.conversation_id)
        if sha256_file(audio) != audio_sha256:
            raise EvaluationError(f"{case.conversation_id}: audio digest mismatch")
        _require_normalized_wav(audio, duration)
        gold = verify_gold(data_root / "annotations" / case.conversation_id / "gold.json")
        gold_sha256 = str(gold.get("content_sha256"))
        turns = gold_turns(gold, duration)
        gold_speakers = len({speaker for _start, _end, speaker in turns})
        reference = _gold_annotation(turns, case.conversation_id)
        asset = AudioAsset(
            source_name=case.conversation_id,
            source_path=str(audio),
            normalized_paths=(str(audio),),
            channel_mode="mixed",
            duration_seconds=duration,
            channels=1,
            sha256=audio_sha256,
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
        nemotron_speakers = len({segment.speaker_id for segment in first.evidence})
        systems: dict[str, Any] = {
            "nemotron_3_local": {
                "exclusive": score_system(
                    first.exclusive, reference, case.conversation_id, duration, gold_speakers
                ),
                "overlap_aware_evidence": score_system(
                    first.evidence, reference, case.conversation_id, duration, gold_speakers
                ),
                "overlap_segments": sum(segment.overlap for segment in first.evidence),
                "timing": timing(walls, duration),
                "timing_includes_process_start_and_model_load": True,
                "repeat_outputs_identical": deterministic,
                # diarize_timelines gates on the evidence timeline's speaker count.
                "provider_two_speaker_gate_on_evidence": "passes"
                if nemotron_speakers == 2
                else "would_raise_unsupported_speaker_count",
            },
            "pyannoteai_precision_2": precision_baseline(
                data_root / "benchmarks" / case.precision_artifact
                if case.precision_artifact
                else None,
                case.conversation_id,
                gold_sha256,
                audio_sha256,
                gold_speakers,
            ),
        }
        if community is not None:
            community_walls: list[float] = []
            community_first: DiarizationTimelines | None = None
            for _ in range(repeats):
                started = time.monotonic()
                community_timelines = _community_turns(community, audio, duration)
                community_walls.append(time.monotonic() - started)
                community_first = community_first or community_timelines
            assert community_first is not None
            systems["pyannote_community_1_local"] = {
                "exclusive": score_system(
                    community_first.exclusive,
                    reference,
                    case.conversation_id,
                    duration,
                    gold_speakers,
                ),
                "overlap_aware_evidence": score_system(
                    community_first.evidence,
                    reference,
                    case.conversation_id,
                    duration,
                    gold_speakers,
                ),
                "timing": timing(community_walls, duration),
            }
        results.append(
            {
                "conversation_id": case.conversation_id,
                "role": case.role,
                "audio_sha256": audio_sha256,
                "gold_sha256": gold_sha256,
                "duration_seconds": round(duration, 3),
                "gold_speakers": gold_speakers,
                "gold_turns": len(turns),
                "systems": systems,
            }
        )
    return results, community_record


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "metal", "cpu"), default="metal")
    parser.add_argument("--runtime-source-commit", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--community-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.output.exists():
            raise EvaluationError("refusing to overwrite an existing report")
        if not 1 <= arguments.repeats <= 10:
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
            evaluations, community = evaluate(
                data_root=arguments.data_root,
                provider=provider,
                repeats=arguments.repeats,
                community_checkpoint=arguments.community_checkpoint,
            )
        if egress.attempted:
            raise EvaluationError("network egress was attempted during a local-only run")
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "transcript_free": True,
            "pilot_not_switch_decision": True,
            "scoring": {
                "metric": "pyannote.metrics DiarizationErrorRate / JaccardErrorRate",
                "collars_seconds": {"strict": 0.0, "collar_250ms": 0.25},
                "skip_overlap": False,
                "uem": "whole recording",
                "headline_timeline": "exclusive (what transcript alignment consumes)",
            },
            "network_isolation": {
                "remote_calls": 0,
                "nemotron_subprocess": f"sandbox-exec profile: {DENY_NETWORK_PROFILE}",
                "nemotron_subprocess_invocations": runner.invocations,
                "python_egress_guard_attempts": egress.attempts,
                "precision_2": "not called; prior artifacts copied when digests match",
            },
            "nemotron": {
                **provider.configuration,
                "provenance_revision": provider.provenance.revision,
                "remote": provider.provenance.remote,
                "runtime_source_commit": commit,
                "runtime_build": "NeMo-Speech.cpp source build, CMake preset metal-diar",
                "executable_sha256": sha256_file(arguments.executable.resolve()),
            },
            "community_1": community,
            "evaluations": evaluations,
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for item in evaluations:
            nemotron = item["systems"]["nemotron_3_local"]
            strict = nemotron["exclusive"]["metrics"]["strict"]
            collar = nemotron["exclusive"]["metrics"]["collar_250ms"]
            print(
                f"{item['conversation_id']} gold_spk={item['gold_speakers']} "
                f"nemotron_spk={nemotron['exclusive']['hypothesis_speakers']} "
                f"strict DER/JER={strict['der']:.4f}/{strict['jer']:.4f} "
                f"collar DER/JER={collar['der']:.4f}/{collar['jer']:.4f} "
                f"RTFx={nemotron['timing']['median_rtfx']}"
            )
        print(f"report: {arguments.output.name}")
        print(f"sha256: {sha256_file(arguments.output)}")
        return 0
    except ProviderError as error:
        print(f"nemotron A/B failed: provider error {error.code}", file=sys.stderr)
        return 2
    except (AnnotationError, EvaluationError, KeyError, ValueError) as error:
        print(f"nemotron A/B failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        # An OS error message can carry a local path; the class name is enough here.
        print(f"nemotron A/B failed: {type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
