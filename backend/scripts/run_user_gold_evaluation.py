"""Evaluate the real VoxDelta path against reviewer-created private gold annotations.

The script is intentionally narrow: every selected recording is submitted to Precision-2
once, Qwen/XLS-R stay local and offline, and the resulting JSON report contains only
digests, counts, and metrics -- never an utterance transcript.  Existing Gemini silver
is scored as an already-created baseline; this script never calls Gemini.

The production pipeline currently requires exactly two diarized speakers to continue past
role confirmation.  A gold recording with another number of speakers still receives DER,
JER and mixed-stream CER, while its XLS-R emotion score is explicitly reported as skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import wave
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never, cast

from pydantic import SecretStr

from voxdelta.annotation.store import AnnotationError, verify_gold
from voxdelta.api.dependencies import ProviderFactories, build_dependencies
from voxdelta.config import Settings
from voxdelta.credentials import load_credentials
from voxdelta.domain.models import AudioAsset, Role, SpeakerSegment, StageName, Utterance
from voxdelta.evaluation.kcsc_asr_benchmark import (
    KcscAsrError,
    characters,
    configure_offline_cache,
    score_sequences,
)
from voxdelta.evaluation.kcsc_diarization_benchmark import _hypothesis_annotation, _score
from voxdelta.evaluation.kcsc_precision_benchmark import CallLedger, LedgerClient
from voxdelta.evaluation.kcsc_qwen_track_benchmark import verify_qwen_identity
from voxdelta.pipeline.stages import DiarizeArtifact, ReportArtifact, TranscribeArtifact
from voxdelta.providers.calibrated_emotion import CalibratedEmotionProvider
from voxdelta.providers.calibration_artifact import verify_calibration_artifact
from voxdelta.providers.pyannote_precision import PRECISION_MODEL_ID, _default_client_factory
from voxdelta.providers.qwen3_asr import ALIGNER_MODEL_ID, DEFAULT_MODEL_ID, Qwen3AsrProvider
from voxdelta.providers.release_bundle import verify_release_bundle
from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = REPOSITORY_ROOT / "data/annotations"
DERIVED_MANIFEST = (
    REPOSITORY_ROOT
    / "data/derived/user-provided/2026-09-11-korean-evaluation-audio/manifest.json"
)
OUTPUT = REPOSITORY_ROOT / "data/benchmarks/user-gold-pipeline-evaluation.json"
SCRATCH_ROOT = REPOSITORY_ROOT / "data/jobs/user-gold-evaluation-scratch"
CONVERSATIONS = ("USER_EVAL_20260911_01", "USER_EVAL_20260911_02")
MAX_REMOTE_JOBS_PER_CONVERSATION = 1
MODEL_CACHE = REPOSITORY_ROOT / "data/models/hf-cache/hub"


class EvaluationError(RuntimeError):
    """A deterministic evaluation precondition or integrity failure."""


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_models(cache_root: Path = MODEL_CACHE) -> dict[str, str]:
    """Pin Qwen's already-present cache offline before an upload can occur."""

    if not cache_root.is_dir():
        raise EvaluationError("missing local Qwen hub cache")
    try:
        asr = verify_qwen_identity(cache_root, DEFAULT_MODEL_ID)
        aligner = verify_qwen_identity(cache_root, ALIGNER_MODEL_ID)
    except KcscAsrError as error:
        raise EvaluationError("Qwen checkpoints are unusable offline") from error
    configure_offline_cache(cache_root)
    return {
        "hub_cache": str(cache_root.resolve()),
        "hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
        "qwen_asr_tree_sha256": asr.tree_sha256,
        "qwen_aligner_tree_sha256": aligner.tree_sha256,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"unreadable private artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"invalid private artifact: {path.name}")
    return value


def _gold_turns(gold: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = gold.get("content")
    turns = content.get("turns") if isinstance(content, dict) else None
    if not isinstance(turns, list) or not turns:
        raise EvaluationError("gold has no turns")
    parsed: list[dict[str, Any]] = []
    for raw in turns:
        if not isinstance(raw, dict):
            raise EvaluationError("gold has malformed turn")
        start, end = raw.get("start"), raw.get("end")
        if not isinstance(start, int | float) or not isinstance(end, int | float) or end <= start:
            raise EvaluationError("gold has invalid timing")
        if not isinstance(raw.get("speaker"), str) or not isinstance(raw.get("transcript"), str):
            raise EvaluationError("gold has malformed speaker or transcript")
        parsed.append(raw)
    return sorted(parsed, key=lambda turn: (float(turn["start"]), float(turn["end"])))


def _manifest_audio(manifest_path: Path = DERIVED_MANIFEST) -> dict[str, tuple[Path, str, float]]:
    raw = _read_json(manifest_path)
    items = raw.get("items")
    if not isinstance(items, list):
        raise EvaluationError("derived user manifest has no items")
    resolved: dict[str, tuple[Path, str, float]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        identifier, filename, digest, duration = (
            item.get("id"),
            item.get("file"),
            item.get("sha256"),
            item.get("duration_seconds"),
        )
        if (
            isinstance(identifier, str)
            and isinstance(filename, str)
            and isinstance(digest, str)
            and isinstance(duration, int | float)
        ):
            resolved[identifier] = (manifest_path.parent / filename, digest, float(duration))
    return resolved


def _settings(scratch: Path) -> Settings:
    return Settings(
        data_root=scratch,
        database_path=scratch / "voxdelta.sqlite3",
        api_capability_token=SecretStr("g" * 43),
        # The user-reviewed evaluation recordings include a 55-second call.  This
        # runner is not the public upload admission path, so retain that gold input
        # without weakening the product's 60-second default.
        min_audio_seconds=1,
        diarization_provider="pyannoteai-precision",
        asr_provider="qwen3",
        qwen_profile="default",
        asr_device="mps",
        asr_fallback_provider="none",
        emotion_provider="wav2vec",
        emotion_device="cpu",
        xlsr_release_enabled=True,
        xlsr_release_path=REPOSITORY_ROOT / "data/models/xls-r-emotion-7class-v1",
        xlsr_calibration_enabled=True,
        xlsr_calibration_path=(
            REPOSITORY_ROOT / "data/models/xls-r-emotion-7class-v1-calibration-v2"
        ),
    )


def _gold_annotation(turns: Sequence[Mapping[str, Any]], conversation_id: str) -> object:
    from pyannote.core import Annotation, Segment  # type: ignore[import-untyped]

    annotation = Annotation(uri=conversation_id)
    for turn in turns:
        annotation[Segment(float(turn["start"]), float(turn["end"]))] = str(turn["speaker"])
    return annotation


def _gold_stream(turns: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(str(turn["transcript"]) for turn in turns)


def _utterance_stream(utterances: Sequence[Utterance]) -> str:
    ordered = sorted(utterances, key=lambda item: (item.start, item.end, item.id))
    return " ".join(item.transcript for item in ordered)


def _best_overlap(
    turn: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    start, end = float(turn["start"]), float(turn["end"])
    best: tuple[float, Mapping[str, Any]] | None = None
    for candidate in candidates:
        overlap = max(
            0.0,
            min(end, float(candidate["end"])) - max(start, float(candidate["start"])),
        )
        if overlap <= 0:
            continue
        if best is None or overlap > best[0]:
            best = (overlap, candidate)
    return best[1] if best else None


def _emotion_agreement(
    gold_turns: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, int | float | None]:
    matched = 0
    correct = 0
    for turn in gold_turns:
        predicted = _best_overlap(turn, predictions)
        if predicted is None:
            continue
        matched += 1
        if predicted.get("emotion") == turn.get("emotion"):
            correct += 1
    return {
        "gold_turn_count": len(gold_turns),
        "matched_turn_count": matched,
        "correct_turn_count": correct,
        "label_accuracy": round(correct / matched, 6) if matched else None,
    }


@contextmanager
def _gold_turn_clip(audio: Path, turn: Mapping[str, Any], scratch: Path) -> Iterator[Path]:
    """Yield a temporary WAV clip for one private reviewer-defined gold turn."""

    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=scratch, prefix=".gold-emotion-", suffix=".wav"
    )
    clip = Path(temporary_name)
    try:
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        with wave.open(str(audio), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getcomptype() != "NONE"
                or source.getframerate() <= 0
            ):
                raise EvaluationError("single-speaker evaluation audio is not normalized WAV")
            rate = source.getframerate()
            start = round(float(turn["start"]) * rate)
            end = round(float(turn["end"]) * rate)
            if start < 0 or end <= start or end > source.getnframes():
                raise EvaluationError("gold turn exceeds single-speaker evaluation audio")
            source.setpos(start)
            frames = source.readframes(end - start)
            if len(frames) != (end - start) * source.getsampwidth():
                raise EvaluationError("single-speaker evaluation audio ended early")
            with wave.open(str(clip), "wb") as output:
                output.setparams(
                    (1, source.getsampwidth(), rate, end - start, "NONE", "not compressed")
                )
                output.writeframes(frames)
        os.chmod(clip, 0o600)
        yield clip
    finally:
        clip.unlink(missing_ok=True)


def _run_single_speaker_local_only(
    conversation_id: str,
    audio: Path,
    expected_digest: str,
    duration: float,
    gold: Mapping[str, Any],
) -> dict[str, Any]:
    """Score local Qwen/XLS-R against one-speaker gold without a Precision retry."""

    gold_turns = _gold_turns(gold)
    if len({str(turn["speaker"]) for turn in gold_turns}) != 1:
        raise EvaluationError("single-speaker local evaluation requires exactly one gold speaker")
    if _sha256(audio) != expected_digest:
        raise EvaluationError(f"{conversation_id}: audio digest mismatch")
    if max(float(turn["end"]) for turn in gold_turns) > duration + 1e-6:
        raise EvaluationError(f"{conversation_id}: gold exceeds audio duration")

    settings = _settings(SCRATCH_ROOT / conversation_id / "single-speaker-local")
    qwen = Qwen3AsrProvider(profile=settings.qwen_profile, device=settings.asr_device)
    release = verify_release_bundle(cast(Path, settings.xlsr_release_path))
    raw_emotion = Wav2VecEmotionProvider(
        release.checkpoint_path,
        base_model_path=release.base_model_path,
        device=settings.emotion_device,
    )
    emotion = CalibratedEmotionProvider(
        raw_emotion,
        verify_calibration_artifact(cast(Path, settings.xlsr_calibration_path), release=release),
    )
    asset = AudioAsset(
        source_name=conversation_id,
        source_path=str(audio),
        normalized_paths=(str(audio),),
        channel_mode="mixed",
        duration_seconds=duration,
        channels=1,
        sha256=expected_digest,
    )
    started = time.monotonic()
    try:
        utterances = qwen.transcribe_single_speaker(asset)
        cer = score_sequences(
            characters(_gold_stream(gold_turns)), characters(_utterance_stream(utterances))
        )
        predictions: list[dict[str, Any]] = []
        unscored = 0
        for index, turn in enumerate(gold_turns, start=1):
            with _gold_turn_clip(audio, turn, SCRATCH_ROOT / conversation_id) as clip:
                try:
                    result = emotion.analyze(f"gold-{index:04d}", clip, str(turn["transcript"]))
                except Exception as error:
                    if getattr(error, "code", None) != "audio_too_short":
                        raise
                    unscored += 1
                    continue
            predictions.append(
                {
                    "start": float(turn["start"]),
                    "end": float(turn["end"]),
                    "emotion": max(result.probabilities, key=result.probabilities.__getitem__),
                }
            )
        coverage = qwen.last_timestamp_coverage
        return {
            "conversation_id": conversation_id,
            "preflight_only": False,
            "evaluation_mode": "single_speaker_local_auxiliary",
            "audio_sha256": expected_digest,
            "gold_sha256": gold.get("content_sha256"),
            "gold_speakers": 1,
            "remote": {
                "provider": PRECISION_MODEL_ID,
                "uploads": 0,
                "jobs": 0,
                "retries": 0,
                "status": "not_retried_after_prior_single_speaker_adapter_rejection",
            },
            "pipeline": {
                "status": "local_single_speaker_auxiliary_completed",
                "hypothesis_speakers": 1,
                "hypothesis_segments": None,
                "utterance_count": len(utterances),
                "timestamp_coverage": coverage.as_dict() if coverage is not None else None,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "real_time_factor": round((time.monotonic() - started) / duration, 6),
            },
            "precision_qwen": {
                "diarization": {
                    "status": "not_measured_single_speaker_precision_result_not_available",
                },
                "mixed_stream_cer": cer.as_dict(),
            },
            "xlsr_emotion": {
                "status": "scored_on_gold_turn_audio_not_role_pipeline",
                **_emotion_agreement(gold_turns, predictions),
                "unscored_audio_too_short_turn_count": unscored,
            },
            "gemini_baseline": _gemini_baseline(gold_turns, conversation_id, duration),
        }
    finally:
        qwen.unload()
        emotion.unload()
        shutil.rmtree(SCRATCH_ROOT / conversation_id, ignore_errors=True)


def _gemini_baseline(
    gold_turns: Sequence[Mapping[str, Any]], conversation_id: str, duration: float
) -> dict[str, Any]:
    silver_path = ANNOTATIONS / conversation_id / "silver.json"
    if not silver_path.is_file():
        return {
            "status": "not_available_for_reviewer_derived_gold_case",
            "source": "no_corresponding_gemini_silver",
        }
    silver = _read_json(silver_path)
    content = silver.get("content")
    raw_turns = content.get("turns") if isinstance(content, dict) else None
    if not isinstance(raw_turns, list):
        raise EvaluationError("silver has no turns")
    turns = [cast(dict[str, Any], item) for item in raw_turns if isinstance(item, dict)]
    hypothesis = [
        SpeakerSegment(
            start=float(turn["start"]),
            end=float(turn["end"]),
            speaker_id=str(turn["speaker"]),
            confidence=float(turn.get("confidence", 0.0)),
        )
        for turn in turns
    ]
    diarization = _score(
        _gold_annotation(gold_turns, conversation_id),
        _hypothesis_annotation(hypothesis, conversation_id),
        duration,
    )
    cer = score_sequences(characters(_gold_stream(gold_turns)), characters(_gold_stream(turns)))
    return {
        "source": "existing_gemini_silver_no_new_call",
        "diarization": diarization,
        "mixed_stream_cer": cer.as_dict(),
        "emotion": _emotion_agreement(gold_turns, turns),
        "silver_content_sha256": silver.get("content_sha256"),
    }


def _pipeline_emotions(
    report: ReportArtifact,
    utterances: Sequence[Utterance],
    gold_turns: Sequence[Mapping[str, Any]],
) -> dict[str, int | float | None]:
    by_id = {utterance.id: utterance for utterance in utterances}
    predictions: list[dict[str, Any]] = []
    for result in report.report.emotions:
        utterance = by_id.get(result.utterance_id)
        if utterance is None:
            continue
        label = max(result.probabilities, key=result.probabilities.__getitem__)
        predictions.append({"start": utterance.start, "end": utterance.end, "emotion": label})
    return _emotion_agreement(gold_turns, predictions)


def _public_pipeline_state(job: Mapping[str, Any]) -> dict[str, Any]:
    """Return only stage statuses and public error codes for an incomplete run.

    A provider can fail before an artifact exists.  The evaluator must preserve that
    boundary in its transcript-free report rather than assuming diarization and ASR
    artifacts were written just because ``run_until_pause`` returned a job record.
    """

    raw_stages = job.get("stages")
    if not isinstance(raw_stages, Mapping):
        return {"status": str(job.get("status", "unknown")), "stages": {}}
    stages: dict[str, dict[str, str | None]] = {}
    for stage, raw in raw_stages.items():
        if not isinstance(stage, str) or not isinstance(raw, Mapping):
            continue
        error_code: str | None = None
        error_json = raw.get("error_json")
        if isinstance(error_json, str):
            try:
                error = json.loads(error_json)
            except json.JSONDecodeError:
                error = None
            if isinstance(error, Mapping) and isinstance(error.get("code"), str):
                error_code = error["code"]
        stages[stage] = {
            "status": str(raw.get("status", "unknown")),
            "error_code": error_code,
        }
    return {"status": str(job.get("status", "unknown")), "stages": stages}


def _run_one(
    conversation_id: str,
    audio: Path,
    expected_digest: str,
    duration: float,
    gold: Mapping[str, Any],
    *,
    confirm_external_upload: bool,
) -> dict[str, Any]:
    gold_turns = _gold_turns(gold)
    if _sha256(audio) != expected_digest:
        raise EvaluationError(f"{conversation_id}: audio digest mismatch")
    if max(float(turn["end"]) for turn in gold_turns) > duration + 1e-6:
        raise EvaluationError(f"{conversation_id}: gold exceeds audio duration")
    if not confirm_external_upload:
        return {
            "conversation_id": conversation_id,
            "preflight_only": True,
            "gold_sha256": gold.get("content_sha256"),
            "audio_sha256": expected_digest,
            "gold_speakers": len({str(turn["speaker"]) for turn in gold_turns}),
            "gemini_baseline": _gemini_baseline(gold_turns, conversation_id, duration),
        }

    scratch = SCRATCH_ROOT / conversation_id
    if scratch.exists():
        raise EvaluationError(f"{conversation_id}: scratch exists")
    credentials = load_credentials()
    if credentials.pyannoteai_api_key is None:
        raise EvaluationError("missing pyannote credential")
    ledger = CallLedger(max_jobs=MAX_REMOTE_JOBS_PER_CONVERSATION)
    started = time.monotonic()
    dependencies = build_dependencies(
        _settings(scratch),
        credentials=credentials,
        provider_factories=ProviderFactories(
            pyannoteai_client=lambda *, timeout_seconds: LedgerClient(
                _default_client_factory(timeout_seconds=timeout_seconds), ledger
            )
        ),
    )
    job_id: str | None = None
    try:
        job_id = dependencies.repository.create_job(str(audio))
        paused = dependencies.runner.run_until_pause(job_id)
        try:
            diarized = dependencies.artifacts.read_model(job_id, StageName.DIARIZE, DiarizeArtifact)
            transcribed = dependencies.artifacts.read_model(
                job_id, StageName.TRANSCRIBE, TranscribeArtifact
            )
        except FileNotFoundError:
            return {
                "conversation_id": conversation_id,
                "preflight_only": False,
                "evaluation_status": "pipeline_failed_before_metrics",
                "audio_sha256": expected_digest,
                "gold_sha256": gold.get("content_sha256"),
                "gold_speakers": len({str(turn["speaker"]) for turn in gold_turns}),
                "remote": {
                    "provider": PRECISION_MODEL_ID,
                    "uploads": ledger.uploads,
                    "jobs": ledger.submissions,
                    "retries": 0,
                    "job_statuses": list(ledger.job_statuses),
                },
                "pipeline": {
                    "status_after_role_gate": str(paused.get("status")),
                    "public_state": _public_pipeline_state(
                        dependencies.repository.get_job(job_id)
                    ),
                },
                "precision_qwen": None,
                "xlsr_emotion": {
                    "status": "skipped_pipeline_failed_before_metrics",
                    "gold_turn_count": len(gold_turns),
                    "matched_turn_count": 0,
                    "correct_turn_count": 0,
                    "label_accuracy": None,
                },
                "gemini_baseline": _gemini_baseline(gold_turns, conversation_id, duration),
            }
        observed = sorted({segment.speaker_id for segment in diarized.alignment_segments})
        hypothesis = _hypothesis_annotation(diarized.alignment_segments, conversation_id)
        diarization = _score(_gold_annotation(gold_turns, conversation_id), hypothesis, duration)
        cer = score_sequences(
            characters(_gold_stream(gold_turns)),
            characters(_utterance_stream(transcribed.utterances)),
        )
        result: dict[str, Any] = {
            "conversation_id": conversation_id,
            "preflight_only": False,
            "audio_sha256": expected_digest,
            "gold_sha256": gold.get("content_sha256"),
            "gold_speakers": len({str(turn["speaker"]) for turn in gold_turns}),
            "remote": {
                "provider": PRECISION_MODEL_ID,
                "uploads": ledger.uploads,
                "jobs": ledger.submissions,
                "retries": 0,
                "job_statuses": list(ledger.job_statuses),
            },
            "pipeline": {
                "status_after_role_gate": str(paused.get("status")),
                "hypothesis_speakers": len(observed),
                "hypothesis_segments": len(diarized.alignment_segments),
                "utterance_count": len(transcribed.utterances),
                "timestamp_coverage": (
                    transcribed.timestamp_coverage.model_dump()
                    if transcribed.timestamp_coverage is not None
                    else None
                ),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "real_time_factor": round((time.monotonic() - started) / duration, 6),
            },
            "precision_qwen": {"diarization": diarization, "mixed_stream_cer": cer.as_dict()},
            "gemini_baseline": _gemini_baseline(gold_turns, conversation_id, duration),
        }
        if len(observed) == 2 and paused.get("status") == "paused":
            dependencies.runner.confirm_roles(
                job_id, {observed[0]: Role.CUSTOMER, observed[1]: Role.AGENT}
            )
            completed = dependencies.runner.run_until_pause(job_id)
            if completed.get("status") != "completed":
                raise EvaluationError(f"{conversation_id}: pipeline did not complete")
            report = dependencies.artifacts.read_model(job_id, StageName.REPORT, ReportArtifact)
            result["xlsr_emotion"] = {
                "status": "scored_with_arbitrary_role_mapping",
                **_pipeline_emotions(report, transcribed.utterances, gold_turns),
            }
        else:
            result["xlsr_emotion"] = {
                "status": (
                    "skipped_pipeline_supports_exactly_two_speakers"
                    if len(observed) != 2
                    else "skipped_role_gate_did_not_pause"
                ),
                "gold_turn_count": len(gold_turns),
                "matched_turn_count": 0,
                "correct_turn_count": 0,
                "label_accuracy": None,
            }
        if ledger.uploads != 1 or ledger.submissions != 1:
            raise EvaluationError(f"{conversation_id}: remote call ceiling violated")
        return result
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--confirm-external-upload", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DERIVED_MANIFEST)
    parser.add_argument("--conversation", action="append")
    parser.add_argument("--single-speaker-local-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        # httpx includes request URLs in INFO lines. Precision uploads use short-lived
        # pre-signed object-storage URLs, which do not belong in terminal history/logs.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        verify_local_models()
        audios = _manifest_audio(arguments.manifest)
        evaluations: list[dict[str, Any]] = []
        selected = tuple(arguments.conversation or tuple(audios))
        if len(set(selected)) != len(selected):
            raise EvaluationError("conversation selection contains duplicates")
        for conversation_id in selected:
            if conversation_id not in audios:
                raise EvaluationError("conversation is absent from selected manifest")
            audio, digest, duration = audios[conversation_id]
            gold = verify_gold(ANNOTATIONS / conversation_id / "gold.json")
            if arguments.single_speaker_local_only:
                if arguments.confirm_external_upload:
                    raise EvaluationError("single-speaker local evaluation must not upload audio")
                evaluations.append(
                    _run_single_speaker_local_only(
                        conversation_id, audio, digest, duration, gold
                    )
                )
            else:
                evaluations.append(
                    _run_one(
                        conversation_id,
                        audio,
                        digest,
                        duration,
                        gold,
                        confirm_external_upload=arguments.confirm_external_upload,
                    )
                )
        report = {
            "schema_version": "1",
            "kind": "user-gold-precision-qwen-xlsr-evaluation",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "transcript_free": True,
            "external_scope": {
                "authorization": "user-approved user-provided audio evaluation",
                "conversations": len(evaluations),
                "max_uploads_per_conversation": 1,
                "retries": 0,
            },
            "local_models": verify_local_models(),
            "evaluations": evaluations,
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"report: {arguments.output}")
        print(f"sha256: {_sha256(arguments.output)}")
        return 0
    except (AnnotationError, EvaluationError, KeyError, ValueError) as error:
        print(f"gold evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
