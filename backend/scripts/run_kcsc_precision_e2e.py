"""One-conversation KCSC end-to-end: remote Precision-2 diarization, local ASR and emotion.

This is the only script in the repository that sends KCSC audio off the machine, and it
does so for exactly one conversation, exactly once. Everything downstream of diarization
runs locally: Qwen3-ASR 1.7B for transcription, the calibrated XLS-R release for emotion.

The transfer is permitted here by an explicit, narrow user override. It is **not** a
rights confirmation: processing rights for this corpus have never been confirmed with the
copyright holder, and the artifact this script writes records that in both directions —
``user_limited_override`` as the permission, ``rights_confirmed: false`` as the fact. The
upload cannot be recalled once made; pyannoteAI states media input may sit in temporary
storage for up to 48 hours and the public API exposes no deletion endpoint.

Three properties are enforced rather than intended:

* **One job.** The outbound client is charged against a ledger with a ceiling of one
  diarization job and one upload. A second submission raises instead of being issued, so
  a retry cannot happen by accident or by exception handling.
* **No transcript anywhere.** The evaluation artifact carries counts, rates, digests, and
  durations. Utterance text is read by the pipeline and never reaches this script's
  output; a test pins that the artifact contains no transcript field.
* **Nothing original is altered.** The conversation is read from the derived set and its
  digest is re-checked after the run. All job state is written under a scratch root that
  is removed at the end.

Default provider configuration is untouched: this script constructs its own Settings and
never reads or writes ``backend/.env``, which keeps faster-whisper selected for every
other caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never

from pydantic import SecretStr

from voxdelta.api.dependencies import ProviderFactories, build_dependencies
from voxdelta.config import Settings
from voxdelta.credentials import load_credentials
from voxdelta.domain.models import Role, StageName
from voxdelta.evaluation.e2e_reconciliation import SCORING_FAILURE_PREFIX, classify
from voxdelta.evaluation.kcsc_asr_benchmark import KcscAsrError, configure_offline_cache
from voxdelta.evaluation.kcsc_diarization_benchmark import (
    KcscBenchmarkError,
    _hypothesis_annotation,
    _reference_annotation,
    _score,
    file_sha256,
    load_manifest,
)
from voxdelta.evaluation.kcsc_precision_benchmark import (
    EXTERNAL_PROCESSING_CAVEAT,
    FROZEN_CONVERSATIONS,
    RIGHTS_HOLDER,
    CallLedger,
    LedgerClient,
)
from voxdelta.evaluation.kcsc_qwen_track_benchmark import verify_qwen_identity
from voxdelta.pipeline.stages import (
    DiarizeArtifact,
    ReportArtifact,
    TranscribeArtifact,
)
from voxdelta.providers.pyannote_precision import PRECISION_MODEL_ID, _default_client_factory
from voxdelta.providers.qwen3_asr import ALIGNER_MODEL_ID, DEFAULT_MODEL_ID
from voxdelta.providers.qwen_timestamps import SANITATION_POLICY

SCHEMA_VERSION = "1"

#: The whole run is one job. Stated once, enforced by the ledger, asserted after the fact.
MAX_DIARIZATION_JOBS = 1

#: Deterministic and minimising: of the frozen benchmark conversations, take the shortest
#: audio, breaking ties by id. Choosing the smallest payload keeps the irrevocable part of
#: this run as small as the approved scope allows.
SELECTION_RULE = "shortest derived audio among the frozen benchmark conversations, ties by id"

#: What this runner is actually permitted to do, which is narrower than the programme-wide
#: scope in ``kcsc_precision_benchmark``. One conversation, one upload, one job, no retries.
E2E_AUTHORIZATION_SCOPE = (
    "Operator override limited to exactly one derived KCSC conversation, one upload and "
    "one diarization job, no retries. Not a general permission to transmit this corpus, "
    "and not evidence that the transfer was licensed."
)

DEFAULT_DERIVED = Path("data/derived/kcsc")
DEFAULT_RECORD = Path("backend/runtime/poc/kcsc-precision2-qwen-xlsr-e2e/EVALUATION.json")
DEFAULT_SCRATCH = Path("data/jobs/e2e-precision-scratch")
MODEL_CACHE = Path("data/models/hf-cache/hub")
XLSR_RELEASE = Path("data/models/xls-r-emotion-7class-v1").resolve()
XLSR_CALIBRATION = Path("data/models/xls-r-emotion-7class-v1-calibration-v2").resolve()
ENV_FILE = Path("backend/.env")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def select_conversation(derived_root: Path) -> dict[str, Any]:
    """Apply :data:`SELECTION_RULE` to the derivation manifest and verify the inputs."""

    manifest = load_manifest(derived_root)
    entries = manifest["conversations"]
    assert isinstance(entries, list)
    candidates = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("conversation_id") in FROZEN_CONVERSATIONS
    ]
    if len(candidates) != len(FROZEN_CONVERSATIONS):
        raise KcscBenchmarkError("derived set does not contain the frozen benchmark conversations")
    chosen = min(
        candidates,
        key=lambda entry: (float(entry["derived_duration_seconds"]), str(entry["conversation_id"])),
    )
    outputs = chosen["outputs"]
    assert isinstance(outputs, dict)
    audio = derived_root / str(outputs["audio"])
    reference = derived_root / str(outputs["reference"])
    for path, expected in (
        (audio, outputs["audio_sha256"]),
        (reference, outputs["reference_sha256"]),
    ):
        if not path.is_file():
            raise KcscBenchmarkError(f"missing derived input: {path}")
        found = file_sha256(path)
        if found != expected:
            raise KcscBenchmarkError(f"{path.name}: checksum mismatch against the manifest")
    source = manifest.get("source")
    return {
        "conversation_id": str(chosen["conversation_id"]),
        "audio_path": audio,
        "reference_path": reference,
        "audio_sha256": str(outputs["audio_sha256"]),
        "reference_sha256": str(outputs["reference_sha256"]),
        "audio_bytes": int(outputs["audio_bytes"]),
        "duration_seconds": float(chosen["derived_duration_seconds"]),
        "speakers": [str(item) for item in chosen["speakers"]],
        "source_revision": str(source.get("revision")) if isinstance(source, dict) else None,
        "manifest_sha256": file_sha256(derived_root / "manifest.json"),
    }


def verify_local_models(cache_root: Path = MODEL_CACHE) -> dict[str, Any]:
    """Confirm every model this run needs is already on disk, and pin the hub client offline.

    An earlier run asserted that it downloaded nothing while never pinning the hub client
    at all, which left the assertion unverifiable. Two things close that here, in this
    order and before any model is constructed:

    * every checkpoint is resolved and hashed out of the local cache, so a missing or
      moved artifact aborts the run rather than becoming a fetch; and
    * ``HF_HUB_CACHE`` and the offline flags are set, so the hub client cannot treat a
      cache miss as a reason to reach out even if one somehow survived the check above.

    Calling this before the external upload matters: a cache problem must stop the run
    while stopping is still free, not after an irrevocable transfer has been made.
    """

    missing = [str(path) for path in (XLSR_RELEASE, XLSR_CALIBRATION) if not path.is_dir()]
    if missing:
        raise KcscBenchmarkError(f"missing local model artifacts: {', '.join(missing)}")
    if not cache_root.is_dir():
        raise KcscBenchmarkError(f"missing local hub cache: {cache_root}")

    try:
        asr = verify_qwen_identity(cache_root, DEFAULT_MODEL_ID)
        aligner = verify_qwen_identity(cache_root, ALIGNER_MODEL_ID)
    except KcscAsrError as error:
        raise KcscBenchmarkError(f"qwen checkpoints unusable offline: {error}") from None

    # Set only after the cache has been proven complete, so the offline pin is never
    # covering for an artifact that is not actually there.
    configure_offline_cache(cache_root)
    return {
        "hub_cache": str(cache_root.resolve()),
        "hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "qwen_asr_revision": asr.revision,
        "qwen_asr_tree_sha256": asr.tree_sha256,
        "qwen_aligner_revision": aligner.revision,
        "qwen_aligner_tree_sha256": aligner.tree_sha256,
        "xlsr_release": str(XLSR_RELEASE),
        "xlsr_release_manifest_sha256": file_sha256(XLSR_RELEASE / "RELEASE.json"),
        "xlsr_calibration": str(XLSR_CALIBRATION),
        "xlsr_calibration_sha256": file_sha256(XLSR_CALIBRATION / "CALIBRATION.json"),
    }


def env_default_provider() -> str | None:
    """Read the shipped default without touching it, so the run can prove it unchanged."""

    if not ENV_FILE.is_file():
        return None
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("VOXDELTA_ASR_PROVIDER="):
            return line.split("=", 1)[1].strip()
    return None


def _settings(scratch: Path) -> Settings:
    """Explicit settings for this run only; the env file is never read or written."""

    return Settings(
        data_root=scratch,
        database_path=scratch / "voxdelta.sqlite3",
        api_capability_token=SecretStr("e" * 43),
        diarization_provider="pyannoteai-precision",
        asr_provider="qwen3",
        qwen_profile="default",
        asr_device="mps",
        emotion_provider="wav2vec",
        emotion_device="cpu",
        xlsr_release_enabled=True,
        xlsr_release_path=XLSR_RELEASE,
        xlsr_calibration_enabled=True,
        xlsr_calibration_path=XLSR_CALIBRATION,
    )


def _provenance_block(provider: Any) -> dict[str, Any]:
    return {
        "name": provider.name,
        "model": provider.model,
        "remote": provider.remote,
        "revision": provider.revision,
    }


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument(
        "--confirm-external-upload",
        action="store_true",
        help="required to transmit audio; without it the run stops after preflight",
    )
    parser.add_argument(
        "--authorization",
        choices=("user_limited_override",),
        help="what permits this transfer. There is no value meaning 'rights confirmed'.",
    )
    parser.add_argument("--keep-scratch", action="store_true", help="skip scratch cleanup")
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="which authorized attempt this is; each attempt is a separate override",
    )
    parser.add_argument(
        "--supersedes",
        type=Path,
        default=None,
        help="a prior run's record, referenced by digest for lineage. Never modified.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911, PLR0915
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("e2e failed: invalid arguments", file=sys.stderr)
        return 2

    try:
        selection = select_conversation(arguments.derived)
        models = verify_local_models()
        credentials = load_credentials()
    except KcscBenchmarkError as error:
        print(f"e2e failed: {error}", file=sys.stderr)
        return 2
    except Exception:
        print("e2e failed: could not complete preflight", file=sys.stderr)
        return 2

    shipped_default = env_default_provider()
    key_present = credentials.pyannoteai_api_key is not None

    print("preflight: KCSC one-conversation E2E (remote diarization, local ASR + emotion)")
    print(f"  selection rule      : {SELECTION_RULE}")
    print(f"  conversation        : {selection['conversation_id']}")
    print(f"  audio sha256        : {selection['audio_sha256']}")
    print(f"  reference sha256    : {selection['reference_sha256']}")
    print(
        f"  audio               : {selection['audio_bytes']} bytes, "
        f"{selection['duration_seconds']:.3f}s, speakers={'+'.join(selection['speakers'])}"
    )
    print(f"  source revision     : {selection['source_revision']}")
    print(f"  remote diarization  : pyannoteai / {PRECISION_MODEL_ID} (remote=True)")
    print("  local asr           : Qwen3-ASR-1.7B (qwen3 profile=default, mps)")
    print(f"  local emotion       : calibrated XLS-R release {models['xlsr_release']}")
    print(f"  hub cache           : {models['hub_cache']} (HF_HUB_OFFLINE={models['hub_offline']})")
    print(f"  qwen asr tree       : sha256={models['qwen_asr_tree_sha256']}")
    print(f"  api key             : {'configured' if key_present else 'MISSING'}")
    print(
        f"  planned remote jobs : {MAX_DIARIZATION_JOBS} "
        f"(ceiling {MAX_DIARIZATION_JOBS}, retries 0)"
    )
    print(f"  shipped .env default: VOXDELTA_ASR_PROVIDER={shipped_default} (not modified)")
    print(f"  rights confirmed    : NO — {E2E_AUTHORIZATION_SCOPE}")
    print(f"  rights holder       : {RIGHTS_HOLDER}")

    if not arguments.confirm_external_upload:
        print()
        print("stopping after preflight: nothing was transmitted.")
        print(f"caveat: {EXTERNAL_PROCESSING_CAVEAT}")
        return 0
    if arguments.authorization is None:
        print("e2e failed: --authorization is required to transmit audio", file=sys.stderr)
        return 2
    if not key_present:
        print("e2e failed: missing pyannote api key", file=sys.stderr)
        return 2

    scratch: Path = arguments.scratch
    if scratch.exists():
        print(f"e2e failed: scratch root already exists: {scratch}", file=sys.stderr)
        return 2

    ledger = CallLedger(max_jobs=MAX_DIARIZATION_JOBS)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    job_id: str | None = None
    try:
        dependencies = build_dependencies(
            _settings(scratch),
            credentials=credentials,
            provider_factories=ProviderFactories(
                pyannoteai_client=lambda *, timeout_seconds: LedgerClient(
                    _default_client_factory(timeout_seconds=timeout_seconds), ledger
                ),
            ),
        )
        job_id = dependencies.repository.create_job(str(selection["audio_path"]))
        paused = dependencies.runner.run_until_pause(job_id)
        if paused.get("status") != "paused":
            raise KcscBenchmarkError(f"pipeline did not pause for roles: {paused.get('status')}")

        diarized = dependencies.artifacts.read_model(job_id, StageName.DIARIZE, DiarizeArtifact)
        speakers = sorted({segment.speaker_id for segment in diarized.alignment_segments})
        if len(speakers) != 2:
            raise KcscBenchmarkError(f"expected two speakers, diarization produced {len(speakers)}")
        # Deterministic and arbitrary: this evaluation has no ground truth for which side
        # is the customer, and nothing downstream of the label is being scored here.
        mapping = {speakers[0]: Role.CUSTOMER, speakers[1]: Role.AGENT}
        dependencies.runner.confirm_roles(job_id, mapping)
        final = dependencies.runner.run_until_pause(job_id)
        if final.get("status") != "completed":
            raise KcscBenchmarkError(f"pipeline did not complete: {final.get('status')}")
    except Exception as error:  # noqa: BLE001 - the ledger state is the diagnosis
        print(f"e2e failed: {type(error).__name__}: {error}", file=sys.stderr)
        print(
            f"  remote jobs submitted before stopping: {ledger.submissions}; no retry attempted",
            file=sys.stderr,
        )
        if job_id and not arguments.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        return 2
    elapsed = time.monotonic() - started

    # ---------------------------------------------------------------- post-run verification
    failures: list[str] = []
    if ledger.submissions != MAX_DIARIZATION_JOBS:
        failures.append(
            f"expected {MAX_DIARIZATION_JOBS} diarization job, ledger counted {ledger.submissions}"
        )
    if ledger.uploads != MAX_DIARIZATION_JOBS:
        failures.append(f"expected {MAX_DIARIZATION_JOBS} upload, ledger counted {ledger.uploads}")
    if len(ledger.job_ids) != MAX_DIARIZATION_JOBS:
        failures.append(f"expected one remote job id, saw {len(ledger.job_ids)}")

    transcribed = dependencies.artifacts.read_model(
        job_id, StageName.TRANSCRIBE, TranscribeArtifact
    )
    report = dependencies.artifacts.read_model(job_id, StageName.REPORT, ReportArtifact).report
    coverage = report.transcription_coverage
    asr_provenance = transcribed.provider
    diarize_provenance = diarized.provider
    if asr_provenance is None or diarize_provenance is None:
        print("e2e failed: a stage artifact carries no provenance", file=sys.stderr)
        return 2

    if asr_provenance.remote or asr_provenance.name != "qwen3-asr":
        failures.append("transcription provenance is not the local Qwen provider")
    if asr_provenance.model != "Qwen3-ASR-1.7B":
        failures.append(f"unexpected ASR model {asr_provenance.model}")
    if not diarize_provenance.remote or diarize_provenance.name != "pyannoteai":
        failures.append("diarization provenance is not the remote Precision provider")
    emotion_providers = {result.provider.name for result in report.emotions}
    if report.emotions and emotion_providers != {"wav2vec-xls-r"}:
        failures.append(f"unexpected emotion provenance {sorted(emotion_providers)}")
    if report.emotions and not any(result.calibration is not None for result in report.emotions):
        failures.append("calibrated XLS-R produced no calibration metadata")
    if coverage is None:
        failures.append("Qwen run produced no timestamp coverage")
    else:
        if coverage.policy != SANITATION_POLICY:
            failures.append(f"unexpected sanitation policy {coverage.policy}")
        if coverage.uncertain != (coverage.omitted_words > 0):
            failures.append("coverage uncertain flag disagrees with its omission count")
        if any(item.end <= item.start for item in report.utterances):
            failures.append("an utterance carries impossible timing")
    if Settings.model_fields["asr_provider"].default != "fake":
        failures.append("the ASR provider default changed")
    if env_default_provider() != shipped_default:
        failures.append("backend/.env was modified by this run")
    if file_sha256(selection["audio_path"]) != selection["audio_sha256"]:
        failures.append("the derived source audio changed during the run")

    # ------------------------------------------------------------------------ diarization DER
    try:
        # _reference_annotation validates the entry against the reference: it requires the
        # declared speakers and cross-checks the duration. Passing only the id made it
        # raise before any DER was computed.
        reference, reference_speakers, duration, _speech, turn_count = _reference_annotation(
            selection["reference_path"],
            {
                "conversation_id": selection["conversation_id"],
                "derived_duration_seconds": selection["duration_seconds"],
                "speakers": selection["speakers"],
            },
        )
        hypothesis = _hypothesis_annotation(
            diarized.alignment_segments, selection["conversation_id"]
        )
        metrics = _score(reference, hypothesis, duration)
    except Exception as error:  # noqa: BLE001 - scoring is a measurement, not the run
        # Recorded with the scoring prefix so the reconciliation taxonomy can tell a
        # measurement that could not be taken from a pipeline that did not work.
        metrics = {}
        reference_speakers, turn_count = (), None
        failures.append(f"{SCORING_FAILURE_PREFIX}: {type(error).__name__}: {error}")

    hypothesis_timeline = [
        {
            "start": round(segment.start, 6),
            "end": round(segment.end, 6),
            "speaker_id": segment.speaker_id,
        }
        for segment in diarized.alignment_segments
    ]

    abstained = sum(
        1 for result in report.emotions if result.calibration and result.calibration.abstained
    )
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "kcsc-precision2-qwen-xlsr-e2e",
        "started_at": started_at,
        "transcript_free": True,
        "rights": {
            "authorization": arguments.authorization,
            "user_limited_override": True,
            "rights_confirmed": False,
            "attempt": arguments.attempt,
            "attempt_note": (
                "each attempt is a separate, independently granted limited override; "
                "it does not extend or inherit any earlier grant"
            ),
            "supersedes": (
                {
                    "path": str(arguments.supersedes),
                    "sha256": file_sha256(arguments.supersedes),
                }
                if arguments.supersedes and arguments.supersedes.is_file()
                else None
            ),
            "rights_holder": RIGHTS_HOLDER,
            "scope": E2E_AUTHORIZATION_SCOPE,
            "conversation_count": 1,
            "caveat": EXTERNAL_PROCESSING_CAVEAT,
            "licensed_or_private_audio_transmitted": True,
        },
        "input": {
            "selection_rule": SELECTION_RULE,
            "conversation_id": selection["conversation_id"],
            "audio_sha256": selection["audio_sha256"],
            "reference_sha256": selection["reference_sha256"],
            "derivation_manifest_sha256": selection["manifest_sha256"],
            "source_revision": selection["source_revision"],
            "audio_bytes": selection["audio_bytes"],
            "duration_seconds": selection["duration_seconds"],
        },
        "remote": {
            "provider": _provenance_block(diarize_provenance),
            "diarization_jobs_submitted": ledger.submissions,
            "uploads": ledger.uploads,
            "remote_job_count": len(ledger.job_ids),
            "retries": 0,
            "max_diarization_jobs": MAX_DIARIZATION_JOBS,
            "call_counts_by_kind": ledger.counts(),
            "hosts_contacted": list(ledger.hosts),
            "job_statuses": list(ledger.job_statuses),
            "transcription_requested": False,
        },
        "local": {
            "asr": _provenance_block(asr_provenance),
            "emotion": sorted(
                {
                    json.dumps(_provenance_block(result.provider), sort_keys=True)
                    for result in report.emotions
                }
            ),
            "models": models,
            # Enforced rather than asserted: the cache was proven complete above and the
            # hub client is pinned offline before any model is constructed.
            "hub_offline_enforced": os.environ.get("HF_HUB_OFFLINE") == "1",
            "hub_cache_pinned": os.environ.get("HF_HUB_CACHE"),
        },
        "metrics": {
            # None rather than 0 when the scorer never ran: zero is a measurement.
            "reference_turn_count": turn_count,
            "reference_speakers": len(reference_speakers) if reference_speakers else None,
            "hypothesis_segment_count": len(diarized.alignment_segments),
            "hypothesis_speaker_count": len({s.speaker_id for s in diarized.alignment_segments}),
            "diarization": metrics,
            "utterance_count": len(report.utterances),
            "emotion_result_count": len(report.emotions),
            "emotion_abstained_count": abstained,
            "transition_count": len(report.transitions),
            "report_warning_count": len(report.warnings),
            "elapsed_seconds": round(elapsed, 3),
            "real_time_factor": round(elapsed / selection["duration_seconds"], 6),
        },
        # Boundaries only, no transcript. Kept so a scoring gap can be closed locally
        # instead of needing another submission, which is what made the first run's
        # missing DER unrecoverable.
        "hypothesis_timeline": hypothesis_timeline,
        "timestamp_coverage": coverage.model_dump() if coverage else None,
        "configuration": {
            "shipped_env_asr_provider": shipped_default,
            "shipped_env_modified": False,
            "settings_asr_provider_default": Settings.model_fields["asr_provider"].default,
        },
        "verification_failures": failures,
    }

    arguments.record.parent.mkdir(parents=True, exist_ok=True)

    print()
    print(f"remote diarization jobs : {ledger.submissions} (uploads {ledger.uploads}, retries 0)")
    print(f"remote call breakdown   : {ledger.counts()}")
    print(f"hosts contacted         : {', '.join(ledger.hosts)}")
    print(f"utterances              : {len(report.utterances)}")
    print(f"emotion results         : {len(report.emotions)} ({abstained} abstained)")
    print(f"timestamp coverage      : {record['timestamp_coverage']}")
    if metrics:
        collar = metrics["collar_250ms"]
        print(
            f"diarization DER (250ms) : {collar['der']:.4f} strict {metrics['strict']['der']:.4f}"
        )
    print(f"elapsed                 : {elapsed:.1f}s")
    print(f"record                  : {arguments.record}")

    outcome = classify(failures)
    record["outcome"] = outcome.as_dict()
    raw = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    arguments.record.write_text(f"{raw}\n", encoding="utf-8")
    digest = hashlib.sha256(arguments.record.read_bytes()).hexdigest()

    # Scratch holds the only copy of the run's intermediate state, so it is removed only
    # after the artifact exists and reads back with everything the run needed to keep.
    # The first run deleted it before that check and lost its diarization timeline.
    readback = json.loads(arguments.record.read_text(encoding="utf-8"))
    persisted = (
        readback.get("outcome") is not None
        and readback.get("hypothesis_timeline")
        and readback.get("timestamp_coverage") is not None
    )
    if not persisted:
        print(
            "e2e failed: evaluation record incomplete; keeping scratch for inspection",
            file=sys.stderr,
        )
        print(f"scratch kept            : {scratch}", file=sys.stderr)
        return 2
    if not arguments.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"scratch removed         : {scratch} (after artifact verified)")

    print(f"record sha256           : {digest}")
    print(f"pipeline status         : {outcome.pipeline_status}")
    print(f"scoring status          : {outcome.scoring_status}")
    for failure in failures:
        print(f"verification failed: {failure}", file=sys.stderr)
    if outcome.pipeline_failures:
        return 2
    if outcome.scoring_failures:
        # The run completed; the measurement did not. Distinct exit code so an operator
        # can tell the two apart without reading the artifact.
        return 3
    print("verification            : all post-run checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
