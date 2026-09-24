"""Create ONE Gemini emotion-only overlay candidate for a fixed KCSC conversation.

Everything is bounded before anything is transmitted: the conversation is fixed in code,
the audio is verified by digest, the clean Silver draft is read and its content hash
pinned into the artifact, consent is required explicitly by name, and the call ledger
allows exactly one upload, one interaction, and one delete. There is no retry — a failure
prints what happened and stops, because a second attempt is a second transmission.

The API key is read only through the protected credentials module and is never printed,
logged, or placed on a command line. Transcript text is transmitted so the model can key
its answer to the right turn, and it goes nowhere else: this script prints counts,
digests, and call facts, and the state file holds the same.

Nothing here writes ``silver.json``, gold, or the local XLS-R candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never

import httpx

from voxdelta.annotation.gemini_emotion_overlay import (
    FILENAME,
    RESPONSE_SCHEMA,
    build_prompt,
    provenance,
    source_turns,
    validate_rows,
    write_candidate,
)
from voxdelta.annotation.gemini_silver import (
    MAX_DELETES,
    MAX_INTERACTIONS,
    MAX_UPLOADS,
    CallLedger,
    CallTrace,
    GeminiAnnotator,
)
from voxdelta.annotation.store import DEFAULT_ROOT, read_silver
from voxdelta.credentials import load_credentials
from voxdelta.evaluation.kcsc_diarization_benchmark import file_sha256

CONVERSATION_ID = "A6000_S0005_0"
AUDIO = Path(f"data/derived/kcsc/audio/{CONVERSATION_ID}.wav")
AUDIO_SHA256 = "01565974fdf7021e0d0afcbae8c040839777434f40416f1e265aeee3926cce27"
DURATION_SECONDS = 583.909562
STATE = Path("data/jobs/benchmarks/gemini-emotion-overlay-A6000_S0005_0.state.json")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=AUDIO)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument(
        "--confirm-external-upload",
        action="store_true",
        help="required to transmit audio to Google; without it the run stops after preflight",
    )
    parser.add_argument(
        "--consent",
        choices=("gemini_audio_transfer",),
        help="the Gemini-specific consent. No other provider's consent substitutes for it.",
    )
    return parser


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911, PLR0915
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("gemini emotion overlay failed: invalid arguments", file=sys.stderr)
        return 2

    audio: Path = arguments.audio
    if not audio.is_file():
        print(f"gemini emotion overlay failed: missing audio {audio}", file=sys.stderr)
        return 2
    digest = file_sha256(audio)
    if digest != AUDIO_SHA256:
        print(
            "gemini emotion overlay failed: audio digest does not match the pinned input",
            file=sys.stderr,
        )
        return 2

    silver_path = arguments.root / CONVERSATION_ID / "silver.json"
    try:
        silver = read_silver(silver_path)
        turns = source_turns(silver)
    except Exception:
        print("gemini emotion overlay failed: clean silver draft is unreadable", file=sys.stderr)
        return 2
    silver_digest = str(silver.get("content_sha256", ""))
    if len(silver_digest) != 64:
        print("gemini emotion overlay failed: silver has no content digest", file=sys.stderr)
        return 2
    # Refuse before transmitting rather than after: the O_EXCL write would reject this
    # anyway, and discovering it post-call would have spent the transmission for nothing.
    candidate_path = arguments.root / CONVERSATION_ID / FILENAME
    if candidate_path.exists():
        print(
            "gemini emotion overlay failed: a candidate already exists for this conversation",
            file=sys.stderr,
        )
        return 2

    try:
        credentials = load_credentials()
    except Exception:
        print("gemini emotion overlay failed: could not load credentials", file=sys.stderr)
        return 2
    key_present = credentials.gemini_api_key is not None

    declared = provenance()
    print("preflight: Gemini emotion-only overlay candidate (one conversation, one call each)")
    print(f"  conversation        : {CONVERSATION_ID}")
    print(f"  audio sha256        : {digest}")
    print(f"  audio bytes         : {audio.stat().st_size}")
    print(f"  silver content sha  : {silver_digest}")
    print(f"  silver turns        : {len(turns)}")
    print(f"  provider            : {declared.name} / {declared.model} (remote={declared.remote})")
    print(f"  transmits           : {', '.join(declared.transmits)}")
    print(f"  retention policy    : {declared.retention_policy_url}")
    print(f"  api key             : {'configured' if key_present else 'MISSING'}")
    print(
        f"  ceilings            : upload={MAX_UPLOADS} interaction={MAX_INTERACTIONS} "
        f"delete={MAX_DELETES}, retries=0"
    )
    print("  overlays            : emotion, emotion_rationale, confidence")
    print("  preserves           : start, end, speaker, transcript (copied from clean silver)")
    print("  produces            : review_required candidate; never silver, never gold")
    sys.stdout.flush()

    if not arguments.confirm_external_upload:
        print()
        print("stopping after preflight: nothing was transmitted.")
        return 0
    if arguments.consent != "gemini_audio_transfer":
        print(
            "gemini emotion overlay failed: --consent gemini_audio_transfer is required; "
            "consent for another provider does not extend to Gemini",
            file=sys.stderr,
        )
        return 2
    if not key_present:
        print("gemini emotion overlay failed: no Gemini API key configured", file=sys.stderr)
        return 2

    state: dict[str, Any] = {
        "job": "gemini-emotion-overlay-A6000_S0005_0",
        "state": "running",
        "conversation_id": CONVERSATION_ID,
        "audio_sha256": digest,
        "source_silver_content_sha256": silver_digest,
        "source_turn_count": len(turns),
        "model": declared.model,
        "ceilings": {
            "uploads": MAX_UPLOADS,
            "interactions": MAX_INTERACTIONS,
            "deletes": MAX_DELETES,
            "retries": 0,
        },
        "consent": "gemini_audio_transfer",
        "remote_audio_transmitted": True,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_state(arguments.state, state)

    secret = credentials.gemini_api_key
    assert secret is not None
    annotator: GeminiAnnotator | None = None
    deleted = False
    detail = "not attempted"
    started = time.monotonic()
    try:
        with httpx.Client(timeout=600.0, follow_redirects=True) as client:
            annotator = GeminiAnnotator(
                api_key=secret.get_secret_value(),
                client=client,  # type: ignore[arg-type]
                consent_granted=True,
            )
            remote_uri, remote_name = annotator._upload(audio, "audio/wav")
            try:
                payload = annotator._interact(
                    remote_uri,
                    "audio/wav",
                    prompt=build_prompt(turns, duration_seconds=DURATION_SECONDS),
                    response_schema=RESPONSE_SCHEMA,
                )
                rows = validate_rows(payload, source_turn_count=len(turns))
            finally:
                # Runs whether the interaction succeeded, failed, or raised: an upload
                # that could not be withdrawn is a fact the reviewer needs either way.
                deleted, detail = annotator._delete(remote_name)
    except Exception as error:  # noqa: BLE001 - no retry; the type is the diagnosis
        trace = annotator.trace.as_dict() if annotator else CallTrace().as_dict()
        counts = dict(annotator.ledger.counts() if annotator else CallLedger().counts())
        state.update(
            {
                "state": "failed",
                "reason": type(error).__name__,
                "trace": trace,
                "call_counts": counts,
                "remote_file_deleted": deleted,
                "deletion_detail": detail,
                "retry_attempted": False,
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        _write_state(arguments.state, state)
        print(f"  stage reached      : {trace.get('stage_reached')}")
        print(f"  last http status   : {trace.get('last_http_status')}")
        print(f"  audio bytes sent   : {trace.get('audio_bytes_sent', 0)}")
        print(f"  remote deleted     : {deleted} ({detail})")
        print(f"gemini emotion overlay failed: {type(error).__name__}", file=sys.stderr)
        print("  no retry attempted; a second attempt would be a second transmission")
        return 2
    elapsed = time.monotonic() - started

    path, summary = write_candidate(
        annotation_root=arguments.root,
        conversation_id=CONVERSATION_ID,
        silver_record=silver,
        rows=rows,
        input_sha256=digest,
        model=annotator.provenance.model,
        remote_file_deleted=deleted,
        deletion_detail=detail,
        call_counts=annotator.ledger.counts(),
    )
    artifact_digest = file_sha256(path)
    candidate = json.loads(path.read_text(encoding="utf-8"))

    state.update(
        {
            "state": "completed",
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 3),
            "call_counts": dict(annotator.ledger.counts()),
            "remote_file_deleted": deleted,
            "deletion_detail": detail,
            "candidate_path": str(path),
            "candidate_content_sha256": str(candidate["content_sha256"]),
            "candidate_artifact_sha256": artifact_digest,
            "review_state": str(candidate["review_state"]),
            "promotable": bool(candidate["promotable"]),
            "overlaid_row_count": len(rows),
            "retry_attempted": False,
        }
    )
    _write_state(arguments.state, state)

    print()
    print(f"call counts            : {dict(annotator.ledger.counts())}")
    print(f"remote file deleted    : {deleted} ({detail})")
    print(f"overlaid rows          : {len(rows)} of {len(turns)} silver turns")
    print(f"uncertain turns        : {summary['uncertain_turns']}")
    print(f"mean confidence        : {summary['mean_confidence']}")
    print(f"emotion histogram      : {summary['emotion_histogram']}")
    print(f"transcript characters  : {summary['transcript_characters']} (text not shown)")
    print(f"review state           : {candidate['review_state']}")
    print(f"promotable             : {candidate['promotable']}")
    print(f"candidate artifact     : {path}")
    print(f"  bound to silver      : {silver_digest}")
    print(f"  content sha256       : {candidate['content_sha256']}")
    print(f"  artifact sha256      : {artifact_digest}")
    print(f"elapsed                : {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
