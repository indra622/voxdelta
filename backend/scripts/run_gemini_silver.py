"""Create ONE Gemini silver annotation for a fixed KCSC conversation.

Everything about this run is bounded before it starts: the conversation is fixed in code,
the audio is verified by digest, consent is required explicitly, and the call ledger allows
exactly one upload, one interaction, and one delete. There is no retry anywhere — a failure
prints what happened and stops, because a second attempt would be a second transmission.

The API key is read only through the protected credentials module and is never printed,
logged, or placed on a command line. Transcript text goes only into the private silver
artifact; this script prints counts, digests, and call facts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never

import httpx

from voxdelta.annotation.gemini_silver import (
    GEMINI_MODEL,
    MAX_DELETES,
    MAX_INTERACTIONS,
    MAX_UPLOADS,
    PROMPT,
    CallLedger,
    CallTrace,
    GeminiAnnotator,
    provenance,
)
from voxdelta.annotation.store import DEFAULT_ROOT, write_silver
from voxdelta.credentials import load_credentials
from voxdelta.evaluation.kcsc_diarization_benchmark import file_sha256

DEFAULT_CONVERSATION_ID = "A6000_S0005_0"


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--conversation-id", default=DEFAULT_CONVERSATION_ID)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--state", type=Path)
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


def _duration_seconds(audio: Path) -> float:
    with wave.open(str(audio), "rb") as source:
        return source.getnframes() / source.getframerate()


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("gemini silver failed: invalid arguments", file=sys.stderr)
        return 2

    conversation_id = str(arguments.conversation_id)
    audio = arguments.audio or Path(f"data/derived/kcsc/audio/{conversation_id}.wav")
    state_path = arguments.state or Path(
        f"data/jobs/benchmarks/gemini-silver-{conversation_id}.state.json"
    )
    if not audio.is_file():
        print(f"gemini silver failed: missing audio {audio}", file=sys.stderr)
        return 2
    digest = file_sha256(audio)
    try:
        duration_seconds = _duration_seconds(audio)
    except (wave.Error, ZeroDivisionError):
        print("gemini silver failed: unreadable wav duration", file=sys.stderr)
        return 2

    try:
        credentials = load_credentials()
    except Exception:
        print("gemini silver failed: could not load credentials", file=sys.stderr)
        return 2
    key_present = credentials.gemini_api_key is not None

    declared = provenance()
    print("preflight: Gemini silver annotation (one conversation, one call each)")
    print(f"  conversation        : {conversation_id}")
    print(f"  audio sha256        : {digest}")
    print(f"  audio bytes         : {audio.stat().st_size}")
    print(f"  provider            : {declared.name} / {declared.model} (remote={declared.remote})")
    print(f"  transmits           : {', '.join(declared.transmits)}")
    print(f"  retention policy    : {declared.retention_policy_url}")
    print(f"  api key             : {'configured' if key_present else 'MISSING'}")
    print(
        f"  ceilings            : upload={MAX_UPLOADS} interaction={MAX_INTERACTIONS} "
        f"delete={MAX_DELETES}, retries=0"
    )
    print("  produces            : silver, review_required; never gold without a human")
    sys.stdout.flush()

    if not arguments.confirm_external_upload:
        print()
        print("stopping after preflight: nothing was transmitted.")
        return 0
    if arguments.consent != "gemini_audio_transfer":
        print(
            "gemini silver failed: --consent gemini_audio_transfer is required; consent "
            "for another provider does not extend to Gemini",
            file=sys.stderr,
        )
        return 2
    if not key_present:
        print("gemini silver failed: no Gemini API key configured", file=sys.stderr)
        return 2

    state: dict[str, Any] = {
        "job": f"gemini-silver-{conversation_id}",
        "state": "running",
        "conversation_id": conversation_id,
        "audio_sha256": digest,
        "model": GEMINI_MODEL,
        "ceilings": {
            "uploads": MAX_UPLOADS,
            "interactions": MAX_INTERACTIONS,
            "deletes": MAX_DELETES,
            "retries": 0,
        },
        "consent": "gemini_audio_transfer",
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_state(state_path, state)

    secret = credentials.gemini_api_key
    assert secret is not None
    annotator: GeminiAnnotator | None = None
    started = time.monotonic()
    try:
        with httpx.Client(timeout=600.0, follow_redirects=True) as client:
            annotator = GeminiAnnotator(
                api_key=secret.get_secret_value(),
                client=client,  # type: ignore[arg-type]
                consent_granted=True,
            )
            outcome = annotator.annotate(
                audio, mime_type="audio/wav", duration_seconds=duration_seconds
            )
    except Exception as error:  # noqa: BLE001 - no retry; the type is the diagnosis
        # The same keys whether or not the annotator was ever constructed: a state file
        # missing "audio_bytes_sent" reads as unknown, and unknown is the one answer this
        # record must never give about whether audio left the machine.
        trace = annotator.trace.as_dict() if annotator else CallTrace().as_dict()
        counts = dict(annotator.ledger.counts() if annotator else CallLedger().counts())
        state.update(
            {
                "state": "failed",
                "reason": type(error).__name__,
                "trace": trace,
                "call_counts": counts,
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "retry_attempted": False,
            }
        )
        print(f"  stage reached      : {trace.get('stage_reached')}")
        print(f"  last http status   : {trace.get('last_http_status')}")
        print(f"  audio bytes sent   : {trace.get('audio_bytes_sent', 0)}")
        _write_state(state_path, state)
        print(f"gemini silver failed: {type(error).__name__}", file=sys.stderr)
        print("  no retry attempted; a second attempt would be a second transmission")
        return 2
    elapsed = time.monotonic() - started

    path, content_digest, summary = write_silver(
        outcome.annotation,
        root=arguments.root,
        conversation_id=conversation_id,
        input_sha256=digest,
        model=outcome.model,
        prompt=PROMPT,
        config={"response_format": "structured-json", "temperature": "unset"},
        remote_file_deleted=outcome.remote_file_deleted,
        deletion_detail=outcome.deletion_detail,
        call_counts=dict(outcome.call_counts),
    )
    artifact_digest = file_sha256(path)

    state.update(
        {
            "state": "completed",
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 3),
            "call_counts": dict(outcome.call_counts),
            "remote_file_deleted": outcome.remote_file_deleted,
            "deletion_detail": outcome.deletion_detail,
            "silver_path": str(path),
            "silver_content_sha256": content_digest,
            "silver_artifact_sha256": artifact_digest,
            "review_state": "review_required",
            "validation": {
                "dropped_turn_count": summary["dropped_turn_count"],
                "dropped_turns_by_rule": summary["dropped_turns_by_rule"],
            },
            "retry_attempted": False,
        }
    )
    _write_state(state_path, state)

    print()
    print(f"call counts           : {dict(outcome.call_counts)}")
    print(f"remote file deleted   : {outcome.remote_file_deleted} ({outcome.deletion_detail})")
    print(f"turns                 : {summary['turn_count']}")
    print(f"speakers              : {summary['speaker_count']}")
    print(f"uncertain turns       : {summary['uncertain_turns']}")
    print(f"mean confidence       : {summary['mean_confidence']}")
    print(f"emotion histogram     : {summary['emotion_histogram']}")
    print(f"transcript characters : {summary['transcript_characters']} (text not shown)")
    print(f"dropped turns         : {summary['dropped_turn_count']}")
    print(f"dropped rules         : {summary['dropped_turns_by_rule']}")
    print("review state          : review_required (not promotable)")
    print(f"silver artifact       : {path}")
    print(f"  content sha256      : {content_digest}")
    print(f"  artifact sha256     : {artifact_digest}")
    print(f"elapsed               : {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
