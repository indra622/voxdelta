"""One bounded Gemini timestamp-only proposal for the suffix of the KCSC Silver draft."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from voxdelta.annotation.alignment import (
    RESPONSE_SCHEMA,
    TARGET_START_POSITION,
    load_source,
    prompt_for_suffix,
    validate_rows,
    write_proposal,
)
from voxdelta.annotation.gemini_silver import GeminiAnnotator
from voxdelta.annotation.store import DEFAULT_ROOT
from voxdelta.credentials import load_credentials
from voxdelta.evaluation.kcsc_diarization_benchmark import file_sha256

CONVERSATION_ID = "A6000_S0005_0"
AUDIO = Path(f"data/derived/kcsc/audio/{CONVERSATION_ID}.wav")
AUDIO_SHA256 = "01565974fdf7021e0d0afcbae8c040839777434f40416f1e265aeee3926cce27"
DURATION_SECONDS = 583.909562
STATE = Path("data/jobs/benchmarks/gemini-alignment-proposal-A6000_S0005_0.state.json")


def _write_state(payload: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if not AUDIO.is_file() or file_sha256(AUDIO) != AUDIO_SHA256:
        print("alignment proposal failed: pinned audio unavailable", file=sys.stderr)
        return 2
    source = load_source(DEFAULT_ROOT, CONVERSATION_ID)
    source_sha = str(source.get("content_sha256", ""))
    state: dict[str, Any] = {
        "job": "gemini-alignment-proposal-A6000_S0005_0",
        "state": "running",
        "conversation_id": CONVERSATION_ID,
        "source_silver_content_sha256": source_sha,
        "target_start_position": TARGET_START_POSITION,
        "audio_sha256": AUDIO_SHA256,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ceilings": {"uploads": 1, "interactions": 1, "deletes": 1, "retries": 0},
    }
    _write_state(state)
    credentials = load_credentials()
    secret = credentials.gemini_api_key
    if secret is None:
        state.update({"state": "failed", "reason": "credentials_missing"})
        _write_state(state)
        return 2
    annotator: GeminiAnnotator | None = None
    try:
        with httpx.Client(timeout=600.0, follow_redirects=True) as client:
            annotator = GeminiAnnotator(
                api_key=secret.get_secret_value(),
                client=client,  # type: ignore[arg-type]
                consent_granted=True,
            )
            remote_uri, remote_name = annotator._upload(AUDIO, "audio/wav")
            try:
                payload = annotator._interact(
                    remote_uri,
                    "audio/wav",
                    prompt=prompt_for_suffix(source, duration_seconds=DURATION_SECONDS),
                    response_schema=RESPONSE_SCHEMA,
                )
                source_turn_count = len(source["content"]["turns"])
                rows, dropped = validate_rows(
                    payload,
                    source_turn_count=source_turn_count,
                    duration_seconds=DURATION_SECONDS,
                )
            finally:
                deleted, detail = annotator._delete(remote_name)
        path, digest = write_proposal(
            root=DEFAULT_ROOT,
            conversation_id=CONVERSATION_ID,
            source_record=source,
            rows=rows,
            dropped_by_rule=dropped,
            model=annotator.provenance.model,
            input_sha256=AUDIO_SHA256,
            remote_file_deleted=deleted,
            deletion_detail=detail,
            call_counts=annotator.ledger.counts(),
        )
    except Exception as error:  # no retry -- another call is another transmission
        trace = annotator.trace.as_dict() if annotator else {}
        counts = (
            annotator.ledger.counts()
            if annotator
            else {"uploads": 0, "interactions": 0, "deletes": 0}
        )
        state.update({
            "state": "failed", "reason": type(error).__name__, "trace": trace,
            "call_counts": counts, "retry_attempted": False,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        })
        _write_state(state)
        print(f"alignment proposal failed: {type(error).__name__}", file=sys.stderr)
        return 2
    state.update({
        "state": "completed", "proposal_path": str(path), "proposal_content_sha256": digest,
        "call_counts": annotator.ledger.counts(), "remote_file_deleted": deleted,
        "deletion_detail": detail, "proposed_row_count": len(rows),
        "dropped_rows_by_rule": dropped, "retry_attempted": False,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    _write_state(state)
    print(f"proposal rows: {len(rows)}")
    print(f"dropped rows: {sum(dropped.values())}")
    print(f"call counts: {annotator.ledger.counts()}")
    print(f"remote deleted: {deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
