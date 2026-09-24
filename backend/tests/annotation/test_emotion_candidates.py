from __future__ import annotations

import hashlib
import json
from pathlib import Path

from voxdelta.annotation.emotion_candidates import FILENAME, KIND, load_local_candidate

CONVERSATION = "A6000_S0005_0"
SOURCE_DIGEST = "a" * 64


def test_local_candidate_is_bound_to_its_silver_and_read_only(tmp_path: Path) -> None:
    directory = tmp_path / CONVERSATION
    directory.mkdir()
    payload = {
        "schema_version": "1",
        "kind": KIND,
        "source": {"silver_content_sha256": SOURCE_DIGEST},
        "content": {
            "turns": [
                {
                    "start": 1.0,
                    "end": 2.0,
                    "speaker": "SPEAKER_00",
                    "transcript": "private",
                    "emotion": "anger",
                    "emotion_rationale": "private",
                    "confidence": 0.7,
                }
            ]
        },
    }
    raw = json.dumps(
        payload["content"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    payload["content_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    path = directory / FILENAME
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    before = path.read_bytes()

    candidate = load_local_candidate(
        annotation_root=tmp_path,
        conversation_id=CONVERSATION,
        source_silver_content_sha256=SOURCE_DIGEST,
    )

    assert candidate is not None
    row = candidate["content"]["turns"][0]
    assert (row["start"], row["end"], row["speaker"], row["transcript"]) == (
        1.0,
        2.0,
        "SPEAKER_00",
        "private",
    )
    assert path.read_bytes() == before
    assert (
        load_local_candidate(
            annotation_root=tmp_path,
            conversation_id=CONVERSATION,
            source_silver_content_sha256="b" * 64,
        )
        is None
    )
