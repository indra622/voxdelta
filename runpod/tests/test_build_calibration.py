from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest
from voxdelta.evaluation.manifest import DatasetItem

from voxdelta_runpod.package import SanitizedRecord

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_calibration.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("build_calibration", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path) -> tuple[list[DatasetItem], tuple[SanitizedRecord, ...]]:
    items: list[DatasetItem] = []
    records: list[SanitizedRecord] = []
    for index, emotion in enumerate(("neutral", "sadness")):
        item_key = f"{index + 1:064x}"
        payload = f"audio-{index}".encode()
        digest = hashlib.sha256(payload).hexdigest()
        path = tmp_path / f"{item_key}.wav"
        path.write_bytes(payload)
        items.append(
            DatasetItem(
                id=item_key,
                call_id=item_key,
                speaker_id=item_key,
                audio_path=str(path),
                transcript="",
                split="validation",
                source="emotion",
                emotion=emotion,
                sha256=digest,
            )
        )
        records.append(
            SanitizedRecord(
                item_key=item_key,
                audio_path=f"audio/{item_key}.wav",
                split="validation",
                emotion=emotion,
                audio_sha256=digest,
            )
        )
    return items, tuple(records)


def test_validation_identity_binds_item_label_digest_and_audio_bytes(tmp_path: Path) -> None:
    module = _module()
    items, records = _fixture(tmp_path)

    module._validate_validation_identity(items, records)

    changed = list(records)
    changed[0] = changed[0].model_copy(update={"item_key": "f" * 64})
    with pytest.raises(module.CalibrationReleaseError, match="^validation_identity_mismatch$"):
        module._validate_validation_identity(items, tuple(changed))

    Path(items[0].audio_path).write_bytes(b"tampered")
    with pytest.raises(module.CalibrationReleaseError, match="^validation_audio_mismatch$"):
        module._validate_validation_identity(items, records)


def test_cache_rows_are_strictly_bound_to_each_validation_item(tmp_path: Path) -> None:
    module = _module()
    items, _ = _fixture(tmp_path)
    cache = tmp_path / "cache.jsonl"
    rows = []
    for index, item in enumerate(items):
        rows.append(
            {
                "i": index,
                "item_id": item.id,
                "audio_sha256": item.sha256,
                "expected": item.emotion,
                "p": [0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1],
            }
        )
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert len(module._load_cache(cache, items)) == 2

    rows[0]["item_id"] = "f" * 64
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(module.CalibrationReleaseError, match="^calibration_cache_out_of_step$"):
        module._load_cache(cache, items)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission modes are required")
def test_cache_rejects_group_or_world_readable_permissions(tmp_path: Path) -> None:
    module = _module()
    items, _ = _fixture(tmp_path)
    item = items[0]
    cache = tmp_path / "cache.jsonl"
    cache.write_text(
        json.dumps(
            {
                "i": 0,
                "item_id": item.id,
                "audio_sha256": item.sha256,
                "expected": item.emotion,
                "p": [0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1],
            }
        )
        + "\n"
    )
    cache.chmod(0o644)

    with pytest.raises(module.CalibrationReleaseError, match="^calibration_cache_not_private$"):
        module._load_cache(cache, items)


@pytest.mark.parametrize(
    "probabilities",
    (
        [0.1] * 7,
        [0.1, 0.1, 0.1, 0.1, 0.4, 0.1, -0.1],
        [0.1, 0.1, 0.1, 0.1, 0.4, 0.1, float("nan")],
    ),
)
def test_cache_rejects_non_distribution_rows(tmp_path: Path, probabilities: list[float]) -> None:
    module = _module()
    items, _ = _fixture(tmp_path)
    item = items[0]
    cache = tmp_path / "cache.jsonl"
    cache.write_text(
        json.dumps(
            {
                "i": 0,
                "item_id": item.id,
                "audio_sha256": item.sha256,
                "expected": item.emotion,
                "p": probabilities,
            }
        )
        + "\n"
    )

    with pytest.raises(module.CalibrationReleaseError, match="^calibration_cache_out_of_step$"):
        module._load_cache(cache, items)
