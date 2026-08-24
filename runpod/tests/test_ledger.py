from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from voxdelta_runpod.ledger import (
    LedgerError,
    RunIdentity,
    append_transition,
    initialize_ledger,
    load_ledger,
)


def _identity(character: str = "a") -> RunIdentity:
    return RunIdentity(
        config_sha256=character * 64,
        code_sha256="b" * 64,
        container_sha256="c" * 64,
        base_model_sha256="d" * 64,
        archive_sha256="e" * 64,
        manifest_sha256="f" * 64,
        sampler_sha256="0" * 64,
    )


def test_ledger_is_append_only_chained_and_private(tmp_path: Path) -> None:
    root = (tmp_path / "ledger").resolve()
    identity = _identity()
    timestamp = datetime(2026, 8, 24, 6, 0, tzinfo=UTC)

    initialized = initialize_ledger(root, identity, recorded_at=timestamp)
    preflight = append_transition(root, "preflight", identity, recorded_at=timestamp)
    pilot_a = append_transition(root, "pilot-a", identity, report_sha256="1" * 64)

    assert initialized.sequence == 0
    assert preflight.previous_record_sha256 == initialized.digest()
    assert pilot_a.from_stage == "preflight"
    assert [record.to_stage for record in load_ledger(root)] == [
        "initialized",
        "preflight",
        "pilot-a",
    ]
    assert root.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o077 == 0 for path in root.iterdir())


def test_ledger_rejects_illegal_transition_and_identity_change(tmp_path: Path) -> None:
    root = (tmp_path / "ledger").resolve()
    initialize_ledger(root, _identity())

    with pytest.raises(LedgerError, match="^illegal_stage_transition$"):
        append_transition(root, "full", _identity())
    with pytest.raises(LedgerError, match="^ledger_identity_mismatch$"):
        append_transition(root, "preflight", _identity("1"))


def test_ledger_detects_tampered_chain_before_append(tmp_path: Path) -> None:
    root = (tmp_path / "ledger").resolve()
    initialize_ledger(root, _identity())
    append_transition(root, "preflight", _identity())
    path = root / "record-000001.json"
    payload = json.loads(path.read_text())
    payload["previous_record_sha256"] = "9" * 64
    path.write_text(json.dumps(payload))

    with pytest.raises(LedgerError, match="^invalid_ledger$"):
        load_ledger(root)


def test_ledger_refuses_existing_target(tmp_path: Path) -> None:
    root = (tmp_path / "ledger").resolve()
    initialize_ledger(root, _identity())
    with pytest.raises(LedgerError, match="^ledger_exists$"):
        initialize_ledger(root, _identity())
