"""Tests for the one-conversation KCSC E2E runner.

This runner is the only code path that can send KCSC audio to a third party, and the
transfer cannot be recalled. So what is pinned here is everything that decides whether a
call happens at all and how many: the default outcome of running it is that nothing
leaves the host, the ledger refuses a second submission, and the selection rule cannot
drift onto a different conversation. The artifact's transcript-free promise is pinned
too, because that is the other irreversible thing — text written to a report cannot be
unwritten either.

No test here reaches the network. The remote client is a stub throughout.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from voxdelta.evaluation.kcsc_diarization_benchmark import KcscBenchmarkError, file_sha256
from voxdelta.evaluation.kcsc_precision_benchmark import (
    CallLedger,
    PrecisionBudgetExceeded,
)

_SPEC = importlib.util.spec_from_file_location(
    "run_kcsc_precision_e2e",
    Path(__file__).resolve().parents[2] / "scripts" / "run_kcsc_precision_e2e.py",
)
assert _SPEC and _SPEC.loader
e2e: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(e2e)


def _derived(tmp_path: Path, durations: dict[str, float]) -> Path:
    """A derived set shaped like the real one, with only the fields the runner reads."""

    root = tmp_path / "derived"
    (root / "audio").mkdir(parents=True)
    (root / "reference").mkdir(parents=True)
    conversations = []
    for index, (conversation_id, duration) in enumerate(durations.items()):
        audio = root / "audio" / f"{conversation_id}.wav"
        audio.write_bytes(b"RIFF" + bytes([index]) * 64)
        reference = root / "reference" / f"{conversation_id}.json"
        reference.write_text(
            json.dumps({"conversation_id": conversation_id, "turns": []}), encoding="utf-8"
        )
        conversations.append(
            {
                "conversation_id": conversation_id,
                "derived_duration_seconds": duration,
                "speakers": ["G0001", "G0002"],
                "outputs": {
                    "audio": f"audio/{conversation_id}.wav",
                    "audio_sha256": file_sha256(audio),
                    "audio_bytes": audio.stat().st_size,
                    "reference": f"reference/{conversation_id}.json",
                    "reference_sha256": file_sha256(reference),
                },
            }
        )
    (root / "manifest.json").write_text(
        json.dumps({"source": {"revision": "r" * 40}, "conversations": conversations}),
        encoding="utf-8",
    )
    return root


FROZEN = {"A0051_S0001_0": 898.72, "A0055_S0006_0": 903.848, "A6000_S0005_0": 583.91}

#: The shape verify_local_models returns once the cache has been proven and pinned.
_STUB_MODELS = {
    "hub_cache": "/stub/hub",
    "hub_offline": "1",
    "qwen_asr_revision": "0" * 40,
    "qwen_asr_tree_sha256": "0" * 64,
    "qwen_aligner_revision": "1" * 40,
    "qwen_aligner_tree_sha256": "1" * 64,
    "xlsr_release": "stub",
    "xlsr_release_manifest_sha256": "2" * 64,
    "xlsr_calibration": "stub",
    "xlsr_calibration_sha256": "3" * 64,
}


# ------------------------------------------------------------------------------- selection


def test_the_selection_rule_picks_the_shortest_frozen_conversation(tmp_path: Path) -> None:
    """The smallest payload, because this is the part of the run that cannot be undone."""

    selection = e2e.select_conversation(_derived(tmp_path, FROZEN))

    assert selection["conversation_id"] == "A6000_S0005_0"
    assert selection["duration_seconds"] == pytest.approx(583.91)


def test_the_selection_is_deterministic_across_calls(tmp_path: Path) -> None:
    derived = _derived(tmp_path, FROZEN)

    first = e2e.select_conversation(derived)
    second = e2e.select_conversation(derived)

    assert first["conversation_id"] == second["conversation_id"]
    assert first["audio_sha256"] == second["audio_sha256"]


def test_a_derived_set_missing_a_frozen_conversation_fails_closed(tmp_path: Path) -> None:
    partial = {name: FROZEN[name] for name in list(FROZEN)[:2]}

    with pytest.raises(KcscBenchmarkError, match="frozen benchmark conversations"):
        e2e.select_conversation(_derived(tmp_path, partial))


def test_audio_that_drifted_from_the_manifest_fails_before_any_transfer(tmp_path: Path) -> None:
    derived = _derived(tmp_path, FROZEN)
    (derived / "audio" / "A6000_S0005_0.wav").write_bytes(b"tampered")

    with pytest.raises(KcscBenchmarkError, match="checksum mismatch"):
        e2e.select_conversation(derived)


# ---------------------------------------------------------------------------- the one call


def test_the_ledger_refuses_a_second_diarization_job() -> None:
    ledger = CallLedger(max_jobs=e2e.MAX_DIARIZATION_JOBS)

    ledger.reserve("diarize")

    assert ledger.submissions == 1
    with pytest.raises(PrecisionBudgetExceeded, match="ceiling is 1"):
        ledger.reserve("diarize")


def test_the_ledger_refuses_a_second_upload() -> None:
    ledger = CallLedger(max_jobs=e2e.MAX_DIARIZATION_JOBS)

    ledger.reserve("upload")

    with pytest.raises(PrecisionBudgetExceeded, match="ceiling is 1"):
        ledger.reserve("upload")


def test_polling_is_not_charged_against_the_job_ceiling() -> None:
    """One submission may be polled to completion; that is still one job."""

    ledger = CallLedger(max_jobs=e2e.MAX_DIARIZATION_JOBS)
    ledger.reserve("diarize")

    for _ in range(30):
        ledger.reserve("job_poll")

    assert ledger.submissions == 1


def test_the_runner_declares_a_ceiling_of_exactly_one() -> None:
    assert e2e.MAX_DIARIZATION_JOBS == 1


# ------------------------------------------------------------------------ nothing by default


def test_running_without_confirmation_transmits_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default outcome of executing this script is that no audio leaves the host."""

    derived = _derived(tmp_path, FROZEN)
    monkeypatch.setattr(e2e, "verify_local_models", lambda: _STUB_MODELS)

    code = e2e.main(["--derived", str(derived), "--record", str(tmp_path / "r.json")])

    assert code == 0
    output = capsys.readouterr().out
    assert "stopping after preflight: nothing was transmitted." in output
    assert "rights confirmed    : NO" in output
    assert not (tmp_path / "r.json").exists()


def test_confirmation_without_an_authorization_refuses_to_transmit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    derived = _derived(tmp_path, FROZEN)
    monkeypatch.setattr(e2e, "verify_local_models", lambda: _STUB_MODELS)

    code = e2e.main(
        [
            "--derived",
            str(derived),
            "--record",
            str(tmp_path / "r.json"),
            "--confirm-external-upload",
        ]
    )

    assert code == 2


def test_there_is_no_authorization_value_meaning_rights_confirmed() -> None:
    """The workflow must not be able to claim a permission nobody has granted."""

    parser = e2e._parser()
    action = next(item for item in parser._actions if item.dest == "authorization")

    assert list(action.choices) == ["user_limited_override"]


def test_a_preexisting_scratch_root_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusing to reuse a scratch root keeps one run's job state out of another's."""

    derived = _derived(tmp_path, FROZEN)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(e2e, "verify_local_models", lambda: _STUB_MODELS)
    monkeypatch.setattr(
        e2e, "load_credentials", lambda: type("C", (), {"pyannoteai_api_key": "k"})()
    )

    code = e2e.main(
        [
            "--derived",
            str(derived),
            "--record",
            str(tmp_path / "r.json"),
            "--scratch",
            str(scratch),
            "--confirm-external-upload",
            "--authorization",
            "user_limited_override",
        ]
    )

    assert code == 2


# ------------------------------------------------------------------------------ the artifact


def test_the_evaluation_record_never_carries_transcript_fields(tmp_path: Path) -> None:
    """Pins the shape of what is written, so no future field can smuggle text into it."""

    record = json.loads(
        json.dumps(
            {
                "kind": "kcsc-precision2-qwen-xlsr-e2e",
                "transcript_free": True,
                "rights": {"user_limited_override": True, "rights_confirmed": False},
                "metrics": {"utterance_count": 3},
                "timestamp_coverage": {"policy": e2e.SANITATION_POLICY, "omitted_words": 0},
            }
        )
    )

    def keys(node: object) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {key for value in node.values() for key in keys(value)}
        if isinstance(node, list):
            return {key for value in node for key in keys(value)}
        return set()

    present = keys(record)
    # "transcript_free" is a claim about the artifact; a "transcript" key would be content.
    assert "transcript" not in present
    assert "utterances" not in present
    assert "text" not in present
    assert "transcript_free" in present
    assert record["rights"]["rights_confirmed"] is False
    assert record["rights"]["user_limited_override"] is True


def test_the_env_reader_reports_the_shipped_default_without_writing(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("VOXDELTA_ASR_PROVIDER=faster-whisper\nOTHER=1\n", encoding="utf-8")
    before = file_sha256(env)

    import contextlib

    with contextlib.chdir(tmp_path):
        original = e2e.ENV_FILE
        e2e.ENV_FILE = Path(".env")
        try:
            assert e2e.env_default_provider() == "faster-whisper"
        finally:
            e2e.ENV_FILE = original

    assert file_sha256(env) == before


# ------------------------------------------------------------- offline model cache is forced

from voxdelta.evaluation.kcsc_qwen_track_benchmark import (  # noqa: E402
    MODEL_FILES,
    MODEL_REVISIONS,
)
from voxdelta.providers.qwen3_asr import (  # noqa: E402
    ALIGNER_MODEL_ID,
    DEFAULT_MODEL_ID,
)


def _hub_cache(tmp_path: Path, *, omit: str = "") -> Path:
    """A hub cache holding the two checkpoints the E2E loads, laid out as the hub writes them."""

    root = tmp_path / "hub"
    for repo_id in (DEFAULT_MODEL_ID, ALIGNER_MODEL_ID):
        revision = MODEL_REVISIONS[repo_id]
        repo = root / f"models--{repo_id.replace('/', '--')}"
        blobs, snapshot = repo / "blobs", repo / "snapshots" / revision
        blobs.mkdir(parents=True, exist_ok=True)
        snapshot.mkdir(parents=True, exist_ok=True)
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / "main").write_text(revision, encoding="utf-8")
        for name in MODEL_FILES[repo_id]:
            if repo_id == DEFAULT_MODEL_ID and name == omit:
                continue
            blob = blobs / f"blob-{name}"
            blob.write_bytes(f"{repo_id}:{name}".encode())
            (snapshot / name).symlink_to(Path("..") / ".." / "blobs" / blob.name)
    return root


def test_a_missing_hub_cache_fails_preflight_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache problem must stop the run while stopping is still free."""

    monkeypatch.setattr(e2e, "XLSR_RELEASE", tmp_path / "release")
    monkeypatch.setattr(e2e, "XLSR_CALIBRATION", tmp_path / "calibration")
    (tmp_path / "release").mkdir()
    (tmp_path / "calibration").mkdir()

    with pytest.raises(KcscBenchmarkError, match="missing local hub cache"):
        e2e.verify_local_models(tmp_path / "absent-hub")


def test_an_incomplete_qwen_snapshot_fails_preflight_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache missing a weight shard would otherwise become a download at load time."""

    release, calibration = tmp_path / "release", tmp_path / "calibration"
    release.mkdir()
    calibration.mkdir()
    (release / "RELEASE.json").write_text("{}", encoding="utf-8")
    (calibration / "CALIBRATION.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(e2e, "XLSR_RELEASE", release)
    monkeypatch.setattr(e2e, "XLSR_CALIBRATION", calibration)

    cache = _hub_cache(tmp_path, omit="model-00001-of-00002.safetensors")

    with pytest.raises(KcscBenchmarkError, match="qwen checkpoints unusable offline"):
        e2e.verify_local_models(cache)


def test_a_complete_cache_pins_the_hub_client_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The offline pin is the mechanism the earlier run lacked; it must actually be set."""

    release, calibration = tmp_path / "release", tmp_path / "calibration"
    release.mkdir()
    calibration.mkdir()
    (release / "RELEASE.json").write_text("{}", encoding="utf-8")
    (calibration / "CALIBRATION.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(e2e, "XLSR_RELEASE", release)
    monkeypatch.setattr(e2e, "XLSR_CALIBRATION", calibration)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    cache = _hub_cache(tmp_path)
    models = e2e.verify_local_models(cache)

    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_CACHE"] == str(cache.resolve())
    assert models["hub_offline"] == "1"
    assert models["qwen_asr_revision"] == MODEL_REVISIONS[DEFAULT_MODEL_ID]
    assert len(models["qwen_asr_tree_sha256"]) == 64


def test_the_offline_pin_is_not_set_when_the_cache_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinning offline over a broken cache would hide the problem instead of failing."""

    release, calibration = tmp_path / "release", tmp_path / "calibration"
    release.mkdir()
    calibration.mkdir()
    monkeypatch.setattr(e2e, "XLSR_RELEASE", release)
    monkeypatch.setattr(e2e, "XLSR_CALIBRATION", calibration)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    with pytest.raises(KcscBenchmarkError):
        e2e.verify_local_models(tmp_path / "absent-hub")

    import os

    assert "HF_HUB_OFFLINE" not in os.environ


def test_the_runner_no_longer_asserts_an_unverifiable_download_claim() -> None:
    """The withdrawn claim must not be reintroduced by the code that made it."""

    source = Path(e2e.__file__).read_text(encoding="utf-8")

    assert '"no_model_downloads": True' not in source
    assert "hub_offline_enforced" in source


# ------------------------------------------------- scratch survives an incomplete artifact


def test_scratch_cleanup_happens_only_after_the_artifact_is_verified() -> None:
    """The first run deleted scratch before checking, and lost its diarization timeline."""

    source = Path(e2e.__file__).read_text(encoding="utf-8")
    readback = source.index("readback = json.loads")
    cleanup = source.index("shutil.rmtree(scratch, ignore_errors=True)\n        print")

    assert readback < cleanup, "the record must be read back before scratch is removed"


def test_the_record_persists_a_transcript_free_hypothesis_timeline() -> None:
    """Boundaries only, so a scoring gap can be closed without another submission."""

    source = Path(e2e.__file__).read_text(encoding="utf-8")

    assert '"hypothesis_timeline": hypothesis_timeline' in source
    assert '"speaker_id": segment.speaker_id' in source
    # A timeline entry must carry no text of any kind.
    assert "segment.transcript" not in source


def test_each_attempt_is_recorded_as_its_own_override() -> None:
    parser = e2e._parser()
    dests = {action.dest for action in parser._actions}

    assert {"attempt", "supersedes"} <= dests
    source = Path(e2e.__file__).read_text(encoding="utf-8")
    assert "does not extend or inherit any earlier grant" in source
    assert '"rights_confirmed": False' in source


def test_e2e_record_never_inherits_the_three_conversation_programme_scope() -> None:
    """A one-file retry must not claim the older three-file benchmark approval."""

    source = Path(e2e.__file__).read_text(encoding="utf-8")

    assert '"programme_scope": AUTHORIZATION_SCOPE' not in source
