from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

import voxdelta.evaluation.emotion_training as emotion_training
from voxdelta.api.app import create_app
from voxdelta.api.dependencies import (
    ProviderConfigurationError,
    ProviderFactories,
    build_dependencies,
)
from voxdelta.config import Settings
from voxdelta.credentials import Credentials
from voxdelta.domain.models import ProviderProvenance, StageName
from voxdelta.evaluation.emotion_training import load_training_examples
from voxdelta.evaluation.manifest import DatasetItem, load_manifest
from voxdelta.pipeline.stages import cache_key_for_stage
from voxdelta.providers.base import ProviderError
from voxdelta.providers.checkpoints import checkpoint_tree_digest
from voxdelta.providers.faster_whisper_asr import FasterWhisperProvider
from voxdelta.providers.qwen3_asr import Qwen3AsrProvider

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_65s.wav"
TOKEN = "integration-capability-token-with-enough-entropy"


def _item(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": "item-1",
        "call_id": "call-1",
        "speaker_id": "speaker-1",
        "audio_path": "/private/audio.wav",
        "transcript": "private transcript sentinel",
        "split": "train",
        "source": "emotion",
        "emotion": "neutral",
        "start": 0.0,
        "end": 1.0,
        "sha256": "a" * 64,
    }
    value.update(updates)
    return value


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (-0.1, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (None, 1.0),
        (0.0, None),
    ],
)
def test_manifest_intervals_are_strict_finite_complete_and_ordered(
    start: float | None, end: float | None
) -> None:
    with pytest.raises(ValidationError):
        DatasetItem.model_validate(_item(start=start, end=end))


def test_manifest_json_constants_are_rejected_without_record_values(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(_item()).replace("0.0", "NaN", 1) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        load_manifest(manifest)

    message = str(raised.value)
    assert "private transcript sentinel" not in message
    assert "/private/audio.wav" not in message


@pytest.mark.parametrize("leaked_field", ["call_id", "speaker_id"])
def test_training_rejects_split_leakage_before_opening_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leaked_field: str,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    records = [
        _item(audio_path=str(tmp_path / "must-not-open-a.wav")),
        _item(
            id="item-2",
            split="validation",
            **({"call_id": "call-2"} if leaked_field == "speaker_id" else {}),
            **({"speaker_id": "speaker-2"} if leaked_field == "call_id" else {}),
            audio_path=str(tmp_path / "must-not-open-b.wav"),
        ),
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    monkeypatch.setattr(
        emotion_training,
        "read_trusted_regular_file",
        lambda _path: pytest.fail("training admission touched audio before split validation"),
    )

    with pytest.raises(ValueError, match="^invalid_training_manifest$") as raised:
        load_training_examples(manifest)

    assert "must-not-open" not in str(raised.value)


def test_provider_revision_changes_stage_cache_key() -> None:
    first = ProviderProvenance(name="local", model="model", remote=False, revision="a" * 64)
    second = ProviderProvenance(name="local", model="model", remote=False, revision="b" * 64)

    assert cache_key_for_stage(StageName.DIARIZE, (), first, {}) != cache_key_for_stage(
        StageName.DIARIZE, (), second, {}
    )


def test_checkpoint_digest_uses_relative_paths_and_content(tmp_path: Path) -> None:
    first = tmp_path / "first" / "same-name"
    second = tmp_path / "second" / "same-name"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "config.json").write_text("one", encoding="utf-8")
    (second / "config.json").write_text("two", encoding="utf-8")

    assert checkpoint_tree_digest(first) != checkpoint_tree_digest(second)


@pytest.mark.skipif(os.name != "posix", reason="POSIX special files are required")
def test_checkpoint_digest_rejects_symlinks_and_special_files(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    target = root / "target"
    target.write_text("weights", encoding="utf-8")
    (root / "link").symlink_to(target)
    with pytest.raises(ValueError, match="invalid_local_checkpoint"):
        checkpoint_tree_digest(root)
    (root / "link").unlink()
    os.mkfifo(root / "fifo")
    with pytest.raises(ValueError, match="invalid_local_checkpoint"):
        checkpoint_tree_digest(root)


def test_asr_replacement_unloads_previous_before_factory_and_failure_leaves_none_active() -> None:
    events: list[str] = []

    class Model:
        def close(self) -> None:
            events.append("faster-close")

    def faster_factory(model_id: str, *, device: str, compute_type: str) -> Model:
        del model_id, device, compute_type
        events.append("faster-load")
        return Model()

    def failing_qwen_factory(model_id: str, *, aligner_id: str, device: str, dtype: str) -> object:
        del model_id, aligner_id, device, dtype
        events.append("qwen-load")
        raise RuntimeError("private factory failure")

    faster = FasterWhisperProvider(model_factory=faster_factory)
    faster._load()
    qwen = Qwen3AsrProvider(
        model_factory=failing_qwen_factory,
        hardware_probe=lambda: (False, False),
    )

    with pytest.raises(ProviderError, match="unavailable"):
        qwen._load()

    assert events == ["faster-load", "faster-close", "qwen-load"]
    faster._load()
    assert events[-1] == "faster-load"
    faster.unload()


def test_missing_local_provider_configuration_fails_with_fixed_safe_error(tmp_path: Path) -> None:
    private = tmp_path / "private-checkpoint-sentinel"
    settings = Settings(
        data_root=tmp_path / "data",
        database_path=tmp_path / "data" / "db.sqlite3",
        emotion_provider="wav2vec",
        emotion_checkpoint_path=private,
    )

    with pytest.raises(ProviderConfigurationError) as raised:
        build_dependencies(settings, credentials=Credentials())

    assert str(raised.value) == "provider_configuration_invalid"
    assert str(private) not in str(raised.value)


def test_selected_real_adapter_boundaries_run_without_weights_or_network(tmp_path: Path) -> None:
    class Segment:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class Annotation:
        def itertracks(self, *, yield_label: bool):  # type: ignore[no-untyped-def]
            assert yield_label is True
            for index in range(6):
                yield Segment(index * 10.0, (index + 1) * 10.0), None, f"raw-{index % 2}"

    class DiarizationOutput:
        speaker_diarization = Annotation()
        exclusive_speaker_diarization = Annotation()

    class Pipeline:
        def __call__(self, audio_path: str, **kwargs: int) -> DiarizationOutput:
            del audio_path, kwargs
            return DiarizationOutput()

    def pipeline_factory(model: str, *, token: str | None) -> Pipeline:
        del model
        assert token is None
        return Pipeline()

    class Word:
        def __init__(self, index: int) -> None:
            self.start = index * 10.0 + 1.0
            self.end = index * 10.0 + 4.0
            self.word = f" 발화{index}"

    class WhisperSegment:
        words = [Word(index) for index in range(6)]

    class WhisperModel:
        def transcribe(self, path: str, **kwargs: object):  # type: ignore[no-untyped-def]
            del path, kwargs
            return [WhisperSegment()], object()

    def whisper_factory(model_id: str, *, device: str, compute_type: str) -> WhisperModel:
        del model_id, device, compute_type
        return WhisperModel()

    checkpoint = tmp_path / "community"
    checkpoint.mkdir()
    (checkpoint / "config.yaml").write_text("pipeline: {}\n", encoding="utf-8")
    settings = Settings(
        data_root=tmp_path / "data",
        database_path=tmp_path / "data" / "db.sqlite3",
        api_capability_token=SecretStr(TOKEN),
        diarization_provider="pyannote-community",
        pyannote_checkpoint_path=checkpoint,
        asr_provider="faster-whisper",
    )
    dependencies = build_dependencies(
        settings,
        credentials=Credentials(),
        provider_factories=ProviderFactories(
            pyannote_pipeline=pipeline_factory,
            faster_whisper_model=whisper_factory,
        ),
    )
    job_id = dependencies.repository.create_job(str(FIXTURE))

    paused = dependencies.runner.run_until_pause(job_id)

    assert paused["status"] == "paused"
    app = create_app(
        repository=dependencies.repository,
        artifacts=dependencies.artifacts,
        runner=dependencies.runner,
        api_capability_token=SecretStr(TOKEN),
    )
    with TestClient(
        app,
        base_url="http://localhost",
        headers={"X-VoxDelta-Token": TOKEN},
    ) as client:
        response = client.get("/api/config/providers")
    assert response.status_code == 200
    serialized = response.text
    assert "pyannote" in serialized
    assert "faster-whisper" in serialized
    assert str(checkpoint) not in serialized
    assert "token" not in serialized.casefold()
