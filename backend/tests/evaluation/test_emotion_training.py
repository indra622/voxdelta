from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest
from pydantic import ValidationError

LABELS = ("happiness", "anger", "disgust", "fear", "neutral", "sadness", "surprise")


def _wav(path: Path, seconds: float, *, channels: int = 1, rate: int = 16_000) -> Path:
    frames = max(1, round(seconds * rate))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(struct.pack(f"<{frames * channels}h", *([100] * frames * channels)))
    return path


def test_profiles_are_strict_and_exact() -> None:
    from voxdelta.evaluation.emotion_training import (
        Emotion2VecTrainingProfile,
        Wav2VecTrainingProfile,
    )

    wav = Wav2VecTrainingProfile()
    assert wav.model_id == "facebook/wav2vec2-xls-r-300m"
    assert wav.seed == 622
    assert (wav.learning_rate, wav.train_batch_size, wav.eval_batch_size) == (2e-5, 8, 8)
    assert (wav.gradient_accumulation_steps, wav.epochs, wav.warmup_ratio) == (2, 10, 0.1)
    assert (wav.evaluation_strategy, wav.selection_metric, wav.early_stopping_patience) == (
        "epoch",
        "macro_f1",
        2,
    )
    modern = Emotion2VecTrainingProfile()
    assert modern.encoder_id == "iic/emotion2vec_plus_large"
    assert modern.encoder_revision == "v2.0.5"
    assert modern.freeze_encoder is True
    assert (modern.hidden_size, modern.dropout, modern.learning_rate) == (256, 0.1, 1e-3)
    assert (modern.batch_size, modern.epochs, modern.early_stopping_patience) == (64, 20, 3)
    assert modern.trainable_components == ("classifier_head",)

    for model, field, value in [
        (Wav2VecTrainingProfile, "seed", "622"),
        (Wav2VecTrainingProfile, "warmup_ratio", True),
        (Wav2VecTrainingProfile, "learning_rate", 3e-5),
        (Wav2VecTrainingProfile, "epochs", 9),
        (Emotion2VecTrainingProfile, "freeze_encoder", 1),
        (Emotion2VecTrainingProfile, "hidden_size", 128),
        (Emotion2VecTrainingProfile, "epochs", 19),
    ]:
        with pytest.raises(ValidationError):
            model.model_validate({field: value})


def test_audio_preprocessing_rejects_wrong_format_and_short_clips(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import load_audio

    for path in (
        _wav(tmp_path / "stereo.wav", 1, channels=2),
        _wav(tmp_path / "rate.wav", 1, rate=8_000),
        _wav(tmp_path / "short.wav", 0.4999375),
    ):
        with pytest.raises(ValueError, match="invalid_audio"):
            load_audio(path)


def test_training_center_crop_and_evaluation_windows_are_deterministic(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import (
        evaluation_logit_mean,
        evaluation_windows,
        load_audio,
        training_clip,
    )

    clip = load_audio(_wav(tmp_path / "long.wav", 45))
    cropped = training_clip(clip)
    windows = evaluation_windows(clip)

    assert len(cropped.samples) == 320_000
    assert cropped.samples == clip.samples[200_000:520_000]
    assert [len(window.samples) for window in windows] == [320_000, 320_000, 80_000]
    assert evaluation_windows(clip) == windows
    assert evaluation_logit_mean(((1.0, 3.0), (3.0, 1.0))) == (2.0, 2.0)
    with pytest.raises(ValueError, match="invalid_evaluation_logits"):
        evaluation_logit_mean(((1.0, float("nan")),))


def test_embedding_cache_key_is_collision_resistant_and_cache_fails_closed(
    tmp_path: Path,
) -> None:
    from voxdelta.evaluation.emotion_training import EmbeddingCache, embedding_cache_key

    base = embedding_cache_key("a" * 64, "iic/emotion2vec_plus_large", "v1")
    assert base != embedding_cache_key("b" * 64, "iic/emotion2vec_plus_large", "v1")
    assert base != embedding_cache_key("a" * 64, "other", "v1")
    assert base != embedding_cache_key("a" * 64, "iic/emotion2vec_plus_large", "v2")
    assert base != embedding_cache_key(
        "a" * 64, "iic/emotion2vec_plus_large@v2.0.5#" + "b" * 64, "v1"
    )
    with pytest.raises(ValueError):
        embedding_cache_key("not-sha", "encoder", "v1")

    cache = EmbeddingCache(tmp_path / "cache")
    cache.store(base, (1.0, 2.0, 3.0))
    assert cache.load(base) == (1.0, 2.0, 3.0)
    cache.path_for(base).write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_embedding_cache"):
        cache.load(base)


def test_default_embedding_cache_root_is_scoped_to_current_user_home() -> None:
    from voxdelta.evaluation.emotion_training import default_embedding_cache_root

    root = default_embedding_cache_root()

    assert root == Path.home() / ".cache" / "voxdelta" / "emotion2vec-v1"


def test_embedding_cache_rejects_symlinked_ancestors(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import EmbeddingCache

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="invalid_embedding_cache"):
        EmbeddingCache(linked / "cache")


def test_embedding_cache_uses_memory_without_secure_directory_primitives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxdelta.evaluation import emotion_training

    monkeypatch.setattr(emotion_training, "_secure_persistent_cache_supported", lambda: False)
    parent = tmp_path / "parent"
    parent.mkdir()
    root = parent / "cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    cache = emotion_training.EmbeddingCache(root)

    root.symlink_to(outside, target_is_directory=True)
    cache.store("a" * 64, (1.0, 2.0))

    assert cache.load("a" * 64) == (1.0, 2.0)
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and mode contract")
def test_embedding_cache_rejects_public_or_foreign_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxdelta.evaluation import emotion_training

    public = tmp_path / "public-cache"
    public.mkdir(mode=0o700)
    public.chmod(0o777)
    with pytest.raises(ValueError, match="invalid_embedding_cache"):
        emotion_training.EmbeddingCache(public)

    private = tmp_path / "private-cache"
    private.mkdir(mode=0o700)
    owner = private.stat().st_uid
    monkeypatch.setattr(emotion_training.os, "geteuid", lambda: owner + 1)
    with pytest.raises(ValueError, match="invalid_embedding_cache"):
        emotion_training.EmbeddingCache(private)


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and mode contract")
def test_embedding_cache_rejects_public_poisoned_entry_with_valid_checksum(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import EmbeddingCache

    root = tmp_path / "cache"
    cache = EmbeddingCache(root)
    key = "a" * 64
    values = [1.0, 2.0]
    canonical = json.dumps(values, separators=(",", ":")).encode()
    cache.path_for(key).write_text(
        json.dumps(
            {"sha256": hashlib.sha256(canonical).hexdigest(), "values": values},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    cache.path_for(key).chmod(0o777)

    with pytest.raises(ValueError, match="invalid_embedding_cache"):
        cache.load(key)
    with pytest.raises(ValueError, match="invalid_embedding_cache"):
        cache.store(key, (3.0, 4.0))


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_embedding_cache_open_failure_does_not_close_negative_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxdelta.evaluation import emotion_training

    closed: list[int] = []

    def fail_open(*_args: object, **_kwargs: object) -> int:
        raise OSError("private path")

    monkeypatch.setattr(emotion_training, "_secure_persistent_cache_supported", lambda: True)
    monkeypatch.setattr(emotion_training.os, "open", fail_open)
    monkeypatch.setattr(emotion_training.os, "close", closed.append)

    with pytest.raises(ValueError, match="invalid_embedding_cache"):
        emotion_training.EmbeddingCache(tmp_path / "cache")
    assert closed == []


@pytest.mark.parametrize("swap", ["root", "ancestor"])
def test_embedding_cache_fails_closed_after_directory_swap(tmp_path: Path, swap: str) -> None:
    from voxdelta.evaluation.emotion_training import EmbeddingCache

    parent = tmp_path / "parent"
    root = parent / "cache"
    parent.mkdir()
    cache = EmbeddingCache(root)
    outside = tmp_path / "outside"
    outside.mkdir()

    if swap == "root":
        original = parent / "original-cache"
        root.rename(original)
        root.symlink_to(outside, target_is_directory=True)
    else:
        original = tmp_path / "original-parent"
        parent.rename(original)
        parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="invalid_embedding_cache"):
        cache.store("a" * 64, (1.0, 2.0))
    assert list(outside.iterdir()) == []


def test_checkpoint_publication_is_atomic_complete_and_hashes_validation(
    tmp_path: Path,
) -> None:
    from voxdelta.evaluation.emotion_training import CheckpointPayload, publish_checkpoint

    output = tmp_path / "model"
    payload = CheckpointPayload(
        architecture="emotion2vec-plus",
        model_id="iic/emotion2vec_plus_large",
        weights=b"weights",
        metrics={"macro_f1": 0.75},
        validation_hash="a" * 64,
        encoder_hash="b" * 64,
        encoder_revision="v2.0.5",
        freeze_encoder=True,
        embedding_size=2,
    )
    publish_checkpoint(output, payload)

    assert sorted(item.name for item in output.iterdir()) == [
        "config.json",
        "label_mapping.json",
        "metrics.json",
        "model.safetensors",
    ]
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["labels"] == list(LABELS)
    assert config["encoder_hash"] == "b" * 64
    assert config["encoder_revision"] == "v2.0.5"
    assert config["embedding_size"] == 2
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["validation_hash"] == "a" * 64


def test_checkpoint_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import CheckpointPayload, publish_checkpoint

    output = tmp_path / "model"
    payload = CheckpointPayload(
        architecture="wav2vec-xls-r",
        model_id="facebook/wav2vec2-xls-r-300m",
        weights=b"weights",
        metrics={"macro_f1": 0.7},
        validation_hash="a" * 64,
    )

    def fail(_path: Path, _data: bytes) -> None:
        raise OSError("/private/transcript-secret")

    with pytest.raises(ValueError, match="checkpoint_publication_failed"):
        publish_checkpoint(output, payload, writer=fail)
    assert not output.exists()


def test_checkpoint_publication_rejects_symlinked_parent(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import CheckpointPayload, publish_checkpoint

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    payload = CheckpointPayload(
        architecture="wav2vec-xls-r",
        model_id="facebook/wav2vec2-xls-r-300m",
        weights=b"weights",
        metrics={"macro_f1": 0.7},
        validation_hash="a" * 64,
    )

    with pytest.raises(ValueError, match="checkpoint_publication_failed"):
        publish_checkpoint(linked / "model", payload)
    assert not (outside / "model").exists()


def test_synthetic_twenty_example_smoke_is_deterministic_and_never_reads_transcript(
    tmp_path: Path,
) -> None:
    from voxdelta.evaluation.emotion_training import smoke_train

    examples = []
    for index in range(20):
        path = _wav(tmp_path / f"{index}.wav", 0.5)
        examples.append(
            {
                "id": f"item-{index}",
                "audio_path": path,
                "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "label": LABELS[index % 7],
                "transcript": f"transcript-secret-{index}",
            }
        )

    first = smoke_train(examples, encoder=lambda clip: (sum(clip.samples), len(clip.samples)))
    second = smoke_train(
        list(reversed(examples)), encoder=lambda clip: (sum(clip.samples), len(clip.samples))
    )

    assert first == second
    assert first.sample_count == 20
    assert 0 <= first.macro_f1 <= 1
    assert "transcript" not in repr(first).lower()


def test_manifest_training_boundary_passes_audio_only_and_exact_profile(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import (
        CheckpointPayload,
        TrainingExample,
        Wav2VecTrainingProfile,
        train_from_manifest,
    )

    audio = _wav(tmp_path / "audio.wav", 0.5)
    audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
    manifest = tmp_path / "emotion.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "item-1",
                "call_id": "call-1",
                "speaker_id": "speaker-1",
                "audio_path": str(audio.resolve()),
                "transcript": "transcript-secret",
                "split": "train",
                "source": "emotion",
                "emotion": "happiness",
                "sha256": audio_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    observed: list[tuple[tuple[TrainingExample, ...], object]] = []

    def backend(examples: tuple[TrainingExample, ...], profile: object) -> CheckpointPayload:
        observed.append((examples, profile))
        return CheckpointPayload(
            architecture="wav2vec-xls-r",
            model_id="facebook/wav2vec2-xls-r-300m",
            weights=b"weights",
            metrics={"macro_f1": 1.0},
            validation_hash="a" * 64,
        )

    output = tmp_path / "output"
    train_from_manifest(
        manifest,
        output,
        profile=Wav2VecTrainingProfile(),
        backend=backend,
    )

    assert output.is_dir()
    assert len(observed) == 1
    examples, profile = observed[0]
    assert isinstance(profile, Wav2VecTrainingProfile)
    assert examples == (
        TrainingExample(
            id="item-1",
            audio_path=audio.resolve(),
            audio_sha256=audio_hash,
            label="happiness",
            split="train",
        ),
    )
    assert "transcript" not in repr(examples).lower()


def test_manifest_training_rejects_wrong_source_hash_and_symlink(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import load_training_examples

    audio = _wav(tmp_path / "audio.wav", 0.5)
    link = tmp_path / "linked.wav"
    link.symlink_to(audio)
    base = {
        "id": "item-1",
        "call_id": "call-1",
        "speaker_id": "speaker-1",
        "audio_path": str(link),
        "transcript": "private-secret",
        "split": "train",
        "source": "emotion",
        "emotion": "happiness",
        "sha256": "0" * 64,
    }
    manifest = tmp_path / "emotion.jsonl"
    manifest.write_text(json.dumps(base) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid_training_manifest") as raised:
        load_training_examples(manifest)
    assert "private" not in str(raised.value).lower()
    assert "secret" not in str(raised.value).lower()


def test_training_cli_errors_are_sanitized_and_imports_are_lazy(tmp_path: Path) -> None:
    before = set(sys.modules)
    import importlib

    importlib.import_module("voxdelta.evaluation.emotion_training")
    added = set(sys.modules) - before
    assert "torch" not in added
    assert "transformers" not in added
    assert "funasr" not in added

    sentinel = tmp_path / "private-transcript-secret.jsonl"
    environment = dict(os.environ)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_emotion.py",
            "--manifest",
            str(sentinel),
            "--output",
            str(tmp_path / "out"),
            "--base-model",
            "facebook/wav2vec2-xls-r-300m",
            "--seed",
            "622",
        ],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "training_error: invalid_training_manifest"
    assert "private" not in result.stderr.lower()
    assert os.environ == environment


def test_evaluation_example_batches_use_exact_profile_batch_size() -> None:
    from voxdelta.evaluation.emotion_training import evaluation_example_batches

    examples = tuple(range(17))
    observed = [len(batch) for batch in evaluation_example_batches(examples, 8)]

    assert observed == [8, 8, 1]
    assert tuple(item for batch in evaluation_example_batches(examples, 8) for item in batch) == (
        examples
    )


def test_xls_r_evaluation_caps_each_model_call_at_eight_windows() -> None:
    from voxdelta.evaluation.emotion_training import (
        AudioClip,
        batched_evaluation_logits,
        evaluation_windows,
    )

    clip = AudioClip(samples=(0.0,) * (45 * 16_000))
    windows_by_item = tuple(evaluation_windows(clip) for _ in range(8))
    call_sizes: list[int] = []

    def predict(windows: tuple[AudioClip, ...]) -> tuple[tuple[float, ...], ...]:
        offset = sum(call_sizes)
        call_sizes.append(len(windows))
        return tuple((float(offset + index),) * 7 for index in range(len(windows)))

    logits = batched_evaluation_logits(windows_by_item, 8, predict)

    assert call_sizes == [8, 8, 8]
    assert len(logits) == 8
    assert all(len(row) == 7 for row in logits)
    assert [row[0] for row in logits] == [float(3 * index + 1) for index in range(8)]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--architecture", "private-transcript-secret"],
        ["--seed", "private-transcript-secret"],
        ["--unknown-private-transcript-secret"],
        ["--manifest", "private-transcript-secret"],
    ],
)
def test_argparse_failures_never_echo_rejected_values(tmp_path: Path, arguments: list[str]) -> None:
    command = [sys.executable, "scripts/train_emotion.py", *arguments]
    result = subprocess.run(
        command,
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "training_error: invalid_training_profile"
    assert "private" not in result.stderr.lower()
    assert "transcript" not in result.stderr.lower()
    assert "secret" not in result.stderr.lower()
