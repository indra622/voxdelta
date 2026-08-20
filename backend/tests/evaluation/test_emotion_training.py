from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

LABELS = ("happiness", "anger", "disgust", "fear", "neutral", "sadness", "surprise")
WAV2VEC_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
WAV2VEC_WEIGHTS_SHA256 = "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"


def test_emotion2vec_input_is_one_float32_waveform() -> None:
    from voxdelta.evaluation.emotion_training import emotion2vec_input

    waveform = emotion2vec_input((0.25, -0.5, 0.75))

    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32
    assert waveform.shape == (3,)
    assert waveform.tolist() == pytest.approx([0.25, -0.5, 0.75])


def _wav(path: Path, seconds: float, *, channels: int = 1, rate: int = 16_000) -> Path:
    frames = max(1, round(seconds * rate))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(struct.pack(f"<{frames * channels}h", *([100] * frames * channels)))
    return path


def test_profiles_are_strict_and_exact(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import (
        Emotion2VecTrainingProfile,
        Wav2VecTrainingProfile,
    )

    wav = Wav2VecTrainingProfile(base_model_path=tmp_path.resolve())
    assert wav.model_id == "facebook/wav2vec2-xls-r-300m"
    assert wav.seed == 622
    assert (wav.learning_rate, wav.train_batch_size, wav.eval_batch_size) == (2e-5, 8, 8)
    assert (wav.gradient_accumulation_steps, wav.epochs, wav.warmup_ratio) == (2, 10, 0.1)
    assert (wav.evaluation_strategy, wav.selection_metric, wav.early_stopping_patience) == (
        "epoch",
        "macro_f1",
        2,
    )
    assert wav.class_weighting == "inverse-frequency"
    assert wav.base_model_path == tmp_path.resolve()
    modern = Emotion2VecTrainingProfile()
    assert modern.encoder_id == "iic/emotion2vec_plus_large"
    assert modern.encoder_revision == "v2.0.5"
    assert modern.freeze_encoder is True
    assert (modern.hidden_size, modern.dropout, modern.learning_rate) == (256, 0.1, 1e-3)
    assert (modern.batch_size, modern.epochs, modern.early_stopping_patience) == (64, 20, 3)
    assert modern.trainable_components == ("classifier_head",)
    assert modern.class_weighting == "inverse-frequency"

    for model, field, value in [
        (Wav2VecTrainingProfile, "seed", "622"),
        (Wav2VecTrainingProfile, "warmup_ratio", True),
        (Wav2VecTrainingProfile, "learning_rate", 3e-5),
        (Wav2VecTrainingProfile, "epochs", 9),
        (Emotion2VecTrainingProfile, "freeze_encoder", 1),
        (Emotion2VecTrainingProfile, "hidden_size", 128),
        (Emotion2VecTrainingProfile, "epochs", 19),
        (Wav2VecTrainingProfile, "class_weighting", "none"),
        (Emotion2VecTrainingProfile, "class_weighting", "none"),
    ]:
        with pytest.raises(ValidationError):
            model.model_validate({field: value})


@pytest.mark.parametrize("micro_batch,accumulation", [(8, 2), (4, 4), (2, 8), (1, 16)])
def test_wav2vec_profiles_preserve_effective_batch_sixteen(
    tmp_path: Path, micro_batch: int, accumulation: int
) -> None:
    from voxdelta.evaluation.emotion_training import Wav2VecTrainingProfile

    profile = Wav2VecTrainingProfile(
        base_model_path=tmp_path.resolve(),
        train_batch_size=micro_batch,
        eval_batch_size=micro_batch,
        gradient_accumulation_steps=accumulation,
    )

    assert profile.train_batch_size * profile.gradient_accumulation_steps == 16


def test_partial_last4_profile_is_exact_and_memory_bounded(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import Wav2VecTrainingProfile

    profile = Wav2VecTrainingProfile(
        base_model_path=tmp_path.resolve(),
        adaptation_strategy="partial-last4",
        train_batch_size=2,
        eval_batch_size=2,
        gradient_accumulation_steps=8,
    )

    assert profile.adaptation_strategy == "partial-last4"
    assert profile.train_batch_size * profile.gradient_accumulation_steps == 16


@pytest.mark.parametrize(
    "metadata",
    [
        {"adaptation_strategy": "partial-last4"},
        {
            "adaptation_strategy": "partial-last4",
            "train_batch_size": 4,
            "eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
        },
        {"adaptation_strategy": "unknown"},
    ],
)
def test_partial_last4_profile_rejects_non_exact_combinations(
    tmp_path: Path, metadata: dict[str, object]
) -> None:
    from voxdelta.evaluation.emotion_training import Wav2VecTrainingProfile

    with pytest.raises(ValidationError):
        Wav2VecTrainingProfile.model_validate({"base_model_path": tmp_path.resolve(), **metadata})


def _fake_wav2vec_model(torch: object, layer_count: int = 24) -> object:
    model = torch.nn.Module()
    model.wav2vec2 = torch.nn.Module()
    model.wav2vec2.feature_extractor = torch.nn.Linear(2, 2)
    model.wav2vec2.feature_projection = torch.nn.Linear(2, 2)
    model.wav2vec2.encoder = torch.nn.Module()
    model.wav2vec2.encoder.layers = torch.nn.ModuleList(
        torch.nn.Linear(2, 2) for _ in range(layer_count)
    )
    model.projector = torch.nn.Linear(2, 2)
    model.classifier = torch.nn.Linear(2, 7)
    return model


def test_partial_last4_enables_only_exact_allowlist() -> None:
    import torch

    from voxdelta.evaluation.emotion_training import (
        PARTIAL_LAST4_PREFIXES,
        configure_wav2vec_trainable_parameters,
    )

    model = _fake_wav2vec_model(torch)
    summary = configure_wav2vec_trainable_parameters(model, "partial-last4")
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert summary.encoder_layers == (20, 21, 22, 23)
    assert summary.module_prefixes == PARTIAL_LAST4_PREFIXES
    assert trainable_names
    assert all(name.startswith(PARTIAL_LAST4_PREFIXES) for name in trainable_names)
    assert summary.trainable_parameter_count == sum(
        parameter.numel() for parameter in summary.parameters
    )
    assert 0 < summary.trainable_parameter_count < summary.total_parameter_count


@pytest.mark.parametrize("layer_count", [23, 25])
def test_partial_last4_rejects_unexpected_encoder_depth(layer_count: int) -> None:
    import torch

    from voxdelta.evaluation.emotion_training import (
        TrainingError,
        configure_wav2vec_trainable_parameters,
    )

    with pytest.raises(TrainingError, match="training_failed"):
        configure_wav2vec_trainable_parameters(
            _fake_wav2vec_model(torch, layer_count), "partial-last4"
        )


def test_partial_last4_rejects_missing_required_prefix() -> None:
    import torch

    from voxdelta.evaluation.emotion_training import (
        TrainingError,
        configure_wav2vec_trainable_parameters,
    )

    model = _fake_wav2vec_model(torch)
    del model.classifier

    with pytest.raises(TrainingError, match="training_failed"):
        configure_wav2vec_trainable_parameters(model, "partial-last4")


def test_partial_last4_optimizer_receives_only_allowlisted_parameters(tmp_path: Path) -> None:
    import torch

    from voxdelta.evaluation.emotion_training import (
        Wav2VecTrainingProfile,
        build_wav2vec_optimizer,
    )

    model = _fake_wav2vec_model(torch)
    profile = Wav2VecTrainingProfile(
        base_model_path=tmp_path.resolve(),
        adaptation_strategy="partial-last4",
        train_batch_size=2,
        eval_batch_size=2,
        gradient_accumulation_steps=8,
    )
    optimizer, summary = build_wav2vec_optimizer(torch, model, profile)

    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_ids == {id(parameter) for parameter in summary.parameters}
    assert optimizer_ids == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


@pytest.mark.parametrize(
    "metadata",
    [
        {"base_model_path": Path("relative")},
        {"train_batch_size": 4, "eval_batch_size": 8, "gradient_accumulation_steps": 4},
        {"train_batch_size": 4, "eval_batch_size": 4, "gradient_accumulation_steps": 2},
        {"train_batch_size": 3, "eval_batch_size": 3, "gradient_accumulation_steps": 5},
    ],
)
def test_wav2vec_profiles_reject_unapproved_path_or_batch_combinations(
    tmp_path: Path, metadata: dict[str, object]
) -> None:
    from voxdelta.evaluation.emotion_training import Wav2VecTrainingProfile

    values: dict[str, object] = {"base_model_path": tmp_path.resolve(), **metadata}
    with pytest.raises(ValidationError):
        Wav2VecTrainingProfile.model_validate(values)


def test_wav2vec_component_load_is_verified_local_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voxdelta.evaluation.emotion_training as training
    from voxdelta.evaluation.emotion_training import load_wav2vec_components
    from voxdelta.evaluation.wav2vec_base import PreparedWav2VecBase

    calls: list[tuple[str, str, dict[str, object]]] = []

    class Loader:
        def __init__(self, kind: str) -> None:
            self.kind = kind

        def from_pretrained(self, source: str, **kwargs: object) -> object:
            calls.append((self.kind, source, kwargs))
            return object()

    class Transformers:
        AutoFeatureExtractor = Loader("extractor")
        AutoModelForAudioClassification = Loader("model")

    base = tmp_path.resolve()
    monkeypatch.setattr(
        training,
        "validate_wav2vec_base",
        lambda path: PreparedWav2VecBase(Path(path)),
    )

    extractor, model, prepared = load_wav2vec_components(Transformers(), base)

    assert extractor is not None and model is not None
    assert prepared.path == base
    assert calls[0] == ("extractor", str(base), {"local_files_only": True})
    assert calls[1][0:2] == ("model", str(base))
    assert calls[1][2]["local_files_only"] is True
    assert calls[1][2]["ignore_mismatched_sizes"] is True


def test_inverse_frequency_weights_are_train_only_canonical_and_normalized() -> None:
    from voxdelta.evaluation.emotion_training import (
        TrainingError,
        TrainingExample,
        inverse_frequency_class_weights,
    )

    training = tuple(
        TrainingExample(
            id=f"train-{label}",
            audio_path=Path(f"/{label}.wav"),
            audio_sha256="a" * 64,
            label=label,
            split="train",
        )
        for label in LABELS
    ) + (
        TrainingExample(
            id="train-sadness-extra",
            audio_path=Path("/sadness-extra.wav"),
            audio_sha256="b" * 64,
            label="sadness",
            split="train",
        ),
    )
    validation = tuple(
        TrainingExample(
            id=f"validation-sadness-{index}",
            audio_path=Path(f"/validation-{index}.wav"),
            audio_sha256=f"{index:064x}",
            label="sadness",
            split="validation",
        )
        for index in range(100)
    )

    weights = inverse_frequency_class_weights(training + validation)

    assert weights == pytest.approx((8 / 7, 8 / 7, 8 / 7, 8 / 7, 8 / 7, 4 / 7, 8 / 7))
    weighted_mean = sum(weights[LABELS.index(item.label)] for item in training) / len(training)
    assert weighted_mean == pytest.approx(1.0)
    with pytest.raises(TrainingError, match="invalid_training_manifest"):
        inverse_frequency_class_weights(
            tuple(item for item in training if item.label != "surprise")
        )
    with pytest.raises(TrainingError, match="invalid_training_manifest"):
        inverse_frequency_class_weights(validation)


def test_weighted_cross_entropy_passes_canonical_weights_to_torch() -> None:
    from voxdelta.evaluation.emotion_training import weighted_cross_entropy

    class FakeFunctional:
        def __init__(self, owner: FakeTorch) -> None:
            self.owner = owner

        def cross_entropy(self, logits: object, labels: object, *, weight: object) -> str:
            self.owner.loss_calls.append((logits, labels, weight))
            return "loss"

    class FakeNN:
        def __init__(self, owner: FakeTorch) -> None:
            self.functional = FakeFunctional(owner)

    class FakeTorch:
        float32 = "float32"

        def __init__(self) -> None:
            self.tensor_calls: list[tuple[tuple[float, ...], object, str]] = []
            self.loss_calls: list[tuple[object, object, object]] = []
            self.nn = FakeNN(self)

        def tensor(self, values: tuple[float, ...], *, dtype: object, device: str) -> str:
            self.tensor_calls.append((values, dtype, device))
            return "weight-tensor"

    fake_torch = FakeTorch()

    result = weighted_cross_entropy(
        fake_torch,
        logits="logits",
        labels="labels",
        class_weights=(1.0, 2.0),
        device="mps",
    )

    assert result == "loss"
    assert fake_torch.tensor_calls == [((1.0, 2.0), "float32", "mps")]
    assert fake_torch.loss_calls == [("logits", "labels", "weight-tensor")]


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
        class_weighting="inverse-frequency",
        class_weights=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
    )
    publish_checkpoint(output, payload)

    assert sorted(item.name for item in output.iterdir()) == [
        "config.json",
        "label_mapping.json",
        "metrics.json",
        "model.safetensors",
    ]
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["schema_version"] == "2"
    assert config["labels"] == list(LABELS)
    assert config["encoder_hash"] == "b" * 64
    assert config["encoder_revision"] == "v2.0.5"
    assert config["embedding_size"] == 2
    assert config["class_weighting"] == "inverse-frequency"
    assert config["class_weights"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["validation_hash"] == "a" * 64


@pytest.mark.parametrize(
    "metadata",
    [
        {"class_weighting": "inverse-frequency"},
        {"class_weights": (1.0,) * 7},
        {
            "class_weighting": "inverse-frequency",
            "class_weights": (1.0,) * 6,
        },
        {
            "class_weighting": "inverse-frequency",
            "class_weights": (0.0,) + (1.0,) * 6,
        },
        {
            "class_weighting": "inverse-frequency",
            "class_weights": (float("nan"),) + (1.0,) * 6,
        },
    ],
)
def test_checkpoint_payload_rejects_incomplete_or_invalid_class_weights(
    metadata: dict[str, object],
) -> None:
    from voxdelta.evaluation.emotion_training import CheckpointPayload

    with pytest.raises(ValidationError):
        CheckpointPayload(
            architecture="wav2vec-xls-r",
            model_id="facebook/wav2vec2-xls-r-300m",
            weights=b"weights",
            metrics={"macro_f1": 0.7},
            validation_hash="a" * 64,
            model_revision=WAV2VEC_REVISION,
            base_model_sha256=WAV2VEC_WEIGHTS_SHA256,
            **metadata,
        )


def test_wav2vec_checkpoint_requires_exact_pinned_base_provenance() -> None:
    from voxdelta.evaluation.emotion_training import CheckpointPayload

    base = {
        "architecture": "wav2vec-xls-r",
        "model_id": "facebook/wav2vec2-xls-r-300m",
        "weights": b"weights",
        "metrics": {"macro_f1": 0.7},
        "validation_hash": "a" * 64,
        "class_weighting": "inverse-frequency",
        "class_weights": (1.0,) * 7,
    }
    with pytest.raises(ValidationError):
        CheckpointPayload(**base)
    with pytest.raises(ValidationError):
        CheckpointPayload(
            **base,
            model_revision="wrong",
            base_model_sha256=WAV2VEC_WEIGHTS_SHA256,
        )


def test_wav2vec_checkpoint_publishes_schema_three_provenance(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import CheckpointPayload, publish_checkpoint

    output = tmp_path / "model"
    payload = CheckpointPayload(
        architecture="wav2vec-xls-r",
        model_id="facebook/wav2vec2-xls-r-300m",
        weights=b"weights",
        metrics={"macro_f1": 0.7},
        validation_hash="a" * 64,
        model_revision=WAV2VEC_REVISION,
        base_model_sha256=WAV2VEC_WEIGHTS_SHA256,
        class_weighting="inverse-frequency",
        class_weights=(1.0,) * 7,
    )

    publish_checkpoint(output, payload)

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["schema_version"] == "3"
    assert config["model_revision"] == WAV2VEC_REVISION
    assert config["base_model_sha256"] == WAV2VEC_WEIGHTS_SHA256


def test_checkpoint_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import CheckpointPayload, publish_checkpoint

    output = tmp_path / "model"
    payload = CheckpointPayload(
        architecture="wav2vec-xls-r",
        model_id="facebook/wav2vec2-xls-r-300m",
        weights=b"weights",
        metrics={"macro_f1": 0.7},
        validation_hash="a" * 64,
        model_revision=WAV2VEC_REVISION,
        base_model_sha256=WAV2VEC_WEIGHTS_SHA256,
        class_weighting="inverse-frequency",
        class_weights=(1.0,) * 7,
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
        model_revision=WAV2VEC_REVISION,
        base_model_sha256=WAV2VEC_WEIGHTS_SHA256,
        class_weighting="inverse-frequency",
        class_weights=(1.0,) * 7,
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
            model_revision=WAV2VEC_REVISION,
            base_model_sha256=WAV2VEC_WEIGHTS_SHA256,
            class_weighting="inverse-frequency",
            class_weights=(1.0,) * 7,
        )

    output = tmp_path / "output"
    train_from_manifest(
        manifest,
        output,
        profile=Wav2VecTrainingProfile(base_model_path=tmp_path.resolve()),
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


def test_manifest_training_rejects_unweighted_backend_payload(tmp_path: Path) -> None:
    from voxdelta.evaluation.emotion_training import (
        CheckpointPayload,
        TrainingError,
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

    def unweighted_backend(
        _examples: tuple[TrainingExample, ...], _profile: object
    ) -> CheckpointPayload:
        return CheckpointPayload(
            architecture="wav2vec-xls-r",
            model_id="facebook/wav2vec2-xls-r-300m",
            weights=b"weights",
            metrics={"macro_f1": 1.0},
            validation_hash="a" * 64,
        )

    output = tmp_path / "output"
    with pytest.raises(TrainingError, match="training_failed"):
        train_from_manifest(
            manifest,
            output,
            profile=Wav2VecTrainingProfile(base_model_path=tmp_path.resolve()),
            backend=unweighted_backend,
        )

    assert not output.exists()


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
            "--base-model-path",
            str(tmp_path.resolve()),
            "--micro-batch-size",
            "4",
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


@pytest.mark.parametrize(
    "arguments",
    [
        ["--micro-batch-size", "4"],
        ["--base-model-path", "/base"],
        ["--base-model-path", "/base", "--micro-batch-size", "3"],
    ],
)
def test_training_cli_requires_both_approved_xls_r_local_flags(
    tmp_path: Path, arguments: list[str]
) -> None:
    command = [
        sys.executable,
        "scripts/train_emotion.py",
        "--manifest",
        str(tmp_path / "manifest.jsonl"),
        "--output",
        str(tmp_path / "output"),
        "--architecture",
        "wav2vec-xls-r",
        "--base-model",
        "facebook/wav2vec2-xls-r-300m",
        *arguments,
    ]
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


@pytest.mark.parametrize(
    "arguments",
    [
        ["--architecture", "emotion2vec-plus", "--adaptation-strategy", "partial-last4"],
        [
            "--architecture",
            "wav2vec-xls-r",
            "--adaptation-strategy",
            "partial-last4",
            "--base-model-path",
            "/base",
            "--micro-batch-size",
            "4",
        ],
    ],
)
def test_training_cli_rejects_incompatible_partial_last4_flags(
    tmp_path: Path, arguments: list[str]
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_emotion.py",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--output",
            str(tmp_path / "output"),
            "--base-model",
            "facebook/wav2vec2-xls-r-300m",
            *arguments,
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "training_error: invalid_training_profile"


def test_training_cli_accepts_partial_last4_flag_before_manifest_validation(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_emotion.py",
            "--manifest",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(tmp_path / "output"),
            "--architecture",
            "wav2vec-xls-r",
            "--adaptation-strategy",
            "partial-last4",
            "--base-model",
            "facebook/wav2vec2-xls-r-300m",
            "--base-model-path",
            str(tmp_path.resolve()),
            "--micro-batch-size",
            "2",
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "training_error: invalid_training_manifest"


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
