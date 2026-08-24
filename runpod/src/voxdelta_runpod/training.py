"""CUDA BF16 XLS-R full-training runtime and isolated memory-probe contracts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from voxdelta.evaluation.emotion_training import (
    AudioClip,
    batched_evaluation_logits,
    evaluation_windows,
    load_audio,
    load_wav2vec_components,
)
from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

from voxdelta_runpod.checkpoint import (
    CheckpointStage,
    CheckpointState,
    publish_epoch_checkpoint,
    select_resume_checkpoint,
)
from voxdelta_runpod.config import CANONICAL_LABELS, ExperimentConfig
from voxdelta_runpod.ledger import RunIdentity
from voxdelta_runpod.recipes import RecipePlan

ProbeStatus = Literal["pass", "oom"]


class TrainingRuntimeError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class BatchProfile(_FrozenModel):
    micro_batch_size: Literal[8, 4, 2, 1]
    gradient_accumulation_steps: Literal[2, 4, 8, 16]

    @model_validator(mode="after")
    def effective_sixteen(self) -> BatchProfile:
        if self.micro_batch_size * self.gradient_accumulation_steps != 16:
            raise ValueError("effective batch must be 16")
        return self


BATCH_PROFILES: tuple[BatchProfile, ...] = (
    BatchProfile(micro_batch_size=8, gradient_accumulation_steps=2),
    BatchProfile(micro_batch_size=4, gradient_accumulation_steps=4),
    BatchProfile(micro_batch_size=2, gradient_accumulation_steps=8),
    BatchProfile(micro_batch_size=1, gradient_accumulation_steps=16),
)


class MemoryProbeAttempt(_FrozenModel):
    profile: BatchProfile
    status: ProbeStatus


class MemoryProbeResult(_FrozenModel):
    selected: BatchProfile
    attempts: tuple[MemoryProbeAttempt, ...]


class _CompletedProcess(Protocol):
    returncode: int
    stdout: str
    stderr: str


ProbeRunner = Callable[..., _CompletedProcess]


def run_isolated_memory_probe(
    command_prefix: Sequence[str],
    *,
    runner: ProbeRunner = subprocess.run,
) -> MemoryProbeResult:
    """Run each batch attempt in a fresh process; only exit 75 authorizes OOM fallback."""

    if not command_prefix:
        raise TrainingRuntimeError("invalid_probe_command")
    attempts: list[MemoryProbeAttempt] = []
    for profile in BATCH_PROFILES:
        completed = runner(
            [
                *command_prefix,
                "--micro-batch-size",
                str(profile.micro_batch_size),
                "--gradient-accumulation-steps",
                str(profile.gradient_accumulation_steps),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout == "probe_ok\n" and not completed.stderr:
            attempts.append(MemoryProbeAttempt(profile=profile, status="pass"))
            return MemoryProbeResult(selected=profile, attempts=tuple(attempts))
        if (
            completed.returncode == 75
            and completed.stdout == ""
            and completed.stderr == "probe_oom\n"
        ):
            attempts.append(MemoryProbeAttempt(profile=profile, status="oom"))
            continue
        raise TrainingRuntimeError("memory_probe_failed")
    raise TrainingRuntimeError("no_batch_profile_fits")


def deterministic_crop_start(
    frame_count: int,
    window_frames: int,
    *,
    seed: int,
    epoch: int,
    semantic_fingerprint: str,
) -> int:
    if frame_count <= 0 or window_frames <= 0 or seed <= 0 or epoch < 0 or not semantic_fingerprint:
        raise TrainingRuntimeError("invalid_crop_key")
    if frame_count <= window_frames:
        return 0
    digest = hashlib.sha256(f"{seed}\0{epoch}\0{semantic_fingerprint}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (frame_count - window_frames + 1)


def deterministic_epoch_records(plan: RecipePlan, *, seed: int, epoch: int) -> tuple[Any, ...]:
    if epoch < 0 or seed <= 0:
        raise TrainingRuntimeError("invalid_epoch_order")
    generator = random.Random(seed + epoch)
    if plan.sampler == "uniform-without-replacement":
        records = list(plan.train)
        generator.shuffle(records)
        return tuple(records)
    by_label = {
        label: [item for item in plan.train if item.emotion == label] for label in CANONICAL_LABELS
    }
    if any(not records for records in by_label.values()):
        raise TrainingRuntimeError("invalid_epoch_order")
    ordered: list[Any] = []
    while len(ordered) < plan.epoch_draws:
        labels = list(CANONICAL_LABELS)
        generator.shuffle(labels)
        for label in labels:
            ordered.append(generator.choice(by_label[label]))
            if len(ordered) == plan.epoch_draws:
                break
    return tuple(ordered)


def configure_full_model(model: Any) -> None:
    parameters = tuple(model.parameters())
    if not parameters:
        raise TrainingRuntimeError("invalid_model_parameters")
    for parameter in parameters:
        parameter.requires_grad_(True)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        raise TrainingRuntimeError("gradient_checkpointing_unavailable") from None
    if not all(bool(parameter.requires_grad) for parameter in parameters):
        raise TrainingRuntimeError("frozen_model_parameter")


def optimizer_parameter_groups(model: Any, config: ExperimentConfig) -> tuple[dict[str, Any], ...]:
    encoder: list[Any] = []
    head: list[Any] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not bool(parameter.requires_grad) or id(parameter) in seen:
            raise TrainingRuntimeError("invalid_model_parameters")
        seen.add(id(parameter))
        (encoder if name.startswith("wav2vec2.") else head).append(parameter)
    if not encoder or not head or len(seen) != len(tuple(model.parameters())):
        raise TrainingRuntimeError("invalid_model_parameters")
    return (
        {"params": encoder, "lr": config.training.encoder_learning_rate},
        {"params": head, "lr": config.training.head_learning_rate},
    )


def validate_cuda_runtime(torch: Any) -> None:
    try:
        if (
            not bool(torch.cuda.is_available())
            or int(torch.cuda.device_count()) != 1
            or not bool(torch.cuda.is_bf16_supported())
        ):
            raise TrainingRuntimeError("invalid_cuda_runtime")
    except TrainingRuntimeError:
        raise
    except Exception:
        raise TrainingRuntimeError("invalid_cuda_runtime") from None


class EpochMetrics(_FrozenModel):
    completed_count: int = Field(gt=0)
    macro_f1: float = Field(ge=0, le=1)
    per_label_f1: dict[str, float]
    predicted_class_count: int = Field(ge=1, le=7)

    @model_validator(mode="after")
    def finite_complete_metrics(self) -> EpochMetrics:
        if set(self.per_label_f1) != set(CANONICAL_LABELS) or any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in self.per_label_f1.values()
        ):
            raise ValueError("invalid epoch metrics")
        return self


class TrainingResult(_FrozenModel):
    best_epoch: int = Field(ge=0)
    epochs_completed: int = Field(gt=0)
    optimizer_steps: int = Field(gt=0)
    metrics: EpochMetrics
    checkpoint_path: Path


class TrainingHistoryRecord(_FrozenModel):
    schema_version: Literal["1"] = "1"
    stage: CheckpointStage
    epoch: int = Field(ge=0)
    optimizer_steps: int = Field(gt=0)
    metrics: EpochMetrics
    best_metric: float = Field(ge=0, le=1)
    best_epoch: int = Field(ge=0)
    patience_used: int = Field(ge=0)
    batch_profile: BatchProfile
    run_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def append_training_history(path: Path, record: TrainingHistoryRecord) -> None:
    """Append one deterministic epoch metric record, accepting an exact crash replay."""

    if not path.is_absolute() or path.is_symlink():
        raise TrainingRuntimeError("invalid_training_history")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        records = (
            tuple(
                TrainingHistoryRecord.model_validate_json(line)
                for line in path.read_bytes().splitlines()
                if line.strip()
            )
            if path.exists()
            else ()
        )
        if any(existing.epoch != index for index, existing in enumerate(records)):
            raise ValueError
        if record.epoch < len(records):
            if records[record.epoch] != record:
                raise ValueError
            return
        if record.epoch != len(records):
            raise ValueError
        with path.open("ab") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write((record.model_dump_json() + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
    except TrainingRuntimeError:
        raise
    except Exception:
        raise TrainingRuntimeError("invalid_training_history") from None


def _metric_report(expected: Sequence[int], predicted: Sequence[int]) -> EpochMetrics:
    if not expected or len(expected) != len(predicted):
        raise TrainingRuntimeError("invalid_validation_metrics")
    per_label: dict[str, float] = {}
    for index, label in enumerate(CANONICAL_LABELS):
        true_positive = sum(
            a == index and b == index for a, b in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            a != index and b == index for a, b in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            a == index and b != index for a, b in zip(expected, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        per_label[label] = 0.0 if denominator == 0 else 2 * true_positive / denominator
    return EpochMetrics(
        completed_count=len(expected),
        macro_f1=math.fsum(per_label.values()) / len(CANONICAL_LABELS),
        per_label_f1=per_label,
        predicted_class_count=len(set(predicted)),
    )


def _torch_bytes(torch: Any, payload: Any) -> bytes:
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


@dataclass(slots=True)
class _Runtime:
    torch: Any
    transformers: Any
    safetensors: Any
    extractor: Any
    model: Any
    optimizer: Any
    scheduler: Any


def _load_runtime(
    base_model_path: Path,
    config: ExperimentConfig,
    profile: BatchProfile,
    total_steps: int,
) -> _Runtime:
    try:
        import importlib

        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        safetensors = importlib.import_module("safetensors.torch")
        validate_cuda_runtime(torch)
        extractor, model, _prepared = load_wav2vec_components(transformers, base_model_path)
        configure_full_model(model)
        model.to("cuda")
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(model, config),
            weight_decay=config.training.weight_decay,
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=round(total_steps * config.training.warmup_ratio),
            num_training_steps=total_steps,
        )
        return _Runtime(torch, transformers, safetensors, extractor, model, optimizer, scheduler)
    except TrainingRuntimeError:
        raise
    except Exception as error:
        if "torch" in locals() and isinstance(error, torch.cuda.OutOfMemoryError):
            raise
        raise TrainingRuntimeError("training_runtime_unavailable") from None


def _inputs(runtime: _Runtime, clips: Sequence[AudioClip]) -> dict[str, Any]:
    encoded = runtime.extractor(
        [list(clip.samples) for clip in clips],
        sampling_rate=16_000,
        padding=True,
        return_tensors="pt",
    )
    return {name: tensor.to("cuda") for name, tensor in encoded.items()}


def _clip_for_record(
    audio_root: Path, record: Any, config: ExperimentConfig, epoch: int
) -> AudioClip:
    clip = load_audio(audio_root / record.audio_path)
    window = config.training.sample_rate * config.training.window_seconds
    start = deterministic_crop_start(
        len(clip.samples),
        window,
        seed=config.seed,
        epoch=epoch,
        semantic_fingerprint=record.audio_sha256,
    )
    return AudioClip(samples=clip.samples[start : start + window])


def _batches(records: Sequence[Any], size: int) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(records[start : start + size]) for start in range(0, len(records), size))


def _evaluate(
    runtime: _Runtime, plan: RecipePlan, audio_root: Path, batch_size: int
) -> EpochMetrics:
    runtime.model.eval()
    expected = [CANONICAL_LABELS.index(record.emotion) for record in plan.validation]
    windows_by_item = tuple(
        evaluation_windows(load_audio(audio_root / record.audio_path)) for record in plan.validation
    )

    def predict(windows: tuple[AudioClip, ...]) -> Sequence[Sequence[float]]:
        with (
            runtime.torch.inference_mode(),
            runtime.torch.autocast(device_type="cuda", dtype=runtime.torch.bfloat16),
        ):
            logits = runtime.model(**_inputs(runtime, windows)).logits
        if not bool(runtime.torch.isfinite(logits).all()):
            raise TrainingRuntimeError("nonfinite_training_state")
        return cast(Sequence[Sequence[float]], logits.detach().float().cpu().tolist())

    averaged = batched_evaluation_logits(windows_by_item, batch_size, predict)
    predicted = [max(range(len(row)), key=row.__getitem__) for row in averaged]
    return _metric_report(expected, predicted)


def _serialize_model(runtime: _Runtime) -> bytes:
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in runtime.model.state_dict().items()
    }
    return cast(bytes, runtime.safetensors.save(state))


def publish_provider_checkpoint(
    output: Path,
    model_weights: bytes,
    plan: RecipePlan,
    metrics: EpochMetrics,
    config: ExperimentConfig,
) -> Path:
    """Publish a PEFT-free schema-v4 checkpoint accepted by the production provider."""

    if not output.is_absolute() or output.exists() or output.is_symlink() or not model_weights:
        raise TrainingRuntimeError("provider_checkpoint_publication_failed")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        class_weighting = "none" if plan.name == "pilot-a" else "sqrt-inverse-frequency"
        payloads = {
            "config.json": (
                json.dumps(
                    {
                        "schema_version": "4",
                        "architecture": "wav2vec-xls-r",
                        "model_id": config.model.model_id,
                        "labels": list(CANONICAL_LABELS),
                        "model_revision": config.model.revision,
                        "base_model_sha256": config.model.weights_sha256,
                        "class_weighting": class_weighting,
                        "class_weights": list(plan.loss_weights),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode(),
            "label_mapping.json": (
                json.dumps(
                    {str(index): label for index, label in enumerate(CANONICAL_LABELS)},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode(),
            "metrics.json": (
                json.dumps(
                    {
                        "macro_f1": metrics.macro_f1,
                        "validation_hash": plan.validation_manifest_sha256,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode(),
            "model.safetensors": model_weights,
        }
        for name, payload in payloads.items():
            with (staging / name).open("xb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(staging, output)
        staging = None
    except Exception:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise TrainingRuntimeError("provider_checkpoint_publication_failed") from None
    return output


def _memory_probe_plan(plan: RecipePlan) -> RecipePlan:
    train = tuple(
        item
        for label in CANONICAL_LABELS
        for item in tuple(record for record in plan.train if record.emotion == label)[:2]
    )
    validation = tuple(
        next(record for record in plan.validation if record.emotion == label)
        for label in CANONICAL_LABELS
    )
    if len(train) != 14 or len(validation) != 7:
        raise TrainingRuntimeError("invalid_memory_probe_slice")
    return RecipePlan(
        name=plan.name,
        scope="pilot",
        sampler=plan.sampler,
        epoch_draws=len(train),
        loss_weights=plan.loss_weights,
        train=train,
        validation=validation,
        train_manifest_sha256=hashlib.sha256(
            b"".join(record.model_dump_json().encode() for record in train)
        ).hexdigest(),
        validation_manifest_sha256=hashlib.sha256(
            b"".join(record.model_dump_json().encode() for record in validation)
        ).hexdigest(),
    )


def run_memory_probe_attempt(
    plan: RecipePlan,
    audio_root: Path,
    base_model_path: Path,
    output_root: Path,
    identity: RunIdentity,
    config: ExperimentConfig,
    batch_profile: BatchProfile,
) -> None:
    """Exercise synthetic and actual data, checkpoint publication, and provider reload."""

    if output_root.exists() or not all(
        path.is_absolute() for path in (audio_root, base_model_path, output_root)
    ):
        raise TrainingRuntimeError("invalid_memory_probe_paths")
    probe = _memory_probe_plan(plan)
    runtime = _load_runtime(base_model_path, config, batch_profile, total_steps=2)
    torch = runtime.torch
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    output_root.mkdir(mode=0o700, parents=True)

    synthetic = AudioClip(
        samples=(0.0,) * (config.training.sample_rate * config.training.window_seconds)
    )
    labels = torch.tensor(
        [index % len(CANONICAL_LABELS) for index in range(batch_profile.micro_batch_size)],
        dtype=torch.long,
        device="cuda",
    )
    runtime.model.train()
    runtime.optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        synthetic_batch = [synthetic] * batch_profile.micro_batch_size
        logits = runtime.model(**_inputs(runtime, synthetic_batch)).logits
        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
        raise TrainingRuntimeError("nonfinite_training_state")
    loss.backward()
    runtime.optimizer.zero_grad(set_to_none=True)

    ordered = deterministic_epoch_records(probe, seed=config.seed, epoch=0)
    batches = _batches(ordered, batch_profile.micro_batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        clips = [_clip_for_record(audio_root, record, config, 0) for record in batch]
        actual_labels = torch.tensor(
            [CANONICAL_LABELS.index(record.emotion) for record in batch],
            dtype=torch.long,
            device="cuda",
        )
        weights = torch.tensor(probe.loss_weights, dtype=torch.float32, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actual_logits = runtime.model(**_inputs(runtime, clips)).logits
            actual_loss = torch.nn.functional.cross_entropy(
                actual_logits.float(), actual_labels, weight=weights
            )
        if not bool(torch.isfinite(actual_logits).all()) or not bool(torch.isfinite(actual_loss)):
            raise TrainingRuntimeError("nonfinite_training_state")
        (actual_loss / batch_profile.gradient_accumulation_steps).backward()
        if batch_index % batch_profile.gradient_accumulation_steps == 0 or batch_index == len(
            batches
        ):
            torch.nn.utils.clip_grad_norm_(
                runtime.model.parameters(), config.training.gradient_clip_norm
            )
            runtime.optimizer.step()
            runtime.scheduler.step()
            runtime.optimizer.zero_grad(set_to_none=True)

    metrics = _evaluate(runtime, probe, audio_root, batch_profile.micro_batch_size)
    weights_payload = _serialize_model(runtime)
    payloads = {
        "model.safetensors": weights_payload,
        "best-model.safetensors": weights_payload,
        "optimizer.pt": _torch_bytes(torch, runtime.optimizer.state_dict()),
        "scheduler.pt": _torch_bytes(torch, runtime.scheduler.state_dict()),
        "rng-python.json": json.dumps(random.getstate(), separators=(",", ":")).encode(),
        "rng-torch-cpu.pt": _torch_bytes(torch, torch.random.get_rng_state()),
        "rng-torch-cuda.pt": _torch_bytes(torch, torch.cuda.get_rng_state_all()),
    }
    state = CheckpointState(
        stage=probe.name,
        epoch=0,
        optimizer_step=1,
        best_metric=metrics.macro_f1,
        best_epoch=0,
        patience_used=0,
        micro_batch_size=batch_profile.micro_batch_size,
        gradient_accumulation_steps=batch_profile.gradient_accumulation_steps,
        identity=identity,
        files={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
    )
    publish_epoch_checkpoint(output_root / "checkpoint", state, payloads)
    provider_root = publish_provider_checkpoint(
        output_root / "provider", weights_payload, probe, metrics, config
    )
    del runtime
    torch.cuda.empty_cache()
    provider = Wav2VecEmotionProvider(
        provider_root,
        base_model_path=base_model_path,
        device="cuda",
    )
    provider_digest: str | None = None
    try:
        for record in probe.validation:
            provider.analyze(record.item_key, audio_root / record.audio_path, "")
        provider_digest = provider.provenance.revision
    finally:
        provider.unload()
    if provider_digest is None:
        raise TrainingRuntimeError("memory_probe_provider_reload_failed")
    shutil.rmtree(output_root / "checkpoint")
    shutil.rmtree(output_root / "provider")
    evidence = {
        "schema_version": "1",
        "micro_batch_size": batch_profile.micro_batch_size,
        "gradient_accumulation_steps": batch_profile.gradient_accumulation_steps,
        "validation_count": metrics.completed_count,
        "macro_f1": metrics.macro_f1,
        "provider_checkpoint_sha256": provider_digest,
    }
    with (output_root / "probe-evidence.json").open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(
            (
                json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode()
        )


def _prune_superseded_checkpoints(root: Path, current: Path) -> None:
    for path in root.iterdir():
        if path == current:
            continue
        if path.is_symlink() or not path.is_dir() or not path.name.startswith("epoch-"):
            raise TrainingRuntimeError("invalid_checkpoint_prune_target")
        shutil.rmtree(path)


def run_cuda_training(
    plan: RecipePlan,
    audio_root: Path,
    base_model_path: Path,
    checkpoint_root: Path,
    identity: RunIdentity,
    config: ExperimentConfig,
    batch_profile: BatchProfile,
    *,
    resume: bool = False,
) -> TrainingResult:
    """Train/resume one frozen recipe and atomically publish every completed epoch."""

    if not all(path.is_absolute() for path in (audio_root, base_model_path, checkpoint_root)):
        raise TrainingRuntimeError("invalid_training_paths")
    max_epochs = (
        config.training.pilot_max_epochs
        if plan.scope == "pilot"
        else config.training.full_max_epochs
    )
    updates_per_epoch = math.ceil(
        math.ceil(plan.epoch_draws / batch_profile.micro_batch_size)
        / batch_profile.gradient_accumulation_steps
    )
    runtime = _load_runtime(base_model_path, config, batch_profile, updates_per_epoch * max_epochs)
    torch = runtime.torch
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    start_epoch = 0
    optimizer_steps = 0
    best_metric = -1.0
    best_epoch = 0
    best_model = b""
    stale_epochs = 0
    last_metrics: EpochMetrics | None = None
    last_checkpoint: Path | None = None
    expected_stage: CheckpointStage = plan.name if plan.scope == "pilot" else "full"
    if resume:
        selection = select_resume_checkpoint(
            checkpoint_root,
            expected_stage=expected_stage,
            expected_identity=identity,
        )
        if (selection.checkpoint is None) != (selection.state is None):
            raise TrainingRuntimeError("resume_checkpoint_invalid")
        if selection.checkpoint is None and selection.incomplete_checkpoint is None:
            raise TrainingRuntimeError("resume_checkpoint_missing")
        if selection.checkpoint is not None and selection.state is not None:
            state = selection.state
            try:
                runtime.model.load_state_dict(
                    runtime.safetensors.load(
                        (selection.checkpoint / "model.safetensors").read_bytes()
                    )
                )
                runtime.optimizer.load_state_dict(
                    torch.load(
                        selection.checkpoint / "optimizer.pt",
                        map_location="cuda",
                        weights_only=False,
                    )
                )
                runtime.scheduler.load_state_dict(
                    torch.load(
                        selection.checkpoint / "scheduler.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                )
                random.setstate(
                    _tuple_tree(json.loads((selection.checkpoint / "rng-python.json").read_text()))
                )
                torch.random.set_rng_state(
                    torch.load(selection.checkpoint / "rng-torch-cpu.pt", weights_only=True)
                )
                torch.cuda.set_rng_state_all(
                    torch.load(selection.checkpoint / "rng-torch-cuda.pt", weights_only=True)
                )
                best_model = (selection.checkpoint / "best-model.safetensors").read_bytes()
            except Exception:
                raise TrainingRuntimeError("resume_restore_failed") from None
            start_epoch = state.epoch + 1
            optimizer_steps = state.optimizer_step
            best_metric = state.best_metric
            best_epoch = state.best_epoch
            stale_epochs = state.patience_used
            last_checkpoint = selection.checkpoint
        if selection.incomplete_checkpoint is not None:
            shutil.rmtree(selection.incomplete_checkpoint)
    if last_checkpoint is not None and (
        start_epoch >= max_epochs or stale_epochs >= config.training.early_stopping_patience
    ):
        runtime.model.load_state_dict(runtime.safetensors.load(best_model))
        recovered_metrics = _evaluate(runtime, plan, audio_root, batch_profile.micro_batch_size)
        if abs(recovered_metrics.macro_f1 - best_metric) > 1e-12:
            raise TrainingRuntimeError("best_checkpoint_mismatch")
        return TrainingResult(
            best_epoch=best_epoch,
            epochs_completed=start_epoch,
            optimizer_steps=optimizer_steps,
            metrics=recovered_metrics,
            checkpoint_path=last_checkpoint,
        )
    for epoch in range(start_epoch, max_epochs):
        runtime.model.train()
        ordered = deterministic_epoch_records(plan, seed=config.seed, epoch=epoch)
        batches = _batches(ordered, batch_profile.micro_batch_size)
        runtime.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(batches, start=1):
            clips = [_clip_for_record(audio_root, record, config, epoch) for record in batch]
            labels = torch.tensor(
                [CANONICAL_LABELS.index(record.emotion) for record in batch],
                dtype=torch.long,
                device="cuda",
            )
            weights = torch.tensor(plan.loss_weights, dtype=torch.float32, device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = runtime.model(**_inputs(runtime, clips)).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), labels, weight=weights)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                raise TrainingRuntimeError("nonfinite_training_state")
            (loss / batch_profile.gradient_accumulation_steps).backward()
            should_step = (
                batch_index % batch_profile.gradient_accumulation_steps == 0
                or batch_index == len(batches)
            )
            if should_step:
                gradients = [
                    parameter.grad
                    for parameter in runtime.model.parameters()
                    if parameter.grad is not None
                ]
                if not gradients or any(
                    not bool(torch.isfinite(gradient).all()) for gradient in gradients
                ):
                    raise TrainingRuntimeError("nonfinite_training_state")
                norm = torch.nn.utils.clip_grad_norm_(
                    runtime.model.parameters(), config.training.gradient_clip_norm
                )
                if not bool(torch.isfinite(norm)):
                    raise TrainingRuntimeError("nonfinite_training_state")
                runtime.optimizer.step()
                runtime.scheduler.step()
                runtime.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        metrics = _evaluate(runtime, plan, audio_root, batch_profile.micro_batch_size)
        current_model = _serialize_model(runtime)
        if metrics.macro_f1 > best_metric + 1e-12:
            best_metric = metrics.macro_f1
            best_epoch = epoch
            best_model = current_model
            stale_epochs = 0
        else:
            stale_epochs += 1
        payloads = {
            "model.safetensors": current_model,
            "best-model.safetensors": best_model,
            "optimizer.pt": _torch_bytes(torch, runtime.optimizer.state_dict()),
            "scheduler.pt": _torch_bytes(torch, runtime.scheduler.state_dict()),
            "rng-python.json": json.dumps(random.getstate(), separators=(",", ":")).encode(),
            "rng-torch-cpu.pt": _torch_bytes(torch, torch.random.get_rng_state()),
            "rng-torch-cuda.pt": _torch_bytes(torch, torch.cuda.get_rng_state_all()),
        }
        state = CheckpointState(
            stage=expected_stage,
            epoch=epoch,
            optimizer_step=optimizer_steps,
            best_metric=best_metric,
            best_epoch=best_epoch,
            patience_used=stale_epochs,
            micro_batch_size=batch_profile.micro_batch_size,
            gradient_accumulation_steps=batch_profile.gradient_accumulation_steps,
            identity=identity,
            files={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        )
        append_training_history(
            checkpoint_root.parent / f"{checkpoint_root.name}-history.jsonl",
            TrainingHistoryRecord(
                stage=expected_stage,
                epoch=epoch,
                optimizer_steps=optimizer_steps,
                metrics=metrics,
                best_metric=best_metric,
                best_epoch=best_epoch,
                patience_used=stale_epochs,
                batch_profile=batch_profile,
                run_identity_sha256=identity.digest(),
            ),
        )
        last_checkpoint = publish_epoch_checkpoint(checkpoint_root, state, payloads)
        _prune_superseded_checkpoints(checkpoint_root, last_checkpoint)
        last_metrics = metrics
        if stale_epochs >= config.training.early_stopping_patience:
            break
    if last_checkpoint is None or last_metrics is None or not best_model:
        raise TrainingRuntimeError("training_did_not_complete")
    runtime.model.load_state_dict(runtime.safetensors.load(best_model))
    best_metrics = _evaluate(runtime, plan, audio_root, batch_profile.micro_batch_size)
    if abs(best_metrics.macro_f1 - best_metric) > 1e-12:
        raise TrainingRuntimeError("best_checkpoint_mismatch")
    return TrainingResult(
        best_epoch=best_epoch,
        epochs_completed=int(last_checkpoint.name.removeprefix("epoch-")) + 1,
        optimizer_steps=optimizer_steps,
        metrics=best_metrics,
        checkpoint_path=last_checkpoint,
    )


__all__ = [
    "BATCH_PROFILES",
    "BatchProfile",
    "EpochMetrics",
    "MemoryProbeAttempt",
    "MemoryProbeResult",
    "TrainingResult",
    "TrainingHistoryRecord",
    "TrainingRuntimeError",
    "append_training_history",
    "configure_full_model",
    "deterministic_crop_start",
    "deterministic_epoch_records",
    "optimizer_parameter_groups",
    "publish_provider_checkpoint",
    "run_cuda_training",
    "run_isolated_memory_probe",
    "run_memory_probe_attempt",
    "validate_cuda_runtime",
]
