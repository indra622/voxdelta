# XLS-R Partial Last-Four Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict `partial-last4` XLS-R training path, prove it on the existing 140/35 smoke, and advance to a deterministic 700/175 development subset only when the smoke gate passes.

**Architecture:** Extend the existing XLS-R profile with an explicit adaptation strategy while preserving the accepted `full` behavior. A focused trainable-parameter boundary freezes the whole model, re-enables only encoder layers 20–23 plus `projector` and `classifier`, and gives AdamW only those parameters. Partial checkpoints use a new strict provenance schema but still contain the complete ordinary XLS-R state dictionary, so the existing production provider topology remains unchanged.

**Tech Stack:** Python 3.12, Pydantic 2, PyTorch 2.13, Transformers 4.57, safetensors, scikit-learn, pytest, Ruff, strict mypy, uv, Apple MPS.

## Global Constraints

- Model ID: `facebook/wav2vec2-xls-r-300m`.
- Model revision: `1a640f32ac3e39899438a2931f9924c02f080a54`.
- Base weight SHA-256: `d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0`.
- Canonical labels remain `happiness`, `anger`, `disgust`, `fear`, `neutral`, `sadness`, `surprise` in that order.
- `partial-last4` trains encoder layers `(20, 21, 22, 23)` and top-level `projector` and `classifier` only.
- Feature extractor, feature projection, positional convolution, encoder layers `0`–`19`, and every non-allowlisted parameter remain frozen.
- Seed `622`, learning rate `2e-5`, ten epochs, warmup `0.1`, macro-F1 selection, patience two, inverse-frequency class weights, and 20-second deterministic center crop remain fixed.
- `partial-last4` uses micro-batch `2`, evaluation batch `2`, gradient accumulation `8`, and effective batch `16`.
- Gradient checkpointing is disabled; no automatic batch, layer, learning-rate, or strategy fallback is allowed.
- The existing `full` profile and schema-three checkpoint behavior remain accepted and unchanged.
- Checkpoints contain the complete model state dictionary, never a delta or adapter.
- Smoke uses the existing balanced 140-train/35-validation manifest. Development uses exactly 700 train and 175 validation items, balanced 100/25 per label.
- No test-split item may be selected, loaded, trained, or evaluated.
- Existing manifests, normalized WAVs, full-smoke checkpoints, and prior reports are immutable and never overwritten.
- User-facing errors and aggregate reports contain no transcript, item ID, audio path, probability vector, or per-item prediction.
- Stage A passes only if 35/35 validation completes, macro-F1 is strictly greater than `0.0357143`, at least two classes are predicted, and the checkpoint reloads through `Wav2VecEmotionProvider`.
- Stage B runs only after Stage A passes; it uses the same profile and must complete 175/175 with macro-F1 strictly greater than `0.0357143` and at least two predicted classes.

## File Map

- Modify `backend/src/voxdelta/evaluation/emotion_training.py`: strategy/profile contract, trainable allowlist, optimizer boundary, partial checkpoint metadata, and full-state publication.
- Modify `backend/scripts/train_emotion.py`: add and validate `--adaptation-strategy full|partial-last4`.
- Modify `backend/src/voxdelta/providers/_emotion_runtime.py`: accept schema-three full checkpoints and strict schema-four partial checkpoints.
- Modify `backend/tests/evaluation/test_emotion_training.py`: profile, freeze mask, optimizer, checkpoint, backend, and CLI coverage.
- Modify `backend/tests/providers/test_wav2vec_emotion.py`: partial checkpoint validation and provider reload coverage.
- Modify `backend/src/voxdelta/evaluation/emotion_experiment.py`: deterministic train/validation-only development manifest builder.
- Create `backend/scripts/build_emotion_development_manifest.py`: privacy-safe fixed 700/175 builder CLI.
- Modify `backend/tests/evaluation/test_emotion_experiment.py`: deterministic, balanced, non-overwriting, transcript-free development-manifest tests.
- Modify `data/README.md`: exact Stage A/Stage B commands, gate checks, output paths, and stop conditions.

---

### Task 1: Explicit Training Strategy and CLI Contract

**Files:**
- Modify: `backend/src/voxdelta/evaluation/emotion_training.py:42-89`
- Modify: `backend/scripts/train_emotion.py:1-95`
- Test: `backend/tests/evaluation/test_emotion_training.py:42-123`
- Test: `backend/tests/evaluation/test_emotion_training.py:785-940`

**Interfaces:**
- Consumes: the existing pinned XLS-R base path and accepted full-profile batch pairs.
- Produces: `AdaptationStrategy`, `Wav2VecTrainingProfile.adaptation_strategy`, and CLI flag `--adaptation-strategy full|partial-last4`.

- [ ] **Step 1: Write failing profile tests**

```python
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
        Wav2VecTrainingProfile.model_validate(
            {"base_model_path": tmp_path.resolve(), **metadata}
        )
```

- [ ] **Step 2: Run the profile tests and confirm RED**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_training.py -k 'partial_last4_profile' -q`

Expected: FAIL because `adaptation_strategy` is currently forbidden.

- [ ] **Step 3: Add the strategy field without changing full defaults**

```python
AdaptationStrategy = Literal["full", "partial-last4"]


class Wav2VecTrainingProfile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    architecture: Literal["wav2vec-xls-r"] = "wav2vec-xls-r"
    adaptation_strategy: AdaptationStrategy = "full"
    model_id: Literal["facebook/wav2vec2-xls-r-300m"] = (
        "facebook/wav2vec2-xls-r-300m"
    )
    base_model_path: Path
    model_revision: Literal[
        "1a640f32ac3e39899438a2931f9924c02f080a54"
    ] = WAV2VEC_MODEL_REVISION
    base_model_sha256: Literal[
        "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"
    ] = WAV2VEC_WEIGHTS_SHA256
    seed: Literal[622] = 622
    learning_rate: float = 2e-5
    train_batch_size: Literal[1, 2, 4, 8] = 8
    eval_batch_size: Literal[1, 2, 4, 8] = 8
    gradient_accumulation_steps: Literal[2, 4, 8, 16] = 2
    epochs: Literal[10] = 10
    warmup_ratio: float = 0.1
    evaluation_strategy: Literal["epoch"] = "epoch"
    selection_metric: Literal["macro_f1"] = "macro_f1"
    early_stopping_patience: Literal[2] = 2
    class_weighting: Literal["inverse-frequency"] = "inverse-frequency"

    @model_validator(mode="after")
    def exact_profile(self) -> Wav2VecTrainingProfile:
        accumulation_by_batch = {8: 2, 4: 4, 2: 8, 1: 16}
        if (
            not self.base_model_path.is_absolute()
            or self.eval_batch_size != self.train_batch_size
            or self.gradient_accumulation_steps
            != accumulation_by_batch[self.train_batch_size]
            or self.learning_rate != 2e-5
            or self.warmup_ratio != 0.1
            or (
                self.adaptation_strategy == "partial-last4"
                and self.train_batch_size != 2
            )
        ):
            raise ValueError("wav2vec training profile is fixed")
        return self
```

- [ ] **Step 4: Write failing CLI matrix tests**

```python
@pytest.mark.parametrize(
    "arguments",
    [
        ["--architecture", "emotion2vec-plus", "--adaptation-strategy", "partial-last4"],
        [
            "--architecture", "wav2vec-xls-r",
            "--adaptation-strategy", "partial-last4",
            "--base-model-path", "/base",
            "--micro-batch-size", "4",
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
            "--manifest", str(tmp_path / "manifest.jsonl"),
            "--output", str(tmp_path / "output"),
            "--base-model", "facebook/wav2vec2-xls-r-300m",
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
```

- [ ] **Step 5: Add and validate the CLI flag**

```python
parser.add_argument(
    "--adaptation-strategy",
    choices=("full", "partial-last4"),
    default="full",
)
```

In the XLS-R branch, construct the profile with:

```python
profile = Wav2VecTrainingProfile(
    adaptation_strategy=arguments.adaptation_strategy,
    base_model_path=arguments.base_model_path,
    train_batch_size=arguments.micro_batch_size,
    eval_batch_size=arguments.micro_batch_size,
    gradient_accumulation_steps={8: 2, 4: 4, 2: 8, 1: 16}[
        arguments.micro_batch_size
    ],
    seed=arguments.seed,
)
```

Before the architecture branches, reject:

```python
if arguments.architecture == "emotion2vec-plus" and arguments.adaptation_strategy != "full":
    raise TrainingError("invalid_training_profile")
```

- [ ] **Step 6: Run focused tests and commit**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_training.py -k 'profile or cli' -q`

Expected: all selected tests pass, including existing full-profile cases.

```bash
git add backend/src/voxdelta/evaluation/emotion_training.py backend/scripts/train_emotion.py backend/tests/evaluation/test_emotion_training.py
git commit -m "feat: add XLS-R partial training profile"
```

### Task 2: Trainable Allowlist and Optimizer Boundary

**Files:**
- Modify: `backend/src/voxdelta/evaluation/emotion_training.py:790-930`
- Test: `backend/tests/evaluation/test_emotion_training.py`

**Interfaces:**
- Consumes: `Wav2VecTrainingProfile.adaptation_strategy` and the loaded Transformers model.
- Produces: `Wav2VecTrainableSummary`, `configure_wav2vec_trainable_parameters(model, strategy)`, and `build_wav2vec_optimizer(torch, model, profile)`.

- [ ] **Step 1: Write failing freeze-mask and failure-closed tests**

```python
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
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_training.py -k 'partial_last4_enables or unexpected_encoder_depth' -q`

Expected: FAIL because the allowlist boundary does not exist.

- [ ] **Step 3: Implement the exact allowlist and summary**

```python
PARTIAL_LAST4_ENCODER_LAYERS = (20, 21, 22, 23)
PARTIAL_LAST4_PREFIXES = (
    "wav2vec2.encoder.layers.20.",
    "wav2vec2.encoder.layers.21.",
    "wav2vec2.encoder.layers.22.",
    "wav2vec2.encoder.layers.23.",
    "projector.",
    "classifier.",
)


@dataclass(frozen=True, slots=True)
class Wav2VecTrainableSummary:
    parameters: tuple[Any, ...]
    encoder_layers: tuple[int, ...]
    module_prefixes: tuple[str, ...]
    trainable_parameter_count: int
    total_parameter_count: int


def configure_wav2vec_trainable_parameters(
    model: Any, strategy: AdaptationStrategy
) -> Wav2VecTrainableSummary:
    named = tuple(model.named_parameters())
    if not named:
        raise TrainingError("training_failed")
    total_count = sum(parameter.numel() for _, parameter in named)
    if strategy == "full":
        parameters = tuple(parameter for _, parameter in named)
        return Wav2VecTrainableSummary(
            parameters=parameters,
            encoder_layers=(),
            module_prefixes=(),
            trainable_parameter_count=sum(parameter.numel() for parameter in parameters),
            total_parameter_count=total_count,
        )

    try:
        layers = model.wav2vec2.encoder.layers
        if len(layers) != 24:
            raise ValueError
    except Exception:
        raise TrainingError("training_failed") from None

    seen_prefixes: set[str] = set()
    trainable: list[Any] = []
    for name, parameter in named:
        parameter.requires_grad = False
        matches = tuple(prefix for prefix in PARTIAL_LAST4_PREFIXES if name.startswith(prefix))
        if len(matches) > 1:
            raise TrainingError("training_failed")
        if matches:
            parameter.requires_grad = True
            trainable.append(parameter)
            seen_prefixes.add(matches[0])

    if seen_prefixes != set(PARTIAL_LAST4_PREFIXES) or not trainable:
        raise TrainingError("training_failed")
    if any(
        parameter.requires_grad
        and not name.startswith(PARTIAL_LAST4_PREFIXES)
        for name, parameter in named
    ):
        raise TrainingError("training_failed")

    return Wav2VecTrainableSummary(
        parameters=tuple(trainable),
        encoder_layers=PARTIAL_LAST4_ENCODER_LAYERS,
        module_prefixes=PARTIAL_LAST4_PREFIXES,
        trainable_parameter_count=sum(parameter.numel() for parameter in trainable),
        total_parameter_count=total_count,
    )
```

- [ ] **Step 4: Write the failing optimizer-input test**

```python
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
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_ids == {id(parameter) for parameter in summary.parameters}
    assert optimizer_ids == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
```

- [ ] **Step 5: Build the optimizer through the allowlist boundary**

```python
def build_wav2vec_optimizer(
    torch: Any, model: Any, profile: Wav2VecTrainingProfile
) -> tuple[Any, Wav2VecTrainableSummary]:
    summary = configure_wav2vec_trainable_parameters(
        model, profile.adaptation_strategy
    )
    optimizer_parameters = (
        summary.parameters
        if profile.adaptation_strategy == "partial-last4"
        else tuple(model.parameters())
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=profile.learning_rate
    )
    return optimizer, summary
```

Replace the direct AdamW construction in `_train_wav2vec` with:

```python
optimizer, trainable_summary = build_wav2vec_optimizer(torch, model, profile)
```

Retain `model.state_dict()` when capturing `best_state`; do not filter it by `requires_grad`.

- [ ] **Step 6: Run focused tests and commit**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_training.py -k 'partial_last4 or wav2vec_optimizer' -q`

Expected: all selected tests pass and the existing full training tests remain green.

```bash
git add backend/src/voxdelta/evaluation/emotion_training.py backend/tests/evaluation/test_emotion_training.py
git commit -m "feat: freeze XLS-R below final four layers"
```

### Task 3: Strict Partial Checkpoint Provenance and Provider Reload

**Files:**
- Modify: `backend/src/voxdelta/evaluation/emotion_training.py:458-595`
- Modify: `backend/src/voxdelta/evaluation/emotion_training.py:815-930`
- Modify: `backend/src/voxdelta/providers/_emotion_runtime.py:35-230`
- Test: `backend/tests/evaluation/test_emotion_training.py`
- Test: `backend/tests/providers/test_wav2vec_emotion.py`

**Interfaces:**
- Consumes: `Wav2VecTrainableSummary` and the complete safetensors state produced by Task 2.
- Produces: schema-four partial checkpoint metadata; schema-three full compatibility; `CheckpointInfo` partial provenance fields.

- [ ] **Step 1: Write failing partial checkpoint publication tests**

```python
def _partial_payload() -> CheckpointPayload:
    return CheckpointPayload(
        architecture="wav2vec-xls-r",
        model_id="facebook/wav2vec2-xls-r-300m",
        weights=b"complete-state",
        metrics={"macro_f1": 0.2},
        validation_hash="a" * 64,
        model_revision=WAV2VEC_REVISION,
        base_model_sha256=WAV2VEC_WEIGHTS_SHA256,
        class_weighting="inverse-frequency",
        class_weights=(1.0,) * 7,
        adaptation_strategy="partial-last4",
        trainable_encoder_layers=(20, 21, 22, 23),
        trainable_module_prefixes=(
            "wav2vec2.encoder.layers.20.",
            "wav2vec2.encoder.layers.21.",
            "wav2vec2.encoder.layers.22.",
            "wav2vec2.encoder.layers.23.",
            "projector.",
            "classifier.",
        ),
        trainable_parameter_count=10,
        total_parameter_count=100,
    )


def test_partial_checkpoint_publishes_strict_schema_four(tmp_path: Path) -> None:
    output = tmp_path / "partial"
    publish_checkpoint(output, _partial_payload())
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))

    assert config["schema_version"] == "4"
    assert config["adaptation_strategy"] == "partial-last4"
    assert config["trainable_encoder_layers"] == [20, 21, 22, 23]
    assert config["trainable_parameter_count"] == 10
    assert config["total_parameter_count"] == 100
    assert (output / "model.safetensors").read_bytes() == b"complete-state"


@pytest.mark.parametrize(
    "changes",
    [
        {"trainable_encoder_layers": (19, 20, 21, 22)},
        {"trainable_module_prefixes": ("classifier.",)},
        {"trainable_parameter_count": 0},
        {"total_parameter_count": 10, "trainable_parameter_count": 10},
    ],
)
def test_partial_checkpoint_rejects_altered_provenance(
    changes: dict[str, object]
) -> None:
    values = _partial_payload().model_dump()
    values.update(changes)
    with pytest.raises(ValidationError):
        CheckpointPayload.model_validate(values)
```

- [ ] **Step 2: Run checkpoint tests and confirm RED**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_training.py -k 'partial_checkpoint' -q`

Expected: FAIL because the partial provenance fields and schema four do not exist.

- [ ] **Step 3: Add conditional partial metadata to `CheckpointPayload`**

```python
adaptation_strategy: Literal["partial-last4"] | None = None
trainable_encoder_layers: tuple[int, ...] | None = None
trainable_module_prefixes: tuple[str, ...] | None = None
trainable_parameter_count: int | None = Field(default=None, gt=0)
total_parameter_count: int | None = Field(default=None, gt=0)
```

Reject all five partial fields in the emotion2vec branch. At the end of the XLS-R branch
in `valid_metadata`, enforce:

```python
partial_values = (
    self.trainable_encoder_layers,
    self.trainable_module_prefixes,
    self.trainable_parameter_count,
    self.total_parameter_count,
)
if self.adaptation_strategy is None:
    if any(value is not None for value in partial_values):
        raise ValueError("wav2vec metadata is incomplete")
elif (
    self.adaptation_strategy != "partial-last4"
    or self.trainable_encoder_layers != PARTIAL_LAST4_ENCODER_LAYERS
    or self.trainable_module_prefixes != PARTIAL_LAST4_PREFIXES
    or self.trainable_parameter_count is None
    or self.total_parameter_count is None
    or self.trainable_parameter_count >= self.total_parameter_count
):
    raise ValueError("wav2vec metadata is incomplete")
```

Publish schema four and the six partial fields only when `adaptation_strategy` is set; otherwise keep the existing schema-three key set byte-for-byte.

- [ ] **Step 4: Populate provenance from the real training summary**

Extend the `_train_wav2vec` return construction with:

```python
adaptation_strategy=(
    "partial-last4"
    if profile.adaptation_strategy == "partial-last4"
    else None
),
trainable_encoder_layers=(
    trainable_summary.encoder_layers
    if profile.adaptation_strategy == "partial-last4"
    else None
),
trainable_module_prefixes=(
    trainable_summary.module_prefixes
    if profile.adaptation_strategy == "partial-last4"
    else None
),
trainable_parameter_count=(
    trainable_summary.trainable_parameter_count
    if profile.adaptation_strategy == "partial-last4"
    else None
),
total_parameter_count=(
    trainable_summary.total_parameter_count
    if profile.adaptation_strategy == "partial-last4"
    else None
),
```

- [ ] **Step 5: Write failing runtime and provider tests**

```python
def test_runtime_accepts_exact_partial_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(
        tmp_path / "checkpoint",
        adaptation_strategy="partial-last4",
    )
    info = validate_checkpoint(
        checkpoint,
        architecture="wav2vec-xls-r",
        model_id="facebook/wav2vec2-xls-r-300m",
    )

    assert info.adaptation_strategy == "partial-last4"
    assert info.trainable_encoder_layers == (20, 21, 22, 23)
    assert info.trainable_parameter_count > 0
    assert info.total_parameter_count > info.trainable_parameter_count


@pytest.mark.parametrize(
    "key,value",
    [
        ("adaptation_strategy", "full"),
        ("trainable_encoder_layers", [19, 20, 21, 22]),
        ("trainable_module_prefixes", ["classifier."]),
        ("trainable_parameter_count", 0),
        ("model_revision", "wrong"),
        ("base_model_sha256", "b" * 64),
    ],
)
def test_runtime_rejects_altered_partial_provenance(
    tmp_path: Path, key: str, value: object
) -> None:
    checkpoint = _checkpoint(
        tmp_path / "checkpoint",
        adaptation_strategy="partial-last4",
    )
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[key] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ProviderError) as raised:
        Wav2VecEmotionProvider(
            checkpoint,
            model_factory=FakeFactory(FakePredictor([[0.0] * 7])),
        )
    assert raised.value.code == "invalid_local_checkpoint"
```

Extend the provider test helper without changing its default schema-three output:

```python
def _checkpoint(
    path: Path, *, adaptation_strategy: str | None = None
) -> Path:
    path.mkdir()
    config: dict[str, object] = {
        "schema_version": "3",
        "architecture": "wav2vec-xls-r",
        "model_id": "facebook/wav2vec2-xls-r-300m",
        "labels": list(LABELS),
        "model_revision": WAV2VEC_REVISION,
        "base_model_sha256": WAV2VEC_WEIGHTS_SHA256,
        "class_weighting": "inverse-frequency",
        "class_weights": [1.0] * 7,
    }
    if adaptation_strategy == "partial-last4":
        config.update(
            {
                "schema_version": "4",
                "adaptation_strategy": "partial-last4",
                "trainable_encoder_layers": [20, 21, 22, 23],
                "trainable_module_prefixes": [
                    "wav2vec2.encoder.layers.20.",
                    "wav2vec2.encoder.layers.21.",
                    "wav2vec2.encoder.layers.22.",
                    "wav2vec2.encoder.layers.23.",
                    "projector.",
                    "classifier.",
                ],
                "trainable_parameter_count": 10,
                "total_parameter_count": 100,
            }
        )
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "label_mapping.json").write_text(
        json.dumps({str(index): label for index, label in enumerate(LABELS)}),
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps({"macro_f1": 0.7, "validation_hash": "a" * 64}),
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(b"fake-weights")
    return path
```

- [ ] **Step 6: Parse schema three and schema four explicitly**

Extend `CheckpointInfo` with optional partial fields. In `validate_checkpoint`, keep the current schema-three XLS-R branch unchanged and add a schema-four branch whose exact key set is:

```python
partial_keys = {
    "adaptation_strategy",
    "trainable_encoder_layers",
    "trainable_module_prefixes",
    "trainable_parameter_count",
    "total_parameter_count",
}
```

The schema-four branch accepts only:

```python
schema_version == "4"
and set(config) == expected_config_keys | {
    "class_weighting", "class_weights", *partial_keys
}
and config["adaptation_strategy"] == "partial-last4"
and config["trainable_encoder_layers"] == [20, 21, 22, 23]
and config["trainable_module_prefixes"] == list(PARTIAL_LAST4_PREFIXES)
and type(config["trainable_parameter_count"]) is int
and type(config["total_parameter_count"]) is int
and 0 < config["trainable_parameter_count"] < config["total_parameter_count"]
```

Return those values on `CheckpointInfo`. `Wav2VecEmotionProvider` continues to reconstruct the ordinary local base model and load the complete `model.safetensors`; it does not reproduce the freeze mask.

In `train_from_manifest`, reject a payload whose strategy metadata does not match the
requested XLS-R profile:

```python
if isinstance(profile, Wav2VecTrainingProfile):
    expected_strategy = (
        "partial-last4"
        if profile.adaptation_strategy == "partial-last4"
        else None
    )
    if payload.adaptation_strategy != expected_strategy:
        raise ValueError
```

Add one unit case where a partial profile receives an otherwise valid schema-three/full
payload from an injected backend and assert `TrainingError("training_failed")` with no
output directory.

- [ ] **Step 7: Run focused tests and commit**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_training.py tests/providers/test_wav2vec_emotion.py -q`

Expected: both files pass, including existing schema-three full checkpoint cases.

```bash
git add backend/src/voxdelta/evaluation/emotion_training.py backend/src/voxdelta/providers/_emotion_runtime.py backend/tests/evaluation/test_emotion_training.py backend/tests/providers/test_wav2vec_emotion.py
git commit -m "feat: validate partial XLS-R checkpoint provenance"
```

### Task 4: Deterministic 700/175 Development Manifest

**Files:**
- Modify: `backend/src/voxdelta/evaluation/emotion_experiment.py:35-225`
- Create: `backend/scripts/build_emotion_development_manifest.py`
- Modify: `backend/tests/evaluation/test_emotion_experiment.py:80-175`

**Interfaces:**
- Consumes: the full trusted emotion manifest and the existing seed-622 `_rank` rule.
- Produces: `DevelopmentManifestSummary`, `build_partial_last4_development_manifest(source, output)`, and a fixed builder CLI.

- [ ] **Step 1: Write failing deterministic manifest tests**

```python
def test_build_development_manifest_is_balanced_deterministic_and_test_free(
    tmp_path: Path,
) -> None:
    source = _write_source_manifest(
        tmp_path / "source",
        train_count=101,
        validation_count=26,
        test_count=2,
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    summary = build_partial_last4_development_manifest(source, first)
    build_partial_last4_development_manifest(source, second)

    assert summary.total_count == 875
    assert summary.split_counts == {"train": 700, "validation": 175}
    selected = load_manifest(first)
    assert Counter((item.split, item.emotion) for item in selected) == Counter(
        {
            **{("train", label): 100 for label in LABELS},
            **{("validation", label): 25 for label in LABELS},
        }
    )
    assert all(item.split != "test" for item in selected)
    assert {item.transcript for item in selected} == {""}
    assert first.read_bytes() == second.read_bytes()
    assert stat.S_IMODE(first.stat().st_mode) == 0o600


def test_build_development_manifest_refuses_existing_output(tmp_path: Path) -> None:
    source = _write_source_manifest(
        tmp_path / "source",
        train_count=101,
        validation_count=26,
        test_count=2,
    )
    output = tmp_path / "existing.jsonl"
    output.write_text("owner-data", encoding="utf-8")

    with pytest.raises(ValueError, match="development_manifest_publication_failed"):
        build_partial_last4_development_manifest(source, output)
    assert output.read_text(encoding="utf-8") == "owner-data"
```

Extend the test module import list with
`build_partial_last4_development_manifest`; do not change production smoke selection.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_experiment.py -k 'development_manifest' -q`

Expected: FAIL because the development builder does not exist.

- [ ] **Step 3: Implement the fixed development builder separately from smoke selection**

```python
class DevelopmentManifestSummary(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    total_count: Literal[875]
    split_counts: dict[Literal["train", "validation"], int]
    label_counts: dict[EmotionLabel, int]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_partial_last4_development_manifest(
    source: Path, output: Path
) -> DevelopmentManifestSummary:
    if not source.is_absolute() or not output.is_absolute():
        raise ValueError("invalid_development_manifest")
    try:
        items = load_manifest(source)
        validate_disjoint_splits(items)
        if any(
            item.source != "emotion" or item.emotion not in CANONICAL_LABELS
            for item in items
        ):
            raise ValueError
        selected: list[DatasetItem] = []
        for split, count in (("train", 100), ("validation", 25)):
            for label in CANONICAL_LABELS:
                candidates = sorted(
                    (
                        item
                        for item in items
                        if item.source == "emotion"
                        and item.split == split
                        and item.emotion == label
                    ),
                    key=lambda item: _rank(item, 622),
                )
                if len(candidates) < count:
                    raise ValueError
                selected.extend(
                    item.model_copy(update={"transcript": ""})
                    for item in candidates[:count]
                )
    except Exception:
        raise ValueError("invalid_development_manifest") from None

    payload = b"".join(
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
        for item in sorted(selected, key=lambda item: item.id)
    )
    _publish_private_file(
        output, payload, "development_manifest_publication_failed"
    )
    return DevelopmentManifestSummary(
        total_count=875,
        split_counts={"train": 700, "validation": 175},
        label_counts={label: 125 for label in CANONICAL_LABELS},
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )
```

- [ ] **Step 4: Add the fixed privacy-safe CLI and its test**

```python
"""Build the fixed train/validation-only partial-last-four development manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.evaluation.emotion_experiment import (
    build_partial_last4_development_manifest,
)


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    try:
        arguments = parser.parse_args(argv)
        summary = build_partial_last4_development_manifest(
            arguments.manifest, arguments.output
        )
    except Exception:
        print("development manifest failed", file=sys.stderr)
        return 2
    print(f"development manifest: {summary.total_count} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The CLI test invokes a missing private-looking path and asserts only `development manifest failed` appears on stderr.

- [ ] **Step 5: Run focused tests and commit**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_experiment.py -k 'development_manifest' -q`

Expected: all selected tests pass.

```bash
git add backend/src/voxdelta/evaluation/emotion_experiment.py backend/scripts/build_emotion_development_manifest.py backend/tests/evaluation/test_emotion_experiment.py
git commit -m "feat: build balanced XLS-R development subset"
```

### Task 5: Runbook, Quality Gates, and Repository Verification

**Files:**
- Modify: `data/README.md:129-190`
- Test: repository verification commands only.

**Interfaces:**
- Consumes: the new strategy CLI, schema-four provider path, development builder, and existing evaluation CLI.
- Produces: exact non-overwriting Stage A/Stage B commands and machine-checkable aggregate gates.

- [ ] **Step 1: Document exact Stage A commands**

```bash
cd /Volumes/nvme1/codes/voxdelta/backend
VOXDELTA_DATA=/Volumes/nvme1/codes/voxdelta/data

uv run python scripts/train_emotion.py \
  --manifest "$VOXDELTA_DATA/manifests/emotion-smoke.jsonl" \
  --output "$VOXDELTA_DATA/models/wav2vec-xls-r-300m-partial-last4-smoke" \
  --architecture wav2vec-xls-r \
  --adaptation-strategy partial-last4 \
  --base-model facebook/wav2vec2-xls-r-300m \
  --base-model-path "$VOXDELTA_DATA/models/base/wav2vec2-xls-r-300m-1a640f3" \
  --micro-batch-size 2 \
  --seed 622

uv run python scripts/evaluate_emotion_checkpoint.py \
  --manifest "$VOXDELTA_DATA/manifests/emotion-smoke.jsonl" \
  --checkpoint "$VOXDELTA_DATA/models/wav2vec-xls-r-300m-partial-last4-smoke" \
  --base-model-path "$VOXDELTA_DATA/models/base/wav2vec2-xls-r-300m-1a640f3" \
  --output "$VOXDELTA_DATA/benchmarks/wav2vec-xls-r-300m-partial-last4-smoke-validation.json" \
  --architecture wav2vec-xls-r \
  --device mps \
  --split validation
```

Document that both targets must be absent before starting and that any failure stops the workflow without retrying a different profile.

- [ ] **Step 2: Document the exact Stage A gate**

```bash
uv run python -c '
import json
from pathlib import Path
r=json.loads(Path("/Volumes/nvme1/codes/voxdelta/data/benchmarks/wav2vec-xls-r-300m-partial-last4-smoke-validation.json").read_text())
m=r["confusion_matrix"]
predicted=sum(any(row[column] for row in m) for column in range(7))
assert r["item_count"] == 35
assert r["completed_count"] == 35
assert r["macro_f1"] > 0.0357143
assert predicted >= 2
print("partial-last4 stage A passed")
'
```

Expected: `partial-last4 stage A passed`. Any assertion failure means Stage B must not run.

- [ ] **Step 3: Document exact Stage B commands after the gate**

```bash
uv run python scripts/build_emotion_development_manifest.py \
  --manifest "$VOXDELTA_DATA/manifests/emotion.jsonl" \
  --output "$VOXDELTA_DATA/manifests/emotion-partial-last4-development.jsonl"

uv run python scripts/train_emotion.py \
  --manifest "$VOXDELTA_DATA/manifests/emotion-partial-last4-development.jsonl" \
  --output "$VOXDELTA_DATA/models/wav2vec-xls-r-300m-partial-last4-development" \
  --architecture wav2vec-xls-r \
  --adaptation-strategy partial-last4 \
  --base-model facebook/wav2vec2-xls-r-300m \
  --base-model-path "$VOXDELTA_DATA/models/base/wav2vec2-xls-r-300m-1a640f3" \
  --micro-batch-size 2 \
  --seed 622

uv run python scripts/evaluate_emotion_checkpoint.py \
  --manifest "$VOXDELTA_DATA/manifests/emotion-partial-last4-development.jsonl" \
  --checkpoint "$VOXDELTA_DATA/models/wav2vec-xls-r-300m-partial-last4-development" \
  --base-model-path "$VOXDELTA_DATA/models/base/wav2vec2-xls-r-300m-1a640f3" \
  --output "$VOXDELTA_DATA/benchmarks/wav2vec-xls-r-300m-partial-last4-development-validation.json" \
  --architecture wav2vec-xls-r \
  --device mps \
  --split validation
```

Add the same gate with `item_count == completed_count == 175`, macro-F1 `> 0.0357143`, and predicted-class count `>= 2`.

- [ ] **Step 4: Run the complete repository gate**

Run:

```bash
cd /Volumes/nvme1/codes/voxdelta/backend
uv run pytest -q
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run mypy src scripts
uv lock --check
cd ..
git diff --check
```

Expected: every command exits `0`; pytest reports no failures; Ruff reports no errors and no formatting changes; mypy reports success; the lock is current; Git reports no whitespace errors.

- [ ] **Step 5: Self-review spec coverage and commit the runbook**

Confirm there are no placeholder markers, incompatible signatures, test-split commands,
overwrite paths, or automatic fallback instructions in the implementation or runbook.

```bash
git add data/README.md
git commit -m "docs: add partial XLS-R experiment gates"
```

### Task 6: Execute Stage A and Conditionally Stage B

**Files:**
- Produce ignored local artifacts only under `/Volumes/nvme1/codes/voxdelta/data`.
- Modify `data/README.md` only to record aggregate accepted evidence after each completed stage.

**Interfaces:**
- Consumes: the verified implementation from Tasks 1–5 and immutable source/base artifacts.
- Produces: aggregate checkpoint/report evidence for Stage A and, only on pass, Stage B.

- [ ] **Step 1: Verify immutable inputs and absent outputs**

Run:

```bash
cd /Volumes/nvme1/codes/voxdelta
shasum -a 256 \
  data/manifests/emotion.jsonl \
  data/manifests/emotion-smoke.jsonl \
  data/benchmarks/wav2vec-xls-r-300m-smoke-validation.json
test ! -e data/models/wav2vec-xls-r-300m-partial-last4-smoke
test ! -e data/benchmarks/wav2vec-xls-r-300m-partial-last4-smoke-validation.json
test ! -e data/manifests/emotion-partial-last4-development.jsonl
test ! -e data/models/wav2vec-xls-r-300m-partial-last4-development
test ! -e data/benchmarks/wav2vec-xls-r-300m-partial-last4-development-validation.json
```

Expected: three digests print and every `test ! -e` exits `0`. Save the three printed digests in the local experiment log before training.

- [ ] **Step 2: Run Stage A and the documented gate**

Run the Task 5 Stage A training, evaluation, and gate commands unchanged. Record elapsed time, peak RSS, trainable/total parameter counts, macro-F1, per-label F1, confusion matrix, predicted-class count, checkpoint digest, and report digest in the local experiment log.

Expected: either the gate prints `partial-last4 stage A passed`, or the workflow stops without creating any development artifact.

- [ ] **Step 3: Recheck immutable fingerprints**

Re-run the exact `shasum -a 256` command from Step 1 and compare all three lines byte-for-byte with the pre-run log.

Expected: all three digests are unchanged.

- [ ] **Step 4: Run Stage B only after a Stage A pass**

Run the Task 5 development builder, training, evaluation, and 175-item gate commands unchanged. Record the same aggregate evidence as Stage A.

Expected: the development manifest contains exactly 875 items and no test items; the gate either passes or the partial-last-four experiment stops without opening the test split.

- [ ] **Step 5: Record aggregate evidence and commit documentation**

Append only aggregate results and artifact digests to `data/README.md`; do not add local paths beyond the already documented canonical data root, item IDs, transcripts, probabilities, or individual predictions.

```bash
git add data/README.md
git commit -m "docs: record partial XLS-R experiment evidence"
```

The final handoff states whether Stage A and Stage B passed, confirms that the test split remained sealed, and recommends either a separately designed larger local run after a Stage B pass or a separately designed LoRA/no-training baseline after failure.
