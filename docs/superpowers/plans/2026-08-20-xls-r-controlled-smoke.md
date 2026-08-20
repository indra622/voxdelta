# XLS-R Controlled Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin XLS-R 300M to verified local NVMe artifacts, support memory-safe effective-batch-16 profiles, and complete one local-only balanced smoke training plus 35-item validation evaluation.

**Architecture:** A focused `wav2vec_base` module owns download, integrity, permissions, and local-path validation. Training and inference consume that verified directory explicitly with Transformers network fallback disabled; XLS-R checkpoints carry the revision and base-weight digest. The smoke starts at micro-batch four and retries in a fresh process at two or one only after an MPS OOM.

**Tech Stack:** Python 3.12, Pydantic 2, PyTorch 2.13, Transformers 4.57, safetensors, pytest, Ruff, mypy, uv, Apple MPS.

## Global Constraints

- Model ID: `facebook/wav2vec2-xls-r-300m`.
- Revision: `1a640f32ac3e39899438a2931f9924c02f080a54`.
- `config.json`: 1,568 bytes, SHA-256 `0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac`.
- `preprocessor_config.json`: 212 bytes, SHA-256 `a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335`.
- `pytorch_model.bin`: 1,269,737,156 bytes, SHA-256 `d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0`.
- Base files and generated outputs live only below the canonical `/Volumes/nvme1/codes/voxdelta/data` tree, not the worktree data shell.
- Directories are mode `0700`; files are mode `0600`; existing targets are never overwritten.
- Allowed `(micro-batch, gradient accumulation)` pairs are `(8,2)`, `(4,4)`, `(2,8)`, `(1,16)`; evaluation batch equals micro-batch.
- Seed 622, learning rate `2e-5`, ten epochs, warmup `0.1`, macro-F1 selection, patience two, and train-only inverse-frequency weights remain fixed.
- Smoke training uses 140 train and 35 validation examples. No test-split item is evaluated.
- The 35 smoke-test members were exposed by the prior emotion2vec integration run; a future
  final comparison must exclude those 35 or freeze a new untouched holdout from the 3,585
  remaining unobserved test items.
- User-facing failures contain stable codes only, never exception text, item IDs, transcripts, paths, probabilities, or predictions.
- Full training, resumable epoch checkpoints, and final test design are out of scope.

## File Map

- Create `backend/src/voxdelta/evaluation/wav2vec_base.py`: pinned artifact identities, atomic preparation, and strict local validation.
- Create `backend/scripts/prepare_wav2vec_base.py`: privacy-safe preparation CLI.
- Create `backend/tests/evaluation/test_wav2vec_base.py`: preparation and validation contract tests.
- Modify `backend/src/voxdelta/evaluation/emotion_training.py`: controlled batches, local-only model load, and checkpoint provenance.
- Modify `backend/scripts/train_emotion.py`: XLS-R base-path and micro-batch CLI flags.
- Modify `backend/src/voxdelta/providers/_emotion_runtime.py`: strict XLS-R checkpoint provenance parsing.
- Modify `backend/src/voxdelta/providers/wav2vec_emotion.py`: verified local-only provider construction.
- Modify `backend/src/voxdelta/evaluation/emotion_experiment.py`: pass the base path into the default XLS-R provider.
- Modify `backend/scripts/evaluate_emotion_checkpoint.py`: XLS-R base-path CLI flag.
- Modify `backend/tests/evaluation/test_emotion_training.py`: profile, local load, payload, and training CLI tests.
- Modify `backend/tests/providers/test_wav2vec_emotion.py`: local base fixture and provider provenance tests.
- Modify `backend/tests/evaluation/test_emotion_experiment.py`: evaluator and CLI base-path tests.
- Modify `data/README.md`: exact prepare, smoke train, fallback, and validation commands.

---

### Task 1: Pinned Base-Model Preparation Boundary

**Files:**
- Create: `backend/src/voxdelta/evaluation/wav2vec_base.py`
- Create: `backend/scripts/prepare_wav2vec_base.py`
- Test: `backend/tests/evaluation/test_wav2vec_base.py`

**Interfaces:**
- Consumes: an absolute, nonexistent output directory and an injectable `Downloader`.
- Produces: `PreparedWav2VecBase`, `prepare_wav2vec_base(output, *, downloader=...)`, and `validate_wav2vec_base(path)`.

- [ ] **Step 1: Write failing identity and validation tests**

```python
def test_pinned_artifacts_have_exact_identity() -> None:
    assert WAV2VEC_MODEL_ID == "facebook/wav2vec2-xls-r-300m"
    assert WAV2VEC_MODEL_REVISION == "1a640f32ac3e39899438a2931f9924c02f080a54"
    assert {item.name: (item.size, item.sha256) for item in WAV2VEC_ARTIFACTS} == {
        "config.json": (1568, "0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac"),
        "preprocessor_config.json": (212, "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335"),
        "pytorch_model.bin": (1269737156, "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"),
    }

def test_prepare_is_atomic_private_and_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        "config.json": b"config",
        "preprocessor_config.json": b"preprocessor",
        "pytorch_model.bin": b"weights",
    }
    artifacts = tuple(
        Wav2VecArtifact(name, len(payload), hashlib.sha256(payload).hexdigest())
        for name, payload in payloads.items()
    )
    monkeypatch.setattr(wav2vec_base, "WAV2VEC_ARTIFACTS", artifacts)
    prepared = prepare_wav2vec_base(
        tmp_path / "base",
        downloader=lambda url, output: output.write_bytes(payloads[Path(url).name]),
    )
    assert prepared.path == (tmp_path / "base").resolve()
    assert stat.S_IMODE(prepared.path.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in prepared.path.iterdir())
    assert validate_wav2vec_base(prepared.path) == prepared
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `cd backend && uv run pytest tests/evaluation/test_wav2vec_base.py -q`

Expected: collection fails because `voxdelta.evaluation.wav2vec_base` does not exist.

- [ ] **Step 3: Implement the pinned artifact module**

```python
WAV2VEC_MODEL_ID = "facebook/wav2vec2-xls-r-300m"
WAV2VEC_MODEL_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"

@dataclass(frozen=True, slots=True)
class Wav2VecArtifact:
    name: str
    size: int
    sha256: str

@dataclass(frozen=True, slots=True)
class PreparedWav2VecBase:
    path: Path
    model_id: str = WAV2VEC_MODEL_ID
    revision: str = WAV2VEC_MODEL_REVISION
    weights_sha256: str = WAV2VEC_WEIGHTS_SHA256

def validate_wav2vec_base(path: str | Path) -> PreparedWav2VecBase:
    candidate = _absolute_private_directory(path)
    if {item.name for item in candidate.iterdir()} != {item.name for item in WAV2VEC_ARTIFACTS}:
        raise ValueError("invalid_wav2vec_base")
    for artifact in WAV2VEC_ARTIFACTS:
        file = candidate / artifact.name
        if file.is_symlink() or stat.S_IMODE(file.stat().st_mode) != 0o600:
            raise ValueError("invalid_wav2vec_base")
        if file.stat().st_size != artifact.size or _sha256(file) != artifact.sha256:
            raise ValueError("invalid_wav2vec_base")
    return PreparedWav2VecBase(candidate)

def prepare_wav2vec_base(
    output: str | Path, *, downloader: Downloader = _download
) -> PreparedWav2VecBase:
    target = _absolute_nonexistent_target(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        os.chmod(staging, 0o700)
        for artifact in WAV2VEC_ARTIFACTS:
            destination = staging / artifact.name
            downloader(_revision_url(artifact.name), destination)
            os.chmod(destination, 0o600)
        validate_wav2vec_base(staging)
        os.replace(staging, target)
        return validate_wav2vec_base(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError("wav2vec_base_preparation_failed") from None
```

The helper implementations reject relative paths, `..`, symlink components, non-private
metadata, unexpected files, hash/size mismatch, and existing targets. `_download` uses
`urllib.request.urlopen` with a finite timeout and streams to a newly created file.

- [ ] **Step 4: Add failure, traversal, cleanup, and CLI tests**

```python
@pytest.mark.parametrize("bad", [Path("relative"), Path("../escape")])
def test_prepare_rejects_untrusted_output(bad: Path) -> None:
    with pytest.raises(ValueError, match="wav2vec_base_preparation_failed"):
        prepare_wav2vec_base(bad, downloader=lambda _url, _path: None)

def test_failed_download_leaves_no_final_or_staging(tmp_path: Path) -> None:
    def fail(_url: str, _path: Path) -> None:
        raise OSError("private detail")
    with pytest.raises(ValueError, match="wav2vec_base_preparation_failed"):
        prepare_wav2vec_base(tmp_path / "base", downloader=fail)
    assert list(tmp_path.iterdir()) == []
```

The CLI calls `prepare_wav2vec_base(Path(args.output))`, prints
`wav2vec base prepared` on success, and prints only
`wav2vec_base_error: wav2vec_base_preparation_failed` on failure.

- [ ] **Step 5: Run tests and commit**

Run: `cd backend && uv run pytest tests/evaluation/test_wav2vec_base.py -q`

Expected: all focused tests pass.

```bash
git add backend/src/voxdelta/evaluation/wav2vec_base.py backend/scripts/prepare_wav2vec_base.py backend/tests/evaluation/test_wav2vec_base.py
git commit -m "feat: pin and prepare XLS-R base artifacts"
```

### Task 2: Strict XLS-R Checkpoint Provenance

**Files:**
- Modify: `backend/src/voxdelta/evaluation/emotion_training.py:436-548`
- Modify: `backend/src/voxdelta/providers/_emotion_runtime.py:42-172`
- Test: `backend/tests/evaluation/test_emotion_training.py`
- Test: `backend/tests/providers/test_wav2vec_emotion.py`

**Interfaces:**
- Consumes: `WAV2VEC_MODEL_REVISION` and `WAV2VEC_WEIGHTS_SHA256` from Task 1.
- Produces: XLS-R schema version 3 config with `model_revision` and `base_model_sha256`; `CheckpointInfo` exposes both.

- [ ] **Step 1: Write failing payload and runtime tests**

```python
def test_wav2vec_payload_requires_pinned_base_provenance() -> None:
    with pytest.raises(ValidationError):
        CheckpointPayload(
            architecture="wav2vec-xls-r",
            model_id=WAV2VEC_MODEL_ID,
            weights=b"weights",
            metrics={"macro_f1": 0.2},
            validation_hash="a" * 64,
            class_weighting="inverse-frequency",
            class_weights=(1.0,) * 7,
        )

def test_published_wav2vec_checkpoint_is_schema_three(tmp_path: Path) -> None:
    publish_checkpoint(tmp_path / "checkpoint", pinned_wav2vec_payload())
    config = json.loads((tmp_path / "checkpoint/config.json").read_text())
    assert config["schema_version"] == "3"
    assert config["model_revision"] == WAV2VEC_MODEL_REVISION
    assert config["base_model_sha256"] == WAV2VEC_WEIGHTS_SHA256
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_training.py tests/providers/test_wav2vec_emotion.py -q`

Expected: assertions fail because the fields and schema do not exist.

- [ ] **Step 3: Implement architecture-specific provenance**

```python
class CheckpointPayload(BaseModel):
    # existing fields remain
    model_revision: str | None = None
    base_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_metadata(self) -> CheckpointPayload:
        # existing class-weight checks remain
        if self.architecture == "wav2vec-xls-r":
            if (
                self.model_revision != WAV2VEC_MODEL_REVISION
                or self.base_model_sha256 != WAV2VEC_WEIGHTS_SHA256
                or any(value is not None for value in (
                    self.encoder_hash, self.encoder_revision, self.freeze_encoder, self.embedding_size
                ))
            ):
                raise ValueError("wav2vec metadata is incomplete")
        else:
            if self.model_revision is not None or self.base_model_sha256 is not None:
                raise ValueError("emotion2vec checkpoint cannot contain wav2vec metadata")
        return self
```

`publish_checkpoint` emits schema `3` only for XLS-R and includes the two new fields.
`validate_checkpoint` requires the exact schema-3 key set and constants for XLS-R while
retaining emotion2vec schema-1 and schema-2 behavior. `CheckpointInfo` gains
`model_revision: str | None` and `base_model_sha256: str | None`.

- [ ] **Step 4: Run focused tests and commit**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_training.py tests/providers/test_wav2vec_emotion.py -q`

Expected: all focused tests pass.

```bash
git add backend/src/voxdelta/evaluation/emotion_training.py backend/src/voxdelta/providers/_emotion_runtime.py backend/tests/evaluation/test_emotion_training.py backend/tests/providers/test_wav2vec_emotion.py
git commit -m "feat: require pinned XLS-R checkpoint provenance"
```

### Task 3: Controlled Local-Only XLS-R Training

**Files:**
- Modify: `backend/src/voxdelta/evaluation/emotion_training.py:47-68,769-901`
- Modify: `backend/scripts/train_emotion.py:26-66`
- Test: `backend/tests/evaluation/test_emotion_training.py`

**Interfaces:**
- Consumes: `validate_wav2vec_base` and strict checkpoint fields from Tasks 1-2.
- Produces: `Wav2VecTrainingProfile(base_model_path, train_batch_size, eval_batch_size, gradient_accumulation_steps)` and CLI flags `--base-model-path`, `--micro-batch-size`.

- [ ] **Step 1: Write failing profile and local-only load tests**

```python
@pytest.mark.parametrize("micro,accum", [(8, 2), (4, 4), (2, 8), (1, 16)])
def test_wav2vec_profiles_keep_effective_batch_sixteen(
    tmp_path: Path, micro: int, accum: int
) -> None:
    profile = Wav2VecTrainingProfile(
        base_model_path=tmp_path.resolve(),
        train_batch_size=micro,
        eval_batch_size=micro,
        gradient_accumulation_steps=accum,
    )
    assert profile.train_batch_size * profile.gradient_accumulation_steps == 16

def test_wav2vec_loads_verified_local_base_only(fake_transformers, prepared_base: Path) -> None:
    run_fake_wav2vec_training(prepared_base, fake_transformers)
    assert fake_transformers.extractor_calls == [(str(prepared_base), True)]
    assert fake_transformers.model_calls == [(str(prepared_base), True)]
```

Also parameterize rejection of mismatched eval batch, `(4,2)`, bools, relative base paths,
missing XLS-R CLI flags, and XLS-R-only flags supplied to emotion2vec+.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_training.py -q`

Expected: profile construction or CLI assertions fail.

- [ ] **Step 3: Implement the exact profile matrix and local load**

```python
AllowedMicroBatch = Literal[1, 2, 4, 8]
AllowedAccumulation = Literal[2, 4, 8, 16]
_ACCUMULATION_BY_BATCH = {8: 2, 4: 4, 2: 8, 1: 16}

class Wav2VecTrainingProfile(BaseModel):
    base_model_path: Path
    model_revision: Literal["1a640f32ac3e39899438a2931f9924c02f080a54"] = (
        WAV2VEC_MODEL_REVISION
    )
    base_model_sha256: Literal[
        "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"
    ] = WAV2VEC_WEIGHTS_SHA256
    train_batch_size: AllowedMicroBatch = 8
    eval_batch_size: AllowedMicroBatch = 8
    gradient_accumulation_steps: AllowedAccumulation = 2

    @model_validator(mode="after")
    def exact_profile(self) -> Wav2VecTrainingProfile:
        if (
            not self.base_model_path.is_absolute()
            or self.eval_batch_size != self.train_batch_size
            or self.gradient_accumulation_steps != _ACCUMULATION_BY_BATCH[self.train_batch_size]
            or self.learning_rate != 2e-5
            or self.warmup_ratio != 0.1
        ):
            raise ValueError("wav2vec training profile is fixed")
        return self
```

At `_train_wav2vec` entry, call `validate_wav2vec_base(profile.base_model_path)`. Pass
`str(prepared.path), local_files_only=True` to both `AutoFeatureExtractor.from_pretrained`
and `AutoModelForAudioClassification.from_pretrained`. Populate the checkpoint payload's
revision and base hash from the validated result.

The training CLI derives accumulation with `{8: 2, 4: 4, 2: 8, 1: 16}` and constructs:

```python
profile = Wav2VecTrainingProfile(
    seed=arguments.seed,
    base_model_path=arguments.base_model_path,
    train_batch_size=arguments.micro_batch_size,
    eval_batch_size=arguments.micro_batch_size,
    gradient_accumulation_steps=accumulation[arguments.micro_batch_size],
)
```

- [ ] **Step 4: Run focused tests and commit**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_training.py -q`

Expected: all focused tests pass.

```bash
git add backend/src/voxdelta/evaluation/emotion_training.py backend/scripts/train_emotion.py backend/tests/evaluation/test_emotion_training.py
git commit -m "feat: add controlled local-only XLS-R training"
```

### Task 4: Local-Only Provider Reload and Evaluation

**Files:**
- Modify: `backend/src/voxdelta/providers/wav2vec_emotion.py:28-125`
- Modify: `backend/src/voxdelta/evaluation/emotion_experiment.py:274-322`
- Modify: `backend/scripts/evaluate_emotion_checkpoint.py:27-60`
- Test: `backend/tests/providers/test_wav2vec_emotion.py`
- Test: `backend/tests/evaluation/test_emotion_experiment.py`

**Interfaces:**
- Consumes: `validate_wav2vec_base`, XLS-R schema-3 `CheckpointInfo`, and published checkpoint.
- Produces: `Wav2VecEmotionProvider(checkpoint_path, *, base_model_path, ...)` and evaluator `base_model_path` forwarding.

- [ ] **Step 1: Write failing provider and evaluator tests**

```python
def test_provider_verifies_matching_base_and_loads_local_only(
    checkpoint: Path, prepared_base: Path, factory: FakeFactory
) -> None:
    provider = Wav2VecEmotionProvider(
        checkpoint,
        base_model_path=prepared_base,
        model_factory=factory,
        hardware_probe=lambda: (False, True),
    )
    provider.analyze("safe-id", audio_path, "")
    assert factory.calls == [(checkpoint.resolve(), prepared_base.resolve(), "mps")]

def test_xls_r_evaluator_requires_base_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid_experiment_report"):
        evaluate_emotion_checkpoint(
            manifest, checkpoint, architecture="wav2vec-xls-r", split="validation"
        )
```

Add parser tests proving `--base-model-path` is required for XLS-R and rejected for
emotion2vec+; existing injected provider-factory tests continue without a real base.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `cd backend && uv run pytest tests/providers/test_wav2vec_emotion.py tests/evaluation/test_emotion_experiment.py -q`

Expected: constructor and evaluator signatures do not accept the base path.

- [ ] **Step 3: Implement local-only provider construction**

```python
class ModelFactory(Protocol):
    def __call__(
        self, checkpoint: Path, *, base_model_path: Path, device: str
    ) -> Predictor: ...

class _TransformersPredictor:
    def __init__(self, checkpoint: Path, base_model_path: Path, device: str) -> None:
        source = str(base_model_path)
        self._extractor = transformers.AutoFeatureExtractor.from_pretrained(
            source, local_files_only=True
        )
        self._model = transformers.AutoModelForAudioClassification.from_pretrained(
            source,
            local_files_only=True,
            num_labels=len(CANONICAL_LABELS),
            id2label=dict(enumerate(CANONICAL_LABELS)),
            label2id={label: index for index, label in enumerate(CANONICAL_LABELS)},
            ignore_mismatched_sizes=True,
        )
```

`Wav2VecEmotionProvider.__init__` validates the base directory and proves its revision and
weight hash equal the checkpoint's schema-3 provenance before any model allocation.
`evaluate_emotion_checkpoint(..., base_model_path: Path | None = None)` passes the path only
to the default XLS-R provider. The script accepts an absolute `--base-model-path`; it rejects
missing XLS-R and extra emotion2vec+ values before evaluation.

- [ ] **Step 4: Run focused tests and commit**

Run: `cd backend && uv run pytest tests/providers/test_wav2vec_emotion.py tests/evaluation/test_emotion_experiment.py -q`

Expected: all focused tests pass.

```bash
git add backend/src/voxdelta/providers/wav2vec_emotion.py backend/src/voxdelta/evaluation/emotion_experiment.py backend/scripts/evaluate_emotion_checkpoint.py backend/tests/providers/test_wav2vec_emotion.py backend/tests/evaluation/test_emotion_experiment.py
git commit -m "feat: reload XLS-R from verified local artifacts"
```

### Task 5: Operator Documentation and Full Static/Test Gate

**Files:**
- Modify: `data/README.md:91-130`
- Modify: implementation files only if gates reveal owned defects.

**Interfaces:**
- Consumes: all Tasks 1-4 CLIs.
- Produces: exact canonical-checkout commands and a fully green implementation branch.

- [ ] **Step 1: Replace the generic XLS-R guidance with exact commands**

```bash
VOXDELTA_DATA=/Volumes/nvme1/codes/voxdelta/data

uv run python scripts/prepare_wav2vec_base.py \
  --output "$VOXDELTA_DATA/models/base/wav2vec2-xls-r-300m-1a640f3"

uv run python scripts/train_emotion.py \
  --manifest "$VOXDELTA_DATA/manifests/emotion-smoke.jsonl" \
  --output "$VOXDELTA_DATA/models/wav2vec-xls-r-300m-smoke" \
  --architecture wav2vec-xls-r \
  --base-model facebook/wav2vec2-xls-r-300m \
  --base-model-path "$VOXDELTA_DATA/models/base/wav2vec2-xls-r-300m-1a640f3" \
  --micro-batch-size 4 \
  --seed 622

uv run python scripts/evaluate_emotion_checkpoint.py \
  --manifest "$VOXDELTA_DATA/manifests/emotion-smoke.jsonl" \
  --checkpoint "$VOXDELTA_DATA/models/wav2vec-xls-r-300m-smoke" \
  --base-model-path "$VOXDELTA_DATA/models/base/wav2vec2-xls-r-300m-1a640f3" \
  --output "$VOXDELTA_DATA/benchmarks/wav2vec-xls-r-300m-smoke-validation.json" \
  --architecture wav2vec-xls-r \
  --device mps \
  --split validation
```

Document that only an observed MPS OOM authorizes retrying the same training command with
`--micro-batch-size 2`, then `1`, after confirming no final checkpoint was published.

- [ ] **Step 2: Run the complete quality gate**

```bash
cd backend
uv run pytest
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run mypy --strict src scripts
uv lock --check
cd ..
git diff --check
```

Expected: `801+` tests pass with the existing single platform skip; Ruff, strict mypy,
lock, and diff checks all exit zero.

- [ ] **Step 3: Commit documentation or owned gate fixes**

```bash
git add data/README.md backend
git commit -m "docs: add reproducible XLS-R smoke workflow"
```

If no gate fix changed `backend`, stage only `data/README.md`.

### Task 6: Real NVMe Preparation and Balanced Smoke Acceptance

**Files:**
- Generate, Git-ignored: `/Volumes/nvme1/codes/voxdelta/data/models/base/wav2vec2-xls-r-300m-1a640f3`
- Generate, Git-ignored: `/Volumes/nvme1/codes/voxdelta/data/models/wav2vec-xls-r-300m-smoke`
- Generate, Git-ignored: `/Volumes/nvme1/codes/voxdelta/data/benchmarks/wav2vec-xls-r-300m-smoke-validation.json`
- Verify only: original ZIP/CSV, normalized audio, full and smoke manifests.

**Interfaces:**
- Consumes: green Tasks 1-5 implementation and canonical data.
- Produces: pinned base, smoke checkpoint, aggregate validation report, selected micro-batch, and go/no-go evidence for full training.

- [ ] **Step 1: Capture source and manifest fingerprints without item-level output**

Run a Python check that writes only filename, size, mtime-ns, and SHA-256 for the six source
ZIP/CSV files plus SHA-256 for `emotion.jsonl` and `emotion-smoke.jsonl` to a private local
acceptance log. Set the log mode to `0600` and do not print paths or manifest rows.

- [ ] **Step 2: Prepare and independently verify the pinned base**

Run the Task 5 preparation command. Then call `validate_wav2vec_base` in a fresh process and
assert revision `1a640f32ac3e39899438a2931f9924c02f080a54`, weight digest
`d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0`, directory `0700`,
three files `0600`, and no unexpected file.

- [ ] **Step 3: Train with the largest safe approved micro-batch**

Run the Task 5 training command with micro-batch four. If and only if diagnostics establish
MPS OOM and the final checkpoint path does not exist, rerun in a fresh process with two;
repeat with one only on a second confirmed OOM. Do not set
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0`.

Expected: one command exits zero and atomically publishes a schema-3 checkpoint.

- [ ] **Step 4: Reload and evaluate validation only**

Run the Task 5 evaluation command. Expected output:

```text
emotion evaluation: 35/35 completed
```

Assert with `jq` that `.split == "validation"`, `.item_count == 35`,
`.completed_count == 35`, all numeric metrics are finite, and the report has no keys named
`id`, `item_id`, `audio_path`, `transcript`, `probabilities`, or `predictions`.

- [ ] **Step 5: Recheck immutability and summarize the gate**

Recompute the Step 1 fingerprints and require exact equality. Verify checkpoint config has
schema `3`, the pinned revision/hash, inverse-frequency weighting, and seven finite positive
class weights. Record selected micro-batch, elapsed time, peak RSS, validation macro-F1,
per-label F1, checkpoint tree digest, and report digest in the private acceptance log.

- [ ] **Step 6: Commit the implementation completion record**

Append an `## Acceptance Result` section to this plan containing the exact observed
micro-batch, elapsed seconds, peak RSS, validation macro-F1, seven per-label F1 values,
checkpoint tree digest, report digest, source-fingerprint equality, and the recommendation
on full training. The recommendation must repeat that final-quality evaluation excludes
the previously exposed 35 test items or uses a newly frozen holdout. Do not add ignored
artifacts or item-level licensed data details.

```bash
git add docs/superpowers/plans/2026-08-20-xls-r-controlled-smoke.md
git commit -m "docs: record XLS-R smoke acceptance"
```

Run `git status --short --branch` and require a clean worktree before merge review.

## Acceptance Result

- Completed on 2026-08-20 KST using the canonical NVMe data tree and Apple MPS.
- The pinned base independently validated at revision
  `1a640f32ac3e39899438a2931f9924c02f080a54` with weight SHA-256
  `d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0`.
  The directory was mode `0700`; its exact three files were mode `0600`.
- Micro-batch four failed after 1,091.40 seconds under confirmed system memory pressure:
  swap peaked above 12 GiB, macOS recorded concurrent jetsam activity, the command returned
  `training_failed`, and no final checkpoint was published.
- A fresh-process retry with micro-batch two and gradient accumulation eight succeeded in
  1,640.61 seconds. `/usr/bin/time -l` recorded a maximum resident set size of
  3,562,782,720 bytes (3,397.73 MiB); Metal/unified-memory use is not included in that RSS,
  and observed swap peaked above 14 GiB.
- The schema-3 checkpoint records the pinned revision and base digest, inverse-frequency
  weighting, seven finite positive class weights, and tree digest
  `7c4d8beff8a3eaef74289857309b4dc61b3ea000b8ed9b9850cf988ce5e67ec0`.
  Because this smoke train split is balanced, all seven class weights are `1.0`.
- Production-provider reload evaluated validation only, completing 35/35 items in 7.4560
  seconds. Validation macro-F1 was `0.03571428571428571`; per-label F1 was happiness
  `0.0`, anger `0.0`, disgust `0.0`, fear `0.0`, neutral `0.0`, sadness `0.0`, and
  surprise `0.25`. All 35 validation items were predicted as surprise.
- The aggregate report is mode `0600`, has SHA-256
  `a9aa8bd0640b47c8dc65c24ca11b3188f8b4be04b130d1536b00a7e739533e2e`, and contains
  no item IDs, audio paths, transcripts, probabilities, or individual predictions.
- Exact post-run fingerprints for the six licensed source ZIP/CSV files and both full and
  smoke manifests equal their pre-run fingerprints. No test-split item was evaluated.
- Recommendation: do not begin the 36,665-item full fine-tune with this exact recipe on the
  24 GiB Mac. The controlled path is technically valid, but the smoke collapsed to one
  class and required severe paging even at micro-batch two. Diagnose optimization on a
  larger balanced development subset and/or move the comparison to higher-memory GPU
  hardware before authorizing full training. This smoke is not evidence that XLS-R itself
  is intrinsically incapable; it is a no-go for the current local recipe.
- Any future final-quality comparison must exclude the 35 test items exposed by the prior
  emotion2vec smoke or freeze a new untouched holdout from the 3,585 remaining unobserved
  test items.
