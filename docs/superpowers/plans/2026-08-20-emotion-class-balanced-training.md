# Class-Balanced Emotion Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic inverse-frequency class weighting to both emotion training backends, preserve legacy checkpoints, and retrain the frozen emotion2vec+ head without opening the full test split.

**Architecture:** A pure helper derives canonical-order weights from train rows only. Both backends call one weighted cross-entropy boundary and publish the exact strategy and weights in schema-v2 checkpoint config, while the runtime continues to accept schema-v1 baselines. The aggregate evaluator gains an explicit validation/test split so the balanced checkpoint can be accepted on validation data after production-provider reload.

**Tech Stack:** Python 3.13, PyTorch, FunASR, Transformers, Pydantic v2, pytest, Ruff, mypy, uv.

## Global Constraints

- Use `weight[c] = training_item_count / (7 * training_count[c])` in canonical label order.
- Derive weights from `train` rows only; validation and test labels must not affect them.
- Keep `data/models/emotion2vec-plus-large-emotion` unchanged.
- Publish the candidate to `data/models/emotion2vec-plus-large-emotion-balanced` without overwriting.
- Keep the full 3,620-item test split sealed during this phase.
- Pass only if validation macro-F1 exceeds `0.2082607` and all seven per-label F1 values exceed zero.
- Do not write transcripts, item IDs, audio paths, raw probabilities, or per-item predictions to reports or logs.

---

### Task 1: Deterministic train-only class weights

**Files:**
- Modify: `backend/src/voxdelta/evaluation/emotion_training.py`
- Test: `backend/tests/evaluation/test_emotion_training.py`

**Interfaces:**
- Consumes: `TrainingExample`, `CANONICAL_LABELS`, and the fixed training profiles.
- Produces: `inverse_frequency_class_weights(examples: Sequence[TrainingExample]) -> tuple[float, ...]` and `class_weighting: Literal["inverse-frequency"]` on both profiles.

- [x] **Step 1: Write failing profile and formula tests**

Add assertions that both profiles expose the fixed strategy and that callers cannot substitute another value:

```python
assert Wav2VecTrainingProfile().class_weighting == "inverse-frequency"
assert Emotion2VecTrainingProfile().class_weighting == "inverse-frequency"
for profile in (Wav2VecTrainingProfile, Emotion2VecTrainingProfile):
    with pytest.raises(ValidationError):
        profile.model_validate({"class_weighting": "none"})
```

Add a helper test with one train item for six labels, two sadness train items, and 100 validation sadness items. Assert canonical-order weights are `(8/7, 8/7, 8/7, 8/7, 8/7, 4/7, 8/7)` and that the sample-weighted mean is one. Add missing-label and no-train cases that expect `TrainingError("invalid_training_manifest")`.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
cd backend
uv run pytest tests/evaluation/test_emotion_training.py::test_profiles_are_strict_and_exact tests/evaluation/test_emotion_training.py::test_inverse_frequency_weights_are_train_only_canonical_and_normalized -q
```

Expected: failure because the profile field and helper do not exist.

- [x] **Step 3: Add the fixed profile field and pure helper**

Add to both profile models:

```python
class_weighting: Literal["inverse-frequency"] = "inverse-frequency"
```

After `TrainingExample`, add:

```python
def inverse_frequency_class_weights(
    examples: Sequence[TrainingExample],
) -> tuple[float, ...]:
    training = tuple(item for item in examples if item.split == "train")
    counts = {label: 0 for label in CANONICAL_LABELS}
    try:
        for item in training:
            counts[item.label] += 1
    except KeyError:
        raise TrainingError("invalid_training_manifest") from None
    if not training or any(count == 0 for count in counts.values()):
        raise TrainingError("invalid_training_manifest")
    total = len(training)
    return tuple(total / (len(CANONICAL_LABELS) * counts[label]) for label in CANONICAL_LABELS)
```

- [x] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command. Expected: pass.

- [x] **Step 5: Commit the weight contract**

```bash
git add backend/src/voxdelta/evaluation/emotion_training.py backend/tests/evaluation/test_emotion_training.py
git commit -m "feat: define inverse-frequency emotion weights"
```

---

### Task 2: Apply one weighted loss boundary to both trainers

**Files:**
- Modify: `backend/src/voxdelta/evaluation/emotion_training.py`
- Test: `backend/tests/evaluation/test_emotion_training.py`

**Interfaces:**
- Consumes: `inverse_frequency_class_weights(...)` from Task 1.
- Produces: `weighted_cross_entropy(torch: Any, logits: Any, labels: Any, class_weights: Sequence[float], device: str) -> Any`; both `_train_wav2vec` and `_train_emotion2vec` use it.

- [x] **Step 1: Write a failing weighted-loss boundary test**

Use a fake torch object whose `tensor` and `nn.functional.cross_entropy` record arguments. Assert the helper constructs a float32 tensor on `mps`, passes it as the `weight=` keyword, and returns the cross-entropy sentinel:

```python
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
```

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
cd backend
uv run pytest tests/evaluation/test_emotion_training.py::test_weighted_cross_entropy_passes_canonical_weights_to_torch -q
```

Expected: failure because `weighted_cross_entropy` does not exist.

- [x] **Step 3: Implement the shared boundary**

```python
def weighted_cross_entropy(
    torch: Any,
    logits: Any,
    labels: Any,
    class_weights: Sequence[float],
    device: str,
) -> Any:
    weights = torch.tensor(tuple(class_weights), dtype=torch.float32, device=device)
    return torch.nn.functional.cross_entropy(logits, labels, weight=weights)
```

In each backend, compute `class_weights = inverse_frequency_class_weights(train_items)` once. Replace XLS-R's model-provided unweighted loss with:

```python
logits = model(**inputs_for(clips)).logits
loss = weighted_cross_entropy(torch, logits, labels, class_weights, device)
```

Replace emotion2vec+'s loss call with:

```python
loss = weighted_cross_entropy(torch, head(features), labels, class_weights, device)
loss.backward()
```

- [x] **Step 4: Run focused and full training-contract tests**

Run:

```bash
cd backend
uv run pytest tests/evaluation/test_emotion_training.py -q
```

Expected: pass with the existing platform-specific skip only if applicable.

- [x] **Step 5: Commit the loss integration**

```bash
git add backend/src/voxdelta/evaluation/emotion_training.py backend/tests/evaluation/test_emotion_training.py
git commit -m "feat: balance both emotion training losses"
```

---

### Task 3: Version weighted checkpoint provenance without breaking baselines

**Files:**
- Modify: `backend/src/voxdelta/evaluation/emotion_training.py`
- Modify: `backend/src/voxdelta/providers/_emotion_runtime.py`
- Test: `backend/tests/evaluation/test_emotion_training.py`
- Test: `backend/tests/providers/test_wav2vec_emotion.py`
- Test: `backend/tests/providers/test_emotion2vec_emotion.py`

**Interfaces:**
- Consumes: fixed strategy and exact weights from Tasks 1-2.
- Produces: optional `class_weighting` and `class_weights` fields on `CheckpointPayload`; schema-v2 config for balanced checkpoints; schema-v1 runtime compatibility.

- [ ] **Step 1: Write failing publication and compatibility tests**

Extend the atomic-publication test payload with:

```python
class_weighting="inverse-frequency",
class_weights=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
```

Assert `config.json` contains schema version `"2"`, method `"inverse-frequency"`, and the exact list. Keep existing provider fixtures at schema v1 to prove legacy loading. Add a schema-v2 provider fixture and parameterize malformed metadata: wrong method, six weights, boolean, zero, negative, NaN, Infinity, and extra keys; each must raise `ProviderError("invalid_local_checkpoint")`.

- [ ] **Step 2: Run checkpoint tests and verify RED**

Run:

```bash
cd backend
uv run pytest tests/evaluation/test_emotion_training.py::test_checkpoint_publication_is_atomic_complete_and_hashes_validation tests/providers/test_wav2vec_emotion.py tests/providers/test_emotion2vec_emotion.py -q
```

Expected: failure because weighted metadata is not accepted or published.

- [ ] **Step 3: Add payload validation and schema-v2 publication**

Add optional fields:

```python
class_weighting: Literal["inverse-frequency"] | None = None
class_weights: tuple[float, ...] | None = None
```

Require both or neither. When present, require exactly seven finite positive non-boolean floats. In `publish_checkpoint`, emit schema `"2"` plus `class_weighting` and `class_weights`; otherwise retain the exact schema-v1 shape.

Pass the profile strategy and computed weights into the payload returned by both training backends.

- [ ] **Step 4: Validate both runtime schemas strictly**

In `_emotion_runtime.validate_checkpoint`, construct the architecture-specific base key set first. Accept exactly:

```python
schema_v1_keys = base_keys
schema_v2_keys = base_keys | {"class_weighting", "class_weights"}
```

Schema v1 must match `schema_v1_keys`. Schema v2 must match `schema_v2_keys`, method `inverse-frequency`, and a seven-element finite positive float list. Reject every other version/key combination before model loading.

- [ ] **Step 5: Run checkpoint and provider tests**

Run the Step 2 command. Expected: all pass.

- [ ] **Step 6: Commit checkpoint provenance**

```bash
git add backend/src/voxdelta/evaluation/emotion_training.py backend/src/voxdelta/providers/_emotion_runtime.py backend/tests/evaluation/test_emotion_training.py backend/tests/providers/test_wav2vec_emotion.py backend/tests/providers/test_emotion2vec_emotion.py
git commit -m "feat: record balanced emotion checkpoint provenance"
```

---

### Task 4: Make aggregate evaluation split-explicit

**Files:**
- Modify: `backend/src/voxdelta/evaluation/emotion_experiment.py`
- Modify: `backend/scripts/evaluate_emotion_checkpoint.py`
- Test: `backend/tests/evaluation/test_emotion_experiment.py`

**Interfaces:**
- Consumes: the existing production-provider aggregate evaluator.
- Produces: `evaluate_emotion_checkpoint(..., split: Literal["validation", "test"] = "test")`; report schema v2 fields `split` and `item_count`; CLI `--split` with default `test`.

- [ ] **Step 1: Write failing validation-split tests**

Call the evaluator with `split="validation"` and assert only validation IDs reach `FakeProvider`, `report.split == "validation"`, and `report.item_count == 7`. Update report fixtures to schema v2 and replace `test_count` assertions with `item_count`. Add a CLI parser test proving `--split validation` is accepted and invalid values fail with the existing sanitized error.

- [ ] **Step 2: Run evaluator tests and verify RED**

Run:

```bash
cd backend
uv run pytest tests/evaluation/test_emotion_experiment.py -q
```

Expected: failure because the split argument and generic count fields do not exist.

- [ ] **Step 3: Generalize the aggregate report and evaluator**

Change the report contract to:

```python
schema_version: Literal["2"] = "2"
split: Literal["validation", "test"]
item_count: int = Field(gt=0)
```

Replace all internal `test_items`/`test_count` names with `selected_items`/`item_count`, filter by the requested split, and reject anything outside validation/test. Keep the default as test for existing smoke commands.

Add to the CLI parser:

```python
parser.add_argument("--split", choices=("validation", "test"), default="test")
```

Pass the cast split into the evaluator and print `completed_count/item_count`.

- [ ] **Step 4: Run evaluator tests and verify GREEN**

Run the Step 2 command. Expected: pass.

- [ ] **Step 5: Commit split-explicit evaluation**

```bash
git add backend/src/voxdelta/evaluation/emotion_experiment.py backend/scripts/evaluate_emotion_checkpoint.py backend/tests/evaluation/test_emotion_experiment.py
git commit -m "feat: evaluate emotion checkpoints by explicit split"
```

---

### Task 5: Quality gate, cached retraining, and validation acceptance

**Files:**
- Modify: `docs/superpowers/plans/2026-08-20-emotion-class-balanced-training.md`
- Runtime artifact: `data/models/emotion2vec-plus-large-emotion-balanced/` (Git-ignored)
- Runtime artifact: `data/benchmarks/emotion2vec-plus-large-emotion-balanced-validation.json` (Git-ignored)
- Runtime log: `/Users/hosungmini/.openclaw/workspace/logs/2026-08-20/voxdelta-emotion2vec-balanced.log`

**Interfaces:**
- Consumes: the balanced trainers, schema-v2 runtime, explicit validation evaluator, fixed manifest, and existing embedding cache.
- Produces: a distinct candidate checkpoint and aggregate validation evidence; no test metrics.

- [ ] **Step 1: Run the complete repository quality gate**

Run:

```bash
cd backend
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts
uv lock --check
git diff --check
```

Expected: all checks pass, with only the existing platform-specific skip.

- [ ] **Step 2: Verify immutable inputs and empty output targets**

Verify the baseline checkpoint digest remains `8a5b7c55e0de52391acba7b7c2c74ac69cfd464eb7ab13002b335f1c1b499cf5`, record the six raw source file sizes/mtimes, verify the manifest hash, and assert neither balanced output path exists. Stop rather than overwrite if either target exists.

- [ ] **Step 3: Run cached full emotion2vec+ head retraining**

Run from `backend` with stdout/stderr captured to the runtime log:

```bash
uv run python scripts/train_emotion.py \
  --manifest ../data/manifests/emotion.jsonl \
  --output ../data/models/emotion2vec-plus-large-emotion-balanced \
  --architecture emotion2vec-plus \
  --base-model iic/emotion2vec_plus_large \
  --freeze-encoder \
  --seed 622
```

Expected: the existing `~/.cache/voxdelta/emotion2vec-v1` embeddings are reused and a private schema-v2 checkpoint is atomically published.

- [ ] **Step 4: Reload and evaluate validation only**

```bash
uv run python scripts/evaluate_emotion_checkpoint.py \
  --manifest ../data/manifests/emotion.jsonl \
  --checkpoint ../data/models/emotion2vec-plus-large-emotion-balanced \
  --output ../data/benchmarks/emotion2vec-plus-large-emotion-balanced-validation.json \
  --architecture emotion2vec-plus \
  --device auto \
  --split validation
```

Expected: 3,569 validation items attempted; report mode 0600; no item-level content; test items untouched.

- [ ] **Step 5: Apply the acceptance rule and verify integrity**

Assert report macro-F1 is greater than `0.2082607`, every canonical per-label F1 is greater than zero, probabilities sum to one on a production-provider reload, the balanced checkpoint config records the exact seven weights, the baseline digest is unchanged, and the six raw source file sizes/mtimes are unchanged.

- [ ] **Step 6: Record aggregate results and rerun focused checks**

Append only aggregate macro-F1, per-label F1, prediction counts, checkpoint digest, validation hash, elapsed time, cache size, and pass/fail outcome to this plan. Then run:

```bash
cd backend
uv run pytest tests/evaluation/test_emotion_training.py tests/evaluation/test_emotion_experiment.py tests/providers/test_wav2vec_emotion.py tests/providers/test_emotion2vec_emotion.py -q
git status --short
```

Expected: focused tests pass and only intentional tracked plan-result changes remain.

- [ ] **Step 7: Commit verified results**

```bash
git add docs/superpowers/plans/2026-08-20-emotion-class-balanced-training.md
git commit -m "docs: record balanced emotion training result"
```
