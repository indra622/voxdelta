# Real-Data Emotion Smoke Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, privacy-minimized real-data smoke training and evaluation path, then run it with `emotion2vec+ large` on AI Hub 263.

**Architecture:** A new experiment module selects a balanced subset from the trusted full manifest and computes aggregate held-out metrics without retaining item-level output. Two thin CLIs build the smoke manifest and evaluate an existing checkpoint; the existing `train_emotion.py` remains the sole training boundary.

**Tech Stack:** Python 3.12, Pydantic v2, scikit-learn metrics, existing FunASR/torch/safetensors training and provider boundaries, pytest, Ruff, strict mypy, uv.

## Global Constraints

- Use the existing canonical label order: `happiness, anger, disgust, fear, neutral, sadness, surprise`.
- Default deterministic sample counts are 20 train, 5 validation, and 5 test items per label with seed 622.
- Preserve original IDs, absolute audio paths, split assignments, labels, and SHA-256 values in the ignored smoke manifest; write an empty transcript only.
- Never write transcript text, item IDs, audio paths, raw probabilities, or per-item predictions to the aggregate report or CLI output.
- Do not mutate the full manifest, normalized audio, source ZIPs, or source CSVs.
- Refuse existing outputs and symlink-traversing paths; publish generated JSON/JSONL atomically with mode 0600.
- Keep the existing fixed production training profiles unchanged.
- Sanitize CLI errors to stable messages without exception details or local paths.

---

### Task 1: Deterministic stratified smoke manifest

**Files:**
- Create: `backend/src/voxdelta/evaluation/emotion_experiment.py`
- Create: `backend/tests/evaluation/test_emotion_experiment.py`

**Interfaces:**
- Consumes: `DatasetItem`, `load_manifest`, `validate_disjoint_splits`, and a trusted full emotion JSONL manifest.
- Produces: `SmokeManifestSummary` and `build_stratified_smoke_manifest(source: Path, output: Path, *, train_per_label: int = 20, validation_per_label: int = 5, test_per_label: int = 5, seed: int = 622) -> SmokeManifestSummary`.

- [x] **Step 1: Write failing deterministic-selection tests**

```python
def test_build_smoke_manifest_is_balanced_deterministic_and_transcript_free(tmp_path: Path) -> None:
    source = _full_fixture_manifest(tmp_path)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    summary = build_stratified_smoke_manifest(
        source, first, train_per_label=2, validation_per_label=1, test_per_label=1, seed=622
    )
    build_stratified_smoke_manifest(
        source, second, train_per_label=2, validation_per_label=1, test_per_label=1, seed=622
    )

    assert summary.total_count == 28
    assert first.read_bytes() == second.read_bytes()
    selected = load_manifest(first)
    assert Counter((item.split, item.emotion) for item in selected) == _expected_cells(2, 1, 1)
    assert {item.transcript for item in selected} == {""}


def test_build_smoke_manifest_rejects_scarce_cells_and_existing_output(tmp_path: Path) -> None:
    source = _full_fixture_manifest(tmp_path, surprise_test_count=0)
    with pytest.raises(ValueError, match="invalid_smoke_manifest"):
        build_stratified_smoke_manifest(source, tmp_path / "smoke.jsonl")

    valid = _full_fixture_manifest(tmp_path / "valid")
    output = tmp_path / "existing.jsonl"
    output.write_text("owner data", encoding="utf-8")
    with pytest.raises(ValueError, match="smoke_manifest_publication_failed"):
        build_stratified_smoke_manifest(valid, output)
    assert output.read_text(encoding="utf-8") == "owner data"
```

- [x] **Step 2: Run the tests and verify RED**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_experiment.py -q`

Expected: collection fails because `voxdelta.evaluation.emotion_experiment` does not exist.

- [x] **Step 3: Implement the selection and atomic writer**

```python
CANONICAL_LABELS: tuple[EmotionLabel, ...] = (
    "happiness", "anger", "disgust", "fear", "neutral", "sadness", "surprise"
)


class SmokeManifestSummary(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    total_count: int = Field(gt=0)
    split_counts: dict[DatasetSplit, int]
    label_counts: dict[EmotionLabel, int]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _rank(item: DatasetItem, seed: int) -> tuple[str, str]:
    payload = f"{seed}\0{item.id}\0{item.sha256}".encode()
    return hashlib.sha256(payload).hexdigest(), item.id


def build_stratified_smoke_manifest(
    source: Path,
    output: Path,
    *,
    train_per_label: int = 20,
    validation_per_label: int = 5,
    test_per_label: int = 5,
    seed: int = 622,
) -> SmokeManifestSummary:
    requested = {"train": train_per_label, "validation": validation_per_label, "test": test_per_label}
    if any(isinstance(value, bool) or value <= 0 for value in (*requested.values(), seed)):
        raise ValueError("invalid_smoke_manifest")
    items = load_manifest(source)
    validate_disjoint_splits(items)
    cells: dict[tuple[DatasetSplit, EmotionLabel], list[DatasetItem]] = {}
    for item in items:
        if item.source != "emotion" or item.emotion not in CANONICAL_LABELS:
            raise ValueError("invalid_smoke_manifest")
        cells.setdefault((item.split, item.emotion), []).append(item)
    selected: list[DatasetItem] = []
    for split, count in requested.items():
        for label in CANONICAL_LABELS:
            candidates = sorted(cells.get((split, label), ()), key=lambda item: _rank(item, seed))
            if len(candidates) < count:
                raise ValueError("invalid_smoke_manifest")
            selected.extend(item.model_copy(update={"transcript": ""}) for item in candidates[:count])
    payload = b"".join(
        json.dumps(item.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        for item in sorted(selected, key=lambda item: item.id)
    )
    _publish_private_file(output, payload, "smoke_manifest_publication_failed")
    return SmokeManifestSummary(
        total_count=len(selected),
        split_counts=Counter(item.split for item in selected),
        label_counts=Counter(cast(EmotionLabel, item.emotion) for item in selected),
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )
```

Implement the private atomic writer used by manifests and reports:

```python
def _publish_private_file(path: Path, payload: bytes, error_code: str) -> None:
    target = Path(os.path.abspath(path.expanduser()))
    current = Path(target.anchor)
    for part in target.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(error_code)
    if target.exists():
        raise ValueError(error_code)
    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise OSError
        os.replace(temporary, target)
        temporary = None
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ValueError(error_code) from None
```

- [x] **Step 4: Run focused tests and static checks**

Run:

```bash
cd backend
uv run pytest tests/evaluation/test_emotion_experiment.py -q
uv run ruff check src/voxdelta/evaluation/emotion_experiment.py tests/evaluation/test_emotion_experiment.py
uv run ruff format --check src/voxdelta/evaluation/emotion_experiment.py tests/evaluation/test_emotion_experiment.py
uv run mypy src/voxdelta/evaluation/emotion_experiment.py
```

Expected: all checks pass.

- [x] **Step 5: Commit the manifest boundary**

```bash
git add backend/src/voxdelta/evaluation/emotion_experiment.py backend/tests/evaluation/test_emotion_experiment.py
git commit -m "feat: build deterministic emotion smoke manifests"
```

---

### Task 2: Aggregate checkpoint evaluation

**Files:**
- Modify: `backend/src/voxdelta/evaluation/emotion_experiment.py`
- Modify: `backend/tests/evaluation/test_emotion_experiment.py`

**Interfaces:**
- Consumes: a smoke manifest, a published local checkpoint, an architecture name, and the production emotion provider interface.
- Produces: `EmotionExperimentReport`, `evaluate_emotion_checkpoint(...)`, and `write_experiment_report(...)`.

- [x] **Step 1: Write failing aggregate-metric and privacy tests**

```python
def test_evaluation_report_has_aggregate_metrics_without_item_content(tmp_path: Path) -> None:
    manifest = _evaluation_fixture(tmp_path)
    provider = FakeProvider(_probability_rows())
    report = evaluate_emotion_checkpoint(
        manifest,
        tmp_path / "checkpoint",
        architecture="emotion2vec-plus",
        provider_factory=lambda *_args, **_kwargs: provider,
    )

    assert report.test_count == 7
    assert report.completed_count == 7
    assert report.completion_rate == 1.0
    assert 0.0 <= report.macro_f1 <= 1.0
    assert tuple(report.per_label_f1) == CANONICAL_LABELS
    assert len(report.confusion_matrix) == 7
    assert all(len(row) == 7 for row in report.confusion_matrix)
    assert 0.0 <= report.expected_calibration_error <= 1.0
    serialized = report.model_dump_json()
    assert "transcript" not in serialized
    assert "audio_path" not in serialized
    assert "utterance-" not in serialized


def test_report_writer_is_atomic_private_and_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    write_experiment_report(output, _report_fixture())
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    original = output.read_bytes()
    with pytest.raises(ValueError, match="experiment_report_publication_failed"):
        write_experiment_report(output, _report_fixture())
    assert output.read_bytes() == original
```

- [x] **Step 2: Run the tests and verify RED**

Run: `cd backend && uv run pytest tests/evaluation/test_emotion_experiment.py -q`

Expected: failures because `EmotionExperimentReport` and evaluator functions are absent.

- [x] **Step 3: Implement report contracts and aggregate metrics**

```python
class EmotionExperimentReport(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    schema_version: Literal["1"] = "1"
    architecture: Literal["emotion2vec-plus", "wav2vec-xls-r"]
    model_id: str
    checkpoint_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_count: int = Field(gt=0)
    completed_count: int = Field(gt=0)
    completion_rate: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    per_label_f1: dict[EmotionLabel, float]
    confusion_matrix: tuple[tuple[int, ...], ...]
    expected_calibration_error: float = Field(ge=0, le=1)
    median_latency_ms: float = Field(ge=0)
    peak_rss_mb: float | None = Field(default=None, ge=0)
    elapsed_seconds: float = Field(ge=0)
    requested_device: Device


def _aggregate_metrics(
    expected: Sequence[EmotionLabel],
    predicted: Sequence[EmotionLabel],
    confidence: Sequence[float],
) -> tuple[float, dict[EmotionLabel, float], tuple[tuple[int, ...], ...], float]:
    matrix = [[0 for _ in CANONICAL_LABELS] for _ in CANONICAL_LABELS]
    for truth, guess in zip(expected, predicted, strict=True):
        matrix[CANONICAL_LABELS.index(truth)][CANONICAL_LABELS.index(guess)] += 1
    scores: dict[EmotionLabel, float] = {}
    for index, label in enumerate(CANONICAL_LABELS):
        true_positive = matrix[index][index]
        false_positive = sum(row[index] for row in matrix) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores[label] = 0.0 if denominator == 0 else 2 * true_positive / denominator
    macro_f1 = math.fsum(scores.values()) / len(CANONICAL_LABELS)
    bins: list[list[tuple[bool, float]]] = [[] for _ in range(10)]
    for truth, guess, score in zip(expected, predicted, confidence, strict=True):
        bins[min(int(score * 10), 9)].append((truth == guess, score))
    ece = math.fsum(
        len(bucket) / len(expected)
        * abs(math.fsum(float(correct) for correct, _ in bucket) / len(bucket)
              - math.fsum(score for _, score in bucket) / len(bucket))
        for bucket in bins
        if bucket
    )
    return macro_f1, scores, tuple(tuple(row) for row in matrix), ece
```

Implement evaluation with an injectable provider factory for unit tests and exact production providers by default:

```python
ProviderFactory = Callable[[Path, Device], EmotionProvider]


def _default_provider(
    checkpoint: Path,
    architecture: Literal["emotion2vec-plus", "wav2vec-xls-r"],
    device: Device,
) -> EmotionProvider:
    if architecture == "emotion2vec-plus":
        return Emotion2VecEmotionProvider(checkpoint, device=device)
    return Wav2VecEmotionProvider(checkpoint, device=device)


def evaluate_emotion_checkpoint(
    manifest: Path,
    checkpoint: Path,
    *,
    architecture: Literal["emotion2vec-plus", "wav2vec-xls-r"],
    device: Device = "auto",
    provider_factory: ProviderFactory | None = None,
) -> EmotionExperimentReport:
    try:
        manifest_payload = read_trusted_regular_file(manifest)
        items = load_manifest(manifest)
        validate_disjoint_splits(items)
        test_items = tuple(item for item in items if item.split == "test")
        if not test_items or any(item.source != "emotion" or item.emotion is None for item in items):
            raise ValueError
        for item in test_items:
            audio = read_trusted_regular_file(item.audio_path)
            if hashlib.sha256(audio).hexdigest() != item.sha256:
                raise ValueError
    except Exception:
        raise ValueError("invalid_experiment_manifest") from None

    provider = (
        provider_factory(checkpoint, device)
        if provider_factory is not None
        else _default_provider(checkpoint, architecture, device)
    )
    expected: list[EmotionLabel] = []
    predicted: list[EmotionLabel] = []
    confidence: list[float] = []
    latency: list[float] = []
    rss: list[float] = []
    started = time.perf_counter()
    try:
        for item in test_items:
            try:
                result = provider.analyze(item.id, Path(item.audio_path), "")
            except ProviderError:
                continue
            expected.append(cast(EmotionLabel, item.emotion))
            predicted.append(max(result.probabilities, key=result.probabilities.__getitem__))
            confidence.append(result.confidence)
            if result.usage is not None:
                latency.append(result.usage.latency_ms)
                if result.usage.peak_rss_mb is not None:
                    rss.append(result.usage.peak_rss_mb)
    finally:
        unload = getattr(provider, "unload", None)
        if callable(unload):
            unload()
    if not expected or not latency:
        raise ValueError("invalid_experiment_report")
    macro_f1, per_label_f1, matrix, ece = _aggregate_metrics(
        expected, predicted, confidence
    )
    provenance = provider.provenance
    if provenance.revision is None:
        raise ValueError("invalid_experiment_report")
    return EmotionExperimentReport(
        architecture=architecture,
        model_id=provenance.model,
        checkpoint_digest=provenance.revision,
        manifest_digest=hashlib.sha256(manifest_payload).hexdigest(),
        test_count=len(test_items),
        completed_count=len(expected),
        completion_rate=len(expected) / len(test_items),
        macro_f1=macro_f1,
        per_label_f1=per_label_f1,
        confusion_matrix=matrix,
        expected_calibration_error=ece,
        median_latency_ms=statistics.median(latency),
        peak_rss_mb=max(rss, default=None),
        elapsed_seconds=max(0.0, time.perf_counter() - started),
        requested_device=device,
    )
```

`write_experiment_report` serializes `report.model_dump(mode="json")` with sorted keys, compact separators, `allow_nan=False`, a trailing newline, and `_publish_private_file(..., "experiment_report_publication_failed")`.

- [x] **Step 4: Add failure-boundary tests**

```python
def test_evaluation_rejects_hash_mismatch_and_zero_completion(tmp_path: Path) -> None:
    manifest = _evaluation_fixture(tmp_path)
    Path(load_manifest(manifest)[-1].audio_path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="invalid_experiment_manifest"):
        evaluate_emotion_checkpoint(
            manifest,
            tmp_path / "checkpoint",
            architecture="emotion2vec-plus",
            provider_factory=lambda *_args, **_kwargs: FakeProvider({}),
        )

    clean = _evaluation_fixture(tmp_path / "clean")
    with pytest.raises(ValueError, match="invalid_experiment_report"):
        evaluate_emotion_checkpoint(
            clean,
            tmp_path / "checkpoint",
            architecture="emotion2vec-plus",
            provider_factory=lambda *_args, **_kwargs: FailingProvider(),
        )
```

- [x] **Step 5: Run focused tests and static checks**

Run:

```bash
cd backend
uv run pytest tests/evaluation/test_emotion_experiment.py -q
uv run ruff check src/voxdelta/evaluation/emotion_experiment.py tests/evaluation/test_emotion_experiment.py
uv run ruff format --check src/voxdelta/evaluation/emotion_experiment.py tests/evaluation/test_emotion_experiment.py
uv run mypy src/voxdelta/evaluation/emotion_experiment.py
```

Expected: all checks pass.

- [x] **Step 6: Commit aggregate evaluation**

```bash
git add backend/src/voxdelta/evaluation/emotion_experiment.py backend/tests/evaluation/test_emotion_experiment.py
git commit -m "feat: evaluate local emotion checkpoints"
```

---

### Task 3: Safe CLIs and emotion2vec inference parity

**Files:**
- Create: `backend/scripts/build_emotion_smoke_manifest.py`
- Create: `backend/scripts/evaluate_emotion_checkpoint.py`
- Modify: `backend/src/voxdelta/providers/emotion2vec_emotion.py`
- Modify: `backend/tests/providers/test_emotion2vec_emotion.py`
- Modify: `backend/tests/evaluation/test_emotion_experiment.py`

**Interfaces:**
- Consumes: Task 1 and Task 2 public functions plus the existing provider predictor.
- Produces: sanitized CLI exit statuses and consistent utterance-embedding extraction in training and inference.

- [x] **Step 1: Write failing CLI tests**

```python
def test_build_cli_writes_summary_without_private_content(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_emotion_smoke_manifest.py", "--manifest", str(source),
         "--output", str(output), "--train-per-label", "2", "--validation-per-label", "1",
         "--test-per-label", "1", "--seed", "622"],
        cwd=BACKEND, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout == "smoke manifest: 28 items\n"


def test_cli_failures_are_sanitized(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_emotion_checkpoint.py", "--manifest", "private-id",
         "--checkpoint", "private-checkpoint", "--output", str(tmp_path / "report.json"),
         "--architecture", "emotion2vec-plus"],
        cwd=BACKEND, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "emotion evaluation failed\n"
    assert "private" not in completed.stderr
```

- [x] **Step 2: Write a failing real-predictor contract test**

```python
def test_predictor_requests_the_same_embedding_contract_as_training(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecordingEncoder:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate(self, **kwargs: object) -> list[dict[str, list[float]]]:
            self.calls.append(kwargs)
            return [{"feats": [1.0, 2.0, 3.0, 4.0]}]

    encoder = RecordingEncoder()
    predictor = _predictor_fixture(monkeypatch, encoder)
    predictor.predict((0.0,) * 16_000, 16_000)
    assert encoder.calls == [
        {"input": [0.0] * 16_000, "granularity": "utterance", "extract_embedding": True}
    ]
```

- [x] **Step 3: Run the tests and verify RED**

Run:

```bash
cd backend
uv run pytest tests/evaluation/test_emotion_experiment.py tests/providers/test_emotion2vec_emotion.py -q
```

Expected: CLI files are missing and inference omits `extract_embedding=True`.

- [x] **Step 4: Implement the two thin CLIs**

Implement `build_emotion_smoke_manifest.py` with this boundary:

```python
def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("positive integer required")
    return parsed


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-per-label", type=_positive_int, default=20)
    parser.add_argument("--validation-per-label", type=_positive_int, default=5)
    parser.add_argument("--test-per-label", type=_positive_int, default=5)
    parser.add_argument("--seed", type=_positive_int, default=622)
    try:
        args = parser.parse_args(argv)
        summary = build_stratified_smoke_manifest(
            args.manifest,
            args.output,
            train_per_label=args.train_per_label,
            validation_per_label=args.validation_per_label,
            test_per_label=args.test_per_label,
            seed=args.seed,
        )
    except Exception:
        print("smoke manifest failed", file=sys.stderr)
        return 2
    print(f"smoke manifest: {summary.total_count} items")
    return 0
```

Implement `evaluate_emotion_checkpoint.py` with this boundary:

```python
class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--architecture", required=True, choices=("emotion2vec-plus", "wav2vec-xls-r")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    try:
        args = parser.parse_args(argv)
        report = evaluate_emotion_checkpoint(
            args.manifest,
            args.checkpoint,
            architecture=args.architecture,
            device=args.device,
        )
        write_experiment_report(args.output, report)
    except Exception:
        print("emotion evaluation failed", file=sys.stderr)
        return 2
    print(f"emotion evaluation: {report.completed_count}/{report.test_count} completed")
    return 0
```

- [x] **Step 5: Align emotion2vec inference embedding extraction**

```python
output = self._encoder.generate(
    input=list(samples),
    granularity="utterance",
    extract_embedding=True,
)
```

- [x] **Step 6: Run focused tests and static checks**

Run:

```bash
cd backend
uv run pytest tests/evaluation/test_emotion_experiment.py tests/providers/test_emotion2vec_emotion.py -q
uv run ruff check scripts/build_emotion_smoke_manifest.py scripts/evaluate_emotion_checkpoint.py src/voxdelta/evaluation/emotion_experiment.py src/voxdelta/providers/emotion2vec_emotion.py tests/evaluation/test_emotion_experiment.py tests/providers/test_emotion2vec_emotion.py
uv run ruff format --check scripts/build_emotion_smoke_manifest.py scripts/evaluate_emotion_checkpoint.py src/voxdelta/evaluation/emotion_experiment.py src/voxdelta/providers/emotion2vec_emotion.py tests/evaluation/test_emotion_experiment.py tests/providers/test_emotion2vec_emotion.py
uv run mypy src scripts
```

Expected: all checks pass.

- [x] **Step 7: Commit CLI and provider parity**

```bash
git add backend/scripts/build_emotion_smoke_manifest.py backend/scripts/evaluate_emotion_checkpoint.py backend/src/voxdelta/providers/emotion2vec_emotion.py backend/tests/evaluation/test_emotion_experiment.py backend/tests/providers/test_emotion2vec_emotion.py
git commit -m "feat: run real-data emotion smoke experiments"
```

---

### Task 4: Documentation, quality gate, and real emotion2vec run

**Files:**
- Modify: `data/README.md`
- Modify: `docs/superpowers/plans/2026-08-20-emotion-real-data-smoke.md`

**Interfaces:**
- Consumes: the generated CLIs, existing `train_emotion.py`, and local AI Hub manifest.
- Produces: reproducible commands, ignored smoke artifacts, and verified aggregate experiment evidence.

- [x] **Step 1: Document the three-command workflow**

```bash
cd backend
uv run python scripts/build_emotion_smoke_manifest.py \
  --manifest ../data/manifests/emotion.jsonl \
  --output ../data/manifests/emotion-smoke.jsonl

uv run python scripts/train_emotion.py \
  --manifest ../data/manifests/emotion-smoke.jsonl \
  --output ../data/models/emotion2vec-plus-large-smoke \
  --architecture emotion2vec-plus \
  --base-model iic/emotion2vec_plus_large \
  --freeze-encoder \
  --seed 622

uv run python scripts/evaluate_emotion_checkpoint.py \
  --manifest ../data/manifests/emotion-smoke.jsonl \
  --checkpoint ../data/models/emotion2vec-plus-large-smoke \
  --output ../data/benchmarks/emotion2vec-plus-large-smoke.json \
  --architecture emotion2vec-plus \
  --device auto
```

Explain that this smoke run is an integration check, not final quality evidence, and that full-training comparison remains emotion2vec+ versus XLS-R on the fixed full splits.

- [x] **Step 2: Run the full repository quality gate**

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

Expected: all checks pass, with only the existing Windows-specific skip on macOS.

- [x] **Step 3: Commit documentation**

```bash
git add data/README.md docs/superpowers/plans/2026-08-20-emotion-real-data-smoke.md
git commit -m "docs: add real-data emotion smoke workflow"
```

- [x] **Step 4: Run the real smoke manifest build**

Run the first documented command. Verify 210 records, exactly 20/5/5 items per label and split, source audio hashes, and empty transcripts.

- [x] **Step 5: Run real emotion2vec+ training with a persistent local log**

Run the second documented command with stdout/stderr redirected to `logs/2026-08-20/voxdelta-emotion2vec-smoke.log` through the task supervisor. Record start/end time, exit status, model/cache disk use, and sanitized last-error context if it fails.

- [x] **Step 6: Evaluate the real checkpoint**

Run the third documented command. Verify the checkpoint contract, 35 test attempts, finite aggregate metrics, report mode 0600, absence of forbidden item-level keys/content, and no changes to source file sizes or mtimes.

- [x] **Step 7: Re-run focused checks after the live integration**

Run:

```bash
cd backend
uv run pytest tests/evaluation/test_emotion_experiment.py tests/providers/test_emotion2vec_emotion.py tests/evaluation/test_emotion_training.py -q
git status --short
```

Expected: tests pass and only intentional tracked documentation changes, if any, remain.

Live-run result: the deterministic manifest contained 210 items with 20/5/5 items per label
across train/validation/test. The production provider completed all 35 held-out test items. The
aggregate report recorded macro-F1 0.082792, ECE 0.186114, median latency 64.996 ms, and peak RSS
4883.484 MB. This is an integration smoke result, not final model-quality evidence.
