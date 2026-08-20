# Real-Data Emotion Smoke Experiment Design

## Goal

Prove that VoxDelta can train and evaluate a real `emotion2vec+ large` seven-emotion checkpoint on the prepared AI Hub 263 audio before committing to the full 36,665-item training run. The experiment must produce aggregate, reproducible evidence about quality, runtime, and memory without copying transcripts or per-item predictions into its report.

## Approved Scope

The source manifest is `data/manifests/emotion.jsonl`: 36,665 Korean dialogue utterances in 16 kHz mono PCM16, labeled as happiness, anger, disgust, fear, neutral, sadness, or surprise. Only rows with at least three matching votes among five annotators are present. The deterministic full split is train 29,476, validation 3,569, and test 3,620.

This phase adds a deterministic real-data smoke path, trains the frozen-encoder `iic/emotion2vec_plus_large@v2.0.5` candidate, evaluates it on a held-out smoke test subset, and emits an aggregate report. Full emotion2vec+ training, XLS-R full fine-tuning, final call-center gold evaluation, and production default selection remain later phases.

## Approaches Considered

1. **Run the full emotion2vec+ job immediately.** This has the shortest code path but risks discovering model-download, FunASR, MPS, memory, or checkpoint-inference failures only after a long run.
2. **Use only the existing synthetic 20-example smoke helper.** This is fast, but it does not load the real model or exercise real AI Hub audio, checkpoint publication, or provider inference.
3. **Run a deterministic stratified real-data smoke experiment.** This adds a small reusable boundary, catches runtime integration failures early, and produces metrics comparable with later XLS-R work. This is the selected approach.

## Architecture

### Deterministic smoke manifest

A focused experiment module loads the trusted full emotion manifest and selects items independently within each `(split, label)` cell. Ordering is the SHA-256 of `seed`, item ID, and audio SHA-256, with the item ID as a stable tie-breaker. The default sample contains 20 training, 5 validation, and 5 test items per label: 210 items total.

The selection fails closed if any of the seven labels lacks the requested count in any split. Selected records preserve their original IDs, absolute audio paths, splits, labels, and audio hashes. The generated JSONL remains under Git-ignored `data/` paths and contains no transcript text.

### Training

The smoke manifest is passed to the existing `train_emotion.py` boundary with the exact production emotion2vec+ profile:

- encoder: `iic/emotion2vec_plus_large`
- revision: `v2.0.5`
- encoder frozen
- seven-class `LayerNorm → Linear(256) → GELU → Dropout(0.1) → Linear(7)` head
- AdamW at `1e-3`, batch size 64, at most 20 epochs, patience 3, seed 622
- validation macro-F1 checkpoint selection

The existing content-addressed embedding cache is reused. The checkpoint is atomically published under `data/models/` and retains its encoder identity hash and validation-set hash.

### Held-out smoke evaluation

A separate evaluation CLI loads the published checkpoint through `Emotion2VecEmotionProvider` and processes only the smoke manifest's test items. Keeping evaluation separate from training ensures the training process has released its model before inference begins.

The evaluator writes one aggregate JSON report under `data/benchmarks/` with:

- schema version, architecture, model ID, checkpoint digest, manifest digest, and deterministic sample counts
- completion rate
- macro-F1 and per-label F1
- a canonical-label-order 7×7 confusion matrix
- expected calibration error using 10 equal-width confidence bins
- median per-item latency and maximum observed process RSS
- total elapsed time and device request

The report must not contain transcripts, item IDs, audio paths, raw probabilities, or per-item predictions. Test items are used only after training and validation are complete.

## Interfaces

`voxdelta.evaluation.emotion_experiment` provides:

- `build_stratified_smoke_manifest(source: Path, output: Path, *, train_per_label: int = 20, validation_per_label: int = 5, test_per_label: int = 5, seed: int = 622) -> SmokeManifestSummary`
- `evaluate_emotion_checkpoint(manifest: Path, checkpoint: Path, *, architecture: Literal["emotion2vec-plus", "wav2vec-xls-r"], device: Device = "auto") -> EmotionExperimentReport`
- `write_experiment_report(path: Path, report: EmotionExperimentReport) -> None`

Two thin CLIs expose those boundaries:

- `scripts/build_emotion_smoke_manifest.py`
- `scripts/evaluate_emotion_checkpoint.py`

The existing `scripts/train_emotion.py` remains the only training entry point.

## Failure and Privacy Boundaries

- Reject relative or symlink-traversing input/output paths through the repository's trusted-file and atomic-publication patterns.
- Reject an existing output manifest or report instead of overwriting it.
- Sanitize CLI failures to stable codes; never print model internals, local data paths, item IDs, transcripts, or exception text.
- Treat provider failure as an incomplete evaluation, but emit a report only when its aggregate fields are internally valid and at least one test item completed.
- Never mutate the full manifest, normalized audio, source ZIPs, or source CSV files.

## Testing

Unit tests use generated WAVs and fake providers to prove deterministic stratification, label/split coverage, scarcity rejection, no transcript retention, metric calculations, canonical confusion-matrix order, ECE bounds, sanitized failures, atomic output, and report privacy. Provider regression tests verify that the real emotion2vec predictor requests utterance embeddings consistently in training and inference.

The quality gate is the full backend pytest suite, Ruff lint and formatting, strict mypy over `src` and `scripts`, `uv lock --check`, and `git diff --check`. The real-data acceptance run additionally verifies a 210-item smoke manifest, a valid published checkpoint, a finite aggregate report, and no source-data mutations.

## Success Criteria

This phase is complete when the real emotion2vec+ model trains on the deterministic 210-item smoke manifest, its checkpoint loads through the production provider, all 35 held-out smoke test items are attempted, and the aggregate report records finite metrics, latency, and RSS without sensitive item-level content. A low macro-F1 does not fail this integration phase; it becomes evidence for the later full-training and class-imbalance experiments.
