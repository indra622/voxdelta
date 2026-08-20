# XLS-R Controlled Smoke Preparation Design

## Goal

Prepare and run one reproducible, privacy-minimized `facebook/wav2vec2-xls-r-300m`
smoke experiment before any full 29,476-item fine-tuning. The run must answer whether
the pinned XLS-R model can train and reload on the current 24 GiB Apple-silicon MPS
host, using the same data split and inverse-frequency loss contract as the accepted
emotion2vec+ comparison candidate.

This phase ends after the pinned base artifacts are present on NVMe, a balanced smoke
checkpoint is published, and all 35 smoke-validation items are evaluated. It does not
start full training or evaluate any additional item from the 3,620-item test split.

## Evidence and Approved Scope

The trusted emotion manifest contains 36,665 Korean dialogue utterances in 16 kHz mono
PCM16. Only examples with at least three matching votes among five annotators are
present. Its deterministic split is train 29,476, validation 3,569, and test 3,620.
The existing smoke manifest contains 20 train, 5 validation, and 5 test examples per
canonical label: 210 examples total.

The 35 smoke-test examples were already evaluated as integration evidence during the
earlier emotion2vec+ smoke. They are therefore exposed members of the nominal 3,620-item
test split, and that full split must not be described as pristine. This phase does not
reuse those 35 examples or inspect any of the remaining 3,585 test examples. A later
final-quality design must either exclude the exposed 35 or create a fresh holdout.

The accepted balanced emotion2vec+ candidate improved validation macro-F1 from
`0.2082607` to `0.2400352`, and all seven label F1 values became non-zero. That proves
inverse-frequency weighting reduces majority-class collapse, but the resulting quality
is still too low to justify more tuning of the same frozen representation before a
different encoder is tested.

The current XLS-R path already performs full-model fine-tuning with AdamW, learning rate
`2e-5`, ten epochs, warmup ratio `0.1`, validation macro-F1 selection, early-stopping
patience two, seed 622, and train-only inverse-frequency class weights. Preparation must
preserve those choices while fixing model identity, local artifacts, and a memory-safe
micro-batch.

## Approaches Considered

1. **Run the current profile directly.** This is shortest, but the base revision is not
   pinned, the model is not cached locally, and micro-batch eight may exhaust MPS memory.
2. **Pin and stage the model, expose controlled micro-batches, then run the balanced
   smoke.** This isolates model identity and hardware capacity before the expensive run.
   This is the selected approach.
3. **Add resumable epoch checkpoints before smoke.** This is valuable for a later full
   run, but it adds recovery semantics that the small smoke does not need. It remains a
   separate decision after smoke timing and stability are known.

## Pinned Base Model Contract

The model identity is fixed to:

- repository: `facebook/wav2vec2-xls-r-300m`
- revision: `1a640f32ac3e39899438a2931f9924c02f080a54`
- `config.json`: 1,568 bytes, SHA-256
  `0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac`
- `preprocessor_config.json`: 212 bytes, SHA-256
  `a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335`
- `pytorch_model.bin`: 1,269,737,156 bytes, SHA-256
  `d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0`

A preparation CLI downloads only those three files from revision-qualified official
Hugging Face URLs. It stages them in a sibling temporary directory, checks exact names,
sizes, and hashes, gives directories mode `0700` and files mode `0600`, then publishes
the complete base-model directory through one atomic rename. An existing output is never
overwritten. The CLI prints only an aggregate success or stable failure code; URLs,
temporary paths, and exception text are not included in user-facing errors.

The approved local target is Git-ignored NVMe storage:

`data/models/base/wav2vec2-xls-r-300m-1a640f3`

All ignored data targets refer to the canonical checkout at
`/Volumes/nvme1/codes/voxdelta/data`, not the isolated implementation worktree's empty
tracked `data/` shell.

Training and inference receive this absolute path explicitly and use Transformers in
local-only mode. They must not silently fall back to a network model ID.

## Training Profile and Memory Selection

The logical effective training batch remains 16. XLS-R supports only these exact
micro-batch profiles:

- micro-batch 8, gradient accumulation 2
- micro-batch 4, gradient accumulation 4
- micro-batch 2, gradient accumulation 8
- micro-batch 1, gradient accumulation 16

Evaluation batch size equals the selected micro-batch. The profile validator rejects
every other batch or accumulation combination. Learning rate, epochs, warmup, seed,
selection metric, early stopping, and inverse-frequency loss remain fixed.

The smoke starts at micro-batch four. If MPS reports an out-of-memory failure, the failed
process must leave no checkpoint directory, and the operator retries at two, then one.
The first profile that completes training and validation becomes the recorded smoke
profile. The retry sequence is explicit rather than an in-process automatic fallback so
one failed attempt releases the full model and optimizer state before the next attempt.

The CLI accepts `--micro-batch-size {1,2,4,8}` only for `wav2vec-xls-r`. Supplying it to
emotion2vec+, omitting the pinned local base path for XLS-R, or combining it with an
unapproved model identity fails with `invalid_training_profile`.

## Checkpoint Provenance

New XLS-R checkpoints record both the exact model revision and the verified base-weight
SHA-256. The runtime checkpoint contract requires those fields for XLS-R and rejects a
checkpoint whose revision or base hash differs from the approved constants. Local cache
paths are never stored in checkpoint metadata because they are machine-specific.

The production provider receives the pinned base-model directory separately, verifies
the three-artifact contract before constructing the model, and calls both the feature
extractor and classification model with the local directory and `local_files_only=True`.
It then loads the fine-tuned safetensors state as it does today. There is no network
fallback during checkpoint reload or evaluation.

Emotion2vec+ checkpoint schemas remain compatible and unchanged. Because no real XLS-R
checkpoint exists yet, legacy XLS-R checkpoint fixtures are migrated to the new strict
provenance contract rather than accepted without a revision.

## Smoke Execution

The run uses the existing absolute `data/manifests/emotion-smoke.jsonl` and publishes to
new, non-overwriting targets:

- base artifacts: `data/models/base/wav2vec2-xls-r-300m-1a640f3`
- checkpoint: `data/models/wav2vec-xls-r-300m-smoke`
- aggregate validation report:
  `data/benchmarks/wav2vec-xls-r-300m-smoke-validation.json`

Training reads only the smoke manifest's 140 train and 35 validation examples. The 35
previously exposed smoke-test examples and all remaining 3,585 test examples are unused
in this phase.
Evaluation explicitly requests `--split validation` and processes the same 35 validation
examples used for checkpoint selection. Its score is readiness evidence, not an unbiased
model-quality estimate.

The evaluator reloads the published checkpoint through `Wav2VecEmotionProvider` with the
pinned local base path. The report retains only aggregate provenance, completion rate,
macro-F1, per-label F1, confusion matrix, ECE, latency, elapsed time, requested device,
and peak RSS. It contains no transcripts, item IDs, audio paths, probabilities, or
per-item predictions.

## Interfaces

The implementation adds or extends these boundaries:

- `WAV2VEC_MODEL_REVISION` and exact artifact metadata beside the existing XLS-R model ID
- `prepare_wav2vec_base(output: Path) -> PreparedBaseModel`
- `validate_wav2vec_base(path: Path) -> PreparedBaseModel`
- `scripts/prepare_wav2vec_base.py --output <absolute-path>`
- `Wav2VecTrainingProfile` with the four exact micro-batch profiles
- `scripts/train_emotion.py --base-model-path <absolute-path>
  --micro-batch-size <1|2|4|8>` for XLS-R
- `Wav2VecEmotionProvider(..., base_model_path=<absolute-path>)`
- `scripts/evaluate_emotion_checkpoint.py --base-model-path <absolute-path>` for XLS-R

The generic experiment evaluator keeps test fakes simple by allowing its provider factory
contract to remain injectable. The default XLS-R provider construction gains the explicit
base-model path; emotion2vec+ construction is unchanged.

## Failure and Privacy Boundaries

- Reject relative, parent-traversing, symlink-traversing, incomplete, oversized, or
  hash-mismatched base-model paths.
- Reject existing preparation, checkpoint, or report targets instead of overwriting them.
- A failed download or training attempt leaves no partially published final directory.
- Never print transcript text, item IDs, audio paths, raw model output, exception text, or
  per-item predictions.
- Never mutate the full manifest, smoke manifest, normalized audio, source ZIPs, source
  CSVs, existing emotion2vec checkpoints, or prior benchmark reports.
- Do not disable PyTorch's MPS memory safety controls to force a larger batch.
- Do not evaluate the previously exposed smoke-test subset or any remaining test item in
  this phase.

## Testing

Unit tests prove:

- the exact model ID, revision, filenames, sizes, and hashes;
- atomic preparation, private permissions, non-overwrite behavior, and cleanup after a
  simulated download failure;
- rejection of relative paths, traversal, symlinks, unexpected files, wrong size, and
  wrong digest;
- local-only Transformers loading in training and provider inference;
- exact acceptance of the four effective-batch-16 profiles and rejection of every other
  batch combination;
- strict XLS-R checkpoint revision and base-hash provenance;
- CLI rejection of missing or architecture-incompatible base-model and micro-batch flags;
- provider and aggregate-report privacy contracts remain intact.

The implementation gate is the full backend pytest suite, Ruff lint and formatting,
strict mypy over `src` and `scripts`, `uv lock --check`, and `git diff --check`.

The real-data acceptance run additionally verifies:

- all three pinned base artifacts and their private permissions;
- no source-data fingerprint changes;
- one successful smoke checkpoint at micro-batch 4, 2, or 1;
- checkpoint reload through the production provider using local-only artifacts;
- completion of all 35 smoke-validation items;
- finite aggregate metrics, latency, elapsed time, and RSS;
- no final artifact or report contains item-level private data;
- no test-split item is evaluated during this phase.

## Success Criteria and Handoff

Preparation is complete when the exact pinned base is on NVMe, one allowed micro-batch
finishes the balanced smoke training, the checkpoint reloads without network access, and
all 35 validation examples produce a valid aggregate report. Low validation macro-F1 does
not fail this preparation gate; it is an early signal to consider before authorizing the
full run.

After completion, the user receives the selected micro-batch, elapsed time, peak RSS,
validation macro-F1 and per-label F1, artifact digests, and a recommendation on whether
full XLS-R training is technically safe and scientifically worthwhile. Full training,
resumable checkpoints, and final test comparison require a separate decision. That final
comparison must explicitly repair the earlier 35-item test exposure by excluding those
items or freezing a new untouched holdout before any model-quality claim is made.
