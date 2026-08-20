# XLS-R Partial Last-Four Fine-Tuning Design

## Goal

Add a local, memory-bounded XLS-R 300M adaptation path that trains only encoder layers
20 through 23 plus the audio-classification projector and classifier. First run the
existing balanced 140-train/35-validation controlled smoke. Advance to a deterministic
700-train/175-validation development subset only if the smoke beats the prior full-model
smoke and no longer predicts a single class for every validation item.

This experiment answers whether high-level task adaptation can produce a useful signal
on the current 24 GiB Apple-silicon host without paying for remote full fine-tuning. It
does not claim final model quality and does not evaluate any test-split item.

## Existing Evidence

The trusted AI Hub emotion manifest contains 36,665 accepted 16 kHz mono utterances:
29,476 train, 3,569 validation, and 3,620 test. Labels are happiness, anger, disgust,
fear, neutral, sadness, and surprise. Training uses train-only inverse-frequency class
weights.

The pinned base is `facebook/wav2vec2-xls-r-300m` revision
`1a640f32ac3e39899438a2931f9924c02f080a54`, with verified weight SHA-256
`d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0`.
The model has 24 Transformer encoder layers indexed 0 through 23.

The prior controlled full-model smoke used balanced train 140 and validation 35. A
micro-batch of four failed under MPS memory pressure. Micro-batch two with gradient
accumulation eight completed in 1,640.61 seconds, but validation macro-F1 was
`0.0357143` and every item was predicted as surprise. That checkpoint remains immutable
readiness evidence, not a quality candidate.

## Approaches Considered

### 1. Train the final four encoder layers

Freeze the feature extractor, feature projection, positional convolution, encoder layers
0 through 19, and all other base parameters. Train encoder layers 20 through 23 plus the
top-level classification `projector` and `classifier`. This is the selected first
approach.

It introduces no model wrapper or new runtime dependency. Frozen lower layers do not
need optimizer state or parameter gradients, while the trainable upper layers can adapt
high-level emotion representations. The published checkpoint retains the ordinary
Transformers XLS-R audio-classification state dictionary, so production inference uses
the existing model topology.

### 2. LoRA on attention projections

Attach low-rank adapters to query and value projections across all 24 encoder layers and
train the classifier. This yields fewer trainable parameters and spreads adaptation
through the network, but it still backpropagates through all layers, retains substantial
activation memory, adds a PEFT dependency, and requires adapter merge and provenance
semantics. It is the next option only if partial-last-four fails.

### 3. Train the final eight encoder layers

This increases adaptation capacity without changing topology, but it also increases
gradient activation and optimizer memory. It is not run automatically. If the final-four
path is stable but only narrowly misses the quality gate, a separate design may compare
final eight against LoRA.

## Training Profile

Add a distinct XLS-R adaptation strategy named `partial-last4`. It is not a modification
of the accepted `full` profile.

The exact first-stage profile is:

- base model and revision: unchanged pinned XLS-R 300M artifacts;
- trainable encoder layers: `(20, 21, 22, 23)`;
- trainable heads: top-level `projector` and `classifier`;
- frozen encoder layers: `0` through `19`;
- feature extractor, feature projection, and remaining base modules: frozen;
- seed: `622`;
- learning rate: `2e-5`;
- epochs: `10`;
- warmup ratio: `0.1`;
- micro-batch: `2`;
- gradient accumulation: `8`;
- effective batch: `16`;
- evaluation: once per epoch;
- selection: validation macro-F1;
- early-stopping patience: `2`;
- loss: train-only inverse-frequency weighted cross-entropy;
- training clip: deterministic center crop capped at 20 seconds;
- evaluation: all deterministic windows, averaged per utterance.

Gradient checkpointing is not enabled in the first experiment. Only four encoder layers
participate in autograd, and adding checkpointing would introduce a second behavioral
change. It can be considered only after an observed memory failure.

## Trainable-Parameter Contract

Before constructing the optimizer, the trainer must inspect the loaded model and require
the expected 24-layer structure. It then freezes all parameters and enables gradients
only for names below these module prefixes:

- `wav2vec2.encoder.layers.20.`
- `wav2vec2.encoder.layers.21.`
- `wav2vec2.encoder.layers.22.`
- `wav2vec2.encoder.layers.23.`
- `projector.`
- `classifier.`

The profile fails closed if any expected prefix has no parameter, if any trainable
parameter is outside the allowlist, or if no parameter is trainable. AdamW receives only
the allowed trainable parameters, not `model.parameters()`.

The trainer records aggregate trainable and total parameter counts. It never prints or
stores data paths or per-item information.

## Checkpoint and Runtime Contract

The previous XLS-R full-fine-tuning checkpoint schema remains accepted and unchanged.
Partial-last-four checkpoints use a new strict XLS-R schema containing:

- `adaptation_strategy: "partial-last4"`;
- `trainable_encoder_layers: [20, 21, 22, 23]`;
- `trainable_module_prefixes` with the six exact prefixes;
- positive `trainable_parameter_count` and `total_parameter_count`;
- the existing pinned model revision and base-weight SHA-256;
- the existing seven-label mapping, class-weighting contract, and class weights.

The model file contains the complete state dictionary after training rather than a delta
or adapter. `Wav2VecEmotionProvider` validates the new provenance fields, reconstructs
the same ordinary audio-classification model from the pinned local base, and loads the
complete state dictionary. Inference does not need to reproduce the training freeze mask.

The new checkpoint and reports use non-overwriting paths distinct from the prior full
smoke artifacts:

- smoke checkpoint: `data/models/wav2vec-xls-r-300m-partial-last4-smoke`;
- smoke report:
  `data/benchmarks/wav2vec-xls-r-300m-partial-last4-smoke-validation.json`;
- development manifest: `data/manifests/emotion-partial-last4-development.jsonl`;
- development checkpoint:
  `data/models/wav2vec-xls-r-300m-partial-last4-development`;
- development report:
  `data/benchmarks/wav2vec-xls-r-300m-partial-last4-development-validation.json`.

## Interfaces

The training profile exposes the strategy explicitly instead of inferring it from frozen
parameters. The training CLI accepts:

```text
--adaptation-strategy full|partial-last4
```

For `partial-last4`, the CLI requires micro-batch two and derives accumulation eight.
The existing `full` profile and commands preserve their current behavior.

A focused deterministic development-manifest builder consumes the full trusted manifest
and publishes only:

- 100 train examples per label, 700 total;
- 25 validation examples per label, 175 total;
- zero test examples.

Selection uses the existing seed-622 stable ranking. The builder strips transcript text,
validates source/split disjointness, refuses an existing output, and writes a private
aggregate-only summary. It does not generalize the existing smoke builder in a way that
could accidentally change the accepted 140/35/35 smoke manifest.

## Execution Flow

### Stage A: Controlled smoke

1. Verify the full manifest, smoke manifest, normalized audio, pinned base, and prior
   artifacts are unchanged.
2. Train `partial-last4` on the existing smoke manifest's 140 train items.
3. Select by its 35 validation items and atomically publish a new checkpoint.
4. Reload through `Wav2VecEmotionProvider` and evaluate the same 35 validation items.
5. Publish only aggregate metrics and compare with the prior full-model smoke.

Stage A passes only when all conditions hold:

- training exits successfully without an MPS memory failure;
- the checkpoint reloads using local-only pinned artifacts;
- validation completes 35 of 35 items;
- macro-F1 is strictly greater than `0.0357143`;
- validation predictions contain at least two distinct canonical classes;
- source, manifest, and prior-artifact fingerprints remain unchanged.

If any condition fails, Stage B is not run. The outcome becomes evidence to design the
LoRA baseline or reconsider a no-training method.

### Stage B: Balanced development subset

Only after Stage A passes, publish the deterministic 700/175 development manifest and
run the identical partial-last-four profile. Stage B passes when:

- validation completes 175 of 175 items;
- macro-F1 is strictly greater than the prior full-model smoke score `0.0357143`;
- predictions contain at least two distinct canonical classes;
- the checkpoint reloads through the production provider;
- no source, full-manifest, smoke-manifest, or prior-artifact fingerprint changes.

Stage B is development evidence, not a final-quality result. Its validation data is used
for selection, so it does not authorize opening the test split or claiming unbiased
generalization.

## Failure and Privacy Boundaries

- Reject an unexpected layer count, missing module prefix, or trainable parameter outside
  the exact allowlist before optimizer creation.
- Reject missing, relative, unverified, or provenance-mismatched base paths.
- Reject existing checkpoint, report, or development-manifest targets instead of
  overwriting them.
- A training failure or MPS OOM leaves no final checkpoint directory.
- Do not disable MPS high-watermark safety controls.
- Do not automatically switch strategy, layer count, batch size, learning rate, or
  gradient checkpointing after a failure.
- Do not modify raw ZIP/CSV files, normalized WAV files, existing manifests, prior
  checkpoints, or prior reports.
- Do not load or evaluate any test-split item.
- User-facing logs and reports contain no transcript, item ID, audio path, probability
  vector, or per-item prediction.

## Testing

Unit and contract tests prove:

- strict acceptance of `full` and `partial-last4` strategies and rejection of unknown
  strategy/profile combinations;
- exactly encoder layers 20 through 23, `projector`, and `classifier` are trainable;
- the optimizer receives only allowed trainable parameters;
- wrong layer counts, missing prefixes, and parameter leaks fail closed;
- partial checkpoints publish the complete model state and exact provenance;
- full-profile checkpoint compatibility remains unchanged;
- provider reload accepts a matching partial checkpoint and rejects altered strategy,
  layer, prefix, count, revision, or base-hash metadata;
- the development manifest is deterministic, balanced 700/175, transcript-free,
  train/validation-only, private, non-overwriting, and split-disjoint;
- CLI flags reject incompatible architecture, strategy, batch, and base-path choices;
- aggregate reports retain existing privacy constraints.

The implementation gate is the full backend pytest suite, Ruff lint and formatting,
strict mypy over `src` and `scripts`, `uv lock --check`, and `git diff --check`.

## Non-Goals

- LoRA, QLoRA, final-eight-layer training, or automatic hyperparameter search;
- changing the seven labels, split, class weights, crop, seed, or model revision;
- training on the full 29,476-item train split;
- evaluating the 3,620 nominal test items or repairing the previously exposed 35-item
  smoke-test subset;
- remote GPU upload or RunPod execution;
- changing the production model topology.

## Handoff

After Stage A, report elapsed time, peak RSS, trainable/total parameter counts,
macro-F1, per-label F1, confusion matrix, predicted-class count, checkpoint digest, and
report digest. If Stage A passes, execute and report the same aggregate evidence for
Stage B. The next decision is based on those two gates: consider a larger local run only
after Stage B passes; otherwise stop partial-last-four and design LoRA or a no-training
baseline.
