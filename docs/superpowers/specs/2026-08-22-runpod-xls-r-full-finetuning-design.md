# RunPod XLS-R Full Fine-Tuning Design

## Goal

Fine-tune every parameter of the pinned `facebook/wav2vec2-xls-r-300m` model on the
accepted AI Hub seven-emotion corpus in a RunPod Secure Cloud 48 GB CUDA Pod. The
experiment must determine whether task-specific encoder adaptation materially beats the
accepted frozen emotion2vec+ validation baseline without repeating the single-class
collapse seen in the local full, last-four-layer, and seeded LoRA smoke runs.

The approved planning assumption is that the required AI Hub overseas-transfer agreement
has been obtained. Its evidence is an execution-time preflight item, not a design blocker.
No data transfer or Pod mutation occurs as part of this design phase.

## Evidence and Fixed Inputs

The canonical emotion corpus contains 36,665 accepted utterances. Each item is 16 kHz
mono PCM16 and has one of seven labels: happiness, anger, disgust, fear, neutral,
sadness, or surprise. Transcripts are not model inputs and are excluded from every
RunPod transfer artifact.

The deterministic split is:

- train: 29,476
- validation: 3,569
- nominal test: 3,620

The train distribution is anger 5,564, disgust 1,689, fear 2,013, happiness 2,791,
neutral 4,343, sadness 12,502, and surprise 574. The accepted frozen
emotion2vec+ checkpoint has validation macro-F1 `0.2400352`; all seven validation
label F1 values are positive.

The pinned XLS-R base identity remains:

- repository: `facebook/wav2vec2-xls-r-300m`
- revision: `1a640f32ac3e39899438a2931f9924c02f080a54`
- base weight SHA-256:
  `d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0`
- seed: 622
- training window: 20 seconds
- selection metric: validation macro-F1

Three local Stage A variants used train 140 and validation 35 and produced the same
macro-F1 `0.0357143` with all predictions assigned to surprise:

1. all XLS-R parameters trainable;
2. encoder layers 20 through 23 plus projector and classifier trainable;
3. seeded rank-8 LoRA on encoder layers 16 through 23 q/v projections plus projector
   and classifier trainable.

The corrected LoRA run removed ambient Torch RNG dependence and still reproduced the
collapse. Those runs remain evidence but are not repeated on RunPod.

## Approaches Considered

### 1. Run the existing inverse-frequency profile on all training data

This is the shortest path, but it takes the most expensive step before separating
sampling, loss weighting, and small-stage optimization failure. It may reproduce the
surprise collapse while revealing little about its cause.

### 2. Compare two bounded pilot recipes, then expand only the winner

This is the selected approach. It compares class-balanced sampling without loss weights
against natural-distribution sampling with moderated weights. Both pilots use the same
number of training examples, maximum optimizer-update budget, validation items, model
identity, and seed. A fixed gate prevents a full run when neither recipe learns a usable
seven-class boundary.

### 3. Change the base model or label ontology before full fine-tuning

WavLM, HuBERT, three-class emotion, valence-arousal, and transcript fusion remain valid
fallbacks. Introducing them now would no longer answer whether full XLS-R adaptation can
beat the existing emotion2vec+ baseline, so they are out of scope.

## Architecture

The implementation is a five-stage, restartable experiment with immutable artifacts:

1. `package`: build the privacy-minimized train/validation bundle and freeze the
   final-holdout identity without reading holdout audio;
2. `preflight`: verify the Pod, code, base model, CUDA profile, data, and output targets;
3. `pilot`: run the two approved 3,500/350 recipes and choose one by a frozen rule;
4. `full`: train on the complete 29,476-item pool and evaluate all 3,569 validation items;
5. `final`: after candidate freeze, package, transfer, and evaluate the untouched
   3,585-item holdout exactly once for both frozen comparison models.

A single experiment ledger records the configuration digest, code commit, environment
image digest, base-model digest, manifest digests, stage status, checkpoint identities,
and aggregate metrics. A stage may resume only from an atomically published checkpoint
whose configuration and input digests exactly match the ledger.

The experiment does not require changes to the production VoxDelta API. Model promotion
and service integration follow only after the final comparison.

## Final Holdout Repair

The nominal 3,620-item test split is not fully untouched. The earlier emotion2vec+
integration smoke evaluated five test items per label, 35 total. Those 35 are permanently
excluded from final-quality claims.

The final holdout contains the remaining 3,585 items:

- anger: 695
- disgust: 212
- fear: 216
- happiness: 330
- neutral: 578
- sadness: 1,485
- surprise: 69

Exclusion uses the immutable semantic fingerprints already present in the smoke and full
manifests, not row position, item ID, or path text. The holdout builder fails unless it
finds exactly five exposed fingerprints per label, removes exactly 35 unique items, and
produces the exact counts above. It publishes a digest and aggregate counts only.

The train/validation bundle and the final-holdout bundle are separate. Before candidate
freeze, the package stage may read existing test-manifest metadata only to derive the
3,585-item identity, counts, and digest. It must not open holdout audio or create the
holdout archive. The final-holdout archive is built and transferred only after the full
checkpoint, configuration, validation report, and comparison decision rule are frozen.

## Local Label and Audio Audit

Before packaging, a local-only audit queue contains 210 train/validation items:

- 15 deterministic random train examples per label;
- 15 highest-loss validation examples per label under the accepted emotion2vec+
  checkpoint.

The reviewer records one of `agree`, `ambiguous`, `disagree`, or `audio_defect` plus an
optional local note. Item-level audit content remains mode `0600`, Git-ignored, and local.
Only counts by label and outcome enter the aggregate report.

The audit does not rewrite labels or exclude subjective disagreements from this controlled
XLS-R comparison. Any manifest/audio mismatch, unreadable file, checksum failure, wrong
sample rate, or wrong channel count is a hard packaging failure. Subjective ambiguity is
reported as an interpretation limit and informs a later ontology decision.

## Privacy-Minimized Transfer Contract

The package boundary creates two artifacts at different times:

1. before training, a train/validation archive with 33,045 normalized WAV files plus a
   sanitized relative-path manifest;
2. after candidate freeze, a final-holdout archive with 3,585 normalized WAV files plus
   its sanitized relative-path manifest.

Transferred manifests contain only a stable opaque item key, relative audio path, split,
canonical emotion, and audio SHA-256. They contain no transcript, call ID, speaker ID,
absolute path, original source filename, free text, or local audit judgment.

Each archive has a sidecar containing its SHA-256, byte length, file count, split counts,
label counts, manifest digest, and schema version. Directories are mode `0700`; files are
mode `0600`. Archive creation rejects symlinks, path traversal, duplicate semantic
fingerprints, duplicate relative paths, missing audio, unexpected formats, and an
existing output target.

Transfer uses SSH `rsync --partial` to a RunPod Secure Cloud Pod. The remote preflight
recomputes archive and manifest digests before extraction, rejects unexpected archive
members, and extracts beneath one private experiment root. Raw ZIPs, source CSVs,
embedding caches, prior item-level outputs, environment files, and credentials are never
transferred.

## RunPod Environment

The planned primary resource is one RunPod Secure Cloud A40 48 GB Pod. An equivalent
48 GB CUDA GPU is allowed only when the entire environment profile below remains fixed.

- one GPU visible to the process;
- private 100 GB volume disk mounted beneath `/workspace`;
- SSH only; no application or notebook port is exposed;
- Python 3.12;
- repository dependencies installed strictly from `uv.lock`;
- container image selected at package time and recorded by immutable digest in the frozen
  experiment configuration; tag-only image identities are rejected;
- CUDA, cuDNN, driver, PyTorch, Transformers, and GPU model recorded in preflight;
- code commit and dirty-state check recorded before any actual-data read.

Preflight fails if the worktree is dirty, the image is not the approved digest, the base
model does not match the pinned revision and hashes, more or fewer than one GPU is
visible, the volume has less than 80 GB free, BF16 is unavailable, or any input/output
target violates the private non-overwrite contract.

The planning assumption about transfer permission is checked once here by confirming the
agreed evidence is present. The ledger records only `transfer_authorization_present: true`;
it does not copy the evidence or its private contents.

## CUDA Memory Profile

The logical effective batch is fixed at 16. Preflight tries these profiles in fresh
processes, in order:

- micro-batch 8, gradient accumulation 2;
- micro-batch 4, gradient accumulation 4;
- micro-batch 2, gradient accumulation 8;
- micro-batch 1, gradient accumulation 16.

Each attempt uses synthetic audio and then a 14-train/7-validation actual-data integration
slice with one item per label in validation. It performs forward, backward, optimizer,
checkpoint publish, reload, and provider evaluation. An OOM attempt leaves no final
checkpoint and the next attempt begins in a new process. The largest passing profile is
frozen for every pilot and full run.

Training uses CUDA BF16 autocast, gradient checkpointing, AdamW, weight decay `0.01`,
linear warmup ratio `0.1`, and gradient norm clipping at `1.0`. The encoder learning rate
is `1e-5`; projector and classifier learning rate is `1e-4`. All XLS-R parameters,
including the convolutional feature extractor, are trainable.

Pilot runs use at most five epochs with early-stopping patience two. The full run uses at
most ten epochs with early-stopping patience two. Best checkpoint selection is validation
macro-F1. Training crops are deterministic per `(seed, epoch, semantic fingerprint)`;
validation uses the existing deterministic multi-window averaging contract.

The experiment records seed and RNG states but does not claim bitwise equivalence across
different CUDA hardware or driver versions.

## Pilot Manifests and Recipes

Both pilots use the same balanced 350-item validation subset: 50 deterministic examples
per label from the canonical validation split. The subset is selected before either run
and its digest is frozen.

### Pilot A: balanced sampler, unweighted loss

- train pool: exactly 500 deterministic examples per label, 3,500 total;
- epoch order: deterministic class-balanced batches;
- loss: ordinary cross-entropy with seven weights equal to `1.0`.

If this recipe wins, the full run uses all 29,476 train items as its source pool and a
deterministic class-balanced sampler with replacement. Each epoch contains exactly
29,476 draws. Loss remains unweighted.

### Pilot B: natural sampler, square-root inverse-frequency loss

- train pool: 3,500 deterministic examples sampled proportionally from the canonical
  train distribution using largest-remainder allocation: anger 661, disgust 201,
  fear 239, happiness 331, neutral 516, sadness 1,484, and surprise 68;
- epoch order: deterministic uniform shuffle without replacement;
- loss weights: `sqrt(N / (K * n_c))`, normalized so their arithmetic mean is `1.0`,
  using only the selected train items.

If this recipe wins, the full run uses every train item once per epoch in a deterministic
uniform shuffle. Square-root inverse-frequency weights are recomputed from all 29,476
train items. Sampling remains unweighted.

No run combines class-balanced sampling and non-uniform loss weights.

## Pilot Gate and Winner Selection

A pilot is eligible only if all conditions hold:

- provider evaluation completes 350/350;
- validation macro-F1 is strictly greater than `0.15`;
- at least six distinct classes are predicted;
- at least five canonical labels have F1 strictly greater than zero;
- loss, logits, probabilities, weights, gradients, and reported metrics are finite;
- the best checkpoint reloads through the ordinary PEFT-free production provider;
- no test or final-holdout input is opened.

If neither pilot is eligible, the runner stops before full training and keeps balanced
emotion2vec+ as the default candidate.

If one pilot is eligible, it wins. If both are eligible, higher macro-F1 wins. When the
absolute macro-F1 difference is less than `0.01`, lower ECE wins. When the ECE difference
is also less than `0.01`, Pilot B wins because it uses every full-train item once per
epoch and introduces less resampling.

The winner decision and all underlying aggregate metrics are published atomically before
the full-run target can be created.

## Full Training and Validation Gate

The full run uses the selected recipe, the frozen CUDA batch profile, all 29,476 train
items as the source pool, and all 3,569 validation items. It restores the best validation
checkpoint before publishing the ordinary full-model safetensors checkpoint.

The full checkpoint is eligible for final holdout only when:

- provider validation completes 3,569/3,569;
- macro-F1 is strictly greater than `0.2400352`;
- every canonical label has F1 strictly greater than zero;
- all seven classes are predicted;
- aggregate ECE, latency, elapsed time, peak CPU RSS, and peak CUDA allocated/reserved
  memory are finite;
- the checkpoint and report pass provenance, permission, non-overwrite, and privacy
  validation.

Failure leaves the accepted emotion2vec+ checkpoint as the default and permanently seals
the final-holdout stage for this recipe.

## Checkpoint and Resume Contract

Every completed epoch publishes an immutable recovery directory containing:

- full model state;
- optimizer and scheduler state;
- BF16 scaler state when the runtime provides one;
- Python, Torch CPU, and Torch CUDA RNG states;
- epoch, optimizer step, best score, stale-epoch count, and selected batch profile;
- code, environment, model, configuration, sampler, and manifest digests.

Publication uses a sibling staging directory, private permissions, complete validation,
and one atomic rename. Existing recovery, best-checkpoint, report, or ledger targets are
never overwritten.

Resume chooses only the highest complete epoch whose digests exactly match the requested
run. A mismatch fails with a stable code and never falls back to weights-only resume.
Connection loss or Pod restart therefore repeats at most one incomplete epoch. Pilot A,
Pilot B, full, and final outputs have distinct roots and cannot resume from one another.

## Final Comparison

After the validation gate passes, the full XLS-R checkpoint, accepted emotion2vec+
checkpoint, both provider configurations, label mapping, final metric schema, and decision
rule are frozen by digest. Only then may the final-holdout archive be transferred and
opened.

Both frozen models evaluate the same 3,585 items exactly once. Reports contain only
aggregate completion, macro-F1, per-label F1, confusion matrix, ECE, latency, elapsed
time, CPU RSS, CUDA memory where applicable, and provenance digests.

XLS-R becomes the VoxDelta default emotion model only when:

- both providers complete 3,585/3,585;
- XLS-R holdout macro-F1 is strictly greater than emotion2vec+ holdout macro-F1;
- all seven XLS-R holdout label F1 values are strictly greater than zero;
- neither report violates its integrity or privacy contract.

Otherwise emotion2vec+ remains the default. Final-holdout results do not authorize
hyperparameter changes, reruns, label remapping, or a second holdout evaluation.

## Data Flow

1. Canonical local manifests and normalized WAV files enter audit and package validation.
2. The package stage emits the train/validation archive and a metadata-only sealed
   final-holdout identity; it does not open holdout audio.
3. `rsync` transfers only train/validation before candidate freeze.
4. RunPod preflight verifies environment, authorization presence, archive, base, code, and
   CUDA profile.
5. The runner creates deterministic pilot manifests, runs both pilots, and publishes the
   winner decision.
6. The winner expands to full train/validation with immutable epoch recovery.
7. A PEFT-free provider reload evaluates full validation and applies the promotion gate.
8. After freeze, the final-holdout archive is built, transferred, and both frozen models
   are evaluated once.
9. Aggregate artifacts return to local storage; their digests and provider reload are
   verified before Pod and volume deletion.

## Interfaces and File Boundaries

The implementation should preserve focused responsibilities:

- `backend/src/voxdelta/evaluation/runpod_package.py`: sanitized manifest models,
  holdout exclusion, archive construction, sidecar digests, and archive validation;
- `backend/src/voxdelta/evaluation/wav2vec_full_training.py`: full-training profile,
  sampling and loss recipes, CUDA precision, parameter groups, evaluation, and payload;
- `backend/src/voxdelta/evaluation/experiment_ledger.py`: immutable stage records,
  checkpoint identities, transition validation, and resume selection;
- `backend/src/voxdelta/evaluation/full_finetuning_gate.py`: pilot eligibility, winner
  selection, validation promotion, and final comparison decisions;
- `backend/scripts/prepare_runpod_emotion_data.py`: local package CLI;
- `backend/scripts/run_wav2vec_full_experiment.py`: RunPod preflight and staged runner;
- `backend/scripts/evaluate_final_emotion_holdout.py`: one-time two-provider comparison;
- `backend/tests/evaluation/`: focused tests matching each boundary;
- `data/README.md`: operator commands, transfer sequence, recovery, result retrieval, and
  deletion checklist.

Existing generic provider and aggregate experiment report contracts are reused. The
ordinary production provider must load the final checkpoint without PEFT or a training
runtime. Existing local smoke, partial, LoRA, emotion2vec+, and benchmark artifacts remain
immutable.

## Failure and Privacy Boundaries

- Fail closed on dirty code, identity mismatch, non-private paths, symlinks, archive
  traversal, unexpected files, duplicate fingerprints, non-finite values, or output
  collisions.
- Never print or publish transcript text, item IDs, call IDs, speaker IDs, absolute audio
  paths, raw per-item logits, probabilities, predictions, audit notes, credentials, SSH
  material, or transfer-authorization contents.
- Stable user-facing errors may identify the stage and a non-sensitive error code only.
  Detailed local and remote logs remain mode `0600` and must still redact private item
  fields and secrets.
- Never read the final holdout before candidate and decision-rule freeze.
- Never use the nominal exposed 35 test items in the final comparison.
- Never overwrite an existing package, checkpoint, ledger, report, or result directory.
- Never silently change GPU type, batch profile, seed, recipe, learning rates, manifest,
  label mapping, or metric schema during resume.
- Pod or network failure never triggers automatic final-holdout evaluation.
- Result retrieval and local digest verification must complete before Pod and volume
  deletion. Deletion is operator-confirmed and recorded as an aggregate timestamp only.

## Testing

Unit and contract tests must prove:

- exact sanitized manifest fields and rejection of private or absolute-path fields;
- deterministic package and sidecar digests, exact split/label counts, private modes,
  non-overwrite, symlink rejection, and archive traversal rejection;
- exact removal of the 35 exposed fingerprints and exact 3,585-item holdout counts;
- deterministic 210-item audit selection without publishing item-level content;
- exact pilot sample counts, no train/validation overlap, and stable digests;
- class-balanced sampling and unweighted loss never combine with non-uniform weights;
- square-root weight formula, normalization, and train-only derivation;
- exact optimizer parameter groups and learning rates;
- CUDA BF16, gradient checkpointing, clipping, and effective-batch-16 profile validation;
- deterministic per-epoch crop and sampler order under seed 622;
- atomic epoch checkpoints, full resume state, digest mismatch rejection, and separation of
  pilot/full/final roots;
- exact pilot eligibility and deterministic tie-breakers;
- exact full validation and final comparison gates;
- final-holdout I/O rejection before candidate freeze and after one completed evaluation;
- provider reload from the ordinary full-model checkpoint;
- aggregate-report privacy and finite-metric contracts;
- stable error codes without private exception text.

Integration tests use synthetic audio and fake transfer/checkpoint backends. They cover an
OOM fallback in fresh processes, interrupted-epoch resume, full stage transitions, sealed
holdout rejection, and one-time final comparison without requiring actual AI Hub audio or
a live RunPod account.

The repository gate is:

- full backend pytest suite;
- Ruff lint and formatting;
- strict mypy over `src` and `scripts`;
- `uv lock --check`;
- `git diff --check`;
- secret, absolute-private-path, transcript, and item-level output scans over tracked
  changes.

The live RunPod acceptance gate additionally verifies the recorded image and GPU profile,
input and output digests, exact item counts, stage ledger, resume boundary, production
provider reload, aggregate reports, local result retrieval, and volume-deletion record.

## Out of Scope

- LoRA, partial encoder training, or repeating the 140/35 local smoke;
- changing the seven canonical labels or relabeling subjective audit disagreements;
- transcript fusion, ASR features, multimodal models, WavLM, HuBERT, or CLAP;
- multi-seed confirmation or hyperparameter search beyond the two fixed pilot recipes;
- distributed or multi-GPU training;
- automatic RunPod account creation, billing, Pod creation, or destructive volume deletion;
- production API default changes before the final comparison passes;
- any second use of the final holdout.

## Success Criteria and Handoff

The implementation phase is successful when it can reproducibly package privacy-minimized
inputs, execute or resume both pilots on the fixed RunPod profile, stop on a failed gate,
expand the deterministic winner to full train/validation, publish a PEFT-free checkpoint,
and keep the final holdout inaccessible until freeze.

The experiment phase is successful when the full validation gate passes, both frozen
models complete the one-time 3,585-item final comparison, all outputs return with matching
digests, and the Pod volume can be deleted after local verification. XLS-R promotion is a
separate outcome determined only by the frozen final comparison rule; a valid experiment
may correctly conclude that emotion2vec+ remains the default.
