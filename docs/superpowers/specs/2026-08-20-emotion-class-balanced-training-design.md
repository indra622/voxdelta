# Class-Balanced Emotion Training Design

## Goal

Correct the majority-class collapse observed in the first full emotion2vec+ run while preserving that checkpoint as an auditable baseline. The balanced run optimizes validation macro-F1 and must produce a non-zero F1 for every canonical emotion before VoxDelta proceeds to the longer XLS-R comparison.

## Evidence and Approved Scope

The fixed training split contains 29,476 utterances. Its largest class is sadness with 12,502 items and its smallest is surprise with 574, a ratio of about 22:1. The unweighted emotion2vec+ head reached validation macro-F1 `0.2082607`; it predicted no disgust or surprise items and only two fear items. This points to label-frequency imbalance rather than damaged audio as the first issue to address.

This phase adds one deterministic class-weighting rule to both supported training architectures, retrains only the frozen-encoder emotion2vec+ classifier head immediately, and compares it with the existing baseline on the unchanged validation split. It does not relabel data, resample audio, unfreeze emotion2vec+, tune multiple weighting strengths, train XLS-R, or open the full test split.

## Approaches Considered

1. **Inverse-frequency class weights.** Keep every training item once per epoch and weight its cross-entropy contribution by its class frequency. This is deterministic, cheap, and applies consistently to both architectures. This is the selected approach.
2. **Minority-class oversampling.** Repeat minority examples until batches are more balanced. This can improve exposure but changes epoch composition and increases overfitting risk for the 729 surprise examples.
3. **Focal loss.** Down-weight easy examples dynamically. This adds another hyperparameter and makes it harder to distinguish class imbalance from example difficulty, so it is deferred unless inverse-frequency weighting fails.

## Weighting Contract

Weights are computed from the training split only, in canonical label order. For class `c`:

`weight[c] = training_item_count / (canonical_label_count * training_count[c])`

This normalization makes the average per-item weight equal to one, so the loss scale remains comparable with the unweighted run. The calculation rejects an empty training split, an unknown label, or any canonical label with zero training items. Validation and test labels never influence the weights.

Both emotion2vec+ and XLS-R use the same helper and weighted cross-entropy contract. The immediate emotion2vec+ run reuses the existing content-addressed frozen-encoder embedding cache; only the classifier head is retrained.

## Checkpoint Compatibility and Provenance

The existing unweighted checkpoint at `data/models/emotion2vec-plus-large-emotion` remains untouched. The balanced candidate is atomically published to a distinct output directory and publication still rejects an existing target.

New checkpoints use config schema version 2 and record the weighting method plus the exact canonical-order weights used for training. Runtime checkpoint validation accepts legacy schema version 1 checkpoints as unweighted baselines and schema version 2 checkpoints only when their class-weight metadata is complete, finite, positive, and in canonical order. Training metadata does not affect inference calculations.

## Validation and Decision Rule

Training still selects the best epoch by validation macro-F1 with seed 622 and the existing early-stopping profile. After publication, the balanced checkpoint is reloaded through the production provider and evaluated on the unchanged validation split only.

The candidate passes this phase when:

- validation macro-F1 is greater than the baseline `0.2082607`;
- all seven per-label F1 values are greater than zero;
- checkpoint provenance, permissions, probability normalization, and source-data integrity checks pass.

If either quality condition fails, the checkpoint remains an experiment artifact but does not become the comparison candidate. The next design decision is then capped inverse-frequency weighting or focal loss; the test split remains sealed.

## Failure and Privacy Boundaries

- Never overwrite the baseline checkpoint, a prior candidate, the manifest, normalized audio, source ZIPs, or source CSVs.
- Fail closed on incomplete label coverage or invalid/non-finite weights.
- Keep logs and reports aggregate-only: no transcripts, item IDs, audio paths, raw probabilities, or per-item predictions.
- Do not evaluate the full 3,620-item test split during tuning.

## Testing

Unit tests prove the exact train-only formula, canonical ordering, average weight normalization, failure on missing labels, and use of weighted cross-entropy in both training paths. Checkpoint tests prove schema v1 compatibility, strict schema v2 metadata validation, atomic publication, and non-overwrite behavior.

The implementation gate is the full backend pytest suite, Ruff lint and formatting, strict mypy over `src` and `scripts`, `uv lock --check`, and `git diff --check`. The real-data acceptance run additionally verifies cache reuse, a distinct balanced checkpoint, production-provider reload, aggregate validation metrics, and unchanged source-data fingerprints.
