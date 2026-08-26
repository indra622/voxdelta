# RunPod XLS-R Full Fine-Tuning Implementation Plan

## Outcome

Build a restartable, privacy-minimized RunPod toolchain inside `runpod/` that can package
the approved AI Hub train/validation corpus, select one of two frozen pilot recipes,
fine-tune the pinned XLS-R 300M model, enforce the validation gate, and perform one sealed
3,585-item final comparison only after an explicit user handoff.

All live registry, SSH, `rsync`, RunPod, and deletion commands are executed manually by
the user. The agent produces the Docker image definition and verified handoff packets,
generates exact commands, and evaluates returned aggregate evidence; it does not log in to
or operate the user's RunPod account.

The scientific constants and gates come from the approved design. This plan is an
execution work order, not a mandatory Superpowers procedure. Implementation should use
small reviewable commits and normal tests; debugging and completion verification use the
shared `systematic-debugging` and `verification-before-completion` skills when applicable.

## Fixed Inputs

- model: `facebook/wav2vec2-xls-r-300m` at revision
  `1a640f32ac3e39899438a2931f9924c02f080a54`;
- base weight SHA-256:
  `d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0`;
- seed: `622`;
- train/validation: `29,476 / 3,569`;
- pilot train/validation: `3,500 / 350` per recipe;
- final holdout: `3,585`, excluding the 35 previously exposed test fingerprints;
- primary hardware: one RunPod Secure Cloud A40 48 GB GPU, private 100 GB volume;
- accepted comparison baseline: frozen emotion2vec+ validation macro-F1 `0.2400352`.

Changing any fixed input requires a new configuration identity and an explicit design
decision. Resume may never absorb such a change silently.

## Implementation Sequence

### 1. Isolated package and configuration contract

Create `runpod/pyproject.toml`, its own `uv.lock`, `config/experiment.toml`, and
`src/voxdelta_runpod/config.py`. The RunPod project may depend on `../backend` as a local
package, but its dependencies and CLIs must run through `uv --project runpod`. Define
strict immutable models for the scientific constants, expected counts, image digest,
volume root, and stage names. Reject secrets, hostnames, absolute local data paths, or
tag-only image identities in tracked configuration.

Tests prove exact defaults, unknown-field rejection, digest stability, and secret/path
rejection.

### 2. Docker image and manual command handoff

Create `runpod/docker/Dockerfile`, `.dockerignore`, build/inspect/smoke scripts, and the
generator for `runpod/dist/<run-id>/push-image.sh`. The image is linux/amd64 CUDA, starts
from an approved digest rather than a mutable tag, installs only the frozen RunPod/backend
environment, and records the Git and lock identities. BuildKit secrets may be mounted for
private registry operations but may never appear in an image layer or build metadata.

The agent builds and smoke-tests a portable OCI archive, emits an SBOM and immutable
expected manifest digest, and creates the user's digest-preserving registry
push/digest-check command. The user authenticates, pushes, and selects the verified digest
in RunPod. Tests inspect the image history/context allowlist and prove that audio,
manifests, `.env`, keys, checkpoints, results, and local paths cannot enter the build
context.

### 3. Local audit and privacy-minimized packaging

Implement `src/voxdelta_runpod/package.py` and a local `scripts/package_data.py` CLI.
Reuse canonical backend manifest/audio validation while keeping all RunPod-specific
archive and transfer rules here. Build the deterministic 210-item local audit queue,
aggregate its outcomes, sanitize train/validation records, validate PCM16 mono 16 kHz
audio, and publish the 33,045-file archive plus sidecar atomically.

Derive and seal the 3,585-item holdout identity from metadata only. Do not open or archive
holdout audio. Tests cover exact counts, five exposed fingerprints per label, field
allowlisting, deterministic digests, private modes, non-overwrite, symlink/path-traversal
rejection, and absence of transcripts or original identifiers.

### 4. Immutable ledger and recovery checkpoints

Implement `src/voxdelta_runpod/ledger.py` and `checkpoint.py`. Record configuration, code,
container, base-model, archive, manifest, sampler, and report digests with legal stage
transitions. Publish epoch recovery directories through sibling staging plus atomic rename.
Persist model, optimizer, scheduler, scaler-if-present, Python/Torch CPU/CUDA RNG states,
epoch, optimizer step, best metric, patience state, and batch profile.

Tests cover interrupted publication, highest-complete-epoch selection, digest mismatch,
wrong-stage recovery, existing-target refusal, and at-most-one-incomplete-epoch replay.

### 5. Deterministic pilot manifests and recipes

Implement `src/voxdelta_runpod/recipes.py`. Freeze the shared balanced 350-item validation
subset. Pilot A uses 500 train items per label, a class-balanced sampler, and unweighted
cross-entropy. Pilot B uses the fixed largest-remainder natural counts, deterministic
shuffle, and mean-one square-root inverse-frequency weights. Implement full-data expansion
for only the selected recipe.

Tests prove sample counts, no train/validation overlap, stable ordering/digests, exact
weights, train-only weight derivation, and the invariant that balanced sampling and
non-uniform loss weights never coexist.

### 6. CUDA full-training runtime

Implement `src/voxdelta_runpod/training.py` on top of the backend checkpoint/provider
contracts. Use full XLS-R parameter training, BF16 autocast, gradient checkpointing,
effective batch 16, encoder LR `1e-5`, head LR `1e-4`, AdamW weight decay `0.01`, linear
warmup `0.1`, clipping `1.0`, deterministic crop keys, epoch validation, best-checkpoint
restore, and patience two.

After promotion, run one local validation-only pass against the immutable release. Fit one
temperature by negative log-likelihood and one abstain threshold at the explicitly selected
target coverage, then publish those scalars as a separate release-bound artifact. The
resumable private cache is per-item bound and never shipped; the sealed final holdout is not
an input.

The preflight memory probe tries micro-batches `8, 4, 2, 1` in fresh processes with
accumulation `2, 4, 8, 16`. Each attempt covers synthetic forward/backward and the fixed
14-train/7-validation actual-data slice, checkpoint publication, reload, and provider
evaluation. Tests use fake CUDA/training backends and assert exact parameter groups,
precision, process isolation, OOM fallback, finite-state checks, and ordinary PEFT-free
provider reload.

### 7. Mechanical gates and sealed final comparison

Implement `src/voxdelta_runpod/gates.py`. Encode pilot eligibility, deterministic winner
selection, the full validation threshold, and the frozen final comparison without operator
judgment. Implement a holdout capability token derived from the frozen candidate,
configuration, reports, metric schema, and decision rule. No other stage may build, open,
or evaluate the final package.

Tests cover every threshold boundary and tie-breaker, all failed-gate stop paths, token
tampering, one-time final use, exact 3,585 completion for both providers, and promotion
only when the frozen rule passes.

### 8. Local/remote orchestration CLIs

Add narrowly scoped scripts:

- `scripts/preflight_local.py`: audit/package/config/code checks with aggregate summary;
- `scripts/render_operator_commands.py`: stage-specific registry, SSH, `rsync`, remote
  execution, resume, and result-download command packets matching `OPERATOR.md`;
- `scripts/preflight_remote.py`: image/GPU/storage/base/code/archive verification;
- `scripts/run_experiment.py`: `pilot`, `full`, and safe `resume` stages;
- `scripts/freeze_candidate.py`: publish validation decision and holdout capability;
- `scripts/package_final_holdout.py`: local one-time package after user authorization,
  including the minimal authenticated full-stage state required by a fresh Pod;
- `scripts/evaluate_final.py`: two-provider aggregate final evaluation exactly once;
- `scripts/verify_retrieved_results.py`: local digest, ledger, privacy, and provider checks;
- `scripts/build_release.py`: assemble the immutable raw release bundle after promotion;
- `scripts/verify_release.py`: re-verify one release bundle's integrity and identity;
- `scripts/smoke_release_inference.py`: offline reload and one synthetic forward pass;
- `scripts/build_calibration.py`: fit validation-only temperature scaling and publish a
  separate release-bound abstention artifact.

CLIs return stable non-sensitive error codes, refuse overwrite, and never print private
paths, item identities, credentials, raw predictions, or evidence contents. The user runs
all generated live commands manually. Training, full/resume, final holdout, and result
download remain separate packets rather than one all-powerful command.

### 9. Synthetic end-to-end and repository integration

Add a synthetic seven-label fixture and fake transfer/GPU/checkpoint backends under
`runpod/tests/`. Exercise package through pilot selection, full training, candidate freeze,
single final evaluation, result retrieval, and deletion-readiness reporting. Add failure
scenarios for corrupt transfer, OOM fallback, interrupted epoch, stale resume, failed pilot
gate, failed validation gate, and forbidden holdout access.

Run the RunPod suite and the existing backend suite together. The production backend may
expose small reusable primitives when necessary, but it must not gain live RunPod state or
depend on `voxdelta_runpod`.

### 10. Operator documentation and live dry run

Keep the Korean command contract in `runpod/OPERATOR.md` synchronized with generated
packets, including expected aggregate output, safe retry, incident log location, and which
manual action is required next. Run a no-data local dry run, a linux/amd64 Docker build and
smoke check, and a synthetic remote dry run before actual-data packaging. Record the exact
image digest, environment versions, GPU profile, and disk check in the ledger.

The actual experiment starts only after all implementation gates pass. The user performs
the registry and RunPod actions with generated command packets; the agent prepares each
packet, verifies returned artifacts, applies gates, and reports the next safe packet.

## Commit Boundaries

Keep implementation reviewable in this order:

1. workspace/configuration;
2. Docker image/manual handoff;
3. packaging/holdout identity;
4. ledger/checkpoint recovery;
5. recipes;
6. CUDA training/preflight;
7. gates/final sealing;
8. CLIs;
9. end-to-end tests/operator docs;
10. calibration/abstention fit and its release surface;
11. release bundle sealing and offline reload smoke.

Each commit must pass its focused tests and static checks. No live RunPod call, data
transfer, full training, holdout opening, model promotion, push, or destructive cleanup is
part of these implementation commits.

## Completion Gates

Implementation is complete only when all of the following pass from a clean worktree:

- RunPod unit, contract, and synthetic integration tests;
- reproducible linux/amd64 Docker build, context allowlist, image-history scan, SBOM, and
  container smoke check with immutable base/final digests;
- full backend pytest suite;
- Ruff lint and formatting for both projects;
- strict mypy for RunPod source/scripts and backend source/scripts;
- both lock-file checks and `git diff --check`;
- tracked-change scans for secrets, private absolute paths, transcript fields, raw
  item-level output, credentials, and SSH material;
- a no-data local dry run and synthetic remote dry run;
- generated command-packet shell syntax checks and a fake-SSH/rsync rehearsal proving the
  user can perform each stage without editing generated scripts;
- independent confirmation that the backend does not import `voxdelta_runpod` and the
  final holdout cannot be opened before freeze or more than once;
- a release bundle round trip: build, re-verify, tamper rejection, and an offline
  reload smoke inference that loads only from the bundle's own `checkpoint/` and
  `base-model/` directories;
- calibration fitted on the development split only, refused on the sealed `test`
  split and on any aggregate that opened holdout items, carried as scalars into
  `results/full/report.json`, published as a separate release-bound artifact that
  leaves the frozen release byte-identical, and applied by the backend's calibrated
  provider after that artifact verifies against the exact release.

Passing these gates produces the first Docker/manual command packet. It does not authorize
billable infrastructure, registry login/push, actual-data transfer, holdout use, or
deletion; the user performs each action deliberately.
