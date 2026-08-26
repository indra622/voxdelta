# XLS-R release canary and rollback runbook

How to bring the promoted XLS-R seven-emotion release and its separately published
calibration into the local backend behind two opt-in flags, prove locally that they are
safe to serve, run a staged shadow/canary, and roll back in one step.

**This runbook performs no deployment and routes no real traffic.** Everything below is
local. The readiness command measures; it does not release. Steps 4 and 5 describe a
staged sequence for an operator to carry out deliberately, on their own infrastructure,
after step 3 passes. Nothing here contacts a cloud service, and nothing here opens the
sealed final holdout — that split is consumed and cannot be rerun.

## 0. What ships, and what stays separate

The release is immutable and carries **no** calibration of its own. The temperature and
abstain threshold are a separately versioned artifact that names the release
cryptographically. Both are verified independently at startup, and the calibration is
accepted only if it binds to the exact release that was just verified.

| Artifact | Location |
| --- | --- |
| Release bundle | `/Volumes/nvme1/codes/voxdelta/data/models/xls-r-emotion-7class-v1` |
| Calibration artifact | `/Volumes/nvme1/codes/voxdelta/data/models/xls-r-emotion-7class-v1-calibration-v2` |

Both are byte-identical copies of the canonical artifacts under
`runpod/dist/xlsr-622-20260825-v7/`. Neither copy may be edited in place.

## 1. Environment

Both flags default to `false`. Until both are set the emotion stage behaves exactly as it
does today, so setting nothing is already the rolled-back state.

```bash
export VOXDELTA_EMOTION_PROVIDER=wav2vec
export VOXDELTA_XLSR_RELEASE_ENABLED=true
export VOXDELTA_XLSR_RELEASE_PATH=/Volumes/nvme1/codes/voxdelta/data/models/xls-r-emotion-7class-v1
export VOXDELTA_XLSR_CALIBRATION_ENABLED=true
export VOXDELTA_XLSR_CALIBRATION_PATH=/Volumes/nvme1/codes/voxdelta/data/models/xls-r-emotion-7class-v1-calibration-v2
export VOXDELTA_EMOTION_DEVICE=auto
```

Both paths must be absolute. `VOXDELTA_XLSR_RELEASE_ENABLED` additionally requires
`VOXDELTA_EMOTION_PROVIDER=wav2vec` and **forbids** `VOXDELTA_EMOTION_CHECKPOINT_PATH`, so
the promoted bundle can never be ambiguous with a loose checkpoint. Enabling calibration
without the release flag is refused before startup.

## 2. Offline preflight

Startup must load from local files only. The readiness runner pins these itself before it
builds anything; export them too if you are starting the service by hand.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
```

Confirm both artifacts still verify before measuring anything:

```bash
uv run --project runpod python runpod/scripts/verify_release.py \
  --bundle /Volumes/nvme1/codes/voxdelta/data/models/xls-r-emotion-7class-v1
```

Only `release_verified` counts as success. On failure, print nothing further and stop:
the bundle must not be repaired in place.

## 3. Readiness measurement

```bash
uv run --project backend python backend/scripts/check_release_readiness.py \
  --release /Volumes/nvme1/codes/voxdelta/data/models/xls-r-emotion-7class-v1 \
  --calibration /Volumes/nvme1/codes/voxdelta/data/models/xls-r-emotion-7class-v1-calibration-v2 \
  --validation-manifest /Volumes/nvme1/codes/voxdelta/data/manifests/validation.jsonl \
  --output "$(pwd)/backend/runtime/readiness/$(date +%Y%m%dT%H%M%S)" \
  --device auto --repeats 3
```

The runner builds the product's own dependency graph with both flags on, so it exercises
the same verification and provider path as startup. It writes one private,
aggregate-only `READINESS.json` plus a `SHA256SUMS`, into a directory that must not
already exist. Exit `0` means every blocking gate passed, `1` means a gate failed, `2`
means the run was refused; failures print only a stable code, never a path.

Omit `--validation-manifest` to fall back to deterministic synthetic audio. Both modes
produce **runtime evidence only** — even validation audio is not scored against its
reference label, so the report marks `quality_evidence false` and says nothing about
accuracy.

The tool has no argument for a holdout archive, package, or report, and a manifest
containing even one non-validation row is refused before any inference runs.

### Go / no-go gates

All blocking gates come from limits the project already enforces. No latency service
level is invented.

| Gate | Meaning |
| --- | --- |
| `identity_exact` | Release and calibration verified and mutually bound |
| `offline_load` | Hugging Face runtime pinned offline before load |
| `verified_before_model_allocation` | Both artifacts verified before any weights allocated |
| `completion_complete` | 100% of attempted items completed |
| `finite_valid_outputs` | Every distribution finite, in range, summing to one |
| `peak_rss_within_ceiling` | Peak RSS within the existing 18,432 MB ceiling |
| `abstention_maps_to_uncertain` | Every abstained result reports `uncertain` |
| `top_label_preserved` | Temperature scaling changed no top label |

`overall_ok` is exactly the conjunction of those eight. `--advisory-p95-latency-ms` is
recorded and compared but is **advisory by contract** and never affects `overall_ok`;
there is no agreed latency SLO to enforce.

`outcome.abstention_mapping_exercised` says whether this particular canary actually
contained an abstention. If it is false, the mapping gate is backed by regression tests,
not by an observed abstention in that runtime sample.

## 4. Local shadow replay (preparation, not a rollout)

Section 3 measures the candidate alone. This step compares it against the provider it
would replace, still entirely locally.

The rollback primary needs a **verified local encoder bundle**. Publish one once, from
the package cache; the cache is read only and nothing is deleted from it.

```bash
uv run --project backend python backend/scripts/build_encoder_bundle.py \
  --snapshot ~/.cache/modelscope/models/iic--emotion2vec_plus_large/snapshots/v2.0.5 \
  --output /Volumes/nvme1/codes/voxdelta/data/models/emotion2vec-plus-large-encoder-v2.0.5
```

Only `encoder_bundle_published` counts as success. The bundle holds exactly the four
files the loader reads — `model.pt`, `tokens.txt`, `config.yaml`, `configuration.json` —
plus `ENCODER.json` and `SHA256SUMS`. Partial downloads, `.DS_Store`, documentation, and
sample audio are excluded, and weights are dereferenced into a real file so the bundle
cannot change when an unrelated cache is pruned.

Content is checked against a pin held in production code — the exact digest and size of
all four files — so a bundle that merely agrees with its own manifest is refused. The
target directory is reserved with an exclusive `mkdir` and committed by writing
`ENCODER.json` last, so a concurrent publisher's directory is never replaced and a
reserved-but-unfinished bundle is never readable as a finished one. Re-publishing over an
existing bundle refuses with `encoder_bundle_already_exists`.

```bash
uv run --project backend python backend/scripts/run_shadow_replay.py \
  --release /Volumes/nvme1/codes/voxdelta/data/models/xls-r-emotion-7class-v1 \
  --calibration /Volumes/nvme1/codes/voxdelta/data/models/xls-r-emotion-7class-v1-calibration-v2 \
  --primary emotion2vec \
  --primary-checkpoint /Volumes/nvme1/codes/voxdelta/data/models/emotion2vec-plus-large-emotion-balanced \
  --primary-encoder-bundle /Volumes/nvme1/codes/voxdelta/data/models/emotion2vec-plus-large-encoder-v2.0.5 \
  --validation-manifest /Volumes/nvme1/codes/voxdelta/data/manifests/validation.jsonl \
  --output "$(pwd)/backend/runtime/shadow/$(date +%Y%m%dT%H%M%S)" \
  --device cpu --repeats 3
```

The bundle is verified before the provider is constructed and before any weight is
allocated. A raw cache path is refused: only a bundle that passes the verifier is accepted.

`shadow_replay_ok` means every blocking gate passed. Exit `1` is a gate failure and exit
`2` is a refusal; a refusal publishes nothing at all.

**This is a replay evaluator, not a live shadow.** It re-runs a canary you chose; it does
not observe production traffic. The report records `mode local-replay` and
`live_shadow_traffic false` so a reader cannot mistake one for the other.

### What this comparison is, and is not

The primary is the incumbent emotion2vec baseline; the candidate is the verified,
calibrated XLS-R release. Both are built and run with network egress blocked, and the
report records `no_network_egress_attempted`.

`top_label_agreement_rate` is **agreement, not accuracy**. Neither side is scored against
a reference label, so the report sets `quality_evidence false` and a low agreement rate
says the two models disagree — not which one is right. The recorded quality difference
between them lives in the release provenance, not here.

The canary is the development-split validation cohort, the same split the calibration was
fitted on. The sealed final holdout is not reachable from this tool.

### Decision fields for shadow → small canary

When the command does produce a report, read these before widening. Each has a companion
`*_exercised` flag; a rate whose flag is false has no runtime evidence behind it and must
not be read as a pass.

- `primary_invariant_holds` — the adapter returned the **exact object** the single
  primary call produced, that result was valid, and exactly one observation was emitted.
  If this is ever false, stop: the observation layer is not safe.
- `errors` — candidate `timeout` / `provider_error` / `invalid_output` /
  `unexpected_error` counts. Any non-zero value is a candidate defect, never a primary one.
- `top_label_agreement_rate` — how often the two agreed. This is agreement, **not**
  accuracy: neither side is scored against a reference label.
- `candidate_abstention_rate` and `candidate_uncertain_rate` against the fitted
  expectation of 0.0997. `uncertain` is the operational state; `abstained` is the
  calibration decision. A result can be uncertain without having abstained.
- `primary_latency` versus `candidate_latency` medians and p95.
- `peak_rss_mb` — the maximum across **both** sides — against the 18,432 MB ceiling.

Blocking gates for this step are `identity_exact`, `offline_load`,
`no_network_egress_attempted`, `verified_before_model_allocation`,
`primary_invariant_holds`, `primary_completion_complete`,
`candidate_completion_complete`, `candidate_valid_outputs`, and
`peak_rss_within_ceiling`. `overall_ok` is exactly their conjunction, so a run in which
the primary completed but every candidate call failed cannot pass.

`no_network_egress_attempted` is stronger than an environment variable: both sides are
constructed, allocated, and run inside an egress guard that fails every outbound
connection and counts the attempt even if the loader swallows its own error.

The replay offers **no candidate timeout**, and the report records
`candidate_timeout_used false`. A timed-out worker cannot be killed, and the egress guard
above is process-wide and temporary, so an abandoned worker could reach the network the
moment the guard is restored. The candidate therefore always runs inline: a slow candidate
makes the run slow rather than leaving something running behind it.

## 5. Staged shadow, then canary

Proceed only when section 3 reports `overall_ok` true. Advance one stage at a time and hold each stage
long enough to see real traffic variety.

1. **Shadow.** Run the calibrated provider alongside the current one without using its
   output. Compare aggregates only.
2. **Canary, small.** Route a small share of live utterances. Watch the observation
   fields below.
3. **Canary, widened.** Increase the share in steps, re-reading the same fields at each
   step.
4. **Full.** Only after a widened canary has been stable across the traffic mix.

Re-run step 3 whenever the release, the calibration, or the host changes.

### Observation fields

Watch these as aggregates. Never log a transcript, an item identity, or a raw
probability — the report format deliberately cannot carry them.

- abstention rate, against the fitted expectation of **0.0997** (coverage 0.9003)
- share of results in the `uncertain` operational state
- mean and minimum calibrated confidence
- per-item latency median and p95, and cold start after each restart
- peak RSS against the 18,432 MB ceiling
- provider error codes and their rate
- `calibration_id` and `bundle_tree_sha256` in startup logs, to prove which pair is live

A rising abstention rate is the calibration working, not a fault, until it materially
exceeds the fitted expectation. Sustained drift there means the audio has moved away
from the development split, and the fit no longer describes it.

## 6. Rollback

Rollback is disabling the two opt-in flags and restarting. Nothing else changes, no
artifact is touched, and no data migration is involved.

```bash
export VOXDELTA_XLSR_CALIBRATION_ENABLED=false
export VOXDELTA_XLSR_RELEASE_ENABLED=false
# then restart the backend
```

Unsetting both variables entirely has the same effect, since both default to `false`.
Leaving the two `_PATH` variables set is harmless: with the flags off they are ignored,
and there is no silent fallback to the release. The emotion stage returns to exactly its
previous behaviour.

To roll back only the calibration while keeping the verified release, set
`VOXDELTA_XLSR_CALIBRATION_ENABLED=false` alone. Results then carry raw provider
confidence and no `calibration` block, which is the signal that scores must not be read
as calibrated.

If startup fails after a rollback, the cause is elsewhere: with both flags off, neither
artifact is opened at all.
