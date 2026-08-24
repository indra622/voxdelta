# VoxDelta RunPod Workspace

This directory is the sole operational boundary for the RunPod XLS-R full-fine-tuning
experiment. RunPod-specific configuration, commands, orchestration code, tests, and
operator documentation belong here. The existing `backend/` remains the reusable
VoxDelta library and production service; it must not accumulate Pod addresses,
credentials, transfer manifests, checkpoints, or live-run state.

The approved scientific design remains available as historical context in
`docs/superpowers/specs/2026-08-22-runpod-xls-r-full-finetuning-design.md`. The executable
work order and ownership boundary in `IMPLEMENTATION_PLAN.md` are canonical for the
implementation phase. The old Superpowers workflow is not a prerequisite for this work.

## Boundary

Tracked RunPod work will use this layout as it is implemented:

```text
runpod/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml                 # RunPod-only environment and backend path dependency
├── uv.lock                        # independently frozen remote environment
├── config/                        # non-secret, immutable experiment profiles
├── scripts/                       # local packaging/transfer and remote runner entry points
├── src/voxdelta_runpod/           # package, ledger, training, gates, and orchestration
└── tests/                         # synthetic/unit/integration tests; no live account required
```

Generated or sensitive material is always ignored beneath this directory:

- `runtime/`: local operator state and authorization-presence marker;
- `packages/`: privacy-minimized transfer archives and sidecars;
- `data/`: extracted remote WAV data and sanitized manifests;
- `models/` and `checkpoints/`: pinned base and recovery checkpoints;
- `results/`: aggregate reports and retrieved artifacts;
- `logs/`: private redacted local/remote logs;
- `ssh/` and `rsync-partial/`: operator-specific connection and partial-transfer state.

No actual audio, source manifest, credential, SSH key, item-level audit record, checkpoint,
or live result may be committed. Code may import stable interfaces from `backend/`; the
production API may not import `runpod/`.

## Responsibility Split

### User

The user performs only actions that require account ownership, private evidence, billing,
or an irreversible decision:

1. keep the overseas-transfer authorization evidence and confirm its presence at
   preflight without sharing its contents;
2. create the RunPod Secure Cloud Pod and private volume in the RunPod UI, approve the
   cost, and provide the temporary SSH endpoint through a local private channel;
3. explicitly authorize the first actual-data upload after reviewing the package summary;
4. explicitly authorize the one-time final-holdout unseal/upload after the validation
   gate and candidate freeze pass;
5. confirm destructive Pod/volume deletion only after local result verification.

The user is not expected to edit manifests, compose shell commands, copy files manually,
interpret raw logs, choose a winning pilot, or decide whether a gate passed.

### Agent

The agent owns all reproducible technical work:

1. implement and test the isolated `runpod/` toolchain;
2. audit local labels/audio and build privacy-minimized packages without exposing private
   fields in chat, Git, or ordinary logs;
3. verify digests, counts, permissions, frozen configuration, image identity, GPU profile,
   and available storage;
4. generate exact Pod creation requirements and SSH/rsync commands;
5. after the user's upload authorization, transfer the approved package, launch and
   monitor preflight/pilots/full training, and resume only from valid checkpoints;
6. apply the frozen pilot, validation, and final-comparison gates mechanically;
7. stop on every failed gate and report a concise diagnosis and next safe decision;
8. retrieve aggregate artifacts, independently verify their digests and provider reload,
   and prepare the deletion confirmation for the user;
9. keep detailed private logs locally and post only concise, non-sensitive status updates.

The agent never creates billable infrastructure, uploads actual data, opens the final
holdout, changes a frozen experiment rule, or deletes a Pod/volume without the explicit
user handoff listed above.

## Handoffs

There are five explicit handoffs; everything between them is agent-owned:

1. **Pod ready:** user supplies the temporary SSH endpoint after creating the approved Pod.
2. **Training upload:** agent presents package counts/digests; user authorizes actual-data
   transfer once.
3. **Candidate freeze:** agent reports the full validation gate and frozen identities.
4. **Final holdout:** user authorizes the single holdout package build and transfer.
5. **Deletion:** agent confirms local retrieval and verification; user authorizes deletion.

An authorization applies only to its named handoff. Network failure, retry, or resume
does not broaden it to a later handoff.
