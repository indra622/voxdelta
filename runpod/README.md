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
├── OPERATOR.md                     # Korean manual RunPod command contract
├── pyproject.toml                 # RunPod-only environment and backend path dependency
├── uv.lock                        # independently frozen remote environment
├── config/                        # non-secret, immutable experiment profiles
├── docker/                        # pinned linux/amd64 CUDA image definition and checks
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
- `dist/`: generated Docker and transfer handoff packets;
- `ssh/` and `rsync-partial/`: operator-specific connection and partial-transfer state.

No actual audio, source manifest, credential, SSH key, item-level audit record, checkpoint,
or live result may be committed. Code may import stable interfaces from `backend/`; the
production API may not import `runpod/`.

## Responsibility Split

### User

The user owns every live RunPod and registry interaction. The agent supplies verified
artifacts and exact command packets, but the user executes them:

1. keep the overseas-transfer authorization evidence and confirm its presence at
   preflight without sharing its contents;
2. authenticate to the user's container registry, push the agent-built image with the
   supplied commands, and select its immutable digest in the RunPod UI;
3. create the RunPod Secure Cloud Pod and private volume and approve the cost;
4. set temporary SSH values only in the user's local shell, then run the supplied verify,
   `rsync`, remote preflight, training, resume, and result-download commands;
5. execute the separate one-time final-holdout command packet only after candidate freeze;
6. delete the Pod/volume after the local result-verification command passes.

The user is not expected to edit manifests, compose shell commands, copy files manually,
interpret raw logs, choose a winning pilot, or decide whether a gate passed. The user may
paste non-sensitive aggregate command output back to the agent for diagnosis; credentials,
SSH keys, private registry tokens, and evidence contents must not be posted in Discord.

### Agent

The agent owns all reproducible technical work:

1. implement and test the isolated `runpod/` toolchain;
2. audit local labels/audio and build privacy-minimized packages without exposing private
   fields in chat, Git, or ordinary logs;
3. create the pinned linux/amd64 CUDA Docker image definition, build and smoke-test a
   portable OCI archive, and deliver its immutable digest, build metadata, and
   digest-preserving registry push commands;
4. verify package digests, counts, permissions, frozen configuration, image identity, and
   expected GPU/storage profile;
5. generate sequential, copy-pasteable command packets for registry push, Pod preflight,
   training-data transfer, pilots, full training/resume, result download, final holdout,
   and deletion readiness;
6. review the aggregate outputs the user returns and apply the frozen pilot, validation,
   and final-comparison gates mechanically;
7. stop on every failed gate and provide a revised safe command packet when recovery is
   allowed;
8. verify downloaded aggregate artifacts locally, including digests and provider reload,
   before producing the deletion-ready command packet;
9. keep detailed private local logs and report only concise non-sensitive status.

The agent never authenticates to the user's registry or RunPod account, creates billable
infrastructure, executes a live upload, opens the remote final holdout, or deletes a
Pod/volume. Docker image content and data packages are separate artifacts; neither embeds
credentials, licensed audio, manifests, checkpoints, or live results in an image layer.

## Handoffs

There are five explicit manual handoffs:

1. **Image:** agent delivers a verified image/digest and commands; user pushes it to the
   user's registry and creates the Pod from that digest.
2. **Training:** agent delivers the sanitized training packet and commands; user verifies,
   transfers, preflights, and starts the pilot/full stages as each gate permits.
3. **Candidate freeze:** user downloads aggregate results; agent verifies them and issues
   either a stop decision or the separately sealed final-holdout packet.
4. **Final holdout:** user transfers and executes the one-time final command packet, then
   downloads the aggregate result packet.
5. **Deletion:** agent verifies the downloaded results and marks deletion ready; user
   removes the Pod/volume in RunPod.

Command packets are stage-specific. A retry or resume packet never authorizes a later
stage, and the final holdout never appears in the training packet.

## Local Build Order

Run these from the repository root. Use private absolute paths outside Git for the source
manifest, authorization marker, and generated packets.

```bash
run_id='xlsr-622-v1'
export VOXDELTA_BUILDER='<docker-container-buildx-builder>'
uv run --project runpod python runpod/scripts/preflight_local.py \
  --config "$(pwd)/runpod/config/experiment.toml" \
  --manifest '<absolute-source-manifest>' \
  --exposed-manifest '<absolute-exposed-manifest>' \
  --authorization-marker '<absolute-private-marker>' \
  --repository "$(pwd)"
bash runpod/scripts/build_image.sh "$run_id-image"
uv run --project runpod python runpod/scripts/package_data.py training \
  --config "$(pwd)/runpod/config/experiment.toml" \
  --manifest '<absolute-source-manifest>' \
  --output "$(pwd)/runpod/packages/$run_id-training"
uv run --project runpod python runpod/scripts/assemble_handoff.py \
  --image-root "$(pwd)/runpod/dist/$run_id-image" \
  --training-root "$(pwd)/runpod/packages/$run_id-training" \
  --output "$(pwd)/runpod/dist/$run_id" \
  --run-id "$run_id"
```

The resulting `runpod/dist/<run-id>/` is the only training handoff. Candidate freeze and
final packaging use `freeze_candidate.py` and `package_final_holdout.py`; final retrieval
is accepted only after `verify_retrieved_results.py final` emits `deletion_ready`.
