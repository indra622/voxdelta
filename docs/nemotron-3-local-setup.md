# Nemotron 3 local diarization: runtime and model setup

This is the one-time, separately performed setup for the `nemotron-3-local` diarization
provider, the PoC default since 2026-09-26. Until it is done, backend startup with the default
configuration is refused with `provider_configuration_invalid` (`local_runtime_missing` /
`local_model_missing`); it never falls back to another provider. Nothing here runs at inference time, and no binaries or model weights
are stored in this repository. Recorded as performed on 2026-09-24 on an Apple M4
(macOS 26.6.2).

## Why a source build

The newest release, NeMo-Speech.cpp `v0.1.0` (2026-08-19), ships a checksummed
`macos-aarch64-metal` archive, but Nemotron 3 Diarization support (NVIDIA/NeMo-Speech.cpp PR
#50) was merged on 2026-09-24, after that release. The runtime was therefore built from source
at a pinned commit. The upstream `curl … | sh` installer was **not** used.

## 1. Build the runtime (Metal, diarization only)

```bash
brew install cmake ninja sentencepiece abseil
git clone https://github.com/NVIDIA/NeMo-Speech.cpp.git ~/opt/NeMo-Speech.cpp
cd ~/opt/NeMo-Speech.cpp
git checkout 97a15afa5caa9bce5baaa86c1184103877af4101   # main after PR #50 and #52
git submodule update --init ggml                         # ggml @ c03b4e2bcece
scripts/configure.sh metal-diar
cmake --build --preset metal-diar
build/metal-diar/bin/nemo-speech --version               # nemo-speech 0.1.0
build/metal-diar/bin/nemo-speech --json doctor           # backend_metal: true, device "Apple M4"
```

`--version` reports `0.1.0` because the version string was not bumped after the release; the
source commit above is the precise runtime identity. The A/B artifact records it, together
with the built executable's SHA-256.

## 2. Pull the model once, with checksum verification

```bash
NEMO_SPEECH_MODEL_DIR=~/opt/nemo-speech-models \
  ~/opt/NeMo-Speech.cpp/build/metal-diar/bin/nemo-speech pull nemotron-3-diarization
```

The runtime's built-in index pins `nvidia/Nemotron-3-Diarization` at revision
`f667ed73aee57d40cc39428eb768b4fd87a0a29e` and accepts the download only after the pinned size
and SHA-256 match. Resulting file:

```
~/opt/nemo-speech-models/nvidia/Nemotron-3-Diarization/f667ed73aee57d40cc39428eb768b4fd87a0a29e/Nemotron-3-Diarization.q8_0.gguf
sha256 08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1  (102.1 MiB)
```

That SHA-256 was independently checked against the Hugging Face LFS `X-Linked-ETag` for the
same file at the same revision. The model is licensed under the NVIDIA Open Model License
(OpenMDW 1.1).

## 3. Configure VoxDelta

`nemotron-3-local` is the default provider and its path settings default to the two locations
above under `$HOME`, so after steps 1 and 2 nothing needs to be set. Only a different install
location, or a pinned device, needs configuration:

```bash
export VOXDELTA_NEMOTRON_EXECUTABLE_PATH=$HOME/opt/NeMo-Speech.cpp/build/metal-diar/bin/nemo-speech
export VOXDELTA_NEMOTRON_MODEL_PATH=$HOME/opt/nemo-speech-models/nvidia/Nemotron-3-Diarization/f667ed73aee57d40cc39428eb768b4fd87a0a29e/Nemotron-3-Diarization.q8_0.gguf
export VOXDELTA_NEMOTRON_DEVICE=metal
```

## What the provider runs

```
nemo-speech diarize <normalized 16 kHz mono WAV> --model <local .gguf> --device <device> --format json
```

- The environment is replaced, not inherited: `PATH=/usr/bin:/bin`, `LC_ALL=C`, and the six
  `NEMO_SPEECH_DIAR_*` geometry keys. No proxy variable, token, or ambient `NEMO_SPEECH_*`
  override reaches the process.
- Geometry is the model card's **Offline, 30.4 s latency** row in 80 ms frames: chunk 340,
  right context 40, left context 0, FIFO 40, speaker cache 264, update period 300. The
  runtime's own `v3-offline` preset is a different geometry (chunk 264, context 1/1, FIFO 0,
  update 188, about 21.3 s), so it is not used. Segmentation thresholds are the runtime
  defaults.
- An existing local GGUF path is the runtime's documented no-network path. In the A/B run the
  subprocess was additionally confined with
  `sandbox-exec -p '(version 1)(allow default)(deny network*)'`, and produced byte-identical
  output to an unconfined run.
- Only the JSON document is parsed; its `file` field and all stderr are discarded.
