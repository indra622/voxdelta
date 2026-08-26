# RunPod 수동 운용 명령 계약

이 문서는 사용자가 직접 RunPod와 컨테이너 레지스트리를 조작할 때 사용할 명령의
형식과 전달물 경계를 고정한다. 실제 구현이 완료되면 에이전트가 이 형식으로
`runpod/dist/<run-id>/` 아래에 값이 채워진 단계별 명령 파일을 만든다.

이미지와 학습 데이터는 각각 비공개 임시 sub-packet으로 만든 뒤
`scripts/assemble_handoff.py`가 허용된 파일만 hard-link(불가능하면 private copy)하여
`runpod/dist/<run-id>/`에 원자적으로 조립하고 `SHA256SUMS`를 새로 계산한다.

여기 있는 `<...>` 표시는 설명용이다. 플레이스홀더가 남아 있는 예시를 실행하지
말고, 에이전트가 생성하고 checksum을 검증한 명령 패킷만 실행한다. SSH 주소,
registry token, private key, 허가 증빙은 Discord에 올리지 않는다.

## 에이전트가 전달하는 것

### 1. Docker 이미지 패킷

- `voxdelta-runpod.oci.tar.zst`: 검증된 linux/amd64 CUDA OCI image archive;
- `image-digest.txt`: registry push 전부터 고정된 OCI manifest digest;
- `image-build.json`: base image digest, Git commit, lock digest, platform, 빌드 시각;
- `image-sbom.spdx.json`: 이미지 안의 패키지 명세;
- `push-image.sh`: archive를 사용자 registry에 digest-preserving push하고 remote
  digest를 확인한 뒤 `image-reference.txt`를 만드는 명령;
- `SHA256SUMS`: 위 파일들의 로컬 checksum.

Docker 이미지는 코드와 고정 dependency만 포함한다. WAV, 전송 manifest, credential,
SSH 자료, checkpoint, audit 기록, 결과 파일은 이미지 layer에 넣지 않는다.

### 2. 학습 데이터 패킷

- `train-validation.tar.zst`: 33,045개 정규화 WAV와 sanitized manifest;
- `train-validation.sidecar.json`: 파일 수, split/label count, archive/manifest digest;
- `xls-r-base.tar.zst`와 sidecar: 로컬에 이미 받은 고정 XLS-R 300M base model;
- `SHA256SUMS`: 전송 전후 검증값;
- `01-preflight-and-pilot.sh`: remote 검증과 두 pilot 실행;
- gate 통과 후 별도 `02-full-or-resume.sh`: full 학습 또는 안전한 resume;
- `03-download-results.sh`: deployable checkpoint, 최신 exact-resume checkpoint, epoch별
  metrics history, pilot/full reports, batch profile, CUDA/driver 환경, ledger, 로그 회수와
  `archive/full-retrieval.tar.zst` 생성·즉시 검증.

원본 CSV/ZIP, transcript, 절대경로, original filename, speaker/call ID, 로컬 audit
note, 기존 item-level output은 보내지 않는다.

### 3. 최종 holdout 패킷

full validation gate와 candidate freeze 검증 후에만 별도로 만든다.

- `final-holdout.tar.zst`: 이전에 노출된 35개를 제외한 3,585개 WAV;
- `final-holdout.sidecar.json`과 `SHA256SUMS`;
- `emotion2vec-baseline.tar.zst`와 sidecar: 고정 baseline checkpoint;
- `04-final-once.sh`: 두 frozen provider의 단 한 번 평가;
- `05-download-final.sh`: aggregate final 결과, final-consumed marker, ledger, 로그 회수와
  `archive/final-retrieval.tar.zst` 생성·즉시 검증.

학습 데이터 패킷에는 final holdout 파일이나 이를 여는 capability가 들어가지 않는다.

## 사용자가 실행할 명령 형식

### A. 로컬 비공개 셸 변수

실제 command packet은 아래 값을 사용자의 로컬 셸에서만 받는다.

```bash
export VOXDELTA_RUN_ID='<validated-run-id>'
export VOXDELTA_HANDOFF='<absolute-path-to-runpod/dist/run-id>'
export RUNPOD_SSH_HOST='<temporary-host>'
export RUNPOD_SSH_PORT='<temporary-port>'
export RUNPOD_SSH_USER='root'
export VOXDELTA_REGISTRY_IMAGE='<user-registry>/voxdelta-runpod:<tag>'
```

token이나 private key를 환경변수 명령 파일에 저장하지 않는다. Registry 로그인과 SSH
key 선택은 사용자가 사용하는 credential helper/SSH agent에 맡긴다.

### B. 전달물 로컬 검증

```bash
cd "$VOXDELTA_HANDOFF"
shasum -a 256 -c SHA256SUMS
```

한 파일이라도 실패하면 push, `rsync`, remote 실행을 중단하고 에이전트에게 실패한
파일명과 checksum 결과만 전달한다.

### C. Docker registry push와 digest 확인

```bash
cd "$VOXDELTA_HANDOFF"
bash ./push-image.sh "$VOXDELTA_REGISTRY_IMAGE"
```

출력된 remote digest가 `image-digest.txt`의 기대값과 일치하고 스크립트가
`image-reference.txt`를 만든 뒤에만 그 immutable reference를 RunPod custom image로
선택한다. 변경 가능한 tag만으로 Pod를 만들지 않는다.

### D. 학습 패킷 전송

실제 생성되는 스크립트는 다음 동작을 수행한다.

```bash
cd "$VOXDELTA_HANDOFF"
bash ./01-preflight-and-pilot.sh \
  "$RUNPOD_SSH_USER" "$RUNPOD_SSH_HOST" "$RUNPOD_SSH_PORT"
```

스크립트는 local checksum 확인, mode `0700` remote root 생성, 대상 파일시스템이
`0700`/`0600`을 실제로 유지하는지 확인하는 capability probe, `rsync --partial`, remote
checksum 재검증, archive member 검사, CUDA/BF16/GPU/disk/base/code preflight, pilot A/B
실행까지만 담당한다. full stage는 포함하지 않는다.

probe가 실패하면 licensed audio와 base model을 전송하기 전에 중단된다. 일부 RunPod
region의 network volume은 요청한 mode를 무시하므로, 그런 Pod에는 데이터를 올리지 않는다.

pilot aggregate 결과를 내려받아 에이전트가 gate를 판정한 뒤, 통과한 경우에만:

```bash
cd "$VOXDELTA_HANDOFF"
bash ./02-full-or-resume.sh \
  "$RUNPOD_SSH_USER" "$RUNPOD_SSH_HOST" "$RUNPOD_SSH_PORT"
bash ./03-download-results.sh \
  "$RUNPOD_SSH_USER" "$RUNPOD_SSH_HOST" "$RUNPOD_SSH_PORT"
```

### E. 최종 holdout 전송과 결과 회수

candidate freeze 뒤 에이전트가 새로 만든 final 전용 디렉터리에서만 실행한다.

```bash
cd "$VOXDELTA_HANDOFF/final"
shasum -a 256 -c SHA256SUMS
bash ./04-final-once.sh \
  "$RUNPOD_SSH_USER" "$RUNPOD_SSH_HOST" "$RUNPOD_SSH_PORT"
bash ./05-download-final.sh \
  "$RUNPOD_SSH_USER" "$RUNPOD_SSH_HOST" "$RUNPOD_SSH_PORT"
```

실패했다고 임의로 `04-final-once.sh`를 재실행하지 않는다. Ledger 결과를 에이전트가
확인한 뒤 같은 실행의 허용된 resume인지, 최종 평가가 이미 소비됐는지 판정한다.

## 삭제 조건

사용자는 다음 세 조건을 에이전트가 로컬에서 확인하기 전까지 Pod/volume을 삭제하지
않는다.

1. 학습·최종 aggregate result packet이 모두 내려와 있고 checksum이 일치한다;
2. ledger, frozen config, checkpoint/report digest와 provider reload 검증이 통과한다;
3. 민감 데이터가 결과 패킷에 없고 로컬 보관 위치가 확인됐다.

확인 후 에이전트는 `deletion-ready`만 알린다. 실제 Pod/volume 삭제는 사용자가 RunPod
UI에서 수행한다.

삭제 직전에는 두 retrieval archive의 `zstd --test`와 `.sha256` 검증을 다시 수행하고,
필요하면 별도 로컬/백업 저장소로 한 번 더 복사한다. 그 뒤 Pod를 중지하고 encrypted
network volume을 삭제한 다음 RunPod 콘솔에서 Pod와 volume이 모두 사라져 추가 과금이
없는지 확인한다. Registry image와 로컬 결과 archive는 학습 재개 필요성에 따라 별도로
보관하거나 정리한다.

## 릴리스 봉인

최종 비교가 XLS-R 승격으로 끝난 뒤에만 수행한다. 전부 로컬 작업이며 RunPod 접속, 추가
과금, holdout 재개봉이 없다. Pod/volume 삭제 전후 어느 쪽에서도 실행할 수 있다.

```bash
uv run --project runpod python runpod/scripts/build_release.py ...
uv run --project runpod python runpod/scripts/verify_release.py \
  --bundle "$(pwd)/runpod/dist/$VOXDELTA_RUN_ID/release/$RELEASE_ID"
uv run --project runpod python runpod/scripts/smoke_release_inference.py \
  --bundle "$(pwd)/runpod/dist/$VOXDELTA_RUN_ID/release/$RELEASE_ID"
```

정확한 `build_release.py` 인자는 `README.md`의 릴리스 블록을 따른다. 세 명령은 각각
`release_built`, `release_verified`, `release_smoke_ok`만 성공 신호로 인정한다. 스모크는
번들 안의 `checkpoint/`와 `base-model/`만 읽어 오프라인으로 재적재하며, 합성 클립 한 개의
집계만 출력한다. 실패 시 코드만 출력되므로 번들을 수정하지 말고 에이전트에게 전달한다.

calibration은 릴리스 번들에 덧쓰지 않고 별도 불변 artifact로 게시한다. 로컬의
train-validation 패키지에서 validation 3,569건만 추출하고, item id·reference label·audio
SHA-256이 packaged manifest와 정확히 일치하는지 확인한 뒤 `build_calibration.py`를 실행한다.
final holdout archive/report는 인자로 전달하지 않는다. 기본 제품 기준은 target coverage
`0.9`이며, 출력의 temperature·abstain threshold·coverage·answered accuracy·ECE 전후 값을
기록한다. backend에서는 release flag와 calibration flag/path를 함께 켜며, artifact가 exact
release/checkpoint에 결속되지 않으면 startup이 fail-closed 해야 한다.

Claude Code가 실제 실행을 맡아도 된다. 단, RunPod MCP 인증, registry credential helper,
SSH agent/키, `rsync`, `zstd`는 각각 실행 환경에 준비되어 있어야 한다. Pod 생성·시작과
volume 생성은 비용 발생 전 승인을 받고, final 1회 실행과 Pod/volume 삭제는 별도 명시
승인을 받는다. Claude Code는 생성된 command packet만 실행하고 checkpoint/result 검증
전에는 volume을 삭제하지 않는다.
