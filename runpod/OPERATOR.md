# RunPod 수동 운용 명령 계약

이 문서는 사용자가 직접 RunPod와 컨테이너 레지스트리를 조작할 때 사용할 명령의
형식과 전달물 경계를 고정한다. 실제 구현이 완료되면 에이전트가 이 형식으로
`runpod/dist/<run-id>/` 아래에 값이 채워진 단계별 명령 파일을 만든다.

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
- `SHA256SUMS`: 전송 전후 검증값;
- `01-preflight-and-pilot.sh`: remote 검증과 두 pilot 실행;
- gate 통과 후 별도 `02-full-or-resume.sh`: full 학습 또는 안전한 resume;
- `03-download-results.sh`: aggregate 결과 패킷 회수.

원본 CSV/ZIP, transcript, 절대경로, original filename, speaker/call ID, 로컬 audit
note, 기존 item-level output은 보내지 않는다.

### 3. 최종 holdout 패킷

full validation gate와 candidate freeze 검증 후에만 별도로 만든다.

- `final-holdout.tar.zst`: 이전에 노출된 35개를 제외한 3,585개 WAV;
- `final-holdout.sidecar.json`과 `SHA256SUMS`;
- `04-final-once.sh`: 두 frozen provider의 단 한 번 평가;
- `05-download-final.sh`: aggregate final 결과 회수.

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

스크립트는 local checksum 확인, mode `0700` remote root 생성, `rsync --partial`, remote
checksum 재검증, archive member 검사, CUDA/BF16/GPU/disk/base/code preflight, pilot A/B
실행까지만 담당한다. full stage는 포함하지 않는다.

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
