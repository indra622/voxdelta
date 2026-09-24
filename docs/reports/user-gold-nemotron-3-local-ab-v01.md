# VoxDelta 사용자 Gold 화자분리 A/B: Nemotron 3 local v01

> **파일럿이며 전환 결정이 아니다.** 2화자 headline case는 두 개뿐이고, 둘 다 동일한
> 사용자·동일한 녹음 세션에서 나왔다. 이 결과만으로 기본 provider를 바꾸지 않는다.

## 범위와 방법

- 대상: `nemotron-3-local` provider (NVIDIA `Nemotron-3-Diarization`, NeMo-Speech.cpp Metal,
  Apple M4). 기본 provider와 두 pyannote provider의 동작은 바꾸지 않았다.
- 입력: 로컬에 오디오와 검수 완료 Gold가 모두 있는 사용자 제공 case 네 개. 모두 16 kHz
  mono PCM16이며, manifest SHA-256과 Gold `content_sha256`을 실행 전에 재검증했다.
- 채점: 기존 user-Gold/KCSC 프로토콜을 그대로 재사용했다 (`pyannote.metrics` DER/JER,
  strict와 250 ms collar, 녹음 전체 UEM, overlap 포함 채점). headline은 transcript 정렬에
  쓰이는 **exclusive timeline**이며, 기존 Precision-2 수치도 같은 timeline으로 채점됐다.
- 설정: 모델 카드의 **Offline 30.4 s** geometry (80 ms frame 기준 chunk 340, right context
  40, left context 0, FIFO 40, speaker cache 264, update period 300). segmentation
  threshold는 runtime 기본값. 런타임의 `v3-offline` preset은 다른 geometry(약 21.3 s)라
  쓰지 않았다.
- 원격 호출 0회. Nemotron subprocess는 `sandbox-exec`의 network 전면 차단 profile 안에서,
  Community-1은 프로세스 egress guard 안에서 실행했다 (차단된 시도 0회). Precision-2는
  **새로 호출하지 않았고**, 기존 artifact의 Gold/오디오 digest가 이번 입력과 일치할 때만
  수치를 복사했다.
- 반복성: 파일마다 3회 실행했고, Nemotron 출력은 3회 모두 동일했다.

## 결과: 2화자 headline case

| Case | 시스템 | strict DER / JER | 250 ms collar DER / JER | 화자 수 (Gold 2) | 중앙값 wall / RTFx |
| --- | --- | --- | --- | --- | --- |
| USER_EVAL_20260911_01 (55.4 s) | Nemotron 3 local | **9.01% / 16.51%** | **5.76% / 10.83%** | 2 (오차 0) | 0.60 s / 92.7× |
| | Precision-2 (기존 artifact) | 9.94% / 17.30% | 6.23% / 11.08% | 2 (오차 0) | 비교 불가 |
| | Community-1 local | 12.40% / 21.67% | 9.01% / 16.39% | 2 (오차 0) | 20.0 s / 2.8× |
| USER_EVAL_20260911_02A (34.0 s) | Nemotron 3 local | 19.81% / 26.34% | 14.02% / 19.74% | 2 (오차 0) | 0.60 s / 56.9× |
| | Precision-2 (기존 artifact) | **14.40% / 16.10%** | **9.39% / 10.04%** | 2 (오차 0) | 비교 불가 |
| | Community-1 local | 23.72% / 30.42% | 19.23% / 26.52% | 2 (오차 0) | 10.5 s / 3.3× |

오류 구성 (strict): 01에서 Nemotron은 miss 1.0%·confusion 8.0%로 Precision-2(miss 2.1%·
confusion 7.9%)와 거의 같다. 02A에서 Nemotron의 열세는 대부분 **화자 혼동**이다
(confusion 8.0% 대 Precision-2 1.8%; miss 7.8% 대 8.3%, false alarm 4.1% 대 4.4%).

## 결과: 보조 case (headline 제외)

| Case | 시스템 | strict DER / JER | 250 ms collar DER / JER | 화자 수 | RTFx |
| --- | --- | --- | --- | --- | --- |
| USER_EVAL_20260911_02 (59.9 s, Gold 3화자) | Nemotron 3 local | 14.36% / 19.39% | 9.44% / 14.11% | 3 (오차 0) | 100.2× |
| | Precision-2 | 수치 없음: 기존 실행이 `unsupported_speaker_count`로 채점 전 실패 | | | |
| | Community-1 local | 19.18% / 24.68% | 14.95% / 21.07% | 3 (오차 0) | 2.6× |
| USER_EVAL_20260911_02B (25.9 s, Gold 1화자) | Nemotron 3 local | 3.75% / 3.62% | 1.05% / 1.04% | exclusive 1 / evidence 2 | 43.3× |
| | Precision-2 | 해당 case 실행 기록 없음 | | | |
| | Community-1 local | 10.59% / 10.22% | 8.02% / 7.94% | exclusive 1 / evidence 2 | 3.3× |

`02`는 `02A`+`02B`와 같은 녹음이라 독립 표본이 아니다. `02B`에서는 두 로컬 시스템 모두
overlap-aware evidence timeline에 짧은 가짜 두 번째 화자를 냈고, exclusive timeline에서는
사라졌다. 즉 제품의 2화자 gate(evidence 기준)가 단일 화자 녹음에서도 통과될 수 있다.

overlap-aware evidence timeline으로 채점하면 두 로컬 시스템 모두 DER이 더 높다. 예를 들어
Nemotron 01 strict DER은 15.63%다. Gold turn에는 겹침이 거의 없는데 evidence는 겹침을
가설로 내기 때문이다. 수치는 JSON artifact에 함께 있다.

## 해석 제한

- 2화자 case가 두 개뿐이고 1분 미만이다. 01에서는 Nemotron이 Precision-2보다 약간 낫고,
  02A에서는 확실히 나쁘다. 방향이 엇갈리므로 두 시스템의 우열을 말할 수 없다.
- 한국어 성능을 일반화할 수 없다. 모델 카드의 학습 언어 목록에 한국어는 없다.
- Precision-2 수치는 2026-09-11 실행에서 복사했다. 같은 입력이지만 같은 시점의 재실행은
  아니다. 당시 wall time은 upload와 ASR을 포함한 전체 파이프라인 시간이라 RTFx를 비교하지
  않았다.
- Nemotron wall time(파일당 약 0.6 s)에는 프로세스 시작과 모델 로드가 포함된다. Community-1
  시간은 모델 로드(2.5 s)를 빼고 CPU에서 잰 값이다. 오디오가 짧아 RTFx는 고정 비용의 영향을
  크게 받는다.
- Community-1은 venv의 `torchcodec`이 FFmpeg 라이브러리(`libavutil.56`)를 찾지 못해 파일
  경로 입력으로는 실행되지 않았다. pyannote의 공식 in-memory waveform 입력으로 같은 샘플을
  넣었다. 이 환경 문제는 제품의 `pyannote-community` 경로에도 영향을 준다.
- Nemotron CLI는 frame 확률을 노출하지 않는다. exclusive timeline은 먼저 말을 시작한 화자가
  겹침 구간을 가져가는 결정적 규칙으로 만든 것이며, 모델이 판단한 결과가 아니다.
- Gold는 Gemini Silver를 사람이 고쳐 만들었다. Gemini는 독립 기준선이 아니라 이 표에서 뺐다.

## 권고

- **기본 provider는 바꾸지 않는다.** `nemotron-3-local`은 opt-in으로 유지한다.
- 로컬 후보로서는 유망하다. 네 case 모두에서 로컬 Community-1보다 DER/JER이 낮고, 중앙값
  wall time 기준 약 13–38배 빠르며, 오디오가 기기를 떠나지 않는다.
- 다음 단계: (1) 이미 Community-1과 Precision-2 benchmark가 있는 KCSC 파생 2화자 평가셋에서
  같은 30.4 s 설정으로 Nemotron을 채점한다. (2) 기준선 보고서 계획대로 동의·검수된 2화자
  사용자 Gold를 최소 세 개 더 모은다. 둘 다 끝난 뒤에만 전환을 판단한다.

## 재현

- Artifact: `data/benchmarks/user-gold-nemotron-3-local-ab-v01.json`
  (sha256 `1556254314606d88da37071da43fa0fa7119f73dcd383d4486b15d865cc71362`,
  transcript·경로 없음)
- 런타임: NeMo-Speech.cpp `97a15afa5caa9bce5baaa86c1184103877af4101`, preset `metal-diar`,
  executable sha256 `a72c0d565f222df8e712818f1b8243b3827f6745f71a421464795449fcdab848`
- 모델: `nvidia/Nemotron-3-Diarization@f667ed73aee57d40cc39428eb768b4fd87a0a29e`,
  `Nemotron-3-Diarization.q8_0.gguf` sha256
  `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`
- Community-1 checkpoint tree sha256
  `afc776102b3aadbf4c7359b6e59f665a8e950bd10c8980234c74a78d0a73d9bb`
- 설치 절차: `docs/nemotron-3-local-setup.md`

`backend/`에서 실행:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/run_nemotron_gold_ab.py \
  --executable ~/opt/NeMo-Speech.cpp/build/metal-diar/bin/nemo-speech \
  --model ~/opt/nemo-speech-models/nvidia/Nemotron-3-Diarization/f667ed73aee57d40cc39428eb768b4fd87a0a29e/Nemotron-3-Diarization.q8_0.gguf \
  --device metal \
  --runtime-source-commit 97a15afa5caa9bce5baaa86c1184103877af4101 \
  --repeats 3 \
  --community-checkpoint ../data/models/speaker-diarization-community-1 \
  --output ../data/benchmarks/user-gold-nemotron-3-local-ab-v02.json
```

이 스크립트는 기존 report를 덮어쓰지 않는다. 새 실행에는 새 버전 파일명을 쓴다.
