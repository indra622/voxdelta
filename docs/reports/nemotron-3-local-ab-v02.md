# VoxDelta 화자분리 A/B v02: Nemotron 3 local, KCSC 2화자 확장 평가

> **2026-09-26 후속 결정:** 공개·로컬 평가 근거를 검토한 사용자 결정으로 `nemotron-3-local`을
> PoC 기본 화자 분리 provider로 승격했다. `pyannoteai-precision`은 명시적 원격 opt-in으로 남는다.
> 아래 본문은 작성 당시(opt-in 유지) 결론을 그대로 보존한다.

> **결론: opt-in 유지.** 기본 provider(`pyannote-community`, `pyannoteai-precision`)는 바꾸지
> 않는다. 로컬 KCSC 7개 화자쌍 전부에서 Nemotron이 Community-1보다 낫다. 하지만 Precision-2와
> 비교할 수 있는 독립 표본은 KCSC 3개, 사용자 Gold 2개(한 세션)뿐이다. 실제 통화 도메인
> 표본은 새로 늘지 않았다.

v01(`user-gold-nemotron-3-local-ab-v01.md`, 2026-09-24)의 후속 평가다. v01 권고의 1단계인
"KCSC 파생 2화자 평가셋 채점"을 수행했다. 2단계인 "동의·검수된 2화자 사용자 Gold 3개 이상 추가
수집"은 아직 수행하지 못했다. v01 artifact와 report는 수정하지 않았다.

## 범위와 방법

- **입력:** 로컬 KCSC 파생 평가셋(`data/derived/kcsc`) 11개 대화 전부. 모두 2화자이고
  16 kHz mono PCM16이며 약 15분 길이다(1개만 9.7분). 실행 전에 audio/reference SHA-256을
  derivation manifest(`0144f428…deb7d2`, source revision `364fb908…ddaf3`)와 대조해
  재검증했다. 제외된 대화는 0개다.
- **독립성:** 11개 대화에 등장하는 화자쌍은 **7개**다. G0101+G0102가 3세션,
  G0143+G0144와 G0359+G0360이 각각 2세션이다. 같은 화자쌍의 세션은 상관된 표본이므로 집계
  단위는 화자쌍이다. 쌍 안에서는 오류 초를 합산한 뒤 나누고 JER은 평균을 낸다. 쌍 사이에서는
  가중하지 않은 요약과 쌍별 차이, 정확 부호검정만 보고한다. 11개 세션 전체를 하나로 합산한
  수치는 보고하지 않는다.
- **시스템:**
  - Nemotron 3 local: **이번 실행**. v01과 같은 NeMo-Speech.cpp Metal runtime, GGUF, 모델
    카드 Offline 30.4 s geometry를 썼다(chunk 340, right context 40, left context 0,
    FIFO 40, speaker cache 264, update period 300). 파일당 3회 실행했고 11개 모두 3회 출력이
    같았다.
  - Community-1 local: **이번 실행**. v01과 같은 checkpoint tree
    `afc77610…73d9bb`를 CPU에서 in-memory waveform으로 파일당 1회 돌렸다.
  - Community-1 historical: 2026-09-01 artifact `kcsc-diarization-community-1.json`
    (sha256 `47227b2f…98c6e8`). 3개 대화만 있다.
  - Precision-2 historical: 2026-09-01 artifact `kcsc-diarization-precision-2.json`
    (sha256 `6a15e389…7a008f0`). 3개 대화만 있다. **이번에 호출하지 않았다.**
  - 두 historical artifact는 manifest digest, source revision, 대화별 reference turn 수와
    길이가 모두 일치할 때만 인용했다. manifest digest가 모든 audio/reference digest를
    고정한다.
- **historical timeline 확인:** historical artifact에는 채점한 timeline이 기록돼 있지 않다.
  이번 Community-1 실행의 **overlap-aware evidence** timeline 수치는 historical Community-1
  수치와 3개 대화 모두에서 소수점 6자리까지 같았다(strict/collar DER·JER). exclusive
  timeline 수치는 달랐다. 따라서 당시 benchmark의 `diarize()`는 evidence timeline을
  반환했다. Precision-2 historical도 같은 benchmark 모듈과 같은 `diarize()` 경로로
  만들어졌으므로 evidence timeline으로 **추정**한다. 이는 재현으로 검증한 사실이 아니라
  추론이다.
- **채점:** v01/KCSC 프로토콜과 같다. `pyannote.metrics` DER/JER, strict와 250 ms collar,
  녹음 전체 UEM, overlap 포함 채점이다. 현재 실행한 시스템은 exclusive와 evidence timeline을
  둘 다 채점했다.
- **네트워크:** 원격 호출은 0회다. Nemotron subprocess 34회는 전부
  `sandbox-exec (deny network*)` 안에서 실행했다. Python egress guard 차단 시도도 0회다.
  오디오는 기기를 떠나지 않았다.

## 결과 1: Nemotron vs Community-1, 둘 다 이번 실행 (7개 화자쌍, 11세션)

화자쌍별 DER이다(쌍 안에서 합산, strict / 250 ms collar).

| 화자쌍 (세션 수) | Nemotron exclusive | Community-1 exclusive | Nemotron evidence | Community-1 evidence |
| --- | --- | --- | --- | --- |
| G0101+G0102 (3) | **14.12%** / **6.64%** | 17.25% / 9.19% | **10.99%** / **3.17%** | 15.34% / 7.08% |
| G0109+G0110 (1) | **27.66%** / **20.16%** | 31.78% / 23.47% | **15.13%** / **5.18%** | 23.99% / 13.84% |
| G0131+G0132 (1) | **25.11%** / **15.83%** | 27.85% / 18.28% | **19.32%** / **8.30%** | 22.93% / 12.43% |
| G0135+G0136 (1) | **18.09%** / **10.64%** | 22.45% / 14.13% | **13.43%** / **5.54%** | 18.34% / 10.03% |
| G0143+G0144 (2) | **22.85%** / **11.60%** | 30.50% / 18.66% | **17.64%** / **5.53%** | 27.18% / 14.92% |
| G0359+G0360 (2) | **19.84%** / **10.88%** | 21.89% / 12.53% | **13.67%** / **4.16%** | 17.81% / 8.07% |
| G5999+G6000 (1) | **19.56%** / **12.26%** | 26.29% / 18.68% | **13.28%** / **5.11%** | 22.60% / 14.53% |

화자쌍 7개에 대한 요약이다(가중 없음).

| Timeline | 지표 | Nemotron 평균 (범위) | Community-1 평균 (범위) | 차이 평균 (Nemotron − C1) | Nemotron 우세 쌍 | 부호검정 p |
| --- | --- | --- | --- | --- | --- | --- |
| exclusive | strict DER | 21.03% (14.12–27.66) | 25.43% (17.25–31.78) | −4.40 pp | 7/7 | 0.016 |
| exclusive | collar DER | 12.57% (6.64–20.16) | 16.42% (9.19–23.47) | −3.85 pp | 7/7 | 0.016 |
| exclusive | strict JER | 20.84% | 27.91% | −7.06 pp | 7/7 | 0.016 |
| evidence | strict DER | 14.78% (10.99–19.32) | 21.17% (15.34–27.18) | −6.39 pp | 7/7 | 0.016 |
| evidence | collar DER | 5.28% (3.17–8.30) | 11.56% (7.08–14.92) | −6.27 pp | 7/7 | 0.016 |
| evidence | strict JER | 14.03% | 23.16% | −9.14 pp | 7/7 | 0.016 |

Nemotron은 세션 단위로도 11/11에서 Community-1보다 낫다(두 timeline, strict와 collar
모두). 다만 세션은 독립 단위가 아니므로 검정은 쌍 단위(n=7)로만 했다. 7/7 결과에서 부호검정이
줄 수 있는 가장 작은 p는 0.016이다.

오류 구성은 두 시스템이 다르다. Community-1은 miss가 크다(strict, evidence 기준 11–21%).
Nemotron은 miss가 작지만(4–10%) false alarm이 더 크다(5–10% 대 Community-1 0.6–1.9%).
화자 혼동은 Nemotron이 모든 세션에서 더 작다(≤1.1% 대 0.7–6.9%).

## 결과 2: historical 기준선과의 비교 (3개 대화, 3개 독립 화자쌍)

Nemotron은 이번 실행이다. Community-1과 Precision-2는 2026-09-01 historical artifact이며
evidence timeline이다(위 절 참조).

| 대화 | 시스템 | strict DER / JER | collar DER / JER | 화자 수 (ref 2) |
| --- | --- | --- | --- | --- |
| A0051_S0001_0 | Nemotron evidence (현재) | 10.29% / 10.24% | **2.61% / 2.78%** | 2 |
| | Nemotron exclusive (현재) | 12.33% / 12.59% | 4.99% / 5.49% | 2 |
| | Precision-2 (historical) | **8.68% / 8.91%** | 4.24% / 4.43% | 2 |
| | Community-1 (historical = 현재 evidence) | 13.13% / 13.83% | 5.17% / 5.89% | 2 |
| A0055_S0006_0 | Nemotron evidence (현재) | **15.13% / 14.31%** | **5.18% / 5.11%** | 2 |
| | Nemotron exclusive (현재) | 27.66% / 26.84% | 20.16% / 20.10% | 2 |
| | Precision-2 (historical) | 23.30% / 23.40% | 18.80% / 19.02% | 2 |
| | Community-1 (historical = 현재 evidence) | 23.99% / 25.46% | 13.84% / 15.62% | 2 |
| A6000_S0005_0 | Nemotron evidence (현재) | **13.28% / 13.28%** | **5.11% / 5.22%** | 2 |
| | Nemotron exclusive (현재) | 19.56% / 20.67% | 12.26% / 13.38% | 2 |
| | Precision-2 (historical) | 16.07% / 17.24% | 10.54% / 11.55% | 2 |
| | Community-1 (historical = 현재 evidence) | 22.60% / 25.97% | 14.53% / 17.80% | 2 |

- **같은 timeline(evidence)에서 Nemotron과 Precision-2를 비교하면:** Nemotron의 strict DER이
  2/3에서 낮다. A0051_S0001_0에서는 Precision-2가 1.6 pp 낫다. collar DER은 3/3에서 Nemotron이
  낮다(차이 평균 −6.9 pp, 부호검정 p=0.25). n=3이라 통계적 결론을 내릴 수 없다.
- **artifact의 `historical_subset_nemotron_exclusive_vs_historical` 집계는 timeline이 맞지 않는
  비교다.** Nemotron exclusive를 기준선 evidence와 비교한 것이며 투명성을 위해 남겼을 뿐이다.
  Precision-2의 exclusive timeline 수치는 KCSC에 존재하지 않는다. 제품이 transcript 정렬에
  쓰는 exclusive timeline에서 Nemotron과 Precision-2를 비교한 자료는 여전히 v01 사용자 Gold
  2개뿐이다(01: Nemotron 우세, 02A: Precision-2 우세).
- A0055_S0006_0에서 Nemotron exclusive DER(27.7%)과 evidence DER(15.1%)의 차이가 크다.
  KCSC reference에는 두 화자 트랙이 겹치는 구간이 있다. exclusive 변환은 겹침 구간을 먼저 말을
  시작한 화자에게 모두 주고 나머지 화자의 발화를 miss로 만든다(miss 22.2% 대 7.6%). 이 변환은
  모델 출력이 아니라 provider의 결정적 규칙이다.

## 화자 수와 속도

- **화자 수 오차:** 11개 대화, 모든 시스템과 timeline에서 0이다(historical 기준선 포함).
- **Nemotron wall time:** 3회 실행의 중앙값은 파일당 3.1–4.7 s이며 RTFx는 187–218×다(Apple
  M4 Metal). 프로세스 시작과 모델 로드가 포함된 값이다.
- **Community-1 wall time:** 259–419 s, RTFx 2.1–2.3×다(CPU, 모델 로드 제외, 1회 실행).
  Community-1은 historical artifact의 CPU 시간(261–409 s)과도 일치한다. 두 시스템은 장치가
  다르므로(GPU/Metal 대 CPU) 속도 차이는 제품 구성의 차이이며 같은 하드웨어에서 비교한 결과는
  아니다.
- **Precision-2 historical elapsed(22–24 s):** upload와 원격 대기가 포함돼 비교할 수 없다.

## 사용자 Gold

**새로 추가할 수 있는 사용자 Gold가 없다.** 로컬에 있는 동의·검수 Gold는 v01의 네 case
(`USER_EVAL_20260911_01`, `_02`, `_02A`, `_02B`)가 전부다. `02A`와 `02B`는 `02` 녹음을
나눈 것이므로 독립적인 2화자 녹음은 여전히 `01`과 `02` 두 개이고, 둘 다 같은 사용자의 같은
세션에서 나왔다. 상관된 녹음을 다시 쪼개 새 표본처럼 쓰지 않았다. `annotations/`에 있는
KCSC와 AI Hub 022 항목은 Gemini Silver이며 사람이 검수한 Gold가 아니어서 쓰지 않았다.

## 해석 제한

- **표본 독립성:** KCSC의 독립 단위는 7개 화자쌍이다. Precision-2와 비교할 수 있는 단위는
  3개뿐이다. 실제 사용자 녹음은 독립 녹음 2개, 세션 1개다.
- **한국어 도메인:** KCSC는 한국어 대화이지만 모집된 화자가 나눈 자유 대화다. 원본은
  화자별로 분리된 wideband 트랙이고 이를 합쳐 16 kHz mono로 만들었다. 전화망 협대역, 코덱
  손상, 상담원-고객 구조, 대기음과 IVR은 없다. 따라서 이 결과는 **실제 통화 도메인 성능을
  대표하지 않는다**. 모델 카드의 학습 언어 목록에 한국어는 없다. Nemotron이 KCSC에서 보인
  우위가 한국어 전반으로 일반화된다고 볼 근거가 없다.
- **reference 특성:** KCSC reference는 화자별 트랙 전사에서 왔다. 짧은 맞장구와 겹침이
  사용자 Gold보다 많다. 그래서 evidence timeline이 유리하고, false alarm이 큰 Nemotron의
  특성이 이 채점에서는 덜 불리하게 나타날 수 있다.
- **Precision-2 historical:** 2026-09-01 실행 결과이고 timeline은 추론이다. 당시 업로드는
  corpus 권리가 확인되지 않은 상태에서 운영자 한정 override로 진행됐다(artifact의
  `processing` 블록). 이번 작업은 그 수치를 인용만 했고 어떤 오디오도 전송하지 않았다.
  KCSC 추가 대화를 Precision-2로 채점하려면 corpus 권리 확인이 먼저 필요하다.
- **런타임 provenance:** executable SHA-256은 v01과 같다(`a72c0d56…a848`). 다만
  NeMo-Speech.cpp checkout의 `ggml` submodule(`c03b4e2b`)에는 커밋되지 않은 로컬 변경이 있다.
  executable hash가 그대로이므로 v01과 같은 바이너리다. 그러나 소스 commit만으로 이 바이너리를
  재빌드해 재현할 수 있다는 보장은 없다.

## 결론: opt-in 유지

**opt-in 유지**를 선택한다. 기본 provider 전환은 권고하지 않는다.

"데이터 부족"을 고르지 않은 이유: 이번 평가로 **opt-in 유지 판단**에는 충분한 근거가 생겼다.
독립 화자쌍 7/7에서 Community-1보다 낫고, 출력이 결정적이며, 화자 수 오차가 없고, 완전히
로컬에서 약 200× 실시간으로 동작한다. 로컬 대안으로 계속 제공할 이유는 분명하다.

"전환 권고"를 고르지 않은 이유:

1. **Precision-2 대비 증거가 약하다.** 독립 비교 단위가 KCSC 3개와 사용자 녹음 2개뿐이다. 제품
   정렬 timeline(exclusive) 기준으로는 사용자 Gold 2개에서 1승 1패다. KCSC에는 Precision-2의
   exclusive 수치가 없다.
2. **실제 통화 도메인 표본이 늘지 않았다.** KCSC는 wideband 자유 대화다. v01에서 정한 전환
   판단 조건(동의·검수된 2화자 사용자 Gold 최소 3개 추가)을 아직 충족하지 못했다.
3. **Nemotron의 exclusive timeline은 휴리스틱으로 만든다.** 겹침이 많은 대화에서 miss가 크게
   늘어난다(A0055_S0006_0: evidence 15.1% → exclusive 27.7%).

한 가지는 KCSC 증거가 분명히 보여 준다. **로컬 기본값인 Community-1과 비교하면** Nemotron
쪽이 우세하다. 로컬 기본값 교체를 다시 논의하려면 다음 두 조건을 채워야 한다. (1) 독립
화자·세션의 동의·검수된 실제 통화 2화자 Gold 3개 이상에서, exclusive timeline 기준으로
Community-1 대비 우위가 유지될 것. (2) exclusive 변환 규칙이 겹침 구간 miss를 얼마나 늘리는지
그 영향을 따로 평가할 것.

## 재현

- Artifact: `data/benchmarks/nemotron-3-local-ab-v02.json`
  (sha256 `fc36b55e430fbd90094d5b1ab04ba927062562eff2abd16f653e89a117f2ae23`, transcript·경로
  없음, 2026-09-26T03:39:19Z)
- 런타임: NeMo-Speech.cpp `97a15afa5caa9bce5baaa86c1184103877af4101`, preset `metal-diar`,
  executable sha256 `a72c0d565f222df8e712818f1b8243b3827f6745f71a421464795449fcdab848`
- 모델: `nvidia/Nemotron-3-Diarization@f667ed73aee57d40cc39428eb768b4fd87a0a29e`,
  `Nemotron-3-Diarization.q8_0.gguf` sha256
  `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`
- Community-1 checkpoint tree sha256
  `afc776102b3aadbf4c7359b6e59f665a8e950bd10c8980234c74a78d0a73d9bb`
- 스크립트: `backend/scripts/run_nemotron_kcsc_ab.py`. 기존 report를 덮어쓰지 않으며, 새 실행에는
  새 버전 파일명을 쓴다.

`backend/`에서 실행한다(`DATA`는 로컬 data 트리):

```bash
HF_HUB_OFFLINE=1 uv run python scripts/run_nemotron_kcsc_ab.py \
  --data-root "$DATA" \
  --executable ~/opt/NeMo-Speech.cpp/build/metal-diar/bin/nemo-speech \
  --model ~/opt/nemo-speech-models/nvidia/Nemotron-3-Diarization/f667ed73aee57d40cc39428eb768b4fd87a0a29e/Nemotron-3-Diarization.q8_0.gguf \
  --device metal \
  --runtime-source-commit 97a15afa5caa9bce5baaa86c1184103877af4101 \
  --repeats 3 \
  --community-checkpoint "$DATA/models/speaker-diarization-community-1" \
  --community-repeats 1 \
  --output "$DATA/benchmarks/nemotron-3-local-ab-v03.json"
```

Community-1을 CPU로 11개 모두 돌리면 약 75분이 걸린다. Nemotron만 실행하려면
`--community-checkpoint`를 빼면 된다(약 1.5분).
