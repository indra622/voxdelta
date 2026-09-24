# VoxDelta — Gemini E2E와 캐스케이드 비교 연구 설계 v01

## 연구 질문

한국어 상담 음성에서 **Gemini E2E**가 `Precision-2 → Qwen3-ASR 1.7B → calibrated XLS-R → ACP Expert` 캐스케이드보다 품질, 비용, 운영 적합성에서 우월한가? 아니면 근거 추적이 필요한 운영에서는 캐스케이드 또는 하이브리드가 더 적합한가?

이 문서는 실제 Runpod을 실행한 결과가 아니다. 공개 GPU 단가와 명시적 처리시간 가정을 사용하는 비용 시뮬레이션이다.

## 비교 경로

1. **Gemini E2E**: 오디오 1회 입력 → 화자 구간·전사·감정·요약·대응 제안.
2. **캐스케이드**: Precision-2 화자분리 → Qwen ASR → XLS-R 감정 → ACP Expert 대응 제안.
3. **하이브리드**: 캐스케이드로 사실·근거·불확실성을 만들고, Gemini 또는 ACP에는 해당 근거 turn만 보내 대응 문구를 생성.

## 왜 E2E만으로 결론 내릴 수 없는가

E2E는 하나의 모델이 통화 전체 맥락을 동시에 보므로 구현이 단순하고, 요약과 대응 문구의 문맥 일관성을 얻기 쉽다. 반면 오류가 생겨도 화자분리·전사·감정·추론 중 어느 단계가 원인인지 분해하기 어렵다.

캐스케이드는 앞 단계의 오류가 뒤로 전파되는 약점이 있지만, 다음 장점이 있다.

- **오류 귀속**: DER/JER, CER, 감정 일치를 따로 측정해 병목을 찾는다.
- **근거 추적**: 대응 제안이 어떤 turn과 감정 신호를 근거로 했는지 남긴다.
- **교체 가능성**: ASR, 감정 모델, Expert 모델을 독립적으로 바꿀 수 있다.
- **데이터 경계**: Qwen·XLS-R을 로컬로 유지하고 필요한 범위만 외부 Expert에 보낼 수 있다.
- **안전한 불확실성 처리**: timestamp·화자 귀속이 애매하면 해당 근거와 권고 강도를 낮출 수 있다.

현재 Gold 2화자 기준선에서 캐스케이드는 `DER/JER`과 `CER`을 별도 수치로 남겼다. 이 자체가 캐스케이드의 관측 가능성 장점이다. 다만 Gemini Silver에서 출발해 사람이 고친 Gold는 Gemini의 독립 성능 비교에 쓰면 안 된다.

## 평가 설계

### 공통 입력과 정답

- 사람이 검수·hash 고정한 실제 2화자 Gold를 사용한다.
- Gold마다 화자 구간, 화자 역할, 전사, 감정 또는 `판단 불확실`을 분리해 둔다.
- 최소 5개 2화자 case가 쌓이기 전에는 평균·우열을 headline으로 제시하지 않는다.
- Gemini가 Silver를 만들었던 case는 **Gold 작성 보조 사례**로는 보존하되, Gemini E2E의 독립 평가 표본에서는 분리하거나 독립 재검수로 앵커링을 낮춘다.

### 품질 지표

| 층위 | 지표 | 해석 |
| --- | --- | --- |
| 화자분리 | strict / 250 ms collar DER, JER, confusion | 누가 말했는지의 오류 |
| 전사 | Korean-normalized CER, 누락 timestamp 비율 | 무엇을 말했는지의 오류 |
| 감정 | Gold turn agreement, abstain 적절성 | 감정을 맞히는 것뿐 아니라 모를 때 멈추는지 |
| 대응 제안 | 근거 turn 정확성, 원인 과단정, 공감성·실행성, 위험 신호 처리 | 제품의 최종 목적 |
| 운영 | p50/p95 지연, 실패·재시도, 모델 버전·출력 schema 재현성 | 실제 운용 가능성 |

대응 제안은 모델 블라인드 상태로 사람이 1–5점 루브릭을 매긴다. `근거 없는 원인 단정`, `의료·법률·금전 보장`, `불확실한데도 확언`은 별도 감점/차단 항목으로 둔다.

## 비용 시뮬레이션

### 공통 가정

- 통화 길이: 10분
- Gemini Flash-Lite Standard 예시: 오디오 25 tokens/s, 출력 175 text tokens/min이라는 Gemini 가격표의 추정 기준을 사용한다.[^gemini-pricing]
- Gemini 3.5 Flash-Lite Standard: input $0.30 / 1M tokens, output $2.50 / 1M tokens.[^gemini-pricing]
- 캐스케이드 GPU 기준: Runpod Secure Cloud RTX A5000 공개 단가 $0.27/h. 실제로 Pod를 만들지 않으며, 단가만 참조한다.[^runpod-pricing]
- Precision-2 요금은 공개 단가로 검증하지 않았으므로 `P_precision`으로 분리한다. 이를 0으로 두고 전체 비용을 비교하면 안 된다.

### Gemini E2E 예시

10분 오디오는 15,000 input tokens, 1,750 output tokens라는 가정이다. 따라서 Flash-Lite Standard의 변동 추정비는 약 **$0.0089/통화**다. 실제 비용은 선택 모델, thinking/output 길이, 캐시, 가격표 개정과 usage metadata에 따라 다시 계산한다.

### 캐스케이드 GPU 계산비 예시

`GPU 계산비 = GPU 시간(분) × $0.27 / 60` 으로 계산한다. 아래는 **Precision-2·저장·네트워크·운영비를 제외한** Qwen/XLS-R 로컬 계산비다.

| 가정 | GPU 시간 | RTX A5000 계산비 / 10분 통화 |
| --- | ---: | ---: |
| warm batch, RTF 0.30 | 3분 | $0.0135 |
| 일반 on-demand, RTF 0.50 | 5분 | $0.0225 |
| 보수적, RTF 1.00 | 10분 | $0.0450 |
| cold start 2분 추가 | +2분 | +$0.0090 |

즉 GPU 계산비만 보면 저처리량 E2E가 경쟁력 있을 수 있다. 캐스케이드의 완전한 총비용은 `GPU + P_precision + 저장/관측/운영`이며, 처리량이 올라가 warm worker와 배치 처리의 비중이 커질 때 다시 달라진다. 반대로 E2E도 긴 구조화 출력, 상위 모델, 재시도, 데이터 보관 정책이 비용을 높일 수 있다.

### 보고용 시나리오

- **저처리량**: 월 100 통화분. cold start·운영 최소화가 중요하므로 E2E 장점이 커질 수 있다.
- **중처리량**: 월 1,000 통화분. p95 지연, 재시도, Precision 비용을 포함한다.
- **고처리량**: 월 10,000 통화분. warm GPU와 batch 조건을 명시하고, GPU·저장·관측 인력을 분리한다.

## 현재의 잠정 결론

- **빠른 PoC·낮은 운영 복잡도**: Gemini E2E가 강한 baseline이다.
- **평가 가능성·감사 가능성·모델 교체**: 캐스케이드가 강하다. 특히 화자 귀속 오류와 전사 오류를 분리해야 감정/대응 결과를 과신하지 않을 수 있다.
- **현실적인 제품 후보**: 캐스케이드로 근거와 uncertainty를 만들고, Gemini/ACP를 Expert mode의 문구·요약 생성에 제한적으로 쓰는 하이브리드다.

이는 현재의 설계 가설이지 성능 결론이 아니다. 다음 실험에서 동일 Gold에 Gemini E2E와 캐스케이드를 실행해 아래 표를 채운 뒤에만 선택한다.

| Gold case | Gemini DER/JER | Cascade DER/JER | Gemini CER | Cascade CER | 감정 agreement | 제안 루브릭 | 비용·지연 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 각 신규 2화자 Gold | 측정 | 측정 | 측정 | 측정 | 모델별 측정 | 블라인드 평가 | usage·wall time |

## 다음 실행

1. `gemini_e2e` runner를 만들되, 화자 수·시간 범위·순서·schema가 깨지면 salvage와 원본 범위를 분리 기록한다.
2. 새 2화자 Gold 최소 3개를 독립 검수해 총 5개 이상으로 늘린다.
3. 동일 Gold에 두 경로를 1회씩 실행하고, usage metadata와 wall time을 원장으로 남긴다.
4. 대응 제안은 근거 turn ID를 숨긴 블라인드 평가본으로 사람 루브릭을 적용한다.

[^gemini-pricing]: [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing) — 2026-09-15 조회.
[^runpod-pricing]: [Runpod GPU pricing](https://www.runpod.io/pricing) — 2026-09-15 조회. RTX A5000 Secure Cloud 공개 단가를 시뮬레이션 기준으로만 사용.
