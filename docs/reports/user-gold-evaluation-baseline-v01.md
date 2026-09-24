# VoxDelta 사용자 제공 Gold 평가 기준선 v01

## 범위

이 문서는 사람이 검수해 확정한 두 개의 2화자 한국어 대화 Gold를 대상으로 한 첫 E2E 평가 기준선이다.
파이프라인은 Precision-2 화자분리, Qwen3-ASR 1.7B 전사, calibrated XLS-R 감정 분류로 구성된다.

## 결과

| Case | strict DER/JER | 250 ms collar DER/JER | Qwen mixed CER | XLS-R 감정 일치 |
| --- | --- | --- | --- | --- |
| USER_EVAL_20260911_01 | 9.94% / 17.30% | 6.23% / 11.08% | 29.32% | 8/17, 47.06% |
| USER_EVAL_20260911_02A | 14.40% / 16.10% | 9.39% / 10.04% | 17.32% | 4/9, 44.44% |

## 해석 제한

- 두 case만으로 일반 성능이나 모델 간 우열을 주장할 수 없다.
- 감정 지표는 정렬된 Gold turn에서만 계산했으며, 역할 임의 매핑 영향을 받는다.
- Gemini Silver를 사람이 고쳐 Gold를 만들었으므로, Gemini 결과는 독립 기준선이 아니다.
- 원본 02의 후반부인 단일화자 `USER_EVAL_20260911_02B`는 2화자 E2E와 계약이 달라 headline 표에서 제외했다.

## 재현 범위

평가 artifact는 transcript-free JSON으로 저장돼 있다.

- `data/benchmarks/user-gold-evaluation-01.json`
- `data/benchmarks/user-gold-evaluation-02A.json`
- `data/benchmarks/user-gold-evaluation-02B-local-auxiliary.json`

각 2화자 case의 Precision-2 호출은 upload 1회·job 1회·retry 0회였다. Qwen3-ASR와 XLS-R은 local offline cache에서 실행했다.

## 다음 단계

동일한 동의·검수 절차를 거친 실제 2화자 Gold를 최소 세 개 더 추가한다. 새 case는 동일 지표를 한 행씩 누적하고, 다섯 case 이상일 때만 평균을 보조 요약으로 제시한다.
