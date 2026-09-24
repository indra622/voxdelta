# VoxDelta 8주 로드맵 (프로젝트 전체)

- 전체 기간: **2026-08-17 ~ 2026-10-11** (8주)
- 프로젝트 첫 커밋: 2026-08-18 (`docs: add VoxDelta design specification`)
- 문서 기준일: 2026-08-25 → **W2 진행 중**
- 완주 정의: 실제 한국어 상담 통화 1건이 업로드부터 리포트 export까지 실모델로 완주하고, 감정 provider 선택이 재현 가능한 근거로 설명되는 상태.

## 한 장 요약

```
W1 ████ 설계 → 코어 파이프라인 → 실모델 어댑터 → 감정 데이터·1차 학습 실험   ✅ 완료
W2 ████ RunPod 전면 파인튜닝 실험 인프라 → 파일럿 실행                      🔵 진행 중
W3 ░░░░ 감정 provider 확정 + 상담 도메인 데이터 도입                        ⬜
W4 ░░░░ ASR/diarization 평가 + 통합 벤치마크 러너·선택 게이트               ⬜
W5 ░░░░ 골드 평가셋 + 부정강도 캘리브레이션                                  ⬜
W6 ░░░░ Gemini·응대전략·요약 provider + 로컬 vs API 봉인 비교                ⬜
W7 ░░░░ 대시보드 (업로드·진행·역할확인·동기화 분석·전이 증거)                ⬜
W8 ░░░░ Export + E2E 수용 + 데모 + 마감                                      ⬜
```

프로젝트는 **파이프라인 → 모델 → 평가 → 화면** 순서로 쌓는다. 화면을 마지막에 두는 이유는 거버닝 메시지가 요구하는 "증거로 되짚을 수 있는 출력"이 먼저 존재해야 화면이 그것을 보여줄 수 있기 때문이다.

## 실제 진행 속도 메모

W1 분량(계획상 3주치 작업: 코어 파이프라인 8태스크 + 데이터 임포트 + 감정 학습 실험 3회)이 **08-18~08-20 사흘**에 끝났고, W2 분량(RunPod 실험 인프라 51파일/12,321줄)이 08-22~08-25 나흘에 들어왔다. 캘린더 주차는 명목 일정이고 실제 소화 속도는 그보다 빠르다. 따라서 아래 W3 이후는 **여유 주차가 아니라, 앞당겨 끝난 만큼 뒤 항목을 당겨오는 버퍼**로 쓴다.

## 주차 표

| 주차 | 기간 | 주제 | 상태 | 종료 게이트 |
|---|---|---|---|---|
| W1 | 08-17 ~ 08-23 | 설계 · 코어 파이프라인 · 실모델 어댑터 · 감정 1차 실험 | ✅ | 로컬 API 수직 슬라이스 완주 + 감정 베이스라인 확보 |
| W2 | 08-24 ~ 08-30 | RunPod 전면 파인튜닝 실험 | 🔵 | 파일럿 A/B 완료 + 게이트 판정 기록 |
| W3 | 08-31 ~ 09-06 | 감정 provider 확정 + 상담 데이터 도입 | ⬜ | 기본 감정 provider 1종 동결 |
| W4 | 09-07 ~ 09-13 | ASR/diarization 평가 + 벤치마크 자동화 | ⬜ | 명령 하나로 선택 결과 재생성 |
| W5 | 09-14 ~ 09-20 | 골드 평가셋 + 강도 캘리브레이션 | ⬜ | 200발화/60triplet + 합치도 보고 |
| W6 | 09-21 ~ 09-27 | Gemini · 응대전략 · 요약 provider | ⬜ | 동결 입력 위 로컬 vs API 비교 1회 |
| W7 | 09-28 ~ 10-04 | 대시보드 | ⬜ | 실모델 job 1건 화면 완주 |
| W8 | 10-05 ~ 10-11 | Export · E2E · 마감 | ⬜ | 실제 통화 완주 + `git status` 클린 |

---

# ✅ W1 (08-17 ~ 08-23) — 설계부터 감정 1차 실험까지

실제 작업일: 08-18 ~ 08-20 (3일, 약 80커밋).

### 1. 설계와 계획 고정 (08-18)

- 설계 명세 승인: 7분류 감정 스키마, 운영 상태 4종 + `uncertain`, `delta ±0.20` 회복/악화 기준, 8스테이지 파이프라인, provider 인터페이스 5종, 인과 주장 금지 원칙.
- 구현 계획 3분할: 코어 파이프라인 / 대시보드 / 모델·평가. 각각 독립적으로 수용·거부할 수 있게 나눔.

### 2. 코어 파이프라인과 로컬 API (08-19)

- 보안 경계: 자격증명 부트스트랩, 환경변수·OS 자격증명 저장소만 허용, 로그·아티팩트에 비밀 미기록.
- 도메인 계약: `AudioAsset` / `SpeakerSegment` / `Utterance` / `EmotionResult` / `ResponseStrategyResult` / `EmotionTransition` / `AnalysisReport`.
- 원자적 아티팩트 저장소 + SQLite job 저장소, 스테이지 캐시 키(입력 해시 + 설정 + provider + 모델 버전).
- 오디오 수용 경계: FFmpeg 기반 검증·정규화, **DB 승인 전 스테이징 검증**, 손상/범위 밖 파일은 job 생성 없이 거절.
- Fake provider 5종 + 계약 테스트 — 실모델 없이 데이터 계약을 먼저 증명.
- 운영 상태 매핑 + 3턴 전이 계산 + 중앙값 스무딩.
- 재개 가능 파이프라인 러너와 **역할 확인 게이트**(사용자 승인 없이는 감정 분석으로 넘어가지 않음).
- FastAPI 수직 슬라이스: 업로드(202) / 상태 / 역할 확인 / 스테이지 재시도 / 리포트 / Range 오디오 스트리밍 / 내구성 있는 삭제. capability 토큰 + Host·Origin 검사로 로컬 전용 고정.

### 3. 실모델 어댑터와 데이터셋 (08-19 ~ 08-20)

- 매니페스트 계약(경로 traversal fail-closed 포함).
- Diarization: pyannote `community-1`(로컬 기본) + `precision-2`(선택적 원격 벤치마크, 암묵 폴백 금지).
- ASR: faster-whisper `large-v3-turbo`(안정) vs Qwen3-ASR `1.7B` + `0.6B` 정렬기(모던).
- 감정: Wav2Vec2-XLS-R 300M(안정) vs emotion2vec+ large 동결 인코더 + 7클래스 헤드(모던).
- AI Hub 263 감정 데이터 임포트: **36,665 발화**(train 29,476 / val 3,569 / test 3,620), 16 kHz mono PCM16 정규화, 화자·대화 누수 방지 분할.

### 4. 감정 학습 실험 1차 (08-20)

| 실험 | 결과 |
|---|---|
| emotion2vec+ 스모크 (35 items) | macro-F1 0.083 |
| **emotion2vec+ 클래스 균형 학습, 전체 validation 3,569** | **macro-F1 0.240 / ECE 0.078 / 7라벨 전부 양수 F1** ← 현행 베이스라인 |
| XLS-R 통제 스모크 — 전체 파라미터 | macro-F1 0.0357 (전부 surprise) |
| XLS-R — encoder 20~23 + 헤드 | macro-F1 0.0357 (전부 surprise) |
| XLS-R — seeded LoRA r8, layer 16~23 q/v | macro-F1 0.0357 (전부 surprise) |

**이 주의 핵심 발견**: XLS-R 로컬 소규모 스모크는 세 변형 모두 **단일 클래스 붕괴**를 재현했다. RNG 의존성을 제거한 재현 실행에서도 동일. → Stage A no-go 판정, Stage B 스킵, **로컬 재시도 중단하고 원격 GPU 전면 파인튜닝으로 분기**. 이 판정이 W2의 존재 이유다.

### W1 산출물 (실재)

| 산출물 | 위치 |
|---|---|
| 설계 명세 1건 + 구현 계획 3건 + 실험 설계·계획 8건 | `docs/superpowers/specs/`, `docs/superpowers/plans/` |
| 백엔드 코어 48개 모듈 | `backend/src/voxdelta/{domain,jobs,audio,pipeline,providers,analysis,api}/` |
| 테스트 스위트 9종 (domain/jobs/audio/pipeline/providers/analysis/api/evaluation/integration) | `backend/tests/` |
| 실모델 어댑터 6종 + fake 5종 | `backend/src/voxdelta/providers/` |
| 데이터 준비·학습·평가 엔트리포인트 7개 | `backend/scripts/` |
| 감정 매니페스트 36,665발화 + 결정적 분할 | `data/manifests/{emotion,train,validation,test,emotion-smoke}.jsonl` |
| 집계 벤치마크 결과 7건 | `data/benchmarks/*.json` |
| 재현 가능한 로컬 스모크 워크플로 (curl+jq 전 구간) | `backend/README.md` |

**W1 게이트 결과**: 통과. 로컬 API가 fake provider로 업로드→리포트→삭제를 완주하고, 감정 쪽은 사용 가능한 베이스라인 1종을 확보했다.

---

# 🔵 W2 (08-24 ~ 08-30) — RunPod 전면 파인튜닝 실험

실제 작업일: 08-22 (설계) + 08-24~08-25 (구현). 브랜치 `feat/runpod-xls-r-full-finetuning-design`, 51파일 / 12,321줄.

### 완료된 것

- **실험 설계 확정**: 핀된 `facebook/wav2vec2-xls-r-300m` (revision `1a640f32…`, 가중치 SHA-256 고정, seed 622, 20초 윈도우, 선택 지표 val macro-F1). 파일럿 2종을 동일 예산으로 비교한 뒤 승자만 확장하는 구조.
  - 파일럿 A: 클래스 균형 샘플링, 손실 가중 없음
  - 파일럿 B: 자연 분포 샘플링 + 완화된 가중
- **경계 격리**: `runpod/` 가 RunPod 작업의 유일한 경계. `backend/` 는 재사용 라이브러리로 유지되고 Pod 주소·자격증명·체크포인트·실행 상태를 절대 흡수하지 않는다. 프로덕션 API는 `runpod/` 를 import 하지 않는다.
- **소유권 분리**: 실계정 조작(레지스트리 로그인, 이미지 push, Pod 생성, 비용 승인, SSH, rsync)은 전부 사용자가 실행. 에이전트는 검증된 아티팩트와 정확한 커맨드 패킷만 제공.
- 5스테이지 재시작 가능 실험(package → transfer → preflight → pilot → full), 불변 아티팩트와 복구 원장.
- 프라이버시 최소화 전송 번들(전사 텍스트 제외), OCI attestation 검증이 붙은 CUDA 이미지 빌드, 봉인된 최종 비교 게이트.

### 남은 것 (이번 주)

1. 브랜치를 main에 정리·머지하고 구현 게이트(pytest / ruff / mypy) 통과.
2. 로컬 preflight — 번들 패키징, PCM16 mono 16 kHz 검증, 매니페스트 봉인, 해외 이전 승인 근거 확인.
3. 이미지 push → Pod 생성 → 원격 preflight.
4. **파일럿 A/B 실행 → 게이트 판정 기록.**

### W2 산출물

**이미 실재하는 것** (브랜치 `feat/runpod-xls-r-full-finetuning-design`)

| 산출물 | 위치 |
|---|---|
| 실험 설계 + 실행 work order + 한국어 운영자 매뉴얼 | `docs/superpowers/specs/2026-08-22-runpod-xls-r-full-finetuning-design.md`, `runpod/IMPLEMENTATION_PLAN.md`, `runpod/OPERATOR.md` |
| 불변 실험 프로파일 (파일럿 A/B 레시피 동결) | `runpod/config/experiment.toml`, `runpod/src/voxdelta_runpod/recipes.py` |
| 핀된 linux/amd64 CUDA 이미지 정의 + attestation 검증 | `runpod/docker/`, `runpod/src/voxdelta_runpod/image.py` |
| 오케스트레이션 12모듈 (원장·패키징·학습·게이트·워크플로) | `runpod/src/voxdelta_runpod/` |
| 로컬/원격 실행 스크립트 15개 | `runpod/scripts/` |
| 실계정 없이 도는 합성·단위·통합 테스트 | `runpod/tests/` |

**이번 주 예상 산출물**

- 파일럿 A/B 집계 리포트 → `runpod/results/pilot-{a,b}/` (비커밋) → 집계만 `data/benchmarks/xls-r-runpod-pilot-{a,b}.json` 로 커밋
- 게이트 판정 문서 → `docs/decisions/2026-08-30-runpod-pilot-gate.md` (통과/미통과 + 근거 수치 + 다음 분기)
- 불변 실행 원장 + 재현 커맨드 로그 → `runpod/runtime/` (비커밋, 요약만 판정 문서에 인용)
- main에 머지된 `runpod/` 워크스페이스 (pytest/ruff/mypy 통과)

**게이트**: 두 파일럿 중 하나라도 7클래스 경계를 학습했는가(단일 클래스 붕괴 아님 + 베이스라인 대비 진전)?
**리스크**: Pod 확보·대용량 전송 지연. 대응 — 주 내에 Pod가 서지 않으면 W3 전반부까지만 연장하고 이후 폴백 경로로 전환.

---

# ⬜ W3 (08-31 ~ 09-06) — 감정 provider 확정 + 상담 데이터 도입

W2 게이트 결과에 따라 분기하되, **주말 전에 기본 감정 provider를 하나로 동결한다.** 이 결정을 미루면 뒤 5주가 전부 흔들린다.

- **게이트 통과 시**: 승자 레시피로 전면 파인튜닝 1회 → 검증 게이트 → 봉인된 최종 비교 1회 → 체크포인트 회수 및 provenance 고정.
- **게이트 미통과 시**: 폴백 하나만 고르고 문서화.
  - (a) 7분류를 3분류(satisfied / stable / negative)로 축소해 신뢰 가능한 축부터 세운다
  - (b) 베이스 인코더 교체(WavLM / HuBERT)
  - (c) emotion2vec+ 0.240을 기본값으로 확정하고 감정 개선을 이번 8주 범위 밖으로 뺀다
- 미선택 후보는 "사용 불가 + 사유"로 기록한다(조용한 폴백 금지).
- 병행: AI Hub 상담 음성 데이터 도입 착수 — 비어 있는 `data/manifests/consultation.jsonl` 을 채우고 다운로드 절차를 문서화(원본은 비커밋).

**예상 산출물**

- 감정 provider 결정 문서 → `docs/decisions/2026-09-06-emotion-provider-selection.md` (선택안 + 탈락 후보별 사유 + 고정된 체크포인트 digest)
- 기본 provider의 전체 validation 결과 → `data/benchmarks/emotion-default-validation.json`
- 전면 파인튜닝 실행 시: 회수된 체크포인트 + provenance 기록 → `runpod/models/` (비커밋), 식별자만 provider 설정에 고정
- 상담 도메인 매니페스트 1차 → `data/manifests/consultation.jsonl` (현재 0줄 → 채움)
- 상담 데이터 반입 절차 문서 → `docs/data/consultation-dataset.md` (다운로드·검증·비커밋 규칙)
**게이트**: `data/benchmarks/` 에 기본 provider의 validation 결과가 있고 7개 라벨 F1이 모두 양수인가?

# ⬜ W4 (09-07 ~ 09-13) — ASR/diarization 평가 + 벤치마크 자동화

- ASR 비교: faster-whisper vs Qwen3-ASR — 상담 test 서브셋에서 CER/WER, 완주율, 중앙 지연, 피크 메모리.
- Diarization: pyannote community-1 의 DER·화자 혼동 측정. precision-2 는 선택적 원격 벤치마크로만.
- 분할 무결성 테스트: 같은 통화·화자가 train/eval에 걸치지 않음을 자동 검증.
- **통합 벤치마크 러너**: 분할 해시, provider·모델 버전, 설정, 타임스탬프, 지연/피크 메모리/API 비용을 한 번에 기록.
- 모델 선택 게이트를 코드로 고정 — ASR(완주율 95%, CER 1%p 절대 개선, 동률 시 지연 2배 이내), 감정(완주율 95%, ECE 0.02 이상 악화 금지, macro-F1 +0.01 개선 시 교체, ±0.01 이내면 ECE → 지연 순).

**예상 산출물**

- ASR 비교 리포트 → `data/benchmarks/asr-consultation-{faster-whisper,qwen3}.json` (CER/WER·완주율·지연·피크 메모리)
- Diarization 리포트 → `data/benchmarks/diarization-community1.json` (DER·화자 혼동)
- 통합 벤치마크 러너 → `backend/src/voxdelta/evaluation/benchmark.py` + `backend/scripts/run_benchmark.py`
- 선택 게이트 구현 확장 → `backend/src/voxdelta/evaluation/selection.py` (ASR·감정 규칙 전부 코드로 고정)
- 선택 결과 스냅샷 → `data/benchmarks/selection-result.json` (분할 해시·모델 버전·설정·타임스탬프 포함)
- 분할 누수 방지 테스트 → `backend/tests/evaluation/test_split_integrity.py`

**게이트**: `git clean` 상태에서 명령 하나로 선택 결과가 동일하게 재생성되는가?

# ⬜ W5 (09-14 ~ 09-20) — 골드 평가셋 + 부정강도 캘리브레이션

여기부터가 VoxDelta가 "또 하나의 감정 분류기"와 갈라지는 지점이다.

- 허용된 상담 샘플로 골드셋 구축: **고객 발화 200개 이상 / 통화 20건 이상 / 3턴 triplet 60개 이상.**
- 2인 독립 어노테이션(운영 상태, 5점 부정강도 루브릭, 전이 클래스) → 불일치 조정 → 어노테이터 간 합치도 보고.
- 7분류 확률 → `negative_intensity` (0.0~1.0) 캘리브레이션을 골드셋 기준으로 provider별 독립 학습.
- 3턴 중앙값 스무딩 확정 + delta 임계 민감도 분석(0.15 / 0.20 / 0.25) 수록.

**예상 산출물**

- 골드 어노테이션 스키마 + 데이터 → `backend/src/voxdelta/evaluation/gold.py` 확장, `data/gold/` (원문 비커밋, 스키마·집계만 커밋)
- 어노테이터 간 합치도 리포트 → `data/benchmarks/gold-agreement.json` (운영 상태·강도·전이 클래스별)
- 부정강도 캘리브레이터 → `backend/src/voxdelta/analysis/calibration.py` + provider별 계수 파일
- 전이 평가 리포트 → `data/benchmarks/transition-gold.json` (회복/안정/악화 macro-F1, 강도 MAE·Spearman)
- 임계 민감도 표 → `docs/analysis/threshold-sensitivity.md` (0.15 / 0.20 / 0.25)
- 골드셋 구축 절차 문서 → `docs/data/gold-annotation-guide.md` (5점 루브릭 포함)

**게이트**: 회복/안정/악화 3분류 macro-F1을 골드셋에서 산출할 수 있고, 강도 MAE·Spearman이 기록되는가?

# ⬜ W6 (09-21 ~ 09-27) — Gemini · 응대전략 · 요약 provider

- Gemini `3.6 Flash` 감정 provider: 구조화 출력 검증, 동일 7라벨 매핑, 자체 캘리브레이션, 비용·지연 메타데이터, 전송 내용 공시.
- 응대전략 provider: 8종 라벨(사과 / 공감·인정 / 확인질문 / 정보·설명 / 해결·다음조치 / 정책·거절 / 인사·마무리 / 기타) 1차 + 보조 다중 라벨.
- 리포트 요약 provider: **인과 표현 금지를 테스트로 강제**(금지 문구 패턴 단위 테스트).
- 로컬 vs API 비교는 **동일한 동결 전사·화자 아티팩트** 위에서만 수행 — ASR 차이가 감정 비교에 섞이지 않게.
- 리포트 JSON 스키마 동결 + 스냅샷 테스트(대시보드와 export가 같은 데이터를 쓰도록).

**예상 산출물**

- Gemini 감정 provider → `backend/src/voxdelta/providers/gemini_emotion.py`
- 응대전략 provider → `backend/src/voxdelta/providers/response_strategy.py` (8종 1차 + 보조 라벨)
- 리포트 요약 provider → `backend/src/voxdelta/providers/report_summary.py`
- 인과 표현 금지 테스트 → `backend/tests/reports/test_causal_language.py` (금지 패턴 목록 포함)
- 동결 입력 위 로컬 vs API 비교 결과 → `data/benchmarks/emotion-local-vs-gemini.json` (품질·캘리브레이션·지연·비용·실패 수)
- 동결된 리포트 JSON 스키마 + 스냅샷 → `backend/src/voxdelta/reports/schema.py`, `backend/tests/reports/__snapshots__/`
- 원격 전송 공시 갱신 → `GET /api/config/providers` 응답에 전송 내용·보존정책 반영

**게이트**: 자격증명 없는 환경에서 전체 테스트가 통과하고, 승인 없이 원격 스테이지가 호출되지 않는가?

# ⬜ W7 (09-28 ~ 10-04) — 대시보드

`docs/superpowers/plans/2026-08-18-voxdelta-dashboard.md` Task 1–5.

- React + TypeScript + Vite, 타입 생성된 로컬 API 클라이언트.
- 업로드 화면: 어떤 스테이지가 로컬이고 어떤 스테이지가 외부 전송인지 **처리 시작 전에** 명시.
- 스테이지 진행 뷰 + 실패 스테이지 재시도.
- 역할 확인 화면(제안 매핑 + swap).
- 분석 화면: 오디오 플레이어–전사–감정 타임라인 동기화, 화자 색상, 불확실 마커.
- 회복/악화 구간을 3턴 증거 카드로 표시하고 응대 전략 배지 부착.

**예상 산출물**

- 프론트엔드 워크스페이스 → `frontend/` (Vite + React + TypeScript, 타입 생성된 API 클라이언트 `frontend/src/api/`)
- 화면 5종 → `frontend/src/features/{jobs,roles,analysis}/`
  - 업로드 + 전송 공시, 스테이지 진행 + 재시도, 역할 확인/swap, 동기화 분석 뷰, 회복·악화 증거 카드
- 컴포넌트 테스트 → `frontend/tests/` (Vitest)
- 화면 캡처 세트 → `docs/demo/screenshots/`

**게이트**: 실모델로 처리된 job 1건을 화면에서 끝까지 볼 수 있는가?

# ⬜ W8 (10-05 ~ 10-11) — Export · E2E · 마감

- 대시보드와 동일한 리포트 JSON으로 HTML / PDF export.
- Playwright E2E: 업로드 → 역할 확인 → 실패 재시도 → 리포트 → export → 삭제.
- 5~15분 실제(또는 라이선스 허용) 한국어 상담 통화 1건 완주 데모 + 스크린샷.
- 실패 경로 점검: 손상 오디오, 자격증명 누락, API 타임아웃, 화자 3인 이상, 중첩 발화.
- 문서 마감: README 상태 갱신, 모델 선택 근거 요약, **알려진 한계를 정직하게 기재**(감정 macro-F1 수준, 골드셋 규모, 인과 미주장).

**예상 산출물**

- HTML / PDF export → `backend/src/voxdelta/reports/export.py` + `frontend/src/features/analysis/` export 액션
- E2E 스위트 → `frontend/tests/e2e/` (Playwright: 업로드→역할확인→재시도→리포트→export→삭제)
- 실패 경로 테스트 → `backend/tests/integration/test_failure_paths.py`
- 데모 리포트 1건 → `docs/demo/sample-report.html` (라이선스 허용 통화 기준, 원본 오디오 비커밋)
- 최종 문서 세트 → `README.md` 상태 갱신, `docs/RESULTS.md` (모델 선택 근거 요약 + 알려진 한계)

**게이트**: 마스터 계획의 완료 정의 충족 + `git status --short` 비어 있음.

---

## 최종 산출물 (W8 종료 시점)

8주가 끝났을 때 저장소에 실제로 존재해야 하는 것들.

### 실행 가능한 소프트웨어

```text
voxdelta/
├── backend/          8스테이지 재개 가능 파이프라인 + 로컬 FastAPI + 실모델 provider 9종
├── frontend/         업로드 → 역할확인 → 분석 → 증거 → export 대시보드
├── runpod/           격리된 전면 파인튜닝 실험 워크스페이스 (재실행 가능)
├── data/manifests/   감정 36,665발화 + 상담 도메인 + 골드 평가셋 분할
└── data/benchmarks/  집계 벤치마크 결과 전량
```

### 증거 산출물

- **모델 선택 근거**: ASR·diarization·감정 후보별 벤치마크 JSON과, 그것을 규칙으로 환원한 `selection-result.json`. "왜 이 모델인가"를 사람 판단이 아니라 재실행 가능한 명령으로 답한다.
- **골드 평가 결과**: 회복/악화 검출 macro-F1, 강도 MAE·Spearman, 어노테이터 합치도, 임계 민감도.
- **로컬 vs 프론티어 비교**: 동일 동결 입력 위에서의 품질·캘리브레이션·지연·비용·실패 수.
- **실험 판정 기록**: `docs/decisions/` — RunPod 파일럿 게이트, 감정 provider 선택. 실패한 분기(XLS-R 로컬 붕괴 포함)도 근거와 함께 남긴다.

### 데모·포트폴리오 산출물

- 5~15분 실제 상담 통화 1건의 완주 리포트(HTML) + 대시보드 스크린샷 세트
- `docs/RESULTS.md` — 최종 수치, 선택 근거, **알려진 한계**(감정 macro-F1 수준, 골드셋 규모, 인과 미주장)
- `README.md` — 이름·거버닝 메시지·현재 상태가 실제와 일치하는 상태

### 끝까지 커밋되지 않는 것

원본 데이터셋, 통화 오디오, 정규화 미디어, 체크포인트, 자격증명, 항목 단위 예측·전사, 생성된 사설 리포트, provider 페이로드, RunPod 실행 상태·SSH·전송 아카이브. 저장소에는 **코드 · 매니페스트 메타데이터 · 집계 결과 · 문서**만 남는다.

## 크로스 컷팅 (매주 반복)

- 금요일 = 게이트 판정일. 미통과 주는 **범위를 줄이되 일정은 밀지 않는다**(미룬 항목을 명시적으로 기록).
- 모든 작업은 TDD, 태스크 단위 커밋.
- 벤치마크는 집계 수준만 커밋(항목 단위 예측·전사·오디오는 비커밋).
- 새 원격 provider가 추가될 때마다 UI 전송 공시와 보존정책 링크를 함께 갱신.
- 인과 표현 금지 테스트는 리포트 문구가 바뀔 때마다 확장.

## 우선순위 (자원이 모자랄 때 버리는 순서)

1. pyannote precision-2 원격 diarization 벤치마크 (선택적)
2. Qwen3-ASR 0.6B 저메모리 프로파일
3. 임계 민감도 분석 확장
4. PDF export (HTML export까지만)
5. 감정 7분류 성능 개선 자체 — **버리더라도 "변화 구간 + 증거" 기능은 유지한다.** 거버닝 메시지상 이 기능이 마지막까지 남아야 한다.

## 주요 리스크

| 리스크 | 영향 | 대응 |
|---|---|---|
| RunPod 파일럿도 단일 클래스 붕괴 | 감정 품질이 0.240에 고정 | W3에 폴백 3안 중 택1을 강제. 8주 완주 자체는 막지 않음 |
| AI Hub 상담 데이터 반입 지연 | W4 ASR/diarization 평가 불가 | 감정 데이터 일부 + 합성 픽스처로 파이프라인 검증 먼저, 실측은 뒤로 |
| 골드셋 어노테이션 인력 부족 | W5 캘리브레이션·전이 평가 축소 | triplet 60개를 최소선으로 사수, 발화 200개는 축소 허용 |
| 대시보드가 2주에 안 들어옴 | 데모 미완 | export를 HTML로 축소하고 비교 화면을 후순위로 |
