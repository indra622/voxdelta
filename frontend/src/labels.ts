import type { EmotionLabel, OperationalState, StageName } from "./api/types";

export const STAGE_LABELS: Record<StageName, string> = {
  normalize: "음성 확인",
  diarize: "화자 구분",
  transcribe: "음성 전사",
  confirm_roles: "역할 확인",
  emotion: "감정 분석",
  response_strategy: "응답 맥락",
  transitions: "변화 계산",
  report: "결과 정리",
};

export const EMOTION_LABELS: Record<EmotionLabel, string> = {
  happiness: "행복",
  anger: "분노",
  disgust: "혐오",
  fear: "불안",
  neutral: "중립",
  sadness: "슬픔",
  surprise: "놀람",
};

/** The annotation vocabulary, which is the report's seven labels plus `uncertain`.
 *
 * Kept separate from `EMOTION_LABELS` on purpose: the analysis report never carries
 * `uncertain` as an emotion, and merging the two would put a label into the result screen
 * that the pipeline does not produce. The backend sends the allowed set with each draft,
 * so this map is display spelling only and falls back to the raw label.
 */
export const ANNOTATION_EMOTION_LABELS: Record<string, string> = {
  ...EMOTION_LABELS,
  uncertain: "판단 불확실",
};

export function annotationEmotionLabel(emotion: string): string {
  return ANNOTATION_EMOTION_LABELS[emotion] ?? emotion;
}

export const STATE_LABELS: Record<OperationalState, string> = {
  satisfied: "만족",
  stable: "안정",
  dissatisfied: "불만족",
  escalated: "격앙",
  uncertain: "판단 불확실",
};

export const TRANSMIT_LABELS: Record<string, string> = {
  audio: "오디오",
  text: "전사문",
  features: "특징값",
};

/** Display spelling for providers the backend names in lowercase provenance. */
const PROVIDER_LABELS: Record<string, string> = {
  pyannoteai: "pyannoteAI",
};

export function providerLabel(name: string): string {
  return PROVIDER_LABELS[name.toLowerCase()] ?? name;
}

/** Ordered severity of the operational read, cool -> warm.
 *
 * `uncertain` is deliberately absent: it is not a point on this scale but a refusal
 * to place one, so it is drawn as a hollow dashed mark rather than a fifth colour.
 * See `--state-*` in styles.css for the validated hexes.
 */
export const STATE_SCALE = ["satisfied", "stable", "dissatisfied", "escalated"] as const;

export function stateTone(state: OperationalState, uncertain = false): string {
  if (uncertain || state === "uncertain") return "var(--state-withheld)";
  return "var(--state-" + state + ")";
}
