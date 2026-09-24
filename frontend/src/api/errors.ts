import { ApiError } from "./client";

export interface ErrorGuidance {
  code: string;
  title: string;
  detail: string;
}

/** Every public error code the local API can surface, in the user's language.
 *
 * The backend deliberately answers in sanitized English so it leaks nothing about
 * the recording; this screen is Korean end to end, so the copy is mapped here and
 * says what the person can actually do next.
 */
const GUIDANCE: Record<string, Omit<ErrorGuidance, "code">> = {
  backend_unreachable: {
    title: "로컬 백엔드에 연결하지 못했어요",
    detail: "터미널에서 npm run poc 가 계속 실행 중인지 확인한 뒤 다시 시도해 주세요.",
  },
  capability_required: {
    title: "로컬 접근 권한이 만료됐어요",
    detail: "capability token은 실행할 때마다 새로 만들어집니다. npm run poc 를 다시 실행하고 이 페이지를 새로 고쳐 주세요.",
  },
  invalid_host: {
    title: "이 주소로는 로컬 API에 접근할 수 없어요",
    detail: "http://127.0.0.1:5173 으로 열어 주세요.",
  },
  origin_not_allowed: {
    title: "이 출처에서는 로컬 API에 접근할 수 없어요",
    detail: "http://127.0.0.1:5173 으로 열어 주세요.",
  },
  unsupported_audio_type: {
    title: "지원하지 않는 파일 형식이에요",
    detail: "WAV, MP3, M4A 파일만 분석할 수 있습니다.",
  },
  empty_upload: {
    title: "파일 내용이 비어 있어요",
    detail: "녹음이 제대로 저장됐는지 확인하고 다시 올려 주세요.",
  },
  upload_too_large: {
    title: "파일이 허용된 크기를 넘었어요",
    detail: "필요한 구간만 잘라서 다시 올려 주세요.",
  },
  audio_rejected: {
    title: "이 오디오는 분석 대상으로 받을 수 없어요",
    detail: "백엔드에 설정된 최소·최대 길이를 벗어났거나 열 수 없는 파일일 수 있습니다. 통화 한 건 분량의 정상 녹음으로 다시 시도해 주세요.",
  },
  audio_not_ready: {
    title: "정규화된 음성이 아직 준비되지 않았어요",
    detail: "음성 확인 단계가 끝난 뒤 다시 시도해 주세요.",
  },
  invalid_normalized_audio: {
    title: "정규화된 음성이 손상됐어요",
    detail: "음성 확인 단계를 다시 시도하거나 새 음성으로 시작해 주세요.",
  },
  job_capacity_reached: {
    title: "동시에 진행할 수 있는 분석 수를 넘었어요",
    detail: "진행 중이던 다른 분석을 끝내거나 삭제한 뒤 다시 시도해 주세요.",
  },
  job_not_found: {
    title: "이 분석 기록은 더 이상 남아 있지 않아요",
    detail: "이미 삭제됐거나 백엔드가 다시 시작됐습니다. 새 음성으로 다시 시작해 주세요.",
  },
  job_deleting: {
    title: "이 분석은 삭제되는 중이에요",
    detail: "삭제가 끝나면 새 음성으로 다시 시작해 주세요.",
  },
  deletion_incomplete: {
    title: "로컬 삭제가 끝나지 않았어요",
    detail: "일부 파일이 이 컴퓨터에 남아 있을 수 있습니다. 삭제를 다시 시도해 주세요.",
  },
  unsupported_speaker_count: {
    title: "고객과 상담원 두 사람의 대화만 분석할 수 있어요",
    detail: "이 PoC는 화자가 정확히 두 명인 녹음만 지원합니다. 한 명이거나 셋 이상이면 분석하지 않습니다.",
  },
  insufficient_emotion_coverage: {
    title: "감정을 판단할 고객 발화가 충분하지 않았어요",
    detail: "고객이 말한 구간이 더 많이 담긴 녹음으로 다시 시도해 주세요.",
  },
  invalid_stage_output: {
    title: "분석 단계가 올바르지 않은 결과를 냈어요",
    detail: "같은 단계를 다시 시도해 보고, 반복되면 다른 녹음으로 확인해 주세요.",
  },
  invalid_report: {
    title: "결과 파일을 읽지 못했어요",
    detail: "결과 정리 단계를 다시 시도해 주세요.",
  },
  report_not_ready: {
    title: "결과가 아직 준비되지 않았어요",
    detail: "남은 단계가 끝난 뒤 다시 불러와 주세요.",
  },
  pipeline_failed: {
    title: "분석 단계 하나가 실패했어요",
    detail: "실패한 단계를 다시 시도할 수 있습니다.",
  },
  role_confirmation_required: {
    title: "역할 확인이 먼저 필요해요",
    detail: "고객과 상담원을 지정한 뒤 분석을 이어가 주세요.",
  },
  role_confirmation_not_ready: {
    title: "지금은 역할을 확인할 단계가 아니에요",
    detail: "화면의 진행 상태를 다시 확인해 주세요.",
  },
  invalid_role_mapping: {
    title: "고객과 상담원을 한 명씩 지정해야 해요",
    detail: "관찰된 두 화자에게 서로 다른 역할을 지정해 주세요.",
  },
  invalid_role_candidate: {
    title: "역할 후보 정보가 더 이상 유효하지 않아요",
    detail: "역할 확인 단계를 다시 시도하거나 새 음성으로 시작해 주세요.",
  },
  stale_role_candidate: {
    title: "역할 확인이 다른 작업과 겹쳤어요",
    detail: "잠시 후 다시 확인해 주세요.",
  },
  invalid_stage: {
    title: "다시 시도할 수 없는 단계예요",
    detail: "새 음성으로 다시 시작해 주세요.",
  },
  annotation_not_found: {
    title: "이 대화의 Silver 초안을 찾지 못했어요",
    detail: "목록을 새로 고친 뒤 다시 선택해 주세요.",
  },
  annotation_unreadable: {
    title: "저장된 Silver 초안을 읽지 못했어요",
    detail: "파일이 손상됐거나 형식이 맞지 않습니다. 초안을 다시 만든 뒤 검수해 주세요.",
  },
  invalid_conversation_id: {
    title: "올바른 대화 식별자가 아니에요",
    detail: "목록에서 초안을 다시 선택해 주세요.",
  },
  reviewer_required: {
    title: "검수자 식별자가 필요해요",
    detail: "확정본에는 누가 확인했는지가 함께 기록됩니다. 본인 식별자를 입력해 주세요.",
  },
  review_acknowledgement_required: {
    title: "확인했다는 표시가 필요해요",
    detail: "초안을 음성과 대조해 확인했다는 항목을 체크한 뒤 다시 시도해 주세요.",
  },
  review_note_too_long: {
    title: "검수 메모가 너무 길어요",
    detail: "메모를 줄인 뒤 다시 시도해 주세요.",
  },
  corrected_turns_required: {
    title: "확정할 발화가 없어요",
    detail: "확정본은 검수자가 제출한 발화로 만들어집니다. 최소 한 개는 있어야 합니다.",
  },
  too_many_turns: {
    title: "발화 수가 허용 범위를 넘었어요",
    detail: "한 번에 확정할 수 있는 발화 수를 넘었습니다.",
  },
  invalid_turn_speaker: {
    title: "화자 값이 올바르지 않아요",
    detail: "백엔드가 알려준 위치의 화자를 고친 뒤 다시 시도해 주세요.",
  },
  invalid_turn_transcript: {
    title: "전사문이 올바르지 않아요",
    detail: "빈 전사문은 확정할 수 없습니다. 해당 발화를 고친 뒤 다시 시도해 주세요.",
  },
  invalid_turn_rationale: {
    title: "감정 근거가 너무 길어요",
    detail: "해당 발화의 근거를 줄인 뒤 다시 시도해 주세요.",
  },
  invalid_turn_emotion: {
    title: "허용되지 않은 감정 라벨이에요",
    detail: "목록에 있는 감정 라벨 중에서 선택해 주세요.",
  },
  invalid_turn_interval: {
    title: "시간 구간이 올바르지 않아요",
    detail: "시작은 0 이상이어야 하고, 종료는 시작보다 커야 합니다.",
  },
  invalid_turn_confidence: {
    title: "확신도 값이 범위를 벗어났어요",
    detail: "확신도는 0에서 1 사이여야 합니다.",
  },
  audio_source_unavailable: {
    title: "이 초안의 원본 음성을 찾지 못했어요",
    detail: "초안은 있지만 원본 녹음이 이 컴퓨터에 없습니다. 음성 없이 글로만 검수할 수 있습니다.",
  },
  audio_source_mismatch: {
    title: "원본 음성이 초안을 만들 때와 다릅니다",
    detail: "저장된 녹음이 초안이 기록한 것과 일치하지 않아 재생하지 않습니다. 원본 파일이 바뀌지 않았는지 확인해 주세요.",
  },
  audio_source_unreadable: {
    title: "원본 음성을 읽지 못했어요",
    detail: "WAV 파일로 열 수 없습니다. 파일이 손상됐는지 확인해 주세요.",
  },
  invalid_clip_range: {
    title: "재생할 구간이 올바르지 않아요",
    detail: "요청한 구간이 녹음 범위를 벗어났습니다. 초안을 다시 불러온 뒤 시도해 주세요.",
  },
  clip_too_long: {
    title: "한 번에 재생하기에 너무 긴 구간이에요",
    detail: "구간을 나눠서 들어 주세요.",
  },
  gold_already_exists: {
    title: "이 대화에는 이미 확정본이 있어요",
    detail: "확정본은 덮어쓸 수 없습니다. 기존 확정본을 그대로 두고 진행해 주세요.",
  },
  promotion_refused: {
    title: "확정 조건을 만족하지 못했어요",
    detail: "검수자, 확인 표시, 수정한 발화를 다시 확인한 뒤 시도해 주세요.",
  },
  silver_modified: {
    title: "Silver 초안이 도중에 바뀌었어요",
    detail: "이 검수 결과는 신뢰할 수 없습니다. 백엔드 로그를 확인한 뒤 다시 검수해 주세요.",
  },
  invalid_request: {
    title: "보낸 내용이 올바르지 않아요",
    detail: "입력한 값의 길이나 형식을 확인한 뒤 다시 시도해 주세요.",
  },
};

const FALLBACK: Omit<ErrorGuidance, "code"> = {
  title: "분석을 이어가지 못했어요",
  detail: "잠시 후 다시 시도해 주세요. 같은 문제가 반복되면 로컬 백엔드 로그를 확인해 주세요.",
};

export function describeError(error: unknown): ErrorGuidance {
  if (error instanceof ApiError) {
    return { code: error.code, ...(GUIDANCE[error.code] ?? FALLBACK) };
  }
  return { code: "unexpected_error", ...FALLBACK };
}

export function isMissingJob(error: unknown): boolean {
  return error instanceof ApiError && (error.status === 404 || error.code === "job_not_found");
}
