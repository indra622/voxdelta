import type {
  AnalysisReport,
  AlignmentProposal,
  AlignmentProposalCollection,
  AnnotationAudioOverview,
  AnnotationReviewIndex,
  ApiClient,
  GoldPromotionResult,
  ProviderConfiguration,
  PublicError,
  PublicJob,
  SilverReviewDraft,
  ReferenceResegmentationCandidate,
  EmotionOverlayCandidate,
  ExpertGuidanceStatus,
  GeminiEmotionOverlayCandidate,
  StageName,
} from "./types";

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(status: number, error: PublicError) {
    super(error.message);
    this.name = "ApiError";
    this.code = error.code;
    this.status = status;
  }
}

async function request<T>(fetcher: typeof fetch, url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetcher(url, init);
  } catch {
    // A transport failure never carries a public error body; name it so the UI can
    // point at the launcher instead of blaming the recording.
    throw new ApiError(0, {
      code: "backend_unreachable",
      message: "로컬 백엔드에 연결하지 못했습니다.",
    });
  }
  if (!response.ok) {
    let detail: Partial<PublicError> = {};
    try {
      const body = (await response.json()) as { detail?: Partial<PublicError> };
      detail = body.detail ?? {};
    } catch {
      // Keep public fallback copy when the response is not JSON.
    }
    throw new ApiError(response.status, {
      code: detail.code ?? "request_failed",
      message: detail.message ?? "요청을 완료하지 못했습니다.",
    });
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function requestBlob(fetcher: typeof fetch, url: string): Promise<Blob> {
  let response: Response;
  try {
    response = await fetcher(url);
  } catch {
    throw new ApiError(0, {
      code: "backend_unreachable",
      message: "로컬 백엔드에 연결하지 못했습니다.",
    });
  }
  if (!response.ok) {
    let detail: Partial<PublicError> = {};
    try {
      const body = (await response.json()) as { detail?: Partial<PublicError> };
      detail = body.detail ?? {};
    } catch {
      // An audio route can fail before it has a JSON body to fail with.
    }
    throw new ApiError(response.status, {
      code: detail.code ?? "request_failed",
      message: detail.message ?? "요청을 완료하지 못했습니다.",
    });
  }
  return await response.blob();
}

export function createApiClient(fetcher: typeof fetch = fetch): ApiClient {
  return {
    async getProviderConfiguration() {
      return request<ProviderConfiguration>(fetcher, "/api/config/providers");
    },
    async createJob(file) {
      const body = new FormData();
      body.append("file", file);
      body.append("diagnostic_capture", "false");
      return request(fetcher, "/api/jobs", { method: "POST", body });
    },
    async getJob(jobId) {
      return request<PublicJob>(fetcher, "/api/jobs/" + encodeURIComponent(jobId));
    },
    async getRoleSampleClip(jobId, speakerId, sampleIndex) {
      return requestBlob(
        fetcher,
        "/api/jobs/" +
          encodeURIComponent(jobId) +
          "/role-samples/" +
          encodeURIComponent(speakerId) +
          "/" +
          encodeURIComponent(String(sampleIndex)) +
          "/audio",
      );
    },
    async confirmRoles(jobId, mapping) {
      return request<PublicJob>(fetcher, "/api/jobs/" + encodeURIComponent(jobId) + "/roles", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ mapping }),
      });
    },
    async retryStage(jobId, stage: StageName) {
      return request<PublicJob>(fetcher, "/api/jobs/" + encodeURIComponent(jobId) + "/retry", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ stage }),
      });
    },
    async getReport(jobId) {
      return request<AnalysisReport>(fetcher, "/api/jobs/" + encodeURIComponent(jobId) + "/report");
    },
    async getExpertGuidance(jobId) {
      return request<ExpertGuidanceStatus>(
        fetcher,
        "/api/jobs/" + encodeURIComponent(jobId) + "/expert-guidance",
      );
    },
    async requestExpertGuidance(jobId, expertRequest) {
      return request<ExpertGuidanceStatus>(
        fetcher,
        "/api/jobs/" + encodeURIComponent(jobId) + "/expert-guidance/request",
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(expertRequest),
        },
      );
    },
    async deleteJob(jobId) {
      await request<void>(fetcher, "/api/jobs/" + encodeURIComponent(jobId), { method: "DELETE" });
    },
    async listSilverAnnotations() {
      return request<AnnotationReviewIndex>(fetcher, "/api/annotations");
    },
    async getSilverAnnotation(conversationId) {
      return request<SilverReviewDraft>(
        fetcher,
        "/api/annotations/" + encodeURIComponent(conversationId),
      );
    },
    async getAlignmentProposal(conversationId) {
      return request<AlignmentProposal | null>(
        fetcher,
        "/api/annotations/" + encodeURIComponent(conversationId) + "/alignment-proposal",
      );
    },
    async getAlignmentProposals(conversationId) {
      return request<AlignmentProposalCollection>(
        fetcher,
        "/api/annotations/" + encodeURIComponent(conversationId) + "/alignment-proposals",
      );
    },
    async getReferenceResegmentationCandidate(conversationId) {
      return request<ReferenceResegmentationCandidate | null>(
        fetcher,
        "/api/annotations/" + encodeURIComponent(conversationId) + "/reference-resegmentation-candidate",
      );
    },
    async getEmotionOverlayCandidate(conversationId) {
      return request<EmotionOverlayCandidate | null>(
        fetcher,
        "/api/annotations/" + encodeURIComponent(conversationId) + "/emotion-overlay-candidate",
      );
    },
    async getGeminiEmotionOverlayCandidate(conversationId) {
      return request<GeminiEmotionOverlayCandidate | null>(
        fetcher,
        "/api/annotations/" +
          encodeURIComponent(conversationId) +
          "/gemini-emotion-overlay-candidate",
      );
    },
    async getAnnotationAudio(conversationId) {
      return request<AnnotationAudioOverview>(
        fetcher,
        "/api/annotations/" + encodeURIComponent(conversationId) + "/audio",
      );
    },
    async getAnnotationClip(conversationId, start, end) {
      const range = new URLSearchParams({ start: String(start), end: String(end) });
      return requestBlob(
        fetcher,
        "/api/annotations/" + encodeURIComponent(conversationId) + "/audio/clip?" + range,
      );
    },
    async promoteSilverToGold(conversationId, submission) {
      return request<GoldPromotionResult>(
        fetcher,
        "/api/annotations/" + encodeURIComponent(conversationId) + "/gold",
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(submission),
        },
      );
    },
  };
}

export const api = createApiClient();
