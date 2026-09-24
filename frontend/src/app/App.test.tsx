import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import type { AnalysisReport, ApiClient, ProviderConfiguration, PublicJob, StageName, StageStatus } from "../api/types";
import { STAGE_NAMES } from "../api/types";
import { App } from "./App";

function stages(overrides: Partial<Record<StageName, StageStatus>> = {}, fallback: StageStatus = "completed") {
  return Object.fromEntries(
    STAGE_NAMES.map((name) => [name, { status: overrides[name] ?? fallback, error: null }]),
  ) as PublicJob["stages"];
}

function job(overrides: Partial<PublicJob> = {}): PublicJob {
  return {
    job_id: "j1",
    status: "completed",
    diagnostic_capture: false,
    created_at: "2026-08-27T00:00:00Z",
    updated_at: "2026-08-27T00:00:09Z",
    stages: stages(),
    role_candidate: null,
    ...overrides,
  };
}

const provider = {
  name: "wav2vec-xls-r",
  model: "xls-r-300m-calibrated",
  remote: false,
  transmits: [] as Array<"audio" | "text" | "features">,
  retention_policy_url: null,
  schema_version: "1",
  revision: "poc-v1",
};

const report: AnalysisReport = {
  job_id: "j1",
  summary: {
    start_state: "dissatisfied",
    end_state: "uncertain",
    peak_customer_utterance_id: "u1",
    overall_delta: -0.1,
    valid_coverage: 0.5,
    recovery_count: 0,
    worsening_count: 0,
    narrative: null,
  },
  utterances: [
    { id: "u1", start: 3, end: 6, speaker_id: "SPEAKER_00", role: "customer", transcript: "배송이 아직 안 왔어요.", overlap: false, confidence: 0.9 },
  ],
  emotions: [
    {
      utterance_id: "u1",
      probabilities: { happiness: 0.05, anger: 0.16, disgust: 0.08, fear: 0.13, neutral: 0.34, sadness: 0.14, surprise: 0.1 },
      operational_state: "uncertain",
      negative_intensity: 0.4,
      smoothed_negative_intensity: null,
      confidence: 0.36,
      provider,
      calibration: { calibration_id: "cal-v2", method: "temperature-scaling", temperature: 1.2, abstain_threshold: 0.55, abstained: true, raw_confidence: 0.44 },
    },
  ],
  strategies: [],
  transitions: [],
  warnings: [],
  transcription_coverage: null,
  schema_version: "1",
};

const localConfiguration: ProviderConfiguration = {
  stages: STAGE_NAMES.map((stage) => ({
    stage,
    provenance: { name: "local", model: "local", remote: false, schema_version: "1", revision: null },
    transmits: [],
    retention_policy_url: null,
    retention_window_hours: null,
  })),
};

/** The Precision-2 PoC configuration: diarization is the one stage that leaves the machine. */
const precisionConfiguration: ProviderConfiguration = {
  stages: localConfiguration.stages.map((disclosure) =>
    disclosure.stage === "diarize"
      ? {
          ...disclosure,
          provenance: { name: "pyannoteai", model: "precision-2", remote: true, schema_version: "1", revision: "precision-2" },
          transmits: ["audio"],
          retention_policy_url: "https://docs.pyannote.ai/data-retention",
          retention_window_hours: 48,
        }
      : disclosure,
  ),
};

function fakeClient(overrides: Partial<ApiClient> = {}): ApiClient {
  return {
    getProviderConfiguration: vi.fn().mockResolvedValue(localConfiguration),
    createJob: vi.fn().mockResolvedValue({ job_id: "j1", status_url: "/api/jobs/j1" }),
    getJob: vi.fn().mockResolvedValue(job()),
    confirmRoles: vi.fn().mockResolvedValue(job()),
    retryStage: vi.fn().mockResolvedValue(job()),
    getReport: vi.fn().mockResolvedValue(report),
    deleteJob: vi.fn().mockResolvedValue(undefined),
    getAnnotationAudio: vi.fn().mockRejectedValue(new Error("no audio in this fixture")),
    getAnnotationClip: vi.fn().mockRejectedValue(new Error("no audio in this fixture")),
    // The analysis journey never reaches these; they exist so the shape stays whole.
    listSilverAnnotations: vi.fn().mockResolvedValue({ annotations: [], unreadable_count: 0 }),
    getSilverAnnotation: vi.fn(),
    getAlignmentProposal: vi.fn().mockResolvedValue(null),
    getAlignmentProposals: vi.fn().mockResolvedValue({ proposals: [] }),
    getReferenceResegmentationCandidate: vi.fn().mockResolvedValue(null),
    promoteSilverToGold: vi.fn(),
    ...overrides,
  };
}

async function choose(user: ReturnType<typeof userEvent.setup>) {
  await user.upload(
    screen.getByLabelText("분석할 음성 파일"),
    new File([new Uint8Array([1, 2, 3])], "call.wav", { type: "audio/wav" }),
  );
  return screen.getByRole("button", { name: "감정 분석 시작" });
}

async function upload(user: ReturnType<typeof userEvent.setup>) {
  const start = await choose(user);
  // The submit control stays disabled until the transfer disclosure has been read back.
  await waitFor(() => expect(start).toBeEnabled());
  await user.click(start);
}

describe("PoC journey", () => {
  let user: ReturnType<typeof userEvent.setup>;

  beforeEach(() => {
    user = userEvent.setup();
  });

  it("confirms roles while paused, then reports the abstained result as uncertain", async () => {
    const paused = job({
      status: "paused",
      stages: stages({ confirm_roles: "paused", emotion: "pending", response_strategy: "pending", transitions: "pending", report: "pending" }),
      role_candidate: { speakers: ["SPEAKER_00", "SPEAKER_01"], suggested_mapping: { SPEAKER_00: "customer", SPEAKER_01: "agent" } },
    });
    const confirmRoles = vi.fn().mockResolvedValue(job());
    const client = fakeClient({ getJob: vi.fn().mockResolvedValue(paused), confirmRoles });

    render(<App client={client} />);
    await upload(user);

    const heading = await screen.findByRole("heading", { name: /역할을 확인해 주세요/ });
    expect(heading).toHaveFocus();
    expect(screen.getByText("SPEAKER_00")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "이 역할로 계속 분석" }));

    expect(confirmRoles).toHaveBeenCalledWith("j1", { SPEAKER_00: "customer", SPEAKER_01: "agent" });
    expect(await screen.findByRole("heading", { name: "분석이 끝났어요" })).toHaveFocus();
    // neutral is the highest probability here, and it must not become the verdict.
    expect(screen.getByRole("heading", { level: 3, name: "판단 불확실" })).toBeInTheDocument();
    expect(within(screen.getByRole("listitem")).getByRole("button")).toHaveTextContent("판단 불확실");
    expect(screen.getByText(/중립으로 해석하지 말고/)).toBeInTheDocument();
  });

  it("shows each diarized speaker's short transcript samples before roles are confirmed", async () => {
    const paused = job({
      status: "paused",
      stages: stages({ confirm_roles: "paused", emotion: "pending", response_strategy: "pending", transitions: "pending", report: "pending" }),
      role_candidate: {
        speakers: ["SPEAKER_00", "SPEAKER_01"],
        suggested_mapping: { SPEAKER_00: "customer", SPEAKER_01: "agent" },
        samples: {
          SPEAKER_00: [{ speaker_id: "SPEAKER_00", index: 0, start: 1.2, end: 5.8, transcript: "배송이 아직 안 왔어요.", clip_truncated: false }],
          SPEAKER_01: [{ speaker_id: "SPEAKER_01", index: 0, start: 6.1, end: 11.4, transcript: "확인해서 안내드리겠습니다.", clip_truncated: false }],
        },
      },
    });
    const client = fakeClient({
      getJob: vi.fn().mockResolvedValue(paused),
      getRoleSampleClip: vi.fn().mockResolvedValue(new Blob(["wav"])),
    });

    render(<App client={client} />);
    await upload(user);

    expect(await screen.findByRole("heading", { name: /대표 구간을 듣고 두 목소리를 비교해 주세요/ })).toBeInTheDocument();
    expect(screen.getByText(/배송이 아직 안 왔어요/)).toBeInTheDocument();
    expect(screen.getByText(/확인해서 안내드리겠습니다/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "SPEAKER_00 대표 구간 1 듣기" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "SPEAKER_01 대표 구간 1 듣기" })).toBeEnabled();
  });

  it("shows the failed stage instead of offering role confirmation again", async () => {
    // The backend accepted the mapping and then a later stage failed. The gate is closed,
    // so re-clicking it would only produce 409s against a decision already recorded.
    const paused = job({
      status: "paused",
      stages: stages({ confirm_roles: "paused", emotion: "pending", response_strategy: "pending", transitions: "pending", report: "pending" }),
      role_candidate: { speakers: ["SPEAKER_00", "SPEAKER_01"], suggested_mapping: { SPEAKER_00: "customer", SPEAKER_01: "agent" } },
    });
    const failed = job({
      status: "failed",
      stages: stages({ emotion: "failed", response_strategy: "pending", transitions: "pending", report: "pending" }),
      role_candidate: null,
    });
    const confirmRoles = vi.fn().mockResolvedValue(failed);
    const client = fakeClient({ getJob: vi.fn().mockResolvedValue(paused), confirmRoles });

    render(<App client={client} />);
    await upload(user);
    await user.click(await screen.findByRole("button", { name: "이 역할로 계속 분석" }));

    await waitFor(() =>
      expect(screen.queryByRole("heading", { name: /역할을 확인해 주세요/ })).not.toBeInTheDocument(),
    );
    expect(screen.queryByRole("button", { name: "이 역할로 계속 분석" })).not.toBeInTheDocument();
    expect(confirmRoles).toHaveBeenCalledTimes(1);
  });

  it("replaces its stale paused state when confirming roles fails outright", async () => {
    const paused = job({
      status: "paused",
      stages: stages({ confirm_roles: "paused", emotion: "pending", response_strategy: "pending", transitions: "pending", report: "pending" }),
      role_candidate: { speakers: ["SPEAKER_00", "SPEAKER_01"], suggested_mapping: { SPEAKER_00: "customer", SPEAKER_01: "agent" } },
    });
    const failed = job({
      status: "failed",
      stages: stages({ emotion: "failed", response_strategy: "pending", transitions: "pending", report: "pending" }),
      role_candidate: null,
    });
    // The request itself errors, but the mapping may already have been recorded, so the
    // screen must re-read the job rather than keep offering the gate.
    const confirmRoles = vi.fn().mockRejectedValue(new ApiError(500, { code: "pipeline_failed", message: "" }));
    const getJob = vi.fn().mockResolvedValueOnce(paused).mockResolvedValue(failed);
    const client = fakeClient({ getJob, confirmRoles });

    render(<App client={client} />);
    await upload(user);
    await user.click(await screen.findByRole("button", { name: "이 역할로 계속 분석" }));

    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "이 역할로 계속 분석" })).not.toBeInTheDocument(),
    );
    expect(confirmRoles).toHaveBeenCalledTimes(1);
  });

  it("swaps the suggested customer and agent before confirming", async () => {
    const paused = job({
      status: "paused",
      stages: stages({ confirm_roles: "paused", emotion: "pending", response_strategy: "pending", transitions: "pending", report: "pending" }),
      role_candidate: { speakers: ["SPEAKER_00", "SPEAKER_01"], suggested_mapping: { SPEAKER_00: "customer", SPEAKER_01: "agent" } },
    });
    const confirmRoles = vi.fn().mockResolvedValue(job());
    const client = fakeClient({ getJob: vi.fn().mockResolvedValue(paused), confirmRoles });

    render(<App client={client} />);
    await upload(user);

    await user.click(await screen.findByRole("button", { name: /역할 바꾸기/ }));
    await user.click(screen.getByRole("button", { name: "이 역할로 계속 분석" }));

    expect(confirmRoles).toHaveBeenCalledWith("j1", { SPEAKER_01: "customer", SPEAKER_00: "agent" });
  });

  it("refuses to submit a mapping the pipeline cannot accept", async () => {
    const paused = job({
      status: "paused",
      stages: stages({ confirm_roles: "paused", emotion: "pending", response_strategy: "pending", transitions: "pending", report: "pending" }),
      role_candidate: { speakers: ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"], suggested_mapping: null },
    });
    const confirmRoles = vi.fn();
    const client = fakeClient({ getJob: vi.fn().mockResolvedValue(paused), confirmRoles });

    render(<App client={client} />);
    await upload(user);

    expect(await screen.findByText(/화자가 3명으로 구분됐습니다/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "이 역할로 계속 분석" })).not.toBeInTheDocument();
    expect(confirmRoles).not.toHaveBeenCalled();
  });

  it("explains a failed stage in Korean and retries exactly that stage", async () => {
    const failed = job({
      status: "failed",
      stages: stages({ transcribe: "failed", confirm_roles: "pending", emotion: "pending", response_strategy: "pending", transitions: "pending", report: "pending" }, "completed"),
    });
    failed.stages.transcribe = { status: "failed", error: { code: "unsupported_speaker_count", message: "Exactly two observed speakers are required." } };
    const retryStage = vi.fn().mockResolvedValue(job());
    const client = fakeClient({ getJob: vi.fn().mockResolvedValue(failed), retryStage });

    render(<App client={client} />);
    await upload(user);

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText(/음성 전사 단계에서 분석이 멈췄습니다/)).toBeInTheDocument();
    expect(within(alert).getByText("unsupported_speaker_count")).toBeInTheDocument();

    await user.click(within(alert).getByRole("button", { name: /다시 시도/ }));
    expect(retryStage).toHaveBeenCalledWith("j1", "transcribe");
  });

  it("deletes the local job when the person starts over", async () => {
    const deleteJob = vi.fn().mockResolvedValue(undefined);
    const client = fakeClient({ deleteJob });

    render(<App client={client} />);
    await upload(user);
    await screen.findByRole("heading", { name: "분석이 끝났어요" });

    await user.click(screen.getByRole("button", { name: "새 음성 분석" }));

    expect(deleteJob).toHaveBeenCalledWith("j1");
    expect(await screen.findByRole("heading", { name: /어떤 목소리를/ })).toBeInTheDocument();
  });

  it("keeps the analysis on screen when local deletion fails", async () => {
    const deleteJob = vi.fn().mockRejectedValue(
      new ApiError(409, { code: "deletion_incomplete", message: "Local deletion did not finish." }),
    );
    const client = fakeClient({ deleteJob });

    render(<App client={client} />);
    await upload(user);
    await screen.findByRole("heading", { name: "분석이 끝났어요" });

    await user.click(screen.getByRole("button", { name: "새 음성 분석" }));

    expect(await screen.findByText("로컬 삭제가 끝나지 않았어요")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "분석이 끝났어요" })).toBeInTheDocument();
  });

  it("does not repeat the local-only promise when a provider transmits data", async () => {
    const client = fakeClient({
      getProviderConfiguration: vi.fn().mockResolvedValue({
        stages: [
          { stage: "emotion", provenance: { name: "hosted", model: "x", remote: true, schema_version: "1", revision: null }, transmits: ["audio"], retention_policy_url: null, retention_window_hours: null },
        ],
      } satisfies ProviderConfiguration),
    });

    render(<App client={client} />);

    expect(await screen.findByText(/로컬 전용이 아닙니다/)).toBeInTheDocument();
    expect(screen.getByText(/감정 분석\(오디오 → hosted 전송\)/)).toBeInTheDocument();
    expect(screen.queryByText(/외부로 전송되지 않습니다/)).not.toBeInTheDocument();
  });

  it("says the local-only claim is unverified when the disclosure cannot be read", async () => {
    const client = fakeClient({
      getProviderConfiguration: vi.fn().mockRejectedValue(new ApiError(0, { code: "backend_unreachable", message: "no" })),
    });

    render(<App client={client} />);

    expect(await screen.findByText(/백엔드 provider 설정을 읽지 못했습니다/)).toBeInTheDocument();
    expect(screen.queryByText(/외부로 전송되지 않습니다/)).not.toBeInTheDocument();
  });

  it("discloses the pyannoteAI transfer and its retention window before anything is sent", async () => {
    const createJob = vi.fn();
    const client = fakeClient({
      getProviderConfiguration: vi.fn().mockResolvedValue(precisionConfiguration),
      createJob,
    });

    render(<App client={client} />);

    const notice = await screen.findByRole("region", { name: /pyannoteAI/ });
    expect(within(notice).getByText(/화자 구분 단계 · pyannoteAI/)).toBeInTheDocument();
    expect(within(notice).getByText(/오디오 — 이 컴퓨터를 떠나 pyannoteAI API로 전송됩니다/)).toBeInTheDocument();
    expect(within(notice).getByText(/최대 48시간까지 남아 있을 수 있고/)).toBeInTheDocument();
    expect(within(notice).getByRole("link", { name: /보관 정책 원문/ })).toHaveAttribute(
      "href",
      "https://docs.pyannote.ai/data-retention",
    );

    const start = await choose(user);
    await waitFor(() => expect(start).toBeDisabled());
    await user.click(start);
    expect(createJob).not.toHaveBeenCalled();
  });

  it("uploads only after the transfer is explicitly acknowledged", async () => {
    const createJob = vi.fn().mockResolvedValue({ job_id: "j1", status_url: "/api/jobs/j1" });
    const client = fakeClient({
      getProviderConfiguration: vi.fn().mockResolvedValue(precisionConfiguration),
      createJob,
    });

    render(<App client={client} />);
    const start = await choose(user);
    await waitFor(() => expect(start).toBeDisabled());

    await user.click(await screen.findByRole("checkbox", { name: /pyannoteAI로 전송하는 데 동의합니다/ }));
    await waitFor(() => expect(start).toBeEnabled());
    await user.click(start);

    expect(createJob).toHaveBeenCalledTimes(1);
    expect(await screen.findByRole("heading", { name: "분석이 끝났어요" })).toBeInTheDocument();
    // The result screen must not settle back into a local-only claim after a remote stage ran.
    expect(screen.getByText(/로컬 전용이 아닙니다/)).toBeInTheDocument();
    expect(screen.getByText("감정 분석 로컬 실행")).toBeInTheDocument();
  });

  it("asks again for the next recording instead of remembering the acknowledgement", async () => {
    const client = fakeClient({
      getProviderConfiguration: vi.fn().mockResolvedValue(precisionConfiguration),
    });

    render(<App client={client} />);
    const start = await choose(user);
    await user.click(await screen.findByRole("checkbox", { name: /동의합니다/ }));
    await waitFor(() => expect(start).toBeEnabled());
    await user.click(start);
    await screen.findByRole("heading", { name: "분석이 끝났어요" });

    await user.click(screen.getByRole("button", { name: "새 음성 분석" }));

    expect(await screen.findByRole("checkbox", { name: /동의합니다/ })).not.toBeChecked();
    expect(screen.getByRole("button", { name: "감정 분석 시작" })).toBeDisabled();
  });

  it("holds submission back while the transfer disclosure is still unread", async () => {
    const createJob = vi.fn();
    const client = fakeClient({
      getProviderConfiguration: vi.fn().mockReturnValue(new Promise(() => {})),
      createJob,
    });

    render(<App client={client} />);
    const start = await choose(user);

    expect(start).toBeDisabled();
    expect(screen.getByText(/확인이 끝나야 분석을 시작할 수 있습니다/)).toBeInTheDocument();
    await user.click(start);
    expect(createJob).not.toHaveBeenCalled();
  });

  it("makes the person acknowledge an unverifiable disclosure before uploading", async () => {
    const createJob = vi.fn().mockResolvedValue({ job_id: "j1", status_url: "/api/jobs/j1" });
    const client = fakeClient({
      getProviderConfiguration: vi.fn().mockRejectedValue(new ApiError(0, { code: "backend_unreachable", message: "no" })),
      createJob,
    });

    render(<App client={client} />);
    const start = await choose(user);
    await waitFor(() => expect(start).toBeDisabled());
    expect(await screen.findByText(/오디오가 외부로 나가는지 확인하지 못했습니다/)).toBeInTheDocument();

    await user.click(screen.getByRole("checkbox", { name: /그래도 분석을 진행합니다/ }));
    await waitFor(() => expect(start).toBeEnabled());
    await user.click(start);

    expect(createJob).toHaveBeenCalledTimes(1);
  });

  it("offers a way out while a report cannot be loaded", async () => {
    const getReport = vi.fn()
      .mockRejectedValueOnce(new ApiError(409, { code: "report_not_ready", message: "not ready" }))
      .mockResolvedValue(report);
    const client = fakeClient({ getReport });

    render(<App client={client} />);
    await upload(user);

    expect(await screen.findByText("결과가 아직 준비되지 않았어요")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /결과 다시 불러오기/ }));

    await waitFor(() => expect(screen.getByRole("heading", { name: "분석이 끝났어요" })).toBeInTheDocument());
  });
});

describe("Silver review mode", () => {
  let user: ReturnType<typeof userEvent.setup>;

  beforeEach(() => {
    user = userEvent.setup();
  });

  it("opens review as a mode of the same app rather than replacing it", async () => {
    render(<App client={fakeClient()} />);

    await user.click(screen.getByRole("button", { name: "Silver 검수" }));

    expect(await screen.findByText("검수할 초안이 없습니다")).toBeInTheDocument();
    expect(screen.queryByLabelText("분석할 음성 파일")).not.toBeInTheDocument();
    // Same shell, same local-posture badge: this is one product with two screens.
    expect(screen.getByRole("link", { name: "VoxDelta 처음으로" })).toBeInTheDocument();
  });

  it("returns to the analysis screen with the flow intact", async () => {
    render(<App client={fakeClient()} />);
    await user.click(screen.getByRole("button", { name: "Silver 검수" }));
    await screen.findByText("검수할 초안이 없습니다");

    await user.click(screen.getByRole("button", { name: "분석" }));

    expect(screen.getByLabelText("분석할 음성 파일")).toBeInTheDocument();
    expect(screen.queryByText("검수할 초안이 없습니다")).not.toBeInTheDocument();
  });

  it("keeps a finished analysis while the reviewer is in the other mode", async () => {
    render(<App client={fakeClient()} />);
    await upload(user);
    await waitFor(() => expect(screen.getByRole("heading", { name: "분석이 끝났어요" })).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "Silver 검수" }));
    expect(screen.queryByRole("heading", { name: "분석이 끝났어요" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "분석" }));
    expect(screen.getByRole("heading", { name: "분석이 끝났어요" })).toBeInTheDocument();
  });

  it("does not read any annotation draft while the analysis screen is open", async () => {
    const listSilverAnnotations = vi.fn().mockResolvedValue({ annotations: [], unreadable_count: 0 });
    render(<App client={fakeClient({ listSilverAnnotations })} />);
    await upload(user);

    expect(listSilverAnnotations).not.toHaveBeenCalled();
  });
});
