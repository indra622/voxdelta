import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../../api/client";
import type {
  AnnotationAudioOverview,
  AlignmentProposal,
  AnnotationReviewIndex,
  AnnotationReviewState,
  ApiClient,
  GoldPromotionResult,
  GeminiEmotionOverlayCandidate,
  ReferenceResegmentationCandidate,
  SilverReviewDraft,
} from "../../api/types";
import { SilverReview } from "./SilverReview";

const CONVERSATION = "A6000_S0005_0";
const TRANSCRIPT = "환불 처리가 아직도 안 됐습니다";

function state(overrides: Partial<AnnotationReviewState> = {}): AnnotationReviewState {
  return {
    conversation_id: CONVERSATION,
    review_state: "review_required",
    promotable: false,
    created_at: "2026-09-02T02:26:24+00:00",
    model: "gemini-3.7-flash",
    input_sha256: "0".repeat(64),
    content_sha256: "1".repeat(64),
    remote_file_deleted: true,
    turn_count: 2,
    speaker_count: 2,
    uncertain_turns: 1,
    mean_confidence: 0.61,
    gold_present: false,
    ...overrides,
  };
}

function draft(overrides: Partial<SilverReviewDraft> = {}): SilverReviewDraft {
  return {
    state: state(),
    speakers: ["SPEAKER_00", "SPEAKER_01"],
    notes: "겹쳐 말한 구간을 먼저 확인해 주세요.",
    turns: [
      {
        start: 0,
        end: 2.5,
        speaker: "SPEAKER_00",
        transcript: TRANSCRIPT,
        emotion: "anger",
        emotion_rationale: "높은 음높이",
        confidence: 0.62,
      },
      {
        start: 2.5,
        end: 5,
        speaker: "SPEAKER_01",
        transcript: "네 확인해 드리겠습니다",
        emotion: "uncertain",
        emotion_rationale: "확실하지 않음",
        confidence: 0.31,
      },
    ],
    emotion_labels: ["happiness", "anger", "neutral", "sadness", "uncertain"],
    warnings: [
      {
        code: "review_required",
        detail: "This draft is a model's provisional guess.",
        count: null,
        by_rule: null,
      },
      {
        code: "salvaged_dropped_turns",
        detail: "Turns were dropped during validation.",
        count: 31,
        by_rule: { "turn.exceeds_duration": 31 },
      },
    ],
    ...overrides,
  };
}

function receipt(overrides: Partial<GoldPromotionResult> = {}): GoldPromotionResult {
  return {
    conversation_id: CONVERSATION,
    reviewer: "reviewer@example.test",
    reviewed_at: "2026-09-02T04:00:00+00:00",
    review_note: "",
    content_sha256: "2".repeat(64),
    parent_silver_sha256: "1".repeat(64),
    unchanged_from_silver: false,
    turn_count: 2,
    speaker_count: 2,
    uncertain_turns: 1,
    mean_confidence: 0.7,
    change_reason_count: 0,
    silver_unmodified: true,
    ...overrides,
  };
}

function index(overrides: Partial<AnnotationReviewIndex> = {}): AnnotationReviewIndex {
  return { annotations: [state()], unreadable_count: 0, ...overrides };
}

function overview(overrides: Partial<AnnotationAudioOverview> = {}): AnnotationAudioOverview {
  return {
    conversation_id: CONVERSATION,
    duration_seconds: 30,
    sample_rate: 16000,
    max_clip_seconds: 120,
    min_gap_seconds: 0.5,
    turn_clips: [
      { position: 1, start: 0, end: 2 },
      { position: 2, start: 5, end: 7 },
    ],
    gaps: [{ start: 2, end: 5 }],
    ...overrides,
  };
}

function alignmentProposal(overrides: Partial<AlignmentProposal> = {}): AlignmentProposal {
  return {
    source_silver_content_sha256: "1".repeat(64),
    target_start_position: 2,
    target_end_position: 2,
    rows: [{ position: 2, start: 3, end: 5.5, confidence: 0.91 }],
    dropped_row_count: 0,
    dropped_rows_by_rule: {},
    ...overrides,
  };
}

function referenceCandidate(
  overrides: Partial<ReferenceResegmentationCandidate> = {},
): ReferenceResegmentationCandidate {
  return {
    source_silver_content_sha256: "1".repeat(64),
    source_reference_sha256: "2".repeat(64),
    reference_turn_count: 2,
    speakers: ["G5999", "G6000"],
    notes: "KCSC source reference candidate",
    turns: [
      {
        start: 10,
        end: 12,
        speaker: "G5999",
        transcript: "reference one",
        emotion: "uncertain",
        emotion_rationale: "reference candidate",
        confidence: 0,
      },
      {
        start: 12,
        end: 14,
        speaker: "G6000",
        transcript: "reference two",
        emotion: "uncertain",
        emotion_rationale: "reference candidate",
        confidence: 0,
      },
    ],
    ...overrides,
  };
}

function geminiEmotionCandidate(
  overrides: Partial<GeminiEmotionOverlayCandidate> = {},
): GeminiEmotionOverlayCandidate {
  return {
    source_silver_content_sha256: "1".repeat(64),
    model: "gemini-3.7-flash",
    remote_audio_transmitted: true,
    review_required: true,
    promotable: false,
    emotion_histogram: { sadness: 1, neutral: 1, uncertain: 0 },
    uncertain_turns: 0,
    mean_confidence: 0.55,
    turns: [
      {
        start: 0,
        end: 2.5,
        speaker: "SPEAKER_00",
        transcript: TRANSCRIPT,
        emotion: "sadness",
        emotion_rationale: "원격 판단 1",
        confidence: 0.6,
      },
      {
        start: 2.5,
        end: 5,
        speaker: "SPEAKER_01",
        transcript: "네 확인해 드리겠습니다",
        emotion: "neutral",
        emotion_rationale: "원격 판단 2",
        confidence: 0.5,
      },
    ],
    ...overrides,
  };
}

function reviewClient(overrides: Partial<ApiClient> = {}): ApiClient {
  return {
    getProviderConfiguration: vi.fn(),
    createJob: vi.fn(),
    getJob: vi.fn(),
    confirmRoles: vi.fn(),
    retryStage: vi.fn(),
    getReport: vi.fn(),
    deleteJob: vi.fn(),
    listSilverAnnotations: vi.fn().mockResolvedValue(index()),
    getSilverAnnotation: vi.fn().mockResolvedValue(draft()),
    getAlignmentProposal: vi.fn().mockResolvedValue(null),
    getAlignmentProposals: vi.fn().mockResolvedValue({ proposals: [] }),
    getReferenceResegmentationCandidate: vi.fn().mockResolvedValue(null),
    getAnnotationAudio: vi.fn().mockResolvedValue(overview()),
    getAnnotationClip: vi.fn().mockResolvedValue(new Blob([new Uint8Array(44)], { type: "audio/wav" })),
    promoteSilverToGold: vi.fn().mockResolvedValue(receipt()),
    ...overrides,
  };
}

async function openDraft(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("button", { name: new RegExp(CONVERSATION) }));
  await screen.findByLabelText("전사문", { selector: "#turn-0-transcript" });
}

async function signOff(user: ReturnType<typeof userEvent.setup>, reviewer = "reviewer@example.test") {
  await user.type(screen.getByLabelText(/검수자/), reviewer);
  await user.click(screen.getByRole("checkbox"));
}

/** A stand-in for the audio element, which jsdom has no implementation for.
 *
 * `play()` is a promise the test controls rather than one that resolves immediately, so a
 * clip can be observed while it is still loading and a second clip can be started before
 * the first has begun. */
class FakeAudio {
  static instances: FakeAudio[] = [];
  src = "";
  paused = true;
  pauseCount = 0;
  onended: (() => void) | null = null;
  onerror: (() => void) | null = null;
  playCalls = 0;
  playResult: Promise<void> = Promise.resolve();

  constructor() {
    FakeAudio.instances.push(this);
  }

  play(): Promise<void> {
    this.playCalls += 1;
    this.paused = false;
    return this.playResult;
  }

  pause(): void {
    this.paused = true;
    this.pauseCount += 1;
  }

  removeAttribute(): void {
    this.src = "";
  }
}

function installAudioDoubles() {
  FakeAudio.instances = [];
  const created: string[] = [];
  const revoked: string[] = [];
  vi.stubGlobal("Audio", FakeAudio);
  vi.stubGlobal("URL", {
    ...URL,
    createObjectURL: vi.fn((blob: Blob) => {
      const url = `blob:clip-${created.length}-${blob.size}`;
      created.push(url);
      return url;
    }),
    revokeObjectURL: vi.fn((url: string) => {
      revoked.push(url);
    }),
  });
  return { created, revoked };
}

const clipButton = (name: RegExp) => screen.findByRole("button", { name });

describe("Silver review mode", () => {
  let user: ReturnType<typeof userEvent.setup>;

  beforeEach(() => {
    user = userEvent.setup();
  });

  afterEach(() => {
    window.localStorage.clear();
    vi.unstubAllGlobals();
  });

  it("shows a loading state before the draft list arrives", async () => {
    let release: (value: AnnotationReviewIndex) => void = () => {};
    const pending = new Promise<AnnotationReviewIndex>((resolve) => {
      release = resolve;
    });
    render(<SilverReview client={reviewClient({ listSilverAnnotations: () => pending })} />);

    expect(screen.getByText(/초안 목록을 읽고 있습니다/)).toBeInTheDocument();
    release(index());
    expect(await screen.findByRole("button", { name: new RegExp(CONVERSATION) })).toBeInTheDocument();
  });

  it("says so plainly when this machine holds no draft", async () => {
    render(
      <SilverReview
        client={reviewClient({
          listSilverAnnotations: vi.fn().mockResolvedValue({ annotations: [], unreadable_count: 0 }),
        })}
      />,
    );

    expect(await screen.findByText("검수할 초안이 없습니다")).toBeInTheDocument();
  });

  it("offers a retry when the list cannot be read", async () => {
    const listSilverAnnotations = vi
      .fn()
      .mockRejectedValueOnce(
        new ApiError(0, { code: "backend_unreachable", message: "unreachable" }),
      )
      .mockResolvedValue(index());
    render(<SilverReview client={reviewClient({ listSilverAnnotations })} />);

    await user.click(await screen.findByRole("button", { name: "다시 시도" }));

    expect(await screen.findByRole("button", { name: new RegExp(CONVERSATION) })).toBeInTheDocument();
  });

  it("counts drafts it could not parse rather than hiding them", async () => {
    render(<SilverReview client={reviewClient({
      listSilverAnnotations: vi.fn().mockResolvedValue(index({ unreadable_count: 2 })),
    })} />);

    expect(await screen.findByText(/읽을 수 없는 초안 2건/)).toBeInTheDocument();
  });

  it("shows the dropped-turn warning with its count before anything is editable", async () => {
    render(<SilverReview client={reviewClient()} />);
    await openDraft(user);

    expect(screen.getByText(/검증에서 탈락한 발화 31개가 이 초안에 없습니다/)).toBeInTheDocument();
    expect(screen.getByText(/turn.exceeds_duration 31건/)).toBeInTheDocument();
    expect(screen.getByText(/이 초안은 아직 정답이 아닙니다/)).toBeInTheDocument();
  });

  it("lets the reviewer edit speaker, time, transcript, and emotion", async () => {
    const promoteSilverToGold = vi.fn().mockResolvedValue(receipt());
    render(<SilverReview client={reviewClient({ promoteSilverToGold })} />);
    await openDraft(user);

    const speaker = screen.getByLabelText("화자", { selector: "#turn-0-speaker" });
    await user.clear(speaker);
    await user.type(speaker, "고객");
    const start = screen.getByLabelText("시작 (초)", { selector: "#turn-0-start" });
    await user.clear(start);
    await user.type(start, "0.5");
    const transcript = screen.getByLabelText("전사문", { selector: "#turn-0-transcript" });
    await user.clear(transcript);
    await user.type(transcript, "환불 처리가 안 됐습니다");
    await user.selectOptions(
      screen.getByLabelText("감정", { selector: "#turn-0-emotion" }),
      "neutral",
    );
    await user.selectOptions(
      screen.getByLabelText(/수정 사유/, { selector: "#turn-0-review-reason" }),
      "emotion_mismatch",
    );

    await signOff(user);
    await user.click(screen.getByRole("button", { name: /검수 완료로 확정/ }));

    await waitFor(() => expect(promoteSilverToGold).toHaveBeenCalledTimes(1));
    const [conversationId, submission] = promoteSilverToGold.mock.calls[0]!;
    expect(conversationId).toBe(CONVERSATION);
    expect(submission.turns[0]).toMatchObject({
      speaker: "고객",
      start: 0.5,
      transcript: "환불 처리가 안 됐습니다",
      emotion: "neutral",
    });
    expect(submission.reviewer).toBe("reviewer@example.test");
    expect(submission.acknowledged).toBe(true);
    expect(submission.change_reasons).toEqual([{ position: 1, reason: "emotion_mismatch" }]);
  });

  it("counts the turns the reviewer actually changed", async () => {
    render(<SilverReview client={reviewClient()} />);
    await openDraft(user);

    expect(screen.getByText(/수정한 발화는 0개입니다/)).toBeInTheDocument();
    await user.selectOptions(
      screen.getByLabelText("감정", { selector: "#turn-1-emotion" }),
      "sadness",
    );
    expect(screen.getByText(/수정한 발화는 1개입니다/)).toBeInTheDocument();
  });

  it("requires a compact reason before a changed turn can become gold", async () => {
    const promoteSilverToGold = vi.fn();
    render(<SilverReview client={reviewClient({ promoteSilverToGold })} />);
    await openDraft(user);

    await user.selectOptions(
      screen.getByLabelText("감정", { selector: "#turn-0-emotion" }),
      "neutral",
    );
    await signOff(user);
    await user.click(screen.getByRole("button", { name: /검수 완료로 확정/ }));

    expect(promoteSilverToGold).not.toHaveBeenCalled();
    expect(screen.getByText(/수정한 발화 1개에 사유를 선택/)).toBeInTheDocument();
  });

  it("applies a matching timing proposal only after the reviewer clicks it", async () => {
    render(
      <SilverReview
        client={reviewClient({
          getAlignmentProposals: vi.fn().mockResolvedValue({ proposals: [alignmentProposal()] }),
        })}
      />,
    );
    await openDraft(user);

    expect(await screen.findByText(/2~2번 시간 정렬 제안 1개/)).toBeInTheDocument();
    expect(screen.getByLabelText("시작 (초)", { selector: "#turn-1-start" })).toHaveValue("2.5");
    expect(window.localStorage).toHaveLength(0);

    await user.click(screen.getByRole("button", { name: "2~2번 시간 제안 적용" }));

    expect(screen.getByLabelText("시작 (초)", { selector: "#turn-0-start" })).toHaveValue("0");
    expect(screen.getByLabelText("시작 (초)", { selector: "#turn-1-start" })).toHaveValue("3");
    expect(screen.getByLabelText("종료 (초)", { selector: "#turn-1-end" })).toHaveValue("5.5");
    expect(screen.getByLabelText("전사문", { selector: "#turn-1-transcript" })).toHaveValue(
      "네 확인해 드리겠습니다",
    );
    expect(window.localStorage).toHaveLength(0);
    expect(screen.getByText(/편집 화면에만 반영했습니다/)).toBeInTheDocument();
  });

  it("keeps a local checkpoint before replacing only the working copy with a KCSC reference candidate", async () => {
    render(
      <SilverReview
        client={reviewClient({
          getReferenceResegmentationCandidate: vi.fn().mockResolvedValue(referenceCandidate()),
        })}
      />,
    );
    await openDraft(user);

    await user.clear(screen.getByLabelText("전사문", { selector: "#turn-0-transcript" }));
    await user.type(screen.getByLabelText("전사문", { selector: "#turn-0-transcript" }), "working copy");
    await user.click(screen.getByRole("button", { name: "KCSC 재분할 후보를 작업본에 적용" }));

    expect(screen.getByDisplayValue("reference one")).toBeInTheDocument();
    expect(screen.getByDisplayValue("reference two")).toBeInTheDocument();
    expect(screen.queryByDisplayValue("working copy")).not.toBeInTheDocument();
    expect(screen.getByText(/현재 작업본을 임시 저장한 뒤 KCSC 기준 재분할 후보 2개/)).toBeInTheDocument();
    expect(window.localStorage.getItem(`voxdelta.review-checkpoint.v1:${CONVERSATION}:${"1".repeat(64)}`)).toContain(
      "working copy",
    );
  });

  it("marks the Gemini candidate as remote and offers it without promoting anything", async () => {
    const promoteSilverToGold = vi.fn().mockResolvedValue(receipt());
    render(
      <SilverReview
        client={reviewClient({
          getGeminiEmotionOverlayCandidate: vi.fn().mockResolvedValue(geminiEmotionCandidate()),
          promoteSilverToGold,
        })}
      />,
    );
    await openDraft(user);

    const card = await screen.findByRole("status", { name: "Gemini 원격 감정 후보" });
    expect(within(card).getByText(/Gemini remote · 외부 전송됨/)).toBeInTheDocument();
    expect(within(card).getByText(/gemini-3.7-flash/)).toBeInTheDocument();
    expect(within(card).getByText(/원격 전송 있음/)).toBeInTheDocument();
    // Merely showing the candidate must never write anything.
    expect(promoteSilverToGold).not.toHaveBeenCalled();
  });

  it("keeps a checkpoint, then applies only emotion fields from the Gemini candidate", async () => {
    render(
      <SilverReview
        client={reviewClient({
          getGeminiEmotionOverlayCandidate: vi.fn().mockResolvedValue(geminiEmotionCandidate()),
        })}
      />,
    );
    await openDraft(user);

    await user.clear(screen.getByLabelText("전사문", { selector: "#turn-0-transcript" }));
    await user.type(screen.getByLabelText("전사문", { selector: "#turn-0-transcript" }), "working copy");
    await user.click(screen.getByRole("button", { name: "Gemini 원격 감정 후보를 작업본에 적용" }));

    // Emotion, rationale, and confidence change...
    expect(
      (screen.getByLabelText("감정", { selector: "#turn-0-emotion" }) as HTMLSelectElement).value,
    ).toBe("sadness");
    expect(
      (screen.getByLabelText("감정", { selector: "#turn-1-emotion" }) as HTMLSelectElement).value,
    ).toBe("neutral");
    // The rationale is shown to the reviewer as text, not as an editable field.
    expect(screen.getByText(/원격 판단 1/)).toBeInTheDocument();
    expect(screen.getByText(/원격 판단 2/)).toBeInTheDocument();
    expect(screen.getByDisplayValue("0.6")).toBeInTheDocument();
    // ...and nothing else does. The reviewer's own transcript edit survives.
    expect(screen.getByDisplayValue("working copy")).toBeInTheDocument();
    expect(
      (screen.getByLabelText("화자", { selector: "#turn-0-speaker" }) as HTMLInputElement).value,
    ).toBe("SPEAKER_00");
    expect(screen.getByText(/Gemini 원격 감정 후보 2개/)).toBeInTheDocument();
    expect(
      window.localStorage.getItem(`voxdelta.review-checkpoint.v1:${CONVERSATION}:${"1".repeat(64)}`),
    ).toContain("working copy");
  });

  it("ignores a Gemini candidate bound to a different Silver digest", async () => {
    render(
      <SilverReview
        client={reviewClient({
          getGeminiEmotionOverlayCandidate: vi
            .fn()
            .mockResolvedValue(
              geminiEmotionCandidate({ source_silver_content_sha256: "9".repeat(64) }),
            ),
        })}
      />,
    );
    await openDraft(user);

    expect(
      screen.queryByRole("button", { name: "Gemini 원격 감정 후보를 작업본에 적용" }),
    ).not.toBeInTheDocument();
  });

  for (const [label, overrides] of [
    ["promotable", { promotable: true }],
    ["already reviewed", { review_required: false }],
  ] as const) {
    it(`ignores a Gemini candidate that claims to be ${label}`, async () => {
      render(
        <SilverReview
          client={reviewClient({
            getGeminiEmotionOverlayCandidate: vi
              .fn()
              .mockResolvedValue(geminiEmotionCandidate(overrides)),
          })}
        />,
      );
      await openDraft(user);

      expect(
        screen.queryByRole("button", { name: "Gemini 원격 감정 후보를 작업본에 적용" }),
      ).not.toBeInTheDocument();
    });
  }

  it("opens the draft normally when the Gemini candidate route is unavailable", async () => {
    const getGeminiEmotionOverlayCandidate = vi
      .fn()
      .mockRejectedValue(new ApiError(404, { code: "not_found", message: "no candidate" }));
    render(<SilverReview client={reviewClient({ getGeminiEmotionOverlayCandidate })} />);

    await openDraft(user);

    expect(getGeminiEmotionOverlayCandidate).toHaveBeenCalled();
    expect(screen.getByDisplayValue(TRANSCRIPT)).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Gemini 원격 감정 후보를 작업본에 적용" }),
    ).not.toBeInTheDocument();
  });

  it("opens the draft normally when the client has no Gemini candidate method at all", async () => {
    render(<SilverReview client={reviewClient()} />);

    await openDraft(user);

    expect(screen.getByDisplayValue(TRANSCRIPT)).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Gemini 원격 감정 후보를 작업본에 적용" }),
    ).not.toBeInTheDocument();
  });

  it("does not offer another reference repair for a clean reference Silver draft", async () => {
    const getReferenceResegmentationCandidate = vi.fn().mockResolvedValue(referenceCandidate());
    render(
      <SilverReview
        client={reviewClient({
          getSilverAnnotation: vi.fn().mockResolvedValue(
            draft({ state: state({ model: "kcsc-human-reference" }) }),
          ),
          getReferenceResegmentationCandidate,
        })}
      />,
    );
    await openDraft(user);

    expect(await screen.findByText("kcsc-human-reference")).toBeInTheDocument();
    expect(getReferenceResegmentationCandidate).not.toHaveBeenCalled();
    expect(screen.queryByLabelText("KCSC 기준 재분할 후보")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("9번 세밀 재분할 후보")).not.toBeInTheDocument();
  });

  it("splits only the ninth long turn with overlapping KCSC reference turns", async () => {
    const turns = Array.from({ length: 9 }, (_, index) => ({
      start: index === 8 ? 58.8 : index * 5,
      end: index === 8 ? 111.4 : index * 5 + 4,
      speaker: "SPEAKER_00",
      transcript: index === 8 ? "긴 아홉번째 발화" : `발화 ${index + 1}`,
      emotion: "neutral",
      emotion_rationale: "초안",
      confidence: 0.9,
    }));
    render(
      <SilverReview
        client={reviewClient({
          getSilverAnnotation: vi.fn().mockResolvedValue(draft({ turns })),
          getReferenceResegmentationCandidate: vi.fn().mockResolvedValue(
            referenceCandidate({
              reference_turn_count: 3,
              turns: [
                {
                  start: 59,
                  end: 60,
                  speaker: "G5999",
                  transcript: "아홉 구간 하나",
                  emotion: "uncertain",
                  emotion_rationale: "reference candidate",
                  confidence: 0,
                },
                {
                  start: 61,
                  end: 63,
                  speaker: "G6000",
                  transcript: "아홉 구간 둘",
                  emotion: "uncertain",
                  emotion_rationale: "reference candidate",
                  confidence: 0,
                },
                {
                  start: 120,
                  end: 121,
                  speaker: "G6000",
                  transcript: "범위 밖",
                  emotion: "uncertain",
                  emotion_rationale: "reference candidate",
                  confidence: 0,
                },
              ],
            }),
          ),
        })}
      />,
    );
    await openDraft(user);

    expect(await screen.findByText(/9번 긴 구간 세밀 재분할 후보 2개/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "9번을 KCSC 기준으로 세분화" }));

    expect(screen.getByDisplayValue("아홉 구간 하나")).toBeInTheDocument();
    expect(screen.getByDisplayValue("아홉 구간 둘")).toBeInTheDocument();
    expect(screen.queryByDisplayValue("긴 아홉번째 발화")).not.toBeInTheDocument();
    expect(screen.queryByDisplayValue("범위 밖")).not.toBeInTheDocument();
    expect(screen.getByText(/9번 긴 구간을 KCSC 기준 2개 발화로 세분화/)).toBeInTheDocument();
    expect(window.localStorage.getItem(`voxdelta.review-checkpoint.v1:${CONVERSATION}:${"1".repeat(64)}`)).toContain(
      "긴 아홉번째 발화",
    );
  });

  it("keeps an explicit browser-local checkpoint across a reload without promoting gold", async () => {
    const first = render(<SilverReview client={reviewClient()} />);
    await openDraft(user);
    const transcript = screen.getByLabelText("전사문", { selector: "#turn-0-transcript" });
    await user.clear(transcript);
    await user.type(transcript, "중간 검수 저장본");
    await user.click(screen.getByRole("button", { name: "임시 저장" }));
    expect(await screen.findByText(/이 브라우저에 .* 저장됨/)).toBeInTheDocument();
    first.unmount();

    render(<SilverReview client={reviewClient()} />);
    await openDraft(user);
    expect(screen.getByLabelText("전사문", { selector: "#turn-0-transcript" })).toHaveValue(
      "중간 검수 저장본",
    );
    expect(screen.queryByText("확정본으로 기록했습니다")).not.toBeInTheDocument();
  });

  it("does not promote without a reviewer identity", async () => {
    const promoteSilverToGold = vi.fn();
    render(<SilverReview client={reviewClient({ promoteSilverToGold })} />);
    await openDraft(user);

    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: /검수 완료로 확정/ }));

    expect(promoteSilverToGold).not.toHaveBeenCalled();
    expect(screen.getByText("확정본에 남길 검수자 식별자를 입력해 주세요.")).toBeInTheDocument();
  });

  it("does not promote a blank reviewer identity", async () => {
    const promoteSilverToGold = vi.fn();
    render(<SilverReview client={reviewClient({ promoteSilverToGold })} />);
    await openDraft(user);

    await user.type(screen.getByLabelText(/검수자/), "   ");
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: /검수 완료로 확정/ }));

    expect(promoteSilverToGold).not.toHaveBeenCalled();
  });

  it("does not promote until the reviewer acknowledges the draft", async () => {
    const promoteSilverToGold = vi.fn();
    render(<SilverReview client={reviewClient({ promoteSilverToGold })} />);
    await openDraft(user);

    await user.type(screen.getByLabelText(/검수자/), "reviewer@example.test");
    await user.click(screen.getByRole("button", { name: /검수 완료로 확정/ }));

    expect(promoteSilverToGold).not.toHaveBeenCalled();
    expect(screen.getByText("확인했다는 표시를 체크해 주세요.")).toBeInTheDocument();
  });

  it("does not send a turn the backend would refuse", async () => {
    const promoteSilverToGold = vi.fn();
    render(<SilverReview client={reviewClient({ promoteSilverToGold })} />);
    await openDraft(user);

    const end = screen.getByLabelText("종료 (초)", { selector: "#turn-0-end" });
    await user.clear(end);
    await user.type(end, "0");
    await signOff(user);
    await user.click(screen.getByRole("button", { name: /검수 완료로 확정/ }));

    expect(promoteSilverToGold).not.toHaveBeenCalled();
    expect(screen.getByText("종료 시각은 시작 시각보다 커야 합니다.")).toBeInTheDocument();
  });

  it("shows the completed gold state with its digest and the silver-untouched check", async () => {
    render(<SilverReview client={reviewClient()} />);
    await openDraft(user);
    await signOff(user);
    await user.click(screen.getByRole("button", { name: /검수 완료로 확정/ }));

    const done = await screen.findByRole("status");
    expect(within(done).getByText("확정본으로 기록했습니다")).toBeInTheDocument();
    expect(within(done).getByText(new RegExp("2".repeat(64)))).toBeInTheDocument();
    expect(within(done).getByText(/Silver 초안은 그대로 남아 있습니다/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /검수 완료로 확정/ })).not.toBeInTheDocument();
    expect(await screen.findByText("확정본 있음")).toBeInTheDocument();
  });

  it("says outright when the backend could not confirm silver was left alone", async () => {
    render(
      <SilverReview
        client={reviewClient({
          promoteSilverToGold: vi.fn().mockResolvedValue(receipt({ silver_unmodified: false })),
        })}
      />,
    );
    await openDraft(user);
    await signOff(user);
    await user.click(screen.getByRole("button", { name: /검수 완료로 확정/ }));

    expect(await screen.findByText(/Silver 초안이 바뀐 것으로 확인됐습니다/)).toBeInTheDocument();
  });

  it("explains a refused promotion in the reviewer's language and keeps the draft open", async () => {
    render(
      <SilverReview
        client={reviewClient({
          promoteSilverToGold: vi
            .fn()
            .mockRejectedValue(
              new ApiError(409, { code: "gold_already_exists", message: "immutable" }),
            ),
        })}
      />,
    );
    await openDraft(user);
    await signOff(user);
    await user.click(screen.getByRole("button", { name: /검수 완료로 확정/ }));

    expect(await screen.findByText("이 대화에는 이미 확정본이 있어요")).toBeInTheDocument();
    expect(screen.getByLabelText("전사문", { selector: "#turn-0-transcript" })).toBeInTheDocument();
  });

  it("refuses to offer promotion for a conversation that already has gold", async () => {
    render(
      <SilverReview
        client={reviewClient({
          listSilverAnnotations: vi
            .fn()
            .mockResolvedValue({ annotations: [state({ gold_present: true })], unreadable_count: 0 }),
          getSilverAnnotation: vi.fn().mockResolvedValue(
            draft({
              state: state({ gold_present: true }),
              warnings: [
                {
                  code: "gold_already_exists",
                  detail: "A reviewed gold annotation already exists.",
                  count: null,
                  by_rule: null,
                },
              ],
            }),
          ),
        })}
      />,
    );
    await openDraft(user);

    expect(screen.getByText("이 대화에는 이미 확정본이 있습니다")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /검수 완료로 확정/ })).toBeDisabled();
    expect(screen.getByLabelText("전사문", { selector: "#turn-0-transcript" })).toBeDisabled();
  });

  it("surfaces a draft that cannot be read without losing the list", async () => {
    render(
      <SilverReview
        client={reviewClient({
          getSilverAnnotation: vi
            .fn()
            .mockRejectedValue(
              new ApiError(409, { code: "annotation_unreadable", message: "unreadable" }),
            ),
        })}
      />,
    );
    await user.click(await screen.findByRole("button", { name: new RegExp(CONVERSATION) }));

    expect(await screen.findByText("저장된 Silver 초안을 읽지 못했어요")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: new RegExp(CONVERSATION) })).toBeInTheDocument();
  });

  it("does not carry an identity or an acknowledgement across drafts", async () => {
    const second = state({ conversation_id: "B0001_S0001_0" });
    render(
      <SilverReview
        client={reviewClient({
          listSilverAnnotations: vi
            .fn()
            .mockResolvedValue({ annotations: [state(), second], unreadable_count: 0 }),
        })}
      />,
    );
    await openDraft(user);
    await signOff(user);

    await user.click(screen.getByRole("button", { name: /B0001_S0001_0/ }));
    await screen.findByLabelText("전사문", { selector: "#turn-0-transcript" });

    expect(screen.getByLabelText(/검수자/)).toHaveValue("");
    expect(screen.getByRole("checkbox")).not.toBeChecked();
  });

  describe("listening to the audio behind the draft", () => {
    it("keeps unclaimed audio collapsed until the reviewer chooses to inspect it", async () => {
      render(<SilverReview client={reviewClient()} />);
      await openDraft(user);

      const section = await screen.findByRole("region", {
        name: /초안에 포함되지 않은 음성 구간/,
      });
      expect(section.querySelector("details")).not.toHaveAttribute("open");
      expect(within(section).getByText("1곳")).toBeInTheDocument();
    });

    it("plays one turn at the current review range, not a stale Silver range", async () => {
      installAudioDoubles();
      const client = reviewClient();
      render(<SilverReview client={client} />);
      await openDraft(user);

      await user.click(await clipButton(/1번 발화.*듣기/));

      await waitFor(() => expect(FakeAudio.instances[0]?.playCalls).toBe(1));
      expect(client.getAnnotationClip).toHaveBeenCalledWith(CONVERSATION, 0, 2.5);
    });

    it("uses a time correction immediately when the reviewer listens again", async () => {
      installAudioDoubles();
      const client = reviewClient();
      render(<SilverReview client={client} />);
      await openDraft(user);

      const start = screen.getByLabelText("시작 (초)", { selector: "#turn-0-start" });
      const end = screen.getByLabelText("종료 (초)", { selector: "#turn-0-end" });
      await user.clear(start);
      await user.type(start, "8");
      await user.clear(end);
      await user.type(end, "10");
      await user.click(await clipButton(/1번 발화.*듣기/));

      await waitFor(() => expect(FakeAudio.instances[0]?.playCalls).toBe(1));
      expect(client.getAnnotationClip).toHaveBeenCalledWith(CONVERSATION, 8, 10);
    });

    it("plays a review gap from the section that lists it", async () => {
      installAudioDoubles();
      const client = reviewClient();
      render(<SilverReview client={client} />);
      await openDraft(user);

      const section = await screen.findByRole("region", {
        name: /초안에 포함되지 않은 음성 구간/,
      });
      const disclosure = section.querySelector("summary");
      expect(disclosure).not.toBeNull();
      await user.click(disclosure!);
      await user.click(await within(section).findByRole("button", { name: /구간.*듣기/ }));

      await waitFor(() => expect(FakeAudio.instances[0]?.playCalls).toBe(1));
      expect(client.getAnnotationClip).toHaveBeenCalledWith(CONVERSATION, 2, 5);
    });

    it("names the gaps as unclaimed audio rather than as the dropped turns", async () => {
      installAudioDoubles();
      render(<SilverReview client={reviewClient()} />);
      await openDraft(user);

      const section = await screen.findByRole("region", {
        name: /초안에 포함되지 않은 음성 구간/,
      });
      expect(within(section).getByText(/아무 말도 하지 않는 구간/)).toBeInTheDocument();
      expect(within(section).getByText("00:02 - 00:05")).toBeInTheDocument();
    });

    it("shows the playing state on the control that started it", async () => {
      installAudioDoubles();
      render(<SilverReview client={reviewClient()} />);
      await openDraft(user);

      const first = await clipButton(/1번 발화.*듣기/);
      expect(first).toHaveAttribute("aria-pressed", "false");

      await user.click(first);

      await waitFor(() => expect(first).toHaveAttribute("aria-pressed", "true"));
      expect(within(first).getByText("정지")).toBeInTheDocument();
      // Only the control that started it: a second turn covering nearby seconds stays idle.
      expect(await clipButton(/2번 발화.*듣기/)).toHaveAttribute("aria-pressed", "false");
    });

    it("stops the previous clip when another one starts", async () => {
      installAudioDoubles();
      render(<SilverReview client={reviewClient()} />);
      await openDraft(user);

      await user.click(await clipButton(/1번 발화.*듣기/));
      await waitFor(() => expect(FakeAudio.instances[0]?.playCalls).toBe(1));
      const element = FakeAudio.instances[0];

      await user.click(await clipButton(/2번 발화.*듣기/));

      await waitFor(() => expect(element?.pauseCount).toBeGreaterThan(0));
      expect(await clipButton(/1번 발화.*듣기/)).toHaveAttribute("aria-pressed", "false");
      await waitFor(() =>
        expect(clipButton(/2번 발화/).then((node) => node.getAttribute("aria-pressed"))).resolves.toBe(
          "true",
        ),
      );
    });

    it("stops what it started when the same control is pressed again", async () => {
      installAudioDoubles();
      render(<SilverReview client={reviewClient()} />);
      await openDraft(user);

      const control = await clipButton(/1번 발화.*듣기/);
      await user.click(control);
      await waitFor(() => expect(control).toHaveAttribute("aria-pressed", "true"));

      await user.click(control);

      expect(control).toHaveAttribute("aria-pressed", "false");
      expect(FakeAudio.instances[0]?.paused).toBe(true);
    });

    it("releases the clip's object URL once it has finished playing", async () => {
      const { created, revoked } = installAudioDoubles();
      render(<SilverReview client={reviewClient()} />);
      await openDraft(user);

      await user.click(await clipButton(/1번 발화.*듣기/));
      await waitFor(() => expect(created).toHaveLength(1));

      FakeAudio.instances[0]?.onended?.();

      await waitFor(() => expect(revoked).toEqual(created));
    });

    it("shows a visible error when a clip cannot be fetched", async () => {
      installAudioDoubles();
      const client = reviewClient({
        getAnnotationClip: vi
          .fn()
          .mockRejectedValue(
            new ApiError(409, {
              code: "audio_source_mismatch",
              message: "The stored recording does not match.",
            }),
          ),
      });
      render(<SilverReview client={client} />);
      await openDraft(user);

      await user.click(await clipButton(/1번 발화.*듣기/));

      const alert = await screen.findByRole("alert");
      expect(alert).toHaveTextContent(/재생|음성|구간/);
      expect(await clipButton(/1번 발화.*듣기/)).toHaveAttribute("aria-pressed", "false");
    });

    it("keeps a failed clip's error on its own control", async () => {
      installAudioDoubles();
      const client = reviewClient({
        getAnnotationClip: vi.fn().mockRejectedValue(new ApiError(0, {
          code: "backend_unreachable",
          message: "unreachable",
        })),
      });
      render(<SilverReview client={client} />);
      await openDraft(user);

      await user.click(await clipButton(/1번 발화.*듣기/));
      await screen.findByRole("alert");

      const second = await clipButton(/2번 발화.*듣기/);
      expect(within(second).queryByText("!")).not.toBeInTheDocument();
    });

    it("says the draft is still reviewable when the recording is missing", async () => {
      installAudioDoubles();
      const client = reviewClient({
        getAnnotationAudio: vi.fn().mockRejectedValue(
          new ApiError(409, {
            code: "audio_source_unavailable",
            message: "The source recording is not available.",
          }),
        ),
      });
      render(<SilverReview client={client} />);
      await openDraft(user);

      expect(await screen.findByText(/음성 없이 글로만 검수할 수 있습니다/)).toBeInTheDocument();
      // No control is offered rather than a dead one being shown.
      expect(screen.queryByRole("button", { name: /1번 발화.*듣기/ })).not.toBeInTheDocument();
      // The draft itself is still fully editable.
      expect(screen.getByDisplayValue(TRANSCRIPT)).toBeInTheDocument();
    });

    it("never puts the capability token or a clip URL in the page", async () => {
      installAudioDoubles();
      render(<SilverReview client={reviewClient()} />);
      await openDraft(user);
      await user.click(await clipButton(/1번 발화.*듣기/));
      await waitFor(() => expect(FakeAudio.instances[0]?.playCalls).toBe(1));

      // The clip is fetched and played from a blob; nothing addressable is rendered.
      expect(document.body.innerHTML).not.toContain("/audio/clip");
      expect(document.querySelector("audio")).toBeNull();
    });
  });
});
