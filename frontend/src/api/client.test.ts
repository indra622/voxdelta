import { describe, expect, it, vi } from "vitest";
import { createApiClient } from "./client";

describe("VoxDelta API client", () => {
  it("uploads the selected file without enabling diagnostic capture", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ job_id: "j1", status_url: "/api/jobs/j1" }), { status: 202 }),
    );
    const client = createApiClient(fetcher);
    const file = new File([new Uint8Array([1, 2, 3])], "voice.wav", { type: "audio/wav" });

    await client.createJob(file);

    const [url, init] = fetcher.mock.calls[0];
    expect(url).toBe("/api/jobs");
    expect(init?.method).toBe("POST");
    expect((init?.body as FormData).get("file")).toBe(file);
    expect((init?.body as FormData).get("diagnostic_capture")).toBe("false");
  });

  it("reads the silver draft list from the capability-fenced route", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ annotations: [], unreadable_count: 0 }), { status: 200 }),
    );

    await createApiClient(fetcher).listSilverAnnotations();

    expect(fetcher.mock.calls[0][0]).toBe("/api/annotations");
  });

  it("escapes the conversation id rather than pasting it into the path", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({}), { status: 200 }),
    );

    await createApiClient(fetcher).getSilverAnnotation("../../etc/passwd");

    expect(fetcher.mock.calls[0][0]).toBe("/api/annotations/..%2F..%2Fetc%2Fpasswd");
  });

  it("reads a timing-only proposal from the capability-fenced route", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response("null", { status: 200 }),
    );

    await createApiClient(fetcher).getAlignmentProposal("A6000_S0005_0");

    expect(fetcher.mock.calls[0][0]).toBe("/api/annotations/A6000_S0005_0/alignment-proposal");
  });

  it("reads all independent timing suggestions from the capability-fenced route", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ proposals: [] }), { status: 200 }),
    );

    await createApiClient(fetcher).getAlignmentProposals("A6000_S0005_0");

    expect(fetcher.mock.calls[0][0]).toBe("/api/annotations/A6000_S0005_0/alignment-proposals");
  });

  it("reads the optional KCSC reference resegmentation candidate from the capability-fenced route", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response("null", { status: 200 }));

    await createApiClient(fetcher).getReferenceResegmentationCandidate("A6000_S0005_0");

    expect(fetcher.mock.calls[0][0]).toBe(
      "/api/annotations/A6000_S0005_0/reference-resegmentation-candidate",
    );
  });

  it("reads the optional Gemini emotion overlay from the capability-fenced route", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response("null", { status: 200 }));

    await createApiClient(fetcher).getGeminiEmotionOverlayCandidate?.("A6000_S0005_0");

    expect(fetcher.mock.calls[0][0]).toBe(
      "/api/annotations/A6000_S0005_0/gemini-emotion-overlay-candidate",
    );
  });

  it("posts the reviewer's sign-off as the promotion body", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({}), { status: 201 }),
    );
    const submission = {
      reviewer: "reviewer@example.test",
      acknowledged: true,
      review_note: "확인함",
      change_reasons: [],
      turns: [
        {
          start: 0,
          end: 1,
          speaker: "SPEAKER_00",
          transcript: "네",
          emotion: "neutral",
          emotion_rationale: "",
          confidence: 0.5,
        },
      ],
    };

    await createApiClient(fetcher).promoteSilverToGold("A6000_S0005_0", submission);

    const [url, init] = fetcher.mock.calls[0];
    expect(url).toBe("/api/annotations/A6000_S0005_0/gold");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(submission);
  });

  it("surfaces the backend's safe public error", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({ detail: { code: "audio_rejected", message: "The uploaded audio was rejected." } }),
        { status: 422, headers: { "content-type": "application/json" } },
      ),
    );
    const client = createApiClient(fetcher);

    await expect(client.getJob("j1")).rejects.toEqual(
      expect.objectContaining({ code: "audio_rejected", status: 422 }),
    );
  });
});
