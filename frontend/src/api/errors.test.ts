import { describe, expect, it, vi } from "vitest";
import { createApiClient, ApiError } from "./client";
import { describeError, isMissingJob } from "./errors";

describe("public error guidance", () => {
  it("turns a backend code into actionable Korean instead of the sanitized English", () => {
    const guidance = describeError(
      new ApiError(422, { code: "audio_rejected", message: "The uploaded audio was rejected." }),
    );
    expect(guidance.code).toBe("audio_rejected");
    expect(guidance.title).toContain("받을 수 없어요");
    expect(guidance.detail).toContain("길이");
    expect(guidance.detail).not.toMatch(/[A-Za-z]{4,}/);
  });

  it("falls back without inventing a cause for an unknown code", () => {
    const guidance = describeError(new ApiError(500, { code: "brand_new_code", message: "?" }));
    expect(guidance.code).toBe("brand_new_code");
    expect(guidance.title).toBe("분석을 이어가지 못했어요");
  });

  it("names a transport failure as a launcher problem, not a bad recording", async () => {
    const fetcher = vi.fn<typeof fetch>().mockRejectedValue(new TypeError("Failed to fetch"));
    const client = createApiClient(fetcher);

    const reason = await client.getJob("j1").catch((error: unknown) => error);

    expect(isMissingJob(reason)).toBe(false);
    expect(describeError(reason).detail).toContain("npm run poc");
  });

  it("recognizes an already-deleted job so starting over is not blocked", () => {
    expect(isMissingJob(new ApiError(404, { code: "job_not_found", message: "gone" }))).toBe(true);
  });
});
