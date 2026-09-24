import { expect, test, type Page } from "@playwright/test";

const STAGE_NAMES = [
  "normalize",
  "diarize",
  "transcribe",
  "confirm_roles",
  "emotion",
  "response_strategy",
  "transitions",
  "report",
] as const;

type StageStatus = "pending" | "running" | "paused" | "completed" | "failed" | "skipped";

function stages(overrides: Partial<Record<string, StageStatus>> = {}) {
  return Object.fromEntries(
    STAGE_NAMES.map((name) => [name, { status: overrides[name] ?? "completed", error: null }]),
  );
}

const provider = {
  name: "wav2vec-xls-r",
  model: "xls-r-300m-calibrated",
  remote: false,
  transmits: [],
  retention_policy_url: null,
  retention_window_hours: null,
  schema_version: "1",
  revision: "poc-v1",
};

const SHOT_DIR = process.env.VOXDELTA_SCREENSHOT_DIR ?? "test-results";

/** Read the disclosure, tick the acknowledgement, and start the analysis. */
async function acknowledgeAndStart(page: Page) {
  const start = page.getByRole("button", { name: "감정 분석 시작" });
  await expect(start).toBeDisabled();
  await page.getByRole("checkbox", { name: /pyannoteAI로 전송하는 데 동의합니다/ }).check();
  await expect(start).toBeEnabled();
  await start.click();
}

const pausedJob = {
  job_id: "j1",
  status: "paused",
  diagnostic_capture: false,
  created_at: "2026-08-27T00:00:00Z",
  updated_at: "2026-08-27T00:00:04Z",
  stages: stages({
    confirm_roles: "paused",
    emotion: "pending",
    response_strategy: "pending",
    transitions: "pending",
    report: "pending",
  }),
  role_candidate: {
    speakers: ["SPEAKER_00", "SPEAKER_01"],
    suggested_mapping: { SPEAKER_00: "customer", SPEAKER_01: "agent" },
  },
};

const completedJob = {
  ...pausedJob,
  status: "completed",
  stages: stages(),
  role_candidate: null,
};

const report = {
  job_id: "j1",
  summary: { start_state: "dissatisfied", end_state: "uncertain", peak_customer_utterance_id: "u2", overall_delta: -0.12, valid_coverage: 0.5, recovery_count: 0, worsening_count: 0, narrative: null },
  utterances: [
    { id: "u1", start: 1, end: 3, speaker_id: "SPEAKER_00", role: "customer", transcript: "배송이 아직 안 왔어요.", overlap: false, confidence: 0.94 },
    { id: "u2", start: 8, end: 10, speaker_id: "SPEAKER_00", role: "customer", transcript: "그럼 오늘 받을 수 있는 건가요?", overlap: false, confidence: 0.92 },
  ],
  emotions: [
    { utterance_id: "u1", probabilities: { happiness: 0.03, anger: 0.52, disgust: 0.08, fear: 0.05, neutral: 0.12, sadness: 0.18, surprise: 0.02 }, operational_state: "dissatisfied", negative_intensity: 0.7, smoothed_negative_intensity: 0.7, confidence: 0.78, provider, calibration: { calibration_id: "cal-v2", method: "temperature-scaling", temperature: 1.2, abstain_threshold: 0.55, abstained: false, raw_confidence: 0.84 } },
    { utterance_id: "u2", probabilities: { happiness: 0.1, anger: 0.14, disgust: 0.06, fear: 0.12, neutral: 0.4, sadness: 0.1, surprise: 0.08 }, operational_state: "uncertain", negative_intensity: 0.44, smoothed_negative_intensity: null, confidence: 0.41, provider, calibration: { calibration_id: "cal-v2", method: "temperature-scaling", temperature: 1.2, abstain_threshold: 0.55, abstained: true, raw_confidence: 0.46 } },
  ],
  strategies: [],
  transitions: [],
  warnings: [],
  schema_version: "1",
};

/** A short silent PCM WAV, so the preview player loads like the real endpoint's. */
function silentWav(seconds = 1, sampleRate = 8000): Buffer {
  const frames = seconds * sampleRate;
  const buffer = Buffer.alloc(44 + frames * 2);
  buffer.write("RIFF", 0, "ascii");
  buffer.writeUInt32LE(36 + frames * 2, 4);
  buffer.write("WAVEfmt ", 8, "ascii");
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36, "ascii");
  buffer.writeUInt32LE(frames * 2, 40);
  return buffer;
}

async function json(page: Page, pattern: string, body: unknown, status = 200) {
  await page.route(pattern, async (route) => {
    await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
  });
}

test.beforeEach(async ({ page }) => {
  let confirmed = false;
  // The Precision-2 PoC configuration: diarization is the one stage that leaves the machine.
  await json(page, "**/api/config/providers", {
    stages: STAGE_NAMES.map((stage) =>
      stage === "diarize"
        ? {
            stage,
            provenance: { name: "pyannoteai", model: "precision-2", remote: true, schema_version: "1", revision: "precision-2" },
            transmits: ["audio"],
            retention_policy_url: "https://docs.pyannote.ai/data-retention",
            retention_window_hours: 48,
          }
        : {
            stage,
            provenance: { name: "local", model: "local", remote: false, schema_version: "1", revision: null },
            transmits: [],
            retention_policy_url: null,
            retention_window_hours: null,
          },
    ),
  });
  await json(page, "**/api/jobs", { job_id: "j1", status_url: "/api/jobs/j1" }, 202);
  await page.route("**/api/jobs/j1", async (route) => {
    if (route.request().method() === "DELETE") {
      await route.fulfill({ status: 204, body: "" });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(confirmed ? completedJob : pausedJob),
    });
  });
  await page.route("**/api/jobs/j1/roles", async (route) => {
    confirmed = true;
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(completedJob) });
  });
  await json(page, "**/api/jobs/j1/report", report);
  await page.route("**/api/jobs/j1/audio", (route) =>
    route.fulfill({
      status: 200,
      contentType: "audio/wav",
      headers: { "Accept-Ranges": "bytes" },
      body: silentWav(),
    }),
  );
});

test("records the whole journey from upload to an explained uncertain result", async ({ page }, testInfo) => {
  const suffix = testInfo.project.name;
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /목소리 속 감정/ })).toBeVisible();
  await expect(page.getByText(/외부로 전송되지 않습니다/)).toHaveCount(0);

  const consent = page.getByRole("region", { name: /pyannoteAI/ });
  await expect(consent).toBeVisible();
  await expect(consent.getByText(/화자 구분 단계 · pyannoteAI/)).toBeVisible();
  await expect(consent.getByText(/최대 48시간까지 남아 있을 수 있고/)).toBeVisible();
  await expect(consent.getByText(/전사문은 어느 단계에서도 전송되지 않습니다/)).toBeVisible();

  await page.getByLabel("분석할 음성 파일").setInputFiles({ name: "sample.wav", mimeType: "audio/wav", buffer: Buffer.from("RIFF") });
  await expect(page.getByText("sample.wav")).toBeVisible();
  await page.screenshot({ path: `${SHOT_DIR}/voxdelta-poc-consent-${suffix}.png`, fullPage: true });

  await acknowledgeAndStart(page);

  await expect(page.getByRole("heading", { name: /역할을 확인해 주세요/ })).toBeVisible();
  await expect(page.getByText("SPEAKER_00")).toBeVisible();
  await page.getByRole("button", { name: "이 역할로 계속 분석" }).click();

  await expect(page.getByRole("heading", { name: "분석이 끝났어요" })).toBeVisible();
  await expect(page.getByText(/1개 구간은 감정을 단정하지 않았어요/)).toBeVisible();
  await expect(page.getByText(/중립으로 해석하지 말고/)).toBeVisible();

  // neutral is the top probability for u2, and the verdict must still refuse to say so.
  const uncertainRow = page.getByRole("button", { name: /판단 불확실/ });
  await expect(uncertainRow).toBeVisible();
  await uncertainRow.click();
  await expect(page.getByRole("heading", { level: 3, name: "판단 불확실" })).toBeVisible();
  await expect(page.getByText(/판단 기준 55%에 못 미쳐 감정을 단정하지 않았습니다/)).toBeVisible();
  await expect(page.getByLabel("분석에 사용된 정규화 음성 미리듣기")).toBeVisible();

  await page.screenshot({ path: `${SHOT_DIR}/voxdelta-poc-result-${suffix}.png`, fullPage: true });
});

test("discards the local job when starting over", async ({ page }) => {
  const deletes: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "DELETE") deletes.push(request.url());
  });

  await page.goto("/");
  await page.getByLabel("분석할 음성 파일").setInputFiles({ name: "sample.wav", mimeType: "audio/wav", buffer: Buffer.from("RIFF") });
  await acknowledgeAndStart(page);
  await page.getByRole("button", { name: "이 역할로 계속 분석" }).click();
  await expect(page.getByRole("heading", { name: "분석이 끝났어요" })).toBeVisible();

  await page.getByRole("button", { name: "새 음성 분석" }).click();

  await expect(page.getByRole("heading", { name: /어떤 목소리를/ })).toBeVisible();
  expect(deletes.some((url) => url.endsWith("/api/jobs/j1"))).toBe(true);
});
