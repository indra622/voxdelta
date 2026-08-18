# VoxDelta Dashboard and Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local VoxDelta dashboard for upload/progress, role confirmation, synchronized audio/transcript/emotion inspection, highlighted transitions, provider comparison, and HTML/PDF export.

**Architecture:** A Vite React SPA uses a typed fetch client and TanStack Query to consume the local FastAPI API from the core plan. Feature folders own UI, state, and tests; all charts and exports render canonical report JSON rather than recomputing analysis. The backend adds only report rendering and comparison-read endpoints in this plan.

**Tech Stack:** Node.js 24, npm 11, React, TypeScript, Vite, React Router, TanStack Query, Zod, Recharts, WaveSurfer.js, Vitest, Testing Library, MSW, Playwright, FastAPI, Jinja2, Playwright PDF

## Global Constraints

- Bind both frontend and backend to loopback addresses only.
- The UI must state whether each configured provider is local or remote and whether it transmits audio, text, or features.
- Role confirmation must precede emotion analysis.
- Charts must retain a text/table equivalent and cannot be the only representation of emotion or confidence.
- `uncertain` and missing emotion values appear as gaps/markers, never as neutral.
- Recovery/worsening copy must use “followed by” or “associated with,” never causal language.
- Dashboard and exported reports must consume the same `AnalysisReport` JSON.
- Tests use fake API fixtures and must not require model credentials.

---

## Task 1: Frontend foundation and typed API contracts

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/app/App.tsx`
- Create: `frontend/src/api/schema.ts`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/test/setup.ts`
- Create: `frontend/src/api/client.test.ts`

**Interfaces:**
- Produces: `api.getProviders`, `api.createJob`, `api.getJob`, `api.getAudioUrl`, `api.deleteJob`, `api.confirmRoles`, `api.retryStage`, `api.getReport`, `api.getComparison`, and Zod-inferred canonical types.
- Consumes: the core plan's `/api/jobs` endpoints and report JSON.

- [ ] **Step 1: Scaffold the Vite application and dependencies**

Run:

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install @tanstack/react-query react-router-dom recharts wavesurfer.js zod
npm install -D @playwright/test @testing-library/jest-dom @testing-library/react @testing-library/user-event jsdom msw vitest
```

Expected: `npm run build` completes on the generated application.

- [ ] **Step 2: Configure Vitest and write a failing client test**

Add `test: "vitest run"` and `test:watch: "vitest"` scripts. Configure jsdom and `src/test/setup.ts` to import `@testing-library/jest-dom/vitest`.

Create `frontend/src/api/client.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { createApiClient } from "./client";

describe("api client", () => {
  it("rejects a report that omits canonical transitions", async () => {
    const fetcher: typeof fetch = async () => new Response(JSON.stringify({ job_id: "j1" }), { status: 200 });
    const api = createApiClient("http://127.0.0.1:8765", fetcher);
    await expect(api.getReport("j1")).rejects.toThrow("transitions");
  });
});
```

- [ ] **Step 3: Run the test to verify failure**

Run: `cd frontend && npm test -- src/api/client.test.ts`
Expected: FAIL because `createApiClient` does not exist.

- [ ] **Step 4: Define Zod schemas and implement the client**

Define `Role`, `StageStatus`, `Utterance`, `EmotionResult`, `ResponseStrategyResult`, `EmotionTransition`, `AnalysisReport`, and `JobStatus` schemas with field names matching `backend/src/voxdelta/domain/models.py` exactly. Define the provider response explicitly as an object containing `stages`, where each stage value has `name`, `model`, `remote`, `transmits`, optional `retention_policy_url`, and `schema_version`; reject unknown transmission values rather than treating them as display text.

Implement:

```typescript
export function createApiClient(baseUrl: string, fetcher: typeof fetch = fetch) {
  async function parse<T>(response: Response, schema: z.ZodType<T>): Promise<T> {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return schema.parse(await response.json());
  }
  return {
    getProviders: async () => parse(await fetcher(`${baseUrl}/api/config/providers`), ProviderConfigSchema),
    createJob: async (file: File, options: { channelPreference: "auto" | "mixed" | "separate"; diagnosticCapture: boolean }) => {
      const body = new FormData();
      body.append("file", file);
      body.append("channel_preference", options.channelPreference);
      body.append("diagnostic_capture", String(options.diagnosticCapture));
      return parse(await fetcher(`${baseUrl}/api/jobs`, { method: "POST", body }), CreateJobResponseSchema);
    },
    getJob: async (id: string) => parse(await fetcher(`${baseUrl}/api/jobs/${id}`), JobStatusSchema),
    getAudioUrl: (id: string) => `${baseUrl}/api/jobs/${encodeURIComponent(id)}/audio`,
    deleteJob: async (id: string) => {
      const response = await fetcher(`${baseUrl}/api/jobs/${id}`, { method: "DELETE" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
    },
    confirmRoles: async (id: string, mapping: Record<string, "customer" | "agent">) => parse(await fetcher(`${baseUrl}/api/jobs/${id}/roles`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ mapping }) }), JobStatusSchema),
    retryStage: async (id: string, stage: string) => parse(await fetcher(`${baseUrl}/api/jobs/${id}/retry`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ stage }) }), JobStatusSchema),
    getReport: async (id: string) => parse(await fetcher(`${baseUrl}/api/jobs/${id}/report`), AnalysisReportSchema),
  };
}
```

- [ ] **Step 5: Add providers and routes**

Mount `QueryClientProvider` and `BrowserRouter` in `main.tsx`. Define `/`, `/jobs/:jobId`, `/jobs/:jobId/analysis`, and `/compare` routes in `App.tsx`. Until each feature task replaces it, the route renders a `NotReady` component with the exact copy `이 화면은 다음 구현 단계에서 활성화됩니다.`

- [ ] **Step 6: Run checks and commit**

Run: `cd frontend && npm test && npm run build`
Expected: client test passes and TypeScript build succeeds.

```bash
git add frontend
git commit -m "feat: scaffold typed dashboard client"
```

## Task 2: Upload and stage-progress experience

**Files:**
- Create: `frontend/src/features/jobs/UploadCall.tsx`
- Create: `frontend/src/features/jobs/JobProgress.tsx`
- Create: `frontend/src/features/jobs/jobs.css`
- Create: `frontend/src/features/jobs/UploadCall.test.tsx`
- Create: `frontend/src/features/jobs/JobProgress.test.tsx`

**Interfaces:**
- Consumes: `api.createJob`, `api.getJob`, `JobStatus`.
- Produces: navigation from `/` to `/jobs/:jobId` and an accessible retry action.

- [ ] **Step 1: Write failing upload tests**

Test that only `.wav`, `.mp3`, and `.m4a` are accepted, selected filename and size are shown, channel preference offers auto/mixed/separate, remote-provider disclosure appears before submission, diagnostic payload capture is off by default, and successful upload navigates to `/jobs/j1`.

```typescript
const file = new File([new Uint8Array(1024)], "sample.wav", { type: "audio/wav" });
await user.upload(screen.getByLabelText("통화 녹음 파일"), file);
expect(screen.getByText("sample.wav")).toBeInTheDocument();
expect(screen.getByText(/원본 음성이 외부 API로 전송될 수 있습니다/)).toBeInTheDocument();
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd frontend && npm test -- src/features/jobs`
Expected: FAIL because job components do not exist.

- [ ] **Step 3: Implement upload with explicit consent**

`UploadCall` requires a file and a checked consent box when any active provider declares `remote=true` and `transmits` includes `audio`. Send `channel_preference` and `diagnostic_capture` form values; diagnostic capture requires a second explicit checkbox warning that provider payloads may contain transcript content. Disable submit while uploading. Surface backend 422 messages without raw response bodies. Do not retain a browser object URL after navigation.

- [ ] **Step 4: Implement stage polling and retry**

Poll `getJob(jobId)` every 1000 ms while any stage is running/pending and stop when paused, failed, or completed. Render all eight stages with text labels and status. On failure, show the public error code/message and a retry button scoped to that stage.

Add a `로컬 작업 삭제` action guarded by a confirmation dialog naming the local filename. Call `DELETE /api/jobs/{jobId}`, clear queries, revoke audio object URLs, and navigate home. State that local deletion does not delete provider-retained remote data.

- [ ] **Step 5: Run tests and commit**

Run: `cd frontend && npm test -- src/features/jobs && npm run build`
Expected: upload and polling tests pass.

```bash
git add frontend/src/features/jobs frontend/src/app/App.tsx
git commit -m "feat: add call upload and progress UI"
```

## Task 3: Customer and agent role confirmation

**Files:**
- Create: `frontend/src/features/roles/RoleConfirmation.tsx`
- Create: `frontend/src/features/roles/RoleConfirmation.test.tsx`
- Modify: `frontend/src/features/jobs/JobProgress.tsx`

**Interfaces:**
- Consumes: paused job utterance preview and `api.confirmRoles`.
- Produces: exactly one-customer/one-agent mapping.

- [ ] **Step 1: Write failing interaction tests**

Fixture two anonymous speakers with one preview audio/text item each. Assert initial suggested roles render, “역할 바꾸기” swaps both values, submit body equals `{SPEAKER_00: "customer", SPEAKER_01: "agent"}`, and duplicate roles cannot be submitted.

- [ ] **Step 2: Run tests to verify failure**

Run: `cd frontend && npm test -- src/features/roles/RoleConfirmation.test.tsx`
Expected: FAIL because the component does not exist.

- [ ] **Step 3: Implement the confirmation gate**

Render two speaker cards with timestamped transcript previews and short audio-preview buttons. Use a single `customerSpeakerId` state; derive the other speaker as agent so duplicate roles are structurally impossible. Submit the exact mapping, invalidate the job query, and resume polling.

- [ ] **Step 4: Connect the paused job state**

Render `RoleConfirmation` only when the `confirm_roles` stage is paused. Do not render emotion or report navigation while confirmation is outstanding.

- [ ] **Step 5: Run checks and commit**

Run: `cd frontend && npm test -- src/features/roles src/features/jobs && npm run build`
Expected: tests and build pass.

```bash
git add frontend/src/features/roles frontend/src/features/jobs/JobProgress.tsx
git commit -m "feat: confirm customer and agent roles"
```

## Task 4: Synchronized analysis dashboard

**Files:**
- Create: `frontend/src/features/analysis/AnalysisPage.tsx`
- Create: `frontend/src/features/analysis/AudioTimeline.tsx`
- Create: `frontend/src/features/analysis/TranscriptPanel.tsx`
- Create: `frontend/src/features/analysis/EmotionDetails.tsx`
- Create: `frontend/src/features/analysis/analysis.css`
- Create: `frontend/src/features/analysis/AnalysisPage.test.tsx`

**Interfaces:**
- Consumes: canonical `AnalysisReport`, `api.getAudioUrl`, `Utterance`, `EmotionResult`.
- Produces: selected timestamp state shared by player, chart, and transcript.

- [ ] **Step 1: Write failing synchronization tests**

Render a report with customer and agent utterances. Click the customer transcript at 12.5 s and assert the mocked player seeks to 12.5; click a chart point and assert the matching transcript gets `aria-current="true"`. Assert uncertain emotion uses label `판단 불확실` and not `안정`.

- [ ] **Step 2: Run tests to verify failure**

Run: `cd frontend && npm test -- src/features/analysis/AnalysisPage.test.tsx`
Expected: FAIL because analysis components do not exist.

- [ ] **Step 3: Implement the report page state**

Fetch report once the report stage is completed. Store `selectedUtteranceId` and `currentTime` in `AnalysisPage`; pass callbacks down. Display start/end state, peak interval, provider/model, valid-emotion coverage, and warnings before the detail panels.

- [ ] **Step 4: Implement WaveSurfer player and Recharts timeline**

Create WaveSurfer once per `api.getAudioUrl(jobId)` value and destroy it on unmount. The browser uses the backend's HTTP byte-range support for seeking; no filesystem path reaches the frontend. Plot customer negative intensity by utterance midpoint. Use `connectNulls={false}` so missing/uncertain values remain gaps. Render an accessible table immediately below the chart with time, state, intensity, and confidence.

- [ ] **Step 5: Implement transcript and details**

Render customer and agent turns with distinct non-color labels, timestamps, transcript, and confidence. Selecting a customer turn shows all seven probabilities, operational state, raw/smoothed intensity, provider/model, and evidence modalities.

- [ ] **Step 6: Run checks and commit**

Run: `cd frontend && npm test -- src/features/analysis && npm run build`
Expected: synchronization and uncertainty tests pass; build succeeds.

```bash
git add frontend/src/features/analysis frontend/src/app/App.tsx
git commit -m "feat: visualize call emotion timeline"
```

## Task 5: Recovery/worsening evidence and provider comparison

**Files:**
- Create: `frontend/src/features/analysis/TransitionList.tsx`
- Create: `frontend/src/features/analysis/TransitionList.test.tsx`
- Create: `frontend/src/features/compare/ComparisonPage.tsx`
- Create: `frontend/src/features/compare/ComparisonPage.test.tsx`
- Modify: `frontend/src/api/schema.ts`
- Modify: `frontend/src/api/client.ts`
- Create: `backend/src/voxdelta/api/comparison.py`
- Create: `backend/tests/api/test_comparison.py`

**Interfaces:**
- Consumes: `EmotionTransition`, response strategies, benchmark summary JSON.
- Produces: evidence-linked transition cards and local/API quality/latency/cost comparison.

- [ ] **Step 1: Write failing transition copy tests**

Assert a recovery card renders previous customer, agent, and next customer turns; displays `-0.32`; includes “직후 회복과 연관”; and contains none of `원인`, `야기`, `때문에`, or `caused`.

- [ ] **Step 2: Implement transition cards**

Sort by absolute delta descending, filter recovery/worsening, label response strategy, and make every card seek to each of its three source utterances. Show stable transitions only behind a “안정 구간 포함” toggle.

- [ ] **Step 3: Add benchmark summary endpoint and schema**

Define a backend `BenchmarkSummary` response with provider/model, dataset hash, macro-F1, expected calibration error, mean latency, p95 latency, cost per audio minute, failures, and uncertain rate. `GET /api/comparisons/latest` reads a versioned local artifact and returns 404 when no benchmark has run.

- [ ] **Step 4: Implement comparison UI**

Display local and API providers as two cards plus a metric table. Do not rank a provider when dataset hashes differ. Format absent API cost as `측정 안 됨`, not zero. Link the run metadata and per-class metrics below the summary.

- [ ] **Step 5: Run checks and commit**

Run: `cd backend && uv run pytest tests/api/test_comparison.py -v && cd ../frontend && npm test -- src/features/analysis/TransitionList.test.tsx src/features/compare/ComparisonPage.test.tsx && npm run build`
Expected: backend and frontend tests pass.

```bash
git add backend/src/voxdelta/api/comparison.py backend/tests/api/test_comparison.py frontend/src/features frontend/src/api
git commit -m "feat: show transition evidence and model comparison"
```

## Task 6: Shared HTML and PDF report export

**Files:**
- Create: `backend/src/voxdelta/reports/render.py`
- Create: `backend/src/voxdelta/reports/templates/report.html.j2`
- Create: `backend/tests/reports/test_render.py`
- Modify: `backend/src/voxdelta/api/app.py`
- Create: `frontend/src/features/analysis/ExportActions.tsx`
- Create: `frontend/src/features/analysis/ExportActions.test.tsx`

**Interfaces:**
- Consumes: `AnalysisReport` only.
- Produces: `GET /api/jobs/{job_id}/report.html` and `GET /api/jobs/{job_id}/report.pdf`.

- [ ] **Step 1: Write failing renderer tests**

Assert HTML contains job ID, emotion summary, all highlighted triplets, provider provenance, warnings, and the phrase “시간상 연관”; assert it excludes causal banned phrases. Assert rendering the same report twice yields byte-identical HTML.

- [ ] **Step 2: Run renderer tests to verify failure**

Run: `cd backend && uv run pytest tests/reports/test_render.py -v`
Expected: FAIL because report renderer does not exist.

- [ ] **Step 3: Implement deterministic HTML rendering**

Add Jinja2 to backend dependencies. Render standalone UTF-8 HTML with inline print CSS. Escape transcripts by default. Serialize chart data into a script tag only after replacing `<` with `\u003c`; do not include provider raw payloads or local absolute paths.

- [ ] **Step 4: Implement PDF export**

Add Playwright to the backend dev/runtime export group, install Chromium during setup, write HTML to a temporary job export path, and call headless Chromium `page.pdf(format="A4", print_background=True)`. Reuse cached PDF when report artifact hash is unchanged.

- [ ] **Step 5: Add frontend export actions**

Render HTML and PDF links only after report completion. Use regular anchor downloads so large files do not enter React state. Display export errors returned by the local API.

- [ ] **Step 6: Run checks and commit**

Run: `cd backend && uv run pytest tests/reports -v && uv run ruff check . && cd ../frontend && npm test -- src/features/analysis/ExportActions.test.tsx && npm run build`
Expected: renderer and UI tests pass.

```bash
git add backend frontend/src/features/analysis
git commit -m "feat: export shared analysis reports"
```

## Task 7: Browser acceptance tests and dashboard handoff

**Files:**
- Create: `frontend/playwright.config.ts`
- Create: `frontend/tests/voxdelta-flow.spec.ts`
- Create: `frontend/README.md`
- Modify: `docs/superpowers/plans/2026-08-18-voxdelta-dashboard.md`

**Interfaces:**
- Consumes: running fake-provider backend and built dashboard.
- Produces: repeatable browser-level proof of the complete local flow.

- [ ] **Step 1: Write the Playwright flow**

The test uploads the generated fixture, waits for role pause, swaps and confirms roles, waits for report completion, selects an emotion point, opens a recovery/worsening card, checks non-causal copy, opens model comparison, and downloads HTML. Use role/label locators rather than CSS selectors.

- [ ] **Step 2: Run the test to verify missing orchestration**

Run: `cd frontend && npx playwright test tests/voxdelta-flow.spec.ts`
Expected: FAIL until `webServer` starts backend and frontend.

- [ ] **Step 3: Configure both local web servers**

Configure Playwright `webServer` entries for `cd ../backend && uv run uvicorn voxdelta.api.app:app --host 127.0.0.1 --port 8765` and `npm run dev -- --host 127.0.0.1 --port 5173`. Reuse existing servers only outside CI.

- [ ] **Step 4: Document frontend operation**

Document Node 24, `npm ci`, `npm run dev`, provider disclosure behavior, test commands, browser support, accessible chart table, and report downloads.

- [ ] **Step 5: Run final dashboard verification**

Run:

```bash
cd backend && uv run pytest -q && uv run ruff check .
cd ../frontend && npm test && npm run build && npx playwright test
cd .. && git diff --check && git status --short
```

Expected: backend tests, frontend unit tests, production build, and Playwright flow all pass; only handoff documentation and checked plan boxes remain.

- [ ] **Step 6: Commit dashboard completion**

```bash
git add frontend/README.md frontend/playwright.config.ts frontend/tests docs/superpowers/plans/2026-08-18-voxdelta-dashboard.md
git commit -m "docs: complete dashboard handoff"
```
