import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { AnalysisReport } from "../../api/types";
import { AnalysisResult } from "./AnalysisResult";

const report: AnalysisReport = {
  job_id: "j1",
  summary: {
    start_state: "uncertain",
    end_state: "stable",
    peak_customer_utterance_id: "u1",
    overall_delta: 0,
    valid_coverage: 0.5,
    recovery_count: 0,
    worsening_count: 0,
    narrative: null,
  },
  utterances: [{ id: "u1", start: 2, end: 4, speaker_id: "S0", role: "customer", transcript: "환불 문의입니다.", overlap: false, confidence: 0.9 }],
  emotions: [{
    utterance_id: "u1",
    probabilities: { happiness: 0.1, anger: 0.2, disgust: 0.1, fear: 0.1, neutral: 0.2, sadness: 0.2, surprise: 0.1 },
    operational_state: "uncertain",
    negative_intensity: 0.5,
    smoothed_negative_intensity: null,
    confidence: 0.42,
    provider: { name: "wav2vec-xls-r", model: "xls-r-300m", remote: false, transmits: [], retention_policy_url: null, schema_version: "1", revision: null },
    calibration: { calibration_id: "c1", method: "temperature-scaling", temperature: 1, abstain_threshold: 0.55, abstained: true, raw_confidence: 0.48 },
  }],
  strategies: [],
  transitions: [],
  warnings: [],
  transcription_coverage: null,
  schema_version: "1",
};

describe("AnalysisResult", () => {
  it("shows abstention as uncertain instead of neutral", () => {
    render(<AnalysisResult report={report} onReset={vi.fn()} />);
    expect(screen.getByRole("heading", { level: 3, name: "판단 불확실" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /판단 불확실/ })).toHaveTextContent("확신도 42%");
    expect(screen.getAllByText("42%")).toHaveLength(2);
    expect(screen.getByText(/중립으로 해석하지 말고/)).toBeInTheDocument();
    expect(screen.getByText(/결론이 아닌 참고 값/)).toBeInTheDocument();
  });

  it("states the abstain comparison that produced the verdict", () => {
    render(<AnalysisResult report={report} onReset={vi.fn()} />);
    expect(screen.getByText(/판단 기준 55%에 못 미쳐 감정을 단정하지 않았습니다/)).toBeInTheDocument();
    expect(screen.getByText("기준 55%")).toBeInTheDocument();
    expect(screen.getByText(/보정 전 48%/)).toBeInTheDocument();
  });

  it("emphasises nothing in an abstained distribution", () => {
    const { container } = render(<AnalysisResult report={report} onReset={vi.fn()} />);
    expect(container.querySelectorAll(".bar-row")).toHaveLength(7);
    expect(container.querySelector(".bar-row.lead")).toBeNull();
  });

  it("emphasises the leading emotion once a result clears the threshold", () => {
    const confident: AnalysisReport = {
      ...report,
      emotions: [{
        ...report.emotions[0],
        operational_state: "escalated",
        confidence: 0.82,
        calibration: { ...report.emotions[0].calibration!, abstained: false, raw_confidence: 0.88 },
      }],
    };
    const { container } = render(<AnalysisResult report={confident} onReset={vi.fn()} />);
    expect(screen.getByText(/판단 기준 55%을 넘어 이 감정을 제시합니다/)).toBeInTheDocument();
    expect(container.querySelector(".bar-row.lead .bar-label")?.textContent).toBe("분노");
  });

  it("says so when a result carries no calibration at all", () => {
    const uncalibrated: AnalysisReport = {
      ...report,
      emotions: [{ ...report.emotions[0], calibration: null }],
    };
    render(<AnalysisResult report={uncalibrated} onReset={vi.fn()} />);
    expect(screen.getByText(/원시 확신도입니다/)).toBeInTheDocument();
  });

  it("still refuses to conclude when an uncalibrated report only carries the uncertain state", () => {
    // What the current backend actually returns: no calibration key on the emotion at all.
    const { calibration: _calibration, ...bare } = report.emotions[0];
    render(<AnalysisResult report={{ ...report, emotions: [bare] }} onReset={vi.fn()} />);

    expect(screen.getByRole("heading", { level: 3, name: "판단 불확실" })).toBeInTheDocument();
    expect(screen.getByText(/원시 확신도입니다/)).toBeInTheDocument();
    expect(screen.getByText(/중립으로 해석하지 말고/)).toBeInTheDocument();
    expect(screen.queryByText(/기준 55%/)).not.toBeInTheDocument();
  });

  it("keeps every customer utterance reachable as a real list item", () => {
    render(<AnalysisResult report={report} onReset={vi.fn()} />);
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(report.emotions.length);
    expect(within(items[0]).getByRole("button")).toHaveAttribute("aria-pressed", "true");
  });

  it("marks a remote provider instead of claiming a local analysis", () => {
    const remote: AnalysisReport = {
      ...report,
      emotions: [{
        ...report.emotions[0],
        provider: { ...report.emotions[0].provider, remote: true, transmits: ["audio"] },
      }],
    };
    render(<AnalysisResult report={remote} onReset={vi.fn()} />);
    expect(screen.getByText(/원격 provider 사용됨/)).toBeInTheDocument();
    expect(screen.queryByText("감정 분석 로컬 실행")).not.toBeInTheDocument();
  });

  it("explains that there is nothing to show rather than rendering an empty table", () => {
    render(<AnalysisResult report={{ ...report, emotions: [] }} onReset={vi.fn()} />);
    expect(screen.getByRole("heading", { name: /표시할 감정 결과가 없어요/ })).toBeInTheDocument();
  });

  it("says nothing about transcript coverage when the recognizer needed no sanitation", () => {
    render(<AnalysisResult report={report} onReset={vi.fn()} />);
    expect(screen.queryByText("전사 반영")).toBeNull();
    expect(screen.queryByText(/전사에서 빠졌어요/)).toBeNull();
  });

  it("shows the attributed ratio and word count when every word was placed", () => {
    const covered: AnalysisReport = {
      ...report,
      transcription_coverage: {
        policy: "strict-interior-point-attribution",
        attributed_words: 120,
        attributed_ratio: 1,
        omitted_words: 0,
        uncertain: false,
      },
    };
    const { container } = render(<AnalysisResult report={covered} onReset={vi.fn()} />);
    expect(screen.getByText("전사 반영")).toBeInTheDocument();
    expect(screen.getByText(/120단어/)).toBeInTheDocument();
    // A complete transcript must not be dressed up as a caveat.
    expect(screen.queryByText(/전사에서 빠졌어요/)).toBeNull();
    expect(container.querySelectorAll(".tile.withheld")).toHaveLength(1);
  });

  it("names the omitted word count and why, when attribution lost words", () => {
    const lossy: AnalysisReport = {
      ...report,
      transcription_coverage: {
        policy: "strict-interior-point-attribution",
        attributed_words: 108,
        attributed_ratio: 0.9,
        omitted_words: 12,
        uncertain: true,
      },
    };
    render(<AnalysisResult report={lossy} onReset={vi.fn()} />);
    expect(screen.getByText("12개 단어가 전사에서 빠졌어요")).toBeInTheDocument();
    expect(screen.getByText(/말한 시각이 명확하지 않아/)).toBeInTheDocument();
    // The reason matters: the words were heard, not missed, and no timing was invented.
    expect(screen.getByText(/시각을 임의로 만들어 넣지 않고/)).toBeInTheDocument();
    expect(screen.getByText("90%")).toBeInTheDocument();
  });

  it("marks the coverage tile as withheld only when words were lost", () => {
    const lossy: AnalysisReport = {
      ...report,
      transcription_coverage: {
        policy: "strict-interior-point-attribution",
        attributed_words: 108,
        attributed_ratio: 0.9,
        omitted_words: 12,
        uncertain: true,
      },
    };
    const { container } = render(<AnalysisResult report={lossy} onReset={vi.fn()} />);
    expect(container.querySelectorAll(".tile.withheld")).toHaveLength(2);
  });
});
