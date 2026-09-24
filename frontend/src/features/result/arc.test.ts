import { describe, expect, it } from "vitest";
import type { EmotionResult, Utterance } from "../../api/types";
import { arcSegments, buildArc, isUncertain, polyline } from "./arc";

const provider = { name: "p", model: "m", remote: false, transmits: [] as never[], retention_policy_url: null, schema_version: "1", revision: null };
const probs = { happiness: 0.1, anger: 0.2, disgust: 0.1, fear: 0.1, neutral: 0.2, sadness: 0.2, surprise: 0.1 };

function emotion(id: string, intensity: number, abstained: boolean): EmotionResult {
  return {
    utterance_id: id,
    probabilities: probs,
    operational_state: abstained ? "uncertain" : "dissatisfied",
    negative_intensity: intensity,
    smoothed_negative_intensity: null,
    confidence: abstained ? 0.3 : 0.8,
    provider,
    calibration: { calibration_id: "c", method: "temperature-scaling", temperature: 1, abstain_threshold: 0.55, abstained, raw_confidence: 0.5 },
  };
}
const utterance = (id: string, start: number): Utterance =>
  ({ id, start, end: start + 2, speaker_id: "S0", role: "customer", transcript: "t", overlap: false, confidence: 0.9 });

const rows = (spec: Array<[string, number, boolean]>) =>
  spec.map(([id, intensity, abstained], index) => ({
    emotion: emotion(id, intensity, abstained),
    utterance: utterance(id, index * 10),
  }));

describe("emotion arc geometry", () => {
  it("spreads points evenly and inverts intensity so distress reads high", () => {
    const points = buildArc(rows([["u1", 0, false], ["u2", 0.5, false], ["u3", 1, false]]));
    expect(points.map((point) => point.x)).toEqual([0, 50, 100]);
    expect(points.map((point) => point.y)).toEqual([100, 50, 0]);
    expect(points[1].start).toBe(10);
  });

  it("prefers the smoothed intensity when the backend supplies one", () => {
    const row = rows([["u1", 0.2, false]])[0];
    const smoothed = { ...row, emotion: { ...row.emotion, smoothed_negative_intensity: 0.8 } };
    expect(buildArc([smoothed])[0].y).toBeCloseTo(20);
  });

  it("breaks the line across an abstained utterance instead of interpolating over it", () => {
    const points = buildArc(rows([["u1", 0.2, false], ["u2", 0.4, false], ["u3", 0.9, true], ["u4", 0.3, false], ["u5", 0.1, false]]));
    const segments = arcSegments(points);
    expect(segments).toHaveLength(2);
    expect(segments[0].map((point) => point.utteranceId)).toEqual(["u1", "u2"]);
    expect(segments[1].map((point) => point.utteranceId)).toEqual(["u4", "u5"]);
    expect(segments.flat().some((point) => point.uncertain)).toBe(false);
  });

  it("drops a run that an abstention reduces to a single point", () => {
    const points = buildArc(rows([["u1", 0.2, false], ["u2", 0.9, true], ["u3", 0.3, false]]));
    expect(arcSegments(points)).toHaveLength(0);
  });

  it("treats the backend's uncertain state as withheld even without a calibration", () => {
    const bare = { ...emotion("u1", 0.5, false), operational_state: "uncertain" as const, calibration: null };
    expect(isUncertain(bare)).toBe(true);
  });

  it("renders a polyline the SVG can consume", () => {
    const points = buildArc(rows([["u1", 0, false], ["u2", 1, false]]));
    expect(polyline(points)).toBe("0.00,100.00 100.00,0.00");
  });
});
