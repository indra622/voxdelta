import type { EmotionResult, OperationalState, Utterance } from "../../api/types";

export interface ArcPoint {
  utteranceId: string;
  index: number;
  /** 0-100 across the plot width. */
  x: number;
  /** 0-100 down the plot height, already inverted so higher distress sits higher. */
  y: number;
  intensity: number;
  state: OperationalState;
  uncertain: boolean;
  start: number | null;
}

export function isUncertain(emotion: EmotionResult): boolean {
  return emotion.operational_state === "uncertain" || Boolean(emotion.calibration?.abstained);
}

function intensityOf(emotion: EmotionResult): number {
  const value = emotion.smoothed_negative_intensity ?? emotion.negative_intensity;
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
}

export function buildArc(
  rows: Array<{ emotion: EmotionResult; utterance: Utterance | undefined }>,
): ArcPoint[] {
  const span = Math.max(1, rows.length - 1);
  return rows.map(({ emotion, utterance }, index) => {
    const intensity = intensityOf(emotion);
    return {
      utteranceId: emotion.utterance_id,
      index,
      x: rows.length === 1 ? 50 : (index / span) * 100,
      y: (1 - intensity) * 100,
      intensity,
      state: emotion.operational_state,
      uncertain: isUncertain(emotion),
      start: utterance ? utterance.start : null,
    };
  });
}

/** Runs of consecutive asserted points.
 *
 * The line is broken across every abstained utterance on purpose: drawing through
 * one would interpolate a trend across a reading the model declined to make.
 */
export function arcSegments(points: ArcPoint[]): ArcPoint[][] {
  const segments: ArcPoint[][] = [];
  let current: ArcPoint[] = [];
  for (const point of points) {
    if (point.uncertain) {
      if (current.length > 1) segments.push(current);
      current = [];
      continue;
    }
    current.push(point);
  }
  if (current.length > 1) segments.push(current);
  return segments;
}

export function polyline(points: ArcPoint[]): string {
  return points.map((point) => point.x.toFixed(2) + "," + point.y.toFixed(2)).join(" ");
}
