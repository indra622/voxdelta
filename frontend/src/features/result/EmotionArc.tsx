import type { EmotionResult, Utterance } from "../../api/types";
import { STATE_LABELS, stateTone } from "../../labels";
import { arcSegments, buildArc, polyline, type ArcPoint } from "./arc";

interface EmotionArcProps {
  rows: Array<{ emotion: EmotionResult; utterance: Utterance | undefined }>;
  selectedId: string;
  onSelect(utteranceId: string, start: number | undefined): void;
}

function timestamp(seconds: number): string {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  return minutes + ":" + Math.floor(seconds % 60).toString().padStart(2, "0");
}

function describe(point: ArcPoint): string {
  const position = point.index + 1;
  const when = point.start === null ? "" : timestamp(point.start) + " ";
  if (point.uncertain) return `${position}번째 고객 발화, ${when}판단 보류`;
  return `${position}번째 고객 발화, ${when}${STATE_LABELS[point.state]}, 부정 강도 ${Math.round(point.intensity * 100)}퍼센트`;
}

export function EmotionArc({ rows, selectedId, onSelect }: EmotionArcProps) {
  const points = buildArc(rows);
  if (points.length < 2) return null;
  const segments = arcSegments(points);

  return (
    <figure className="arc">
      <figcaption>
        <span>통화 중 감정 흐름</span>
        <span className="arc-legend">
          <i className="arc-swatch" style={{ background: "var(--state-escalated)" }} aria-hidden="true" />강함
          <i className="arc-swatch" style={{ background: "var(--state-satisfied)" }} aria-hidden="true" />약함
          <i className="arc-swatch withheld" aria-hidden="true" />판단 보류
        </span>
      </figcaption>

      <div className="arc-plot">
        <span className="arc-axis high" aria-hidden="true">강함</span>
        <span className="arc-axis low" aria-hidden="true">약함</span>
        <div className="arc-frame">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="고객 발화별 부정 감정 강도의 변화">
          <line className="arc-base" x1="0" y1="100" x2="100" y2="100" vectorEffect="non-scaling-stroke" />
          {points.filter((point) => point.uncertain).map((point) => (
            <line
              key={point.utteranceId}
              className="arc-gap"
              x1={point.x}
              y1={point.y}
              x2={point.x}
              y2="100"
              vectorEffect="non-scaling-stroke"
            />
          ))}
          {segments.map((segment) => (
            <polyline
              key={segment[0].utteranceId}
              className="arc-line"
              points={polyline(segment)}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>

        {/* Buttons sit above the plot so every point is focusable and hit-target sized. */}
        <ul className="arc-points">
          {points.map((point) => (
            <li key={point.utteranceId} style={{ left: point.x + "%", top: point.y + "%" }}>
              <button
                type="button"
                className={
                  "arc-point" +
                  (point.uncertain ? " withheld" : "") +
                  (point.utteranceId === selectedId ? " selected" : "")
                }
                style={point.uncertain ? undefined : { "--point-tone": stateTone(point.state) } as React.CSSProperties}
                aria-pressed={point.utteranceId === selectedId}
                aria-label={describe(point)}
                onClick={() => onSelect(point.utteranceId, point.start ?? undefined)}
              />
            </li>
          ))}
        </ul>
        </div>
      </div>

      <p className="arc-note">
        판단을 보류한 구간에서는 선을 잇지 않습니다. 보류 구간의 값은 참고용이며 결론이 아닙니다.
      </p>
    </figure>
  );
}
