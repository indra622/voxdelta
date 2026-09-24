import { AlertCircle, CheckCircle2, ShieldAlert, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { AnalysisReport, ApiClient, EmotionLabel, EmotionResult } from "../../api/types";
import { ExpertGuidanceCard } from "../expert/ExpertGuidanceCard";
import { EMOTION_LABELS, STATE_LABELS, TRANSMIT_LABELS, stateTone } from "../../labels";
import { EmotionArc } from "./EmotionArc";
import { isUncertain } from "./arc";

function percent(value: number): string {
  return Math.round(value * 100) + "%";
}

function timestamp(seconds: number): string {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  return minutes + ":" + Math.floor(seconds % 60).toString().padStart(2, "0");
}

function ranked(emotion: EmotionResult): Array<[EmotionLabel, number]> {
  return (Object.entries(emotion.probabilities) as Array<[EmotionLabel, number]>)
    .sort((a, b) => b[1] - a[1]);
}

function dominantEmotion(emotion: EmotionResult): EmotionLabel {
  return ranked(emotion)[0][0];
}

interface AnalysisResultProps {
  report: AnalysisReport;
  client?: ApiClient;
  onReset(): void;
  resetPending?: boolean;
}

export function AnalysisResult({ report, client, onReset, resetPending = false }: AnalysisResultProps) {
  const headingRef = useRef<HTMLHeadingElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [audioBroken, setAudioBroken] = useState(false);
  const rows = useMemo(
    () => report.emotions.map((emotion) => ({
      emotion,
      utterance: report.utterances.find((utterance) => utterance.id === emotion.utterance_id),
    })),
    [report],
  );
  const [selectedId, setSelectedId] = useState(report.summary.peak_customer_utterance_id || rows[0]?.emotion.utterance_id);
  const selected = rows.find(({ emotion }) => emotion.utterance_id === selectedId) ?? rows[0];
  const uncertainCount = rows.filter(({ emotion }) => isUncertain(emotion)).length;
  // Absent for a recognizer whose timestamps needed no sanitation; see TranscriptionCoverage.
  const coverage = report.transcription_coverage;

  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  function pick(utteranceId: string, start: number | undefined) {
    setSelectedId(utteranceId);
    const element = audioRef.current;
    if (element && typeof start === "number" && Number.isFinite(start)) {
      try {
        element.currentTime = start;
      } catch {
        // Seeking before metadata loads is not an error worth surfacing.
      }
    }
  }

  if (!selected) {
    return (
      <section className="panel result-panel empty-result" aria-labelledby="result-heading">
        <AlertCircle role="img" aria-label="결과 없음" size={26} />
        <h2 id="result-heading" ref={headingRef} tabIndex={-1}>표시할 감정 결과가 없어요</h2>
        <p>유효한 고객 발화가 충분하지 않았습니다. 고객이 말한 구간이 더 많이 담긴 녹음으로 다시 시도해 주세요.</p>
        <button className="button ghost" type="button" disabled={resetPending} onClick={onReset}>
          {resetPending ? "정리하는 중" : "새 음성 분석"}
        </button>
      </section>
    );
  }

  const { emotion, utterance } = selected;
  const uncertain = isUncertain(emotion);
  const tone = stateTone(emotion.operational_state, uncertain);
  const threshold = emotion.calibration?.abstain_threshold ?? null;
  const remote = emotion.provider.remote || emotion.provider.transmits.length > 0;
  const distribution = ranked(emotion);

  return (
    <section className="panel result-panel" aria-labelledby="result-heading">
      <header className="panel-head">
        <div>
          <span className="kicker">03 / RESULT</span>
          <h2 id="result-heading" ref={headingRef} tabIndex={-1}>분석이 끝났어요</h2>
        </div>
        {remote ? (
          <span className="badge alert">
            <ShieldAlert aria-hidden="true" size={14} /> 원격 provider 사용됨
            {emotion.provider.transmits.length > 0 &&
              " · " + emotion.provider.transmits.map((item) => TRANSMIT_LABELS[item] ?? item).join(", ") + " 전송"}
          </span>
        ) : (
          <span className="badge"><ShieldCheck aria-hidden="true" size={14} /> 감정 분석 로컬 실행</span>
        )}
      </header>

      <div className="tiles">
        <div className="tile">
          <span>통화 흐름</span>
          <strong>
            <i className="dot" style={{ background: stateTone(report.summary.start_state) }} aria-hidden="true" />
            {STATE_LABELS[report.summary.start_state]}
            <em aria-label="에서">→</em>
            <i className="dot" style={{ background: stateTone(report.summary.end_state) }} aria-hidden="true" />
            {STATE_LABELS[report.summary.end_state]}
          </strong>
        </div>
        <div className="tile">
          <span>회복 / 악화</span>
          <strong>{report.summary.recovery_count}회 <em>/</em> {report.summary.worsening_count}회</strong>
        </div>
        <div className="tile">
          <span>유효 분석</span>
          <strong>{percent(report.summary.valid_coverage)}</strong>
        </div>
        <div className={uncertainCount > 0 ? "tile withheld" : "tile"}>
          <span>판단 보류</span>
          <strong>{uncertainCount}개 <em>/ {rows.length}개</em></strong>
        </div>
        {coverage && (
          <div className={coverage.uncertain ? "tile withheld" : "tile"}>
            <span>전사 반영</span>
            <strong>
              {percent(coverage.attributed_ratio)} <em>/ {coverage.attributed_words}단어</em>
            </strong>
          </div>
        )}
      </div>

      {coverage?.uncertain && (
        <div className="notice" role="note">
          <AlertCircle aria-hidden="true" size={18} />
          <div>
            <strong>{coverage.omitted_words}개 단어가 전사에서 빠졌어요</strong>
            <p>
              음성 인식은 이 단어들을 들었지만 말한 시각이 명확하지 않아 어느 화자의 발화인지 정할 수 없었습니다.
              시각을 임의로 만들어 넣지 않고 제외했습니다. 아래 전사는 이만큼 비어 있습니다.
            </p>
          </div>
        </div>
      )}

      {uncertainCount > 0 && (
        <div className="notice" role="note">
          <AlertCircle aria-hidden="true" size={18} />
          <div>
            <strong>{uncertainCount}개 구간은 감정을 단정하지 않았어요</strong>
            <p>확신도가 보정된 기준보다 낮은 결과입니다. 중립으로 해석하지 말고, 아래에서 해당 구간의 음성과 전사 내용을 직접 확인해 주세요.</p>
          </div>
        </div>
      )}

      <EmotionArc rows={rows} selectedId={emotion.utterance_id} onSelect={pick} />

      <div className="result-layout">
        <div className="track-column">
          <div className="preview-audio">
            <span className="field-label">분석에 사용된 정규화 음성</span>
            {audioBroken ? (
              <p className="preview-audio-note">음성 미리듣기를 불러오지 못했습니다. 원본 파일을 직접 열어 확인해 주세요.</p>
            ) : (
              <audio
                ref={audioRef}
                aria-label="분석에 사용된 정규화 음성 미리듣기"
                controls
                preload="metadata"
                src={"/api/jobs/" + encodeURIComponent(report.job_id) + "/audio"}
                onError={() => setAudioBroken(true)}
              />
            )}
          </div>

          <h3 id="utterance-list-heading" className="field-label">고객 발화별 결과</h3>
          <ul className="utterance-list" aria-labelledby="utterance-list-heading">
            {rows.map(({ emotion: item, utterance: itemUtterance }, index) => {
              const itemUncertain = isUncertain(item);
              const active = item.utterance_id === emotion.utterance_id;
              return (
                <li key={item.utterance_id}>
                  <button
                    className={active ? "utterance-row selected" : "utterance-row"}
                    type="button"
                    aria-pressed={active}
                    onClick={() => pick(item.utterance_id, itemUtterance?.start)}
                  >
                    <i
                      className={itemUncertain ? "state-mark withheld" : "state-mark"}
                      style={itemUncertain ? undefined : { background: stateTone(item.operational_state) }}
                      aria-hidden="true"
                    />
                    <span className="utterance-copy">
                      <strong>
                        {itemUncertain ? "판단 불확실" : EMOTION_LABELS[dominantEmotion(item)]}
                        <small>{itemUtterance ? timestamp(itemUtterance.start) : "#" + (index + 1)}</small>
                      </strong>
                      <small className="utterance-text">{itemUtterance?.transcript ?? item.utterance_id}</small>
                    </span>
                    <span className={itemUncertain ? "confidence withheld" : "confidence"}>
                      <span className="visually-hidden">확신도 </span>{percent(item.confidence)}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>

        <article
          className={uncertain ? "detail withheld" : "detail"}
          style={{ "--tone": tone } as React.CSSProperties}
          aria-live="polite"
        >
          <div className="detail-head">
            <span className="field-label">
              {utterance ? timestamp(utterance.start) + " 고객 발화" : "선택한 고객 발화"}
            </span>
            <h3>
              {uncertain ? "판단 불확실" : EMOTION_LABELS[dominantEmotion(emotion)]}
              {uncertain
                ? <AlertCircle aria-hidden="true" size={20} />
                : <CheckCircle2 aria-hidden="true" size={20} />}
            </h3>
            {!uncertain && (
              <p className="detail-state">
                이 구간의 상태는 <strong>{STATE_LABELS[emotion.operational_state]}</strong>입니다.
              </p>
            )}
          </div>

          {utterance?.transcript && <blockquote>{utterance.transcript}</blockquote>}

          <div className="gauge">
            <div className="gauge-head">
              <span className="field-label">모델 확신도</span>
              <strong>{percent(emotion.confidence)}</strong>
            </div>
            <div className="gauge-track" aria-hidden="true">
              <span className="gauge-fill" style={{ width: percent(emotion.confidence) }} />
              {threshold !== null && (
                <span className="gauge-threshold" style={{ left: percent(threshold) }}>
                  <i />
                  <em>기준 {percent(threshold)}</em>
                </span>
              )}
            </div>
            <p className="gauge-verdict">
              {threshold === null
                ? "보정 정보가 없어 모델의 원시 확신도입니다. 보정된 값으로 해석하지 마세요."
                : uncertain
                  ? `보정 후 확신도가 판단 기준 ${percent(threshold)}에 못 미쳐 감정을 단정하지 않았습니다.`
                  : `보정 후 확신도가 판단 기준 ${percent(threshold)}을 넘어 이 감정을 제시합니다.`}
            </p>
            {emotion.calibration && (
              <p className="gauge-meta">
                temperature {emotion.calibration.temperature} 보정 · 보정 전 {percent(emotion.calibration.raw_confidence)}
              </p>
            )}
          </div>

          <div className="distribution">
            <span className="field-label">
              감정 분포
              {uncertain && <em> · 결론이 아닌 참고 값</em>}
            </span>
            {distribution.map(([label, value], index) => (
              <div
                className={!uncertain && index === 0 ? "bar-row lead" : "bar-row"}
                key={label}
              >
                <span className="bar-label">{EMOTION_LABELS[label]}</span>
                <span className="bar-track" aria-hidden="true">
                  <i style={{ width: Math.max(value * 100, 0.8) + "%" }} />
                </span>
                <span className="bar-value">{percent(value)}</span>
              </div>
            ))}
          </div>

          <footer>
            <span className="field-label">모델</span>
            <span className="mono">{emotion.provider.model || emotion.provider.name}</span>
          </footer>
        </article>
      </div>

      {report.warnings.length > 0 && (
        <div className="notice quiet" role="note">
          <AlertCircle aria-hidden="true" size={18} />
          <div>
            <strong>분석 중 확인된 사항</strong>
            <p>{report.warnings.join(" · ")}</p>
          </div>
        </div>
      )}

      {client ? <ExpertGuidanceCard jobId={report.job_id} client={client} /> : null}

      <footer className="panel-foot">
        <button className="button ghost" type="button" disabled={resetPending} onClick={onReset}>
          {resetPending ? "정리하는 중" : "새 음성 분석"}
        </button>
        <p>새 분석을 시작하면 이 분석의 오디오와 중간 결과가 이 컴퓨터에서 삭제됩니다. 결과가 필요하면 먼저 화면을 저장해 두세요.</p>
      </footer>
    </section>
  );
}
