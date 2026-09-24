import { AlertCircle, Bot, CheckCircle2, Send } from "lucide-react";
import { useEffect, useState } from "react";
import type { ApiClient, ExpertGuidanceStatus } from "../../api/types";

interface ExpertGuidanceCardProps {
  jobId: string;
  client: ApiClient;
}

const PENDING: ExpertGuidanceStatus = {
  status: "not_requested",
  target: null,
  transport: null,
  request_sha256: null,
  transcription_uncertain: null,
  evidence_turn_count: 0,
  guidance: null,
};

/** Expert mode never infers a diagnosis in the browser. It only renders a response which
 * the API has already tied to a bounded evidence packet and contract-validated. */
export function ExpertGuidanceCard({ jobId, client }: ExpertGuidanceCardProps) {
  const [status, setStatus] = useState<ExpertGuidanceStatus>(PENDING);
  const [target, setTarget] = useState<"claude" | "codex">("claude");
  const [acknowledged, setAcknowledged] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!client.getExpertGuidance) return undefined;
    void client.getExpertGuidance(jobId)
      .then((next) => { if (!cancelled) setStatus(next); })
      .catch(() => { if (!cancelled) setError("Expert mode 상태를 불러오지 못했습니다."); });
    return () => { cancelled = true; };
  }, [client, jobId]);

  async function requestGuidance() {
    if (!client.requestExpertGuidance || !acknowledged) return;
    setPending(true);
    setError(null);
    try {
      setStatus(await client.requestExpertGuidance(jobId, {
        target,
        acknowledge_text_transfer: true,
      }));
    } catch {
      setError("Expert mode 요청을 만들지 못했습니다. 분석 결과는 변경되지 않았습니다.");
    } finally {
      setPending(false);
    }
  }

  if (!client.getExpertGuidance || !client.requestExpertGuidance) return null;

  const guidance = status.guidance;
  return (
    <section className="expert-card" aria-labelledby="expert-guidance-heading">
      <header>
        <div>
          <span className="kicker">EXPERT MODE</span>
          <h3 id="expert-guidance-heading"><Bot aria-hidden="true" size={19} /> 근거 기반 대응 제안</h3>
        </div>
        {guidance ? <span className="badge"><CheckCircle2 aria-hidden="true" size={14} /> 검증 완료</span> : null}
      </header>

      {guidance ? (
        <div className="expert-guidance">
          <div>
            <span className="field-label">관찰된 흐름</span>
            <ul>{guidance.observations.map((item, index) => <li key={index}>{item.statement}</li>)}</ul>
          </div>
          {guidance.hypotheses.length > 0 ? (
            <div>
              <span className="field-label">가능한 원인 · 단정 금지</span>
              <ul>{guidance.hypotheses.map((item, index) => <li key={index}>{item.statement}</li>)}</ul>
            </div>
          ) : null}
          <blockquote>{guidance.suggested_message}</blockquote>
          <p className="expert-question"><strong>다음 확인 질문</strong>{guidance.next_question}</p>
          <p className="expert-meta">안전 수준: {guidance.safety_level === "watch" ? "확인 필요" : guidance.safety_level === "urgent" ? "즉시 확인" : "일반"}</p>
        </div>
      ) : status.status === "queued" ? (
        <div className="expert-queued">
          <AlertCircle aria-hidden="true" size={18} />
          <p><strong>{status.target === "claude" ? "Claude" : "Codex"} 해석 요청을 준비했어요.</strong> 근거 발화 {status.evidence_turn_count}개만 ACP 작업으로 전달됩니다. 완료된 응답이 검증되면 이 카드에 표시됩니다.</p>
        </div>
      ) : (
        <div className="expert-request">
          <p>분석 결과의 강한 감정·낮은 확신도 구간만 최대 6개를 골라, 원인 가설과 상담 대응 문구를 요청합니다. 음성은 전송하지 않습니다.</p>
          <label className="expert-target">해석 대상
            <select value={target} onChange={(event) => setTarget(event.target.value as "claude" | "codex")}>
              <option value="claude">Claude</option>
              <option value="codex">Codex</option>
            </select>
          </label>
          <label className="expert-ack">
            <input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} />
            근거 전사 텍스트가 연결된 Expert 세션으로 전달됨을 확인했습니다.
          </label>
          <button className="button" type="button" disabled={!acknowledged || pending} onClick={requestGuidance}>
            <Send aria-hidden="true" size={15} /> {pending ? "요청 만드는 중" : "Expert 대응 제안 요청"}
          </button>
        </div>
      )}
      {status.transcription_uncertain ? <p className="expert-caveat"><AlertCircle aria-hidden="true" size={15} /> 전사 반영이 불확실한 구간이 있어, 제안은 확인 질문을 우선합니다.</p> : null}
      {error ? <p className="field-error">{error}</p> : null}
      <p className="expert-policy">음성은 전송하지 않으며, 근거 발화와 불확실성 정보만 명시적 요청 시 전달됩니다.</p>
    </section>
  );
}
