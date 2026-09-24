import { ExternalLink, ShieldAlert, ShieldQuestion } from "lucide-react";
import type { PrivacyPosture } from "./posture";

interface TransferConsentProps {
  posture: PrivacyPosture;
  acknowledged: boolean;
  disabled: boolean;
  onAcknowledge(value: boolean): void;
}

/** The pre-submission transfer disclosure and its acknowledgement.
 *
 * This is the only thing standing between a recording and a third party, so it says the
 * provider's name, the data, and the retention window in the first two sentences rather
 * than folding them into the general privacy copy at the bottom of the panel. Every fact
 * comes from `GET /api/config/providers`; nothing here is asserted from the client's own
 * idea of how the backend is configured.
 */
export function TransferConsent({ posture, acknowledged, disabled, onAcknowledge }: TransferConsentProps) {
  if (posture.kind === "local" || posture.kind === "checking") return null;

  if (posture.kind === "unknown") {
    return (
      <section className="consent" aria-labelledby="consent-heading">
        <div className="consent-head">
          <ShieldQuestion aria-hidden="true" size={20} />
          <div>
            <span className="kicker">전송 고지</span>
            <h3 id="consent-heading">오디오가 외부로 나가는지 확인하지 못했습니다</h3>
          </div>
        </div>
        <p className="consent-body">
          백엔드의 provider 설정을 읽지 못해, 이 분석이 로컬에서만 실행된다고 보증할 수 없습니다. 그대로 진행하면
          오디오가 외부 서비스로 전송될 수도 있습니다.
        </p>
        <label className="consent-ack">
          <input
            type="checkbox"
            checked={acknowledged}
            disabled={disabled}
            onChange={(event) => onAcknowledge(event.target.checked)}
          />
          <span>전송 여부를 확인할 수 없는 상태라는 점을 이해했고, 그래도 분석을 진행합니다.</span>
        </label>
      </section>
    );
  }

  const providers = [...new Set(posture.transfers.map((transfer) => transfer.provider))];
  const scope = [
    posture.transcriptStaysLocal ? "전사문은 어느 단계에서도 전송되지 않습니다." : "",
    posture.localStages.length > 0
      ? `나머지 단계(${posture.localStages.join(", ")})는 이 컴퓨터에서만 실행됩니다.`
      : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <section className="consent alert" aria-labelledby="consent-heading">
      <div className="consent-head">
        <ShieldAlert aria-hidden="true" size={20} />
        <div>
          <span className="kicker">전송 고지</span>
          <h3 id="consent-heading">이 분석은 오디오를 {providers.join(", ")}로 보냅니다</h3>
        </div>
      </div>

      <ul className="consent-list">
        {posture.transfers.map((transfer) => (
          <li key={transfer.stage}>
            <strong>
              {transfer.stageLabel} 단계 · {transfer.provider}
              {transfer.model && <span className="mono"> {transfer.model}</span>}
            </strong>
            <p>
              {transfer.transmits.length > 0
                ? `${transfer.transmits.join(", ")} — 이 컴퓨터를 떠나 ${transfer.provider} API로 전송됩니다.`
                : `${transfer.provider}의 원격 모델이 이 단계를 처리합니다.`}
            </p>
            <p>
              {transfer.retentionWindowHours !== null
                ? `전송된 입력은 ${transfer.provider} Media API에 최대 ${transfer.retentionWindowHours}시간까지 남아 있을 수 있고, 공개 API에는 이를 지우는 방법이 없습니다.`
                : `${transfer.provider}가 전송된 입력을 얼마나 보관하는지는 이 화면에서 확인할 수 없습니다.`}
              {transfer.retentionPolicyUrl && (
                <>
                  {" "}
                  <a href={transfer.retentionPolicyUrl} target="_blank" rel="noreferrer noopener">
                    보관 정책 원문 <ExternalLink aria-hidden="true" size={12} />
                  </a>
                </>
              )}
            </p>
          </li>
        ))}
      </ul>

      {scope && <p className="consent-scope">{scope}</p>}

      <label className="consent-ack">
        <input
          type="checkbox"
          checked={acknowledged}
          disabled={disabled}
          onChange={(event) => onAcknowledge(event.target.checked)}
        />
        <span>
          위 내용을 확인했고, 이 오디오를 {providers.join(", ")}로 전송하는 데 동의합니다.
        </span>
      </label>
    </section>
  );
}
