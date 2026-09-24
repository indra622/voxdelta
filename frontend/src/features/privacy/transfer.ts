import type { ProviderConfiguration, ProviderDisclosure, StageName } from "../../api/types";
import { STAGE_LABELS, TRANSMIT_LABELS, providerLabel } from "../../labels";

/** One pipeline stage that moves data off this machine, named the way it will be said aloud. */
export interface RemoteTransfer {
  stage: StageName;
  stageLabel: string;
  provider: string;
  model: string;
  /** Localized data kinds, in the order the backend declared them. */
  transmits: string[];
  retentionPolicyUrl: string | null;
  retentionWindowHours: number | null;
}

/** A stage leaves this machine when its provider is remote or it declares any transmission.
 *
 * Either one is enough: a local model that ships features away is no more local-only than a
 * hosted one, so both have to be disclosed under the same rule. */
export function leavesThisMachine(disclosure: ProviderDisclosure): boolean {
  return disclosure.provenance?.remote === true || disclosure.transmits.length > 0;
}

export function remoteTransfers(configuration: ProviderConfiguration): RemoteTransfer[] {
  return configuration.stages.filter(leavesThisMachine).map((disclosure) => ({
    stage: disclosure.stage,
    stageLabel: STAGE_LABELS[disclosure.stage] ?? disclosure.stage,
    provider: providerLabel(disclosure.provenance?.name ?? "알 수 없는 provider"),
    model: disclosure.provenance?.model ?? "",
    transmits: disclosure.transmits.map((item) => TRANSMIT_LABELS[item] ?? item),
    retentionPolicyUrl: disclosure.retention_policy_url,
    retentionWindowHours: disclosure.retention_window_hours,
  }));
}

/** Stages that stay here, so the notice can bound the transfer instead of only naming it. */
export function localStageLabels(configuration: ProviderConfiguration): string[] {
  return configuration.stages
    .filter((disclosure) => disclosure.provenance !== null && !leavesThisMachine(disclosure))
    .map((disclosure) => STAGE_LABELS[disclosure.stage] ?? disclosure.stage);
}

/** True when no declared transfer carries transcript text anywhere.
 *
 * The remote diarization request disables transcription, so this holds today; it is read
 * back off the disclosure rather than asserted, because only the backend can promise it. */
export function transcriptStaysLocal(transfers: RemoteTransfer[]): boolean {
  const text = TRANSMIT_LABELS.text;
  return transfers.every((transfer) => !transfer.transmits.includes(text));
}

export function describeTransfer(transfer: RemoteTransfer): string {
  return transfer.transmits.length > 0
    ? `${transfer.stageLabel}(${transfer.transmits.join(", ")} → ${transfer.provider} 전송)`
    : `${transfer.stageLabel}(원격 모델 ${transfer.provider})`;
}
