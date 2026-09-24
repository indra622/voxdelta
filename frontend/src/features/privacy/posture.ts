import type { ProviderConfiguration } from "../../api/types";
import {
  localStageLabels,
  remoteTransfers,
  transcriptStaysLocal,
  type RemoteTransfer,
} from "./transfer";

export type PrivacyPosture =
  | { kind: "checking" }
  | { kind: "local"; stageCount: number }
  | {
      kind: "remote";
      transfers: RemoteTransfer[];
      localStages: string[];
      transcriptStaysLocal: boolean;
    }
  | { kind: "unknown" };

/** Read the local-only claim off the backend instead of asserting it in copy.
 *
 * A stage counts as leaving this machine when its provider is remote or when it
 * declares that it transmits anything at all; either one makes the "외부로 전송되지
 * 않습니다" promise false, so the UI has to say so rather than repeat it.
 */
export function readPosture(configuration: ProviderConfiguration): PrivacyPosture {
  const transfers = remoteTransfers(configuration);
  if (transfers.length > 0) {
    return {
      kind: "remote",
      transfers,
      localStages: localStageLabels(configuration),
      transcriptStaysLocal: transcriptStaysLocal(transfers),
    };
  }
  return { kind: "local", stageCount: configuration.stages.length };
}

/** Whether audio may leave this machine, or whether that cannot be ruled out.
 *
 * `checking` is not yet an answer and `unknown` is a failure to get one; neither may be
 * treated as a local-only run, so both hold submission back. */
export function needsAcknowledgement(posture: PrivacyPosture): boolean {
  return posture.kind === "remote" || posture.kind === "unknown";
}
