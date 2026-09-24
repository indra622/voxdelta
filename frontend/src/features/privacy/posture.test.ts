import { describe, expect, it } from "vitest";
import type { ProviderConfiguration, StageName } from "../../api/types";
import { needsAcknowledgement, readPosture } from "./posture";
import { describeTransfer } from "./transfer";

interface StageOverride {
  name?: string;
  remote?: boolean;
  transmits?: string[];
  retentionPolicyUrl?: string | null;
  retentionWindowHours?: number | null;
}

const ALL_STAGES: StageName[] = [
  "normalize",
  "diarize",
  "transcribe",
  "confirm_roles",
  "emotion",
  "response_strategy",
  "transitions",
  "report",
];

function configuration(overrides: Partial<Record<StageName, StageOverride>>): ProviderConfiguration {
  return {
    stages: ALL_STAGES.map((stage) => {
      const override = overrides[stage];
      return {
        stage,
        provenance: {
          name: override?.name ?? stage,
          model: "local",
          remote: override?.remote ?? false,
          schema_version: "1",
          revision: null,
        },
        transmits: override?.transmits ?? [],
        retention_policy_url: override?.retentionPolicyUrl ?? null,
        retention_window_hours: override?.retentionWindowHours ?? null,
      };
    }),
  };
}

describe("privacy posture", () => {
  it("confirms local-only from the backend disclosure rather than from copy", () => {
    const posture = readPosture(configuration({}));
    expect(posture).toEqual({ kind: "local", stageCount: 8 });
    expect(needsAcknowledgement(posture)).toBe(false);
  });

  it("names the stage, the data, and the provider when a provider is remote", () => {
    const posture = readPosture(
      configuration({ diarize: { name: "pyannoteai", remote: true, transmits: ["audio"] } }),
    );
    if (posture.kind !== "remote") throw new Error("expected a remote posture");
    expect(posture.transfers.map(describeTransfer)).toEqual(["화자 구분(오디오 → pyannoteAI 전송)"]);
    expect(needsAcknowledgement(posture)).toBe(true);
  });

  it("carries the declared retention window so the consent copy can state it", () => {
    const posture = readPosture(
      configuration({
        diarize: {
          name: "pyannoteai",
          remote: true,
          transmits: ["audio"],
          retentionPolicyUrl: "https://docs.pyannote.ai/data-retention",
          retentionWindowHours: 48,
        },
      }),
    );
    if (posture.kind !== "remote") throw new Error("expected a remote posture");
    expect(posture.transfers[0]).toMatchObject({
      provider: "pyannoteAI",
      retentionWindowHours: 48,
      retentionPolicyUrl: "https://docs.pyannote.ai/data-retention",
    });
  });

  it("reports the stages that stay local, so the transfer can be bounded", () => {
    const posture = readPosture(
      configuration({ diarize: { name: "pyannoteai", remote: true, transmits: ["audio"] } }),
    );
    if (posture.kind !== "remote") throw new Error("expected a remote posture");
    expect(posture.localStages).toContain("음성 전사");
    expect(posture.localStages).not.toContain("화자 구분");
    expect(posture.transcriptStaysLocal).toBe(true);
  });

  it("stops claiming the transcript stays local once a stage transmits text", () => {
    const posture = readPosture(
      configuration({ transcribe: { name: "hosted", remote: true, transmits: ["audio", "text"] } }),
    );
    if (posture.kind !== "remote") throw new Error("expected a remote posture");
    expect(posture.transcriptStaysLocal).toBe(false);
  });

  it("treats a local model that still transmits as not local-only", () => {
    expect(readPosture(configuration({ emotion: { remote: false, transmits: ["features"] } })).kind)
      .toBe("remote");
  });

  it("ignores stages that run without any provider", () => {
    const empty: ProviderConfiguration = {
      stages: [
        {
          stage: "report",
          provenance: null,
          transmits: [],
          retention_policy_url: null,
          retention_window_hours: null,
        },
      ],
    };
    expect(readPosture(empty)).toEqual({ kind: "local", stageCount: 1 });
  });

  it("requires an acknowledgement when the disclosure could not be read at all", () => {
    expect(needsAcknowledgement({ kind: "unknown" })).toBe(true);
    expect(needsAcknowledgement({ kind: "checking" })).toBe(false);
  });
});
