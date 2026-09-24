import { describe, expect, it } from "vitest";
import type { ReviewTurn, ReviewWarning } from "../../api/types";
import {
  changedTurnCount,
  reviewQuality,
  clockTime,
  isBlockingWarning,
  toEditable,
  toSubmission,
  turnIssues,
  warningCopy,
  type EditableTurn,
} from "./review";

const EMOTIONS = ["happiness", "anger", "neutral", "uncertain"];

function turn(overrides: Partial<ReviewTurn> = {}): ReviewTurn {
  return {
    start: 0,
    end: 2.5,
    speaker: "SPEAKER_00",
    transcript: "환불 처리가 아직 안 됐습니다",
    emotion: "anger",
    emotion_rationale: "높은 음높이",
    confidence: 0.62,
    ...overrides,
  };
}

function editable(overrides: Partial<EditableTurn> = {}): EditableTurn {
  return { ...toEditable(turn()), ...overrides };
}

describe("review turn validation", () => {
  it("accepts a draft turn as loaded", () => {
    expect(turnIssues([editable()], EMOTIONS)).toEqual([]);
  });

  it("refuses a blank transcript rather than promoting an empty utterance", () => {
    const [issue] = turnIssues([editable({ transcript: "   " })], EMOTIONS);
    expect(issue).toMatchObject({ index: 0, field: "transcript" });
  });

  it("refuses a blank speaker", () => {
    expect(turnIssues([editable({ speaker: " " })], EMOTIONS)[0]?.field).toBe("speaker");
  });

  it("refuses an emotion label the backend would not accept", () => {
    expect(turnIssues([editable({ emotion: "frustrated" })], EMOTIONS)[0]?.field).toBe("emotion");
  });

  it("refuses a range that ends at or before it starts", () => {
    expect(turnIssues([editable({ start: "4", end: "4" })], EMOTIONS)[0]?.field).toBe("end");
    expect(turnIssues([editable({ start: "4", end: "3" })], EMOTIONS)[0]?.field).toBe("end");
  });

  it("refuses a negative start", () => {
    expect(turnIssues([editable({ start: "-1" })], EMOTIONS)[0]?.field).toBe("start");
  });

  it("refuses a confidence outside zero to one", () => {
    expect(turnIssues([editable({ confidence: "1.4" })], EMOTIONS)[0]?.field).toBe("confidence");
    expect(turnIssues([editable({ confidence: "-0.1" })], EMOTIONS)[0]?.field).toBe("confidence");
  });

  it("treats a half-typed number as not yet valid instead of coercing it to zero", () => {
    expect(turnIssues([editable({ start: "" })], EMOTIONS)[0]?.field).toBe("start");
    expect(turnIssues([editable({ confidence: "abc" })], EMOTIONS)[0]?.field).toBe("confidence");
  });

  it("reports the position of every invalid turn", () => {
    const issues = turnIssues([editable(), editable({ emotion: "frustrated" })], EMOTIONS);
    expect(issues).toHaveLength(1);
    expect(issues[0]?.index).toBe(1);
  });
});

describe("submission building", () => {
  it("returns null while anything is invalid, so nothing unchecked is sent", () => {
    expect(toSubmission([editable({ transcript: "" })], EMOTIONS)).toBeNull();
  });

  it("parses the typed numbers and trims the speaker", () => {
    const submission = toSubmission(
      [editable({ start: " 1.5 ", end: "3", speaker: " SPEAKER_01 ", confidence: "0.9" })],
      EMOTIONS,
    );
    expect(submission).toEqual([
      {
        start: 1.5,
        end: 3,
        speaker: "SPEAKER_01",
        transcript: "환불 처리가 아직 안 됐습니다",
        emotion: "anger",
        emotion_rationale: "높은 음높이",
        confidence: 0.9,
      },
    ]);
  });

  it("keeps the transcript verbatim, including its surrounding spacing", () => {
    const submission = toSubmission([editable({ transcript: " 네 알겠습니다 " })], EMOTIONS);
    expect(submission?.[0]?.transcript).toBe(" 네 알겠습니다 ");
  });
});

describe("edit accounting", () => {
  it("counts nothing when the reviewer changed nothing", () => {
    const original = [turn(), turn({ start: 3, end: 5 })];
    expect(changedTurnCount(original, original.map(toEditable))).toBe(0);
  });

  it("counts each turn the reviewer touched, once", () => {
    const original = [turn(), turn({ start: 3, end: 5 })];
    const edited = original.map(toEditable);
    edited[0] = { ...edited[0]!, transcript: "환불 처리 안 됐습니다", emotion: "neutral" };
    expect(changedTurnCount(original, edited)).toBe(1);
  });
});

describe("review-quality triage", () => {
  it("prioritises low-confidence, short, and overlapping turns without calling them errors", () => {
    const quality = reviewQuality([
      editable({ start: "0", end: "2", confidence: "0.9" }),
      editable({ start: "1.5", end: "2.2", confidence: "0.4", emotion: "uncertain" }),
    ]);
    expect(quality.uncertainPositions).toEqual([1]);
    expect(quality.shortPositions).toEqual([1]);
    expect(quality.overlapPositions).toEqual([0, 1]);
    expect(quality.priorityPositions).toEqual([0, 1]);
  });
});

describe("warning copy", () => {
  const warning = (overrides: Partial<ReviewWarning>): ReviewWarning => ({
    code: "review_required",
    detail: "english fallback",
    count: null,
    by_rule: null,
    ...overrides,
  });

  it("states the dropped-turn count and the rules that dropped them", () => {
    const copy = warningCopy(
      warning({
        code: "salvaged_dropped_turns",
        count: 31,
        by_rule: { "turn.exceeds_duration": 31 },
      }),
    );
    expect(copy.title).toContain("31");
    expect(copy.detail).toContain("turn.exceeds_duration 31건");
  });

  it("keeps an unmapped code visible with the backend's own text", () => {
    const copy = warningCopy(warning({ code: "something_new", detail: "english fallback" }));
    expect(copy.title).toBe("something_new");
    expect(copy.detail).toBe("english fallback");
  });

  it("treats an existing gold artifact as blocking and review_required as not", () => {
    expect(isBlockingWarning("gold_already_exists")).toBe(true);
    expect(isBlockingWarning("review_required")).toBe(false);
  });
});

describe("clock formatting", () => {
  it("renders seconds as mm:ss", () => {
    expect(clockTime(0)).toBe("00:00");
    expect(clockTime(75.9)).toBe("01:15");
  });

  it("refuses to invent a time for a value that is not one", () => {
    expect(clockTime(Number.NaN)).toBe("--:--");
    expect(clockTime(-1)).toBe("--:--");
  });
});
