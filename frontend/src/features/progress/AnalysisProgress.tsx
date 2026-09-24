import { Check, LoaderCircle, RotateCcw, TriangleAlert } from "lucide-react";
import type { ErrorGuidance } from "../../api/errors";
import { STAGE_NAMES } from "../../api/types";
import type { PublicJob, StageName, StageStatus } from "../../api/types";
import { STAGE_LABELS } from "../../labels";

const STATUS_LABELS: Record<StageStatus, string> = {
  pending: "대기",
  running: "진행 중",
  paused: "확인 필요",
  completed: "완료",
  failed: "멈춤",
  skipped: "생략",
};

const SETTLED: ReadonlySet<StageStatus> = new Set<StageStatus>(["completed", "skipped"]);

interface AnalysisProgressProps {
  job: PublicJob | null;
  uploadPending: boolean;
  busy: boolean;
  reportError: ErrorGuidance | null;
  onRetry(stage: StageName): void;
  onReloadReport(): void;
  onDiscard(): void;
  /** True while the role checkpoint is open below; the two render as one card. */
  attached?: boolean;
}

export function AnalysisProgress({
  job,
  uploadPending,
  busy,
  reportError,
  onRetry,
  onReloadReport,
  onDiscard,
  attached = false,
}: AnalysisProgressProps) {
  const settled = job
    ? STAGE_NAMES.filter((name) => SETTLED.has(job.stages[name]?.status)).length
    : 0;
  // Report only stages that actually settled: a decorative sliver during upload
  // reads as progress the pipeline has not made, and flickers back to 0 on first poll.
  const percentage = Math.round((settled / STAGE_NAMES.length) * 100);
  const failed = job ? STAGE_NAMES.find((name) => job.stages[name]?.status === "failed") : undefined;
  const active = job
    ? STAGE_NAMES.find((name) => job.stages[name]?.status === "running" || job.stages[name]?.status === "paused")
    : undefined;
  // A job can terminalize without any single stage owning the failure.
  const stalled = job?.status === "failed" && !failed;
  const focus = failed ?? active;
  // The track runs node-centre to node-centre, so the teal length is the run of
  // settled nodes it connects — not the completion fraction, which the count and
  // the progressbar value already carry.
  const connector = settled > 1 ? ((settled - 1) / (STAGE_NAMES.length - 1)) * 100 : 0;

  const paused = Boolean(active && job?.stages[active]?.status === "paused");
  const headline = failed
    ? STAGE_LABELS[failed] + " 단계에서 멈췄어요"
    : stalled
      ? "분석이 중단됐어요"
      : busy
        ? "로컬 모델이 계산하고 있어요"
        : paused && active
          ? STAGE_LABELS[active] + "이 필요해요"
          : active
            ? STAGE_LABELS[active] + " 단계를 진행하고 있어요"
            : uploadPending
              ? "음성을 올리고 있어요"
              : "분석을 준비하고 있어요";

  const support = paused
    ? "아래에서 두 화자의 역할을 확인하면 남은 단계가 이어집니다."
    : failed || stalled
    ? "아래에서 다시 시도하거나, 새 음성으로 시작할 수 있어요."
    : busy
      ? "이 단계는 로컬에서 실행되므로 진행률이 잠시 멈춘 것처럼 보일 수 있어요."
      : "여덟 단계 중 " + settled + "단계가 끝났어요.";

  const announcement = uploadPending && !job
    ? "음성을 업로드하는 중입니다."
    : failed
      ? STAGE_LABELS[failed] + " 단계에서 분석이 멈췄습니다."
      : stalled
        ? "분석이 중단됐습니다."
        : active
          ? "분석 진행률 " + percentage + "퍼센트, 현재 " + STAGE_LABELS[active] + " 단계입니다."
          : "분석 진행률 " + percentage + "퍼센트입니다.";

  return (
    <section className={attached ? "panel progress-panel attached" : "panel progress-panel"} aria-labelledby="progress-heading">
      <header className="panel-head">
        <div>
          <span className="kicker">02 / ANALYSIS</span>
          <h2 id="progress-heading">{headline}</h2>
          <p>{support}</p>
        </div>
        <span className="progress-count" aria-hidden="true">
          <strong>{settled}</strong>/{STAGE_NAMES.length}
        </span>
      </header>

      <p className="visually-hidden" aria-live="polite">{announcement}</p>

      <div
        className="rail"
        role="progressbar"
        aria-label="분석 진행률"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percentage}
        aria-valuetext={percentage + "퍼센트"}
      >
        <span className="rail-track" aria-hidden="true">
          <span className="rail-fill" style={{ width: connector + "%" }} />
        </span>
        <ol className="rail-nodes">
          {STAGE_NAMES.map((name, index) => {
            const status: StageStatus =
              job?.stages[name]?.status ?? (uploadPending && index === 0 ? "running" : "pending");
            return (
              <li key={name} className={"rail-node " + status + (name === focus ? " focus" : "")}>
                <span className="rail-dot" aria-hidden="true">
                  {SETTLED.has(status) ? <Check size={13} strokeWidth={3} />
                    : status === "running" ? <LoaderCircle className="spin" size={14} />
                    : status === "failed" ? <TriangleAlert size={13} />
                    : index + 1}
                </span>
                <span className="rail-label">
                  {STAGE_LABELS[name]}
                  <small>{STATUS_LABELS[status]}</small>
                </span>
              </li>
            );
          })}
        </ol>
      </div>

      {failed && job && (
        <div className="notice alert" role="alert">
          <TriangleAlert aria-hidden="true" size={18} />
          <div>
            <strong>{STAGE_LABELS[failed]} 단계에서 분석이 멈췄습니다</strong>
            <p>오류 코드: <span className="mono">{job.stages[failed].error?.code ?? "pipeline_failed"}</span></p>
          </div>
          <button className="button ghost" type="button" disabled={busy} onClick={() => onRetry(failed)}>
            <RotateCcw aria-hidden="true" size={15} /> {busy ? "다시 시도하는 중" : "다시 시도"}
          </button>
        </div>
      )}

      {stalled && (
        <div className="notice alert" role="alert">
          <TriangleAlert aria-hidden="true" size={18} />
          <div>
            <strong>분석이 중단됐습니다</strong>
            <p>어느 단계도 이어서 실행할 수 없습니다. 새 음성으로 다시 시작해 주세요.</p>
          </div>
        </div>
      )}

      {reportError && (
        <div className="notice alert" role="alert">
          <TriangleAlert aria-hidden="true" size={18} />
          <div>
            <strong>{reportError.title}</strong>
            <p>{reportError.detail}</p>
          </div>
          <button className="button ghost" type="button" disabled={busy} onClick={onReloadReport}>
            <RotateCcw aria-hidden="true" size={15} /> 결과 다시 불러오기
          </button>
        </div>
      )}

      {job && !attached && (
        <footer className="panel-foot">
          <button className="button ghost" type="button" disabled={busy} onClick={onDiscard}>
            {busy ? "정리하는 중" : "분석 중단하고 삭제"}
          </button>
          <p>진행 중인 단계가 끝나는 대로 중단하고, 로컬에 저장된 오디오와 중간 결과를 삭제합니다.</p>
        </footer>
      )}
    </section>
  );
}
