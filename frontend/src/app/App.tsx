import { AudioLines, LockKeyhole, TriangleAlert, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api as defaultApi } from "../api/client";
import { describeError, isMissingJob, type ErrorGuidance } from "../api/errors";
import type { AnalysisReport, ApiClient, PublicJob, StageName } from "../api/types";
import { AudioInput } from "../features/input/AudioInput";
import { PrivacyNote, privacyBadgeText } from "../features/privacy/PrivacyNote";
import { needsAcknowledgement, readPosture, type PrivacyPosture } from "../features/privacy/posture";
import { AnalysisProgress } from "../features/progress/AnalysisProgress";
import { RoleConfirmation } from "../features/progress/RoleConfirmation";
import { AnalysisResult } from "../features/result/AnalysisResult";
import { SilverReview } from "../features/review/SilverReview";

interface AppProps {
  client?: ApiClient;
}

const POLL_INTERVAL_MS = 900;

/** The two things this app does. Review is a mode of the same app rather than a separate
 *  one: it reads the same local backend behind the same capability, and a reviewer moving
 *  between an analysis and a draft should not be moving between products. */
type Mode = "analysis" | "review";

export function App({ client = defaultApi }: AppProps) {
  const [mode, setMode] = useState<Mode>("analysis");
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<PublicJob | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [uploadPending, setUploadPending] = useState(false);
  const [actionPending, setActionPending] = useState(false);
  const [error, setError] = useState<ErrorGuidance | null>(null);
  const [reportError, setReportError] = useState<ErrorGuidance | null>(null);
  const [posture, setPosture] = useState<PrivacyPosture>({ kind: "checking" });
  const [acknowledged, setAcknowledged] = useState(false);
  const consentRequired = needsAcknowledgement(posture);
  const submittable = posture.kind !== "checking" && (!consentRequired || acknowledged);
  const reportLoading = useRef(false);

  useEffect(() => {
    let cancelled = false;
    void client.getProviderConfiguration()
      .then((configuration) => { if (!cancelled) setPosture(readPosture(configuration)); })
      .catch(() => { if (!cancelled) setPosture({ kind: "unknown" }); });
    return () => { cancelled = true; };
  }, [client]);

  const loadReport = useCallback((jobId: string) => {
    if (reportLoading.current) return;
    reportLoading.current = true;
    setReportError(null);
    void client.getReport(jobId)
      .then(setReport)
      .catch((reason: unknown) => setReportError(describeError(reason)))
      .finally(() => { reportLoading.current = false; });
  }, [client]);

  useEffect(() => {
    if (!job || report || actionPending) return undefined;

    if (job.stages.report?.status === "completed") {
      if (!reportError) loadReport(job.job_id);
      return undefined;
    }
    // A paused job waits on the person, and a failed one waits on an explicit retry;
    // polling either one only burns requests against a state that cannot move alone.
    if (job.status === "paused" || job.status === "failed") return undefined;

    let cancelled = false;
    const timer = window.setTimeout(() => {
      void client.getJob(job.job_id)
        .then((next) => { if (!cancelled) setJob(next); })
        .catch((reason: unknown) => { if (!cancelled) setError(describeError(reason)); });
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [actionPending, client, job, loadReport, report, reportError]);

  function clearSession() {
    setFile(null);
    // Consent is given for one recording, not for the session.
    setAcknowledged(false);
    setJob(null);
    setReport(null);
    setError(null);
    setReportError(null);
    reportLoading.current = false;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function analyze() {
    // The gate is enforced here as well as on the button: nothing may be uploaded while the
    // transfer question is unanswered, whatever state the control happens to be in.
    if (!file || !submittable) return;
    setUploadPending(true);
    setError(null);
    try {
      const created = await client.createJob(file);
      setJob(await client.getJob(created.job_id));
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setUploadPending(false);
    }
  }

  async function confirmRoles(mapping: Record<string, "customer" | "agent">) {
    if (!job) return;
    setActionPending(true);
    setError(null);
    try {
      setJob(await client.confirmRoles(job.job_id, mapping));
    } catch (reason) {
      setError(describeError(reason));
      // The mapping may already have been recorded before the failure, so this screen's
      // paused copy of the job is no longer trustworthy. Re-read it rather than keep
      // offering a gate the backend has closed; a second submission would only 409.
      try {
        setJob(await client.getJob(job.job_id));
      } catch {
        // Leave the existing error visible: a failed refresh says nothing new.
      }
    } finally {
      setActionPending(false);
    }
  }

  async function retry(stage: StageName) {
    if (!job) return;
    setActionPending(true);
    setError(null);
    try {
      setJob(await client.retryStage(job.job_id, stage));
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setActionPending(false);
    }
  }

  /** Starting over always discards the local job, so the audio and intermediate
   *  artifacts this PoC wrote to disk do not outlive the screen that produced them. */
  async function discard() {
    if (!job) {
      clearSession();
      return;
    }
    setActionPending(true);
    setError(null);
    try {
      await client.deleteJob(job.job_id);
    } catch (reason) {
      if (!isMissingJob(reason)) {
        setError(describeError(reason));
        setActionPending(false);
        return;
      }
    }
    setActionPending(false);
    clearSession();
  }

  const busy = uploadPending || actionPending;
  const started = Boolean(job) || uploadPending || Boolean(report);
  const working = busy || Boolean(job && !report);
  const pausedForRoles = job?.stages.confirm_roles?.status === "paused" ? job.role_candidate : null;
  // The analysis keeps its state while the reviewer is elsewhere. Polling stays on, so
  // switching back shows where the job actually got to rather than where it was left.
  const reviewing = mode === "review";

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="VoxDelta 처음으로">
          <span><AudioLines aria-hidden="true" size={21} /></span>
          <strong>VoxDelta</strong>
        </a>
        <nav className="mode-switch" aria-label="화면 전환">
          <button
            type="button"
            className={mode === "analysis" ? "mode-tab selected" : "mode-tab"}
            aria-pressed={mode === "analysis"}
            onClick={() => setMode("analysis")}
          >
            분석
          </button>
          <button
            type="button"
            className={mode === "review" ? "mode-tab selected" : "mode-tab"}
            aria-pressed={mode === "review"}
            onClick={() => setMode("review")}
          >
            Silver 검수
          </button>
        </nav>
        <span className={posture.kind === "local" ? "badge" : "badge alert"}>
          <LockKeyhole aria-hidden="true" size={13} /> {privacyBadgeText(posture)}
        </span>
      </header>

      <main>
        {reviewing ? (
          <div className="hero collapsed">
            <span className="kicker">SILVER REVIEW</span>
            <h1>사람이 확인한 것만 정답이 됩니다.</h1>
          </div>
        ) : (
          <div className={started ? "hero collapsed" : "hero"}>
            <span className="kicker">KOREAN SPEECH EMOTION</span>
            {started ? (
              <h1>목소리 속 감정, 과장 없이.</h1>
            ) : (
              <>
                <h1>목소리 속 감정,<br />과장 없이 읽어냅니다.</h1>
                <p>감정과 보정된 확신도를 함께 보여주고, 애매한 순간은 애매하다고 말하는 로컬 음성 분석 도구입니다.</p>
              </>
            )}
          </div>
        )}

        {reviewing && <SilverReview client={client} />}

        {!reviewing && error && (
          <div className="notice alert global" role="alert">
            <TriangleAlert aria-hidden="true" size={18} />
            <div><strong>{error.title}</strong><p>{error.detail}</p></div>
            <button className="icon-button" type="button" aria-label="오류 닫기" onClick={() => setError(null)}>
              <X aria-hidden="true" size={16} />
            </button>
          </div>
        )}

        {!reviewing && !report && !job && (
          <AudioInput
            disabled={uploadPending}
            file={file}
            posture={posture}
            acknowledged={acknowledged}
            submittable={submittable}
            onAcknowledge={setAcknowledged}
            onFile={setFile}
            onAnalyze={analyze}
          />
        )}
        {!reviewing && !report && (job || uploadPending) && (
          <AnalysisProgress
            job={job}
            uploadPending={uploadPending}
            busy={busy}
            reportError={reportError}
            onRetry={retry}
            onReloadReport={() => job && loadReport(job.job_id)}
            onDiscard={discard}
            attached={Boolean(pausedForRoles)}
          />
        )}
        {!reviewing && !report && pausedForRoles && job && (
          <RoleConfirmation
            jobId={job.job_id}
            client={client}
            candidate={pausedForRoles}
            busy={busy}
            onConfirm={confirmRoles}
            onDiscard={discard}
          />
        )}
        {!reviewing && report && (
          <AnalysisResult report={report} client={client} onReset={discard} resetPending={actionPending} />
        )}

        {!reviewing && working && !pausedForRoles && !report && (
          <p className="footnote">
            분석은 로컬 백엔드에서 계속됩니다. 다만 이 페이지를 새로 고치면 진행 중인 분석을 다시 열 수 없으니, 끝날 때까지 이 화면을 열어 두세요.
          </p>
        )}

        {!reviewing && report && <PrivacyNote posture={posture} />}
      </main>

      <footer className="site-footer">
        <span>VoxDelta / Proof of Concept</span>
        {/* No standing local-inference claim here: the header badge and the transfer notice
            are the only places that answer that, and they answer it from the disclosure. */}
        <span>Calibrated XLS-R</span>
      </footer>
    </div>
  );
}
