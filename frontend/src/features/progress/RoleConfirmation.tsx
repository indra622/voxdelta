import { ArrowLeftRight, Headphones, LoaderCircle, Play, Square, TriangleAlert, UserRound } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { describeError, type ErrorGuidance } from "../../api/errors";
import type { ApiClient, RoleCandidate, RoleSample } from "../../api/types";

interface RoleConfirmationProps {
  jobId: string;
  client: ApiClient;
  candidate: RoleCandidate;
  busy: boolean;
  onConfirm(mapping: Record<string, "customer" | "agent">): void;
  onDiscard(): void;
}

export function RoleConfirmation({ jobId, client, candidate, busy, onConfirm, onDiscard }: RoleConfirmationProps) {
  const [swapped, setSwapped] = useState(false);
  const [activeSample, setActiveSample] = useState<string | null>(null);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const [sampleError, setSampleError] = useState<ErrorGuidance | null>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const baseCustomer = useMemo(() => {
    const suggested = Object.entries(candidate.suggested_mapping ?? {}).find(([, role]) => role === "customer")?.[0];
    return suggested ?? candidate.speakers[0];
  }, [candidate]);

  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  const stopSample = useCallback(() => {
    audioRef.current?.pause();
    audioRef.current?.removeAttribute("src");
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    setActiveSample(null);
    setLoadingSample(null);
  }, []);

  useEffect(() => () => stopSample(), [stopSample]);

  const playSample = useCallback((sample: RoleSample) => {
    const key = `${sample.speaker_id}:${sample.index}`;
    if (activeSample === key || loadingSample === key) {
      stopSample();
      return;
    }
    if (!client.getRoleSampleClip) return;
    stopSample();
    setSampleError(null);
    setLoadingSample(key);
    void client.getRoleSampleClip(jobId, sample.speaker_id, sample.index)
      .then((blob) => {
        const element = audioRef.current ?? new Audio();
        audioRef.current = element;
        const url = URL.createObjectURL(blob);
        objectUrlRef.current = url;
        element.src = url;
        element.onended = stopSample;
        return element.play();
      })
      .then(() => {
        setLoadingSample(null);
        setActiveSample(key);
      })
      .catch((reason: unknown) => {
        stopSample();
        setSampleError(describeError(reason));
      });
  }, [activeSample, client, jobId, loadingSample, stopSample]);

  // The pipeline only accepts a mapping over exactly the two observed speakers, so
  // anything else has to be said outright rather than sent and rejected.
  const pairable = candidate.speakers.length === 2;
  const baseAgent = candidate.speakers.find((speaker) => speaker !== baseCustomer);
  const customer = swapped ? baseAgent : baseCustomer;
  const agent = swapped ? baseCustomer : baseAgent;
  const hasSamples = candidate.speakers.some((speaker) => (candidate.samples?.[speaker] ?? []).length > 0);

  return (
    <section className="panel role-panel" aria-labelledby="role-heading">
      <header className="panel-head">
        <div>
          <span className="kicker">CHECKPOINT</span>
          <h2 id="role-heading" ref={headingRef} tabIndex={-1}>두 목소리의 역할을 확인해 주세요</h2>
          <p>고객 목소리에만 감정 분석을 적용합니다. 잘못 배정됐다면 한 번 바꿔 주세요.</p>
        </div>
      </header>

      {!pairable ? (
        <div className="notice alert" role="alert">
          <TriangleAlert aria-hidden="true" size={18} />
          <div>
            <strong>화자가 {candidate.speakers.length}명으로 구분됐습니다</strong>
            <p>이 PoC는 고객과 상담원, 정확히 두 사람의 대화만 분석합니다. 두 사람의 목소리가 또렷하게 담긴 녹음으로 다시 시도해 주세요.</p>
          </div>
          <button className="button ghost" type="button" disabled={busy} onClick={onDiscard}>
            {busy ? "정리하는 중" : "새 음성 분석"}
          </button>
        </div>
      ) : (
        <>
          <div className="role-grid">
            <div className="role-card customer">
              <UserRound aria-hidden="true" size={20} />
              <span>고객</span>
              <strong>{customer}</strong>
            </div>
            <button className="swap-button" type="button" onClick={() => setSwapped((value) => !value)} disabled={busy}>
              <ArrowLeftRight aria-hidden="true" size={17} /> 역할 바꾸기
            </button>
            <div className="role-card agent">
              <Headphones aria-hidden="true" size={20} />
              <span>상담원</span>
              <strong>{agent}</strong>
            </div>
          </div>
          {hasSamples && <section className="role-samples" aria-label="화자별 대표 음성 및 전사">
            <div className="role-samples-head">
              <div>
                <span className="kicker">VOICE CHECK</span>
                <h3>대표 구간을 듣고 두 목소리를 비교해 주세요</h3>
              </div>
              <p>전사는 이미 화자 구간에 맞춰 생성됐습니다. 역할 확정 전에도 듣고 대조할 수 있어요.</p>
            </div>
            <div className="role-samples-grid">
              {candidate.speakers.map((speaker) => (
                <article className="role-sample-card" key={speaker}>
                  <strong>{speaker}</strong>
                  {(candidate.samples?.[speaker] ?? []).map((sample) => {
                    const key = `${sample.speaker_id}:${sample.index}`;
                    const playing = activeSample === key;
                    const loading = loadingSample === key;
                    return (
                      <div className="role-sample" key={key}>
                        <button
                          className="role-sample-play"
                          type="button"
                          disabled={busy || !client.getRoleSampleClip}
                          onClick={() => playSample(sample)}
                          aria-label={`${speaker} 대표 구간 ${sample.index + 1} 듣기`}
                        >
                          {loading ? <LoaderCircle className="spin" aria-hidden="true" size={15} /> : playing ? <Square aria-hidden="true" size={14} /> : <Play aria-hidden="true" size={15} />}
                          {loading ? "불러오는 중" : playing ? "중지" : "듣기"}
                        </button>
                        <div>
                          <span className="mono">{sample.start.toFixed(1)}–{sample.end.toFixed(1)}초</span>
                          <p>“{sample.transcript}”{sample.clip_truncated ? " …" : ""}</p>
                        </div>
                      </div>
                    );
                  })}
                  {(candidate.samples?.[speaker] ?? []).length === 0 && <p className="role-sample-empty">대표 전사 구간이 없습니다.</p>}
                </article>
              ))}
            </div>
            {sampleError && <p className="field-error" role="alert">{sampleError.title}. {sampleError.detail}</p>}
          </section>}
          <button
            className="button primary"
            type="button"
            disabled={busy || !customer || !agent}
            onClick={() => customer && agent && onConfirm({ [customer]: "customer", [agent]: "agent" })}
          >
            {busy ? "확인한 역할로 분석하는 중" : "이 역할로 계속 분석"}
          </button>
          <p className="role-footnote">
            확인을 누르면 남은 단계가 로컬에서 이어서 실행됩니다. 녹음 길이에 따라 몇 분 걸릴 수 있습니다.
          </p>
          <footer className="panel-foot">
            <button className="button ghost" type="button" disabled={busy} onClick={onDiscard}>
              {busy ? "정리하는 중" : "분석 중단하고 삭제"}
            </button>
            <p>진행 중인 단계가 끝나는 대로 중단하고, 로컬에 저장된 오디오와 중간 결과를 삭제합니다.</p>
          </footer>
        </>
      )}
    </section>
  );
}
