import { FileAudio, Upload, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { PrivacyNote } from "../privacy/PrivacyNote";
import { TransferConsent } from "../privacy/TransferConsent";
import type { PrivacyPosture } from "../privacy/posture";
import { RecordingControl, formatDuration } from "./RecordingControl";

interface AudioInputProps {
  disabled: boolean;
  file: File | null;
  posture: PrivacyPosture;
  acknowledged: boolean;
  /** False until the transfer question is both answered and, if needed, acknowledged. */
  submittable: boolean;
  onAcknowledge(value: boolean): void;
  onFile(file: File | null): void;
  onAnalyze(): void;
}

// MP4 is accepted as an audio container only. The backend extracts and normalizes its
// audio track locally; it never treats video frames as analysis input.
const ACCEPTED = [".wav", ".mp3", ".m4a", ".mp4"];

function prettyBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function acceptedFile(file: File): boolean {
  return ACCEPTED.some((suffix) => file.name.toLowerCase().endsWith(suffix));
}

export function AudioInput({
  disabled,
  file,
  posture,
  acknowledged,
  submittable,
  onAcknowledge,
  onFile,
  onAnalyze,
}: AudioInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const chooseRef = useRef<HTMLButtonElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [recordedSeconds, setRecordedSeconds] = useState<number | null>(null);
  const [restoreFocus, setRestoreFocus] = useState(false);
  const gated = posture.kind === "remote" || posture.kind === "unknown";

  // Removing the file unmounts the chip that held focus, so hand it back to the
  // control that replaces it rather than dropping the keyboard user on <body>.
  useEffect(() => {
    if (!restoreFocus || file) return;
    chooseRef.current?.focus();
    setRestoreFocus(false);
  }, [file, restoreFocus]);

  function select(candidate: File | undefined) {
    setDragging(false);
    if (!candidate) return;
    if (!acceptedFile(candidate)) {
      setError("WAV, MP3, M4A, MP4 파일만 분석할 수 있어요.");
      return;
    }
    setError(null);
    setRecordedSeconds(null);
    onFile(candidate);
  }

  function clearSelection() {
    setRecordedSeconds(null);
    setRestoreFocus(true);
    onFile(null);
  }

  return (
    <section className="panel input-panel" aria-labelledby="input-heading">
      <header className="panel-head">
        <div>
          <span className="kicker">01 / INPUT</span>
          <h2 id="input-heading">어떤 목소리를 살펴볼까요?</h2>
          <p>파일을 올리거나 지금 바로 녹음하세요. 고객과 상담원, 두 사람의 통화 녹음을 분석합니다.</p>
        </div>
      </header>

      {!file ? (
        <div
          className={dragging ? "dropzone dragging" : "dropzone"}
          onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => { event.preventDefault(); select(event.dataTransfer.files[0]); }}
        >
          <Upload aria-hidden="true" size={26} strokeWidth={1.7} />
          <p><strong>여기에 음성 파일을 놓으세요</strong></p>
          <p className="dropzone-note">WAV, MP3, M4A, MP4</p>
          <button ref={chooseRef} className="secondary-button" type="button" disabled={disabled} onClick={() => inputRef.current?.click()}>
            파일 고르기
          </button>
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept=".wav,.mp3,.m4a,.mp4,audio/wav,audio/mpeg,audio/mp4,video/mp4"
            aria-label="분석할 음성 파일"
            onChange={(event) => select(event.target.files?.[0])}
          />
        </div>
      ) : (
        <div className="selected-file" aria-live="polite">
          <div className="file-mark"><FileAudio aria-hidden="true" size={24} /></div>
          <div>
            <strong>{file.name}</strong>
            <span>
              {prettyBytes(file.size)}
              {recordedSeconds !== null && " · " + formatDuration(recordedSeconds) + " 녹음"}
              {" · 분석 준비됨"}
            </span>
          </div>
          <button className="icon-button" type="button" aria-label="선택한 파일 제거" disabled={disabled} onClick={clearSelection}>
            <X aria-hidden="true" size={18} />
          </button>
        </div>
      )}

      {error && <p className="field-error" role="alert">{error}</p>}
      <div className="input-divider"><span>또는</span></div>

      {/* Stacked while the transfer gate is up, so the disclosure sits between choosing a
          recording and starting the analysis instead of beside it. */}
      <div className={gated ? "input-actions stacked" : "input-actions"}>
        <RecordingControl
          disabled={disabled}
          onReady={(recorded, durationSeconds) => {
            setError(null);
            setRecordedSeconds(durationSeconds);
            onFile(recorded);
          }}
        />
        <TransferConsent
          posture={posture}
          acknowledged={acknowledged}
          disabled={disabled}
          onAcknowledge={onAcknowledge}
        />
        <button
          className="button primary analyze-button"
          type="button"
          disabled={!file || disabled || !submittable}
          onClick={onAnalyze}
        >
          {disabled ? "분석을 시작하는 중" : "감정 분석 시작"}
        </button>
      </div>
      {!submittable && (
        <p className="input-gate" role="note">
          {posture.kind === "checking"
            ? "오디오가 외부로 나가는지 백엔드 설정에서 확인하는 중입니다. 확인이 끝나야 분석을 시작할 수 있습니다."
            : "위 전송 고지에 동의해야 분석을 시작할 수 있습니다."}
        </p>
      )}
      <PrivacyNote posture={posture} />
    </section>
  );
}
