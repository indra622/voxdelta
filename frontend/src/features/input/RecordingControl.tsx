import { Mic, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { recordingToWav } from "./audio";

interface RecordingControlProps {
  disabled: boolean;
  onReady(file: File, durationSeconds: number): void;
}

export function formatDuration(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(whole / 60).toString().padStart(2, "0");
  const remainder = (whole % 60).toString().padStart(2, "0");
  return minutes + ":" + remainder;
}

export function RecordingControl({ disabled, onReady }: RecordingControlProps) {
  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    if (!recording) return undefined;
    const timer = window.setInterval(() => setSeconds((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [recording]);

  useEffect(
    () => () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    },
    [],
  );

  async function startRecording() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("이 브라우저에서는 마이크 녹음을 사용할 수 없습니다. 파일 업로드를 이용해 주세요.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const preferred = "audio/webm;codecs=opus";
      const options = MediaRecorder.isTypeSupported(preferred) ? { mimeType: preferred } : undefined;
      const recorder = new MediaRecorder(stream, options);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        setRecording(false);
        setProcessing(true);
        try {
          const blob = new Blob(chunksRef.current, { type: recorder.mimeType });
          const prepared = await recordingToWav(blob);
          onReady(prepared.file, prepared.durationSeconds);
        } catch {
          setError("녹음을 WAV 파일로 준비하지 못했습니다. 파일 업로드를 이용해 주세요.");
        } finally {
          chunksRef.current = [];
          setProcessing(false);
        }
      };
      recorderRef.current = recorder;
      setSeconds(0);
      setRecording(true);
      recorder.start(250);
    } catch {
      setError("마이크 권한을 확인해 주세요. 브라우저 주소창의 권한 설정에서 마이크를 허용할 수 있습니다.");
    }
  }

  function stopRecording() {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  }

  const status = recording
    ? "녹음 중입니다."
    : processing
      ? "녹음을 WAV 파일로 변환하는 중입니다."
      : "";

  return (
    <div className="recording-control">
      <button
        className={recording ? "record-button recording" : "record-button"}
        type="button"
        disabled={disabled || processing}
        onClick={recording ? stopRecording : startRecording}
      >
        {recording ? <Square aria-hidden="true" size={17} /> : <Mic aria-hidden="true" size={18} />}
        <span>{processing ? "녹음 준비 중" : recording ? "녹음 끝내기" : "바로 녹음하기"}</span>
        {/* The seconds tick every second; announcing them would talk over everything else. */}
        {recording && <span className="record-timer" aria-hidden="true">{formatDuration(seconds)}</span>}
      </button>
      <p className="record-status" role="status">{status}</p>
      <p className="record-hint">
        너무 짧은 녹음은 백엔드가 받지 않습니다. 통화 한 건 분량으로 녹음해 주세요.
      </p>
      {error && <p className="field-error" role="alert">{error}</p>}
    </div>
  );
}
