import { Loader2, Play, Square } from "lucide-react";
import { sameClip, type ClipId, type Playback } from "./playback";

interface ClipButtonProps {
  clip: ClipId;
  start: number;
  end: number;
  playback: Playback;
  label: string;
  /** Named separately from the visible label so the control still says what it plays when
   *  it is read out of context, which is how a screen reader reaches it. */
  accessibleLabel: string;
}

/** One play/stop control with its state visible rather than only announced.
 *
 * The same button stops what it started: a reviewer who pressed play looks for the way
 * back at the place they pressed, and a separate stop control elsewhere is a second thing
 * to find. `aria-pressed` carries the same state the icon does, so the control is not
 * relying on colour or shape alone. */
export function ClipButton({
  clip,
  start,
  end,
  playback,
  label,
  accessibleLabel,
}: ClipButtonProps) {
  const current = sameClip(playback.active, clip);
  const loading = current && playback.status === "loading";
  const playing = current && playback.status === "playing";
  const failed = sameClip(playback.errorClip, clip) && playback.error !== null;

  return (
    <button
      type="button"
      className={playing ? "clip-button playing" : "clip-button"}
      aria-pressed={playing}
      aria-label={accessibleLabel}
      onClick={() => playback.play(clip, start, end)}
    >
      {loading ? (
        <Loader2 aria-hidden="true" size={14} className="spin" />
      ) : playing ? (
        <Square aria-hidden="true" size={14} />
      ) : (
        <Play aria-hidden="true" size={14} />
      )}
      <span>{loading ? "불러오는 중" : playing ? "정지" : label}</span>
      {failed && <span className="clip-failed" aria-hidden="true">!</span>}
    </button>
  );
}
