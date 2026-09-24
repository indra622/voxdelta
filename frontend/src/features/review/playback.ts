import { useCallback, useEffect, useRef, useState } from "react";
import { describeError, type ErrorGuidance } from "../../api/errors";
import type { ApiClient } from "../../api/types";

/** Which clip a reviewer is listening to, named by where it came from.
 *
 * A turn and a gap can cover the same seconds, so a clip is identified by its origin
 * rather than by its time range: pressing play on a gap must not light up the turn
 * beside it. */
export type ClipKind = "turn" | "gap";

export interface ClipId {
  kind: ClipKind;
  index: number;
}

export type ClipStatus = "idle" | "loading" | "playing";

export interface Playback {
  status: ClipStatus;
  /** The clip currently loading or playing, or null when nothing is. */
  active: ClipId | null;
  error: ErrorGuidance | null;
  /** The clip the error belongs to, so one failure does not mark every control. */
  errorClip: ClipId | null;
  play(clip: ClipId, start: number, end: number): void;
  stop(): void;
}

export function sameClip(left: ClipId | null, right: ClipId | null): boolean {
  return left !== null && right !== null && left.kind === right.kind && left.index === right.index;
}

/** One audio element for the whole screen, so starting a clip always stops the last one.
 *
 * Clips are fetched and played from an object URL rather than pointed at with a plain
 * `<audio src>`. The capability token is attached by the dev proxy to `fetch` requests
 * for `/api`; a media element's own request would not carry it, and putting the token in
 * the URL to compensate would place it in history, referrers, and any log that records
 * paths. The blob is revoked as soon as it is replaced or the screen unmounts, so a clip
 * of someone's speech is not left addressable after the reviewer has moved on. */
export function useClipPlayback(client: ApiClient, conversationId: string | null): Playback {
  const [status, setStatus] = useState<ClipStatus>("idle");
  const [active, setActive] = useState<ClipId | null>(null);
  const [error, setError] = useState<ErrorGuidance | null>(null);
  const [errorClip, setErrorClip] = useState<ClipId | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  // Every play is given a number, and a fetch that finishes after a newer one started is
  // discarded. Without it, a slow first clip would begin playing over the second.
  const requestRef = useRef(0);

  const releaseUrl = useCallback(() => {
    if (objectUrlRef.current !== null) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    requestRef.current += 1;
    const element = audioRef.current;
    if (element) {
      element.pause();
      element.removeAttribute("src");
    }
    releaseUrl();
    setStatus("idle");
    setActive(null);
  }, [releaseUrl]);

  // Switching drafts must not leave the previous conversation's audio playing.
  useEffect(() => {
    stop();
    setError(null);
    setErrorClip(null);
  }, [conversationId, stop]);

  useEffect(() => {
    return () => {
      requestRef.current += 1;
      audioRef.current?.pause();
      releaseUrl();
    };
  }, [releaseUrl]);

  const play = useCallback(
    (clip: ClipId, start: number, end: number) => {
      if (!conversationId) return;
      if (sameClip(active, clip) && status !== "idle") {
        stop();
        return;
      }
      requestRef.current += 1;
      const ticket = requestRef.current;
      audioRef.current?.pause();
      releaseUrl();
      setError(null);
      setErrorClip(null);
      setActive(clip);
      setStatus("loading");

      void client
        .getAnnotationClip(conversationId, start, end)
        .then((blob) => {
          if (ticket !== requestRef.current) return;
          const element = audioRef.current ?? new Audio();
          audioRef.current = element;
          const url = URL.createObjectURL(blob);
          objectUrlRef.current = url;
          element.src = url;
          element.onended = () => {
            if (ticket !== requestRef.current) return;
            setStatus("idle");
            setActive(null);
            releaseUrl();
          };
          element.onerror = () => {
            if (ticket !== requestRef.current) return;
            setStatus("idle");
            setActive(null);
            setErrorClip(clip);
            setError({
              code: "clip_playback_failed",
              title: "이 구간을 재생하지 못했어요",
              detail: "브라우저가 오디오를 재생하지 못했습니다. 다시 시도해 주세요.",
            });
            releaseUrl();
          };
          return element
            .play()
            .then(() => {
              if (ticket !== requestRef.current) return;
              setStatus("playing");
            })
            .catch(() => {
              if (ticket !== requestRef.current) return;
              // Autoplay policies reject play() without a gesture. The reviewer did make
              // one, so this is worth saying plainly rather than failing silently.
              setStatus("idle");
              setActive(null);
              setErrorClip(clip);
              setError({
                code: "clip_playback_blocked",
                title: "브라우저가 재생을 막았어요",
                detail: "재생 버튼을 다시 한 번 눌러 주세요.",
              });
            });
        })
        .catch((reason: unknown) => {
          if (ticket !== requestRef.current) return;
          setStatus("idle");
          setActive(null);
          setErrorClip(clip);
          setError(describeError(reason));
        });
    },
    [active, client, conversationId, releaseUrl, status, stop],
  );

  return { status, active, error, errorClip, play, stop };
}
