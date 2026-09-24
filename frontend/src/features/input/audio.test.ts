import { describe, expect, it } from "vitest";
import { encodeWav } from "./audio";
import { formatDuration } from "./RecordingControl";

function fakeBuffer(channels: Float32Array[], sampleRate = 16_000): AudioBuffer {
  return {
    length: channels[0].length,
    numberOfChannels: channels.length,
    sampleRate,
    getChannelData: (index: number) => channels[index],
  } as unknown as AudioBuffer;
}

describe("browser recording WAV encoder", () => {
  it("writes a mono PCM WAV that the backend can admit", async () => {
    const channel = new Float32Array([0, 0.5, -0.5, 1]);
    const bytes = new Uint8Array(await encodeWav(fakeBuffer([channel])).arrayBuffer());
    const header = new DataView(bytes.buffer);

    expect(new TextDecoder().decode(bytes.slice(0, 4))).toBe("RIFF");
    expect(new TextDecoder().decode(bytes.slice(8, 12))).toBe("WAVE");
    expect(header.getUint16(20, true)).toBe(1);
    expect(header.getUint16(22, true)).toBe(1);
    expect(header.getUint32(24, true)).toBe(16_000);
    expect(header.getUint16(34, true)).toBe(16);
    expect(bytes).toHaveLength(44 + channel.length * 2);
  });

  it("downmixes a stereo capture instead of dropping a channel", async () => {
    const left = new Float32Array([1, -1]);
    const right = new Float32Array([0, -1]);
    const bytes = await encodeWav(fakeBuffer([left, right])).arrayBuffer();
    const samples = new DataView(bytes);

    expect(samples.getInt16(44, true)).toBe(Math.trunc(0.5 * 0x7fff));
    expect(samples.getInt16(46, true)).toBe(-0x8000);
  });
});

describe("recording duration", () => {
  it("formats elapsed seconds as mm:ss", () => {
    expect(formatDuration(0)).toBe("00:00");
    expect(formatDuration(9.7)).toBe("00:09");
    expect(formatDuration(605)).toBe("10:05");
  });
});
