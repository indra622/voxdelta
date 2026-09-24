function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

export function encodeWav(buffer: AudioBuffer): Blob {
  const frameCount = buffer.length;
  const channelCount = buffer.numberOfChannels;
  const samples = new Float32Array(frameCount);

  for (let channel = 0; channel < channelCount; channel += 1) {
    const input = buffer.getChannelData(channel);
    for (let frame = 0; frame < frameCount; frame += 1) {
      samples[frame] += input[frame] / channelCount;
    }
  }

  const output = new ArrayBuffer(44 + frameCount * 2);
  const view = new DataView(output);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + frameCount * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, frameCount * 2, true);

  for (let frame = 0; frame < frameCount; frame += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[frame]));
    view.setInt16(44 + frame * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
  }

  return new Blob([output], { type: "audio/wav" });
}

export interface PreparedRecording {
  file: File;
  durationSeconds: number;
}

export async function recordingToWav(blob: Blob): Promise<PreparedRecording> {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await blob.arrayBuffer());
    if (decoded.length === 0) throw new Error("empty recording");
    return {
      file: new File([encodeWav(decoded)], "recording.wav", { type: "audio/wav" }),
      durationSeconds: decoded.length / decoded.sampleRate,
    };
  } finally {
    await context.close();
  }
}
