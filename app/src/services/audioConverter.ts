/**
 * Convert WebM audio to WAV format using Web Audio API
 */
export async function convertWebMToWav(webmBlob: Blob): Promise<Blob> {
  console.log(`[AudioConverter] Converting WebM blob: ${webmBlob.size} bytes, type: ${webmBlob.type}`);
  
  const audioContext = new AudioContext();
  
  try {
    // Read the WebM blob as ArrayBuffer
    const arrayBuffer = await webmBlob.arrayBuffer();
    console.log(`[AudioConverter] Array buffer size: ${arrayBuffer.byteLength} bytes`);
    
    if (arrayBuffer.byteLength === 0) {
      throw new Error("Audio file is empty (0 bytes)");
    }
    
    // Decode the audio data
    console.log("[AudioConverter] Decoding audio data...");
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    console.log(`[AudioConverter] Decoded: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.sampleRate}Hz, ${audioBuffer.numberOfChannels} channels`);
    
    if (audioBuffer.duration === 0) {
      throw new Error("Audio has zero duration");
    }
    
    // Convert to mono if stereo
    const numberOfChannels = 1;
    const sampleRate = audioBuffer.sampleRate;
    const length = audioBuffer.length;
    
    // Get audio data from first channel (or mix if stereo)
    const channelData = audioBuffer.getChannelData(0);
    
    // If stereo, mix down to mono
    let audioData: Float32Array;
    if (audioBuffer.numberOfChannels > 1) {
      audioData = new Float32Array(length);
      const left = audioBuffer.getChannelData(0);
      const right = audioBuffer.getChannelData(1);
      for (let i = 0; i < length; i++) {
        audioData[i] = (left[i] + right[i]) / 2;
      }
    } else {
      audioData = channelData;
    }
    
    // Trim silence from beginning and end (improves voice cloning quality)
    const trimmedAudio = trimSilence(audioData, sampleRate);
    if (trimmedAudio.length < audioData.length) {
      console.log(`[AudioConverter] Trimmed ${((audioData.length - trimmedAudio.length) / sampleRate).toFixed(2)}s of silence`);
    }
    
    // Convert to 16-bit PCM
    const pcmData = new Int16Array(trimmedAudio.length);
    for (let i = 0; i < trimmedAudio.length; i++) {
      const sample = Math.max(-1, Math.min(1, trimmedAudio[i]));
      pcmData[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    }
    
    // Create WAV file
    console.log("[AudioConverter] Creating WAV buffer...");
    const wavBuffer = createWavBuffer(pcmData, sampleRate, numberOfChannels);
    
    const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
    console.log(`[AudioConverter] WAV conversion complete: ${wavBlob.size} bytes`);
    
    return wavBlob;
  } catch (error) {
    console.error("[AudioConverter] Conversion failed:", error);
    throw error;
  } finally {
    audioContext.close();
  }
}

function createWavBuffer(pcmData: Int16Array, sampleRate: number, numChannels: number): ArrayBuffer {
  const headerSize = 44;
  const dataSize = pcmData.length * 2; // 16-bit = 2 bytes per sample
  const buffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buffer);
  
  // Write WAV header
  // "RIFF" chunk descriptor
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true); // File size - 8
  writeString(view, 8, 'WAVE');
  
  // "fmt " sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk size (16 for PCM)
  view.setUint16(20, 1, true); // Audio format (1 = PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true); // Byte rate
  view.setUint16(32, numChannels * 2, true); // Block align
  view.setUint16(34, 16, true); // Bits per sample
  
  // "data" sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);
  
  // Write PCM data
  const dataView = new Int16Array(buffer, headerSize);
  dataView.set(pcmData);
  
  return buffer;
}

function writeString(view: DataView, offset: number, string: string): void {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

/**
 * Trim silence from beginning and end of audio
 * This improves voice cloning by removing dead air
 * Threshold: -40dB (0.01 amplitude) - good balance for voice
 * Minimum silence to keep at edges: 0.1 seconds
 */
function trimSilence(audioData: Float32Array, sampleRate: number): Float32Array {
  const threshold = 0.01; // -40dB threshold
  const minSilenceKeep = Math.floor(0.1 * sampleRate); // Keep 100ms silence at edges
  
  let start = 0;
  let end = audioData.length;
  
  // Find first non-silent sample (from beginning)
  for (let i = 0; i < audioData.length; i++) {
    if (Math.abs(audioData[i]) > threshold) {
      start = Math.max(0, i - minSilenceKeep); // Keep a little silence before
      break;
    }
  }
  
  // Find last non-silent sample (from end)
  for (let i = audioData.length - 1; i >= 0; i--) {
    if (Math.abs(audioData[i]) > threshold) {
      end = Math.min(audioData.length, i + minSilenceKeep + 1); // Keep a little silence after
      break;
    }
  }
  
  // If no audio found above threshold, return original (don't trim everything)
  if (start >= end) {
    console.log("[AudioConverter] No significant audio detected, returning original");
    return audioData;
  }
  
  // Return trimmed audio
  return audioData.slice(start, end);
}
