/**
 * Puter.js Service Wrapper
 * Provides AI transcription and other Puter.js services
 */

// Type definitions for Puter.js
declare global {
  interface Window {
    puter?: {
      ai?: {
        speech2txt?: (audioDataUrl: string) => Promise<string | { text: string }>;
        transcribe?: (audio: Blob) => Promise<{ text: string }>;
        chat?: any;
      };
    };
  }
}

export interface PuterTranscriptionResult {
  text: string;
}

// Check if Puter.js is available
const checkPuterAvailable = (): boolean => {
  return typeof window !== 'undefined' && !!(window as any).puter && !!(window as any).puter.ai;
};

// Convert audio blob to WAV format for Puter.js
const convertToWav = async (audioBlob: Blob): Promise<Blob> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const arrayBuffer = e.target?.result as ArrayBuffer;
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        
        // Decode audio
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // Create WAV from audio buffer
        const numberOfChannels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;
        
        const bytesPerSample = bitDepth / 8;
        const blockAlign = numberOfChannels * bytesPerSample;
        
        const dataLength = audioBuffer.length * numberOfChannels * bytesPerSample;
        const buffer = new ArrayBuffer(44 + dataLength);
        const view = new DataView(buffer);
        
        // Write WAV header
        const writeString = (view: DataView, offset: number, string: string) => {
          for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
          }
        };
        
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numberOfChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(view, 36, 'data');
        view.setUint32(40, dataLength, true);
        
        // Write audio data
        const offset = 44;
        const channels = [];
        for (let i = 0; i < numberOfChannels; i++) {
          channels.push(audioBuffer.getChannelData(i));
        }
        
        let index = 0;
        for (let i = 0; i < audioBuffer.length; i++) {
          for (let channel = 0; channel < numberOfChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, channels[channel][i]));
            view.setInt16(offset + index, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            index += 2;
          }
        }
        
        const wavBlob = new Blob([buffer], { type: 'audio/wav' });
        resolve(wavBlob);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = reject;
    reader.readAsArrayBuffer(audioBlob);
  });
};

class PuterService {
  private listeners: (() => void)[] = [];

  constructor() {
    this.checkAvailability();
    
    if (typeof window !== 'undefined') {
      setTimeout(() => this.checkAvailability(), 100);
      setTimeout(() => this.checkAvailability(), 500);
      setTimeout(() => this.checkAvailability(), 1000);
      setTimeout(() => this.checkAvailability(), 2000);
    }
  }

  private checkAvailability(): void {
    const available = checkPuterAvailable();
    if (available) {
      console.log("[PuterService] Puter.js is available");
      this.listeners.forEach(listener => listener());
    }
  }

  onAvailabilityChange(listener: () => void): () => void {
    this.listeners.push(listener);
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) this.listeners.splice(index, 1);
    };
  }

  /**
   * Check if Puter.js transcription is available
   */
  isTranscriptionAvailable(): boolean {
    return checkPuterAvailable();
  }

  /**
   * Transcribe audio using Puter.js AI
   * Converts to WAV format first for better compatibility
   */
  async transcribe(audioBlob: Blob): Promise<PuterTranscriptionResult> {
    if (!checkPuterAvailable()) {
      throw new Error("Puter.js not available");
    }

    const puter = (window as any).puter;
    
    console.log("[PuterService] Converting audio to WAV format...");
    const wavBlob = await convertToWav(audioBlob);
    
    // Convert blob to data URL for speech2txt
    const dataUrl = await this.blobToDataUrl(wavBlob);
    
    console.log("[PuterService] Starting transcription with speech2txt...");
    
    // Use speech2txt (the actual Puter.js API)
    if (puter.ai?.speech2txt) {
      const result = await puter.ai.speech2txt(dataUrl);
      console.log("[PuterService] speech2txt result:", result);
      
      // Result can be string or { text: string }
      const text = typeof result === 'string' ? result : (result?.text || "");
      return { text };
    }
    
    // Fallback to transcribe if available
    if (puter.ai?.transcribe) {
      const result = await puter.ai.transcribe(wavBlob);
      console.log("[PuterService] transcribe result:", result);
      return result;
    }
    
    throw new Error("Puter.ai.speech2txt not available");
  }
  
  private blobToDataUrl(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }
}

export const puter = new PuterService();
export default puter;
