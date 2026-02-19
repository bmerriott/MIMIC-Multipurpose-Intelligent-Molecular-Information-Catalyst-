/**
 * Speech Recognition using Puter.js (preferred) or Python backend (fallback)
 * Puter.js offers free OpenAI Whisper-1 transcription via cloud API
 * Falls back to local backend Whisper only if Puter.js is unavailable
 */

import { convertWebMToWav } from "./audioConverter";

interface SpeechRecognitionResult {
  transcript: string;
  isFinal: boolean;
}

type SpeechRecognitionCallback = (result: SpeechRecognitionResult) => void;
type ErrorCallback = (error: string) => void;

// Check if Puter.js is available
const isPuterAvailable = (): boolean => {
  return typeof window !== 'undefined' && (window as any).puter && (window as any).puter.ai;
};

export class LocalSpeechRecognizer {
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private stream: MediaStream | null = null;
  private isListening = false;
  private onResultCallback: SpeechRecognitionCallback | null = null;
  private onErrorCallback: ErrorCallback | null = null;
  private backendUrl: string;
  private usePuter: boolean;

  private minRecordingDuration = 500; // Minimum 500ms recording
  private maxRecordingDuration = 10000; // Maximum 10 seconds
  private recordingStartTime = 0;

  constructor(backendUrl: string = "http://localhost:8000", usePuter: boolean = true) {
    this.backendUrl = backendUrl;
    this.usePuter = usePuter && isPuterAvailable();
    if (this.usePuter) {
      console.log("🎤 Using Puter.js for free speech recognition");
    } else {
      console.log("🎤 Using local backend for speech recognition");
    }
  }

  setBackendUrl(url: string) {
    this.backendUrl = url;
  }
  
  setUsePuter(usePuter: boolean) {
    this.usePuter = usePuter && isPuterAvailable();
    console.log(`🎤 Puter.js ${this.usePuter ? 'enabled' : 'disabled'}`);
  }

  onResult(callback: SpeechRecognitionCallback) {
    this.onResultCallback = callback;
  }

  onError(callback: ErrorCallback) {
    this.onErrorCallback = callback;
  }

  async start(deviceId?: string): Promise<boolean> {
    console.log(`🎤 LocalSpeechRecognizer.start() called, deviceId: ${deviceId || 'default'}, isListening: ${this.isListening}`);
    
    if (this.isListening) {
      console.log("🎤 Already listening");
      return true;
    }

    try {
      // Get microphone access with specific device if provided
      const constraints: MediaStreamConstraints = {
        audio: deviceId 
          ? {
              deviceId: { exact: deviceId },
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
              sampleRate: 16000,
            }
          : {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
              sampleRate: 16000,
            }
      };
      
      console.log('🎤 Requesting getUserMedia with constraints:', JSON.stringify(constraints));
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log('🎤 getUserMedia succeeded');
      
      // Log which device we're using
      const track = this.stream.getAudioTracks()[0];
      if (track) {
        console.log(`🎤 Using microphone: ${track.label}`);
      }

      this.audioChunks = [];
      
      // Use webm/opus for best compatibility
      const mimeType = MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/mp4";
      
      this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });
      
      this.mediaRecorder.ondataavailable = (event) => {
        console.log(`🎤 MediaRecorder data available: ${event.data.size} bytes`);
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        this.processRecording();
      };

      this.mediaRecorder.onerror = (event) => {
        console.error("MediaRecorder error:", event);
        this.onErrorCallback?.("Recording error");
      };

      // Start recording
      console.log(`🎤 Starting MediaRecorder with mimeType: ${mimeType}`);
      this.mediaRecorder.start(100); // Collect data every 100ms
      this.recordingStartTime = Date.now();
      this.isListening = true;
      
      console.log("🎤 Local speech recognition started - state:", this.mediaRecorder.state);
      
      // Set max duration timeout
      setTimeout(() => {
        if (this.isListening) {
          console.log("Max recording duration reached");
          this.stop();
        }
      }, this.maxRecordingDuration);

      return true;
    } catch (error: any) {
      console.error("🎤 Failed to start recording:", error);
      console.error("🎤 Error name:", error.name);
      console.error("🎤 Error message:", error.message);
      this.onErrorCallback?.(error.message || "Microphone access denied");
      return false;
    }
  }

  stop() {
    if (!this.isListening) {
      console.log("🎤 Stop called but not listening");
      return;
    }
    
    const recordingDuration = Date.now() - this.recordingStartTime;
    console.log(`🎤 Stopping recording... duration: ${recordingDuration}ms, chunks: ${this.audioChunks.length}`);
    
    if (recordingDuration < this.minRecordingDuration) {
      // Wait until minimum duration
      const waitTime = this.minRecordingDuration - recordingDuration;
      console.log(`🎤 Waiting ${waitTime}ms for minimum duration...`);
      setTimeout(() => this.stop(), waitTime);
      return;
    }

    this.isListening = false;

    if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
      console.log("🎤 Calling mediaRecorder.stop()...");
      this.mediaRecorder.stop();
    } else {
      console.log("🎤 MediaRecorder already inactive or null");
    }

    // Stop all tracks
    this.stream?.getTracks().forEach(track => track.stop());
  }

  abort() {
    this.isListening = false;
    if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
      this.mediaRecorder.stop();
    }
    this.stream?.getTracks().forEach(track => track.stop());
    this.audioChunks = [];
  }

  private async processRecording() {
    if (this.audioChunks.length === 0) {
      console.log("No audio recorded");
      return;
    }

    const audioBlob = new Blob(this.audioChunks, { type: "audio/webm" });
    console.log(`🎤 Processing ${audioBlob.size} bytes of WebM audio...`);
    
    // Skip tiny audio chunks that can't be decoded (often happen when stopping recording)
    if (audioBlob.size < 10000) {
      console.log(`🎤 Audio too small (${audioBlob.size} bytes) - skipping`);
      return;
    }

    try {
      // Convert WebM to WAV for analysis and transcription
      console.log("🎤 Converting to WAV format...");
      const wavBlob = await convertWebMToWav(audioBlob);
      console.log(`🎤 Converted to WAV: ${wavBlob.size} bytes`);
      
      // Check if audio contains speech (VAD - Voice Activity Detection)
      const hasSpeech = await this.checkVoiceActivity(wavBlob);
      if (!hasSpeech) {
        console.log("🎤 No voice activity detected - skipping transcription");
        return;
      }
      
      let transcript: string;
      
      // Use Puter.js if available, otherwise use backend
      if (this.usePuter) {
        transcript = await this.transcribeWithPuter(wavBlob);
      } else {
        transcript = await this.transcribeWithBackend(wavBlob);
      }

      if (transcript) {
        console.log("🎤 Transcribed:", transcript);
        this.onResultCallback?.({ transcript, isFinal: true });
      } else {
        console.log("🎤 No speech detected");
      }
    } catch (error: any) {
      console.error("Transcription error:", error);
      this.onErrorCallback?.(error.message);
    }
  }
  
  private async transcribeWithPuter(wavBlob: Blob): Promise<string> {
    console.log("🎤 Using Puter.js for transcription...");
    
    // Convert blob to data URL
    const dataUrl = await this.blobToDataUrl(wavBlob);
    
    // Use Puter.js AI speech-to-text
    const puter = (window as any).puter;
    const result = await puter.ai.speech2txt(dataUrl);
    
    console.log("🎤 Puter.js result:", result);
    return result?.text || result || "";
  }
  
  private async transcribeWithBackend(wavBlob: Blob): Promise<string> {
    // Convert to base64
    const base64Audio = await this.blobToBase64(wavBlob);
    
    // Send to backend for transcription
    const response = await fetch(`${this.backendUrl}/api/transcribe`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audio_data: base64Audio }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Transcription failed: ${error}`);
    }

    const result = await response.json();
    return result.text?.trim() || "";
  }
  
  private blobToDataUrl(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }
  
  // Simple Voice Activity Detection - check if audio has sufficient volume
  private async checkVoiceActivity(wavBlob: Blob): Promise<boolean> {
    try {
      const arrayBuffer = await wavBlob.arrayBuffer();
      const audioContext = new AudioContext();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Get audio data from first channel
      const channelData = audioBuffer.getChannelData(0);
      
      // Calculate RMS (Root Mean Square) volume
      let sum = 0;
      for (let i = 0; i < channelData.length; i++) {
        sum += channelData[i] * channelData[i];
      }
      const rms = Math.sqrt(sum / channelData.length);
      
      // Convert to dB
      const db = 20 * Math.log10(rms);
      
      console.log(`🎤 VAD: Audio level is ${db.toFixed(1)} dB`);
      
      // Threshold: -40 dB (typical for speech detection)
      // Silence is usually below -50 dB
      const hasSpeech = db > -45;
      
      console.log(`🎤 VAD: ${hasSpeech ? 'Speech detected' : 'No speech (silence)'}`);
      
      audioContext.close();
      return hasSpeech;
    } catch (error) {
      console.error("VAD error:", error);
      // If VAD fails, assume there's speech to be safe
      return true;
    }
  }

  private blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = (reader.result as string).split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  getListening(): boolean {
    return this.isListening;
  }
}

// Singleton instance
export const localSpeechRecognizer = new LocalSpeechRecognizer();
