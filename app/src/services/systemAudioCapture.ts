/**
 * System Audio Capture for Lip Sync Testing
 * 
 * Captures system audio output during Browser TTS playback
 * to analyze amplitude in real-time, similar to Qwen TTS pre-analysis.
 */

export interface CapturedAudioFrame {
  time: number;
  amplitude: number;
}

export class SystemAudioCapture {
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private mediaStream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private isCapturingFlag = false;
  private captureStartTime = 0;
  private capturedFrames: CapturedAudioFrame[] = [];
  private rafId: number | null = null;
  private onAmplitudeCallback: ((amplitude: number) => void) | null = null;

  /**
   * Start capturing system audio
   * Uses getDisplayMedia to capture audio output
   * Compatible with various audio setups including VoiceMeeter
   */
  async start(): Promise<boolean> {
    try {
      console.log('[SystemAudioCapture] Starting audio capture...');
      
      // Create audio context
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      
      // Try different capture methods
      // Method 1: getDisplayMedia with video (required for some browsers to capture audio)
      try {
        this.mediaStream = await (navigator.mediaDevices as any).getDisplayMedia({
          video: true,  // Some browsers require video for audio capture
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            sampleRate: 48000,
            channelCount: 2
          }
        });
      } catch (e) {
        console.log('[SystemAudioCapture] getDisplayMedia with video failed, trying without...');
        // Method 2: getDisplayMedia without video
        this.mediaStream = await (navigator.mediaDevices as any).getDisplayMedia({
          video: false,
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            sampleRate: 48000,
            channelCount: 2
          }
        });
      }
      
      if (!this.mediaStream) {
        console.warn('[SystemAudioCapture] No media stream obtained');
        return false;
      }
      
      // Check if we got an audio track
      const audioTracks = this.mediaStream.getAudioTracks();
      console.log('[SystemAudioCapture] Audio tracks:', audioTracks.length);
      
      if (audioTracks.length === 0) {
        console.warn('[SystemAudioCapture] No audio track in media stream');
        this.stop();
        return false;
      }
      
      // Log audio track info for debugging
      audioTracks.forEach((track, i) => {
        console.log(`[SystemAudioCapture] Audio track ${i}:`, track.label, track.readyState);
      });
      
      // Create analyzer
      this.source = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 512;  // Higher resolution
      this.analyser.smoothingTimeConstant = 0.2;  // Less smoothing for faster response
      
      this.source.connect(this.analyser);
      
      this.isCapturingFlag = true;
      this.captureStartTime = performance.now();
      this.capturedFrames = [];
      
      // Start analyzing
      this.analyze();
      
      console.log('[SystemAudioCapture] Capture started successfully');
      return true;
      
    } catch (error) {
      console.error('[SystemAudioCapture] Failed to start capture:', error);
      this.stop();
      return false;
    }
  }

  /**
   * Real-time audio analysis loop
   */
  private analyze = () => {
    if (!this.isCapturingFlag || !this.analyser) return;
    
    const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(dataArray);
    
    // Calculate RMS amplitude
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const normalized = dataArray[i] / 255;
      sum += normalized * normalized;
    }
    const rms = Math.sqrt(sum / dataArray.length);
    
    // Store frame
    const elapsed = (performance.now() - this.captureStartTime) / 1000;
    this.capturedFrames.push({
      time: elapsed,
      amplitude: rms
    });
    
    // Notify callback
    if (this.onAmplitudeCallback) {
      this.onAmplitudeCallback(rms);
    }
    
    // Continue analyzing
    this.rafId = requestAnimationFrame(this.analyze);
  };

  /**
   * Check if audio is actually being captured (not just silence)
   * Useful for detecting if the correct audio source is selected
   */
  isCapturingAudio(): boolean {
    if (this.capturedFrames.length < 10) return false;
    
    // Check last 10 frames for any non-zero amplitude
    const recentFrames = this.capturedFrames.slice(-10);
    const hasAudio = recentFrames.some(f => f.amplitude > 0.01);
    
    return hasAudio;
  }

  /**
   * Stop capturing audio
   */
  stop(): void {
    console.log('[SystemAudioCapture] Stopping capture...');
    
    this.isCapturingFlag = false;
    
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    
    if (this.source) {
      try {
        this.source.disconnect();
      } catch (e) {}
      this.source = null;
    }
    
    if (this.analyser) {
      try {
        this.analyser.disconnect();
      } catch (e) {}
      this.analyser = null;
    }
    
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    
    if (this.audioContext && this.audioContext.state !== 'closed') {
      try {
        this.audioContext.close();
      } catch (e) {}
      this.audioContext = null;
    }
    
    console.log(`[SystemAudioCapture] Capture stopped. Captured ${this.capturedFrames.length} frames`);
  }

  /**
   * Get current amplitude (for real-time lip sync)
   */
  getCurrentAmplitude(): number {
    if (!this.isCapturingFlag || !this.analyser) return 0;
    
    const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(dataArray);
    
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const normalized = dataArray[i] / 255;
      sum += normalized * normalized;
    }
    
    return Math.sqrt(sum / dataArray.length);
  }

  /**
   * Get all captured frames (for post-analysis)
   */
  getCapturedFrames(): CapturedAudioFrame[] {
    return [...this.capturedFrames];
  }

  /**
   * Check if currently capturing
   */
  isCapturing(): boolean {
    return this.isCapturingFlag;
  }

  /**
   * Set callback for real-time amplitude updates
   */
  onAmplitude(callback: (amplitude: number) => void): void {
    this.onAmplitudeCallback = callback;
  }

  /**
   * Convert captured frames to LipSyncFrame format for unified playback
   */
  toLipSyncFrames(): { time: number; amplitude: number }[] {
    return this.capturedFrames.map(f => ({
      time: f.time,
      amplitude: f.amplitude
    }));
  }
}

// Singleton instance
export const systemAudioCapture = new SystemAudioCapture();
