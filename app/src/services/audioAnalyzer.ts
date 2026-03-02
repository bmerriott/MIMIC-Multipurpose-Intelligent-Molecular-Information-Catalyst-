/**
 * Audio Analyzer Service
 * Tracks audio amplitude from TTS and microphone for avatar mouth animation
 */

import { useRef, useCallback } from 'react';

export interface AudioAnalysisData {
  amplitude: number;      // 0-1 overall volume
  frequencyData: Uint8Array | number[]; // Frequency spectrum
  isSpeaking: boolean;    // Derived state
}

class AudioAnalyzerService {
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private dataArray: Uint8Array | Uint8Array<ArrayBufferLike> | null = null;
  private source: MediaElementAudioSourceNode | null = null;
  private currentAmplitude: number = 0;
  private isActive: boolean = false;
  private rafId: number | null = null;
  private listeners: Set<(amplitude: number) => void> = new Set();

  constructor() {
    this.analyze = this.analyze.bind(this);
  }

  /**
   * Initialize the audio analyzer with an HTMLAudioElement
   * Call this when TTS audio starts playing
   */
  initialize(audioElement?: HTMLAudioElement): boolean {
    try {
      // Clean up existing
      this.cleanup();

      // Create audio context
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      
      // Create analyser
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 64; // Small for performance
      this.analyser.smoothingTimeConstant = 0.3; // Lower smoothing for faster response to pauses

      const bufferLength = this.analyser.frequencyBinCount;
      this.dataArray = new Uint8Array(bufferLength);

      // Connect source if provided
      if (audioElement) {
        this.source = this.audioContext.createMediaElementSource(audioElement);
        this.source.connect(this.analyser);
        this.analyser.connect(this.audioContext.destination);
      }

      this.isActive = true;
      this.analyze();
      return true;
    } catch (error) {
      console.error('[AudioAnalyzer] Initialization failed:', error);
      return false;
    }
  }

  /**
   * Create analyzer from microphone stream
   * For wake word listening visualization
   */
  initializeFromStream(stream: MediaStream): boolean {
    try {
      this.cleanup();

      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 64;
      this.analyser.smoothingTimeConstant = 0.3;

      const bufferLength = this.analyser.frequencyBinCount;
      this.dataArray = new Uint8Array(bufferLength);

      const source = this.audioContext.createMediaStreamSource(stream);
      source.connect(this.analyser);
      // Don't connect to destination for microphone (would cause feedback)

      this.isActive = true;
      this.analyze();
      return true;
    } catch (error) {
      console.error('[AudioAnalyzer] Stream initialization failed:', error);
      return false;
    }
  }

  /**
   * Update from existing audio element (for TTS playback)
   */
  connectToElement(audioElement: HTMLAudioElement): boolean {
    if (!this.audioContext) {
      return this.initialize(audioElement);
    }

    try {
      // Disconnect existing source
      if (this.source) {
        this.source.disconnect();
      }

      this.source = this.audioContext.createMediaElementSource(audioElement);
      this.source.connect(this.analyser!);
      this.analyser!.connect(this.audioContext.destination);
      return true;
    } catch (error) {
      console.error('[AudioAnalyzer] Connect failed:', error);
      return false;
    }
  }

  /**
   * Analyze audio and update amplitude
   */
  private analyze(): void {
    if (!this.isActive || !this.analyser || !this.dataArray) return;

    (this.analyser.getByteFrequencyData as any)(this.dataArray);
    
    // Debug: log if we have non-zero data
    const max = Math.max(...this.dataArray);
    if (max > 0 && Math.random() < 0.01) { // Log 1% of frames with audio
      console.log('[AudioAnalyzer] Max frequency:', max, 'avg:', (this.dataArray.reduce((a,b)=>a+b,0)/this.dataArray.length).toFixed(1));
    }

    // Calculate average amplitude (0-255 from analyser -> 0-1)
    const sum = this.dataArray.reduce((a, b) => a + b, 0);
    const average = sum / this.dataArray.length;
    
    // Focus on speech frequencies (roughly 85-255 Hz range maps to lower bins)
    // Use first few bins which typically contain voice fundamentals
    const voiceBins = this.dataArray.slice(0, Math.floor(this.dataArray.length * 0.3));
    const voiceSum = voiceBins.reduce((a, b) => a + b, 0);
    const voiceAverage = voiceSum / voiceBins.length;
    
    // Combine overall and voice-specific for mouth animation
    // Weight voice frequencies more heavily for speech detection
    const rawAmplitude = ((average * 0.3) + (voiceAverage * 0.7)) / 255;
    
    // Apply threshold - values below this are considered silence
    // This helps distinguish between actual speech pauses and low-level noise
    const SILENCE_THRESHOLD = 0.05;
    this.currentAmplitude = rawAmplitude > SILENCE_THRESHOLD ? rawAmplitude : 0;
    
    // Notify listeners
    this.listeners.forEach(listener => listener(this.currentAmplitude));

    // Continue loop
    this.rafId = requestAnimationFrame(this.analyze);
  }

  /**
   * Get current amplitude (0-1)
   */
  getAmplitude(): number {
    return this.currentAmplitude;
  }

  /**
   * Get full analysis data
   */
  getAnalysisData(): AudioAnalysisData | null {
    if (!this.analyser || !this.dataArray) return null;
    
    (this.analyser.getByteFrequencyData as any)(this.dataArray);
    
    return {
      amplitude: this.currentAmplitude,
      frequencyData: Array.from(this.dataArray),
      isSpeaking: this.currentAmplitude > 0.1, // Threshold for speech
    };
  }

  /**
   * Connect to an existing AnalyserNode (for integration with audioEffects)
   */
  connectNode(analyser: AnalyserNode): void {
    console.log('[AudioAnalyzer] Connecting to analyser node');
    
    // Don't clean up - just update the analyser reference
    // The old analyser might still be useful
    this.analyser = analyser;
    const bufferLength = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(bufferLength);
    
    if (!this.isActive) {
      this.isActive = true;
      this.analyze();
    }
    
    console.log('[AudioAnalyzer] Connected, fftSize:', analyser.fftSize, 'bufferLength:', bufferLength);
  }

  /**
   * Subscribe to amplitude updates
   */
  subscribe(listener: (amplitude: number) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Resume audio context (needed after user interaction)
   */
  async resume(): Promise<void> {
    if (this.audioContext?.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  /**
   * Clean up resources
   */
  cleanup(): void {
    this.isActive = false;
    
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }

    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }

    if (this.analyser) {
      this.analyser.disconnect();
      this.analyser = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    this.listeners.clear();
    this.currentAmplitude = 0;
  }
}

// Singleton instance
export const audioAnalyzer = new AudioAnalyzerService();

/**
 * React hook for using audio amplitude in components
 */
export function useAudioAmplitude() {
  const amplitudeRef = useRef(0);

  const subscribe = useCallback((callback: (amplitude: number) => void) => {
    return audioAnalyzer.subscribe(callback);
  }, []);

  const getAmplitude = useCallback(() => {
    return audioAnalyzer.getAmplitude();
  }, []);

  return {
    amplitudeRef,
    subscribe,
    getAmplitude,
    initialize: audioAnalyzer.initialize.bind(audioAnalyzer),
    initializeFromStream: audioAnalyzer.initializeFromStream.bind(audioAnalyzer),
    connectToElement: audioAnalyzer.connectToElement.bind(audioAnalyzer),
    cleanup: audioAnalyzer.cleanup.bind(audioAnalyzer),
  };
}
