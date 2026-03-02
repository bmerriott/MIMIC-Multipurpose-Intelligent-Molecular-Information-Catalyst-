/**
 * Audio Analysis Service for Lip Sync
 * 
 * Unified lip sync system that works with both Qwen TTS (audio data available)
 * and Browser TTS (text + timing events).
 */

export interface LipSyncFrame {
  time: number;  // seconds from start
  amplitude: number;  // 0-1, mouth openness
}

// ============================================
// Qwen TTS: Pre-analyze audio data
// ============================================

export async function analyzeQwenAudio(audioData: string): Promise<LipSyncFrame[]> {
  try {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    
    // Decode base64 audio
    const binaryString = atob(audioData);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    
    const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
    const channelData = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;
    
    // Analyze amplitude every 16ms (~60fps) for smooth animation
    const interval = 1 / 60;
    const samplesPerFrame = Math.floor(sampleRate * interval);
    const frames: LipSyncFrame[] = [];
    
    // Calculate global stats for normalization
    let globalMax = 0;
    
    for (let i = 0; i < channelData.length; i += samplesPerFrame) {
      let sumSquares = 0;
      let peak = 0;
      const end = Math.min(i + samplesPerFrame, channelData.length);
      
      for (let j = i; j < end; j++) {
        const sample = channelData[j];
        sumSquares += sample * sample;
        peak = Math.max(peak, Math.abs(sample));
      }
      
      // RMS amplitude with peak boost for transients
      const rms = Math.sqrt(sumSquares / (end - i));
      let amplitude = (rms * 3) + (peak * 0.5);
      
      // Apply silence threshold - close mouth during quiet periods
      const SILENCE_THRESHOLD = 0.02;
      if (amplitude < SILENCE_THRESHOLD) {
        amplitude = 0; // Force mouth closed during silence
      } else {
        // Normalize and apply curve for more dynamic range
        amplitude = Math.min(1, (amplitude - SILENCE_THRESHOLD) * 2);
        // Apply power curve for more snap (close faster, open slower)
        amplitude = Math.pow(amplitude, 0.7);
      }
      
      globalMax = Math.max(globalMax, amplitude);
      
      frames.push({
        time: i / sampleRate,
        amplitude
      });
    }
    
    audioContext.close();
    console.log(`[AudioAnalysis] Analyzed ${frames.length} frames from Qwen audio (${audioBuffer.duration.toFixed(2)}s), max amplitude: ${globalMax.toFixed(2)}`);
    return frames;
    
  } catch (error) {
    console.error('[AudioAnalysis] Failed to analyze Qwen audio:', error);
    return [];
  }
}

// ============================================
// Browser TTS: Text-based timing with phoneme parsing
// ============================================

interface Phoneme {
  char: string;
  isVowel: boolean;
  isPause: boolean;
  duration: number; // relative duration multiplier
}

function parsePhonemes(text: string): Phoneme[] {
  const phonemes: Phoneme[] = [];
  const vowels = new Set('aeiouyAEIOUY');
  const pauses = new Set('.!?');
  const shortPauses = new Set(',;: -');
  
  for (const char of text) {
    if (pauses.has(char)) {
      phonemes.push({ char, isVowel: false, isPause: true, duration: 8 }); // 400ms
    } else if (shortPauses.has(char)) {
      phonemes.push({ char, isVowel: false, isPause: true, duration: 4 }); // 200ms
    } else if (vowels.has(char)) {
      phonemes.push({ char, isVowel: true, isPause: false, duration: 3 }); // 150ms
    } else {
      phonemes.push({ char, isVowel: false, isPause: false, duration: 1.5 }); // 75ms
    }
  }
  
  return phonemes;
}

export function generateBrowserTTSLipSync(text: string, wpm: number = 130): LipSyncFrame[] {
  const phonemes = parsePhonemes(text);
  const frames: LipSyncFrame[] = [];
  
  // Slightly slower WPM for more realistic timing with system TTS
  const baseMsPerUnit = 60;
  const msPerUnit = (60000 / wpm / 5) / baseMsPerUnit * baseMsPerUnit;
  
  let currentTime = 0;
  const frameInterval = 1 / 60; // 60fps
  
  // Add startup delay to match Browser TTS audio initialization
  const startupDelay = 0.35;
  currentTime += startupDelay;
  
  // Envelope follower for natural syllable transitions
  let envelopeValue = 0;
  const attackTime = 0.02;  // 20ms attack
  const releaseTime = 0.08; // 80ms release
  const attackCoef = Math.exp(-1 / (60 * attackTime));
  const releaseCoef = Math.exp(-1 / (60 * releaseTime));
  
  // Track syllable boundaries for emphasis
  let lastWasVowel = false;
  let syllableCounter = 0;
  
  for (let p = 0; p < phonemes.length; p++) {
    const phoneme = phonemes[p];
    
    // Detect syllable boundaries (vowel -> consonant transitions)
    const isSyllableStart = phoneme.isVowel && !lastWasVowel;
    if (isSyllableStart) {
      syllableCounter++;
    }
    lastWasVowel = phoneme.isVowel;
    
    const duration = phoneme.duration * (msPerUnit / 1000);
    const endTime = currentTime + duration;
    
    // Generate frames for this phoneme
    while (currentTime < endTime) {
      let targetAmplitude: number;
      const progress = (currentTime - (endTime - duration)) / duration;
      
      if (phoneme.isPause) {
        targetAmplitude = 0;
      } else if (phoneme.isVowel) {
        // Vowel: more exaggerated opening for better syllable visibility
        let envelope: number;
        if (progress < 0.2) {
          // Fast attack (first 20%)
          envelope = Math.pow(progress / 0.2, 0.5);
        } else if (progress < 0.6) {
          // Sustain (20-60%)
          envelope = 1 - ((progress - 0.2) / 0.4) * 0.2;
        } else {
          // Release (60-100%)
          envelope = 0.8 * Math.pow(1 - (progress - 0.6) / 0.4, 1.5);
        }
        
        // Higher amplitude for stressed syllables
        targetAmplitude = 0.25 + (envelope * 0.75);
        
        // Boost amplitude for better visibility
        targetAmplitude = Math.min(1, targetAmplitude * 1.15);
      } else {
        // Consonant: more movement for visibility
        const isPlosive = 'pbtdkg'.includes(phoneme.char.toLowerCase());
        const isFricative = 'fvszsh'.includes(phoneme.char.toLowerCase());
        const isNasal = 'mn'.includes(phoneme.char.toLowerCase());
        
        if (isPlosive) {
          // Quick close then slight open
          targetAmplitude = progress < 0.3 ? 0 : 0.1 + Math.sin((progress - 0.3) / 0.7 * Math.PI) * 0.15;
        } else if (isFricative) {
          // Slightly open for airflow
          targetAmplitude = 0.2 + Math.sin(progress * Math.PI) * 0.15;
        } else if (isNasal) {
          // Moderate opening
          targetAmplitude = 0.18 + Math.sin(progress * Math.PI) * 0.2;
        } else {
          // Other consonants
          targetAmplitude = 0.12 + Math.sin(progress * Math.PI) * 0.2;
        }
      }
      
      // Apply envelope follower for smooth transitions
      if (targetAmplitude > envelopeValue) {
        envelopeValue = targetAmplitude + attackCoef * (envelopeValue - targetAmplitude);
      } else {
        envelopeValue = targetAmplitude + releaseCoef * (envelopeValue - targetAmplitude);
      }
      
      // Apply power curve for more visible movement
      let finalAmplitude = Math.pow(envelopeValue, 0.65);
      
      // Boost mid-range for more syllable definition
      if (finalAmplitude > 0.15 && finalAmplitude < 0.6) {
        finalAmplitude = 0.15 + (finalAmplitude - 0.15) * 1.25;
        finalAmplitude = Math.min(1, finalAmplitude);
      }
      
      frames.push({ time: currentTime, amplitude: finalAmplitude });
      currentTime += frameInterval;
    }
  }
  
  console.log(`[AudioAnalysis] Generated ${frames.length} frames for Browser TTS (${syllableCounter} syllables, ${startupDelay}s startup)`);
  return frames;
}

// ============================================
// Browser TTS with System Audio Capture (Testing)
// ============================================

import { systemAudioCapture } from './systemAudioCapture';

/**
 * Lip sync player that uses real-time system audio capture
 * This provides the same effect as Qwen TTS but for Browser TTS
 */
export class SystemAudioLipSync {
  private isActiveFlag = false;
  private onAmplitudeCallback: ((amplitude: number) => void) | null = null;
  private rafId: number | null = null;

  /**
   * Start lip sync with system audio capture
   * This will prompt user for screen/audio capture permission
   */
  async start(onAmplitude: (amplitude: number) => void): Promise<boolean> {
    // Start system audio capture
    const success = await systemAudioCapture.start();
    
    if (!success) {
      console.warn('[SystemAudioLipSync] Failed to start system audio capture');
      return false;
    }
    
    this.isActiveFlag = true;
    this.onAmplitudeCallback = onAmplitude;
    
    // Start real-time amplitude polling
    this.pollAmplitude();
    
    console.log('[SystemAudioLipSync] Started with system audio capture');
    return true;
  }

  private pollAmplitude = () => {
    if (!this.isActiveFlag) return;
    
    // Get current amplitude from system audio capture
    const amplitude = systemAudioCapture.getCurrentAmplitude();
    
    if (this.onAmplitudeCallback) {
      this.onAmplitudeCallback(amplitude);
    }
    
    // Continue polling at 60fps
    this.rafId = requestAnimationFrame(this.pollAmplitude);
  };

  /**
   * Stop lip sync and audio capture
   */
  stop(): void {
    this.isActiveFlag = false;
    
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    
    systemAudioCapture.stop();
    
    if (this.onAmplitudeCallback) {
      this.onAmplitudeCallback(0);
    }
    
    console.log('[SystemAudioLipSync] Stopped');
    
    // Log captured frames for analysis
    const frames = systemAudioCapture.getCapturedFrames();
    console.log(`[SystemAudioLipSync] Captured ${frames.length} frames total`);
  }

  /**
   * Check if currently active
   */
  isActive(): boolean {
    return this.isActiveFlag;
  }

  /**
   * Get captured frames for post-analysis
   */
  getCapturedFrames() {
    return systemAudioCapture.getCapturedFrames();
  }
}

// ============================================
// Lip Sync Player
// ============================================

export class LipSyncPlayer {
  private frames: LipSyncFrame[] = [];
  private startTime = 0;
  private isPlayingFlag = false;
  private onAmplitudeCallback: ((amplitude: number) => void) | null = null;
  private rafId: number | null = null;

  loadFrames(frames: LipSyncFrame[]) {
    this.frames = frames;
  }

  getFrames(): LipSyncFrame[] {
    return this.frames;
  }

  start(onAmplitude: (amplitude: number) => void) {
    this.startTime = performance.now() / 1000;
    this.isPlayingFlag = true;
    this.onAmplitudeCallback = onAmplitude;
    this.tick();
  }

  private tick = () => {
    if (!this.isPlayingFlag) return;
    
    const elapsed = (performance.now() / 1000) - this.startTime;
    
    // Find current frame
    const frameIndex = Math.floor(elapsed * 60); // 60fps
    
    if (frameIndex >= this.frames.length) {
      // End of audio
      this.onAmplitudeCallback?.(0);
      this.isPlayingFlag = false;
      return;
    }
    
    const amplitude = this.frames[frameIndex]?.amplitude || 0;
    this.onAmplitudeCallback?.(amplitude);
    
    this.rafId = requestAnimationFrame(this.tick);
  };

  stop() {
    this.isPlayingFlag = false;
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.onAmplitudeCallback?.(0);
  }

  isPlaying(): boolean {
    return this.isPlayingFlag;
  }

  dispose() {
    this.stop();
    this.frames = [];
    this.onAmplitudeCallback = null;
  }
}
