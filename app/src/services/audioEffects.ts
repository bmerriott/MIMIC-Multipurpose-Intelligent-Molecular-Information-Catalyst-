/**
 * Audio Effects Service
 * Applies real-time audio effects using Web Audio API
 * Used for previewing voice tuning parameters without regeneration
 */

export interface AudioEffectParams {
  // Basic
  pitchShift: number;      // -1 to 1 (semitones)
  speed: number;           // 0.5 to 2.0
  
  // Voice characteristics (affect filter/EQ)
  warmth: number;          // 0 to 1 (low-mid boost)
  clarity: number;         // 0 to 1 (high boost)
  breathiness: number;     // 0 to 1 (high freq noise/air)
  resonance: number;       // 0 to 1 (mid boost)
  
  // Audio effects
  reverb: number;          // 0 to 1
  eqLow: number;           // 0 to 1 (100Hz)
  eqMid: number;           // 0 to 1 (1kHz)
  eqHigh: number;          // 0 to 1 (10kHz)
  compression: number;     // 0 to 1
}

interface PlaybackState {
  source: AudioBufferSourceNode;
  gainNode: GainNode;
  eqNodes: BiquadFilterNode[];
  reverbNode: ConvolverNode | null;
  compressor: DynamicsCompressorNode | null;
  context: AudioContext;
  isPlaying: boolean;
  startTime: number;
  pausedAt: number;
  buffer: AudioBuffer;
  playbackRate: number;
}

class AudioEffectsService {
  private currentPlayback: PlaybackState | null = null;
  private reverbImpulse: AudioBuffer | null = null;

  /**
   * Generate a reverb impulse response
   */
  private async generateReverbImpulse(context: AudioContext, duration: number = 2.0): Promise<AudioBuffer> {
    const sampleRate = context.sampleRate;
    const length = sampleRate * duration;
    const impulse = context.createBuffer(2, length, sampleRate);
    
    for (let channel = 0; channel < 2; channel++) {
      const channelData = impulse.getChannelData(channel);
      for (let i = 0; i < length; i++) {
        // Exponential decay with random noise
        const decay = Math.pow(1 - i / length, 2);
        channelData[i] = (Math.random() * 2 - 1) * decay * 0.5;
      }
    }
    
    return impulse;
  }

  /**
   * Decode base64 audio to AudioBuffer
   */
  async decodeAudio(base64Data: string): Promise<AudioBuffer> {
    const context = new (window.AudioContext || (window as any).webkitAudioContext)();
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return await context.decodeAudioData(bytes.buffer);
  }

  /**
   * Create EQ nodes for the effect chain
   */
  private createEQNodes(context: AudioContext, params: AudioEffectParams): BiquadFilterNode[] {
    const nodes: BiquadFilterNode[] = [];
    
    // Low shelf (100Hz) - warmth and eqLow
    const lowShelf = context.createBiquadFilter();
    lowShelf.type = 'lowshelf';
    lowShelf.frequency.value = 100;
    // Combine warmth and eqLow
    const lowGain = (params.warmth * 10) + (params.eqLow * 10) - 5;
    lowShelf.gain.value = lowGain;
    nodes.push(lowShelf);
    
    // Peaking (1kHz) - resonance and eqMid
    const midPeak = context.createBiquadFilter();
    midPeak.type = 'peaking';
    midPeak.frequency.value = 1000;
    midPeak.Q.value = 1;
    const midGain = (params.resonance * 10) + (params.eqMid * 10) - 5;
    midPeak.gain.value = midGain;
    nodes.push(midPeak);
    
    // High shelf (10kHz) - clarity, breathiness, eqHigh
    const highShelf = context.createBiquadFilter();
    highShelf.type = 'highshelf';
    highShelf.frequency.value = 10000;
    const highGain = (params.clarity * 8) + (params.breathiness * 5) + (params.eqHigh * 10) - 5;
    highShelf.gain.value = highGain;
    nodes.push(highShelf);
    
    return nodes;
  }

  /**
   * Create dynamics compressor
   */
  private createCompressor(context: AudioContext, amount: number): DynamicsCompressorNode {
    const compressor = context.createDynamicsCompressor();
    compressor.threshold.value = -24 * (1 - amount);
    compressor.knee.value = 10;
    compressor.ratio.value = 4 * amount + 1;
    compressor.attack.value = 0.005;
    compressor.release.value = 0.1;
    return compressor;
  }

  /**
   * Create reverb convolver
   */
  private async createReverb(context: AudioContext, amount: number): Promise<ConvolverNode | null> {
    if (amount <= 0.05) return null;
    
    if (!this.reverbImpulse) {
      this.reverbImpulse = await this.generateReverbImpulse(context, 1.5);
    }
    
    const convolver = context.createConvolver();
    convolver.buffer = this.reverbImpulse;
    return convolver;
  }

  /**
   * Build the effect chain and play audio
   */
  async playWithEffects(
    base64Audio: string, 
    params: AudioEffectParams,
    onEnded?: () => void
  ): Promise<void> {
    // Stop any current playback
    this.stop();

    const context = new (window.AudioContext || (window as any).webkitAudioContext)();
    
    // Decode audio
    let buffer: AudioBuffer;
    try {
      buffer = await this.decodeAudio(base64Audio);
    } catch (e) {
      console.error('[AudioEffects] Failed to decode audio:', e);
      throw new Error('Failed to decode audio');
    }

    // Create source
    const source = context.createBufferSource();
    source.buffer = buffer;
    
    // Apply speed (playback rate)
    // Pitch shift is simulated by combining playback rate with resampling
    // Positive pitchShift = higher pitch = faster playback
    // Negative pitchShift = lower pitch = slower playback
    const pitchMultiplier = 1 + (params.pitchShift * 0.1); // ±10% per semitone step
    source.playbackRate.value = params.speed * pitchMultiplier;

    // Create effect nodes
    const eqNodes = this.createEQNodes(context, params);
    const compressor = params.compression > 0.1 ? this.createCompressor(context, params.compression) : null;
    const reverbNode = await this.createReverb(context, params.reverb);

    // Build chain: source -> EQ -> compressor -> reverb -> gain -> destination
    let lastNode: AudioNode = source;

    // Connect EQ
    for (const eqNode of eqNodes) {
      lastNode.connect(eqNode);
      lastNode = eqNode;
    }

    // Connect compressor
    if (compressor) {
      lastNode.connect(compressor);
      lastNode = compressor;
    }

    // Create dry/wet mix for reverb
    const gainNode = context.createGain();
    gainNode.gain.value = 0.8;

    if (reverbNode) {
      // Split: dry path and wet path
      const dryGain = context.createGain();
      dryGain.gain.value = 1 - (params.reverb * 0.5);
      
      const wetGain = context.createGain();
      wetGain.gain.value = params.reverb * 0.5;

      lastNode.connect(dryGain);
      lastNode.connect(reverbNode);
      reverbNode.connect(wetGain);

      dryGain.connect(gainNode);
      wetGain.connect(gainNode);
    } else {
      lastNode.connect(gainNode);
    }

    gainNode.connect(context.destination);

    // Store playback state
    this.currentPlayback = {
      source,
      gainNode,
      eqNodes,
      reverbNode,
      compressor,
      context,
      isPlaying: true,
      startTime: context.currentTime,
      pausedAt: 0,
      buffer,
      playbackRate: params.speed * pitchMultiplier
    };

    // Handle end
    source.onended = () => {
      if (this.currentPlayback?.source === source) {
        this.currentPlayback.isPlaying = false;
        onEnded?.();
      }
    };

    // Start playback
    source.start(0);
    console.log('[AudioEffects] Started playback with effects:', params);
  }

  /**
   * Stop playback
   */
  stop(): void {
    if (this.currentPlayback) {
      try {
        this.currentPlayback.source.stop();
        this.currentPlayback.source.disconnect();
        this.currentPlayback.gainNode.disconnect();
        this.currentPlayback.eqNodes.forEach(n => n.disconnect());
        if (this.currentPlayback.reverbNode) {
          this.currentPlayback.reverbNode.disconnect();
        }
        if (this.currentPlayback.compressor) {
          this.currentPlayback.compressor.disconnect();
        }
        this.currentPlayback.context.close();
      } catch (e) {
        // Ignore errors during cleanup
      }
      this.currentPlayback = null;
      console.log('[AudioEffects] Stopped playback');
    }
  }

  /**
   * Check if currently playing
   */
  isPlaying(): boolean {
    return this.currentPlayback?.isPlaying ?? false;
  }

  /**
   * Update effects in real-time (for parameter tweaking while playing)
   * Note: Some parameters require regeneration for full effect
   */
  updateEffects(params: AudioEffectParams): void {
    if (!this.currentPlayback) return;

    const { eqNodes, compressor, gainNode } = this.currentPlayback;

    // Update EQ
    if (eqNodes.length >= 3) {
      // Low shelf
      eqNodes[0].gain.value = (params.warmth * 10) + (params.eqLow * 10) - 5;
      // Mid peak
      eqNodes[1].gain.value = (params.resonance * 10) + (params.eqMid * 10) - 5;
      // High shelf
      eqNodes[2].gain.value = (params.clarity * 8) + (params.breathiness * 5) + (params.eqHigh * 10) - 5;
    }

    // Update compressor
    if (compressor && params.compression > 0.1) {
      compressor.threshold.value = -24 * (1 - params.compression);
      compressor.ratio.value = 4 * params.compression + 1;
    }

    // Update gain (master volume adjustment based on compression)
    gainNode.gain.value = 0.8 + (params.compression * 0.2);

    console.log('[AudioEffects] Updated effects in real-time');
  }

  /**
   * Convert persona voice params to effect params
   */
  static fromPersonaParams(params: {
    pitch?: number;
    speed?: number;
    warmth?: number;
    expressiveness?: number;
    stability?: number;
    clarity?: number;
    breathiness?: number;
    resonance?: number;
    reverb?: number;
    eq_low?: number;
    eq_mid?: number;
    eq_high?: number;
    compression?: number;
  }): AudioEffectParams {
    return {
      pitchShift: (params.pitch ?? 0) / 100, // Convert to -1 to 1 range
      speed: params.speed ?? 1.0,
      warmth: params.warmth ?? 0.5,
      clarity: params.clarity ?? 0.5,
      breathiness: params.breathiness ?? 0.3,
      resonance: params.resonance ?? 0.5,
      reverb: params.reverb ?? 0,
      eqLow: params.eq_low ?? 0.5,
      eqMid: params.eq_mid ?? 0.5,
      eqHigh: params.eq_high ?? 0.5,
      compression: params.compression ?? 0,
    };
  }
}

export const audioEffects = new AudioEffectsService();
