/**
 * Gapless Streaming TTS Service
 * Uses Web Audio API for seamless audio playback
 * Audio is queued and played back-to-back without gaps
 */

import { ttsService, type SyntheticVoiceParams } from "./tts";

interface AudioChunk {
  audioData: string;
  durationMs: number;
  decodedBuffer?: AudioBuffer;
}

export interface GaplessVoiceOptions {
  voiceId: string;
  speed: number;
  volume: number;
  voiceMode: "default" | "synthetic" | "legacy";
  syntheticParams?: SyntheticVoiceParams;
  // Legacy fields (deprecated)
  iscreated?: boolean;
  createAudioData?: string;
  createReferenceText?: string;
}

export class GaplessStreamingTTSService {
  private audioContext: AudioContext | null = null;
  private audioQueue: AudioChunk[] = [];
  private isGenerating = false;
  private isPlaying = false;
  private currentSource: AudioBufferSourceNode | null = null;
  private nextStartTime = 0;
  private abortController: AbortController | null = null;
  private volume = 1.0;
  private gainNode: GainNode | null = null;
  
  // Buffer settings
  private readonly INITIAL_BUFFER_MS = 500; // Start playing after 500ms of audio buffered

  private async initAudioContext(): Promise<void> {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.gainNode = this.audioContext.createGain();
      this.gainNode.connect(this.audioContext.destination);
      this.updateVolume();
    }
    
    // Resume if suspended (browser autoplay policy)
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  private updateVolume(): void {
    if (this.gainNode) {
      this.gainNode.gain.value = this.volume;
    }
  }

  // Split text into natural chunks (sentences or phrases)
  private splitIntoChunks(text: string): string[] {
    // First try to split by sentence endings
    const sentenceRegex = /[^.!?。！？]+[.!?。！？]+["\']*/g;
    const sentences = text.match(sentenceRegex);
    
    if (!sentences || sentences.length <= 1) {
      // If only one sentence, split by clauses (commas, semicolons, etc.)
      const clauseRegex = /[^,;:\n]+[,;:\n]*/g;
      const clauses = text.match(clauseRegex);
      if (clauses && clauses.length > 1) {
        // Group clauses into chunks of ~100 chars
        const chunks: string[] = [];
        let currentChunk = "";
        for (const clause of clauses) {
          if (currentChunk.length + clause.length > 100 && currentChunk.length > 0) {
            chunks.push(currentChunk.trim());
            currentChunk = clause;
          } else {
            currentChunk += clause;
          }
        }
        if (currentChunk.trim()) {
          chunks.push(currentChunk.trim());
        }
        return chunks.length > 0 ? chunks : [text.trim()];
      }
      return [text.trim()];
    }

    return sentences.map(s => s.trim()).filter(s => s.length > 0);
  }

  // Decode base64 audio to AudioBuffer
  private async decodeAudio(audioData: string): Promise<AudioBuffer> {
    if (!this.audioContext) {
      throw new Error("AudioContext not initialized");
    }

    const binaryString = atob(audioData);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    return await this.audioContext.decodeAudioData(bytes.buffer);
  }

  // Generate audio for a chunk
  private async generateChunkAudio(
    chunk: string,
    options: GaplessVoiceOptions
  ): Promise<AudioChunk | null> {
    try {
      if (options.voiceMode === "synthetic" && options.syntheticParams) {
        // NEW: Synthetic voice generation with parameters
        const response = await ttsService.synthesizeSynthetic({
          text: chunk,
          params: {
            ...options.syntheticParams,
            speed: (options.syntheticParams.speed || 1.0) * options.speed,
          },
          language: "English",
        });
        const decodedBuffer = await this.decodeAudio(response.audio_data);
        return {
          audioData: response.audio_data,
          durationMs: response.duration_ms,
          decodedBuffer,
        };
      } else if (options.iscreated && options.createAudioData) {
        // LEGACY: Created voice (deprecated)
        console.warn("GaplessStreamingTTS: Legacy created voices are deprecated. Use synthetic voices instead.");
        // Fallback to default voice
        const response = await ttsService.generateSpeech({
          text: chunk,
          voice_id: options.voiceId,
          speed: options.speed,
        });
        const decodedBuffer = await this.decodeAudio(response.audio_data);
        return {
          audioData: response.audio_data,
          durationMs: response.duration_ms,
          decodedBuffer,
        };
      } else {
        // Default voice
        const response = await ttsService.generateSpeech({
          text: chunk,
          voice_id: options.voiceId,
          speed: options.speed,
        });
        const decodedBuffer = await this.decodeAudio(response.audio_data);
        return {
          audioData: response.audio_data,
          durationMs: response.duration_ms,
          decodedBuffer,
        };
      }
    } catch (error) {
      console.error("Failed to generate audio for chunk:", chunk, error);
      return null;
    }
  }

  // Schedule audio playback using precise timing
  private schedulePlayback(): void {
    if (!this.audioContext || !this.gainNode) return;

    // Process queued chunks that have decoded buffers
    while (this.audioQueue.length > 0 && this.audioQueue[0].decodedBuffer) {
      const chunk = this.audioQueue.shift()!;
      
      // Create source node
      const source = this.audioContext.createBufferSource();
      source.buffer = chunk.decodedBuffer!;
      source.connect(this.gainNode);

      // Calculate precise start time
      const currentTime = this.audioContext.currentTime;
      const startTime = Math.max(this.nextStartTime, currentTime);
      
      // Schedule playback
      source.start(startTime);
      this.currentSource = source;
      
      // Update next start time for gapless playback
      this.nextStartTime = startTime + chunk.decodedBuffer!.duration;

      // Clean up when done
      source.onended = () => {
        if (this.currentSource === source) {
          this.currentSource = null;
        }
      };

      console.log(`🎵 Scheduled chunk at ${startTime.toFixed(3)}s, duration: ${chunk.decodedBuffer!.duration.toFixed(3)}s`);
    }
  }

  // Main streaming speak method
  async speakStreaming(
    text: string,
    options: GaplessVoiceOptions & { onProgress?: (current: number, total: number) => void }
  ): Promise<void> {
    // Cancel any ongoing speech
    this.stop();
    
    this.abortController = new AbortController();
    const { signal } = this.abortController;
    
    this.volume = options.volume;
    await this.initAudioContext();
    this.updateVolume();

    const chunks = this.splitIntoChunks(text);
    console.log(`🎵 Gapless Streaming TTS: ${chunks.length} chunks (mode: ${options.voiceMode})`);
    
    this.isGenerating = true;
    this.isPlaying = true;
    this.nextStartTime = 0;
    
    // Pre-generate first chunk(s) before starting playback
    const preGenCount = Math.min(2, chunks.length);
    const generatePromises: Promise<AudioChunk | null>[] = [];
    
    for (let i = 0; i < preGenCount; i++) {
      generatePromises.push(
        this.generateChunkAudio(chunks[i], options)
      );
    }

    let totalBufferedMs = 0;

    // Process each chunk
    for (let i = 0; i < chunks.length; i++) {
      if (signal.aborted) break;

      // Wait for current chunk to be ready
      const audioChunk = await generatePromises[i];

      // Start generating next chunk while processing current
      if (i + preGenCount < chunks.length) {
        generatePromises.push(
          this.generateChunkAudio(chunks[i + preGenCount], options)
        );
      }

      if (signal.aborted) break;

      if (audioChunk?.decodedBuffer) {
        this.audioQueue.push(audioChunk);
        totalBufferedMs += audioChunk.durationMs;
        options.onProgress?.(i + 1, chunks.length);

        // Start playback after initial buffer is ready
        if (!this.isPlaying && totalBufferedMs >= this.INITIAL_BUFFER_MS) {
          this.isPlaying = true;
          this.schedulePlayback();
        }

        // Continue scheduling as we generate
        if (this.isPlaying) {
          this.schedulePlayback();
        }
      } else {
        console.warn(`Skipping chunk ${i + 1} - audio generation failed`);
      }
    }

    // Ensure all remaining chunks are scheduled
    this.schedulePlayback();

    // Wait for all audio to finish playing
    if (this.audioContext && this.nextStartTime > 0) {
      const remainingDuration = this.nextStartTime - this.audioContext.currentTime;
      if (remainingDuration > 0) {
        await new Promise(resolve => setTimeout(resolve, remainingDuration * 1000));
      }
    }

    this.isGenerating = false;
    this.isPlaying = false;
  }

  // Stop current playback
  stop(): void {
    this.abortController?.abort();
    this.abortController = null;

    // Stop current source
    if (this.currentSource) {
      try {
        this.currentSource.stop();
        this.currentSource.disconnect();
      } catch (e) {
        // May already be stopped
      }
      this.currentSource = null;
    }

    // Clear queue
    this.audioQueue = [];
    this.nextStartTime = 0;
    this.isGenerating = false;
    this.isPlaying = false;
  }

  // Check if currently speaking
  isSpeaking(): boolean {
    return this.isPlaying || this.isGenerating;
  }
}

// Singleton instance
export const gaplessStreamingTTS = new GaplessStreamingTTSService();
