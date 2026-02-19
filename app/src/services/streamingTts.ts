/**
 * Streaming TTS Service
 * Splits text into sentences and plays them sequentially
 * Reduces perceived latency by starting playback immediately
 */

import { ttsService, type SyntheticVoiceParams } from "./tts";

interface QueuedAudio {
  audioData: string;
  durationMs: number;
}

export interface StreamingVoiceOptions {
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

export class StreamingTTSService {
  private isPlaying = false;
  private abortController: AbortController | null = null;
  private currentAudio: HTMLAudioElement | null = null;
  private onProgressCallback?: (currentSentence: number, totalSentences: number) => void;

  // Split text into sentences (handles common punctuation)
  private splitIntoSentences(text: string): string[] {
    // Match sentences ending with . ! ? followed by space or end of string
    // Also handle Chinese/Japanese punctuation
    const sentenceRegex = /[^.!?。！？]+[.!?。！？]+["\']*/g;
    const matches = text.match(sentenceRegex);
    
    if (!matches || matches.length === 0) {
      // If no sentence breaks found, treat entire text as one sentence
      return [text.trim()];
    }

    // Clean up and filter empty sentences
    return matches
      .map(s => s.trim())
      .filter(s => s.length > 0);
  }

  // Pre-generate audio for a sentence
  private async generateSentenceAudio(
    sentence: string,
    options: StreamingVoiceOptions
  ): Promise<QueuedAudio | null> {
    try {
      if (options.voiceMode === "synthetic" && options.syntheticParams) {
        // NEW: Synthetic voice generation with parameters
        const response = await ttsService.synthesizeSynthetic({
          text: sentence,
          params: {
            ...options.syntheticParams,
            speed: (options.syntheticParams.speed || 1.0) * options.speed,
          },
          language: "English",
        });
        return {
          audioData: response.audio_data,
          durationMs: response.duration_ms,
        };
      } else if (options.iscreated && options.createAudioData) {
        // LEGACY: Created voice (deprecated)
        console.warn("StreamingTTS: Legacy created voices are deprecated. Use synthetic voices instead.");
        // Fallback to default voice
        const response = await ttsService.generateSpeech({
          text: sentence,
          voice_id: options.voiceId,
          speed: options.speed,
        });
        return {
          audioData: response.audio_data,
          durationMs: response.duration_ms,
        };
      } else {
        // Default voice
        const response = await ttsService.generateSpeech({
          text: sentence,
          voice_id: options.voiceId,
          speed: options.speed,
        });
        return {
          audioData: response.audio_data,
          durationMs: response.duration_ms,
        };
      }
    } catch (error) {
      console.error("Failed to generate audio for sentence:", sentence, error);
      return null;
    }
  }

  // Play a single audio blob
  private playAudioBlob(audioData: string, volume: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const audio = new Audio(`data:audio/wav;base64,${audioData}`);
      audio.volume = volume;
      this.currentAudio = audio;

      audio.onended = () => {
        this.currentAudio = null;
        resolve();
      };

      audio.onerror = (e) => {
        this.currentAudio = null;
        reject(e);
      };

      audio.play().catch((err) => {
        this.currentAudio = null;
        reject(err);
      });
    });
  }

  // Main streaming speak method
  async speakStreaming(
    text: string,
    options: StreamingVoiceOptions & { onProgress?: (current: number, total: number) => void }
  ): Promise<void> {
    // Cancel any ongoing speech
    this.stop();
    this.abortController = new AbortController();
    const { signal } = this.abortController;

    const sentences = this.splitIntoSentences(text);
    console.log(`🎵 Streaming TTS: ${sentences.length} sentences (mode: ${options.voiceMode})`);
    
    this.onProgressCallback = options.onProgress;
    this.isPlaying = true;

    // Start generating first sentence immediately
    const generatePromises: Promise<QueuedAudio | null>[] = [];
    
    // Pre-generate first 2 sentences in parallel
    const preGenCount = Math.min(2, sentences.length);
    for (let i = 0; i < preGenCount; i++) {
      generatePromises.push(
        this.generateSentenceAudio(sentences[i], options)
      );
    }

    // Play sentences as they become ready
    for (let i = 0; i < sentences.length; i++) {
      if (signal.aborted) break;

      // Wait for current sentence audio to be ready
      const audio = await generatePromises[i];
      
      // Start generating next sentence while playing current
      if (i + preGenCount < sentences.length) {
        generatePromises.push(
          this.generateSentenceAudio(sentences[i + preGenCount], options)
        );
      }

      if (signal.aborted) break;

      if (audio) {
        this.onProgressCallback?.(i + 1, sentences.length);
        console.log(`🎵 Playing sentence ${i + 1}/${sentences.length}: "${sentences[i].substring(0, 50)}..."`);
        
        try {
          await this.playAudioBlob(audio.audioData, options.volume);
        } catch (error) {
          console.warn(`Failed to play sentence ${i + 1}:`, error);
        }
      } else {
        console.warn(`Skipping sentence ${i + 1} - audio generation failed`);
      }

      // Small pause between sentences for natural flow
      if (i < sentences.length - 1 && !signal.aborted) {
        await new Promise(resolve => setTimeout(resolve, 150));
      }
    }

    this.isPlaying = false;
    this.onProgressCallback = undefined;
  }

  // Stop current playback
  stop() {
    this.abortController?.abort();
    this.abortController = null;
    
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio = null;
    }
    
    this.isPlaying = false;
  }

  // Check if currently speaking
  isSpeaking(): boolean {
    return this.isPlaying;
  }
}

// Singleton instance
export const streamingTTS = new StreamingTTSService();
