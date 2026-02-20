import type { TTSCreateRequest, TTSGenerateRequest, TTSResponse } from "@/types";

// TTS Engine types
export type TTSEngine = "off" | "browser" | "qwen3";
export type Qwen3ModelSize = "0.6B" | "1.7B";

// Voice creation parameters - Comprehensive tuning
export interface VoiceCreationParams {
  // Reference audio (optional for StyleTTS2, required for Qwen3)
  reference_audio?: string; // Base64 encoded
  reference_text?: string;  // Transcript of reference audio
  
  // Basic voice tuning (required)
  pitch_shift: number;  // -1.0 to 1.0
  speed: number;        // 0.5 to 2.0
  
  // Advanced voice characteristics (optional with defaults)
  warmth?: number;           // 0.0 to 1.0 - naturalness
  expressiveness?: number;   // 0.0 to 1.0 - emotional variation
  stability?: number;        // 0.0 to 1.0 - consistency
  clarity?: number;          // 0.0 to 1.0 - articulation
  breathiness?: number;      // 0.0 to 1.0 - air in voice
  resonance?: number;        // 0.0 to 1.0 - depth/fullness
  
  // Speech characteristics (optional with defaults)
  emotion?: "neutral" | "happy" | "sad" | "angry" | "excited" | "calm";
  emphasis?: number;         // 0.0 to 1.0 - word stress
  pauses?: number;           // 0.0 to 1.0 - pause length
  energy?: number;           // 0.0 to 1.0 - vocal energy
  
  // Audio effects (post-processing, optional with defaults)
  reverb?: number;           // 0.0 to 1.0 - room ambiance
  eq_low?: number;           // 0.0 to 1.0 - bass
  eq_mid?: number;           // 0.0 to 1.0 - mids
  eq_high?: number;          // 0.0 to 1.0 - treble
  compression?: number;      // 0.0 to 1.0 - dynamic compression
  
  // Engine selection
  engine: TTSEngine;
  qwen3_model_size: Qwen3ModelSize;
  use_flash_attention: boolean;
  
  // Voice profile extraction (1.7B for quality, playback with 0.6B)
  extraction_model_size?: "0.6B" | "1.7B";
  save_voice_profile?: boolean;
  persona_id?: string;
  
  // Seed for reproducibility
  seed?: number;
}

// Legacy synthetic params (for backward compatibility)
export interface SyntheticVoiceParams {
  gender: "neutral" | "masculine" | "feminine";
  age: "young" | "adult" | "mature";
  pitch: number;
  speed: number;
  warmth: number;
  expressiveness: number;
  stability: number;
  seed?: number;
}

export interface EngineStatus {
  qwen3_available: boolean;
  browser_tts_available: boolean;
  cuda_available: boolean;
  current_engine: string;
  qwen3_loaded_size: string | null;
}

export class TTSService {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  setBaseUrl(url: string) {
    this.baseUrl = url;
  }

  async checkConnection(): Promise<boolean> {
    // Try with CORS mode first
    try {
      const response = await fetch(`${this.baseUrl}/health`, { 
        method: "GET",
        signal: AbortSignal.timeout(3000)
      });
      if (response.ok) {
        console.log("[TTS] Connection check successful (CORS mode)");
        return true;
      }
    } catch (error: any) {
      console.log("[TTS] CORS check failed:", error.message || error);
    }
    
    // Try with no-cors mode for Tauri WebView
    try {
      await fetch(`${this.baseUrl}/health`, { 
        method: "GET",
        mode: "no-cors",
        signal: AbortSignal.timeout(3000)
      });
      // With no-cors, we can't read response but if no error, assume OK
      console.log("[TTS] Connection check successful (no-cors mode)");
      return true;
    } catch (error: any) {
      console.error("[TTS] Both connection checks failed:", error.message || error);
      return false;
    }
  }

  // Get engine status
  async getEngineStatus(): Promise<EngineStatus> {
    const response = await fetch(`${this.baseUrl}/api/engines/status`, {
      method: "GET",
    });

    if (!response.ok) {
      throw new Error("Failed to get engine status");
    }

    return await response.json();
  }

  // Upload reference audio for voice creation
  async uploadReferenceAudio(
    voiceId: string,
    audioData: string,
    referenceText?: string
  ): Promise<{
    status: string;
    voice_id: string;
    reference_path: string;
    message: string;
  }> {
    const response = await fetch(`${this.baseUrl}/api/voice/upload-reference`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        voice_id: voiceId,
        audio_data: audioData,
        reference_text: referenceText,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to upload reference audio");
    }

    return await response.json();
  }

  // Create voice from reference audio
  async createVoice(
    text: string,
    params: VoiceCreationParams
  ): Promise<TTSResponse & { engine_used: string }> {
    const processedRequest = {
      text: this.preprocessTextForTTS(text),
      reference_audio: params.reference_audio,
      reference_text: params.reference_text,
      // Basic tuning
      pitch_shift: params.pitch_shift,
      speed: params.speed,
      // Voice characteristics
      warmth: params.warmth,
      expressiveness: params.expressiveness,
      stability: params.stability,
      clarity: params.clarity,
      breathiness: params.breathiness,
      resonance: params.resonance,
      // Speech characteristics
      emotion: params.emotion,
      emphasis: params.emphasis,
      pauses: params.pauses,
      energy: params.energy,
      // Audio effects
      reverb: params.reverb,
      eq_low: params.eq_low,
      eq_mid: params.eq_mid,
      eq_high: params.eq_high,
      compression: params.compression,
      // Engine selection
      engine: params.engine,
      qwen3_model_size: params.qwen3_model_size,
      use_flash_attention: params.use_flash_attention,
      // Voice profile
      extraction_model_size: params.extraction_model_size,
      save_voice_profile: params.save_voice_profile,
      persona_id: params.persona_id,
      // Reproducibility
      seed: params.seed,
    };
    
    console.log(`[TTS] Creating voice with engine: ${params.engine}, qwen3_size: ${params.qwen3_model_size}`);

    const response = await fetch(`${this.baseUrl}/api/voice/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(processedRequest),
    });

    if (!response.ok) {
      let errorMessage = "Voice creation failed";
      try {
        const error = await response.json();
        errorMessage = error.detail || JSON.stringify(error) || errorMessage;
      } catch (e) {
        errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      }
      throw new Error(errorMessage);
    }

    return await response.json();
  }

  // Extract voice profile using 1.7B model for high-quality capture
  async extractVoiceProfile(params: {
    persona_id: string;
    reference_audio: string;
    reference_text: string;
    extraction_model_size?: "0.6B" | "1.7B";
    voice_params?: Partial<VoiceCreationParams>;
  }): Promise<{
    status: string;
    persona_id: string;
    extraction_model: string;
    profile_path?: string;
    message: string;
  }> {
    const response = await fetch(`${this.baseUrl}/api/voice/extract-profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        persona_id: params.persona_id,
        reference_audio: params.reference_audio,
        reference_text: params.reference_text,
        extraction_model_size: params.extraction_model_size || "1.7B",
        save_profile: true,
        voice_params: params.voice_params || {},
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Voice profile extraction failed");
    }

    return await response.json();
  }

  // Generate voice using saved profile (0.6B for fast playback)
  async generateWithProfile(
    text: string,
    persona_id: string,
    params: Partial<VoiceCreationParams> & { playback_model_size?: "0.6B" | "1.7B" }
  ): Promise<TTSResponse & { engine_used: string; voice_profile_used: boolean }> {
    const processedRequest = {
      text: this.preprocessTextForTTS(text),
      persona_id,
      playback_model_size: params.playback_model_size || "0.6B",
      use_flash_attention: params.use_flash_attention ?? true,
      // All voice effect parameters
      pitch_shift: params.pitch_shift ?? 0,
      speed: params.speed ?? 1.0,
      warmth: params.warmth ?? 0.5,
      expressiveness: params.expressiveness ?? 0.5,
      stability: params.stability ?? 0.5,
      clarity: params.clarity ?? 0.5,
      breathiness: params.breathiness ?? 0.3,
      resonance: params.resonance ?? 0.5,
      emotion: params.emotion ?? "neutral",
      emphasis: params.emphasis ?? 0.5,
      pauses: params.pauses ?? 0.5,
      energy: params.energy ?? 0.5,
      reverb: params.reverb ?? 0,
      eq_low: params.eq_low ?? 0.5,
      eq_mid: params.eq_mid ?? 0.5,
      eq_high: params.eq_high ?? 0.5,
      compression: params.compression ?? 0.3,
    };

    const response = await fetch(`${this.baseUrl}/api/voice/generate-with-profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(processedRequest),
    });

    if (!response.ok) {
      let errorMessage = "Voice generation with profile failed";
      try {
        const error = await response.json();
        errorMessage = error.detail || JSON.stringify(error) || errorMessage;
      } catch (e) {
        errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      }
      throw new Error(errorMessage);
    }

    return await response.json();
  }

  // Default TTS (StyleTTS2 without reference)
  async generateSpeech(request: TTSGenerateRequest): Promise<TTSResponse> {
    const processedRequest = {
      ...request,
      text: this.preprocessTextForTTS(request.text),
    };

    const response = await fetch(`${this.baseUrl}/api/tts/default`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(processedRequest),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Speech generation failed");
    }

    return await response.json();
  }

  // Preprocess text for better TTS - removes Unicode chars that Windows can't handle
  private preprocessTextForTTS(text: string): string {
    // Remove extra whitespace
    let processed = text.replace(/\s{2,}/g, ' ').trim();
    
    // Replace problematic Unicode characters with ASCII equivalents
    const replacements: Record<string, string> = {
      // IPA vowels
      '\u025b': 'e',  // Latin Small Letter Open E (ɛ)
      '\u0259': 'e',  // Schwa (ə)
      '\u025c': 'e',  // Reversed Open E (ɜ)
      '\u026a': 'i',  // Latin Small Letter Iota (ɪ)
      '\u028a': 'u',  // Latin Small Letter Upsilon (ʊ)
      '\u0254': 'o',  // Latin Small Letter Open O (ɔ)
      '\u00e6': 'ae', // Latin Small Letter AE (æ)
      '\u0153': 'oe', // Latin Small Ligature OE (œ)
      // IPA consonants
      '\u0283': 'sh', // Latin Small Letter Esh (ʃ)
      '\u0292': 'zh', // Latin Small Letter Ezh (ʒ)
      '\u03b8': 'th', // Greek Small Letter Theta (θ)
      '\u00f0': 'th', // Latin Small Letter Eth (ð)
      '\u014b': 'ng', // Latin Small Letter Eng (ŋ)
      // Smart quotes
      '\u2019': "'",  // Right Single Quotation Mark
      '\u2018': "'",  // Left Single Quotation Mark
      '\u201c': '"',  // Left Double Quotation Mark
      '\u201d': '"',  // Right Double Quotation Mark
      // Dashes
      '\u2013': '-',  // En Dash
      '\u2014': '-',  // Em Dash
      '\u2026': '...', // Horizontal Ellipsis
    };
    
    for (const [char, replacement] of Object.entries(replacements)) {
      processed = processed.split(char).join(replacement);
    }
    
    // Remove any remaining non-ASCII characters
    processed = processed.replace(/[^\x00-\x7F]/g, '');
    
    return processed;
  }

  // Play audio
  playAudio(audioData: string, volume: number = 1.0): HTMLAudioElement {
    const audio = new Audio(`data:audio/wav;base64,${audioData}`);
    audio.volume = volume;
    return audio;
  }

  // Browser TTS fallback
  speakWithBrowserTTS(text: string, volume: number = 1.0, rate: number = 1.0): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!("speechSynthesis" in window)) {
        reject(new Error("Browser TTS not supported"));
        return;
      }

      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = rate;
      utterance.volume = volume;
      utterance.pitch = 1;

      utterance.onend = () => resolve();
      utterance.onerror = (e) => reject(e);

      window.speechSynthesis.speak(utterance);
    });
  }

  // Unload models to free GPU memory
  async unloadModels(): Promise<{ status: string; message: string }> {
    const response = await fetch(`${this.baseUrl}/unload-models`, {
      method: "POST",
    });

    if (!response.ok) {
      throw new Error("Failed to unload models");
    }

    return await response.json();
  }

  // Detect watermark
  async detectWatermark(audioData: string): Promise<{
    detected: boolean;
    confidence: number;
    message: string;
  }> {
    const response = await fetch(`${this.baseUrl}/api/watermark/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audio_data: audioData }),
    });

    if (!response.ok) {
      throw new Error("Watermark detection failed");
    }

    return await response.json();
  }

  // ==================== LEGACY METHODS (for backward compatibility) ====================

  // Deprecated: Use createVoice instead
  async synthesizeSynthetic(request: {
    text: string;
    params: SyntheticVoiceParams;
    language?: string;
  }): Promise<TTSResponse> {
    console.warn("synthesizeSynthetic is deprecated. Use createVoice instead.");
    
    // Convert old synthetic params to new voice creation params
    const creationParams: VoiceCreationParams = {
      reference_audio: undefined,
      reference_text: undefined,
      pitch_shift: request.params.pitch,
      speed: request.params.speed,
      // Voice characteristics
      warmth: request.params.warmth,
      expressiveness: request.params.expressiveness,
      stability: request.params.stability,
      // Engine selection
      engine: "browser",
      qwen3_model_size: "0.6B",
      use_flash_attention: true,
      seed: request.params.seed,
    };

    const result = await this.createVoice(request.text, creationParams);
    return {
      audio_data: result.audio_data,
      duration_ms: result.duration_ms,
      sample_rate: result.sample_rate,
    };
  }

  // Deprecated: Use uploadReferenceAudio + createVoice instead
  async createVoiceRequest(_request: TTSCreateRequest): Promise<{ success: boolean; voice_id?: string; error?: string }> {
    console.warn("createVoiceRequest is deprecated. Use uploadReferenceAudio + createVoice instead.");
    return {
      success: false,
      error: "createVoiceRequest is deprecated. Use the new voice creation flow.",
    };
  }

  // Deprecated: Use createVoice instead
  async synthesizeWithcreate(
    text: string,
    audioData: string,
    referenceText?: string,
    speed: number = 1.0
  ): Promise<TTSResponse> {
    console.warn("synthesizeWithcreate is deprecated. Use createVoice instead.");
    
    const result = await this.createVoice(text, {
      reference_audio: audioData,
      reference_text: referenceText,
      pitch_shift: 0,
      speed: speed,
      engine: "browser",
      qwen3_model_size: "0.6B",
      use_flash_attention: true,
      // Optional voice params with defaults
      warmth: 0.5,
      expressiveness: 0.5,
      stability: 0.5,
    });

    return {
      audio_data: result.audio_data,
      duration_ms: result.duration_ms,
      sample_rate: result.sample_rate,
    };
  }

  // Stub methods for compatibility - TODO: implement or remove calls
  async preloadModel(): Promise<void> {
    console.log("[TTSService] preloadModel called (stub)");
    // TODO: Implement or remove from calling code
  }

  async synthesizeWithSavedVoice(_voiceId: string, _text: string, _language: string): Promise<TTSResponse> {
    console.warn("[TTSService] synthesizeWithSavedVoice is deprecated, use createVoice instead");
    throw new Error("synthesizeWithSavedVoice is deprecated. Use createVoice with reference audio.");
  }

  async synthesizeWithEnrolledVoice(_voiceId: string, _text: string, _language: string): Promise<TTSResponse> {
    console.warn("[TTSService] synthesizeWithEnrolledVoice is deprecated, use createVoice instead");
    throw new Error("synthesizeWithEnrolledVoice is deprecated. Use createVoice with reference audio.");
  }

  async enrollVoice(_voiceId: string, _audioData: string, _text: string): Promise<{ success: boolean }> {
    console.warn("[TTSService] enrollVoice is deprecated, use uploadReferenceAudio instead");
    throw new Error("enrollVoice is deprecated. Use uploadReferenceAudio.");
  }
}

export const ttsService = new TTSService();
