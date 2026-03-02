import type { TTSCreateRequest, TTSGenerateRequest, TTSResponse } from "@/types";

// TTS Engine types
export type TTSEngine = "off" | "qwen3" | "kitten";
export type Qwen3ModelSize = "0.6B" | "1.7B";

// Voice creation parameters - Comprehensive tuning
export interface VoiceCreationParams {
  // Reference audio (required for Qwen3, optional for KittenTTS)
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
  cuda_available: boolean;
  current_engine: string;
  qwen3_loaded_size: string | null;
}

export class TTSService {
  private baseUrl: string;
  private lastRequestTime: number = 0;
  private minRequestInterval: number = 500; // Minimum 500ms between requests

  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  setBaseUrl(url: string) {
    this.baseUrl = url;
  }

  // Rate limiting helper - ensures we don't overwhelm the backend
  private async rateLimit(): Promise<void> {
    const now = Date.now();
    const timeSinceLastRequest = now - this.lastRequestTime;
    if (timeSinceLastRequest < this.minRequestInterval) {
      const delay = this.minRequestInterval - timeSinceLastRequest;
      await new Promise(resolve => setTimeout(resolve, delay));
    }
    this.lastRequestTime = Date.now();
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
    // Qwen3 handles text naturally - don't preprocess to avoid compound word separation
    const shouldPreprocess = params.engine !== 'qwen3';
    const processedRequest = {
      text: shouldPreprocess ? this.preprocessTextForTTS(text) : text,
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
    // generateWithProfile is only used for Qwen3 - don't preprocess
    const processedRequest = {
      text: text,
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

  // Default TTS using KittenTTS (no reference audio required)
  async generateSpeech(request: TTSGenerateRequest): Promise<TTSResponse> {
    const processedText = this.preprocessTextForTTS(request.text);

    const response = await fetch(`${this.baseUrl}/api/tts/kitten`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: processedText,
        voice: request.voice_id || "Bella",
        model: "KittenML/kitten-tts-nano-0.8",
        speed: request.speed || 1.0,
        pitch: 0,
      }),
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

  // ============================================
  // KittenTTS Integration (Local Backend)
  // ============================================
  
  async generateKittenTTS(
    text: string, 
    voice: string = "Bella",
    model: string = "nano",
    speed: number = 1.0,
    retryCount: number = 0
  ): Promise<{ audio_data: string; format: string; sample_rate: number }> {
    const maxRetries = 3;
    const retryDelay = 1000; // 1 second

    // Apply rate limiting to prevent overwhelming the backend
    await this.rateLimit();

    try {
      // Call local backend
      const response = await fetch(`${this.baseUrl}/api/tts/kitten`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          voice,
          model, // nano, micro, or mini
          speed, // 0.5 to 2.0
        }),
      });

      if (!response.ok) {
        // If server error and we haven't exhausted retries, try again
        if (response.status >= 500 && retryCount < maxRetries) {
          console.warn(`[KittenTTS] Server error ${response.status}, retrying ${retryCount + 1}/${maxRetries}...`);
          await new Promise(resolve => setTimeout(resolve, retryDelay * (retryCount + 1)));
          return this.generateKittenTTS(text, voice, model, speed, retryCount + 1);
        }
        
        let errorDetail = `KittenTTS request failed: ${response.status}`;
        try {
          const error = await response.json();
          errorDetail = error.detail || JSON.stringify(error);
        } catch (e) {
          // If JSON parsing fails, use status text
          errorDetail = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorDetail);
      }

      const result = await response.json();
      
      return {
        audio_data: result.audio_data,
        format: result.format || "wav",
        sample_rate: result.sample_rate || 24000
      };
    } catch (error) {
      // Network or other errors - retry if we haven't exhausted retries
      if (retryCount < maxRetries) {
        console.warn(`[KittenTTS] Request failed, retrying ${retryCount + 1}/${maxRetries}...`, error);
        await new Promise(resolve => setTimeout(resolve, retryDelay * (retryCount + 1)));
        return this.generateKittenTTS(text, voice, model, speed, retryCount + 1);
      }
      throw error;
    }
  }

  // Available KittenTTS voices (from huggingface.co/spaces/KittenML/KittenTTS-Demo)
  getKittenTTSVoices(): { id: string; name: string; description: string }[] {
    return [
      { id: "Bella", name: "Bella", description: "Female voice" },
      { id: "Jasper", name: "Jasper", description: "Male voice" },
      { id: "Luna", name: "Luna", description: "Female voice" },
      { id: "Bruno", name: "Bruno", description: "Male voice" },
      { id: "Rosie", name: "Rosie", description: "Female voice" },
      { id: "Hugo", name: "Hugo", description: "Male voice" },
      { id: "Kiki", name: "Kiki", description: "Female voice" },
      { id: "Leo", name: "Leo", description: "Male voice" },
    ];
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
      engine: "kitten",
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
      engine: "kitten",
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

  // ==================== STREAMING TTS ====================
  
  /**
   * Split text into natural chunks for streaming TTS
   * Chunks are sized for optimal generation speed and smooth playback
   */
  private splitTextIntoChunks(text: string, maxChunkLength: number = 150): string[] {
    // Split by sentence endings first
    const sentenceRegex = /[^.!?]+[.!?]+["'\s]*/g;
    const sentences = text.match(sentenceRegex) || [text];
    
    const chunks: string[] = [];
    let currentChunk = "";
    
    for (const sentence of sentences) {
      const trimmed = sentence.trim();
      if (!trimmed) continue;
      
      // If sentence itself is too long, split by clauses (commas, semicolons)
      if (trimmed.length > maxChunkLength) {
        // Save current chunk if exists
        if (currentChunk.trim()) {
          chunks.push(currentChunk.trim());
          currentChunk = "";
        }
        
        // Split long sentence by clauses
        const clauseRegex = /[^,;]+[,;]*/g;
        const clauses = trimmed.match(clauseRegex) || [trimmed];
        let clauseChunk = "";
        
        for (const clause of clauses) {
          if (clauseChunk.length + clause.length > maxChunkLength && clauseChunk.length > 0) {
            chunks.push(clauseChunk.trim());
            clauseChunk = clause;
          } else {
            clauseChunk += clause;
          }
        }
        
        if (clauseChunk.trim()) {
          chunks.push(clauseChunk.trim());
        }
      } else if (currentChunk.length + trimmed.length > maxChunkLength && currentChunk.length > 0) {
        // Start new chunk
        chunks.push(currentChunk.trim());
        currentChunk = trimmed;
      } else {
        currentChunk += " " + trimmed;
      }
    }
    
    // Add final chunk
    if (currentChunk.trim()) {
      chunks.push(currentChunk.trim());
    }
    
    return chunks.length > 0 ? chunks : [text];
  }
  
  /**
   * Streaming TTS - generates audio chunks in parallel for smooth playback
   * Similar to YouTube/Spotify streaming approach
   */
  async *streamTTS(
    text: string,
    options: {
      engine: "qwen3" | "kitten";
      voiceParams?: VoiceCreationParams;
      referenceAudio?: string;
      referenceText?: string;
      kittenVoice?: string;
      kittenModel?: string;
      speed?: number;
      maxChunkLength?: number;
    }
  ): AsyncGenerator<{ chunkIndex: number; totalChunks: number; audioData: string; text: string }, void, unknown> {
    const chunks = this.splitTextIntoChunks(text, options.maxChunkLength || 150);
    const totalChunks = chunks.length;
    
    console.log(`[StreamingTTS] Split text into ${totalChunks} chunks`);
    
    if (totalChunks === 1) {
      // Single chunk - generate directly
      let audioData: string;
      
      if (options.engine === "qwen3" && options.referenceAudio) {
        const result = await this.createVoice(text, {
          reference_audio: options.referenceAudio,
          reference_text: options.referenceText,
          pitch_shift: options.voiceParams?.pitch_shift ?? 0,
          speed: (options.voiceParams?.speed ?? 1.0) * (options.speed ?? 1.0),
          warmth: options.voiceParams?.warmth ?? 0.6,
          expressiveness: options.voiceParams?.expressiveness ?? 0.7,
          stability: options.voiceParams?.stability ?? 0.5,
          clarity: options.voiceParams?.clarity ?? 0.6,
          breathiness: options.voiceParams?.breathiness ?? 0.3,
          resonance: options.voiceParams?.resonance ?? 0.5,
          emotion: options.voiceParams?.emotion ?? "neutral",
          emphasis: options.voiceParams?.emphasis ?? 0.5,
          pauses: options.voiceParams?.pauses ?? 0.5,
          energy: options.voiceParams?.energy ?? 0.6,
          reverb: options.voiceParams?.reverb ?? 0,
          eq_low: options.voiceParams?.eq_low ?? 0.5,
          eq_mid: options.voiceParams?.eq_mid ?? 0.5,
          eq_high: options.voiceParams?.eq_high ?? 0.5,
          compression: options.voiceParams?.compression ?? 0.3,
          engine: "qwen3",
          qwen3_model_size: options.voiceParams?.qwen3_model_size ?? "0.6B",
          use_flash_attention: options.voiceParams?.use_flash_attention ?? true,
          seed: options.voiceParams?.seed,
        });
        audioData = result.audio_data;
      } else {
        const result = await this.generateKittenTTS(
          text,
          options.kittenVoice || "Bella",
          options.kittenModel || "nano",
          options.speed ?? 1.0
        );
        audioData = result.audio_data;
      }
      
      yield { chunkIndex: 0, totalChunks: 1, audioData, text };
      return;
    }
    
    // Multiple chunks - generate first chunk immediately, rest in parallel
    const chunkPromises: Promise<{ index: number; audioData: string; error?: string }>[] = [];
    
    // Start generating all chunks in parallel
    for (let i = 0; i < totalChunks; i++) {
      const chunkText = chunks[i];
      
      chunkPromises.push(
        (async () => {
          try {
            let audioData: string;
            
            if (options.engine === "qwen3" && options.referenceAudio) {
              const result = await this.createVoice(chunkText, {
                reference_audio: options.referenceAudio,
                reference_text: options.referenceText,
                pitch_shift: options.voiceParams?.pitch_shift ?? 0,
                speed: (options.voiceParams?.speed ?? 1.0) * (options.speed ?? 1.0),
                warmth: options.voiceParams?.warmth ?? 0.6,
                expressiveness: options.voiceParams?.expressiveness ?? 0.7,
                stability: options.voiceParams?.stability ?? 0.5,
                clarity: options.voiceParams?.clarity ?? 0.6,
                breathiness: options.voiceParams?.breathiness ?? 0.3,
                resonance: options.voiceParams?.resonance ?? 0.5,
                emotion: options.voiceParams?.emotion ?? "neutral",
                emphasis: options.voiceParams?.emphasis ?? 0.5,
                pauses: options.voiceParams?.pauses ?? 0.5,
                energy: options.voiceParams?.energy ?? 0.6,
                reverb: options.voiceParams?.reverb ?? 0,
                eq_low: options.voiceParams?.eq_low ?? 0.5,
                eq_mid: options.voiceParams?.eq_mid ?? 0.5,
                eq_high: options.voiceParams?.eq_high ?? 0.5,
                compression: options.voiceParams?.compression ?? 0.3,
                engine: "qwen3",
                qwen3_model_size: options.voiceParams?.qwen3_model_size ?? "0.6B",
                use_flash_attention: options.voiceParams?.use_flash_attention ?? true,
                seed: options.voiceParams?.seed,
              });
              audioData = result.audio_data;
            } else {
              const result = await this.generateKittenTTS(
                chunkText,
                options.kittenVoice || "Bella",
                options.kittenModel || "nano",
                options.speed ?? 1.0
              );
              audioData = result.audio_data;
            }
            
            return { index: i, audioData };
          } catch (error) {
            return { 
              index: i, 
              audioData: "", 
              error: error instanceof Error ? error.message : "Unknown error" 
            };
          }
        })()
      );
    }
    
    // Yield chunks as they complete (in order)
    const results = await Promise.all(chunkPromises);
    
    // Sort by index to maintain order
    results.sort((a, b) => a.index - b.index);
    
    for (const result of results) {
      if (result.error) {
        console.error(`[StreamingTTS] Chunk ${result.index} failed:`, result.error);
        continue;
      }
      
      yield {
        chunkIndex: result.index,
        totalChunks: totalChunks,
        audioData: result.audioData,
        text: chunks[result.index],
      };
    }
  }
}

export const ttsService = new TTSService();
