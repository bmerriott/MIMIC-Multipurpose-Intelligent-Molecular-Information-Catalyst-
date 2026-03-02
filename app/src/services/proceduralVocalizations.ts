/**
 * Procedural Vocalization Service
 * Generates and manages non-verbal vocalizations (giggles, sighs, etc.) using Qwen TTS
 * Only for Qwen TTS - Browser TTS does not support this
 */

import type { Persona } from "@/types";
import { ttsService } from "./tts";
import { personaLearning } from "./personaLearning";

// Vocalization types with example prompts for Qwen
const VOCALIZATION_PROMPTS: Record<string, { text: string; description: string }> = {
  giggle: {
    text: "tee hee",
    description: "light playful laugh"
  },
  sigh: {
    text: "sigh",
    description: "soft contented or resigned sigh"
  },
  hum: {
    text: "hmm",
    description: "thoughtful humming sound"
  },
  gasp: {
    text: "gasp",
    description: "surprised intake of breath"
  },
  yawn: {
    text: "yawn",
    description: "sleepy yawn sound"
  },
  laugh: {
    text: "ha ha",
    description: "brief cheerful laugh"
  },
  hmm: {
    text: "hmm",
    description: "thinking or considering sound"
  }
};

// Maximum storage per persona (in MB, approximate)
const MAX_STORAGE_MB = 5;
const AVG_VOCALIZATION_SIZE_KB = 30; // ~30KB per vocalization
const MAX_VOCALIZATIONS = Math.floor((MAX_STORAGE_MB * 1024) / AVG_VOCALIZATION_SIZE_KB);

export class ProceduralVocalizationService {
  private generating: Set<string> = new Set();

  /**
   * Check if a persona has a specific vocalization
   */
  hasVocalization(personaId: string, type: string): boolean {
    const vocalization = personaLearning.getProceduralVocalization(personaId, type);
    return vocalization !== null;
  }

  /**
   * Get a vocalization (plays from cache or returns null if not cached)
   */
  async playVocalization(
    personaId: string, 
    type: string
  ): Promise<{ audio_data: string; duration_ms: number } | null> {
    // Check cache first
    const cached = personaLearning.getProceduralVocalization(personaId, type);
    if (cached) {
      console.log(`[ProceduralVocal] Playing cached ${type} for ${personaId}`);
      return { 
        audio_data: cached.audio_data, 
        duration_ms: 1000 // Approximate
      };
    }

    // Generate if not cached
    return this.generateVocalization(personaId, type);
  }

  /**
   * Generate a new vocalization using Qwen TTS
   * Only works with Qwen TTS engine
   */
  async generateVocalization(
    personaId: string,
    type: string,
    forceRegenerate: boolean = false
  ): Promise<{ audio_data: string; duration_ms: number } | null> {
    // Prevent duplicate generation
    const key = `${personaId}_${type}`;
    if (this.generating.has(key)) {
      console.log(`[ProceduralVocal] Already generating ${type} for ${personaId}`);
      return null;
    }

    // Check if already exists (unless force regenerate)
    if (!forceRegenerate && this.hasVocalization(personaId, type)) {
      const cached = personaLearning.getProceduralVocalization(personaId, type);
      if (cached) {
        return { 
          audio_data: cached.audio_data, 
          duration_ms: 1000 
        };
      }
    }

    const prompt = VOCALIZATION_PROMPTS[type];
    if (!prompt) {
      console.error(`[ProceduralVocal] Unknown vocalization type: ${type}`);
      return null;
    }

    this.generating.add(key);
    console.log(`[ProceduralVocal] Generating ${type} for ${personaId}`);

    try {
      // Get persona voice config
      const persona = this.getPersona(personaId);
      if (!persona?.voice_create?.voice_config) {
        console.log("[ProceduralVocal] No voice config found");
        return null;
      }

      const config = persona.voice_create.voice_config;
      
      // Only works with Qwen TTS
      if (config.params?.engine !== 'qwen3') {
        console.log("[ProceduralVocal] Procedural vocalizations only available with Qwen TTS");
        return null;
      }

      // Generate using voice profile if available
      let result;
      if (config.params?.use_voice_profile) {
        result = await ttsService.generateWithProfile(
          prompt.text,
          personaId,
          {
            playback_model_size: config.params.qwen3_model_size || '0.6B',
            use_flash_attention: true,
            pitch_shift: config.params.pitch || 0,
            speed: (config.params.speed || 1.0) * 1.1, // Slightly faster for vocalizations
            warmth: config.params.warmth ?? 0.5,
            expressiveness: 0.8, // Higher expressiveness for vocalizations
            stability: 0.4, // Lower stability for more variation
          }
        );
      } else {
        // Fallback to createVoice with reference
        result = await ttsService.createVoice(prompt.text, {
          reference_audio: persona.voice_create.audio_data,
          reference_text: persona.voice_create.reference_text,
          pitch_shift: config.params?.pitch || 0,
          speed: (config.params?.speed || 1.0) * 1.1,
          warmth: config.params?.warmth ?? 0.5,
          expressiveness: 0.8,
          stability: 0.4,
          engine: 'qwen3',
          qwen3_model_size: config.params?.qwen3_model_size || '0.6B',
          use_flash_attention: true,
        });
      }

      // Store in learning data
      const stored = personaLearning.addProceduralVocalization(
        personaId,
        type,
        result.audio_data
      );

      if (stored) {
        console.log(`[ProceduralVocal] Generated and stored ${type} for ${personaId}`);
      } else {
        console.log(`[ProceduralVocal] Generated ${type} but storage limit reached`);
      }

      return {
        audio_data: result.audio_data,
        duration_ms: result.duration_ms
      };

    } catch (error) {
      console.error(`[ProceduralVocal] Failed to generate ${type}:`, error);
      return null;
    } finally {
      this.generating.delete(key);
    }
  }

  /**
   * Pre-generate common vocalizations for a persona
   * Call this when a persona's voice is first created
   */
  async pregenerateCommonVocalizations(personaId: string): Promise<void> {
    const persona = this.getPersona(personaId);
    if (!persona?.voice_create?.voice_config?.params?.use_voice_profile) {
      console.log("[ProceduralVocal] Skipping pregeneration - no voice profile");
      return;
    }

    const commonTypes = ['giggle', 'sigh', 'hum', 'gasp'];
    
    console.log(`[ProceduralVocal] Pregenerating vocalizations for ${personaId}`);
    
    for (const type of commonTypes) {
      // Skip if already exists
      if (this.hasVocalization(personaId, type)) continue;
      
      try {
        await this.generateVocalization(personaId, type);
        // Small delay to not overwhelm the backend
        await new Promise(r => setTimeout(r, 500));
      } catch (e) {
        console.error(`[ProceduralVocal] Failed to pregenerate ${type}:`, e);
      }
    }
  }

  /**
   * Get storage stats for a persona's vocalizations
   */
  getStorageStats(personaId: string): { count: number; estimatedSizeKB: number; limit: number } {
    const persona = this.getPersona(personaId);
    const count = persona?.procedural_vocalizations?.length || 0;
    
    return {
      count,
      estimatedSizeKB: count * AVG_VOCALIZATION_SIZE_KB,
      limit: MAX_VOCALIZATIONS
    };
  }

  /**
   * Clear all vocalizations for a persona
   */
  clearVocalizations(personaId: string): void {
    const persona = this.getPersona(personaId);
    if (persona) {
      persona.procedural_vocalizations = [];
      // Update through store
      const { updatePersona } = useStore.getState();
      updatePersona(persona);
      console.log(`[ProceduralVocal] Cleared all vocalizations for ${personaId}`);
    }
  }

  /**
   * Trigger appropriate vocalization based on context
   */
  async triggerContextualVocalization(
    personaId: string,
    context: 'happy_moment' | 'thinking' | 'surprised' | 'tired' | 'agreement'
  ): Promise<{ audio_data: string } | null> {
    const mapping: Record<string, string> = {
      happy_moment: 'giggle',
      thinking: 'hum',
      surprised: 'gasp',
      tired: 'yawn',
      agreement: 'sigh'
    };

    const type = mapping[context];
    if (!type) return null;

    const result = await this.playVocalization(personaId, type);
    return result ? { audio_data: result.audio_data } : null;
  }

  private getPersona(personaId: string): Persona | null {
    const { personas } = useStore.getState();
    return personas.find(p => p.id === personaId) || null;
  }
}

import { useStore } from "@/store";

export const proceduralVocalizations = new ProceduralVocalizationService();
