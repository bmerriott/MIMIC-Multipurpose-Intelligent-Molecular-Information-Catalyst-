/**
 * Personality Derivation Service
 * Derives avatar traits and voice parameters from personality prompts
 */

import type { AvatarPersonalityTraits, VoiceTuningParams } from "@/types";
import { ollamaService } from "./ollama";

// Keyword-based trait extraction (fallback when LLM unavailable)
const TRAIT_KEYWORDS: Record<keyof AvatarPersonalityTraits, { high: string[]; low: string[] }> = {
  energy_level: {
    high: ["energetic", "enthusiastic", "lively", "active", "vibrant", "dynamic", "spirited", "peppy", "bubbly", "excited"],
    low: ["calm", "relaxed", "laid-back", "chill", "serene", "tranquil", "peaceful", "slow", "gentle", "mellow"]
  },
  playfulness: {
    high: ["playful", "fun-loving", "mischievous", "witty", "humorous", "joking", "teasing", "lighthearted", "silly", "cheeky"],
    low: ["serious", "formal", "stern", "strict", "professional", "businesslike", "reserved", "dignified", "proper", "grave"]
  },
  expressiveness: {
    high: ["expressive", "animated", "dramatic", "emotional", "passionate", "intense", "vivid", "flamboyant", "theatrical", "demonstrative"],
    low: ["stoic", "reserved", "restrained", "subdued", "muted", "understated", "composed", "detached", "unemotional", "stoical"]
  },
  curiosity: {
    high: ["curious", "inquisitive", "questioning", "exploring", "investigative", "nosy", "interested", "eager", "enthusiastic", "probing"],
    low: ["indifferent", "uninterested", "apathetic", "disinterested", "aloof", "distant", "detached", "uncaring", "passive", "uninvolved"]
  },
  empathy: {
    high: ["empathetic", "compassionate", "caring", "understanding", "sympathetic", "warm", "kind", "nurturing", "supportive", "sensitive"],
    low: ["cold", "distant", "uncaring", "unsympathetic", "harsh", "cruel", "indifferent", "callous", "insensitive", "ruthless"]
  },
  formality: {
    high: ["formal", "proper", "professional", "courteous", "polite", "dignified", "elegant", "sophisticated", "refined", "respectful"],
    low: ["casual", "informal", "relaxed", "laid-back", "easygoing", "familiar", "intimate", "friendly", "approachable", "unpretentious"]
  }
};

// Voice parameter keywords
const VOICE_KEYWORDS = {
  pitch: {
    high: ["high-pitched", "squeaky", "youthful", "childlike", "feminine", "bright", "cheerful", "energetic"],
    low: ["deep", "low-pitched", "mature", "masculine", "authoritative", "serious", "calm", "measured"]
  },
  speed: {
    fast: ["fast", "quick", "rapid", "energetic", "nervous", "excited", "enthusiastic"],
    slow: ["slow", "deliberate", "measured", "calm", "thoughtful", "relaxed", "contemplative"]
  },
  warmth: {
    high: ["warm", "friendly", "kind", "gentle", "caring", "soft", "smooth", "welcoming"],
    low: ["cold", "harsh", "stern", "clinical", "distant", "detached", "professional"]
  },
  expressiveness: {
    high: ["expressive", "animated", "dramatic", "varied", "emotional", "passionate"],
    low: ["monotone", "flat", "even", "steady", "controlled", "restrained"]
  },
  breathiness: {
    high: ["breathy", "soft", "whispery", "airy", "gentle", "intimate"],
    low: ["clear", "crisp", "sharp", "strong", "powerful", "authoritative"]
  }
};

export class PersonalityDerivationService {
  /**
   * Derive personality traits from a personality prompt
   * Uses keyword analysis as fallback if LLM is unavailable
   */
  async deriveTraits(personalityPrompt: string): Promise<AvatarPersonalityTraits> {
    // Try LLM-based analysis first
    try {
      const llmTraits = await this.deriveTraitsWithLLM(personalityPrompt);
      if (llmTraits) return llmTraits;
    } catch (e) {
      console.log("[PersonalityDerivation] LLM analysis failed, using keyword fallback");
    }
    
    // Fallback to keyword analysis
    return this.deriveTraitsWithKeywords(personalityPrompt);
  }
  
  /**
   * Use lightweight LLM to analyze personality
   */
  private async deriveTraitsWithLLM(prompt: string): Promise<AvatarPersonalityTraits | null> {
    const analysisPrompt = `Analyze this character description and return ONLY a JSON object with these exact fields (values 0.0-1.0):
- energy_level: How energetic/active (0=very calm, 1=very energetic)
- playfulness: How playful/humorous (0=very serious, 1=very playful)  
- expressiveness: How emotionally expressive (0=stoic, 1=very expressive)
- curiosity: How inquisitive/proactive (0=indifferent, 1=very curious)
- empathy: How caring/understanding (0=cold, 1=very empathetic)
- formality: How formal in speech (0=very casual, 1=very formal)

Character: "${prompt}"

Return ONLY the JSON object, no other text:`;

    const response = await ollamaService.generate("llama3.2:1b", analysisPrompt, undefined, {
      temperature: 0.1
    });
    
    try {
      // Extract JSON from response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        
        // Validate and clamp values
        const traits: AvatarPersonalityTraits = {
          energy_level: this.clamp(parsed.energy_level ?? 0.5),
          playfulness: this.clamp(parsed.playfulness ?? 0.5),
          expressiveness: this.clamp(parsed.expressiveness ?? 0.5),
          curiosity: this.clamp(parsed.curiosity ?? 0.5),
          empathy: this.clamp(parsed.empathy ?? 0.5),
          formality: this.clamp(parsed.formality ?? 0.5)
        };
        
        console.log("[PersonalityDerivation] LLM-derived traits:", traits);
        return traits;
      }
    } catch (e) {
      console.error("[PersonalityDerivation] Failed to parse LLM response:", e);
    }
    
    return null;
  }
  
  /**
   * Keyword-based trait extraction (fallback)
   */
  private deriveTraitsWithKeywords(prompt: string): AvatarPersonalityTraits {
    const lower = prompt.toLowerCase();
    const traits: Partial<AvatarPersonalityTraits> = {};
    
    for (const [traitName, keywords] of Object.entries(TRAIT_KEYWORDS)) {
      let score = 0.5; // Neutral default
      
      const highMatches = keywords.high.filter(k => lower.includes(k)).length;
      const lowMatches = keywords.low.filter(k => lower.includes(k)).length;
      
      if (highMatches > 0 || lowMatches > 0) {
        // Adjust score based on keyword matches
        score = 0.5 + (highMatches * 0.15) - (lowMatches * 0.15);
        score = this.clamp(score);
      }
      
      traits[traitName as keyof AvatarPersonalityTraits] = score;
    }
    
    console.log("[PersonalityDerivation] Keyword-derived traits:", traits);
    return traits as AvatarPersonalityTraits;
  }
  
  /**
   * Derive voice parameters from personality prompt and traits
   */
  deriveVoiceParams(
    personalityPrompt: string, 
    traits: AvatarPersonalityTraits
  ): Partial<VoiceTuningParams> {
    const lower = personalityPrompt.toLowerCase();
    const params: Partial<VoiceTuningParams> = {};
    
    // Pitch - based on energy and keywords
    let pitchShift = 0;
    if (VOICE_KEYWORDS.pitch.high.some(k => lower.includes(k))) pitchShift += 0.3;
    if (VOICE_KEYWORDS.pitch.low.some(k => lower.includes(k))) pitchShift -= 0.3;
    // Energy affects pitch
    pitchShift += (traits.energy_level - 0.5) * 0.2;
    params.pitchShift = this.clamp(pitchShift, -0.5, 0.5);
    
    // Speed - based on energy
    let speed = 1.0;
    if (VOICE_KEYWORDS.speed.fast.some(k => lower.includes(k))) speed += 0.2;
    if (VOICE_KEYWORDS.speed.slow.some(k => lower.includes(k))) speed -= 0.2;
    speed += (traits.energy_level - 0.5) * 0.3;
    params.speed = this.clamp(speed, 0.7, 1.3);
    
    // Warmth - based on empathy
    let warmth = 0.5;
    if (VOICE_KEYWORDS.warmth.high.some(k => lower.includes(k))) warmth += 0.2;
    if (VOICE_KEYWORDS.warmth.low.some(k => lower.includes(k))) warmth -= 0.2;
    warmth += (traits.empathy - 0.5) * 0.3;
    params.warmth = this.clamp(warmth);
    
    // Expressiveness - directly from trait
    params.expressiveness = traits.expressiveness;
    
    // Stability - inverse of playfulness (playful = less stable/more varied)
    params.stability = this.clamp(1 - (traits.playfulness - 0.5) * 0.4);
    
    // Clarity - based on formality
    params.clarity = this.clamp(0.5 + (traits.formality - 0.5) * 0.3);
    
    // Breathiness - based on warmth and keywords
    let breathiness = 0.3;
    if (VOICE_KEYWORDS.breathiness.high.some(k => lower.includes(k))) breathiness += 0.2;
    if (VOICE_KEYWORDS.breathiness.low.some(k => lower.includes(k))) breathiness -= 0.1;
    breathiness += (traits.empathy - 0.5) * 0.2;
    params.breathiness = this.clamp(breathiness);
    
    // Resonance - based on energy and formality
    params.resonance = this.clamp(0.5 + (traits.energy_level - 0.5) * 0.2 + (traits.formality - 0.5) * 0.2);
    
    console.log("[PersonalityDerivation] Derived voice params:", params);
    return params;
  }
  
  /**
   * Derive emotional state from conversation context
   */
  deriveEmotionalState(
    recentMessages: { role: string; content: string }[],
    currentTraits: AvatarPersonalityTraits
  ): { emotion: string; intensity: number } {
    // Simple keyword-based emotion detection
    const emotionKeywords: Record<string, string[]> = {
      happy: ["happy", "joy", "excited", "great", "wonderful", "fantastic", "love", "like", "good", "best"],
      sad: ["sad", "sorry", "unfortunate", "regret", "miss", "loss", "cry", "upset"],
      angry: ["angry", "mad", "frustrated", "annoying", "hate", "bad", "terrible", "awful"],
      curious: ["curious", "wonder", "question", "how", "why", "what if", "interesting"],
      empathetic: ["understand", "feel", "empathy", "care", "support", "here for you", "listening"]
    };
    
    // Analyze last few messages
    const recentText = recentMessages.slice(-3).map(m => m.content.toLowerCase()).join(" ");
    
    const scores: Record<string, number> = {};
    for (const [emotion, keywords] of Object.entries(emotionKeywords)) {
      scores[emotion] = keywords.filter(k => recentText.includes(k)).length;
    }
    
    // Find dominant emotion
    const dominant = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
    
    if (dominant && dominant[1] > 0) {
      // Calculate intensity based on frequency and expressiveness trait
      const baseIntensity = Math.min(1, dominant[1] * 0.3);
      const traitMultiplier = 0.7 + currentTraits.expressiveness * 0.6;
      return {
        emotion: dominant[0],
        intensity: this.clamp(baseIntensity * traitMultiplier)
      };
    }
    
    // Default to neutral with low intensity
    return { emotion: "neutral", intensity: 0.3 };
  }
  
  private clamp(value: number, min = 0, max = 1): number {
    return Math.max(min, Math.min(max, value));
  }
}

export const personalityDerivation = new PersonalityDerivationService();
