/**
 * Persona Learning Service
 * Tracks user interactions and helps personas develop their personality over time
 */

import type { Persona, PersonaLearningData } from "@/types";
import { useStore } from "@/store";

// Maximum stored vocalizations per persona (to prevent storage bloat)
const MAX_PROCEDURAL_VOCALIZATIONS = 10;
const MAX_FAVORITE_TOPICS = 20;
const MAX_INSIDE_JOKES = 10;

interface InteractionEvent {
  type: 'conversation' | 'emote_played' | 'emote_interrupted' | 'user_laughed' | 'user_reaction';
  timestamp: number;
  data: Record<string, any>;
}

export class PersonaLearningService {
  private interactionBuffer: Map<string, InteractionEvent[]> = new Map();
  private bufferFlushInterval: number = 30000; // 30 seconds
  private lastFlush: number = Date.now();

  /**
   * Initialize learning data for a new persona
   */
  initializeLearningData(): PersonaLearningData {
    return {
      interaction_count: 0,
      total_conversation_time: 0,
      favorite_topics: [],
      user_preferences: {},
      animation_preferences: {
        favorites: [],
        avoided: [],
        last_played: {}
      },
      emotional_history: [],
      milestones: {
        first_conversation: new Date().toISOString(),
        conversations_count: 0,
        inside_jokes: []
      }
    };
  }

  /**
   * Record a conversation interaction
   */
  recordConversation(personaId: string, durationMinutes: number, topics: string[]) {
    const event: InteractionEvent = {
      type: 'conversation',
      timestamp: Date.now(),
      data: { durationMinutes, topics }
    };
    this.bufferEvent(personaId, event);
    
    // Update favorite topics immediately
    this.updateFavoriteTopics(personaId, topics);
  }

  /**
   * Record when an emote is played
   */
  recordEmotePlayed(personaId: string, emoteName: string, userReaction?: 'positive' | 'negative' | 'neutral') {
    const event: InteractionEvent = {
      type: 'emote_played',
      timestamp: Date.now(),
      data: { emoteName, userReaction }
    };
    this.bufferEvent(personaId, event);

    // Immediate update to animation preferences
    this.updateAnimationPreference(personaId, emoteName, userReaction);
  }

  /**
   * Record when an emote is interrupted by user
   */
  recordEmoteInterrupted(personaId: string, emoteName: string, durationPlayed: number) {
    const event: InteractionEvent = {
      type: 'emote_interrupted',
      timestamp: Date.now(),
      data: { emoteName, durationPlayed }
    };
    this.bufferEvent(personaId, event);

    // If interrupted quickly, mark as potentially disliked
    if (durationPlayed < 2000) { // Less than 2 seconds
      this.markAnimationAvoided(personaId, emoteName);
    }
  }

  /**
   * Record user laughter/positive reaction
   */
  recordUserLaughed(personaId: string, triggerText: string) {
    const event: InteractionEvent = {
      type: 'user_laughed',
      timestamp: Date.now(),
      data: { trigger: triggerText }
    };
    this.bufferEvent(personaId, event);

    // Check if this could be an inside joke
    this.checkForInsideJoke(personaId, triggerText);
  }

  /**
   * Record emotional state change
   */
  recordEmotionalState(personaId: string, emotion: string, intensity: number, _trigger?: string) {
    const persona = this.getPersona(personaId);
    if (!persona?.learning_data) return;

    persona.learning_data.emotional_history.push({
      timestamp: new Date().toISOString(),
      emotion,
      intensity
    });

    // Keep only last 100 emotional states
    if (persona.learning_data.emotional_history.length > 100) {
      persona.learning_data.emotional_history = persona.learning_data.emotional_history.slice(-100);
    }

    this.updatePersona(persona);
  }

  /**
   * Get animation selection weights based on learned preferences
   */
  getAnimationWeights(personaId: string, availableEmotes: string[]): Map<string, number> {
    const persona = this.getPersona(personaId);
    if (!persona?.learning_data) {
      return new Map(availableEmotes.map(e => [e, 1]));
    }

    const weights = new Map<string, number>();
    const prefs = persona.learning_data.animation_preferences;

    for (const emote of availableEmotes) {
      let weight = 1; // Base weight

      // Boost favorites
      if (prefs.favorites.includes(emote)) {
        weight *= 2;
      }

      // Reduce avoided
      if (prefs.avoided.includes(emote)) {
        weight *= 0.3;
      }

      // Check recency (avoid recently played)
      const lastPlayed = prefs.last_played[emote];
      if (lastPlayed) {
        const minutesSince = (Date.now() - new Date(lastPlayed).getTime()) / 60000;
        if (minutesSince < 5) {
          weight *= 0.5; // Played in last 5 minutes
        } else if (minutesSince > 30) {
          weight *= 1.2; // Not played in 30+ minutes (boost)
        }
      } else {
        // Never played - slight boost to try new animations
        weight *= 1.1;
      }

      weights.set(emote, weight);
    }

    return weights;
  }

  /**
   * Get suggested topics based on favorites
   */
  getSuggestedTopics(personaId: string, count: number = 3): string[] {
    const persona = this.getPersona(personaId);
    if (!persona?.learning_data?.favorite_topics.length) {
      return [];
    }

    // Return random subset of favorite topics
    const topics = persona.learning_data.favorite_topics;
    return topics
      .sort(() => Math.random() - 0.5)
      .slice(0, count);
  }

  /**
   * Get inside jokes for personalization
   */
  getInsideJokes(personaId: string): string[] {
    const persona = this.getPersona(personaId);
    return persona?.learning_data?.milestones.inside_jokes || [];
  }

  /**
   * Add a procedural vocalization (Qwen TTS only)
   * Returns true if added, false if storage limit reached
   */
  addProceduralVocalization(
    personaId: string, 
    type: string, 
    audioData: string
  ): boolean {
    const persona = this.getPersona(personaId);
    if (!persona) return false;

    // Initialize array if needed
    if (!persona.procedural_vocalizations) {
      persona.procedural_vocalizations = [];
    }

    // Check if at limit
    if (persona.procedural_vocalizations.length >= MAX_PROCEDURAL_VOCALIZATIONS) {
      // Remove least used
      persona.procedural_vocalizations.sort((a, b) => a.usage_count - b.usage_count);
      persona.procedural_vocalizations.shift();
    }

    // Add new vocalization
    persona.procedural_vocalizations.push({
      id: `${type}_${Date.now()}`,
      type: type as any,
      audio_data: audioData,
      created_at: new Date().toISOString(),
      usage_count: 0
    });

    this.updatePersona(persona);
    return true;
  }

  /**
   * Get a procedural vocalization
   */
  getProceduralVocalization(
    personaId: string, 
    type: string
  ): { audio_data: string; id: string } | null {
    const persona = this.getPersona(personaId);
    if (!persona?.procedural_vocalizations) return null;

    const vocalization = persona.procedural_vocalizations.find(v => v.type === type);
    if (vocalization) {
      vocalization.usage_count++;
      this.updatePersona(persona);
      return { audio_data: vocalization.audio_data, id: vocalization.id };
    }

    return null;
  }

  /**
   * Get relationship stats for display
   */
  getRelationshipStats(personaId: string) {
    const persona = this.getPersona(personaId);
    if (!persona?.learning_data) return null;

    const data = persona.learning_data;
    const daysSinceFirst = Math.floor(
      (Date.now() - new Date(data.milestones.first_conversation).getTime()) / 86400000
    );

    return {
      daysTogether: daysSinceFirst,
      conversationCount: data.milestones.conversations_count,
      totalChatTime: Math.floor(data.total_conversation_time),
      favoriteTopics: data.favorite_topics.slice(0, 5),
      topEmotes: Object.entries(data.animation_preferences.last_played)
        .sort((a, b) => new Date(b[1]).getTime() - new Date(a[1]).getTime())
        .slice(0, 3)
        .map(([name]) => name),
      insideJokes: data.milestones.inside_jokes.length
    };
  }

  // ============================================
  // PRIVATE METHODS
  // ============================================

  private bufferEvent(personaId: string, event: InteractionEvent) {
    if (!this.interactionBuffer.has(personaId)) {
      this.interactionBuffer.set(personaId, []);
    }
    this.interactionBuffer.get(personaId)!.push(event);

    // Flush if buffer is large or time has passed
    const buffer = this.interactionBuffer.get(personaId)!;
    if (buffer.length > 10 || Date.now() - this.lastFlush > this.bufferFlushInterval) {
      this.flushBuffer(personaId);
    }
  }

  private flushBuffer(personaId: string) {
    const buffer = this.interactionBuffer.get(personaId);
    if (!buffer || buffer.length === 0) return;

    const persona = this.getPersona(personaId);
    if (!persona || !persona.learning_data) {
      this.interactionBuffer.delete(personaId);
      return;
    }

    // Process buffered events
    for (const event of buffer) {
      switch (event.type) {
        case 'conversation':
          persona.learning_data.interaction_count++;
          persona.learning_data.total_conversation_time += event.data.durationMinutes;
          persona.learning_data.milestones.conversations_count++;
          break;
        // Other types handled immediately in record methods
      }
    }

    this.updatePersona(persona);
    this.interactionBuffer.delete(personaId);
    this.lastFlush = Date.now();
  }

  private updateFavoriteTopics(personaId: string, topics: string[]) {
    const persona = this.getPersona(personaId);
    if (!persona?.learning_data) return;

    for (const topic of topics) {
      if (!persona.learning_data.favorite_topics.includes(topic)) {
        persona.learning_data.favorite_topics.push(topic);
      }
    }

    // Limit stored topics
    if (persona.learning_data.favorite_topics.length > MAX_FAVORITE_TOPICS) {
      persona.learning_data.favorite_topics = persona.learning_data.favorite_topics.slice(-MAX_FAVORITE_TOPICS);
    }

    this.updatePersona(persona);
  }

  private updateAnimationPreference(
    personaId: string, 
    emoteName: string, 
    reaction?: 'positive' | 'negative' | 'neutral'
  ) {
    const persona = this.getPersona(personaId);
    if (!persona?.learning_data) return;

    const prefs = persona.learning_data.animation_preferences;
    prefs.last_played[emoteName] = new Date().toISOString();

    if (reaction === 'positive') {
      if (!prefs.favorites.includes(emoteName)) {
        prefs.favorites.push(emoteName);
      }
      // Remove from avoided if present
      prefs.avoided = prefs.avoided.filter(e => e !== emoteName);
    }

    this.updatePersona(persona);
  }

  private markAnimationAvoided(personaId: string, emoteName: string) {
    const persona = this.getPersona(personaId);
    if (!persona?.learning_data) return;

    const prefs = persona.learning_data.animation_preferences;
    if (!prefs.avoided.includes(emoteName)) {
      prefs.avoided.push(emoteName);
    }

    this.updatePersona(persona);
  }

  private checkForInsideJoke(personaId: string, trigger: string) {
    const persona = this.getPersona(personaId);
    if (!persona?.learning_data) return;

    // Simple heuristic: if same trigger causes laughter multiple times, it's an inside joke
    const recentLaughs = persona.learning_data.emotional_history.filter(
      e => e.emotion === 'laugh' && 
      (Date.now() - new Date(e.timestamp).getTime()) < 604800000 // Within a week
    ).length;

    if (recentLaughs >= 3 && !persona.learning_data.milestones.inside_jokes.includes(trigger)) {
      persona.learning_data.milestones.inside_jokes.push(trigger);
      
      // Limit stored jokes
      if (persona.learning_data.milestones.inside_jokes.length > MAX_INSIDE_JOKES) {
        persona.learning_data.milestones.inside_jokes.shift();
      }

      this.updatePersona(persona);
    }
  }

  private getPersona(personaId: string): Persona | null {
    const store = useStore.getState();
    return store.personas.find(p => p.id === personaId) || null;
  }

  private updatePersona(persona: Persona) {
    const store = useStore.getState();
    store.updatePersona({
      ...persona,
      updated_at: new Date().toISOString()
    });
  }
}

export const personaLearning = new PersonaLearningService();
