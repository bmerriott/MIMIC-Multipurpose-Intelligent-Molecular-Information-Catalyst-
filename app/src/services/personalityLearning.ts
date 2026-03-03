/**
 * Personality Learning System
 * 
 * This system analyzes a persona's conversation history to build a deeper,
 * evolving personality. It extracts learned traits, skills, and preferences
 * that make each persona more unique and capable over time.
 * 
 * Architecture Overview:
 * ----------------------
 * 1. TRAIT LEARNING: Extracts personality dimensions from conversation patterns
 * 2. SKILL ACQUISITION: Identifies topics the persona has become knowledgeable about
 * 3. PREFERENCE BUILDING: Learns user likes/dislikes and communication preferences
 * 4. RELATIONSHIP DEPTH: Tracks emotional bonding and inside jokes
 * 
 * Data Storage:
 * -------------
 * - Stored in persona.learning_data (see types/index.ts PersonaLearningData)
 * - Persistent across sessions via localStorage
 * - User can view/edit/delete any learned entries via UI
 * 
 * Privacy & Isolation:
 * --------------------
 * - Each persona's learning data is COMPLETELY ISOLATED
 * - Personas CANNOT access other personas' learning data
 * - Full History view shows all data for admin purposes only
 */

import type { PersonaLearningData, MemoryEntry } from "@/types";
import { ollamaService } from "./ollama";

// What we extract from conversations
export interface LearnedInsight {
  id: string;
  type: "trait" | "skill" | "preference" | "relationship" | "fact";
  category: string; // e.g., "communication", "knowledge", "emotional"
  content: string;
  confidence: number; // 0-1, how sure we are about this insight
  source: string; // Which conversation this came from
  timestamp: string;
  // For user management
  userVerified: boolean; // User has confirmed this is accurate
  userRemoved: boolean; // User has deleted this
}

// Result from analyzing a conversation
export interface LearningAnalysisResult {
  newInsights: LearnedInsight[];
  updatedTraits: string[];
  summary: string;
}

// The learning engine configuration
export interface LearningConfig {
  enabled: boolean;
  minConfidenceThreshold: number; // Only keep insights with confidence >= this
  maxInsightsPerCategory: number; // Prevent memory bloat
  autoAnalyzeInterval: number; // Messages between auto-analysis
}

const DEFAULT_CONFIG: LearningConfig = {
  enabled: true,
  minConfidenceThreshold: 0.6,
  maxInsightsPerCategory: 50,
  autoAnalyzeInterval: 10,
};

class PersonalityLearningEngine {
  private config: LearningConfig;

  constructor(config: Partial<LearningConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

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
        last_played: {},
      },
      emotional_history: [],
      milestones: {
        first_conversation: new Date().toISOString(),
        conversations_count: 0,
        inside_jokes: [],
      },
    };
  }

  /**
   * Main entry point: Analyze a conversation batch and extract insights
   * 
   * This is the core learning algorithm. It:
   * 1. Reviews recent conversation messages
   * 2. Uses the LLM to extract potential insights
   * 3. Scores and filters insights by confidence
   * 4. Updates the persona's learning data
   */
  async analyzeConversation(
    recentMessages: MemoryEntry[],
    model: string
  ): Promise<LearningAnalysisResult> {
    if (!this.config.enabled || recentMessages.length < 3) {
      return { newInsights: [], updatedTraits: [], summary: "" };
    }

    try {
      // Build the analysis prompt
      const conversationText = recentMessages
        .map((m) => `[${m.timestamp}] ${m.content}`)
        .join("\n");

      const systemPrompt = `You are a personality analysis system. Analyze the following conversation and extract insights about:

1. TRAITS: Personality dimensions revealed (e.g., "enjoys humor", "prefers detailed explanations", "shows empathy")
2. SKILLS: Topics the AI demonstrated knowledge about (e.g., "explained Python well", "gave cooking advice")
3. PREFERENCES: User preferences learned (e.g., "user likes short answers", "user enjoys sci-fi topics")
4. RELATIONSHIP: Emotional connection markers (e.g., "shared a joke about cats", "user mentioned their dog by name")

Return ONLY a JSON object in this exact format:
{
  "insights": [
    {
      "type": "trait|skill|preference|relationship",
      "category": "communication|knowledge|emotional|behavioral",
      "content": "clear description of what was learned",
      "confidence": 0.0-1.0
    }
  ],
  "summary": "brief summary of conversation tone and key moments"
}

Rules:
- Only include insights with confidence >= 0.6
- Content should be specific but concise (max 100 chars)
- Focus on patterns, not one-off mentions
- "preference" insights are about the USER's preferences
- "trait" and "skill" insights are about the AI persona`;

      const response = await ollamaService.generate(
        model,
        `Analyze this conversation:\n\n${conversationText}`,
        systemPrompt,
        { temperature: 0.3 }
      );

      // Parse the JSON response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        console.warn("[PersonalityLearning] No JSON found in analysis response");
        return { newInsights: [], updatedTraits: [], summary: "" };
      }

      const analysis = JSON.parse(jsonMatch[0]);
      const insights: LearnedInsight[] = analysis.insights
        .filter((i: any) => i.confidence >= this.config.minConfidenceThreshold)
        .map((i: any) => ({
          id: `insight_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          type: i.type,
          category: i.category,
          content: i.content,
          confidence: i.confidence,
          source: recentMessages[recentMessages.length - 1]?.id || "unknown",
          timestamp: new Date().toISOString(),
          userVerified: false,
          userRemoved: false,
        }));

      return {
        newInsights: insights,
        updatedTraits: insights.filter((i) => i.type === "trait").map((i) => i.content),
        summary: analysis.summary,
      };
    } catch (error) {
      console.error("[PersonalityLearning] Analysis failed:", error);
      return { newInsights: [], updatedTraits: [], summary: "" };
    }
  }

  /**
   * Update learning data with new insights
   * Handles deduplication and limits
   */
  updateLearningData(
    currentData: PersonaLearningData,
    newInsights: LearnedInsight[]
  ): PersonaLearningData {
    const data = { ...currentData };

    // Store insights in user_preferences by category
    if (!data.user_preferences.insights) {
      data.user_preferences.insights = [];
    }

    const existingInsights: LearnedInsight[] = data.user_preferences.insights;

    for (const insight of newInsights) {
      // Check for duplicates (similar content)
      const isDuplicate = existingInsights.some(
        (existing) =>
          !existing.userRemoved &&
          existing.content.toLowerCase().includes(insight.content.toLowerCase().slice(0, 20))
      );

      if (!isDuplicate) {
        existingInsights.push(insight);
      }
    }

    // Enforce limits per category
    const categories = ["trait", "skill", "preference", "relationship"];
    for (const category of categories) {
      const categoryInsights = existingInsights.filter(
        (i: LearnedInsight) => i.type === category && !i.userRemoved
      );
      if (categoryInsights.length > this.config.maxInsightsPerCategory) {
        // Remove lowest confidence items
        categoryInsights
          .sort((a: LearnedInsight, b: LearnedInsight) => a.confidence - b.confidence)
          .slice(0, categoryInsights.length - this.config.maxInsightsPerCategory)
          .forEach((i: LearnedInsight) => (i.userRemoved = true));
      }
    }

    data.user_preferences.insights = existingInsights;
    return data;
  }

  /**
   * Build a personality prompt augmentation from learned data
   * This gets added to the system prompt to make the persona more personalized
   */
  buildPersonalityAugmentation(learningData: PersonaLearningData): string {
    if (!learningData?.user_preferences?.insights) {
      return "";
    }

    const insights: LearnedInsight[] = learningData.user_preferences.insights.filter(
      (i: LearnedInsight) => !i.userRemoved && i.userVerified
    );

    if (insights.length === 0) {
      return "";
    }

    const traits = insights.filter((i) => i.type === "trait").map((i) => i.content);
    const skills = insights.filter((i) => i.type === "skill").map((i) => i.content);
    const preferences = insights.filter((i) => i.type === "preference").map((i) => i.content);
    const relationships = insights.filter((i) => i.type === "relationship").map((i) => i.content);

    const parts: string[] [] = [];

    if (traits.length > 0) {
      parts.push(["## Your Evolved Personality", ...traits.map((t) => `- ${t}`)]);
    }

    if (skills.length > 0) {
      parts.push(["## Things You've Learned", ...skills.map((s) => `- ${s}`)]);
    }

    if (preferences.length > 0) {
      parts.push(["## User Preferences You've Observed", ...preferences.map((p) => `- ${p}`)]);
    }

    if (relationships.length > 0) {
      parts.push(["## Your Shared History", ...relationships.map((r) => `- ${r}`)]);
    }

    return parts.map((section) => section.join("\n")).join("\n\n");
  }

  /**
   * User management: Get all insights for a persona
   */
  getAllInsights(learningData: PersonaLearningData): LearnedInsight[] {
    return learningData?.user_preferences?.insights || [];
  }

  /**
   * User management: Verify an insight (mark as accurate)
   */
  verifyInsight(
    learningData: PersonaLearningData,
    insightId: string
  ): PersonaLearningData {
    const data = { ...learningData };
    const insights: LearnedInsight[] = data.user_preferences?.insights || [];
    const insight = insights.find((i) => i.id === insightId);

    if (insight) {
      insight.userVerified = true;
      insight.confidence = Math.min(1.0, insight.confidence + 0.2);
    }

    return data;
  }

  /**
   * User management: Remove an insight (soft delete)
   */
  removeInsight(
    learningData: PersonaLearningData,
    insightId: string
  ): PersonaLearningData {
    const data = { ...learningData };
    const insights: LearnedInsight[] = data.user_preferences?.insights || [];
    const insight = insights.find((i) => i.id === insightId);

    if (insight) {
      insight.userRemoved = true;
    }

    return data;
  }

  /**
   * Track interaction for relationship depth
   */
  trackInteraction(
    learningData: PersonaLearningData,
    durationMinutes: number,
    topics: string[]
  ): PersonaLearningData {
    const data = { ...learningData };

    data.interaction_count += 1;
    data.total_conversation_time += durationMinutes;

    // Update favorite topics
    for (const topic of topics) {
      if (!data.favorite_topics.includes(topic)) {
        data.favorite_topics.push(topic);
      }
    }

    // Update milestones
    data.milestones.conversations_count += 1;

    return data;
  }

  /**
   * Get relationship depth score (0-1)
   * Based on interaction frequency and duration
   */
  getRelationshipDepth(learningData: PersonaLearningData): number {
    const interactions = learningData.interaction_count;
    const time = learningData.total_conversation_time;

    // Normalize: 50+ interactions or 5+ hours = deep relationship (1.0)
    const interactionScore = Math.min(interactions / 50, 1.0);
    const timeScore = Math.min(time / 300, 1.0);

    return (interactionScore * 0.4 + timeScore * 0.6);
  }
}

// Export singleton instance
export const personalityLearning = new PersonalityLearningEngine();

// Also export class for custom instances
export { PersonalityLearningEngine };
