/**
 * usePersonalityLearning Hook
 * 
 * Integrates the personality learning system with the chat flow.
 * Analyzes conversations periodically to extract insights and evolve the persona.
 * 
 * Usage:
 * ```tsx
 * const { triggerAnalysis, isAnalyzing } = usePersonalityLearning(persona);
 * 
 * // Call after significant conversations
 * useEffect(() => {
 *   if (messages.length > 0 && messages.length % 10 === 0) {
 *     triggerAnalysis(messages.slice(-10));
 *   }
 * }, [messages]);
 * ```
 */

import { useState, useCallback, useRef } from "react";
import type { Persona, MemoryEntry, ChatMessage } from "@/types";
import { personalityLearning } from "@/services/personalityLearning";
import { useStore } from "@/store";
import { toast } from "sonner";

interface UsePersonalityLearningOptions {
  model?: string;
  minMessages?: number;
}

export function usePersonalityLearning(
  persona: Persona | null,
  options: UsePersonalityLearningOptions = {}
) {
  const { model = "qwen3:0.6b", minMessages = 5 } = options;
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [lastAnalysis, setLastAnalysis] = useState<Date | null>(null);
  const analysisInProgress = useRef(false);
  const { updatePersona, settings } = useStore();

  /**
   * Check if a message is a ChatMessage (has role property)
   */
  const isChatMessage = (m: MemoryEntry | ChatMessage): m is ChatMessage => {
    return "role" in m && typeof m.role === "string";
  };

  /**
   * Trigger personality analysis on a batch of messages
   */
  const triggerAnalysis = useCallback(
    async (messages: (MemoryEntry | ChatMessage)[]) => {
      if (!persona || analysisInProgress.current) return;
      if (messages.length < minMessages) return;
      if (!settings.enable_memory) return;

      analysisInProgress.current = true;
      setIsAnalyzing(true);

      try {
        // Convert ChatMessages to MemoryEntry format if needed
        const entries: MemoryEntry[] = messages.map((m) => {
          const role = isChatMessage(m) ? m.role : "unknown";
          return {
            id: m.id || Date.now().toString(),
            content: `${role}: ${m.content}`,
            timestamp: m.timestamp || new Date().toISOString(),
            importance: 0.5,
          };
        });

        // Run the analysis
        const result = await personalityLearning.analyzeConversation(
          entries,
          model
        );

        if (result.newInsights.length > 0) {
          // Update the persona's learning data
          const updatedLearningData = personalityLearning.updateLearningData(
            persona.learning_data || personalityLearning.initializeLearningData(),
            result.newInsights
          );

          // Track the interaction
          const trackedData = personalityLearning.trackInteraction(
            updatedLearningData,
            1, // Assume 1 minute for now
            [] // Topics could be extracted separately
          );

          // Save back to persona
          updatePersona({
            ...persona,
            learning_data: trackedData,
          });

          // Notify user of new insights (optional - could be subtle)
          if (result.newInsights.some((i) => i.confidence > 0.8)) {
            toast.info(`${persona.name} is learning about you`, {
              description: `Discovered ${result.newInsights.length} new insight${result.newInsights.length > 1 ? "s" : ""} from your conversation.`,
              duration: 3000,
            });
          }

          console.log("[PersonalityLearning] Analysis complete:", {
            newInsights: result.newInsights.length,
            summary: result.summary,
          });
        }

        setLastAnalysis(new Date());
      } catch (error) {
        console.error("[PersonalityLearning] Analysis failed:", error);
      } finally {
        setIsAnalyzing(false);
        analysisInProgress.current = false;
      }
    },
    [persona, model, minMessages, settings.enable_memory, updatePersona]
  );

  /**
   * Get the personality augmentation for system prompts
   */
  const getPersonalityAugmentation = useCallback((): string => {
    if (!persona?.learning_data) return "";
    return personalityLearning.buildPersonalityAugmentation(persona.learning_data);
  }, [persona]);

  /**
   * Get relationship depth score
   */
  const getRelationshipDepth = useCallback((): number => {
    if (!persona?.learning_data) return 0;
    return personalityLearning.getRelationshipDepth(persona.learning_data);
  }, [persona]);

  return {
    triggerAnalysis,
    isAnalyzing,
    lastAnalysis,
    getPersonalityAugmentation,
    getRelationshipDepth,
  };
}

export default usePersonalityLearning;
