/**
 * Agent System Prompt Service
 * 
 * PRIORITY ORDER (highest to lowest):
 * 1. Personality Prompt (dominant - defines who the persona is)
 * 2. Learned Personality (evolved traits from conversations)
 * 3. Context/Timestamps (conversation history with timing)
 * 4. Capabilities (what tools are available)
 * 5. Minimal behavior constraints (NO EMOJIS only)
 * 
 * The personality prompt is ALWAYS first and defines the response style,
 * tone, behavior, and character of the AI.
 * 
 * LEARNED PERSONALITY:
 * As conversations progress, the AI develops learned traits, skills, and
 * relationship understanding. These are integrated to make each persona
 * more unique and personalized over time.
 */

import type { Persona } from "@/types";
import { personalityLearning } from "@/services/personalityLearning";

export interface AgentContext {
  hasMemoryAccess: boolean;
  hasWebSearch: boolean;
  hasVision: boolean;
  canWriteFiles: boolean;
  toolPermissionRequired: boolean;
}

/**
 * Build the complete agent system prompt
 * Personality is DOMINANT - everything else supports it
 */
export function buildAgentSystemPrompt(
  persona: Persona,
  context: AgentContext
): string {
  // 1. PERSONALITY PROMPT (DOMINANT - First and most important)
  const personalityPrompt = persona.personality_prompt?.trim() 
    || `You are ${persona.name}, a helpful AI assistant.`;

  // 2. LEARNED PERSONALITY (evolved from conversations - verified insights only)
  const learnedPersonality = personalityLearning.buildPersonalityAugmentation(
    persona.learning_data || personalityLearning.initializeLearningData()
  );
  const learnedSection = learnedPersonality 
    ? `\n\n${learnedPersonality}` 
    : "";

  // 3. Self-awareness context (minimal, supports personality)
  const selfAwareness = `

You are operating in MIMIC Desktop Assistant. You have access to tools but should use them naturally as befits your character.`;

  // 4. Capabilities (what the persona CAN do, not restrictions)
  const capabilities: string[] = [];
  if (context.hasMemoryAccess) {
    capabilities.push("access your memory files and conversation history");
  }
  if (context.hasWebSearch) {
    capabilities.push("search the web for current information");
  }
  if (context.hasVision) {
    capabilities.push("analyze images");
  }
  
  const capabilityText = capabilities.length > 0 
    ? `

CAPABILITIES: You can ${capabilities.join(", ")}. Use these as your character would.`
    : "";

  // 5. Tool guidance for memory recall
  const toolGuidance = context.hasMemoryAccess
    ? `

MEMORY: When asked about past conversations, timing, or "what was first/second/third", use get_conversation_history or search_conversation_history tools. History includes timestamps.`
    : "";

  // 6. ABSOLUTE MINIMUM constraints (only non-negotiable rules)
  // These do NOT override personality - they are technical requirements
  const constraints = `

TECHNICAL CONSTRAINTS:
- NO EMOJIS: Do not use emojis as TTS reads them aloud. Express emotion through words as your character would.`;

  // ASSEMBLE: Personality first, learned personality second, everything else supports it
  return `${personalityPrompt}${learnedSection}${selfAwareness}${capabilityText}${toolGuidance}${constraints}`;
}

/**
 * Build a minimal system prompt when agent context isn't needed
 * Still puts personality first
 */
export function buildMinimalSystemPrompt(persona: Persona): string {
  const personalityPrompt = persona.personality_prompt?.trim() 
    || `You are ${persona.name}, a helpful AI assistant.`;

  // Add learned personality even in minimal mode
  const learnedPersonality = personalityLearning.buildPersonalityAugmentation(
    persona.learning_data || personalityLearning.initializeLearningData()
  );
  const learnedSection = learnedPersonality 
    ? `\n\n${learnedPersonality}` 
    : "";

  return `${personalityPrompt}${learnedSection}\n\nYou are in MIMIC Desktop Assistant. NO EMOJIS (TTS reads them).`;
}

/**
 * Router guidance prompt - Situational guidance only
 * Does NOT override personality
 */
export function buildRouterGuidance(
  approach: string,
  tone: string,
  confidence: number
): string | null {
  if (confidence < 0.6) {
    return null;
  }

  const parts: string[] = [];

  if (approach && approach.length > 5) {
    parts.push(`Situation: ${approach}`);
  }

  if (tone && tone !== "neutral" && tone.length > 2) {
    parts.push(`Suggested tone: ${tone}`);
  }

  if (parts.length === 0) {
    return null;
  }

  // Router guidance is SUGGESTIVE, not prescriptive
  return `

[Context: ${parts.join(" | ")}]`;
}

/**
 * Tool confirmation prompt
 */
export function formatToolConfirmation(
  toolName: string,
  args: Record<string, any>
): { title: string; description: string; preview: string } {
  switch (toolName) {
    case "write_memory":
      return {
        title: "Create/Modify File?",
        description: `${args.personaName || "The AI"} wants to write to a memory file.`,
        preview: `File: ${args.filename}\n\nContent:\n${args.content?.substring(0, 500)}${args.content?.length > 500 ? "..." : ""}`,
      };

    case "delete_memory":
      return {
        title: "Delete File?",
        description: `${args.personaName || "The AI"} wants to delete a memory file.`,
        preview: `File: ${args.filename}`,
      };

    case "clear_conversation_history":
      return {
        title: "Clear History?",
        description: `${args.personaName || "The AI"} wants to clear conversation history.`,
        preview: `This will delete all conversation messages for this persona.`,
      };

    default:
      return {
        title: "Execute Operation?",
        description: `${args.personaName || "The AI"} wants to perform an operation.`,
        preview: `Tool: ${toolName}\nArgs: ${JSON.stringify(args, null, 2)}`,
      };
  }
}

/**
 * Check if a tool requires user confirmation
 */
export function requiresConfirmation(toolName: string): boolean {
  const writeOperations = [
    "write_memory",
    "delete_memory",
    "clear_conversation_history",
    "delete_conversation_memory",
  ];

  return writeOperations.includes(toolName);
}
