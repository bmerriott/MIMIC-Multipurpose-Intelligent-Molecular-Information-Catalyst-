/**
 * Agent with Tools Hook
 * Manages the agent loop for models that support tool calling (arcee-agent, llama3.2, etc.)
 * Provides mediated access to memory tools with user confirmation for writes
 */

import { useState, useCallback } from "react";
import { memoryToolsService, type ToolCall, type ToolResult } from "@/services/memoryTools";
import { ollamaService } from "@/services/ollama";
import type { Persona } from "@/types";

export interface AgentStep {
  type: "thinking" | "tool_call" | "tool_result" | "response";
  content: string;
  toolCall?: ToolCall;
  toolResult?: ToolResult;
}

export interface PendingConfirmation {
  type: "write" | "delete";
  filename: string;
  content?: string;
  toolCall: ToolCall;
}

export interface UseAgentWithToolsOptions {
  maxSteps?: number;
  model?: string;
}

export function useAgentWithTools(options: UseAgentWithToolsOptions = {}) {
  const { maxSteps = 5, model = "arcee-agent" } = options;

  const [isProcessing, setIsProcessing] = useState(false);
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);

  /**
   * Check if a model response contains a tool call
   */
  const extractToolCall = useCallback((response: string): ToolCall | null => {
    // Look for tool call patterns in the response
    // Format: <tool>name</tool> or ```tool:name``` or JSON format

    // Try JSON format first (Ollama tool calling format)
    try {
      const jsonMatch = response.match(/\{[\s\S]*"name"[\s\S]*"arguments"[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        if (parsed.name && parsed.arguments) {
          return {
            name: parsed.name,
            arguments: parsed.arguments,
          };
        }
      }
    } catch {
      // Not JSON, try other formats
    }

    // Try XML-like format: <tool name="read_memory"><filename>test.txt</filename></tool>
    const xmlMatch = response.match(/<tool\s+name="([^"]+)"[^>]*>([\s\S]*?)<\/tool>/);
    if (xmlMatch) {
      const name = xmlMatch[1];
      const inner = xmlMatch[2];
      // Parse inner XML as simple key-value
      const args: Record<string, any> = {};
      const argMatches = inner.matchAll(/<(\w+)>([^<]*)<\/\w+>/g);
      for (const match of argMatches) {
        args[match[1]] = match[2];
      }
      return { name, arguments: args };
    }

    // Try markdown code block format: ```tool:read_memory\n{"filename": "test.txt"}\n```
    const mdMatch = response.match(/```tool:(\w+)\n([\s\S]*?)\n```/);
    if (mdMatch) {
      try {
        const name = mdMatch[1];
        const args = JSON.parse(mdMatch[2]);
        return { name, arguments: args };
      } catch {
        // Invalid JSON
      }
    }

    return null;
  }, []);

  /**
   * Build system prompt with tool instructions
   */
  const buildAgentSystemPrompt = useCallback(
    (persona: Persona, context: string = ""): string => {
      const toolDescriptions = `
MEMORY ACCESS PROTOCOL:
You have read-only access to a personal memory folder with files saved by the user.

CRITICAL RULES:
1. When asked about memory files, notes, or documents, you MUST use read_memory tool
2. You must NEVER make up or hallucinate file contents - always read actual files
3. After receiving TOOL_RESULT with file contents, report EXACTLY what was in the file
4. Tools are strictly read-only - never attempt write/delete/execute operations

Available tools:
- list_memories: List all files in your memory folder
- read_memory: Read a file (args: {filename}) - USE THIS when asked about file contents
- search_memories: Search file contents (args: {query})
- get_memory_info: Get file metadata (args: {filename})

To use a tool, output ONLY this JSON format:
{"name": "read_memory", "arguments": {"filename": "example.txt"}}

After receiving tool results, use the actual data to respond truthfully.
`;

      return `You are ${persona.name}, an AI assistant with access to a personal memory folder.
${persona.personality_prompt ? `Personality: ${persona.personality_prompt}` : ""}

${toolDescriptions}

${context}

IMPORTANT: Always use read_memory tool when asked about memory files. Never guess file contents.`;
    },
    []
  );

  /**
   * Execute a single step of the agent loop
   */
  const executeStep = useCallback(
    async (
      userMessage: string,
      persona: Persona,
      conversationHistory: Array<{ role: "user" | "assistant"; content: string }>,
      currentContext: string = ""
    ): Promise<{ response: string; done: boolean }> => {
      const systemPrompt = buildAgentSystemPrompt(persona, currentContext);

      const messages = [
        { role: "system" as const, content: systemPrompt },
        ...conversationHistory.map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        })),
        { role: "user" as const, content: userMessage },
      ];

      const response = await ollamaService.chat(model, messages, {
        temperature: 0.7,
      });

      // Check if response contains a tool call
      const toolCall = extractToolCall(response);

      if (toolCall) {
        // Execute the tool
        setSteps((prev) => [
          ...prev,
          { type: "tool_call", content: `Using tool: ${toolCall.name}`, toolCall },
        ]);

        const result = await memoryToolsService.executeReadOnlyToolCall(toolCall);
        
        // Add clear instructions based on tool type
        let instruction = "";
        if (toolCall.name === "read_memory" && result.success) {
          instruction = "\n\nCRITICAL: Report the EXACT file contents above. Do NOT make up different content.";
        }

        setSteps((prev) => [
          ...prev,
          { type: "tool_result", content: result.message, toolResult: result },
        ]);

        // Return the result with instruction as context for next iteration
        return {
          response: JSON.stringify(result) + instruction,
          done: false,
        };
      }

      // No tool call, this is the final response
      return { response, done: true };
    },
    [buildAgentSystemPrompt, extractToolCall, model]
  );

  /**
   * Run the full agent loop
   */
  const runAgent = useCallback(
    async (
      userMessage: string,
      persona: Persona,
      conversationHistory: Array<{ role: "user" | "assistant"; content: string }>
    ): Promise<string> => {
      setIsProcessing(true);
      setSteps([]);
      setPendingConfirmation(null);

      let context = "";
      let finalResponse = "";

      try {
        for (let step = 0; step < maxSteps; step++) {
          const { response, done } = await executeStep(
            step === 0 ? userMessage : "Continue",
            persona,
            conversationHistory,
            context
          );

          if (done) {
            finalResponse = response;
            setSteps((prev) => [...prev, { type: "response", content: response }]);
            break;
          } else if (pendingConfirmation) {
            // Waiting for user confirmation
            finalResponse = response;
            break;
          } else {
            // Add tool result to context for next iteration
            context += `\nTool result: ${response}`;
          }
        }

        if (!finalResponse) {
          finalResponse = "I've analyzed your request but reached my thinking limit. Could you rephrase?";
        }

        return finalResponse;
      } finally {
        setIsProcessing(false);
      }
    },
    [executeStep, maxSteps, pendingConfirmation]
  );

  /**
   * Confirm a pending write operation
   */
  const confirmWrite = useCallback(async (): Promise<string> => {
    setPendingConfirmation(null);
    return "Write operations are disabled in read-only agent mode.";
  }, [pendingConfirmation]);

  /**
   * Cancel a pending operation
   */
  const cancelOperation = useCallback(() => {
    setPendingConfirmation(null);
  }, []);

  /**
   * Clear steps
   */
  const clearSteps = useCallback(() => {
    setSteps([]);
  }, []);

  return {
    isProcessing,
    steps,
    pendingConfirmation,
    runAgent,
    confirmWrite,
    cancelOperation,
    clearSteps,
  };
}
