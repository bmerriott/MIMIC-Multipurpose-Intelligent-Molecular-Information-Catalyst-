/**
 * Tool Confirmation Hook
 * 
 * Manages tool execution with mandatory user confirmation for write operations.
 * Provides preview of exact command and changes before execution.
 */

import { useState, useCallback } from "react";
import { memoryToolsService, type ToolCall, type ToolResult } from "@/services/memoryTools";
import { formatToolConfirmation, requiresConfirmation } from "@/services/agentSystem";

export interface PendingToolConfirmation {
  id: string;
  toolName: string;
  args: Record<string, any>;
  personaName: string;
  preview: {
    title: string;
    description: string;
    content: string;
  };
  resolve: (result: ToolResult) => void;
  reject: (error: Error) => void;
}

export interface UseToolConfirmationReturn {
  pendingConfirmation: PendingToolConfirmation | null;
  isExecuting: boolean;
  executeWithConfirmation: (
    toolCall: ToolCall,
    personaName: string,
    personaId: string
  ) => Promise<ToolResult>;
  confirmExecution: () => Promise<void>;
  cancelExecution: () => void;
  clearConfirmation: () => void;
}

export function useToolConfirmation(): UseToolConfirmationReturn {
  const [pendingConfirmation, setPendingConfirmation] = useState<PendingToolConfirmation | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);

  /**
   * Execute a tool call, requiring confirmation for write operations
   */
  const executeWithConfirmation = useCallback(async (
    toolCall: ToolCall,
    personaName: string,
    personaId: string
  ): Promise<ToolResult> => {
    // Check if this tool requires confirmation
    if (!requiresConfirmation(toolCall.name)) {
      // Read-only tool - execute immediately
      setIsExecuting(true);
      try {
        const result = await memoryToolsService.executeReadOnlyToolCall(toolCall, personaId);
        return result;
      } finally {
        setIsExecuting(false);
      }
    }

    // Write operation - require confirmation
    return new Promise((resolve, reject) => {
      const formatted = formatToolConfirmation(toolCall.name, {
        ...toolCall.arguments,
        personaName,
      });

      const pending: PendingToolConfirmation = {
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        toolName: toolCall.name,
        args: { ...toolCall.arguments, personaId },
        personaName,
        preview: {
          title: formatted.title,
          description: formatted.description,
          content: formatted.preview,
        },
        resolve,
        reject,
      };

      setPendingConfirmation(pending);
    });
  }, []);

  /**
   * Confirm and execute the pending tool
   */
  const confirmExecution = useCallback(async () => {
    if (!pendingConfirmation) return;

    setIsExecuting(true);
    
    try {
      let result: ToolResult;

      switch (pendingConfirmation.toolName) {
        case "write_memory":
          result = await memoryToolsService.writeMemory(
            pendingConfirmation.args.filename,
            pendingConfirmation.args.content,
            pendingConfirmation.args.personaId,
            true, // confirm = true
            pendingConfirmation.args.folder || "user_files"
          );
          break;

        case "delete_memory":
          result = await memoryToolsService.deleteMemory(
            pendingConfirmation.args.filename,
            pendingConfirmation.args.personaId,
            true // confirm = true
          );
          break;

        case "clear_conversation_history":
          result = await memoryToolsService.clearConversationHistory(
            pendingConfirmation.args.personaId,
            true // confirm = true
          );
          break;

        default:
          result = {
            success: false,
            message: `Unknown tool: ${pendingConfirmation.toolName}`,
          };
      }

      pendingConfirmation.resolve(result);
    } catch (error) {
      pendingConfirmation.reject(error instanceof Error ? error : new Error(String(error)));
    } finally {
      setIsExecuting(false);
      setPendingConfirmation(null);
    }
  }, [pendingConfirmation]);

  /**
   * Cancel the pending tool execution
   */
  const cancelExecution = useCallback(() => {
    if (!pendingConfirmation) return;

    pendingConfirmation.resolve({
      success: false,
      message: "Operation cancelled by user",
    });
    setPendingConfirmation(null);
  }, [pendingConfirmation]);

  /**
   * Clear the confirmation state without resolving
   */
  const clearConfirmation = useCallback(() => {
    setPendingConfirmation(null);
  }, []);

  return {
    pendingConfirmation,
    isExecuting,
    executeWithConfirmation,
    confirmExecution,
    cancelExecution,
    clearConfirmation,
  };
}
