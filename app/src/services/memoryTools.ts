/**
 * Secure Memory Tools Service
 * Frontend interface for the mediated file access system
 */

export interface MemoryFile {
  name: string;
  path: string;
  size: number;
  modified: string;
  preview: string;
}

export interface MemorySearchMatch {
  filename: string;
  snippet: string;
  matches: number;
}

export interface ToolCall {
  name: string;
  arguments: Record<string, any>;
}

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  type?: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export type ReadOnlyToolName =
  | "list_memories"
  | "read_memory"
  | "search_memories"
  | "get_memory_info"
  | "get_conversation_history"
  | "search_conversation_history";

export interface ToolResult {
  success: boolean;
  data?: any;
  message: string;
}

class MemoryToolsService {
  private baseUrl: string;
  private isAvailable: boolean = false;
  
  constructor() {
    // Use direct backend URL in production, relative path in dev
    const isDevServer = window.location.hostname === 'localhost' && window.location.port === '5173';
    this.baseUrl = isDevServer ? "/api" : "http://localhost:8000/api";
  }

  async checkAvailability(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/memory/list`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });
      this.isAvailable = response.ok;
      return this.isAvailable;
    } catch {
      this.isAvailable = false;
      return false;
    }
  }

  isEnabled(): boolean {
    return this.isAvailable;
  }

  /**
   * List all memory files for a persona
   * @param personaId The persona identifier
   * @param folder 'user_files', 'conversations', or undefined for both
   */
  async listMemories(personaId: string = "default", folder?: "user_files" | "conversations" | "all"): Promise<MemoryFile[]> {
    try {
      const url = new URL(`${this.baseUrl}/memory/list`, window.location.origin);
      if (folder) {
        url.searchParams.append("folder", folder);
      }
      url.searchParams.append("persona_id", personaId);
      
      const response = await fetch(url.toString(), {
        signal: AbortSignal.timeout(10000),
      });
      if (!response.ok) {
        const errorText = await response.text();
        console.error("[MemoryTools] listMemories failed:", response.status, errorText);
        throw new Error(`Failed to list memories: ${response.status}`);
      }
      const data = await response.json();
      return data.files || [];
    } catch (error) {
      console.error("[MemoryTools] listMemories error:", error);
      // Return empty array instead of crashing
      return [];
    }
  }

  /**
   * Read a specific memory file
   * @param filename Name of file (can include folder prefix like "user_files/" or "conversations/")
   * @param personaId The persona identifier
   */
  async readMemory(filename: string, personaId: string = "default"): Promise<string> {
    try {
      const response = await fetch(`${this.baseUrl}/memory/read`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, persona_id: personaId }),
        signal: AbortSignal.timeout(10000),
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to read memory");
      }
      const data = await response.json();
      return data.content;
    } catch (error: any) {
      console.error("[MemoryTools] readMemory error:", error);
      throw new Error(error.message || "Failed to read memory file");
    }
  }

  /**
   * Search memories for keywords
   * @param query Search query
   * @param personaId The persona identifier
   * @param folder 'user_files', 'conversations', or undefined for both
   */
  async searchMemories(query: string, personaId: string = "default", folder?: "user_files" | "conversations" | "all"): Promise<MemorySearchMatch[]> {
    try {
      const response = await fetch(`${this.baseUrl}/memory/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, folder, persona_id: personaId }),
        signal: AbortSignal.timeout(10000),
      });
      if (!response.ok) {
        throw new Error("Failed to search memories");
      }
      const data = await response.json();
      return data.matches || [];
    } catch (error) {
      console.error("[MemoryTools] searchMemories error:", error);
      return [];
    }
  }

  /**
   * Write to a memory file
   * REQUIRES user confirmation
   * @param filename Name of file
   * @param content Content to write
   * @param personaId The persona identifier
   * @param folder 'user_files' (default) or 'conversations'
   */
  async writeMemory(
    filename: string, 
    content: string, 
    personaId: string = "default",
    confirm: boolean = false,
    folder: "user_files" | "conversations" = "user_files"
  ): Promise<ToolResult> {
    const response = await fetch(`${this.baseUrl}/memory/write`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename, content, confirm, folder, persona_id: personaId }),
    });
    const data = await response.json();
    return {
      success: data.success,
      message: data.message,
    };
  }

  /**
   * Delete a memory file
   * REQUIRES user confirmation
   * @param filename Name of file
   * @param personaId The persona identifier
   */
  async deleteMemory(filename: string, personaId: string = "default", confirm: boolean = false): Promise<ToolResult> {
    const response = await fetch(`${this.baseUrl}/memory/delete?filename=${encodeURIComponent(filename)}&confirm=${confirm}&persona_id=${encodeURIComponent(personaId)}`, {
      method: "DELETE",
    });
    const data = await response.json();
    return {
      success: data.success,
      message: data.message,
    };
  }

  // =========================================================================
  // CONVERSATION HISTORY API
  // =========================================================================

  /**
   * Save a conversation message to the persona's history
   */
  async saveConversationMessage(
    personaId: string,
    role: "user" | "assistant",
    content: string,
    messageType: string = "text",
    metadata?: Record<string, any>
  ): Promise<ToolResult> {
    const response = await fetch(`${this.baseUrl}/memory/conversation/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        persona_id: personaId,
        role,
        content,
        message_type: messageType,
        metadata: {
          ...metadata,
          timestamp: new Date().toISOString(),
        },
      }),
    });
    const data = await response.json();
    return {
      success: data.success,
      message: data.message,
      data: data.data,
    };
  }

  /**
   * Get the full conversation history for a persona
   */
  async getConversationHistory(personaId: string, limit?: number): Promise<ConversationMessage[]> {
    const url = new URL(`${this.baseUrl}/memory/conversation/history`, window.location.origin);
    url.searchParams.append("persona_id", personaId);
    if (limit) {
      url.searchParams.append("limit", limit.toString());
    }
    
    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error("Failed to get conversation history");
    }
    const data = await response.json();
    return data.history || [];
  }

  /**
   * Search conversation history for specific content
   */
  async searchConversationHistory(personaId: string, query: string): Promise<ConversationMessage[]> {
    const response = await fetch(`${this.baseUrl}/memory/conversation/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persona_id: personaId, query }),
    });
    if (!response.ok) {
      throw new Error("Failed to search conversation history");
    }
    const data = await response.json();
    return data.matches || [];
  }

  /**
   * Clear all conversation history for a persona
   */
  async clearConversationHistory(personaId: string, confirm: boolean = false): Promise<ToolResult> {
    const url = new URL(`${this.baseUrl}/memory/conversation/clear`, window.location.origin);
    url.searchParams.append("persona_id", personaId);
    url.searchParams.append("confirm", confirm.toString());
    
    const response = await fetch(url.toString(), { method: "DELETE" });
    const data = await response.json();
    return {
      success: data.success,
      message: data.message,
    };
  }

  /**
   * Initialize persona folders (creates user_files and conversations folders)
   * Call this when creating a new persona
   */
  async initializePersonaFolders(personaId: string, personaName: string): Promise<boolean> {
    try {
      // Create a welcome/info file in the persona's folder
      const welcomeContent = `# ${personaName}'s Memory Folder

This folder contains saved information for ${personaName}.
- Files in this folder can be read by ${personaName} during conversations
- Use this to store facts, preferences, notes, and other information

## Folder Structure
- user_files/ - Your saved notes and documents
- conversations/ - Auto-saved conversation summaries
`;
      const result = await this.writeMemory(
        "README.md",
        welcomeContent,
        personaId,
        true,
        "user_files"
      );
      console.log(`[MemoryTools] Initialized folders for persona ${personaId}:`, result.message);
      return result.success;
    } catch (error) {
      console.error("[MemoryTools] Failed to initialize persona folders:", error);
      return false;
    }
  }

  /**
   * Get the base path for a persona's memory storage
   * This is useful for understanding where files are stored
   */
  getPersonaMemoryPath(personaId: string): string {
    // This is for information purposes - actual path is on the backend
    return `~/MimicAI/Memories/${personaId}/`;
  }

  /**
   * Get tool schema for Ollama
   */
  async getToolSchema(): Promise<any[]> {
    try {
      const response = await fetch(`${this.baseUrl}/memory/tools`, {
        signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) {
        return [];
      }
      return response.json();
    } catch (error) {
      console.error("[MemoryTools] getToolSchema error:", error);
      return [];
    }
  }

  getReadOnlyToolSchema(): Array<{
    name: ReadOnlyToolName;
    description: string;
    args: string;
  }> {
    return [
      {
        name: "list_memories",
        description: "List all files in the memory folder for the current persona",
        args: "{}",
      },
      {
        name: "read_memory",
        description: "Read memory file content by filename",
        args: '{"filename":"example.txt"}',
      },
      {
        name: "search_memories",
        description: "Search all memory files for a query string",
        args: '{"query":"keyword"}',
      },
      {
        name: "get_memory_info",
        description: "Get metadata for one memory file by filename",
        args: '{"filename":"example.txt"}',
      },
      {
        name: "get_conversation_history",
        description: "Get the full conversation history with the user",
        args: '{"limit":50}',
      },
      {
        name: "search_conversation_history",
        description: "Search conversation history for specific content",
        args: '{"query":"keyword"}',
      },
    ];
  }

  /**
   * Execute ONLY read-only tool calls.
   * Any write/delete/unknown tool is rejected.
   */
  async executeReadOnlyToolCall(toolCall: ToolCall, personaId: string = "default"): Promise<ToolResult> {
    const { name, arguments: args } = toolCall;
    if (!["list_memories", "read_memory", "search_memories", "get_memory_info", "get_conversation_history", "search_conversation_history"].includes(name)) {
      return {
        success: false,
        message: `Tool not allowed in read-only mode: ${name}`,
      };
    }

    try {
      switch (name as ReadOnlyToolName) {
        case "list_memories": {
          const files = await this.listMemories(personaId);
          return {
            success: true,
            data: files,
            message: `Found ${files.length} memory files`,
          };
        }
        case "read_memory": {
          const content = await this.readMemory(args.filename, personaId);
          return {
            success: true,
            data: { filename: args.filename, content },
            message: `Read ${args.filename}`,
          };
        }
        case "search_memories": {
          const matches = await this.searchMemories(args.query, personaId);
          return {
            success: true,
            data: matches,
            message: `Found ${matches.length} matches for "${args.query}"`,
          };
        }
        case "get_memory_info": {
          const allFiles = await this.listMemories(personaId);
          const file = allFiles.find((f) => f.name === args.filename);
          if (file) {
            return {
              success: true,
              data: file,
              message: `Retrieved info for ${args.filename}`,
            };
          }
          return {
            success: false,
            message: `File not found: ${args.filename}`,
          };
        }
        case "get_conversation_history": {
          const history = await this.getConversationHistory(personaId, args.limit);
          return {
            success: true,
            data: history,
            message: `Retrieved ${history.length} conversation messages`,
          };
        }
        case "search_conversation_history": {
          const matches = await this.searchConversationHistory(personaId, args.query);
          return {
            success: true,
            data: matches,
            message: `Found ${matches.length} matching messages`,
          };
        }
      }
    } catch (error: any) {
      return {
        success: false,
        message: error.message || "Tool execution failed",
      };
    }
  }

  /**
   * Execute a tool call from the agent
   */
  async executeToolCall(toolCall: ToolCall, personaId: string = "default"): Promise<ToolResult> {
    const { name, arguments: args } = toolCall;

    try {
      switch (name) {
        case "list_memories":
          const files = await this.listMemories(personaId);
          return {
            success: true,
            data: files,
            message: `Found ${files.length} memory files`,
          };

        case "read_memory":
          const content = await this.readMemory(args.filename, personaId);
          return {
            success: true,
            data: { filename: args.filename, content },
            message: `Read ${args.filename}`,
          };

        case "search_memories":
          const matches = await this.searchMemories(args.query, personaId);
          return {
            success: true,
            data: matches,
            message: `Found ${matches.length} matches for "${args.query}"`,
          };

        case "get_memory_info":
          const allFiles = await this.listMemories(personaId);
          const file = allFiles.find((f) => f.name === args.filename);
          if (file) {
            return {
              success: true,
              data: file,
              message: `Retrieved info for ${args.filename}`,
            };
          } else {
            return {
              success: false,
              message: `File not found: ${args.filename}`,
            };
          }

        case "get_conversation_history":
          const history = await this.getConversationHistory(personaId, args.limit);
          return {
            success: true,
            data: history,
            message: `Retrieved ${history.length} conversation messages`,
          };

        case "search_conversation_history":
          const convMatches = await this.searchConversationHistory(personaId, args.query);
          return {
            success: true,
            data: convMatches,
            message: `Found ${convMatches.length} matching messages`,
          };

        case "write_memory":
          if (!args.confirm) {
            return {
              success: false,
              message: "Write operation requires user confirmation",
            };
          }
          return await this.writeMemory(args.filename, args.content, personaId, true);

        default:
          return {
            success: false,
            message: `Unknown tool: ${name}`,
          };
      }
    } catch (error: any) {
      return {
        success: false,
        message: error.message || "Tool execution failed",
      };
    }
  }
}

export const memoryToolsService = new MemoryToolsService();
