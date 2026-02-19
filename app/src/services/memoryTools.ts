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

export type ReadOnlyToolName =
  | "list_memories"
  | "read_memory"
  | "search_memories"
  | "get_memory_info";

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
   * List all memory files
   */
  async listMemories(): Promise<MemoryFile[]> {
    const response = await fetch(`${this.baseUrl}/memory/list`);
    if (!response.ok) {
      throw new Error("Failed to list memories");
    }
    const data = await response.json();
    return data.files || [];
  }

  /**
   * Read a specific memory file
   */
  async readMemory(filename: string): Promise<string> {
    const response = await fetch(`${this.baseUrl}/memory/read`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to read memory");
    }
    const data = await response.json();
    return data.content;
  }

  /**
   * Search memories for keywords
   */
  async searchMemories(query: string): Promise<MemorySearchMatch[]> {
    const response = await fetch(`${this.baseUrl}/memory/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!response.ok) {
      throw new Error("Failed to search memories");
    }
    const data = await response.json();
    return data.matches || [];
  }

  /**
   * Write to a memory file
   * REQUIRES user confirmation
   */
  async writeMemory(filename: string, content: string, confirm: boolean = false): Promise<ToolResult> {
    const response = await fetch(`${this.baseUrl}/memory/write`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename, content, confirm }),
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
   */
  async deleteMemory(filename: string, confirm: boolean = false): Promise<ToolResult> {
    const response = await fetch(`${this.baseUrl}/memory/delete?filename=${encodeURIComponent(filename)}&confirm=${confirm}`, {
      method: "DELETE",
    });
    const data = await response.json();
    return {
      success: data.success,
      message: data.message,
    };
  }

  /**
   * Get tool schema for Ollama
   */
  async getToolSchema(): Promise<any[]> {
    const response = await fetch(`${this.baseUrl}/memory/tools`);
    if (!response.ok) {
      return [];
    }
    return response.json();
  }

  getReadOnlyToolSchema(): Array<{
    name: ReadOnlyToolName;
    description: string;
    args: string;
  }> {
    return [
      {
        name: "list_memories",
        description: "List all files in the memory folder",
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
    ];
  }

  /**
   * Execute ONLY read-only tool calls.
   * Any write/delete/unknown tool is rejected.
   */
  async executeReadOnlyToolCall(toolCall: ToolCall): Promise<ToolResult> {
    const { name, arguments: args } = toolCall;
    if (!["list_memories", "read_memory", "search_memories", "get_memory_info"].includes(name)) {
      return {
        success: false,
        message: `Tool not allowed in read-only mode: ${name}`,
      };
    }

    try {
      switch (name as ReadOnlyToolName) {
        case "list_memories": {
          const files = await this.listMemories();
          return {
            success: true,
            data: files,
            message: `Found ${files.length} memory files`,
          };
        }
        case "read_memory": {
          const content = await this.readMemory(args.filename);
          return {
            success: true,
            data: { filename: args.filename, content },
            message: `Read ${args.filename}`,
          };
        }
        case "search_memories": {
          const matches = await this.searchMemories(args.query);
          return {
            success: true,
            data: matches,
            message: `Found ${matches.length} matches for "${args.query}"`,
          };
        }
        case "get_memory_info": {
          const allFiles = await this.listMemories();
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
  async executeToolCall(toolCall: ToolCall): Promise<ToolResult> {
    const { name, arguments: args } = toolCall;

    try {
      switch (name) {
        case "list_memories":
          const files = await this.listMemories();
          return {
            success: true,
            data: files,
            message: `Found ${files.length} memory files`,
          };

        case "read_memory":
          const content = await this.readMemory(args.filename);
          return {
            success: true,
            data: { filename: args.filename, content },
            message: `Read ${args.filename}`,
          };

        case "search_memories":
          const matches = await this.searchMemories(args.query);
          return {
            success: true,
            data: matches,
            message: `Found ${matches.length} matches for "${args.query}"`,
          };

        case "get_memory_info":
          // This is a simplified version - backend doesn't have a separate endpoint
          const allFiles = await this.listMemories();
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

        case "write_memory":
          if (!args.confirm) {
            return {
              success: false,
              message: "Write operation requires user confirmation",
            };
          }
          return await this.writeMemory(args.filename, args.content, true);

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
