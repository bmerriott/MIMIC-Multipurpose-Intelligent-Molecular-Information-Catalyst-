/**
 * Intent Router Service
 * Uses a lightweight local model to classify user intent
 * Routes to appropriate minimal system prompt
 */

export type Intent = 
  | "memory_read"      // User wants to read memory files
  | "memory_write"     // User wants to write to memory
  | "web_search"       // User wants current info/search
  | "general_chat";    // Regular conversation

export interface RouteResult {
  intent: Intent;
  confidence: number;
  systemPrompt: string;
  needsTools: boolean;
}

// Minimal prompts for each intent - no bloat
const MINIMAL_PROMPTS: Record<Intent, string> = {
  memory_read: `You are {name}. You have access to memory files.

CRITICAL: You CANNOT see file contents. You MUST use the tool.

TO READ A FILE - Output ONLY this JSON, nothing else:
{"name":"read_memory","arguments":{"filename":"test.txt"}}

THEN you will receive TOOL_RESULT with the actual content.

FINAL ANSWER FORMAT:
"The file contains: [paste exact text from TOOL_RESULT]"

NEVER:
- Guess or make up content
- Repeat content from previous files
- Say "I love Tacos" unless that's in TOOL_RESULT
- Be conversational or add commentary`,

  memory_write: `You are {name}. User wants to save information.

TO WRITE A FILE:
Output: {"name":"write_memory","arguments":{"filename":"notes.txt","content":"...","confirm":true}}

Ask user to confirm before writing.`,

  web_search: `You are {name}. You have web search results in context.

Answer based on the provided search results.
Include source URLs when relevant.

Search results are in [INTERNET DATA] section.`,

  general_chat: `You are {name}, an AI assistant.

Be helpful, natural, and conversational.
No stage directions or action markers.`,
};

// Intent detection keywords (fast, no model needed for simple cases)
const INTENT_PATTERNS: Record<Intent, RegExp[]> = {
  memory_read: [
    /file\s+named/i,
    /contents?\s+of/i,
    /read\s+(the\s+)?file/i,
    /what\s+(is|does)\s+.*\s+say/i,
    /test\d*\.txt/i,  // test.txt, test1.txt, test2.txt, test3.txt, etc
    /\w+\.txt/i,      // ANY filename ending in .txt
    /\.txt\s+(file|contains?)/i,
    /memory\s+file/i,
    /saved\s+file/i,
    /provide\s+.*contents?/i,
  ],
  memory_write: [
    /save\s+(this|that|it)\s+to/i,
    /write\s+to\s+file/i,
    /create\s+a?\s*file/i,
    /remember\s+this/i,
    /add\s+to\s+memory/i,
  ],
  web_search: [
    /search\s+(for|the)/i,
    /look\s+up/i,
    /find\s+(me|out)/i,
    /latest\s+(news|info)/i,
    /current\s+(weather|price|news)/i,
    /what\s+happened\s+(today|yesterday|recently)/i,
  ],
  general_chat: [],
};

class IntentRouter {
  /**
   * Route user query to appropriate intent and system prompt
   * Uses fast keyword matching
   */
  async route(
    query: string, 
    personaName: string,
    hasWebSearchContext: boolean = false
  ): Promise<RouteResult> {
    // Fast keyword matching
    for (const [intent, patterns] of Object.entries(INTENT_PATTERNS) as [Intent, RegExp[]][]) {
      for (const pattern of patterns) {
        if (pattern.test(query)) {
          return {
            intent,
            confidence: 0.9,
            systemPrompt: MINIMAL_PROMPTS[intent].replace("{name}", personaName),
            needsTools: intent === "memory_read" || intent === "memory_write",
          };
        }
      }
    }

    // If web search context exists, assume web_search intent for time-sensitive queries
    if (hasWebSearchContext) {
      return {
        intent: "web_search",
        confidence: 0.8,
        systemPrompt: MINIMAL_PROMPTS.web_search.replace("{name}", personaName),
        needsTools: false,
      };
    }

    // Default to general chat
    return {
      intent: "general_chat",
      confidence: 1.0,
      systemPrompt: MINIMAL_PROMPTS.general_chat.replace("{name}", personaName),
      needsTools: false,
    };
  }

  /**
   * Get minimal system prompt for memory read with file list
   */
  getMemoryReadPrompt(personaName: string, fileNames: string[]): string {
    const basePrompt = MINIMAL_PROMPTS.memory_read.replace("{name}", personaName);
    const filesList = fileNames.length > 0 
      ? `Files: ${fileNames.join(", ")}` 
      : "No files found.";
    
    return `${basePrompt}\n\n${filesList}\n\nQuote file contents exactly when asked.`;
  }
}

export const intentRouter = new IntentRouter();
