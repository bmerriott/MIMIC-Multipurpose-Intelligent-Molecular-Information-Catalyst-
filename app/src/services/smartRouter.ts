/**
 * Smart Router Service
 * Uses a lightweight LLM to intelligently route user queries
 * Provides agent behavior guidance without being verbose
 */

import { ollamaService } from "./ollama";
import type { Persona } from "@/types";

export type InputType = "voice" | "text";
export type Intent = 
  | "general_chat"
  | "memory_read"
  | "memory_write"
  | "web_search"
  | "vision_analysis"
  | "creative_writing"
  | "technical_help"
  | "personal_topic"
  | "source_request";

export interface RouteResult {
  inputType: InputType;
  primaryIntent: Intent;
  confidence: number;
  suggestedTools: string[];
  needsVisionModel: boolean;
  needsWebSearch: boolean;
  needsMemoryAccess: boolean;
  processingNotes: string;
  suggestedApproach: string;
  emotionalTone: string;
  requiresToolConfirmation: boolean;
}

// Models suitable for routing (must be small and fast)
export const ROUTER_MODELS = [
  { id: "qwen3:0.6b", name: "Qwen3 0.6B", vram: "~1GB", recommended: true },
  { id: "deepseek-r1:1.5b", name: "DeepSeek-R1 1.5B", vram: "~2GB", recommended: true },
  { id: "qwen2.5:0.5b", name: "Qwen2.5 0.5B", vram: "~1GB", recommended: false },
  { id: "phi3:mini", name: "Phi-3 Mini", vram: "~2GB", recommended: false },
  { id: "gemma2:2b", name: "Gemma 2 2B", vram: "~2GB", recommended: false },
];

// Check if a model is suitable for routing
export function isRouterModel(modelName: string): boolean {
  const lowerModel = modelName.toLowerCase();
  return ROUTER_MODELS.some(m => lowerModel.includes(m.id.split(':')[0]));
}

// Get recommended router models from available models
export function getAvailableRouterModels(allModels: string[]): string[] {
  return allModels.filter(isRouterModel);
}

class SmartRouter {
  private defaultModel = "qwen3:0.6b";

  /**
   * Intelligently route user input
   * Uses lightweight LLM to understand intent and context
   */
  async route(
    query: string,
    inputType: InputType,
    _persona: Persona,
    hasImages: boolean,
    _conversationHistory: Array<{ role: string; content: string }>,
    routerModel?: string
  ): Promise<RouteResult> {
    // Fast path: Check if user is asking for sources/URLs
    if (this.isSourceRequest(query)) {
      return {
        inputType,
        primaryIntent: "source_request",
        confidence: 0.95,
        suggestedTools: [],
        needsVisionModel: false,
        needsWebSearch: false,
        needsMemoryAccess: false,
        emotionalTone: "helpful",
        processingNotes: "User is requesting URLs/sources from previous search",
        suggestedApproach: "Provide source URLs from search results using markdown format.",
        requiresToolConfirmation: false,
      };
    }

    const model = routerModel || this.defaultModel;

    // Minimal router prompt for intent detection only
    const routerPrompt = `Analyze this input and respond with ONLY JSON:
{
  "primaryIntent": "general_chat|web_search|memory_read|memory_write|vision_analysis",
  "confidence": 0.0-1.0,
  "emotionalTone": "neutral|playful|serious|curious",
  "suggestedApproach": "Brief style guidance"
}

Input: "${query.substring(0, 200)}"
Has images: ${hasImages}`;

    try {
      const response = await ollamaService.generate(
        model,
        routerPrompt,
        undefined,
        { temperature: 0.3 }
      );

      // Parse JSON response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error("Router returned invalid JSON");
      }

      const parsed = JSON.parse(jsonMatch[0]);

      const result: RouteResult = {
        inputType,
        primaryIntent: parsed.primaryIntent || "general_chat",
        confidence: parsed.confidence || 0.5,
        suggestedTools: [],
        needsVisionModel: hasImages,
        needsWebSearch: parsed.primaryIntent === "web_search",
        needsMemoryAccess: parsed.primaryIntent === "memory_read" || parsed.primaryIntent === "memory_write",
        emotionalTone: parsed.emotionalTone || "neutral",
        processingNotes: "",
        suggestedApproach: parsed.suggestedApproach || "",
        requiresToolConfirmation: parsed.primaryIntent === "memory_write",
      };

      return result;
    } catch (error) {
      // Router error - use fallback
      // Fallback to safe defaults
      return this.getFallbackResult(inputType, hasImages);
    }
  }

  /**
   * Quick check if query needs web search
   * Lightweight keyword check for fast path
   */
  async needsWebSearch(query: string, routerModel?: string): Promise<boolean> {
    const model = routerModel || this.defaultModel;
    
    const prompt = `Does this query need current/real-time information from the web? "${query}"

Respond with ONLY "yes" or "no".

Needs web search if:
- Current events, news, weather
- Recent information (after 2023)
- Real-time data (prices, scores)
- Specific facts that might change

Does NOT need web search if:
- Personal conversation
- Creative writing
- General knowledge
- Opinion/advice`;

    try {
      const response = await ollamaService.generate(model, prompt, undefined, { temperature: 0.1 });
      return response.toLowerCase().includes("yes");
    } catch {
      return false;
    }
  }

  /**
   * Summarize search results for the brain model
   * Uses router model to extract key information and truncate
   */
  async summarizeSearchResults(
    query: string,
    searchResults: string,
    routerModel?: string
  ): Promise<string> {
    const model = routerModel || this.defaultModel;
    
    // If results are already short, return as-is
    if (searchResults.length < 2000) {
      return searchResults;
    }

    const prompt = `Summarize these search results for: "${query.substring(0, 100)}"

Search Results:
${searchResults.substring(0, 8000)}

Instructions:
1. Extract the most relevant facts to answer the query
2. Include source URLs for key information
3. Keep under 1000 words
4. Format: Brief answer + bullet points with sources

Summary:`;

    try {
      const response = await ollamaService.generate(model, prompt, undefined, { 
        temperature: 0.3
      });
      return response;
    } catch {
      // Fallback: truncate and return
      return searchResults.substring(0, 4000) + "\n...[Results truncated]";
    }
  }

  /**
   * Quick keyword check if user is asking for sources/URLs
   */
  isSourceRequest(query: string): boolean {
    const sourceKeywords = [
      'url', 'urls', 'link', 'links', 
      'source', 'sources', 
      'citation', 'citations', 
      'reference', 'references',
      'where did you get',
      'provide the link',
      'provide the url',
      'give me the link',
      'give me the url',
      'what are the sources',
      'what websites'
    ];
    
    const lowerQuery = query.toLowerCase();
    return sourceKeywords.some(keyword => lowerQuery.includes(keyword));
  }

  /**
   * Get fallback result if router fails
   */
  private getFallbackResult(inputType: InputType, hasImages: boolean): RouteResult {
    return {
      inputType,
      primaryIntent: "general_chat",
      confidence: 0.5,
      suggestedTools: hasImages ? ["vision"] : [],
      needsVisionModel: hasImages,
      needsWebSearch: false,
      needsMemoryAccess: false,
      emotionalTone: "neutral",
      processingNotes: "",
      suggestedApproach: "",
      requiresToolConfirmation: false,
    };
  }
}

export const smartRouter = new SmartRouter();
