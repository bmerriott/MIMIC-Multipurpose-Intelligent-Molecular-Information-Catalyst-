import type { 
  OllamaModel, 
  OllamaChatRequest, 
  OllamaChatResponse,
  OllamaGenerateRequest,
  OllamaGenerateResponse,
  Persona,
  PersonaMemory 
} from "@/types";

export class OllamaService {
  private baseUrl: string;
  private useProxy: boolean;

  constructor(baseUrl: string = "http://localhost:11434") {
    this.baseUrl = baseUrl;
    // Only use proxy in Vite development server
    // In Tauri production, use direct connection (no CORS issues in WebView)
    const isDevServer = window.location.hostname === 'localhost' && window.location.port === '5173';
    this.useProxy = isDevServer && baseUrl.includes('localhost:11434');
  }

  setBaseUrl(url: string) {
    this.baseUrl = url;
    const isDevServer = window.location.hostname === 'localhost' && window.location.port === '5173';
    this.useProxy = isDevServer && url.includes('localhost:11434');
  }

  private getUrl(endpoint: string): string {
    // Use Vite proxy in development to bypass CORS
    if (this.useProxy) {
      return `/api/ollama${endpoint}`;
    }
    return `${this.baseUrl}${endpoint}`;
  }

  async checkConnection(): Promise<boolean> {
    const url = this.getUrl('/api/tags');
    console.log('[Ollama] Checking connection at:', url);
    
    // Try with regular CORS first
    try {
      const response = await fetch(url, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(3000)
      });
      
      if (response.ok) {
        console.log('[Ollama] Connection successful, status:', response.status);
        return true;
      }
    } catch (error: any) {
      console.log('[Ollama] CORS check failed:', error.message || error);
    }
    
    // Try with no-cors mode for Tauri WebView
    try {
      await fetch(url, {
        method: "GET",
        mode: "no-cors",
        signal: AbortSignal.timeout(3000)
      });
      // With no-cors, assume OK if no error
      console.log('[Ollama] Connection successful (no-cors mode)');
      return true;
    } catch (error: any) {
      console.error("[Ollama] Both connection checks failed:", error.message || error);
      return false;
    }
  }

  async listModels(): Promise<OllamaModel[]> {
    try {
      const url = this.getUrl('/api/tags');
      console.log('Fetching models from:', url);
      
      const response = await fetch(url, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to fetch models: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      const allModels = data.models || [];
      console.log('Models received:', allModels.length);
      
      // Filter out embedding models that don't support chat
      const chatModels = this.filterChatModels(allModels);
      console.log('Chat-capable models:', chatModels.length);
      
      return chatModels;
    } catch (error) {
      console.error("Failed to list Ollama models:", error);
      throw error;
    }
  }

  private normalizeModelName(model: string): string {
    // Ollama uses hyphens, not underscores in model names
    // e.g., "qwen3_tts_0_6b" -> "qwen3-tts-0.6b"
    // But LLM models like "llama3.2" should stay as-is
    // Also preserve tags like "qwen3:30b"
    
    // IMPORTANT: Don't modify if it already looks like a valid Ollama model name
    // Valid formats: "llama3.2:latest", "qwen3:30b", "mistral:latest"
    if (model.includes(':') && !model.includes('_')) {
      return model;
    }
    
    // Split by colon to preserve tag
    const [name, tag] = model.split(':');
    
    // Normalize the name part (replace underscores with hyphens)
    let normalizedName = name;
    if (normalizedName.includes('_')) {
      normalizedName = normalizedName.replace(/_/g, '-');
    }
    
    // Reconstruct with tag if present
    return tag ? `${normalizedName}:${tag}` : normalizedName;
  }
  
  /**
   * Filter out models that don't support chat (embeddings, etc.)
   */
  filterChatModels(models: OllamaModel[]): OllamaModel[] {
    return models.filter(model => {
      const name = model.name.toLowerCase();
      // Filter out embedding models
      if (name.includes('embed')) return false;
      if (name.includes('embedding')) return false;
      // Keep all other models
      return true;
    });
  }

  async chat(
    model: string,
    messages: { role: "system" | "user" | "assistant"; content: string; images?: string[] }[],
    options?: { temperature?: number; top_p?: number; top_k?: number; repeat_penalty?: number }
  ): Promise<string> {
    // Normalize model name for Ollama
    const normalizedModel = this.normalizeModelName(model);
    if (normalizedModel !== model) {
      console.log(`[Ollama] Normalized model name: ${model} -> ${normalizedModel}`);
    }
    // Support vision models by including images in the request
    const formattedMessages = messages.map(m => ({
      role: m.role,
      content: m.content,
      ...(m.images && m.images.length > 0 ? { images: m.images } : {}),
    }));
    
    const request: OllamaChatRequest = {
      model: normalizedModel,
      messages: formattedMessages,
      stream: false,
      options: {
        temperature: options?.temperature ?? 0.7,
        top_p: options?.top_p ?? 0.9,
        top_k: options?.top_k ?? 40,
        repeat_penalty: options?.repeat_penalty ?? 1.1,
      },
    };

    try {
      const url = this.getUrl('/api/chat');
      console.log('Sending chat request to:', url, 'with model:', model);
      console.log('Request messages count:', request.messages.length);
      console.log('System prompt length:', request.messages[0]?.content?.length || 0);
      console.log('Full system prompt:', request.messages[0]?.content);
      
      // Add timeout for Ollama request (5 minutes max)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 min timeout
      
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Ollama HTTP error:', response.status, errorText);
        console.error('Request details:', {
          url: url,
          model: model,
          messageCount: request.messages.length
        });
        // Try to parse error for more details
        let errorDetail = errorText;
        try {
          const errorJson = JSON.parse(errorText);
          errorDetail = errorJson.error || errorText;
        } catch {}
        throw new Error(`Ollama error: ${response.status} - ${errorDetail}`);
      }

      const data: OllamaChatResponse = await response.json();
      console.log('Ollama raw response:', data);
      
      let content = data.message?.content;
      if (!content) {
        console.error('Empty content in response:', data);
        throw new Error('Model returned empty response');
      }
      
      console.log('Chat response received, length:', content.length);
      
      // Clean stage directions from response
      content = this.cleanResponse(content);
      console.log('Cleaned response, length:', content.length);
      
      return content;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Ollama request timed out after 5 minutes. The model may be stuck or overloaded.');
      }
      console.error("Ollama chat error:", error);
      throw error;
    }
  }

  async generate(
    model: string,
    prompt: string,
    systemPrompt?: string,
    options?: { temperature?: number }
  ): Promise<string> {
    const request: OllamaGenerateRequest = {
      model,
      prompt,
      system: systemPrompt,
      stream: false,
      options: {
        temperature: options?.temperature ?? 0.7,
      },
    };

    try {
      const response = await fetch(this.getUrl('/api/generate'), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Generate request failed: ${response.status} ${errorText}`);
      }

      const data: OllamaGenerateResponse = await response.json();
      return data.response;
    } catch (error) {
      console.error("Ollama generate error:", error);
      throw error;
    }
  }

  async generateAvatarDescription(
    model: string,
    persona: Persona
  ): Promise<{
    description: string;
    primary_color: string;
    secondary_color: string;
    glow_color: string;
    shape_type: string;
    animation_style: string;
    complexity: number;
    reasoning: string;
  }> {
    const systemPrompt = `You are an AI avatar designer. Based on the personality and characteristics of an AI assistant, design a 3D avatar appearance.

Respond with a JSON object containing:
- description: A vivid description of the avatar's visual appearance
- primary_color: Primary color in hex format (e.g., "#6366f1")
- secondary_color: Secondary color in hex format (e.g., "#8b5cf6") 
- glow_color: Glow/emission color in hex format (e.g., "#a78bfa")
- shape_type: One of: "sphere", "cube", "torus", "icosahedron"
- animation_style: One of: "flowing", "pulsing", "wave", "static"
- complexity: A number between 0.3 and 1.0 representing visual complexity
- reasoning: Brief explanation of why these choices fit the personality

Colors should reflect the personality traits. Energetic personalities might use warm colors (oranges, reds), calm personalities might use cool colors (blues, purples), nature-themed might use greens, etc.`;

    const userPrompt = `Design an avatar for an AI assistant with the following characteristics:

Name: ${persona.name}
Description: ${persona.description}
Personality: ${persona.personality_prompt}

Create a cohesive visual design that embodies this personality. Return only the JSON object.`;

    try {
      const response = await this.generate(model, userPrompt, systemPrompt, {
        temperature: 0.8,
      });

      // Extract JSON from response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
      
      throw new Error("Could not parse avatar description from response");
    } catch (error) {
      console.error("Avatar generation error:", error);
      // Return default values
      return {
        description: "A mystical sphere with flowing energy patterns",
        primary_color: "#6366f1",
        secondary_color: "#8b5cf6",
        glow_color: "#a78bfa",
        shape_type: "sphere",
        animation_style: "flowing",
        complexity: 0.7,
        reasoning: "Default design",
      };
    }
  }

  async summarizeMemory(
    model: string,
    memory: PersonaMemory,
    currentPersona: Persona
  ): Promise<{ summary: string; importantMemories: string[] }> {
    const systemPrompt = `You are a memory summarization AI. Summarize the conversation history and extract important memories for an AI assistant.

Respond with a JSON object:
- summary: A concise summary of the conversation and relationship context
- importantMemories: Array of 3-5 key facts/memories to remember about the user`;

    const recentMessages = memory.short_term
      .slice(-10)
      .map((m) => `- ${m.content}`)
      .join("\n");

    const userPrompt = `As ${currentPersona.name}, summarize these recent interactions:

Previous Summary: ${memory.summary || "None yet"}

Recent Messages:
${recentMessages}

Extract key information about the user, their preferences, and ongoing topics. Return only the JSON object.`;

    try {
      const response = await this.generate(model, userPrompt, systemPrompt, {
        temperature: 0.5,
      });

      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          summary: parsed.summary || memory.summary,
          importantMemories: parsed.importantMemories || [],
        };
      }
      
      throw new Error("Could not parse memory summary");
    } catch (error) {
      console.error("Memory summarization error:", error);
      return {
        summary: memory.summary,
        importantMemories: [],
      };
    }
  }

  buildPersonaSystemPrompt(
    persona: Persona,
    hasImages: boolean = false,
    includeMemory: boolean = false,
    searchContext?: string,
    fileContext?: string,
    rulesContext?: string,
    toolPolicy?: string
  ): string {
    // CONCISE system prompt to save tokens
    const parts: string[] = [];
    
    // Core identity
    parts.push(`You are ${persona.name}, an AI Digital Assistant.`);
    parts.push("Respond naturally without referencing technical modes or input methods.");

    if (rulesContext) {
      parts.push(`\n[SYSTEM RULES]\n${rulesContext}\n[END SYSTEM RULES]`);
    }
    
    // Context-specific capabilities (current session only)
    if (searchContext) {
      parts.push("\nYou have been provided with real-time web search results above. Use them to answer accurately.");
    }
    if (fileContext) {
      parts.push("\nYou have been provided with user file contents above. Reference them as needed.");
    }
    if (hasImages) {
      parts.push("\nThe user has shared images for you to analyze.");
    }
    
    // IMPORTANT: When search results provided, THIS IS THE INTERNET
    if (searchContext) {
      parts.push(`\n[INTERNET DATA ${new Date().toLocaleDateString()}]\n${searchContext}\n[END INTERNET DATA]`);
      parts.push(`The above search results have been provided to you. Answer directly using this data. Include relevant source URLs in your response.`);
    }
    
    // User file content only (rules are not files)
    if (fileContext) {
      parts.push(`\n[USER FILES]\n${fileContext}\n[END USER FILES]`);
    }

    if (toolPolicy) {
      parts.push(`\n[TOOL POLICY]\n${toolPolicy}\n[END TOOL POLICY]`);
    }
    
    // Memory
    if (includeMemory && persona.memory.summary) {
      parts.push(`\nMemory: ${persona.memory.summary}`);
    }
    
    // Image note
    if (hasImages) {
      parts.push("\nImages provided above.");
    }
    
    // Capability explanation (only if asked - brief version)
    parts.push(`If asked about capabilities: Explain you run on Mimic AI, a multi-modal system combining local LLMs with web search, file reading, vision, and voice synthesis.`);
    
    // Safety constraint only
    parts.push(`No hate speech, violence, or bigotry.`);
    
    return parts.join("\n");
  }

  // Clean response by removing stage directions (text between asterisks or brackets)
  cleanResponse(response: string): string {
    // Remove text between asterisks (stage directions like *smiles*, *looks around*)
    let cleaned = response.replace(/\*[^*]+\*/g, '');
    
    // Remove text between brackets (stage directions like [smiles], [pauses, looks around])
    cleaned = cleaned.replace(/\[[^\]]+\]/g, '');
    
    // Remove multiple consecutive newlines
    cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
    
    // Trim whitespace
    cleaned = cleaned.trim();
    
    // If cleaning removed everything, return original (fallback)
    if (!cleaned) {
      return response.trim();
    }
    
    return cleaned;
  }
}

export const ollamaService = new OllamaService();
