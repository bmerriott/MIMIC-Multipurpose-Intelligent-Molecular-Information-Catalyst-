import type { Persona, PersonaMemory, MemoryEntry, ChatMessage } from "@/types";
import { ollamaService } from "./ollama";

export class MemoryService {
  private summarizeThreshold: number;
  private importanceThreshold: number;

  constructor(summarizeThreshold: number = 20, importanceThreshold: number = 0.5) {
    this.summarizeThreshold = summarizeThreshold;
    this.importanceThreshold = importanceThreshold;
  }

  setThreshold(threshold: number) {
    this.summarizeThreshold = threshold;
  }

  setImportanceThreshold(threshold: number) {
    this.importanceThreshold = threshold;
  }

  async addMessage(
    persona: Persona,
    message: ChatMessage,
    model: string
  ): Promise<PersonaMemory> {
    const importance = this.calculateImportance(message);
    
    // Skip storing if below importance threshold (unless it's the first message or a question)
    if (importance < this.importanceThreshold && persona.memory.short_term.length > 0 && !message.content.includes("?")) {
      return persona.memory;
    }

    const newEntry: MemoryEntry = {
      id: Date.now().toString(),
      content: `${message.role}: ${message.content}`,
      timestamp: new Date().toISOString(),
      importance,
    };

    const updatedMemory: PersonaMemory = {
      ...persona.memory,
      short_term: [...persona.memory.short_term, newEntry],
    };

    // Check if we need to summarize
    if (updatedMemory.short_term.length >= this.summarizeThreshold) {
      return await this.summarize(updatedMemory, persona, model);
    }

    return updatedMemory;
  }

  private calculateImportance(message: ChatMessage): number {
    // Simple importance calculation based on content length and keywords
    let importance = 0.5;
    const content = message.content.toLowerCase();
    
    // Personal information indicators increase importance
    const personalKeywords = ["like", "love", "hate", "prefer", "name", "live", "work", "job", "family", "pet"];
    if (personalKeywords.some(kw => content.includes(kw))) {
      importance += 0.2;
    }

    // Questions might be important context
    if (content.includes("?")) {
      importance += 0.1;
    }

    // Longer messages might have more context
    if (message.content.length > 100) {
      importance += 0.1;
    }

    return Math.min(importance, 1.0);
  }

  async summarize(
    memory: PersonaMemory,
    persona: Persona,
    model: string
  ): Promise<PersonaMemory> {
    try {
      const result = await ollamaService.summarizeMemory(model, memory, persona);

      // Convert important memories to entries
      const newLongTermEntries: MemoryEntry[] = result.importantMemories.map((content, index) => ({
        id: `lt-${Date.now()}-${index}`,
        content,
        timestamp: new Date().toISOString(),
        importance: 0.9, // High importance for summarized memories
      }));

      // Keep only recent short-term memories (last 5)
      const recentShortTerm = memory.short_term.slice(-5);

      return {
        short_term: recentShortTerm,
        long_term: [...memory.long_term, ...newLongTermEntries].slice(-20), // Keep last 20 long-term memories
        summary: result.summary,
        last_summarized: new Date().toISOString(),
      };
    } catch (error) {
      console.error("Memory summarization failed:", error);
      return memory;
    }
  }

  buildContextPrompt(memory: PersonaMemory): string {
    const parts: string[] = [];

    if (memory.summary) {
      parts.push(`Previous context: ${memory.summary}`);
    }

    if (memory.long_term.length > 0) {
      parts.push("Important memories:");
      memory.long_term.forEach((entry) => {
        parts.push(`- ${entry.content}`);
      });
    }

    if (memory.short_term.length > 0) {
      parts.push("Recent conversation:");
      memory.short_term.slice(-5).forEach((entry) => {
        parts.push(`- ${entry.content}`);
      });
    }

    return parts.join("\n");
  }

  // Get relevant memories for a query
  getRelevantMemories(memory: PersonaMemory, query: string, limit: number = 5): MemoryEntry[] {
    const queryWords = query.toLowerCase().split(" ");
    
    const allMemories = [...memory.long_term, ...memory.short_term];
    
    // Score each memory based on word overlap
    const scored = allMemories.map((entry) => {
      const entryWords = entry.content.toLowerCase().split(" ");
      const overlap = queryWords.filter(w => entryWords.includes(w)).length;
      return { entry, score: overlap * entry.importance };
    });

    // Sort by score and return top memories
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit).map(s => s.entry);
  }

  clearMemory(_personaId: string): PersonaMemory {
    return {
      short_term: [],
      long_term: [],
      summary: "",
      last_summarized: new Date().toISOString(),
    };
  }
}

export const memoryService = new MemoryService();
