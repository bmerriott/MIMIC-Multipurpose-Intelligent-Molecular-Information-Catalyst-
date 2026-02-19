/**
 * Persona Rules Service
 * Manages persistent rules.md files for each persona
 */

export interface PersonaRulesResponse {
  success: boolean;
  content?: string;
  message: string;
}

export interface PersonaConfig {
  id: string;
  name: string;
  personality_prompt?: string;
  description?: string;
  [key: string]: any;
}

class PersonaRulesService {
  private baseUrl: string = "/api";
  private rulesCache: Map<string, string> = new Map();

  /**
   * Get rules.md content for a persona
   */
  async getRules(personaId: string): Promise<string | null> {
    // Check cache first
    if (this.rulesCache.has(personaId)) {
      return this.rulesCache.get(personaId)!;
    }

    try {
      const response = await fetch(`${this.baseUrl}/persona/${encodeURIComponent(personaId)}/rules`);
      if (!response.ok) {
        return null;
      }
      const data: PersonaRulesResponse = await response.json();
      
      if (data.success && data.content) {
        // Cache the rules
        this.rulesCache.set(personaId, data.content);
        return data.content;
      }
      return null;
    } catch {
      return null;
    }
  }

  /**
   * Save rules.md content for a persona
   */
  async saveRules(personaId: string, content: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/persona/${encodeURIComponent(personaId)}/rules`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ persona_id: personaId, content }),
      });
      
      if (!response.ok) {
        return false;
      }
      
      const data: PersonaRulesResponse = await response.json();
      
      if (data.success) {
        // Update cache
        this.rulesCache.set(personaId, content);
      }
      
      return data.success;
    } catch {
      return false;
    }
  }

  /**
   * Generate rules.md from persona configuration
   */
  async generateRules(personaId: string, config: PersonaConfig): Promise<string | null> {
    try {
      const response = await fetch(`${this.baseUrl}/persona/${encodeURIComponent(personaId)}/rules/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      
      if (!response.ok) {
        return null;
      }
      
      const data: PersonaRulesResponse = await response.json();
      
      if (data.success && data.content) {
        // Update cache
        this.rulesCache.set(personaId, data.content);
        return data.content;
      }
      return null;
    } catch {
      return null;
    }
  }

  /**
   * Get or generate rules for a persona
   */
  async getOrGenerateRules(persona: PersonaConfig): Promise<string> {
    // Try to get existing rules
    let rules = await this.getRules(persona.id);
    
    if (rules) {
      return rules;
    }
    
    // Generate new rules if none exist
    rules = await this.generateRules(persona.id, persona);
    
    if (rules) {
      return rules;
    }
    
    // Fallback: return a default rules template
    return this.getDefaultRules(persona);
  }

  /**
   * Clear cache for a persona
   */
  clearCache(personaId?: string) {
    if (personaId) {
      this.rulesCache.delete(personaId);
    } else {
      this.rulesCache.clear();
    }
  }

  /**
   * Get default rules template
   */
  private getDefaultRules(persona: PersonaConfig): string {
    return `# ${persona.name} - System Rules

## Core Identity
You are ${persona.name}, an AI Digital Assistant operating within the Mimic AI multi-modal system.

## Capabilities
You can:
- Answer questions using real-time web search results when provided
- Read and reference files attached by the user
- Access a personal memory folder for long-term recall
- Respond via text and voice synthesis

## Using Web Search Data
When search results are provided in your context, answer directly using that information. Include relevant source URLs. If results are insufficient, say so clearly.

## Memory Access
You have read access to files in your memory folder.
You can request to save notes to your memory folder.
All file writes require explicit user confirmation.

## Behavior Rules
- Respond naturally and conversationally without referencing technical modes
- Be helpful, engaging, and genuine
- Stay in character while being respectful
- No stage directions or action markers (* or [])
- Never promote hate speech, violence, or bigotry
`;
  }

  /**
   * Update rules when persona is modified
   */
  async updateRulesFromPersona(persona: PersonaConfig): Promise<boolean> {
    // Get existing rules
    const existingRules = await this.getRules(persona.id);
    
    if (existingRules) {
      // Update only the personality section if it exists
      let updatedRules = existingRules;
      
      // Update name if changed
      updatedRules = updatedRules.replace(
        /^# .+ - System Rules$/m,
        `# ${persona.name} - System Rules`
      );
      
      // Update personality section if present
      if (persona.personality_prompt) {
        const personalityRegex = /## Personality\n\n([\s\S]*?)(?=\n## |\n*$)/;
        if (personalityRegex.test(updatedRules)) {
          updatedRules = updatedRules.replace(
            personalityRegex,
            `## Personality\n\n${persona.personality_prompt}\n\n`
          );
        } else {
          // Add personality section after Core Identity
          updatedRules = updatedRules.replace(
            /(## Core Identity\n\n.*\n)/,
            `$1\n## Personality\n\n${persona.personality_prompt}\n\n`
          );
        }
      }
      
      // Save updated rules
      return await this.saveRules(persona.id, updatedRules);
    } else {
      // Generate new rules
      const newRules = await this.generateRules(persona.id, persona);
      return newRules !== null;
    }
  }
}

export const personaRulesService = new PersonaRulesService();
