/**
 * SearXNG Local Search Service
 * Self-hosted search engine aggregator for privacy-focused web search
 * Routes all requests through backend proxy to avoid CORS issues
 */

export interface SearXNGResult {
  title: string;
  url: string;
  content: string;
  engine: string;
  score?: number;
}

export interface SearXNGResponse {
  query: string;
  results: SearXNGResult[];
  answers: string[];
  suggestions: string[];
  infobox?: {
    title: string;
    content: string;
    url?: string;
  };
}

export interface SearchContext {
  query: string;
  results: string;
  sources: string[];
}

class SearXNGService {
  private enabled: boolean = false;
  private available: boolean = false;
  private backendUrl: string = "http://localhost:8000";
  private useProxy: boolean;

  constructor() {
    // Only use proxy in Vite development server
    // In Tauri production, use direct connection (no CORS issues in WebView)
    const isDevServer = window.location.hostname === 'localhost' && window.location.port === '5173';
    this.useProxy = isDevServer;
  }

  setEnabled(enabled: boolean) {
    this.enabled = enabled;
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  isAvailable(): boolean {
    return this.available;
  }

  private getUrl(endpoint: string): string {
    if (this.useProxy) {
      return `/api/search${endpoint}`;
    }
    // In production (Tauri), always use Python backend proxy for SearXNG
    // The backend handles the SearXNG communication and CORS
    return `${this.backendUrl}/api/search${endpoint}`;
  }

  /**
   * Check if SearXNG is running (via backend proxy or direct)
   */
  async checkStatus(): Promise<boolean> {
    // First try direct connection to SearXNG root page
    try {
      console.log('[SearXNG] Trying direct connection to localhost:8080/');
      // SearXNG root returns 200 when ready (not /health)
      const response = await fetch('http://localhost:8080/', {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      if (response.ok || response.status === 200) {
        console.log('[SearXNG] Direct connection succeeded, status:', response.status);
        this.available = true;
        return true;
      }
    } catch (e: any) {
      console.log('[SearXNG] Direct connection failed:', e.message || e);
    }
    
    // Try with no-cors mode as fallback
    try {
      console.log('[SearXNG] Trying no-cors mode...');
      await fetch('http://localhost:8080/', {
        method: 'GET',
        mode: 'no-cors',
        signal: AbortSignal.timeout(5000)
      });
      console.log('[SearXNG] No-cors connection succeeded');
      this.available = true;
      return true;
    } catch (e: any) {
      console.log('[SearXNG] No-cors connection failed:', e.message || e);
    }
    
    // Try with regular CORS mode as fallback
    try {
      console.log('[SearXNG] Trying with CORS mode...');
      const corsResponse = await fetch('http://localhost:8080/', {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      if (corsResponse.ok || corsResponse.status === 200) {
        console.log('[SearXNG] CORS connection successful, status:', corsResponse.status);
        this.available = true;
        return true;
      } else {
        console.log(`[SearXNG] CORS connection returned status: ${corsResponse.status}`);
      }
    } catch (e: any) {
      console.log('[SearXNG] CORS connection failed:', e.message || e);
    }
    
    // Fall back to backend proxy as last resort
    try {
      const url = this.getUrl('/status');
      console.log('[SearXNG] Checking status via backend proxy at:', url);
      const response = await fetch(url, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      if (response.ok) {
        const data = await response.json();
        console.log('[SearXNG] Backend proxy response:', data);
        this.available = data.available === true;
        return this.available;
      }
      console.log('[SearXNG] Backend proxy returned status:', response.status);
      this.available = false;
      return false;
    } catch (error: any) {
      console.log('[SearXNG] Backend proxy check failed:', error.message || error);
      this.available = false;
      return false;
    }
  }

  /**
   * Check SearXNG status with retries (for auto-detection)
   */
  async checkStatusWithRetries(maxRetries: number = 10, delayMs: number = 2000): Promise<boolean> {
    for (let i = 0; i < maxRetries; i++) {
      console.log(`[SearXNG] Check attempt ${i + 1}/${maxRetries}`);
      const isReady = await this.checkStatus();
      if (isReady) {
        console.log('[SearXNG] Ready!');
        return true;
      }
      if (i < maxRetries - 1) {
        console.log(`[SearXNG] Not ready, waiting ${delayMs}ms...`);
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
    return false;
  }

  /**
   * Search using SearXNG via backend proxy or direct
   */
  async search(request: { query: string }): Promise<SearchContext> {
    if (!this.enabled) {
      throw new Error('SearXNG search is disabled');
    }

    if (!this.available) {
      const isRunning = await this.checkStatus();
      if (!isRunning) {
        throw new Error('SearXNG is not running. Please start the service.');
      }
    }

    // Try direct SearXNG first (no CORS in Tauri WebView)
    try {
      const directUrl = `http://localhost:8080/search?q=${encodeURIComponent(request.query)}&format=json`;
      console.log('[SearXNG] Trying direct search:', directUrl);
      const directResponse = await fetch(directUrl, {
        method: 'GET',
        signal: AbortSignal.timeout(15000)
      });
      
      if (directResponse.ok) {
        const data = await directResponse.json();
        console.log('[SearXNG] Direct search successful');
        return this.formatResults(request.query, data);
      }
    } catch (e) {
      console.log('[SearXNG] Direct search failed, trying backend proxy...');
    }

    // Fall back to backend proxy
    try {
      const url = this.getUrl('');
      console.log('[SearXNG] Searching at:', url);
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: request.query }),
        signal: AbortSignal.timeout(15000)
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Search failed: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log('[SearXNG] Search results:', data);
      
      return this.formatResults(request.query, data);
    } catch (error) {
      console.error('[SearXNG] Search failed:', error);
      throw error;
    }
  }

  /**
   * Format SearXNG results for AI context - CONCISE to save tokens
   */
  private formatResults(query: string, data: any): SearchContext {
    const sources: string[] = [];
    let formattedResults = '';

    // Add search results concisely
    if (data.results && data.results.length > 0) {
      data.results.slice(0, 5).forEach((result: any, i: number) => {
        formattedResults += `${i + 1}. ${result.title}\n   URL: ${result.url}\n   ${result.content}\n\n`;
        sources.push(result.url);
      });
    } else {
      formattedResults = 'Search returned no results. Inform the user that no relevant information was found online.';
    }

    return {
      query,
      results: formattedResults,
      sources
    };
  }

  /**
   * Format for inclusion in AI system prompt
   */
  formatForPrompt(context: SearchContext): string {
    if (!context.results) {
      return '';
    }
    return `Query: "${context.query}"\n${context.results}`;
  }

  /**
   * Detect if query needs current web information
   */
  needsCurrentInfo(query: string): boolean {
    const timeSensitiveKeywords = [
      // Time-based
      'today', 'now', 'current', 'latest', 'recent', 'news', 'update',
      'this year', 'this month', 'this week', 'yesterday', 'tomorrow',
      '2024', '2025', '2026',
      
      // Real-time data
      'weather', 'price', 'stock', 'market', 'crypto',
      'score', 'live', 'result', 'election',
      
      // Location-based
      'restaurant', 'restaurants', 'open now', 'hours', 'nearby',
      'closest', 'best', 'top rated',
      
      // Current events
      'who won', 'who is the', 'what happened', 'when is', 'where is',
      'how much does', 'what time', 'is it open',
      
      // Research
      'find me', 'look up', 'search for', 'browse'
    ];
    
    const lowerQuery = query.toLowerCase();
    return timeSensitiveKeywords.some(keyword => lowerQuery.includes(keyword));
  }
}

export const searxngService = new SearXNGService();
