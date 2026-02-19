/**
 * Web Search Service
 * Provides real-time web search using multiple providers with fallback
 * 
 * Supported providers (priority order):
 * 1. Tavily AI Search - Optimized for AI apps (requires TAVILY_API_KEY)
 * 2. Brave Search - Privacy-focused (requires BRAVE_API_KEY)  
 * 3. SerpAPI - Google results (requires SERPAPI_KEY)
 * 4. Wikipedia - Factual queries
 * 5. DuckDuckGo - Fallback (no API key needed)
 */

export interface SearchSource {
  title: string;
  url: string;
}

export interface SearchResponse {
  answer: string;
  source: string;
  query: string;
  sources: SearchSource[];
  success: boolean;
}

export interface SearchRequest {
  query: string;
  includeSources?: boolean;
}

class WebSearchService {
  private enabled: boolean = false;
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

  private getUrl(): string {
    if (this.useProxy) {
      return '/api/search';
    }
    return `${this.backendUrl}/api/search`;
  }

  /**
   * Search the web with multiple provider fallback
   */
  async search(request: SearchRequest): Promise<SearchResponse> {
    if (!this.enabled) {
      throw new Error('Web search is disabled');
    }

    try {
      const url = this.getUrl();
      console.log('[WebSearch] Searching at:', url);
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: request.query,
          include_sources: request.includeSources ?? true
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Search failed: ${response.status} - ${errorText}`);
      }

      const data: SearchResponse = await response.json();
      
      console.log('[WebSearch] Result from:', data.source);
      console.log('[WebSearch] Sources:', data.sources);
      
      return data;
    } catch (error) {
      console.error('[WebSearch] Error:', error);
      throw error;
    }
  }

  /**
   * Format search answer for inclusion in AI context
   */
  formatAnswerForContext(response: SearchResponse): string {
    if (!response.success) {
      return '';
    }

    let context = `Search Query: "${response.query}"\n`;
    
    if (response.answer) {
      context += `Answer: ${response.answer}\n`;
    }
    
    // Include source URLs if available
    if (response.sources && response.sources.length > 0) {
      context += '\nSources:\n';
      response.sources.forEach((source, i) => {
        context += `${i + 1}. ${source.title}${source.url ? ` - ${source.url}` : ''}\n`;
      });
    } else {
      context += '\nNo specific sources available.';
    }
    
    return context;
  }

  /**
   * Detect if a query likely needs current information
   */
  needsCurrentInfo(query: string): boolean {
    const timeSensitiveKeywords = [
      // Time-based
      'today', 'now', 'current', 'latest', 'recent', 'news', 'update',
      'this year', 'this month', 'this week', 'yesterday', 'tomorrow',
      '2024', '2025', '2026', '2027',
      
      // Real-time data
      'weather', 'price', 'stock', 'market', 'crypto', 'bitcoin',
      'score', 'live', 'result', 'election',
      
      // Location-based current info
      'restaurant', 'restaurants', 'food near me', 'open now', 'hours',
      'nearby', 'closest', 'best', 'top rated',
      
      // Questions about current state
      'who won', 'who is the', 'what happened', 'when is', 'where is',
      'how much does', 'what time', 'is it open',
      
      // Research queries
      'find me', 'look up', 'search for', 'browse', 'internet',
      'website', 'link', 'official', 'contact'
    ];
    
    const lowerQuery = query.toLowerCase();
    return timeSensitiveKeywords.some(keyword => lowerQuery.includes(keyword));
  }

  /**
   * Check which search providers are configured
   */
  async checkProviders(): Promise<{ 
    available: boolean; 
    providers: string[];
    recommended: string;
  }> {
    try {
      const response = await fetch('/api/search/status');
      if (response.ok) {
        return await response.json();
      }
    } catch {
      // Status endpoint not available, check via search
    }
    
    // Fallback: try a test search
    try {
      await this.search({ query: 'test' });
      return { 
        available: true, 
        providers: ['Unknown'], 
        recommended: 'Configure API keys for better results' 
      };
    } catch {
      return { 
        available: false, 
        providers: [], 
        recommended: 'Install requests library and configure API keys' 
      };
    }
  }
}

export const webSearchService = new WebSearchService();
