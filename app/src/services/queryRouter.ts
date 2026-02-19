/**
 * Simple Rule-Based Query Router
 * 
 * Analyzes user queries and determines which capabilities are needed.
 * This is a lightweight alternative to a dedicated router model.
 * 
 * Benefits:
 * - 0 VRAM cost (rule-based, no ML model)
 * - Fast (simple pattern matching)
 * - Deterministic (predictable behavior)
 * - Saves VRAM by only loading needed models
 */

export interface RouteDecision {
  needsVision: boolean;
  needsWebSearch: boolean;
  needsCodeModel: boolean;
  needsReasoning: boolean;
  confidence: number; // 0-1, how confident we are in this routing
  suggestedTags: string[];
}

// Keyword patterns for routing decisions
const ROUTING_RULES = {
  vision: {
    keywords: [
      // Image-related
      'image', 'picture', 'photo', 'look at', 'what do you see', 'describe this',
      'analyze this image', 'what is in this', 'show me', 'screenshot',
      // Visual analysis
      'chart', 'graph', 'diagram', 'table', 'document', 'text in image',
      'ocr', 'read this', 'what does this say',
    ],
    patterns: [
      /\b(see|look|show|image|picture|photo)\b/i,
      /\b(describe|analyze|what is in|what's in)\s+(this|the|that)\s+(image|picture|photo)\b/i,
      /\b(read|extract)\s+(the\s+)?text\b/i,
    ],
    weight: 1.0,
  },
  
  webSearch: {
    keywords: [
      // Time-sensitive
      'latest', 'recent', 'news', 'today', 'yesterday', 'this week',
      'current', 'now', 'happening', 'update', 'just announced',
      // Factual queries
      'weather', 'stock price', 'who won', 'score', 'election results',
      'when did', 'how old is', 'net worth', 'population of',
      // Research
      'search for', 'find information', 'look up', 'google',
    ],
    patterns: [
      /\b(latest|recent|current|today|news about)\b/i,
      /\b(weather in|stock price of|who won)\b/i,
      /\b(when did|how old is|what happened to)\b/i,
      /\b(search|look up|find)\s+(for\s+)?\b/i,
    ],
    weight: 0.8,
  },
  
  code: {
    keywords: [
      // Programming
      'code', 'function', 'bug', 'error', 'debug', 'fix this',
      'program', 'script', 'algorithm', 'implement',
      // Languages
      'python', 'javascript', 'typescript', 'java', 'c++', 'rust',
      'react', 'node', 'api', 'database', 'sql',
      // Development
      'compile', 'runtime error', 'syntax', 'refactor', 'optimize',
    ],
    patterns: [
      /\b(code|function|bug|debug|error|fix)\s+(this|the)\b/i,
      /\b(write|create)\s+(a\s+)?(function|script|program)\b/i,
      /\b(syntax|runtime|compile)\s+error\b/i,
      /\b(python|javascript|typescript|java|c\+\+|rust|go)\s+(code|function)\b/i,
    ],
    weight: 0.9,
  },
  
  reasoning: {
    keywords: [
      // Complex thinking
      'explain', 'why', 'how does', 'compare', 'contrast',
      'analyze', 'evaluate', 'pros and cons', 'advantages',
      'step by step', 'walk me through', 'break down',
      // Math/Logic
      'calculate', 'solve', 'equation', 'math', 'logic',
      'prove', 'deduce', 'infer',
    ],
    patterns: [
      /\b(explain|why|how does|compare|analyze)\b/i,
      /\b(step by step|walk me through|break down)\b/i,
      /\b(calculate|solve|equation|math problem)\b/i,
      /\b(pros and cons|advantages|disadvantages)\b/i,
    ],
    weight: 0.6,
  },
};

/**
 * Analyze a query and determine routing decision
 */
export function analyzeQuery(query: string, hasImages: boolean = false): RouteDecision {
  const lowerQuery = query.toLowerCase();
  
  // Score each category
  let visionScore = 0;
  let webSearchScore = 0;
  let codeScore = 0;
  let reasoningScore = 0;
  
  // Check vision (always true if images present)
  if (hasImages) {
    visionScore = 1.0;
  } else {
    visionScore = calculateScore(lowerQuery, ROUTING_RULES.vision);
  }
  
  // Check other categories
  webSearchScore = calculateScore(lowerQuery, ROUTING_RULES.webSearch);
  codeScore = calculateScore(lowerQuery, ROUTING_RULES.code);
  reasoningScore = calculateScore(lowerQuery, ROUTING_RULES.reasoning);
  
  // Determine which capabilities are needed
  const needsVision = visionScore > 0.3;
  const needsWebSearch = webSearchScore > 0.3;
  const needsCodeModel = codeScore > 0.4; // Higher threshold for code
  const needsReasoning = reasoningScore > 0.3 || query.length > 100;
  
  // Build suggested tags
  const suggestedTags: string[] = [];
  if (needsVision) suggestedTags.push('vision');
  if (needsWebSearch) suggestedTags.push('web-search');
  if (needsCodeModel) suggestedTags.push('code');
  if (needsReasoning) suggestedTags.push('reasoning');
  if (suggestedTags.length === 0) suggestedTags.push('general');
  
  // Calculate overall confidence
  const maxScore = Math.max(visionScore, webSearchScore, codeScore, reasoningScore);
  const confidence = Math.min(maxScore + 0.3, 1.0); // Boost confidence slightly
  
  return {
    needsVision,
    needsWebSearch,
    needsCodeModel,
    needsReasoning,
    confidence,
    suggestedTags,
  };
}

/**
 * Calculate score for a category based on keywords and patterns
 */
function calculateScore(query: string, rules: typeof ROUTING_RULES.vision): number {
  let score = 0;
  
  // Check keywords
  for (const keyword of rules.keywords) {
    if (query.includes(keyword.toLowerCase())) {
      score += 0.2 * rules.weight;
    }
  }
  
  // Check patterns (stronger signal)
  for (const pattern of rules.patterns) {
    if (pattern.test(query)) {
      score += 0.4 * rules.weight;
    }
  }
  
  return Math.min(score, 1.0);
}

/**
 * Get a human-readable explanation of the routing decision
 */
export function explainRouting(decision: RouteDecision): string {
  const parts: string[] = [];
  
  if (decision.needsVision) parts.push('👁️ Vision analysis');
  if (decision.needsWebSearch) parts.push('🔍 Web search');
  if (decision.needsCodeModel) parts.push('💻 Code assistance');
  if (decision.needsReasoning) parts.push('🧠 Complex reasoning');
  
  if (parts.length === 0) {
    parts.push('💬 General conversation');
  }
  
  return parts.join(' + ');
}

/**
 * Determine if a query can be handled by a lightweight model
 * Returns true if the query is simple enough for a fast model
 */
export function isSimpleQuery(query: string): boolean {
  const lowerQuery = query.toLowerCase();
  
  // Simple greeting patterns
  const greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'];
  if (greetings.some(g => lowerQuery.includes(g)) && query.length < 30) {
    return true;
  }
  
  // Simple questions
  if (query.length < 50 && !lowerQuery.includes('explain') && !lowerQuery.includes('analyze')) {
    return true;
  }
  
  return false;
}

// Export rules for debugging/testing
export { ROUTING_RULES };
