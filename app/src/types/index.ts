export interface AppState {
  current_persona_id: string;
  is_listening: boolean;
  is_speaking: boolean;
  ollama_connected: boolean;
  current_model: string;
  tts_backend_connected: boolean;
}

export interface MemoryEntry {
  id: string;
  content: string;
  timestamp: string;
  importance: number; // 0-1 scale for memory retention
}

export interface ChatImage {
  data: string; // Base64 encoded image
  mimeType: string; // image/jpeg, image/png, etc.
}

export interface PersonaMemory {
  short_term: MemoryEntry[]; // Recent conversations
  long_term: MemoryEntry[]; // Summarized important memories
  summary: string; // Current summary of the persona's understanding
  last_summarized: string;
}

export interface Persona {
  id: string;
  name: string;
  description: string;
  personality_prompt: string;
  wake_words: string[]; // Multiple wake words (e.g., ["Jarvis", "Hey Jarvis"])
  response_words: string[]; // Multiple response words (e.g., ["Yes?", "I'm here"])
  voice_id: string;
  voice_create?: {
    audio_data?: string; // Base64 encoded audio sample for preview
    has_audio?: boolean; // Flag indicating voice is configured
    reference_text?: string; // Sample text used for the voice
    created_at: string;
    voice_config?: {
      type: "synthetic" | "created"; // Type of voice creation
      params?: {
        // Basic tuning
        pitch: number;
        speed: number;
        // Voice characteristics
        warmth?: number;
        expressiveness?: number;
        stability?: number;
        clarity?: number;
        breathiness?: number;
        resonance?: number;
        // Speech characteristics
        emotion?: "neutral" | "happy" | "sad" | "angry" | "excited" | "calm";
        emphasis?: number;
        pauses?: number;
        energy?: number;
        // Audio effects
        reverb?: number;
        eq_low?: number;
        eq_mid?: number;
        eq_high?: number;
        compression?: number;
        // Engine info
        engine?: "off" | "browser" | "qwen3";
        qwen3_model_size?: "0.6B" | "1.7B";
        extraction_model_size?: "0.6B" | "1.7B";
        use_voice_profile?: boolean;
        // Legacy
        gender?: "neutral" | "masculine" | "feminine";
        age?: "young" | "adult" | "mature";
        seed?: number; // Random seed for reproducibility
      };
      name?: string;
    };
  } | null;
  avatar_config: AvatarConfig;
  avatar_description?: string; // LLM-generated avatar description
  memory: PersonaMemory;
  created_at: string;
  updated_at: string;
}

export interface AvatarConfig {
  primary_color: string;
  secondary_color: string;
  glow_color: string;
  shape_type: "sphere" | "cube" | "torus" | "icosahedron" | "llm_generated";
  animation_style: "flowing" | "pulsing" | "wave" | "static" | "llm_generated";
  complexity: number;
  llm_prompt?: string; // The prompt used to generate the avatar
}

export interface AppSettings {
  ollama_url: string;
  default_model: string; // Brain model - for conversation/text
  vision_model: string;  // Vision model - for image understanding
  tts_backend_url: string;
  wake_word_sensitivity: number;
  voice_volume: number;
  speech_rate: number;
  auto_listen: boolean;
  show_avatar: boolean;
  theme: string;
  language: string;
  enable_memory: boolean;
  memory_importance_threshold: number; // Minimum importance (0-1) for storing memories
  memory_summarize_threshold: number; // Number of messages before summarizing
  tts_mode: "browser" | "qwen3" | "auto"; // Legacy setting, now use tts_engine
  tts_engine?: "off" | "browser" | "qwen3"; // TTS engine selection (off = text only, browser = system TTS, qwen3 = AI voice)
  qwen3_model_size?: "0.6B" | "1.7B"; // Qwen3 model size (0.6B = faster, 1.7B = better quality)
  qwen3_flash_attention?: boolean; // Use flash attention for Qwen3 (reduces VRAM)
  microphone_device?: string; // Device ID for speech recognition (empty = default)
  enable_web_search: boolean; // Enable real-time web search for current information
}

export interface OllamaModel {
  name: string;
  modified_at: string;
  size: number;
  digest?: string;
  details?: {
    parent_model?: string;
    format?: string;
    family?: string;
    families?: string[];
    parameter_size?: string;
    quantization_level?: string;
  };
}

export interface OllamaGenerateRequest {
  model: string;
  prompt: string;
  system?: string;
  stream?: boolean;
  options?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    repeat_penalty?: number;
  };
}

export interface OllamaGenerateResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  context?: number[];
}

export interface OllamaChatRequest {
  model: string;
  messages: {
    role: "system" | "user" | "assistant";
    content: string;
  }[];
  stream?: boolean;
  options?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    repeat_penalty?: number;
  };
}

export interface OllamaChatResponse {
  model: string;
  created_at: string;
  message: {
    role: string;
    content: string;
  };
  done: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: string;
  id?: string;
  images?: string[]; // Base64 encoded images for vision models
}

export interface Voice {
  id: string;
  name: string;
  description: string;
  language: string;
  is_created?: boolean;
  persona_id?: string;
}

export interface TTSCreateRequest {
  audio_data: string; // Base64 encoded
  reference_text: string;
  target_text: string;
  persona_id: string;
}

export interface TTSGenerateRequest {
  text: string;
  voice_id?: string;
  voice_create_id?: string;
  persona_id?: string;
  speed?: number;
}

export interface TTSResponse {
  audio_data: string; // Base64 encoded
  duration_ms: number;
  sample_rate?: number;
  transcribed_text?: string;  // Auto-transcribed reference text (if applicable)
}

// Voice Enrollment Types
export interface VoiceEnrollRequest {
  voice_id: string;
  audio_data: string; // Base64 encoded
  reference_text?: string;
}

export interface VoiceEnrollResponse {
  voice_id: string;
  success: boolean;
  sample_rate: number;
  audio_duration_sec: number;
  has_reference_text: boolean;
  message: string;
}

export interface EnrolledVoiceInfo {
  voice_id: string;
  sample_rate: number;
  audio_duration_sec: number;
  has_reference_text: boolean;
  enrolled_at: number;
}

export interface VoiceListResponse {
  voices: EnrolledVoiceInfo[];
  count: number;
}

export interface VoiceSynthesizeRequest {
  voice_id: string;
  text: string;
  language?: string;
}

export interface AvatarGenerationRequest {
  personality_prompt: string;
  persona_name: string;
  current_config?: AvatarConfig;
}

export interface AvatarGenerationResponse {
  description: string;
  primary_color: string;
  secondary_color: string;
  glow_color: string;
  shape_type: string;
  animation_style: string;
  complexity: number;
  reasoning: string;
}
