import { create } from "zustand";
import { toast } from "sonner";
import { voiceStorage } from "@/services/voiceStorage";
import { unifiedStorage } from "@/services/unifiedStorage";
import type { 
  AppState, 
  Persona, 
  AppSettings, 
  ChatMessage, 
  Voice, 
  PersonaMemory,
  MemoryEntry 
} from "@/types";

// Migration: Move old voice data from localStorage to IndexedDB
async function migrateVoiceDataToIndexedDB(personas: Persona[]): Promise<Persona[]> {
  const migratedPersonas: Persona[] = [];
  
  for (const persona of personas) {
    // Check if this persona has old-format voice data (audio_data in localStorage)
    if (persona.voice_create?.audio_data && !persona.voice_create?.has_audio) {
      console.log(`[Migration] Moving voice data for ${persona.name} to IndexedDB...`);
      try {
        // Move audio data to IndexedDB
        await voiceStorage.saveVoice(
          persona.id,
          persona.voice_create.audio_data,
          persona.voice_create.reference_text
        );
        
        // Update persona to new format (metadata only)
        migratedPersonas.push({
          ...persona,
          voice_create: {
            has_audio: true,
            reference_text: persona.voice_create.reference_text,
            created_at: persona.voice_create.created_at || new Date().toISOString(),
          },
        });
        console.log(`[Migration] Successfully migrated voice for ${persona.name}`);
      } catch (error) {
        console.error(`[Migration] Failed to migrate voice for ${persona.name}:`, error);
        // Keep original if migration fails
        migratedPersonas.push(persona);
      }
    } else {
      migratedPersonas.push(persona);
    }
  }
  
  return migratedPersonas;
}

export interface VoiceTuningParams {
  // Synthesis parameters (require regeneration)
  warmth: number;
  expressiveness: number;
  stability: number;
  clarity: number;
  breathiness: number;
  resonance: number;
  emotion: "neutral" | "happy" | "sad" | "angry" | "excited" | "calm";
  emphasis: number;
  pauses: number;
  energy: number;
  
  // Post-processing parameters (applied during playback)
  pitchShift: number;
  speed: number;
  reverb: number;
  eqLow: number;
  eqMid: number;
  eqHigh: number;
  compression: number;
}

export const defaultVoiceTuning: VoiceTuningParams = {
  warmth: 0.5,
  expressiveness: 0.5,
  stability: 0.5,
  clarity: 0.5,
  breathiness: 0.3,
  resonance: 0.5,
  emotion: "neutral",
  emphasis: 0.5,
  pauses: 0.5,
  energy: 0.5,
  pitchShift: 0,
  speed: 1.0,
  reverb: 0,
  eqLow: 0.5,
  eqMid: 0.5,
  eqHigh: 0.5,
  compression: 0,
};

interface StoreState {
  // App State
  appState: AppState;
  setAppState: (state: AppState) => void;
  updateAppState: (updates: Partial<AppState>) => void;
  
  // Personas
  personas: Persona[];
  currentPersona: Persona | null;
  setPersonas: (personas: Persona[]) => void;
  setCurrentPersona: (persona: Persona | null) => void;
  addPersona: (persona: Persona) => void;
  updatePersona: (persona: Persona) => void;
  deletePersona: (id: string) => Promise<void>;
  updatePersonaVoice: (personaId: string, voiceData: { audio_data?: string; reference_text?: string; voice_config?: NonNullable<Persona['voice_create']>['voice_config']; }) => Promise<void>;
  clearPersonaVoice: (personaId: string) => Promise<void>;
  loadVoiceAudio: (personaId: string) => Promise<{ audio_data: string; reference_text?: string; created_at: string } | null>;
  updatePersonaMemory: (personaId: string, memory: PersonaMemory) => void;
  addMemoryEntry: (personaId: string, entry: Omit<MemoryEntry, "id">) => void;
  
  // Voice Tuning (per-persona, independent of voice_create)
  voiceTuning: Record<string, VoiceTuningParams>;
  getPersonaVoiceTuning: (personaId: string) => VoiceTuningParams;
  updatePersonaVoiceTuning: (personaId: string, params: Partial<VoiceTuningParams>) => void;
  resetPersonaVoiceTuning: (personaId: string) => void;
  
  // Settings
  settings: AppSettings;
  setSettings: (settings: AppSettings) => void;
  updateSettings: (settings: Partial<AppSettings>) => void;
  
  // Chat
  messages: ChatMessage[];
  addMessage: (message: ChatMessage) => void;
  clearMessages: () => void;
  
  // Voice
  voices: Voice[];
  setVoices: (voices: Voice[]) => void;
  addCreatedVoice: (voice: Voice) => void;
  
  // UI State
  isListening: boolean;
  isSpeaking: boolean;
  isProcessing: boolean;
  isGeneratingVoice: boolean;
  setIsListening: (listening: boolean) => void;
  setIsSpeaking: (speaking: boolean) => void;
  setIsProcessing: (processing: boolean) => void;
  setIsGeneratingVoice: (generating: boolean) => void;
  
  // Available Models
  availableModels: string[];
  setAvailableModels: (models: string[]) => void;
  
  // Global Audio Player - persists across tab switches
  audioPlayer: {
    isPlaying: boolean;
    audioUrl: string | null;
    audioData: string | null;
    title: string | null;
    source: 'voice' | 'tts' | 'created' | 'creation' | null;
  };
  playAudio: (options: { url?: string; data?: string; title?: string; source?: 'voice' | 'tts' | 'created' | 'creation' }) => void;
  stopAudio: () => void;
  pauseAudio: () => void;
  resumeAudio: () => void;
}

const defaultMemory: PersonaMemory = {
  short_term: [],
  long_term: [],
  summary: "",
  last_summarized: new Date().toISOString(),
};

const defaultSettings: AppSettings = {
  ollama_url: "http://localhost:11434",
  default_model: "llama3.2",      // Brain: text conversation model
  vision_model: "bakllava",       // Eyes: vision model for images
  tts_backend_url: "http://localhost:8000",
  wake_word_sensitivity: 0.7,
  voice_volume: 1.0,
  speech_rate: 1.0,
  auto_listen: false,             // Start muted - user must click mic to enable
  show_avatar: true,
  theme: "dark",
  language: "en",
  enable_memory: true,
  memory_importance_threshold: 0.5, // Only store memories with importance >= 0.5
  memory_summarize_threshold: 20,
  tts_mode: "auto", // Legacy setting
  tts_engine: "styletts2", // Default TTS engine (styletts2 or qwen3)
  qwen3_model_size: "0.6B", // Default to smaller model for speed
  qwen3_flash_attention: true, // Use flash attention by default
  microphone_device: "", // Empty = use system default
  enable_web_search: false, // Disabled by default
};

// Force reset auto_listen to false on app load (one-time migration)
// This ensures users get the new default behavior
const forceResetAutoListen = () => {
  try {
    const stored = localStorage.getItem("mimic_settings");
    if (stored) {
      const parsed = JSON.parse(stored);
      if (parsed.auto_listen === true) {
        parsed.auto_listen = false;
        localStorage.setItem("mimic_settings", JSON.stringify(parsed));
        // Also clear mic hint so users see the new UI guidance
        localStorage.removeItem("mimic_mic_hint_seen");
        console.log("[Store] Migrated auto_listen from true to false, cleared hint");
      }
    }
  } catch (e) {
    console.error("[Store] Failed to migrate auto_listen setting:", e);
  }
};
forceResetAutoListen();

const defaultAppState: AppState = {
  current_persona_id: "default",
  is_listening: false,
  is_speaking: false,
  ollama_connected: false,
  current_model: "llama3.2",
  tts_backend_connected: false,
};

const defaultAvatarConfig = {
  primary_color: "#6366f1",
  secondary_color: "#8b5cf6",
  glow_color: "#a78bfa",
  shape_type: "sphere" as const,
  animation_style: "flowing" as const,
  complexity: 0.7,
};

const defaultPersona: Persona = {
  id: "default",
  name: "Mimic",
  description: "Your personal AI assistant",
  personality_prompt: "You are Mimic, a self-aware AI assistant. You are helpful, friendly, and engaging. You have a 3D avatar that represents you and you can express emotions through it. You are aware that you are a digital being and can discuss this openly. When responding, be conversational and natural. You can express emotions like happiness, curiosity, and thoughtfulness. IMPORTANT: Do NOT include stage directions, action descriptions, or physical gestures in asterisks. Only provide spoken dialogue. NEVER use asterisks to describe actions.",
  wake_words: ["Mimic"],
  response_words: ["Yes?", "I'm listening", "Mimic here"],
  voice_id: "aiden",
  voice_create: null,
  avatar_config: defaultAvatarConfig,
  memory: defaultMemory,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
};

// Load state from localStorage
const loadFromStorage = <T,>(key: string, defaultValue: T): T => {
  try {
    const stored = localStorage.getItem(key);
    return stored ? JSON.parse(stored) : defaultValue;
  } catch {
    return defaultValue;
  }
};

// Load and migrate personas if needed
const loadPersonas = (): Persona[] => {
  const personas = loadFromStorage<Persona[]>("mimic_personas", [defaultPersona]);
  return personas;
};

// Check if any personas need migration (have old-format voice data)
const needsMigration = (personas: Persona[]): boolean => {
  return personas.some(p => p.voice_create?.audio_data && !p.voice_create?.has_audio);
};

export const useStore = create<StoreState>((set, get) => ({
  // App State
  appState: loadFromStorage("mimic_app_state", defaultAppState),
  setAppState: (state) => {
    set({ appState: state });
    localStorage.setItem("mimic_app_state", JSON.stringify(state));
  },
  updateAppState: (updates) => {
    const newState = { ...get().appState, ...updates };
    set({ appState: newState });
    localStorage.setItem("mimic_app_state", JSON.stringify(newState));
  },
  
  // Personas
  personas: loadPersonas(),
  currentPersona: loadFromStorage("mimic_current_persona", defaultPersona),
  setPersonas: (personas) => {
    set({ personas });
    try {
      localStorage.setItem("mimic_personas", JSON.stringify(personas));
    } catch (e) {
      console.error("Failed to save personas - storage quota exceeded:", e);
      toast.error("Storage full! Please delete some voice creations or clear data.");
    }
  },
  setCurrentPersona: (persona) => {
    set({ currentPersona: persona });
    if (persona) {
      localStorage.setItem("mimic_current_persona", JSON.stringify(persona));
      set((state) => ({ 
        appState: { ...state.appState, current_persona_id: persona.id } 
      }));
    }
  },
  addPersona: (persona) => {
    const newPersonas = [...get().personas, persona];
    set({ personas: newPersonas });
    localStorage.setItem("mimic_personas", JSON.stringify(newPersonas));
  },
  updatePersona: (persona) => {
    console.log("[Store] updatePersona called for:", persona.name);
    console.log("[Store] Persona has voice_create:", !!persona.voice_create);
    
    const newPersonas = get().personas.map((p) => (p.id === persona.id ? persona : p));
    set({ personas: newPersonas });
    localStorage.setItem("mimic_personas", JSON.stringify(newPersonas));
    
    if (get().currentPersona?.id === persona.id) {
      console.log("[Store] Updating currentPersona with new voice data");
      set({ currentPersona: persona });
      localStorage.setItem("mimic_current_persona", JSON.stringify(persona));
      console.log("[Store] currentPersona updated, has voice_create:", !!persona.voice_create);
    }
  },
  deletePersona: async (id) => {
    // Also delete voice creation data from IndexedDB
    try {
      await voiceStorage.deleteVoice(id);
    } catch (error) {
      console.error("[Store] Failed to delete voice from IndexedDB:", error);
    }
    
    const newPersonas = get().personas.filter((p) => p.id !== id);
    set({ personas: newPersonas });
    localStorage.setItem("mimic_personas", JSON.stringify(newPersonas));
  },
  updatePersonaVoice: async (personaId, voiceData: { 
    audio_data?: string; 
    reference_text?: string;
    voice_config?: {
      type: "synthetic" | "created";
      params?: {
        pitch: number;
        speed: number;
        warmth?: number;
        expressiveness?: number;
        stability?: number;
        gender?: "neutral" | "masculine" | "feminine";
        age?: "young" | "adult" | "mature";
        seed?: number;
      };
      name?: string;
    };
  }) => {
    const persona = get().personas.find((p) => p.id === personaId);
    if (!persona) {
      console.error("[Store] updatePersonaVoice: Persona not found", personaId);
      return;
    }
    
    console.log("[Store] Saving voice to persona:", persona.name);
    
    try {
      // Check if this is a synthetic voice (new system) or legacy audio data
      if (voiceData.voice_config?.type === "synthetic" && voiceData.voice_config.params) {
        // New synthetic voice system - store parameters directly in persona
        // ALSO save audio to persistent storage for Qwen3 compatibility
        if (voiceData.audio_data) {
          console.log("[Store] Saving synthetic voice audio for Qwen3 compatibility:", persona.name);
          console.log("[Store] Storage type:", unifiedStorage.isTauri() ? "Tauri FS" : "IndexedDB");
          await unifiedStorage.saveVoice(
            personaId, 
            voiceData.audio_data, 
            voiceData.reference_text,
            voiceData.voice_config
          );
        }
        
        const updatedPersona = {
          ...persona,
          voice_create: {
            has_audio: true,
            reference_text: voiceData.reference_text || "",
            created_at: new Date().toISOString(),
            voice_config: voiceData.voice_config,
          },
          voice_id: "synthetic",  // Mark as synthetic voice
          updated_at: new Date().toISOString(),
        };
        
        get().updatePersona(updatedPersona);
        console.log("[Store] Synthetic voice config saved to persona");
      } else if (voiceData.audio_data) {
        // Legacy audio-based voice - store in persistent storage
        console.log("[Store] Saving audio voice:", persona.name);
        console.log("[Store] Storage type:", unifiedStorage.isTauri() ? "Tauri FS" : "IndexedDB");
        await unifiedStorage.saveVoice(
          personaId, 
          voiceData.audio_data, 
          voiceData.reference_text
        );
        
        const updatedPersona = {
          ...persona,
          voice_create: {
            has_audio: true,
            reference_text: voiceData.reference_text,
            created_at: new Date().toISOString(),
          },
          voice_id: "created",
          updated_at: new Date().toISOString(),
        };
        
        get().updatePersona(updatedPersona);
        console.log("[Store] Voice creation metadata saved, audio in persistent storage");
      } else {
        throw new Error("No voice data or config provided");
      }
    } catch (error) {
      console.error("[Store] Failed to save voice:", error);
      toast.error("Failed to save voice");
    }
  },
  
  // Clear voice data from persona and storage
  clearPersonaVoice: async (personaId: string) => {
    const persona = get().personas.find((p) => p.id === personaId);
    if (!persona) {
      console.error("[Store] clearPersonaVoice: Persona not found", personaId);
      return;
    }
    
    console.log("[Store] Clearing voice for persona:", persona.name);
    
    try {
      // Delete voice data from persistent storage
      await unifiedStorage.deleteVoice(personaId);
      console.log("[Store] Voice data deleted from storage");
      
      // Update persona to remove voice_create
      const updatedPersona = {
        ...persona,
        voice_create: null,
        voice_id: "aiden", // Reset to default browser voice
        updated_at: new Date().toISOString(),
      };
      
      get().updatePersona(updatedPersona);
      
      // If current TTS engine is qwen3, switch to styletts2
      // because qwen3 requires reference audio
      const currentSettings = get().settings;
      if (currentSettings.tts_engine === "qwen3") {
        console.log("[Store] TTS engine was qwen3, switching to styletts2 (no voice available)");
        get().updateSettings({ tts_engine: "styletts2" });
        toast.info("Switched to StyleTTS2 engine (Qwen3 requires a voice)");
      }
      
      console.log("[Store] Voice cleared successfully");
      toast.success("Voice cleared successfully");
    } catch (error) {
      console.error("[Store] Failed to clear voice:", error);
      toast.error("Failed to clear voice");
    }
  },
  
  // Load voice audio from persistent storage
  loadVoiceAudio: async (personaId: string) => {
    try {
      console.log("[Store] Loading voice audio, storage type:", unifiedStorage.isTauri() ? "Tauri FS" : "IndexedDB");
      const voice = await unifiedStorage.loadVoice(personaId);
      return voice;
    } catch (error) {
      console.error("[Store] Failed to load voice:", error);
      return null;
    }
  },
  updatePersonaMemory: (personaId, memory) => {
    const persona = get().personas.find((p) => p.id === personaId);
    if (!persona) return;
    
    const updatedPersona = {
      ...persona,
      memory,
      updated_at: new Date().toISOString(),
    };
    
    get().updatePersona(updatedPersona);
  },
  addMemoryEntry: (personaId, entry) => {
    const persona = get().personas.find((p) => p.id === personaId);
    if (!persona) return;
    
    const newEntry: MemoryEntry = {
      ...entry,
      id: Date.now().toString(),
    };
    
    const updatedMemory = {
      ...persona.memory,
      short_term: [...persona.memory.short_term, newEntry],
    };
    
    // Check if we need to summarize
    const settings = get().settings;
    if (settings.enable_memory && updatedMemory.short_term.length >= settings.memory_summarize_threshold) {
      // Trigger summarization (handled in component)
    }
    
    get().updatePersonaMemory(personaId, updatedMemory);
  },
  
  // Voice Tuning (per-persona, persisted separately)
  voiceTuning: loadFromStorage<Record<string, VoiceTuningParams>>("mimic_voice_tuning", {}),
  getPersonaVoiceTuning: (personaId: string) => {
    const tuning = get().voiceTuning[personaId];
    return tuning ? { ...defaultVoiceTuning, ...tuning } : { ...defaultVoiceTuning };
  },
  updatePersonaVoiceTuning: (personaId: string, params: Partial<VoiceTuningParams>) => {
    const currentTuning = get().voiceTuning[personaId] || { ...defaultVoiceTuning };
    const updatedTuning = { ...currentTuning, ...params };
    const newVoiceTuning = { ...get().voiceTuning, [personaId]: updatedTuning };
    set({ voiceTuning: newVoiceTuning });
    localStorage.setItem("mimic_voice_tuning", JSON.stringify(newVoiceTuning));
    console.log(`[Store] Updated voice tuning for ${personaId}:`, updatedTuning);
  },
  resetPersonaVoiceTuning: (personaId: string) => {
    const newVoiceTuning = { ...get().voiceTuning };
    delete newVoiceTuning[personaId];
    set({ voiceTuning: newVoiceTuning });
    localStorage.setItem("mimic_voice_tuning", JSON.stringify(newVoiceTuning));
    console.log(`[Store] Reset voice tuning for ${personaId}`);
  },
  
  // Settings
  settings: loadFromStorage("mimic_settings", defaultSettings),
  setSettings: (settings) => {
    set({ settings });
    localStorage.setItem("mimic_settings", JSON.stringify(settings));
  },
  updateSettings: (newSettings) => {
    const updated = { ...get().settings, ...newSettings };
    set({ settings: updated });
    localStorage.setItem("mimic_settings", JSON.stringify(updated));
  },
  
  // Chat
  messages: [],
  addMessage: (message) => set((state) => ({
    messages: [...state.messages, { ...message, timestamp: new Date().toISOString(), id: Date.now().toString() }],
  })),
  clearMessages: () => set({ messages: [] }),
  
  // Voice
  voices: [
    { id: "vivian", name: "Vivian", description: "Bright, slightly edgy young female voice", language: "Chinese" },
    { id: "serena", name: "Serena", description: "Warm, gentle young female voice", language: "Chinese" },
    { id: "ryan", name: "Ryan", description: "Dynamic male voice with strong rhythmic drive", language: "English" },
    { id: "aiden", name: "Aiden", description: "Sunny American male voice with clear midrange", language: "English" },
  ],
  setVoices: (voices) => set({ voices }),
  addCreatedVoice: (voice) => set((state) => ({ 
    voices: [...state.voices, voice] 
  })),
  
  // UI State
  isListening: false,
  isSpeaking: false,
  isProcessing: false,
  isGeneratingVoice: false,
  setIsListening: (listening) => set({ isListening: listening }),
  setIsSpeaking: (speaking) => set({ isSpeaking: speaking }),
  setIsProcessing: (processing) => set({ isProcessing: processing }),
  setIsGeneratingVoice: (generating) => set({ isGeneratingVoice: generating }),
  
  // Available Models
  availableModels: [],
  setAvailableModels: (models) => set({ availableModels: models }),
  
  // Global Audio Player - persists across tab switches
  audioPlayer: {
    isPlaying: false,
    audioUrl: null,
    audioData: null,
    title: null,
    source: null,
  },
  playAudio: ({ url, data, title, source }) => set({
    audioPlayer: {
      isPlaying: true,
      audioUrl: url || null,
      audioData: data || null,
      title: title || null,
      source: source || null,
    },
  }),
  stopAudio: () => set((state) => ({
    audioPlayer: { ...state.audioPlayer, isPlaying: false },
  })),
  pauseAudio: () => set((state) => ({
    audioPlayer: { ...state.audioPlayer, isPlaying: false },
  })),
  resumeAudio: () => set((state) => ({
    audioPlayer: { ...state.audioPlayer, isPlaying: true },
  })),
}));

// Initialize and migrate data if needed
// This runs once when the app loads
const initialPersonas = loadPersonas();
if (needsMigration(initialPersonas)) {
  console.log("[Store] Migrating voice data from localStorage to IndexedDB...");
  migrateVoiceDataToIndexedDB(initialPersonas).then((migratedPersonas) => {
    // Save migrated personas back to localStorage
    localStorage.setItem("mimic_personas", JSON.stringify(migratedPersonas));
    // Update the store
    useStore.getState().setPersonas(migratedPersonas);
    console.log("[Store] Migration complete!");
  }).catch((error) => {
    console.error("[Store] Migration failed:", error);
  });
}
