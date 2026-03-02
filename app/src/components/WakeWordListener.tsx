import { useEffect, useRef, useCallback } from "react";
import { useStore } from "@/store";
import { toast } from "sonner";
import { ollamaService } from "@/services/ollama";
import { ttsService, type Qwen3ModelSize } from "@/services/tts";
import { searxngService } from "@/services/searxng";
import { memoryService } from "@/services/memory";
import { memoryToolsService, type ToolCall } from "@/services/memoryTools";
import { smartRouter, type RouteResult } from "@/services/smartRouter";
import { localSpeechRecognizer } from "@/services/localSpeechRecognition";
import { audioEffects } from "@/services/audioEffects";
import { buildAgentSystemPrompt, buildRouterGuidance } from "@/services/agentSystem";

// Type declarations
interface SpeechRecognitionEvent extends Event { results: SpeechRecognitionResultList; resultIndex: number; }
interface SpeechRecognitionErrorEvent extends Event { error: string; message?: string; }
interface SpeechRecognition extends EventTarget {
  lang: string; interimResults: boolean; maxAlternatives: number; continuous: boolean;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null; onstart: (() => void) | null;
  start(): void; stop(): void; abort(): void;
}
interface SpeechRecognitionConstructor { new (): SpeechRecognition; }
declare global { interface Window { SpeechRecognition: SpeechRecognitionConstructor; webkitSpeechRecognition: SpeechRecognitionConstructor; } }

export function WakeWordListener() {
  const { personas, currentPersona, setCurrentPersona, settings, isListening, setIsListening, setIsSpeaking, setIsGeneratingVoice, addMessage, addMemoryEntry, updatePersona, loadVoiceAudio, appState, messages } = useStore();
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const isProcessingRef = useRef(false);
  const shouldBeListeningRef = useRef(false);
  const userEnabledListeningRef = useRef(false); // Track if user manually enabled mic
  const isWaitingForCommandRef = useRef(false);
  const transcriptBufferRef = useRef("");
  const commandBufferRef = useRef("");
  const restartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const commandTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const silenceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const processCommandRef = useRef<((text: string) => Promise<void>) | null>(null);
  
  // Silence detection delay - wait this long after speech stops before processing (ms)
  // Increased to 5s to allow natural pauses during longer speech without cutting off
  const SILENCE_DELAY = 5000;
  
  // Note: Old voice enrollment system removed - using unified createVoice API
  
  // Use Puter.js-based recognition (free, cloud-based OpenAI Whisper-1)
  const useLocalRecognitionRef = useRef(false);
  const browserApiFailedRef = useRef(false);
  
  // Track if TTS backend is available - use local recognition if so
  const ttsBackendAvailableRef = useRef(appState.tts_backend_connected);
  
  // Refs to always have latest values without triggering re-renders
  const settingsRef = useRef(settings);
  const currentPersonaRef = useRef(currentPersona);
  const personasRef = useRef(personas);
  
  useEffect(() => { settingsRef.current = settings; }, [settings]);
  useEffect(() => { currentPersonaRef.current = currentPersona; }, [currentPersona]);
  useEffect(() => { personasRef.current = personas; }, [personas]);
  useEffect(() => { 
    ttsBackendAvailableRef.current = appState.tts_backend_connected;
    // If backend becomes available, use Puter.js-based recognition
    if (appState.tts_backend_connected && !useLocalRecognitionRef.current) {
      useLocalRecognitionRef.current = true;
    }
  }, [appState.tts_backend_connected]);

  const stripPromptTags = useCallback((text: string): string => {
    return text.replace(/\[TAG:\s*[^\]]+\]\s*/gi, "").trim();
  }, []);

  const extractToolCalls = useCallback((response: string): ToolCall[] => {
    const toolCalls: ToolCall[] = [];
    
    // Find all JSON objects that look like tool calls
    const jsonPattern = /\{[\s\S]*?"name"\s*:\s*"[^"]+"[\s\S]*?"arguments"\s*:\s*\{[^}]*\}[\s\S]*?\}/g;
    const matches = response.match(jsonPattern) || [];
    
    for (const match of matches) {
      try {
        const parsed = JSON.parse(match) as ToolCall;
        if (parsed?.name && parsed?.arguments && typeof parsed.arguments === "object") {
          toolCalls.push(parsed);
        }
      } catch {
        // Not valid JSON, skip
      }
    }
    
    return toolCalls;
  }, []);

  const chatWithReadOnlyTools = useCallback(
    async (
      model: string,
      messages: { role: "system" | "user" | "assistant"; content: string }[],
      options: { temperature?: number; top_p?: number; repeat_penalty?: number },
      maxToolSteps: number = 4
    ): Promise<string> => {
      let workingMessages = [...messages];

      for (let step = 0; step <= maxToolSteps; step++) {
        const response = await ollamaService.chat(model, workingMessages, options);
        const toolCalls = extractToolCalls(response);
        
        if (toolCalls.length === 0) {
          return response;
        }

        // Execute all tool calls
        const toolResponses: string[] = [];
        for (const toolCall of toolCalls) {
          const toolResult = await memoryToolsService.executeReadOnlyToolCall(toolCall);
          
          if (toolCall.name === "read_memory" && toolResult.success && toolResult.data?.content) {
            const content = toolResult.data.content;
            const filename = toolResult.data.filename || toolCall.arguments.filename;
            toolResponses.push(`FILE CONTENT [${filename}]:\n\n${content}\n\n[END FILE]`);
          } else {
            toolResponses.push(`TOOL_RESULT [${toolCall.name}]: ${JSON.stringify(toolResult)}`);
          }
        }

        // Add the tool results as user message with explicit instruction to quote exactly
        const toolResultsContent = toolResponses.join("\n\n") + 
          "\n\nQUOTE THE EXACT TEXT ABOVE. Format: 'The file contains: [exact text]'";
        
        workingMessages = [
          ...workingMessages,
          { role: "assistant", content: `I found the file contents.` },
          { role: "user", content: toolResultsContent },
        ];
        
        // Continue to next iteration to get the natural language response
        continue;
      }

      return "I reached the tool-call step limit. Please ask again with a specific memory filename.";
    },
    [extractToolCalls]
  );

  const playWakeSound = useCallback(() => {
    try {
      const audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      oscillator.frequency.setValueAtTime(880, audioContext.currentTime);
      oscillator.frequency.exponentialRampToValueAtTime(440, audioContext.currentTime + 0.1);
      gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.3);
    } catch (e) {}
  }, []);

  processCommandRef.current = async (transcript: string) => {
    const persona = currentPersonaRef.current;
    if (!persona) {
      toast.error("No persona selected");
      return;
    }
    
    if (isProcessingRef.current) {
      return;
    }
    
    isProcessingRef.current = true;
    isWaitingForCommandRef.current = false;
    commandBufferRef.current = "";
    setIsListening(false);
    shouldBeListeningRef.current = false;
    
    if (commandTimeoutRef.current) {
      clearTimeout(commandTimeoutRef.current);
      commandTimeoutRef.current = null;
    }
    
    const cleanedTranscript = stripPromptTags(transcript);
    
    // Add user message to chat (show original transcript without tag)
    addMessage({ role: "user", content: cleanedTranscript, inputType: "voice" });
    
    // Add to memory if enabled
    if (settingsRef.current.enable_memory) {
      addMemoryEntry(persona.id, { 
        content: `User: ${transcript}`, 
        timestamp: new Date().toISOString(), 
        importance: 0.6 
      });
    }

    let response: string | null = null;
    let placeholderId: string | null = null;

    try {
      ollamaService.setBaseUrl(settingsRef.current.ollama_url);
      
      // Check if web search is enabled and might be needed
      let searchContext = "";
      if (settingsRef.current.enable_web_search && searxngService.isEnabled()) {
        // Check if query needs current info
        if (searxngService.needsCurrentInfo(cleanedTranscript)) {
          toast.info("Searching for current information...", { duration: 2000 });
          try {
            const searchResult = await searxngService.search({ query: cleanedTranscript, deepSearch: true });
            searchContext = searxngService.formatForPrompt(searchResult);
          } catch {
            // Continue without search context
          }
        }
      }
      
      // Include ALL conversation history for voice (same as text)
      const conversationHistory = messages.map((msg) => {
        const timestamp = msg.timestamp || new Date().toISOString();
        const timeStr = new Date(timestamp).toLocaleString();
        return {
          role: msg.role as "user" | "assistant" | "system",
          content: `[${timeStr}] ${msg.content}`,
        };
      });

      // SMART ROUTING - Use lightweight LLM to classify intent for voice input
      let routeResult: RouteResult | null = null;
      try {
        routeResult = await smartRouter.route(
          cleanedTranscript,
          "voice",
          persona,
          false, // no images in voice path currently
          conversationHistory.slice(-10), // router only needs recent context
          settingsRef.current.router_model
        );
        
        // Override search decision if router suggests it
        if (routeResult.needsWebSearch && !searchContext) {
          try {
            const searchResult = await searxngService.search({ query: cleanedTranscript, deepSearch: true });
            searchContext = searxngService.formatForPrompt(searchResult);
          } catch {
            // Continue without search context
          }
        }
      } catch {
        // Router failed, use defaults
      }
      
      // Use router to summarize search results if available
      let processedSearchContext = searchContext;
      if (searchContext && searchContext.length > 2000) {
        try {
          processedSearchContext = await smartRouter.summarizeSearchResults(
            cleanedTranscript,
            searchContext,
            settingsRef.current.router_model
          );
        } catch {
          processedSearchContext = searchContext.substring(0, 4000) + "\n...[truncated]";
        }
      }

      // Build agent-aware system prompt
      const agentContext = {
        hasMemoryAccess: settingsRef.current.enable_memory,
        hasWebSearch: !!searchContext,
        hasVision: false,
        canWriteFiles: false,
        toolPermissionRequired: true,
      };
      
      let systemPrompt = buildAgentSystemPrompt(persona, agentContext);
      
      // Add router guidance if available
      if (routeResult?.suggestedApproach && routeResult.confidence > 0.6) {
        const routerGuidance = buildRouterGuidance(
          routeResult.suggestedApproach,
          routeResult.emotionalTone,
          routeResult.confidence
        );
        if (routerGuidance) {
          systemPrompt += routerGuidance;
        }
      }
      
      // Add memory files if available
      if (agentContext.hasMemoryAccess) {
        try {
          const memoryFiles = await memoryToolsService.listMemories(persona.id || "default");
          const fileNames = memoryFiles.map(f => f.name);
          if (fileNames.length > 0) {
            systemPrompt += `\n\nAvailable files: ${fileNames.join(", ")}`;
          }
        } catch (e) {
          // Memory access unavailable
        }
      }

      // Build user message with context
      let userMessageContent = cleanedTranscript;
      if (processedSearchContext) {
        userMessageContent = `[Search Results]\n${processedSearchContext}\n\n[Question]\n${cleanedTranscript}`;
      }

      const chatMessages = [
        { role: "system" as const, content: systemPrompt },
        ...conversationHistory,
        { role: "user" as const, content: userMessageContent },
      ];
      
      // Show generating toast AND chat placeholder
      const generatingToast = toast.loading(`${persona.name} is thinking...`, {
        duration: 60000, // 1 minute max
      });
      
      // Add placeholder to chat while generating
      placeholderId = `voice-${Date.now()}`;
      addMessage({ 
        role: "assistant", 
        content: "", 
        isLoading: true,
        id: placeholderId 
      });
      
      // Generate text response - NO num_predict limit (model uses full context window)
      response = await chatWithReadOnlyTools(
        settingsRef.current.default_model, 
        chatMessages, 
        { temperature: 0.8, top_p: 0.9, repeat_penalty: 1.1 }
      );
      
      toast.dismiss(generatingToast);

      // Add to memory if enabled
      if (settingsRef.current.enable_memory && response) {
        addMemoryEntry(persona.id, { 
          content: `${persona.name}: ${response}`, 
          timestamp: new Date().toISOString(), 
          importance: 0.5 
        });
        
        // Check if we need to summarize
        if (persona.memory.short_term.length >= settingsRef.current.memory_summarize_threshold) {
          memoryService.summarize(persona.memory, persona, settingsRef.current.default_model)
            .then((updatedMemory) => {
              const freshPersona = useStore.getState().personas.find(p => p.id === persona.id);
              if (freshPersona) {
                updatePersona({ ...freshPersona, memory: updatedMemory });
              }
            }).catch(() => {
              // Summarization failed, continue without update
            });
        }
      }

      // Generate voice and output both text and audio
      if (response) {
        await speakResponse(response, persona, (text) => {
          // Replace placeholder with actual message when audio starts
          const messages = useStore.getState().messages;
          const idx = messages.findIndex(m => m.id === placeholderId);
          if (idx !== -1) {
            useStore.setState(state => ({
              messages: state.messages.map((m, i) => 
                i === idx ? { ...m, content: text, isLoading: false } : m
              )
            }));
          } else {
            // Fallback: add new message if placeholder not found
            addMessage({ role: "assistant", content: text });
          }
        });
      } else {
        // No response - remove placeholder
        useStore.setState(state => ({
          messages: state.messages.filter(m => m.id !== placeholderId)
        }));
      }
    } catch (error) {
      toast.error(`Sorry, I couldn't process that: ${error instanceof Error ? error.message : 'Unknown error'}`);
      // Remove placeholder on error
      if (placeholderId) {
        useStore.setState(state => ({
          messages: state.messages.filter(m => m.id !== placeholderId)
        }));
      }
    } finally {
      isProcessingRef.current = false;
      isWaitingForCommandRef.current = false;
      commandBufferRef.current = "";
      setIsSpeaking(false);
      setIsGeneratingVoice(false);
      
      // Re-enable listening if user had manually enabled it
      if (userEnabledListeningRef.current) {
        shouldBeListeningRef.current = true;
        setIsListening(true);
      }
    }
  };

  // Track if user has interacted (clicked) - required for audio autoplay
  const userInteractedRef = useRef(false);
  
  // Set up user interaction listener
  useEffect(() => {
    const handleInteraction = () => {
      if (!userInteractedRef.current) {
        userInteractedRef.current = true;
      }
    };
    
    window.addEventListener('click', handleInteraction);
    window.addEventListener('keydown', handleInteraction);
    
    return () => {
      window.removeEventListener('click', handleInteraction);
      window.removeEventListener('keydown', handleInteraction);
    };
  }, []);
  
  // Track user-enabled listening state from localStorage and persist changes
  useEffect(() => {
    // Load saved state on mount
    const savedState = localStorage.getItem('mimic_user_enabled_listening');
    if (savedState === 'true') {
      userEnabledListeningRef.current = true;
    }
  }, []);
  
  // Watch isListening changes to track user intent
  useEffect(() => {
    // Only update ref if this is a user-initiated change (not from auto-processing)
    // We detect this by checking if we're not currently processing a command
    if (!isProcessingRef.current) {
      userEnabledListeningRef.current = isListening;
      localStorage.setItem('mimic_user_enabled_listening', isListening ? 'true' : 'false');
    }
  }, [isListening]);

  const speakResponse = async (text: string, persona: typeof currentPersona, onBeforePlay?: (text: string) => void): Promise<void> => {
    // Check if TTS is disabled
    if (settingsRef.current.tts_engine === 'off') {
      onBeforePlay?.(text);
      return;
    }
    
    if (settingsRef.current.voice_volume <= 0) {
      // No audio, just show text immediately
      onBeforePlay?.(text);
      return;
    }
    
    // Check if user has interacted - if not, show toast and skip audio
    if (!userInteractedRef.current) {
      onBeforePlay?.(text); // Still show text
      toast.info('Click the microphone button to enable voice responses', {
        duration: 5000,
        icon: '🔇'
      });
      return;
    }
    
    // Get persona's voice config
    const voiceConfig = persona?.voice_create?.voice_config;
    const voiceParams = voiceConfig?.params;
    
    // Determine which engine to use - simplified logic
    const useQwen3 = settingsRef.current.tts_engine === "qwen3";
    const useKitten = settingsRef.current.tts_engine === "kitten";

    
    try {
      let audioData: string | null = null;
      
      // PHASE 1: GENERATION - Generate all audio first (blocking)
      setIsGeneratingVoice(true);
      
      ttsService.setBaseUrl(settingsRef.current.tts_backend_url);
      
      if (useQwen3) {
        // Load voice data for synthetic/created voices
        try {
          const voiceData = await loadVoiceAudio(persona!.id);
          
          let referenceAudio: string | undefined = undefined;
          let referenceText: string | undefined = undefined;
          
          // Guard: If Qwen3 is selected but no reference audio available, fallback to browser TTS
          let effectiveEngine: "qwen3" | "kitten" = useQwen3 ? "qwen3" : "kitten";
          
          if (voiceData?.audio_data) {
            referenceAudio = voiceData.audio_data;
            referenceText = voiceData.reference_text || persona!.voice_create?.reference_text || undefined;
          } else {
            if (useQwen3) {
              effectiveEngine = 'kitten';
              toast.info('Qwen3 requires a voice sample. Using KittenTTS fallback.');
            }
          }
          
          // Use the unified createVoice API (same as ChatPanel)
          const response = await ttsService.createVoice(text, {
            reference_audio: referenceAudio,
            reference_text: referenceText,
            // Basic tuning from saved voice config
            pitch_shift: voiceParams?.pitch ?? 0,
            speed: (voiceParams?.speed ?? 1.0) * settingsRef.current.speech_rate,
            // Voice characteristics from saved config
            warmth: voiceParams?.warmth,
            expressiveness: voiceParams?.expressiveness,
            stability: voiceParams?.stability,
            clarity: voiceParams?.clarity,
            breathiness: voiceParams?.breathiness,
            resonance: voiceParams?.resonance,
            // Speech characteristics
            emotion: voiceParams?.emotion || 'neutral',
            emphasis: voiceParams?.emphasis,
            pauses: voiceParams?.pauses,
            energy: voiceParams?.energy,
            // Audio effects
            reverb: voiceParams?.reverb,
            eq_low: voiceParams?.eq_low,
            eq_mid: voiceParams?.eq_mid,
            eq_high: voiceParams?.eq_high,
            compression: voiceParams?.compression,
            // Engine selection - use effectiveEngine (may fallback to browser)
            engine: effectiveEngine,
            qwen3_model_size: (voiceParams?.qwen3_model_size as Qwen3ModelSize) || settingsRef.current.qwen3_model_size || '0.6B',
            use_flash_attention: settingsRef.current.qwen3_flash_attention !== false,
            seed: voiceParams?.seed,
          });
          
          audioData = response.audio_data;
        } catch (error) {
          setIsGeneratingVoice(false);
          toast.error(`Voice generation failed: ${error instanceof Error ? error.message : 'Unknown error'}. TTS disabled.`);
          onBeforePlay?.(text); // Still show text even if voice fails
          return;
        }
      } else if (useKitten) {
        // KittenTTS voice generation
        try {
          const voice = settingsRef.current.kitten_voice || "Bella";
          const model = settingsRef.current.kitten_model || "nano";
          const speed = settingsRef.current.kitten_speed || 1.0;
          
          const response = await ttsService.generateKittenTTS(text, voice, model, speed);
          audioData = response.audio_data;
        } catch (error) {
          setIsGeneratingVoice(false);
          toast.error(`KittenTTS failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
          onBeforePlay?.(text); // Still show text even if voice fails
          return;
        }
      }
      
      // PHASE 2: PLAYBACK - Output text and audio simultaneously
      setIsGeneratingVoice(false);
      setIsSpeaking(true, text, audioData || null);  // Pass text and audio data for lip sync
      
      // Show text in chat right before audio starts
      onBeforePlay?.(text);
      
      if (audioData) {
        // Get persona's voice tuning for post-processing
        const { getPersonaVoiceTuning } = useStore.getState();
        const voiceTuning = persona ? getPersonaVoiceTuning(persona.id) : null;
        
        if (voiceTuning) {
          try {
            await audioEffects.playWithEffects(audioData, {
              pitchShift: voiceTuning.pitchShift,
              speed: voiceTuning.speed * settingsRef.current.speech_rate,
              warmth: voiceTuning.warmth,
              clarity: voiceTuning.clarity,
              breathiness: voiceTuning.breathiness,
              resonance: voiceTuning.resonance,
              reverb: voiceTuning.reverb,
              eqLow: voiceTuning.eqLow,
              eqMid: voiceTuning.eqMid,
              eqHigh: voiceTuning.eqHigh,
              compression: voiceTuning.compression,
            });
            
            // Wait for playback to complete
            await new Promise<void>((resolve) => {
              const checkEnded = setInterval(() => {
                if (!audioEffects.isPlaying()) {
                  clearInterval(checkEnded);
                  resolve();
                }
              }, 100);
              setTimeout(() => { clearInterval(checkEnded); resolve(); }, 300000);
            });
          } catch {
            // Fall back to standard audio
            const audio = new Audio(`data:audio/wav;base64,${audioData}`);
            audio.volume = settingsRef.current.voice_volume;
            await new Promise<void>((resolve, reject) => {
              audio.onended = () => resolve();
              audio.onerror = () => reject();
              audio.play().catch(reject);
            });
          }
        } else {
          // No voice tuning - play raw audio
          const audio = new Audio(`data:audio/wav;base64,${audioData}`);
          audio.volume = settingsRef.current.voice_volume;
          await new Promise<void>((resolve, reject) => {
            audio.onended = () => resolve();
            audio.onerror = () => reject();
            audio.play().catch(reject);
          });
        }
      } else if (settingsRef.current.tts_engine === 'kitten') {
        // KittenTTS (Local) - Uses audio-based lip sync
        // Get fresh settings at generation time to pick up voice changes
        const currentSettings = useStore.getState().settings;
        const voice = currentSettings.kitten_voice || "Bella";
        const model = currentSettings.kitten_model || "nano";
        
        try {
          const kittenResponse = await ttsService.generateKittenTTS(text, voice, model);
          audioData = kittenResponse.audio_data;
          
          // Show text first, then start lip sync and audio together
          onBeforePlay?.(text);
          
          // Use global audio player so VrmAvatar can analyze the audio
          const { playAudio } = useStore.getState();
          playAudio({
            data: audioData,
            title: "AI Response",
            source: 'tts'
          });
          
          // Start lip sync after a short delay to let audio begin
          setTimeout(() => {
            setIsSpeaking(true, text, audioData);
          }, 50);
          
          // Wait for playback to complete
          await new Promise<void>((resolve) => {
            const checkEnded = setInterval(() => {
              const state = useStore.getState();
              if (!state.audioPlayer.isPlaying && state.audioPlayer.audioData === audioData) {
                clearInterval(checkEnded);
                resolve();
              }
            }, 100);
            setTimeout(() => { clearInterval(checkEnded); resolve(); }, 300000);
          });
        } catch (kittenError) {
          toast.error(`KittenTTS failed: ${kittenError instanceof Error ? kittenError.message : 'Unknown error'}. TTS disabled.`);
        }
      } else {
        // TTS is off - just show text without audio
        onBeforePlay?.(text);
      }
    } catch {
      // Still show text even if audio failed
      onBeforePlay?.(text);
    } finally {
      setIsGeneratingVoice(false);
      setIsSpeaking(false);
      // Re-enable listening if user had manually enabled it
      if (userEnabledListeningRef.current && !isProcessingRef.current) {
        shouldBeListeningRef.current = true;
        setIsListening(true);
      }
      // Dismiss the generating toast if it exists
      // (toast.dismiss is handled automatically by duration)
    }
  };

  const handleSwitchPersona = (targetWakeWord: string) => {
    
    const targetPersona = personas.find(p => 
      p.wake_words?.some(w => w.toLowerCase() === targetWakeWord.toLowerCase())
    );
    
    if (targetPersona) {
      if (targetPersona.id === currentPersona?.id) {
        toast.info(`Already talking to ${targetPersona.name}`);
        speakResponse("I'm already here.", targetPersona, (text) => addMessage({ role: "assistant", content: text })).catch(console.error);
      } else {
        setCurrentPersona(targetPersona);
        toast.success(`Switched to ${targetPersona.name}`);
        
        const responseWords = targetPersona.response_words || ["Yes?", "I'm listening"];
        const randomResponse = responseWords[Math.floor(Math.random() * responseWords.length)];
        speakResponse(randomResponse, targetPersona, (text) => addMessage({ role: "assistant", content: text })).catch(console.error);
      }
    } else {
      toast.error(`No persona found with wake word "${targetWakeWord}"`);
      speakResponse(`I couldn't find a persona called ${targetWakeWord}.`, currentPersona, (text) => addMessage({ role: "assistant", content: text })).catch(console.error);
    }
    
    isProcessingRef.current = false;
  };

  // Initialize recognition mode based on backend availability
  useEffect(() => {
    if (appState.tts_backend_connected) {
      useLocalRecognitionRef.current = true;
    }
  }, []); // Run once on mount

  // Note: Microphone always starts disabled on launch
  // This ensures user has clicked/interacted with the page before any audio plays
  // (Required for browser autoplay policies)
  useEffect(() => {
    // Always reset to disabled on launch
    setIsListening(false);
    userEnabledListeningRef.current = false;
    // Don't auto-start even if previously enabled - forces user interaction first
  }, []);

  // Track last restart time to prevent rapid restart loops
  const lastRestartTimeRef = useRef<number>(0);
  const consecutiveErrorCountRef = useRef<number>(0);

  useEffect(() => {
    // Skip browser SpeechRecognition if we're using Puter.js mode
    if (useLocalRecognitionRef.current) {
      return;
    }
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      return;
    }

    shouldBeListeningRef.current = isListening;

    if (!isListening) {
      isWaitingForCommandRef.current = false;
      commandBufferRef.current = "";
      consecutiveErrorCountRef.current = 0;
      if (commandTimeoutRef.current) { clearTimeout(commandTimeoutRef.current); commandTimeoutRef.current = null; }
      if (silenceTimeoutRef.current) { clearTimeout(silenceTimeoutRef.current); silenceTimeoutRef.current = null; }
      if (recognitionRef.current) { try { recognitionRef.current.abort(); } catch (e) {} recognitionRef.current = null; }
      return;
    }

    if (isProcessingRef.current) return;
    
    // Prevent rapid restarts - if we just restarted within 1 second, wait
    const now = Date.now();
    const timeSinceLastRestart = now - lastRestartTimeRef.current;
    if (timeSinceLastRestart < 1000) {
      const delayTimer = setTimeout(() => {
        if (shouldBeListeningRef.current && !isProcessingRef.current) {
          setIsListening(false);
          setTimeout(() => setIsListening(true), 100);
        }
      }, 1000 - timeSinceLastRestart);
      return () => clearTimeout(delayTimer);
    }

    // Try/catch around recognition creation in case constructor fails
    let recognition: SpeechRecognition;
    try {
      recognition = new SpeechRecognition();
    } catch {
      return;
    }
    
    const currentSettings = settingsRef.current;
    
    recognition.lang = currentSettings.language === "en" ? "en-US" : currentSettings.language === "zh" ? "zh-CN" : currentSettings.language === "ja" ? "ja-JP" : currentSettings.language === "ko" ? "ko-KR" : currentSettings.language === "de" ? "de-DE" : currentSettings.language === "fr" ? "fr-FR" : currentSettings.language === "es" ? "es-ES" : currentSettings.language === "it" ? "it-IT" : currentSettings.language === "pt" ? "pt-PT" : currentSettings.language === "ru" ? "ru-RU" : "en-US";
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.continuous = true;

    recognition.onstart = () => { 
      // Listening started
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let finalTranscript = "", interimTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalTranscript += transcript; else interimTranscript += transcript;
      }

      // Reset error count when we get any results
      if ((interimTranscript || finalTranscript) && consecutiveErrorCountRef.current > 0) {
        consecutiveErrorCountRef.current = 0;
      }



      if (isWaitingForCommandRef.current) {
        // Clear any existing silence timeout when we get speech
        if (silenceTimeoutRef.current) {
          clearTimeout(silenceTimeoutRef.current);
          silenceTimeoutRef.current = null;
        }
        
        if (finalTranscript) {
          commandBufferRef.current += " " + finalTranscript;
          commandBufferRef.current = commandBufferRef.current.trim();
          
          if (commandBufferRef.current.length > 2) {
            // Wait for SILENCE_DELAY ms of silence before processing
            silenceTimeoutRef.current = setTimeout(() => {
              if (isWaitingForCommandRef.current && commandBufferRef.current.length > 2) {
                const command = commandBufferRef.current;
                commandBufferRef.current = "";
                isWaitingForCommandRef.current = false;
                if (commandTimeoutRef.current) { clearTimeout(commandTimeoutRef.current); commandTimeoutRef.current = null; }
                processCommandRef.current?.(command);
              }
            }, SILENCE_DELAY);
            return;
          }
        }
        return;
      }

      if (finalTranscript) {
        transcriptBufferRef.current += " " + finalTranscript;
        transcriptBufferRef.current = transcriptBufferRef.current.trim();
      }

      if (transcriptBufferRef.current && !isProcessingRef.current && !isWaitingForCommandRef.current) {
        const wakeWords = (currentPersonaRef.current?.wake_words || ["Mimic"]).map(w => w.toLowerCase());
        const buffer = transcriptBufferRef.current.toLowerCase();
        
        let detectedWakeWord = "";
        let wakeIndex = -1;
        
        for (const wakeWord of wakeWords) {
          const idx = buffer.indexOf(wakeWord);
          if (idx !== -1 && (wakeIndex === -1 || idx < wakeIndex)) {
            wakeIndex = idx;
            detectedWakeWord = wakeWord;
          }
        }
        
        if (detectedWakeWord) {
          const afterWake = buffer.slice(wakeIndex + detectedWakeWord.length).trim();
          const command = afterWake.replace(/^[,.!?\s]+/, '');
          transcriptBufferRef.current = "";
          
          playWakeSound();
          
          if (command) {
            
            const switchMatch = command.match(/^switch\s+to\s+(.+)$/i);
            if (switchMatch) {
              const targetPersona = switchMatch[1].trim();
              handleSwitchPersona(targetPersona);
              return;
            }
            
            setTimeout(() => { processCommandRef.current?.(command); }, 100);
          } else {
            isWaitingForCommandRef.current = true;
            commandBufferRef.current = "";
            
            const responseWords = currentPersonaRef.current?.response_words || ["Yes?", "I'm listening"];
            const randomResponse = responseWords[Math.floor(Math.random() * responseWords.length)];
            
            toast.info(`${currentPersonaRef.current?.name || "Mimic"}: ${randomResponse}`, { description: "Speak your command", duration: 10000 });
            
            setTimeout(() => {
              if (isWaitingForCommandRef.current) {
                speakResponse(randomResponse, currentPersonaRef.current, (text) => addMessage({ role: "assistant", content: text })).catch(console.error);
              }
            }, 500);
            
            if (commandTimeoutRef.current) clearTimeout(commandTimeoutRef.current);
            if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);
            commandTimeoutRef.current = setTimeout(() => {
              if (isWaitingForCommandRef.current) {
                isWaitingForCommandRef.current = false;
                commandBufferRef.current = "";
                isProcessingRef.current = false;
                if (silenceTimeoutRef.current) { clearTimeout(silenceTimeoutRef.current); silenceTimeoutRef.current = null; }
              }
            }, 10000);
          }
        }
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (event.error === "aborted") return;
      
      consecutiveErrorCountRef.current++;
      
      if (event.error === "not-allowed") {
        toast.error("Microphone permission denied.");
      } else if (event.error === "network") {
        // Switch to local recognition after 3 network errors
        if (consecutiveErrorCountRef.current >= 3 && !useLocalRecognitionRef.current) {
          useLocalRecognitionRef.current = true;
          browserApiFailedRef.current = true;
          toast.info('Switching to Puter.js speech recognition...');
          // Stop browser recognition, local mode will take over
          setIsListening(false);
        }
      } else if (event.error === "no-speech") {
        // This is normal if user isn't speaking
      }
    };

    recognition.onend = () => {
      lastRestartTimeRef.current = Date.now();
      
      // Always restart if we should be listening (for continuous listening)
      if (shouldBeListeningRef.current && settingsRef.current.auto_listen) {
        // Small delay to prevent rapid restart loops
        setTimeout(() => {
          if (shouldBeListeningRef.current) {
            try { 
              recognition.start(); 
            } catch {
              // Failed to restart, will try again
            }
          }
        }, 300);
        return;
      }
      
      // Stop if too many consecutive errors
      if (consecutiveErrorCountRef.current > 10) {
        toast.error('Speech recognition failed repeatedly. Please refresh the page.');
        setIsListening(false);
        return;
      }
    };

    recognitionRef.current = recognition;
    lastRestartTimeRef.current = Date.now();
    
    // Delay start slightly to ensure clean state
    const startTimer = setTimeout(() => { 
      try { 
        recognition.start(); 
      } catch {
        consecutiveErrorCountRef.current++;
      }
    }, 300);

    return () => {
      clearTimeout(startTimer);
      if (restartTimeoutRef.current) clearTimeout(restartTimeoutRef.current);
      if (commandTimeoutRef.current) clearTimeout(commandTimeoutRef.current);
      if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);
      if (recognitionRef.current) { 
        try { recognitionRef.current.abort(); } catch (e) {} 
        recognitionRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isListening]);

  // Puter.js-based speech recognition (uses OpenAI Whisper-1 via Puter)
  useEffect(() => {
    if (!useLocalRecognitionRef.current) return;
    
    // Update the ref to match current isListening state
    shouldBeListeningRef.current = isListening;
    
    localSpeechRecognizer.setBackendUrl(settings.tts_backend_url);
    
    localSpeechRecognizer.onResult((result) => {
      const transcript = result.transcript.trim();
      const lowerTranscript = transcript.toLowerCase();
      
      const wakeWords = (currentPersonaRef.current?.wake_words || ["Mimic"]).map(w => w.toLowerCase());
      
      // Check if wake word is in transcript
      for (const wakeWord of wakeWords) {
        if (lowerTranscript.includes(wakeWord)) {
          const afterWake = lowerTranscript.split(wakeWord)[1]?.trim() || "";
          
          playWakeSound();
          
          if (afterWake) {
            // Command included after wake word - process immediately
            const command = afterWake.replace(/^[,.!?\s]+/, ''); // Remove leading punctuation
            
            // Stop current recognition before processing
            localSpeechRecognizer.abort();
            
            // Process the command
            setTimeout(() => {
              if (processCommandRef.current) {
                processCommandRef.current(command);
              }
            }, 100);
          } else {
            // Just wake word, show "I'm listening" and record next utterance
            isWaitingForCommandRef.current = true;
            const responseWords = currentPersonaRef.current?.response_words || ["Yes?", "I'm listening"];
            const randomResponse = responseWords[Math.floor(Math.random() * responseWords.length)];
            toast.info(`${currentPersonaRef.current?.name || "Mimic"}: ${randomResponse}`, { 
              description: "Speak your command", 
              duration: 10000 
            });
            
            // Speak the response
            speakResponse(randomResponse, currentPersonaRef.current, (text) => addMessage({ role: "assistant", content: text })).catch(console.error);
            
            // Start another recording to capture the command
            setTimeout(() => {
              if (isWaitingForCommandRef.current) {
                localSpeechRecognizer.start();
              }
            }, 500);
            
            // Timeout after 8 seconds
            setTimeout(() => {
              if (isWaitingForCommandRef.current) {
                isWaitingForCommandRef.current = false;
                commandBufferRef.current = "";
                if (silenceTimeoutRef.current) { clearTimeout(silenceTimeoutRef.current); silenceTimeoutRef.current = null; }
              }
            }, 10000);
          }
          return;
        }
      }
      
      // If waiting for command, buffer it and wait for silence
      if (isWaitingForCommandRef.current && transcript) {
        // Clear any existing silence timeout when we get speech
        if (silenceTimeoutRef.current) {
          clearTimeout(silenceTimeoutRef.current);
          silenceTimeoutRef.current = null;
        }
        
        commandBufferRef.current += " " + transcript;
        commandBufferRef.current = commandBufferRef.current.trim();
        
        if (commandBufferRef.current.length > 2) {
          // Wait for SILENCE_DELAY ms of silence before processing
          silenceTimeoutRef.current = setTimeout(() => {
            if (isWaitingForCommandRef.current && commandBufferRef.current.length > 2) {
              const command = commandBufferRef.current;
              commandBufferRef.current = "";
              isWaitingForCommandRef.current = false;
              localSpeechRecognizer.abort();
              processCommandRef.current?.(command);
            }
          }, SILENCE_DELAY);
        }
      }
    });
    
    localSpeechRecognizer.onError(() => {
      // Don't show toast for every error to avoid spam
    });
    
    // Start listening loop
    const listenLoop = async () => {
      if (!shouldBeListeningRef.current) {
        return;
      }
      
      try {
        // Use selected microphone device
        const deviceId = settingsRef.current.microphone_device || undefined;
        const started = await localSpeechRecognizer.start(deviceId);
        
        if (started) {
          // Wait for recording to complete (max 10 seconds)
          await new Promise(resolve => setTimeout(resolve, 10500));
          
          // Loop if still listening
          if (shouldBeListeningRef.current && !isProcessingRef.current) {
            listenLoop();
          }
        } else {
          // Retry after delay
          setTimeout(listenLoop, 2000);
        }
      } catch {
        // Retry after delay
        setTimeout(listenLoop, 2000);
      }
    };
    
    if (isListening) {
      listenLoop();
    } else {
      localSpeechRecognizer.abort();
    }
    
    return () => {
      localSpeechRecognizer.abort();
    };
  }, [isListening, settings.tts_backend_url]);

  return null;
}
