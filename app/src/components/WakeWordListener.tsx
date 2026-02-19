import { useEffect, useRef, useCallback } from "react";
import { useStore } from "@/store";
import { toast } from "sonner";
import { ollamaService } from "@/services/ollama";
import { ttsService, type Qwen3ModelSize } from "@/services/tts";
import { searxngService } from "@/services/searxng";
import { memoryService } from "@/services/memory";
import { memoryToolsService, type ToolCall } from "@/services/memoryTools";
import { localSpeechRecognizer } from "@/services/localSpeechRecognition";
import { audioEffects } from "@/services/audioEffects";

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
  const { personas, currentPersona, setCurrentPersona, settings, isListening, setIsListening, setIsSpeaking, setIsGeneratingVoice, addMessage, addMemoryEntry, updatePersona, loadVoiceAudio, appState } = useStore();
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
  // Increased to 3.5s to allow natural pauses during speech without cutting off
  const SILENCE_DELAY = 3500;
  
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
      console.log('🎤 TTS backend connected - using Puter.js for speech recognition');
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
    console.log('🎤 PROCESSING COMMAND:', transcript);
    console.log('🎤 isProcessingRef.current:', isProcessingRef.current);
    
    const persona = currentPersonaRef.current;
    if (!persona) {
      console.error('🎤 No current persona!');
      toast.error("No persona selected");
      return;
    }
    
    if (isProcessingRef.current) {
      console.log('🎤 Already processing, skipping');
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
    addMessage({ role: "user", content: cleanedTranscript });
    
    // Add to memory if enabled
    if (settingsRef.current.enable_memory) {
      addMemoryEntry(persona.id, { 
        content: `User: ${transcript}`, 
        timestamp: new Date().toISOString(), 
        importance: 0.6 
      });
    }

    let response: string | null = null;

    try {
      console.log('🎤 Starting Ollama chat...');
      console.log('🎤 Persona:', persona.name);
      console.log('🎤 Model:', settingsRef.current.default_model);
      
      ollamaService.setBaseUrl(settingsRef.current.ollama_url);
      
      // Check if web search is enabled and might be needed
      let searchContext = "";
      if (settingsRef.current.enable_web_search && searxngService.isEnabled()) {
        // Check if query needs current info
        if (searxngService.needsCurrentInfo(cleanedTranscript)) {
          console.log('🎤 Performing web search for current info...');
          toast.info("Searching for current information...", { duration: 2000 });
          try {
            const searchResult = await searxngService.search({ query: cleanedTranscript });
            searchContext = searxngService.formatForPrompt(searchResult);
            console.log('🎤 Search context:', searchContext);
          } catch (searchError) {
            console.error('🎤 Web search failed:', searchError);
            // Continue without search context
          }
        }
      }
      
      // Fetch persona rules as system instruction context
      let rulesContext = "";
      try {
        const { personaRulesService } = await import('@/services/personaRules');
        const personaRules = await personaRulesService.getOrGenerateRules({
          id: persona.id,
          name: persona.name,
          personality_prompt: persona.personality_prompt,
          description: persona.description,
        });
        rulesContext = personaRules;
      } catch (error) {
        console.error('Failed to load persona rules:', error);
      }

      // Check if query is about memory files - only then include detailed tool policy
      const queryLower = cleanedTranscript.toLowerCase();
      const isMemoryQuery = queryLower.includes('memory') || 
                            queryLower.includes('file') || 
                            queryLower.includes('document') ||
                            queryLower.includes('saved') ||
                            queryLower.includes('note') ||
                            queryLower.includes('test1') ||
                            queryLower.includes('test2') ||
                            queryLower.includes('test.') ||
                            queryLower.includes('.txt') ||
                            queryLower.includes('content') ||
                            queryLower.includes('contents') ||
                            queryLower.includes('says') ||
                            queryLower.includes('what is in') ||
                            queryLower.includes('what does') ||
                            (queryLower.includes('read') && queryLower.includes('file'));
      
      let toolPolicy = "";
      if (isMemoryQuery) {
        // Proactively fetch memory files list for memory-related queries
        let memoryFilesList = "";
        try {
          const memoryFiles = await memoryToolsService.listMemories();
          if (memoryFiles.length > 0) {
            memoryFilesList = memoryFiles.map(f => `- ${f.name}`).join('\n');
          } else {
            memoryFilesList = "(No memory files found)";
          }
        } catch (e) {
          memoryFilesList = "(Unable to list memory files)";
        }
        
        const fileNames = memoryFilesList.split(/\n/).map(f => f.replace('- ', '').trim()).filter(f => f);
        toolPolicy = [
          "MANDATORY TOOL PROTOCOL - YOU MUST FOLLOW THIS:",
          "Files in memory folder: " + fileNames.join(", "),
          "",
          "IMPORTANT: You have ZERO knowledge of file contents.",
          "You cannot see, guess, or infer what files contain.",
          "Filename patterns (like 'taco', 'banana') mean NOTHING to you.",
          "",
          "TO ANSWER THIS QUESTION:",
          "1. Output ONLY this exact JSON:",
          '   {"name":"read_memory","arguments":{"filename":"test.txt"}}',
          "2. NOTHING ELSE. No text, no explanation, no guessing.",
          "3. The system will give you the actual file content.",
          "",
          "WHEN REPORTING FILE CONTENTS:",
          "- Quote the EXACT text from the file, verbatim",
          "- Do not paraphrase, interpret, or be conversational",
          "- Do not add commentary about 'fun messages' or 'enthusiasm'",
          "- Just state: 'The file contains: [exact content]'",
          "",
          "IF YOU HAVEN'T RECEIVED TOOL_RESULT:",
          "- You don't know the answer.",
          "- Say 'I need to check that file for you.' and use the tool.",
          "",
          "NEVER:",
          "- Guess file contents based on filenames",
          "- Make up content about tacos, bananas, or anything",
          "- Respond as if you know the content without using the tool",
          "- Continue the conversation without calling the tool",
          "- Paraphrase or interpret the content - quote it exactly",
        ].join("\n");
      } else {
        // Minimal tool info for non-memory queries
        toolPolicy = "Memory tools available. Use read_memory to access file contents when needed.";
      }
      
      // Build system prompt with optional search context
      const systemPrompt = ollamaService.buildPersonaSystemPrompt(
        persona, 
        false, // hasImages
        settingsRef.current.enable_memory, // includeMemory
        searchContext || undefined,
        undefined, // no user files in voice path
        rulesContext || undefined,
        toolPolicy
      );
      
      // Log the complete prompt structure being sent to Ollama
      console.log('');
      console.log('╔══════════════════════════════════════════════════════════════╗');
      console.log('║                    PROMPT SENT TO OLLAMA                      ║');
      console.log('╠══════════════════════════════════════════════════════════════╣');
      console.log('║ SYSTEM PROMPT:');
      console.log('╠──────────────────────────────────────────────────────────────╣');
      console.log(systemPrompt.split('\n').map((line: string) => '║ ' + line.substring(0, 75)).join('\n'));
      console.log('╠──────────────────────────────────────────────────────────────╣');
      console.log('║ USER MESSAGE:');
      console.log('╠──────────────────────────────────────────────────────────────╣');
      console.log('║ ' + cleanedTranscript);
      if (searchContext) {
        console.log('╠──────────────────────────────────────────────────────────────╣');
        console.log('║ SEARCH CONTEXT INCLUDED: Yes');
        console.log('║ ' + searchContext.substring(0, 150) + '...');
      }
      console.log('╚══════════════════════════════════════════════════════════════╝');
      console.log('');
      
      const chatMessages = [
        { role: "system" as const, content: systemPrompt }, 
        { role: "user" as const, content: cleanedTranscript }
      ];
      
      // Show generating toast
      const generatingToast = toast.loading(`${persona.name} is thinking...`, {
        duration: 60000, // 1 minute max
      });
      
      // Generate text response
      console.log('🎤 Calling Ollama chat...');
      response = await chatWithReadOnlyTools(
        settingsRef.current.default_model, 
        chatMessages, 
        { temperature: 0.8, top_p: 0.9, repeat_penalty: 1.1 }
      );
      
      toast.dismiss(generatingToast);
      
      // Log the complete response from Ollama
      console.log('');
      console.log('╔══════════════════════════════════════════════════════════════╗');
      console.log('║                 RESPONSE RECEIVED FROM OLLAMA                 ║');
      console.log('╠══════════════════════════════════════════════════════════════╣');
      console.log(response ? response.split('\n').map((line: string) => '║ ' + line.substring(0, 75)).join('\n') : '║ (empty response)');
      console.log('╚══════════════════════════════════════════════════════════════╝');
      console.log('');

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
            }).catch(console.error);
        }
      }

      // Generate voice and output both text and audio
      if (response) {
        await speakResponse(response, persona, (text) => {
          // Callback called right before audio starts playing
          addMessage({ role: "assistant", content: text });
        });
      }
    } catch (error) {
      console.error('🎤 Error processing command:', error);
      toast.error(`Sorry, I couldn't process that: ${error instanceof Error ? error.message : 'Unknown error'}`);
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
        console.log('🎵 User interaction detected - audio enabled');
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
      console.log('🎤 User listening preference updated:', isListening);
    }
  }, [isListening]);

  const speakResponse = async (text: string, persona: typeof currentPersona, onBeforePlay?: (text: string) => void): Promise<void> => {
    // Check if TTS is disabled
    if (settingsRef.current.tts_engine === 'off') {
      console.log('🎵 TTS is disabled (off), skipping voice generation');
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
      console.log('🎵 Audio skipped - waiting for user interaction');
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
    
    // Determine which engine to use
    const hasVoicecreate = !!persona?.voice_create?.has_audio;
    const isSyntheticVoice = persona?.voice_id === "synthetic" || persona?.voice_id === "created";
    const useQwen3 = hasVoicecreate && isSyntheticVoice && settingsRef.current.tts_engine === "qwen3";
    const useStyleTTS2 = hasVoicecreate && isSyntheticVoice && settingsRef.current.tts_engine === "styletts2";
    
    // Debug logging for voice selection
    console.log('🎵 Voice selection check:', {
      personaName: persona?.name,
      personaId: persona?.id,
      hasVoicecreate,
      voiceId: persona?.voice_id,
      ttsEngine: settingsRef.current.tts_engine,
      useQwen3,
      useStyleTTS2,
      voiceParams: voiceParams ? 'present' : 'missing'
    });
    
    try {
      let audioData: string | null = null;
      
      // PHASE 1: GENERATION - Generate all audio first (blocking)
      console.log('🎵 Starting voice generation...');
      setIsGeneratingVoice(true);
      
      ttsService.setBaseUrl(settingsRef.current.tts_backend_url);
      
      if (useQwen3 || useStyleTTS2) {
        // Load voice data for synthetic/created voices
        try {
          const voiceData = await loadVoiceAudio(persona!.id);
          
          let referenceAudio: string | undefined = undefined;
          let referenceText: string | undefined = undefined;
          
          // Guard: If Qwen3 is selected but no reference audio available, fallback to StyleTTS2
          let effectiveEngine: "qwen3" | "styletts2" = useQwen3 ? "qwen3" : "styletts2";
          
          if (voiceData?.audio_data) {
            referenceAudio = voiceData.audio_data;
            referenceText = voiceData.reference_text || persona!.voice_create?.reference_text || undefined;
            console.log('🎵 Reference audio loaded:', referenceAudio.length, 'chars');
          } else {
            console.warn('🎵 No reference audio found');
            if (useQwen3) {
              console.warn('🎵 Qwen3 requires reference audio - falling back to StyleTTS2');
              effectiveEngine = 'styletts2';
              toast.info('Qwen3 requires a voice. Using StyleTTS2 fallback.');
            }
          }
          
          // Use the unified createVoice API (same as ChatPanel)
          const ttsStartTime = performance.now();
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
            // Engine selection - use effectiveEngine (may fallback to styletts2)
            engine: effectiveEngine,
            qwen3_model_size: (voiceParams?.qwen3_model_size as Qwen3ModelSize) || settingsRef.current.qwen3_model_size || '0.6B',
            use_flash_attention: settingsRef.current.qwen3_flash_attention !== false,
            seed: voiceParams?.seed,
          });
          
          console.log(`[TIMING] Voice creation (${response.engine_used}): ${(performance.now() - ttsStartTime).toFixed(0)}ms`);
          
          audioData = response.audio_data;
          console.log('🎵 Voice generation complete');
        } catch (error) {
          console.error("Voice creation failed:", error);
          toast.error(`Voice generation failed: ${error instanceof Error ? error.message : 'Unknown error'}. Using browser TTS.`);
          // Fall through to browser TTS
        }
      }
      
      // PHASE 2: PLAYBACK - Output text and audio simultaneously
      setIsGeneratingVoice(false);
      setIsSpeaking(true);
      
      // Show text in chat right before audio starts
      onBeforePlay?.(text);
      
      if (audioData) {
        // Get persona's voice tuning for post-processing
        const { getPersonaVoiceTuning } = useStore.getState();
        const voiceTuning = persona ? getPersonaVoiceTuning(persona.id) : null;
        
        if (voiceTuning) {
          // Apply post-processing effects during playback
          console.log('🎵 Applying voice tuning during playback:', voiceTuning);
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
          } catch (error) {
            console.error('🎵 Audio effects playback failed:', error);
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
      } else {
        // Browser TTS fallback
        await ttsService.speakWithBrowserTTS(text, settingsRef.current.voice_volume, settingsRef.current.speech_rate);
      }
    } catch (error) {
      console.error('TTS failed:', error);
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
    console.log('Switching to persona with wake word:', targetWakeWord);
    
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
      console.log('No persona found with wake word:', targetWakeWord);
      toast.error(`No persona found with wake word "${targetWakeWord}"`);
      speakResponse(`I couldn't find a persona called ${targetWakeWord}.`, currentPersona, (text) => addMessage({ role: "assistant", content: text })).catch(console.error);
    }
    
    isProcessingRef.current = false;
  };

  // Initialize recognition mode based on backend availability
  useEffect(() => {
    if (appState.tts_backend_connected) {
      console.log('🎤 TTS backend available - using Puter.js for speech recognition');
      useLocalRecognitionRef.current = true;
    }
  }, []); // Run once on mount

  // Note: Microphone always starts disabled on launch
  // This ensures user has clicked/interacted with the page before any audio plays
  // (Required for browser autoplay policies)
  useEffect(() => {
    console.log('🎤 Microphone starts disabled - user must click mic button to enable');
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
      console.log('🎤 Using Puter.js recognition - skipping browser SpeechRecognition');
      return;
    }
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.error('SpeechRecognition not supported');
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
      console.log(`🎤 Restarting too fast (${timeSinceLastRestart}ms), delaying...`);
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
    } catch (e) {
      console.error('Failed to create SpeechRecognition:', e);
      return;
    }
    
    const currentSettings = settingsRef.current;
    
    recognition.lang = currentSettings.language === "en" ? "en-US" : currentSettings.language === "zh" ? "zh-CN" : currentSettings.language === "ja" ? "ja-JP" : currentSettings.language === "ko" ? "ko-KR" : currentSettings.language === "de" ? "de-DE" : currentSettings.language === "fr" ? "fr-FR" : currentSettings.language === "es" ? "es-ES" : currentSettings.language === "it" ? "it-IT" : currentSettings.language === "pt" ? "pt-PT" : currentSettings.language === "ru" ? "ru-RU" : "en-US";
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.continuous = true;

    recognition.onstart = () => { 
      console.log('🎤 Listening for wake word:', currentPersonaRef.current?.wake_words?.[0] || "Mimic"); 
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let finalTranscript = "", interimTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalTranscript += transcript; else interimTranscript += transcript;
      }

      // Reset error count when we get any results
      if ((interimTranscript || finalTranscript) && consecutiveErrorCountRef.current > 0) {
        console.log('🎤 Speech detected - resetting error count');
        consecutiveErrorCountRef.current = 0;
      }

      if (interimTranscript) console.log('Interim:', interimTranscript);

      if (isWaitingForCommandRef.current) {
        // Clear any existing silence timeout when we get speech
        if (silenceTimeoutRef.current) {
          clearTimeout(silenceTimeoutRef.current);
          silenceTimeoutRef.current = null;
        }
        
        if (finalTranscript) {
          commandBufferRef.current += " " + finalTranscript;
          commandBufferRef.current = commandBufferRef.current.trim();
          console.log('Command buffer:', commandBufferRef.current);
          
          if (commandBufferRef.current.length > 2) {
            // Wait for SILENCE_DELAY ms of silence before processing
            console.log('Command captured, waiting for silence...');
            silenceTimeoutRef.current = setTimeout(() => {
              if (isWaitingForCommandRef.current && commandBufferRef.current.length > 2) {
                console.log('Silence detected, processing command:', commandBufferRef.current);
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
        console.log('Buffer:', transcriptBufferRef.current);
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
          console.log('WAKE WORD DETECTED:', detectedWakeWord);
          const afterWake = buffer.slice(wakeIndex + detectedWakeWord.length).trim();
          const command = afterWake.replace(/^[,.!?\s]+/, '');
          transcriptBufferRef.current = "";
          
          playWakeSound();
          
          if (command) {
            console.log('Command found, processing:', command);
            
            const switchMatch = command.match(/^switch\s+to\s+(.+)$/i);
            if (switchMatch) {
              const targetPersona = switchMatch[1].trim();
              handleSwitchPersona(targetPersona);
              return;
            }
            
            setTimeout(() => { processCommandRef.current?.(command); }, 100);
          } else {
            console.log('No command yet, waiting...');
            isWaitingForCommandRef.current = true;
            commandBufferRef.current = "";
            
            const responseWords = currentPersonaRef.current?.response_words || ["Yes?", "I'm listening"];
            const randomResponse = responseWords[Math.floor(Math.random() * responseWords.length)];
            
            toast.info(`${currentPersonaRef.current?.name || "Mimic"}: ${randomResponse}`, { description: "Speak your command", duration: 8000 });
            
            setTimeout(() => {
              if (isWaitingForCommandRef.current) {
                speakResponse(randomResponse, currentPersonaRef.current, (text) => addMessage({ role: "assistant", content: text })).catch(console.error);
              }
            }, 500);
            
            if (commandTimeoutRef.current) clearTimeout(commandTimeoutRef.current);
            if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);
            commandTimeoutRef.current = setTimeout(() => {
              if (isWaitingForCommandRef.current) {
                console.log('Command timeout');
                isWaitingForCommandRef.current = false;
                commandBufferRef.current = "";
                isProcessingRef.current = false;
                if (silenceTimeoutRef.current) { clearTimeout(silenceTimeoutRef.current); silenceTimeoutRef.current = null; }
              }
            }, 8000);
          }
        }
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (event.error === "aborted") return;
      
      consecutiveErrorCountRef.current++;
      console.error(`Recognition error (${consecutiveErrorCountRef.current}):`, event.error);
      
      if (event.error === "not-allowed") {
        toast.error("Microphone permission denied.");
      } else if (event.error === "network") {
        console.error('Network error - speech recognition service unavailable.');
        // Switch to local recognition after 3 network errors
        if (consecutiveErrorCountRef.current >= 3 && !useLocalRecognitionRef.current) {
          console.log('🎤 Switching to Puter.js recognition...');
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
      console.log('Recognition ended');
      lastRestartTimeRef.current = Date.now();
      
      if (isWaitingForCommandRef.current && !isProcessingRef.current) {
        setTimeout(() => {
          if (isWaitingForCommandRef.current && !isProcessingRef.current) {
            try { recognition.start(); } catch (e) {}
          }
        }, 200);
        return;
      }
      
      // Stop if too many consecutive errors
      if (consecutiveErrorCountRef.current > 10) {
        console.error('🎤 Too many errors, stopping wake word listener');
        toast.error('Speech recognition failed repeatedly. Please refresh the page.');
        setIsListening(false);
        return;
      }
      
      if (shouldBeListeningRef.current && !isProcessingRef.current && settingsRef.current.auto_listen) {
        if (restartTimeoutRef.current) clearTimeout(restartTimeoutRef.current);
        const delay = Math.min(2000, 500 + (consecutiveErrorCountRef.current * 200)); // Progressive backoff
        console.log(`🎤 Will restart in ${delay}ms (error count: ${consecutiveErrorCountRef.current})`);
        restartTimeoutRef.current = setTimeout(() => {
          if (shouldBeListeningRef.current && !isProcessingRef.current) {
            setIsListening(false);
            setTimeout(() => setIsListening(true), 100);
          }
        }, delay);
      }
    };

    recognitionRef.current = recognition;
    lastRestartTimeRef.current = Date.now();
    
    // Delay start slightly to ensure clean state
    const startTimer = setTimeout(() => { 
      try { 
        recognition.start(); 
        console.log('🎤 Speech recognition started successfully');
      } catch (error) {
        console.error('🎤 Failed to start recognition:', error);
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
    console.log('🎤 Puter.js recognition effect - isListening:', isListening, 'shouldBeListening:', shouldBeListeningRef.current);
    
    localSpeechRecognizer.setBackendUrl(settings.tts_backend_url);
    
    localSpeechRecognizer.onResult((result) => {
      const transcript = result.transcript.trim();
      const lowerTranscript = transcript.toLowerCase();
      
      console.log('🎤 Puter.js recognition result:', transcript);
      console.log('🎤 isProcessingRef:', isProcessingRef.current, 'isWaitingForCommand:', isWaitingForCommandRef.current);
      
      const wakeWords = (currentPersonaRef.current?.wake_words || ["Mimic"]).map(w => w.toLowerCase());
      console.log('🎤 Wake words to check:', wakeWords);
      
      // Check if wake word is in transcript
      for (const wakeWord of wakeWords) {
        console.log(`🎤 Checking if "${lowerTranscript}" includes "${wakeWord}"`);
        
        if (lowerTranscript.includes(wakeWord)) {
          const afterWake = lowerTranscript.split(wakeWord)[1]?.trim() || "";
          console.log('✅ WAKE WORD DETECTED (local):', wakeWord);
          console.log('🎤 After wake word:', afterWake || '(empty - waiting for command)');
          
          playWakeSound();
          
          if (afterWake) {
            // Command included after wake word - process immediately
            console.log('🎤 Processing command immediately:', afterWake);
            const command = afterWake.replace(/^[,.!?\s]+/, ''); // Remove leading punctuation
            console.log('🎤 Cleaned command:', command);
            
            // Stop current recognition before processing
            localSpeechRecognizer.abort();
            
            // Process the command
            setTimeout(() => {
              console.log('🎤 Calling processCommandRef.current...');
              if (processCommandRef.current) {
                processCommandRef.current(command);
              } else {
                console.error('🎤 processCommandRef.current is null!');
              }
            }, 100);
          } else {
            // Just wake word, show "I'm listening" and record next utterance
            console.log('🎤 No command after wake word, waiting...');
            isWaitingForCommandRef.current = true;
            const responseWords = currentPersonaRef.current?.response_words || ["Yes?", "I'm listening"];
            const randomResponse = responseWords[Math.floor(Math.random() * responseWords.length)];
            toast.info(`${currentPersonaRef.current?.name || "Mimic"}: ${randomResponse}`, { 
              description: "Speak your command", 
              duration: 8000 
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
                console.log('🎤 Command wait timeout');
                isWaitingForCommandRef.current = false;
                commandBufferRef.current = "";
                if (silenceTimeoutRef.current) { clearTimeout(silenceTimeoutRef.current); silenceTimeoutRef.current = null; }
              }
            }, 8000);
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
        console.log('🎤 Command buffer:', commandBufferRef.current);
        
        if (commandBufferRef.current.length > 2) {
          // Wait for SILENCE_DELAY ms of silence before processing
          console.log('🎤 Command captured, waiting for silence...');
          silenceTimeoutRef.current = setTimeout(() => {
            if (isWaitingForCommandRef.current && commandBufferRef.current.length > 2) {
              console.log('🎤 Silence detected, processing command:', commandBufferRef.current);
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
    
    localSpeechRecognizer.onError((error) => {
      console.error('Local recognition error:', error);
      // Don't show toast for every error to avoid spam
    });
    
    // Start listening loop
    const listenLoop = async () => {
      if (!shouldBeListeningRef.current) {
        console.log('🎤 listenLoop: shouldBeListening is false, exiting');
        return;
      }
      
      try {
        // Use selected microphone device
        const deviceId = settingsRef.current.microphone_device || undefined;
        console.log('🎤 listenLoop: Starting recognition with device:', deviceId || 'default');
        const started = await localSpeechRecognizer.start(deviceId);
        console.log('🎤 listenLoop: start() returned:', started);
        
        if (started) {
          // Wait for recording to complete (max 10 seconds)
          console.log('🎤 listenLoop: Waiting for recording to complete...');
          await new Promise(resolve => setTimeout(resolve, 10500));
          console.log('🎤 listenLoop: Recording period complete');
          
          // Loop if still listening
          if (shouldBeListeningRef.current && !isProcessingRef.current) {
            console.log('🎤 listenLoop: Continuing to next loop');
            listenLoop();
          } else {
            console.log('🎤 listenLoop: Stopping - shouldBeListening:', shouldBeListeningRef.current, 'isProcessing:', isProcessingRef.current);
          }
        } else {
          console.log('🎤 listenLoop: start() returned false, retrying in 2s');
          // Retry after delay
          setTimeout(listenLoop, 2000);
        }
      } catch (error) {
        console.error('🎤 listenLoop: ERROR:', error);
        // Retry after delay
        setTimeout(listenLoop, 2000);
      }
    };
    
    if (isListening) {
      console.log('🎤 Starting local speech recognition...');
      listenLoop();
    } else {
      console.log('🎤 Stopping local speech recognition (isListening is false)');
      localSpeechRecognizer.abort();
    }
    
    return () => {
      localSpeechRecognizer.abort();
    };
  }, [isListening, settings.tts_backend_url]);

  return null;
}
