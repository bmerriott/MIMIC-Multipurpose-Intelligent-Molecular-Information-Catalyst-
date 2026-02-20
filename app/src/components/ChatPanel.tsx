import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, User, Bot, Sparkles, RefreshCw, ChevronDown, Video, X, Wifi, Paperclip, Zap, BrainCircuit } from "lucide-react";
import { MimicLogo } from "./MimicLogo";
import { useStore } from "@/store";
import type { ChatMessage } from "@/types";
import { Button } from "./ui/button";

import { Textarea } from "./ui/textarea";
import { toast } from "sonner";
import { ollamaService } from "@/services/ollama";
import { ttsService, type TTSEngine, type Qwen3ModelSize } from "@/services/tts";
import { memoryService } from "@/services/memory";
import { memoryToolsService, type ToolCall } from "@/services/memoryTools";
import { searxngService } from "@/services/searxng";
import { audioEffects } from "@/services/audioEffects";

import { intentRouter } from "@/services/router";

import { LiveVideoModal } from "./LiveVideoModal";
import { FileAttachmentModal } from "./FileAttachmentModal";
import { MemoryManager, MemoryWriteConfirmation } from "./MemoryManager";

export function ChatPanel() {
  const [input, setInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSearchingWeb, setIsSearchingWeb] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [attachedImages, setAttachedImages] = useState<string[]>([]);
  const [attachedFiles, setAttachedFiles] = useState<{name: string, content: string}[]>([]);
  const [showLiveVideo, setShowLiveVideo] = useState(false);
  const [showFileModal, setShowFileModal] = useState(false);
  const [showMemoryManager, setShowMemoryManager] = useState(false);
  const [memoryWritePending, setMemoryWritePending] = useState<{
    filename: string;
    content?: string;
    confirmCallback: () => void;
    cancelCallback: () => void;
  } | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const lastMessageCountRef = useRef(0);
  
  // Cache for search results to avoid redundant searches in follow-up questions
  const searchCacheRef = useRef<{query: string, context: string, timestamp: number} | null>(null);
  
  // Expandable chat window state
  // Start at minimum height, user drags DOWN to expand chat area
  const MIN_CHAT_HEIGHT = 150;
  const MAX_CHAT_HEIGHT = 600;
  const [chatHeight, setChatHeight] = useState<number>(MIN_CHAT_HEIGHT); 
  const [isDragging, setIsDragging] = useState(false);
  const dragStartY = useRef(0);
  const dragStartHeight = useRef(0);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const chatAreaRef = useRef<HTMLDivElement>(null);
  
  const {
    messages,
    addMessage,
    currentPersona,
    settings,
    setIsSpeaking,
    setIsGeneratingVoice,
    updatePersona,
    loadVoiceAudio,
    addMemoryEntry,
    updateAppState,
    appState,
    isListening,
    isGeneratingVoice,
    updateSettings,
  } = useStore();
  
  // Global audio player hook
  const globalPlayAudio = useStore(state => state.playAudio);
  
  // Ref to always have latest currentPersona (avoids stale closure issues)
  const currentPersonaRef = useRef(currentPersona);
  useEffect(() => {
    currentPersonaRef.current = currentPersona;
  }, [currentPersona]);

  // Smooth scroll to bottom
  const scrollToBottom = useCallback((smooth = true) => {
    if (scrollContainerRef.current) {
      const container = scrollContainerRef.current;
      container.scrollTo({
        top: container.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto'
      });
    }
  }, []);

  // Check if user is near bottom
  const isNearBottom = useCallback(() => {
    if (!scrollContainerRef.current) return true;
    const container = scrollContainerRef.current;
    const threshold = 150; // pixels from bottom
    return container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
  }, []);

  // Handle scroll events
  const handleScroll = useCallback(() => {
    if (scrollContainerRef.current) {
      setShowScrollButton(!isNearBottom());
    }
  }, [isNearBottom]);

  // Resize handlers for expandable chat window
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    dragStartY.current = e.clientY;
    dragStartHeight.current = chatHeight;
  }, [chatHeight]);

  const handleResizeMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;
    // Drag DOWN (e.clientY increases) = EXPAND chat area (increase height)
    // Drag UP (e.clientY decreases) = SHRINK chat area (decrease height)
    const delta = e.clientY - dragStartY.current;
    const newHeight = Math.max(MIN_CHAT_HEIGHT, Math.min(MAX_CHAT_HEIGHT, dragStartHeight.current + delta));
    setChatHeight(newHeight);
  }, [isDragging]);

  const handleResizeEnd = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Attach resize event listeners
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleResizeMove);
      document.addEventListener('mouseup', handleResizeEnd);
      return () => {
        document.removeEventListener('mousemove', handleResizeMove);
        document.removeEventListener('mouseup', handleResizeEnd);
      };
    }
  }, [isDragging, handleResizeMove, handleResizeEnd]);

  // Auto-scroll to bottom when new messages arrive (only if near bottom)
  useEffect(() => {
    if (messages.length > lastMessageCountRef.current) {
      if (isNearBottom()) {
        // Use requestAnimationFrame for smooth scroll after render
        requestAnimationFrame(() => {
          scrollToBottom(true);
        });
      } else {
        setShowScrollButton(true);
      }
      lastMessageCountRef.current = messages.length;
    }
  }, [messages, isNearBottom, scrollToBottom]);

  // Scroll to bottom on initial load
  useEffect(() => {
    const timer = setTimeout(() => {
      scrollToBottom(false);
    }, 100);
    return () => clearTimeout(timer);
  }, [scrollToBottom]);

  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);

  // Preload TTS model on mount to avoid timeout on first request
  useEffect(() => {
    const preloadTTSModel = async () => {
      if (settings.tts_mode !== "browser") {
        try {
          console.log("ðŸŽµ Preloading TTS model...");
          ttsService.setBaseUrl(settings.tts_backend_url);
          await ttsService.preloadModel();
          console.log("ðŸŽµ TTS model preloaded successfully");
        } catch (error) {
          // Non-critical error, just log it
          console.log("ðŸŽµ TTS model preload skipped (backend may not be ready):", error);
        }
      }
    };
    
    preloadTTSModel();
  }, [settings.tts_backend_url, settings.tts_mode]);
  
  // Initialize web search service
  useEffect(() => {
    searxngService.setEnabled(settings.enable_web_search);
  }, [settings.enable_web_search]);

  // ===== VISION / IMAGE HANDLING =====

  const handlePaste = (e: ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const blob = item.getAsFile();
        if (blob) {
          const reader = new FileReader();
          reader.onload = (e) => {
            const base64 = (e.target?.result as string).split(',')[1];
            setAttachedImages(prev => [...prev, base64]);
            toast.success("Image pasted! ðŸ“¸");
          };
          reader.readAsDataURL(blob);
        }
      }
    }
  };

  const handleLiveVideoCapture = (imageBase64: string) => {
    setAttachedImages(prev => [...prev, imageBase64]);
  };

  const removeImage = (index: number) => {
    setAttachedImages(prev => prev.filter((_, i) => i !== index));
  };

  const removeFile = (index: number) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleFileAttach = (name: string, content: string) => {
    setAttachedFiles(prev => [...prev, { name, content }]);
    toast.success(`Attached: ${name}`);
  };

  const stripPromptTags = (text: string): string => {
    return text.replace(/\[TAG:\s*[^\]]+\]\s*/gi, "").trim();
  };

  const extractToolCalls = (response: string): ToolCall[] => {
    const toolCalls: ToolCall[] = [];
    
    // Find all JSON objects that look like tool calls
    // Match { ... "name": ... "arguments": ... } patterns
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
  };

  const chatWithReadOnlyTools = async (
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
        // No tool calls found, return the response as-is
        return response;
      }

      // Execute all tool calls
      const toolResponses: string[] = [];
      for (const toolCall of toolCalls) {
        const toolResult = await memoryToolsService.executeReadOnlyToolCall(toolCall);
        
        // Format tool result - for read_memory, make content very prominent
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

    return "I reached the tool-call step limit. Please ask again with a more specific memory file name.";
  };

  // Attach paste listener
  useEffect(() => {
    const handlePasteEvent = (e: ClipboardEvent) => handlePaste(e);
    document.addEventListener('paste', handlePasteEvent);
    return () => document.removeEventListener('paste', handlePasteEvent);
  }, []);

  const generateResponse = async (
    userMessage: string, 
    images?: string[], 
    files?: {name: string, content: string}[],
    personaOverride?: typeof currentPersona
  ): Promise<string> => {
    // Use provided persona or fall back to currentPersona
    const persona = personaOverride || currentPersona;
    if (!persona) {
      throw new Error("No persona selected");
    }

    console.log("ðŸ¤– [generateResponse] Using persona:", persona.name, "id:", persona.id);
    console.log("ðŸ¤– [generateResponse] Persona has voice:", !!persona.voice_create?.has_audio);

    ollamaService.setBaseUrl(settings.ollama_url);

    const cleanUserMessage = stripPromptTags(userMessage);
    let enhancedMessage = cleanUserMessage;
    let visionDescription = "";

    // DUAL-MODEL ARCHITECTURE:
    // If images are present, use vision model first to describe them,
    // then feed that description to the brain model
    if (images && images.length > 0) {
      console.log("ðŸ‘ï¸ [ChatPanel] Images detected, using vision model:", settings.vision_model);
      
      try {
        // Ask vision model to describe the images
        const visionMessages = [
          { 
            role: "user" as const, 
            content: "Describe what you see in this image in detail:",
            images: images 
          },
        ];
        
        visionDescription = await ollamaService.chat(
          settings.vision_model,
          visionMessages,
          { temperature: 0.3, top_p: 0.9 } // Lower temp for factual description
        );
        
        console.log("ðŸ‘ï¸ [ChatPanel] Vision model description:", visionDescription.substring(0, 100) + "...");
        
        // Enhance the user message with the vision description
        // Add explicit instruction to acknowledge what was seen
        enhancedMessage = `[INSTRUCTION: The user has shared ${images.length} image(s). You MUST acknowledge what you see in your response. Do not ignore the image content.]

Image analysis: ${visionDescription}

User's request: ${cleanUserMessage}

[REMEMBER: Address the image content directly in your response while staying in character]`;
      } catch (error) {
        console.error("Vision model error:", error);
        toast.error("Vision model failed, proceeding with text only");
        enhancedMessage = `[User shared ${images.length} image(s) but I couldn't analyze them]\n\nUser's message: ${cleanUserMessage}`;
      }
    }

    // FILE ATTACHMENTS: Build file context for system prompt
    let fileContext = "";
    if (files && files.length > 0) {
      fileContext = files.map(f => `FILE: ${f.name}\n${f.content}`).join('\n\n');
      console.log("ðŸ“Ž [ChatPanel] Files attached:", files.map(f => f.name).join(', '));
    }

    // WEB SEARCH: Check if query needs current information
    let searchContext = "";
    const CACHE_DURATION_MS = 5 * 60 * 1000; // 5 minutes
    
    if (settings.enable_web_search && searxngService.isEnabled() && searxngService.needsCurrentInfo(cleanUserMessage)) {
      // Check if we have cached results for a similar query
      const cached = searchCacheRef.current;
      const now = Date.now();
      if (cached && (now - cached.timestamp) < CACHE_DURATION_MS && 
          (cleanUserMessage.toLowerCase().includes(cached.query.toLowerCase()) || 
           cached.query.toLowerCase().includes(cleanUserMessage.toLowerCase()))) {
        console.log("ðŸ” [ChatPanel] Using cached search results for:", cached.query);
        searchContext = cached.context;
      } else {
        // Perform new search
        try {
          console.log("ðŸ” [ChatPanel] Web search enabled, searching for:", cleanUserMessage);
          setIsSearchingWeb(true);
          const searchResult = await searxngService.search({ query: cleanUserMessage });
          searchContext = searxngService.formatForPrompt(searchResult);
          
          // Cache the results
          searchCacheRef.current = {
            query: cleanUserMessage,
            context: searchContext,
            timestamp: now
          };
          console.log("ðŸ” [ChatPanel] Web search completed, results cached");
        } catch (error) {
          console.error("ðŸ” [ChatPanel] Web search failed:", error);
          // Continue without search results
        } finally {
          setIsSearchingWeb(false);
        }
      }
    }

    // INTENT ROUTING - Get minimal system prompt based on query type
    const route = await intentRouter.route(cleanUserMessage, persona.name, !!searchContext);
    console.log("ðŸŽ¯ [Router] Intent:", route.intent, "Confidence:", route.confidence);
    
    // For memory_read intent, fetch file list and build minimal prompt
    let systemPrompt = route.systemPrompt;
    if (route.intent === "memory_read") {
      try {
        const memoryFiles = await memoryToolsService.listMemories();
        const fileNames = memoryFiles.map(f => f.name);
        systemPrompt = intentRouter.getMemoryReadPrompt(persona.name, fileNames);
        console.log("ðŸ“Ž [Router] Memory files:", fileNames.join(", "));
      } catch (e) {
        systemPrompt = intentRouter.getMemoryReadPrompt(persona.name, []);
      }
    }

    const conversationHistory = messages.slice(-10).map((msg) => ({
      role: msg.role as "user" | "assistant" | "system",
      content: msg.content, // No more tag stripping needed
    }));

    // Add web search context to system prompt if available
    if (searchContext) {
      systemPrompt += "\n\n[SEARCH RESULTS]\n" + searchContext + "\n[END SEARCH]";
    }
    
    // Add file attachments to system prompt if available
    if (fileContext) {
      systemPrompt += "\n\n[ATTACHED FILES]\n" + fileContext + "\n[END FILES]";
    }
    
    console.log("ðŸ¤– [generateResponse] System prompt length:", systemPrompt.length);
    console.log("ðŸ¤– [generateResponse] System prompt preview:", systemPrompt.substring(0, 200) + "...");

    const chatMessages = [
      { role: "system" as const, content: systemPrompt },
      ...conversationHistory,
      { role: "user" as const, content: enhancedMessage },
    ];

    try {
      console.log("ðŸ¤– [ChatPanel] Sending to brain model:", settings.default_model);
      const response = await chatWithReadOnlyTools(
        settings.default_model,
        chatMessages,
        {
          temperature: 0.8,
          top_p: 0.95,
          repeat_penalty: 1.0,
        }
      );
      console.log("ðŸ¤– [ChatPanel] Brain response received, length:", response.length);
      
      if (!response || response.trim() === '') {
        throw new Error("Received empty response from language model");
      }

      return response;
    } catch (error) {
      console.error("LLM Error:", error);
      throw new Error("Failed to get response from language model. Is Ollama running?");
    }
  };

  const speakResponse = async (text: string, persona: typeof currentPersona, onBeforePlay?: (text: string) => void): Promise<void> => {
    // Check if TTS is disabled
    if (settings.tts_engine === 'off') {
      console.log("ðŸ”Š [ChatPanel] TTS is disabled (off), skipping voice generation");
      onBeforePlay?.(text);
      return;
    }
    
    if (settings.voice_volume <= 0) {
      // No audio, just show text immediately
      onBeforePlay?.(text);
      return;
    }
    
    console.log("ðŸ”Š [ChatPanel] speakResponse called");
    console.log("ðŸ”Š [ChatPanel] TTS mode:", settings.tts_mode);
    console.log("ðŸ”Š [ChatPanel] Persona for voice:", persona?.name, "id:", persona?.id);
    console.log("ðŸ”Š [ChatPanel] Voice check:", {
      hasVoice: !!persona?.voice_create?.has_audio,
      isSynthetic: persona?.voice_create?.voice_config?.type === "synthetic",
      voiceId: persona?.voice_id,
      voiceConfig: persona?.voice_create?.voice_config
    });
    
    // Check if persona has a synthetic voice configuration (new system)
    const hasSyntheticVoice = persona?.voice_create?.has_audio && 
                              persona.voice_create?.voice_config?.type === "synthetic" &&
                              persona.voice_create?.voice_config?.params;
    
    // Check if persona has a legacy created voice
    const hasCreatedVoice = persona?.voice_create?.has_audio && 
                           persona.voice_id === "created" &&
                           !persona.voice_create?.voice_config;
    
    // Determine which voice system to use based on tts_engine setting
    // tts_engine: 'off' | 'browser' | 'qwen3'
    const useQwen3Voice = hasSyntheticVoice && settings.tts_engine === "qwen3";
    const useBackendTTS = settings.tts_engine === "qwen3";
    
    console.log("ðŸ”Š [ChatPanel] Voice selection:", { 
      useQwen3Voice,
      useBackendTTS,
      hasSyntheticVoice,
      hasCreatedVoice,
      tts_engine: settings.tts_engine
    });
    
    
    try {
      let audioData: string | null = null;
      
      // PHASE 1: GENERATION - Generate all audio first (blocking)
      console.log('ðŸŽµ Starting voice generation...');
      setIsGeneratingVoice(true);
      
      // Note: No timeout - let the backend process as long as needed
      // Long LLM responses can take significant time to synthesize
      
      if (useQwen3Voice && persona?.voice_create?.voice_config?.params) {
        // ===== VOICE CREATION SYSTEM =====
        // Create voice using unified TTS API
        try {
          ttsService.setBaseUrl(settings.tts_backend_url);
          
          const ttsStartTime = performance.now();
          const params = persona.voice_create.voice_config.params;
          const engine = (settings.tts_engine as TTSEngine) || 'browser';
          
          console.log('ðŸŽµ Creating voice with unified TTS:', {
            engine: engine,
            pitch: params.pitch,
            speed: params.speed,
          });
          
          // Load reference audio from IndexedDB if needed (Qwen3 requires it)
          let referenceAudio: string | undefined = undefined;
          let referenceText: string | undefined = persona.voice_create?.reference_text;
          
          // Guard: If Qwen3 is selected but no reference audio available, fallback to StyleTTS2
          let effectiveEngine = engine;
          if (engine === 'qwen3') {
            console.log('ðŸŽµ Qwen3 selected - loading reference audio from storage...');
            console.log('ðŸŽµ Persona ID:', persona.id);
            console.log('ðŸŽµ Voice create config:', persona.voice_create);
            const voiceData = await loadVoiceAudio(persona.id);
            console.log('ðŸŽµ Voice data from IndexedDB:', voiceData ? `found (${voiceData.audio_data?.length} chars)` : 'not found');
            if (voiceData?.audio_data) {
              referenceAudio = voiceData.audio_data;
              console.log('ðŸŽµ Reference audio loaded:', referenceAudio.length, 'chars');
              // Check if it's valid base64
              if (!referenceAudio.match(/^[A-Za-z0-9+/=]+$/)) {
                console.error('ðŸŽµ WARNING: Reference audio does not look like valid base64!');
              }
            } else {
              console.warn('ðŸŽµ No reference audio found for Qwen3 - falling back to Browser TTS');
              console.warn('ðŸŽµ Qwen3 requires reference audio. Please create a voice in Voice Studio first.');
              effectiveEngine = 'browser';
              toast.info('Qwen3 requires a voice sample. Falling back to Browser TTS. Please create a voice first.');
            }
          }
          
          // Use the new unified createVoice API
          const response = await ttsService.createVoice(text, {
            reference_audio: referenceAudio,
            reference_text: referenceText,
            // Basic tuning
            pitch_shift: params.pitch || 0,
            speed: (params.speed || 1.0) * settings.speech_rate,
            // Voice characteristics from persona config
            warmth: params.warmth,
            expressiveness: params.expressiveness,
            stability: params.stability,
            clarity: (params as any).clarity,
            breathiness: (params as any).breathiness,
            resonance: (params as any).resonance,
            // Speech characteristics
            emotion: (params as any).emotion || 'neutral',
            emphasis: (params as any).emphasis,
            pauses: (params as any).pauses,
            energy: (params as any).energy,
            // Audio effects
            reverb: (params as any).reverb,
            eq_low: (params as any).eq_low,
            eq_mid: (params as any).eq_mid,
            eq_high: (params as any).eq_high,
            compression: (params as any).compression,
            // Engine selection - use effectiveEngine (may be fallback to styletts2)
            engine: effectiveEngine,
            qwen3_model_size: (settings.qwen3_model_size as Qwen3ModelSize) || '0.6B',
            use_flash_attention: settings.qwen3_flash_attention !== false,
            seed: params.seed,
          });
          
          console.log(`[TIMING] Voice creation (${response.engine_used}): ${(performance.now() - ttsStartTime).toFixed(0)}ms`);
          
          audioData = response.audio_data;
          console.log('ðŸŽµ Voice creation complete');
        } catch (error) {
          console.error("Voice creation failed:", error);
          toast.error(`Voice creation failed: ${error instanceof Error ? error.message : 'Unknown error'}. Using browser TTS.`);
          // Fall through to browser TTS
        }
      } else if (useBackendTTS) {
        try {
          ttsService.setBaseUrl(settings.tts_backend_url);
          
          // Generate voice - no timeout, let it complete naturally
          const response = await ttsService.generateSpeech({
            text,
            voice_id: persona?.voice_id || "default",
            speed: settings.speech_rate,
          });
          
          audioData = response.audio_data;
          console.log('ðŸŽµ Voice generation complete (StyleTTS 2)');
        } catch (error) {
          console.error("Qwen3 TTS failed:", error);
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
          console.log('ðŸŽµ Applying voice tuning during playback:', voiceTuning);
          try {
            await audioEffects.playWithEffects(audioData, {
              pitchShift: voiceTuning.pitchShift,
              speed: voiceTuning.speed * settings.speech_rate,
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
              
              // Safety timeout (5 minutes max)
              setTimeout(() => {
                clearInterval(checkEnded);
                resolve();
              }, 300000);
            });
          } catch (error) {
            console.error('ðŸŽµ Audio effects playback failed:', error);
            // Fall back to global player
            globalPlayAudio({
              data: audioData,
              title: "AI Response",
              source: 'tts'
            });
            
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
          }
        } else {
          // No voice tuning - play raw audio
          globalPlayAudio({
            data: audioData,
            title: "AI Response",
            source: 'tts'
          });
          
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
        }
      } else {
        // Browser TTS fallback
        await ttsService.speakWithBrowserTTS(text, settings.voice_volume, settings.speech_rate);
      }
    } catch (error) {
      console.error('TTS failed:', error);
      // Still show text even if audio failed
      onBeforePlay?.(text);
    } finally {
      setIsGeneratingVoice(false);
      setIsSpeaking(false);
    }
  };

  const handleSend = async () => {
    if ((!input.trim() && attachedImages.length === 0 && attachedFiles.length === 0) || isProcessing) return;

    if (!currentPersona) {
      toast.error("No persona selected");
      return;
    }

    // Build message content with file references
    let messageContent = input.trim() || (attachedImages.length > 0 ? "What do you see in this image?" : "");
    
    const userMessage: ChatMessage = {
      role: "user",
      content: messageContent,
      ...(attachedImages.length > 0 ? { images: attachedImages } : {}),
    };

    addMessage(userMessage);
    const imagesToSend = attachedImages;
    const filesToSend = attachedFiles;
    setInput("");
    setAttachedImages([]);
    setAttachedFiles([]);
    setIsProcessing(true);

    // CAPTURE persona data at start - don't use currentPersona during async (it may change)
    const personaAtStart = currentPersona;
    console.log(" [ChatPanel] handleSend started for persona:", personaAtStart.name, "id:", personaAtStart.id);

    try {
      if (settings.enable_memory) {
        const memoryContent = `User: ${messageContent}${imagesToSend.length > 0 ? ' [attached image]' : ''}${filesToSend.length > 0 ? ` [attached ${filesToSend.length} file(s)]` : ''}`;
        addMemoryEntry(personaAtStart.id, {
          content: memoryContent,
          timestamp: new Date().toISOString(),
          importance: 0.6,
        });
      }

      // Generate text response (but don't show it yet)
      console.log(" [ChatPanel] Calling generateResponse...");
      const response = await generateResponse(userMessage.content, imagesToSend, filesToSend, personaAtStart);
      console.log(" [ChatPanel] generateResponse returned, response length:", response.length);

      if (settings.enable_memory) {
        // Update memory service thresholds from settings
        memoryService.setThreshold(settings.memory_summarize_threshold);
        memoryService.setImportanceThreshold(settings.memory_importance_threshold || 0.5);
        
        // Use memory service to add message (respects importance threshold)
        const userMessage: ChatMessage = { role: 'user', content: input };
        memoryService.addMessage(personaAtStart, userMessage, settings.default_model)
          .then((updatedMemory) => {
            if (updatedMemory !== personaAtStart.memory) {
              updatePersona({
                ...personaAtStart,
                memory: updatedMemory,
              });
            }
          })
          .catch(console.error);

        // Add AI response memory
        const aiMessage: ChatMessage = { role: 'assistant', content: response };
        memoryService.addMessage(
          { ...personaAtStart, memory: personaAtStart.memory },
          aiMessage,
          settings.default_model
        )
          .then((updatedMemory) => {
            if (updatedMemory !== personaAtStart.memory) {
              updatePersona({
                ...personaAtStart,
                memory: updatedMemory,
              });
            }
          })
          .catch(console.error);
      }
      
      // Generate voice and wait for it to be ready (only if TTS is enabled)
      if (settings.tts_engine === 'off') {
        // TTS is disabled - just add text response immediately
        console.log(" [ChatPanel] TTS is off, adding text response only");
        addMessage({ role: "assistant", content: response });
      } else {
        // TTS is enabled - generate voice and output both text and audio
        console.log(" [ChatPanel] Calling speakResponse with persona:", personaAtStart.name);
        await speakResponse(response, personaAtStart, (text) => {
          // Callback called right before audio starts playing
          addMessage({ role: "assistant", content: text });
        });
        console.log(" [ChatPanel] speakResponse completed");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      toast.error("Failed to get response", {
        description: errorMessage,
      });
      
      addMessage({
        role: "assistant",
        content: "I apologize, but I'm having trouble connecting to my language model. Please make sure Ollama is running and try again.",
      });
    } finally {
      setIsProcessing(false);
      setIsGeneratingVoice(false);
      setIsSpeaking(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };
  
  // Auto-resize textarea based on content
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

  const clearChat = () => {
    useStore.getState().clearMessages();
    toast.success("Chat history cleared");
  };

  const testConnection = async () => {
    ollamaService.setBaseUrl(settings.ollama_url);
    const connected = await ollamaService.checkConnection();
    updateAppState({ ollama_connected: connected });
    
    if (connected) {
      toast.success("Connected to Ollama successfully");
    } else {
      toast.error("Could not connect to Ollama", {
        description: "Make sure Ollama is running at " + settings.ollama_url,
      });
    }
  };

  return (
    <div className="flex flex-col h-full relative" ref={chatContainerRef}>
      {/* Chat messages area - flex:1 to fill space, explicit height when resizing */}
      <div 
        ref={chatAreaRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto overflow-x-hidden scroll-smooth min-h-[120px]"
        style={{ 
          scrollbarWidth: 'thin', 
          scrollbarColor: 'var(--scrollbar-thumb) var(--scrollbar-track)',
          height: `${chatHeight}px`
        }}
      >
        <div className="p-4 space-y-4 min-h-full">
          {messages.length === 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center py-12"
            >
              <motion.div 
                className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center overflow-hidden"
                animate={{ scale: [1, 1.05, 1] }}
                transition={{ duration: 3, repeat: Infinity }}
              >
                <MimicLogo size={48} rotated={false} className="drop-shadow-lg" />
              </motion.div>
              
              {!isListening ? (
                // Mic is OFF - show enable prompt
                <>
                  <h3 className="text-lg font-semibold mb-2">
                     Welcome!
                  </h3>
                  <motion.p 
                    className="text-primary font-medium text-sm max-w-xs mx-auto mb-2"
                    animate={{ opacity: [1, 0.6, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    Enable the microphone to begin!
                  </motion.p>
                  <p className="text-muted-foreground text-xs max-w-xs mx-auto mb-4">
                    Or type a message below to start chatting
                  </p>
                </>
              ) : (
                // Mic is ON - show wake word prompt
                <>
                  <h3 className="text-lg font-semibold mb-2">
                     I'm listening!
                  </h3>
                  <motion.p 
                    className="text-primary font-medium text-sm max-w-xs mx-auto mb-2"
                    animate={{ opacity: [1, 0.6, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    Say "{currentPersona?.wake_words?.[0] || "Mimic"}" to wake me up!
                  </motion.p>
                  <p className="text-muted-foreground text-xs max-w-xs mx-auto mb-4">
                    Or type a message to start chatting
                  </p>
                </>
              )}
              
              {!appState.ollama_connected && (
                <Button variant="outline" size="sm" onClick={testConnection}>
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Test Ollama Connection
                </Button>
              )}
            </motion.div>
          )}

          <AnimatePresence>
            {messages.map((message, index) => (
              <motion.div
                key={message.id || index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className={`flex gap-3 ${
                  message.role === "user" ? "flex-row-reverse" : ""
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    message.role === "user"
                      ? "bg-secondary"
                      : "bg-gradient-to-br from-indigo-500 to-purple-600"
                  }`}
                >
                  {message.role === "user" ? (
                    <User className="w-4 h-4" />
                  ) : (
                    <Bot className="w-4 h-4 text-white" />
                  )}
                </div>
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm whitespace-pre-wrap break-words ${
                    message.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  }`}
                >
                  {message.content}
                  {/* Display attached images */}
                  {message.images && message.images.length > 0 && (
                    <div className="flex gap-2 mt-2 flex-wrap">
                      {message.images.map((img, i) => (
                        <img
                          key={i}
                          src={`data:image/jpeg;base64,${img}`}
                          alt={`Image ${i + 1}`}
                          className="max-w-[200px] max-h-[200px] rounded-lg object-cover cursor-pointer hover:opacity-90 transition-opacity"
                          onClick={() => window.open(`data:image/jpeg;base64,${img}`, '_blank')}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {isProcessing && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex gap-3"
            >
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="bg-muted rounded-2xl px-4 py-3 flex items-center gap-1">
                <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </motion.div>
          )}
          
          {isGeneratingVoice && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex gap-3"
            >
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="bg-muted rounded-2xl px-4 py-3 flex items-center gap-2">
                {/* Pulsing dots */}
                <div className="flex items-center gap-1">
                  {[0, 1, 2].map((i) => (
                    <motion.span
                      key={i}
                      className="w-2 h-2 bg-primary rounded-full"
                      animate={{
                        scale: [1, 1.3, 1],
                        opacity: [0.4, 1, 0.4],
                      }}
                      transition={{
                        duration: 0.6,
                        repeat: Infinity,
                        delay: i * 0.15,
                      }}
                    />
                  ))}
                </div>
                <span className="text-xs text-muted-foreground">Generating voice...</span>
              </div>
            </motion.div>
          )}
          
          {/* Spacer for scroll padding */}
          <div className="h-4" />
        </div>
      </div>

      {/* Scroll to bottom button */}
      {showScrollButton && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          className="absolute bottom-20 left-1/2 -translate-x-1/2 z-10"
        >
          <Button
            variant="secondary"
            size="sm"
            onClick={() => scrollToBottom(true)}
            className="shadow-lg rounded-full px-3 py-1 text-xs"
          >
            <ChevronDown className="w-4 h-4 mr-1" />
            New messages
          </Button>
        </motion.div>
      )}

      {/* Live Video Modal */}
      <LiveVideoModal
        isOpen={showLiveVideo}
        onClose={() => setShowLiveVideo(false)}
        onCapture={handleLiveVideoCapture}
      />

      {/* File Attachment Modal */}
      <FileAttachmentModal
        isOpen={showFileModal}
        onClose={() => setShowFileModal(false)}
        onAttach={handleFileAttach}
      />

      {/* Memory Manager */}
      <MemoryManager
        isOpen={showMemoryManager}
        onClose={() => setShowMemoryManager(false)}
      />

      {/* Memory Write Confirmation */}
      {memoryWritePending && (
        <MemoryWriteConfirmation
          isOpen={true}
          filename={memoryWritePending.filename}
          content={memoryWritePending.content}
          onConfirm={() => {
            memoryWritePending.confirmCallback();
            setMemoryWritePending(null);
          }}
          onCancel={() => {
            memoryWritePending.cancelCallback();
            setMemoryWritePending(null);
          }}
        />
      )}

      {/* Draggable divider between chat and input */}
      <div 
        className="h-3 flex items-center justify-center cursor-ns-resize bg-border/30 hover:bg-border/50 transition-colors select-none group"
        onMouseDown={handleResizeStart}
        title="Drag to resize chat area"
      >
        <div className="w-12 h-1 bg-muted-foreground/30 rounded-full group-hover:bg-muted-foreground/50 transition-colors" />
      </div>

      <div className="p-4 border-t border-border bg-card/50 backdrop-blur-sm">
        {/* Attached Images Preview */}
        {attachedImages.length > 0 && (
          <div className="flex gap-2 mb-2 flex-wrap">
            {attachedImages.map((img, i) => (
              <div key={i} className="relative group">
                <img
                  src={`data:image/jpeg;base64,${img}`}
                  alt={`Attached ${i + 1}`}
                  className="w-16 h-16 object-cover rounded-lg border"
                />
                <button
                  onClick={() => removeImage(i)}
                  className="absolute -top-1 -right-1 bg-destructive text-white rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}
        
        {/* Attached Files Preview */}
        {attachedFiles.length > 0 && (
          <div className="flex gap-2 mb-2 flex-wrap">
            {attachedFiles.map((file, i) => (
              <div key={i} className="relative group flex items-center gap-2 bg-muted px-3 py-1.5 rounded-lg text-sm">
                <Paperclip className="w-4 h-4 text-muted-foreground" />
                <span className="truncate max-w-[150px]">{file.name}</span>
                <button
                  onClick={() => removeFile(i)}
                  className="text-muted-foreground hover:text-destructive transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}
        
        <div className="flex gap-2">
          {/* Input Buttons - 2x2 Grid */}
          <div className="grid grid-cols-2 gap-1 shrink-0">
            {/* File Attachment Button */}
            <Button
              variant="outline"
              size="icon"
              onClick={() => setShowFileModal(true)}
              disabled={isProcessing}
              title="Attach file (PDF, TXT, DOCX, etc.)"
              className="h-8 w-8"
            >
              <Paperclip className="w-3.5 h-3.5" />
            </Button>
            
            {/* Live Video Button */}
            <Button
              variant={showLiveVideo ? "default" : "outline"}
              size="icon"
              onClick={() => setShowLiveVideo(true)}
              disabled={isProcessing}
              title="Live video feed"
              className="h-8 w-8"
            >
              <Video className="w-3.5 h-3.5" />
            </Button>
            
            {/* Memory Manager Button */}
            <Button
              variant="outline"
              size="icon"
              onClick={() => setShowMemoryManager(true)}
              disabled={isProcessing}
              title="Open Memory Manager"
              className="h-8 w-8"
            >
              <BrainCircuit className="w-3.5 h-3.5" />
            </Button>
            
            {/* Web Search Button - WiFi Icon */}
            <Button
              variant={settings.enable_web_search ? "default" : "outline"}
              size="icon"
              onClick={() => updateSettings({ enable_web_search: !settings.enable_web_search })}
              title={settings.enable_web_search ? "SearXNG web search enabled" : "Web search disabled"}
              className="h-8 w-8"
            >
              <Wifi className={`w-3.5 h-3.5 ${isSearchingWeb ? 'animate-pulse' : ''}`} />
            </Button>
          </div>
          
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Message ${currentPersona?.name || "Mimic"}...`}
            disabled={isProcessing}
            className="flex-1 min-h-[40px] max-h-[200px] resize-none"
            rows={1}
          />
          <Button
            onClick={handleSend}
            disabled={(!input.trim() && attachedImages.length === 0) || isProcessing}
            size="icon"
            className="shrink-0"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
        
        {/* TTS Configuration and Status - Always visible */}
        <div className="flex justify-between items-center mt-2">
          <div className="flex items-center gap-3 flex-wrap">
            {/* TTS Engine Toggle with Off option */}
            <div className="flex items-center gap-1 bg-muted rounded-lg p-0.5">
              <button
                onClick={() => updateSettings({ tts_engine: 'off' })}
                className={`px-2 py-0.5 text-xs rounded transition-colors flex items-center gap-1 ${
                  settings.tts_engine === 'off' 
                    ? 'bg-primary text-primary-foreground' 
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                title="Voice disabled - text only chat"
              >
                <span className="w-3 h-3 flex items-center justify-center text-[10px]">X</span>
                Off
              </button>
              <button
                onClick={() => updateSettings({ tts_engine: 'browser' })}
                className={`px-2 py-0.5 text-xs rounded transition-colors flex items-center gap-1 ${
                  settings.tts_engine === 'browser' 
                    ? 'bg-primary text-primary-foreground' 
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                title="Browser TTS - Uses your system's built-in text-to-speech"
              >
                <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
                Browser
              </button>
              <button
                onClick={() => updateSettings({ tts_engine: 'qwen3' })}
                className={`px-2 py-0.5 text-xs rounded transition-colors flex items-center gap-1 ${
                  settings.tts_engine === 'qwen3' 
                    ? 'bg-primary text-primary-foreground' 
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                title="Qwen3 - Higher quality, reference-based synthesis. Voices created with 1.7B work with 0.6B and vice versa."
              >
                <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>
                Qwen3
              </button>
            </div>
            
            {/* Qwen3 Model Size Toggle - Only show when Qwen3 selected and TTS not off */}
            {settings.tts_engine === 'qwen3' && (
              <div className="flex items-center gap-1 bg-muted rounded-lg p-0.5">
                <button
                  onClick={() => {
                    console.log('[ChatPanel] Switching to Qwen3 0.6B model');
                    updateSettings({ qwen3_model_size: '0.6B' });
                  }}
                  className={`px-2 py-0.5 text-xs rounded transition-colors flex items-center gap-1 ${
                    settings.qwen3_model_size !== '1.7B' 
                      ? 'bg-primary text-primary-foreground' 
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                  title="0.6B - Faster, ~3GB VRAM"
                >
                  <Zap className="w-3 h-3" />
                  0.6B
                </button>
                <button
                  onClick={() => {
                    console.log('[ChatPanel] Switching to Qwen3 1.7B model');
                    updateSettings({ qwen3_model_size: '1.7B' });
                  }}
                  className={`px-2 py-0.5 text-xs rounded transition-colors flex items-center gap-1 ${
                    settings.qwen3_model_size === '1.7B' 
                      ? 'bg-primary text-primary-foreground' 
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                  title="1.7B - Better quality, ~6GB VRAM"
                >
                  <Sparkles className="w-3 h-3" />
                  1.7B
                </button>
              </div>
            )}
          </div>
          {messages.length > 0 && (
            <Button variant="ghost" size="sm" onClick={clearChat} className="text-xs">
              Clear chat
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

