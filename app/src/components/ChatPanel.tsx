import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, User, Bot, Sparkles, RefreshCw, ChevronDown, Video, X, Wifi, Paperclip, Zap, BrainCircuit, Brain } from "lucide-react";
import { MimicLogo } from "./MimicLogo";
import { useStore } from "@/store";
import type { ChatMessage } from "@/types";
import { Button } from "./ui/button";

import { Textarea } from "./ui/textarea";
import { toast } from "sonner";
import { ollamaService } from "@/services/ollama";
import { ttsService, type Qwen3ModelSize } from "@/services/tts";
import { memoryService } from "@/services/memory";
import { memoryToolsService, type ToolCall } from "@/services/memoryTools";
import { searxngService } from "@/services/searxng";
import { audioEffects } from "@/services/audioEffects";

import { smartRouter, type RouteResult } from "@/services/smartRouter";
import { buildAgentSystemPrompt, buildRouterGuidance } from "@/services/agentSystem";
import { useToolConfirmation } from "@/hooks/useToolConfirmation";

import { LiveVideoModal } from "./LiveVideoModal";
import { FileAttachmentModal } from "./FileAttachmentModal";
import { MemoryManager, MemoryWriteConfirmation } from "./MemoryManager";
import { PersonalityManager } from "./PersonalityManager";
import { ToolConfirmationModal } from "./ToolConfirmationModal";

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
  const [showPersonalityManager, setShowPersonalityManager] = useState(false);
  const [memoryWritePending, setMemoryWritePending] = useState<{
    filename: string;
    content?: string;
    confirmCallback: () => void;
    cancelCallback: () => void;
  } | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const lastMessageCountRef = useRef(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  const isMountedRef = useRef(true);
  
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
    updateAppState,
    appState,
    isListening,
    updateSettings,
  } = useStore();
  
  // Tool confirmation hook for write operations
  const {
    pendingConfirmation: toolConfirmation,
    isExecuting: isToolExecuting,
    confirmExecution: confirmToolExecution,
    cancelExecution: cancelToolExecution,
  } = useToolConfirmation();
  
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

  // Cleanup on unmount - cancel ongoing requests
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, []);

  // Preload TTS model on mount to avoid timeout on first request
  useEffect(() => {
    const preloadTTSModel = async () => {
      if (settings.tts_engine !== "off") {
        try {
          ttsService.setBaseUrl(settings.tts_backend_url);
          await ttsService.preloadModel();
        } catch (error) {
          // Non-critical error, preload failed
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
    personaOverride?: typeof currentPersona,
    inputType: "voice" | "text" = "text"
  ): Promise<string> => {
    // Use provided persona or fall back to currentPersona
    const persona = personaOverride || currentPersona;
    if (!persona) {
      throw new Error("No persona selected");
    }

    ollamaService.setBaseUrl(settings.ollama_url);

    const cleanUserMessage = stripPromptTags(userMessage);
    let enhancedMessage = cleanUserMessage;
    let visionDescription = "";

    // DUAL-MODEL ARCHITECTURE:
    // If images are present, use vision model first to describe them,
    // then feed that description to the brain model
    if (images && images.length > 0) {
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
        
        // Enhance the user message with the vision description
        // Add explicit instruction to acknowledge what was seen
        enhancedMessage = `[INSTRUCTION: The user has shared ${images.length} image(s). You MUST acknowledge what you see in your response. Do not ignore the image content.]

Image analysis: ${visionDescription}

User's request: ${cleanUserMessage}

[REMEMBER: Address the image content directly in your response while staying in character]`;
      } catch (error) {

        toast.error("Vision model failed, proceeding with text only");
        enhancedMessage = `[User shared ${images.length} image(s) but I couldn't analyze them]\n\nUser's message: ${cleanUserMessage}`;
      }
    }

    // FILE ATTACHMENTS: Build file context for system prompt
    let fileContext = "";
    if (files && files.length > 0) {
      fileContext = files.map(f => `FILE: ${f.name}\n${f.content}`).join('\n\n');
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
        searchContext = cached.context;
      } else {
        // Perform new search
        try {
          setIsSearchingWeb(true);
          const searchResult = await searxngService.search({ query: cleanUserMessage, deepSearch: true });
          searchContext = searxngService.formatForPrompt(searchResult);
          
          // Cache the results
          searchCacheRef.current = {
            query: cleanUserMessage,
            context: searchContext,
            timestamp: now
          };
        } catch (error) {
          // Continue without search results
        } finally {
          setIsSearchingWeb(false);
        }
      }
    }

    // SMART ROUTING - Use lightweight LLM to classify intent and determine processing
    const routeResult: RouteResult = await smartRouter.route(
      cleanUserMessage,
      inputType,
      persona,
      !!(images && images.length > 0),
      messages.slice(-5).map(m => ({ role: m.role, content: m.content })),
      settings.router_model
    );
    
    // Build agent-aware system prompt
    const agentContext = {
      hasMemoryAccess: routeResult.needsMemoryAccess || routeResult.primaryIntent === "memory_read",
      hasWebSearch: settings.enable_web_search && !!searchContext,
      hasVision: !!(images && images.length > 0),
      canWriteFiles: false, // Always require confirmation
      toolPermissionRequired: true,
    };
    
    let systemPrompt = buildAgentSystemPrompt(persona, agentContext);
    
    // Add router guidance
    const routerGuidance = buildRouterGuidance(
      routeResult.suggestedApproach,
      routeResult.emotionalTone,
      routeResult.confidence
    );
    if (routerGuidance) {
      systemPrompt += routerGuidance;
    }
    
    // Handle memory access if router suggests it
    if (agentContext.hasMemoryAccess) {
      try {
        const personaId = persona.id || "default";
        const memoryFiles = await memoryToolsService.listMemories(personaId);
        const fileNames = memoryFiles.map(f => f.name);
        if (fileNames.length > 0) {
          systemPrompt += `\n\nAvailable files: ${fileNames.join(", ")}`;
        }
      } catch (e) {
        // Memory access unavailable
      }
    }

    // Include ALL messages - never slice. User controls memory via memory manager.
    const conversationHistory = messages.map((msg) => {
      const timestamp = msg.timestamp || new Date().toISOString();
      const timeStr = new Date(timestamp).toLocaleString();
      return {
        role: msg.role as "user" | "assistant" | "system",
        content: `[${timeStr}] ${msg.content}`,
      };
    });

    // Use router to summarize search results if available
    let processedSearchContext = searchContext;
    if (searchContext && searchContext.length > 2000) {
      try {
        processedSearchContext = await smartRouter.summarizeSearchResults(
          cleanUserMessage,
          searchContext,
          settings.router_model
        );
      } catch {
        // Fallback to truncation
        processedSearchContext = searchContext.substring(0, 4000) + "\n...[truncated]";
      }
    }

    // Build user message with context
    let userMessageContent = enhancedMessage;
    
    // Add processed search context to user message
    if (processedSearchContext) {
      userMessageContent = `[Search Results]\n${processedSearchContext}\n\n[Question]\n${enhancedMessage}`;
    }
    
    // Add file attachments to user message
    if (fileContext) {
      userMessageContent = `[Files]\n${fileContext}\n\n${userMessageContent}`;
    }

    const chatMessages = [
      { role: "system" as const, content: systemPrompt },
      ...conversationHistory,
      { role: "user" as const, content: userMessageContent },
    ];

    try {
      const response = await chatWithReadOnlyTools(
        settings.default_model,
        chatMessages,
        {
          temperature: 0.8,
          top_p: 0.95,
          repeat_penalty: 1.0,
          // No num_predict - model uses full context window
        }
      );
      
      if (!response || response.trim() === '') {
        throw new Error("Received empty response from language model");
      }

      return response;
    } catch (error) {

      throw new Error("Failed to get response from language model. Is Ollama running?");
    }
  };

  const speakResponse = async (text: string, persona: typeof currentPersona, onBeforePlay?: (text: string) => void): Promise<void> => {
    // Check if TTS is disabled
    if (settings.tts_engine === 'off') {
      onBeforePlay?.(text);
      return;
    }
    
    if (settings.voice_volume <= 0) {
      // No audio, just show text immediately
      onBeforePlay?.(text);
      return;
    }
    
    // Determine which voice system to use based on tts_engine setting
    // tts_engine: 'off' | 'qwen3' | 'kitten'
    const useKittenTTS = settings.tts_engine === "kitten";
    
    try {
      let audioData: string | null = null;
      
      // PHASE 1: GENERATION - Generate all audio first (blocking)
      setIsGeneratingVoice(true);
      
      // Note: No timeout - let the backend process as long as needed
      // Long LLM responses can take significant time to synthesize
      
      // ===== QWEN3 VOICE CREATION =====
      // Use when Qwen3 engine is selected (voice created or default)
      if (settings.tts_engine === "qwen3") {
        try {
          ttsService.setBaseUrl(settings.tts_backend_url);
          
          // Load reference audio from IndexedDB/Tauri FS
          const voiceData = await loadVoiceAudio(persona?.id || "default");
          
          // Guard: If no reference audio available, try loading default persona voice
          let referenceAudio = voiceData?.audio_data;
          let referenceText = voiceData?.reference_text || persona?.voice_create?.reference_text;
          
          if (!referenceAudio) {
            // Try loading default voice as fallback
            const defaultVoiceData = await loadVoiceAudio("default");
            if (defaultVoiceData?.audio_data) {
              referenceAudio = defaultVoiceData.audio_data;
              referenceText = defaultVoiceData.reference_text || "Hello! I'm Mimic, your personal AI assistant.";
            } else {
              // No voice sample available - show text only
              toast.info('Qwen3 requires a voice sample. Please create a voice in Voice Creator first.');
              setIsGeneratingVoice(false);
              onBeforePlay?.(text); // Show text even without voice
              return;
            }
          }
          
          // Get voice params from persona config or use defaults
          const params = persona?.voice_create?.voice_config?.params || {
            pitch: 0,
            speed: 1.0,
            warmth: 0.6,
            expressiveness: 0.7,
            stability: 0.5,
            clarity: 0.6,
            breathiness: 0.3,
            resonance: 0.5,
            emotion: 'neutral',
            emphasis: 0.5,
            pauses: 0.5,
            energy: 0.6,
            seed: undefined,
          };
          
          // Use the unified createVoice API with Qwen3
          const response = await ttsService.createVoice(text, {
            reference_audio: referenceAudio,
            reference_text: referenceText,
            // Basic tuning
            pitch_shift: params.pitch || 0,
            speed: (params.speed || 1.0) * settings.speech_rate,
            // Voice characteristics
            warmth: params.warmth ?? 0.6,
            expressiveness: params.expressiveness ?? 0.7,
            stability: params.stability ?? 0.5,
            clarity: (params as any).clarity ?? 0.6,
            breathiness: (params as any).breathiness ?? 0.3,
            resonance: (params as any).resonance ?? 0.5,
            // Speech characteristics
            emotion: (params as any).emotion || 'neutral',
            emphasis: (params as any).emphasis ?? 0.5,
            pauses: (params as any).pauses ?? 0.5,
            energy: (params as any).energy ?? 0.6,
            // Audio effects
            reverb: (params as any).reverb ?? 0.0,
            eq_low: (params as any).eq_low ?? 0.5,
            eq_mid: (params as any).eq_mid ?? 0.5,
            eq_high: (params as any).eq_high ?? 0.5,
            compression: (params as any).compression ?? 0.3,
            // Engine selection
            engine: 'qwen3',
            qwen3_model_size: (settings.qwen3_model_size as Qwen3ModelSize) || '0.6B',
            use_flash_attention: settings.qwen3_flash_attention !== false,
            seed: params.seed,
          });
          
          audioData = response.audio_data;
        } catch (error) {
          toast.error(`Voice creation failed: ${error instanceof Error ? error.message : 'Unknown error'}. TTS disabled.`);
          return;
        }
      }
      
      // PHASE 2: PLAYBACK - Output text and audio simultaneously
      setIsGeneratingVoice(false);
      
      if (audioData) {
        // Get persona's voice tuning for post-processing
        const { getPersonaVoiceTuning } = useStore.getState();
        const voiceTuning = persona ? getPersonaVoiceTuning(persona.id) : null;
        
        if (voiceTuning) {
          // Apply post-processing effects during playback

          try {
            // Show text immediately when audio starts
            onBeforePlay?.(text);
            setIsSpeaking(true, text, audioData);
            
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

            // Fall back to global player
            globalPlayAudio({
              data: audioData,
              title: "AI Response",
              source: 'tts'
            });
            
            // Show text immediately
            onBeforePlay?.(text);
            setIsSpeaking(true, text, audioData);
            
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
          
          // Show text immediately
          onBeforePlay?.(text);
          setIsSpeaking(true, text, audioData);
          
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
      } else if (useKittenTTS) {
        // KittenTTS (Local) - Uses audio-based lip sync like Qwen3
        // Get fresh settings at generation time to pick up voice changes
        const currentSettings = useStore.getState().settings;
        const voice = currentSettings.kitten_voice || "Bella";
        const model = currentSettings.kitten_model || "nano";
        
        try {
          // PHASE 1: GENERATION - Generate audio first (blocking, no UI updates yet)
          const speed = currentSettings.kitten_speed || 1.0;

          const kittenResponse = await ttsService.generateKittenTTS(
            text,
            voice,
            model,
            speed
          );
          audioData = kittenResponse.audio_data;

          
          // PHASE 2: PLAYBACK - Everything starts together
          // 1. Start audio playback (but don't show UI yet)
          globalPlayAudio({
            data: audioData,
            title: "AI Response",
            source: 'tts'
          });
          
          // 2. Wait for audio to actually start playing
          await new Promise<void>((resolve) => {
            const checkStarted = setInterval(() => {
              const state = useStore.getState();
              if (state.audioPlayer.isPlaying && state.audioPlayer.audioData === audioData) {
                clearInterval(checkStarted);
                resolve();
              }
            }, 50);
            // Timeout after 500ms to prevent hanging
            setTimeout(() => { clearInterval(checkStarted); resolve(); }, 500);
          });
          
          // 3. NOW show text and start lip sync together with audio

          onBeforePlay?.(text);
          setIsSpeaking(true, text, audioData);
          
          // 4. Wait for playback to complete
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
          toast.error(`KittenTTS failed: ${kittenError instanceof Error ? kittenError.message : 'Unknown error'}`);
          // Show text even if TTS failed
          onBeforePlay?.(text);
        }
      } else {
        // This should not happen - if we get here, tts_engine is not set correctly
        toast.error('Please select Qwen3-TTS or KittenTTS in settings');
        onBeforePlay?.(text);
      }
    } catch (error) {
      // Still show text even if audio failed
      onBeforePlay?.(text);
    } finally {
      setIsGeneratingVoice(false);
      setIsSpeaking(false);
    }
  };

  const handleSend = async (isVoiceInput = false) => {
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
      inputType: isVoiceInput ? "voice" : "text",
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


    try {
      // Generate text response (but don't show it yet)

      const response = await generateResponse(userMessage.content, imagesToSend, filesToSend, personaAtStart, userMessage.inputType || "text");


      if (settings.enable_memory) {
        // Update memory service thresholds from settings
        memoryService.setThreshold(settings.memory_summarize_threshold);
        memoryService.setImportanceThreshold(settings.memory_importance_threshold || 0.5);
        
        // Use memory service to add message (respects importance threshold)
        const userMessageForMemory: ChatMessage = { role: 'user', content: messageContent };
        memoryService.addMessage(personaAtStart, userMessageForMemory, settings.default_model)
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

        // ALSO save to file-based conversation history for persistent storage
        // Save user message
        memoryToolsService.saveConversationMessage(
          personaAtStart.id,
          "user",
          messageContent,
          imagesToSend.length > 0 ? "image" : filesToSend.length > 0 ? "file" : "text",
          { 
            hasImages: imagesToSend.length > 0,
            hasFiles: filesToSend.length > 0,
            inputType: userMessage.inputType || "text"
          }
        ).catch(err => console.error("Failed to save user message to history:", err));

        // Save assistant response
        memoryToolsService.saveConversationMessage(
          personaAtStart.id,
          "assistant",
          response,
          "text",
          { inputType: "response" }
        ).catch(err => console.error("Failed to save assistant response to history:", err));
      }
      
      // Unified flow: Match voice input behavior from WakeWordListener
      if (settings.tts_engine !== 'off') {
        try {
          // Generate voice and output both text and audio
          // Callback is called when audio starts playing
          await speakResponse(response, personaAtStart, (text) => {
            addMessage({ role: "assistant", content: text });
          });
        } catch (ttsError) {
          // TTS failed - show text anyway
          console.error("TTS failed:", ttsError);
          addMessage({ role: "assistant", content: response });
          toast.error("Voice generation failed", {
            description: "Showing text response only."
          });
        }
      } else {
        // TTS off - show text immediately
        addMessage({ role: "assistant", content: response });
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
      handleSend(false);
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
                  {message.isLoading ? (
                    <div className="flex items-center gap-2">
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
                      <span className="text-xs text-muted-foreground">Generating response...</span>
                    </div>
                  ) : (
                    message.content
                  )}
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
        onAttachImage={(base64) => {
          setAttachedImages(prev => [...prev, base64]);
          toast.success("Image attached! 📸");
        }}
      />

      {/* Memory Manager */}
      <MemoryManager
        isOpen={showMemoryManager}
        onClose={() => setShowMemoryManager(false)}
      />
      
      {/* Personality Manager */}
      <PersonalityManager
        isOpen={showPersonalityManager}
        onClose={() => setShowPersonalityManager(false)}
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

      {/* Tool Execution Confirmation */}
      <ToolConfirmationModal
        pending={toolConfirmation}
        isExecuting={isToolExecuting}
        onConfirm={confirmToolExecution}
        onCancel={cancelToolExecution}
      />

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
            
            {/* Personality Manager Button */}
            <Button
              variant="outline"
              size="icon"
              onClick={() => setShowPersonalityManager(true)}
              disabled={isProcessing}
              title="Open Personality Development"
              className="h-8 w-8"
            >
              <Brain className="w-3.5 h-3.5" />
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
            onClick={() => handleSend(false)}
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
              <button
                onClick={() => updateSettings({ tts_engine: 'kitten' })}
                className={`px-2 py-0.5 text-xs rounded transition-colors flex items-center gap-1 ${
                  settings.tts_engine === 'kitten' 
                    ? 'bg-primary text-primary-foreground' 
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                title="KittenTTS - Cloud TTS with multiple voices (Bella, Jasper, Luna, etc.)"
              >
                <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"></path><path d="M8 14s1.5 2 4 2 4-2 4-2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>
                Kitten
              </button>
            </div>
            
            {/* Qwen3 Model Size Toggle - Only show when Qwen3 selected and TTS not off */}
            {settings.tts_engine === 'qwen3' && (
              <div className="flex items-center gap-1 bg-muted rounded-lg p-0.5">
                <button
                  onClick={() => {

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
            
            {/* KittenTTS Model Toggle - Only show when KittenTTS selected */}
            {settings.tts_engine === 'kitten' && (
              <div className="flex items-center gap-1 bg-muted rounded-lg p-0.5">
                <button
                  onClick={() => {

                    updateSettings({ kitten_model: 'nano' });
                  }}
                  className={`px-2 py-0.5 text-xs rounded transition-colors flex items-center gap-1 ${
                    settings.kitten_model === 'nano' || settings.kitten_model === undefined
                      ? 'bg-primary text-primary-foreground' 
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                  title="Nano - Fastest, 15M params"
                >
                  Nano
                </button>
                <button
                  onClick={() => {

                    updateSettings({ kitten_model: 'micro' });
                  }}
                  className={`px-2 py-0.5 text-xs rounded transition-colors flex items-center gap-1 ${
                    settings.kitten_model === 'micro' 
                      ? 'bg-primary text-primary-foreground' 
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                  title="Micro - 40M params"
                >
                  Micro
                </button>
                <button
                  onClick={() => {

                    updateSettings({ kitten_model: 'mini' });
                  }}
                  className={`px-2 py-0.5 text-xs rounded transition-colors flex items-center gap-1 ${
                    settings.kitten_model === 'mini' 
                      ? 'bg-primary text-primary-foreground' 
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                  title="Mini - Best quality, 80M params"
                >
                  Mini
                </button>
              </div>
            )}
            
            {/* KittenTTS Speed Control */}
            {settings.tts_engine === 'kitten' && (
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs text-muted-foreground">Speed:</span>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={settings.kitten_speed || 1.0}
                  onChange={(e) => updateSettings({ kitten_speed: parseFloat(e.target.value) })}
                  className="w-24 h-1 bg-muted rounded-lg appearance-none cursor-pointer"
                  title={`Speech speed: ${(settings.kitten_speed || 1.0).toFixed(1)}x`}
                />
                <span className="text-xs text-muted-foreground w-8">{(settings.kitten_speed || 1.0).toFixed(1)}x</span>
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

