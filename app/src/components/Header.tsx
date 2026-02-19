import { motion, AnimatePresence } from "framer-motion";
import { Mic, MicOff, Bot, Bug, ArrowRight, ChevronDown, Brain, Eye, Wifi, WifiOff } from "lucide-react";
import { useStore } from "@/store";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";
import { useState, useEffect, useCallback } from "react";
import { ollamaService } from "@/services/ollama";
import { searxngService } from "@/services/searxng";

interface HeaderProps {
  onToggleDebug?: () => void;
  showDebug?: boolean;
}

interface OllamaModel {
  name: string;
  size?: number;
  modified_at?: string;
}

export function Header({ onToggleDebug, showDebug }: HeaderProps) {
  const { 
    isListening, 
    isSpeaking, 
    isGeneratingVoice,
    setIsListening, 
    currentPersona,
    appState,
    settings,
    updateSettings
  } = useStore();
  
  // Show microphone hint on first visit
  const [showMicHint, setShowMicHint] = useState(false);
  
  // Model selection state
  const [availableModels, setAvailableModels] = useState<OllamaModel[]>([]);
  const [showBrainDropdown, setShowBrainDropdown] = useState(false);
  const [showVisionDropdown, setShowVisionDropdown] = useState(false);
  const [searxngStatus, setSearxngStatus] = useState<boolean>(false);
  
  // Fetch available models
  const fetchModels = useCallback(async () => {
    try {
      ollamaService.setBaseUrl(settings.ollama_url);
      const models = await ollamaService.listModels();
      setAvailableModels(models);
    } catch (error) {
      console.error("Failed to fetch models:", error);
    }
  }, [settings.ollama_url]);
  
  useEffect(() => {
    fetchModels();
    // Refresh every 30 seconds
    const interval = setInterval(fetchModels, 30000);
    return () => clearInterval(interval);
  }, [fetchModels]);
  
  // Check SearXNG status with retry logic for startup
  useEffect(() => {
    let retryCount = 0;
    const maxRetries = 12; // Retry for up to ~2 minutes (exponential backoff)
    let timeoutId: ReturnType<typeof setTimeout>;
    
    const checkSearxng = async () => {
      const available = await searxngService.checkStatus();
      setSearxngStatus(available);
      
      // If not available and we haven't maxed out retries, retry with exponential backoff
      if (!available && retryCount < maxRetries) {
        retryCount++;
        const delay = Math.min(1000 * Math.pow(1.5, retryCount), 15000); // Max 15 seconds between retries
        console.log(`[SearXNG] Not ready, retrying in ${Math.round(delay/1000)}s (attempt ${retryCount}/${maxRetries})`);
        timeoutId = setTimeout(checkSearxng, delay);
      } else if (available) {
        console.log('[SearXNG] Connected successfully');
        retryCount = 0; // Reset retry count on success
      } else {
        console.log('[SearXNG] Max retries reached, will check again in 30s');
      }
    };
    
    // Start checking immediately
    checkSearxng();
    
    // Also check periodically to detect if it goes down or comes back up
    const interval = setInterval(() => {
      retryCount = 0; // Reset retry count for periodic checks
      checkSearxng();
    }, 30000);
    
    return () => {
      clearInterval(interval);
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, []);
  
  useEffect(() => {
    // Check if user has seen the hint before
    const hasSeenHint = localStorage.getItem('mimic_mic_hint_seen');
    if (!hasSeenHint && !isListening) {
      setShowMicHint(true);
    }
  }, [isListening]);
  
  // Hide hint when listening starts
  useEffect(() => {
    if (isListening && showMicHint) {
      setShowMicHint(false);
      localStorage.setItem('mimic_mic_hint_seen', 'true');
    }
  }, [isListening, showMicHint]);

  const toggleListening = () => {
    setIsListening(!isListening);
    // Hide hint when user clicks the button
    if (showMicHint) {
      setShowMicHint(false);
      localStorage.setItem('mimic_mic_hint_seen', 'true');
    }
  };
  
  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      setShowBrainDropdown(false);
      setShowVisionDropdown(false);
    };
    if (showBrainDropdown || showVisionDropdown) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [showBrainDropdown, showVisionDropdown]);

  return (
    <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm flex items-center justify-between px-4 lg:px-6 relative z-50">
      {/* Left: Persona Info */}
      <div className="flex items-center gap-3">
        <motion.div 
          className="w-10 h-10 rounded-full flex items-center justify-center shrink-0"
          style={{ 
            background: `linear-gradient(135deg, ${currentPersona?.avatar_config.primary_color || '#6366f1'}, ${currentPersona?.avatar_config.secondary_color || '#8b5cf6'})` 
          }}
          animate={isSpeaking ? { scale: [1, 1.1, 1] } : {}}
          transition={{ duration: 0.3, repeat: isSpeaking ? Infinity : 0 }}
        >
          <Bot className="w-5 h-5 text-white" />
        </motion.div>
        <div className="min-w-0">
          <h2 className="font-semibold text-foreground truncate">
            {currentPersona?.name || "Mimic"}
          </h2>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className={appState.ollama_connected ? "text-green-400" : "text-amber-400"}>
              {appState.ollama_connected ? "● Connected to Ollama" : "● Offline"}
            </span>
            {settings.enable_web_search && (
              <span className={searxngStatus ? "text-green-400" : "text-red-400"} title={searxngStatus ? "SearXNG ready" : "SearXNG not running"}>
                {searxngStatus ? <Wifi className="w-3 h-3 inline" /> : <WifiOff className="w-3 h-3 inline" />}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Center: Model Selection Dropdowns */}
      <div className="hidden md:flex items-center gap-2">
        {/* Brain Model Dropdown */}
        <div className="relative">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowBrainDropdown(!showBrainDropdown);
              setShowVisionDropdown(false);
            }}
            className="flex items-center gap-2 px-3 py-1.5 bg-muted/50 hover:bg-muted rounded-lg text-sm transition-colors"
          >
            <Brain className="w-4 h-4 text-primary" />
            <span className="max-w-[120px] truncate">
              {settings.default_model || 'Select model'}
            </span>
            <ChevronDown className="w-3 h-3 text-muted-foreground" />
          </button>
          
          <AnimatePresence>
            {showBrainDropdown && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute top-full left-0 mt-1 w-64 bg-popover border rounded-lg shadow-[0_8px_30px_rgb(0,0,0,0.4)] z-[9999] max-h-64 overflow-y-auto"
              >
                <div className="p-2">
                  <p className="text-xs text-muted-foreground px-2 py-1">Brain Model (Text)</p>
                  {availableModels.length === 0 ? (
                    <div className="px-2 py-2">
                      <p className="text-xs text-muted-foreground">No models found</p>
                      <p className="text-xs text-amber-400 mt-1">Run: ollama pull llama3.2</p>
                    </div>
                  ) : (
                    availableModels.map((model) => (
                      <button
                        key={model.name}
                        onClick={() => {
                          updateSettings({ default_model: model.name });
                          setShowBrainDropdown(false);
                        }}
                        className={cn(
                          "w-full text-left px-3 py-2 rounded text-sm transition-colors",
                          settings.default_model === model.name
                            ? "bg-primary text-primary-foreground"
                            : "hover:bg-muted"
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <span className="truncate">{model.name}</span>
                          {model.size && (
                            <span className="text-xs opacity-70">
                              {(model.size / 1e9).toFixed(1)}GB
                            </span>
                          )}
                        </div>
                      </button>
                    ))
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Vision Model Dropdown */}
        <div className="relative">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowVisionDropdown(!showVisionDropdown);
              setShowBrainDropdown(false);
            }}
            className="flex items-center gap-2 px-3 py-1.5 bg-muted/50 hover:bg-muted rounded-lg text-sm transition-colors"
          >
            <Eye className="w-4 h-4 text-emerald-400" />
            <span className="max-w-[120px] truncate">
              {!settings.vision_model || settings.vision_model === 'none' || !availableModels.some(m => m.name === settings.vision_model)
                ? 'No vision model' 
                : settings.vision_model}
            </span>
            <ChevronDown className="w-3 h-3 text-muted-foreground" />
          </button>
          
          <AnimatePresence>
            {showVisionDropdown && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute top-full left-0 mt-1 w-64 bg-popover border rounded-lg shadow-[0_8px_30px_rgb(0,0,0,0.4)] z-[9999] max-h-64 overflow-y-auto"
              >
                <div className="p-2">
                  <p className="text-xs text-muted-foreground px-2 py-1">Vision Model (Images)</p>
                  {availableModels.filter(m => 
                    m.name.includes('llava') || 
                    m.name.includes('bakllava') ||
                    m.name.includes('moondream') ||
                    m.name.includes('vision')
                  ).length === 0 ? (
                    <p className="text-xs text-muted-foreground px-2 py-2">No vision models found. Install llava, bakllava, or moondream.</p>
                  ) : (
                    availableModels.filter(m => 
                      m.name.includes('llava') || 
                      m.name.includes('bakllava') ||
                      m.name.includes('moondream') ||
                      m.name.includes('vision')
                    ).map((model) => (
                      <button
                        key={model.name}
                        onClick={() => {
                          updateSettings({ vision_model: model.name });
                          setShowVisionDropdown(false);
                        }}
                        className={cn(
                          "w-full text-left px-3 py-2 rounded text-sm transition-colors",
                          settings.vision_model === model.name
                            ? "bg-emerald-500 text-white"
                            : "hover:bg-muted"
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <span className="truncate">{model.name}</span>
                          {model.size && (
                            <span className="text-xs opacity-70">
                              {(model.size / 1e9).toFixed(1)}GB
                            </span>
                          )}
                        </div>
                      </button>
                    ))
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Right: Status & Controls */}
      <div className="flex items-center gap-2">
        {/* Status Indicators */}
        <div className="hidden sm:flex items-center gap-3 mr-2">
          {isListening && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center gap-2 px-2 py-1 rounded-full bg-primary/10 text-primary text-xs"
            >
              <span className="relative flex h-1.5 w-1.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-primary" />
              </span>
              Listening
            </motion.div>
          )}
          
          {isGeneratingVoice && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center gap-1"
            >
              {[0, 1, 2].map((i) => (
                <motion.div
                  key={i}
                  className="w-1 h-1 bg-primary rounded-full"
                  animate={{ scale: [1, 1.5, 1], opacity: [0.4, 1, 0.4] }}
                  transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.15 }}
                />
              ))}
            </motion.div>
          )}
          
          {isSpeaking && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center gap-0.5"
            >
              {[...Array(4)].map((_, i) => (
                <motion.div
                  key={i}
                  className="w-0.5 bg-gradient-to-t from-indigo-500 to-purple-500 rounded-full"
                  animate={{ height: [4, 16, 4] }}
                  transition={{ duration: 0.5, repeat: Infinity, delay: i * 0.1 }}
                />
              ))}
            </motion.div>
          )}
        </div>

        {/* Microphone Button with Hint */}
        <div className="relative">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleListening}
            className={cn(
              "relative",
              isListening && "text-primary bg-primary/10"
            )}
            title={isListening ? "Stop listening" : "Start listening"}
          >
            {isListening ? (
              <Mic className="w-5 h-5" />
            ) : (
              <MicOff className="w-5 h-5" />
            )}
          </Button>
          
          {/* Microphone hint - pulsing arrow only */}
          <AnimatePresence>
            {showMicHint && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="absolute right-full mr-1 top-1/2 -translate-y-1/2 z-50"
              >
                <motion.div
                  animate={{ x: [0, -6, 0] }}
                  transition={{ duration: 0.8, repeat: Infinity, ease: "easeInOut" }}
                >
                  <ArrowRight className="w-5 h-5 text-primary" />
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {onToggleDebug && (
          <Button 
            variant="ghost" 
            size="icon"
            onClick={onToggleDebug}
            className={cn(
              showDebug && "text-primary bg-primary/10"
            )}
            title="Toggle debug panel"
          >
            <Bug className="w-5 h-5" />
          </Button>
        )}
      </div>
    </header>
  );
}
