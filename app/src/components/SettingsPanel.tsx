import { useState, useEffect, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { 
  Server, 
  Volume2, 
  Mic, 
  Eye, 
  Globe,
  RefreshCw,
  Check,
  AlertCircle,
  Cpu,
  Brain,
  MemoryStick,
  Trash2,
  Activity,
  Zap,
  Wifi,
  WifiOff
} from "lucide-react";
import { useStore } from "@/store";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Slider } from "./ui/slider";
import { Switch } from "./ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { toast } from "sonner";
import { ollamaService } from "@/services/ollama";
import { ttsService } from "@/services/tts";
import { searxngService } from "@/services/searxng";

// SearXNG Status Component
function SearXNGStatus() {
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  const checkStatus = async () => {
    setStatus('checking');
    const available = await searxngService.checkStatus();
    setStatus(available ? 'online' : 'offline');
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="mt-2 p-3 bg-blue-500/10 border border-blue-500/20 rounded">
      <div className="flex items-center gap-2 mb-2">
        {status === 'checking' && <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />}
        {status === 'online' && <Wifi className="w-4 h-4 text-green-400" />}
        {status === 'offline' && <WifiOff className="w-4 h-4 text-red-400" />}
        <span className="text-xs font-medium">
          {status === 'checking' && 'Checking SearXNG...'}
          {status === 'online' && 'SearXNG is running on localhost:8080'}
          {status === 'offline' && 'SearXNG is not running'}
        </span>
      </div>
      
      {status === 'offline' && (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">
            To enable web search, start SearXNG with Docker:
          </p>
          <code className="block p-2 bg-black/30 rounded text-[10px] font-mono text-muted-foreground">
            docker run -d -p 8888:8080 --name mimic-searxng searxng/searxng
          </code>
          <Button 
            size="sm" 
            variant="outline" 
            className="w-full text-xs"
            onClick={checkStatus}
          >
            <RefreshCw className="w-3 h-3 mr-1" />
            Check Again
          </Button>
        </div>
      )}
      
      {status === 'online' && (
        <p className="text-xs text-green-400">
          Web search is ready. Privacy-first search aggregated from multiple engines.
        </p>
      )}
    </div>
  );
}

export function SettingsPanel() {
  const { 
    settings, 
    updateSettings, 
    appState, 
    updateAppState, 
    currentPersona,
    updatePersona,
  } = useStore();
  
  const [isTestingOllama, setIsTestingOllama] = useState(false);
  const [isTestingTTS, setIsTestingTTS] = useState(false);
  
  // Microphone test state
  const [isTestingMic, setIsTestingMic] = useState(false);
  const [micLevel, setMicLevel] = useState(0);
  const [micDevices, setMicDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedMicDevice, setSelectedMicDevice] = useState<string>("");
  const [micTestStatus, setMicTestStatus] = useState<"idle" | "testing" | "success" | "failed">("idle");
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);

  // Load models on mount and when URL changes
  useEffect(() => {
    if (appState.ollama_connected) {
      fetchModels();
    }
  }, [appState.ollama_connected]);

  // Sync web search setting with service
  useEffect(() => {
    searxngService.setEnabled(settings.enable_web_search);
    console.log('[Settings] SearXNG web search enabled:', settings.enable_web_search);
  }, [settings.enable_web_search]);

  const fetchModels = async () => {
    try {
      ollamaService.setBaseUrl(settings.ollama_url);
      const modelList = await ollamaService.listModels();
      // Models are used directly in the dropdown, no need to store in state
      console.log("[Settings] Loaded models:", modelList.map(m => m.name));
    } catch (error) {
      console.error("Failed to fetch models:", error);
      toast.error("Failed to fetch models from Ollama");
    }
  };

  const testOllamaConnection = async () => {
    setIsTestingOllama(true);
    try {
      ollamaService.setBaseUrl(settings.ollama_url);
      const connected = await ollamaService.checkConnection();
      updateAppState({ ollama_connected: connected });
      
      if (connected) {
        toast.success("Successfully connected to Ollama");
        await fetchModels();
      } else {
        toast.error("Could not connect to Ollama. Please check your settings.");
      }
    } catch (error) {
      updateAppState({ ollama_connected: false });
      toast.error("Connection test failed");
    } finally {
      setIsTestingOllama(false);
    }
  };

  const testTTSConnection = async () => {
    setIsTestingTTS(true);
    try {
      ttsService.setBaseUrl(settings.tts_backend_url);
      const connected = await ttsService.checkConnection();
      updateAppState({ tts_backend_connected: connected });
      
      if (connected) {
        toast.success("Successfully connected to TTS backend");
      } else {
        toast.error("Could not connect to TTS backend.");
      }
    } catch (error) {
      updateAppState({ tts_backend_connected: false });
      toast.error("TTS connection test failed");
    } finally {
      setIsTestingTTS(false);
    }
  };

  const clearPersonaMemory = () => {
    if (!currentPersona) return;
    
    if (!confirm(`Are you sure you want to clear ${currentPersona.name}'s memory? This cannot be undone.`)) {
      return;
    }

    // Get fresh persona data from store to avoid overwriting voice_create
    const freshPersona = useStore.getState().personas.find(p => p.id === currentPersona.id);
    if (!freshPersona) return;

    updatePersona({
      ...freshPersona,
      memory: {
        short_term: [],
        long_term: [],
        summary: "",
        last_summarized: new Date().toISOString(),
      },
    });

    toast.success(`${freshPersona.name}'s memory has been cleared`);
  };

  // Load microphone devices
  useEffect(() => {
    const loadDevices = async () => {
      try {
        // Request permission first to get device labels
        await navigator.mediaDevices.getUserMedia({ audio: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(d => d.kind === "audioinput");
        setMicDevices(audioInputs);
        
        // Use setting if available, otherwise use first device
        if (settings.microphone_device) {
          setSelectedMicDevice(settings.microphone_device);
        } else if (audioInputs.length > 0 && !selectedMicDevice) {
          // Prefer Blue Snowball if available
          const snowball = audioInputs.find(d => d.label.toLowerCase().includes("snowball"));
          const defaultDevice = audioInputs.find(d => d.deviceId === "default");
          setSelectedMicDevice(snowball?.deviceId || defaultDevice?.deviceId || audioInputs[0].deviceId);
        }
      } catch (error) {
        console.error("Failed to load microphone devices:", error);
      }
    };
    loadDevices();
  }, [settings.microphone_device]);

  // Cleanup mic test on unmount
  useEffect(() => {
    return () => {
      stopMicTest();
    };
  }, []);

  const stopMicTest = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsTestingMic(false);
    setMicLevel(0);
  }, []);

  const testMicrophone = async () => {
    setIsTestingMic(true);
    setMicTestStatus("testing");
    setMicLevel(0);
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: selectedMicDevice ? { exact: selectedMicDevice } : undefined,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });
      streamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      let maxLevel = 0;
      let samples = 0;

      const checkLevel = () => {
        if (!analyserRef.current) return;
        
        analyserRef.current.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        const normalizedLevel = Math.min(100, (average / 128) * 100);
        
        setMicLevel(normalizedLevel);
        maxLevel = Math.max(maxLevel, normalizedLevel);
        samples++;

        // Test for ~3 seconds
        if (samples < 180) {
          animationRef.current = requestAnimationFrame(checkLevel);
        } else {
          // Test complete
          stopMicTest();
          if (maxLevel > 10) {
            setMicTestStatus("success");
            toast.success(`Microphone working! Peak level: ${Math.round(maxLevel)}%`);
          } else {
            setMicTestStatus("failed");
            toast.error("No audio detected. Please check your microphone isn't muted.");
          }
        }
      };

      animationRef.current = requestAnimationFrame(checkLevel);
    } catch (error: any) {
      stopMicTest();
      setMicTestStatus("failed");
      if (error.name === "NotAllowedError") {
        toast.error("Microphone permission denied.");
      } else if (error.name === "NotFoundError") {
        toast.error("No microphone found.");
      } else {
        toast.error(`Microphone error: ${error.message}`);
      }
    }
  };

  return (
    <div className="h-full overflow-auto p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-8"
      >
        <div>
          <h2 className="text-2xl font-bold mb-2">Settings</h2>
          <p className="text-muted-foreground">Configure your Mimic assistant</p>
        </div>

        {/* Ollama Settings */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-lg font-semibold">
            <Brain className="w-5 h-5" />
            <h3>Ollama Configuration</h3>
            {appState.ollama_connected && (
              <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full">
                Connected
              </span>
            )}
          </div>
          
          <div className="space-y-4 pl-7">
            <div className="space-y-2">
              <Label>Ollama URL</Label>
              <div className="flex gap-2">
                <Input
                  value={settings.ollama_url}
                  onChange={(e) => {
                    updateSettings({ ollama_url: e.target.value });
                    updateAppState({ ollama_connected: false });
                  }}
                  placeholder="http://localhost:11434"
                />
                <Button
                  variant="outline"
                  onClick={testOllamaConnection}
                  disabled={isTestingOllama}
                >
                  {isTestingOllama ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : appState.ollama_connected ? (
                    <Check className="w-4 h-4 text-green-500" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-amber-500" />
                  )}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Make sure Ollama is running. Default: http://localhost:11434
              </p>
            </div>

            {!appState.ollama_connected && (
              <div className="text-sm text-muted-foreground bg-muted p-3 rounded">
                <p className="font-medium mb-1">Getting Started with Ollama:</p>
                <ol className="list-decimal list-inside space-y-1">
                  <li>Install Ollama from <a href="https://ollama.ai" target="_blank" rel="noopener noreferrer" className="text-primary underline">ollama.ai</a></li>
                  <li>Start Ollama (it runs automatically after install)</li>
                  <li>Pull a model: <code className="bg-secondary px-1 rounded">ollama pull llama3.2</code></li>
                  <li>Click the test button above to connect</li>
                </ol>
              </div>
            )}
          </div>
        </section>

        {/* TTS Backend Settings */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-lg font-semibold">
            <Server className="w-5 h-5" />
            <h3>TTS Backend</h3>
            {appState.tts_backend_connected && (
              <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full">
                Connected
              </span>
            )}
          </div>
          
          <div className="space-y-4 pl-7">
            <div className="space-y-2">
              <Label>TTS Backend URL</Label>
              <div className="flex gap-2">
                <Input
                  value={settings.tts_backend_url}
                  onChange={(e) => {
                    updateSettings({ tts_backend_url: e.target.value });
                    updateAppState({ tts_backend_connected: false });
                  }}
                  placeholder="http://localhost:8000"
                />
                <Button
                  variant="outline"
                  onClick={testTTSConnection}
                  disabled={isTestingTTS}
                >
                  {isTestingTTS ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : appState.tts_backend_connected ? (
                    <Check className="w-4 h-4 text-green-500" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-amber-500" />
                  )}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Python backend for Qwen3-TTS voice creation. Default: http://localhost:8000
              </p>
            </div>

            {!appState.tts_backend_connected && (
              <div className="text-sm text-muted-foreground bg-muted p-3 rounded">
                <p className="font-medium mb-1">TTS Backend Setup:</p>
                <ol className="list-decimal list-inside space-y-1">
                  <li>Install Python dependencies: <code className="bg-secondary px-1 rounded">pip install -r requirements.txt</code></li>
                  <li>Start the backend: <code className="bg-secondary px-1 rounded">python backend/tts_server.py</code></li>
                  <li>Click the test button to verify connection</li>
                </ol>
              </div>
            )}
          </div>
        </section>

        {/* Voice Settings */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-lg font-semibold">
            <Volume2 className="w-5 h-5" />
            <h3>Voice Settings</h3>
          </div>
          
          <div className="space-y-6 pl-7">
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Volume</Label>
                <span className="text-sm text-muted-foreground">
                  {Math.round(settings.voice_volume * 100)}%
                </span>
              </div>
              <Slider
                value={[settings.voice_volume * 100]}
                onValueChange={([value]) => updateSettings({ voice_volume: value / 100 })}
                max={100}
                step={5}
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Speech Rate</Label>
                <span className="text-sm text-muted-foreground">
                  {settings.speech_rate.toFixed(1)}x
                </span>
              </div>
              <Slider
                value={[settings.speech_rate * 100]}
                onValueChange={([value]) => updateSettings({ speech_rate: value / 100 })}
                min={50}
                max={200}
                step={10}
              />
            </div>

            <div className="space-y-2">
              <Label>TTS Mode</Label>
              <Select 
                value={settings.tts_mode} 
                onValueChange={(value: "browser" | "qwen3" | "auto") => updateSettings({ tts_mode: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="browser">
                    <div className="flex flex-col items-start">
                      <span>Browser TTS (Fastest)</span>
                      <span className="text-xs text-muted-foreground">Always use browser TTS - fastest responses</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="auto">
                    <div className="flex flex-col items-start">
                      <span>Auto (Recommended)</span>
                      <span className="text-xs text-muted-foreground">Qwen3 for created voices, Browser for others</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="qwen3">
                    <div className="flex flex-col items-start">
                      <span>Qwen3-TTS (High Quality)</span>
                      <span className="text-xs text-muted-foreground">Always use Qwen3 backend - slower but best quality</span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Requires TTS Backend running for Qwen3 modes.
              </p>
            </div>

            {/* TTS Engine Selection */}
            <div className="space-y-2 pt-4 border-t">
              <Label className="flex items-center gap-2">
                <Cpu className="w-4 h-4" />
                TTS Engine
              </Label>
              <Select 
                value={settings.tts_engine || "styletts2"} 
                onValueChange={(value: "styletts2" | "qwen3" | "off") => updateSettings({ tts_engine: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="off">
                    <div className="flex flex-col items-start">
                      <span>Off</span>
                      <span className="text-xs text-muted-foreground">Text only, no voice output</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="styletts2">
                    <div className="flex flex-col items-start">
                      <span>StyleTTS2</span>
                      <span className="text-xs text-muted-foreground">Fast, lightweight, works without reference</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="qwen3">
                    <div className="flex flex-col items-start">
                      <span>Qwen3-TTS</span>
                      <span className="text-xs text-muted-foreground">Higher quality, requires reference audio</span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Qwen3 Voice Warning */}
            {settings.tts_engine === "qwen3" && !currentPersona?.voice_create?.has_audio && (
              <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg mt-4">
                <p className="text-sm text-amber-400 font-medium">Qwen3 Requires a Voice</p>
                <p className="text-xs text-muted-foreground mt-1">
                  The current persona ({currentPersona?.name || "Unknown"}) does not have a created voice. 
                  Qwen3 will fallback to StyleTTS2 until you create a voice in Voice Studio.
                </p>
              </div>
            )}

            {/* Qwen3 Options */}
            {settings.tts_engine === "qwen3" && (
              <div className="space-y-4 pt-4 border-t">
                <div className="space-y-2">
                  <Label>Qwen3 Model Size</Label>
                  <Select 
                    value={settings.qwen3_model_size || "0.6B"} 
                    onValueChange={(value: "0.6B" | "1.7B") => updateSettings({ qwen3_model_size: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0.6B">
                        <div className="flex flex-col items-start">
                          <span>0.6B (Fast)</span>
                          <span className="text-xs text-muted-foreground">~3GB VRAM, faster inference</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="1.7B">
                        <div className="flex flex-col items-start">
                          <span>1.7B (Quality)</span>
                          <span className="text-xs text-muted-foreground">~6GB VRAM, higher quality</span>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label className="flex items-center gap-2">
                      <Zap className="w-4 h-4" />
                      Flash Attention
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      Reduces VRAM usage. Disable if you encounter errors.
                    </p>
                  </div>
                  <Switch
                    checked={settings.qwen3_flash_attention !== false}
                    onCheckedChange={(checked) => updateSettings({ qwen3_flash_attention: checked })}
                  />
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Memory Settings */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-lg font-semibold">
            <MemoryStick className="w-5 h-5" />
            <h3>Memory</h3>
          </div>
          
          <div className="space-y-4 pl-7">
            <div className="flex items-center justify-between">
              <div>
                <Label>Enable Memory</Label>
                <p className="text-sm text-muted-foreground">
                  Personas remember conversations and build context over time
                </p>
              </div>
              <Switch
                checked={settings.enable_memory}
                onCheckedChange={(checked) => updateSettings({ enable_memory: checked })}
              />
            </div>

            {settings.enable_memory && (
              <>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label>Importance Threshold</Label>
                      <span className="text-sm text-muted-foreground">
                        {Math.round((settings.memory_importance_threshold || 0.5) * 100)}%
                      </span>
                    </div>
                    <Slider
                      value={[settings.memory_importance_threshold || 0.5]}
                      onValueChange={([value]) => updateSettings({ memory_importance_threshold: value })}
                      min={0}
                      max={1}
                      step={0.1}
                    />
                    <p className="text-xs text-muted-foreground">
                      Only store memories with importance above this threshold (0% = store everything, 100% = store only critical info)
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label>Summarize After</Label>
                      <span className="text-sm text-muted-foreground">
                        {settings.memory_summarize_threshold} messages
                      </span>
                    </div>
                    <Slider
                      value={[settings.memory_summarize_threshold]}
                      onValueChange={([value]) => updateSettings({ memory_summarize_threshold: value })}
                      min={5}
                      max={50}
                      step={5}
                    />
                    <p className="text-xs text-muted-foreground">
                      Automatically summarize conversation history when threshold is reached
                    </p>
                  </div>
                </div>
              </>
            )}

            {currentPersona && settings.enable_memory && (
              <div className="flex items-center justify-between p-3 bg-muted rounded">
                <div>
                  <p className="font-medium">{currentPersona.name}'s Memory</p>
                  <p className="text-sm text-muted-foreground">
                    {currentPersona.memory.short_term.length} short-term entries
                    {currentPersona.memory.long_term.length > 0 && 
                      ` • ${currentPersona.memory.long_term.length} long-term memories`
                    }
                  </p>
                </div>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={clearPersonaMemory}
                >
                  <Trash2 className="w-4 h-4 mr-1" />
                  Clear
                </Button>
              </div>
            )}

            <div className="flex items-center justify-between">
              <div>
                <Label>Enable Web Search (SearXNG)</Label>
                <p className="text-sm text-muted-foreground">
                  Allow AI to search the web using local SearXNG instance
                </p>
                {settings.enable_web_search && <SearXNGStatus />}
              </div>
              <Switch
                checked={settings.enable_web_search}
                onCheckedChange={(checked) => {
                  updateSettings({ enable_web_search: checked });
                  searxngService.setEnabled(checked);
                  toast.success(checked ? "SearXNG web search enabled" : "Web search disabled");
                }}
              />
            </div>
          </div>
        </section>

        {/* Wake Word Settings */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-lg font-semibold">
            <Mic className="w-5 h-5" />
            <h3>Wake Word Detection</h3>
          </div>
          
          <div className="space-y-6 pl-7">
            <div className="flex items-center justify-between">
              <div>
                <Label>Auto Listen</Label>
                <p className="text-sm text-muted-foreground">
                  Automatically listen for wake word
                </p>
                <p className="text-xs text-amber-400 mt-1">
                  Note: Requires Chrome/Edge. Brave blocks speech recognition.
                </p>
              </div>
              <Switch
                checked={settings.auto_listen}
                onCheckedChange={(checked) => updateSettings({ auto_listen: checked })}
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Wake Word Sensitivity</Label>
                <span className="text-sm text-muted-foreground">
                  {Math.round(settings.wake_word_sensitivity * 100)}%
                </span>
              </div>
              <Slider
                value={[settings.wake_word_sensitivity * 100]}
                onValueChange={([value]) => updateSettings({ wake_word_sensitivity: value / 100 })}
                max={100}
                step={5}
              />
            </div>

            {/* Microphone Test */}
            <div className="space-y-3 border rounded-lg p-4 bg-muted/50">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4" />
                <Label className="text-sm font-medium">Microphone Test</Label>
              </div>
              
              {micDevices.length > 0 && (
                <div className="space-y-2">
                  <Select 
                    value={selectedMicDevice} 
                    onValueChange={(value) => {
                      setSelectedMicDevice(value);
                      updateSettings({ microphone_device: value });
                      toast.success("Microphone selected. Refresh page to apply.");
                    }}
                  >
                    <SelectTrigger className="text-sm">
                      <SelectValue placeholder="Select microphone" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">🎤 System Default</SelectItem>
                      {micDevices
                        .filter(d => d.deviceId !== "default" && d.deviceId !== "communications")
                        .map((device) => (
                          <SelectItem key={device.deviceId} value={device.deviceId}>
                            {device.label.includes("Snowball") && "🔵 "}
                            {device.label || `Microphone ${micDevices.indexOf(device) + 1}`}
                          </SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              <div className="flex items-center gap-3">
                <Button
                  variant={isTestingMic ? "destructive" : "outline"}
                  size="sm"
                  onClick={isTestingMic ? stopMicTest : testMicrophone}
                  className="flex-1"
                >
                  {isTestingMic ? (
                    <>
                      <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                      Stop Test
                    </>
                  ) : micTestStatus === "success" ? (
                    <>
                      <Check className="w-4 h-4 mr-2 text-green-500" />
                      Test Again
                    </>
                  ) : (
                    <>
                      <Mic className="w-4 h-4 mr-2" />
                      Test Microphone
                    </>
                  )}
                </Button>
              </div>

              {/* Audio Level Meter */}
              {isTestingMic && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Speak now...</span>
                    <span>{Math.round(micLevel)}%</span>
                  </div>
                  <div className="h-2 bg-secondary rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-green-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${micLevel}%` }}
                      transition={{ duration: 0.1 }}
                    />
                  </div>
                </div>
              )}

              {micTestStatus === "failed" && (
                <div className="text-xs text-red-400 bg-red-500/10 p-2 rounded">
                  <p className="font-medium mb-1">No audio detected. Try these fixes:</p>
                  <ul className="list-disc list-inside space-y-1">
                    <li><strong>VoiceMeeter users:</strong> Set Blue Snowball as Input Device in VoiceMeeter, OR disable VoiceMeeter temporarily</li>
                    <li>Check Windows Sound Settings → Recording → Blue Snowball set as Default</li>
                    <li>Check the mic isn't muted (physical mute switch on Blue Snowball)</li>
                    <li>Try selecting Blue Snowball directly from the dropdown above</li>
                    <li>Close other apps that might be using the microphone</li>
                  </ul>
                </div>
              )}

              {micTestStatus === "success" && (
                <div className="text-xs text-green-400 bg-green-500/10 p-2 rounded flex items-center gap-2">
                  <Check className="w-4 h-4" />
                  Microphone is working correctly!
                </div>
              )}

              <p className="text-xs text-muted-foreground">
                If speech recognition isn't working, use this test to verify your microphone is set up correctly.
              </p>
            </div>
          </div>
        </section>

        {/* Display Settings */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-lg font-semibold">
            <Eye className="w-5 h-5" />
            <h3>Display</h3>
          </div>
          
          <div className="space-y-4 pl-7">
            <div className="flex items-center justify-between">
              <div>
                <Label>Show Avatar</Label>
                <p className="text-sm text-muted-foreground">
                  Display the 3D avatar in chat
                </p>
              </div>
              <Switch
                checked={settings.show_avatar}
                onCheckedChange={(checked) => updateSettings({ show_avatar: checked })}
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label>Dark Mode</Label>
                <p className="text-sm text-muted-foreground">
                  Use dark theme
                </p>
              </div>
              <Switch
                checked={settings.theme === "dark"}
                onCheckedChange={(checked) => updateSettings({ theme: checked ? "dark" : "light" })}
              />
            </div>
          </div>
        </section>

        {/* Language Settings */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-lg font-semibold">
            <Globe className="w-5 h-5" />
            <h3>Language</h3>
          </div>
          
          <div className="pl-7">
            <Select
              value={settings.language}
              onValueChange={(value) => updateSettings({ language: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select language" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="en">English</SelectItem>
                <SelectItem value="zh">Chinese</SelectItem>
                <SelectItem value="ja">Japanese</SelectItem>
                <SelectItem value="ko">Korean</SelectItem>
                <SelectItem value="de">German</SelectItem>
                <SelectItem value="fr">French</SelectItem>
                <SelectItem value="es">Spanish</SelectItem>
                <SelectItem value="it">Italian</SelectItem>
                <SelectItem value="pt">Portuguese</SelectItem>
                <SelectItem value="ru">Russian</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </section>

        {/* System Info */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-lg font-semibold">
            <Cpu className="w-5 h-5" />
            <h3>System Info</h3>
          </div>
          
          <div className="pl-7 space-y-2 text-sm text-muted-foreground">
            <div className="grid grid-cols-2 gap-2">
              <span>Mimic Assistant</span>
              <span>v1.0.0</span>
              
              <span>Ollama Status</span>
              <span className={appState.ollama_connected ? "text-green-400" : "text-red-400"}>
                {appState.ollama_connected ? "Connected" : "Disconnected"}
              </span>
              
              <span>TTS Backend</span>
              <span className={appState.tts_backend_connected ? "text-green-400" : "text-red-400"}>
                {appState.tts_backend_connected ? "Connected" : "Disconnected"}
              </span>
              
              <span>Current Model</span>
              <span>{appState.current_model}</span>
              
              <span>Active Persona</span>
              <span>{currentPersona?.name || "None"}</span>
            </div>
          </div>
        </section>
      </motion.div>
    </div>
  );
}
