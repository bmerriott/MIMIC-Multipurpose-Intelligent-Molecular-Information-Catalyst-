import { useState, useRef, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { 
  Play, 
  Save,
  Wand2,
  Sparkles,
  User,
  Volume2,
  SlidersHorizontal,
  RefreshCw,
  Download,
  AlertCircle,
  Check,
  Mic,
  Upload,
  Cpu,
  Zap,
  Info,
  X,
  Edit3,
  WandSparkles,
  Ear,
  Headphones,
  Shield,
  FileAudio
} from "lucide-react";
import { useStore, type VoiceTuningParams, defaultVoiceTuning } from "@/store";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Textarea } from "./ui/textarea";
import { Slider } from "./ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Switch } from "./ui/switch";
import { toast } from "sonner";
import { ttsService, type TTSEngine, type Qwen3ModelSize } from "@/services/tts";
import { puter } from "@/services/puter";
import { audioEffects, type AudioEffectParams } from "@/services/audioEffects";
import { 
  POST_PROCESSING_PARAMS, 
  requiresRegeneration
} from "@/services/voiceTuning";
import type { Persona } from "@/types";

// Helper: Convert audio blob to WAV format
async function blobToWav(blob: Blob): Promise<Blob> {
  const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
  const arrayBuffer = await blob.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  
  // Convert to mono if stereo
  const numberOfChannels = 1;
  const sampleRate = audioBuffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numberOfChannels * bytesPerSample;
  
  // Get audio data as Float32 and convert to Int16
  const channelData = audioBuffer.getChannelData(0);
  const samples = new Int16Array(channelData.length);
  for (let i = 0; i < channelData.length; i++) {
    const s = Math.max(-1, Math.min(1, channelData[i]));
    samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  
  const dataLength = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);
  
  // Write WAV header
  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataLength, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numberOfChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, 'data');
  view.setUint32(40, dataLength, true);
  
  // Write audio data
  const bytes = new Uint8Array(buffer, 44);
  for (let i = 0; i < samples.length; i++) {
    bytes[i * 2] = samples[i] & 0xFF;
    bytes[i * 2 + 1] = (samples[i] >> 8) & 0xFF;
  }
  
  return new Blob([buffer], { type: 'audio/wav' });
}

interface VoiceParams {
  // Engine
  engine: TTSEngine;
  qwen3Size: Qwen3ModelSize;
  extractionModelSize: "0.6B" | "1.7B";
  useFlashAttention: boolean;
  
  // Reference
  hasReference: boolean;
  referenceAudio: string | null;
  referenceText: string;
  isEditingReferenceText: boolean;
  
  // Basic tuning
  pitchShift: number;
  speed: number;
  seed?: number;
  
  // Voice characteristics
  warmth: number;
  expressiveness: number;
  stability: number;
  clarity: number;
  breathiness: number;
  resonance: number;
  
  // Speech characteristics
  emotion: "neutral" | "happy" | "sad" | "angry" | "excited" | "calm";
  emphasis: number;
  pauses: number;
  energy: number;
  
  // Audio effects
  reverb: number;
  eqLow: number;
  eqMid: number;
  eqHigh: number;
  compression: number;
  
  // Voice profile
  saveAsVoiceProfile: boolean;
  useVoiceProfile: boolean;
}

const defaultParams: VoiceParams = {
  engine: "styletts2",
  qwen3Size: "0.6B",
  extractionModelSize: "1.7B",
  useFlashAttention: true,
  hasReference: false,
  referenceAudio: null,
  referenceText: "",
  isEditingReferenceText: false,
  pitchShift: 0,
  speed: 1.0,
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
  reverb: 0,
  eqLow: 0.5,
  eqMid: 0.5,
  eqHigh: 0.5,
  compression: 0.3,
  saveAsVoiceProfile: false,
  useVoiceProfile: false,
};

const QWEN3_INSTALL_INSTRUCTIONS = `cd app/backend
pip install qwen-tts

# Optional: Flash Attention for lower VRAM
pip install flash-attn --no-build-isolation`;

// Helper function to generate gradient style from persona colors
const getPersonaGradientStyle = (persona: Persona | undefined): React.CSSProperties => {
  if (!persona?.avatar_config) return {};
  
  const primary = persona.avatar_config.primary_color || '#6366f1';
  const secondary = persona.avatar_config.secondary_color || '#8b5cf6';
  
  return {
    background: `linear-gradient(135deg, ${primary}15 0%, ${secondary}10 50%, transparent 100%)`,
    borderColor: `${primary}30`,
  };
};

export function VoiceCreator() {
  const personas = useStore(state => state.personas);
  const currentPersona = useStore(state => state.currentPersona);
  const updatePersonaVoice = useStore(state => state.updatePersonaVoice);
  const settings = useStore(state => state.settings);
  const setCurrentPersona = useStore(state => state.setCurrentPersona);
  const getPersonaVoiceTuning = useStore(state => state.getPersonaVoiceTuning);
  const updatePersonaVoiceTuning = useStore(state => state.updatePersonaVoiceTuning);
  const resetPersonaVoiceTuning = useStore(state => state.resetPersonaVoiceTuning);
  const updateSettings = useStore(state => state.updateSettings);
  
  const [selectedPersonaId, setSelectedPersonaId] = useState(currentPersona?.id || "");
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [engineStatus, setEngineStatus] = useState<{
    styletts2_available: boolean;
    qwen3_available: boolean;
  } | null>(null);
  
  const [params, setParams] = useState<VoiceParams>({ ...defaultParams });
  const [targetText, setTargetText] = useState("Hello! This is my custom voice created with Mimic AI.");
  const [isProcessing, setIsProcessing] = useState(false);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [lastUsedEngine, setLastUsedEngine] = useState<string>("");
  const [isPreviewPlaying, setIsPreviewPlaying] = useState(false);
  const [voiceName, setVoiceName] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [activeTab, setActiveTab] = useState<"create" | "detect">("create");
  const [watermarkDetectionResult, setWatermarkDetectionResult] = useState<{
    detected: boolean;
    confidence: number;
    message: string;
    details?: any;
  } | null>(null);
  const [isDetectingWatermark, setIsDetectingWatermark] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>(() => {
    // Load saved device from localStorage
    return localStorage.getItem("mimic_recording_device") || "";
  });
  const [puterAvailable, setPuterAvailable] = useState(false);
  
  // Preview state
  const [previewTarget, setPreviewTarget] = useState<'generated' | 'persona'>('generated');
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [previewPersonaVoice, setPreviewPersonaVoice] = useState<string | null>(null);
  
  // Voice tuning state
  const [, setOriginalTuning] = useState<VoiceTuningParams>(defaultVoiceTuning);
  const [modifiedParams, setModifiedParams] = useState<Set<keyof VoiceTuningParams>>(new Set());
  const [showRegenNotice, setShowRegenNotice] = useState(false);
  
  // Load voice tuning when persona changes
  // Note: selectedPersona is defined later, so we use selectedPersonaId in deps
  useEffect(() => {
    if (selectedPersonaId) {
      const tuning = getPersonaVoiceTuning(selectedPersonaId);
      setOriginalTuning(tuning);
      setParams(p => ({
        ...p,
        ...tuning,
      }));
      setModifiedParams(new Set());
      setShowRegenNotice(false);
    }
  }, [selectedPersonaId, getPersonaVoiceTuning]);

  // Sync TTS engine settings from global settings on mount
  useEffect(() => {
    setParams(p => ({
      ...p,
      engine: settings.tts_engine || "styletts2",
      extractionModelSize: settings.qwen3_model_size || "0.6B",
      useFlashAttention: settings.qwen3_flash_attention !== false,
    }));
  }, []);


  
  // Track parameter changes
  const updateParam = useCallback((key: keyof VoiceParams, value: any) => {
    setParams(p => {
      const newParams = { ...p, [key]: value };
      
      // Check if this is a voice tuning param
      if (key in defaultVoiceTuning) {
        setModifiedParams(prev => new Set([...prev, key as keyof VoiceTuningParams]));
        
        // Check if synthesis param changed
        if (requiresRegeneration(key as keyof VoiceTuningParams)) {
          setShowRegenNotice(true);
        }
        
        // Save to store immediately for post-processing params
        if (selectedPersonaId && POST_PROCESSING_PARAMS.includes(key as keyof VoiceTuningParams)) {
          updatePersonaVoiceTuning(selectedPersonaId, { [key]: value });
        }
      }
      
      return newParams;
    });
    
    // Sync TTS engine settings to global settings
    if (key === "engine") {
      updateSettings({ tts_engine: value });
    } else if (key === "extractionModelSize") {
      updateSettings({ qwen3_model_size: value });
    } else if (key === "useFlashAttention") {
      updateSettings({ qwen3_flash_attention: value });
    }
  }, [selectedPersonaId, updatePersonaVoiceTuning, updateSettings]);
  
  // Save all tuning params (call this when regenerating voice)
  const saveVoiceTuning = useCallback(() => {
    if (selectedPersonaId) {
      const tuning: Partial<VoiceTuningParams> = {};
      modifiedParams.forEach(param => {
        tuning[param] = params[param] as any;
      });
      updatePersonaVoiceTuning(selectedPersonaId, tuning);
      setOriginalTuning(getPersonaVoiceTuning(selectedPersonaId));
      setModifiedParams(new Set());
      setShowRegenNotice(false);
      toast.success('Voice tuning saved for this persona');
    }
  }, [selectedPersonaId, modifiedParams, params, updatePersonaVoiceTuning, getPersonaVoiceTuning]);
  
  // Reset tuning to defaults
  const resetVoiceTuning = useCallback(() => {
    if (selectedPersonaId) {
      resetPersonaVoiceTuning(selectedPersonaId);
      setParams(p => ({ ...p, ...defaultVoiceTuning }));
      setOriginalTuning(defaultVoiceTuning);
      setModifiedParams(new Set());
      setShowRegenNotice(false);
      toast.info('Voice tuning reset to defaults');
    }
  }, [selectedPersonaId, resetPersonaVoiceTuning]);
  
  // Check if synthesis params need regeneration
  const needsRegeneration = useCallback(() => {
    return Array.from(modifiedParams).some(param => requiresRegeneration(param));
  }, [modifiedParams]);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const selectedPersona = personas.find(p => p.id === selectedPersonaId);
  
  const log = useCallback((message: string, data?: any) => {
    const prefix = "[VoiceCreator]";
    if (data !== undefined) {
      console.log(`${prefix} ${message}`, data);
    } else {
      console.log(`${prefix} ${message}`);
    }
  }, []);

  // Load persona voice from IndexedDB for preview
  // Note: Uses selectedPersonaId instead of selectedPersona (which is defined later)
  const loadPersonaVoiceForPreview = useCallback(async (): Promise<string | null> => {
    if (!selectedPersonaId) {
      toast.error('No persona selected');
      return null;
    }
    
    // Get fresh persona data to check for voice
    const persona = personas.find(p => p.id === selectedPersonaId);
    if (!persona?.voice_create?.has_audio) {
      toast.error('No voice saved for this persona');
      return null;
    }

    setIsPreviewLoading(true);
    try {
      log('Loading voice from IndexedDB for preview:', selectedPersonaId);
      const { unifiedStorage } = await import('@/services/unifiedStorage');
      const voiceData = await unifiedStorage.loadVoice(selectedPersonaId);
      
      if (voiceData?.audio_data) {
        log('Voice loaded successfully, audio length:', voiceData.audio_data.length);
        setPreviewPersonaVoice(voiceData.audio_data);
        return voiceData.audio_data;
      } else {
        toast.error('Voice data not found in storage');
        return null;
      }
    } catch (error) {
      log('Failed to load persona voice:', error);
      toast.error('Failed to load voice for preview');
      return null;
    } finally {
      setIsPreviewLoading(false);
    }
  }, [selectedPersonaId, personas, log]);

  // Handle preview with real-time effects
  const handlePreview = useCallback(async () => {
    if (isPreviewPlaying) {
      audioEffects.stop();
      setIsPreviewPlaying(false);
      return;
    }

    // Determine which audio to preview
    let audioToPreview: string | null = null;
    
    if (previewTarget === 'generated' && generatedAudio) {
      audioToPreview = generatedAudio;
    } else if (previewTarget === 'persona') {
      if (previewPersonaVoice) {
        audioToPreview = previewPersonaVoice;
      } else {
        audioToPreview = await loadPersonaVoiceForPreview();
      }
    }

    if (!audioToPreview) {
      if (previewTarget === 'generated') {
        toast.error('No generated voice to preview. Create a voice first.');
      }
      return;
    }

    // Convert params to audio effect params
    const effectParams: AudioEffectParams = {
      pitchShift: params.pitchShift,
      speed: params.speed,
      warmth: params.warmth,
      clarity: params.clarity,
      breathiness: params.breathiness,
      resonance: params.resonance,
      reverb: params.reverb,
      eqLow: params.eqLow,
      eqMid: params.eqMid,
      eqHigh: params.eqHigh,
      compression: params.compression,
    };

    try {
      setIsPreviewPlaying(true);
      log('Starting preview with effects:', effectParams);
      
      await audioEffects.playWithEffects(audioToPreview, effectParams, () => {
        setIsPreviewPlaying(false);
      });
    } catch (error) {
      log('Preview failed:', error);
      toast.error('Failed to play preview');
      setIsPreviewPlaying(false);
    }
  }, [isPreviewPlaying, previewTarget, generatedAudio, previewPersonaVoice, params, loadPersonaVoiceForPreview, log]);

  // Stop preview when component unmounts
  useEffect(() => {
    return () => {
      audioEffects.stop();
    };
  }, []);

  // Update effects in real-time when params change during preview
  useEffect(() => {
    if (isPreviewPlaying && audioEffects.isPlaying()) {
      audioEffects.updateEffects({
        pitchShift: params.pitchShift,
        speed: params.speed,
        warmth: params.warmth,
        clarity: params.clarity,
        breathiness: params.breathiness,
        resonance: params.resonance,
        reverb: params.reverb,
        eqLow: params.eqLow,
        eqMid: params.eqMid,
        eqHigh: params.eqHigh,
        compression: params.compression,
      });
    }
  }, [params, isPreviewPlaying]);

  useEffect(() => {
    const checkPuter = () => {
      const available = puter.isTranscriptionAvailable();
      setPuterAvailable(available);
    };
    
    checkPuter();
    const timeouts = [100, 500, 1000, 2000].map(ms => setTimeout(checkPuter, ms));
    const unsubscribe = puter.onAvailabilityChange(() => checkPuter());
    
    return () => {
      timeouts.forEach(clearTimeout);
      unsubscribe();
    };
  }, []);

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const connected = await ttsService.checkConnection();
        setIsBackendConnected(connected);
        if (connected) {
          const status = await ttsService.getEngineStatus();
          setEngineStatus(status);
        }
      } catch (error) {
        setIsBackendConnected(false);
      }
    };
    checkConnection();
  }, [settings.tts_backend_url]);

  useEffect(() => {
    if (currentPersona) {
      setSelectedPersonaId(currentPersona.id);
      setVoiceName(`${currentPersona.name}'s Voice`);
    }
  }, [currentPersona]);

  // Enumerate audio input devices
  useEffect(() => {
    const enumerateDevices = async () => {
      try {
        // Request permission first to get labels
        await navigator.mediaDevices.getUserMedia({ audio: true });
        
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(d => d.kind === 'audioinput');
        setAudioDevices(audioInputs);
        
        console.log("[VoiceCreator] Available audio devices:", audioInputs.map(d => ({
          id: d.deviceId.slice(0, 8) + '...',
          label: d.label,
          kind: d.kind
        })));
      } catch (err) {
        console.error("[VoiceCreator] Failed to enumerate devices:", err);
      }
    };
    
    enumerateDevices();
    
    // Listen for device changes
    navigator.mediaDevices.addEventListener('devicechange', enumerateDevices);
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', enumerateDevices);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      if (previewAudioRef.current) {
        previewAudioRef.current.pause();
        previewAudioRef.current = null;
      }
      setCountdown(null);
    };
  }, []);



  const transcribeAudio = async (audioBlob: Blob, source: "upload" | "record") => {
    setIsTranscribing(true);
    try {
      log("Transcribing with Puter.js...");
      const result = await puter.transcribe(audioBlob);
      const transcribedText = result.text || "";
      log("Transcribed:", transcribedText);
      
      setParams(p => ({ ...p, referenceText: transcribedText }));
      toast.success(`${source === "upload" ? "Audio file" : "Recording"} transcribed!`);
    } catch (err) {
      log("Transcription failed:", err);
      toast.error("Transcription failed. Please enter the text manually.");
    } finally {
      setIsTranscribing(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      log("Loading audio file:", file.name);
      
      // Check if file needs conversion (not WAV)
      const needsConversion = !file.name.toLowerCase().endsWith('.wav');
      let processedFile: Blob = file;
      
      if (needsConversion) {
        log("Converting to WAV format...");
        toast.info("Converting audio to WAV format...");
        try {
          processedFile = await blobToWav(file);
          log("Conversion successful");
        } catch (err) {
          log("Conversion failed, using original file:", err);
          toast.warning("Could not convert to WAV. Backend may not accept this format.");
        }
      }
      
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64 = (reader.result as string).split(',')[1];
        
        setParams(p => ({
          ...p,
          hasReference: true,
          referenceAudio: base64,
          referenceText: "",
        }));
        
        if (puter.isTranscriptionAvailable()) {
          await transcribeAudio(file, "upload");
        } else {
          toast.info("Audio loaded. Enter the spoken text manually or use Transcribe.");
        }
        
        toast.success(`Loaded: ${file.name}`);
      };
      reader.readAsDataURL(processedFile);
    } catch (error) {
      log("File upload failed:", error);
      toast.error("Failed to load audio file");
    }
  };

  // Play beep sound
  const playBeep = (frequency: number = 800, duration: number = 200) => {
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      oscillator.frequency.value = frequency;
      oscillator.type = 'sine';
      
      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + duration / 1000);
      
      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + duration / 1000);
    } catch (e) {
      console.error("Beep failed:", e);
    }
  };

  const startRecording = async () => {
    try {
      // Build audio constraints
      const audioConstraints: MediaTrackConstraints = selectedDeviceId
        ? {
            deviceId: { exact: selectedDeviceId },
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          }
        : {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          };
      
      console.log("[VoiceCreator] Starting recording with device:", selectedDeviceId || "default", audioConstraints);
      
      // Get microphone access with selected device
      const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
      
      // Log the actual device being used
      const track = stream.getAudioTracks()[0];
      if (track) {
        console.log("[VoiceCreator] Using audio track:", track.label);
      }
      
      // Setup MediaRecorder BEFORE countdown for instant start
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        const webmBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        try {
          // Convert webm to wav for backend compatibility
          const wavBlob: Blob = await blobToWav(webmBlob);
          
          const reader = new FileReader();
          reader.onloadend = async () => {
            const base64 = (reader.result as string).split(',')[1];
            
            setParams(p => ({
              ...p,
              hasReference: true,
              referenceAudio: base64,
              referenceText: "",
            }));
            
            if (puter.isTranscriptionAvailable()) {
              await transcribeAudio(webmBlob, "record");
            } else {
              toast.info("Recording saved. Enter the spoken text manually or use Transcribe.");
            }
            
            toast.success("Recording saved!");
          };
          reader.readAsDataURL(wavBlob);
        } catch (err) {
          console.error("[VoiceCreator] Failed to convert audio:", err);
          toast.error("Failed to process recording. Please try uploading a WAV file instead.");
        }
        
        stream.getTracks().forEach(track => track.stop());
      };
      
      // Countdown with precise timing
      // 3 - Beep
      setCountdown(3);
      playBeep(800, 200);
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // 2 - Beep  
      setCountdown(2);
      playBeep(800, 200);
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // 1 - Silent
      setCountdown(1);
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Start actual recording IMMEDIATELY
      setCountdown(null);
      mediaRecorder.start(100); // Collect data every 100ms
      setIsRecording(true);
      setRecordingDuration(0);
      
      recordingTimerRef.current = setInterval(() => {
        setRecordingDuration(d => d + 1);
      }, 1000);
      
    } catch (error) {
      log("Recording failed:", error);
      setCountdown(null);
      toast.error("Could not access microphone");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
    }
    setIsRecording(false);
    setCountdown(null);
  };

  const clearReference = () => {
    setParams(p => ({
      ...p,
      hasReference: false,
      referenceAudio: null,
      referenceText: "",
      isEditingReferenceText: false,
    }));
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current = null;
      setIsPreviewPlaying(false);
    }
  };

  const togglePreview = () => {
    if (!params.referenceAudio) return;
    
    if (isPreviewPlaying && previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current = null;
      setIsPreviewPlaying(false);
    } else {
      const audio = new Audio(`data:audio/wav;base64,${params.referenceAudio}`);
      previewAudioRef.current = audio;
      audio.onended = () => {
        setIsPreviewPlaying(false);
        previewAudioRef.current = null;
      };
      audio.onerror = () => {
        setIsPreviewPlaying(false);
        previewAudioRef.current = null;
        toast.error("Failed to play preview");
      };
      audio.play().then(() => {
        setIsPreviewPlaying(true);
      }).catch(() => {
        setIsPreviewPlaying(false);
        toast.error("Failed to play preview");
      });
    }
  };

  const handleGenerate = async () => {
    if (!targetText.trim()) {
      toast.error("Please enter text to synthesize");
      return;
    }

    if (!isBackendConnected) {
      toast.error("TTS backend not connected");
      return;
    }

    if (params.engine === "qwen3" && !params.hasReference) {
      toast.error("Qwen3 requires reference audio. Please upload or record audio first.");
      return;
    }

    setIsProcessing(true);
    log("Creating voice with params:", {
      engine: params.engine,
      modelSize: params.extractionModelSize,
      hasReference: params.hasReference,
    });

    try {
      const startTime = Date.now();
      
      // Standard voice creation with selected model size
      const response = await ttsService.createVoice(targetText, {
          reference_audio: params.referenceAudio || undefined,
          reference_text: params.referenceText,
          // Basic tuning
          pitch_shift: params.pitchShift,
          speed: params.speed,
          // Voice characteristics
          warmth: params.warmth,
          expressiveness: params.expressiveness,
          stability: params.stability,
          clarity: params.clarity,
          breathiness: params.breathiness,
          resonance: params.resonance,
          // Speech characteristics
          emotion: params.emotion,
          emphasis: params.emphasis,
          pauses: params.pauses,
          energy: params.energy,
          // Audio effects
          reverb: params.reverb,
          eq_low: params.eqLow,
          eq_mid: params.eqMid,
          eq_high: params.eqHigh,
          compression: params.compression,
          // Engine selection
          engine: params.engine,
          qwen3_model_size: params.extractionModelSize,
          use_flash_attention: params.useFlashAttention,
          // Seed
          seed: params.seed || Math.floor(Math.random() * 1000000),
        });

        // Track timing for debugging
        console.log(`[VoiceCreator] Voice creation took ${Date.now() - startTime}ms`);
        log(`Voice creation complete! Engine: ${response.engine_used}`);
        
        setGeneratedAudio(response.audio_data);
        setLastUsedEngine(response.engine_used);
        toast.success(`Voice created with ${response.engine_used}!`);
    } catch (error) {
      log("Voice creation failed:", error);
      toast.error(`Voice creation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const saveVoiceToPersona = async () => {
    if (!generatedAudio) {
      toast.error("Generate a voice first before saving");
      return;
    }

    if (!selectedPersonaId) {
      toast.error("Please select a persona to save this voice to");
      return;
    }

    log(`Saving voice to persona: ${selectedPersona?.name}`);

    try {
      await updatePersonaVoice(selectedPersonaId, {
        audio_data: generatedAudio,
        reference_text: targetText,
        voice_config: {
          type: "synthetic",
          params: {
            // Basic
            pitch: params.pitchShift,
            speed: params.speed,
            // Voice characteristics
            warmth: params.warmth,
            expressiveness: params.expressiveness,
            stability: params.stability,
            clarity: params.clarity,
            breathiness: params.breathiness,
            resonance: params.resonance,
            // Speech
            emotion: params.emotion,
            emphasis: params.emphasis,
            pauses: params.pauses,
            energy: params.energy,
            // Audio effects
            reverb: params.reverb,
            eq_low: params.eqLow,
            eq_mid: params.eqMid,
            eq_high: params.eqHigh,
            compression: params.compression,
            // Engine info
            engine: params.engine,
            qwen3_model_size: params.extractionModelSize,
            gender: "neutral",
            age: "adult",
            seed: params.seed,
          },
          name: voiceName || `${selectedPersona?.name}'s Voice`,
        }
      });

      toast.success(`Voice saved to ${selectedPersona?.name}!`);
      
      setTimeout(() => {
        const { personas: freshPersonas } = useStore.getState();
        const freshPersona = freshPersonas.find(p => p.id === selectedPersonaId);
        if (freshPersona) {
          setCurrentPersona(freshPersona);
        }
      }, 100);
    } catch (error) {
      log("Save voice error:", error);
      toast.error("Failed to save voice");
    }
  };

  // Handle watermark detection
  const handleWatermarkDetection = async (file: File) => {
    setIsDetectingWatermark(true);
    setWatermarkDetectionResult(null);
    
    // Determine file format from extension
    const fileExt = file.name.split('.').pop()?.toLowerCase() || 'wav';
    console.log(`[Watermark Detection] Uploading file: ${file.name}, size: ${file.size} bytes, format: ${fileExt}`);
    
    try {
      const reader = new FileReader();
      
      reader.onerror = () => {
        toast.error("Failed to read audio file");
        setIsDetectingWatermark(false);
      };
      
      reader.onloadend = async () => {
        try {
          const base64 = (reader.result as string).split(',')[1];
          
          if (!base64 || base64.length < 100) {
            toast.error("Audio file is too small or empty");
            setIsDetectingWatermark(false);
            return;
          }
          
          console.log(`[Watermark Detection] Sending ${base64.length} bytes to backend`);
          
          const response = await fetch(`${settings.tts_backend_url}/api/watermark/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              audio_data: base64,
              format: fileExt
            })
          });
          
          if (response.ok) {
            const result = await response.json();
            console.log(`[Watermark Detection] Result:`, result);
            setWatermarkDetectionResult({
              detected: result.detected,
              confidence: result.confidence,
              message: result.message,
              details: result.details
            });
            
            // Show toast with result
            if (result.detected) {
              toast.success("AI watermark detected!", { 
                description: `${Math.round(result.confidence * 100)}% confidence`
              });
            } else {
              toast.info("No AI watermark found", { 
                description: `${Math.round(result.confidence * 100)}% confidence`
              });
            }
          } else {
            // Try to get error message from backend
            let errorMsg = "Watermark detection failed";
            try {
              const errorData = await response.json();
              errorMsg = errorData.detail || errorMsg;
            } catch {
              errorMsg = `Server error: ${response.status}`;
            }
            console.error(`[Watermark Detection] Error: ${errorMsg}`);
            toast.error(errorMsg);
          }
        } catch (error) {
          console.error("[Watermark Detection] Network error:", error);
          toast.error("Could not connect to TTS backend. Ensure the backend is running.");
        } finally {
          setIsDetectingWatermark(false);
        }
      };
      
      reader.readAsDataURL(file);
    } catch (error) {
      console.error("[Watermark Detection] File read error:", error);
      toast.error("Failed to read audio file");
      setIsDetectingWatermark(false);
    }
  };

  return (
    <div className="h-full overflow-auto p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6 max-w-3xl mx-auto"
      >
        <div>
          <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-primary" />
            Voice Studio
          </h2>
          <p className="text-muted-foreground">
            Create unique AI voices and verify audio authenticity.
          </p>
          {!isBackendConnected && (
            <div className="mt-2 text-sm text-amber-400 bg-amber-400/10 px-3 py-2 rounded flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              TTS backend not connected. Start the Python backend.
            </div>
          )}
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2 border-b">
          <button
            onClick={() => setActiveTab("create")}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-[2px] ${
              activeTab === "create"
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            <div className="flex items-center gap-2">
              <Wand2 className="w-4 h-4" />
              Create Voice
            </div>
          </button>
          <button
            onClick={() => setActiveTab("detect")}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-[2px] ${
              activeTab === "detect"
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4" />
              Watermark Detection
            </div>
          </button>
        </div>

        {activeTab === "create" ? (
          <>
        <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
          <div className="flex items-start gap-3">
            <Check className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-green-400">Voice Creation</h3>
              <p className="text-sm text-muted-foreground">
                This tool creates <strong>synthetic voices</strong> inspired by reference audio. 
                The AI generates a new voice profile. 
                You are responsible for obtaining consent from anyone whose voice you reference.
              </p>
            </div>
          </div>
        </div>

        {/* Target Persona - Moved to top */}
        <div 
          className="space-y-4 border rounded-lg p-4 transition-all duration-300"
          style={getPersonaGradientStyle(selectedPersona)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {selectedPersona?.avatar_config && (
                <div 
                  className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold border-2 border-white/20 shadow-inner"
                  style={{ 
                    background: `linear-gradient(135deg, ${selectedPersona.avatar_config.primary_color}, ${selectedPersona.avatar_config.secondary_color})`,
                  }}
                >
                  {selectedPersona.name.charAt(0).toUpperCase()}
                </div>
              )}
              <Label className="flex items-center gap-2">
                <User className="w-4 h-4" />
                Target Persona
              </Label>
            </div>
          </div>
          <Select value={selectedPersonaId} onValueChange={setSelectedPersonaId}>
            <SelectTrigger>
              <SelectValue placeholder="Select a persona" />
            </SelectTrigger>
            <SelectContent>
              {personas.map((persona) => (
                <SelectItem key={persona.id} value={persona.id} textValue={persona.name}>
                  {persona.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          <div className="space-y-2 mt-3">
            <Label>Voice Name</Label>
            <Input
              value={voiceName}
              onChange={(e) => setVoiceName(e.target.value)}
              placeholder={`${selectedPersona?.name || "Persona"}'s Voice`}
            />
          </div>
        </div>

        {/* TTS Engine Selection */}
        <div className="space-y-4 border rounded-lg p-4 bg-gradient-to-br from-indigo-500/10 via-purple-500/5 to-blue-500/10 border-indigo-500/20 hover:border-indigo-500/30 transition-all">
          <Label className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-indigo-400" />
            TTS Engine
          </Label>
          
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setParams(p => ({ ...p, engine: "styletts2" }))}
              className={`p-3 rounded-lg border text-left transition-all ${
                params.engine === "styletts2"
                  ? "border-primary bg-primary/10"
                  : "border-border hover:border-primary/50"
              }`}
            >
              <div className="font-medium text-sm">StyleTTS2</div>
              <div className="text-xs text-muted-foreground">Fast, lightweight</div>
              {engineStatus && (
                <div className={`text-xs mt-1 ${engineStatus.styletts2_available ? 'text-green-400' : 'text-amber-400'}`}>
                  {engineStatus.styletts2_available ? 'Available' : 'Not installed'}
                </div>
              )}
            </button>
            
            <button
              onClick={() => setParams(p => ({ ...p, engine: "qwen3" }))}
              className={`p-3 rounded-lg border text-left transition-all ${
                params.engine === "qwen3"
                  ? "border-primary bg-primary/10"
                  : "border-border hover:border-primary/50"
              }`}
            >
              <div className="font-medium text-sm">Qwen3-TTS</div>
              <div className="text-xs text-muted-foreground">Higher quality</div>
              {engineStatus && (
                <div className={`text-xs mt-1 ${engineStatus.qwen3_available ? 'text-green-400' : 'text-amber-400'}`}>
                  {engineStatus.qwen3_available ? 'Available' : 'Not installed'}
                </div>
              )}
            </button>
          </div>
          
          {params.engine === "qwen3" && engineStatus && !engineStatus.qwen3_available && (
            <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
              <p className="text-sm text-amber-400 font-medium mb-2">Qwen3-TTS Not Installed</p>
              <p className="text-xs text-muted-foreground mb-2">To use Qwen3-TTS, run these commands:</p>
              <pre className="text-xs bg-black/50 p-2 rounded overflow-x-auto">
                {QWEN3_INSTALL_INSTRUCTIONS}
              </pre>
            </div>
          )}
          
          {params.engine === "qwen3" && (
            <div className="space-y-3 pt-3 border-t">
              {!isBackendConnected && (
                <div className="p-2 bg-amber-500/10 border border-amber-500/30 rounded text-xs text-amber-400 mb-2">
                  Backend not connected. Install dependencies and restart the backend.
                </div>
              )}
              
              <div className="flex items-center justify-between">
                <Label className="text-xs flex items-center gap-1">
                  <Zap className="w-3 h-3" />
                  Flash Attention
                </Label>
                <Switch
                  checked={params.useFlashAttention}
                  onCheckedChange={(v) => setParams(p => ({ ...p, useFlashAttention: v }))}
                  disabled={!isBackendConnected}
                />
              </div>
              
              <p className="text-xs text-muted-foreground">
                Flash Attention reduces VRAM usage by ~30%. Uses bfloat16 for compatibility.
              </p>
              
              {/* Qwen3 Model Size Selector */}
              <div className="space-y-3 pt-3 border-t">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Model Size</Label>
                  <Select 
                    value={params.extractionModelSize} 
                    onValueChange={(v: "0.6B" | "1.7B") => setParams(p => ({ ...p, extractionModelSize: v, qwen3Size: v }))}
                    disabled={!isBackendConnected}
                  >
                    <SelectTrigger className="w-36">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0.6B" textValue="0.6B">0.6B (~3GB VRAM)</SelectItem>
                      <SelectItem value="1.7B" textValue="1.7B">1.7B (~6GB VRAM)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <p className="text-xs text-muted-foreground">
                  {params.extractionModelSize === "1.7B" ? "Higher quality, more VRAM required" : "Faster generation, less VRAM required"}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Reference Audio */}
        <div className="space-y-4 border rounded-lg p-4 bg-gradient-to-br from-amber-500/5 via-orange-500/3 to-yellow-500/5 border-amber-500/20">
          <Label className="flex items-center gap-2">
            <Mic className="w-4 h-4 text-amber-400" />
            Reference Audio
            {params.engine === "qwen3" && <span className="text-amber-400 text-xs">(Required)</span>}
            {params.engine === "styletts2" && <span className="text-muted-foreground text-xs">(Optional)</span>}
          </Label>
          
          <div className={`flex items-center gap-2 text-xs px-3 py-2 rounded ${puterAvailable ? 'text-green-400 bg-green-400/10' : 'text-amber-400 bg-amber-400/10'}`}>
            {puterAvailable ? (
              <>
                <Check className="w-3 h-3" />
                Puter.js transcription available
              </>
            ) : (
              <>
                <AlertCircle className="w-3 h-3" />
                Puter.js not detected - enter text manually
              </>
            )}
          </div>

          {/* Audio Device Selector */}
          {audioDevices.length > 0 && (
            <div className="space-y-2">
              <Label className="text-xs flex items-center gap-1">
                <Headphones className="w-3 h-3" />
                Recording Device
              </Label>
              <Select 
                value={selectedDeviceId || "default"} 
                onValueChange={(value) => {
                  const deviceId = value === "default" ? "" : value;
                  setSelectedDeviceId(deviceId);
                  localStorage.setItem("mimic_recording_device", deviceId);
                }}
              >
                <SelectTrigger className="text-xs">
                  <SelectValue placeholder="Default Microphone" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default" textValue="Default Microphone">Default Microphone</SelectItem>
                  {audioDevices.map((device) => (
                    <SelectItem key={device.deviceId} value={device.deviceId} textValue={device.label || `Device ${device.deviceId.slice(0, 8)}`}>
                      {device.label || `Device ${device.deviceId.slice(0, 8)}...`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedDeviceId && (
                <p className="text-xs text-muted-foreground">
                  Recording from: {audioDevices.find(d => d.deviceId === selectedDeviceId)?.label || "Selected device"}
                </p>
              )}
            </div>
          )}
          
          {!params.hasReference ? (
            <div className="grid grid-cols-2 gap-3">
              <input
                type="file"
                accept="audio/*"
                className="hidden"
                ref={fileInputRef}
                onChange={handleFileUpload}
              />
              
              <Button
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                disabled={isTranscribing}
                className="h-20 flex flex-col items-center justify-center gap-2"
              >
                <Upload className="w-5 h-5" />
                <span className="text-xs">Upload Audio</span>
                {isTranscribing && <span className="text-xs text-muted-foreground">Transcribing...</span>}
              </Button>
              
              <Button
                variant={countdown ? "default" : (isRecording ? "destructive" : "outline")}
                onClick={isRecording ? stopRecording : (countdown ? undefined : startRecording)}
                disabled={isTranscribing || !!countdown}
                className={`h-20 flex flex-col items-center justify-center gap-2 ${countdown ? "text-2xl font-bold" : ""}`}
              >
                {countdown ? (
                  <span className="text-3xl font-bold">{countdown}</span>
                ) : (
                  <>
                    <Mic className={`w-5 h-5 ${isRecording ? "animate-pulse" : ""}`} />
                    <span className="text-xs">
                      {isRecording ? `Recording ${recordingDuration}s` : "Record"}
                    </span>
                  </>
                )}
              </Button>
            </div>
          ) : (
            <div className="p-3 bg-muted rounded-lg space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-green-400" />
                  <span className="text-sm font-medium">Reference audio loaded</span>
                </div>
                <div className="flex items-center gap-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={togglePreview}
                    className="flex items-center gap-1"
                  >
                    <Ear className="w-4 h-4" />
                    {isPreviewPlaying ? "Stop" : "Preview"}
                  </Button>
                  <Button variant="ghost" size="sm" onClick={clearReference}>
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs flex items-center gap-1">
                    <Edit3 className="w-3 h-3" />
                    Reference Text
                    <span className="text-muted-foreground">(edit if needed)</span>
                  </Label>
                  
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      if (!params.referenceAudio) return;
                      try {
                        const byteString = atob(params.referenceAudio);
                        const bytes = new Uint8Array(byteString.length);
                        for (let i = 0; i < byteString.length; i++) {
                          bytes[i] = byteString.charCodeAt(i);
                        }
                        const blob = new Blob([bytes], { type: 'audio/wav' });
                        await transcribeAudio(blob, "upload");
                      } catch (err) {
                        toast.error("Could not transcribe. Enter text manually.");
                      }
                    }}
                    disabled={isTranscribing}
                    className="flex items-center gap-1 text-xs h-7"
                  >
                    {isTranscribing ? (
                      <>
                        <RefreshCw className="w-3 h-3 animate-spin" />
                        Transcribing...
                      </>
                    ) : (
                      <>
                        <WandSparkles className="w-3 h-3" />
                        {puterAvailable ? "Transcribe with Puter" : "Try Transcribe"}
                      </>
                    )}
                  </Button>
                </div>
                <Textarea
                  value={params.referenceText}
                  onChange={(e) => setParams(p => ({ ...p, referenceText: e.target.value }))}
                  placeholder="Enter the spoken text from your reference audio..."
                  rows={3}
                  className="text-sm"
                />
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">
                    {params.referenceText.length} characters
                  </span>
                  <button
                    onClick={() => setParams(p => ({ ...p, referenceText: "" }))}
                    className="text-muted-foreground hover:text-primary underline"
                    title="Use voice characteristics only (x_vector mode) - more stable but less accurate"
                  >
                    Clear text (voice only mode)
                  </button>
                </div>
                <p className="text-xs text-muted-foreground">
                  This text helps the AI understand your voice better. Edit if the transcription is incorrect.
                  Voice-only mode (no text) is more stable but less accurate.
                </p>
              </div>
            </div>
          )}
          
          <p className="text-xs text-muted-foreground">
            {params.engine === "qwen3" 
              ? "Qwen3 requires reference audio to create a voice profile. The reference text helps improve quality."
              : "StyleTTS2 can work with or without reference audio. Adding reference audio improves voice similarity."
            }
          </p>
        </div>

        <div className="space-y-4 border rounded-lg p-4 bg-gradient-to-br from-emerald-500/8 via-teal-500/5 to-cyan-500/8 border-emerald-500/20">
          <div className="flex items-center justify-between">
            <div>
              <Label className="flex items-center gap-2">
                <SlidersHorizontal className="w-4 h-4 text-emerald-400" />
                Voice Tuning
              </Label>
              <p className="text-xs text-muted-foreground mt-0.5">
                Persistent for {selectedPersona?.name || 'selected persona'}
              </p>
            </div>
            <div className="flex gap-2">
              <Button variant="ghost" size="sm" onClick={resetVoiceTuning}>
                <RefreshCw className="w-4 h-4 mr-1" />
                Reset
              </Button>
              {(modifiedParams.size > 0 || showRegenNotice) && (
                <Button 
                  variant="default" 
                  size="sm" 
                  onClick={saveVoiceTuning}
                  className="bg-emerald-600 hover:bg-emerald-700"
                >
                  <Check className="w-4 h-4 mr-1" />
                  Save Tuning
                </Button>
              )}
            </div>
          </div>

          {/* Regeneration Notice */}
          {showRegenNotice && needsRegeneration() && (
            <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
                <div>
                  <p className="text-xs text-amber-400 font-medium">
                    Voice Character settings changed
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Changes to Warmth, Clarity, Expressiveness, etc. require voice regeneration to take full effect. 
                    Post-processing effects (Pitch, Speed, Reverb, EQ) apply instantly during preview.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Preview Controls */}
          <div className="p-3 bg-black/20 rounded-lg space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-xs flex items-center gap-2">
                <Play className="w-3 h-3" />
                Preview Voice
              </Label>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPreviewTarget('generated')}
                  className={`px-3 py-1 text-xs rounded-full transition-colors ${
                    previewTarget === 'generated'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted hover:bg-muted/80'
                  } ${!generatedAudio ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={!generatedAudio}
                  title={!generatedAudio ? 'Generate a voice first' : 'Preview generated voice'}
                >
                  Generated
                </button>
                <button
                  onClick={() => setPreviewTarget('persona')}
                  className={`px-3 py-1 text-xs rounded-full transition-colors ${
                    previewTarget === 'persona'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted hover:bg-muted/80'
                  } ${!selectedPersona?.voice_create?.has_audio ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={!selectedPersona?.voice_create?.has_audio}
                  title={!selectedPersona?.voice_create?.has_audio ? 'No saved voice for this persona' : `Preview ${selectedPersona.name}'s voice`}
                >
                  {selectedPersona?.name || 'Persona'}
                </button>
              </div>
            </div>
            
            <Button
              onClick={handlePreview}
              disabled={isPreviewLoading || (previewTarget === 'generated' && !generatedAudio) || (previewTarget === 'persona' && !selectedPersona?.voice_create?.has_audio)}
              variant={isPreviewPlaying ? "destructive" : "default"}
              size="sm"
              className="w-full"
            >
              {isPreviewLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Loading...
                </>
              ) : isPreviewPlaying ? (
                <>
                  <X className="w-4 h-4 mr-2" />
                  Stop Preview
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  {previewTarget === 'generated' 
                    ? 'Preview Generated Voice' 
                    : `Preview ${selectedPersona?.name || 'Persona'} Voice`}
                </>
              )}
            </Button>
            
            <p className="text-xs text-muted-foreground">
              Preview applies tuning effects in real-time. Adjust sliders while playing to hear changes.
            </p>
          </div>

          {/* Basic Tuning - Post-processing params */}
          <div className="space-y-4 border-b pb-4">
            <div className="flex items-center gap-2">
              <div className="text-xs font-medium text-emerald-400 uppercase tracking-wide">Instant Effects</div>
              <span className="text-[10px] px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 rounded">No regeneration needed</span>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label className="text-xs flex items-center gap-1">
                  Pitch Shift
                  {modifiedParams.has('pitchShift') && <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />}
                </Label>
                <span className="text-xs text-muted-foreground">
                  {params.pitchShift < 0 ? "Lower" : params.pitchShift > 0 ? "Higher" : "Normal"}
                </span>
              </div>
              <Slider
                value={[params.pitchShift]}
                onValueChange={([v]) => updateParam('pitchShift', v)}
                min={-1}
                max={1}
                step={0.1}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Lower</span>
                <span>Normal</span>
                <span>Higher</span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label className="text-xs flex items-center gap-1">
                  Speed
                  {modifiedParams.has('speed') && <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />}
                </Label>
                <span className="text-xs text-muted-foreground">{params.speed.toFixed(1)}x</span>
              </div>
              <Slider
                value={[params.speed]}
                onValueChange={([v]) => updateParam('speed', v)}
                min={0.5}
                max={2.0}
                step={0.1}
              />
            </div>
          </div>

          {/* Voice Characteristics - Synthesis params (require regeneration) */}
          <div className="space-y-4 border-b pb-4">
            <div className="flex items-center gap-2">
              <div className="text-xs font-medium text-amber-400 uppercase tracking-wide">Voice Character</div>
              <span className="text-[10px] px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded">Requires regeneration</span>
            </div>
            
            {[
              { key: 'warmth', label: 'Warmth', desc: 'Natural, mellow tone' },
              { key: 'expressiveness', label: 'Expressiveness', desc: 'Emotional variation' },
              { key: 'stability', label: 'Stability', desc: 'Consistency vs creativity' },
              { key: 'clarity', label: 'Clarity', desc: 'Articulation sharpness' },
              { key: 'breathiness', label: 'Breathiness', desc: 'Airiness in voice' },
              { key: 'resonance', label: 'Resonance', desc: 'Depth and fullness' },
            ].map(({ key, label, desc }) => (
              <div key={key} className="space-y-1">
                <div className="flex justify-between">
                  <Label className="text-xs flex items-center gap-1">
                    {label}
                    {modifiedParams.has(key as keyof VoiceTuningParams) && (
                      <span className="w-1.5 h-1.5 bg-amber-400 rounded-full" />
                    )}
                  </Label>
                  <span className="text-xs text-muted-foreground">{Math.round(params[key as keyof VoiceParams] as number * 100)}%</span>
                </div>
                <Slider
                  value={[params[key as keyof VoiceParams] as number]}
                  onValueChange={([v]) => updateParam(key as keyof VoiceParams, v)}
                  min={0}
                  max={1}
                  step={0.05}
                />
                <p className="text-[10px] text-muted-foreground">{desc}</p>
              </div>
            ))}
          </div>

          {/* Speech Characteristics - Synthesis params (require regeneration) */}
          <div className="space-y-4 border-b pb-4">
            <div className="flex items-center gap-2">
              <div className="text-xs font-medium text-amber-400 uppercase tracking-wide">Speech</div>
              <span className="text-[10px] px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded">Requires regeneration</span>
            </div>
            
            <div className="space-y-2">
              <Label className="text-xs flex items-center gap-1">
                Emotion
                {modifiedParams.has('emotion') && <span className="w-1.5 h-1.5 bg-amber-400 rounded-full" />}
              </Label>
              <Select 
                value={params.emotion} 
                onValueChange={(v: VoiceParams['emotion']) => updateParam('emotion', v)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="neutral" textValue="Neutral">Neutral</SelectItem>
                  <SelectItem value="happy" textValue="Happy">Happy</SelectItem>
                  <SelectItem value="sad" textValue="Sad">Sad</SelectItem>
                  <SelectItem value="angry" textValue="Angry">Angry</SelectItem>
                  <SelectItem value="excited" textValue="Excited">Excited</SelectItem>
                  <SelectItem value="calm" textValue="Calm">Calm</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {[
              { key: 'emphasis', label: 'Emphasis', desc: 'Word stress intensity' },
              { key: 'pauses', label: 'Pauses', desc: 'Pause length between phrases' },
              { key: 'energy', label: 'Energy', desc: 'Overall vocal energy' },
            ].map(({ key, label, desc }) => (
              <div key={key} className="space-y-1">
                <div className="flex justify-between">
                  <Label className="text-xs flex items-center gap-1">
                    {label}
                    {modifiedParams.has(key as keyof VoiceTuningParams) && (
                      <span className="w-1.5 h-1.5 bg-amber-400 rounded-full" />
                    )}
                  </Label>
                  <span className="text-xs text-muted-foreground">{Math.round(params[key as keyof VoiceParams] as number * 100)}%</span>
                </div>
                <Slider
                  value={[params[key as keyof VoiceParams] as number]}
                  onValueChange={([v]) => updateParam(key as keyof VoiceParams, v)}
                  min={0}
                  max={1}
                  step={0.05}
                />
                <p className="text-[10px] text-muted-foreground">{desc}</p>
              </div>
            ))}
          </div>

          {/* Audio Effects - Post-processing params */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="text-xs font-medium text-emerald-400 uppercase tracking-wide">Audio Effects</div>
              <span className="text-[10px] px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 rounded">No regeneration needed</span>
            </div>
            
            <div className="space-y-1">
              <div className="flex justify-between">
                <Label className="text-xs flex items-center gap-1">
                  Reverb
                  {modifiedParams.has('reverb') && <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />}
                </Label>
                <span className="text-xs text-muted-foreground">{Math.round(params.reverb * 100)}%</span>
              </div>
              <Slider
                value={[params.reverb]}
                onValueChange={([v]) => updateParam('reverb', v)}
                min={0}
                max={1}
                step={0.05}
              />
              <p className="text-[10px] text-muted-foreground">Room ambiance (0 = dry, 1 = cathedral)</p>
            </div>

            <div className="space-y-2">
              <Label className="text-xs">3-Band EQ</Label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { key: 'eqLow', label: 'Low', color: 'text-amber-400' },
                  { key: 'eqMid', label: 'Mid', color: 'text-green-400' },
                  { key: 'eqHigh', label: 'High', color: 'text-blue-400' },
                ].map(({ key, label, color }) => (
                  <div key={key} className="space-y-1">
                    <span className={`text-xs ${color} flex items-center gap-1`}>
                      {label}
                      {modifiedParams.has(key as keyof VoiceTuningParams) && (
                        <span className="w-1 h-1 bg-emerald-400 rounded-full" />
                      )}
                    </span>
                    <Slider
                      value={[params[key as keyof VoiceParams] as number]}
                      onValueChange={([v]) => updateParam(key as keyof VoiceParams, v)}
                      min={0}
                      max={1}
                      step={0.05}
                      className="h-16 flex-col"
                      orientation="vertical"
                    />
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between">
                <Label className="text-xs flex items-center gap-1">
                  Compression
                  {modifiedParams.has('compression') && <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />}
                </Label>
                <span className="text-xs text-muted-foreground">{Math.round(params.compression * 100)}%</span>
              </div>
              <Slider
                value={[params.compression]}
                onValueChange={([v]) => updateParam('compression', v)}
                min={0}
                max={1}
                step={0.05}
              />
              <p className="text-[10px] text-muted-foreground">Dynamic range compression</p>
            </div>
          </div>
        </div>

        <div className="space-y-2 border rounded-lg p-4 bg-gradient-to-br from-blue-500/8 via-sky-500/5 to-cyan-500/8 border-blue-500/20">
          <Label className="flex items-center gap-2">
            <Volume2 className="w-4 h-4 text-blue-400" />
            Text To Generate
          </Label>
          <Textarea
            value={targetText}
            onChange={(e) => setTargetText(e.target.value)}
            placeholder="Enter text to test your voice..."
            rows={3}
          />
        </div>

        <div className="flex gap-3">
          <Button
            onClick={handleGenerate}
            disabled={isProcessing || !targetText.trim() || (params.engine === "qwen3" && !params.hasReference)}
            className="flex-1"
            size="lg"
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <Wand2 className="w-4 h-4 mr-2" />
                Create Voice
              </>
            )}
          </Button>
        </div>

        {generatedAudio && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 border rounded-lg space-y-3 bg-gradient-to-br from-violet-500/10 via-purple-500/8 to-fuchsia-500/10 border-violet-500/30"
          >
            <div className="flex items-center justify-between">
              <h3 className="font-semibold flex items-center gap-2">
                <Volume2 className="w-4 h-4 text-violet-400" />
                Generated Voice
                <span className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded">
                  {lastUsedEngine}
                </span>
                <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded">
                  AI-Generated
                </span>
              </h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  const a = document.createElement('a');
                  a.href = `data:audio/wav;base64,${generatedAudio}`;
                  a.download = `${voiceName || 'voice'}.wav`;
                  a.click();
                }}
              >
                <Download className="w-4 h-4" />
              </Button>
            </div>
            <audio
              src={`data:audio/wav;base64,${generatedAudio}`}
              controls
              className="w-full"
            />
          </motion.div>
        )}

        {generatedAudio && (
          <Button
            onClick={saveVoiceToPersona}
            variant="outline"
            className="w-full"
            size="lg"
          >
            <Save className="w-4 h-4 mr-2" />
            Save Voice to {selectedPersona?.name || "Persona"}
          </Button>
        )}

        <div className="flex items-start gap-2 text-sm text-muted-foreground bg-muted p-3 rounded-lg">
          <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <ul className="list-disc list-inside space-y-1">
            <li>StyleTTS2 is faster and works without reference audio</li>
            <li>Qwen3 produces higher quality but requires reference audio and more VRAM</li>
            <li>Use 0.6B model with Flash Attention for lower memory usage (~3GB VRAM)</li>
            <li>All generated audio is watermarked for identification</li>
          </ul>
        </div>
        </>
        ) : (
          /* Watermark Detection Tab */
          <div className="space-y-6">
            <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <div className="flex items-start gap-3">
                <Shield className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-blue-400">AI Audio Verification</h3>
                  <p className="text-sm text-muted-foreground">
                    Upload an audio file to check if it contains an AI-generated watermark. 
                    This helps verify the authenticity of audio content.
                  </p>
                </div>
              </div>
            </div>

            {/* Upload Area */}
            <div className="border-2 border-dashed border-muted-foreground/25 rounded-xl p-8 text-center hover:border-muted-foreground/50 transition-colors">
              <input
                type="file"
                accept="audio/*"
                className="hidden"
                id="watermark-audio-upload"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleWatermarkDetection(file);
                }}
              />
              <label htmlFor="watermark-audio-upload" className="cursor-pointer">
                <FileAudio className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-sm font-medium mb-1">Drop audio file here or click to browse</p>
                <p className="text-xs text-muted-foreground">Supports WAV, MP3, and other audio formats</p>
              </label>
            </div>

            {/* Detection Result */}
            {isDetectingWatermark && (
              <div className="p-6 bg-muted rounded-lg text-center">
                <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2 text-primary" />
                <p className="text-sm text-muted-foreground">Analyzing audio for AI watermark...</p>
              </div>
            )}

            {watermarkDetectionResult && !isDetectingWatermark && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`p-4 rounded-lg border ${
                  watermarkDetectionResult.detected
                    ? "bg-amber-500/10 border-amber-500/30"
                    : "bg-green-500/10 border-green-500/30"
                }`}
              >
                <div className="flex items-start gap-3">
                  {watermarkDetectionResult.detected ? (
                    <AlertCircle className="w-6 h-6 text-amber-400 flex-shrink-0 mt-0.5" />
                  ) : (
                    <Check className="w-6 h-6 text-green-400 flex-shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <h3 className={`font-semibold ${
                      watermarkDetectionResult.detected ? "text-amber-400" : "text-green-400"
                    }`}>
                      {watermarkDetectionResult.detected
                        ? "AI-Generated Watermark Detected"
                        : "No AI Watermark Detected"
                      }
                    </h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      {watermarkDetectionResult.message}
                    </p>
                    <div className="mt-3">
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-muted-foreground">Confidence</span>
                        <span className={watermarkDetectionResult.detected ? "text-amber-400" : "text-green-400"}>
                          {Math.round(watermarkDetectionResult.confidence * 100)}%
                        </span>
                      </div>
                      <div className="h-2 bg-secondary rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            watermarkDetectionResult.detected ? "bg-amber-400" : "bg-green-400"
                          }`}
                          style={{ width: `${watermarkDetectionResult.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                    {watermarkDetectionResult.details && (
                      <div className="mt-3 text-xs text-muted-foreground">
                        <p>Detection layers: {watermarkDetectionResult.details.layers_detected || 0}/3</p>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}

            <div className="flex items-start gap-2 text-sm text-muted-foreground bg-muted p-3 rounded-lg">
              <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium text-foreground mb-1">About Audio Watermarking</p>
                <p>All AI-generated audio from Mimic AI contains invisible watermarks for identification purposes. This helps prevent misuse and provides transparency about the origin of audio content.</p>
              </div>
            </div>
          </div>
        )}

      </motion.div>
    </div>
  );
}
