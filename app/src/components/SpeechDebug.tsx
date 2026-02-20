import { useState, useEffect } from "react";
import { useStore } from "@/store";
import { Mic, MicOff, Activity, AlertCircle, X, Volume2 } from "lucide-react";
import { Button } from "./ui/button";
import { localSpeechRecognizer } from "@/services/localSpeechRecognition";

interface SpeechDebugProps {
  onClose?: () => void;
}

export function SpeechDebug({ onClose }: SpeechDebugProps) {
  const { isListening, isSpeaking, settings, currentPersona, setIsListening } = useStore();
  const [lastHeard, setLastHeard] = useState("");
  const [isSupported, setIsSupported] = useState(true);
  const [error, setError] = useState("");
  const [isTesting, setIsTesting] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [isCheckingAudio, setIsCheckingAudio] = useState(false);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    setIsSupported(!!SpeechRecognition);
    
    if (!SpeechRecognition) {
      setError("Speech recognition not supported. Use Chrome or Edge.");
    }
  }, []);

  // Hook into console logs to capture transcripts
  useEffect(() => {
    const originalLog = console.log;
    console.log = (...args) => {
      originalLog.apply(console, args);
      const msg = args.join(' ');
      if (msg.includes('Buffer:') || msg.includes('Interim:')) {
        const match = msg.match(/Buffer:|Interim:\s*(.+)/);
        if (match) {
          setLastHeard(match[1].trim());
        }
      }
    };

    return () => {
      console.log = originalLog;
    };
  }, []);

  // Check which microphone is actually receiving audio
  const checkAudioInput = async () => {
    console.log('🔊 Starting audio input check...');
    setIsCheckingAudio(true);
    setAudioLevel(0);
    
    let stream: MediaStream | null = null;
    let audioContext: AudioContext | null = null;
    
    try {
      const deviceId = settings.microphone_device || undefined;
      const constraints: MediaStreamConstraints = {
        audio: deviceId 
          ? { deviceId: { exact: deviceId } }
          : true
      };
      
      console.log('🔊 Requesting getUserMedia...');
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      const track = stream.getAudioTracks()[0];
      console.log('🔊 Using device:', track.label);
      
      audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      let checkCount = 0;
      let rafId: number;
      
      const checkLevel = () => {
        try {
          analyser.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
          const normalized = Math.min(100, (average / 128) * 100);
          setAudioLevel(normalized);
          checkCount++;
          
          if (checkCount < 100) { // Check for ~1.6 seconds at 60fps
            rafId = requestAnimationFrame(checkLevel);
          } else {
            console.log('🔊 Audio check complete');
            cleanup();
          }
        } catch (err) {
          console.error('🔊 Error in checkLevel:', err);
          cleanup();
        }
      };
      
      const cleanup = () => {
        console.log('🔊 Cleaning up audio check');
        if (rafId) cancelAnimationFrame(rafId);
        stream?.getTracks().forEach(t => t.stop());
        audioContext?.close();
        setIsCheckingAudio(false);
      };
      
      // Start checking
      checkLevel();
      
      // Safety timeout - force cleanup after 3 seconds
      setTimeout(() => {
        if (checkCount < 100) {
          console.log('🔊 Safety timeout triggered');
          cleanup();
        }
      }, 3000);
      
    } catch (error: any) {
      console.error('🔊 Failed to check audio input:', error);
      console.error('🔊 Error name:', error.name);
      console.error('🔊 Error message:', error.message);
      stream?.getTracks().forEach(t => t.stop());
      audioContext?.close();
      setIsCheckingAudio(false);
    }
  };

  // Test using Puter.js-based recognition (free OpenAI Whisper-1)
  const testMicrophone = async () => {
    if (isTesting) {
      // Stop testing
      stopTest();
      return;
    }
    
    setError("");
    setIsTesting(true);
    setLastHeard("🎤 Listening... speak now");
    
    localSpeechRecognizer.setBackendUrl(settings.tts_backend_url);
    
    localSpeechRecognizer.onResult((result) => {
      setLastHeard(`✅ Heard: "${result.transcript}"`);
      console.log('SpeechDebug - Result:', result.transcript);
      setIsTesting(false);
    });
    
    localSpeechRecognizer.onError((err) => {
      setError(`❌ Error: ${err}`);
      setIsTesting(false);
    });
    
    // Use selected device
    const deviceId = settings.microphone_device || undefined;
    const started = await localSpeechRecognizer.start(deviceId);
    
    if (!started) {
      setError("❌ Failed to start microphone");
      setIsTesting(false);
    }
  };
  
  // Stop the test
  const stopTest = () => {
    setIsTesting(false);
    localSpeechRecognizer.stop();
  };

  if (!isSupported) {
    return (
      <div className="fixed bottom-4 right-4 bg-destructive/90 text-white p-4 rounded-lg text-sm max-w-xs z-50">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4" />
            <span className="font-bold">Browser Not Supported</span>
          </div>
          {onClose && (
            <Button variant="ghost" size="icon" onClick={onClose} className="h-6 w-6 text-white">
              <X className="w-4 h-4" />
            </Button>
          )}
        </div>
        <p>Speech recognition requires Chrome or Edge.</p>
      </div>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 bg-card border border-border rounded-lg p-4 shadow-lg max-w-xs z-50">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-semibold flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary" />
          Voice Debug
        </span>
        <div className="flex items-center gap-1">
          <Button 
            variant={isListening ? "default" : "outline"} 
            size="sm"
            onClick={() => setIsListening(!isListening)}
            className="h-7 text-xs px-2"
          >
            {isListening ? (
              <><Mic className="w-3 h-3 mr-1" /> On</>
            ) : (
              <><MicOff className="w-3 h-3 mr-1" /> Off</>
            )}
          </Button>
          {onClose && (
            <Button variant="ghost" size="icon" onClick={onClose} className="h-7 w-7">
              <X className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>
      
      {error && (
        <div className="mb-3 p-2 bg-destructive/10 text-destructive text-xs rounded">
          {error}
        </div>
      )}
      
      <div className="space-y-1 text-xs">
        <div className="flex justify-between">
          <span className="text-muted-foreground">Status:</span>
          <span className={isListening ? "text-green-500 font-medium" : "text-red-400"}>
            {isListening ? "● Listening" : "○ Paused"}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-muted-foreground">Wake Word:</span>
          <span className="text-primary font-mono">"{currentPersona?.wake_words?.[0] || "Mimic"}"</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-muted-foreground">Speaking:</span>
          <span>{isSpeaking ? "Yes" : "No"}</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-muted-foreground">Auto Listen:</span>
          <span className={settings.auto_listen ? "text-green-500" : "text-muted-foreground"}>
            {settings.auto_listen ? "On" : "Off"}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-muted-foreground">Mic Device:</span>
          <span className="text-xs truncate max-w-[150px]" title={settings.microphone_device || "System Default"}>
            {settings.microphone_device ? "Custom Device" : "System Default"}
          </span>
        </div>
        
        <div className="mt-2 p-2 bg-amber-500/10 border border-amber-500/20 rounded text-[10px]">
          <strong className="text-amber-400">Virtual Audio Software:</strong>
          <p className="mt-1">
            If using VoiceMeeter or similar software, ensure your physical microphone 
            is selected as the input device in the virtual audio settings.
          </p>
        </div>
        
        {lastHeard && (
          <div className="mt-3 pt-2 border-t border-border">
            <span className="text-muted-foreground">Last:</span>
            <p className="text-foreground mt-1 font-mono text-[10px] bg-muted p-1.5 rounded">
              {lastHeard}
            </p>
          </div>
        )}
      </div>
      
      {/* Audio Input Check */}
      <div className="mt-3 pt-2 border-t border-border space-y-2">
        <Button 
          variant={isCheckingAudio ? "destructive" : "secondary"}
          size="sm" 
          onClick={isCheckingAudio ? () => setIsCheckingAudio(false) : checkAudioInput}
          className="w-full h-7 text-xs"
        >
          <Volume2 className="w-3 h-3 mr-1" />
          {isCheckingAudio ? "Stop Check" : "Check Audio Input"}
        </Button>
        
        {isCheckingAudio && (
          <div className="space-y-1">
            <div className="flex justify-between text-[10px]">
              <span>Input Level:</span>
              <span>{Math.round(audioLevel)}%</span>
            </div>
            <div className="h-2 bg-secondary rounded-full overflow-hidden">
              <div 
                className="h-full bg-green-500 transition-all duration-100"
                style={{ width: `${audioLevel}%` }}
              />
            </div>
            <p className="text-[10px] text-muted-foreground">
              {audioLevel > 10 ? "✅ Audio detected!" : "Speak now..."}
            </p>
          </div>
        )}
      </div>

      {/* Puter.js Speech Recognition Test */}
      <div className="mt-3 pt-2 border-t border-border space-y-2">
        <Button 
          variant={isTesting ? "destructive" : "outline"}
          size="sm" 
          onClick={isTesting ? stopTest : testMicrophone}
          className="w-full h-7 text-xs"
        >
          <Mic className="w-3 h-3 mr-1" />
          {isTesting ? "Stop Test" : "Test Microphone (Puter.js)"}
        </Button>
        
        <p className="text-[10px] text-amber-400">
          Uses Puter.js (free OpenAI Whisper-1). No backend required for transcription.
        </p>
        
        <p className="text-[10px] text-muted-foreground">
          Say "{currentPersona?.wake_words?.[0] || "Mimic"}, your command"
        </p>
      </div>
    </div>
  );
}
