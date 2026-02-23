import { useEffect, useRef } from "react";
import { useStore } from "@/store";
import { audioAnalyzer } from "@/services/audioAnalyzer";

/**
 * Global Audio Player - Persists across tab switches
 * This component lives in App.tsx and handles all audio playback
 * so that audio continues playing even when switching tabs.
 */
export function GlobalAudioPlayer() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaElementAudioSourceNode | null>(null);
  const analyserNodeRef = useRef<AnalyserNode | null>(null);
  const { audioPlayer, stopAudio } = useStore();
  
  const settings = useStore(state => state.settings);
  
  useEffect(() => {
    // Create audio element if it doesn't exist
    if (!audioRef.current) {
      audioRef.current = new Audio();
      
      audioRef.current.onended = () => {
        stopAudio();
      };
      
      audioRef.current.onerror = () => {
        console.error("[GlobalAudioPlayer] Audio playback error");
        stopAudio();
      };
    }
    
    const audio = audioRef.current;
    
    // Set volume from settings
    audio.volume = settings.voice_volume;
    
    // Update audio source if changed
    if (audioPlayer.audioData) {
      const newSrc = `data:audio/wav;base64,${audioPlayer.audioData}`;
      if (audio.src !== newSrc) {
        audio.src = newSrc;
      }
    } else if (audioPlayer.audioUrl) {
      if (audio.src !== audioPlayer.audioUrl) {
        audio.src = audioPlayer.audioUrl;
      }
    }
    
    // Handle play/pause state
    if (audioPlayer.isPlaying) {
      if (audio.paused) {
        // Set up audio analysis for avatar mouth animation
        try {
          if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
          }
          
          // Create source from audio element (only once per audio element)
          if (!sourceNodeRef.current && audioContextRef.current) {
            sourceNodeRef.current = audioContextRef.current.createMediaElementSource(audio);
            
            // Create analyser
            analyserNodeRef.current = audioContextRef.current.createAnalyser();
            analyserNodeRef.current.fftSize = 64;
            analyserNodeRef.current.smoothingTimeConstant = 0.8;
            
            // Connect: source -> analyser -> destination
            sourceNodeRef.current.connect(analyserNodeRef.current);
            analyserNodeRef.current.connect(audioContextRef.current.destination);
            
            // Connect to global analyzer for avatar
            if (analyserNodeRef.current) {
              audioAnalyzer.connectNode(analyserNodeRef.current);
            }
          }
          
          // Resume context if suspended
          if (audioContextRef.current.state === 'suspended') {
            audioContextRef.current.resume();
          }
        } catch (err) {
          console.error("[GlobalAudioPlayer] Audio analysis setup failed:", err);
        }
        
        audio.play().catch((err) => {
          console.error("[GlobalAudioPlayer] Play failed:", err);
          stopAudio();
        });
      }
    } else {
      if (!audio.paused) {
        audio.pause();
      }
    }
    
  }, [audioPlayer, settings.voice_volume, stopAudio]);
  
  // Cleanup on unmount (shouldn't happen in normal use)
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      
      // Cleanup audio context and analysis
      if (sourceNodeRef.current) {
        sourceNodeRef.current.disconnect();
        sourceNodeRef.current = null;
      }
      if (analyserNodeRef.current) {
        analyserNodeRef.current.disconnect();
        analyserNodeRef.current = null;
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      audioAnalyzer.cleanup();
    };
  }, []);
  
  // This is a non-visual component
  return null;
}
