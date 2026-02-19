import { useEffect, useRef } from "react";
import { useStore } from "@/store";

/**
 * Global Audio Player - Persists across tab switches
 * This component lives in App.tsx and handles all audio playback
 * so that audio continues playing even when switching tabs.
 */
export function GlobalAudioPlayer() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
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
    };
  }, []);
  
  // This is a non-visual component
  return null;
}
