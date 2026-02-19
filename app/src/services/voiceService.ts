export interface VoiceCreateData {
  audio_data: string;
  reference_text: string;
  created_at: string;
}

export interface TranscriptionResult {
  text: string;
  success: boolean;
  error?: string;
}

export interface SystemAudioResult {
  success: boolean;
  stream?: MediaStream;
  error?: string;
  requiresVirtualCable?: boolean;
}

/**
 * Service for voice-related operations including:
 * - Audio transcription using browser-based SpeechRecognition
 * - Voice creation data management
 * - System audio recording (with proper setup)
 */
export class VoiceService {
  /**
   * Transcribe audio using browser's SpeechRecognition API
   */
  async transcribeAudio(audioBlob: Blob): Promise<TranscriptionResult> {
    console.log("🎤 Starting audio transcription...");

    if (this.isBrowserTranscriptionAvailable()) {
      console.log("🎤 Using browser SpeechRecognition for transcription");
      return this.transcribeWithBrowser(audioBlob);
    }

    console.warn("🎤 No automatic transcription available");
    return {
      text: "",
      success: false,
      error: "Automatic transcription not available. Please type the spoken text manually.",
    };
  }

  /**
   * Check if browser supports SpeechRecognition
   */
  private isBrowserTranscriptionAvailable(): boolean {
    return (
      "webkitSpeechRecognition" in window ||
      "SpeechRecognition" in window
    );
  }

  /**
   * Transcribe using browser's SpeechRecognition API
   * 
   * BROWSER LIMITATION: The Web Speech API's SpeechRecognition interface
   * always listens to the system's default microphone input device. It CANNOT
   * be configured to listen to audio playback or audio files.
   * 
   * This means we cannot transcribe pre-recorded audio - the API will only
   * transcribe what the microphone hears in real-time.
   * 
   * For accurate transcription of audio files, server-side solutions like
   * OpenAI Whisper would be needed.
   */
  private async transcribeWithBrowser(_audioBlob: Blob): Promise<TranscriptionResult> {
    // Browser SpeechRecognition cannot transcribe audio files - it only
    // listens to the microphone. Return failure so user enters text manually.
    console.log("🎤 Browser SpeechRecognition cannot transcribe audio files - only microphone input");
    return {
      text: "",
      success: false,
      error: "Please type the spoken text manually. Browser speech recognition only works with live microphone input.",
    };
  }

  /**
   * Request system audio stream
   * 
   * BROWSER LIMITATION: Most browsers do not support direct system audio capture.
   * The getDisplayMedia API with audio:true only works in specific scenarios.
   * 
   * For reliable system audio recording, users need:
   * 1. A virtual audio cable (VB-Cable, BlackHole, etc.)
   * 2. Route system audio through the virtual cable
   * 3. Select the virtual cable as microphone input
   * 
   * This method will return success:false with requiresVirtualCable=true
   * if system audio capture is not available.
   */
  async requestSystemAudioStream(): Promise<SystemAudioResult> {
    console.log("🔊 Requesting system audio access...");
    
    // Check if we're in a context that supports display media
    if (!(navigator.mediaDevices as any).getDisplayMedia) {
      console.warn("🔊 getDisplayMedia not supported in this browser");
      return {
        success: false,
        error: "System audio recording not supported in this browser.",
        requiresVirtualCable: true,
      };
    }

    try {
      // Try to get system audio through display capture
      // Note: This only works when sharing a tab/window with audio in supported browsers
      const stream = await (navigator.mediaDevices as any).getDisplayMedia({
        video: false,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      // Check if we actually got audio tracks
      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) {
        console.warn("🔊 No audio tracks received from getDisplayMedia");
        // Stop any video tracks we might have received
        stream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
        return {
          success: false,
          error: "No audio captured. System audio requires a virtual audio cable setup.",
          requiresVirtualCable: true,
        };
      }

      console.log("🔊 System audio access granted, audio tracks:", audioTracks.length);
      
      // Listen for the 'ended' event which fires when user stops sharing
      audioTracks[0].onended = () => {
        console.log("🔊 System audio sharing stopped by user");
      };
      
      return {
        success: true,
        stream: stream,
      };
    } catch (error: any) {
      console.error("🔊 System audio access denied:", error);
      
      // Provide specific error messages
      let errorMessage = "Could not access system audio.";
      let requiresCable = false;
      
      if (error.name === "NotAllowedError") {
        errorMessage = "Permission denied. Please allow screen/audio sharing when prompted.";
      } else if (error.name === "NotSupportedError") {
        errorMessage = "System audio capture not supported in this browser.";
        requiresCable = true;
      } else if (error.name === "AbortError") {
        errorMessage = "System audio sharing was cancelled.";
      }
      
      return {
        success: false,
        error: errorMessage,
        requiresVirtualCable: requiresCable,
      };
    }
  }

  /**
   * Check if system audio recording might be supported
   * Note: This only checks API availability, actual support varies
   */
  isSystemAudioSupported(): boolean {
    const supported = "getDisplayMedia" in navigator.mediaDevices;
    console.log("🔊 System audio API support:", supported);
    return supported;
  }

  /**
   * Get instructions for setting up virtual audio cable
   */
  getVirtualCableInstructions(): string {
    return `To record system audio, you need a virtual audio cable:

Windows: Install VB-Cable (free)
1. Download from vb-audio.com/Cable
2. Set "CABLE Output" as your playback device
3. Select "CABLE Input" as microphone in this app

macOS: Use BlackHole (free)
1. Install via brew: brew install blackhole-2ch
2. Set BlackHole as audio device
3. Select it as microphone input

Linux: Use PulseAudio or JACK
1. Create a null sink: pactl load-module module-null-sink sink_name=VirtualCable
2. Route audio to it and select as input`;
  }
}

export const voiceService = new VoiceService();
