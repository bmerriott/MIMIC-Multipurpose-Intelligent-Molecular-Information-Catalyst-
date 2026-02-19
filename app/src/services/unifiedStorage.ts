/**
 * Unified Storage Service
 * Uses Tauri filesystem API when available (desktop app)
 * Falls back to IndexedDB when running in browser
 */

import { voiceStorage } from "./voiceStorage";
import { tauriFs, type VoiceFileData } from "./tauriFs";
import { invoke } from "@tauri-apps/api/tauri";

export interface VoiceStorageData {
  audio_data: string;
  reference_text?: string;
  created_at: string;
  voice_config?: {
    type: "synthetic" | "created";
    params?: {
      pitch: number;
      speed: number;
      warmth?: number;
      expressiveness?: number;
      stability?: number;
      gender?: "neutral" | "masculine" | "feminine";
      age?: "young" | "adult" | "mature";
      seed?: number;
    };
    name?: string;
  };
}

export interface SetupStatus {
  python_installed: boolean;
  python_version?: string;
  dependencies_installed: boolean;
  missing_deps: string[];
}

class UnifiedStorageService {
  private useTauri: boolean = false;

  constructor() {
    // Check if we're in Tauri environment
    this.useTauri = typeof window !== "undefined" && "__TAURI__" in window;
    console.log("[UnifiedStorage] Using Tauri FS:", this.useTauri);
  }

  /**
   * Save voice data to persistent storage
   */
  async saveVoice(
    personaId: string,
    audioData: string,
    referenceText?: string,
    voiceConfig?: VoiceStorageData["voice_config"]
  ): Promise<void> {
    if (this.useTauri) {
      const data: VoiceFileData = {
        persona_id: personaId,
        audio_data: audioData,
        reference_text: referenceText,
        voice_config: voiceConfig,
        created_at: new Date().toISOString(),
      };
      await tauriFs.saveVoice(data);
    } else {
      // Fallback to IndexedDB
      await voiceStorage.saveVoice(personaId, audioData, referenceText);
    }
  }

  /**
   * Load voice data from persistent storage
   */
  async loadVoice(personaId: string): Promise<VoiceStorageData | null> {
    if (this.useTauri) {
      const data = await tauriFs.loadVoice(personaId);
      if (!data) return null;
      return {
        audio_data: data.audio_data,
        reference_text: data.reference_text,
        created_at: data.created_at,
        voice_config: data.voice_config,
      };
    } else {
      // Fallback to IndexedDB
      const data = await voiceStorage.getVoice(personaId);
      if (!data) return null;
      return {
        audio_data: data.audio_data,
        reference_text: data.reference_text,
        created_at: data.created_at,
      };
    }
  }

  /**
   * Delete voice data from persistent storage
   */
  async deleteVoice(personaId: string): Promise<void> {
    if (this.useTauri) {
      await tauriFs.deleteVoice(personaId);
    } else {
      // Fallback to IndexedDB
      await voiceStorage.deleteVoice(personaId);
    }
  }

  /**
   * List all saved voice IDs
   */
  async listVoices(): Promise<string[]> {
    if (this.useTauri) {
      return await tauriFs.listVoices();
    } else {
      // IndexedDB doesn't have a list function, return empty
      return [];
    }
  }

  /**
   * Check if running in Tauri environment
   */
  isTauri(): boolean {
    return this.useTauri;
  }

  /**
   * Get storage info (for UI display)
   */
  async getStorageInfo(): Promise<{ type: string; path?: string }> {
    if (this.useTauri) {
      try {
        const path = await tauriFs.getAppDataPath();
        return { type: "Tauri Filesystem", path };
      } catch {
        return { type: "Tauri Filesystem" };
      }
    } else {
      return { type: "Browser IndexedDB" };
    }
  }

  /**
   * Check setup status (Python, dependencies)
   */
  async checkSetupStatus(): Promise<SetupStatus | null> {
    if (!this.useTauri) return null;
    try {
      return await invoke<SetupStatus>("check_setup_status");
    } catch (e) {
      console.error("Failed to check setup status:", e);
      return null;
    }
  }

  /**
   * Install Python dependencies
   */
  async installPythonDeps(): Promise<void> {
    if (!this.useTauri) throw new Error("Not in Tauri environment");
    await invoke("install_python_deps");
  }
}

// Singleton instance
export const unifiedStorage = new UnifiedStorageService();
