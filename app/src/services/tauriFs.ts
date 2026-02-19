/**
 * Tauri Filesystem API Service
 * Provides persistent file storage for voice data and app configuration
 * This replaces IndexedDB for desktop app builds
 */

import { invoke } from "@tauri-apps/api/tauri";

export interface VoiceFileData {
  persona_id: string;
  audio_data: string; // base64
  reference_text?: string;
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
  created_at: string;
}

class TauriFilesystemService {
  private isAvailable: boolean = false;

  constructor() {
    // Check if Tauri API is available
    this.isAvailable = typeof window !== "undefined" && "__TAURI__" in window;
    console.log("[TauriFS] Available:", this.isAvailable);
  }

  checkAvailability(): boolean {
    return this.isAvailable;
  }

  /**
   * Save voice data to persistent file storage
   */
  async saveVoice(data: VoiceFileData): Promise<string> {
    if (!this.isAvailable) {
      throw new Error("Tauri filesystem API not available");
    }
    
    console.log("[TauriFS] Saving voice for persona:", data.persona_id);
    const path = await invoke<string>("save_voice_to_file", { data });
    console.log("[TauriFS] Voice saved to:", path);
    return path;
  }

  /**
   * Load voice data from persistent file storage
   */
  async loadVoice(personaId: string): Promise<VoiceFileData | null> {
    if (!this.isAvailable) {
      throw new Error("Tauri filesystem API not available");
    }
    
    console.log("[TauriFS] Loading voice for persona:", personaId);
    const data = await invoke<VoiceFileData | null>("load_voice_from_file", { personaId });
    console.log("[TauriFS] Voice loaded:", data ? "found" : "not found");
    return data;
  }

  /**
   * Delete voice data from persistent file storage
   */
  async deleteVoice(personaId: string): Promise<void> {
    if (!this.isAvailable) {
      throw new Error("Tauri filesystem API not available");
    }
    
    console.log("[TauriFS] Deleting voice for persona:", personaId);
    await invoke("delete_voice_file", { personaId });
    console.log("[TauriFS] Voice deleted");
  }

  /**
   * List all saved voice IDs
   */
  async listVoices(): Promise<string[]> {
    if (!this.isAvailable) {
      throw new Error("Tauri filesystem API not available");
    }
    
    const voices = await invoke<string[]>("list_saved_voices");
    console.log("[TauriFS] Found voices:", voices.length);
    return voices;
  }

  /**
   * Get the backend port (dynamically assigned by Tauri)
   */
  async getBackendPort(): Promise<number> {
    if (!this.isAvailable) {
      // Default port for development without Tauri
      return 8000;
    }
    
    const port = await invoke<number>("get_backend_port");
    console.log("[TauriFS] Backend port:", port);
    return port;
  }

  /**
   * Get the app data directory path
   */
  async getAppDataPath(): Promise<string> {
    if (!this.isAvailable) {
      throw new Error("Tauri filesystem API not available");
    }
    
    const path = await invoke<string>("get_app_data_path");
    console.log("[TauriFS] App data path:", path);
    return path;
  }

  /**
   * Show a file in the system's file manager
   */
  async showInFolder(path: string): Promise<void> {
    if (!this.isAvailable) {
      throw new Error("Tauri filesystem API not available");
    }
    
    await invoke("show_in_folder", { path });
  }
}

// Singleton instance
export const tauriFs = new TauriFilesystemService();
