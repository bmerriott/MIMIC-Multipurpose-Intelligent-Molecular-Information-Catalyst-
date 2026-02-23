/**
 * Default Assets Service
 * Handles initialization of bundled default persona assets (VRM + voice)
 */

import { invoke } from "@tauri-apps/api/tauri";
import { useStore } from "@/store";
import { unifiedStorage } from "./unifiedStorage";

export interface DefaultAssetsInfo {
  vrm: {
    id: string;
    name: string;
    filename: string;
    path: string;
  };
  voice: {
    persona_id: string;
    json_path: string;
    wav_path: string;
    reference_text: string;
  };
}

/**
 * Check and initialize default assets on app startup
 * This should be called once when the app loads
 */
export async function initializeDefaultAssets(): Promise<DefaultAssetsInfo | null> {
  // Only run in Tauri environment
  if (!unifiedStorage.isTauri()) {
    console.log("[DefaultAssets] Not in Tauri environment, skipping");
    return null;
  }

  try {
    console.log("[DefaultAssets] Checking for default assets...");
    const info = await invoke<DefaultAssetsInfo | null>("check_and_initialize_defaults");
    
    if (info) {
      console.log("[DefaultAssets] Assets available:", info);
      
      // Always update the default persona to ensure it's synced
      // This fixes stale localStorage data from previous installs
      updateDefaultPersona(info);
      
      return info;
    } else {
      console.log("[DefaultAssets] Assets not available");
    }
    
    return null;
  } catch (error) {
    console.error("[DefaultAssets] Failed to initialize:", error);
    return null;
  }
}

/**
 * Get info about default assets without initializing
 */
export async function getDefaultAssetsInfo(): Promise<DefaultAssetsInfo | null> {
  if (!unifiedStorage.isTauri()) {
    return null;
  }

  try {
    return await invoke<DefaultAssetsInfo | null>("get_default_assets_info");
  } catch (error) {
    console.error("[DefaultAssets] Failed to get info:", error);
    return null;
  }
}

/**
 * Update the default persona to use the initialized assets
 * This runs on every app launch to ensure the default persona is synced
 */
function updateDefaultPersona(info: DefaultAssetsInfo): void {
  const store = useStore.getState();
  const { personas, currentPersona, setPersonas, setCurrentPersona } = store;
  
  // Find the default persona
  const defaultPersonaIndex = personas.findIndex(p => p.id === "default");
  if (defaultPersonaIndex === -1) {
    console.log("[DefaultAssets] No default persona found");
    return;
  }
  
  const defaultPersona = personas[defaultPersonaIndex];
  
  // ALWAYS update the VRM ID to ensure it's using the bundled one
  // This fixes issues where localStorage has stale VRM references
  const currentVrmId = defaultPersona.avatar_config?.vrm_id;
  const needsVrmUpdate = currentVrmId !== info.vrm.id;
  
  // Check if voice needs update (only if no voice exists)
  const needsVoiceUpdate = !defaultPersona.voice_create?.has_audio;
  
  if (!needsVrmUpdate && !needsVoiceUpdate) {
    console.log("[DefaultAssets] Default persona already up to date");
    return;
  }
  
  console.log("[DefaultAssets] Updating default persona:", {
    oldVrmId: currentVrmId,
    newVrmId: info.vrm.id,
    needsVoiceUpdate
  });
  
  // Build updated persona
  const updatedPersona = {
    ...defaultPersona,
    avatar_config: {
      ...defaultPersona.avatar_config,
      type: "vrm" as const,
      vrm_id: info.vrm.id,
    },
    updated_at: new Date().toISOString(),
  };
  
  // Only update voice if needed (don't overwrite user's custom voice)
  if (needsVoiceUpdate) {
    updatedPersona.voice_create = {
      has_audio: true,
      reference_text: info.voice.reference_text,
      created_at: new Date().toISOString(),
      voice_config: {
        type: "created" as const,
        name: "Mimic Default",
        params: {
          pitch: 0,
          speed: 1.0,
          warmth: 0.6,
          expressiveness: 0.7,
          stability: 0.5,
          clarity: 0.6,
          breathiness: 0.3,
          resonance: 0.5,
          emotion: "neutral" as const,
          emphasis: 0.5,
          pauses: 0.5,
          energy: 0.6,
          engine: "qwen3" as const,
          qwen3_model_size: "0.6B" as const,
        }
      }
    };
    updatedPersona.voice_id = "qwen3";
  }
  
  // Update personas array
  const updatedPersonas = [...personas];
  updatedPersonas[defaultPersonaIndex] = updatedPersona;
  setPersonas(updatedPersonas);
  
  // Update current persona if it's the default
  if (currentPersona?.id === "default") {
    setCurrentPersona(updatedPersona);
  }
  
  console.log("[DefaultAssets] Default persona updated successfully");
}

/**
 * Check if the bundled VRM is available in the library
 */
export async function isBundledVrmAvailable(): Promise<boolean> {
  if (!unifiedStorage.isTauri()) {
    return false;
  }
  
  try {
    const info = await getDefaultAssetsInfo();
    return info !== null;
  } catch {
    return false;
  }
}

/**
 * Get the bundled VRM ID for use in avatar config
 */
export async function getBundledVrmId(): Promise<string | null> {
  if (!unifiedStorage.isTauri()) {
    return null;
  }
  
  try {
    const info = await getDefaultAssetsInfo();
    return info?.vrm.id || null;
  } catch {
    return null;
  }
}
