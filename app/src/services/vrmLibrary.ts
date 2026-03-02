//! VRM Library Service
//! Handles communication with the Rust backend for VRM file management

import { invoke } from "@tauri-apps/api/tauri";
import { readBinaryFile } from "@tauri-apps/api/fs";

export interface VrmEntry {
  id: string;
  name: string;
  filename: string;
  size_bytes: number;
  created_at: string;
  thumbnail?: string;
}

export interface VrmLibrary {
  entries: VrmEntry[];
}

/**
 * List all VRMs in the local library
 */
export async function listVrmLibrary(): Promise<VrmLibrary> {
  return await invoke("list_vrm_library");
}

/**
 * Save a VRM file to the library
 */
export async function saveVrmToLibrary(
  name: string,
  file: File
): Promise<VrmEntry> {
  // Check if we're running in Tauri
  if (!isTauriEnv()) {
    throw new Error("VRM upload requires the desktop app. Please run Mimic AI from the built application.");
  }
  
  // Read file as array buffer and convert to byte array
  const arrayBuffer = await file.arrayBuffer();
  const data = new Uint8Array(arrayBuffer);
  
  // Convert to regular array for Tauri
  const byteArray = Array.from(data);
  
  return await invoke("save_vrm_to_library", {
    name,
    data: byteArray,
  });
}

/**
 * Get the file path/URL for a VRM (returns file:// URL)
 */
export async function getVrmFilePath(vrmId: string): Promise<string> {
  return await invoke("get_vrm_file_path", { vrmId });
}

/// Check if running inside Tauri
function isTauriEnv(): boolean {
  return typeof window !== "undefined" && "__TAURI__" in window;
}

/**
 * Load a VRM file as a blob URL for Three.js
 * This is necessary because Three.js GLTFLoader can't load file:// URLs directly
 */
export async function loadVrmAsBlobUrl(vrmId: string): Promise<string> {
  // Check if we're running in Tauri
  const isTauri = isTauriEnv();
  console.log("[VRM] loadVrmAsBlobUrl called, vrmId:", vrmId, "isTauri:", isTauri);
  
  if (!isTauri) {
    throw new Error("VRM loading requires the desktop app. Please run Mimic AI from the built application, not the browser.");
  }
  
  try {
    // Get the filesystem path from Rust
    console.log("[VRM] Invoking get_vrm_file_path...");
    const filePath = await invoke<string>("get_vrm_file_path", { vrmId });
    console.log("[VRM] Got file path:", filePath);
    
    // Read the file as binary using Tauri FS API
    console.log("[VRM] Reading file with readBinaryFile...");
    const bytes = await readBinaryFile(filePath);
    console.log("[VRM] Read bytes:", bytes.length);
    
    // Create a blob from the bytes
    console.log("[VRM] Creating blob...");
    const blob = new Blob([bytes as unknown as BlobPart], { type: "model/gltf-binary" });
    console.log("[VRM] Blob created, size:", blob.size);
    
    // Create a blob URL that Three.js can load
    const blobUrl = URL.createObjectURL(blob);
    console.log("[VRM] Created blob URL:", blobUrl);
    return blobUrl;
  } catch (error) {
    console.error("[VRM] Failed to load VRM:", error);
    // Log more details about the error
    if (error instanceof Error) {
      console.error("[VRM] Error name:", error.name);
      console.error("[VRM] Error message:", error.message);
      console.error("[VRM] Error stack:", error.stack);
    }
    throw error;
  }
}

/**
 * Revoke a blob URL to free memory
 */
export function revokeVrmBlobUrl(url: string): void {
  if (url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}

/**
 * Delete a VRM from the library
 */
export async function deleteVrmFromLibrary(vrmId: string): Promise<void> {
  return await invoke("delete_vrm_from_library", { vrmId });
}

/**
 * Rename a VRM in the library
 */
export async function renameVrmInLibrary(
  vrmId: string,
  newName: string
): Promise<VrmEntry> {
  return await invoke("rename_vrm_in_library", { vrmId, newName });
}

/**
 * Format file size for display
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// ============================================
// GENERIC 3D MODEL SUPPORT (GLB/GLTF)
// ============================================

export interface ModelEntry {
  id: string;
  name: string;
  filename: string;
  size_bytes: number;
  created_at: string;
  model_type: "vrm" | "glb" | "gltf";
  thumbnail?: string;
}

export interface ModelLibrary {
  entries: ModelEntry[];
}

/**
 * List all 3D models in the library (VRM + GLB/GLTF)
 */
export async function listModelLibrary(): Promise<ModelLibrary> {
  return await invoke("list_model_library");
}

/**
 * Save a generic 3D model (GLB/GLTF) to the library
 */
export async function saveModelToLibrary(
  name: string,
  file: File
): Promise<ModelEntry> {
  if (!isTauriEnv()) {
    throw new Error("Model upload requires the desktop app.");
  }
  
  const arrayBuffer = await file.arrayBuffer();
  const data = new Uint8Array(arrayBuffer);
  const byteArray = Array.from(data);
  
  // Determine model type from extension
  const ext = file.name.split('.').pop()?.toLowerCase() || 'glb';
  const modelType = ext === 'vrm' ? 'vrm' : ext === 'gltf' ? 'gltf' : 'glb';
  
  return await invoke("save_model_to_library", {
    name,
    data: byteArray,
    modelType,
  });
}

/**
 * Load any model file as a blob URL
 */
export async function loadModelAsBlobUrl(modelId: string): Promise<string> {
  const isTauri = isTauriEnv();
  
  if (!isTauri) {
    throw new Error("Model loading requires the desktop app.");
  }
  
  try {
    const filePath = await invoke<string>("get_model_file_path", { modelId });
    const bytes = await readBinaryFile(filePath);
    
    // Determine MIME type from extension
    const ext = filePath.split('.').pop()?.toLowerCase();
    const mimeType = ext === 'gltf' ? 'model/gltf+json' : 'model/gltf-binary';
    
    const blob = new Blob([bytes as unknown as BlobPart], { type: mimeType });
    return URL.createObjectURL(blob);
  } catch (error) {
    console.error("[Model] Failed to load model:", error);
    throw error;
  }
}

/**
 * Delete a model from the library
 */
export async function deleteModelFromLibrary(modelId: string): Promise<void> {
  return await invoke("delete_model_from_library", { modelId });
}

/**
 * Rename a model in the library
 */
export async function renameModelInLibrary(
  modelId: string,
  newName: string
): Promise<ModelEntry> {
  return await invoke("rename_model_in_library", { modelId, newName });
}
