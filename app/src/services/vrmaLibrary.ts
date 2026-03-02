//! VRMA Animation Library Service
//! Stores persona-specific VRMA animation file references
//! Note: In production, these would be stored in a proper file system

export interface VrmaEntry {
  id: string;
  name: string;
  path: string; // blob URL
  size: number;
  uploadedAt: string;
}

// In-memory storage for VRMA entries per persona
const vrmaStorage = new Map<string, VrmaEntry[]>();

/**
 * Get all VRMA animations for a persona
 */
export async function listPersonaVrmas(personaId: string): Promise<VrmaEntry[]> {
  return vrmaStorage.get(personaId) || [];
}

/**
 * Get VRMA paths as a record for VrmAvatar component
 */
export async function getPersonaVrmaPaths(personaId: string): Promise<Record<string, string>> {
  const entries = await listPersonaVrmas(personaId);
  const paths: Record<string, string> = {};
  
  for (const entry of entries) {
    paths[entry.name] = entry.path;
  }
  
  return paths;
}

/**
 * Save a VRMA file for a persona
 * Creates a blob URL that can be used by VrmAvatar
 */
export async function saveVrmaToPersona(
  personaId: string,
  file: File
): Promise<VrmaEntry> {
  try {
    // Create blob URL from file
    const blobUrl = URL.createObjectURL(file);
    
    const id = `${personaId}_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    const name = file.name.replace(/\.vrma$/i, "");
    
    const entry: VrmaEntry = {
      id,
      name,
      path: blobUrl,
      size: file.size,
      uploadedAt: new Date().toISOString(),
    };
    
    // Add to storage
    const existing = vrmaStorage.get(personaId) || [];
    vrmaStorage.set(personaId, [...existing, entry]);
    
    console.log(`[VRMA] Saved ${name} for persona ${personaId}`);
    return entry;
  } catch (error) {
    console.error("[VRMA] Failed to save VRMA:", error);
    throw error;
  }
}

/**
 * Delete a VRMA file
 */
export async function deleteVrma(personaId: string, vrmaId: string): Promise<void> {
  try {
    const existing = vrmaStorage.get(personaId) || [];
    const entry = existing.find(e => e.id === vrmaId);
    
    if (entry) {
      // Revoke blob URL to free memory
      URL.revokeObjectURL(entry.path);
      
      // Remove from storage
      vrmaStorage.set(personaId, existing.filter(e => e.id !== vrmaId));
    }
    
    console.log(`[VRMA] Deleted ${vrmaId}`);
  } catch (error) {
    console.error("[VRMA] Failed to delete VRMA:", error);
    throw error;
  }
}

/**
 * Delete all VRMAs for a persona
 */
export async function deleteAllPersonaVrmas(personaId: string): Promise<void> {
  try {
    const existing = vrmaStorage.get(personaId) || [];
    
    // Revoke all blob URLs
    for (const entry of existing) {
      URL.revokeObjectURL(entry.path);
    }
    
    vrmaStorage.delete(personaId);
    console.log(`[VRMA] Deleted all VRMAs for persona ${personaId}`);
  } catch (error) {
    console.error("[VRMA] Failed to delete persona VRMAs:", error);
    throw error;
  }
}
