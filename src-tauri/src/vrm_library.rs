//! VRM Library Management
//! Handles saving, loading, and managing VRM avatar files locally

use std::fs;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use tauri::AppHandle;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VrmEntry {
    pub id: String,
    pub name: String,
    pub filename: String,
    pub size_bytes: u64,
    pub created_at: String,
    pub thumbnail: Option<String>, // Base64 encoded thumbnail (optional)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VrmLibrary {
    pub entries: Vec<VrmEntry>,
}

/// Get the VRM library directory path
fn get_vrm_dir(app_handle: &AppHandle) -> Result<PathBuf, String> {
    let app_data_dir = app_handle.path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data directory")?;
    
    let vrm_dir = app_data_dir.join("vrm_library");
    
    // Create directory if it doesn't exist
    if !vrm_dir.exists() {
        fs::create_dir_all(&vrm_dir)
            .map_err(|e| format!("Failed to create VRM directory: {}", e))?;
    }
    
    Ok(vrm_dir)
}

/// Get the library index file path
fn get_library_index_path(app_handle: &AppHandle) -> Result<PathBuf, String> {
    let vrm_dir = get_vrm_dir(app_handle)?;
    Ok(vrm_dir.join("library.json"))
}

/// Load the VRM library index
fn load_library(app_handle: &AppHandle) -> Result<VrmLibrary, String> {
    let index_path = get_library_index_path(app_handle)?;
    
    if !index_path.exists() {
        return Ok(VrmLibrary { entries: vec![] });
    }
    
    let content = fs::read_to_string(&index_path)
        .map_err(|e| format!("Failed to read library index: {}", e))?;
    
    let library: VrmLibrary = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse library index: {}", e))?;
    
    Ok(library)
}

/// Save the VRM library index
fn save_library(app_handle: &AppHandle, library: &VrmLibrary) -> Result<(), String> {
    let index_path = get_library_index_path(app_handle)?;
    let content = serde_json::to_string_pretty(library)
        .map_err(|e| format!("Failed to serialize library: {}", e))?;
    
    fs::write(&index_path, content)
        .map_err(|e| format!("Failed to write library index: {}", e))?;
    
    Ok(())
}

/// List all VRMs in the library
#[tauri::command]
pub async fn list_vrm_library(app_handle: AppHandle) -> Result<VrmLibrary, String> {
    load_library(&app_handle)
}

/// Save a VRM file to the library
#[tauri::command]
pub async fn save_vrm_to_library(
    app_handle: AppHandle,
    name: String,
    data: Vec<u8>,
) -> Result<VrmEntry, String> {
    let vrm_dir = get_vrm_dir(&app_handle)?;
    
    // Generate unique ID
    let id = format!("vrm_{}", chrono::Utc::now().timestamp_millis());
    let filename = format!("{}.vrm", id);
    let filepath = vrm_dir.join(&filename);
    
    // Save the file
    fs::write(&filepath, &data)
        .map_err(|e| format!("Failed to save VRM file: {}", e))?;
    
    // Create entry
    let entry = VrmEntry {
        id: id.clone(),
        name: name.clone(),
        filename: filename.clone(),
        size_bytes: data.len() as u64,
        created_at: chrono::Utc::now().to_rfc3339(),
        thumbnail: None,
    };
    
    // Update library index
    let mut library = load_library(&app_handle)?;
    library.entries.push(entry.clone());
    save_library(&app_handle, &library)?;
    
    Ok(entry)
}

/// Get the filesystem path to a VRM file (for Tauri FS API)
#[tauri::command]
pub async fn get_vrm_file_path(
    app_handle: AppHandle,
    vrm_id: String,
) -> Result<String, String> {
    println!("[VRM] get_vrm_file_path called for vrm_id: {}", vrm_id);
    
    let library = load_library(&app_handle)?;
    println!("[VRM] Loaded library with {} entries", library.entries.len());
    
    let entry = library.entries
        .iter()
        .find(|e| e.id == vrm_id)
        .ok_or("VRM not found in library")?;
    
    println!("[VRM] Found entry: {} (filename: {})", entry.name, entry.filename);
    
    let vrm_dir = get_vrm_dir(&app_handle)?;
    let filepath = vrm_dir.join(&entry.filename);
    
    println!("[VRM] Full filepath: {:?}", filepath);
    println!("[VRM] File exists: {}", filepath.exists());
    
    // Return the actual filesystem path (not a URL)
    // This is what Tauri's readBinaryFile needs
    let path_str = filepath.to_string_lossy().to_string();
    println!("[VRM] Returning path: {}", path_str);
    
    Ok(path_str)
}

/// Delete a VRM from the library
#[tauri::command]
pub async fn delete_vrm_from_library(
    app_handle: AppHandle,
    vrm_id: String,
) -> Result<(), String> {
    let mut library = load_library(&app_handle)?;
    
    // Find and remove entry
    let entry_index = library.entries
        .iter()
        .position(|e| e.id == vrm_id)
        .ok_or("VRM not found in library")?;
    
    let entry = library.entries.remove(entry_index);
    
    // Delete the file
    let vrm_dir = get_vrm_dir(&app_handle)?;
    let filepath = vrm_dir.join(&entry.filename);
    
    if filepath.exists() {
        fs::remove_file(&filepath)
            .map_err(|e| format!("Failed to delete VRM file: {}", e))?;
    }
    
    // Save updated library
    save_library(&app_handle, &library)?;
    
    Ok(())
}

/// Rename a VRM in the library
#[tauri::command]
pub async fn rename_vrm_in_library(
    app_handle: AppHandle,
    vrm_id: String,
    new_name: String,
) -> Result<VrmEntry, String> {
    let mut library = load_library(&app_handle)?;
    
    let entry = library.entries
        .iter_mut()
        .find(|e| e.id == vrm_id)
        .ok_or("VRM not found in library")?;
    
    entry.name = new_name.clone();
    
    // Clone entry for return before saving (to avoid borrow conflict)
    let entry_clone = entry.clone();
    
    save_library(&app_handle, &library)?;
    
    Ok(entry_clone)
}
