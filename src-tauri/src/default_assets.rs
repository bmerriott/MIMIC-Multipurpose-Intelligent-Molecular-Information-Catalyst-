//! Default Assets Management
//! Handles copying bundled default persona assets to user data on first launch

use std::fs;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use tauri::AppHandle;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DefaultVoiceConfig {
    pub reference_text: String,
    pub voice_config: serde_json::Value,
}

/// Check if default assets have been initialized
pub fn is_default_assets_initialized(app_handle: &AppHandle) -> bool {
    let marker_file = get_marker_path(app_handle);
    marker_file.exists()
}

/// Get the marker file path
fn get_marker_path(app_handle: &AppHandle) -> PathBuf {
    let app_data_dir = app_handle.path_resolver()
        .app_data_dir()
        .unwrap_or_else(|| PathBuf::from("."));
    app_data_dir.join(".default_assets_initialized")
}

/// Initialize default assets - copy from resources to user data
pub fn initialize_default_assets(app_handle: &AppHandle) -> Result<DefaultAssetsInfo, String> {
    println!("[DefaultAssets] Initializing default persona assets...");
    
    let app_data_dir = app_handle.path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data directory")?;
    
    // Create necessary directories
    let voices_dir = app_data_dir.join("voices");
    let vrm_dir = app_data_dir.join("vrm_library");
    fs::create_dir_all(&voices_dir).map_err(|e| format!("Failed to create voices dir: {}", e))?;
    fs::create_dir_all(&vrm_dir).map_err(|e| format!("Failed to create vrm dir: {}", e))?;
    
    // Copy VRM file
    let vrm_info = copy_default_vrm(app_handle, &vrm_dir)?;
    
    // Copy voice files
    let voice_info = copy_default_voice(app_handle, &voices_dir)?;
    
    // Create marker file to indicate initialization is complete
    let marker_file = get_marker_path(app_handle);
    fs::write(&marker_file, "initialized").map_err(|e| format!("Failed to create marker: {}", e))?;
    
    println!("[DefaultAssets] Default assets initialized successfully");
    
    Ok(DefaultAssetsInfo {
        vrm: vrm_info,
        voice: voice_info,
    })
}

#[derive(Debug, Serialize, Clone)]
pub struct VrmAssetInfo {
    pub id: String,
    pub name: String,
    pub filename: String,
    pub path: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct VoiceAssetInfo {
    pub persona_id: String,
    pub json_path: String,
    pub wav_path: String,
    pub reference_text: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct DefaultAssetsInfo {
    pub vrm: VrmAssetInfo,
    pub voice: VoiceAssetInfo,
}

/// Copy the default VRM from resources to user data and add to library
fn copy_default_vrm(app_handle: &AppHandle, vrm_dir: &PathBuf) -> Result<VrmAssetInfo, String> {
    // Get the resource directory
    let resource_dir = app_handle.path_resolver()
        .resource_dir()
        .ok_or("Failed to get resource directory")?;
    
    let source_vrm = resource_dir.join("resources").join("default_persona").join("avatar.vrm");
    
    // Check if bundled VRM exists
    if !source_vrm.exists() {
        return Err(format!("Bundled VRM not found at: {:?}", source_vrm));
    }
    
    // Copy VRM to user's vrm_library
    let vrm_id = "default_bundled";
    let filename = format!("{}.vrm", vrm_id);
    let dest_vrm = vrm_dir.join(&filename);
    
    fs::copy(&source_vrm, &dest_vrm)
        .map_err(|e| format!("Failed to copy VRM: {}", e))?;
    
    // Add to library index
    let library_path = vrm_dir.join("library.json");
    let mut library = if library_path.exists() {
        let content = fs::read_to_string(&library_path)
            .map_err(|e| format!("Failed to read library: {}", e))?;
        serde_json::from_str(&content)
            .unwrap_or_else(|_| crate::vrm_library::VrmLibrary { entries: vec![] })
    } else {
        crate::vrm_library::VrmLibrary { entries: vec![] }
    };
    
    // Check if already in library
    if !library.entries.iter().any(|e| e.id == vrm_id) {
        let metadata = fs::metadata(&dest_vrm)
            .map_err(|e| format!("Failed to get VRM metadata: {}", e))?;
        
        let entry = crate::vrm_library::VrmEntry {
            id: vrm_id.to_string(),
            name: "Mimic (Default)".to_string(),
            filename: filename.clone(),
            size_bytes: metadata.len(),
            created_at: chrono::Utc::now().to_rfc3339(),
            thumbnail: None,
        };
        
        library.entries.push(entry);
        
        let library_json = serde_json::to_string_pretty(&library)
            .map_err(|e| format!("Failed to serialize library: {}", e))?;
        fs::write(&library_path, library_json)
            .map_err(|e| format!("Failed to write library: {}", e))?;
    }
    
    println!("[DefaultAssets] VRM copied to: {:?}", dest_vrm);
    
    Ok(VrmAssetInfo {
        id: vrm_id.to_string(),
        name: "Mimic (Default)".to_string(),
        filename,
        path: dest_vrm.to_string_lossy().to_string(),
    })
}

/// Copy the default voice from resources to user data
fn copy_default_voice(app_handle: &AppHandle, voices_dir: &PathBuf) -> Result<VoiceAssetInfo, String> {
    // Get the resource directory
    let resource_dir = app_handle.path_resolver()
        .resource_dir()
        .ok_or("Failed to get resource directory")?;
    
    let source_wav = resource_dir.join("resources").join("default_persona").join("voice.wav");
    let source_json = resource_dir.join("resources").join("default_persona").join("voice.json");
    
    // Check if bundled voice exists
    if !source_wav.exists() {
        return Err(format!("Bundled voice WAV not found at: {:?}", source_wav));
    }
    if !source_json.exists() {
        return Err(format!("Bundled voice JSON not found at: {:?}", source_json));
    }
    
    // Read the voice JSON to get reference text and config
    let voice_json_content = fs::read_to_string(&source_json)
        .map_err(|e| format!("Failed to read voice JSON: {}", e))?;
    let voice_data: serde_json::Value = serde_json::from_str(&voice_json_content)
        .map_err(|e| format!("Failed to parse voice JSON: {}", e))?;
    
    let reference_text = voice_data.get("reference_text")
        .and_then(|v| v.as_str())
        .unwrap_or("Hello! I'm Mimic, your personal AI assistant.")
        .to_string();
    
    // Read WAV file and encode as base64
    let wav_bytes = fs::read(&source_wav)
        .map_err(|e| format!("Failed to read voice WAV: {}", e))?;
    let audio_base64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &wav_bytes);
    
    // Create voice data for storage
    let persona_id = "default";
    let voice_config = voice_data.get("voice_config").cloned()
        .unwrap_or_else(|| serde_json::json!({
            "type": "created",
            "name": "Mimic Default",
            "params": {
                "pitch": 0,
                "speed": 1.0,
                "warmth": 0.6,
                "expressiveness": 0.7,
                "stability": 0.5,
                "clarity": 0.6,
                "breathiness": 0.3,
                "resonance": 0.5,
                "emotion": "neutral",
                "emphasis": 0.5,
                "pauses": 0.5,
                "energy": 0.6,
                "engine": "qwen3",
                "qwen3_model_size": "0.6B"
            }
        }));
    
    let voice_storage_data = crate::VoiceData {
        persona_id: persona_id.to_string(),
        audio_data: audio_base64,
        reference_text: Some(reference_text.clone()),
        voice_config: Some(voice_config),
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    
    // Save voice data
    let json_path = voices_dir.join(format!("voice_{}.json", persona_id));
    let wav_path = voices_dir.join(format!("voice_{}.wav", persona_id));
    
    let voice_json = serde_json::to_string_pretty(&voice_storage_data)
        .map_err(|e| format!("Failed to serialize voice data: {}", e))?;
    fs::write(&json_path, voice_json)
        .map_err(|e| format!("Failed to write voice JSON: {}", e))?;
    fs::write(&wav_path, &wav_bytes)
        .map_err(|e| format!("Failed to write voice WAV: {}", e))?;
    
    println!("[DefaultAssets] Voice copied to: {:?}", json_path);
    
    Ok(VoiceAssetInfo {
        persona_id: persona_id.to_string(),
        json_path: json_path.to_string_lossy().to_string(),
        wav_path: wav_path.to_string_lossy().to_string(),
        reference_text,
    })
}

/// Tauri command to check and initialize default assets
/// Returns asset info if available (even if already initialized)
#[tauri::command]
pub async fn check_and_initialize_defaults(app_handle: AppHandle) -> Result<Option<DefaultAssetsInfo>, String> {
    // First, try to initialize if not already done
    if !is_default_assets_initialized(&app_handle) {
        println!("[DefaultAssets] First launch - initializing assets...");
        match initialize_default_assets(&app_handle) {
            Ok(info) => return Ok(Some(info)),
            Err(e) => {
                eprintln!("[DefaultAssets] Initialization failed: {}", e);
                // Continue to try to get existing info
            }
        }
    }
    
    // Return asset info if assets exist (even if already initialized)
    // This allows frontend to sync persona config on every launch
    get_default_assets_info_sync(&app_handle)
}

/// Get default assets info synchronously (for use in check_and_initialize_defaults)
fn get_default_assets_info_sync(app_handle: &AppHandle) -> Result<Option<DefaultAssetsInfo>, String> {
    let app_data_dir = app_handle.path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data directory")?;
    
    let vrm_dir = app_data_dir.join("vrm_library");
    let voices_dir = app_data_dir.join("voices");
    
    let vrm_path = vrm_dir.join("default_bundled.vrm");
    let voice_json_path = voices_dir.join("voice_default.json");
    
    // Check if assets actually exist
    if !vrm_path.exists() {
        println!("[DefaultAssets] VRM not found at: {:?}", vrm_path);
        return Ok(None);
    }
    
    // Read voice JSON to get reference text
    let reference_text = if voice_json_path.exists() {
        match fs::read_to_string(&voice_json_path) {
            Ok(content) => {
                match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(voice_data) => voice_data.get("reference_text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Hello! I'm Mimic, your personal AI assistant.")
                        .to_string(),
                    Err(_) => "Hello! I'm Mimic, your personal AI assistant.".to_string(),
                }
            }
            Err(_) => "Hello! I'm Mimic, your personal AI assistant.".to_string(),
        }
    } else {
        "Hello! I'm Mimic, your personal AI assistant.".to_string()
    };
    
    let voice_wav_path = voices_dir.join("voice_default.wav");
    
    Ok(Some(DefaultAssetsInfo {
        vrm: VrmAssetInfo {
            id: "default_bundled".to_string(),
            name: "Mimic (Default)".to_string(),
            filename: "default_bundled.vrm".to_string(),
            path: vrm_path.to_string_lossy().to_string(),
        },
        voice: VoiceAssetInfo {
            persona_id: "default".to_string(),
            json_path: voice_json_path.to_string_lossy().to_string(),
            wav_path: voice_wav_path.to_string_lossy().to_string(),
            reference_text,
        },
    }))
}

/// Tauri command to get default assets info
#[tauri::command]
pub async fn get_default_assets_info(app_handle: AppHandle) -> Result<Option<DefaultAssetsInfo>, String> {
    get_default_assets_info_sync(&app_handle)
}
