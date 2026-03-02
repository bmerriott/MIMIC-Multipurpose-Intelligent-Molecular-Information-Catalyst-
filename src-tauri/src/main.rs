// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{Manager, State};
use serde::{Serialize, Deserialize};
use serde_json;
use std::path::PathBuf;
use base64::{Engine as _, engine::general_purpose};

// Windows-specific: flag to hide console windows
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;
#[cfg(target_os = "windows")]
const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
#[cfg(target_os = "windows")]
const DETACHED_PROCESS: u32 = 0x00000008;

mod setup;
mod dependency_manager;
mod vrm_library;
mod license_manager;
mod default_assets;
use setup::{check_setup_status, install_python_deps, find_python_executable, install_dependencies_visible};
use dependency_manager::{check_dependencies, install_dependencies_command, get_python_path};
use license_manager::{get_machine_id, verify_license_key};
use default_assets::{check_and_initialize_defaults, get_default_assets_info};

// Helper to write startup logs
fn log_startup(message: &str) {
    println!("{}", message);
    // Also try to write to a log file in app data (append mode)
    if let Some(data_dir) = dirs::data_dir() {
        let log_path = data_dir.join("com.mimicai.app").join("startup.log");
        let _ = std::fs::create_dir_all(log_path.parent().unwrap());
        // Append to log file
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .and_then(|mut file| {
                use std::io::Write;
                writeln!(file, "{}", message)
            });
    }
}

/// Start all backends via PowerShell script
fn start_all_backends() {
    log_startup("[*] Starting all backends via PowerShell script...");
    
    // Find the PowerShell script
    let exe_path = std::env::current_exe().unwrap();
    let exe_dir = exe_path.parent().unwrap();
    
    // Try multiple possible locations for the script
    let script_paths = vec![
        exe_dir.join("start-backends.ps1"),
        exe_dir.join("resources").join("start-backends.ps1"),
        std::env::current_dir().unwrap().join("src-tauri").join("start-backends.ps1"),
    ];
    
    let script_file = match script_paths.iter().find(|p| p.exists()) {
        Some(p) => p.clone(),
        None => {
            log_startup("    Could not find start-backends.ps1");
            return;
        }
    };
    
    log_startup(&format!("    Found script: {:?}", script_file));
    
    // Run the PowerShell script - use -WindowStyle Hidden to run silently
    #[cfg(target_os = "windows")]
    {
        let result = Command::new("powershell")
            .args(&[
                "-ExecutionPolicy", "Bypass",
                "-WindowStyle", "Hidden",
                "-File", &script_file.to_string_lossy()
            ])
            .creation_flags(CREATE_NO_WINDOW)
            .spawn();
        
        match result {
            Ok(child) => {
                log_startup(&format!("    PowerShell script started, PID: {}", child.id()));
            }
            Err(e) => {
                log_startup(&format!("    Failed to start PowerShell script: {}", e));
            }
        }
    }
}

// Global state for the Python backend process
struct BackendState {
    process: Mutex<Option<Child>>,
    port: Mutex<u16>,
}

// Data structures for voice storage
#[derive(Serialize, Deserialize, Debug)]
struct VoiceData {
    persona_id: String,
    audio_data: String, // base64
    reference_text: Option<String>,
    voice_config: Option<serde_json::Value>,
    created_at: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct AppConfig {
    voices_dir: PathBuf,
    data_dir: PathBuf,
}

fn main() {
    // Launch application immediately - backends start in background
    log_startup("[*] Launching Mimic AI Desktop...");
    
    // Get app data directory for backend
    let app_data_dir = dirs::data_dir()
        .map(|d| d.join("com.mimicai.app"))
        .unwrap_or_else(|| PathBuf::from("."));
    
    // Create directories
    let _ = std::fs::create_dir_all(app_data_dir.join("voices"));
    let _ = std::fs::create_dir_all(app_data_dir.join("backend_data"));
    
    // Start backends in INDEPENDENT background threads so they don't block each other
    // Each service starts in its own thread to prevent one from delaying the others
    log_startup("[*] Spawning backend services in independent threads...");
    
    // 1. Start Ollama in its own thread - completely isolated from Docker
    let ollama_thread = std::thread::spawn(move || {
        log_startup("    [Ollama Thread] Starting Ollama server...");
        start_ollama_server_blocking();
        log_startup("    [Ollama Thread] Ollama startup complete");
    });
    
    // 2. Start SearXNG Docker in its own thread - isolated from Ollama
    let docker_thread = std::thread::spawn(move || {
        log_startup("    [Docker Thread] Starting SearXNG container...");
        start_searxng_docker();
        log_startup("    [Docker Thread] Docker startup complete");
    });
    
    // 2.5 Install Python dependencies if needed (before starting Python backend)
    let setup_status = setup::run_setup_check();
    log_startup(&format!("[*] Python check: installed={}, deps={}", 
        setup_status.python_installed, setup_status.dependencies_installed));
    if !setup_status.missing_deps.is_empty() {
        log_startup(&format!("    Missing packages: {}", setup_status.missing_deps.join(", ")));
    }
    
    if setup_status.python_installed && !setup_status.dependencies_installed {
        log_startup("[*] Installing Python dependencies...");
        log_startup(&format!("    Missing: {}", setup_status.missing_deps.join(", ")));
        log_startup("    This may take 2-5 minutes on first launch...");
        
        // Install with visible window so user sees progress
        match setup::install_dependencies_visible() {
            Ok(_) => log_startup("    Dependencies installed successfully"),
            Err(e) => {
                log_startup(&format!("    ERROR: Failed to install dependencies: {}", e));
                log_startup("    TTS features will not work. Please install manually:");
                log_startup("    pip install fastapi uvicorn python-dotenv torch numpy");
            }
        }
    } else if !setup_status.python_installed {
        log_startup("[!] WARNING: Python not found. TTS will not work.");
        log_startup("    Please install Python 3.10-3.12 from https://python.org");
    } else {
        log_startup("    Python dependencies OK");
    }
    
    // 2.6 Install espeak-ng if needed (required for KittenTTS)
    // CRITICAL: Install BEFORE starting Python backend and wait for completion
    let espeak_just_installed = if !setup_status.espeak_ng_installed {
        log_startup("[*] espeak-ng not found, installing...");
        log_startup("    This is required for KittenTTS voice engine");
        
        // Use blocking installation with visible window so user sees progress
        match setup::install_espeak_ng_blocking() {
            Ok(_) => {
                log_startup("    espeak-ng installed successfully");
                // Verify installation worked
                if setup::check_espeak_ng() {
                    log_startup("    espeak-ng verified OK");
                    true
                } else {
                    log_startup("    WARNING: espeak-ng installation may require restart");
                    false
                }
            }
            Err(e) => {
                log_startup(&format!("    WARNING: Failed to install espeak-ng: {}", e));
                log_startup("    KittenTTS voice engine will not work without espeak-ng.");
                log_startup("    Please install manually from: https://github.com/espeak-ng/espeak-ng/releases");
                false
            }
        }
    } else {
        log_startup("    espeak-ng OK");
        false
    };
    
    // 3. Start Python TTS server in its own thread - isolated from others
    let app_data_dir_create = app_data_dir.clone();
    let python_thread = std::thread::spawn(move || {
        log_startup("    [Python Thread] Starting Python TTS server...");
        start_python_backend_blocking(&app_data_dir_create);
        log_startup("    [Python Thread] Python TTS startup complete");
    });
    
    // 3.5 If we just installed espeak-ng, restart Python backend after it starts
    // This ensures Python picks up the newly installed espeak-ng
    if espeak_just_installed {
        log_startup("[*] espeak-ng was just installed, will restart Python backend...");
        
        // Spawn a thread to wait and restart
        let app_data_dir_restart = app_data_dir.clone();
        std::thread::spawn(move || {
            // Wait for initial Python startup (10 seconds)
            log_startup("    [Restart Thread] Waiting for initial Python startup...");
            std::thread::sleep(std::time::Duration::from_secs(10));
            
            // Kill Python process
            log_startup("    [Restart Thread] Stopping Python for restart...");
            #[cfg(target_os = "windows")]
            {
                let _ = Command::new("taskkill")
                    .args(&["/F", "/IM", "python.exe"])
                    .creation_flags(CREATE_NO_WINDOW)
                    .output();
            }
            
            // Wait for process to die
            std::thread::sleep(std::time::Duration::from_secs(3));
            
            // Restart Python backend
            log_startup("    [Restart Thread] Restarting Python with espeak-ng support...");
            start_python_backend_blocking(&app_data_dir_restart);
            log_startup("    [Restart Thread] Python restart complete");
        });
    }
    
    // Note: We don't join these threads - they run independently
    // This ensures Ollama starts even if Docker takes a long time
    
    tauri::Builder::default()
        .manage(BackendState {
            process: Mutex::new(None),
            port: Mutex::new(8000),
        })
        .setup(|app| {
            let app_handle = app.handle();
            
            // ENSURE WINDOW IS SHOWN FIRST - regardless of other setup issues
            if let Some(main_window) = app_handle.get_window("main") {
                let _ = main_window.show();
                let _ = main_window.set_focus();
                let _ = main_window.center();
                log_startup("[*] Main window shown and focused");
            } else {
                log_startup("[!] Warning: Main window not found");
            }
            
            // Try to check for Python, but don't crash if it fails
            let inner_setup_status = setup::run_setup_check();
            
            if !inner_setup_status.python_installed {
                println!("Warning: Python not found - TTS features will not work");
                println!("Please install Python 3.10-3.12 from https://python.org");
            } else if !inner_setup_status.dependencies_installed {
                println!("Warning: Python dependencies not fully installed");
                println!("Some features may not work properly");
            }
            
            // Get app data directory (with error handling)
            let app_data_dir = match app_handle.path_resolver().app_data_dir() {
                Some(dir) => dir,
                None => {
                    eprintln!("Warning: Could not get app data directory");
                    // Continue anyway - window is already shown
                    return Ok(());
                }
            };
            
            // Create necessary directories (with error handling)
            if let Err(e) = std::fs::create_dir_all(app_data_dir.join("voices")) {
                eprintln!("Warning: Could not create voices directory: {}", e);
            }
            if let Err(e) = std::fs::create_dir_all(app_data_dir.join("backend_data")) {
                eprintln!("Warning: Could not create backend_data directory: {}", e);
            }

            println!("App data directory: {:?}", app_data_dir);

            // Store port
            {
                let state: State<BackendState> = app_handle.state();
                *state.port.lock().unwrap() = 8000;
            }
            
            // Initialize default assets (VRM + voice) on first launch
            std::thread::spawn(move || {
                // Small delay to let window show first
                std::thread::sleep(std::time::Duration::from_millis(500));
                
                match default_assets::initialize_default_assets(&app_handle) {
                    Ok(info) => {
                        log_startup(&format!("[DefaultAssets] Initialized: VRM={}, Voice={}", 
                            info.vrm.id, info.voice.persona_id));
                    }
                    Err(e) => {
                        // Don't fail if assets can't be initialized - app can still work
                        log_startup(&format!("[DefaultAssets] Warning: {}", e));
                    }
                }
            });

            Ok(())
        })
        .on_window_event(|event| match event.event() {
            tauri::WindowEvent::CloseRequested { api: _, .. } => {
                // Kill backends before closing
                log_startup("[*] Shutting down backends...");
                
                // Kill Python TTS server
                let _ = Command::new("taskkill")
                    .args(&["/F", "/IM", "python.exe"])
                    .creation_flags(CREATE_NO_WINDOW)
                    .output();
                
                // Kill Ollama
                let _ = Command::new("taskkill")
                    .args(&["/F", "/IM", "ollama.exe"])
                    .creation_flags(CREATE_NO_WINDOW)
                    .output();
                
                // Stop SearXNG container (optional - comment out if you want to keep it running)
                let _ = Command::new("docker")
                    .args(&["stop", "mimic-searxng"])
                    .creation_flags(CREATE_NO_WINDOW)
                    .output();
                
                log_startup("    Backends stopped");
                
                // Actually close the app now
                std::process::exit(0);
            }
            _ => {}
        })

        .invoke_handler(tauri::generate_handler![
            save_voice_to_file,
            save_voice_to_file,
            load_voice_from_file,
            delete_voice_file,
            list_saved_voices,
            get_backend_port,
            get_app_data_path,
            show_in_folder,
            scan_all_models,
            get_model_directories,
            vrm_library::list_vrm_library,
            vrm_library::save_vrm_to_library,
            vrm_library::delete_vrm_from_library,
            vrm_library::rename_vrm_in_library,
            vrm_library::get_vrm_file_path,
            check_setup_status,
            install_python_deps,
            check_dependencies,
            install_dependencies_command,
            get_python_path,
            get_machine_id,
            verify_license_key,
            check_and_initialize_defaults,
            get_default_assets_info,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn start_python_backend(app_data_dir: &PathBuf) -> u16 {
    // Use fixed port 8000 for TTS server (frontend expects this port)
    const PORT: u16 = 8000;
    let port = PORT;
    
    log_startup("[*] Starting Python backend...");
    
    // Determine Python executable and script path
    let python_exe = if cfg!(windows) {
        "python.exe"
    } else {
        "python3"
    };

    // First, check if Python is available
    #[cfg(target_os = "windows")]
    let python_check = Command::new(python_exe)
        .args(&["--version"])
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    
    #[cfg(not(target_os = "windows"))]
    let python_check = Command::new(python_exe)
        .args(&["--version"])
        .output();
    
    let python_available = match python_check {
        Ok(output) => {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                println!("    Python found: {}", version);
                log_startup(&format!("    Python found: {}", version));
                true
            } else {
                false
            }
        }
        Err(e) => {
            println!("[*] Python not found: {}", e);
            println!("    Please install Python 3.10-3.12 from https://python.org");
            log_startup(&format!("[*] Python not found: {}", e));
            return port;
        }
    };
    
    if !python_available {
        println!("[*] Python not available");
        log_startup("[*] Python not available");
        return port;
    }
    
    println!("[*] Python found, looking for backend...");
    log_startup("[*] Python found, looking for backend...");
    
    // Find the backend script - try multiple locations
    let exe_binding = std::env::current_exe().unwrap();
    let exe_dir = exe_binding.parent().unwrap();
    let current_dir = std::env::current_dir().unwrap();
    
    // Try multiple paths in order of likelihood
    let possible_paths = vec![
        // Packaged: resources folder next to exe
        exe_dir.join("resources").join("backend").join("tts_server_unified.py"),
        // Dev: running from project root
        current_dir.join("app").join("backend").join("tts_server_unified.py"),
        // Dev: running from src-tauri directory
        current_dir.join("..").join("app").join("backend").join("tts_server_unified.py"),
        // Dev: running from src-tauri/target/release
        exe_dir.join("..").join("..").join("..").join("app").join("backend").join("tts_server_unified.py"),
        // Dev: resources in src-tauri
        current_dir.join("resources").join("backend").join("tts_server_unified.py"),
        // Dev: resources next to exe (for src-tauri/target/release)
        exe_dir.join("..").join("..").join("resources").join("backend").join("tts_server_unified.py"),
    ];
    
    println!("Checking paths:");
    for p in &possible_paths {
        println!("      - {:?} (exists: {})", p, p.exists());
    }

    // Find existing path or skip backend startup
    let backend_script = match possible_paths.iter().find(|p| p.exists()) {
        Some(p) => {
            println!("    Found backend at: {:?}", p);
            log_startup(&format!("    Found backend at: {:?}", p));
            p.clone()
        }
        None => {
            eprintln!("Warning: Could not find tts_server_unified.py - Python backend will not start automatically");
            eprintln!("Please run the backend manually: python app/backend/tts_server_unified.py");
            log_startup("Warning: Could not find tts_server_unified.py");
            return port; // Return port anyway, backend can be started separately
        }
    };

    println!("Starting Python backend: {:?}", backend_script);
    println!("Port: {}", PORT);
    println!("Data directory: {:?}", app_data_dir);
    
    log_startup(&format!("    Starting backend on port {}", PORT));

    // Set environment variables
    let voices_dir = app_data_dir.join("voices");
    let backend_data_dir = app_data_dir.join("backend_data");
    
    // Get the backend directory (parent of the script)
    let backend_dir = backend_script.parent().unwrap().to_path_buf();
    
    // Check if .env exists in backend directory, if not create one from bundled template
    let env_file = backend_dir.join(".env");
    let user_env_file = app_data_dir.join("backend.env");
    
    if !env_file.exists() && user_env_file.exists() {
        // Use user's env file from app data
        println!("Using .env from app data directory");
    } else if !env_file.exists() {
        // No .env found - the app will use environment variables or defaults
        println!("Warning: No .env file found - using defaults");
    }

    // Try to start the backend, but don't crash if it fails
    #[cfg(target_os = "windows")]
    {
        let start_result = Command::new(python_exe)
            .arg(&backend_script)
            .current_dir(&backend_dir)  // Set working directory so Python can find .env
            .env("MIMIC_PORT", port.to_string())
            .env("MIMIC_VOICES_DIR", &voices_dir)
            .env("MIMIC_DATA_DIR", &backend_data_dir)
            // Pass SearXNG URL - backend expects it on port 8080
            .env("SEARXNG_URL", "http://localhost:8080")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .creation_flags(CREATE_NO_WINDOW)  // Just hide window, don't detach
            .spawn();
        
        match start_result {
            Ok(child) => {
                println!("    Python backend started, PID: {}", child.id());
                log_startup(&format!("    Python backend started, PID: {}", child.id()));
            }
            Err(e) => {
                eprintln!("Warning: Failed to start Python backend: {}", e);
                eprintln!("Please run the backend manually: python app/backend/tts_server_unified.py");
                log_startup(&format!("    Failed to start Python backend: {}", e));
            }
        }
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        if let Err(e) = Command::new(python_exe)
            .arg(&backend_script)
            .current_dir(&backend_dir)
            .env("MIMIC_PORT", port.to_string())
            .env("MIMIC_VOICES_DIR", &voices_dir)
            .env("MIMIC_DATA_DIR", &backend_data_dir)
            .env("SEARXNG_URL", "http://localhost:8080")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            eprintln!("Warning: Failed to start Python backend: {}", e);
            log_startup(&format!("    Failed to start Python backend: {}", e));
        }
    }

    port
}

/// Information about a discovered Ollama model
#[derive(Serialize, Deserialize, Debug, Clone)]
struct DiscoveredModel {
    name: String,
    model: String,
    size: u64,
    modified_at: String,
    digest: String,
    source_dir: String,
    details: ModelDetails,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
struct ModelDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_model: Option<String>,
    format: String,
    family: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    families: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameter_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantization_level: Option<String>,
}

/// Detect ALL Ollama model directories on the system
fn detect_all_ollama_model_dirs() -> Vec<PathBuf> {
    let mut found_dirs = Vec::new();
    
    // Check if OLLAMA_MODELS env var is already set
    if let Ok(models_dir) = std::env::var("OLLAMA_MODELS") {
        let path = PathBuf::from(models_dir);
        if path.exists() {
            log_startup(&format!("    Found OLLAMA_MODELS env var: {:?}", path));
            found_dirs.push(path);
        }
    }
    
    // Check common locations
    let possible_locations = vec![
        // User's home directory (most common for Ollama Desktop and CLI)
        dirs::home_dir().map(|h| h.join(".ollama").join("models")),
        // LocalAppData (alternative location)
        std::env::var("LOCALAPPDATA").ok().map(|p| PathBuf::from(p).join("Ollama").join("models")),
        // AppData (another alternative)
        std::env::var("APPDATA").ok().map(|p| PathBuf::from(p).join("Ollama").join("models")),
        // ProgramData (system-wide installation)
        std::env::var("ProgramData").ok().map(|p| PathBuf::from(p).join("Ollama").join("models")),
    ];
    
    for location in possible_locations.iter().flatten() {
        if location.exists() && !found_dirs.contains(location) {
            // Verify it actually has models
            if let Ok(entries) = std::fs::read_dir(location) {
                let has_content = entries.count() > 0;
                if has_content {
                    log_startup(&format!("    Found Ollama models directory: {:?}", location));
                    found_dirs.push(location.clone());
                }
            }
        }
    }
    
    found_dirs
}

/// Scan all Ollama model directories and return unified model list
fn scan_all_ollama_models() -> Vec<DiscoveredModel> {
    let dirs = detect_all_ollama_model_dirs();
    let mut all_models: Vec<DiscoveredModel> = Vec::new();
    
    log_startup(&format!("[Model Scanner] Scanning {} Ollama directories...", dirs.len()));
    
    for models_dir in &dirs {
        let manifests_dir = models_dir.join("manifests");
        if !manifests_dir.exists() {
            continue;
        }
        
        // Scan registry.ollama.ai/library
        let library_dir = manifests_dir.join("registry.ollama.ai").join("library");
        if library_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&library_dir) {
                for entry in entries.flatten() {
                    let model_name = entry.file_name().to_string_lossy().to_string();
                    if let Ok(tags) = std::fs::read_dir(&entry.path()) {
                        for tag_entry in tags.flatten() {
                            if let Some(model) = parse_model_manifest(&tag_entry.path(), &model_name, models_dir) {
                                // Check if we already have this model (avoid duplicates)
                                if !all_models.iter().any(|m| m.name == model.name) {
                                    all_models.push(model);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Scan other namespaces (like aeline/, etc.)
        if let Ok(entries) = std::fs::read_dir(&library_dir) {
            for entry in entries.flatten() {
                let namespace = entry.file_name().to_string_lossy().to_string();
                if namespace == "library" { continue; }
                
                if let Ok(models) = std::fs::read_dir(&entry.path()) {
                    for model_entry in models.flatten() {
                        let model_name = format!("{}/{}", namespace, model_entry.file_name().to_string_lossy());
                        if let Ok(tags) = std::fs::read_dir(&model_entry.path()) {
                            for tag_entry in tags.flatten() {
                                if let Some(model) = parse_model_manifest(&tag_entry.path(), &model_name, models_dir) {
                                    if !all_models.iter().any(|m| m.name == model.name) {
                                        all_models.push(model);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    log_startup(&format!("[Model Scanner] Found {} unique models", all_models.len()));
    all_models
}

/// Parse a single model manifest file
fn parse_model_manifest(manifest_path: &PathBuf, model_name: &str, source_dir: &PathBuf) -> Option<DiscoveredModel> {
    use std::io::Read;
    
    let tag = manifest_path.file_name()?.to_string_lossy().to_string();
    let full_name = format!("{}:{}", model_name, tag);
    
    // Read manifest file
    let mut file = std::fs::File::open(manifest_path).ok()?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).ok()?;
    
    // Parse JSON manifest
    let manifest: serde_json::Value = serde_json::from_str(&contents).ok()?;
    
    // Get config digest for the model identifier
    let digest = manifest.get("config")?.get("digest")?.as_str()?.to_string();
    
    // Calculate TOTAL model size from all layers (not just config size)
    let mut total_size: u64 = 0;
    
    // Add config size
    if let Some(config_size) = manifest.get("config").and_then(|c| c.get("size")).and_then(|s| s.as_u64()) {
        total_size += config_size;
    }
    
    // Add all layer sizes
    if let Some(layers) = manifest.get("layers").and_then(|l| l.as_array()) {
        for layer in layers {
            if let Some(layer_size) = layer.get("size").and_then(|s| s.as_u64()) {
                total_size += layer_size;
            }
        }
    }
    
    // Get modified time from file metadata
    let modified_at = std::fs::metadata(manifest_path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let modified_at_iso = chrono::DateTime::from_timestamp(modified_at as i64, 0)
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_default();
    
    // Extract details from manifest - try to get parameter size from layer media types
    let mut parameter_size = None;
    let mut quantization_level = None;
    let mut families = Vec::new();
    
    // Parse layer media types for model info
    if let Some(layers) = manifest.get("layers").and_then(|l| l.as_array()) {
        for layer in layers {
            if let Some(media_type) = layer.get("mediaType").and_then(|m| m.as_str()) {
                // Look for model layer with quantization info
                if media_type.contains("model") {
                    // Try to extract parameter size and quantization from media type or annotations
                    if let Some(annotations) = layer.get("annotations").and_then(|a| a.as_object()) {
                        if let Some(param) = annotations.get("org.opencontainers.image.title").and_then(|t| t.as_str()) {
                            // Extract parameter size from title like "qwen3-30b-q4_k_m.gguf"
                            if param.contains("0.5b") { parameter_size = Some("0.5B".to_string()); }
                            else if param.contains("0.6b") { parameter_size = Some("0.6B".to_string()); }
                            else if param.contains("1b") { parameter_size = Some("1B".to_string()); }
                            else if param.contains("1.5b") { parameter_size = Some("1.5B".to_string()); }
                            else if param.contains("3b") { parameter_size = Some("3B".to_string()); }
                            else if param.contains("7b") { parameter_size = Some("7B".to_string()); }
                            else if param.contains("8b") { parameter_size = Some("8B".to_string()); }
                            else if param.contains("14b") { parameter_size = Some("14B".to_string()); }
                            else if param.contains("30b") { parameter_size = Some("30B".to_string()); }
                            else if param.contains("32b") { parameter_size = Some("32B".to_string()); }
                            else if param.contains("70b") { parameter_size = Some("70B".to_string()); }
                            
                            // Extract quantization
                            if param.to_lowercase().contains("q4_") || param.to_lowercase().contains("q4-") {
                                quantization_level = Some("Q4".to_string());
                            } else if param.to_lowercase().contains("q5_") || param.to_lowercase().contains("q5-") {
                                quantization_level = Some("Q5".to_string());
                            } else if param.to_lowercase().contains("q6_") || param.to_lowercase().contains("q6-") {
                                quantization_level = Some("Q6".to_string());
                            } else if param.to_lowercase().contains("q8_") || param.to_lowercase().contains("q8-") {
                                quantization_level = Some("Q8".to_string());
                            } else if param.to_lowercase().contains("fp16") {
                                quantization_level = Some("FP16".to_string());
                            }
                        }
                    }
                    
                    // Extract family from media type
                    if media_type.contains("llama") { families.push("llama".to_string()); }
                    else if media_type.contains("qwen") { families.push("qwen".to_string()); }
                    else if media_type.contains("mistral") { families.push("mistral".to_string()); }
                    else if media_type.contains("phi") { families.push("phi".to_string()); }
                }
            }
        }
    }
    
    // If no family found, try to infer from model name
    if families.is_empty() {
        let name_lower = full_name.to_lowercase();
        if name_lower.contains("llama") { families.push("llama".to_string()); }
        else if name_lower.contains("qwen") { families.push("qwen".to_string()); }
        else if name_lower.contains("mistral") { families.push("mistral".to_string()); }
        else if name_lower.contains("phi") { families.push("phi".to_string()); }
        else if name_lower.contains("gemma") { families.push("gemma".to_string()); }
        else { families.push("unknown".to_string()); }
    }
    
    // Extract details from manifest
    let details = ModelDetails {
        parent_model: manifest.get("parent_model").and_then(|v| v.as_str()).map(|s| s.to_string()),
        format: "gguf".to_string(), // Ollama models are typically GGUF
        family: families.get(0).cloned().unwrap_or_else(|| "unknown".to_string()),
        families: if families.len() > 1 { Some(families) } else { None },
        parameter_size,
        quantization_level,
    };
    
    Some(DiscoveredModel {
        name: full_name.clone(),
        model: full_name,
        size: total_size,
        modified_at: modified_at_iso,
        digest,
        source_dir: source_dir.to_string_lossy().to_string(),
        details,
    })
}

/// Detect the primary Ollama models directory (for backward compatibility)
fn detect_ollama_models_dir() -> Option<PathBuf> {
    let dirs = detect_all_ollama_model_dirs();
    dirs.into_iter().next()
}

/// Start Ollama server silently in the background
fn start_ollama_server() {
    log_startup("[*] Checking for Ollama...");
    
    // Check if Ollama is already running and responding
    match reqwest::blocking::get("http://localhost:11434/api/tags") {
        Ok(response) if response.status().is_success() => {
            println!("    Ollama is already running and responding");
            log_startup("    Ollama is already running and responding");
            return;
        }
        _ => {
            println!("    Ollama not responding, will start it");
            log_startup("    Ollama not responding, will start it");
        }
    }
    
    // Find Ollama executable
    let ollama_paths = vec![
        PathBuf::from("C:/Program Files/Ollama/ollama.exe"),
        PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_default()).join("Programs/Ollama/ollama.exe"),
    ];
    
    let ollama_path = ollama_paths.iter().find(|p| p.exists());
    
    match ollama_path {
        Some(path) => {
            println!("    Starting Ollama server from: {:?}", path);
            log_startup(&format!("    Starting Ollama from: {:?}", path));
            
            #[cfg(target_os = "windows")]
            {
                // Detect and set Ollama models directory
                let models_dir = detect_ollama_models_dir();
                
                // Start Ollama serve silently - don't kill existing, just start new if needed
                // Ollama will handle the port binding itself
                // Use null stdout/stderr to avoid issues with CREATE_NO_WINDOW and pipes
                // Set required environment variables for Tauri WebView compatibility
                let mut cmd = Command::new(path);
                cmd.arg("serve")
                    .env("OLLAMA_ORIGINS", "*")
                    .env("OLLAMA_HOST", "0.0.0.0:11434")
                    .creation_flags(CREATE_NO_WINDOW)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null());
                
                // Set OLLAMA_MODELS if we detected a directory
                if let Some(ref dir) = models_dir {
                    log_startup(&format!("    Setting OLLAMA_MODELS={:?}", dir));
                    cmd.env("OLLAMA_MODELS", dir);
                }
                
                // Set HOME/USERPROFILE to ensure Ollama uses correct user directory
                if let Some(home) = dirs::home_dir() {
                    cmd.env("USERPROFILE", &home);
                    log_startup(&format!("    Setting USERPROFILE={:?}", home));
                }
                
                let start_result = cmd.spawn();
                
                match start_result {
                    Ok(child) => {
                        println!("    Ollama server started, PID: {}", child.id());
                        log_startup(&format!("    Ollama server started, PID: {}", child.id()));
                        
                        // Wait a moment for Ollama to start
                        std::thread::sleep(std::time::Duration::from_secs(2));
                        
                        // Verify it's running
                        match reqwest::blocking::get("http://127.0.0.1:11434/api/tags") {
                            Ok(response) if response.status().is_success() => {
                                println!("    Ollama is now responding");
                                log_startup("    Ollama is now responding");
                            }
                            _ => {
                                println!("    Warning: Ollama started but not yet responding");
                                log_startup("    Warning: Ollama started but not yet responding");
                            }
                        }
                    }
                    Err(e) => {
                        println!("    Failed to start Ollama: {}", e);
                        log_startup(&format!("    Failed to start Ollama: {}", e));
                    }
                }
            }
            
            #[cfg(not(target_os = "windows"))]
            {
                let _ = Command::new(path)
                    .arg("serve")
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn();
                    
                println!("    Ollama server started");
                log_startup("    Ollama server started");
            }
        }
        None => {
            println!("    Ollama not found - please install from https://ollama.com");
            log_startup("    Ollama not found - please install from https://ollama.com");
        }
    }
    
    // Check and install required models
    check_and_install_ollama_models();
}

/// Check for required Ollama models and install if missing
fn check_and_install_ollama_models() {
    log_startup("[*] Checking required Ollama models...");
    
    let required_models = vec![
        ("qwen3:0.6b", "Qwen3 0.6B (Router)"),
        ("llama3.2:latest", "Llama 3.2 (Brain)"),
    ];
    
    // Check if Ollama is responding
    let models_list = match reqwest::blocking::get("http://localhost:11434/api/tags") {
        Ok(response) if response.status().is_success() => {
            match response.json::<serde_json::Value>() {
                Ok(json) => {
                    json.get("models")
                        .and_then(|m| m.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|m| m.get("name").and_then(|n| n.as_str()))
                                .map(|s| s.to_string())
                                .collect::<Vec<String>>()
                        })
                        .unwrap_or_default()
                }
                Err(_) => {
                    log_startup("    Warning: Could not parse Ollama models list");
                    return;
                }
            }
        }
        _ => {
            log_startup("    Warning: Ollama not responding, skipping model check");
            return;
        }
    };
    
    // Check each required model
    for (model_name, display_name) in required_models {
        let is_installed = models_list.iter().any(|m| {
            m.starts_with(model_name) || 
            (model_name.contains(":") && m == model_name.split(':').next().unwrap_or(""))
        });
        
        if is_installed {
            log_startup(&format!("    ✓ {} is installed", display_name));
        } else {
            log_startup(&format!("    → Installing {}...", display_name));
            
            // Find Ollama executable
            let ollama_exe = find_ollama_executable();
            
            if let Some(ollama_path) = ollama_exe {
                // Pull the model in background
                let model = model_name.to_string();
                std::thread::spawn(move || {
                    log_startup(&format!("    Starting background pull for {}", model));
                    
                    let pull_result = std::process::Command::new(&ollama_path)
                        .args(&["pull", &model])
                        .creation_flags(CREATE_NO_WINDOW)
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .output();
                    
                    match pull_result {
                        Ok(output) if output.status.success() => {
                            log_startup(&format!("    ✓ {} installed successfully", model));
                        }
                        Ok(_) => {
                            log_startup(&format!("    ✗ Failed to install {} (will retry on next start)", model));
                        }
                        Err(e) => {
                            log_startup(&format!("    ✗ Error installing {}: {}", model, e));
                        }
                    }
                });
            } else {
                log_startup(&format!("    ✗ Cannot install {} - Ollama executable not found", display_name));
            }
        }
    }
}

/// Find Ollama executable path
fn find_ollama_executable() -> Option<PathBuf> {
    let paths = vec![
        PathBuf::from("C:/Program Files/Ollama/ollama.exe"),
        PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_default()).join("Programs/Ollama/ollama.exe"),
        PathBuf::from("ollama"),  // Try PATH
    ];
    
    paths.into_iter().find(|p| {
        if p == &PathBuf::from("ollama") {
            // Check if command exists in PATH
            std::process::Command::new("ollama")
                .arg("--version")
                .creation_flags(CREATE_NO_WINDOW)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        } else {
            p.exists()
        }
    })
}

/// Start Docker with SearXNG container silently
fn start_searxng_docker() {
    log_startup("[*] Checking for Docker/SearXNG...");
    
    // Check if Docker is installed (hidden)
    #[cfg(target_os = "windows")]
    let docker_check = Command::new("docker")
        .args(&["info"])
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    
    #[cfg(not(target_os = "windows"))]
    let docker_check = Command::new("docker")
        .args(&["info"])
        .output();
    
    // Check if Docker is installed AND running
    let docker_running = match &docker_check {
        Ok(output) => output.status.success(),
        Err(_) => false,
    };
    
    if !docker_running {
        log_startup("    Docker daemon not running, checking if Docker Desktop is installed...");
        
        // Try with --version as fallback check to see if Docker CLI is installed
        #[cfg(target_os = "windows")]
        let docker_check = Command::new("docker")
            .args(&["--version"])
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();
        
        let docker_installed = match docker_check {
            Ok(output) if output.status.success() => {
                log_startup("    Docker CLI is installed but daemon not running");
                true
            }
            _ => {
                log_startup("    Docker not found - web search will use fallback");
                false
            }
        };
        
        // If Docker is installed but not running, try to start Docker Desktop
        if docker_installed {
            #[cfg(target_os = "windows")]
            {
                let docker_desktop_paths = vec![
                    PathBuf::from("C:/Program Files/Docker/Docker/Docker Desktop.exe"),
                    PathBuf::from("C:/Program Files (x86)/Docker/Docker/Docker Desktop.exe"),
                ];
                
                let mut docker_launched = false;
                for docker_path in &docker_desktop_paths {
                    if docker_path.exists() {
                        log_startup(&format!("    Found Docker Desktop at: {:?}", docker_path));
                        println!("    Starting Docker Desktop...");
                        
                        // Launch Docker Desktop directly (no PowerShell to avoid path issues)
                        let launch_result = Command::new(docker_path)
                            .creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP)
                            .stdout(Stdio::null())
                            .stderr(Stdio::null())
                            .spawn();
                        
                        match launch_result {
                            Ok(_) => {
                                log_startup("    Docker Desktop launch command executed");
                                docker_launched = true;
                                
                                // Wait for Docker to be ready (up to 2 minutes)
                                println!("    Waiting for Docker to start...");
                                let mut docker_ready = false;
                                for i in 0..120 {
                                    std::thread::sleep(std::time::Duration::from_secs(1));
                                    
                                    if i % 10 == 0 {
                                        print!(".");
                                    }
                                    
                                    // Check if Docker daemon is responding
                                    let docker_check = Command::new("docker")
                                        .args(&["info"])
                                        .creation_flags(CREATE_NO_WINDOW)
                                        .stdout(Stdio::null())
                                        .stderr(Stdio::null())
                                        .output();
                                    
                                    if let Ok(output) = docker_check {
                                        if output.status.success() {
                                            println!("\n    Docker is ready! (took {}s)", i + 1);
                                            log_startup(&format!("    Docker is ready! (took {}s)", i + 1));
                                            docker_ready = true;
                                            break;
                                        }
                                    }
                                    
                                    if i == 119 {
                                        println!();
                                        log_startup("    Docker did not start within 2 minutes");
                                        println!("    Please start Docker Desktop manually to enable web search");
                                        return;
                                    }
                                }
                                
                                if !docker_ready {
                                    return; // Docker didn't start, exit the function
                                }
                                // Continue to SearXNG container startup
                            }
                            Err(e) => {
                                log_startup(&format!("    Failed to launch Docker Desktop: {}", e));
                            }
                        }
                        break;
                    }
                }
                
                if !docker_launched {
                    println!("    Docker Desktop not found - please start it manually to enable web search");
                    log_startup("    Docker Desktop not found in standard locations");
                }
            }
            
            #[cfg(not(target_os = "windows"))]
            {
                println!("    Please start Docker Desktop to enable web search");
            }
        }
    }
    
    println!("    Docker is running");
    log_startup("    Docker is running");
    
    // Check if SearXNG container is already running (hidden)
    #[cfg(target_os = "windows")]
    let container_check = Command::new("docker")
        .args(&["ps", "--filter", "name=mimic-searxng", "--format", "{{.Names}}"])
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    
    #[cfg(not(target_os = "windows"))]
    let container_check = Command::new("docker")
        .args(&["ps", "--filter", "name=mimic-searxng", "--format", "{{.Names}}"])
        .output();
    
    if let Ok(output) = container_check {
        let container_name = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !container_name.is_empty() {
            println!("    SearXNG container is already running");
            log_startup("    SearXNG container is already running");
            return;
        }
    }
    
    // Check if container exists but is stopped - if so, start it
    #[cfg(target_os = "windows")]
    let stopped_check = Command::new("docker")
        .args(&["ps", "-a", "--filter", "name=mimic-searxng", "--format", "{{.Names}}"])
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    
    #[cfg(not(target_os = "windows"))]
    let stopped_check = Command::new("docker")
        .args(&["ps", "-a", "--filter", "name=mimic-searxng", "--format", "{{.Names}}"])
        .output();
    
    if let Ok(output) = stopped_check {
        let container_name = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !container_name.is_empty() {
            // Container exists but is stopped - start it
            println!("    SearXNG container exists but stopped, starting it...");
            log_startup("    SearXNG container exists but stopped, starting it...");
            
            #[cfg(target_os = "windows")]
            let start_result = Command::new("docker")
                .args(&["start", "mimic-searxng"])
                .creation_flags(CREATE_NO_WINDOW)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output();
            
            #[cfg(not(target_os = "windows"))]
            let start_result = Command::new("docker")
                .args(&["start", "mimic-searxng"])
                .output();
            
            match start_result {
                Ok(out) if out.status.success() => {
                    println!("    SearXNG container started successfully");
                    log_startup("    SearXNG container started successfully");
                    return;
                }
                _ => {
                    println!("    Failed to start existing container, will try to recreate");
                    log_startup("    Failed to start existing container, will try to recreate");
                    // Remove the broken container
                    #[cfg(target_os = "windows")]
                    let _ = Command::new("docker")
                        .args(&["rm", "-f", "mimic-searxng"])
                        .creation_flags(CREATE_NO_WINDOW)
                        .output();
                    #[cfg(not(target_os = "windows"))]
                    let _ = Command::new("docker")
                        .args(&["rm", "-f", "mimic-searxng"])
                        .output();
                }
            }
        }
    }
    
    // Start SearXNG container
    println!("    Starting SearXNG container on port 8080...");
    log_startup("    Starting SearXNG container on port 8080...");
    
    #[cfg(target_os = "windows")]
    {
        // NOTE: Backend expects SearXNG on port 8080 (default), so we map 8080:8080
        // Container internal port 8080 -> Host port 8080
        
        // Pull image first (wait for it to complete)
        println!("    Pulling SearXNG image (if not present)...");
        log_startup("    Pulling SearXNG image (if not present)...");
        let _ = Command::new("docker")
            .args(&["pull", "searxng/searxng"])
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();  // Wait for pull to complete
        
        // Then start the container (wait for it to complete)
        let result = Command::new("docker")
            .args(&["run", "-d", "--name", "mimic-searxng", "-p", "8080:8080", "--restart", "unless-stopped",
                   "-e", "SEARXNG_BASE_URL=http://localhost:8080",
                   "-e", "SEARXNG_SECRET=your-secret-key-change-in-production",
                   "searxng/searxng"])
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();  // Wait for container to start
        
        match result {
            Ok(out) => {
                if out.status.success() {
                    println!("    SearXNG container started successfully");
                    log_startup("    SearXNG container started successfully");
                    
                    // Wait for container to be ready
                    println!("    Waiting for SearXNG to be ready...");
                    log_startup("    Waiting for SearXNG to be ready...");
                    for i in 0..10 {
                        std::thread::sleep(std::time::Duration::from_secs(1));
                        if let Ok(resp) = reqwest::blocking::get("http://localhost:8080/") {
                            if resp.status().is_success() {
                                println!("    SearXNG is ready!");
                                log_startup("    SearXNG is ready!");
                                break;
                            }
                        }
                        if i == 9 {
                            println!("    Warning: SearXNG may not be fully ready yet");
                            log_startup("    Warning: SearXNG may not be fully ready yet");
                        }
                    }
                } else {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    println!("    Failed to start SearXNG: {}", stderr);
                    log_startup(&format!("    Failed to start SearXNG: {}", stderr));
                }
            }
            Err(e) => {
                println!("    Could not start SearXNG: {}", e);
                log_startup(&format!("    Could not start SearXNG: {}", e));
            }
        }
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        let result = Command::new("docker")
            .args(&["run", "-d", "--name", "mimic-searxng", "-p", "8080:8080", "--restart", "unless-stopped",
                   "-e", "SEARXNG_BASE_URL=http://localhost:8080",
                   "-e", "SEARXNG_SECRET=your-secret-key-change-in-production",
                   "searxng/searxng"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();
        
        match result {
            Ok(out) if out.status.success() => {
                println!("    SearXNG container started on port 8080 (mimic-searxng)");
                log_startup("    SearXNG container started on port 8080 (mimic-searxng)");
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                println!("    Could not start SearXNG: {}", stderr);
                log_startup(&format!("    Could not start SearXNG: {}", stderr));
            }
            Err(e) => {
                println!("    Could not start SearXNG: {}", e);
                log_startup(&format!("    Could not start SearXNG: {}", e));
            }
        }
    }
}

/// Start Ollama server and BLOCK until it's ready
fn start_ollama_server_blocking() {
    log_startup("[*] Starting Ollama (blocking)...");

    // Check if Ollama API is already responding (from Ollama Desktop or previous instance)
    // Use 127.0.0.1 explicitly to avoid IPv6/IPv4 confusion
    log_startup("    Checking Ollama at http://127.0.0.1:11434/api/tags...");
    
    // First, check which process is using port 11434
    let netstat_check = Command::new("cmd")
        .args(&["/c", "netstat -ano | findstr :11434"])
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(Stdio::piped())
        .output();
    if let Ok(output) = netstat_check {
        let output_str = String::from_utf8_lossy(&output.stdout);
        log_startup(&format!("    Port 11434 usage:\n{}", output_str));
    }
    
    // Also fetch and log the model list for diagnostics
    let (already_running, models_info) = match reqwest::blocking::get("http://127.0.0.1:11434/api/tags") {
        Ok(response) if response.status().is_success() => {
            let models_text = response.text().unwrap_or_default();
            // Parse to extract model names
            let model_names: Vec<String> = if let Ok(json) = serde_json::from_str::<serde_json::Value>(&models_text) {
                json.get("models")
                    .and_then(|m| m.as_array())
                    .map(|arr| arr.iter()
                        .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(|s| s.to_string()))
                        .collect())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            
            println!("    Ollama API is already responding (Ollama Desktop may be running)");
            log_startup(&format!("    Ollama API is already responding"));
            log_startup(&format!("    Models from existing Ollama: {:?}", model_names));
            
            // Log warning if qwen3:0.6b is not found
            if !model_names.iter().any(|m| m == "qwen3:0.6b") {
                log_startup("    WARNING: qwen3:0.6b NOT found in existing Ollama instance!");
                log_startup("    This may be a different Ollama instance than your CLI.");
            }
            
            (true, models_text)
        }
        Ok(response) => {
            log_startup(&format!("    Ollama responded with status: {}", response.status()));
            (false, String::new())
        }
        Err(e) => {
            log_startup(&format!("    Ollama not responding yet: {}", e));
            (false, String::new())
        }
    };
    
    // Even if API is responding, we still try to launch ollama serve
    // If Ollama Desktop is running, the new serve will fail (port in use) which is fine
    // If it's not actually running as a server, we need to start it
    if already_running {
        log_startup("    Attempting to start ollama serve anyway (will fail if port in use)...");
        log_startup(&format!("    Current models available: {} bytes of JSON", models_info.len()));
    }

    // Find Ollama executable - check PATH first, then known locations
    let mut ollama_path: Option<PathBuf> = None;
    
    // Try using 'which' to find in PATH
    log_startup("    Searching for Ollama in PATH...");
    if let Ok(which_result) = which::which("ollama") {
        log_startup(&format!("    Found Ollama in PATH: {:?}", which_result));
        ollama_path = Some(which_result);
    }
    
    // If not in PATH, check known locations
    if ollama_path.is_none() {
        let local_app_data = std::env::var("LOCALAPPDATA").unwrap_or_default();
        let program_files = std::env::var("ProgramFiles").unwrap_or_else(|_| "C:\\Program Files".to_string());
        let program_files_x86 = std::env::var("ProgramFiles(x86)").unwrap_or_else(|_| "C:\\Program Files (x86)".to_string());
        
        let ollama_paths = vec![
            PathBuf::from(&program_files).join("Ollama").join("ollama.exe"),
            PathBuf::from(&program_files_x86).join("Ollama").join("ollama.exe"),
            PathBuf::from(&local_app_data).join("Programs").join("Ollama").join("ollama.exe"),
            PathBuf::from("C:/Program Files/Ollama/ollama.exe"),
            PathBuf::from("C:/Program Files (x86)/Ollama/ollama.exe"),
        ];
        
        log_startup("    Checking Ollama locations:");
        for path in &ollama_paths {
            let exists = path.exists();
            log_startup(&format!("      - {:?} (exists: {})", path, exists));
            if exists && ollama_path.is_none() {
                ollama_path = Some(path.clone());
            }
        }
    }
    
    let path = match ollama_path {
        Some(p) => p,
        None => {
            println!("    Ollama not found - please install from https://ollama.com");
            log_startup("    ERROR: Ollama executable not found in PATH or standard locations");
            return;
        }
    };
    
    println!("    Starting Ollama server from: {:?}", path);
    log_startup(&format!("    Starting Ollama from: {:?}", path));
    
    #[cfg(target_os = "windows")]
    {
        // First, try to check if we can execute Ollama (get version)
        log_startup("    Testing Ollama executable...");
        let version_check = Command::new(&path)
            .arg("--version")
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();
        
        match version_check {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout);
                log_startup(&format!("    Ollama version: {}", version.trim()));
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                log_startup(&format!("    Ollama version check failed: {}", stderr));
            }
            Err(e) => {
                log_startup(&format!("    Cannot execute Ollama: {}", e));
                return;
            }
        }
        
        // Now try to start Ollama serve directly with CREATE_NEW_CONSOLE
        log_startup("    Starting Ollama serve directly...");
        
        // Detect and set Ollama models directory
        let models_dir = detect_ollama_models_dir();
        
        // Use CREATE_NEW_CONSOLE to create a new console window that we can hide
        // This avoids the \\?\ prefix issues with cmd /c
        // Set required environment variables for Tauri WebView compatibility
        let mut cmd = Command::new(&path);
        cmd.arg("serve")
            .env("OLLAMA_ORIGINS", "*")
            .env("OLLAMA_HOST", "0.0.0.0:11434")
            .creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        
        // Set OLLAMA_MODELS if we detected a directory
        if let Some(ref dir) = models_dir {
            log_startup(&format!("    Setting OLLAMA_MODELS={:?}", dir));
            cmd.env("OLLAMA_MODELS", dir);
        }
        
        // Set HOME/USERPROFILE to ensure Ollama uses correct user directory
        if let Some(home) = dirs::home_dir() {
            cmd.env("USERPROFILE", &home);
            log_startup(&format!("    Setting USERPROFILE={:?}", home));
        }
        
        let start_result = cmd.spawn();
        
        match start_result {
            Ok(mut child) => {
                // Wait a moment for cmd to execute
                std::thread::sleep(std::time::Duration::from_millis(1000));
                
                // Check if cmd executed successfully
                match child.try_wait() {
                    Ok(Some(status)) if status.success() => {
                        log_startup("    cmd started Ollama successfully");
                    }
                    Ok(Some(status)) => {
                        let mut stderr = String::new();
                        if let Some(mut err) = child.stderr.take() {
                            use std::io::Read;
                            let _ = err.read_to_string(&mut stderr);
                        }
                        log_startup(&format!("    cmd failed with status: {:?}, stderr: {}", status.code(), stderr));
                    }
                    _ => {
                        log_startup("    cmd command executing...");
                    }
                }
                
                // Don't wait for the cmd process - let it complete
                let _ = child;
                
                // Wait for Ollama API to be ready (up to 60 seconds)
                println!("    Waiting for Ollama API to be ready...");
                for i in 0..60 {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    
                    if i % 5 == 0 {
                        print!(".");
                    }
                    
                    // Check if Ollama process is running via tasklist
                    if i == 3 || i == 10 {
                        let tasklist_check = Command::new("tasklist")
                            .args(&["/FI", "IMAGENAME eq ollama.exe", "/NH"])
                            .creation_flags(CREATE_NO_WINDOW)
                            .stdout(Stdio::piped())
                            .output();
                        
                        if let Ok(output) = tasklist_check {
                            let output_str = String::from_utf8_lossy(&output.stdout);
                            if output_str.contains("ollama.exe") {
                                log_startup(&format!("    ollama.exe found in tasklist at {}s", i));
                            } else {
                                log_startup(&format!("    ollama.exe NOT in tasklist at {}s", i));
                            }
                        }
                    }
                    
                    match reqwest::blocking::get("http://localhost:11434/api/tags") {
                        Ok(response) if response.status().is_success() => {
                            println!("\n    Ollama is ready! (took {}s)", i + 1);
                            log_startup(&format!("    Ollama is ready! (took {}s)", i + 1));
                            return;
                        }
                        Ok(response) => {
                            log_startup(&format!("    Ollama responded with status {} at {}s", response.status(), i + 1));
                        }
                        Err(_) => {
                            // Still not ready, continue waiting
                        }
                    }
                }
                println!();
                log_startup("    WARNING: Ollama not responding after 60s");
                
                // Check if process exists
                let tasklist_check = Command::new("tasklist")
                    .args(&["/FI", "IMAGENAME eq ollama.exe", "/NH"])
                    .creation_flags(CREATE_NO_WINDOW)
                    .stdout(Stdio::piped())
                    .output();
                
                if let Ok(output) = tasklist_check {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    if output_str.contains("ollama.exe") {
                        log_startup("    ollama.exe is still running but API not available");
                    } else {
                        log_startup("    ollama.exe not found in tasklist");
                    }
                }
            }
            Err(e) => {
                println!("    Failed to start Ollama: {}", e);
                log_startup(&format!("    ERROR: Failed to spawn Ollama: {}", e));
            }
        }
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        match Command::new(&path)
            .arg("serve")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn() 
        {
            Ok(_) => {
                log_startup("    Ollama spawned, waiting for API...");
                for i in 0..30 {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    if let Ok(response) = reqwest::blocking::get("http://localhost:11434/api/tags") {
                        if response.status().is_success() {
                            println!("    Ollama is ready!");
                            log_startup("    Ollama is ready!");
                            return;
                        }
                    }
                }
                log_startup("    WARNING: Ollama not responding after 30s");
            }
            Err(e) => {
                log_startup(&format!("    ERROR: Failed to start Ollama: {}", e));
            }
        }
    }
}

/// Start Python TTS server and BLOCK until it's ready
fn start_python_backend_blocking(app_data_dir: &PathBuf) {
    const PORT: u16 = 8000;
    
    log_startup("[*] Starting Python TTS backend (blocking)...");
    
    // Try to find Python executable using multiple methods
    let python_exe_path = find_python_executable();
    
    let python_exe = match &python_exe_path {
        Some(path) => {
            let path_str = path.to_string_lossy().to_string();
            log_startup(&format!("    Found Python at: {}", path_str));
            path_str
        }
        None => {
            log_startup("    Python not found in registry or common paths, trying PATH...");
            // Fallback to simple check
            let fallback = if cfg!(windows) { "python.exe" } else { "python3" };
            let python_check = Command::new(fallback)
                .arg("--version")
                .creation_flags(CREATE_NO_WINDOW)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output();
            
            match python_check {
                Ok(output) if output.status.success() => fallback.to_string(),
                _ => {
                    println!("    Python not available - TTS will not work");
                    log_startup("    Python not available - TTS will not work");
                    log_startup("    Please install Python 3.10-3.12 from https://python.org");
                    return;
                }
            }
        }
    };
    
    // Find the backend script - check multiple possible locations
    let exe_binding = std::env::current_exe().unwrap();
    let exe_dir = exe_binding.parent().unwrap();
    let current_dir = std::env::current_dir().unwrap();
    
    // Try multiple paths in order of likelihood
    let possible_paths = vec![
        // Packaged: resources folder next to exe
        exe_dir.join("resources").join("backend").join("tts_server_unified.py"),
        // Dev: running from project root
        current_dir.join("app").join("backend").join("tts_server_unified.py"),
        // Dev: running from src-tauri directory
        current_dir.join("..").join("app").join("backend").join("tts_server_unified.py"),
        // Dev: running from src-tauri/target/release
        exe_dir.join("..").join("..").join("..").join("app").join("backend").join("tts_server_unified.py"),
        // Dev: resources in src-tauri
        current_dir.join("resources").join("backend").join("tts_server_unified.py"),
        // Dev: resources next to exe (for src-tauri/target/release)
        exe_dir.join("..").join("..").join("resources").join("backend").join("tts_server_unified.py"),
    ];
    
    let backend_script = match possible_paths.iter().find(|p| p.exists()) {
        Some(p) => {
            log_startup(&format!("    Found tts_server_unified.py at: {:?}", p));
            p.clone()
        }
        None => {
            eprintln!("    Could not find tts_server_unified.py in any of:");
            log_startup("    Could not find tts_server_unified.py in any of:");
            for p in &possible_paths {
                eprintln!("      - {:?}", p);
                log_startup(&format!("      - {:?}", p));
            }
            return;
        }
    };
    
    let backend_dir = backend_script.parent().unwrap().to_path_buf();
    
    println!("    Starting: {:?}", backend_script);
    log_startup(&format!("    Starting: {:?}", backend_script));
    
    // Set environment variables
    let voices_dir = app_data_dir.join("voices");
    let backend_data_dir = app_data_dir.join("backend_data");
    
    // Add espeak-ng to PATH if installed (required for KittenTTS)
    let espeak_paths = [
        r"C:\Program Files\eSpeak NG",
        r"C:\Program Files (x86)\eSpeak NG",
    ];
    let mut path_env = std::env::var("PATH").unwrap_or_default();
    let mut espeak_found = false;
    for path in &espeak_paths {
        let espeak_exe = std::path::Path::new(path).join("espeak-ng.exe");
        if espeak_exe.exists() {
            if !path_env.contains(path) {
                path_env = format!("{};{}", path, path_env);
            }
            espeak_found = true;
            log_startup(&format!("    Added espeak-ng to PATH: {}", path));
            break;
        }
    }
    
    #[cfg(target_os = "windows")]
    {
        // Create log files for backend output
        let log_dir = app_data_dir.join("logs");
        let _ = std::fs::create_dir_all(&log_dir);
        let stdout_file = std::fs::File::create(log_dir.join("backend_stdout.log")).ok();
        let stderr_file = std::fs::File::create(log_dir.join("backend_stderr.log")).ok();
        
        let mut cmd_builder = Command::new(python_exe);
        cmd_builder.arg(&backend_script)
            .current_dir(&backend_dir)
            .env("MIMIC_PORT", PORT.to_string())
            .env("MIMIC_VOICES_DIR", &voices_dir)
            .env("MIMIC_DATA_DIR", &backend_data_dir)
            .env("SEARXNG_URL", "http://localhost:8080")
            .env("PATH", &path_env);
        
        // Add PHONEMIZER_ESPEAK_PATH/LIBRARY if espeak found
        if espeak_found {
            for path in &espeak_paths {
                let espeak_exe = std::path::Path::new(path).join("espeak-ng.exe");
                let espeak_dll = std::path::Path::new(path).join("libespeak-ng.dll");
                if espeak_exe.exists() {
                    cmd_builder.env("PHONEMIZER_ESPEAK_PATH", &espeak_exe);
                    if espeak_dll.exists() {
                        cmd_builder.env("PHONEMIZER_ESPEAK_LIBRARY", &espeak_dll);
                    }
                    let data_path = std::path::Path::new(path).join("espeak-ng-data");
                    if data_path.exists() {
                        cmd_builder.env("ESPEAK_DATA_PATH", &data_path);
                    }
                    break;
                }
            }
        }
        
        let start_result = cmd_builder
            .stdout(stdout_file.map(|f| Stdio::from(f)).unwrap_or(Stdio::null()))
            .stderr(stderr_file.map(|f| Stdio::from(f)).unwrap_or(Stdio::null()))
            .creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP)
            .spawn();
        
        match start_result {
            Ok(_child) => {
                println!("    TTS server started (PID: {}), waiting for it to be ready...", _child.id());
                log_startup(&format!("    TTS server started (PID: {})", _child.id()));
                
                // BLOCK and wait for TTS server to be ready (up to 60 seconds)
                for i in 0..60 {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    if let Ok(response) = reqwest::blocking::get(format!("http://localhost:{}/health", PORT)) {
                        if response.status().is_success() {
                            println!("    TTS server is ready! (took {}s)", i + 1);
                            log_startup("    TTS server is ready!");
                            return;
                        }
                    }
                    print!(".");
                }
                println!();
                log_startup("    Warning: TTS started but not responding after 60s");
            }
            Err(e) => {
                eprintln!("    Failed to start TTS server: {}", e);
                log_startup(&format!("    Failed to start TTS server: {}", e));
            }
        }
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        if let Err(e) = Command::new(python_exe)
            .arg(&backend_script)
            .current_dir(&backend_dir)
            .env("MIMIC_PORT", PORT.to_string())
            .env("MIMIC_VOICES_DIR", &voices_dir)
            .env("MIMIC_DATA_DIR", &backend_data_dir)
            .env("SEARXNG_URL", "http://localhost:8080")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            eprintln!("    Failed to start TTS server: {}", e);
            log_startup(&format!("    Failed to start TTS server: {}", e));
            return;
        }
        
        // Wait for TTS to be ready
        for i in 0..60 {
            std::thread::sleep(std::time::Duration::from_secs(1));
            if let Ok(response) = reqwest::blocking::get(format!("http://localhost:{}/health", PORT)) {
                if response.status().is_success() {
                    println!("    TTS server is ready!");
                    log_startup("    TTS server is ready!");
                    return;
                }
            }
        }
    }
}

async fn wait_for_backend(port: u16) {
    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/health", port);
    
    for _ in 0..60 { // Try for 60 seconds
        if let Ok(response) = client.get(&url).timeout(std::time::Duration::from_secs(1)).send().await {
            if response.status().is_success() {
                return;
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
    
    eprintln!("Warning: Backend did not become ready within 60 seconds");
}

// Tauri Commands for File System Operations

#[tauri::command]
fn save_voice_to_file(
    app_handle: tauri::AppHandle,
    data: VoiceData,
) -> Result<String, String> {
    let app_data_dir = app_handle.path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data directory")?;
    
    let voices_dir = app_data_dir.join("voices");
    std::fs::create_dir_all(&voices_dir).map_err(|e| e.to_string())?;

    // Save voice data as JSON file
    let filename = format!("voice_{}.json", data.persona_id);
    let filepath = voices_dir.join(&filename);
    
    let json = serde_json::to_string_pretty(&data).map_err(|e| e.to_string())?;
    std::fs::write(&filepath, json).map_err(|e| e.to_string())?;
    
    // Also save the audio separately as a WAV file for easy access
    if let Ok(audio_bytes) = general_purpose::STANDARD.decode(&data.audio_data) {
        let audio_path = voices_dir.join(format!("voice_{}.wav", data.persona_id));
        std::fs::write(&audio_path, audio_bytes).map_err(|e| e.to_string())?;
    }
    
    Ok(filepath.to_string_lossy().to_string())
}

#[tauri::command]
fn load_voice_from_file(
    app_handle: tauri::AppHandle,
    persona_id: String,
) -> Result<Option<VoiceData>, String> {
    let app_data_dir = app_handle.path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data directory")?;
    
    let voices_dir = app_data_dir.join("voices");
    let filepath = voices_dir.join(format!("voice_{}.json", persona_id));
    
    if !filepath.exists() {
        return Ok(None);
    }
    
    let content = std::fs::read_to_string(&filepath).map_err(|e| e.to_string())?;
    let data: VoiceData = serde_json::from_str(&content).map_err(|e| e.to_string())?;
    
    Ok(Some(data))
}

#[tauri::command]
fn delete_voice_file(
    app_handle: tauri::AppHandle,
    persona_id: String,
) -> Result<(), String> {
    let app_data_dir = app_handle.path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data directory")?;
    
    let voices_dir = app_data_dir.join("voices");
    
    // Delete JSON file
    let json_path = voices_dir.join(format!("voice_{}.json", persona_id));
    if json_path.exists() {
        std::fs::remove_file(&json_path).map_err(|e| e.to_string())?;
    }
    
    // Delete WAV file
    let wav_path = voices_dir.join(format!("voice_{}.wav", persona_id));
    if wav_path.exists() {
        std::fs::remove_file(&wav_path).map_err(|e| e.to_string())?;
    }
    
    Ok(())
}

#[tauri::command]
fn list_saved_voices(
    app_handle: tauri::AppHandle,
) -> Result<Vec<String>, String> {
    let app_data_dir = app_handle.path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data directory")?;
    
    let voices_dir = app_data_dir.join("voices");
    
    if !voices_dir.exists() {
        return Ok(vec![]);
    }
    
    let mut voices = vec![];
    for entry in std::fs::read_dir(&voices_dir).map_err(|e| e.to_string())? {
        if let Ok(entry) = entry {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("voice_") && name.ends_with(".json") {
                    // Extract persona_id from filename
                    let id = name.trim_start_matches("voice_").trim_end_matches(".json");
                    voices.push(id.to_string());
                }
            }
        }
    }
    
    Ok(voices)
}

#[tauri::command]
fn get_backend_port(state: State<BackendState>) -> u16 {
    *state.port.lock().unwrap()
}

#[tauri::command]
fn get_app_data_path(app_handle: tauri::AppHandle) -> Result<String, String> {
    app_handle.path_resolver()
        .app_data_dir()
        .map(|p| p.to_string_lossy().to_string())
        .ok_or("Failed to get app data directory".to_string())
}

#[tauri::command]
async fn show_in_folder(path: String) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .args(["/select,", &path])
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .args(["-R", &path])
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(&path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Tauri command: Scan all Ollama model directories and return unified list
#[tauri::command]
async fn scan_all_models() -> Result<serde_json::Value, String> {
    let models = scan_all_ollama_models();
    
    // Convert to Ollama API-compatible format
    let models_json: Vec<serde_json::Value> = models.into_iter().map(|m| {
        serde_json::json!({
            "name": m.name,
            "model": m.model,
            "size": m.size,
            "modified_at": m.modified_at,
            "digest": m.digest,
            "source_dir": m.source_dir,
            "details": m.details,
        })
    }).collect();
    
    Ok(serde_json::json!({ "models": models_json }))
}

/// Tauri command: Get all Ollama model directories found on system
#[tauri::command]
fn get_model_directories() -> Vec<String> {
    detect_all_ollama_model_dirs()
        .into_iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect()
}
