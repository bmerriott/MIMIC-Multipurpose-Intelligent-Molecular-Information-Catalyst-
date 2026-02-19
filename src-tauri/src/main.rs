// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{Manager, State};
use serde::{Serialize, Deserialize};
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
use setup::{check_setup_status, install_python_deps, find_python_executable};
use dependency_manager::{check_dependencies, install_dependencies_command, get_python_path};

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
    
    // 3. Start Python TTS server in its own thread - isolated from others
    let app_data_dir_create = app_data_dir.clone();
    let python_thread = std::thread::spawn(move || {
        log_startup("    [Python Thread] Starting Python TTS server...");
        start_python_backend_blocking(&app_data_dir_create);
        log_startup("    [Python Thread] Python TTS startup complete");
    });
    
    // Note: We don't join these threads - they run independently
    // This ensures Ollama starts even if Docker takes a long time
    
    tauri::Builder::default()
        .manage(BackendState {
            process: Mutex::new(None),
            port: Mutex::new(8000),
        })
        .setup(|app| {
            let app_handle = app.handle();
            
            // Try to check for Python, but don't crash if it fails
            let setup_status = setup::run_setup_check();
            if !setup_status.python_installed {
                println!("Warning: Python not found - TTS features will not work");
                println!("Please install Python 3.10-3.12 from https://python.org");
            } else if !setup_status.dependencies_installed {
                println!("Warning: Python dependencies not fully installed");
                println!("Some features may not work properly");
            }
            
            // Get app data directory (with error handling)
            let app_data_dir = match app_handle.path_resolver().app_data_dir() {
                Some(dir) => dir,
                None => {
                    eprintln!("Warning: Could not get app data directory");
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
            
            // Ensure main window is visible and focused
            if let Some(main_window) = app_handle.get_window("main") {
                let _ = main_window.show();
                let _ = main_window.set_focus();
                let _ = main_window.center();
                log_startup("[*] Main window shown and focused");
            } else {
                log_startup("[!] Warning: Main window not found");
            }

            Ok(())
        })
        .on_window_event(|event| match event.event() {
            tauri::WindowEvent::CloseRequested { api, .. } => {
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
            load_voice_from_file,
            delete_voice_file,
            list_saved_voices,
            get_backend_port,
            get_app_data_path,
            show_in_folder,
            check_setup_status,
            install_python_deps,
            check_dependencies,
            install_dependencies_command,
            get_python_path,
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
    // 1. Development: current directory (for running from source)
    // 2. Packaged: resources folder bundled with the app
    let dev_path = std::env::current_dir().unwrap().join("app").join("backend").join("tts_server_unified.py");
    
    // For packaged app, get exe directory first
    let exe_binding = std::env::current_exe().unwrap();
    let exe_dir = exe_binding.parent().unwrap();
    
    // Tauri bundles resources at resources/ (not resources/app/)
    let packaged_path = exe_dir.join("resources").join("backend").join("tts_server_unified.py");
    
    println!("createhecking paths:");
    println!("      - Dev: {:?}", dev_path);
    println!("      - Packaged: {:?}", packaged_path);

    // Check both dev and packaged paths
    let possible_paths = vec![dev_path, packaged_path];
    
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
                // Start Ollama serve silently - don't kill existing, just start new if needed
                // Ollama will handle the port binding itself
                // Use null stdout/stderr to avoid issues with CREATE_NO_WINDOW and pipes
                let start_result = Command::new(path)
                    .arg("serve")
                    .creation_flags(CREATE_NO_WINDOW)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn();
                
                match start_result {
                    Ok(child) => {
                        println!("    Ollama server started, PID: {}", child.id());
                        log_startup(&format!("    Ollama server started, PID: {}", child.id()));
                        
                        // Wait a moment for Ollama to start
                        std::thread::sleep(std::time::Duration::from_secs(2));
                        
                        // Verify it's running
                        match reqwest::blocking::get("http://localhost:11434/api/tags") {
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
    let already_running = match reqwest::blocking::get("http://localhost:11434/api/tags") {
        Ok(response) if response.status().is_success() => {
            println!("    Ollama API is already responding (Ollama Desktop may be running)");
            log_startup("    Ollama API is already responding");
            true
        }
        Ok(response) => {
            log_startup(&format!("    Ollama responded with status: {}", response.status()));
            false
        }
        Err(e) => {
            log_startup(&format!("    Ollama not responding yet: {}", e));
            false
        }
    };
    
    // Even if API is responding, we still try to launch ollama serve
    // If Ollama Desktop is running, the new serve will fail (port in use) which is fine
    // If it's not actually running as a server, we need to start it
    if already_running {
        log_startup("    Attempting to start ollama serve anyway (will fail if port in use)...");
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
        
        // Use CREATE_NEW_CONSOLE to create a new console window that we can hide
        // This avoids the \\?\ prefix issues with cmd /c
        let start_result = Command::new(&path)
            .arg("serve")
            .creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        
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
    
    // Find the backend script
    let dev_path = std::env::current_dir().unwrap().join("app").join("backend").join("tts_server_unified.py");
    let exe_binding = std::env::current_exe().unwrap();
    let exe_dir = exe_binding.parent().unwrap();
    let packaged_path = exe_dir.join("resources").join("backend").join("tts_server_unified.py");
    
    let backend_script = if dev_path.exists() {
        dev_path
    } else if packaged_path.exists() {
        packaged_path
    } else {
        eprintln!("    Could not find tts_server_unified.py");
        log_startup("    Could not find tts_server_unified.py");
        return;
    };
    
    let backend_dir = backend_script.parent().unwrap().to_path_buf();
    
    println!("    Starting: {:?}", backend_script);
    log_startup(&format!("    Starting: {:?}", backend_script));
    
    // Set environment variables
    let voices_dir = app_data_dir.join("voices");
    let backend_data_dir = app_data_dir.join("backend_data");
    
    #[cfg(target_os = "windows")]
    {
        let start_result = Command::new(python_exe)
            .arg(&backend_script)
            .current_dir(&backend_dir)
            .env("MIMIC_PORT", PORT.to_string())
            .env("MIMIC_VOICES_DIR", &voices_dir)
            .env("MIMIC_DATA_DIR", &backend_data_dir)
            .env("SEARXNG_URL", "http://localhost:8080")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
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
