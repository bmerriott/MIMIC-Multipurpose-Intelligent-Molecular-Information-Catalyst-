//! Dependency Manager
//! Handles checking, downloading, and installing Python + required packages

use std::process::{Command, Stdio};
use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use tauri::Window;

// Windows-specific: flag to hide console windows
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DependencyStatus {
    pub python_installed: bool,
    pub python_version: Option<String>,
    pub python_path: Option<String>,
    pub pip_installed: bool,
    pub dependencies_installed: bool,
    pub missing_packages: Vec<String>,
    pub venv_exists: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InstallProgress {
    pub stage: String,
    pub message: String,
    pub percent: u8,
    pub is_complete: bool,
    pub error: Option<String>,
}

const REQUIRED_PACKAGES: &[&str] = &[
    "fastapi",
    "uvicorn",
    "numpy",
    "scipy",
    "soundfile",
    "python-dotenv",
    "requests",
    "torch",
    "torchaudio",
    "styletts2",
    "qwen-tts",
];

impl DependencyStatus {
    pub fn check() -> Self {
        println!("[Deps] Checking dependencies...");
        
        // Check Python
        let python_check = Self::check_python();
        let python_installed = python_check.is_some();
        let (python_version, python_path) = python_check
            .map(|(v, p)| (Some(v), Some(p)))
            .unwrap_or((None, None));
        
        // Check pip
        let pip_installed = if python_installed {
            Self::check_pip(&python_path.as_ref().unwrap())
        } else {
            false
        };
        
        // Check dependencies
        let (dependencies_installed, missing_packages) = if python_installed {
            Self::check_packages(&python_path.as_ref().unwrap())
        } else {
            (false, REQUIRED_PACKAGES.iter().map(|n| n.to_string()).collect())
        };
        
        // Check for virtual environment
        let venv_exists = Self::check_venv();
        
        let status = DependencyStatus {
            python_installed,
            python_version,
            python_path,
            pip_installed,
            dependencies_installed,
            missing_packages,
            venv_exists,
        };
        
        println!("[Deps] Status: {:?}", status);
        status
    }
    
    fn check_python() -> Option<(String, String)> {
        // Try 'python' first (Windows), then 'python3' (Unix)
        let commands = vec!["python", "python3"];
        
        for cmd in commands {
            let output = Self::run_command_hidden(cmd, &["--version"]);
            if let Some(output) = output {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout)
                        .trim()
                        .to_string();
                    let version = if version.is_empty() {
                        String::from_utf8_lossy(&output.stderr).trim().to_string()
                    } else {
                        version
                    };
                    
                    // Get the full path
                    let path_output = Self::run_command_hidden(cmd, &["-c", "import sys; print(sys.executable)"]);
                    if let Some(path_out) = path_output {
                        let path = String::from_utf8_lossy(&path_out.stdout).trim().to_string();
                        return Some((version, path));
                    }
                    return Some((version, cmd.to_string()));
                }
            }
        }
        None
    }
    
    fn run_command_hidden(program: &str, args: &[&str]) -> Option<std::process::Output> {
        #[cfg(target_os = "windows")]
        {
            Command::new(program)
                .args(args)
                .creation_flags(CREATE_NO_WINDOW)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .ok()
        }
        #[cfg(not(target_os = "windows"))]
        {
            Command::new(program)
                .args(args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .ok()
        }
    }
    
    fn run_command_with_hidden(pid: &str, args: &[&str]) -> Option<std::process::Output> {
        #[cfg(target_os = "windows")]
        {
            Command::new(pid)
                .args(args)
                .creation_flags(CREATE_NO_WINDOW)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .ok()
        }
        #[cfg(not(target_os = "windows"))]
        {
            Command::new(pid)
                .args(args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .ok()
        }
    }
    
    fn check_pip(python_path: &str) -> bool {
        Self::run_command_with_hidden(python_path, &["-m", "pip", "--version"])
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
    
    fn check_packages(python_path: &str) -> (bool, Vec<String>) {
        let mut missing = Vec::new();
        
        for package_name in REQUIRED_PACKAGES {
            let check_cmd = format!("import {}; print('OK')", package_name);
            let result = Self::run_command_with_hidden(python_path, &["-c", &check_cmd]);
            
            match result {
                Some(output) if output.status.success() => {}
                _ => {
                    missing.push(package_name.to_string());
                }
            }
        }
        
        (missing.is_empty(), missing)
    }
    
    fn check_venv() -> bool {
        if let Some(data_dir) = Self::get_app_data_dir() {
            let venv_path = data_dir.join("venv");
            venv_path.exists()
        } else {
            false
        }
    }
    
    fn get_app_data_dir() -> Option<PathBuf> {
        dirs::data_dir().map(|d| d.join("com.mimicai.app"))
    }
}

/// Install all dependencies with progress reporting
pub async fn install_all(window: Window) -> Result<(), String> {
    let status = DependencyStatus::check();
    
    // Emit initial progress
    emit_progress(&window, "checking", "Checking system...", 0);
    
    // Step 1: Ensure Python
    if !status.python_installed {
        emit_progress(&window, "error", "Python not found", 0);
        return Err("Python is not installed. Please install Python 3.10 or higher from https://python.org".to_string());
    }
    
    let python_path = status.python_path.unwrap();
    emit_progress(&window, "python", &format!("Found {}", status.python_version.unwrap_or_default()), 10);
    
    // Step 2: Create virtual environment
    emit_progress(&window, "venv", "Creating virtual environment...", 15);
    let venv_path = DependencyStatus::get_app_data_dir()
        .ok_or("Failed to get app data directory")?
        .join("venv");
    
    if !venv_path.exists() {
        let output = DependencyStatus::run_command_hidden(&python_path, &["-m", "venv", venv_path.to_str().unwrap()]);
        if let Some(output) = output {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("Failed to create virtual environment: {}", stderr));
            }
        } else {
            return Err("Failed to create virtual environment: command failed to run".to_string());
        }
    }
    
    // Get venv Python path
    let venv_python = if cfg!(windows) {
        venv_path.join("Scripts").join("python.exe")
    } else {
        venv_path.join("bin").join("python")
    };
    
    // Step 3: Upgrade pip
    emit_progress(&window, "pip", "Upgrading pip...", 20);
    let venv_python_str = venv_python.to_string_lossy().to_string();
    let _ = DependencyStatus::run_command_hidden(&venv_python_str, &["-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]);
    
    // Step 4: Install packages
    let total_packages = REQUIRED_PACKAGES.len();
    let base_progress = 25u8;
    let progress_per_package = (75u8 - base_progress) / total_packages as u8;
    
    for (idx, package) in REQUIRED_PACKAGES.iter().enumerate() {
        let percent = base_progress + (idx as u8 * progress_per_package);
        emit_progress(&window, "package", &format!("Installing {}...", package), percent);
        
        // Try to install package
        let output = DependencyStatus::run_command_hidden(&venv_python_str, &["-m", "pip", "install", package]);
        
        if let Some(output) = output {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("Warning: Failed to install {}: {}", package, stderr);
                // Continue anyway, some packages might be optional
            }
        }
    }
    
    emit_progress(&window, "complete", "Installation complete!", 100);
    Ok(())
}

/// Get the Python executable path (prefer venv)
pub fn get_python_executable() -> Option<String> {
    // First check for venv
    if let Some(data_dir) = DependencyStatus::get_app_data_dir() {
        let venv_python = if cfg!(windows) {
            data_dir.join("venv").join("Scripts").join("python.exe")
        } else {
            data_dir.join("venv").join("bin").join("python")
        };
        
        if venv_python.exists() {
            return Some(venv_python.to_string_lossy().to_string());
        }
    }
    
    // Fall back to system Python
    DependencyStatus::check().python_path
}

fn emit_progress(window: &Window, stage: &str, message: &str, percent: u8) {
    let progress = InstallProgress {
        stage: stage.to_string(),
        message: message.to_string(),
        percent,
        is_complete: percent >= 100,
        error: None,
    };
    
    let _ = window.emit("install-progress", progress);
}

// Tauri commands
#[tauri::command]
pub async fn check_dependencies() -> DependencyStatus {
    DependencyStatus::check()
}

#[tauri::command]
pub async fn install_dependencies_command(window: Window) -> Result<(), String> {
    install_all(window).await
}

#[tauri::command]
pub fn get_python_path() -> Option<String> {
    get_python_executable()
}
