//! First-time setup wizard for Mimic AI
//! Checks for Python and dependencies, guides user through installation

use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use tauri::{AppHandle, Manager, WindowBuilder, WindowUrl};

// Windows-specific: flag to hide console windows
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[cfg(target_os = "windows")]
use winreg::enums::*;
#[cfg(target_os = "windows")]
use winreg::RegKey;
#[cfg(target_os = "windows")]
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SetupStatus {
    pub python_installed: bool,
    pub python_version: Option<String>,
    pub dependencies_installed: bool,
    pub missing_deps: Vec<String>,
}

const REQUIRED_PACKAGES: &[&str] = &[
    "fastapi",
    "uvicorn",
    "numpy",
    "scipy",
    "soundfile",
    "python-dotenv",
    "requests",
];

// TTS engine - Qwen3 only (StyleTTS2 removed due to dependency conflicts)
const TTS_PACKAGES: &[&str] = &["qwen-tts"];

/// Check if Python is installed and get version
pub fn check_python() -> Option<String> {
    // Try 'python' first (Windows), then 'python3' (Unix)
    // Use hidden execution to avoid flashing windows
    let output = run_command_hidden("python", &["--version"])
        .or_else(|| run_command_hidden("python3", &["--version"]))?;
    
    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout)
            .trim()
            .to_string();
        Some(version)
    } else {
        None
    }
}

/// Find Python executable path from registry and common locations (Windows only)
#[cfg(target_os = "windows")]
pub fn find_python_executable() -> Option<PathBuf> {
    // First try the PATH
    if let Ok(output) = Command::new("where").arg("python.exe").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout);
            let first_line = path.lines().next()?;
            let trimmed = first_line.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
    }
    
    // Try Windows registry for Python installations
    let reg_paths = [
        ("SOFTWARE\\Python\\PythonCore", HKEY_LOCAL_MACHINE),
        ("SOFTWARE\\Python\\PythonCore", HKEY_CURRENT_USER),
        ("SOFTWARE\\WOW6432Node\\Python\\PythonCore", HKEY_LOCAL_MACHINE),
    ];
    
    for (path, hive) in &reg_paths {
        if let Ok(key) = RegKey::predef(*hive).open_subkey(path) {
            if let Ok(subkeys) = key.enum_keys().collect::<Result<Vec<_>, _>>() {
                // Sort to get the newest version first
                let mut versions: Vec<_> = subkeys.into_iter()
                    .filter(|v| v.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false))
                    .collect();
                versions.sort_by(|a, b| b.cmp(a)); // Descending order
                
                for version in versions {
                    let install_path = format!("{}\\{}\\InstallPath", path, version);
                    if let Ok(install_key) = RegKey::predef(*hive).open_subkey(&install_path) {
                        if let Ok(exe_path) = install_key.get_value::<String, _>("ExecutablePath") {
                            let exe = PathBuf::from(exe_path);
                            if exe.exists() {
                                return Some(exe);
                            }
                        }
                        // Try default value which is the install directory
                        if let Ok(dir) = install_key.get_value::<String, _>("") {
                            let python_exe = PathBuf::from(dir).join("python.exe");
                            if python_exe.exists() {
                                return Some(python_exe);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Try common installation paths
    let common_paths = [
        r"C:\Python312\python.exe",
        r"C:\Python311\python.exe",
        r"C:\Python310\python.exe",
        r"C:\Python39\python.exe",
        r"C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe",
        r"C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe",
        r"C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe",
        r"C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python39\python.exe",
    ];
    
    for path in &common_paths {
        // Expand %USERNAME% if present
        let expanded = if path.contains("%USERNAME%") {
            if let Ok(username) = std::env::var("USERNAME") {
                path.replace("%USERNAME%", &username)
            } else {
                continue;
            }
        } else {
            path.to_string()
        };
        
        let exe = PathBuf::from(expanded);
        if exe.exists() {
            return Some(exe);
        }
    }
    
    // Try Microsoft Store Python path
    let ms_store_path = PathBuf::from(r"C:\Users")
        .join(std::env::var("USERNAME").unwrap_or_default())
        .join(r"AppData\Local\Microsoft\WindowsApps\python.exe");
    if ms_store_path.exists() {
        return Some(ms_store_path);
    }
    
    None
}

/// Non-Windows fallback
#[cfg(not(target_os = "windows"))]
pub fn find_python_executable() -> Option<PathBuf> {
    // On Unix, just check if python3 or python is available
    if let Ok(output) = Command::new("which").arg("python3").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }
    
    if let Ok(output) = Command::new("which").arg("python").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
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

/// Check if Python dependencies are installed using pip list
pub fn check_dependencies() -> (bool, Vec<String>) {
    let mut missing = Vec::new();
    
    // Check base packages using pip list (more reliable than import)
    let pip_list_output = run_command_hidden("python", &["-m", "pip", "list"]);
    let installed_packages: String = match pip_list_output {
        Some(out) if out.status.success() => {
            String::from_utf8_lossy(&out.stdout).to_lowercase()
        }
        _ => String::new(),
    };
    
    // Check base packages
    for package in REQUIRED_PACKAGES {
        // Package names in pip list may have dashes or underscores
        // Check both the original name and variations
        let package_check = package.to_lowercase();
        let package_underscore = package_check.replace("-", "_");
        let package_dash = package_check.replace("_", "-");
        
        let is_installed = installed_packages.contains(&package_check) || 
                          installed_packages.contains(&package_underscore) ||
                          installed_packages.contains(&package_dash);
        
        if !is_installed {
            missing.push(package.to_string());
        }
    }
    
    (missing.is_empty(), missing)
}

/// Run the complete setup check
pub fn run_setup_check() -> SetupStatus {
    let python_version = check_python();
    let python_installed = python_version.is_some();
    
    let (dependencies_installed, missing_deps) = if python_installed {
        check_dependencies()
    } else {
        (false, REQUIRED_PACKAGES.iter().map(|s| s.to_string()).collect())
    };
    
    SetupStatus {
        python_installed,
        python_version,
        dependencies_installed,
        missing_deps,
    }
}

/// Install Python dependencies (hidden)
pub fn install_dependencies() -> Result<(), String> {
    println!("Installing Python dependencies...");
    
    // Install base packages one by one
    for package in REQUIRED_PACKAGES {
        println!("  Installing {}...", package);
        let output = run_command_hidden("python", &["-m", "pip", "install", package])
            .ok_or_else(|| format!("Failed to run pip install for {}", package))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("pip install failed for {}: {}", package, stderr));
        }
    }
    
    // Install TTS packages separately to handle dependency conflicts
    println!("  Installing TTS packages (this may take a while)...");
    
    // Install PyTorch CPU first (stable base)
    println!("    Installing PyTorch CPU...");
    let _ = run_command_hidden("python", &["-m", "pip", "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]);
    
    // Install qwen-tts (AI voice creation engine)
    println!("    Installing qwen-tts...");
    let _ = run_command_hidden("python", &["-m", "pip", "install", "qwen-tts"]);
    
    println!("Dependencies installed successfully");
    Ok(())
}

/// Install Python dependencies with visible window (for first-time setup)
#[cfg(target_os = "windows")]
pub fn install_dependencies_visible() -> Result<(), String> {
    use std::process::Stdio;
    use std::os::windows::process::CommandExt;
    
    const CREATE_NEW_CONSOLE: u32 = 0x08000000;
    
    println!("Installing Python dependencies (visible mode)...");
    
    // Create a batch file that shows the install progress
    let temp_dir = std::env::temp_dir();
    let batch_path = temp_dir.join("mimic_install_deps.bat");
    
    // Build pip install arguments with each package quoted separately
    let pip_args: Vec<String> = REQUIRED_PACKAGES.iter().map(|p| format!("\"{}\"", p)).collect();
    let packages_list = REQUIRED_PACKAGES.join(" ");
    let tts_packages_list = TTS_PACKAGES.join(" ");
    
    let batch_content = format!(
        r#"@echo off
echo =========================================
echo  Mimic AI - Installing Python Dependencies
echo =========================================
echo.
echo This may take 5-10 minutes on first launch.
echo.

:: Step 1: Install base packages
echo [1/3] Installing base packages: {}
python.exe -m pip install {}
if %errorlevel% neq 0 (
    echo ERROR: Base packages installation failed!
    pause
    exit /b 1
)

:: Step 2: Install PyTorch CPU (stable base)
echo [2/3] Installing PyTorch (CPU version for stability)...
python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

:: Step 3: Install Qwen3-TTS (AI voice creation engine)
echo [3/3] Installing Qwen3-TTS (AI voice engine)...
python.exe -m pip install qwen-tts
if %errorlevel% neq 0 (
    echo WARNING: Qwen3-TTS installation failed. Browser TTS will be used as fallback.
) else (
    echo Qwen3-TTS installed successfully!
)

echo.
echo =========================================
echo  Installation Complete!
echo =========================================
echo.
echo Voice Engines:
echo   - Qwen3-TTS: AI voice creation (if installed)
echo   - Browser TTS: Always available (system voice)
echo.
timeout /t 3 /nobreak >nul
exit /b 0
"#,
        packages_list, pip_args.join(" ")
    );
    
    std::fs::write(&batch_path, batch_content)
        .map_err(|e| format!("Failed to create batch file: {}", e))?;
    
    // Run the batch file in a new visible window
    let output = Command::new("cmd")
        .args(&["/c", "start", "/wait", "/min", batch_path.to_str().unwrap()])
        .creation_flags(CREATE_NEW_CONSOLE)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run install: {}", e))?;
    
    // Clean up batch file
    let _ = std::fs::remove_file(&batch_path);
    
    if output.status.success() {
        println!("Dependencies installed successfully");
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("pip install failed: {}", stderr))
    }
}

/// Non-Windows fallback
#[cfg(not(target_os = "windows"))]
pub fn install_dependencies_visible() -> Result<(), String> {
    install_dependencies()
}

/// Show setup window if needed
pub fn show_setup_if_needed(app_handle: &AppHandle) -> bool {
    let status = run_setup_check();
    
    if status.python_installed && status.dependencies_installed {
        // Everything is set up, continue
        return true;
    }
    
    // Create setup window
    let setup_window = WindowBuilder::new(
        app_handle,
        "setup",
        WindowUrl::App("setup.html".into()),
    )
    .title("Mimic AI Setup")
    .inner_size(600.0, 500.0)
    .center()
    .resizable(false)
    .build();
    
    if let Ok(window) = setup_window {
        // Send setup status to window
        let _ = window.emit("setup-status", status);
        
        // Hide main window until setup is complete
        if let Some(main_window) = app_handle.get_window("main") {
            let _ = main_window.hide();
        }
        
        false
    } else {
        // Failed to create setup window, continue anyway
        true
    }
}

#[tauri::command]
pub async fn check_setup_status() -> SetupStatus {
    run_setup_check()
}

#[tauri::command]
pub async fn install_python_deps() -> Result<(), String> {
    install_dependencies()
}
