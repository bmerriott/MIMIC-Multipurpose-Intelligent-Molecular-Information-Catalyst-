# Mimic AI Backend Launcher (PowerShell)
# This script starts all required backend services silently

param(
    [string]$BackendDir = ""
)

$ErrorActionPreference = "Continue"

# Log to file
$LogFile = "$env:APPDATA\com.mimicai.app\backend-startup.log"
function Log {
    param([string]$msg)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp $msg" | Out-File -FilePath $LogFile -Append
}

Log "[*] Starting Mimic AI Backends..."

# Find backend directory - try more locations
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ExeDir = Split-Path -Parent $ScriptDir

Log "    Script dir: $ScriptDir"
Log "    Exe dir: $ExeDir"

$PossibleDirs = @(
    "$ExeDir\resources\backend",
    "$ExeDir\app\backend",
    "$ScriptDir\resources\backend",
    "$ScriptDir\app\backend",
    "$ExeDir\..\app\backend"
)

foreach ($dir in $PossibleDirs) {
    $testScript = Join-Path $dir "tts_server_unified.py"
    if (Test-Path $testScript) {
        $BackendDir = $dir
        Log "    Found backend at: $dir"
        break
    }
}

if ($BackendDir -eq "") {
    Log "    ERROR: Could not find backend directory"
}

# ============================================
# 1. Start Ollama Server
# ============================================
Log "[*] Starting Ollama..."

$OllamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($OllamaProcess) {
    Log "    Ollama is already running"
} else {
    # Kill any stale processes
    Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
    
    $OllamaPaths = @(
        "C:\Program Files\Ollama\ollama.exe",
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    )
    
    foreach ($path in $OllamaPaths) {
        if (Test-Path $path) {
            try {
                # Start Ollama with no window - use Start-Process for reliability
                Start-Process -FilePath $path -ArgumentList "serve" -WindowStyle Hidden -ErrorAction Stop
                Log "    Ollama started: $path"
            } catch {
                Log "    ERROR starting Ollama: $_"
            }
            break
        }
    }
}

Start-Sleep -Seconds 2

# Wait for Ollama to be ready (it can take a few seconds to start)
Log "    Checking if Ollama is ready..."
$OllamaReady = $false
for ($i = 0; $i -lt 5; $i++) {
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
        $OllamaReady = $true
        Log "    Ollama is ready!"
        break
    } catch {}
    Start-Sleep -Seconds 1
}
if (-not $OllamaReady) {
    Log "    WARNING: Ollama may not be fully ready yet (this is OK - it will work shortly)"
}

# ============================================
# 2. Start SearXNG Docker Container
# ============================================
Log "[*] Starting SearXNG..."

# Check if Docker is running - try both process and CLI
$dockerRunning = $false
try {
    $null = docker version 2>$null
    if ($LASTEXITCODE -eq 0) { $dockerRunning = $true }
} catch {}

if (-not $dockerRunning) {
    Log "    Docker is not running"
} else {
    # Check if container is already running
    $container = docker ps --filter "name=mimic-searxng" --format "{{.Names}}" 2>$null
    if ($container -eq "mimic-searxng") {
        Log "    SearXNG container is already running"
    } else {
        # Check if container exists but is stopped
        $stoppedContainer = docker ps -a --filter "name=mimic-searxng" --format "{{.Names}}" 2>$null
        if ($stoppedContainer -eq "mimic-searxng") {
            Log "    SearXNG container exists but stopped, starting it..."
            docker start mimic-searxng 2>$null
            if ($LASTEXITCODE -eq 0) {
                Log "    SearXNG container started successfully"
            } else {
                Log "    Failed to start existing container, recreating..."
                docker rm -f mimic-searxng 2>$null | Out-Null
                # Pull image first (async)
                Log "    Pulling SearXNG image..."
                Start-Process -FilePath "docker" -ArgumentList "pull searxng/searxng" -WindowStyle Hidden -ErrorAction SilentlyContinue
                # Start container
                $err = docker run -d --name mimic-searxng -p 8080:8080 --restart unless-stopped `
                    -e SEARXNG_BASE_URL=http://localhost:8080 `
                    -e SEARXNG_SECRET=your-secret-key-change-in-production `
                    searxng/searxng 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Log "    SearXNG container started"
                } else {
                    Log "    ERROR starting SearXNG: $err"
                }
            }
        } else {
            # No container exists, create new one
            Log "    Creating new SearXNG container..."
            # Pull image first (async)
            Start-Process -FilePath "docker" -ArgumentList "pull searxng/searxng" -WindowStyle Hidden -ErrorAction SilentlyContinue
            # Start container
            $err = docker run -d --name mimic-searxng -p 8080:8080 --restart unless-stopped `
                -e SEARXNG_BASE_URL=http://localhost:8080 `
                -e SEARXNG_SECRET=your-secret-key-change-in-production `
                searxng/searxng 2>&1
            if ($LASTEXITCODE -eq 0) {
                Log "    SearXNG container started"
            } else {
                Log "    ERROR starting SearXNG: $err"
            }
        }
    }
}

# ============================================
# 3. Start Python TTS Server
# ============================================
Log "[*] Starting TTS Server..."

if ($BackendDir -eq "") {
    Log "    ERROR: Could not find backend directory"
} else {
    $BackendScript = Join-Path $BackendDir "tts_server_unified.py"
    
    if (Test-Path $BackendScript) {
        Log "    Starting: $BackendScript"
        
        # Set environment variables
        $env:MIMIC_PORT = "8000"
        $env:SEARXNG_URL = "http://localhost:8080"
        
        try {
            # Start Python TTS server using Start-Process for reliability
            Start-Process -FilePath "python.exe" -ArgumentList "`"$BackendScript`"" -WorkingDirectory $BackendDir -WindowStyle Hidden -ErrorAction Stop
            Log "    TTS server started"
        } catch {
            Log "    ERROR starting TTS: $_"
        }
    } else {
        Log "    ERROR: Backend script not found: $BackendScript"
    }
}

Log "[*] All backends started!"
