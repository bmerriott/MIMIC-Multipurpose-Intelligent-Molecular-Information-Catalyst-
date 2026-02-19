# Mimic AI - Diagnostic Tool
# Run this to check all services and diagnose issues

param(
    [switch]$Verbose,
    [switch]$Fix
)

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Mimic AI Diagnostic Tool" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$IssuesFound = @()
$AllGood = $true

function Test-Port {
    param($Port, $Name)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $result = $client.BeginConnect("localhost", $Port, $null, $null)
        $success = $result.AsyncWaitHandle.WaitOne(1000, $false)
        $client.Close()
        return $success
    } catch {
        return $false
    }
}

# 1. Check Ollama Service
Write-Host "[1/6] Checking Ollama..." -ForegroundColor Yellow
$OllamaRunning = Test-Port -Port 11434 -Name "Ollama"
if ($OllamaRunning) {
    Write-Host "  [OK] Ollama is running on port 11434" -ForegroundColor Green
    
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
        $modelCount = $resp.models.Count
        Write-Host "  [OK] Models installed: $modelCount" -ForegroundColor Green
        
        if ($modelCount -eq 0) {
            Write-Host "  [WARN] No models found! Run: ollama pull llama3.2" -ForegroundColor Yellow
            $IssuesFound += "No Ollama models installed"
            $AllGood = $false
        } else {
            Write-Host "  Models: $($resp.models.name -join ', ')" -ForegroundColor Gray
        }
    } catch {
        Write-Host "  [ERROR] Cannot query Ollama API: $_" -ForegroundColor Red
        $IssuesFound += "Ollama API not responding"
        $AllGood = $false
    }
} else {
    Write-Host "  [ERROR] Ollama is not running on port 11434" -ForegroundColor Red
    Write-Host "  Fix: Install from https://ollama.com and run 'ollama serve'" -ForegroundColor Gray
    $IssuesFound += "Ollama not running"
    $AllGood = $false
}
Write-Host ""

# 2. Check TTS Backend
Write-Host "[2/6] Checking TTS Backend..." -ForegroundColor Yellow
$TTSRunning = Test-Port -Port 8000 -Name "TTS"
if ($TTSRunning) {
    Write-Host "  [OK] TTS Backend is running on port 8000" -ForegroundColor Green
    
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
        Write-Host "  [OK] TTS health check passed" -ForegroundColor Green
    } catch {
        Write-Host "  [WARN] TTS health check failed: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [ERROR] TTS Backend is not running on port 8000" -ForegroundColor Red
    Write-Host "  Fix: Restart Mimic AI or run Python backend manually" -ForegroundColor Gray
    $IssuesFound += "TTS backend not running"
    $AllGood = $false
}
Write-Host ""

# 3. Check Docker
Write-Host "[3/6] Checking Docker..." -ForegroundColor Yellow
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Docker is running" -ForegroundColor Green
        
        # Check SearXNG
        $containers = docker ps --filter "name=mimic-searxng" --format "{{.Names}}"
        if ($containers -contains "mimic-searxng") {
            Write-Host "  [OK] SearXNG container is running" -ForegroundColor Green
            
            # Test SearXNG
            try {
                $resp = Invoke-WebRequest -Uri "http://localhost:8080/" -TimeoutSec 5 -ErrorAction Stop
                Write-Host "  [OK] SearXNG is responding" -ForegroundColor Green
            } catch {
                Write-Host "  [WARN] SearXNG container running but not responding yet" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  [INFO] SearXNG container not running (optional for web search)" -ForegroundColor Gray
            Write-Host "  It will be created automatically when you enable web search" -ForegroundColor Gray
        }
    } else {
        Write-Host "  [INFO] Docker is not running (optional - only needed for web search)" -ForegroundColor Gray
    }
} catch {
    Write-Host "  [INFO] Docker not installed (optional - only needed for web search)" -ForegroundColor Gray
}
Write-Host ""

# 4. Check Python
Write-Host "[4/6] Checking Python..." -ForegroundColor Yellow
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) { $pythonCmd = "python" }
elseif (Get-Command python3 -ErrorAction SilentlyContinue) { $pythonCmd = "python3" }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $pythonCmd = "py" }

if ($pythonCmd) {
    $pyVersion = & $pythonCmd --version 2>&1
    Write-Host "  [OK] Python found: $pyVersion" -ForegroundColor Green
    
    # Check required modules
    $required = @("fastapi", "uvicorn", "torch", "numpy")
    $missing = @()
    foreach ($mod in $required) {
        try {
            $result = & $pythonCmd -c "import $mod; print('OK')" 2>&1
            if ($result -ne "OK") { $missing += $mod }
        } catch {
            $missing += $mod
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-Host "  [WARN] Missing Python modules: $($missing -join ', ')" -ForegroundColor Yellow
        if ($Fix) {
            Write-Host "  Installing missing modules..." -ForegroundColor Yellow
            & $pythonCmd -m pip install $missing
        } else {
            Write-Host "  Fix: pip install $($missing -join ' ')" -ForegroundColor Gray
        }
        $IssuesFound += "Missing Python modules: $($missing -join ', ')"
    } else {
        Write-Host "  [OK] All required Python modules installed" -ForegroundColor Green
    }
} else {
    Write-Host "  [ERROR] Python not found in PATH" -ForegroundColor Red
    Write-Host "  Fix: Install Python 3.10-3.12 from https://python.org" -ForegroundColor Gray
    Write-Host "       Make sure to check 'Add Python to PATH' during installation!" -ForegroundColor Gray
    $IssuesFound += "Python not found"
    $AllGood = $false
}
Write-Host ""

# 5. Check Mimic AI Installation
Write-Host "[5/6] Checking Mimic AI Installation..." -ForegroundColor Yellow
$mimicPaths = @(
    "$env:LOCALAPPDATA\Mimic AI",
    "$env:ProgramFiles\Mimic AI",
    "$env:USERPROFILE\AppData\Local\Mimic AI"
)

$found = $false
foreach ($path in $mimicPaths) {
    if (Test-Path $path) {
        Write-Host "  [OK] Found at: $path" -ForegroundColor Green
        $found = $true
        break
    }
}

if (-not $found) {
    Write-Host "  [INFO] Installation not found in standard locations" -ForegroundColor Gray
}
Write-Host ""

# 6. Check Logs
Write-Host "[6/6] Checking Logs..." -ForegroundColor Yellow
$logPath = "$env:APPDATA\com.mimicai.app\startup.log"
if (Test-Path $logPath) {
    Write-Host "  [OK] Startup log found" -ForegroundColor Green
    $lastLines = Get-Content $logPath -Tail 5
    Write-Host "  Recent entries:" -ForegroundColor Gray
    foreach ($line in $lastLines) {
        Write-Host "    $line" -ForegroundColor DarkGray
    }
} else {
    Write-Host "  [INFO] No startup log found (Mimic AI may not have run yet)" -ForegroundColor Gray
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
if ($AllGood) {
    Write-Host "  All checks passed!" -ForegroundColor Green
    Write-Host "  Mimic AI should be working correctly." -ForegroundColor Green
} else {
    Write-Host "  Issues Found:" -ForegroundColor Yellow
    foreach ($issue in $IssuesFound) {
        Write-Host "    - $issue" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "  Please address the issues above." -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Quick fixes
if (-not $AllGood -and -not $Fix) {
    Write-Host "Run with -Fix flag to attempt automatic fixes:" -ForegroundColor Cyan
    Write-Host "  .\diagnose.ps1 -Fix" -ForegroundColor White
    Write-Host ""
}
