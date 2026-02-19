# Mimic AI - Installation Verification Script
# Run this to check if all components are working correctly

param(
    [switch]$Fix
)

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Mimic AI - Installation Check" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$AllGood = $true

# 1. Check Python
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
$PythonVersion = python --version 2>&1
if ($PythonVersion -match "Python 3\.(10|11|12)") {
    Write-Host "  ✓ Python found: $PythonVersion" -ForegroundColor Green
} elseif ($PythonVersion -match "Python 3\.(13|[0-9])") {
    Write-Host "  ⚠ Python version may not be compatible: $PythonVersion" -ForegroundColor Yellow
    Write-Host "      Recommended: Python 3.10, 3.11, or 3.12" -ForegroundColor Gray
    $AllGood = $false
} else {
    Write-Host "  ✗ Python not found or not in PATH" -ForegroundColor Red
    Write-Host "      Please install Python 3.10-3.12 from https://python.org" -ForegroundColor Gray
    $AllGood = $false
}
Write-Host ""

# 2. Check Python dependencies
Write-Host "[2/6] Checking Python dependencies..." -ForegroundColor Yellow
$RequiredModules = @("fastapi", "uvicorn", "torch", "numpy", "soundfile")
$MissingModules = @()

foreach ($module in $RequiredModules) {
    $result = python -c "import $module; print('OK')" 2>&1
    if ($result -eq "OK") {
        Write-Host "  ✓ $module" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $module (missing)" -ForegroundColor Red
        $MissingModules += $module
        $AllGood = $false
    }
}

if ($Fix -and $MissingModules.Count -gt 0) {
    Write-Host "  → Installing missing modules..." -ForegroundColor Yellow
    $modulesString = $MissingModules -join " "
    python -m pip install $modulesString
}
Write-Host ""

# 3. Check Ollama
Write-Host "[3/6] Checking Ollama..." -ForegroundColor Yellow
try {
    $OllamaResponse = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
    $OllamaData = $OllamaResponse.Content | ConvertFrom-Json
    $ModelCount = $OllamaData.models.Count
    Write-Host "  ✓ Ollama is running" -ForegroundColor Green
    Write-Host "      Models available: $ModelCount" -ForegroundColor Gray
    if ($ModelCount -eq 0) {
        Write-Host "      ⚠ No models found. Run: ollama pull llama3.2" -ForegroundColor Yellow
        $AllGood = $false
    }
} catch {
    Write-Host "  ✗ Ollama is not running or not installed" -ForegroundColor Red
    Write-Host "      Please install from https://ollama.com" -ForegroundColor Gray
    Write-Host "      Then pull a model: ollama pull llama3.2" -ForegroundColor Gray
    $AllGood = $false
}
Write-Host ""

# 4. Check Docker (optional)
Write-Host "[4/6] Checking Docker (optional)..." -ForegroundColor Yellow
try {
    $null = docker version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Docker is running" -ForegroundColor Green
        
        # Check SearXNG container
        $SearxContainer = docker ps --filter "name=mimic-searxng" --format "{{.Names}}" 2>&1
        if ($SearxContainer -eq "mimic-searxng") {
            Write-Host "  ✓ SearXNG container is running" -ForegroundColor Green
        } else {
            Write-Host "  ⚠ SearXNG container not found (will be created on first use)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ⚠ Docker is installed but not running" -ForegroundColor Yellow
        Write-Host "      Please start Docker Desktop" -ForegroundColor Gray
    }
} catch {
    Write-Host "  ⚠ Docker not installed (optional for web search)" -ForegroundColor Yellow
}
Write-Host ""

# 5. Check Mimic AI installation
Write-Host "[5/6] Checking Mimic AI installation..." -ForegroundColor Yellow
$MimicPaths = @(
    "$env:LOCALAPPDATA\Mimic AI",
    "$env:ProgramFiles\Mimic AI",
    "$env:USERPROFILE\AppData\Local\Mimic AI"
)

$MimicFound = $false
foreach ($path in $MimicPaths) {
    if (Test-Path $path) {
        Write-Host "  ✓ Mimic AI found at: $path" -ForegroundColor Green
        $MimicFound = $true
        break
    }
}

if (-not $MimicFound) {
    Write-Host "  ⚠ Mimic AI installation not found in standard locations" -ForegroundColor Yellow
}
Write-Host ""

# 6. Check startup log
Write-Host "[6/6] Checking startup log..." -ForegroundColor Yellow
$LogPath = "$env:APPDATA\com.mimicai.app\startup.log"
if (Test-Path $LogPath) {
    Write-Host "  ✓ Startup log found" -ForegroundColor Green
    $LastLines = Get-Content $LogPath -Tail 10
    Write-Host "      Recent entries:" -ForegroundColor Gray
    foreach ($line in $LastLines) {
        Write-Host "        $line" -ForegroundColor DarkGray
    }
} else {
    Write-Host "  ⚠ No startup log found (app may not have run yet)" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
if ($AllGood) {
    Write-Host "  ✓ All checks passed!" -ForegroundColor Green
    Write-Host "  Mimic AI should be ready to use." -ForegroundColor Green
} else {
    Write-Host "  ⚠ Some checks failed" -ForegroundColor Yellow
    Write-Host "  Please address the issues above." -ForegroundColor Yellow
    if (-not $Fix) {
        Write-Host "  Run with -Fix flag to auto-fix some issues:" -ForegroundColor Gray
        Write-Host "    .\verify-installation.ps1 -Fix" -ForegroundColor White
    }
}
Write-Host "========================================" -ForegroundColor Cyan

exit $(if ($AllGood) { 0 } else { 1 })
