# Mimic AI Installer Build Script
# Run this script to build the one-click installer

param(
    [switch]$Dev,
    [switch]$SkipFrontend,
    [switch]$SkipRust
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Mimic AI - Installer Builder" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

# Check Rust
$rustVersion = rustc --version 2>$null
if (-not $rustVersion) {
    Write-Host "ERROR: Rust not found. Install with: winget install Rustlang.Rustup" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Rust: $rustVersion" -ForegroundColor Green

# Check Node.js
$nodeVersion = node --version 2>$null
if (-not $nodeVersion) {
    Write-Host "ERROR: Node.js not found. Install from https://nodejs.org" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Node.js: $nodeVersion" -ForegroundColor Green

# Check Python (warn only)
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "not found") {
    Write-Host "  ⚠ Python not found (will be required at runtime)" -ForegroundColor Yellow
} else {
    Write-Host "  ✓ Python: $pythonVersion" -ForegroundColor Green
}

Write-Host ""

# Install npm dependencies
if (-not $SkipFrontend) {
    Write-Host "Installing npm dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: npm install failed" -ForegroundColor Red
        exit 1
    }
}

# Build frontend
if (-not $SkipFrontend) {
    Write-Host "Building frontend..." -ForegroundColor Yellow
    Push-Location app
    npm install
    npm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Frontend build failed" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Pop-Location
}

# Build Tauri app
if (-not $SkipRust) {
    Write-Host "Building Tauri application..." -ForegroundColor Yellow
    Write-Host "  (This may take 10-20 minutes on first build)" -ForegroundColor Gray
    
    if ($Dev) {
        cargo tauri dev
    } else {
        cargo tauri build
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Tauri build failed" -ForegroundColor Red
        exit 1
    }
}

# Show output location
if (-not $Dev) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Build Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Installer location:" -ForegroundColor Cyan
    
    $installerPath = "src-tauri\target\release\bundle\nsis\Mimic-AI-Setup.exe"
    if (Test-Path $installerPath) {
        $fullPath = Resolve-Path $installerPath
        Write-Host "  $fullPath" -ForegroundColor White
        Write-Host ""
        Write-Host "File size: $([math]::Round((Get-Item $installerPath).Length / 1MB, 2)) MB" -ForegroundColor Gray
    } else {
        Write-Host "  (Check src-tauri/target/release/bundle/)" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "To distribute:" -ForegroundColor Cyan
    Write-Host "  1. Upload $installerPath to your releases page" -ForegroundColor White
    Write-Host "  2. Users download and double-click to install" -ForegroundColor White
    Write-Host ""
    Write-Host "Prerequisites for users:" -ForegroundColor Yellow
    Write-Host "  - Windows 10/11 (64-bit)" -ForegroundColor Gray
    Write-Host "  - Python 3.10+ (will be checked at runtime)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
