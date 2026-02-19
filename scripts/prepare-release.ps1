# Mimic AI - Release Preparation Script
# This script prepares a complete release package for GitHub

param(
    [string]$Version = "1.0.0",
    [switch]$SkipBuild,
    [switch]$SkipSync,
    [switch]$CreateZip
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ReleaseDir = Join-Path $ProjectRoot "releases"
$ReleaseVersionDir = Join-Path $ReleaseDir "Mimic-AI-v$Version"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Mimic AI - Release Preparation" -ForegroundColor Cyan
Write-Host "  Version: $Version" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Sync Resources
if (-not $SkipSync) {
    Write-Host "[1/5] Syncing backend resources..." -ForegroundColor Yellow
    $SyncScript = Join-Path $ProjectRoot "scripts\sync-resources.ps1"
    if (Test-Path $SyncScript) {
        & $SyncScript
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Resource sync failed" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "WARNING: Sync script not found, skipping..." -ForegroundColor Yellow
    }
    Write-Host ""
}

# Step 2: Build Frontend
if (-not $SkipBuild) {
    Write-Host "[2/5] Building frontend..." -ForegroundColor Yellow
    Push-Location (Join-Path $ProjectRoot "app")
    
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: npm install failed" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    
    npm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Frontend build failed" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    
    Pop-Location
    Write-Host "  ✓ Frontend built successfully" -ForegroundColor Green
    Write-Host ""
}

# Step 3: Build Tauri App
if (-not $SkipBuild) {
    Write-Host "[3/5] Building Tauri application..." -ForegroundColor Yellow
    Write-Host "  (This may take 10-20 minutes on first build)" -ForegroundColor Gray
    
    Push-Location $ProjectRoot
    
    npx tauri build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Tauri build failed" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    
    Pop-Location
    Write-Host "  ✓ Tauri build complete" -ForegroundColor Green
    Write-Host ""
}

# Step 4: Prepare Release Directory
Write-Host "[4/5] Preparing release package..." -ForegroundColor Yellow

# Clean and create release directory
if (Test-Path $ReleaseVersionDir) {
    Remove-Item -Path $ReleaseVersionDir -Recurse -Force
}
New-Item -ItemType Directory -Path $ReleaseVersionDir -Force | Out-Null

# Copy installer
$InstallerSource = Join-Path $ProjectRoot "src-tauri\target\release\bundle\nsis\Mimic AI_${Version}_x64-setup.exe"
$InstallerDest = Join-Path $ReleaseVersionDir "Mimic-AI-Setup-v$Version.exe"

if (Test-Path $InstallerSource) {
    Copy-Item -Path $InstallerSource -Destination $InstallerDest
    Write-Host "  ✓ Installer copied" -ForegroundColor Green
} else {
    # Try alternative naming
    $InstallerSource = Join-Path $ProjectRoot "src-tauri\target\release\bundle\nsis\Mimic-AI-Setup.exe"
    if (Test-Path $InstallerSource) {
        Copy-Item -Path $InstallerSource -Destination $InstallerDest
        Write-Host "  ✓ Installer copied" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Installer not found at expected location" -ForegroundColor Yellow
        Write-Host "    Checked: $InstallerSource" -ForegroundColor Gray
    }
}

# Copy MSI if it exists
$MsiSource = Join-Path $ProjectRoot "src-tauri\target\release\bundle\msi\Mimic AI_${Version}_x64_en-US.msi"
$MsiDest = Join-Path $ReleaseVersionDir "Mimic-AI-v$Version.msi"
if (Test-Path $MsiSource) {
    Copy-Item -Path $MsiSource -Destination $MsiDest
    Write-Host "  ✓ MSI installer copied" -ForegroundColor Green
}

# Copy README
$ReadmeSource = Join-Path $ProjectRoot "scripts\RELEASE-README.txt"
$ReadmeDest = Join-Path $ReleaseVersionDir "README.txt"
if (Test-Path $ReadmeSource) {
    Copy-Item -Path $ReadmeSource -Destination $ReadmeDest
    Write-Host "  ✓ README copied" -ForegroundColor Green
}

# Copy quick start guide
$QuickstartSource = Join-Path $ProjectRoot "QUICKSTART.md"
if (Test-Path $QuickstartSource) {
    Copy-Item -Path $QuickstartSource -Destination (Join-Path $ReleaseVersionDir "QUICKSTART.md")
    Write-Host "  ✓ Quick start guide copied" -ForegroundColor Green
}

Write-Host ""

# Step 5: Create ZIP (optional)
if ($CreateZip) {
    Write-Host "[5/5] Creating ZIP archive..." -ForegroundColor Yellow
    $ZipPath = "$ReleaseVersionDir.zip"
    if (Test-Path $ZipPath) {
        Remove-Item -Path $ZipPath -Force
    }
    
    Compress-Archive -Path $ReleaseVersionDir -DestinationPath $ZipPath
    Write-Host "  ✓ ZIP archive created" -ForegroundColor Green
    Write-Host "    Location: $ZipPath" -ForegroundColor Gray
    Write-Host ""
}

# Summary
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Release Package Ready!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Location: $ReleaseVersionDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "Files to upload to GitHub:" -ForegroundColor Yellow

$Files = Get-ChildItem -Path $ReleaseVersionDir
foreach ($file in $Files) {
    $Size = [math]::Round($file.Length / 1MB, 2)
    Write-Host "  • $($file.Name) ($Size MB)" -ForegroundColor White
}

if ($CreateZip) {
    $ZipSize = [math]::Round((Get-Item "$ReleaseVersionDir.zip").Length / 1MB, 2)
    Write-Host "  • Mimic-AI-v$Version.zip ($ZipSize MB) [Archive]" -ForegroundColor White
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Test the installer on a clean machine" -ForegroundColor White
Write-Host "  2. Create a new release on GitHub" -ForegroundColor White
Write-Host "  3. Upload the files above to the release" -ForegroundColor White
Write-Host "  4. Add release notes with installation instructions" -ForegroundColor White
Write-Host ""
