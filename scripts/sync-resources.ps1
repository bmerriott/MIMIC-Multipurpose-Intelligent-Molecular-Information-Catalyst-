# Mimic AI - Resource Sync Script
# Run this before building to ensure all backend files are in place

param(
    [switch]$Clean,
    [switch]$ShowVerbose
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BackendSource = Join-Path $ProjectRoot "app\backend"
$BackendDest = Join-Path $ProjectRoot "src-tauri\resources\backend"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Mimic AI - Resource Sync" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verify source exists
if (-not (Test-Path $BackendSource)) {
    Write-Host "ERROR: Backend source not found at $BackendSource" -ForegroundColor Red
    exit 1
}

# Create destination if needed
if (-not (Test-Path $BackendDest)) {
    Write-Host "Creating resources directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $BackendDest -Force | Out-Null
}

# Clean destination if requested
if ($Clean) {
    Write-Host "Cleaning destination directory..." -ForegroundColor Yellow
    Remove-Item -Path "$BackendDest\*" -Recurse -Force -ErrorAction SilentlyContinue
}

# Files to copy
$FilesToCopy = @(
    "tts_server_unified.py",
    "watermarker.py",
    "detect_watermark.py",
    "memory_tools.py",
    "persona_rules.py",
    "streaming_tts.py",
    "voice_profile_manager.py",
    "tts_server_styletts2.py",
    "requirements.txt",
    ".env",
    "searxng_settings.yml"
)

Write-Host "Copying backend files..." -ForegroundColor Yellow
$CopiedCount = 0
$SkippedCount = 0

foreach ($file in $FilesToCopy) {
    $SourcePath = Join-Path $BackendSource $file
    $DestPath = Join-Path $BackendDest $file
    
    if (Test-Path $SourcePath) {
        $SourceHash = Get-FileHash $SourcePath -Algorithm SHA256 -ErrorAction SilentlyContinue
        $DestHash = if (Test-Path $DestPath) { Get-FileHash $DestPath -Algorithm SHA256 -ErrorAction SilentlyContinue } else { $null }
        
        $NeedsCopy = $true
        if ($SourceHash -and $DestHash) {
            if ($SourceHash.Hash -eq $DestHash.Hash) {
                $NeedsCopy = $false
            }
        }
        
        if ($NeedsCopy) {
            Copy-Item -Path $SourcePath -Destination $DestPath -Force
            Write-Host "  -> $file" -ForegroundColor Green
            $CopiedCount++
        } else {
            if ($ShowVerbose) { 
                Write-Host "  [OK] $file (up to date)" -ForegroundColor Gray 
            }
            $SkippedCount++
        }
    } else {
        if ($ShowVerbose) { 
            Write-Host "  [WARN] $file (not found in source)" -ForegroundColor Yellow 
        }
    }
}

# Create directories if they don't exist
$DirsToCreate = @("enrolled_voices", "saved_voices", "voice_references")
foreach ($dir in $DirsToCreate) {
    $DirPath = Join-Path $BackendDest $dir
    if (-not (Test-Path $DirPath)) {
        New-Item -ItemType Directory -Path $DirPath -Force | Out-Null
        if ($ShowVerbose) { 
            Write-Host "  [DIR] Created: $dir" -ForegroundColor Green 
        }
    }
}

Write-Host ""
Write-Host "Sync complete!" -ForegroundColor Green
Write-Host "  Copied: $CopiedCount files" -ForegroundColor White
Write-Host "  Skipped: $SkippedCount files (unchanged)" -ForegroundColor Gray
Write-Host ""
Write-Host "Resources ready for build." -ForegroundColor Cyan
