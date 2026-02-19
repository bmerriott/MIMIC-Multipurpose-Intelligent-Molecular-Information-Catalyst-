# Mimic AI - Build Cleanup Script
# Removes unnecessary build artifacts to free disk space

param(
    [switch]$DeepClean,
    [switch]$WhatIf
)

$ErrorActionPreference = "Continue"

function Get-FolderSize {
    param($Path)
    $size = (Get-ChildItem $Path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    return [math]::Round($size / 1MB, 2)
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Mimic AI - Build Cleanup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check current sizes
Write-Host "Current folder sizes:" -ForegroundColor Yellow
$tauriTarget = "src-tauri\target"
$tauriSize = if (Test-Path $tauriTarget) { Get-FolderSize $tauriTarget } else { 0 }
$releaseSize = if (Test-Path "releases") { Get-FolderSize "releases" } else { 0 }

Write-Host "  src-tauri/target: $tauriSize MB" -ForegroundColor White
Write-Host "  releases: $releaseSize MB" -ForegroundColor White
Write-Host ""

if ($WhatIf) {
    Write-Host "WhatIf mode - no changes will be made" -ForegroundColor Yellow
    Write-Host ""
}

# Safe cleanup: Remove debug builds only
Write-Host "[1/3] Cleaning debug build artifacts..." -ForegroundColor Yellow
$debugPaths = @(
    "src-tauri\target\debug",
    "src-tauri\target\x86_64-pc-windows-msvc\debug"
)

foreach ($path in $debugPaths) {
    if (Test-Path $path) {
        $size = Get-FolderSize $path
        if ($WhatIf) {
            Write-Host "  Would remove: $path ($size MB)" -ForegroundColor Gray
        } else {
            Write-Host "  Removing: $path ($size MB)..." -ForegroundColor Gray
            Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# Clean old release bundles (keep only latest)
Write-Host "[2/3] Cleaning old release bundles..." -ForegroundColor Yellow
$bundleDir = "src-tauri\target\release\bundle"
if (Test-Path $bundleDir) {
    # Keep NSIS and MSI, but remove other formats if they exist
    $toRemove = Get-ChildItem -Path $bundleDir -Directory | Where-Object { $_.Name -notin @('nsis', 'msi') }
    foreach ($dir in $toRemove) {
        $size = Get-FolderSize $dir.FullName
        if ($WhatIf) {
            Write-Host "  Would remove: $($dir.FullName) ($size MB)" -ForegroundColor Gray
        } else {
            Write-Host "  Removing: $($dir.Name) ($size MB)..." -ForegroundColor Gray
            Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# Deep clean: Remove entire target (requires rebuild)
if ($DeepClean) {
    Write-Host "[3/3] DEEP CLEAN: Removing entire target folder..." -ForegroundColor Red
    if ($WhatIf) {
        Write-Host "  Would remove: src-tauri\target ($tauriSize MB)" -ForegroundColor Red
    } else {
        $confirm = Read-Host "This will delete $tauriSize MB and require a full rebuild. Continue? (y/n)"
        if ($confirm -eq 'y') {
            Remove-Item -Path "src-tauri\target" -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "  Target folder removed." -ForegroundColor Green
        } else {
            Write-Host "  Cancelled." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "[3/3] Skipping deep clean (use -DeepClean to remove entire target)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Cleanup complete!" -ForegroundColor Green

# Show new sizes
Write-Host ""
Write-Host "Remaining folder sizes:" -ForegroundColor Yellow
$newTauriSize = if (Test-Path $tauriTarget) { Get-FolderSize $tauriTarget } else { 0 }
$newReleaseSize = if (Test-Path "releases") { Get-FolderSize "releases" } else { 0 }
Write-Host "  src-tauri/target: $newTauriSize MB" -ForegroundColor White
Write-Host "  releases: $newReleaseSize MB" -ForegroundColor White
Write-Host ""
Write-Host "Space saved: $([math]::Round($tauriSize - $newTauriSize, 2)) MB" -ForegroundColor Green
