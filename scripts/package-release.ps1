# Mimic AI - Package Release Script
# Packages the existing build with README and creates ZIP

param(
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BundleDir = Join-Path $ProjectRoot "src-tauri\target\release\bundle\nsis"
$ReleaseDir = Join-Path $ProjectRoot "releases"
$ReleaseVersionDir = Join-Path $ReleaseDir "Mimic-AI-v$Version"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Mimic AI - Package Release" -ForegroundColor Cyan
Write-Host "  Version: $Version" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if build exists
if (-not (Test-Path $BundleDir)) {
    Write-Host "ERROR: Build directory not found: $BundleDir" -ForegroundColor Red
    Write-Host "Please run 'npm run tauri build' first." -ForegroundColor Yellow
    exit 1
}

# Find the installer
$Installer = Get-ChildItem -Path $BundleDir -Filter "*.exe" | Select-Object -First 1
if (-not $Installer) {
    Write-Host "ERROR: No installer found in $BundleDir" -ForegroundColor Red
    exit 1
}

Write-Host "Found installer: $($Installer.Name)" -ForegroundColor Green
Write-Host ""

# Create release directory
Write-Host "Creating release package..." -ForegroundColor Yellow

if (Test-Path $ReleaseVersionDir) {
    Remove-Item -Path $ReleaseVersionDir -Recurse -Force
}
New-Item -ItemType Directory -Path $ReleaseVersionDir -Force | Out-Null

# Copy installer with cleaner name
$InstallerDestName = "Mimic-AI-Setup-v$Version.exe"
$InstallerDest = Join-Path $ReleaseVersionDir $InstallerDestName
Copy-Item -Path $Installer.FullName -Destination $InstallerDest
Write-Host "  [OK] Installer copied -> $InstallerDestName" -ForegroundColor Green

# Copy README
$ReadmeSource = Join-Path $ProjectRoot "scripts\RELEASE-README.txt"
$ReadmeDest = Join-Path $ReleaseVersionDir "README.txt"
if (Test-Path $ReadmeSource) {
    Copy-Item -Path $ReadmeSource -Destination $ReadmeDest
    Write-Host "  [OK] README copied" -ForegroundColor Green
} else {
    Write-Host "  [WARN] README not found" -ForegroundColor Yellow
}

# Copy QUICKSTART
$QuickstartSource = Join-Path $ProjectRoot "QUICKSTART.md"
if (Test-Path $QuickstartSource) {
    Copy-Item -Path $QuickstartSource -Destination (Join-Path $ReleaseVersionDir "QUICKSTART.md")
    Write-Host "  [OK] QUICKSTART copied" -ForegroundColor Green
}

# Create ZIP
Write-Host ""
Write-Host "Creating ZIP archive..." -ForegroundColor Yellow
$ZipPath = "$ReleaseVersionDir.zip"
if (Test-Path $ZipPath) {
    Remove-Item -Path $ZipPath -Force
}

Compress-Archive -Path $ReleaseVersionDir -DestinationPath $ZipPath
Write-Host "  [OK] ZIP created" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Package Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Location: $ReleaseVersionDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "Files to upload to GitHub:" -ForegroundColor Yellow

$Files = Get-ChildItem -Path $ReleaseVersionDir
foreach ($file in $Files) {
    $Size = [math]::Round($file.Length / 1MB, 2)
    Write-Host "  - $($file.Name) ($Size MB)" -ForegroundColor White
}

$ZipSize = [math]::Round((Get-Item $ZipPath).Length / 1MB, 2)
Write-Host "  - Mimic-AI-v$Version.zip ($ZipSize MB) [Archive]" -ForegroundColor White

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Upload these files to your GitHub release" -ForegroundColor White
Write-Host "  2. Copy RELEASE-NOTES.md content to release description" -ForegroundColor White
Write-Host ""
