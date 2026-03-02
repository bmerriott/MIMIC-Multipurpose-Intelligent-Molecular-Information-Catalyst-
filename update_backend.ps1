# Run this script as Administrator to update the production backend
# Right-click -> Run with PowerShell (as Administrator)

$source = "C:\Users\merri\Downloads\Mimic AI Desktop Assistant\app\backend\tts_server_unified.py"
$dest = "C:\Program Files\Mimic AI\resources\backend\tts_server_unified.py"
$backup = "C:\Program Files\Mimic AI\resources\backend\tts_server_unified.py.backup"

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "ERROR: Please run this script as Administrator!" -ForegroundColor Red
    Write-Host "Right-click -> Run with PowerShell (as Administrator)"
    pause
    exit 1
}

# Create backup
if (Test-Path $dest) {
    Copy-Item $dest $backup -Force
    Write-Host "Created backup: $backup" -ForegroundColor Green
}

# Copy new backend
try {
    Copy-Item $source $dest -Force
    Write-Host "Successfully updated production backend!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Please restart Mimic AI for changes to take effect." -ForegroundColor Yellow
} catch {
    Write-Host "ERROR: Failed to update: $_" -ForegroundColor Red
}

pause
