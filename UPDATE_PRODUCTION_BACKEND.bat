@echo off
:: Update Production Backend - Run as Administrator
:: This script updates the installed Mimic AI backend with the optimized version

echo ============================================
echo  Mimic AI - Update Production Backend
echo ============================================
echo.

:: Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo.
    echo Please:
    echo 1. Right-click on UPDATE_PRODUCTION_BACKEND.bat
    echo 2. Select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo [OK] Running with Administrator privileges
echo.

:: Define paths
set "SOURCE=%~dp0app\backend\tts_server_unified.py"
set "DEST=C:\Program Files\Mimic AI\resources\backend\tts_server_unified.py"
set "BACKUP=C:\Program Files\Mimic AI\resources\backend\tts_server_unified.py.backup"

echo Source: %SOURCE%
echo Destination: %DEST%
echo.

:: Check if source exists
if not exist "%SOURCE%" (
    echo ERROR: Source file not found!
    echo %SOURCE%
    echo.
    pause
    exit /b 1
)

:: Check if destination exists
if not exist "%DEST%" (
    echo WARNING: Destination not found. Is Mimic AI installed?
    echo %DEST%
    echo.
    pause
    exit /b 1
)

:: Create backup
echo [1/3] Creating backup...
copy /Y "%DEST%" "%BACKUP%" >nul
if %errorLevel% neq 0 (
    echo WARNING: Could not create backup, continuing anyway...
) else (
    echo [OK] Backup created: %BACKUP%
)

:: Stop any running Python backend processes
echo [2/3] Stopping running backend processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM pythonw.exe >nul 2>&1
timeout /t 1 /nobreak >nul
echo [OK] Backend processes stopped

:: Copy new backend
echo [3/3] Installing updated backend...
copy /Y "%SOURCE%" "%DEST%" >nul
if %errorLevel% neq 0 (
    echo ERROR: Failed to copy file!
    echo.
    pause
    exit /b 1
)

echo [OK] Backend updated successfully!
echo.
echo ============================================
echo  What's New in This Update:
echo ============================================
echo.
echo Qwen3-TTS Performance:
echo   - Added parallel chunking for long text
echo   - Text split at sentence boundaries (~150 chars)
echo   - Chunks processed in parallel (4 workers)
echo   - Crossfade for smooth audio concatenation
echo   - Expected RTF improvement: 3-4x to ~0.5-0.8
echo.
echo KittenTTS Long Text Support:
echo   - Added paragraph-level chunking (~380 chars)
echo   - Automatic splitting at sentence boundaries
echo   - Sequential chunk processing with crossfade
echo   - Fixes "invalid expand shape" errors
echo.
echo ============================================
echo.
echo Please restart Mimic AI to use the updated backend.
echo.
pause
