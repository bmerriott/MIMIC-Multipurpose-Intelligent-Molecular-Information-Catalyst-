@echo off
REM Mimic AI Backend Launcher
REM This script starts all required backend services silently

setlocal enabledelayedexpansion

echo [*] Starting Mimic AI Backends...

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"

REM ============================================
REM 1. Start Ollama Server
REM ============================================
echo [*] Starting Ollama...

REM Check if Ollama process is actually running (not just port check)
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | findstr /I "ollama.exe" >nul
if %ERRORLEVEL% EQU 0 (
    echo     Ollama is already running (process found)
) else (
    REM Check if port is in use (might be leftover)
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo     Port 11434 in use - trying to start Ollama anyway
    )
    
    REM Kill any existing process that might be hanging
    taskkill /F /IM ollama.exe >nul 2>&1
    timeout /t 1 /nobreak >nul
    
    REM Find and start Ollama
    if exist "C:\Program Files\Ollama\ollama.exe" (
        start /B "" "C:\Program Files\Ollama\ollama.exe" serve >nul 2>&1
        echo     Ollama started from Program Files
    ) else (
        set "OLLAMA_PATH=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
        if exist "!OLLAMA_PATH!" (
            start /B "" "!OLLAMA_PATH!" serve >nul 2>&1
            echo     Ollama started from AppData
        ) else (
            echo     Ollama not found - please install from https://ollama.com
        )
    )
)

REM Wait a bit for Ollama to start
timeout /t 3 /nobreak >nul

REM ============================================
REM 2. Start SearXNG Docker Container
REM ============================================
echo [*] Starting SearXNG...

REM Check if Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo     Docker is not running - please start Docker Desktop
    goto :tts
)

REM Check if container already exists and is running
docker ps --filter "name=mimic-searxng" --format "{{.Names}}" | findstr "mimic-searxng" >nul
if %ERRORLEVEL% EQU 0 (
    echo     SearXNG container is already running
    goto :tts
)

REM Check if container exists but is stopped
docker ps -a --filter "name=mimic-searxng" --format "{{.Names}}" | findstr "mimic-searxng" >nul
if %ERRORLEVEL% EQU 0 (
    echo     SearXNG container exists but stopped, starting it...
    docker start mimic-searxng >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo     SearXNG container started successfully
        goto :tts
    ) else (
        echo     Failed to start existing container, recreating...
        docker rm -f mimic-searxng >nul 2>&1
    )
)

REM Pull the image first (this may take time on first run)
echo     Pulling SearXNG image (this may take a few minutes on first run)...
docker pull searxng/searxng >nul 2>&1

REM Start the container
docker run -d --name mimic-searxng -p 8080:8080 --restart unless-stopped ^
    -e SEARXNG_BASE_URL=http://localhost:8080 ^
    -e SEARXNG_SECRET=your-secret-key-change-in-production ^
    searxng/searxng >nul 2>&1
    
if %ERRORLEVEL% EQU 0 (
    echo     SearXNG container started on port 8080
) else (
    echo     Failed to start SearXNG container
)

:tts

REM ============================================
REM 3. Start Python TTS Server
REM ============================================
echo [*] Starting TTS Server...

REM Find Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo     Python not found - please install Python 3.10-3.12
    goto :done
)

REM Find the backend script - check multiple locations
set "BACKEND_SCRIPT="
set "BACKEND_DIR="

if exist "%SCRIPT_DIR%app\backend\tts_server_unified.py" (
    set "BACKEND_DIR=%SCRIPT_DIR%app\backend"
    set "BACKEND_SCRIPT=%SCRIPT_DIR%app\backend\tts_server_unified.py"
) else if exist "%SCRIPT_DIR%resources\backend\tts_server_unified.py" (
    set "BACKEND_DIR=%SCRIPT_DIR%resources\backend"
    set "BACKEND_SCRIPT=%SCRIPT_DIR%resources\backend\tts_server_unified.py"
) else if exist "%SCRIPT_DIR%..\app\backend\tts_server_unified.py" (
    set "BACKEND_DIR=%SCRIPT_DIR%..\app\backend"
    set "BACKEND_SCRIPT=%SCRIPT_DIR%..\app\backend\tts_server_unified.py"
) else (
    REM Try current directory
    cd /d "%SCRIPT_DIR%..\app\backend"
    if exist "tts_server_unified.py" (
        set "BACKEND_DIR=%SCRIPT_DIR%..\app\backend"
        set "BACKEND_SCRIPT=%SCRIPT_DIR%..\app\backend\tts_server_unified.py"
    )
)

if "%BACKEND_SCRIPT%"=="" (
    echo     Could not find tts_server_unified.py
    goto :done
)

REM Start the TTS server in background
start /B cmd /c "cd /d "%BACKEND_DIR%" && set MIMIC_PORT=8000 && set SEARXNG_URL=http://localhost:8080 && python.exe tts_server_unified.py"

echo     TTS server starting on port 8000...

:done

echo [*] All backends started!
echo.
echo Services:
echo   - Ollama:     http://localhost:11434
echo   - TTS Server: http://localhost:8000
echo   - SearXNG:    http://localhost:8080

timeout /t 2 /nobreak >nul
