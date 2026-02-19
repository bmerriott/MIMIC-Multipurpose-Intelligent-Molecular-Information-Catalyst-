@echo off
chcp 65001 >nul
echo ======================================
echo     Mimic AI Assistant Launcher
echo ======================================
echo.

:: Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo [OK] Node.js found
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Python not found. TTS backend will not be available.
    echo For voice cloning, install Python 3.10+ from https://python.org/
    echo.
)

:: Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama is not running!
    echo Please start Ollama first:
    echo   1. Install from https://ollama.ai
    echo   2. Pull a model: ollama pull llama3.2
    echo   3. Start Ollama
    echo.
    choice /C YN /M "Continue without Ollama"
    if errorlevel 2 exit /b 1
) else (
    echo [OK] Ollama is running
)

echo.
echo ======================================
echo Starting Mimic AI Assistant...
echo ======================================
echo.

:: Start TTS backend in new window (if Python is available)
python --version >nul 2>&1
if errorlevel 0 (
    echo Starting TTS Backend...
    start "Mimic TTS Backend" cmd /k "cd backend && python tts_server_simple.py"
    timeout /t 3 /nobreak >nul
)

:: Start frontend
echo Starting Frontend...
echo.
npm run dev

pause
