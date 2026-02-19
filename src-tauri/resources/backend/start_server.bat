@echo off
cd /d "%~dp0"

echo ==========================================
echo Mimic AI TTS Backend
echo ==========================================
echo.

echo Setting up environment...
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo.
echo Checking Python...
python --version
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo.
echo Testing imports...
python -c "import fastapi, uvicorn, numpy" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Missing required packages!
    echo Run: pip install fastapi uvicorn numpy scipy soundfile python-dotenv requests
    pause
    exit /b 1
)

echo.
echo Starting server on port 8000...
echo Press Ctrl+C to stop
echo.
echo If this window closes immediately, check:
echo   1. Is port 8000 already in use? (check with: netstat -ano | findstr :8000)
echo   2. Are there any import errors above?
echo   3. Check Windows Defender or antivirus blocking Python
echo.

:: Run server and capture exit code
python -u tts_server_unified.py

set EXITCODE=%ERRORLEVEL%

echo.
echo ==========================================
if %EXITCODE% equ 0 (
    echo [INFO] Server stopped normally
) else (
    echo [ERROR] Server crashed with exit code: %EXITCODE%
)
echo ==========================================
echo.
pause
