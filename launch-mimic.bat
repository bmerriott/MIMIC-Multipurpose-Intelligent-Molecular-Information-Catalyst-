@echo off
echo ==========================================
echo    Mimic AI - Launch Script  
echo ==========================================
echo.
cd /d "%~dp0"

echo [*] Checking dependencies...

:: Check Python packages
python -c "import fastapi" >nul 2>&1
if %ERRORLEVEL% neq 0 goto :install_python_packages
goto :python_ok

:install_python_packages
echo     Installing Python packages (first run may take a few minutes)...
python -m pip install -q fastapi uvicorn python-multipart pydantic python-dotenv numpy scipy soundfile librosa requests torch torchaudio --index-url https://download.pytorch.org/whl/cpu
echo     Python packages installed
goto :python_done

:python_ok
echo     Python packages OK
:python_done

:: Check Node dependencies  
if exist "app\node_modules" goto :node_ok
echo [*] Installing Node dependencies...
cd "app"
call npm install
cd ".."
echo     Node dependencies installed
goto :node_done

:node_ok
echo     Node dependencies OK
:node_done

echo [*] Starting Ollama Server...

:: Find Ollama
set OLLAMA_PATH=C:\Program Files\Ollama\ollama.exe
if exist "%OLLAMA_PATH%" goto :ollama_found
set OLLAMA_PATH=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
if exist "%OLLAMA_PATH%" goto :ollama_found
echo [ERROR] Ollama not found! Install from https://ollama.com
pause
exit /b 1
:ollama_found

:: Kill existing Ollama
taskkill /F /IM ollama.exe >nul 2>&1
timeout /t 2 >nul

:: Start Ollama server
start "Ollama Server" "%OLLAMA_PATH%" serve
echo     Ollama started, waiting...
timeout /t 5 >nul

echo [*] Checking Docker...
tasklist | findstr /I "Docker Desktop.exe" >nul
if %ERRORLEVEL% equ 0 goto :docker_ready

:: Try to start Docker
if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" (
    start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
    echo     Waiting for Docker Desktop...
    timeout /t 30 >nul
) else (
    echo     Docker not found - web search disabled
    goto :skip_docker
)

:docker_ready
echo     Docker is running
echo [*] Setting up SearXNG...
docker rm -f mimic-searxng >nul 2>&1
timeout /t 2 >nul
docker run -d -p 8080:8080 --name mimic-searxng searxng/searxng:latest >nul 2>&1
timeout /t 3 >nul
docker ps | findstr mimic-searxng >nul
if %ERRORLEVEL% equ 0 (echo     SearXNG ready) else (echo     SearXNG failed)

:skip_docker
echo [*] Starting Mimic AI...

:: Kill old processes
taskkill /F /FI "WINDOWTITLE eq TTS Backend" >nul 2>&1

:: Create launch scripts
echo @echo off > "%TEMP%\mimic_backend.bat"
echo cd /d "%~dp0\app\backend" >> "%TEMP%\mimic_backend.bat"
echo python tts_server_unified.py >> "%TEMP%\mimic_backend.bat"
echo pause >> "%TEMP%\mimic_backend.bat"

echo @echo off > "%TEMP%\mimic_frontend.bat"
echo cd /d "%~dp0\app" >> "%TEMP%\mimic_frontend.bat"
echo npm run dev >> "%TEMP%\mimic_frontend.bat"
echo pause >> "%TEMP%\mimic_frontend.bat"

:: Start services
start "TTS Backend" "%TEMP%\mimic_backend.bat"
timeout /t 15 >nul

start "Frontend" "%TEMP%\mimic_frontend.bat"
timeout /t 8 >nul

start http://localhost:5173

echo.
echo ==========================================
echo    Mimic AI is Starting!
echo ==========================================
echo.
pause

echo.
echo [*] Stopping services...
taskkill /F /IM ollama.exe >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq TTS Backend" >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /F /PID %%a >nul 2>&1
docker stop mimic-searxng >nul 2>&1
echo     Done!
timeout /t 2 >nul
