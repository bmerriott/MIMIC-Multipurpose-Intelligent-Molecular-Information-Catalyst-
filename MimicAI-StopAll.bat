@echo off
:: Stop all Mimic AI services

title Mimic AI - Stop All Services
echo ============================================
echo     Mimic AI - Stopping All Services
echo ============================================
echo.

:: Kill by window title
echo [1/4] Stopping TTS Backend...
taskkill /F /FI "WINDOWTITLE eq *TTS Backend*" 2>NUL
taskkill /F /FI "WINDOWTITLE eq Mimic AI - TTS*" 2>NUL
echo     Done

echo [2/4] Stopping Frontend...
taskkill /F /FI "WINDOWTITLE eq *Frontend*" 2>NUL
taskkill /F /FI "WINDOWTITLE eq Mimic AI - Frontend*" 2>NUL
taskkill /F /FI "WINDOWTITLE eq *Vite*" 2>NUL
echo     Done

:: Kill by port (forcefully)
echo [3/4] Clearing port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000"') do (
    echo     Killing PID %%a
    taskkill /F /PID %%a 2>NUL
)
echo     Done

echo [4/4] Clearing port 5173...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5173"') do (
    echo     Killing PID %%a
    taskkill /F /PID %%a 2>NUL
)
echo     Done

echo.
echo ============================================
echo     All services stopped
echo ============================================
echo.
timeout /t 2
