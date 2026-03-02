@echo off
echo =========================================
echo  Mimic AI - Dependency Installer
echo =========================================
echo.

:: Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This installer requires administrator privileges.
    echo Please right-click and select "Run as administrator"
    pause
    exit /b 1
)

:: Check for espeak-ng
if exist "C:\Program Files\eSpeak NG\espeak-ng.exe" (
    echo espeak-ng is already installed.
    goto :espeak_done
)

:: Install espeak-ng from bundled MSI
echo Installing espeak-ng (required for KittenTTS)...
if exist "%~dp0espeak-ng.msi" (
    echo Found espeak-ng.msi, installing...
    msiexec /i "%~dp0espeak-ng.msi" /qn /norestart
    if %errorlevel% == 0 (
        echo espeak-ng installed successfully!
    ) else (
        echo Warning: espeak-ng installation may have failed (code: %errorlevel%)
        echo KittenTTS voice engine may not work without espeak-ng.
    )
) else (
    echo WARNING: espeak-ng.msi not found!
    echo Please download from: https://github.com/espeak-ng/espeak-ng/releases
)

:espeak_done
echo.
echo =========================================
echo  Installation Complete
echo =========================================
echo.
pause
