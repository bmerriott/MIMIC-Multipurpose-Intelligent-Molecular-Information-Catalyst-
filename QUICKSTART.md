# Mimic AI - Quick Start Guide

Get Mimic AI running in minutes with this streamlined setup guide.

---

## Prerequisites

Install these three components before launching Mimic AI:

| Requirement | Version | Download Link |
|-------------|---------|---------------|
| **Python** | 3.10, 3.11, or 3.12 | https://python.org/downloads |
| **Ollama** | Latest | https://ollama.com |
| **Docker Desktop** | Latest | https://docker.com |

**Important:** Python 3.13+ is **NOT supported** due to PyTorch compatibility issues.

---

## Step 1: Configure Ollama

After installing Ollama, open a terminal or the application and pull/use at least one AI model:

```powershell
ollama pull llama3.2
```

**Optional models for additional features:**
```powershell
ollama pull bakllava      # For image/vision support
ollama pull mistral       # Alternative conversation model
```

---

## Step 2: Close Background Services

**Critical:** Close these applications completely before launching Mimic AI. Check the system tray (bottom-right corner of your screen) and quit both:

1. **Ollama** - Right-click icon → Quit Ollama
2. **Docker Desktop** - Right-click icon → Quit Docker Desktop

**Why?** Mimic AI manages these services automatically. If they're already running, it will increase memory usage.

### Verify They're Closed

Press `Ctrl+Shift+Esc` to open Task Manager:
- **Processes** tab: Ensure `ollama.exe` is not running
- **System tray**: No Docker or Ollama icons visible

---

## Step 3: Launch Mimic AI

Choose how you want to run the application:

### Option A: Desktop Application (Recommended)

Simply run the installer in the root folder:

```powershell
.\Mimic AI Setup.exe
```

Or double-click the installer file.

**What happens:**
- Installs Mimic AI to your system
- Creates Start Menu and Desktop shortcuts
- All services run in background (no terminal windows)
- Auto-starts/stops Ollama, TTS backend, and SearXNG

**Best for:** Daily use, cleaner experience, system tray integration

---

### Option B: Web Browser Mode

Run the launcher script from the project folder:

```powershell
.\launch-mimic.bat
```

**What happens:**
- Starts all backend services (Ollama, TTS server, optional SearXNG)
- Opens your default browser at `http://localhost:5173`
- Keep the terminal window open (services stop when closed)

**Best for:** Development, quick testing, or if you prefer browser tabs

---

## Mode Comparison

| Feature | Desktop App | Web Browser |
|---------|-------------|-------------|
| Setup | Run installer | Run batch script |
| Window type | Native window | Browser tab |
| Backend visibility | Hidden processes | Terminal windows |
| Service management | Automatic | Manual |
| Voice data storage | Windows AppData | Browser IndexedDB |
| Install location | `Program Files` | Any folder |
| Uninstall | Windows Add/Remove Programs | Delete folder |

---

## First Launch Checklist

When you first open Mimic AI:

1. **Accept Terms** - Review and accept the AI disclosure agreement
2. **Select Model** - Choose your downloaded Ollama model from the dropdown
3. **Test Voice** - Create a persona and test voice synthesis in Voice Studio
4. **Enable Search** - Toggle Web Search in Settings (requires Docker)

---

## Troubleshooting

### "Python not found" error
- Install Python 3.10-3.12 from python.org
- **Must check** "Add Python to PATH" during installation
- Verify: Open new PowerShell and run `python --version`

### "Ollama not connected"
1. Ensure Ollama is fully closed (Task Manager)
2. Restart Mimic AI
3. Verify at least one model is pulled: `ollama list`

### "Port 8000/11434/8080 already in use"
- Open Task Manager → Details tab
- End any `ollama.exe`, `python.exe`, or `docker` processes
- Retry launch

### Docker/SearXNG not starting
1. Start Docker Desktop manually
2. Wait 30 seconds for the Docker engine
3. Restart Mimic AI

### Voice synthesis not working
1. Check TTS backend connection (green indicator in top-right)
2. Ensure voice is created and saved to persona (Voice Studio tab)
3. Try switching TTS engine: Settings → Voice → TTS Engine

---

## Next Steps

- **Create a Persona:** Personas tab → New Persona → Customize personality and voice
- **Upload Memories:** Memory Manager → New Memory File → Add knowledge for AI to reference
- **Enable Wake Word:** Settings → Voice → Enable Auto-Listen for hands-free activation
- **Voice Creation:** Voice Studio → Record or upload reference audio → Create custom voice

---

## Need Help?

- Check the full documentation: [README.md](README.md)
- Enable debug mode: Open browser console (F12) and run `localStorage.setItem('mimic_debug', 'true')`
- View backend logs: `%APPDATA%\com.mimicai.app\startup.log` (Desktop mode)
