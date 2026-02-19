# Mimic AI v1.0.0 - Initial Release

## 🎉 First Official Release

Mimic AI - Your privacy-first, AI-powered desktop assistant with voice synthesis, persistent memory, and local model inference.

## 📦 What's Included

| File | Description | Size |
|------|-------------|------|
| `Mimic-AI-Setup-v1.0.exe` | One-click installer (recommended) | ~121 MB |
| `Mimic-AI-v1.0.zip` | Portable ZIP archive | ~121 MB |
| `README.txt` | Installation and troubleshooting guide | - |
| `QUICKSTART.md` | Quick start documentation | - |

## 🚀 Getting Started

### Prerequisites

Before installing Mimic AI, you need:

1. **Python 3.10-3.12** 
   - Download: https://python.org/downloads/
   - ⚠️ **Important**: Check "Add Python to PATH" during installation!

2. **Ollama** (for AI models)
   - Download: https://ollama.com
   - **First-time setup**: After install, open terminal and run `ollama serve` once
   - Then pull a model: `ollama pull llama3.2`

3. **Docker Desktop** (optional, for web search)
   - Download: https://docker.com
   - Only needed if you want web search functionality

### Installation

1. Download `Mimic-AI-Setup-v1.0.exe`
2. Double-click to run the installer
3. Follow the installation wizard
4. Launch Mimic AI from Start Menu or Desktop

### First Launch

On first run, Mimic AI will:
- Configure Ollama for Tauri WebView compatibility (automatic)
- Verify Python and install required dependencies
- Start the TTS (Text-to-Speech) backend
- Start Ollama (if not already running)
- Initialize SearXNG in Docker (if enabled)

**Note**: First launch may take 30-60 seconds as services start up.

## ✨ Features

- 🤖 **Local AI Inference** - Runs entirely on your machine using Ollama
- 🗣️ **Voice Synthesis** - Dual TTS engine support (StyleTTS2 & Qwen3-TTS)
- 🧠 **Persistent Memory** - Conversation memory with per-persona context
- 🔍 **Web Search** - Privacy-focused SearXNG integration
- 🎙️ **Wake Word Detection** - Hands-free activation
- 👁️ **Vision Support** - Image analysis with vision-capable models
- 🔒 **Audio Watermarking** - All generated audio is watermarked

## 🔧 Key Features

### Automatic Ollama Configuration
Mimic AI automatically configures Ollama with the required environment variables (`OLLAMA_ORIGINS=*`) for Tauri WebView compatibility. No manual setup needed!

### Auto Model Selection
If your default model isn't installed, Mimic AI automatically selects the first available model from your Ollama installation.

### Direct SearXNG Connection
SearXNG connection is checked directly (bypassing the Python backend), so web search works immediately when Docker is running.

## 🐛 Known Issues

- First launch may take longer as Python dependencies are installed
- Docker must be running before enabling web search
- Some antivirus software may flag the installer (false positive)

## 🔧 Troubleshooting

### "Python not found" error
- Ensure Python 3.10-3.12 is installed and in PATH
- Restart Mimic AI after installing Python

### "Ollama not connected"
- **First time only**: Run `ollama serve` once to initialize Ollama
- Pull a model: `ollama pull llama3.2`
- Mimic AI will auto-start Ollama for you on subsequent launches
- Check Ollama is running: http://localhost:11434/api/tags

### "TTS backend not connected"
- Ensure Python dependencies are installed
- Check the startup log: `%APPDATA%\com.mimicai.app\startup.log`

### Web search not working
- Ensure Docker Desktop is installed and running
- Enable web search in Mimic AI settings
- SearXNG container will be created automatically

## 📋 System Requirements

- **OS**: Windows 10/11 (64-bit)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **GPU**: Optional but recommended for faster inference

## 📝 Checksums

```
SHA256: (to be added after build)
```

## 🙏 Credits

- Built with Tauri, React, and Rust
- Uses Ollama for local LLM inference
- Voice synthesis powered by StyleTTS2 and Qwen3-TTS

## 📄 License

MIT License - See LICENSE file for details

---

**Full Documentation**: See README.md in the repository
**Issue Tracker**: https://github.com/bmerriott/MIMIC-Multipurpose-Intelligent-Molecular-Information-Catalyst-/issues
