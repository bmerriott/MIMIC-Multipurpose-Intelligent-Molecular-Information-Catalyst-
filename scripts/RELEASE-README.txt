================================================================================
  MIMIC AI - Desktop Assistant v1.0
  Multipurpose Intelligent Molecular Information Catalyst
================================================================================

SYSTEM REQUIREMENTS
--------------------------------------------------------------------------------
  OS: Windows 10/11 (64-bit)
  RAM: 8GB minimum (16GB recommended)
  Storage: 5GB free space
  GPU: Optional but recommended for faster AI inference

PREREQUISITES (Must be installed before running Mimic AI)
--------------------------------------------------------------------------------
  1. Python 3.10, 3.11, or 3.12
     Download: https://python.org/downloads/
     ⚠ IMPORTANT: Check "Add Python to PATH" during installation!

  2. Ollama (for AI models)
     Download: https://ollama.com
     After install, pull a model: ollama pull llama3.2
     
     ⚠ IMPORTANT: First-time Ollama setup:
        - After installing Ollama, open a terminal and run: ollama serve
        - This initializes Ollama's configuration
        - You can close it after the first run - Mimic AI will auto-start it

  3. Docker Desktop (optional, for web search)
     Download: https://docker.com
     Required only if you want to use web search feature

NOTE: The app now automatically configures Ollama for Tauri WebView compatibility.
If you experience Ollama connection issues, restart your computer after first install.

INSTALLATION
--------------------------------------------------------------------------------
  1. Run "Mimic-AI-Setup-v1.0.0.exe"
  2. Follow the installation wizard
  3. Launch Mimic AI from the Start Menu or Desktop shortcut

FIRST RUN
--------------------------------------------------------------------------------
  On first launch, Mimic AI will:
  - Check for Python and required dependencies
  - Start the TTS (Text-to-Speech) backend server
  - Start Ollama (if not already running)
  - Start SearXNG in Docker (if enabled and Docker is available)

  This may take 30-60 seconds. Please be patient!

TROUBLESHOOTING
--------------------------------------------------------------------------------

  Problem: "Python not found" error
  Solution: 
    - Install Python 3.10-3.12 from https://python.org
    - Make sure to check "Add Python to PATH" during installation
    - Restart Mimic AI after installing Python

  Problem: "Ollama not connected"
  Solution:
    - Install Ollama from https://ollama.com
    - Open a terminal and run: ollama pull llama3.2
    - Restart Mimic AI

  Problem: "TTS backend not connected"
  Solution:
    - Ensure Python is installed and in PATH
    - The app will attempt to install dependencies automatically
    - If it fails, manually run: pip install fastapi uvicorn torch torchaudio

  Problem: Web search not working
  Solution:
    - Install Docker Desktop from https://docker.com
    - Start Docker Desktop
    - Enable web search in Mimic AI settings
    - The SearXNG container will be created automatically

  Problem: Black screen or UI not loading
  Solution:
    - Check that all prerequisites are installed
    - Look for error logs at: %APPDATA%\com.mimicai.app\startup.log
    - Try running Mimic AI as Administrator

CHECKING SERVICE STATUS
--------------------------------------------------------------------------------
  You can verify services are running by opening PowerShell and running:

  # Check Ollama
  curl http://localhost:11434/api/tags

  # Check TTS Backend
  curl http://localhost:8000/health

  # Check SearXNG (if Docker is running)
  curl http://localhost:8080/

FEATURES
--------------------------------------------------------------------------------
  ✓ Local AI Inference (via Ollama)
  ✓ Voice Synthesis (Browser TTS & Qwen3-TTS)
  ✓ Persistent Memory (per persona)
  ✓ Web Search (SearXNG via Docker)
  ✓ Wake Word Detection
  ✓ Vision Support (image analysis)
  ✓ Audio Watermarking

SUPPORT
--------------------------------------------------------------------------------
  For issues and feature requests, please use the GitHub issue tracker:
  https://github.com/bmerriott/MIMIC-Multipurpose-Intelligent-Molecular-Information-Catalyst-/issues

LICENSE
--------------------------------------------------------------------------------
  MIT License - See LICENSE file for details

================================================================================
  Thank you for using Mimic AI!
================================================================================
