#!/bin/bash

echo "======================================"
echo "     Mimic AI Assistant Launcher"
echo "======================================"
echo

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js is not installed"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo "[OK] Node.js found: $(node --version)"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[WARNING] Python not found. TTS backend will not be available."
    echo "For voice cloning, install Python 3.10+ from https://python.org/"
    echo
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[WARNING] Ollama is not running!"
    echo "Please start Ollama first:"
    echo "  1. Install from https://ollama.ai"
    echo "  2. Pull a model: ollama pull llama3.2"
    echo "  3. Start Ollama"
    echo
    read -p "Continue without Ollama? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "[OK] Ollama is running"
fi

echo
echo "======================================"
echo "Starting Mimic AI Assistant..."
echo "======================================"
echo

# Start TTS backend in background (if Python is available)
if command -v python3 &> /dev/null || command -v python &> /dev/null; then
    echo "Starting TTS Backend..."
    PYTHON_CMD=$(command -v python3 || command -v python)
    cd backend && $PYTHON_CMD tts_server_simple.py &
    TTS_PID=$!
    sleep 3
fi

# Function to cleanup on exit
cleanup() {
    echo
    echo "Shutting down..."
    if [ -n "$TTS_PID" ]; then
        kill $TTS_PID 2>/dev/null
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start frontend
echo "Starting Frontend..."
echo
npm run dev

# Cleanup
cleanup
