# Mimic AI Desktop Assistant

A privacy-first, AI-powered desktop assistant with voice synthesis, persistent memory, and local model inference. Mimic AI runs entirely on your local machine using Ollama for language models and supports Browser TTS and Qwen3-TTS for voice synthesis.

## Table of Contents
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Data Flow](#data-flow)
- [Component Interaction](#component-interaction)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)
- [Privacy & Security](#privacy--security)

---

## Features

### Core Capabilities
- **Local AI Inference**: Uses Ollama for running LLMs locally (Llama, Mistral, etc.)
- **Voice Synthesis**: Browser TTS (no setup) and Qwen3-TTS (AI voice cloning)
- **Persistent Memory**: Conversation memory with per-persona context
- **Web Search**: Privacy-focused SearXNG integration
- **Wake Word Detection**: Hands-free activation with custom wake words
- **Vision Support**: Image analysis with vision-capable models
- **Audio Watermarking**: All generated audio is invisibly watermarked for identification

### Persona System
- Create multiple AI personas with unique personalities
- Each persona has independent memory and voice profiles
- Voice creation with adjustable parameters (pitch, speed, emotion, etc.)
- Avatar customization with dynamic 3D visuals

### Memory Management
- **File Memories**: Upload text files for the AI to reference
- **Conversation Memories**: Automatic extraction of important conversation points
- **Memory Controls**: Per-persona memory management with delete/recall capabilities
- **Summary Generation**: Automatic conversation summarization

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 12+, or Linux
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space
- **GPU**: Optional but recommended for faster inference

### Software Dependencies
- **Python**: 3.10, 3.11, or 3.12 (3.13+ not supported due to PyTorch compatibility)
- **Node.js**: 18.x or 20.x LTS
- **Ollama**: Latest version from [ollama.com](https://ollama.com)
- **Docker Desktop**: Optional, for SearXNG web search

### VRAM Requirements (Voice Synthesis)
- Browser TTS: No VRAM required (uses system voice)
- Qwen3-TTS 0.6B: ~3GB VRAM  
- Qwen3-TTS 1.7B: ~6GB VRAM

---

## Installation

### Step 1: Install Prerequisites

1. **Install Python 3.11** (recommended):
   ```bash
   # Download from https://python.org/downloads/
   # During installation, check "Add Python to PATH"
   ```

2. **Install Node.js**:
   ```bash
   # Download from https://nodejs.org (LTS version)
   # Check "Add to PATH" during installation
   ```

3. **Install Ollama**:
   ```bash
   # Download from https://ollama.com
   # Pull a model: ollama pull llama3.2
   ```

4. **Install Docker Desktop** (optional, for web search):
   ```bash
   # Download from https://docker.com
   ```

### Step 2: Choose Installation Method

#### Option A: Desktop Application (Recommended)

Run the installer in the root folder:

```powershell
.\Mimic AI Setup.exe
```

Or double-click the installer file.

**What happens:**
- Installs Mimic AI to your system
- Creates Start Menu and Desktop shortcuts
- All services run in background (no terminal windows)
- Auto-starts/stops Ollama, TTS backend, and SearXNG

#### Option B: Web Browser Mode (Development)

```bash
# Clone the repository
git clone <repository-url>
cd "Mimic AI Desktop Assistant - Copy"

# Install frontend dependencies
cd app
npm install
cd ..

# Launch (uses browser)
.\launch-mimic.bat
```

The launcher will:
1. Check Python version (3.10-3.12 required)
2. Install Python dependencies (fastapi, uvicorn, torch, etc.)
3. Install Node dependencies if needed
4. Start Ollama server
5. Start Docker Desktop and SearXNG (if available)
6. Launch the TTS backend and frontend
7. Open your browser

---

## Quick Start

### First Launch
1. Accept the terms and conditions
2. The app will auto-detect Ollama models
3. Create your first persona or use the default "Mimic" assistant

### Creating a Voice
1. Go to **Voice Studio** tab
2. Select or create a persona
3. Choose TTS Engine (Browser for simplicity, Qwen3 for AI voice cloning)
4. Upload reference audio or record your voice
5. Adjust voice parameters (pitch, speed, emotion)
6. Click "Create Voice" and save to persona

### Using Web Search
1. Enable "Web Search" in Settings
2. Docker Desktop will auto-start if installed
3. SearXNG container will be created and started
4. Ask questions like "What's the latest news?"

---

## Architecture Overview

### System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │  Chat    │ │ Settings │ │  Voice   │ │ Personas │       │
│  │  Panel   │ │  Panel   │ │  Studio  │ │ Manager  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                         │                                    │
│              ┌──────────┴──────────┐                        │
│              │     Zustand Store    │                        │
│              │  (State Management)  │                        │
│              └──────────┬──────────┘                        │
└─────────────────────────┼───────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────┐
│              TTS Backend (Python FastAPI)                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Browser TTS │ │  Qwen3-TTS  │ │  Legal Watermarker  │   │
│  │  Engine     │ │   Engine    │ │    (AudioSeal)      │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│                         │                                    │
│              ┌──────────┴──────────┐                        │
│              │   Unified TTS Server  │                      │
│              │     (Port 8000)       │                      │
│              └───────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────┐
│              Ollama Server (Local LLM)                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │   Llama     │ │   Mistral   │ │   Vision Models     │   │
│  │   Models    │ │   Models    │ │  (LLaVA, BakLLaVA)  │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│                         │                                    │
│              Port 11434 (Default)                           │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────┐
│              Optional Services                               │
│  ┌─────────────────┐ ┌─────────────────────────────────────┐│
│  │  Docker Desktop │ │           SearXNG                   ││
│  │   (Web Search)  │ │   (Privacy Search Aggregator)       ││
│  │                 │ │         Port 8080                   ││
│  └─────────────────┘ └─────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Data Storage
```
┌─────────────────────────────────────────────────────────────┐
│                    Data Persistence                          │
├─────────────────────────────────────────────────────────────┤
│  localStorage (Browser)                                     │
│  ├── mimic_personas       - Persona configurations          │
│  ├── mimic_settings       - App settings                    │
│  ├── mimic_app_state      - Current state                   │
│  ├── mimic_voice_tuning   - Per-persona voice params        │
│  └── mimic_terms_accepted - Legal acceptance                │
│                                                             │
│  IndexedDB (Browser)                                        │
│  └── Voice audio data (large binary data)                   │
│                                                             │
│  File System (~/MimicAI/Memories/)                          │
│  └── Memory files (.txt, .md, .json)                        │
│                                                             │
│  Tauri FS (Desktop App Only)                                │
│  └── Voice data in AppData (Windows) or ~/Library (Mac)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Chat Message Flow
```
User Input
    ↓
┌─────────────────────────────────────────────┐
│  1. Query Router (Keyword Analysis)         │
│     - Determines: VISION / WEB_SEARCH /     │
│       CODE / REASONING / GENERAL            │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  2. Context Assembly                        │
│     - System prompt (persona rules)         │
│     - Memory files (if any)                 │
│     - Web search results (if enabled)       │
│     - Image context (if uploaded)           │
│     - Conversation history                  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  3. Ollama API Request                      │
│     POST /api/generate                      │
│     - model: selected model                 │
│     - prompt: assembled context             │
│     - stream: true                          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  4. Response Processing                     │
│     - Stream chunks to UI                   │
│     - Memory extraction (if enabled)        │
│     - TTS generation (if voice enabled)     │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  5. Audio Output (if TTS enabled)           │
│     - Persona voice profile loaded          │
│     - TTS engine called (Browser/Qwen3)   │
│     - Audio watermark embedded              │
│     - Audio played via Web Audio API        │
└─────────────────────────────────────────────┘
    ↓
User Hears Response
```

### 2. Voice Creation Flow
```
User Uploads Reference Audio
    ↓
┌─────────────────────────────────────────────┐
│  1. Audio Processing                        │
│     - Convert to WAV if needed              │
│     - Transcribe with Puter.js (optional)   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  2. TTS Backend Request                     │
│     POST /api/voice/create                  │
│     - reference_audio: base64 WAV           │
│     - reference_text: transcription         │
│     - engine: browser or qwen3            │
│     - voice_params: pitch, speed, etc.      │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  3. Voice Synthesis                         │
│     - Load reference audio                  │
│     - Extract voice characteristics         │
│     - Generate target text speech           │
│     - Apply audio watermark                 │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  4. Storage                                 │
│     - Audio saved to IndexedDB              │
│     - Voice config saved to persona         │
│     - localStorage updated                  │
└─────────────────────────────────────────────┘
    ↓
Voice Ready for Use
```

### 3. Memory System Flow
```
Conversation Happens
    ↓
┌─────────────────────────────────────────────┐
│  1. Memory Extraction                       │
│     - LLM analyzes conversation             │
│     - Extracts key facts (importance > 0.5) │
│     - Stores in persona.short_term[]        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  2. Memory Summarization (every N msgs)     │
│     - When threshold reached (default 20)   │
│     - LLM summarizes conversation           │
│     - Stores in persona.memory.summary      │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  3. Memory Retrieval                        │
│     - On new message, relevant memories     │
│       are retrieved via semantic search     │
│     - Added to context window               │
└─────────────────────────────────────────────┘
```

---

## Component Interaction

### Frontend Components

#### 1. ChatPanel.tsx
- **Purpose**: Main chat interface
- **Inputs**: User text, voice input, file attachments, images
- **Outputs**: AI responses, TTS audio
- **Integrates with**: 
  - `ollama.ts` (LLM API)
  - `tts.ts` (voice synthesis)
  - `memoryTools.ts` (file operations)
  - `queryRouter.ts` (query classification)

#### 2. VoiceCreator.tsx
- **Purpose**: Voice creation and watermark detection
- **Features**:
  - Reference audio upload/recording
  - Voice parameter tuning (pitch, speed, emotion)
  - TTS engine selection (Browser/Qwen3)
  - Audio watermark detection
- **Integrates with**:
  - `tts.ts` (voice creation API)
  - `audioEffects.ts` (real-time preview)
  - `puter.ts` (transcription)

#### 3. MemoryManager.tsx
- **Purpose**: Browse and manage memories
- **Features**:
  - File memory browsing
  - Conversation memory viewing
  - Per-persona memory filtering
  - Delete individual/all memories
- **Integrates with**:
  - `memoryTools.ts` (file CRUD)
  - Store (persona memory access)

#### 4. SettingsPanel.tsx
- **Purpose**: App configuration
- **Settings**:
  - Ollama/TTS backend URLs
  - Voice settings (volume, rate, TTS mode)
  - Memory settings (thresholds, summarization)
  - Wake word sensitivity
  - Microphone selection

### Backend Components

#### 1. tts_server_unified.py
- **Purpose**: Unified TTS backend
- **Endpoints**:
  - `POST /voice/create` - Create voice from reference
  - `POST /tts/synthesize` - Text-to-speech
  - `POST /watermark/detect` - Detect AI watermark
  - `GET /health` - Health check
- **Engines**:
  - Browser TTS: Uses system voice, no setup required
  - Qwen3-TTS: AI voice cloning, requires reference audio

#### 2. watermarker.py
- **Purpose**: Multi-layer audio watermarking
- **Layers**:
  1. Spread-spectrum (survives compression)
  2. Echo-based (survives filtering)
  3. Phase coding (survives amplitude changes)
- **Usage**: All generated audio is automatically watermarked

#### 3. detect_watermark.py
- **Purpose**: Standalone watermark detection tool
- **Usage**: `python detect_watermark.py audio.wav`
- **Returns**: Detection result with confidence score

### Service Layer

#### 1. ollama.ts
- **Purpose**: Ollama API client
- **Methods**:
  - `generate()` - Text generation
  - `listModels()` - Get available models
  - `checkConnection()` - Health check
  - `filterVisionModels()` - Filter vision-capable models

#### 2. tts.ts
- **Purpose**: TTS API client
- **Methods**:
  - `createVoice()` - Voice creation
  - `synthesize()` - Text-to-speech
  - `checkConnection()` - Health check
  - `getEngineStatus()` - Get available engines

#### 3. searxng.ts
- **Purpose**: Web search integration
- **Features**:
  - Privacy-first search aggregation
  - No API keys required
  - Results injected into context

#### 4. memoryTools.ts
- **Purpose**: File-based memory operations
- **Methods**:
  - `listMemories()` - List memory files
  - `readMemory()` - Read file content
  - `writeMemory()` - Create/update file
  - `searchMemories()` - Search file contents

### State Management

#### Zustand Store (store/index.ts)
```typescript
// Key state slices:
- appState: Connection status, current model
- personas: Persona configurations and memory
- settings: User preferences
- messages: Chat history
- voiceTuning: Per-persona voice parameters
```

---

## Configuration

### Environment Variables
Create a `.env` file in `app/` directory:

```env
# Ollama Configuration
VITE_OLLAMA_URL=http://localhost:11434

# TTS Backend
VITE_TTS_BACKEND_URL=http://localhost:8000

# SearXNG (Web Search)
VITE_SEARXNG_URL=http://localhost:8080
```

### Settings File
Settings are stored in `localStorage` under `mimic_settings`:

```json
{
  "ollama_url": "http://localhost:11434",
  "default_model": "llama3.2",
  "vision_model": "bakllava",
  "tts_backend_url": "http://localhost:8000",
  "tts_engine": "browser",
  "qwen3_model_size": "0.6B",
  "qwen3_flash_attention": true,
  "voice_volume": 1.0,
  "speech_rate": 1.0,
  "enable_memory": true,
  "memory_importance_threshold": 0.5,
  "enable_web_search": false,
  "wake_word_sensitivity": 0.7,
  "auto_listen": false
}
```

---

## Usage Guide

### Chat Interface
1. **Text Input**: Type messages and press Enter
2. **Voice Input**: Click microphone icon or use wake word
3. **File Attachment**: Click paperclip to upload files
4. **Image Upload**: Click camera icon for vision analysis

### Persona Management
1. Go to **Personas** tab
2. Click "New Persona" to create
3. Configure:
   - Name and description
   - Personality prompt
   - Wake words
   - Avatar colors
   - Voice settings

### Memory Files
1. Go to **Memory Manager**
2. **Memory Files** tab: Manage uploaded documents
3. **Conversation Memories** tab: View extracted memories
4. Use "New Memory File" to add knowledge

### Voice Studio
1. Select persona to voice
2. Upload reference audio or record
3. Adjust parameters:
   - **Instant Effects**: Pitch, speed (no regeneration)
   - **Voice Character**: Warmth, clarity, expressiveness
   - **Speech**: Emotion, emphasis, pauses
4. Click "Create Voice"
5. Save to persona

### Watermark Detection
1. Go to **Voice Studio** → **Watermark Detection** tab
2. Upload audio file
3. View detection results:
   - Whether AI watermark is present
   - Confidence score
   - Layer analysis

---

## Troubleshooting

### Common Issues

**"Python not found" error**
- Install Python 3.10-3.12 from python.org
- Ensure "Add to PATH" is checked

**"Ollama not connected"**
- Verify Ollama is running: `ollama serve`
- Check Ollama URL in Settings
- Ensure at least one model is pulled: `ollama pull llama3.2`

**"TTS backend not connected"**
- Check if backend is running on port 8000
- Restart with `launch-mimic.bat`
- Check firewall settings

**No audio output**
- Check system volume
- Verify voice is saved to persona
- Try different TTS engine

**Web search not working**
- Ensure Docker Desktop is installed
- Check if SearXNG container is running: `docker ps`
- Verify port 8080 is not in use

### Debug Mode
Enable debug logging in browser console:
```javascript
localStorage.setItem('mimic_debug', 'true');
location.reload();
```

---

## Privacy & Security

### Data Locality
- **All AI processing runs locally** - No cloud services required
- **Voice data never leaves your machine**
- **Chat history stored locally** in browser

### Audio Watermarking
All AI-generated audio contains invisible watermarks for:
- **Transparency**: Identifies AI-generated content
- **Legal protection**: Forensic evidence of origin
- **Misuse prevention**: Deters fraudulent use

### File Security
- Memory files stored in user directory (`~/MimicAI/Memories/`)
- Path traversal attacks prevented
- Only allowed extensions permitted (.txt, .md, .json, etc.)
- File size limits enforced (10MB max)

### Consent Requirements
- First-launch consent dialog for AI disclosure
- Voice creation requires explicit consent
- Prohibited uses clearly stated (no fraud/impersonation)

---

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Support

For issues and feature requests, please use the GitHub issue tracker.
