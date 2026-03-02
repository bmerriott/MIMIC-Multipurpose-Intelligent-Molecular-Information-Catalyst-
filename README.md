<p align="center">
  <img width="256" height="256" alt="icon" src="https://github.com/user-attachments/assets/b035d3cf-8579-47a1-b910-69af3e00218b" />
</p>

# MIMIC AI Desktop Assistant

A privacy-first, AI-powered desktop assistant with voice synthesis, persistent memory, and local model inference. MIMIC runs entirely on your local machine using Ollama for language models and supports **KittenTTS** (8 selectable AI voices) and **Qwen3-TTS** (AI voice cloning) for voice synthesis.

> **Latest Release: v1.2.0** - See [Release Notes](#release-notes) for new features!

## Table of Contents
- [Features](#features)
- [Video Showcase](#video-showcase)
- [Release Notes](#release-notes)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)
- [Privacy & Security](#privacy--security)
- [License](#license)

---

## Video Showcase

[![MIMIC AI Demo](https://img.youtube.com/vi/iltqKnsCTks/maxresdefault.jpg)](https://www.youtube.com/watch?v=iltqKnsCTks)

*Click to watch MIMIC AI in action - Voice cloning, multi-persona support, and real-time conversation*

**🎥 Watch the Demo:** [MIMIC AI Desktop Assistant v1.2.0](https://www.youtube.com/watch?v=iltqKnsCTks)

---

## Features

### Core Capabilities
- **Local AI Inference**: Uses Ollama for running LLMs locally (Llama, Mistral, etc.)
- **Voice Synthesis**: 
  - **KittenTTS** (Default): 8 selectable AI voices (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo) with adjustable speed
  - **Qwen3-TTS**: AI voice cloning with reference audio (~3-6GB VRAM)
- **Persistent Memory**: Conversation memory with per-persona context and file-based storage
- **Web Search**: Privacy-focused SearXNG integration with smart result summarization
- **Wake Word Detection**: Hands-free activation with custom wake words
- **Vision Support**: Image analysis with vision-capable models
- **Audio Watermarking**: All generated audio is invisibly watermarked for identification

### Persona System
- Create multiple AI personas with unique personalities
- **Per-Persona Memory Isolation**: Each persona has their own memory folder (`~/MimicAI/Memories/{persona_id}/`)
- Full conversation history stored per persona
- Voice creation with adjustable parameters (pitch, speed, emotion, etc.)
- Avatar customization with dynamic 3D visuals

### Avatar Personality System
- **Automatic Trait Extraction**: Derives avatar personality traits from personality prompts
- **Voice Parameter Derivation**: Automatically adjusts voice parameters based on personality
- **Dynamic Emotional State**: Avatar tracks emotional states during conversations
- **Height-Based Camera**: Camera intelligently positions based on avatar model height
- **Procedural Vocalizations**: AI-generated non-verbal sounds (giggles, sighs, hums)

### Smart Router System
- **Intent Classification**: Lightweight LLM routes queries to appropriate handlers
- **Search Summarization**: Router automatically summarizes web search results to reduce token usage
- **Minimal System Prompts**: System prompts stay under 500 tokens for faster inference
- **No Emoji Policy**: Responses exclude emojis (TTS reads them aloud)

### Memory Management (Improved in v1.2.0)
- **Per-Persona Storage**: Each persona gets isolated memory folder with:
  - `user_files/` - Uploaded documents
  - `conversations/` - Saved conversation files
  - `history.json` - Full conversation history
- **AI Memories**: Important conversation points automatically extracted
- **Full History**: Complete conversation log with search capability
- **No File Size Limits**: Upload any size file (user-managed local storage)

### Licensing
- **Freemium Model**: Core functionality free, premium assets available in-store
- **Proprietary License**: See [LICENSE.txt](LICENSE.txt) for full EULA
- **Contact**: deadheadstudios@atomicmail.io

---

## Release Notes

### v1.2.0 (Latest)

#### Major Changes
- **🔊 KittenTTS Replaces Browser TTS**: Default TTS is now KittenTTS with 8 selectable voices
  - Female: Bella, Luna, Rosie, Kiki
  - Male: Jasper, Bruno, Hugo, Leo
  - Adjustable speech speed (0.5x - 2.0x)
- **🧠 Improved Memory System**: Per-persona memory folders with full conversation history
- **🎯 Smart Router**: Intent routing and automatic search result summarization
- **📷 Camera Improvements**: Better avatar following and height-based positioning
- **🧹 Clean Console**: Removed debug logging for production builds
- **📝 Minimal System Prompts**: Reduced token usage for faster responses
- **🚫 No Emojis**: Responses exclude emojis for better TTS experience

#### Technical Improvements
- File size limits removed (user-managed storage)
- Search results processed by router to reduce context size
- AI Memories tab shows current persona only
- Voice selection persists across sessions

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
- **KittenTTS**: No VRAM required (CPU-based, 15M-80M parameter models)
- **Qwen3-TTS 0.6B**: ~3GB VRAM  
- **Qwen3-TTS 1.7B**: ~6GB VRAM

---

## Installation

### Desktop Application (Recommended)

Run the installer:

```powershell
.\Mimic AI_1.0.0_x64-setup.exe
```

**What happens:**
- Installs MIMIC to your system
- Creates Start Menu and Desktop shortcuts
- All services run in background (no terminal windows)
- Auto-starts/stops Ollama, TTS backend, and SearXNG

---

## Quick Start

### First Launch
1. Accept the End User License Agreement (EULA)
2. The app will auto-detect Ollama models
3. Create your first persona or use the default "Mimic" assistant

### Creating a Voice (KittenTTS - Default)
1. Go to **Voice Studio** tab
2. Select or create a persona
3. Choose **KittenTTS** engine
4. Select one of 8 voices (Bella, Jasper, Luna, etc.)
5. Adjust speech speed if desired
6. Save to persona

### Creating a Voice (Qwen3-TTS - AI Cloning)
1. Go to **Voice Studio** tab
2. Select or create a persona
3. Choose **Qwen3-TTS** engine
4. Upload reference audio or record your voice
5. Adjust voice parameters (pitch, speed, emotion)
6. Click "Create Voice" and save to persona

### Using Web Search
1. Enable "Web Search" in Settings
2. Docker Desktop will auto-start if installed
3. SearXNG container will be created and started
4. Ask questions like "What's the weather today?"
5. Router automatically summarizes results

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
│  │  KittenTTS  │ │  Qwen3-TTS  │ │  Legal Watermarker  │   │
│  │   Engine    │ │   Engine    │ │    (AudioSeal)      │   │
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
```

### Data Storage (Per-Persona in v1.2.0)
```
┌─────────────────────────────────────────────────────────────┐
│                    Data Persistence                          │
├─────────────────────────────────────────────────────────────┤
│  localStorage (Browser)                                     │
│  ├── mimic_personas       - Persona configurations          │
│  ├── mimic_settings       - App settings                    │
│  └── mimic_app_state      - Current state                   │
│                                                             │
│  IndexedDB (Browser)                                        │
│  └── Voice audio data (large binary data)                   │
│                                                             │
│  File System (~/MimicAI/Memories/{persona_id}/)             │
│  ├── user_files/          - Uploaded documents              │
│  ├── conversations/       - Conversation exports            │
│  └── history.json         - Full conversation history       │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Settings
Settings are stored in `localStorage` under `mimic_settings`:

```json
{
  "ollama_url": "http://localhost:11434",
  "default_model": "llama3.2",
  "vision_model": "bakllava",
  "tts_engine": "kitten",
  "kitten_voice": "Bella",
  "kitten_model": "nano",
  "kitten_speed": 1.0,
  "voice_volume": 1.0,
  "enable_memory": true,
  "enable_web_search": false
}
```

---

## Usage Guide

### Chat Interface
1. **Text Input**: Type messages and press Enter
2. **Voice Input**: Click microphone icon or use wake word
3. **File Attachment**: Click paperclip to upload files (no size limit)
4. **Image Upload**: Click camera icon for vision analysis

### Persona Management
1. Go to **Personas** tab
2. Click "New Persona" to create
3. Configure name, personality, wake words, avatar
4. Each persona gets isolated memory storage

### Memory Manager
1. Go to **Memory Manager**
2. **Memory Files**: Manage uploaded documents (per-persona)
3. **AI Memories**: View important extracted memories
4. **Full History**: Complete conversation log

### Voice Studio
1. Select persona to voice
2. Choose engine:
   - **KittenTTS**: Select from 8 voices, adjust speed
   - **Qwen3-TTS**: Upload reference audio for cloning
3. Adjust parameters (pitch, speed, emotion)
4. Save to persona

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
- Restart the application
- Check firewall settings

**No audio output**
- Check system volume
- Verify voice is saved to persona
- Try different TTS engine (KittenTTS requires no setup)

**Web search not working**
- Ensure Docker Desktop is installed
- Check if SearXNG container is running: `docker ps`
- Verify port 8080 is not in use

---

## Privacy & Security

### Data Locality
- **All AI processing runs locally** - No cloud services required
- **Voice data never leaves your machine**
- **Chat history stored locally** in per-persona folders

### Audio Watermarking
All AI-generated audio contains invisible watermarks for:
- **Transparency**: Identifies AI-generated content
- **Legal protection**: Forensic evidence of origin
- **Misuse prevention**: Deters fraudulent use

### Per-Persona Memory Isolation
Each persona's data is stored in separate folders:
- No cross-contamination between personas
- Easy backup/restore per persona
- Delete one persona without affecting others

---

## Media Gallery

### Adding Demo Media to README

You can embed demo media in this README using GitHub's file hosting:

#### Option 1: GitHub Repository Media (Recommended)
1. Create a `media/` folder in your repository
2. Add GIFs/videos to the folder
3. Reference them in README:
   ```markdown
   ![Demo](media/demo.gif)
   ```

#### Option 2: GitHub Issue Attachments
1. Create a new GitHub issue
2. Drag and drop your GIF/video into the issue
3. Copy the generated URL
4. Use that URL in your README

#### Option 3: GitHub Repository Assets
1. Go to your repository's "Releases" section
2. Create a new release
3. Upload GIFs/videos as release assets
4. Link to them in README

**Recommended Format**: 
- **GIFs**: Use for short demos (under 10 seconds) - autoplay in README
- **Videos**: Use MP4 for longer demos - users click to play

---

## Attribution

### Character Model
- **Default Avatar**: PeePhanthong (VRoid Hub)

### Animations
- **VRoid Project** by pixiv Inc.
- **八ツ橋まろんのお店** (Yatsuhashi Maron's Shop)
- **Kannaku @ Nekokoya**

See [ATTRIBUTION.md](ATTRIBUTION.md) for full credits and license terms.

---

## License

**Proprietary License - See [LICENSE.txt](LICENSE.txt)**

Copyright (c) 2026 Dead Head Studios. All Rights Reserved.

This software is provided under a Proprietary License. The Software is licensed, not sold. See the EULA for complete terms including:
- Freemium model with optional premium content
- Usage restrictions
- Intellectual property rights
- Termination conditions

**Contact:** deadheadstudios@atomicmail.io

---

## Support

- **Email**: deadheadstudios@atomicmail.io
- **Web**: https://github.com/bmerriott/MIMIC-Multipurpose-Intelligent-Molecular-Information-Catalyst-
- **Documentation**: See [QUICKSTART.md](QUICKSTART.md)
