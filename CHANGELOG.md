# Changelog

All notable changes to Mimic AI Desktop Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-02-28

### Added
- **KittenTTS Integration**: New default voice engine with 8 selectable voices
  - Female voices: Bella, Luna, Rosie, Kiki
  - Male voices: Jasper, Bruno, Hugo, Leo
  - Adjustable speech speed (0.5x to 2.0x)
  - No VRAM required, CPU-based inference
- **Smart Router System**: Intent classification using lightweight LLM
  - Routes queries to appropriate processing pipeline
  - Search result summarization (60-80% token reduction)
  - Style guidance only when confidence > 60%
- **Per-Persona Memory Isolation**: Each persona has dedicated storage
  - Folder structure: `~/MimicAI/Memories/{persona_id}/`
  - user_files/, conversations/, history.json
  - Full conversation history tracking
- **Image Attachment Support**: Paperclip button now supports image upload
  - JPG, PNG, GIF, WebP, BMP, SVG formats
  - 10MB max size for images
  - Vision model analyzes attached images
- **Tool Confirmation System**: Explicit user approval for write/delete operations
  - Modal shows operation preview before execution
  - Yes/No confirmation with content preview
- **Agent Self-Awareness System**: Personas know their capabilities without verbosity
  - Implicit awareness of MIMIC AI environment
  - No "As an AI..." meta-commentary
  - Natural responses without announcing tools

### Changed
- **Default TTS Engine**: Browser TTS → KittenTTS
- **System Prompts**: Reduced from 5000+ tokens to under 500
- **Memory Storage**: Single shared folder → Per-persona isolated folders
- **Console Output**: Removed 50+ debug console.log statements
- **Search Results**: Moved from system prompt to user message
- **AI Memories Tab**: Shows current persona only (was multi-persona selector)
- **Version Display**: Updated to v1.2.0 in System Info panel

### Removed
- Browser TTS engine (completely replaced by KittenTTS)
- Debug console logging in production builds
- File size upload limits (user-managed local storage)
- Emoji usage in AI responses (TTS compatibility)

### Fixed
- **Image Attachment**: Paperclip button now properly supports image files
- **Voice Selection**: KittenTTS voices properly persist across sessions
- **Memory Manager**: Per-persona memory properly isolated

### Technical
- Token usage reduced by ~94% for system prompts
- Search result summarization for faster inference
- Clean production builds with no debug output
- Router-guided style application (confidence threshold)

## [1.1.0] - 2026-02-22

### Added
- **VRM Avatar Support**: Full 3D avatar support using VRoid models
  - Load custom VRM files via VRoid Hub or custom models
  - Automatic VRM loading with bundled default avatar
  - Expression and lip-sync support
- **VRMA Animation Support**: VRM animation file support
  - Play VRMA animations on VRM avatars
  - Includes bundled animation library (greetings, poses, dances, idle)
  - Auto-play animations based on conversation context
- **Avatar-Persona Memory Integration**: Avatars now use contextual memory from their persona
  - Avatar expressions reflect persona's emotional state
  - Personality traits influence avatar behavior
  - Conversational memory shared between avatar and persona
- **Bundled Default Persona**: Ships with pre-configured VRM avatar and Qwen3 voice
  - Default avatar automatically loaded on fresh installs
  - Default voice included for immediate Qwen3 TTS use
  - Assets copied to user data on first launch
- **Avatar Personality System**: Automatic trait extraction from personality prompts
  - Derives energy, playfulness, expressiveness, curiosity, empathy, formality
  - Maps traits to voice parameters (pitch, speed, warmth)
- **Persona Learning System**: Tracks user interactions and builds relationship memory
  - Records conversation history and sentiment
  - Learns preferred emotes and animations
  - Calculates rapport, familiarity, and trust scores
  - Stores favorite topics and inside jokes
- **Procedural Vocalizations**: AI-generated non-verbal sounds (giggles, sighs, hums)
  - Uses Qwen3 TTS for generation
  - IndexedDB caching (max 10 per persona)
- **Height-Based Camera**: Camera positions based on avatar model height
  - Automatically adjusts to show full avatar
  - Prevents zoom-in issues with different model sizes
- **Improved Lip Sync**: Enhanced mouth animation with volume-based detection

### Changed
- **License**: Changed from MIT to Proprietary with 7-day trial
- **Camera Position**: Adjusted default zoom for better avatar visibility
- **Patreon URL**: Updated to https://www.patreon.com/c/MimicAIDigitalAssistant
- **Default Avatar Type**: Fresh installs now use VRM avatar by default (was abstract blob)

### Fixed
- **Audio Playback**: Fixed AudioContext suspension issue causing no sound
- **Camera Snapping**: Fixed camera zooming in too close after model load
- **VRM Positioning**: Better default positioning for VRM avatars
- **Voice Reference Text**: Corrected default voice reference text to match bundled audio

### Attribution
- **Animations**: VRoid Project by pixiv Inc.
- **Character Model**: PeePhanthong (VRoid Hub)
- **Animation Store**: 八ツ橋まろんのお店, Kannaku @ Nekokoya

## [1.0.0] - 2026-02-20

### Added
- Initial release of Mimic AI Desktop Assistant
- Local AI inference via Ollama
- Voice synthesis (Browser TTS and Qwen3-TTS)
- Persistent memory system
- Web search via SearXNG
- Wake word detection
- Vision support for image analysis
- Abstract avatar (animated blob/sphere)
- Persona system with customizable personalities
- Voice creation and cloning
- Audio watermarking for identification

### Technical
- Tauri (Rust) + React + TypeScript stack
- Three.js + @react-three/fiber for 3D
- IndexedDB for voice data storage
- LocalStorage for personas and settings
