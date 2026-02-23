# Changelog

All notable changes to Mimic AI Desktop Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
