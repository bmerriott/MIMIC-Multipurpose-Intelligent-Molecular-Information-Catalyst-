# Release Notes - Mimic AI v1.1.0

**Release Date:** February 22, 2026  
**Version:** 1.1.0

---

## 🎉 What's New

### VRM Avatar Support (New!)
- **Full 3D Avatar Support**: Load and display VRM models from VRoid Hub or custom creations
- **Expression & Lip-Sync**: VRM avatars show expressions and lip-sync to speech
- **Bundled Default Avatar**: Ships with PeePhanthong's character model ready to use
- **Height-Based Camera**: Camera intelligently positions based on avatar model height for optimal viewing

### VRMA Animation Support (New!)
- **Animation Library**: Play VRMA animation files on VRM avatars
- **Bundled Animations**: Includes greetings, poses, dances, and idle animations
- **Context-Aware**: Avatars automatically play appropriate animations based on conversation

### Avatar-Persona Memory Integration (New!)
- **Shared Context**: Avatars now access their persona's contextual memory
- **Emotional Reflection**: Avatar expressions reflect the persona's emotional state
- **Personality-Driven**: Avatar behavior influenced by persona's personality traits

### Bundled Default Persona
- **Pre-configured VRM Avatar**: Ships with character model ready to use
- **Default Voice Included**: Bundled Qwen3 voice reference for immediate voice synthesis
- **Automatic Setup**: Assets copied to user data on first launch - no manual configuration needed
- **Fresh Install Experience**: New users get a fully functional 3D avatar out of the box

### Avatar Personality System
- **Automatic Trait Extraction**: Derives personality traits (energy, playfulness, expressiveness, curiosity, empathy, formality) from personality prompts
- **Voice Parameter Derivation**: Automatically adjusts voice parameters based on extracted personality traits
- **Dynamic Emotional State**: Avatar tracks and displays emotional states during conversations

### Persona Learning System
- **Interaction Tracking**: Records conversation history and user preferences per persona
- **Emote Learning**: Tracks preferred emotes and animations based on user feedback
- **Relationship Metrics**: Calculates rapport, familiarity, and trust scores over time
- **Favorite Topics**: Automatically identifies and remembers user's favorite conversation topics
- **Emotional History**: Maintains emotional state history for more natural interactions

### Procedural Vocalizations
- AI-generated non-verbal sounds (giggles, sighs, hums) using Qwen3 TTS
- Cached in IndexedDB (max 10 per persona)
- Adds natural expressiveness to conversations

---

## 🔧 Improvements

### Camera & Visuals
- **Fixed Camera Snapping**: Avatar no longer zooms in too close after loading (base distance increased to 4.5 units)
- **Real-Time Color Updates**: Blob avatar colors update instantly when using color wheel

### Audio & Voice
- **Fixed Audio Playback**: Resolved AudioContext suspension issue causing silent audio
- **Longer Voice Prompts**: Increased silence delay from 3.5s to 5s (allows natural pauses)
- **Extended Command Timeout**: Increased from 8s to 10s for longer speech

### Licensing
- **7-Day Free Trial**: Full functionality for 7 days from first launch
- **Subscription Model**: $5/month via Patreon for continued use after trial
- **Support Development**: Subscribe to help fund ongoing development

---

## 🐛 Bug Fixes

| Issue | Fix |
|-------|-----|
| Camera snaps too close on load | Camera only adjusts if too close, respects user's zoom |
| No audio from Browser/Qwen TTS | Added AudioContext.resume() for suspended contexts |
| Voice prompts cut off early | Increased silence detection to 5 seconds |
| Blob colors don't update live | Real-time uniform updates in shader |

---

## 🎨 Attribution

### Character Model
- **Default Avatar**: PeePhanthong (VRoid Hub)

### Animations
- **VRoid Project** by pixiv Inc.
- **八ツ橋まろんのお店** (Yatsuhashi Maron's Shop)
- **Kannaku @ Nekokoya**

See [ATTRIBUTION.md](ATTRIBUTION.md) for full credits and license terms.

---

## 📦 Assets

### Bundled with Release
- Default VRM avatar (`personas/default/avatar.vrm`)
- Default voice reference (`personas/default/voice.wav`)
- Voice configuration (`personas/default/voice.json`)
- VRMA animation library

### Default Persona
The default "Mimic" persona now ships with:
- VRM avatar loaded by default
- Qwen3 voice configured
- Can be customized or replaced by user

---

## 🚀 Installation

### Windows
1. Download `Mimic-AI-v1.1.0.zip` from GitHub Releases
2. Extract the ZIP file
3. Run `Mimic-AI-Setup.exe` to install
4. Launch from Start Menu or Desktop

### From Source
See [QUICKSTART.md](QUICKSTART.md) for development setup.

---

## 💻 System Requirements

- **OS**: Windows 10/11, macOS 12+, or Linux
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space
- **Python**: 3.10-3.12 (3.13+ not supported)
- **Ollama**: Latest version from [ollama.com](https://ollama.com)

---

## 🔐 Privacy & Security

- All AI processing runs locally
- Voice data never leaves your machine
- Chat history stored locally
- AI-generated audio contains invisible watermarks
- No telemetry or analytics

---

## 📝 License

Proprietary License with 7-Day Trial

Copyright (c) 2026 Mimic AI. All Rights Reserved.

See [LICENSE.txt](LICENSE.txt) for full terms.

---

## 🙏 Support

- **Patreon**: https://www.patreon.com/c/MimicAIDigitalAssistant
- **Issues**: GitHub issue tracker
- **Documentation**: [README.md](README.md) | [QUICKSTART.md](QUICKSTART.md)

---

## 🔄 Previous Versions

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

---

**Full Changelog**: Compare with v1.0.0
