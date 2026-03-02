# MIMIC AI v1.2.0 Release Notes

**Release Date:** February 28, 2026  
**Codename:** "Smart Voice"  
**Licensor:** Dead Head Studios

---

## 🎉 Major Features

### 🔊 KittenTTS - New Default Voice Engine
- **Replaces Browser TTS** as the default voice engine
- **8 Selectable AI Voices:**
  - Female: Bella, Luna, Rosie, Kiki
  - Male: Jasper, Bruno, Hugo, Leo
- **Adjustable Speech Speed:** 0.5x to 2.0x
- **No VRAM Required:** CPU-based inference (15M-80M parameter models)
- **Fast and Reliable:** No setup or dependencies required

### 🧠 Improved Memory System
- **Per-Persona Memory Isolation:** Each persona has their own folder:
  ```
  ~/MimicAI/Memories/{persona_id}/
  ├── user_files/          # Uploaded documents
  ├── conversations/       # Conversation exports
  └── history.json         # Full conversation history
  ```
- **No File Size Limits:** Upload any size file (user-managed local storage)
- **Full History Tracking:** Complete conversation log with search capability
- **Memory Manager Updates:**
  - AI Memories tab shows current persona only
  - Full History tab with persona selector
  - File-based storage for persistence

### 🎯 Smart Router System
- **Intent Classification:** Lightweight LLM (Qwen3:0.6B) routes queries
- **Search Result Summarization:** Automatically compresses web search results
  - Reduces token usage by 60-80%
  - Faster inference times
  - More focused responses
- **Minimal System Prompts:** Under 500 tokens for faster processing

### 📷 Image Attachment Support
- **Paperclip Button:** Now supports image upload
- **Supported Formats:** JPG, PNG, GIF, WebP, BMP, SVG
- **Vision Analysis:** Attached images analyzed by vision model
- **10MB Max Size:** For images (5MB for documents)

### 🎥 Camera & Animation Improvements
- **Improved Camera Follow:** Smoother tracking during avatar movement
- **Height-Based Positioning:** Camera adjusts based on avatar model height
- **Enhanced Lip Sync:** Better syllable detection and mouth movement
- **18 Emote Animations:** Full set of idle and expressive animations

### 💳 Licensing Changes
- **Subscription Removed:** No longer required for full access
- **Freemium Model:** Core features free, premium assets available via in-app purchases
- **Proprietary License:** See LICENSE.txt for full EULA

---

## 🔧 Technical Improvements

### Console & Debugging
- **Clean Console:** Removed 50+ debug console.log statements
- **Production Ready:** No debug output in release builds
- **Error Handling:** User-facing toast notifications preserved

### System Prompt Optimization
- **Minimal Prompts:** Reduced from 5000+ tokens to under 500
- **No Emoji Policy:** Responses exclude emojis (TTS reads them aloud)
- **Search Results in User Message:** Keeps system prompt clean
- **Router-Guided Style:** Only adds style guidance when confidence > 60%

### Voice & TTS
- **Consistent Voice Selection:** KittenTTS voice persists across sessions
- **Speed Control:** Per-persona speech speed settings
- **Qwen3 Fallback:** Falls back to KittenTTS if reference audio missing

---

## 📋 Changes from v1.1.0

### Added
- KittenTTS integration with 8 selectable voices
- Smart router for intent classification
- Search result summarization
- Per-persona memory folders
- Full conversation history tracking
- Image attachment support (JPG, PNG, GIF, WebP, BMP, SVG)
- Speech speed control for KittenTTS
- Tool confirmation system for write/delete operations
- Agent self-awareness system
- "NO EMOJIS" system prompt rule

### Changed
- **Default TTS:** Browser TTS → KittenTTS
- **Memory Storage:** Single folder → Per-persona folders
- **System Prompts:** Bloated → Minimal
- **Console:** Debug logging → Clean production output
- **AI Memories Tab:** Multi-persona selector → Current persona only
- **Version Display:** Updated to v1.2.0 in System Info

### Removed
- Browser TTS engine (completely removed)
- Debug console logging
- File size upload limits
- Emoji usage in responses

---

## 🚀 Performance Improvements

| Metric | v1.1.0 | v1.2.0 | Improvement |
|--------|--------|--------|-------------|
| System Prompt Size | ~5500 tokens | ~300 tokens | 94% reduction |
| Search Result Size | Full raw results | Summarized | 60-80% reduction |
| Console Output | 50+ log lines | 0 (clean) | 100% reduction |
| TTS Setup Time | Variable | Instant | Always available |

---

## 📁 File Structure

### Installer
```
Mimic AI_1.2.0_x64-setup.exe (164.7 MB)
```

### Memory Storage
```
~/MimicAI/Memories/
├── default/                    # Default persona
│   ├── user_files/
│   ├── conversations/
│   └── history.json
└── persona_*/                  # Custom personas
    ├── user_files/
    ├── conversations/
    └── history.json
```

---

## 🎓 Usage Tips

### Voice Setup (New Users)
1. Go to **Voice Studio** tab
2. Select **KittenTTS** engine
3. Choose your preferred voice (Bella recommended for female, Jasper for male)
4. Adjust speed if needed (1.0 is normal)
5. Save to persona

### Memory Management
1. Go to **Memory Manager** (paperclip menu)
2. **Memory Files** tab: Upload documents for the current persona
3. **AI Memories**: View important extracted points
4. **Full History**: Browse complete conversation log

### Web Search
1. Enable in Settings
2. Ask questions naturally
3. Router automatically summarizes results
4. No need to ask for "search" explicitly

### Image Analysis
1. Click the **paperclip** button
2. Select an image file (JPG, PNG, GIF, etc.)
3. Add optional text prompt
4. Send - the AI will analyze the image

---

## 🐛 Known Issues

- Qwen3-TTS requires ~3-6GB VRAM (use KittenTTS if VRAM limited)
- Docker Desktop required for web search functionality
- First launch may take 30-60 seconds to initialize

---

## 📞 Support

- **Email:** deadheadstudios@atomicmail.io
- **GitHub:** https://github.com/bmerriott/MIMIC-Multipurpose-Intelligent-Molecular-Information-Catalyst-
- **Documentation:** See README.md and QUICKSTART.md

---

## 📄 License

**Proprietary License** - See LICENSE.txt for full EULA

Copyright (c) 2026 Dead Head Studios. All Rights Reserved.

---

**Enjoy MIMIC AI v1.2.0!**
