## Mimic AI v1.2.1

### Bug Fixes
- **Fixed Memory Manager crashes** - All tabs (Saved Files, Conversational, Full History) now work correctly
- **Fixed React error #31** - Object content in memories now renders as JSON instead of crashing
- **Fixed summary rendering** - Added safeContent() wrapper for all memory summaries
- **Fixed insight rendering** - Personality Manager now handles object content safely

### Changes
- Added `safeContent()` helper function to convert any value to string
- Applied safe content rendering to all memory/insight display locations
- Filter invalid memory entries during loading
- Defensive coding for corrupted data in localStorage

### Files
| File | Size | Description |
|------|------|-------------|
| Mimic AI_1.2.1_x64-setup.exe | 164 MB | Windows Installer |
| Mimic-AI-v1.2.1-x64.zip | 164 MB | Portable ZIP |

### Installation
1. Download the installer or ZIP
2. Run installer (or extract ZIP and run `Mimic AI.exe`)
3. Your existing personas and settings will be preserved
