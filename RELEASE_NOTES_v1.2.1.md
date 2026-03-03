## Mimic AI v1.2.1

### Bug Fixes
- **Fixed Memory Manager blank screen crashes** - All tabs (Saved Files, Conversational, Full History) now work correctly
- **Fixed tab switching crashes** - Replaced Radix Tabs with custom implementation for stability
- **Improved error handling** - Safe date parsing and defensive memory mapping

### UI Improvements
- **Cleaner Memory Manager layout** - Better spacing and responsive design
- **Simpler tab interface** - Custom tab buttons instead of complex Tabs component
- **Better empty states** - Clearer messaging when no memories exist

### Technical Changes
- Replaced `@radix-ui/react-tabs` implementation with custom tab state management
- Added comprehensive null checking for persona memory data
- Fixed type safety issues with memory entries

### Files
- Windows Installer: `Mimic AI_1.2.1_x64-setup.exe` (164 MB)
- ZIP Archive: `Mimic-AI-v1.2.1-x64.zip` (164 MB)

### Installation
1. Download the installer or ZIP
2. Run the installer (or extract ZIP and run `Mimic AI.exe`)
3. The app will update automatically if you have a previous version installed
