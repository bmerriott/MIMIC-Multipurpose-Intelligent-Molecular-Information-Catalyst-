# GitHub Release Commands for Mimic AI v1.2.1

## Files Location
- **Installer:** `src-tauri\target\release\bundle\nsis\Mimic AI_1.2.1_x64-setup.exe`
- **ZIP:** `src-tauri\target\release\bundle\nsis\Mimic-AI-v1.2.1-x64.zip`

---

## Option 1: GitHub CLI (Recommended)

### Install GitHub CLI if not already installed:
```powershell
winget install GitHub.cli
```

### Login to GitHub:
```powershell
gh auth login
```

### Create Release with Assets:
```powershell
cd "C:\Users\merri\Downloads\Mimic AI Desktop Assistant"

# Create release
gh release create v1.2.1 `
  --title "Mimic AI v1.2.1" `
  --notes-file RELEASE_NOTES_v1.2.1.md `
  "src-tauri\target\release\bundle\nsis\Mimic AI_1.2.1_x64-setup.exe#Windows Installer" `
  "src-tauri\target\release\bundle\nsis\Mimic-AI-v1.2.1-x64.zip#Windows ZIP"
```

---

## Option 2: Manual Git Commands

### Step 1: Commit version changes
```powershell
cd "C:\Users\merri\Downloads\Mimic AI Desktop Assistant"
git add package.json app/package.json src-tauri/tauri.conf.json src-tauri/Cargo.toml
git commit -m "Bump version to 1.2.1 - Fix Memory Manager crashes and UI"
git push origin main
```

### Step 2: Create and push tag
```powershell
git tag -a v1.2.1 -m "Release v1.2.1 - Memory Manager fixes"
git push origin v1.2.1
```

### Step 3: Upload assets via GitHub web interface
1. Go to https://github.com/YOUR_USERNAME/mimic-ai/releases
2. Click "Draft a new release"
3. Select tag "v1.2.1"
4. Title: "Mimic AI v1.2.1"
5. Paste release notes from below
6. Upload files:
   - `Mimic AI_1.2.1_x64-setup.exe`
   - `Mimic-AI-v1.2.1-x64.zip`
7. Click "Publish release"

---

## Release Notes (v1.2.1)

```markdown
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
```

---

## Quick Copy-Paste Commands

```powershell
# Navigate to project
cd "C:\Users\merri\Downloads\Mimic AI Desktop Assistant"

# Commit version bump
git add .
git commit -m "v1.2.1 - Fix Memory Manager crashes"
git push

# Create and push tag
git tag v1.2.1
git push origin v1.2.1

# Create GitHub release with CLI
gh release create v1.2.1 `
  --title "Mimic AI v1.2.1" `
  --notes "Fixed Memory Manager blank screen crashes. All tabs now work correctly." `
  "src-tauri\target\release\bundle\nsis\Mimic AI_1.2.1_x64-setup.exe" `
  "src-tauri\target\release\bundle\nsis\Mimic-AI-v1.2.1-x64.zip"
```

---

## Verify Release

Check the release was created successfully:
```powershell
gh release view v1.2.1
```

Open in browser:
```powershell
gh release view v1.2.1 --web
```
