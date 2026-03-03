# Mimic AI v1.2.1 Release Guide

## Release Files Location
```
releases/
├── Mimic AI_1.2.1_x64-setup.exe  (164.12 MB) - Windows Installer
└── Mimic-AI-v1.2.1-x64.zip       (164.15 MB) - ZIP Archive
```

---

## Option 1: GitHub CLI (Recommended)

### Prerequisites
```powershell
# Install GitHub CLI if not already installed
winget install GitHub.cli

# Login to GitHub
gh auth login
```

### Create Release with Assets
```powershell
cd "C:\Users\merri\Downloads\Mimic AI Desktop Assistant"

# Create release with all assets
gh release create v1.2.1 `
  --title "Mimic AI v1.2.1" `
  --notes-file RELEASE_NOTES.md `
  "releases/Mimic AI_1.2.1_x64-setup.exe#Windows Installer" `
  "releases/Mimic-AI-v1.2.1-x64.zip#Windows ZIP Archive"
```

### Verify Release
```powershell
# View release details
gh release view v1.2.1

# Open in browser
gh release view v1.2.1 --web
```

---

## Option 2: Git Commands + Manual Upload

### Step 1: Commit Changes
```powershell
cd "C:\Users\merri\Downloads\Mimic AI Desktop Assistant"

# Add all changes
git add .

# Commit with version message
git commit -m "Release v1.2.1 - Fix Memory Manager crashes and UI improvements

Changes:
- Fix React crash when rendering object content in memories
- Fix Full History tab crash with safe content rendering  
- Fix Conversational Memories tab with proper data sanitization
- Add safeContent() helper for all memory/insight rendering
- Update Memory Manager UI layout"

# Push to main
git push origin main
```

### Step 2: Create and Push Tag
```powershell
# Create annotated tag
git tag -a v1.2.1 -m "Release v1.2.1"

# Push tag to remote
git push origin v1.2.1
```

### Step 3: Manual Release Upload
1. Go to: https://github.com/YOUR_USERNAME/mimic-ai/releases
2. Click "Draft a new release"
3. Choose tag: `v1.2.1`
4. Release title: `Mimic AI v1.2.1`
5. Paste release notes (see RELEASE_NOTES.md)
6. Upload files from `releases/` folder:
   - `Mimic AI_1.2.1_x64-setup.exe`
   - `Mimic-AI-v1.2.1-x64.zip`
7. Click "Publish release"

---

## Quick One-Liner Commands

### Complete Release (CLI)
```powershell
cd "C:\Users\merri\Downloads\Mimic AI Desktop Assistant"; git add .; git commit -m "v1.2.1 - Memory Manager fixes"; git push; git tag v1.2.1; git push origin v1.2.1; gh release create v1.2.1 --title "Mimic AI v1.2.1" --notes "Fix Memory Manager crashes - all tabs working" "releases/Mimic AI_1.2.1_x64-setup.exe" "releases/Mimic-AI-v1.2.1-x64.zip"
```

### Just Create Release (if already committed)
```powershell
cd "C:\Users\merri\Downloads\Mimic AI Desktop Assistant"
gh release create v1.2.1 --title "Mimic AI v1.2.1" --notes-file RELEASE_NOTES.md releases/*
```

---

## Release Notes Template

Create `RELEASE_NOTES.md`:
```markdown
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
```

---

## Troubleshooting

### If gh CLI fails
```powershell
# Check you're authenticated
gh auth status

# Re-authenticate if needed
gh auth login
```

### If release already exists
```powershell
# Delete existing release
gh release delete v1.2.1 --yes

# Recreate
gh release create v1.2.1 --title "Mimic AI v1.2.1" --notes "..." releases/*
```

### Update existing release
```powershell
# Upload additional files
gh release upload v1.2.1 "releases/NEW_FILE.exe"

# Remove and re-add
gh release upload v1.2.1 --clobber "releases/Mimic AI_1.2.1_x64-setup.exe"
```

---

## Files Checksum (for verification)

```powershell
cd "C:\Users\merri\Downloads\Mimic AI Desktop Assistant\releases"

# Generate SHA256 checksums
Get-FileHash "Mimic AI_1.2.1_x64-setup.exe" -Algorithm SHA256
Get-FileHash "Mimic-AI-v1.2.1-x64.zip" -Algorithm SHA256
```

Add checksums to release notes for security.
