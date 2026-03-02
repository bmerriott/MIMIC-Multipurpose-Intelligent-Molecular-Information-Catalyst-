; Custom NSIS installer script for Mimic AI
; Installs espeak-ng as a dependency

!include "MUI2.nsh"
!include "LogicLib.nsh"

; Define installer name and output
Name "Mimic AI"
OutFile "Mimic AI_1.0.0_x64-setup.exe"

; Default installation directory
InstallDir "$PROGRAMFILES64\Mimic AI"

; Request admin privileges
RequestExecutionLevel admin

; Interface settings
!define MUI_ABORTWARNING

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; Languages
!insertmacro MUI_LANGUAGE "English"

; Installation section
Section "Install"
  SetOutPath "$INSTDIR"
  
  ; Extract main application files (these come from Tauri build)
  File /r "..\target\release\bundle\nsis\*.*"
  
  ; Install espeak-ng if not already installed
  DetailPrint "Checking for espeak-ng..."
  IfFileExists "$PROGRAMFILES64\eSpeak NG\espeak-ng.exe" espeak_installed 0
  
  DetailPrint "Installing espeak-ng (required for KittenTTS)..."
  SetOutPath "$TEMP"
  File "..\resources\espeak-ng.msi"
  
  ; Run MSI installer silently
  ExecWait '"msiexec" /i "$TEMP\espeak-ng.msi" /qn /norestart' $0
  
  ${If} $0 == 0
    DetailPrint "espeak-ng installed successfully"
  ${Else}
    DetailPrint "espeak-ng installation may have failed (code: $0)"
    DetailPrint "KittenTTS voice engine may not work without espeak-ng"
  ${EndIf}
  
  Delete "$TEMP\espeak-ng.msi"
  
  espeak_installed:
  
  ; Create shortcuts
  CreateDirectory "$SMPROGRAMS\Mimic AI"
  CreateShortcut "$SMPROGRAMS\Mimic AI\Mimic AI.lnk" "$INSTDIR\mimic-ai.exe"
  CreateShortcut "$DESKTOP\Mimic AI.lnk" "$INSTDIR\mimic-ai.exe"
  
  ; Write uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"
  
  ; Registry entries
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Mimic AI" "DisplayName" "Mimic AI"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Mimic AI" "UninstallString" "$INSTDIR\uninstall.exe"
SectionEnd

; Uninstallation section
Section "Uninstall"
  ; Remove shortcuts
  Delete "$SMPROGRAMS\Mimic AI\Mimic AI.lnk"
  RMDir "$SMPROGRAMS\Mimic AI"
  Delete "$DESKTOP\Mimic AI.lnk"
  
  ; Remove installed files
  RMDir /r "$INSTDIR"
  
  ; Remove registry entries
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Mimic AI"
SectionEnd
