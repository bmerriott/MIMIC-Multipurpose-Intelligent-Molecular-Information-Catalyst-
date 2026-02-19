; Mimic AI Installer Script
; Creates a one-click installer for Windows

!include "MUI2.nsh"
!include "LogicLib.nsh"

; App information
!define APPNAME "Mimic AI"
!define COMPANYNAME "Mimic AI"
!define DESCRIPTION "AI Desktop Assistant with Voice Synthesis"
!define VERSION "1.0.0"
!define INSTALLSIZE "500000" ; KB

; Installer configuration
Name "${APPNAME}"
OutFile "..\Mimic-AI-Setup.exe"
InstallDir "$LOCALAPPDATA\${APPNAME}"
InstallDirRegKey HKCU "Software\${APPNAME}" "InstallDir"
RequestExecutionLevel user

; Interface settings
!define MUI_ABORTWARNING
!define MUI_ICON "icons\icon.ico"
!define MUI_UNICON "icons\icon.ico"
!define MUI_WELCOMEPAGE_TEXT "Welcome to Mimic AI Setup.$\r$\n$\r$\nThis will install Mimic AI on your computer.$\r$\n$\r$\nMimic AI requires Python 3.10+ to be installed.$\r$\nIf Python is not installed, please install it first from python.org"

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

; Sections
Section "Install" SecInstall
    SetOutPath "$INSTDIR"
    
    ; Copy application files
    File /r "target\release\*.*"
    
    ; Create data directories
    CreateDirectory "$APPDATA\com.mimicai.app"
    CreateDirectory "$APPDATA\com.mimicai.app\voices"
    CreateDirectory "$APPDATA\com.mimicai.app\backend_data"
    
    ; Create shortcuts
    CreateDirectory "$SMPROGRAMS\${APPNAME}"
    CreateShortcut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\mimic-ai.exe"
    CreateShortcut "$SMPROGRAMS\${APPNAME}\Uninstall.lnk" "$INSTDIR\uninstall.exe"
    CreateShortcut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\mimic-ai.exe"
    
    ; Write registry keys
    WriteRegStr HKCU "Software\${APPNAME}" "InstallDir" "$INSTDIR"
    WriteRegStr HKCU "Software\${APPNAME}" "Version" "${VERSION}"
    
    ; Write uninstaller
    WriteUninstaller "$INSTDIR\uninstall.exe"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayName" "${APPNAME}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayIcon" "$INSTDIR\mimic-ai.exe"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "Publisher" "${COMPANYNAME}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayVersion" "${VERSION}"
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "EstimatedSize" ${INSTALLSIZE}
    
    ; Check for Python and show message if not found
    nsExec::ExecToStack 'python --version'
    Pop $0
    ${If} $0 != "0"
        MessageBox MB_OK "Python was not detected on your system.$\r$\n$\r$\nPlease install Python 3.10 or higher from https://python.org$\r$\n$\r$\nThe app will not function without Python."
    ${EndIf}
    
SectionEnd

Section "Uninstall"
    ; Remove files
    RMDir /r "$INSTDIR"
    
    ; Remove shortcuts
    Delete "$DESKTOP\${APPNAME}.lnk"
    RMDir /r "$SMPROGRAMS\${APPNAME}"
    
    ; Remove registry keys
    DeleteRegKey HKCU "Software\${APPNAME}"
    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
    
    ; Ask about removing user data
    MessageBox MB_YESNO "Do you want to remove all user data (voices, settings)?$\r$\nThis cannot be undone." /SD IDNO IDNO SkipDataRemoval
        RMDir /r "$APPDATA\com.mimicai.app"
    SkipDataRemoval:
    
SectionEnd
