; NSIS Installer Hooks for Mimic AI
; Installs espeak-ng dependency during setup

!macro NSIS_HOOK_PREINSTALL
  ; This runs before files are extracted
!macroend

!macro NSIS_HOOK_POSTINSTALL
  ; This runs after files are extracted - install espeak-ng here
  DetailPrint "Checking for espeak-ng..."
  
  ; Check if espeak-ng is already installed
  IfFileExists "$PROGRAMFILES64\eSpeak NG\espeak-ng.exe" espeak_already_installed
  
  ; Install espeak-ng from bundled MSI
  DetailPrint "Installing espeak-ng (required for KittenTTS)..."
  
  ; The MSI should be in the resources folder after extraction
  IfFileExists "$INSTDIR\resources\espeak-ng.msi" espeak_msi_found
  
  ; Try alternate location
  IfFileExists "$EXEDIR\espeak-ng.msi" espeak_msi_found_alt
  
  DetailPrint "WARNING: espeak-ng.msi not found in bundle"
  DetailPrint "KittenTTS may not work without espeak-ng"
  Goto espeak_install_done
  
  espeak_msi_found:
    ExecWait '"msiexec" /i "$INSTDIR\resources\espeak-ng.msi" /qn /norestart' $0
    Goto espeak_install_check_result
    
  espeak_msi_found_alt:
    ExecWait '"msiexec" /i "$EXEDIR\espeak-ng.msi" /qn /norestart' $0
    Goto espeak_install_check_result
  
  espeak_install_check_result:
    ${If} $0 == 0
      DetailPrint "espeak-ng installed successfully"
    ${Else}
      DetailPrint "espeak-ng installation may have failed (code: $0)"
      DetailPrint "KittenTTS voice engine may not work without espeak-ng"
    ${EndIf}
    Goto espeak_install_done
  
  espeak_already_installed:
    DetailPrint "espeak-ng already installed"
  
  espeak_install_done:
!macroend
