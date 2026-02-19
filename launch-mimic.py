#!/usr/bin/env python3
"""
Mimic AI Launcher Script
Checks dependencies and launches services using system Python
"""

import subprocess
import sys
import os
import time
import re
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"    Found Python {version.major}.{version.minor}.{version.micro}")
    
    # Check if version is 3.10-3.12
    if version.major != 3 or version.minor < 10 or version.minor > 12:
        print("\n" + "="*50)
        print("   PYTHON VERSION ERROR")
        print("="*50)
        print(f"\nYour Python version ({version.major}.{version.minor}) is not compatible.")
        print("\nSupported versions: Python 3.10, 3.11, or 3.12")
        
        if version.minor > 12:
            print("\nIf you have Python 3.13 or newer:")
            print("  PyTorch (required for TTS) does not yet support Python 3.13+")
            print("  You need to install Python 3.11 alongside your current version.")
        
        print("\nSOLUTION:")
        print("1. Go to https://python.org/downloads/")
        print("2. Download Python 3.11.8")
        print("3. Run the installer and CHECK 'Add Python to PATH'")
        print("4. Install")
        print("\nPython versions can coexist - installing 3.11 will NOT")
        print("remove your current Python installation.")
        print("="*50 + "\n")
        return False
    
    print("    OK (Compatible)")
    return True

def check_nodejs():
    """Check Node.js installation"""
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        print(f"    Found {version}")
        
        # Parse version number
        major = int(version.lstrip('v').split('.')[0])
        if major < 18:
            print(f"\n[WARNING] Node.js {version} may be too old!")
            print("          Recommended: 18.x or 20.x LTS")
            response = input("\nContinue anyway? (y/n): ").lower()
            return response == 'y'
        
        print("    OK")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n" + "="*50)
        print("   NODE.JS NOT FOUND")
        print("="*50)
        print("\nNode.js is required to run Mimic AI.")
        print("\nPlease install Node.js 20 LTS from:")
        print("  https://nodejs.org")
        print("\nDuring installation, check 'Add to PATH'")
        print("="*50 + "\n")
        input("Press Enter to exit...")
        return False

def install_dependencies():
    """Install all required dependencies to system Python"""
    script_dir = Path(__file__).parent.absolute()
    
    # Frontend dependencies
    print("    Checking frontend dependencies...")
    if not (script_dir / "app" / "node_modules").exists():
        print("    Installing npm packages (this may take a few minutes)...")
        result = subprocess.run(['npm', 'install'], 
                              cwd=script_dir / "app",
                              capture_output=False)
        if result.returncode != 0:
            print("[ERROR] npm install failed!")
            return False
    else:
        print("    Frontend dependencies OK")
    
    # Python dependencies
    print("    Checking Python packages...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'python-multipart', 'pydantic', 
        'python-dotenv', 'numpy', 'scipy', 'soundfile', 
        'librosa', 'requests'
    ]
    
    # Check which packages are missing
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg.replace('-', '_').split('[')[0])
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"    Installing: {', '.join(missing)}")
        print("    This may take a few minutes...")
        result = subprocess.run([sys.executable, "-m", "pip", "install"] + missing,
                              capture_output=False)
        if result.returncode != 0:
            print("[ERROR] Failed to install Python packages!")
            print("        Try: python -m pip install --upgrade pip")
            return False
    
    # Check PyTorch
    try:
        import torch
        print("    PyTorch OK")
    except ImportError:
        print("    Installing PyTorch (CPU version)...")
        print("    This may take several minutes...")
        result = subprocess.run([sys.executable, "-m", "pip", "install",
                               "torch", "torchaudio",
                               "--index-url", "https://download.pytorch.org/whl/cpu"],
                              capture_output=False)
        if result.returncode != 0:
            print("[ERROR] Failed to install PyTorch!")
            return False
        print("    [INFO] For GPU support, run:")
        print("           pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("    OK")
    return True

def start_ollama():
    """Start Ollama server"""
    print("[*] Starting Ollama Server...")
    
    # Find Ollama
    possible_paths = [
        Path("C:/Program Files/Ollama/ollama.exe"),
        Path(os.environ.get('LOCALAPPDATA', '')) / "Programs/Ollama/ollama.exe",
        Path(os.environ.get('USERPROFILE', '')) / "AppData/Local/Programs/Ollama/ollama.exe",
    ]
    
    ollama_path = None
    for path in possible_paths:
        if path.exists():
            ollama_path = path
            break
    
    if not ollama_path:
        print("\n" + "="*50)
        print("   OLLAMA NOT FOUND")
        print("="*50)
        print("\nOllama is required for AI model inference.")
        print("\nPlease install Ollama from https://ollama.com")
        print("\nAfter installation, pull at least one model:")
        print("  ollama pull llama3.2")
        print("\nVisit https://ollama.com/library for more models")
        print("="*50 + "\n")
        input("Press Enter to exit...")
        return False
    
    # Kill existing processes
    print("    Stopping existing Ollama processes...")
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/IM', 'ollama.exe'],
                          capture_output=True)
        else:
            subprocess.run(['pkill', '-f', 'ollama'],
                          capture_output=True)
        time.sleep(3)
    except:
        pass
    
    # Start server
    print("    Starting ollama serve...")
    if os.name == 'nt':
        subprocess.Popen([str(ollama_path), 'serve'],
                        creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen([str(ollama_path), 'serve'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
    
    # Wait for server
    print("    Waiting for server...")
    for i in range(15):
        time.sleep(2)
        try:
            import urllib.request
            req = urllib.request.Request('http://localhost:11434/api/tags', 
                                        method='GET', timeout=3)
            with urllib.request.urlopen(req) as resp:
                if resp.status == 200:
                    print("    OK (Server ready)")
                    return True
        except:
            pass
    
    print("[WARNING] Ollama server may not have started properly")
    response = input("Continue anyway? (y/n): ").lower()
    return response == 'y'

def check_models():
    """Check if any models are available"""
    print("[*] Checking for Ollama models...")
    
    try:
        import urllib.request
        import json
        
        req = urllib.request.Request('http://localhost:11434/api/tags',
                                    method='GET', timeout=5)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            models = data.get('models', [])
            
            if not models:
                print("\n[WARNING] No models found!")
                print("\nPlease pull a model:")
                print("  ollama pull llama3.2")
                print("\nVisit https://ollama.com/library for more models")
                response = input("\nContinue anyway? (y/n): ").lower()
                return response == 'y'
            
            print(f"    OK ({len(models)} models available)")
            for model in models[:3]:
                print(f"      - {model.get('name', 'unknown')}")
            if len(models) > 3:
                print(f"      ... and {len(models) - 3} more")
            return True
    except Exception as e:
        print(f"[WARNING] Could not check models: {e}")
        return True

def start_services():
    """Start TTS backend and frontend"""
    script_dir = Path(__file__).parent.absolute()
    
    print("[*] Starting Mimic AI...")
    
    # Kill old processes
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/IM', 'python.exe',
                           '/FI', 'WINDOWTITLE eq TTS Backend'],
                          capture_output=True)
    except:
        pass
    
    # Start backend
    print("[*] Starting TTS Backend...")
    if os.name == 'nt':
        backend_bat = Path(os.environ.get('TEMP', '/tmp')) / 'mimic_backend.bat'
        with open(backend_bat, 'w') as f:
            f.write(f'@echo off\n')
            f.write(f'cd /d "{script_dir}"\n')
            f.write(f'cd app\\backend\n')
            f.write(f'python tts_server_unified.py\n')
            f.write(f'pause\n')
        subprocess.Popen([str(backend_bat)], 
                        creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        # Linux/Mac - run in background
        subprocess.Popen([sys.executable, 'tts_server_unified.py'],
                        cwd=script_dir / 'app' / 'backend',
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
    
    time.sleep(15)
    
    # Start frontend
    print("[*] Starting Frontend...")
    if os.name == 'nt':
        frontend_bat = Path(os.environ.get('TEMP', '/tmp')) / 'mimic_frontend.bat'
        with open(frontend_bat, 'w') as f:
            f.write(f'@echo off\n')
            f.write(f'cd /d "{script_dir}\\app"\n')
            f.write(f'npm run dev\n')
            f.write(f'pause\n')
        subprocess.Popen([str(frontend_bat)],
                        creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen(['npm', 'run', 'dev'],
                        cwd=script_dir / 'app',
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
    
    time.sleep(10)
    
    # Open browser
    print("[*] Opening browser...")
    if os.name == 'nt':
        subprocess.Popen(['start', 'http://localhost:5173'], shell=True)
    elif os.uname().sysname == 'Darwin':
        subprocess.Popen(['open', 'http://localhost:5173'])
    else:
        subprocess.Popen(['xdg-open', 'http://localhost:5173'])
    
    print("\n" + "="*50)
    print("   Mimic AI is Starting!")
    print("="*50)
    print("\nPress Enter to STOP all services...")
    input()
    
    return True

def cleanup():
    """Stop all services"""
    print("\n[*] Stopping services...")
    
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/IM', 'python.exe',
                           '/FI', 'WINDOWTITLE eq TTS Backend'],
                          capture_output=True)
            subprocess.run(['taskkill', '/F', '/IM', 'ollama.exe'],
                          capture_output=True)
        else:
            subprocess.run(['pkill', '-f', 'tts_server_unified'],
                          capture_output=True)
            subprocess.run(['pkill', '-f', 'ollama'],
                          capture_output=True)
    except:
        pass
    
    print("\n" + "="*50)
    print("   All Services Stopped")
    print("="*50)

def main():
    """Main launcher function"""
    print("="*50)
    print("   Mimic AI - Launcher")
    print("="*50)
    print()
    
    os.chdir(Path(__file__).parent.absolute())
    
    # Step 1: Check Python version
    print("[*] Checking Python...")
    if not check_python_version():
        input("\nPress Enter to exit...")
        return 1
    
    # Step 2: Check Node.js
    if not check_nodejs():
        return 1
    
    # Step 3: Install dependencies
    print("[*] Installing dependencies...")
    if not install_dependencies():
        input("\nPress Enter to exit...")
        return 1
    
    # Step 4: Start Ollama
    if not start_ollama():
        return 1
    
    # Step 5: Check models
    if not check_models():
        return 0
    
    # Step 6: Start services
    if not start_services():
        return 1
    
    # Step 7: Cleanup on exit
    cleanup()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cleanup()
        sys.exit(0)
