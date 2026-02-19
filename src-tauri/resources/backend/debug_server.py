"""
Debug script to test TTS server startup step by step
"""
import sys
import os

print("=" * 60)
print("Mimic AI TTS Backend - Debug Mode")
print("=" * 60)
print()

# Step 1: Environment
print("Step 1: Checking environment...")
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print()

# Step 2: Test imports
print("Step 2: Testing imports...")
try:
    import fastapi
    print("[OK] fastapi")
except Exception as e:
    print(f"[FAIL] fastapi: {e}")
    sys.exit(1)

try:
    import uvicorn
    print("[OK] uvicorn")
except Exception as e:
    print(f"[FAIL] uvicorn: {e}")
    sys.exit(1)

try:
    import numpy
    print("[OK] numpy")
except Exception as e:
    print(f"[FAIL] numpy: {e}")
    sys.exit(1)

print()

# Step 3: Import the app module
print("Step 3: Importing tts_server_unified...")
try:
    import tts_server_unified
    print("[OK] Module imported successfully")
    print(f"  - StyleTTS2: {tts_server_unified.STYLETTS2_AVAILABLE}")
    print(f"  - Qwen3: {tts_server_unified.QWEN_TTS_AVAILABLE}")
except Exception as e:
    print(f"[FAIL] Could not import module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 4: Check if port is available
print("Step 4: Checking port 8000...")
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 8000))
    if result == 0:
        print("[WARNING] Port 8000 is already in use!")
        print("          Another instance may be running.")
    else:
        print("[OK] Port 8000 is available")
    sock.close()
except Exception as e:
    print(f"[WARNING] Could not check port: {e}")

print()

# Step 5: Try to start the server
print("Step 5: Starting server...")
print("Press Ctrl+C to stop")
print()

try:
    # Use the same settings as the main script
    port = int(os.environ.get("MIMIC_PORT", "8000"))
    
    uvicorn.run(
        "tts_server_unified:app",
        host="127.0.0.1",
        port=port,
        reload=False,
        log_level="debug"  # More verbose logging
    )
except KeyboardInterrupt:
    print("\n[INFO] Server stopped by user")
except Exception as e:
    print(f"\n[ERROR] Server failed: {e}")
    import traceback
    traceback.print_exc()
    input("\nPress Enter to exit...")
    sys.exit(1)

print("[INFO] Server exited")
input("Press Enter to exit...")
