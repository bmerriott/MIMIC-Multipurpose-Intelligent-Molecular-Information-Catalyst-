#!/usr/bin/env python3
"""
Mimic AI - Audio Watermark Detection Tool

This tool detects the AI-generated watermark embedded in audio files
created by the Mimic AI TTS system.

Usage:
    python detect_watermark.py <audio_file>
    python detect_watermark.py <audio_file> --detailed
    
The watermark is designed for legal evidentiary purposes:
- Multi-layer encoding for redundancy
- Survives compression and common processing
- Provides confidence scores for legal proceedings

Exit codes:
    0 - Watermark detected (AI-generated)
    1 - No watermark detected (likely human/original)
    2 - Error during analysis
"""

import sys
import os
import argparse

# Try to import the legal watermarker
try:
    from watermarker import LegalWatermarker, detect_ai_watermark
    LEGAL_MODE = True
except ImportError:
    LEGAL_MODE = False
    print("Note: watermarker.py not found. Using basic detection mode.")
    print("For full detection capabilities, ensure watermarker.py is in the same directory.")
    print()

import numpy as np
import wave
import hashlib
from typing import Tuple


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file (WAV format). Supports both standard and compressed WAV."""
    # Try reading as standard WAV first
    try:
        with wave.open(file_path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            raw_data = wav_file.readframes(n_frames)
            
            # Convert to numpy array based on sample width
            if sample_width == 2:
                audio_array = np.frombuffer(raw_data, dtype=np.int16)
            elif sample_width == 4:
                audio_array = np.frombuffer(raw_data, dtype=np.int32)
            else:
                audio_array = np.frombuffer(raw_data, dtype=np.uint8)
            
            # Convert to float32 in range [-1, 1]
            audio_array = audio_array.astype(np.float32) / 32767.0
            
            # Convert to mono if stereo
            if n_channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1)
            
            return audio_array, sample_rate
    except wave.Error:
        pass
    
    # Try using soundfile for other formats (MP3, OGG, FLAC, etc.)
    try:
        import soundfile as sf
        audio_array, sample_rate = sf.read(file_path, dtype='float32')
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        return audio_array, sample_rate
    except ImportError:
        print("Error: soundfile not installed. Install with: pip install soundfile")
        print("       (Required for non-WAV formats)")
        sys.exit(2)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(2)


def basic_detect(audio: np.ndarray, sample_rate: int) -> Tuple[bool, float, str]:
    """Basic spread-spectrum detection (fallback mode)."""
    key = "AI-generated"
    np.random.seed(int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32))
    chip_rate = 100
    
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    num_chips = int(len(audio) * chip_rate / sample_rate) + 1
    chips = np.random.choice([-1, 1], size=num_chips)
    chip_samples = int(sample_rate / chip_rate)
    watermark = np.repeat(chips, chip_samples)[:len(audio)]
    
    correlation = np.correlate(audio, watermark, mode='valid')
    max_corr = np.max(np.abs(correlation))
    mean_corr = np.mean(np.abs(correlation))
    
    confidence = max_corr / (mean_corr + 1e-8)
    detected = confidence > 2.0
    
    message = "AI watermark detected" if detected else "No AI watermark detected"
    return detected, float(confidence), message


def main():
    parser = argparse.ArgumentParser(
        description='Detect AI-generated watermarks in audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python detect_watermark.py recording.wav
    python detect_watermark.py song.mp3 --detailed
    python detect_watermark.py voice.ogg --json
        """
    )
    parser.add_argument('audio_file', help='Path to audio file (WAV, MP3, OGG, FLAC)')
    parser.add_argument('--detailed', '-d', action='store_true', 
                        help='Show detailed layer-by-layer analysis')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output results as JSON')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(2)
    
    # Header
    if not args.json:
        print("=" * 70)
        print("Mimic AI - Audio Watermark Detection Tool")
        print("Legal Evidence and Forensic Analysis")
        print("=" * 70)
    
    # Load audio
    try:
        audio, sample_rate = load_audio(args.audio_file)
        duration = len(audio) / sample_rate
        
        if not args.json:
            print(f"\nFile: {args.audio_file}")
            print(f"Sample rate: {sample_rate} Hz")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Samples: {len(audio):,}")
            print(f"Detection mode: {'Legal (multi-layer)' if LEGAL_MODE else 'Basic (single-layer)'}")
            print("\nAnalyzing for AI-generated watermark...")
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(2)
    
    # Detect watermark
    try:
        if LEGAL_MODE and not args.json:
            # Use full legal watermarker with detailed output
            watermarker = LegalWatermarker()
            detected, confidence, details = watermarker.detect_watermark(audio, sample_rate)
            
            print("\n" + "=" * 70)
            if detected:
                print("RESULT: AI-GENERATED WATERMARK DETECTED")
                print(f"Confidence: {confidence:.1%}")
                print(f"\nLayer Analysis:")
                for layer, info in details.items():
                    if isinstance(info, dict) and 'detected' in info:
                        status = "✓ DETECTED" if info['detected'] else "✗ Not found"
                        print(f"  {layer:20s}: {status} (conf: {info['confidence']:.2f})")
                print(f"\nTotal layers verified: {details.get('layers_detected', 0)}/{details.get('total_layers', 3)}")
                print("\nThis audio was generated by Mimic AI.")
                print("This finding can be used as evidence in legal proceedings.")
            else:
                print("RESULT: NO AI-GENERATED WATERMARK DETECTED")
                print(f"Confidence: {confidence:.1%}")
                print("\nThis audio appears to be original (no AI watermark found).")
                if details and isinstance(details, dict):
                    print(f"\nLayer Analysis:")
                    for layer, info in details.items():
                        if isinstance(info, dict) and 'detected' in info:
                            status = "✓ DETECTED" if info['detected'] else "✗ Not found"
                            print(f"  {layer:20s}: {status} (conf: {info['confidence']:.2f})")
            print("=" * 70)
            
        elif LEGAL_MODE and args.json:
            # JSON output mode
            import json
            watermarker = LegalWatermarker()
            detected, confidence, details = watermarker.detect_watermark(audio, sample_rate)
            
            result = {
                'file': args.audio_file,
                'detected': bool(detected),
                'confidence': float(confidence),
                'sample_rate': sample_rate,
                'duration_sec': duration,
                'detection_mode': 'legal_multi_layer',
                'layer_details': details if isinstance(details, dict) else {}
            }
            print(json.dumps(result, indent=2))
            
        else:
            # Basic detection mode
            detected, confidence, message = basic_detect(audio, sample_rate)
            
            if not args.json:
                print("\n" + "=" * 70)
                if detected:
                    print(f"RESULT: {message}")
                    print(f"Confidence: {confidence:.2f}")
                    print("\nThis audio may be AI-generated.")
                else:
                    print(f"RESULT: {message}")
                    print(f"Confidence: {confidence:.2f}")
                    print("\nThis audio appears to be original.")
                print("=" * 70)
            else:
                import json
                result = {
                    'file': args.audio_file,
                    'detected': bool(detected),
                    'confidence': float(confidence),
                    'sample_rate': sample_rate,
                    'duration_sec': duration,
                    'detection_mode': 'basic_single_layer'
                }
                print(json.dumps(result, indent=2))
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
    
    # Exit code
    sys.exit(0 if detected else 1)


if __name__ == "__main__":
    main()
