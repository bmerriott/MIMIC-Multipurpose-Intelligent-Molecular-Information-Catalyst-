"""
Mimic AI - Unified TTS Backend Server
Supports multiple TTS engines: StyleTTS2 and QWEN3-TTS
Provides voice creation from reference audio (NOT cloning - synthesis from reference)
"""

import os
import io
import base64
import tempfile
import time
import traceback
import hashlib
import random
import sys
import unicodedata
from typing import Optional, Union, List, Tuple, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum

# Force UTF-8 encoding for stdout/stderr on Windows to prevent charmap codec errors
# Handle case where stdout/stderr may be redirected (e.g., when running as daemon)
if sys.platform == 'win32':
    import codecs
    try:
        # Only wrap if stdout has a buffer attribute (not when redirected to null)
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except (AttributeError, IOError):
        # stdout/stderr may be redirected, skip encoding wrapper
        pass
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn
import numpy as np

# ============ TORCH SETUP ============
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch/torchaudio not installed.")

# Fix for PyTorch 2.6+ weights_only default change
if TORCH_AVAILABLE:
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

# ============ STYLETTS2 SETUP ============
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
except ImportError:
    pass

try:
    from styletts2 import tts as styletts2_module
    STYLETTS2_AVAILABLE = True
    print("[OK] StyleTTS2 available")
except ImportError:
    STYLETTS2_AVAILABLE = False
    print("[MISSING] StyleTTS2 not available")

# ============ QWEN3-TTS SETUP ============
try:
    from qwen_tts import Qwen3TTSModel
    QWEN_TTS_AVAILABLE = True
    print("[OK] Qwen3-TTS available")
except ImportError:
    QWEN_TTS_AVAILABLE = False
    print("[MISSING] Qwen3-TTS not available")

# ============ AUDIO PROCESSING ============
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# ============ WATERMARKING ============
try:
    from watermarker import LegalWatermarker
    LEGAL_WATERMARKER_AVAILABLE = True
except ImportError:
    LEGAL_WATERMARKER_AVAILABLE = False

class AudioWatermarker:
    """Basic audio watermarking with spread-spectrum encoding - WORKING VERSION"""
    def __init__(self, watermark_key: str = "AI-generated"):
        self.watermark_key = watermark_key
        self.seed = int(hashlib.md5(watermark_key.encode()).hexdigest(), 16) % (2**32)
        self.chip_rate = 100
        self.strength = 0.05  # STRONGER watermark for reliable detection
        
    def _generate_watermark(self, num_samples: int, sample_rate: int) -> np.ndarray:
        """Generate the watermark pattern for given audio length"""
        np.random.seed(self.seed)
        chip_samples = int(sample_rate / self.chip_rate)
        num_chips = int(num_samples / chip_samples) + 1
        chips = np.random.choice([-1, 1], size=num_chips)
        watermark = np.repeat(chips, chip_samples)[:num_samples]
        return watermark
        
    def embed_watermark(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Embed watermark into audio"""
        if len(audio.shape) > 1:
            result = np.copy(audio)
            for i in range(audio.shape[1]):
                result[:, i] = self.embed_watermark(audio[:, i], sample_rate)
            return result
        
        watermark = self._generate_watermark(len(audio), sample_rate)
        alpha = self.strength
        # Embed: add watermark scaled by amplitude with DC offset
        watermarked = audio + alpha * watermark * (np.abs(audio) + 0.1)
        
        max_val = np.max(np.abs(watermarked))
        if max_val > 1.0:
            watermarked = watermarked / max_val * 0.99
            
        return watermarked.astype(np.float32)
    
    def detect_watermark(self, audio: np.ndarray, sample_rate: int) -> Tuple[bool, float, dict]:
        """
        Detect if watermark is present in audio - SIGN-BASED detection.
        
        Uses the sign of audio samples to detect watermark pattern.
        More robust than amplitude-based detection.
        
        Returns:
            (detected, confidence, details)
        """
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Convert to sign (-1, 0, +1)
        audio_sign = np.sign(audio)
        
        # Generate expected watermark
        watermark = self._generate_watermark(len(audio), sample_rate)
        
        if len(audio) != len(watermark):
            min_len = min(len(audio_sign), len(watermark))
            audio_sign = audio_sign[:min_len]
            watermark = watermark[:min_len]
        
        # Chip-level sign agreement detection
        chip_samples = int(sample_rate / self.chip_rate)
        num_chips = len(audio_sign) // chip_samples
        
        if num_chips < 10:
            return False, 0.0, {"method": "sign_correlation", "error": "audio too short"}
        
        # Compute sign agreement per chip
        chip_agreements = []
        for i in range(num_chips):
            start = i * chip_samples
            end = start + chip_samples
            chip_signs = audio_sign[start:end]
            chip_watermark = watermark[start:end]
            
            # Count agreements (same sign) vs disagreements
            agreements = np.sum(chip_signs * chip_watermark > 0)
            total = np.sum(chip_signs != 0)  # Exclude zero crossings
            if total > 0:
                chip_agreements.append(agreements / total)
            else:
                chip_agreements.append(0.5)
        
        chip_agreements = np.array(chip_agreements)
        watermark_chips = watermark[::chip_samples][:num_chips]
        
        # Chips with watermark=+1 should have higher agreement
        pos_mask = watermark_chips > 0
        neg_mask = watermark_chips < 0
        
        pos_agreement = np.mean(chip_agreements[pos_mask]) if np.any(pos_mask) else 0.5
        neg_agreement = np.mean(chip_agreements[neg_mask]) if np.any(neg_mask) else 0.5
        
        # Difference should be positive for watermarked audio
        diff = pos_agreement - neg_agreement
        
        # Statistical test
        pos_std = np.std(chip_agreements[pos_mask]) if np.sum(pos_mask) > 1 else 0.1
        neg_std = np.std(chip_agreements[neg_mask]) if np.sum(neg_mask) > 1 else 0.1
        
        n_pos = np.sum(pos_mask)
        n_neg = np.sum(neg_mask)
        se = np.sqrt((pos_std**2 / n_pos) + (neg_std**2 / n_neg) + 1e-10)
        
        z_score = abs(diff) / se
        
        # Confidence based on z-score (z > 2 means 95% confidence)
        confidence = min(1.0, z_score / 3.0)
        detected = diff > 0.02 and z_score > 1.5  # Positive diff with statistical significance
        
        details = {
            "method": "sign_correlation",
            "pos_agreement": float(pos_agreement),
            "neg_agreement": float(neg_agreement),
            "diff": float(diff),
            "z_score": float(z_score),
        }
        
        return detected, confidence, details


# ============ TTS ENGINE ENUM ============
class TTSEngine(str, Enum):
    STYLETTS2 = "styletts2"
    QWEN3 = "qwen3"


class Qwen3ModelSize(str, Enum):
    SMALL = "0.6B"  # 600M parameter model
    LARGE = "1.7B"  # 1.7B parameter model


# ============ DATA CLASSES ============
@dataclass
class VoiceCreationParams:
    """Parameters for voice creation from reference with comprehensive tuning"""
    # Reference audio (required for voice creation)
    reference_audio_path: Optional[str] = None
    reference_text: Optional[str] = None
    
    # Fine-tuning parameters - Basic
    pitch_shift: float = 0.0  # -1.0 to 1.0
    speed: float = 1.0  # 0.5 to 2.0
    
    # Advanced Voice Characteristics
    warmth: float = 0.5  # 0.0 to 1.0 - warmth/naturalness of voice
    expressiveness: float = 0.5  # 0.0 to 1.0 - emotional variation
    stability: float = 0.5  # 0.0 to 1.0 - consistency vs creativity
    clarity: float = 0.5  # 0.0 to 1.0 - articulation sharpness
    breathiness: float = 0.3  # 0.0 to 1.0 - air in voice
    resonance: float = 0.5  # 0.0 to 1.0 - depth/fullness
    
    # Speech Characteristics
    emotion: str = "neutral"  # neutral, happy, sad, angry, excited, calm
    emphasis: float = 0.5  # 0.0 to 1.0 - word stress intensity
    pauses: float = 0.5  # 0.0 to 1.0 - pause length between phrases
    energy: float = 0.5  # 0.0 to 1.0 - overall vocal energy
    
    # Audio Effects (post-processing)
    reverb: float = 0.0  # 0.0 to 1.0 - room ambiance
    eq_low: float = 0.5  # 0.0 to 1.0 - bass boost/cut
    eq_mid: float = 0.5  # 0.0 to 1.0 - mid range
    eq_high: float = 0.5  # 0.0 to 1.0 - treble
    compression: float = 0.3  # 0.0 to 1.0 - dynamic range compression
    
    # TTS Engine selection
    engine: TTSEngine = TTSEngine.STYLETTS2
    qwen3_model_size: Qwen3ModelSize = Qwen3ModelSize.SMALL
    
    # For QWEN3: Use flash attention
    use_flash_attention: bool = True
    
    # For QWEN3: Voice profile extraction model (1.7B for extraction, 0.6B for playback)
    extraction_model_size: Optional[str] = None  # "1.7B" or None (uses creation model)
    
    # Seed for reproducibility
    seed: Optional[int] = None
    
    # Voice profile for reuse
    voice_profile_id: Optional[str] = None  # Use saved voice profile instead of reference audio


# ============ AUDIO UTILITIES ============
def numpy_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert numpy array to WAV bytes"""
    import wave
    
    if audio_array.dtype != np.int16:
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        audio_array = (audio_array * 32767).astype(np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())
    
    return buffer.getvalue()


def audio_to_numpy(audio_bytes: bytes, format_hint: str = None) -> Tuple[np.ndarray, int]:
    """Convert audio bytes to numpy array. Supports WAV, MP3, and other formats."""
    
    # Try soundfile first (best for WAV, supports some other formats)
    if SOUNDFILE_AVAILABLE:
        try:
            with io.BytesIO(audio_bytes) as buffer:
                audio_array, sample_rate = sf.read(buffer, dtype='float32')
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                print(f"[audio_to_numpy] Loaded via soundfile: sr={sample_rate}, samples={len(audio_array)}")
                return audio_array, sample_rate
        except Exception as e:
            print(f"[audio_to_numpy] soundfile failed: {e}")
    
    # Try librosa for MP3 and other formats (requires soundfile or audioread backend)
    if LIBROSA_AVAILABLE:
        try:
            with io.BytesIO(audio_bytes) as buffer:
                # librosa.load uses soundfile first, then audioread fallback
                audio_array, sample_rate = librosa.load(buffer, sr=None, mono=True)
                print(f"[audio_to_numpy] Loaded via librosa: sr={sample_rate}, samples={len(audio_array)}")
                return audio_array, sample_rate
        except Exception as e:
            print(f"[audio_to_numpy] librosa failed: {e}")
    
    # Final fallback to wave (WAV only)
    try:
        import wave
        with io.BytesIO(audio_bytes) as buffer:
            with wave.open(buffer, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                raw_data = wav_file.readframes(n_frames)
                
                if sample_width == 2:
                    audio_array = np.frombuffer(raw_data, dtype=np.int16)
                elif sample_width == 4:
                    audio_array = np.frombuffer(raw_data, dtype=np.int32)
                else:
                    audio_array = np.frombuffer(raw_data, dtype=np.uint8)
                
                audio_array = audio_array.astype(np.float32) / 32767.0
                if n_channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                
                print(f"[audio_to_numpy] Loaded via wave: sr={sample_rate}, samples={len(audio_array)}")
                return audio_array, sample_rate
    except Exception as e:
        print(f"[audio_to_numpy] wave fallback failed: {e}")
        raise ValueError(f"Could not decode audio. Supported formats: WAV, MP3 (with ffmpeg). Error: {e}")


def resample_audio(audio_array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    if orig_sr == target_sr:
        return audio_array
    
    if LIBROSA_AVAILABLE:
        return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)
    else:
        from scipy import signal
        return signal.resample(audio_array, int(len(audio_array) * target_sr / orig_sr))


def preprocess_text_for_tts(text: str) -> str:
    """
    Preprocess text to handle Unicode characters that may cause encoding issues.
    Converts problematic Unicode characters to ASCII equivalents.
    """
    # Common phonetic character mappings to ASCII equivalents
    phonetic_replacements = {
        # IPA vowels
        '\u025b': 'e',  # Latin Small Letter Open E (ɛ) -> e
        '\u0259': 'e',  # Schwa (ə) -> e
        '\u025c': 'e',  # Reversed Open E (ɜ) -> e
        '\u026a': 'i',  # Latin Small Letter Iota (ɪ) -> i
        '\u028a': 'u',  # Latin Small Letter Upsilon (ʊ) -> u
        '\u0254': 'o',  # Latin Small Letter Open O (ɔ) -> o
        '\u00e6': 'ae', # Latin Small Letter AE (æ) -> ae
        '\u0153': 'oe', # Latin Small Ligature OE (œ) -> oe
        # IPA consonants
        '\u0283': 'sh', # Latin Small Letter Esh (ʃ) -> sh
        '\u0292': 'zh', # Latin Small Letter Ezh (ʒ) -> zh
        '\u03b8': 'th', # Greek Small Letter Theta (θ) -> th
        '\u00f0': 'th', # Latin Small Letter Eth (ð) -> th
        '\u014b': 'ng', # Latin Small Letter Eng (ŋ) -> ng
        # Other common problematic chars
        '\u2019': "'",  # Right Single Quotation Mark -> '
        '\u2018': "'",  # Left Single Quotation Mark -> '
        '\u201c': '"',  # Left Double Quotation Mark -> "
        '\u201d': '"',  # Right Double Quotation Mark -> "
        '\u2013': '-',  # En Dash -> -
        '\u2014': '-',  # Em Dash -> -
        '\u2026': '...', # Horizontal Ellipsis -> ...
    }
    
    # Apply specific replacements first
    for char, replacement in phonetic_replacements.items():
        text = text.replace(char, replacement)
    
    # Normalize remaining Unicode to NFKD form and encode to ASCII, ignoring non-ASCII chars
    try:
        normalized = unicodedata.normalize('NFKD', text)
        ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
        return ascii_text
    except Exception as e:
        print(f"Warning: Text normalization failed ({e}), using original text")
        return text


def apply_pitch_shift(audio: np.ndarray, sample_rate: int, n_steps: float) -> np.ndarray:
    """Apply pitch shifting using librosa"""
    if not LIBROSA_AVAILABLE or abs(n_steps) < 0.1:
        return audio
    
    try:
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
    except Exception as e:
        print(f"Pitch shift failed: {e}")
        return audio


def apply_reverb(audio: np.ndarray, sample_rate: int, amount: float) -> np.ndarray:
    """Apply simple reverb effect using convolution"""
    if amount < 0.01 or not TORCH_AVAILABLE:
        return audio
    
    try:
        # Create simple room impulse response
        decay = int(0.3 * sample_rate)  # 300ms decay
        ir = np.exp(-np.linspace(0, 5, decay)) * amount
        
        # Convolve
        convolved = np.convolve(audio, ir, mode='full')[:len(audio)]
        
        # Mix dry and wet
        mixed = audio * (1 - amount * 0.5) + convolved * (amount * 0.5)
        return mixed / np.max(np.abs(mixed) + 1e-8)
    except Exception as e:
        print(f"Reverb failed: {e}")
        return audio


def apply_eq(audio: np.ndarray, sample_rate: int, low: float, mid: float, high: float) -> np.ndarray:
    """Apply 3-band EQ"""
    if not TORCH_AVAILABLE:
        return audio
    
    try:
        # Simple shelving EQ using FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
        
        # Define bands
        low_cutoff = 250
        high_cutoff = 4000
        
        # Apply gains (0.5 = flat, <0.5 = cut, >0.5 = boost)
        low_gain = 0.5 + (low - 0.5) * 0.6  # ±30%
        mid_gain = 0.5 + (mid - 0.5) * 0.6
        high_gain = 0.5 + (high - 0.5) * 0.6
        
        # Create smooth transitions
        mask_low = np.exp(-((freqs - low_cutoff/2) / low_cutoff) ** 2)
        mask_high = np.exp(-((freqs - high_cutoff) / (sample_rate/4 - high_cutoff)) ** 2)
        mask_mid = 1 - mask_low - mask_high
        
        # Apply gains
        gains = low_gain * mask_low + mid_gain * mask_mid + high_gain * mask_high
        fft *= gains
        
        result = np.fft.irfft(fft, n=len(audio))
        return result / np.max(np.abs(result) + 1e-8)
    except Exception as e:
        print(f"EQ failed: {e}")
        return audio


def apply_compression(audio: np.ndarray, amount: float) -> np.ndarray:
    """Apply dynamic range compression"""
    if amount < 0.01:
        return audio
    
    try:
        # Simple compression
        threshold = 0.5 - amount * 0.3  # 0.2 to 0.5
        ratio = 1 + amount * 4  # 1:1 to 5:1
        
        # Soft knee compression
        abs_audio = np.abs(audio)
        gain_reduction = np.ones_like(audio)
        
        mask = abs_audio > threshold
        gain_reduction[mask] = threshold + (abs_audio[mask] - threshold) / ratio
        gain_reduction[mask] /= abs_audio[mask]
        
        compressed = audio * gain_reduction
        return compressed / (np.max(np.abs(compressed)) + 1e-8)
    except Exception as e:
        print(f"Compression failed: {e}")
        return audio


def apply_voice_effects(audio: np.ndarray, sample_rate: int, params: 'VoiceCreationParams') -> np.ndarray:
    """Apply all voice effects based on params"""
    # Apply reverb
    if params.reverb > 0:
        audio = apply_reverb(audio, sample_rate, params.reverb)
    
    # Apply EQ
    if params.eq_low != 0.5 or params.eq_mid != 0.5 or params.eq_high != 0.5:
        audio = apply_eq(audio, sample_rate, params.eq_low, params.eq_mid, params.eq_high)
    
    # Apply compression
    if params.compression > 0:
        audio = apply_compression(audio, params.compression)
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95
    
    return audio


# ============ UNIFIED TTS MANAGER ============
class UnifiedTTSManager:
    """Manages multiple TTS engines: StyleTTS2 and QWEN3"""
    
    def __init__(self):
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.target_sample_rate = 24000
        
        # Models
        self.styletts2_model = None
        self.qwen3_model = None
        self.qwen3_current_size = None
        
        # Watermarker
        if LEGAL_WATERMARKER_AVAILABLE:
            self.watermarker = LegalWatermarker(user_id="", timestamp="")
        else:
            self.watermarker = AudioWatermarker("AI-generated")
        self.enable_watermarking = True
        
        # Reference audio storage - use environment variable if set (Tauri desktop app)
        env_voices_dir = os.environ.get("MIMIC_VOICES_DIR")
        if env_voices_dir:
            self.voices_dir = env_voices_dir
        else:
            self.voices_dir = os.path.join(os.path.dirname(__file__), "voice_references")
        os.makedirs(self.voices_dir, exist_ok=True)
        
        print(f"UnifiedTTSManager initialized. Device: {self.device}")
        print(f"Available engines: StyleTTS2={STYLETTS2_AVAILABLE}, Qwen3={QWEN_TTS_AVAILABLE}")
    
    def load_styletts2(self):
        """Lazy load StyleTTS2 model"""
        if not STYLETTS2_AVAILABLE:
            raise RuntimeError("StyleTTS2 not installed. Run: pip install styletts2")
        
        if self.styletts2_model is None:
            print("Loading StyleTTS2 model...")
            try:
                self.styletts2_model = styletts2_module.StyleTTS2()
                print("[OK] StyleTTS2 loaded")
            except Exception as e:
                print(f"Failed to load StyleTTS2: {e}")
                raise
        return self.styletts2_model
    
    def load_qwen3(self, model_size: Qwen3ModelSize = Qwen3ModelSize.SMALL, use_flash_attention: bool = True):
        """Lazy load QWEN3-TTS model with specified size"""
        if not QWEN_TTS_AVAILABLE:
            raise RuntimeError("Qwen3-TTS not installed. Run: pip install qwen-tts")
        
        # Unload if different size is loaded
        if self.qwen3_model is not None and self.qwen3_current_size != model_size:
            print(f"Switching Qwen3 model from {self.qwen3_current_size} to {model_size}")
            del self.qwen3_model
            self.qwen3_model = None
            import gc
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.qwen3_model is None:
            model_name = f"Qwen/Qwen3-TTS-12Hz-{model_size.value}-Base"
            print(f"Loading Qwen3-TTS model: {model_name}")
            print(f"Flash Attention: {use_flash_attention}")
            
            try:
                # Clear cache before loading
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"GPU memory before load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
                # Configure flash attention and dtype
                # Flash Attention requires fp16 or bf16, not float32
                if use_flash_attention and TORCH_AVAILABLE and torch.cuda.is_available():
                    attn_implementation = "flash_attention_2"
                    dtype = torch.bfloat16  # bfloat16 is more stable than float16
                    print(f"Using Flash Attention with bfloat16")
                else:
                    attn_implementation = None
                    dtype = torch.float32
                    print(f"Using standard attention with float32")
                
                self.qwen3_model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map=self.device,
                    torch_dtype=dtype,
                    attn_implementation=attn_implementation,
                )
                self.qwen3_current_size = model_size
                print(f"[OK] Qwen3-TTS ({model_size.value}) loaded")
                
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    
            except Exception as e:
                print(f"Failed to load Qwen3-TTS: {e}")
                # Fallback without flash attention
                if use_flash_attention:
                    print("Retrying without flash attention...")
                    return self.load_qwen3(model_size, use_flash_attention=False)
                raise
        
        return self.qwen3_model
    
    def create_voice_styletts2(
        self, 
        text: str, 
        params: VoiceCreationParams
    ) -> Tuple[np.ndarray, int]:
        """Create voice using StyleTTS2 from reference audio"""
        model = self.load_styletts2()
        
        # Preprocess text to handle Unicode characters that cause encoding issues
        text = preprocess_text_for_tts(text)
        
        if not params.reference_audio_path or not os.path.exists(params.reference_audio_path):
            # No reference - use default voice
            print("StyleTTS2: Using default voice (no reference)")
            audio_output = model.inference(
                text=text,
                diffusion_steps=5,
                embedding_scale=1.0
            )
        else:
            # Use reference audio for voice creation
            print(f"StyleTTS2: Creating voice from reference: {params.reference_audio_path}")
            
            # Adjust diffusion steps based on pitch shift
            diffusion_steps = 5
            embedding_scale = 1.0
            
            audio_output = model.inference(
                text=text,
                target_voice_path=params.reference_audio_path,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
                alpha=0.3,
                beta=0.7
            )
        
        # Apply pitch shift if specified
        if abs(params.pitch_shift) > 0.05:
            n_steps = params.pitch_shift * 4  # -4 to +4 semitones
            audio_output = apply_pitch_shift(audio_output, self.target_sample_rate, n_steps)
        
        # Apply speed adjustment
        if abs(params.speed - 1.0) > 0.05:
            new_sample_rate = int(self.target_sample_rate * params.speed)
            audio_output = resample_audio(audio_output, new_sample_rate, self.target_sample_rate)
        
        # Apply all voice effects (reverb, EQ, compression)
        audio_output = apply_voice_effects(audio_output, self.target_sample_rate, params)
        
        return audio_output, self.target_sample_rate
    
    def create_voice_qwen3(
        self,
        text: str,
        params: VoiceCreationParams
    ) -> Tuple[np.ndarray, int]:
        """Create voice using QWEN3-TTS from reference audio"""
        import traceback
        
        try:
            model = self.load_qwen3(params.qwen3_model_size, params.use_flash_attention)
        except Exception as e:
            print(f"Failed to load Qwen3 model: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Qwen3 model loading failed: {e}")
        
        print(f"Qwen3 ({params.qwen3_model_size.value}): Creating voice from reference")
        
        if not params.reference_audio_path or not os.path.exists(params.reference_audio_path):
            raise ValueError("Qwen3 requires reference audio. Please upload or record audio first.")
        
        # Load reference audio
        try:
            with open(params.reference_audio_path, 'rb') as f:
                ref_audio_bytes = f.read()
            ref_audio_array, sample_rate = audio_to_numpy(ref_audio_bytes)
            print(f"Reference audio: shape={ref_audio_array.shape}, sr={sample_rate}, duration={len(ref_audio_array)/sample_rate:.2f}s")
        except Exception as e:
            print(f"Failed to load reference audio: {e}")
            traceback.print_exc()
            raise ValueError(f"Invalid reference audio: {e}")
        
        # Resample to 24kHz if needed (Qwen3 expects 24kHz)
        if sample_rate != 24000:create
            print(f"Resampling from {sample_rate} to 24000 Hz")
            ref_audio_array = resample_audio(ref_audio_array, sample_rate, 24000)
            sample_rate = 24000
        
        # Ensure audio is float32 and normalized
        if ref_audio_array.dtype != np.float32:
            ref_audio_array = ref_audio_array.astype(np.float32)
        
        # Check for valid audiocreate
        if len(ref_audio_array) < 1600:  # At least 0.067 seconds (67ms)
            raise ValueError("Reference audio too short. Minimum 0.1 seconds required.")
        
        if np.max(np.abs(ref_audio_array)) < 0.01:
            raise ValueError("Reference audio appears to be silent.")
        
        # Normalize to prevent clipping issues
        ref_audio_array = ref_audio_array / (np.max(np.abs(ref_audio_array)) + 1e-8)
        
        # Debug: print audio statistics
        print(f"Audio stats: min={ref_audio_array.min():.4f}, max={ref_audio_array.max():.4f}, mean={ref_audio_array.mean():.4f}")
        print(f"Audio dtype: {ref_audio_array.dtype}, shape: {ref_audio_array.shape}")
        
        # Determine if we have reference text
        has_ref_text = params.reference_text and params.reference_text.strip()
        x_vector_only = not has_ref_text
        ref_text = params.reference_text or ""
        
        # Limit reference text length (Qwen3 works best with shorter reference texts)
        MAX_REF_TEXT_LENGTH = 500
        if len(ref_text) > MAX_REF_TEXT_LENGTH:
            print(f"Truncating reference text from {len(ref_text)} to {MAX_REF_TEXT_LENGTH} chars")
            ref_text = ref_text[:MAX_REF_TEXT_LENGTH].rsplit(' ', 1)[0] + "..."
        
        # Limit reference audio length (max 30 seconds to prevent OOM)
        MAX_REF_AUDIO_SAMPLES = 30 * 24000  # 30 seconds at 24kHz
        if len(ref_audio_array) > MAX_REF_AUDIO_SAMPLES:
            print(f"Truncating reference audio from {len(ref_audio_array)/24000:.1f}s to 30s")
            ref_audio_array = ref_audio_array[:MAX_REF_AUDIO_SAMPLES]
        
        # Estimate max tokens - add buffer to prevent cutting off last word
        # Using 2.5 chars per token average, plus 50 token buffer for safety
        estimated_tokens = min(2000, max(150, int(len(text) * 2.5) + 50))
        
        print(f"Generating with x_vector_only={x_vector_only}, has_ref_text={has_ref_text}")
        print(f"Text length: {len(text)}, estimated_tokens: {estimated_tokens}")
        
        # Generate voice
        try:
            wavs, sr = model.generate_voice_create(
                text=text,
                language="English",
                ref_audio=(ref_audio_array, sample_rate),
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only,
                max_new_tokens=estimated_tokens,
            )
        except Exception as e:
            print(f"Qwen3 generate_voice_create failed: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Voice generation failed: {e}")
        
        # Extract audio from response
        if isinstance(wavs, list) and len(wavs) > 0:
            audio_output = wavs[0]
        else:
            audio_output = wavs
        
        if isinstance(audio_output, torch.Tensor):
            audio_output = audio_output.cpu().numpy()
        
        # Validate output
        if audio_output is None or len(audio_output) == 0:
            raise RuntimeError("Qwen3 produced empty audio output")
        
        if np.isnan(audio_output).any() or np.isinf(audio_output).any():
            print("WARNING: Qwen3 output contains NaN/Inf values, cleaning up...")
            audio_output = np.nan_to_num(audio_output, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize output
        max_val = np.max(np.abs(audio_output))
        if max_val > 0:
            audio_output = audio_output / max_val * 0.9  # Leave some headroom
        
        print(f"Output audio: {len(audio_output)} samples, range=[{audio_output.min():.4f}, {audio_output.max():.4f}]")
        
        # Apply pitch shift if specified
        if abs(params.pitch_shift) > 0.05:
            n_steps = params.pitch_shift * 4
            audio_output = apply_pitch_shift(audio_output, sr, n_steps)
        
        # Apply speed adjustment
        if abs(params.speed - 1.0) > 0.05:
            new_sample_rate = int(sr * params.speed)
            audio_output = resample_audio(audio_output, new_sample_rate, sr)
            sr = self.target_sample_rate
        
        # Apply all voice effects (reverb, EQ, compression)
        audio_output = apply_voice_effects(audio_output, sr, params)
        
        return audio_output, sr
    
    def create_voice(
        self,
        text: str,
        params: VoiceCreationParams
    ) -> Tuple[np.ndarray, int]:
        """Unified voice creation - selects appropriate engine"""
        start_time = time.time()
        
        if params.engine == TTSEngine.QWEN3:
            if not QWEN_TTS_AVAILABLE:
                raise RuntimeError("Qwen3-TTS not available. Install with: pip install qwen-tts")
            audio_output, sample_rate = self.create_voice_qwen3(text, params)
        else:
            if not STYLETTS2_AVAILABLE:
                raise RuntimeError("StyleTTS2 not available. Install with: pip install styletts2")
            audio_output, sample_rate = self.create_voice_styletts2(text, params)
        
        # Apply watermark
        if self.enable_watermarking:
            audio_output = self.watermarker.embed_watermark(audio_output, sample_rate)
        
        synthesis_time = time.time() - start_time
        rtf = synthesis_time / (len(audio_output) / sample_rate)
        print(f"Voice creation complete: {len(audio_output)} samples, RTF={rtf:.3f}")
        
        return audio_output, sample_rate
    
    def save_reference_audio(self, audio_data: bytes, voice_id: str) -> str:
        """Save uploaded/recorded reference audio"""
        safe_id = "".join(c for c in voice_id if c.isalnum() or c in ('_', '-')).rstrip()
        ref_path = os.path.join(self.voices_dir, f"{safe_id}_reference.wav")
        create
        # Convert and save as WAV
        audio_array, sample_ratcreateudio_to_numpy(audio_data)
        create
        import scipy.io.wavfile as wavfile
        wavfile.write(ref_path, sample_rate, (audio_array * 32767).astype(np.int16))
        
        print(f"Reference audio saved: {ref_path}")
        return ref_path
    
    def extract_voice_profile(
        self,
        reference_audio_path: str,
        reference_text: str,
        extraction_model_size: str = "1.7B",
        use_flash_attention: bool = True
    ) -> Dict[str, Any]:
        """
        Extract voice profile (speaker embedding) using larger 1.7B model.
        The extracted profile can be used with 0.6B model for faster inference.
        
        This implements Method 1 from the user's specification:
        - Use 1.7B model for high-quality voice feature extraction
        - Save the voice prompt/embedding for reuse
        - Later use with 0.6B model for playback
        """
        if not QWEN_TTS_AVAILABLE:
            raise RuntimeError("Qwen3-TTS not available")
        
        print(f"[VoiceProfile] Extracting voice profile with {extraction_model_size} model...")
        
        # Load the extraction model (1.7B)
        model_size_enum = Qwen3ModelSize.LARGE if extraction_model_size == "1.7B" else Qwen3ModelSize.SMALL
        model = self.load_qwen3(model_size_enum, use_flash_attention)
        
        # Load reference audio
        with open(reference_audio_path, 'rb') as f:
            ref_audio_bytes = f.read()
        ref_audio_array, sample_rate = audio_to_numpy(ref_audio_bytes)
        
        # Resample to 24kHz if needed
        if sample_rate != 24000:
            ref_audio_array = resample_audio(ref_audio_array, sample_rate, 24000)
            sample_rate = 24000
        
        # Normalize
        ref_audio_array = ref_audio_array.astype(np.float32)
        ref_audio_array = ref_audio_array / (np.max(np.abs(ref_audio_array)) + 1e-8)
        
        # Create voice create prompt using Qwen3
        try:
            # Use create_voice_create_prompt to extract the speaker representation
            voice_prompt = model.create_voice_create_prompt(
                ref_audio=(ref_audio_array, sample_rate),
                ref_text=reference_text,
                x_vector_only_mode=False  # Use full prompt for better quality
            )
            
            print(f"[VoiceProfile] Extracted voice prompt with {len(voice_prompt)} items")
            
            # Convert to serializable format
            profile_data = {
                'extraction_model': extraction_model_size,
                'sample_rate': sample_rate,
                'reference_text': reference_text,
                'prompt_items': []
            }
            
            # Extract tensors from prompt items
            for idx, item in enumerate(voice_prompt):
                item_data = {'index': idx}
                if hasattr(item, 'speaker_embed') and item.speaker_embed is not None:
                    item_data['speaker_embed'] = item.speaker_embed.cpu().numpy().tolist()
                if hasattr(item, 'codec_tokens') and item.codec_tokens is not None:
                    item_data['codec_tokens'] = [t.cpu().numpy().tolist() if hasattr(t, 'cpu') else t for t in item.codec_tokens]
                profile_data['prompt_items'].append(item_data)
            
            return profile_data
            
        except Exception as e:
            print(f"[VoiceProfile] Extraction failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Voice profile extraction failed: {e}")
    
    def save_voice_profile(
        self,
        persona_id: str,
        profile_data: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save voice profile to disk"""
        from voice_profile_manager import get_voice_profile_manager
        
        manager = get_voice_profile_manager(self.voices_dir)
        
        # Create voice profile entry
        success, message = manager.enroll_voice(
            persona_id=persona_id,
            reference_audio=np.array([]),  # Empty - we use profile data instead
            reference_text=profile_data.get('reference_text', ''),
            engine="qwen3",
            model_size=profile_data.get('extraction_model', '0.6B'),
            params=params or {}
        )
        
        if success:
            # Save the actual profile data
            profile_path = self.voices_dir / f"{persona_id}_voice_profile.json"
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            print(f"[VoiceProfile] Saved to {profile_path}")
            return str(profile_path)
        else:
            raise RuntimeError(f"Failed to save voice profile: {message}")
    
    def load_voice_profile(self, persona_id: str) -> Optional[Dict[str, Any]]:
        """Load voice profile from disk"""
        profile_path = self.voices_dir /createersona_id}_voice_profile.json"
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                recreatejson.load(f)
        return None
    
    def create_voice_with_profile(
        self,
        text: str,
        profile_data: Dict[str, Any],
        playback_model_size: str = "0.6B",
        params: Optional[VoiceCreationParams] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate voice using saved profile with 0.6B model (faster).
        Uses voice profile extracted with 1.7B model for quality.
        """
        if not QWEN_TTS_AVAILABLE:
            raise RuntimeError("Qwen3-TTS not available")
        
        print(f"[VoiceProfile] Creating voice with 0.6B model using saved profile...")
        
        # Load the playback model (0.6B for speed)
        model_size_enum = Qwen3ModelSize.SMALL if playback_model_size == "0.6B" else Qwen3ModelSize.LARGE
        model = self.load_qwen3(model_size_enum, params.use_flash_attention if params else True)
        
        # Reconstruct voice prompt from profile data
        prompt_items = []
        for item_data in profile_data.get('prompt_items', []):
            # Reconstruct the prompt item
            # This is a simplified version - actual implementation depends on Qwen3's internal structure
            speaker_embed = None
            if 'speaker_embed' in item_data:
                speaker_embed = torch.tensor(item_data['speaker_embed'])
            
            codec_tokens = None
            if 'codec_tokens' in item_data:
                codec_tokens = [torch.tensor(t) for t in item_data['codec_tokens']]
            
            # Create a simple object to hold the data
            class PromptItem:
                pass
            
            item = PromptItem()
            item.speaker_embed = speaker_embed
            item.codec_tokens = codec_tokens
            prompt_items.append(item)
        
        # Generate with the loaded profile
        wavs, sr = model.generate_voice_create(
            text=text,
            language="English",
            voice_create_prompt=prompt_items,
            max_new_tokens=min(2000, max(100, len(text) * 2)),
        )
        
        # Extract and process audio
        if isinstance(wavs, list) and len(wavs) > 0:
            audio_output = wavs[0]
        else:
            audio_output = wavs
        
        if isinstance(audio_output, torch.Tensor):
            audio_output = audio_output.cpu().numpy()
        
        # Apply effects
        if params:
            if abs(params.pitch_shift) > 0.05:
                audio_output = apply_pitch_shift(audio_output, sr, params.pitch_shift * 4)
            if abs(params.speed - 1.0) > 0.05:
                new_sr = int(sr * params.speed)
                audio_output = resample_audio(audio_output, new_sr, sr)
            audio_output = apply_voice_effects(audio_output, sr, params)
        
        return audio_output, sr
    
    def unload_models(self):
        """Unload all models to free memory"""
        if self.styletts2_model:
            del self.styletts2_model
            self.styletts2_model = None
        if self.qwen3_model:
            del self.qwen3_model
            self.qwen3_model = None
        
        import gc
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("All models unloaded")


# Global manager
manager = UnifiedTTSManager()


# ============ PYDANTIC MODELS ============
class VoiceCreationRequest(BaseModel):
    """Request to create voice from reference with comprehensive tuning"""
    text: str = Field(..., description="Text to synthesize")
    reference_audio: Optional[str] = Field(None, description="Base64 encoded reference audio (optional)")
    reference_text: Optional[str] = Field(None, description="Transcript of reference audio (optional)")
    
    # Basic voice tuning
    pitch_shift: float = Field(default=0.0, ge=-1.0, le=1.0, description="Pitch shift -1.0 to 1.0")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speed 0.5x to 2.0x")
    
    # Advanced voice characteristics
    warmth: float = Field(default=0.5, ge=0.0, le=1.0, description="Voice warmth 0.0 to 1.0")
    expressiveness: float = Field(default=0.5, ge=0.0, le=1.0, description="Emotional variation 0.0 to 1.0")
    stability: float = Field(default=0.5, ge=0.0, le=1.0, description="Consistency vs creativity 0.0 to 1.0")
    clarity: float = Field(default=0.5, ge=0.0, le=1.0, description="Articulation sharpness 0.0 to 1.0")
    breathiness: float = Field(default=0.3, ge=0.0, le=1.0, description="Air in voice 0.0 to 1.0")
    resonance: float = Field(default=0.5, ge=0.0, le=1.0, description="Depth/fullness 0.0 to 1.0")
    
    # Speech characteristics
    emotion: str = Field(default="neutral", description="Emotion: neutral, happy, sad, angry, excited, calm")
    emphasis: float = Field(default=0.5, ge=0.0, le=1.0, description="Word stress intensity 0.0 to 1.0")
    pauses: float = Field(default=0.5, ge=0.0, le=1.0, description="Pause length between phrases 0.0 to 1.0")
    energy: float = Field(default=0.5, ge=0.0, le=1.0, description="Overall vocal energy 0.0 to 1.0")
    
    # Audio effects (post-processing)
    reverb: float = Field(default=0.0, ge=0.0, le=1.0, description="Room ambiance 0.0 to 1.0")
    eq_low: float = Field(default=0.5, ge=0.0, le=1.0, description="Bass boost/cut 0.0 to 1.0")
    eq_mid: float = Field(default=0.5, ge=0.0, le=1.0, description="Mid range 0.0 to 1.0")
    eq_high: float = Field(default=0.5, ge=0.0, le=1.0, description="Treble 0.0 to 1.0")
    compression: float = Field(default=0.3, ge=0.0, le=1.0, description="Dynamic range compression 0.0 to 1.0")
    
    # Engine selection
    engine: str = Field(default="styletts2", description="TTS engine: styletts2 or qwen3")
    qwen3_model_size: str = Field(default="0.6B", description="Qwen3 model size: 0.6B or 1.7B")
    use_flash_attention: bool = Field(default=True, description="Use flash attention for Qwen3")
    
    # Voice profile extraction
    extraction_model_size: Optional[str] = Field(default=None, description="Model for extraction: 1.7B for quality")
    save_voice_profile: bool = Field(default=False, description="Save as voice profile for reuse")
    persona_id: Optional[str] = Field(default=None, description="Persona ID for voice profile")
    
    # Reproducibility
    seed: Optional[int] = Field(None, description="Random seed")


class VoiceReferenceRequest(BaseModel):
    """Request to upload/save reference audio"""
    voice_id: str = Field(..., description="Unique identifier for this voice")
    audio_data: str = Field(..., description="Base64 encoded audio")
    reference_text: Optional[str] = Field(None, description="Transcript of the audio")


class TTSResponse(BaseModel):
    """Response with synthesized audio"""
    audio_data: str = Field(..., description="Base64 encoded WAV")
    duration_ms: int = Field(..., description="Audio duration in milliseconds")
    sample_rate: int = Field(default=24000)
    engine_used: str = Field(..., description="Which TTS engine was used")


class EngineStatusResponse(BaseModel):
    """Status of TTS engines"""
    styletts2_available: bool
    qwen3_available: bool
    cuda_available: bool
    current_engine: str
    qwen3_loaded_size: Optional[str]


# ============ FASTAPI APP ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    print("=" * 60)
    print("Mimic AI Unified TTS Backend Starting...")
    print(f"StyleTTS2: {'OK' if STYLETTS2_AVAILABLE else 'MISSING'}")
    print(f"Qwen3-TTS: {'OK' if QWEN_TTS_AVAILABLE else 'MISSING'}")
    print(f"CUDA: {'OK' if TORCH_AVAILABLE and torch.cuda.is_available() else 'MISSING'}")
    print("=" * 60)
    yield
    print("Shutting down TTS backend...")
    manager.unload_models()


app = FastAPI(
    title="Mimic AI Unified TTS Backend",
    description="Voice creation from reference audio using StyleTTS2 or Qwen3-TTS",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "Validation error"}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = {}
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "gpu_memory_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        }
    
    return {
        "status": "healthy",
        "styletts2_available": STYLETTS2_AVAILABLE,
        "qwen3_available": QWEN_TTS_AVAILABLE,
        "cuda_available": TORCH_AVAILABLE and torch.cuda.is_available(),
        **gpu_info,
    }


@app.get("/api/engines/status", response_model=EngineStatusResponse)
async def engine_status():
    """Get status of TTS engines"""
    return EngineStatusResponse(
        styletts2_available=STYLETTS2_AVAILABLE,
        qwen3_available=QWEN_TTS_AVAILABLE,
        cuda_available=TORCH_AVAILABLE and torch.cuda.is_available(),
        current_engine="styletts2" if manager.styletts2_model else ("qwen3" if manager.qwen3_model else "none"),
        qwen3_loaded_size=manager.qwen3_current_size.value if manager.qwen3_current_size else None
    )


@app.post("/api/voice/upload-reference")
async def upload_reference(request: VoiceReferenceRequest):
    """
    Upload reference audio for voice creation.
    This is NOT voice cloning - it's creating a synthetic voice inspired by the reference.
    """
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_data)
        print(f"Received reference audio for '{request.voice_id}': {len(audio_bytes)} bytes")
        
        # Save reference audio
        ref_path = manager.save_reference_audio(audio_bytes, request.voice_id)
        
        return {
            "status": "success",
            "voice_id": request.voice_id,
            "reference_path": ref_path,
            "message": "Reference audio saved. Use with /api/voice/create endpoint."
        }
    
    except Exception as e:
        print(f"Failed to save reference audio: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/create", response_model=TTSResponse)
async def create_voice(request: VoiceCreationRequest):
    """
    Create voice from text using reference audio.
    Supports both StyleTTS2 and Qwen3-TTS engines.
    """
    # Validate engine selection
    engine = request.engine.lower()
    if engine not in ["styletts2", "qwen3"]:
        raise HTTPException(status_code=400, detail=f"Invalid engine: {engine}. Use 'styletts2' or 'qwen3'.")
    
    # Check engine availability
    if engine == "qwen3" and not QWEN_TTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Qwen3-TTS not available")
    if engine == "styletts2" and not STYLETTS2_AVAILABLE:
        raise HTTPException(status_code=503, detail="StyleTTS2 not available")
    
    # Parse Qwen3 model size
    qwen_size = Qwen3ModelSize.SMALL
    if request.qwen3_model_size == "1.7B":
        qwen_size = Qwen3ModelSize.LARGE
    
    # Preprocess text to handle Unicode characters BEFORE any operations
    # This prevents 'charmap' codec errors when printing or processing
    original_text = request.text
    safe_text = preprocess_text_for_tts(original_text)
    if safe_text != original_text:
        print(f"[TTS] Text preprocessed: removed Unicode characters")
    
    try:
        # Save reference audio if provided
        ref_path = None
        if request.reference_audio:
            try:
                audio_bytes = base64.b64decode(request.reference_audio)
                print(f"Received reference audio: {len(audio_bytes)} bytes, engine={engine}")
                # Create temporary voice ID based on hash
                voice_hash = hashlib.md5(audio_bytes).hexdigest()[:12]
                ref_path = manager.save_reference_audio(audio_bytes, f"temp_{voice_hash}")
                print(f"Reference audio saved to: {ref_path}")
            except Exception as e:
                print(f"Failed to decode/save reference audio: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=400, detail=f"Invalid reference audio: {e}")
        
        # Create parameters
        print(f"Creating voice with ref_path={ref_path}, engine={engine}")
        params = VoiceCreationParams(
            reference_audio_path=ref_path,
            reference_text=request.reference_text,
            pitch_shift=request.pitch_shift,
            speed=request.speed,
            engine=TTSEngine.QWEN3 if engine == "qwen3" else TTSEngine.STYLETTS2,
            qwen3_model_size=qwen_size,
            use_flash_attention=request.use_flash_attention,
            seed=request.seed
        )
        
        # Generate voice (using preprocessed text)
        audio_output, sample_rate = manager.create_voice(safe_text, params)
        
        # Convert to WAV bytes
        wav_bytes = numpy_to_wav_bytes(audio_output, sample_rate)
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        duration_ms = int(len(audio_output) / sample_rate * 1000)
        
        return TTSResponse(
            audio_data=audio_base64,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            engine_used=engine
        )
    
    except Exception as e:
        print(f"Voice creation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts/default")
async def default_tts(request: dict):
    """Default TTS without reference (StyleTTS2 only)"""
    text = request.get("text", "")
    speed = request.get("speed", 1.0)
    
    if not STYLETTS2_AVAILABLE:
        raise HTTPException(status_code=503, detail="StyleTTS2 not available")
    
    # Preprocess text to handle Unicode characters
    safe_text = preprocess_text_for_tts(text)
    if safe_text != text:
        print(f"[TTS] Text preprocessed: removed Unicode characters")
    
    try:
        params = VoiceCreationParams(
            engine=TTSEngine.STYLETTS2,
            speed=speed
        )
        audio_output, sample_rate = manager.create_voice_styletts2(safe_text, params)
        
        # Apply watermark
        if manager.enable_watermarking:
            audio_output = manager.watermarker.embed_watermark(audio_output, sample_rate)
        
        wav_bytes = numpy_to_wav_bytes(audio_output, sample_rate)
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        duration_ms = int(len(audio_output) / sample_rate * 1000)
        
        return TTSResponse(
            audio_data=audio_base64,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            engine_used="styletts2"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload-models")
async def unload_models():
    """Unload all models to free GPU memory"""
    manager.unload_models()
    return {"status": "success", "message": "All models unloaded"}


@app.post("/api/watermark/detect")
async def detect_watermark(request: dict):
    """Detect AI watermark in audio. Accepts various formats (WAV, MP3, etc.)."""
    try:
        # Validate request
        audio_data_b64 = request.get("audio_data", "")
        if not audio_data_b64:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        # Get format hint if provided
        format_hint = request.get("format", "auto")
        
        print(f"[Watermark Detect] Received request, format_hint={format_hint}, data_length={len(audio_data_b64)}")
        
        # Decode base64
        try:
            audio_bytes = base64.b64decode(audio_data_b64)
            print(f"[Watermark Detect] Decoded {len(audio_bytes)} bytes")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {e}")
        
        if len(audio_bytes) < 100:
            raise HTTPException(status_code=400, detail="Audio data too small (less than 100 bytes)")
        
        # Convert to numpy array (supports WAV, MP3, etc.)
        try:
            audio_array, sample_rate = audio_to_numpy(audio_bytes, format_hint)
            print(f"[Watermark Detect] Audio loaded: sr={sample_rate}, duration={len(audio_array)/sample_rate:.2f}s")
        except Exception as e:
            print(f"[Watermark Detect] Audio conversion failed: {e}")
            raise HTTPException(status_code=400, detail=f"Could not process audio file. Ensure it's a valid audio format (WAV, MP3). Error: {e}")
        
        # Validate audio
        if len(audio_array) < 1000:
            raise HTTPException(status_code=400, detail="Audio too short (less than ~0.1 seconds)")
        
        # Run watermark detection using the same watermarker type used for embedding
        # The TTSManager uses AudioWatermarker for embedding, so we use it for detection too
        print(f"[Watermark Detect] Running detection with AudioWatermarker...")
        detector = AudioWatermarker("AI-generated")
        detected, confidence, details = detector.detect_watermark(audio_array, sample_rate)
        
        print(f"[Watermark Detect] Result: detected={detected}, confidence={confidence:.2%}")
        
        # Build response message
        if detected:
            message = f"AI-generated watermark DETECTED ({confidence:.0%} confidence)"
            if details.get('layers_detected', 0) >= 2:
                message += f" - {details['layers_detected']}/{details.get('total_layers', 3)} verification layers confirmed"
        else:
            message = f"No AI watermark detected ({confidence:.0%} confidence)"
            if confidence > 0.3:
                message += " - Some signal artifacts present but insufficient for detection"
        
        return {
            "detected": bool(detected),
            "confidence": float(confidence),
            "details": details,
            "message": message,
            "sample_rate": sample_rate,
            "duration_seconds": round(len(audio_array) / sample_rate, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[Watermark Detect] Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Watermark detection failed: {str(e)}")


# SearXNG Local Search - Privacy-focused, no API keys required
try:
    import requests
    SEARXNG_AVAILABLE = True
except ImportError:
    SEARXNG_AVAILABLE = False

SEARXNG_URL = os.environ.get('SEARXNG_URL', 'http://localhost:8080')


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]] = []
    answers: List[str] = []
    infobox: Optional[Dict[str, str]] = None
    available: bool = False


@app.get("/api/search/status")
async def search_status():
    """Check if SearXNG is available"""
    if not SEARXNG_AVAILABLE:
        return {'available': False, 'message': 'requests library not installed'}
    
    try:
        # SearXNG doesn't have a standard health endpoint, check the base URL
        response = requests.get(
            f"{SEARXNG_URL}/", 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'},
            timeout=5,
            allow_redirects=True
        )
        # SearXNG returns 200 for the main page
        available = response.status_code == 200
        return {
            'available': available,
            'url': SEARXNG_URL,
            'message': 'SearXNG is running' if available else 'SearXNG not found on localhost:8080'
        }
    except Exception as e:
        return {
            'available': False,
            'url': SEARXNG_URL,
            'message': f'SearXNG not found. Run: docker run -d -p 8080:8080 searxng/searxng ({str(e)})'
        }


@app.post("/api/search", response_model=SearchResponse)
async def web_search(request: SearchRequest):
    """Proxy search to local SearXNG instance"""
    if not SEARXNG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Search not available")
    
    query = request.query
    print(f"[SearXNG] Searching: {query}")
    
    try:
        # Try JSON format first (for properly configured SearXNG)
        params = {
            'q': query,
            'format': 'json',
            'language': 'en-US',
            'safesearch': '0',
            'categories': 'general'
        }
        
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        
        print(f"[SearXNG] Querying: {SEARXNG_URL}/search?q={query}")
        
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params=params,
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 403:
            # JSON API blocked by bot detection, fall back to HTML parsing
            print(f"[SearXNG] JSON blocked (403), falling back to HTML parsing")
            return await _search_html_fallback(query)
        
        if response.status_code != 200:
            print(f"[SearXNG] Error {response.status_code}: {response.text[:200]}")
            raise HTTPException(status_code=502, detail=f"SearXNG error: {response.status_code}")
        
        data = response.json()
        
        return SearchResponse(
            query=query,
            results=data.get('results', [])[:10],
            answers=data.get('answers', []),
            infobox=data.get('infoboxes', [{}])[0] if data.get('infoboxes') else None,
            available=True
        )
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="SearXNG not running on localhost:8080")
    except Exception as e:
        print(f"[SearXNG] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _search_html_fallback(query: str) -> SearchResponse:
    """Fallback to HTML parsing when JSON API is blocked"""
    try:
        params = {
            'q': query,
            'language': 'en-US',
            'safesearch': '0',
            'categories': 'general'
        }
        
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0'
        }
        
        print(f"[SearXNG] HTML fallback: {SEARXNG_URL}/search?q={query}")
        
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params=params,
            headers=headers,
            timeout=15
        )
        
        if response.status_code != 200:
            print(f"[SearXNG] HTML fallback error {response.status_code}")
            raise HTTPException(status_code=502, detail=f"SearXNG HTML error: {response.status_code}")
        
        # Parse HTML to extract results
        from html.parser import HTMLParser
        
        class SearxResultParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current_result = {}
                self.in_result = False
                self.in_title = False
                self.in_url = False
                self.in_content = False
                self.current_tag = None
                
            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                self.current_tag = tag
                
                # Look for result articles
                if tag == 'article' and attrs_dict.get('class', '').startswith('result'):
                    self.in_result = True
                    self.current_result = {}
                    
                # Title and URL in h3 > a
                if self.in_result and tag == 'a' and attrs_dict.get('href', '').startswith('http'):
                    if 'result-header' in str(attrs_dict.get('class', '')) or not self.current_result.get('url'):
                        self.current_result['url'] = attrs_dict.get('href', '')
                        self.in_title = True
                        
            def handle_data(self, data):
                if self.in_result and self.in_title and data.strip():
                    if 'title' not in self.current_result:
                        self.current_result['title'] = data.strip()
                    else:
                        self.current_result['title'] += data.strip()
                        
                # Look for content in result-content or result-abstract
                if self.in_result and not self.in_title and data.strip():
                    if 'content' not in self.current_result:
                        self.current_result['content'] = data.strip()
                    else:
                        self.current_result['content'] += ' ' + data.strip()
                        
            def handle_endtag(self, tag):
                if tag == 'a' and self.in_title:
                    self.in_title = False
                if tag == 'article' and self.in_result:
                    if self.current_result.get('title') and self.current_result.get('url'):
                        self.results.append(self.current_result)
                    self.in_result = False
                    self.current_result = {}
        
        # Try simple regex-based parsing first
        import re
        html = response.text
        
        results = []
        
        # Pattern 1: article.result with h3 > a for title/url and p for content
        result_pattern = r'<article[^>]*class="result[^"]*"[^>]*>.*?<h3[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?</h3>(?:.*?<p[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</p>)?.*?</article>'
        matches = re.findall(result_pattern, html, re.DOTALL | re.IGNORECASE)
        
        for url, title_html, content_html in matches[:10]:
            # Clean up HTML tags from title and content
            title = re.sub(r'<[^>]+>', '', title_html).strip()
            content = re.sub(r'<[^>]+>', '', content_html).strip() if content_html else ''
            
            if title and url.startswith('http'):
                results.append({
                    'title': title,
                    'url': url,
                    'content': content,
                    'engine': 'searxng'
                })
        
        # If no results, try simpler pattern
        if not results:
            # Look for any link with title
            link_pattern = r'<a[^>]*href="(https?://[^"]+)"[^>]*>([^<]{10,200})</a>'
            matches = re.findall(link_pattern, html, re.IGNORECASE)
            for url, title in matches[:10]:
                title = re.sub(r'<[^>]+>', '', title).strip()
                if title and len(title) > 10:
                    results.append({
                        'title': title,
                        'url': url,
                        'content': '',
                        'engine': 'searxng'
                    })
        
        print(f"[SearXNG] HTML fallback found {len(results)} results")
        
        return SearchResponse(
            query=query,
            results=results,
            answers=[],
            infobox=None,
            available=True
        )
        
    except Exception as e:
        print(f"[SearXNG] HTML fallback error: {e}")
        raise HTTPException(status_code=502, detail=f"SearXNG HTML fallback failed: {str(e)}")


# ============ VOICE PROFILE & STREAMING ENDPOINTS ============
from voice_profile_manager import VoiceProfileManager, get_voice_profile_manager
from streaming_tts import StreamingTTS, chunk_text_intelligently

# Initialize voice profile manager
profile_manager = get_voice_profile_manager()


class VoiceProfileRequest(BaseModel):
    """Request to enroll a voice profile"""
    persona_id: str
    reference_audio: str  # Base64 encoded
    reference_text: str
    engine: str = "qwen3"
    model_size: str = "0.6B"
    params: Optional[Dict[str, Any]] = None


@app.post("/api/voice/enroll")
async def enroll_voice(request: VoiceProfileRequest):
    """
    Enroll a voice profile from reference audio.
    Extracts and saves the voice embedding for future use.
    """
    try:
        # Decode reference audio
        audio_bytes = base64.b64decode(request.reference_audio)
        audio_array, sample_rate = audio_to_numpy(audio_bytes)
        
        # Resample to 24kHz if needed
        if sample_rate != 24000:
            audio_array = resample_audio(audio_array, sample_rate, 24000)
        
        # Get Qwen model if using qwen3
        qwen_model = None
        if request.engine == "qwen3":
            if not QWEN_TTS_AVAILABLE:
                raise HTTPException(status_code=503, detail="Qwen3-TTS not available")
            qwen_size = Qwen3ModelSize.SMALL if request.model_size == "0.6B" else Qwen3ModelSize.LARGE
            qwen_model = manager.load_qwen3(qwen_size, use_flash_attention=True)
        
        # Enroll the voice
        success, message = profile_manager.enroll_voice(
            persona_id=request.persona_id,
            reference_audio=audio_array,
            reference_text=request.reference_text,
            engine=request.engine,
            model_size=request.model_size,
            params=request.params or {},
            qwen_model=qwen_model
        )
        
        if success:
            return {
                "status": "success",
                "persona_id": request.persona_id,
                "message": "Voice profile enrolled successfully"
            }
        else:
            raise HTTPException(status_code=500, detail=message)
            
    except Exception as e:
        print(f"Voice enrollment failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voice/profile/{persona_id}")
async def get_voice_profile(persona_id: str):
    """Get voice profile information for a persona"""
    profile = profile_manager.load_voice_profile(persona_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    
    return {
        "exists": True,
        "persona_id": profile.persona_id,
        "engine": profile.engine,
        "model_size": profile.model_size,
        "reference_text": profile.reference_text,
        "params": profile.params
    }


@app.delete("/api/voice/profile/{persona_id}")
async def delete_voice_profile(persona_id: str):
    """Delete a voice profile"""
    success = profile_manager.delete_voice_profile(persona_id)
    if success:
        return {"status": "success", "message": "Voice profile deleted"}
    else:
        raise HTTPException(status_code=404, detail="Voice profile not found")


@app.get("/api/voice/profiles")
async def list_voice_profiles():
    """List all enrolled voice profiles"""
    profiles = profile_manager.list_profiles()
    return {
        "profiles": [
            {
                "persona_id": p.persona_id,
                "engine": p.engine,
                "model_size": p.model_size,
                "created_at": p.created_at
            }
            for p in profiles.values()
        ]
    }


class VoiceProfileExtractionRequest(BaseModel):
    """Request to extract voice profile using 1.7B model"""
    persona_id: str
    reference_audio: str = Field(..., description="Base64 encoded reference audio")
    reference_text: str = Field(..., description="Transcript of reference audio")
    extraction_model_size: str = Field(default="1.7B", description="Model size for extraction: 1.7B")
    use_flash_attention: bool = Field(default=True, description="Use flash attention")
    save_profile: bool = Field(default=True, description="Save the extracted profile")
    voice_params: Optional[Dict[str, Any]] = Field(default=None, description="Voice tuning parameters")


@app.post("/api/voice/extract-profile")
async def extract_voice_profile_endpoint(request: VoiceProfileExtractionRequest):
    """
    Extract voice profile using 1.7B model for high-quality voice capture.
    The extracted profile can be used with 0.6B model for faster playback.
    """
    try:
        print(f"[API] Extracting voice profile for {request.persona_id} using {request.extraction_model_size}")
        
        if not QWEN_TTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Qwen3-TTS not available")
        
        # Decode reference audio
        audio_bytes = base64.b64decode(request.reference_audio)
        audio_array, sample_rate = audio_to_numpy(audio_bytes)
        
        # Save temporarily
        temp_path = os.path.join(manager.voices_dir, f"temp_{request.persona_id}_ref.wav")
        import scipy.io.wavfile as wavfile
        wavfile.write(temp_path, sample_rate, (audio_array * 32767).astype(np.int16))
        
        # Extract voice profile
        profile_data = manager.extract_voice_profile(
            reference_audio_path=temp_path,
            reference_text=request.reference_text,
            extraction_model_size=request.extraction_model_size,
            use_flash_attention=request.use_flash_attention
        )
        
        # Save if requested
        if request.save_profile:
            profile_path = manager.save_voice_profile(
                persona_id=request.persona_id,
                profile_data=profile_data,
                params=request.voice_params or {}
            )
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {
                "status": "success",
                "persona_id": request.persona_id,
                "extraction_model": request.extraction_model_size,
                "profile_path": profile_path,
                "message": f"Voice profile extracted with {request.extraction_model_size} and saved"
            }
        else:
            return {
                "status": "success",
                "profile_data": profile_data,
                "message": f"Voice profile extracted with {request.extraction_model_size} (not saved)"
            }
            
    except Exception as e:
        print(f"[API] Voice profile extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class VoiceProfileGenerateRequest(BaseModel):
    """Request to generate voice using saved profile"""
    text: str
    persona_id: str
    playback_model_size: str = Field(default="0.6B", description="Model size for playback: 0.6B")
    use_flash_attention: bool = Field(default=True)
    # All voice effect parameters
    pitch_shift: float = Field(default=0.0, ge=-1.0, le=1.0)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    warmth: float = Field(default=0.5, ge=0.0, le=1.0)
    expressiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    stability: float = Field(default=0.5, ge=0.0, le=1.0)
    clarity: float = Field(default=0.5, ge=0.0, le=1.0)
    breathiness: float = Field(default=0.3, ge=0.0, le=1.0)
    resonance: float = Field(default=0.5, ge=0.0, le=1.0)
    emotion: str = Field(default="neutral")
    emphasis: float = Field(default=0.5, ge=0.0, le=1.0)
    pauses: float = Field(default=0.5, ge=0.0, le=1.0)
    energy: float = Field(default=0.5, ge=0.0, le=1.0)
    reverb: float = Field(default=0.0, ge=0.0, le=1.0)
    eq_low: float = Field(default=0.5, ge=0.0, le=1.0)
    eq_mid: float = Field(default=0.5, ge=0.0, le=1.0)
    eq_high: float = Field(default=0.5, ge=0.0, le=1.0)
    compression: float = Field(default=0.3, ge=0.0, le=1.0)


@app.post("/api/voice/generate-with-profile")
async def generate_with_profile(request: VoiceProfileGenerateRequest):
    """
    Generate voice using saved voice profile.
    Uses 0.6B model for fast playback of profiles extracted with 1.7B model.
    """
    try:
        print(f"[API] Generating voice with profile for {request.persona_id}")
        
        # Load voice profile
        profile_data = manager.load_voice_profile(request.persona_id)
        if not profile_data:
            raise HTTPException(status_code=404, detail="Voice profile not found. Extract profile first.")
        
        # Create params from request
        params = VoiceCreationParams(
            pitch_shift=request.pitch_shift,
            speed=request.speed,
            warmth=request.warmth,
            expressiveness=request.expressiveness,
            stability=request.stability,
            clarity=request.clarity,
            breathiness=request.breathiness,
            resonance=request.resonance,
            emotion=request.emotion,
            emphasis=request.emphasis,
            pauses=request.pauses,
            energy=request.energy,
            reverb=request.reverb,
            eq_low=request.eq_low,
            eq_mid=request.eq_mid,
            eq_high=request.eq_high,
            compression=request.compression,
            engine=TTSEngine.QWEN3,
            qwen3_model_size=Qwen3ModelSize.SMALL if request.playback_model_size == "0.6B" else Qwen3ModelSize.LARGE,
            use_flash_attention=request.use_flash_attention
        )
        
        # Generate voice with profile
        audio_output, sample_rate = manager.create_voice_with_profile(
            text=request.text,
            profile_data=profile_data,
            playback_model_size=request.playback_model_size,
            params=params
        )
        
        # Apply watermark
        audio_output = manager.watermarker.embed_watermark(audio_output, sample_rate)
        
        # Convert to base64
        audio_base64 = base64.b64encode(numpy_to_wav_bytes(audio_output, sample_rate)).decode('utf-8')
        
        return {
            "audio_data": audio_base64,
            "sample_rate": sample_rate,
            "engine_used": f"qwen3-{request.playback_model_size}",
            "voice_profile_used": True,create
            "duration_seconds": len(audio_output) / sample_rate
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Voice generation with profile failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class StreamingTTSRequest(BaseModel):
    """Request for streaming TTS"""
    text: str
    persona_id: str
    chunk_max_length: int = 200
    crossfade_ms: int = 50


@app.post("/api/tts/stream-chunks")
async def tts_stream_chunks(request: StreamingTTSRequest):
    """
    Generate TTS in chunks and return as separate audio segments.
    This allows the client to implement streaming playback.
    """
    try:
        # Load voice profile
        profile = profile_manager.load_voice_profile(request.persona_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Voice profile not found")
        
        # Load reference audio
        ref_audio, _ = audio_to_numpy(open(profile.reference_audio_path, 'rb').read())
        
        # Chunk the text
        chunks = chunk_text_intelligently(request.text, request.chunk_max_length)
        
        # Generate audio for each chunk
        chunk_audios = []
        for i, chunk_text in enumerate(chunks):
            # Generate using the appropriate engine
            if profile.engine == "qwen3":
                # Load Qwen model
                qwen_size = Qwen3ModelSize.SMALL if profile.model_size == "0.6B" else Qwen3ModelSize.LARGE
                model = manager.load_qwen3(qwen_size, profile.params.get('use_flash_attention', True))
                
                # Generate
                wavs, sr = model.generate_voice_create(
                    text=chunk_text,
                    language="English",
                    ref_audio=(ref_audio, 24000),
                    ref_text=profile.reference_text,
                    x_vector_only_mode=False
                )
                
                audio = wavs[0] if isinstance(wavs, list) else wavs
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
            else:
                # StyleTTS2
                params = VoiceCreationParams(
                    engine=TTSEngine.STYLETTS2,
                    reference_audio_path=profile.reference_audio_path,
                    reference_text=profile.reference_text,
                    **profile.params
                )
                audio, sr = manager.create_voice_styletts2(chunk_text, params)
            
            # Convert to base64
            wav_bytes = numpy_to_wav_bytes(audio, 24000)
            chunk_audios.append({
                "text": chunk_text,
                "audio": base64.b64encode(wav_bytes).decode('utf-8'),
                "index": i,
                "is_first": i == 0,
                "is_last": i == len(chunks) - 1
            })
        
        return {
            "chunks": chunk_audios,
            "total_chunks": len(chunks),
            "engine": profile.engine
        }
        
    except Exception as e:
        print(f"Streaming TTS failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ MEMORY TOOLS API ============
# Secure file access for agent models - Read-only by default, writes require confirmation

# Import memory tools
try:
    from memory_tools import get_memory_tools, MEMORY_TOOLS_SCHEMA, SecurityError
    MEMORY_TOOLS_AVAILABLE = True
except ImportError:
    MEMORY_TOOLS_AVAILABLE = False
    print("Warning: memory_tools module not available")

# Pydantic models for memory tool requests
class MemoryListResponse(BaseModel):
    success: bool
    files: List[Dict[str, Any]] = []
    message: str

class MemoryReadRequest(BaseModel):
    filename: str

class MemoryReadResponse(BaseModel):
    success: bool
    filename: str
    content: str = ""
    message: str

class MemorySearchRequest(BaseModel):
    query: str

class MemorySearchResponse(BaseModel):
    success: bool
    matches: List[Dict[str, Any]] = []
    message: str

class MemoryWriteRequest(BaseModel):
    filename: str
    content: str
    confirm: bool = False  # Safety: must be True to actually write

class MemoryWriteResponse(BaseModel):
    success: bool
    filename: str
    message: str

@app.get("/api/memory/tools", response_model=List[Dict[str, Any]])
async def get_memory_tool_schema():
    """Get the tool schema for agent models (for Ollama tool calling)"""
    if not MEMORY_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Memory tools not available")
    return MEMORY_TOOLS_SCHEMA

@app.get("/api/memory/list", response_model=MemoryListResponse)
async def list_memories():
    """List all files in the memory folder"""
    if not MEMORY_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Memory tools not available")
    
    try:
        tools = get_memory_tools()
        result = tools.list_memories()
        
        # Convert MemoryFile objects to dicts
        files = []
        for f in result.data:
            files.append({
                "name": f.name,
                "path": f.path,
                "size": f.size,
                "modified": f.modified.isoformat() if f.modified else None,
                "preview": f.content_preview
            })
        
        return MemoryListResponse(
            success=result.success,
            files=files,
            message=result.message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/read", response_model=MemoryReadResponse)
async def read_memory(request: MemoryReadRequest):
    """Read a specific memory file"""
    if not MEMORY_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Memory tools not available")
    
    try:
        tools = get_memory_tools()
        result = tools.read_memory(request.filename)
        
        return MemoryReadResponse(
            success=result.success,
            filename=request.filename,
            content=result.data.get("content", "") if result.data else "",
            message=result.message
        )
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/search", response_model=MemorySearchResponse)
async def search_memories(request: MemorySearchRequest):
    """Search memory contents for keywords"""
    if not MEMORY_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Memory tools not available")
    
    try:
        tools = get_memory_tools()
        result = tools.search_memories(request.query)
        
        return MemorySearchResponse(
            success=result.success,
            matches=result.data if result.data else [],
            message=result.message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/write", response_model=MemoryWriteResponse)
async def write_memory(request: MemoryWriteRequest):
    """
    Write content to a memory file.
    REQUIRES confirm=True to actually write (safety check).
    """
    if not MEMORY_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Memory tools not available")
    
    try:
        tools = get_memory_tools()
        result = tools.write_memory(
            filename=request.filename,
            content=request.content,
            confirm=request.confirm
        )
        
        return MemoryWriteResponse(
            success=result.success,
            filename=request.filename,
            message=result.message
        )
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/memory/delete")
async def delete_memory(filename: str, confirm: bool = False):
    """
    Delete a memory file.
    REQUIRES confirm=True to actually delete.
    """
    if not MEMORY_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Memory tools not available")
    
    try:
        tools = get_memory_tools()
        result = tools.delete_memory(filename=filename, confirm=confirm)
        
        return {
            "success": result.success,
            "filename": filename,
            "message": result.message
        }
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ PERSONA RULES API ============
# Persistent rules.md files for each persona

try:
    from persona_rules import get_persona_rules_manager
    PERSONA_RULES_AVAILABLE = True
except ImportError:
    PERSONA_RULES_AVAILABLE = False
    print("Warning: persona_rules module not available")

class PersonaRulesRequest(BaseModel):
    persona_id: str
    content: Optional[str] = None

class PersonaRulesResponse(BaseModel):
    success: bool
    content: Optional[str] = None
    message: str

@app.get("/api/persona/{persona_id}/rules", response_model=PersonaRulesResponse)
async def get_persona_rules(persona_id: str):
    """Get the rules.md content for a persona"""
    if not PERSONA_RULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Persona rules not available")
    
    try:
        manager = get_persona_rules_manager()
        content = manager.get_rules(persona_id)
        
        if content is None:
            return PersonaRulesResponse(
                success=False,
                content=None,
                message=f"No rules found for persona {persona_id}"
            )
        
        return PersonaRulesResponse(
            success=True,
            content=content,
            message="Rules retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/persona/{persona_id}/rules", response_model=PersonaRulesResponse)
async def save_persona_rules(persona_id: str, request: PersonaRulesRequest):
    """Save rules.md content for a persona"""
    if not PERSONA_RULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Persona rules not available")
    
    try:
        manager = get_persona_rules_manager()
        success = manager.save_rules(persona_id, request.content or "")
        
        return PersonaRulesResponse(
            success=success,
            content=request.content,
            message="Rules saved successfully" if success else "Failed to save rules"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/persona/{persona_id}/rules/generate", response_model=PersonaRulesResponse)
async def generate_persona_rules(persona_id: str, request: Dict[str, Any]):
    """Generate rules.md from persona configuration"""
    if not PERSONA_RULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Persona rules not available")
    
    try:
        manager = get_persona_rules_manager()
        success = manager.update_rules_from_persona(persona_id, request)
        
        if success:
            content = manager.get_rules(persona_id)
            return PersonaRulesResponse(
                success=True,
                content=content,
                message="Rules generated successfully"
            )
        else:
            return PersonaRulesResponse(
                success=False,
                content=None,
                message="Failed to generate rules"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ MAIN ENTRY POINT ============

def setup_logging_for_daemon():
    """Setup logging to file when running as a daemon (no console)"""
    import logging
    
    # Check if stdout is a proper console or redirected to null
    try:
        # Try to access stdout.buffer - this fails when stdout is redirected to null
        import sys
        sys.stdout.buffer
        return False  # Console is available, use default logging
    except (AttributeError, IOError):
        # stdout is redirected (e.g., to null), setup file logging
        pass
    
    # Setup logging to file
    log_dir = os.environ.get("MIMIC_DATA_DIR")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "tts_server.log")
    else:
        log_file = "tts_server.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
        ]
    )
    
    # Redirect stdout/stderr to the log file to prevent uvicorn errors
    class LoggerWriter:
        def __init__(self, level):
            self.level = level
        def write(self, message):
            if message and message.strip():
                self.level(message.strip())
        def flush(self):
            pass
        def isatty(self):
            return False
        def fileno(self):
            # Return a dummy file descriptor (can't use -1, so use 1 for stdout-like)
            return 1
    
    # Replace stdout/stderr with our logger wrappers
    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)
    
    return True

def main():
    # Setup logging for daemon mode (no console)
    is_daemon = setup_logging_for_daemon()
    
    print("=" * 60)
    print("Mimic AI Unified TTS Backend")
    print("Voice Creation - NOT Voice Cloning")
    print("Supports: StyleTTS2, Qwen3-TTS (0.6B & 1.7B)")
    print("=" * 60)
    
    # Read configuration from environment variables (set by Tauri)
    port = int(os.environ.get("MIMIC_PORT", "8000"))
    voices_dir = os.environ.get("MIMIC_VOICES_DIR")
    data_dir = os.environ.get("MIMIC_DATA_DIR")
    
    print(f"Port: {port}")
    print(f"Voices directory: {voices_dir or 'default'}")
    print(f"Data directory: {data_dir or 'default'}")
    print()
    print("Starting uvicorn server...")
    print(f"API docs will be at: http://127.0.0.1:{port}/docs")
    print()
    
    try:
        if is_daemon:
            # When running as daemon, disable access logs to avoid stdout issues
            uvicorn.run(
                "tts_server_unified:app",
                host="127.0.0.1",
                port=port,
                reload=False,
                log_level="warning",  # Reduce logging when daemonized
                access_log=False      # Disable access log to avoid stdout issues
            )
        else:
            uvicorn.run(
                "tts_server_unified:app",
                host="127.0.0.1",
                port=port,
                reload=False,
                log_level="info"
            )
    except Exception as e:
        print(f"[ERROR] Server failed to start: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
