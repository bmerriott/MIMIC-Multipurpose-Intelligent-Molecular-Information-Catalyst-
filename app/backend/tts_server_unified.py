"""
Mimic AI - Unified TTS Backend Server
Supports multiple TTS engines: QWEN3-TTS and KittenTTS
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
import re
import gc
from typing import Optional, Union, List, Tuple, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# Force UTF-8 encoding for stdout/stderr on Windows to prevent charmap codec errors
if sys.platform == 'win32':
    import codecs
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except (AttributeError, TypeError):
        pass
    os.environ['PYTHONIOENCODING'] = 'utf-8'

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

if TORCH_AVAILABLE:
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

# ============ NLTK SETUP ============
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

# ============ QWEN3-TTS SETUP ============
try:
    from qwen_tts import Qwen3TTSModel
    QWEN_TTS_AVAILABLE = True
    print("[OK] Qwen3-TTS available")
except ImportError:
    QWEN_TTS_AVAILABLE = False
    print("[MISSING] Qwen3-TTS not available")

# ============ KITTENTTS SETUP ============
# KittenTTS can use GPU if available for better performance
# ONNX Runtime will automatically use CUDA if available

espeak_paths = [
    r"C:\Program Files\eSpeak NG",
    r"C:\Program Files (x86)\eSpeak NG",
]
espeak_found = False
_espeak_dll_path = None

for path in espeak_paths:
    espeak_exe = os.path.join(path, "espeak-ng.exe")
    espeak_dll = os.path.join(path, "libespeak-ng.dll")
    if os.path.exists(espeak_exe):
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_exe
        if os.path.exists(espeak_dll):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_dll
            _espeak_dll_path = path
            print(f"[OK] Found espeak-ng at: {path}")
            espeak_found = True
            break

if sys.platform == 'win32' and _espeak_dll_path and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(_espeak_dll_path)
        print(f"[OK] Added DLL directory: {_espeak_dll_path}")
    except Exception as e:
        print(f"[WARN] Could not add DLL directory: {e}")

KITTENTTS_INSTALLED = False
KITTEN_TTS_AVAILABLE = False
KittenTTS = None

try:
    import importlib.util
    KITTENTTS_INSTALLED = importlib.util.find_spec("kittentts") is not None
    if KITTENTTS_INSTALLED:
        print("[OK] kittentts package found, attempting import...")
        try:
            from phonemizer.backend import EspeakBackend
            _test_backend = EspeakBackend(language="en-us")
            from kittentts import KittenTTS as KT
            KittenTTS = KT
            KITTEN_TTS_AVAILABLE = True
            print("[OK] KittenTTS imported successfully at startup")
        except Exception as e:
            print(f"[WARN] KittenTTS import failed at startup: {e}")
    else:
        print("[MISSING] kittentts not installed - pip install kittentts")
except Exception as e:
    print(f"[MISSING] kittentts check failed: {e}")

# ============ ESPEAK PATH HELPER ============
_espreak_dll_dir_added = False

def ensure_espeak_in_path():
    global _espreak_dll_dir_added
    for path in espeak_paths:
        espeak_exe = os.path.join(path, "espeak-ng.exe")
        espeak_dll = os.path.join(path, "libespeak-ng.dll")
        if os.path.exists(espeak_exe):
            current_path = os.environ.get("PATH", "")
            if path not in current_path:
                os.environ["PATH"] = path + os.pathsep + current_path
            os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_exe
            if os.path.exists(espeak_dll):
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_dll
            data_path = os.path.join(path, "espeak-ng-data")
            if os.path.exists(data_path):
                os.environ["ESPEAK_DATA_PATH"] = data_path
            if sys.platform == 'win32' and not _espreak_dll_dir_added and hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(path)
                    _espreak_dll_dir_added = True
                except:
                    pass
            return True
    return False

def remove_cached_module(module_name):
    modules_to_remove = [name for name in sys.modules.keys() if name == module_name or name.startswith(module_name + ".")]
    for name in modules_to_remove:
        del sys.modules[name]

def reload_phonemizer_with_espeak():
    modules_to_clear = [
        'phonemizer', 'phonemizer.backend', 'phonemizer.backend.espeak',
        'phonemizer.backend.espeak.espeak', 'kittentts'
    ]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
    import gc
    gc.collect()

# ============ AUDIO WATERMARKING SETUP ============
try:
    import wavmark
    import wavmark.models.encoder as watermark_encoder
    WAVMARK_AVAILABLE = True
    print("[OK] WavMark available for audio watermarking")
except ImportError:
    WAVMARK_AVAILABLE = False
    print("[MISSING] WavMark not available - watermarking disabled")

# ============ Pydantic Models ============

class Qwen3ModelSize(str, Enum):
    SMALL = "0.6B"
    LARGE = "1.7B"

class KittenModelSize(str, Enum):
    NANO = "KittenML/kitten-tts-nano-0.8"
    MICRO = "KittenML/kitten-tts-micro-0.8"
    MINI = "KittenML/kitten-tts-mini-0.8"

class TTSEngine(str, Enum):
    QWEN3 = "qwen3"
    KITTEN = "kitten"

class VoiceCreationRequest(BaseModel):
    text: str
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None  # Text spoken in reference audio (required for Qwen3 voice clone)
    engine: str = "qwen3"
    qwen3_model_size: str = "0.6B"
    voice: Optional[str] = None
    kitten_model_size: str = "KittenML/kitten-tts-nano-0.8"
    pitch: int = 0
    speed: float = 1.0
    
    @field_validator('pitch')
    @classmethod
    def validate_pitch(cls, v):
        return max(-50, min(50, v))
    
    @field_validator('speed')
    @classmethod
    def validate_speed(cls, v):
        return max(0.5, min(2.0, v))

class KittenTTSRequest(BaseModel):
    text: str
    voice: str = "Bella"
    model_size: str = "KittenML/kitten-tts-nano-0.8"
    pitch: int = 0
    speed: float = 1.0
    
    @field_validator('pitch')
    @classmethod
    def validate_pitch(cls, v):
        return max(-50, min(50, v))
    
    @field_validator('speed')
    @classmethod
    def validate_speed(cls, v):
        return max(0.5, min(2.0, v))

class VoiceCreationResponse(BaseModel):
    audio_data: str
    sample_rate: int
    duration: float
    format: str = "wav"
    engine: str

class TTSStatusResponse(BaseModel):
    status: str
    models: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    models: Dict[str, Any]
    cuda_available: bool = False
    cuda_devices: int = 0

class SaveVoiceProfileRequest(BaseModel):
    persona_id: str
    reference_audio: str  # Base64 encoded
    reference_text: str
    qwen3_model_size: str = "0.6B"
    
class VoiceProfileResponse(BaseModel):
    status: str
    persona_id: str
    voice_id: str
    message: str

class GenerateVoiceRequest(BaseModel):
    text: str
    voice_id: str  # persona_id or hash
    qwen3_model_size: str = "0.6B"
    speed: float = 1.0
    language: str = "Auto"

# ============ Helper Functions ============

def preprocess_text_for_tts(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF\u2060\u00AD]', '', text)
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('…', '...')
    text = text.replace('\u00A0', ' ')
    return text

def audio_to_numpy(audio_data: bytes) -> Tuple[np.ndarray, int]:
    import wave
    wav_io = io.BytesIO(audio_data)
    with wave.open(wav_io, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(n_frames)
        
        if sample_width == 2:
            audio_array = np.frombuffer(raw_data, dtype=np.int16)
        elif sample_width == 4:
            audio_array = np.frombuffer(raw_data, dtype=np.int32)
        elif sample_width == 1:
            audio_array = np.frombuffer(raw_data, dtype=np.uint8)
            audio_array = (audio_array.astype(np.float32) - 128) / 128.0
            audio_array = (audio_array * 32767).astype(np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        audio_array = audio_array.astype(np.float32) / 32767.0
        if n_channels == 2:
            audio_array = audio_array.reshape(-1, 2).mean(axis=1)
        return audio_array, sample_rate

def numpy_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    import wave
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    return wav_io.getvalue()

@dataclass
class VoiceCreationParams:
    reference_audio: Optional[np.ndarray] = None
    reference_sample_rate: int = 24000
    reference_text: Optional[str] = None  # Text spoken in reference audio (required for Qwen3 voice clone)
    engine: TTSEngine = TTSEngine.QWEN3
    qwen3_model_size: Qwen3ModelSize = Qwen3ModelSize.SMALL
    voice: Optional[str] = None
    kitten_model_size: KittenModelSize = KittenModelSize.NANO
    pitch: int = 0
    speed: float = 1.0
    
    @classmethod
    def from_request(cls, request: VoiceCreationRequest) -> 'VoiceCreationParams':
        params = cls(
            engine=TTSEngine.QWEN3 if request.engine == "qwen3" else TTSEngine.KITTEN,
            qwen3_model_size=Qwen3ModelSize.SMALL if request.qwen3_model_size == "0.6B" else Qwen3ModelSize.LARGE,
            voice=request.voice,
            kitten_model_size=KittenModelSize.NANO if request.kitten_model_size == "KittenML/kitten-tts-nano-0.8" else KittenModelSize.MICRO,
            pitch=request.pitch,
            speed=request.speed,
            reference_text=request.reference_text  # Pass reference text for voice cloning
        )
        if request.reference_audio:
            audio_bytes = base64.b64decode(request.reference_audio)
            params.reference_audio, params.reference_sample_rate = audio_to_numpy(audio_bytes)
        return params

class AudioWatermarker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and WAVMARK_AVAILABLE
        self.model = None
        if self.enabled:
            try:
                self.model = watermark_encoder.Encoder()
                print("[OK] Audio watermarking initialized")
            except Exception as e:
                print(f"[WARN] Failed to initialize watermarking: {e}")
                self.enabled = False
    
    def embed_watermark(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or self.model is None:
            return audio
        try:
            if sample_rate != 16000:
                return audio
            watermarked = self.model.encode(audio, sample_rate)
            return watermarked
        except Exception as e:
            print(f"[WARN] Watermark embedding failed: {e}")
            return audio

class MimicTTSEngine:
    def __init__(self, enable_watermarking: bool = True):
        self.enable_watermarking = enable_watermarking
        self.watermarker = AudioWatermarker(enable_watermarking)
        self.qwen3_models: Dict[Qwen3ModelSize, Any] = {}
        self.qwen3_config: Dict[Qwen3ModelSize, Any] = {}
        self.kitten_models: Dict[KittenModelSize, Any] = {}
        self.voices_dir = os.path.join(tempfile.gettempdir(), "mimic_voices")
        os.makedirs(self.voices_dir, exist_ok=True)
        # Cache for voice clone prompts (key: hash of ref_audio, value: VoiceClonePromptItem)
        self.voice_clone_cache: Dict[str, Any] = {}
        print(f"[OK] MimicTTSEngine initialized (watermarking: {enable_watermarking})")
    
    def unload_qwen3(self, model_size: Qwen3ModelSize = None):
        """Unload Qwen3 model(s) to free VRAM. Call when switching engines or personas."""
        if model_size is None:
            # Unload all Qwen3 models
            for size in list(self.qwen3_models.keys()):
                if size in self.qwen3_models:
                    print(f"[GPU] Unloading Qwen3-TTS {size.value}...")
                    del self.qwen3_models[size]
            self.qwen3_models.clear()
        elif model_size in self.qwen3_models:
            print(f"[GPU] Unloading Qwen3-TTS {model_size.value}...")
            del self.qwen3_models[model_size]
        
        # Clear CUDA cache
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            print(f"[GPU] CUDA cache cleared. Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    def load_qwen3(self, model_size: Qwen3ModelSize = Qwen3ModelSize.SMALL, use_flash_attention: bool = True):
        # Check if we need to unload a different size to save VRAM
        if self.qwen3_models:
            for loaded_size in list(self.qwen3_models.keys()):
                if loaded_size != model_size:
                    self.unload_qwen3(loaded_size)
        
        if model_size in self.qwen3_models:
            return self.qwen3_models[model_size]
        
        if not QWEN_TTS_AVAILABLE:
            raise RuntimeError("Qwen3-TTS not available. Install with: pip install qwen-tts")
        
        model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base" if model_size == Qwen3ModelSize.SMALL else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        
        # Check available VRAM before loading 1.7B model
        if TORCH_AVAILABLE and torch.cuda.is_available() and model_size == Qwen3ModelSize.LARGE:
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_memory_gb = free_memory / 1024**3
            print(f"[GPU] Free VRAM: {free_memory_gb:.2f}GB")
            if free_memory_gb < 6.0:
                print(f"[WARN] Insufficient VRAM for 1.7B model. Free: {free_memory_gb:.2f}GB, Required: ~6GB")
                print(f"[WARN] Falling back to 0.6B model")
                return self.load_qwen3(Qwen3ModelSize.SMALL, use_flash_attention)
        
        print(f"Loading {model_name}...")
        print(f"[INFO] Loading Qwen3-TTS model on {'CUDA' if torch.cuda.is_available() else 'CPU'}...")
        
        try:
            # Clear cache before loading new model
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load model with device_map to ensure it uses GPU if available
            if torch.cuda.is_available():
                model = Qwen3TTSModel.from_pretrained(model_name, device_map='cuda')
            else:
                model = Qwen3TTSModel.from_pretrained(model_name)
            
            self.qwen3_models[model_size] = model
            print(f"[OK] {model_name} loaded on {next(model.model.parameters()).device}")
            
            # Report GPU memory
            if TORCH_AVAILABLE and torch.cuda.is_available():
                print(f"[GPU] Memory used: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            
            return model
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"[ERROR] GPU out of memory loading {model_name}: {e}")
                # Try to clear memory and fallback to 0.6B
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                if model_size == Qwen3ModelSize.LARGE:
                    print(f"[FALLBACK] Attempting to load 0.6B model instead...")
                    return self.load_qwen3(Qwen3ModelSize.SMALL, use_flash_attention)
            raise
    
    def save_voice_profile(self, persona_id: str, reference_audio: np.ndarray, 
                           reference_sample_rate: int, reference_text: str,
                           model_size: Qwen3ModelSize = Qwen3ModelSize.SMALL) -> str:
        """Save voice profile and create voice clone prompt for fast TTS."""
        if not QWEN_TTS_AVAILABLE:
            raise RuntimeError("Qwen3-TTS not available")
        
        # Create voice_id from persona_id
        voice_id = f"persona_{persona_id}"
        profile_dir = os.path.join(self.voices_dir, voice_id)
        os.makedirs(profile_dir, exist_ok=True)
        
        # Save reference audio as WAV
        ref_path = os.path.join(profile_dir, "reference.wav")
        wav_bytes = numpy_to_wav_bytes(reference_audio, reference_sample_rate)
        with open(ref_path, 'wb') as f:
            f.write(wav_bytes)
        
        # Save reference text
        text_path = os.path.join(profile_dir, "reference.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(reference_text)
        
        # Create and cache voice clone prompt
        model = self.load_qwen3(model_size)
        print(f"[VoiceProfile] Creating voice clone prompt for {voice_id}...")
        ref_audio_tuple = (reference_audio, reference_sample_rate)
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio_tuple,
            ref_text=reference_text,
            x_vector_only_mode=False
        )
        
        # Cache the prompt
        self.voice_clone_cache[voice_id] = voice_clone_prompt
        
        # Try to pickle the prompt for persistence (optional)
        try:
            import pickle
            cache_path = os.path.join(profile_dir, "voice_prompt.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(voice_clone_prompt, f)
            print(f"[VoiceProfile] Saved voice profile to {profile_dir}")
        except Exception as e:
            print(f"[VoiceProfile] Could not pickle prompt (will recreate on restart): {e}")
        
        return voice_id
    
    def load_voice_profile(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Load voice profile data."""
        profile_dir = os.path.join(self.voices_dir, voice_id)
        if not os.path.exists(profile_dir):
            return None
        
        ref_path = os.path.join(profile_dir, "reference.wav")
        text_path = os.path.join(profile_dir, "reference.txt")
        cache_path = os.path.join(profile_dir, "voice_prompt.pkl")
        
        if not os.path.exists(ref_path) or not os.path.exists(text_path):
            return None
        
        # Load reference text
        with open(text_path, 'r', encoding='utf-8') as f:
            reference_text = f.read()
        
        # Load reference audio
        with open(ref_path, 'rb') as f:
            ref_audio, ref_sr = audio_to_numpy(f.read())
        
        # Try to load cached voice clone prompt
        voice_clone_prompt = None
        if voice_id in self.voice_clone_cache:
            voice_clone_prompt = self.voice_clone_cache[voice_id]
        elif os.path.exists(cache_path):
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    voice_clone_prompt = pickle.load(f)
                self.voice_clone_cache[voice_id] = voice_clone_prompt
                print(f"[VoiceProfile] Loaded cached prompt for {voice_id}")
            except Exception as e:
                print(f"[VoiceProfile] Could not load cached prompt: {e}")
        
        return {
            "voice_id": voice_id,
            "reference_audio": ref_audio,
            "reference_sample_rate": ref_sr,
            "reference_text": reference_text,
            "voice_clone_prompt": voice_clone_prompt
        }
    
    def generate_voice_with_profile(self, text: str, voice_id: str,
                                    model_size: Qwen3ModelSize = Qwen3ModelSize.SMALL,
                                    speed: float = 1.0, language: str = "Auto") -> Tuple[np.ndarray, int]:
        """Generate TTS using cached voice profile (fast)."""
        if not QWEN_TTS_AVAILABLE:
            raise RuntimeError("Qwen3-TTS not available")
        
        model = self.load_qwen3(model_size)
        
        # Check if prompt is in memory cache
        if voice_id in self.voice_clone_cache:
            voice_clone_prompt = self.voice_clone_cache[voice_id]
            print(f"[VoiceProfile] Using cached prompt for {voice_id}")
        else:
            # Load profile and create/recreate prompt
            profile = self.load_voice_profile(voice_id)
            if profile is None:
                raise ValueError(f"Voice profile not found: {voice_id}")
            
            if profile.get("voice_clone_prompt"):
                voice_clone_prompt = profile["voice_clone_prompt"]
            else:
                # Recreate prompt from reference audio
                print(f"[VoiceProfile] Recreating voice clone prompt for {voice_id}...")
                ref_audio_tuple = (profile["reference_audio"], profile["reference_sample_rate"])
                voice_clone_prompt = model.create_voice_clone_prompt(
                    ref_audio=ref_audio_tuple,
                    ref_text=profile["reference_text"],
                    x_vector_only_mode=False
                )
                self.voice_clone_cache[voice_id] = voice_clone_prompt
        
        # Generate audio using cached prompt
        print(f"[VoiceProfile] Generating TTS for text: {text[:50]}...")
        audio_list, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            non_streaming_mode=True
        )
        
        # Process output
        if isinstance(audio_list, list) and len(audio_list) > 0:
            audio_output = np.concatenate(audio_list) if len(audio_list) > 1 else audio_list[0]
        else:
            audio_output = np.array(audio_list) if audio_list is not None else np.array([])
        
        audio_output = np.array(audio_output).flatten()
        if audio_output.size == 0:
            raise RuntimeError("Generated audio is empty")
        
        max_val = np.max(np.abs(audio_output))
        if max_val > 0:
            audio_output = audio_output / max_val * 0.95
        
        return audio_output, sample_rate
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 150) -> List[str]:
        text = text.strip()
        if not text:
            return []
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text] if len(text) <= max_chunk_size else [text[:max_chunk_size]]
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                if ',' in sentence:
                    parts = sentence.split(',')
                    temp_chunk = ""
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        if len(temp_chunk) + len(part) + 2 <= max_chunk_size:
                            temp_chunk = (temp_chunk + ", " + part) if temp_chunk else part
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = part
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    for i in range(0, len(sentence), max_chunk_size):
                        chunk = sentence[i:i + max_chunk_size].strip()
                        if chunk:
                            chunks.append(chunk)
            else:
                if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                    current_chunk = (current_chunk + " " + sentence) if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks if chunks else [text[:max_chunk_size]]
    
    def _concatenate_audio_chunks(self, chunks: List[Tuple[np.ndarray, int]], crossfade_ms: int = 50) -> Tuple[np.ndarray, int]:
        if not chunks:
            return np.array([], dtype=np.float32), 24000
        if len(chunks) == 1:
            return chunks[0]
        sample_rate = chunks[0][1]
        crossfade_samples = int(sample_rate * crossfade_ms / 1000)
        result = chunks[0][0]
        for i in range(1, len(chunks)):
            next_chunk = chunks[i][0]
            if len(result) == 0:
                result = next_chunk
                continue
            if crossfade_samples > 0 and len(result) > crossfade_samples and len(next_chunk) > crossfade_samples:
                fade_out = result[-crossfade_samples:]
                fade_in = next_chunk[:crossfade_samples]
                fade_curve = np.linspace(0, 1, crossfade_samples)
                crossfaded = fade_out * (1 - fade_curve) + fade_in * fade_curve
                result = np.concatenate([result[:-crossfade_samples], crossfaded, next_chunk[crossfade_samples:]])
            else:
                result = np.concatenate([result, next_chunk])
        return result, sample_rate

    def create_voice_qwen3(self, text: str, params: VoiceCreationParams) -> Tuple[np.ndarray, int]:
        model = self.load_qwen3(params.qwen3_model_size)
        print(f"Qwen3-TTS ({params.qwen3_model_size.value}): Generating voice...")
        print(f"[DEBUG] Has reference audio: {params.reference_audio is not None}")
        print(f"[DEBUG] Reference text: {params.reference_text[:50] if params.reference_text else 'None'}...")
        
        # Use generate_voice_clone for voice creation from reference audio
        if params.reference_audio is not None:
            # Create cache key from audio data hash + ref_text
            audio_hash = hashlib.md5(params.reference_audio.tobytes()).hexdigest()[:16]
            ref_text = params.reference_text if params.reference_text else "Hello! This is my custom voice created with Mimic AI."
            cache_key = f"{audio_hash}_{ref_text[:50]}"
            
            # Check if voice clone prompt is cached
            if cache_key in self.voice_clone_cache:
                print(f"[DEBUG] Using cached voice clone prompt")
                voice_clone_prompt = self.voice_clone_cache[cache_key]
            else:
                print(f"[DEBUG] Creating voice clone prompt from reference audio...")
                # ref_audio must be a tuple of (audio_array, sample_rate) per qwen-tts API
                ref_audio_tuple = (params.reference_audio, params.reference_sample_rate)
                # Extract voice clone prompt using Base model capabilities
                # Returns List[VoiceClonePromptItem] - cache the whole list
                voice_clone_prompt = model.create_voice_clone_prompt(
                    ref_audio=ref_audio_tuple,
                    ref_text=ref_text,
                    x_vector_only_mode=False  # Use ICL mode for better quality
                )
                if voice_clone_prompt:
                    self.voice_clone_cache[cache_key] = voice_clone_prompt
                    print(f"[DEBUG] Voice clone prompt cached ({len(voice_clone_prompt)} items)")
            
            # Generate using cached voice clone prompt
            # voice_clone_prompt is List[VoiceClonePromptItem] as expected by generate_voice_clone
            print(f"[DEBUG] Generating voice for text: {text[:50]}...")
            if voice_clone_prompt:
                print(f"[DEBUG] Using voice_clone_prompt with {len(voice_clone_prompt)} items")
                audio_list, sample_rate = model.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=voice_clone_prompt,
                    non_streaming_mode=True
                )
            else:
                # Fallback to direct method if prompt creation failed
                print(f"[WARN] Voice clone prompt creation failed, falling back to direct method")
                ref_audio_tuple = (params.reference_audio, params.reference_sample_rate)
                audio_list, sample_rate = model.generate_voice_clone(
                    text=text,
                    ref_audio=ref_audio_tuple,
                    ref_text=ref_text,
                    non_streaming_mode=True
                )
        else:
            # No reference audio - use a default approach (generate_custom_voice with a default speaker)
            # Default to 'Bella' speaker as a fallback
            print(f"[DEBUG] No reference audio, using custom voice with Bella speaker")
            audio_list, sample_rate = model.generate_custom_voice(
                text=text,
                speaker="Bella",
                non_streaming_mode=True
            )
        
        # Debug the output
        print(f"[DEBUG] audio_list type: {type(audio_list)}, len: {len(audio_list) if isinstance(audio_list, list) else 'N/A'}")
        print(f"[DEBUG] sample_rate: {sample_rate}")
        
        # audio_list is a list of numpy arrays, concatenate them
        if isinstance(audio_list, list) and len(audio_list) > 0:
            audio_output = np.concatenate(audio_list) if len(audio_list) > 1 else audio_list[0]
        else:
            audio_output = np.array(audio_list) if audio_list is not None else np.array([])
        
        print(f"[DEBUG] audio_output shape: {audio_output.shape}, size: {audio_output.size}")
        
        audio_output = np.array(audio_output).flatten()
        print(f"[DEBUG] Flattened audio_output shape: {audio_output.shape}, size: {audio_output.size}")
        
        if audio_output.size == 0:
            raise RuntimeError("Generated audio is empty - voice cloning failed")
        
        max_val = np.max(np.abs(audio_output))
        if max_val > 0:
            audio_output = audio_output / max_val * 0.95
        return audio_output, sample_rate
    
    def create_voice_qwen3_chunked(self, text: str, params: VoiceCreationParams, max_workers: int = 1) -> Tuple[np.ndarray, int]:
        """Generate voice without chunking for better performance"""
        # Skip chunking - generate full text at once for better GPU utilization
        return self.create_voice_qwen3(text, params)

    def create_voice(self, text: str, params: VoiceCreationParams) -> Tuple[np.ndarray, int]:
        start_time = time.time()
        if params.engine == TTSEngine.QWEN3:
            if not QWEN_TTS_AVAILABLE:
                raise RuntimeError("Qwen3-TTS not available. Install with: pip install qwen-tts")
            audio_output, sample_rate = self.create_voice_qwen3_chunked(text, params)
        elif params.engine == TTSEngine.KITTEN:
            audio_output, sample_rate = self.generate_kitten_tts(
                text=text, voice=params.voice or "Bella",
                model_size=params.kitten_model_size, pitch=params.pitch, speed=params.speed
            )
        else:
            raise RuntimeError(f"Unknown engine: {params.engine}")
        if self.enable_watermarking:
            audio_output = self.watermarker.embed_watermark(audio_output, sample_rate)
        synthesis_time = time.time() - start_time
        rtf = synthesis_time / (len(audio_output) / sample_rate)
        print(f"Voice creation complete: {len(audio_output)} samples, RTF={rtf:.3f}")
        return audio_output, sample_rate
    
    def load_kitten(self, model_size: KittenModelSize = KittenModelSize.NANO):
        global KITTEN_TTS_AVAILABLE, KittenTTS
        if model_size in self.kitten_models:
            return self.kitten_models[model_size]
        if not KITTEN_TTS_AVAILABLE:
            print("[KittenTTS] Retrying import...")
            ensure_espeak_in_path()
            try:
                reload_phonemizer_with_espeak()
            except:
                pass
            try:
                from kittentts import KittenTTS as KT
                KittenTTS = KT
                KITTEN_TTS_AVAILABLE = True
                print("[OK] KittenTTS imported successfully on retry")
            except Exception as e:
                print(f"[ERROR] KittenTTS import failed: {e}")
                raise RuntimeError(f"KittenTTS not available: {e}")
        if not KITTEN_TTS_AVAILABLE:
            raise RuntimeError("KittenTTS not available. Install with: pip install kittentts")
        
        # KittenTTS accepts HuggingFace repo name directly
        repo_name = model_size.value  # e.g., "KittenML/kitten-tts-nano-0.8"
        print(f"Loading KittenTTS ({repo_name})...")
        
        try:
            model = KittenTTS(repo_name)
            self.kitten_models[model_size] = model
            print(f"[OK] KittenTTS ({repo_name}) loaded")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load KittenTTS model: {e}")
            print("[WARN] Falling back to default model...")
            try:
                model = KittenTTS()  # Use default model
                self.kitten_models[model_size] = model
                print(f"[OK] KittenTTS (default) loaded")
                return model
            except Exception as e2:
                raise RuntimeError(f"KittenTTS failed to load: {e}, fallback also failed: {e2}")

    def _split_text_for_kitten(self, text: str, max_length: int = 380) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= max_length:
            return [text]
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                if ',' in sentence:
                    parts = sentence.split(',')
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        if len(current_chunk) + len(part) + 2 <= max_length:
                            current_chunk = (current_chunk + ", " + part) if current_chunk else part
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = part
                else:
                    for i in range(0, len(sentence), max_length):
                        part = sentence[i:i + max_length].strip()
                        if part:
                            if current_chunk and len(current_chunk) + len(part) + 1 <= max_length:
                                current_chunk += " " + part
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = part
            else:
                if len(current_chunk) + len(sentence) + 1 <= max_length:
                    current_chunk = (current_chunk + " " + sentence) if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks if chunks else [text[:max_length]]
    
    def generate_kitten_tts(self, text: str, voice: str = "expr-voice-2-m",
                            model_size: KittenModelSize = KittenModelSize.NANO,
                            pitch: int = 0, speed: float = 1.0) -> Tuple[np.ndarray, int]:
        model = self.load_kitten(model_size)
        voice_map = {
            "Bella": "expr-voice-2-f", "Luna": "expr-voice-3-f", "Rosie": "expr-voice-4-f", "Kiki": "expr-voice-5-f",
            "Nicole": "expr-voice-2-f", "Emma": "expr-voice-3-f", "Sophia": "expr-voice-4-f",
            "Jasper": "expr-voice-2-m", "Bruno": "expr-voice-3-m", "Hugo": "expr-voice-4-m", "Leo": "expr-voice-5-m",
            "Adam": "expr-voice-2-m", "Michael": "expr-voice-3-m", "James": "expr-voice-4-m", "William": "expr-voice-5-m",
        }
        kitten_voice = voice_map.get(voice, "expr-voice-2-f")
        text = preprocess_text_for_tts(text)
        if speed == 1.0 and pitch != 0:
            speed = 1.0 - (pitch / 100.0)
        speed = max(0.5, min(2.0, speed))
        MAX_TEXT_LENGTH = 380
        chunks = self._split_text_for_kitten(text, max_length=MAX_TEXT_LENGTH)
        if len(chunks) == 1:
            print(f"KittenTTS ({model_size.value}): Generating with voice '{voice}' -> '{kitten_voice}'")
            try:
                audio_output = model.generate(text=chunks[0], voice=kitten_voice, speed=speed)
                sample_rate = 24000
                audio_output = np.array(audio_output).flatten()
                max_val = np.max(np.abs(audio_output))
                if max_val > 0:
                    audio_output = audio_output / max_val * 0.95
                return audio_output, sample_rate
            except Exception as e:
                print(f"KittenTTS generation failed: {e}")
                raise RuntimeError(f"KittenTTS generation failed: {e}")
        else:
            print(f"KittenTTS ({model_size.value}): Text length {len(text)} -> {len(chunks)} chunks")
            all_audio = []
            for i, chunk in enumerate(chunks):
                print(f"KittenTTS: Generating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                try:
                    audio_output = model.generate(text=chunk, voice=kitten_voice, speed=speed)
                    audio_output = np.array(audio_output).flatten()
                    max_val = np.max(np.abs(audio_output))
                    if max_val > 0:
                        audio_output = audio_output / max_val * 0.95
                    all_audio.append((audio_output, 24000))
                except Exception as e:
                    print(f"KittenTTS chunk {i+1} failed: {e}")
                    raise RuntimeError(f"KittenTTS chunk {i+1} failed: {e}")
            final_audio, sample_rate = self._concatenate_audio_chunks(all_audio, crossfade_ms=30)
            print(f"KittenTTS: Concatenated {len(chunks)} chunks into {len(final_audio)} samples")
            return final_audio, sample_rate

tts_engine: Optional[MimicTTSEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_engine
    print("="*60)
    print("Starting Mimic AI TTS Server")
    print("="*60)
    tts_engine = MimicTTSEngine(enable_watermarking=True)
    try:
        if QWEN_TTS_AVAILABLE:
            print("Preloading Qwen3-TTS 0.6B model...")
            tts_engine.load_qwen3(Qwen3ModelSize.SMALL)
    except Exception as e:
        print(f"Warning: Could not preload Qwen3 model: {e}")
    print("="*60)
    print("TTS Server Ready")
    print("="*60)
    yield
    print("Shutting down TTS Server...")

app = FastAPI(title="Mimic AI TTS Server", description="Unified Text-to-Speech API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        models={"qwen3": QWEN_TTS_AVAILABLE, "kitten": KITTEN_TTS_AVAILABLE},
        cuda_available=torch.cuda.is_available() if TORCH_AVAILABLE else False,
        cuda_devices=torch.cuda.device_count() if TORCH_AVAILABLE else 0
    )

@app.post("/api/voice/create", response_model=VoiceCreationResponse)
async def create_voice(request: VoiceCreationRequest):
    engine = request.engine.lower()
    if engine not in ["qwen3", "kitten"]:
        raise HTTPException(status_code=400, detail=f"Invalid engine: {engine}")
    if engine == "qwen3" and not QWEN_TTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Qwen3-TTS not available")
    try:
        params = VoiceCreationParams.from_request(request)
        ref_path = None
        if request.reference_audio:
            audio_bytes = base64.b64decode(request.reference_audio)
            ref_path = os.path.join(tempfile.gettempdir(), f"ref_{hashlib.md5(audio_bytes).hexdigest()}.wav")
            with open(ref_path, 'wb') as f:
                f.write(audio_bytes)
            params.reference_audio, params.reference_sample_rate = audio_to_numpy(audio_bytes)
        audio_output, sample_rate = tts_engine.create_voice(text=request.text, params=params)
        wav_bytes = numpy_to_wav_bytes(audio_output, sample_rate)
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
        return VoiceCreationResponse(
            audio_data=base64.b64encode(wav_bytes).decode('utf-8'),
            sample_rate=sample_rate, duration=len(audio_output) / sample_rate,
            format="wav", engine=engine
        )
    except Exception as e:
        print(f"Voice creation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts/kitten")
async def generate_kitten_tts(request: KittenTTSRequest):
    if not KITTEN_TTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="KittenTTS not available")
    try:
        # Map short names to full model names
        model_size_map = {
            "nano": KittenModelSize.NANO,
            "micro": KittenModelSize.MICRO,
            "mini": KittenModelSize.MINI,
            "KittenML/kitten-tts-nano-0.8": KittenModelSize.NANO,
            "KittenML/kitten-tts-micro-0.8": KittenModelSize.MICRO,
            "KittenML/kitten-tts-mini-0.8": KittenModelSize.MINI,
        }
        model_size = model_size_map.get(request.model_size, KittenModelSize.NANO)
        audio_output, sample_rate = tts_engine.generate_kitten_tts(
            text=request.text, voice=request.voice, model_size=model_size,
            pitch=request.pitch, speed=request.speed
        )
        wav_bytes = numpy_to_wav_bytes(audio_output, sample_rate)
        return {
            "audio_data": base64.b64encode(wav_bytes).decode('utf-8'),
            "sample_rate": sample_rate, "duration": len(audio_output) / sample_rate,
            "format": "wav", "voice": request.voice,
            "text_length": len(request.text), "chunks": len(tts_engine._split_text_for_kitten(request.text))
        }
    except Exception as e:
        print(f"KittenTTS error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tts/status")
async def get_tts_status():
    return TTSStatusResponse(
        status="healthy",
        models={
            "qwen3": {
                "available": QWEN_TTS_AVAILABLE,
                "loaded": {size.value: size in tts_engine.qwen3_models if tts_engine else False for size in Qwen3ModelSize}
            },
            "kitten": {
                "available": KITTEN_TTS_AVAILABLE,
                "loaded": {size.value: size in tts_engine.kitten_models if tts_engine else False for size in KittenModelSize}
            }
        }
    )

@app.get("/api/engines/status")
async def get_engines_status():
    """Get engine status for Voice Studio"""
    qwen3_loaded_size = None
    if tts_engine and tts_engine.qwen3_models:
        # Get the first loaded model size
        for size in Qwen3ModelSize:
            if size in tts_engine.qwen3_models:
                qwen3_loaded_size = size.value
                break
    
    return {
        "qwen3_available": QWEN_TTS_AVAILABLE,
        "cuda_available": torch.cuda.is_available() if TORCH_AVAILABLE else False,
        "current_engine": "qwen3",
        "qwen3_loaded_size": qwen3_loaded_size
    }

@app.post("/api/voice/profile/save", response_model=VoiceProfileResponse)
async def save_voice_profile(request: SaveVoiceProfileRequest):
    """Save a voice profile for fast TTS generation."""
    if not QWEN_TTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Qwen3-TTS not available")
    try:
        # Decode reference audio
        audio_bytes = base64.b64decode(request.reference_audio)
        ref_audio, ref_sr = audio_to_numpy(audio_bytes)
        
        # Map model size
        model_size = Qwen3ModelSize.SMALL if request.qwen3_model_size == "0.6B" else Qwen3ModelSize.LARGE
        
        # Save profile and create voice clone prompt
        voice_id = tts_engine.save_voice_profile(
            persona_id=request.persona_id,
            reference_audio=ref_audio,
            reference_sample_rate=ref_sr,
            reference_text=request.reference_text,
            model_size=model_size
        )
        
        return VoiceProfileResponse(
            status="success",
            persona_id=request.persona_id,
            voice_id=voice_id,
            message=f"Voice profile saved. Use voice_id '{voice_id}' for TTS generation."
        )
    except Exception as e:
        print(f"Save voice profile error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/profile/generate")
async def generate_with_voice_profile(request: GenerateVoiceRequest):
    """Generate TTS using a saved voice profile (fast - uses cached prompt)."""
    if not QWEN_TTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Qwen3-TTS not available")
    try:
        model_size = Qwen3ModelSize.SMALL if request.qwen3_model_size == "0.6B" else Qwen3ModelSize.LARGE
        
        audio_output, sample_rate = tts_engine.generate_voice_with_profile(
            text=request.text,
            voice_id=request.voice_id,
            model_size=model_size,
            speed=request.speed,
            language=request.language
        )
        
        wav_bytes = numpy_to_wav_bytes(audio_output, sample_rate)
        
        return {
            "audio_data": base64.b64encode(wav_bytes).decode('utf-8'),
            "sample_rate": sample_rate,
            "duration": len(audio_output) / sample_rate,
            "format": "wav",
            "voice_id": request.voice_id
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Generate with profile error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voice/profile/{persona_id}")
async def get_voice_profile(persona_id: str):
    """Check if a voice profile exists."""
    voice_id = f"persona_{persona_id}"
    profile = tts_engine.load_voice_profile(voice_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Voice profile not found for persona: {persona_id}")
    return {
        "voice_id": voice_id,
        "has_cached_prompt": profile.get("voice_clone_prompt") is not None,
        "reference_text": profile.get("reference_text", "")[:100] + "..."
    }

@app.post("/api/tts/unload")
async def unload_tts_models(engine: str = "qwen3"):
    """Unload TTS models to free GPU memory when switching engines or personas."""
    try:
        if engine.lower() == "qwen3":
            tts_engine.unload_qwen3()
            return {"status": "success", "message": "Qwen3 models unloaded", "engine": engine}
        elif engine.lower() == "all":
            tts_engine.unload_qwen3()
            # Clear Kitten models too
            for size in list(tts_engine.kitten_models.keys()):
                del tts_engine.kitten_models[size]
            tts_engine.kitten_models.clear()
            return {"status": "success", "message": "All TTS models unloaded", "engine": engine}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")
    except Exception as e:
        print(f"Unload error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("TTS_PORT", 8000))
    host = os.environ.get("TTS_HOST", "127.0.0.1")
    print(f"Starting TTS server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
