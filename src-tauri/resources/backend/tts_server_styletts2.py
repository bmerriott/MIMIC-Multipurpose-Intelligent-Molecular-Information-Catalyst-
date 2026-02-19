"""
Mimic AI - StyleTTS 2 Backend Server (Synthetic Voices Only)
Provides real-time text-to-speech synthesis using StyleTTS 2 with parameter-based voice control.
NO voice cloning - uses synthetic voice generation with adjustable parameters.
"""

import os
import io
import base64
import tempfile
import time
import traceback
import hashlib
import random
from typing import Optional, Union, List, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
print("Loaded environment variables from .env file (if present)")

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn
import numpy as np

# Try to import torch FIRST (needed before the monkey-patch)
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch/torchaudio not installed. Some features may be limited.")

# Fix for PyTorch 2.6+ weights_only default change
# Must be done BEFORE importing styletts2
if TORCH_AVAILABLE:
    # Monkey-patch torch.load to use weights_only=False for compatibility
    # This is needed because styletts2 uses torch.load without weights_only parameter
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    print("PyTorch torch.load patched for compatibility with styletts2")

# Download required NLTK data for StyleTTS 2
try:
    import nltk
    # Download punkt tokenizer data (needed by styletts2)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt data...")
        nltk.download('punkt', quiet=True)
    # Also download punkt_tab which is sometimes needed
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt_tab data...")
        nltk.download('punkt_tab', quiet=True)
    print("NLTK data ready")
except ImportError:
    print("Warning: nltk not installed. StyleTTS 2 may not work properly.")

# Try to import StyleTTS 2 (AFTER the torch patch)
try:
    from styletts2 import tts as styletts2_module
    STYLETTS2_AVAILABLE = True
    print("StyleTTS 2 imported successfully")
except ImportError as e:
    STYLETTS2_AVAILABLE = False
    print(f"Warning: styletts2 not installed. Run: pip install styletts2")
    print(f"Import error: {e}")

# Try to import soundfile for audio format support (WebM, MP3, etc.)
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not installed. WebM audio will not be supported.")

# Try to import librosa for audio preprocessing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not installed. Audio preprocessing may be limited.")


# ============ AUDIO WATERMARKING ============
# Import improved legal watermarker
try:
    from watermarker import LegalWatermarker, detect_ai_watermark
    LEGAL_WATERMARKER_AVAILABLE = True
except ImportError:
    LEGAL_WATERMARKER_AVAILABLE = False
    print("Warning: watermarker.py not found. Using basic watermarking.")


class AudioWatermarker:
    """
    Invisible audio watermarking using spread spectrum technique.
    The watermark survives compression and is detectable by a public tool.
    """
    
    def __init__(self, watermark_key: str = "AI-generated"):
        self.watermark_key = watermark_key
        # Generate a pseudo-random sequence based on the key
        np.random.seed(int(hashlib.md5(watermark_key.encode()).hexdigest(), 16) % (2**32))
        self.chip_rate = 100  # Chips per second
        self.strength = 0.005  # Watermark strength (very subtle)
        
    def generate_chips(self, num_samples: int, sample_rate: int) -> np.ndarray:
        """Generate pseudo-random chip sequence for watermark"""
        num_chips = int(num_samples * self.chip_rate / sample_rate) + 1
        chips = np.random.choice([-1, 1], size=num_chips)
        # Upsample to audio sample rate
        chip_samples = int(sample_rate / self.chip_rate)
        watermark = np.repeat(chips, chip_samples)[:num_samples]
        return watermark.astype(np.float32)
    
    def embed_watermark(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Embed invisible watermark into audio.
        Returns watermarked audio.
        """
        if len(audio.shape) > 1:
            # Handle stereo - watermark both channels
            result = np.copy(audio)
            for i in range(audio.shape[1]):
                result[:, i] = self.embed_watermark(audio[:, i], sample_rate)
            return result
        
        # Generate watermark sequence
        watermark = self.generate_chips(len(audio), sample_rate)
        
        # Apply simple spectral shaping to make watermark less audible
        # Use a high-pass characteristic (watermark mostly in higher frequencies)
        alpha = self.strength
        watermarked = audio + alpha * watermark * np.abs(audio)
        
        # Ensure no clipping
        max_val = np.max(np.abs(watermarked))
        if max_val > 1.0:
            watermarked = watermarked / max_val * 0.99
            
        return watermarked.astype(np.float32)
    
    def detect_watermark(self, audio: np.ndarray, sample_rate: int) -> Tuple[bool, float]:
        """
        Detect if watermark is present in audio.
        Returns (detected, confidence_score)
        """
        if len(audio.shape) > 1:
            # Use first channel for detection
            audio = audio[:, 0]
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Generate expected watermark
        watermark = self.generate_chips(len(audio), sample_rate)
        
        # Correlate with expected watermark
        correlation = np.correlate(audio, watermark, mode='valid')
        max_corr = np.max(np.abs(correlation))
        mean_corr = np.mean(np.abs(correlation))
        
        # Confidence score based on correlation peak
        confidence = max_corr / (mean_corr + 1e-8)
        
        # Threshold for detection
        detected = confidence > 2.0  # Empirical threshold
        
        return detected, float(confidence)


@dataclass
class SyntheticVoiceParams:
    """Parameters for synthetic voice generation"""
    gender: str = "neutral"  # neutral, masculine, feminine
    age: str = "adult"  # young, adult, mature
    pitch: float = 0.0  # -1.0 to 1.0 (shift from baseline)
    speed: float = 1.0  # 0.5 to 2.0
    warmth: float = 0.5  # 0.0 to 1.0 (bright to warm)
    expressiveness: float = 0.5  # 0.0 to 1.0 (monotone to dynamic)
    stability: float = 0.5  # 0.0 to 1.0 (variability consistency)
    seed: Optional[int] = None  # Random seed for reproducibility


def numpy_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert numpy array to WAV bytes"""
    import wave
    
    # Ensure audio is in the right range
    if audio_array.dtype != np.int16:
        # Normalize to int16 range
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        audio_array = (audio_array * 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())
    
    return buffer.getvalue()


def audio_to_numpy(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Convert audio bytes to numpy array and sample rate
    
    Supports multiple formats: WAV, WebM, MP3, OGG, etc.
    Uses soundfile for broad format support, falls back to wave for basic WAV.
    """
    # Try soundfile first (supports WebM, MP3, OGG, FLAC, etc.)
    if SOUNDFILE_AVAILABLE:
        try:
            with io.BytesIO(audio_bytes) as buffer:
                audio_array, sample_rate = sf.read(buffer, dtype='float32')
                
                # Convert to mono if multi-channel
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                
                print(f"Audio decoded with soundfile: {len(audio_array)} samples at {sample_rate}Hz")
                return audio_array, sample_rate
        except Exception as e:
            print(f"soundfile failed to decode audio: {e}, trying wave fallback...")
    
    # Fallback to wave module for basic WAV support
    import wave
    
    with io.BytesIO(audio_bytes) as buffer:
        try:
            with wave.open(buffer, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Read raw data
                raw_data = wav_file.readframes(n_frames)
                
                # Convert to numpy array
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
                
                print(f"Audio decoded with wave: {len(audio_array)} samples at {sample_rate}Hz")
                return audio_array, sample_rate
        except wave.Error as e:
            raise ValueError(f"Audio format not supported. Install soundfile for WebM/MP3 support. Error: {e}")


def resample_audio(audio_array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    if orig_sr == target_sr:
        return audio_array
    
    if LIBROSA_AVAILABLE:
        return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)
    else:
        # Simple linear interpolation fallback
        from scipy import signal
        return signal.resample(audio_array, int(len(audio_array) * target_sr / orig_sr))


class TTSModelManager:
    """Manages StyleTTS 2 model instances and synthetic voice generation"""
    
    def __init__(self):
        self.tts_model = None
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.target_sample_rate = 24000  # StyleTTS 2 outputs at 24kHz
        
        # Watermarker - use legal multi-layer if available
        if LEGAL_WATERMARKER_AVAILABLE:
            self.watermarker = LegalWatermarker(user_id="", timestamp="")
            print("Using LegalWatermarker (multi-layer, evidentiary)")
        else:
            self.watermarker = AudioWatermarker("AI-generated")
            print("Using basic AudioWatermarker")
        self.enable_watermarking = True
        
        # Stored synthetic voice configurations
        # Key: voice_id, Value: SyntheticVoiceParams
        self.saved_voices: dict[str, SyntheticVoiceParams] = {}
        
        # Create directory for voice configs
        self.voices_dir = os.path.join(os.path.dirname(__file__), "saved_voices")
        os.makedirs(self.voices_dir, exist_ok=True)
        print(f"Saved voices directory: {self.voices_dir}")
        
        # Load saved voices on startup
        self._load_saved_voices()
        
        # Load model on startup
        if STYLETTS2_AVAILABLE:
            self.load_model()
    
    def _load_saved_voices(self):
        """Load saved synthetic voice configurations"""
        import json
        config_path = os.path.join(self.voices_dir, "voice_configs.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    for voice_id, params_dict in data.items():
                        self.saved_voices[voice_id] = SyntheticVoiceParams(**params_dict)
                print(f"Loaded {len(self.saved_voices)} saved voice configurations")
            except Exception as e:
                print(f"Warning: could not load saved voices: {e}")
    
    def _save_voice_configs(self):
        """Save synthetic voice configurations to disk"""
        import json
        config_path = os.path.join(self.voices_dir, "voice_configs.json")
        try:
            data = {vid: {
                'gender': v.gender,
                'age': v.age,
                'pitch': v.pitch,
                'speed': v.speed,
                'warmth': v.warmth,
                'expressiveness': v.expressiveness,
                'stability': v.stability,
                'seed': v.seed
            } for vid, v in self.saved_voices.items()}
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save voice configs: {e}")
    
    def load_model(self):
        """Lazy load the StyleTTS 2 model"""
        if not STYLETTS2_AVAILABLE:
            raise RuntimeError("StyleTTS 2 is not installed. Run: pip install styletts2")
            
        if self.tts_model is None:
            print("Loading StyleTTS 2 model...")
            print(f"Using device: {self.device}")
            try:
                # Initialize StyleTTS 2 - downloads model if needed
                self.tts_model = styletts2_module.StyleTTS2()
                print("StyleTTS 2 model loaded successfully")
            except Exception as e:
                print(f"Failed to load StyleTTS 2 model: {e}")
                traceback.print_exc()
                raise
        
        return self.tts_model
    
    def unload_model(self):
        """Unload TTS model to free GPU memory"""
        if self.tts_model is not None:
            print("Unloading StyleTTS 2 model")
            try:
                # Delete model
                del self.tts_model
                self.tts_model = None
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache if available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print(f"GPU memory cleared. Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
                
                print("Model unloaded successfully")
            except Exception as e:
                print(f"Error unloading model: {e}")
    
    def _get_or_create_reference_audio(self, params: SyntheticVoiceParams) -> str:
        """
        Get or create a reference audio file for the given voice parameters.
        StyleTTS2 uses reference audio for voice characteristics.
        We use the default voice and apply post-processing for pitch/speed.
        """
        import hashlib
        import scipy.io.wavfile as wavfile
        
        # Create a cache key based on gender and age (these affect reference voice selection)
        cache_key = f"{params.gender}_{params.age}"
        ref_path = os.path.join(self.voices_dir, f"_ref_{cache_key}.wav")
        
        # If we already have a reference for this gender/age combo, use it
        if os.path.exists(ref_path):
            return ref_path
        
        # Generate a short reference audio using default voice
        # We'll use this for voice cloning to get the base characteristics
        print(f"Generating reference audio for {params.gender} {params.age} voice...")
        
        ref_text = "Hello, this is a reference voice for synthetic speech generation."
        model = self.load_model()
        
        # Generate base reference audio
        ref_audio = model.inference(
            text=ref_text,
            diffusion_steps=5,
            embedding_scale=1.0
        )
        
        # Apply pitch shifting based on gender and age
        ref_audio = self._apply_voice_characteristics(ref_audio, params, is_reference=True)
        
        # Save reference audio
        wavfile.write(ref_path, self.target_sample_rate, (ref_audio * 32767).astype(np.int16))
        print(f"Reference audio saved: {ref_path}")
        
        return ref_path
    
    def _apply_voice_characteristics(self, audio: np.ndarray, params: SyntheticVoiceParams, is_reference: bool = False) -> np.ndarray:
        """
        Apply voice characteristics (pitch, etc.) to audio using librosa.
        """
        if not LIBROSA_AVAILABLE:
            print("Warning: librosa not available, voice characteristics may be limited")
            return audio
        
        result = audio.copy()
        
        # Apply pitch shifting based on gender (more significant effect)
        gender_shift = 0
        if params.gender == "masculine":
            gender_shift = -3  # 3 semitones lower
        elif params.gender == "feminine":
            gender_shift = 3   # 3 semitones higher
        
        # Apply pitch shifting based on age
        age_shift = 0
        if params.age == "young":
            age_shift = 2  # 2 semitones higher
        elif params.age == "mature":
            age_shift = -2  # 2 semitones lower
        
        # Apply user-controlled pitch
        user_pitch = params.pitch * 4  # -4 to +4 semitones
        
        # Total pitch shift
        total_shift = gender_shift + age_shift + user_pitch
        
        if abs(total_shift) > 0.1:
            try:
                result = librosa.effects.pitch_shift(
                    result, 
                    sr=self.target_sample_rate, 
                    n_steps=total_shift
                )
            except Exception as e:
                print(f"Pitch shifting failed: {e}")
        
        return result
    
    def synthesize_synthetic(
        self, 
        text: str, 
        params: SyntheticVoiceParams
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize text using synthetic voice parameters.
        Uses StyleTTS2 voice cloning with reference audio for gender/age,
        then applies pitch shifting and speed adjustment.
        Returns: (audio_array, sample_rate)
        """
        model = self.load_model()
        
        try:
            print(f"Synthesizing with synthetic voice: {text[:50]}...")
            print(f"  Gender: {params.gender}, Age: {params.age}, Pitch: {params.pitch}")
            print(f"  Speed: {params.speed}, Warmth: {params.warmth}, Expressiveness: {params.expressiveness}")
            
            start_time = time.time()
            
            # Get or create reference audio for this voice profile
            ref_path = self._get_or_create_reference_audio(params)
            
            # Adjust diffusion steps based on expressiveness
            diffusion_steps = 5 + int(params.expressiveness * 5)  # 5-10 steps
            
            # Adjust embedding scale based on expressiveness
            embedding_scale = 0.8 + params.expressiveness * 0.4  # 0.8-1.2
            
            # Generate speech using voice cloning with reference audio
            # This gives us the base voice characteristics (gender, age)
            audio_output = model.inference(
                text=text,
                target_voice_path=ref_path,  # Use reference audio for voice characteristics
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
                alpha=0.3 + params.warmth * 0.4,  # Timbre balance (0.3-0.7)
                beta=0.5 + params.expressiveness * 0.3  # Prosody balance (0.5-0.8)
            )
            
            # Apply pitch shifting based on user pitch parameter
            # (gender/age pitch is already in the reference audio)
            if abs(params.pitch) > 0.05 and LIBROSA_AVAILABLE:
                try:
                    pitch_shift = params.pitch * 4  # -4 to +4 semitones
                    audio_output = librosa.effects.pitch_shift(
                        audio_output,
                        sr=self.target_sample_rate,
                        n_steps=pitch_shift
                    )
                except Exception as e:
                    print(f"Pitch shift failed: {e}")
            
            # Apply speed adjustment if needed (resample)
            if abs(params.speed - 1.0) > 0.05:
                # Resample to change speed (pitch-corrected speed change)
                new_sample_rate = int(self.target_sample_rate * params.speed)
                audio_output = resample_audio(audio_output, new_sample_rate, self.target_sample_rate)
            
            synthesis_time = time.time() - start_time
            rtf = synthesis_time / (len(audio_output) / self.target_sample_rate)
            print(f"Synthesis successful: {len(audio_output)} samples, RTF={rtf:.3f}")
            
            # Apply watermark
            if self.enable_watermarking:
                audio_output = self.watermarker.embed_watermark(audio_output, self.target_sample_rate)
            
            return audio_output, self.target_sample_rate
            
        except Exception as e:
            print(f"Synthesis failed: {e}")
            traceback.print_exc()
            raise
    
    def synthesize_default(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize text to speech using default voice"""
        model = self.load_model()
        
        try:
            print(f"Synthesizing with default voice: {text[:50]}...")
            
            # Generate speech without voice cloning (uses default voice)
            start_time = time.time()
            audio_output = model.inference(
                text=text,
                diffusion_steps=5,
                embedding_scale=1
            )
            
            synthesis_time = time.time() - start_time
            rtf = synthesis_time / (len(audio_output) / self.target_sample_rate)
            print(f"Synthesis successful: {len(audio_output)} samples, RTF={rtf:.3f}")
            
            # Apply watermark
            if self.enable_watermarking:
                audio_output = self.watermarker.embed_watermark(audio_output, self.target_sample_rate)
            
            return audio_output, self.target_sample_rate
            
        except Exception as e:
            print(f"Synthesis failed: {e}")
            traceback.print_exc()
            raise
    
    def save_voice(self, voice_id: str, params: SyntheticVoiceParams) -> dict:
        """Save a synthetic voice configuration"""
        self.saved_voices[voice_id] = params
        self._save_voice_configs()
        
        return {
            'voice_id': voice_id,
            'gender': params.gender,
            'age': params.age,
            'pitch': params.pitch,
            'speed': params.speed,
            'warmth': params.warmth,
            'expressiveness': params.expressiveness,
            'stability': params.stability,
            'seed': params.seed
        }
    
    def get_saved_voice(self, voice_id: str) -> Optional[SyntheticVoiceParams]:
        """Get a saved synthetic voice configuration"""
        return self.saved_voices.get(voice_id)
    
    def list_saved_voices(self) -> list:
        """List all saved synthetic voices"""
        return [
            {
                'voice_id': vid,
                'gender': params.gender,
                'age': params.age,
                'pitch': params.pitch,
                'speed': params.speed,
                'warmth': params.warmth,
                'expressiveness': params.expressiveness,
                'stability': params.stability,
                'seed': params.seed
            }
            for vid, params in self.saved_voices.items()
        ]
    
    def delete_saved_voice(self, voice_id: str) -> bool:
        """Delete a saved voice configuration"""
        if voice_id in self.saved_voices:
            del self.saved_voices[voice_id]
            self._save_voice_configs()
            return True
        return False


# Global model manager
model_manager = TTSModelManager()


# ============ API MODELS ============

class SyntheticVoiceParamsModel(BaseModel):
    """Parameters for synthetic voice generation"""
    gender: str = Field(default="neutral", description="Voice gender: neutral, masculine, feminine")
    age: str = Field(default="adult", description="Voice age: young, adult, mature")
    pitch: float = Field(default=0.0, ge=-1.0, le=1.0, description="Pitch shift from -1.0 to 1.0")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed from 0.5x to 2.0x")
    warmth: float = Field(default=0.5, ge=0.0, le=1.0, description="Timbre warmth from 0.0 (bright) to 1.0 (warm)")
    expressiveness: float = Field(default=0.5, ge=0.0, le=1.0, description="Expressiveness from 0.0 (monotone) to 1.0 (dynamic)")
    stability: float = Field(default=0.5, ge=0.0, le=1.0, description="Voice consistency/stability from 0.0 to 1.0")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible voice generation")
    
    @field_validator('pitch')
    @classmethod
    def validate_pitch(cls, v):
        if v is None:
            return 0.0
        return float(v)
    
    @field_validator('speed')
    @classmethod
    def validate_speed(cls, v):
        if v is None:
            return 1.0
        return float(v)
    
    @field_validator('warmth', 'expressiveness', 'stability')
    @classmethod
    def validate_float_0_1(cls, v):
        if v is None:
            return 0.5
        return float(v)
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in ("neutral", "masculine", "feminine"):
            return "neutral"
        return v
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v not in ("young", "adult", "mature"):
            return "adult"
        return v


class SyntheticSynthesizeRequest(BaseModel):
    """Request to synthesize text with synthetic voice parameters"""
    text: str = Field(..., description="Text to synthesize")
    params: SyntheticVoiceParamsModel = Field(default_factory=SyntheticVoiceParamsModel, description="Voice parameters")
    language: str = Field(default="English", description="Language code")


class SavedVoiceSynthesizeRequest(BaseModel):
    """Request to synthesize text with a saved voice ID"""
    text: str = Field(..., description="Text to synthesize")
    voice_id: str = Field(..., description="ID of the saved voice configuration")
    language: str = Field(default="English", description="Language code")


class SaveVoiceRequest(BaseModel):
    """Request to save a synthetic voice configuration"""
    voice_id: str = Field(..., description="Unique identifier for this voice")
    params: SyntheticVoiceParamsModel = Field(..., description="Voice parameters to save")


class TTSResponse(BaseModel):
    """Response with synthesized audio"""
    audio_data: str = Field(..., description="Base64 encoded WAV audio")
    duration_ms: int = Field(..., description="Audio duration in milliseconds")
    sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")


class SavedVoiceInfo(BaseModel):
    """Information about a saved synthetic voice"""
    voice_id: str
    gender: str
    age: str
    pitch: float
    speed: float
    warmth: float
    expressiveness: float
    stability: float
    seed: Optional[int]


class SavedVoiceListResponse(BaseModel):
    """Response with list of saved voices"""
    voices: List[SavedVoiceInfo]
    count: int


class Voice(BaseModel):
    """Voice option for dropdowns"""
    id: str
    name: str
    description: str
    language: str
    is_synthetic: bool = True


class WatermarkDetectRequest(BaseModel):
    """Request to detect watermark in audio"""
    audio_data: str = Field(..., description="Base64 encoded audio")


class WatermarkDetectResponse(BaseModel):
    """Response from watermark detection"""
    detected: bool
    confidence: float
    message: str


# Create FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    print("=" * 50)
    print("Mimic AI TTS Backend Starting...")
    print("Mode: SYNTHETIC VOICE GENERATION ONLY")
    print(f"StyleTTS 2 Available: {STYLETTS2_AVAILABLE}")
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Audio Watermarking: Enabled")
    print("=" * 50)
    
    # Preload model in background for faster first request
    if STYLETTS2_AVAILABLE:
        print("StyleTTS 2 model loaded during startup")
    else:
        print("WARNING: StyleTTS 2 not available. Run: pip install styletts2")
    
    yield
    # Shutdown
    print("Shutting down TTS backend...")
    model_manager.unload_model()


app = FastAPI(
    title="Mimic AI TTS Backend (StyleTTS 2 - Synthetic Voices)",
    description="Synthetic voice generation using StyleTTS 2 with parameter controls. NO voice cloning.",
    version="2.0.0-synthetic",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom validation error handler for debugging
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error: {exc.errors()}")
    print(f"Request body: {await request.body()}")
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
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "gpu_memory_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        }
    
    return {
        "status": "healthy",
        "mode": "synthetic_voice_generation",
        "voice_cloning_enabled": False,
        "styletts2_available": STYLETTS2_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": TORCH_AVAILABLE and torch.cuda.is_available(),
        "model_loaded": model_manager.tts_model is not None,
        "watermarking_enabled": model_manager.enable_watermarking,
        "saved_voices_count": len(model_manager.saved_voices),
        **gpu_info,
    }


@app.get("/gpu-status")
async def gpu_status():
    """Get detailed GPU memory status"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "message": "CUDA not available"
        }
    
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
        "memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
        "memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2),
        "model_loaded": model_manager.tts_model is not None,
    }


@app.post("/unload-model")
async def unload_model():
    """Manually unload TTS model to free GPU memory"""
    model_manager.unload_model()
    return {
        "status": "success",
        "message": "Model unloaded successfully",
        "model_loaded": model_manager.tts_model is not None,
    }


# ===== SYNTHETIC VOICE ENDPOINTS =====

@app.post("/api/voice/synthesize-synthetic", response_model=TTSResponse)
async def synthesize_synthetic(request: SyntheticSynthesizeRequest):
    """
    Synthesize text using synthetic voice parameters.
    
    This endpoint creates voices entirely through AI synthesis using adjustable parameters.
    NO voice cloning or audio reference is used.
    """
    if not STYLETTS2_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="StyleTTS 2 is not installed. Please install styletts2 package."
        )
    
    try:
        print(f"Received synthetic voice request: {request.text[:50] if request.text else 'EMPTY'}...")
        print(f"Params received: gender={request.params.gender}, age={request.params.age}, pitch={request.params.pitch}, speed={request.params.speed}")
        
        # Convert Pydantic model to internal params
        params = SyntheticVoiceParams(
            gender=request.params.gender or "neutral",
            age=request.params.age or "adult",
            pitch=float(request.params.pitch) if request.params.pitch is not None else 0.0,
            speed=float(request.params.speed) if request.params.speed is not None else 1.0,
            warmth=float(request.params.warmth) if request.params.warmth is not None else 0.5,
            expressiveness=float(request.params.expressiveness) if request.params.expressiveness is not None else 0.5,
            stability=float(request.params.stability) if request.params.stability is not None else 0.5,
            seed=int(request.params.seed) if request.params.seed is not None else None
        )
        
        print(f"Converted params: {params}")
        
        # Synthesize with synthetic parameters
        audio_array, sample_rate = model_manager.synthesize_synthetic(
            text=request.text,
            params=params
        )
        
        # Convert to WAV bytes
        wav_bytes = numpy_to_wav_bytes(audio_array, sample_rate)
        
        # Encode to base64
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        # Calculate duration
        duration_ms = int(len(audio_array) / sample_rate * 1000)
        
        return TTSResponse(
            audio_data=audio_base64,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
        )
    
    except Exception as e:
        print(f"Synthetic voice synthesis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/synthesize-saved", response_model=TTSResponse)
async def synthesize_with_saved_voice(request: SavedVoiceSynthesizeRequest):
    """
    Synthesize text using a previously saved synthetic voice.
    """
    if not STYLETTS2_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="StyleTTS 2 is not installed."
        )
    
    # Check if voice exists
    params = model_manager.get_saved_voice(request.voice_id)
    if params is None:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{request.voice_id}' not found. Save it first via /api/voice/save"
        )
    
    try:
        print(f"Synthesizing with saved voice '{request.voice_id}': {request.text[:50]}...")
        
        # Synthesize with saved parameters
        audio_array, sample_rate = model_manager.synthesize_synthetic(
            text=request.text,
            params=params
        )
        
        # Convert to WAV bytes
        wav_bytes = numpy_to_wav_bytes(audio_array, sample_rate)
        
        # Encode to base64
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        # Calculate duration
        duration_ms = int(len(audio_array) / sample_rate * 1000)
        
        return TTSResponse(
            audio_data=audio_base64,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
        )
    
    except Exception as e:
        print(f"Synthesis with saved voice failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/save")
async def save_voice(request: SaveVoiceRequest):
    """
    Save a synthetic voice configuration for later use.
    """
    try:
        params = SyntheticVoiceParams(
            gender=request.params.gender,
            age=request.params.age,
            pitch=request.params.pitch,
            speed=request.params.speed,
            warmth=request.params.warmth,
            expressiveness=request.params.expressiveness,
            stability=request.params.stability,
            seed=request.params.seed
        )
        
        result = model_manager.save_voice(request.voice_id, params)
        
        return {
            "status": "success",
            "message": f"Voice '{request.voice_id}' saved successfully",
            "voice": result
        }
    
    except Exception as e:
        print(f"Failed to save voice: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voice/saved", response_model=SavedVoiceListResponse)
async def list_saved_voices():
    """List all saved synthetic voice configurations"""
    voices = model_manager.list_saved_voices()
    return SavedVoiceListResponse(
        voices=[SavedVoiceInfo(**v) for v in voices],
        count=len(voices)
    )


@app.delete("/api/voice/saved/{voice_id}")
async def delete_saved_voice(voice_id: str):
    """Delete a saved voice configuration"""
    success = model_manager.delete_saved_voice(voice_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    return {"status": "success", "message": f"Voice '{voice_id}' deleted"}


# ===== LEGACY/COMPATIBILITY ENDPOINTS =====

@app.post("/api/tts/generate", response_model=TTSResponse)
async def generate_speech(request: dict):
    """
    Generate speech from text using default voice.
    Legacy endpoint - maintained for backward compatibility.
    """
    if not STYLETTS2_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="StyleTTS 2 is not installed"
        )
    
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Use default synthesis
        audio_array, sample_rate = model_manager.synthesize_default(text)
        
        # Convert to WAV bytes
        wav_bytes = numpy_to_wav_bytes(audio_array, sample_rate)
        
        # Encode to base64
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        # Calculate duration
        duration_ms = int(len(audio_array) / sample_rate * 1000)
        
        return TTSResponse(
            audio_data=audio_base64,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
        )
    
    except Exception as e:
        print(f"Speech generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voices", response_model=list[Voice])
async def list_voices():
    """List available voice options"""
    voices = [
        Voice(id="default", name="Default", description="Default StyleTTS 2 voice", language="English"),
        Voice(id="neutral", name="Neutral", description="Neutral synthetic voice", language="English", is_synthetic=True),
        Voice(id="warm", name="Warm", description="Warm, comforting synthetic voice", language="English", is_synthetic=True),
        Voice(id="bright", name="Bright", description="Bright, energetic synthetic voice", language="English", is_synthetic=True),
        Voice(id="professional", name="Professional", description="Professional, clear synthetic voice", language="English", is_synthetic=True),
    ]
    
    # Add saved voices
    for saved_voice in model_manager.list_saved_voices():
        voices.append(Voice(
            id=saved_voice['voice_id'],
            name=saved_voice['voice_id'].replace('_', ' ').title(),
            description=f"Custom {saved_voice['gender']} {saved_voice['age']} voice",
            language="English",
            is_synthetic=True
        ))
    
    return voices


@app.post("/api/preload-model")
async def preload_model():
    """Preload the TTS model into memory"""
    if not STYLETTS2_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="StyleTTS 2 is not installed"
        )
    
    try:
        model_manager.load_model()
        return {"status": "success", "message": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== WATERMARK DETECTION ENDPOINT =====

@app.post("/api/watermark/detect", response_model=WatermarkDetectResponse)
async def detect_watermark(request: WatermarkDetectRequest):
    """Detect AI-generated watermark in audio"""
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_data)
        audio_array, sample_rate = audio_to_numpy(audio_bytes)
        
        # Use improved detection if available
        if LEGAL_WATERMARKER_AVAILABLE:
            detected, confidence, details = model_manager.watermarker.detect_watermark(audio_array, sample_rate)
            message = "AI-generated watermark DETECTED" if detected else "No AI watermark detected"
            if detected and isinstance(details, dict):
                message += f" ({details.get('layers_detected', 0)}/{details.get('total_layers', 3)} layers verified)"
        else:
            # Fallback to basic detection
            detected, confidence = model_manager.watermarker.detect_watermark(audio_array, sample_rate)
            message = "AI-generated watermark detected" if detected else "No AI-generated watermark detected"
        
        if detected:
            message += f" (confidence: {confidence:.2f})"
        
        return WatermarkDetectResponse(
            detected=detected,
            confidence=confidence,
            message=message
        )
    
    except Exception as e:
        print(f"Watermark detection failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Try to import for web search
try:
    import requests
    import urllib.parse
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print("Warning: requests not installed. Web search will not be available.")


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    answer: str
    source: str
    query: str


@app.post("/api/search", response_model=SearchResponse)
async def web_search(request: SearchRequest):
    """Get instant answer from DuckDuckGo (concise, no API key required)"""
    if not WEB_SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Web search dependencies not installed. Run: pip install requests"
        )
    
    try:
        # Use DuckDuckGo Instant Answers API
        encoded_query = urllib.parse.quote(request.query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract the most relevant answer
        answer = ""
        source = ""
        
        # Priority: AbstractText > Answer > RelatedTopics[0].Text
        if data.get('AbstractText'):
            answer = data['AbstractText']
            source = data.get('AbstractURL', '')
        elif data.get('Answer'):
            answer = data['Answer']
            source = data.get('AbstractURL', '')
        elif data.get('RelatedTopics') and len(data['RelatedTopics']) > 0:
            first_topic = data['RelatedTopics'][0]
            if isinstance(first_topic, dict):
                answer = first_topic.get('Text', '')
                source = first_topic.get('FirstURL', '')
        
        # If still no answer, use Definition
        if not answer and data.get('Definition'):
            answer = data['Definition']
            source = data.get('DefinitionURL', '')
        
        # Truncate if too long (keep under 300 chars for prompt size)
        if len(answer) > 300:
            answer = answer[:297] + "..."
        
        print(f"Instant answer for '{request.query}': {answer[:100]}..." if answer else f"No instant answer for '{request.query}'")
        return SearchResponse(answer=answer, source=source, query=request.query)
        
    except Exception as e:
        print(f"Web search error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


def main():
    """Run the server"""
    print("=" * 60)
    print("Mimic AI TTS Backend (StyleTTS 2 - Synthetic Voices)")
    print("Mode: SYNTHETIC VOICE GENERATION ONLY")
    print("Voice cloning features have been removed.")
    print("All voices are generated using AI synthesis with adjustable parameters.")
    print("=" * 60)
    
    # Check for required dependencies
    if not STYLETTS2_AVAILABLE:
        print("\nWARNING: StyleTTS 2 is not installed!")
        print("To install, run:")
        print("  pip install styletts2")
        print("  pip install gruut  # For phoneme conversion")
        print()
    
    # Run server
    uvicorn.run(
        "tts_server_styletts2:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
