"""
Voice Profile Manager
Handles saving, loading, and managing voice embeddings for persistent voice synthesis.
"""

import os
import hashlib
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
from dataclasses import dataclass, asdict


@dataclass
class VoiceProfile:
    """Voice profile metadata and paths"""
    persona_id: str
    engine: str  # "qwen3" | "styletts2"
    model_size: str  # "0.6B" | "1.7B" for qwen3
    embedding_path: str
    reference_audio_path: str
    reference_text: str
    created_at: str
    params: Dict[str, Any]  # pitch, speed, etc.


class VoiceProfileManager:
    """
    Manages voice profile enrollment, storage, and retrieval.
    
    Voice profiles are stored in:
    - Windows: %APPDATA%/com.mimicai.app/voice_profiles/
    - Linux: ~/.local/share/com.mimicai.app/voice_profiles/
    - macOS: ~/Library/Application Support/com.mimicai.app/voice_profiles/
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        if base_dir:
            self.profiles_dir = Path(base_dir)
        else:
            # Use platform-appropriate app data directory
            self.profiles_dir = self._get_default_profiles_dir()
        
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.profiles_dir / "profiles.json"
        self.profiles: Dict[str, VoiceProfile] = {}
        self._load_metadata()
    
    def _get_default_profiles_dir(self) -> Path:
        """Get platform-specific profiles directory"""
        if os.name == 'nt':  # Windows
            app_data = os.environ.get('APPDATA')
            if not app_data:
                app_data = os.path.expanduser('~')
            return Path(app_data) / "com.mimicai.app" / "voice_profiles"
        elif os.name == 'posix':
            # Check if macOS
            if os.uname().sysname == 'Darwin':
                return Path.home() / "Library" / "Application Support" / "com.mimicai.app" / "voice_profiles"
            else:  # Linux
                data_home = os.environ.get('XDG_DATA_HOME')
                if not data_home:
                    data_home = Path.home() / ".local" / "share"
                return Path(data_home) / "com.mimicai.app" / "voice_profiles"
        else:
            return Path.home() / ".mimic_ai" / "voice_profiles"
    
    def _load_metadata(self):
        """Load profile metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                for pid, pdata in data.items():
                    self.profiles[pid] = VoiceProfile(**pdata)
            except Exception as e:
                print(f"[VoiceProfileManager] Failed to load metadata: {e}")
    
    def _save_metadata(self):
        """Save profile metadata to disk"""
        try:
            data = {pid: asdict(profile) for pid, profile in self.profiles.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[VoiceProfileManager] Failed to save metadata: {e}")
    
    def enroll_voice(
        self,
        persona_id: str,
        reference_audio: np.ndarray,
        reference_text: str,
        engine: str = "qwen3",
        model_size: str = "0.6B",
        params: Optional[Dict[str, Any]] = None,
        qwen_model=None  # Qwen3TTS model instance for extraction
    ) -> Tuple[bool, str]:
        """
        Enroll a new voice by extracting and saving the voice embedding.
        
        Args:
            persona_id: Unique identifier for the persona
            reference_audio: Reference audio array (24kHz)
            reference_text: Text spoken in reference audio
            engine: TTS engine type
            model_size: Model size for qwen3
            params: Voice parameters (pitch, speed, etc.)
            qwen_model: Loaded Qwen3TTS model for embedding extraction
        
        Returns:
            (success: bool, message: str)
        """
        try:
            # Generate unique filename
            safe_id = "".join(c for c in persona_id if c.isalnum() or c in ('_', '-')).rstrip()
            
            # Save reference audio
            ref_path = self.profiles_dir / f"{safe_id}_reference.wav"
            self._save_audio(reference_audio, ref_path)
            
            # Extract voice embedding using Qwen3
            embedding_path = self.profiles_dir / f"{safe_id}_embedding.pt"
            
            if engine == "qwen3" and qwen_model is not None:
                # Extract x-vector from reference audio
                # Note: This depends on Qwen3's API for speaker embedding extraction
                embedding = self._extract_qwen_embedding(
                    qwen_model, reference_audio, reference_text
                )
                torch.save(embedding, embedding_path)
            else:
                # For StyleTTS2 or fallback, save reference audio as "embedding"
                # (StyleTTS2 uses reference audio directly)
                torch.save({
                    'type': 'reference_audio',
                    'audio': reference_audio,
                    'text': reference_text
                }, embedding_path)
            
            # Create profile record
            profile = VoiceProfile(
                persona_id=persona_id,
                engine=engine,
                model_size=model_size,
                embedding_path=str(embedding_path),
                reference_audio_path=str(ref_path),
                reference_text=reference_text,
                created_at=str(np.datetime64('now')),
                params=params or {}
            )
            
            self.profiles[persona_id] = profile
            self._save_metadata()
            
            print(f"[VoiceProfileManager] Enrolled voice for {persona_id}")
            print(f"  - Embedding: {embedding_path}")
            print(f"  - Reference: {ref_path}")
            
            return True, "Voice enrolled successfully"
            
        except Exception as e:
            print(f"[VoiceProfileManager] Enrollment failed: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def _extract_qwen_embedding(
        self,
        qwen_model,
        reference_audio: np.ndarray,
        reference_text: str
    ) -> torch.Tensor:
        """
        Extract speaker embedding (x-vector) from reference audio using Qwen3.
        
        This uses Qwen3's voice cloning capability to extract the speaker embedding
        that can be reused for future synthesis without the reference audio.
        """
        # Qwen3 specific embedding extraction
        # The model generates an internal speaker representation during voice cloning
        # We capture this for reuse
        
        # Note: Exact implementation depends on Qwen3's API
        # This is a placeholder - actual implementation would use qwen_model's
        # internal methods to extract the speaker embedding
        
        # For now, save a placeholder that includes reference info
        # In production, this would extract the actual x-vector
        embedding = {
            'type': 'qwen3_speaker_embedding',
            'reference_audio_shape': reference_audio.shape,
            'reference_text': reference_text,
            'model': 'qwen3',
            # Actual embedding would be extracted here
            'embedding': None  # Placeholder
        }
        
        return embedding
    
    def _save_audio(self, audio: np.ndarray, path: Path):
        """Save audio array to WAV file"""
        import scipy.io.wavfile as wavfile
        wavfile.write(path, 24000, (audio * 32767).astype(np.int16))
    
    def load_voice_profile(self, persona_id: str) -> Optional[VoiceProfile]:
        """Load a voice profile by persona ID"""
        profile = self.profiles.get(persona_id)
        if profile:
            # Verify files exist
            if not Path(profile.embedding_path).exists():
                print(f"[VoiceProfileManager] Embedding missing for {persona_id}")
                return None
            return profile
        return None
    
    def load_embedding(self, persona_id: str) -> Optional[torch.Tensor]:
        """Load just the embedding tensor for TTS"""
        profile = self.load_voice_profile(persona_id)
        if profile and Path(profile.embedding_path).exists():
            try:
                return torch.load(profile.embedding_path)
            except Exception as e:
                print(f"[VoiceProfileManager] Failed to load embedding: {e}")
        return None
    
    def delete_voice_profile(self, persona_id: str) -> bool:
        """Delete a voice profile and associated files"""
        profile = self.profiles.get(persona_id)
        if not profile:
            return False
        
        try:
            # Delete files
            if Path(profile.embedding_path).exists():
                Path(profile.embedding_path).unlink()
            if Path(profile.reference_audio_path).exists():
                Path(profile.reference_audio_path).unlink()
            
            # Remove from registry
            del self.profiles[persona_id]
            self._save_metadata()
            
            print(f"[VoiceProfileManager] Deleted voice profile for {persona_id}")
            return True
        except Exception as e:
            print(f"[VoiceProfileManager] Failed to delete profile: {e}")
            return False
    
    def list_profiles(self) -> Dict[str, VoiceProfile]:
        """List all enrolled voice profiles"""
        return self.profiles.copy()
    
    def has_profile(self, persona_id: str) -> bool:
        """Check if a voice profile exists"""
        return persona_id in self.profiles


# Singleton instance
_voice_profile_manager: Optional[VoiceProfileManager] = None

def get_voice_profile_manager(base_dir: Optional[str] = None) -> VoiceProfileManager:
    """Get or create the global VoiceProfileManager instance"""
    global _voice_profile_manager
    if _voice_profile_manager is None:
        _voice_profile_manager = VoiceProfileManager(base_dir)
    return _voice_profile_manager
