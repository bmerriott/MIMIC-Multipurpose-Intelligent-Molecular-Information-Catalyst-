"""
Mimic AI - Legal Watermarking System

Multi-layer watermarking for evidentiary purposes.
Provides forensic proof that audio was AI-generated.
"""

import numpy as np
import hashlib
from typing import Tuple, Optional
from scipy import signal


class LegalWatermarker:
    """
    Multi-layer audio watermarking for legal protection.
    
    Layers:
    1. Spread-spectrum (survives compression)
    2. Echo-based (survives filtering)
    3. Phase coding (survives amplitude changes)
    
    Each layer uses different frequencies and encoding methods
    for redundancy and robustness.
    """
    
    def __init__(self, user_id: str = "", timestamp: str = ""):
        """
        Args:
            user_id: Optional user identifier for tracking
            timestamp: Optional timestamp for evidence
        """
        # Layer 1: Public identifier (anyone can detect)
        self.public_key = "MIMIC-AI-GENERATED-AUDIO"
        
        # Layer 2: User-specific (if available)
        self.user_key = hashlib.sha256(user_id.encode()).hexdigest()[:16] if user_id else "ANONYMOUS"
        
        # Layer 3: Timestamp evidence
        self.time_key = timestamp if timestamp else ""
        
        # Configuration
        self.spread_strength = 0.003  # Very subtle
        self.echo_strength = 0.005
        self.phase_strength = 0.01
        
    def embed_watermark(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Embed multi-layer watermark into audio.
        
        Args:
            audio: Input audio array (-1.0 to 1.0)
            sample_rate: Sample rate in Hz
            
        Returns:
            Watermarked audio array
        """
        if len(audio.shape) > 1:
            # Handle stereo/multichannel
            result = np.copy(audio)
            for i in range(audio.shape[1]):
                result[:, i] = self.embed_watermark(audio[:, i], sample_rate)
            return result
        
        result = audio.copy().astype(np.float64)
        
        # Layer 1: Spread spectrum in mid frequencies (800-4000 Hz)
        result = self._embed_spread_spectrum(
            result, sample_rate, self.public_key, 
            strength=self.spread_strength
        )
        
        # Layer 2: Echo-based encoding (very robust)
        result = self._embed_echo_pattern(
            result, sample_rate, 
            strength=self.echo_strength
        )
        
        # Layer 3: Phase coding for user identification
        if self.user_key:
            result = self._embed_phase_coding(
                result, sample_rate, self.user_key,
                strength=self.phase_strength
            )
        
        # Ensure no clipping
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val * 0.99
            
        return result.astype(np.float32)
    
    def detect_watermark(self, audio: np.ndarray, sample_rate: int) -> Tuple[bool, float, dict]:
        """
        Detect watermark in audio.
        
        Args:
            audio: Audio to analyze
            sample_rate: Sample rate in Hz
            
        Returns:
            (detected, confidence_score, details)
            detected: True if watermark found
            confidence: 0.0-1.0 confidence score
            details: Dict with per-layer detection results
        """
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Detect each layer
        spread_detected, spread_conf = self._detect_spread_spectrum(
            audio, sample_rate, self.public_key
        )
        
        echo_detected, echo_conf = self._detect_echo_pattern(
            audio, sample_rate
        )
        
        phase_detected, phase_conf = False, 0.0
        if self.user_key:
            phase_detected, phase_conf = self._detect_phase_coding(
                audio, sample_rate, self.user_key
            )
        
        # Combined detection logic
        # At least 2 layers must be detected for confidence
        detections = sum([spread_detected, echo_detected, phase_detected])
        avg_confidence = np.mean([spread_conf, echo_conf, phase_conf])
        
        detected = detections >= 2 or (spread_detected and spread_conf > 0.8)
        
        details = {
            'spread_spectrum': {'detected': spread_detected, 'confidence': spread_conf},
            'echo_pattern': {'detected': echo_detected, 'confidence': echo_conf},
            'phase_coding': {'detected': phase_detected, 'confidence': phase_conf},
            'layers_detected': detections,
            'total_layers': 3 if self.user_key else 2
        }
        
        return detected, avg_confidence, details
    
    def _embed_spread_spectrum(self, audio: np.ndarray, sr: int, 
                                key: str, strength: float) -> np.ndarray:
        """Embed spread-spectrum watermark in mid frequencies."""
        # Bandpass filter audio to mid frequencies
        sos = signal.butter(4, [800, 4000], 'bandpass', fs=sr, output='sos')
        filtered = signal.sosfilt(sos, audio)
        
        # Generate chip sequence
        np.random.seed(int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32))
        chip_rate = 100
        num_chips = len(audio) // (sr // chip_rate) + 1
        chips = np.random.choice([-1, 1], size=num_chips)
        
        # Upsample to audio length
        chip_samples = sr // chip_rate
        watermark = np.repeat(chips, chip_samples)[:len(audio)]
        
        # Apply to filtered audio and mix back
        watermarked_filtered = filtered + strength * watermark * np.abs(filtered)
        
        # Replace mid frequencies in original
        result = audio - filtered + watermarked_filtered
        return result
    
    def _detect_spread_spectrum(self, audio: np.ndarray, sr: int, 
                                 key: str) -> Tuple[bool, float]:
        """Detect spread-spectrum watermark."""
        # Bandpass filter
        sos = signal.butter(4, [800, 4000], 'bandpass', fs=sr, output='sos')
        filtered = signal.sosfilt(sos, audio)
        
        # Generate expected watermark
        np.random.seed(int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32))
        chip_rate = 100
        num_chips = len(audio) // (sr // chip_rate) + 1
        chips = np.random.choice([-1, 1], size=num_chips)
        
        chip_samples = sr // chip_rate
        watermark = np.repeat(chips, chip_samples)[:len(audio)]
        
        # Correlate
        correlation = np.correlate(filtered, watermark, mode='valid')
        
        if len(correlation) == 0:
            return False, 0.0
        
        max_corr = np.max(np.abs(correlation))
        mean_corr = np.mean(np.abs(correlation))
        std_corr = np.std(np.abs(correlation))
        
        # Signal-to-noise ratio
        snr = (max_corr - mean_corr) / (std_corr + 1e-8)
        confidence = min(1.0, snr / 5.0)  # Normalize to 0-1
        
        detected = snr > 2.0
        return detected, confidence
    
    def _embed_echo_pattern(self, audio: np.ndarray, sr: int, 
                            strength: float) -> np.ndarray:
        """Embed echo-based watermark (very robust to compression)."""
        # Use multiple echo delays to encode data
        delays = [40, 60, 80]  # ms
        
        result = audio.copy()
        for i, delay_ms in enumerate(delays):
            delay_samples = int(sr * delay_ms / 1000)
            if delay_samples < len(audio):
                # Positive echo = 1, Negative = 0 (simplified)
                echo_strength = strength if i % 2 == 0 else -strength
                echo = np.zeros_like(audio)
                echo[delay_samples:] = audio[:-delay_samples] * echo_strength
                result = result + echo
        
        return result
    
    def _detect_echo_pattern(self, audio: np.ndarray, sr: int) -> Tuple[bool, float]:
        """Detect echo-based watermark using autocorrelation."""
        delays = [40, 60, 80]  # ms
        
        # Compute autocorrelation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
        
        scores = []
        for delay_ms in delays:
            delay_samples = int(sr * delay_ms / 1000)
            if delay_samples < len(autocorr):
                # Check for peaks at expected delays
                peak = autocorr[delay_samples]
                baseline = np.mean(autocorr[delay_samples-10:delay_samples+10])
                scores.append(peak / (baseline + 1e-8))
        
        if not scores:
            return False, 0.0
        
        avg_score = np.mean(scores)
        confidence = min(1.0, (avg_score - 1.0) * 2)
        detected = avg_score > 1.3
        
        return detected, confidence
    
    def _embed_phase_coding(self, audio: np.ndarray, sr: int, 
                            key: str, strength: float) -> np.ndarray:
        """Embed data in phase of frequency components."""
        # Process in overlapping windows
        window_size = 2048
        hop_size = 1024
        
        result = np.zeros_like(audio)
        window = np.hanning(window_size)
        
        for start in range(0, len(audio) - window_size, hop_size):
            end = start + window_size
            segment = audio[start:end] * window
            
            # FFT
            fft = np.fft.rfft(segment)
            magnitude = np.abs(fft)
            phase = np.angle(fft)
            
            # Embed key bits in phase of mid frequencies
            seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)
            np.random.seed(seed + start)  # Vary per window
            
            # Select bins to modify
            num_bins = len(phase)
            bins = np.random.choice(
                range(num_bins // 8, num_bins // 2),  # Mid freq range
                size=min(50, num_bins // 4),
                replace=False
            )
            
            # Modify phase based on key (use hash of key, not direct hex parsing)
            key_hash = hashlib.md5(key.encode()).hexdigest()
            for i, b in enumerate(bins):
                bit = int(key_hash[i % len(key_hash)], 16) % 2
                phase_shift = strength if bit else -strength
                phase[b] += phase_shift
            
            # Reconstruct
            fft_new = magnitude * np.exp(1j * phase)
            segment_new = np.fft.irfft(fft_new, n=window_size) * window
            
            # Overlap-add
            result[start:end] += segment_new
        
        # Normalize for overlap-add
        result *= 0.5
        
        # Mix with original (preserve some original characteristics)
        return 0.7 * audio + 0.3 * result
    
    def _detect_phase_coding(self, audio: np.ndarray, sr: int, 
                             key: str) -> Tuple[bool, float]:
        """Detect phase-coded watermark."""
        window_size = 2048
        hop_size = 1024
        
        correlations = []
        
        for start in range(0, len(audio) - window_size, hop_size):
            end = start + window_size
            segment = audio[start:end] * np.hanning(window_size)
            
            # FFT
            fft = np.fft.rfft(segment)
            phase = np.angle(fft)
            
            # Expected phase shifts
            seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)
            np.random.seed(seed + start)
            
            num_bins = len(phase)
            bins = np.random.choice(
                range(num_bins // 8, num_bins // 2),
                size=min(50, num_bins // 4),
                replace=False
            )
            
            # Check correlation with expected pattern
            expected_shifts = []
            actual_shifts = []
            
            key_hash = hashlib.md5(key.encode()).hexdigest()
            for i, b in enumerate(bins):
                bit = int(key_hash[i % len(key_hash)], 16) % 2
                expected = self.phase_strength if bit else -self.phase_strength
                
                # Need reference - compare with unmarked expectation
                # This is simplified detection
                expected_shifts.append(expected)
                actual_shifts.append(np.sin(phase[b]))  # Approximate
            
            if expected_shifts and actual_shifts:
                corr = np.corrcoef(expected_shifts, actual_shifts)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if not correlations:
            return False, 0.0
        
        avg_corr = np.mean(correlations)
        confidence = (avg_corr + 1) / 2  # Normalize -1,1 to 0,1
        detected = avg_corr > 0.3
        
        return detected, confidence


# Simple public detection function for the standalone tool
def detect_ai_watermark(audio: np.ndarray, sample_rate: int) -> Tuple[bool, float, str]:
    """
    Public detection function.
    
    Returns:
        (detected, confidence, message)
    """
    watermarker = LegalWatermarker()
    detected, confidence, details = watermarker.detect_watermark(audio, sample_rate)
    
    if detected:
        message = f"AI-generated watermark DETECTED (confidence: {confidence:.1%})"
        if details['layers_detected'] >= 2:
            message += f" - {details['layers_detected']}/{details['total_layers']} layers verified"
    else:
        message = f"No AI watermark detected (confidence: {confidence:.1%})"
    
    return detected, confidence, message
