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
        
        # Configuration - INCREASED strengths for better detection
        self.spread_strength = 0.008  # Increased from 0.003
        self.echo_strength = 0.015    # Increased from 0.005
        self.phase_strength = 0.02    # Increased from 0.01
        
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
        # Spread spectrum is the primary layer; echo and phase are secondary
        
        # Primary: Spread spectrum detection
        primary_detected = spread_detected
        primary_confidence = spread_conf
        
        # Secondary: Echo and phase provide additional confidence
        secondary_detections = sum([echo_detected, phase_detected])
        secondary_boost = 0.1 * secondary_detections  # Small boost per secondary layer
        
        # Final confidence combines primary with secondary boost
        final_confidence = min(1.0, primary_confidence + secondary_boost)
        
        # Detection decision: primary must be detected, or 2+ secondary layers
        detected = primary_detected or (secondary_detections >= 2)
        
        details = {
            'spread_spectrum': {'detected': spread_detected, 'confidence': spread_conf},
            'echo_pattern': {'detected': echo_detected, 'confidence': echo_conf},
            'phase_coding': {'detected': phase_detected, 'confidence': phase_conf},
            'layers_detected': (1 if spread_detected else 0) + secondary_detections,
            'total_layers': 3,
            'primary_layer': 'spread_spectrum'
        }
        
        return detected, final_confidence, details
        
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
        
        # Generate chip sequence - FIXED: consistent calculation with detection
        np.random.seed(int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32))
        chip_rate = 100
        chip_samples = int(sr / chip_rate)
        num_chips = int(len(audio) / chip_samples) + 1
        chips = np.random.choice([-1, 1], size=num_chips)
        
        # Upsample to audio length
        watermark = np.repeat(chips, chip_samples)[:len(audio)]
        
        # Apply to filtered audio and mix back - ENHANCED strength
        watermarked_filtered = filtered + strength * watermark * (np.abs(filtered) + 0.1)
        
        # Replace mid frequencies in original
        result = audio - filtered + watermarked_filtered
        return result
    
    def _detect_spread_spectrum(self, audio: np.ndarray, sr: int, 
                                 key: str) -> Tuple[bool, float]:
        """Detect spread-spectrum watermark - ENHANCED detection."""
        # Bandpass filter
        sos = signal.butter(4, [800, 4000], 'bandpass', fs=sr, output='sos')
        filtered = signal.sosfilt(sos, audio)
        
        # Generate expected watermark - FIXED: consistent calculation with embedding
        np.random.seed(int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32))
        chip_rate = 100
        chip_samples = int(sr / chip_rate)
        num_chips = int(len(audio) / chip_samples) + 1
        chips = np.random.choice([-1, 1], size=num_chips)
        
        watermark = np.repeat(chips, chip_samples)[:len(audio)]
        
        # Multi-lag correlation for better detection
        correlations = []
        for lag in range(0, min(100, len(filtered) - len(watermark)), 10):
            segment = filtered[lag:lag+len(watermark)]
            if len(segment) == len(watermark):
                corr = np.corrcoef(segment, watermark)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if not correlations:
            # Fallback to simple correlation
            correlation = np.correlate(filtered, watermark, mode='valid')
            if len(correlation) == 0:
                return False, 0.0
            max_corr = np.max(np.abs(correlation))
            mean_corr = np.mean(np.abs(correlation))
            std_corr = np.std(np.abs(correlation))
            snr = (max_corr - mean_corr) / (std_corr + 1e-8)
            confidence = min(1.0, snr / 3.0)
            detected = snr > 1.5
            return detected, confidence
        
        # Use best correlation found
        best_corr = max(correlations)
        confidence = min(1.0, best_corr)
        detected = best_corr > 0.3  # Lower threshold for correlation
        
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
        """Detect echo-based watermark using enhanced cepstral analysis."""
        delays = [40, 60, 80]  # ms
        
        # Use power spectrum for better echo detection
        frame_size = int(sr * 0.025)  # 25ms frames
        hop_size = int(sr * 0.010)    # 10ms hop
        
        all_scores = []
        
        # Process in frames for better SNR
        for frame_start in range(0, len(audio) - frame_size, hop_size * 5):  # Skip some frames
            frame = audio[frame_start:frame_start + frame_size]
            if len(frame) < frame_size:
                continue
                
            # Window the frame
            windowed = frame * np.hanning(len(frame))
            
            # Compute power spectrum
            fft = np.fft.rfft(windowed)
            power = np.abs(fft) ** 2
            
            # Cepstrum (inverse FFT of log power)
            log_power = np.log(power + 1e-10)
            cepstrum = np.fft.irfft(log_power)
            
            # Check for peaks at expected quefrencies
            for delay_ms in delays:
                quefrency_samples = int(delay_ms * sr / 1000)
                if quefrency_samples < len(cepstrum):
                    # Look for peak in a small window
                    window_start = max(0, quefrency_samples - 2)
                    window_end = min(len(cepstrum), quefrency_samples + 3)
                    peak = np.max(cepstrum[window_start:window_end])
                    
                    # Compare to surrounding
                    surrounding = np.concatenate([
                        cepstrum[max(0, quefrency_samples-10):window_start],
                        cepstrum[window_end:min(len(cepstrum), quefrency_samples+10)]
                    ])
                    if len(surrounding) > 0:
                        baseline = np.mean(np.abs(surrounding))
                        score = peak / (baseline + 1e-8)
                        all_scores.append(score)
        
        if not all_scores:
            # Fallback to simple autocorrelation
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            scores = []
            for delay_ms in delays:
                delay_samples = int(sr * delay_ms / 1000)
                if delay_samples < len(autocorr):
                    peak = autocorr[delay_samples]
                    baseline = np.mean(autocorr[delay_samples-10:delay_samples+10])
                    scores.append(peak / (baseline + 1e-8))
            
            if not scores:
                return False, 0.0
            
            avg_score = np.mean(scores)
            confidence = min(1.0, max(0, (avg_score - 1.1) * 5))
            detected = avg_score > 1.25
            return detected, confidence
        
        # Use aggregated scores
        avg_score = np.mean(all_scores)
        confidence = min(1.0, max(0, (avg_score - 1.0) * 3))
        detected = avg_score > 1.2
        
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
