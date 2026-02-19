"""
Streaming TTS Pipeline
Implements low-latency voice synthesis with sentence-level chunking and audio streaming.
"""

import re
import numpy as np
import threading
import queue
import asyncio
from collections import deque
from typing import Optional, Callable, AsyncGenerator, List, Tuple
import time
from dataclasses import dataclass


@dataclass
class AudioChunk:
    """Represents a chunk of audio with metadata"""
    audio: np.ndarray
    text: str
    is_first: bool = False
    is_last: bool = False
    chunk_index: int = 0


class AudioRingBuffer:
    """
    Thread-safe ring buffer for audio chunks.
    Producer (TTS generation) writes to buffer.
    Consumer (audio playback) reads from buffer.
    """
    
    def __init__(self, max_chunks: int = 5):
        self.max_chunks = max_chunks
        self._buffer: deque = deque(maxlen=max_chunks)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._closed = False
    
    def write(self, chunk: AudioChunk, timeout: Optional[float] = None) -> bool:
        """
        Write audio chunk to buffer.
        Blocks if buffer is full until space available.
        """
        with self._not_full:
            if self._closed:
                return False
            
            # Wait until buffer has space
            if len(self._buffer) >= self.max_chunks:
                if not self._not_full.wait(timeout):
                    return False  # Timeout
            
            if self._closed:
                return False
            
            self._buffer.append(chunk)
            self._not_empty.notify()
            return True
    
    def read(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        """
        Read audio chunk from buffer.
        Blocks if buffer is empty until data available.
        Returns None if buffer is closed and empty.
        """
        with self._not_empty:
            # Wait until buffer has data or is closed
            while len(self._buffer) == 0 and not self._closed:
                if not self._not_empty.wait(timeout):
                    return None  # Timeout
            
            if len(self._buffer) == 0:
                return None  # Closed and empty
            
            chunk = self._buffer.popleft()
            self._not_full.notify()
            return chunk
    
    def available(self) -> int:
        """Number of chunks currently in buffer"""
        with self._lock:
            return len(self._buffer)
    
    def close(self):
        """Signal that no more data will be written"""
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
    
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed


def chunk_text_intelligently(text: str, max_chunk_length: int = 200) -> List[str]:
    """
    Split text into natural chunks at sentence boundaries.
    
    Strategy:
    1. Split on sentence boundaries (. ! ?)
    2. If sentence is too long, split on clause boundaries (, ; :)
    3. If still too long, split on word boundaries
    """
    if len(text) <= max_chunk_length:
        return [text]
    
    chunks = []
    
    # First, try to split on sentence boundaries
    # Pattern matches sentence-ending punctuation followed by space or end
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If adding this sentence would exceed limit
        if len(current_chunk) + len(sentence) + 1 > max_chunk_length:
            # Save current chunk if not empty
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If sentence itself is too long, split it
            if len(sentence) > max_chunk_length:
                # Split on clause boundaries
                clause_chunks = _split_on_clauses(sentence, max_chunk_length)
                chunks.extend(clause_chunks)
                current_chunk = ""
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_on_clauses(text: str, max_length: int) -> List[str]:
    """Split long sentences on clause boundaries"""
    # Split on clause boundaries
    clause_pattern = r'(?<=[,;:])\s+'
    clauses = re.split(clause_pattern, text)
    
    chunks = []
    current = ""
    
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        
        if len(current) + len(clause) + 1 > max_length:
            if current:
                chunks.append(current.strip())
            
            # If clause is still too long, split on words
            if len(clause) > max_length:
                word_chunks = _split_on_words(clause, max_length)
                chunks.extend(word_chunks)
                current = ""
            else:
                current = clause
        else:
            if current:
                current += " " + clause
            else:
                current = clause
    
    if current:
        chunks.append(current.strip())
    
    return chunks


def _split_on_words(text: str, max_length: int) -> List[str]:
    """Split on word boundaries as last resort"""
    words = text.split()
    chunks = []
    current = ""
    
    for word in words:
        if len(current) + len(word) + 1 > max_length:
            if current:
                chunks.append(current.strip())
            current = word
        else:
            if current:
                current += " " + word
            else:
                current = word
    
    if current:
        chunks.append(current.strip())
    
    return chunks


def crossfade_chunks(
    chunk1: np.ndarray,
    chunk2: np.ndarray,
    crossfade_samples: int = 1200,  # 50ms at 24kHz
    fade_type: str = "linear"
) -> np.ndarray:
    """
    Apply crossfade between two audio chunks to smooth transitions.
    
    Args:
        chunk1: First audio chunk
        chunk2: Second audio chunk
        crossfade_samples: Number of samples to crossfade
        fade_type: "linear" or "equal_power"
    
    Returns:
        Combined audio with crossfade
    """
    if len(chunk1) < crossfade_samples or len(chunk2) < crossfade_samples:
        # Not enough samples for crossfade, just concatenate
        return np.concatenate([chunk1, chunk2])
    
    if fade_type == "linear":
        # Linear fade
        fade_out = np.linspace(1, 0, crossfade_samples)
        fade_in = np.linspace(0, 1, crossfade_samples)
    elif fade_type == "equal_power":
        # Equal power fade (better for audio)
        t = np.linspace(0, 1, crossfade_samples)
        fade_out = np.cos(t * np.pi / 2)
        fade_in = np.sin(t * np.pi / 2)
    else:
        raise ValueError(f"Unknown fade type: {fade_type}")
    
    # Apply fades
    chunk1_faded = chunk1.copy()
    chunk1_faded[-crossfade_samples:] *= fade_out
    
    chunk2_faded = chunk2.copy()
    chunk2_faded[:crossfade_samples] *= fade_in
    
    # Mix the crossfade region
    crossfade_region = (
        chunk1_faded[-crossfade_samples:] + 
        chunk2_faded[:crossfade_samples]
    )
    
    # Combine: chunk1 (minus crossfade tail) + crossfade + chunk2 (minus crossfade head)
    result = np.concatenate([
        chunk1_faded[:-crossfade_samples],
        crossfade_region,
        chunk2_faded[crossfade_samples:]
    ])
    
    return result


class StreamingTTS:
    """
    Streaming TTS system with producer-consumer architecture.
    
    Usage:
        tts = StreamingTTS(qwen_model, voice_profile_id)
        
        # Start generation and playback
        await tts.speak_streaming("Your long text here")
    """
    
    def __init__(
        self,
        tts_model,
        voice_profile_id: str,
        sample_rate: int = 24000,
        buffer_size: int = 3,
        chunk_max_length: int = 200,
        crossfade_ms: int = 50
    ):
        self.tts_model = tts_model
        self.voice_profile_id = voice_profile_id
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.chunk_max_length = chunk_max_length
        self.crossfade_samples = int(sample_rate * crossfade_ms / 1000)
        
        self.ring_buffer = AudioRingBuffer(max_chunks=buffer_size)
        self.generation_thread: Optional[threading.Thread] = None
        self.is_generating = False
        
        # Stats
        self.chunks_generated = 0
        self.total_generation_time = 0.0
        self.first_chunk_latency = 0.0
    
    def generate_chunks(
        self,
        text: str,
        reference_audio: Optional[np.ndarray] = None,
        reference_text: Optional[str] = None
    ):
        """
        Producer: Generate audio chunks in background thread.
        Writes chunks to ring buffer.
        """
        self.is_generating = True
        start_time = time.time()
        
        try:
            # Split text into chunks
            text_chunks = chunk_text_intelligently(text, self.chunk_max_length)
            total_chunks = len(text_chunks)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_start = time.time()
                
                # Generate audio for this chunk
                # Note: Actual implementation depends on Qwen3's API
                # This is a placeholder structure
                try:
                    # Generate using saved voice profile
                    audio_chunk = self._generate_chunk(
                        chunk_text,
                        reference_audio,
                        reference_text
                    )
                    
                    # Track latency for first chunk
                    if i == 0:
                        self.first_chunk_latency = time.time() - start_time
                    
                    # Create AudioChunk
                    chunk = AudioChunk(
                        audio=audio_chunk,
                        text=chunk_text,
                        is_first=(i == 0),
                        is_last=(i == total_chunks - 1),
                        chunk_index=i
                    )
                    
                    # Write to ring buffer (blocks if full)
                    success = self.ring_buffer.write(chunk, timeout=30.0)
                    if not success:
                        print("[StreamingTTS] Buffer write timeout")
                        break
                    
                    self.chunks_generated += 1
                    self.total_generation_time += time.time() - chunk_start
                    
                except Exception as e:
                    print(f"[StreamingTTS] Chunk generation failed: {e}")
                    break
        
        finally:
            # Signal completion
            self.ring_buffer.close()
            self.is_generating = False
    
    def _generate_chunk(
        self,
        text: str,
        reference_audio: Optional[np.ndarray],
        reference_text: Optional[str]
    ) -> np.ndarray:
        """
        Generate audio for a single text chunk.
        
        In production, this would:
        1. Load the saved voice embedding
        2. Use Qwen3's generate method with the embedding
        3. Return the audio array
        """
        # Placeholder: return silence
        # Actual implementation would use:
        # audio = self.tts_model.generate(
        #     text=text,
        #     voice_profile=self.voice_profile_id,
        #     ...
        # )
        duration_samples = int(self.sample_rate * 0.5)  # 0.5s placeholder
        return np.zeros(duration_samples, dtype=np.float32)
    
    async def stream_audio(self) -> AsyncGenerator[AudioChunk, None]:
        """
        Consumer: Async generator that yields audio chunks as they're ready.
        
        Usage:
            async for chunk in tts.stream_audio():
                play_audio(chunk.audio)
        """
        while True:
            chunk = self.ring_buffer.read(timeout=1.0)
            
            if chunk is None:
                # Buffer closed and empty
                break
            
            yield chunk
    
    def get_stats(self) -> dict:
        """Get generation statistics"""
        avg_chunk_time = (
            self.total_generation_time / self.chunks_generated
            if self.chunks_generated > 0 else 0
        )
        return {
            "chunks_generated": self.chunks_generated,
            "first_chunk_latency_ms": self.first_chunk_latency * 1000,
            "avg_chunk_time_ms": avg_chunk_time * 1000,
            "buffer_available": self.ring_buffer.available(),
            "is_generating": self.is_generating
        }
    
    def stop(self):
        """Stop generation and clear buffer"""
        self.ring_buffer.close()
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=5.0)


class AdaptiveBufferManager:
    """
    Dynamically adjusts buffer size based on generation speed.
    
    If generation is slower than real-time, increase buffer.
    If generation is faster, decrease buffer for lower latency.
    """
    
    def __init__(
        self,
        min_buffer: int = 2,
        max_buffer: int = 8,
        target_latency_ms: float = 500
    ):
        self.min_buffer = min_buffer
        self.max_buffer = max_buffer
        self.target_latency_ms = target_latency_ms
        self.current_buffer_size = min_buffer
        
        # Running averages
        self.recent_latencies: deque = deque(maxlen=10)
    
    def update(self, chunk_latency_ms: float):
        """Update buffer size based on recent latency"""
        self.recent_latencies.append(chunk_latency_ms)
        
        if len(self.recent_latencies) < 3:
            return  # Not enough data
        
        avg_latency = np.mean(self.recent_latencies)
        
        # Adjust buffer size
        if avg_latency > self.target_latency_ms * 1.5:
            # Generation is slow, increase buffer
            self.current_buffer_size = min(
                self.current_buffer_size + 1,
                self.max_buffer
            )
        elif avg_latency < self.target_latency_ms * 0.5:
            # Generation is fast, can decrease buffer
            self.current_buffer_size = max(
                self.current_buffer_size - 1,
                self.min_buffer
            )
    
    def get_buffer_size(self) -> int:
        return self.current_buffer_size
