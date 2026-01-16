"""
ASR Model Service
Manages Whisper model lifecycle, transcription with quality filtering, and batch processing.
"""

import os
import re
import time
import asyncio
import logging
import threading
from typing import Optional, Tuple, List, Sequence, Dict, Any
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

import numpy as np
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from Asr.utils import weighted_vote
from Asr._preprocessing import clean_audio

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

class ASRConfig:
    """Centralized ASR configuration with validation."""
    
    # Model
    DEFAULT_MODEL: str = os.getenv("WHISPER_MODEL", "tiny")
    DEVICE: str = os.getenv("WHISPER_DEVICE", "auto")
    COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
    
    # Quality thresholds
    NO_SPEECH_MAX: float = float(os.getenv("NO_SPEECH_MAX", "0.50"))
    AVG_LOGPROB_MIN: float = float(os.getenv("AVG_LOGPROB_MIN", "-1.10"))
    COMPRESSION_RATIO_MAX: float = float(os.getenv("COMPRESSION_RATIO_MAX", "2.4"))
    MIN_CHARS: int = int(os.getenv("MIN_CHARS", "6"))
    
    # Banned phrases
    BANNED_SHORT: set = {
        x.strip().lower()
        for x in os.getenv("BANNED_SHORT", "bye,thanks,thank you,okay,ok").split(",")
        if x.strip()
    }
    
    # Performance
    TRANSCRIBE_TIMEOUT: float = float(os.getenv("TRANSCRIBE_TIMEOUT", "30.0"))
    MAX_AUDIO_LENGTH: float = float(os.getenv("MAX_AUDIO_LENGTH", "30.0"))  # seconds
    BEAM_SIZE: int = int(os.getenv("BEAM_SIZE", "5"))
    BEST_OF: int = int(os.getenv("BEST_OF", "5"))
    
    # Retry
    LOAD_RETRIES: int = int(os.getenv("LOAD_RETRIES", "3"))
    LOAD_BACKOFF: float = float(os.getenv("LOAD_BACKOFF", "2.0"))
    
    # Features
    ENABLE_VAD: bool = os.getenv("ENABLE_VAD", "true").lower() == "true"
    ENABLE_PREPROCESSING: bool = os.getenv("ENABLE_PREPROCESSING", "false").lower() == "true"
    ENABLE_TEXT_CLEANING: bool = os.getenv("ENABLE_TEXT_CLEANING", "true").lower() == "true"
    
    # Batch processing
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "32"))
    BATCH_TIMEOUT: float = float(os.getenv("BATCH_TIMEOUT", "300.0"))
    
    @classmethod
    def validate(cls):
        """Validate configuration values."""
        if cls.NO_SPEECH_MAX < 0 or cls.NO_SPEECH_MAX > 1:
            raise ValueError("NO_SPEECH_MAX must be between 0 and 1")
        
        if cls.COMPRESSION_RATIO_MAX < 1:
            raise ValueError("COMPRESSION_RATIO_MAX must be >= 1")
        
        if cls.MIN_CHARS < 0:
            raise ValueError("MIN_CHARS must be non-negative")
        
        if cls.TRANSCRIBE_TIMEOUT <= 0:
            raise ValueError("TRANSCRIBE_TIMEOUT must be positive")
        
        if cls.MAX_BATCH_SIZE < 1:
            raise ValueError("MAX_BATCH_SIZE must be positive")


ASRConfig.validate()

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ============================================================================
# Metrics
# ============================================================================

class ASRMetrics:
    """Track ASR performance metrics."""
    
    def __init__(self):
        self.transcription_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0
        self.filtered_segments = 0
        self.total_segments = 0
        self.lock = threading.Lock()
    
    def record_transcription(self, latency_ms: float, segments_total: int, segments_kept: int):
        with self.lock:
            self.transcription_count += 1
            self.total_latency_ms += latency_ms
            self.total_segments += segments_total
            self.filtered_segments += (segments_total - segments_kept)
    
    def record_error(self):
        with self.lock:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            avg_latency = (
                self.total_latency_ms / self.transcription_count
                if self.transcription_count > 0
                else 0
            )
            filter_rate = (
                self.filtered_segments / self.total_segments
                if self.total_segments > 0
                else 0
            )
            
            return {
                "transcriptions_total": self.transcription_count,
                "errors_total": self.error_count,
                "avg_latency_ms": round(avg_latency, 2),
                "segments_filtered_rate": round(filter_rate, 3),
                "segments_total": self.total_segments,
            }
    
    def reset(self):
        with self.lock:
            self.transcription_count = 0
            self.error_count = 0
            self.total_latency_ms = 0.0
            self.filtered_segments = 0
            self.total_segments = 0


metrics = ASRMetrics()

# ============================================================================
# Device Selection
# ============================================================================

class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class ComputeType(Enum):
    INT8 = "int8"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    AUTO = "auto"


def select_device_compute(
    device: str = "auto",
    compute: str = "auto"
) -> Tuple[str, str]:
    """
    Select optimal device and compute type with fallback chain.
    
    Returns:
        (device, compute_type) tuple
    """
    try:
        # Device selection
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                if device == "cuda":
                    # Log GPU info
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    log.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
            except ImportError:
                log.warning("PyTorch not available, falling back to CPU")
                device = "cpu"
        
        # Compute type selection
        if compute == "auto":
            if device == "cuda":
                # Try float16 first, fallback to int8 if unsupported
                compute = "float16"
            else:
                compute = "int8"
        
        log.info(f"Selected device={device}, compute_type={compute}")
        return device, compute
    
    except Exception as e:
        log.error(f"Device selection error: {e}", exc_info=True)
        log.warning("Falling back to CPU/int8")
        return "cpu", "int8"


# ============================================================================
# Model Management
# ============================================================================

@dataclass
class ModelInfo:
    """Metadata about loaded model."""
    size: str
    device: str
    compute_type: str
    loaded_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "device": self.device,
            "compute_type": self.compute_type,
            "loaded_at": self.loaded_at,
            "uptime_seconds": time.time() - self.loaded_at,
        }


class ModelManager:
    """Thread-safe Whisper model manager with lifecycle control."""
    
    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self.model_info: Optional[ModelInfo] = None
        self.lock = threading.RLock()  # Reentrant lock
    
    @contextmanager
    def _model_context(self):
        """Context manager for thread-safe model access."""
        self.lock.acquire()
        try:
            if self.model is None:
                raise RuntimeError(
                    "Whisper model not loaded. Call load_model() first."
                )
            yield self.model
        finally:
            self.lock.release()
    
    def load_model(
        self,
        model_size: str = ASRConfig.DEFAULT_MODEL,
        device: str = ASRConfig.DEVICE,
        compute: str = ASRConfig.COMPUTE_TYPE,
        force_reload: bool = False,
    ) -> ModelInfo:
        """
        Load Whisper model with retry logic and validation.
        
        Args:
            model_size: Model variant (tiny, base, small, medium, large-v3)
            device: Device to run on (cpu, cuda, auto)
            compute: Compute type (int8, float16, float32, auto)
            force_reload: Force reload even if already loaded
        
        Returns:
            ModelInfo with metadata
        
        Raises:
            RuntimeError: If loading fails after retries
        """
        with self.lock:
            # Check if already loaded
            if not force_reload and self.model is not None:
                if self.model_info and self.model_info.size == model_size:
                    log.info(f"Model already loaded: {model_size}")
                    return self.model_info
                else:
                    log.info("Unloading existing model for different size")
                    self._unload_unsafe()
            
            # Select device/compute
            device, compute = select_device_compute(device, compute)
            
            # Retry loop
            last_error = None
            for attempt in range(1, ASRConfig.LOAD_RETRIES + 1):
                try:
                    log.info(
                        f"Loading Whisper model (attempt {attempt}/{ASRConfig.LOAD_RETRIES}): "
                        f"size={model_size}, device={device}, compute={compute}"
                    )
                    
                    start = time.perf_counter()
                    loaded_model = WhisperModel(
                        model_size,
                        device=device,
                        compute_type=compute,
                    )
                    elapsed = time.perf_counter() - start
                    
                    # Validate model by running quick test
                    try:
                        test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                        segments, _ = loaded_model.transcribe(test_audio, beam_size=1)
                        list(segments)  # Force evaluation
                        log.info("Model validation successful")
                    except Exception as e:
                        raise RuntimeError(f"Model validation failed: {e}")
                    
                    # Success - store model
                    self.model = loaded_model
                    self.model_info = ModelInfo(
                        size=model_size,
                        device=device,
                        compute_type=compute,
                        loaded_at=time.time(),
                    )
                    
                    log.info(
                        f"Whisper model loaded successfully in {elapsed:.1f}s: "
                        f"{model_size} ({device}/{compute})"
                    )
                    return self.model_info
                
                except Exception as e:
                    last_error = e
                    log.error(
                        f"Attempt {attempt} failed: {e}",
                        exc_info=(attempt == ASRConfig.LOAD_RETRIES)  # Full trace on last attempt
                    )
                    
                    if attempt < ASRConfig.LOAD_RETRIES:
                        sleep_time = ASRConfig.LOAD_BACKOFF ** attempt
                        log.info(f"Retrying in {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
            
            # All retries exhausted
            raise RuntimeError(
                f"Failed to load Whisper model after {ASRConfig.LOAD_RETRIES} attempts"
            ) from last_error
    
    def unload_model(self):
        """Unload model and free resources."""
        with self.lock:
            self._unload_unsafe()
    
    def _unload_unsafe(self):
        """Internal unload (assumes lock is held)."""
        if self.model is not None:
            try:
                # faster-whisper doesn't have explicit cleanup, but we can del
                del self.model
                self.model = None
                self.model_info = None
                log.info("Whisper model unloaded successfully")
            except Exception as e:
                log.warning(f"Error unloading model: {e}")
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get current model metadata."""
        with self.lock:
            return self.model_info
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        with self.lock:
            return self.model is not None
    
    def health_check(self) -> Dict[str, Any]:
        """Return model health status."""
        with self.lock:
            if self.model is None:
                return {
                    "loaded": False,
                    "status": "not_loaded",
                }
            
            return {
                "loaded": True,
                "status": "healthy",
                "model_info": self.model_info.to_dict() if self.model_info else None,
            }


# Global model manager
model_manager = ModelManager()


# ============================================================================
# Segment Quality Filtering
# ============================================================================

@dataclass
class SegmentStats:
    """Statistics about segment filtering."""
    total: int = 0
    kept: int = 0
    filtered_no_speech: int = 0
    filtered_low_prob: int = 0
    filtered_compression: int = 0
    filtered_too_short: int = 0
    filtered_banned: int = 0


def filter_segment(seg, stats: Optional[SegmentStats] = None) -> bool:
    """
    Check if segment passes quality thresholds.
    
    Args:
        seg: Whisper segment object
        stats: Optional stats object to track filtering reasons
    
    Returns:
        True if segment should be kept
    """
    if stats:
        stats.total += 1
    
    # Extract text
    txt = (seg.text or "").strip().lower()
    
    # No speech probability
    no_speech_prob = getattr(seg, "no_speech_prob", 0)
    if no_speech_prob > ASRConfig.NO_SPEECH_MAX:
        if stats:
            stats.filtered_no_speech += 1
        return False
    
    # Average log probability (confidence)
    avg_logprob = getattr(seg, "avg_logprob", 0)
    if avg_logprob < ASRConfig.AVG_LOGPROB_MIN:
        if stats:
            stats.filtered_low_prob += 1
        return False
    
    # Compression ratio (repetitiveness)
    compression_ratio = getattr(seg, "compression_ratio", 1.0)
    if compression_ratio > ASRConfig.COMPRESSION_RATIO_MAX:
        if stats:
            stats.filtered_compression += 1
        return False
    
    # Minimum length
    if len(txt) < ASRConfig.MIN_CHARS:
        if stats:
            stats.filtered_too_short += 1
        return False
    
    # Banned phrases
    if txt in ASRConfig.BANNED_SHORT:
        if stats:
            stats.filtered_banned += 1
        return False
    
    if stats:
        stats.kept += 1
    return True


# ============================================================================
# Text Cleaning
# ============================================================================

# Regex patterns (compiled once)
_NGRAM_RE = re.compile(
    r'\b(\w+(?:\s+\w+){1,5})\b(?:\s+\1\b)+',
    flags=re.IGNORECASE
)
_WORD_REPEAT_RE = re.compile(
    r'\b(\w+)( \1){1,}\b',
    flags=re.IGNORECASE
)
_SPACE_RE = re.compile(r'\s+')

# Filler words to remove
FILLER_WORDS = [
    "okay", "um", "uh", "like", "you know", "i mean",
    "just got", "let's go", "sort of", "kind of"
]


def clean_text(text: str) -> str:
    """
    Normalize ASR output by removing fillers, duplicates, and extra spaces.
    
    Args:
        text: Raw transcription text
    
    Returns:
        Cleaned text
    """
    if not text or not ASRConfig.ENABLE_TEXT_CLEANING:
        return text.strip()
    
    original_text = text
    
    # Remove filler words
    for filler in FILLER_WORDS:
        pattern = rf"\b{re.escape(filler)}\b"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Remove repeated single words ("the the the" -> "the")
    text = _WORD_REPEAT_RE.sub(r"\1", text)
    
    # Remove repeated n-grams (up to 6 words)
    text = _NGRAM_RE.sub(lambda m: m.group(1), text)
    
    # Deduplicate sentences
    sentences = []
    seen_sentences = set()
    for sentence in re.split(r'(?<=[.!?])\s+', text.strip()):
        sentence_norm = sentence.lower().strip()
        if sentence_norm and sentence_norm not in seen_sentences:
            sentences.append(sentence.strip())
            seen_sentences.add(sentence_norm)
    text = " ".join(sentences)
    
    # Normalize whitespace
    text = _SPACE_RE.sub(" ", text).strip()
    
    # Log if significant cleaning occurred
    if len(original_text) - len(text) > 50:
        log.debug(f"Text cleaning reduced length by {len(original_text) - len(text)} chars")
    
    return text


# ============================================================================
# Audio Validation
# ============================================================================

def validate_audio(audio: np.ndarray) -> np.ndarray:
    """
    Validate and normalize audio input.
    
    Args:
        audio: Input audio array
    
    Returns:
        Validated audio array
    
    Raises:
        ValueError: If audio is invalid
    """
    if audio is None or audio.size == 0:
        raise ValueError("Audio array is empty")
    
    # Check dtype
    if audio.dtype not in (np.float32, np.float64, np.int16):
        raise ValueError(f"Unsupported audio dtype: {audio.dtype}")
    
    # Convert to float32 if needed
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)
    
    # Check shape (should be 1D)
    if audio.ndim != 1:
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio[:, 0]
        else:
            raise ValueError(f"Audio must be 1D, got shape {audio.shape}")
    
    # Check length
    sample_rate = 16000  # Whisper expects 16kHz
    duration_seconds = len(audio) / sample_rate
    if duration_seconds > ASRConfig.MAX_AUDIO_LENGTH:
        log.warning(
            f"Audio length {duration_seconds:.1f}s exceeds max {ASRConfig.MAX_AUDIO_LENGTH}s, "
            f"truncating"
        )
        audio = audio[:int(ASRConfig.MAX_AUDIO_LENGTH * sample_rate)]
    
    # Check for silence
    if np.max(np.abs(audio)) < 1e-6:
        log.warning("Audio appears to be silent (max amplitude < 1e-6)")
    
    return audio


# ============================================================================
# Core Transcription
# ============================================================================

@dataclass
class TranscriptionResult:
    """Result of transcription with metadata."""
    text: str
    latency_ms: float
    segments_total: int
    segments_kept: int
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "latency_ms": round(self.latency_ms, 2),
            "segments_total": self.segments_total,
            "segments_kept": self.segments_kept,
            "language": self.language,
        }


def transcribe(
    audio: np.ndarray,
    timeout: float = ASRConfig.TRANSCRIBE_TIMEOUT,
    language: Optional[str] = None,
) -> TranscriptionResult:
    """
    Transcribe audio array with Whisper.
    
    Args:
        audio: Audio array (float32, 16kHz, mono)
        timeout: Maximum transcription time in seconds
        language: Optional language code (e.g., 'en')
    
    Returns:
        TranscriptionResult with text and metadata
    
    Raises:
        RuntimeError: If model not loaded or transcription fails
        ValueError: If audio is invalid
        TimeoutError: If transcription exceeds timeout
    """
    # Validate input
    audio = validate_audio(audio)
    
    # Preprocess if enabled
    if ASRConfig.ENABLE_PREPROCESSING:
        try:
            audio = clean_audio(audio)
        except Exception as e:
            log.warning(f"Audio preprocessing failed: {e}")
    
    # Transcribe with model
    with model_manager._model_context() as model:
        try:
            start = time.perf_counter()
            
            # Configure transcription
            vad_params = None
            if ASRConfig.ENABLE_VAD:
                vad_params = {
                    "min_silence_duration_ms": 300,
                    "threshold": 0.5,
                }
            
            # Run transcription
            segments_gen, info = model.transcribe(
                audio,
                beam_size=ASRConfig.BEAM_SIZE,
                best_of=ASRConfig.BEST_OF,
                patience=1.2,
                condition_on_previous_text=False,
                temperature=[0.0, 0.2, 0.4],
                vad_filter=ASRConfig.ENABLE_VAD,
                vad_parameters=vad_params,
                language=language,
            )
            
            # Process segments with timeout
            stats = SegmentStats()
            texts = []
            
            for seg in segments_gen:
                # Check timeout
                if (time.perf_counter() - start) > timeout:
                    log.warning(f"Transcription timeout ({timeout}s), returning partial result")
                    break
                
                # Filter segment
                if filter_segment(seg, stats):
                    text = (seg.text or "").strip()
                    if text:
                        text = clean_text(text)
                        if text:  # Check again after cleaning
                            texts.append(text)
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Combine texts
            final_text = " ".join(texts).strip()
            
            # Record metrics
            metrics.record_transcription(
                latency_ms=elapsed_ms,
                segments_total=stats.total,
                segments_kept=stats.kept,
            )
            
            log.info(
                f"Transcription complete: {elapsed_ms:.0f}ms | "
                f"segments={stats.kept}/{stats.total} | "
                f"text_len={len(final_text)}"
            )
            
            if stats.total > 0 and stats.kept == 0:
                log.warning("All segments filtered out - check quality thresholds")
            
            return TranscriptionResult(
                text=final_text,
                latency_ms=elapsed_ms,
                segments_total=stats.total,
                segments_kept=stats.kept,
                language=info.language if hasattr(info, 'language') else None,
            )
        
        except Exception as e:
            metrics.record_error()
            log.error(f"Transcription error: {e}", exc_info=True)
            raise RuntimeError(f"Transcription failed: {e}") from e


async def transcribe_async(
    audio: np.ndarray,
    timeout: float = ASRConfig.TRANSCRIBE_TIMEOUT,
    language: Optional[str] = None,
) -> TranscriptionResult:
    """
    Async wrapper for transcription.
    Runs transcription in thread pool to avoid blocking event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        transcribe,
        audio,
        timeout,
        language,
    )


# ============================================================================
# Batch Transcription
# ============================================================================

def transcribe_batch(
    audio_list: Sequence[np.ndarray],
    timeout: float = ASRConfig.BATCH_TIMEOUT,
    language: Optional[str] = None,
) -> List[TranscriptionResult]:
    """
    Transcribe multiple audio clips sequentially.
    
    Args:
        audio_list: Sequence of audio arrays
        timeout: Total timeout for entire batch
        language: Optional language code
    
    Returns:
        List of TranscriptionResult objects
    """
    if len(audio_list) > ASRConfig.MAX_BATCH_SIZE:
        raise ValueError(
            f"Batch size {len(audio_list)} exceeds maximum {ASRConfig.MAX_BATCH_SIZE}"
        )
    
    results: List[TranscriptionResult] = []
    start = time.perf_counter()
    
    for i, audio in enumerate(audio_list, 1):
        # Check batch timeout
        elapsed = time.perf_counter() - start
        if elapsed > timeout:
            log.warning(f"Batch timeout ({timeout}s) reached at item {i}/{len(audio_list)}")
            break
        
        # Transcribe with remaining time
        remaining = timeout - elapsed
        try:
            result = transcribe(
                audio,
                timeout=min(remaining, ASRConfig.TRANSCRIBE_TIMEOUT),
                language=language,
            )
            results.append(result)
            log.info(f"Batch item {i}/{len(audio_list)}: '{result.text[:50]}'")
        
        except Exception as e:
            log.error(f"Batch item {i} failed: {e}")
            # Add empty result to maintain list alignment
            results.append(TranscriptionResult(
                text="",
                latency_ms=0,
                segments_total=0,
                segments_kept=0,
            ))
    
    total_elapsed = time.perf_counter() - start
    log.info(f"Batch completed: {len(results)}/{len(audio_list)} items in {total_elapsed:.1f}s")
    
    return results


async def transcribe_batch_async(
    audio_list: Sequence[np.ndarray],
    timeout: float = ASRConfig.BATCH_TIMEOUT,
    language: Optional[str] = None,
    max_concurrency: int = 4,
) -> List[TranscriptionResult]:
    """
    Transcribe multiple audio clips concurrently.
    
    Args:
        audio_list: Sequence of audio arrays
        timeout: Timeout per individual transcription
        language: Optional language code
        max_concurrency: Maximum concurrent transcriptions
    
    Returns:
        List of TranscriptionResult objects (order preserved)
    """
    if len(audio_list) > ASRConfig.MAX_BATCH_SIZE:
        raise ValueError(
            f"Batch size {len(audio_list)} exceeds maximum {ASRConfig.MAX_BATCH_SIZE}"
        )
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def transcribe_with_semaphore(audio: np.ndarray, index: int):
        async with semaphore:
            try:
                result = await transcribe_async(audio, timeout, language)
                log.info(f"Async batch item {index+1}: '{result.text[:50]}'")
                return result
            except Exception as e:
                log.error(f"Async batch item {index+1} failed: {e}")
                return TranscriptionResult(
                    text="",
                    latency_ms=0,
                    segments_total=0,
                    segments_kept=0,
                )
    
    # Run all transcriptions concurrently with limit
    tasks = [
        transcribe_with_semaphore(audio, i)
        for i, audio in enumerate(audio_list)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results


# ============================================================================
# Public API (Backward Compatibility)
# ============================================================================

# Expose model manager methods
load_whisper = model_manager.load_model
unload_whisper = model_manager.unload_model

# Legacy function names
_transcribe = transcribe  # Internal name used in original


# ============================================================================
# Health & Diagnostics
# ============================================================================

def get_health() -> Dict[str, Any]:
    """Get ASR service health status."""
    return {
        "model": model_manager.health_check(),
        "metrics": metrics.get_stats(),
        "config": {
            "no_speech_max": ASRConfig.NO_SPEECH_MAX,
            "avg_logprob_min": ASRConfig.AVG_LOGPROB_MIN,
            "compression_ratio_max": ASRConfig.COMPRESSION_RATIO_MAX,
            "vad_enabled": ASRConfig.ENABLE_VAD,
            "preprocessing_enabled": ASRConfig.ENABLE_PREPROCESSING,
        }
    }


def get_metrics() -> Dict[str, Any]:
    """Get current metrics."""
    return metrics.get_stats()


def reset_metrics():
    """Reset metrics (useful for testing)."""
    metrics.reset()


# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    import json
    
    # Load model
    print("Loading Whisper model...")
    model_info = load_whisper()
    print(f"Model loaded: {json.dumps(model_info.to_dict(), indent=2)}")
    
    # Health check
    print("\n=== Health Check ===")
    print(json.dumps(get_health(), indent=2))
    
    # Test with synthetic audio
    print("\n=== Testing Transcription ===")
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1  # 1 second of noise
    
    try:
        result = transcribe(test_audio)
        print(f"Result: {json.dumps(result.to_dict(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Metrics
    print("\n=== Metrics ===")
    print(json.dumps(get_metrics(), indent=2))
    
    # Cleanup
    print("\n=== Unloading Model ===")
    unload_whisper()
    print("Done")

