"""
Audio Preprocessing Module
==========================
Robust speech preprocessing pipeline with VAD, denoising, and enhancement.

Features:
- Voice Activity Detection (WebRTC VAD + energy-based fallback)
- Spectral gating denoiser (noise estimation from non-speech frames)
- Bandpass filtering, DC removal, normalization
- Trimming, noise gating, and RMS normalization
- Thread-safe, configurable, with comprehensive error handling

Dependencies:
- Required: numpy, scipy
- Optional: webrtcvad (for improved VAD)

Usage:
    from Asr.preprocessing import Preprocessor, clean_audio
    
    # Simple usage
    cleaned = clean_audio(raw_audio, sample_rate=16000)
    
    # Advanced usage
    preprocessor = Preprocessor(
        sample_rate=16000,
        denoise_enabled=True,
        vad_aggressiveness=2
    )
    cleaned = preprocessor.process(raw_audio)
"""

import os
import time
import logging
from typing import Optional, Tuple, Dict, Any, Generator
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.signal import butter, lfilter, sosfilt, stft, istft, medfilt

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

# ============================================================================
# Optional Dependencies
# ============================================================================

try:
    import webrtcvad
    _HAS_WEBRTC = True
except ImportError:
    _HAS_WEBRTC = False
    log.info("webrtcvad not available - using energy-based VAD fallback")

# ============================================================================
# Configuration
# ============================================================================

class PreprocessingConfig:
    """Centralized preprocessing configuration."""
    
    # Audio format
    DEFAULT_SAMPLE_RATE: int = 16000
    TARGET_DTYPE: np.dtype = np.float32
    
    # Normalization
    TARGET_DBFS: float = -20.0
    PEAK_NORMALIZATION: bool = False
    
    # Filtering
    LOWCUT_HZ: int = 80
    HIGHCUT_HZ: int = 8000
    FILTER_ORDER: int = 5
    
    # VAD
    VAD_MODE: int = 2  # 0-3, higher = more aggressive
    VAD_FRAME_MS: int = 30  # Must be 10, 20, or 30 for WebRTC
    ENERGY_THRESHOLD: float = 0.01
    
    # Denoising
    DENOISE_ENABLED: bool = True
    DENOISE_METHOD: str = "spectral_gating"
    N_FFT: int = 1024
    HOP_LENGTH: int = 256
    NOISE_REDUCTION_FACTOR: float = 1.5
    SPECTRAL_FLOOR_DB: float = -40.0
    
    # Trimming & Gating
    TRIM_ENABLED: bool = True
    PADDING_MS: int = 50
    NOISE_GATE_THRESHOLD: float = 0.005
    
    # Performance
    MAX_AUDIO_LENGTH_SECONDS: float = 300.0  # 5 minutes
    MIN_AUDIO_LENGTH_SECONDS: float = 0.1
    
    @classmethod
    def validate(cls):
        """Validate configuration values."""
        if cls.VAD_FRAME_MS not in (10, 20, 30):
            raise ValueError("VAD_FRAME_MS must be 10, 20, or 30")
        
        if cls.VAD_MODE < 0 or cls.VAD_MODE > 3:
            raise ValueError("VAD_MODE must be 0-3")
        
        if cls.LOWCUT_HZ >= cls.HIGHCUT_HZ:
            raise ValueError("LOWCUT_HZ must be less than HIGHCUT_HZ")
        
        if cls.N_FFT < cls.HOP_LENGTH:
            raise ValueError("N_FFT must be >= HOP_LENGTH")


PreprocessingConfig.validate()

# ============================================================================
# Metrics
# ============================================================================

@dataclass
class PreprocessingMetrics:
    """Metrics for preprocessing operation."""
    duration_ms: float
    original_length: int
    processed_length: int
    original_rms: float
    processed_rms: float
    vad_speech_ratio: float
    denoise_applied: bool
    trim_applied: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_ms": round(self.duration_ms, 2),
            "original_length": self.original_length,
            "processed_length": self.processed_length,
            "length_reduction_pct": round(
                100 * (1 - self.processed_length / self.original_length),
                1
            ) if self.original_length > 0 else 0,
            "original_rms": round(self.original_rms, 4),
            "processed_rms": round(self.processed_rms, 4),
            "vad_speech_ratio": round(self.vad_speech_ratio, 3),
            "denoise_applied": self.denoise_applied,
            "trim_applied": self.trim_applied,
        }


# ============================================================================
# VAD Implementation
# ============================================================================

class VADMethod(Enum):
    """Available VAD methods."""
    WEBRTC = "webrtc"
    ENERGY = "energy"
    AUTO = "auto"


class VoiceActivityDetector:
    """
    Voice Activity Detection with multiple backends.
    
    Supports WebRTC VAD (if available) and energy-based fallback.
    """
    
    def __init__(
        self,
        sample_rate: int = PreprocessingConfig.DEFAULT_SAMPLE_RATE,
        mode: int = PreprocessingConfig.VAD_MODE,
        method: VADMethod = VADMethod.AUTO,
        energy_threshold: float = PreprocessingConfig.ENERGY_THRESHOLD,
    ):
        """
        Initialize VAD.
        
        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
            mode: Aggressiveness (0-3, higher = more aggressive)
            method: VAD method to use
            energy_threshold: Threshold for energy-based VAD
        """
        self.sample_rate = sample_rate
        self.mode = max(0, min(3, mode))
        self.energy_threshold = energy_threshold
        
        # Validate sample rate for WebRTC
        if sample_rate not in (8000, 16000, 32000, 48000):
            log.warning(
                f"Sample rate {sample_rate} not supported by WebRTC VAD, "
                f"forcing energy-based VAD"
            )
            method = VADMethod.ENERGY
        
        # Initialize backend
        if method == VADMethod.AUTO:
            self.method = VADMethod.WEBRTC if _HAS_WEBRTC else VADMethod.ENERGY
        else:
            self.method = method
        
        self.vad = None
        if self.method == VADMethod.WEBRTC:
            if _HAS_WEBRTC:
                try:
                    self.vad = webrtcvad.Vad(self.mode)
                    log.debug(f"WebRTC VAD initialized (mode={self.mode})")
                except Exception as e:
                    log.warning(f"WebRTC VAD init failed: {e}, using energy-based")
                    self.method = VADMethod.ENERGY
            else:
                self.method = VADMethod.ENERGY
        
        log.info(f"VAD initialized: method={self.method.value}, mode={self.mode}")
    
    def is_speech(self, frame: np.ndarray) -> bool:
        """
        Check if frame contains speech.
        
        Args:
            frame: Audio frame (float32, mono)
        
        Returns:
            True if speech detected
        """
        if self.method == VADMethod.WEBRTC and self.vad is not None:
            return self._is_speech_webrtc(frame)
        else:
            return self._is_speech_energy(frame)
    
    def _is_speech_webrtc(self, frame: np.ndarray) -> bool:
        """WebRTC VAD implementation."""
        try:
            # Convert to 16-bit PCM
            pcm16 = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
            return self.vad.is_speech(pcm16.tobytes(), self.sample_rate)
        except Exception as e:
            log.debug(f"WebRTC VAD error: {e}, falling back to energy")
            return self._is_speech_energy(frame)
    
    def _is_speech_energy(self, frame: np.ndarray) -> bool:
        """Energy-based VAD implementation."""
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        return float(rms) > self.energy_threshold
    
    def get_speech_mask(
        self,
        audio: np.ndarray,
        frame_ms: int = PreprocessingConfig.VAD_FRAME_MS,
    ) -> np.ndarray:
        """
        Generate per-sample speech mask.
        
        Args:
            audio: Audio array (float32, mono)
            frame_ms: Frame duration in milliseconds
        
        Returns:
            Boolean mask (True = speech)
        """
        if audio.size == 0:
            return np.zeros(0, dtype=bool)
        
        frame_len = int(self.sample_rate * frame_ms / 1000)
        num_frames = int(np.ceil(len(audio) / frame_len))
        
        mask = np.zeros_like(audio, dtype=bool)
        
        for i in range(num_frames):
            start = i * frame_len
            end = min(start + frame_len, len(audio))
            frame = audio[start:end]
            
            # Pad frame if needed
            if len(frame) < frame_len:
                frame = np.pad(frame, (0, frame_len - len(frame)), mode='constant')
            
            is_speech = self.is_speech(frame)
            mask[start:end] = is_speech
        
        return mask


# ============================================================================
# Audio Utilities
# ============================================================================

def validate_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Validate and prepare audio input.
    
    Args:
        audio: Input audio
        sample_rate: Expected sample rate
    
    Returns:
        Validated audio array
    
    Raises:
        ValueError: If audio is invalid
    """
    if audio is None or audio.size == 0:
        raise ValueError("Audio array is empty")
    
    # Check length limits
    max_samples = int(PreprocessingConfig.MAX_AUDIO_LENGTH_SECONDS * sample_rate)
    if len(audio) > max_samples:
        raise ValueError(
            f"Audio too long: {len(audio)/sample_rate:.1f}s "
            f"(max: {PreprocessingConfig.MAX_AUDIO_LENGTH_SECONDS}s)"
        )
    
    min_samples = int(PreprocessingConfig.MIN_AUDIO_LENGTH_SECONDS * sample_rate)
    if len(audio) < min_samples:
        raise ValueError(
            f"Audio too short: {len(audio)/sample_rate:.3f}s "
            f"(min: {PreprocessingConfig.MIN_AUDIO_LENGTH_SECONDS}s)"
        )
    
    return audio


def to_float32(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to float32 in range [-1, 1].
    
    Args:
        audio: Input audio (any numeric dtype)
    
    Returns:
        Float32 audio
    """
    if audio.dtype == np.float32:
        return audio
    
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    
    if audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    
    if audio.dtype == np.float64:
        return audio.astype(np.float32)
    
    # Generic conversion
    log.warning(f"Converting from unusual dtype: {audio.dtype}")
    return audio.astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono.
    
    Args:
        audio: Input audio (1D or 2D)
    
    Returns:
        Mono audio (1D)
    """
    audio = to_float32(audio)
    
    if audio.ndim == 1:
        return audio
    
    if audio.ndim == 2:
        # (samples, channels) or (channels, samples)
        if audio.shape[0] < audio.shape[1]:
            # Likely (channels, samples)
            audio = audio.T
        
        # Average channels
        if audio.shape[1] > 1:
            return audio.mean(axis=1, dtype=np.float32)
        else:
            return audio[:, 0]
    
    raise ValueError(f"Unsupported audio shape: {audio.shape}")


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS (Root Mean Square) of audio."""
    return float(np.sqrt(np.mean(audio ** 2) + 1e-12))


def db_to_linear(db: float) -> float:
    """Convert dB to linear scale."""
    return 10 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear scale to dB."""
    return 20 * np.log10(max(linear, 1e-12))


# ============================================================================
# Signal Processing
# ============================================================================

class SignalProcessor:
    """Collection of signal processing operations."""
    
    @staticmethod
    def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
        """Remove DC offset (mean) from audio."""
        return audio - np.mean(audio)
    
    @staticmethod
    def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """
        Normalize audio to target peak amplitude.
        
        Args:
            audio: Input audio
            target_peak: Target peak amplitude (0-1)
        
        Returns:
            Normalized audio
        """
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (target_peak / peak)
        return audio
    
    @staticmethod
    def normalize_rms(
        audio: np.ndarray,
        target_dbfs: float = PreprocessingConfig.TARGET_DBFS,
    ) -> np.ndarray:
        """
        Normalize audio to target RMS level in dBFS.
        
        Args:
            audio: Input audio
            target_dbfs: Target RMS in dBFS
        
        Returns:
            Normalized audio
        """
        rms = compute_rms(audio)
        if rms <= 0:
            return audio
        
        target_linear = db_to_linear(target_dbfs)
        scale = target_linear / rms
        
        audio = audio * scale
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    @staticmethod
    def bandpass_filter(
        audio: np.ndarray,
        sample_rate: int,
        lowcut: int = PreprocessingConfig.LOWCUT_HZ,
        highcut: int = PreprocessingConfig.HIGHCUT_HZ,
        order: int = PreprocessingConfig.FILTER_ORDER,
    ) -> np.ndarray:
        """
        Apply bandpass filter.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            lowcut: Low cutoff frequency (Hz)
            highcut: High cutoff frequency (Hz)
            order: Filter order
        
        Returns:
            Filtered audio
        """
        nyquist = 0.5 * sample_rate
        low = max(lowcut / nyquist, 1e-5)
        high = min(highcut / nyquist, 0.99999)
        
        if low >= high:
            log.warning(
                f"Invalid bandpass range: [{lowcut}, {highcut}] Hz, "
                f"skipping filter"
            )
            return audio
        
        try:
            # Use second-order sections for better numerical stability
            sos = butter(order, [low, high], btype='band', output='sos')
            filtered = sosfilt(sos, audio).astype(np.float32)
            return filtered
        except Exception as e:
            log.error(f"Bandpass filter failed: {e}")
            return audio
    
    @staticmethod
    def apply_noise_gate(
        audio: np.ndarray,
        threshold: float = PreprocessingConfig.NOISE_GATE_THRESHOLD,
        smooth_ms: int = 10,
        sample_rate: int = PreprocessingConfig.DEFAULT_SAMPLE_RATE,
    ) -> np.ndarray:
        """
        Apply noise gate with smoothing.
        
        Args:
            audio: Input audio
            threshold: Gate threshold
            smooth_ms: Smoothing window in milliseconds
            sample_rate: Sample rate
        
        Returns:
            Gated audio
        """
        # Create gate mask
        mask = np.abs(audio) > threshold
        
        # Smooth mask to avoid clicks
        if smooth_ms > 0:
            smooth_samples = int(sample_rate * smooth_ms / 1000)
            if smooth_samples > 1:
                # Moving average
                kernel = np.ones(smooth_samples) / smooth_samples
                mask = np.convolve(mask.astype(float), kernel, mode='same') > 0.5
        
        return audio * mask
    
    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        mask: Optional[np.ndarray] = None,
        padding_ms: int = PreprocessingConfig.PADDING_MS,
        sample_rate: int = PreprocessingConfig.DEFAULT_SAMPLE_RATE,
        threshold: float = 0.01,
    ) -> np.ndarray:
        """
        Trim leading/trailing silence.
        
        Args:
            audio: Input audio
            mask: Optional VAD mask (True = speech)
            padding_ms: Padding to keep around speech (ms)
            sample_rate: Sample rate
            threshold: Amplitude threshold if no mask provided
        
        Returns:
            Trimmed audio
        """
        if mask is None:
            # Fallback: amplitude-based trimming
            indices = np.where(np.abs(audio) > threshold)[0]
            if indices.size == 0:
                return audio
            return audio[indices[0]:indices[-1] + 1]
        
        # Use VAD mask
        speech_indices = np.where(mask)[0]
        if speech_indices.size == 0:
            log.warning("No speech detected in trimming")
            return audio
        
        # Add padding
        padding_samples = int(sample_rate * padding_ms / 1000)
        start = max(0, speech_indices[0] - padding_samples)
        end = min(len(audio), speech_indices[-1] + padding_samples + 1)
        
        return audio[start:end]


# ============================================================================
# Denoising
# ============================================================================

class SpectralGatingDenoiser:
    """
    Spectral gating denoiser using STFT.
    
    Estimates noise from non-speech frames and applies soft gating.
    """
    
    def __init__(
        self,
        sample_rate: int = PreprocessingConfig.DEFAULT_SAMPLE_RATE,
        n_fft: int = PreprocessingConfig.N_FFT,
        hop_length: int = PreprocessingConfig.HOP_LENGTH,
        reduction_factor: float = PreprocessingConfig.NOISE_REDUCTION_FACTOR,
        spectral_floor_db: float = PreprocessingConfig.SPECTRAL_FLOOR_DB,
    ):
        """
        Initialize denoiser.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length
            reduction_factor: Noise reduction strength
            spectral_floor_db: Minimum gain in dB
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.reduction_factor = reduction_factor
        self.spectral_floor_db = spectral_floor_db
        self.spectral_floor_linear = db_to_linear(spectral_floor_db)
    
    def estimate_noise_spectrum(
        self,
        magnitude: np.ndarray,
        speech_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate noise magnitude spectrum.
        
        Args:
            magnitude: STFT magnitude (freq_bins, frames)
            speech_mask: Per-frame speech mask (True = speech)
        
        Returns:
            Noise spectrum (freq_bins, 1)
        """
        if speech_mask is None or np.all(speech_mask):
            # No mask or all speech: use minimum across time
            return np.min(magnitude, axis=1, keepdims=True)
        
        # Use non-speech frames
        nonspeech_mask = ~speech_mask
        if np.sum(nonspeech_mask) < 1:
            return np.min(magnitude, axis=1, keepdims=True)
        
        # Average non-speech frames
        noise = np.mean(magnitude[:, nonspeech_mask], axis=1, keepdims=True)
        
        # Ensure reasonable noise estimate
        noise = np.maximum(noise, np.percentile(magnitude, 5, axis=1, keepdims=True))
        
        return noise
    
    def compute_gain(
        self,
        magnitude: np.ndarray,
        noise_spectrum: np.ndarray,
    ) -> np.ndarray:
        """
        Compute spectral gain for noise reduction.
        
        Args:
            magnitude: Signal magnitude
            noise_spectrum: Estimated noise magnitude
        
        Returns:
            Gain mask (0-1)
        """
        # Soft thresholding
        threshold = noise_spectrum * self.reduction_factor
        gain = np.maximum(0.0, (magnitude - threshold) / (magnitude + 1e-12))
        
        # Apply floor
        gain = np.maximum(gain, self.spectral_floor_linear)
        
        # Smooth gain across frequency to reduce musical noise
        # Use median filter
        gain_smoothed = np.zeros_like(gain)
        for i in range(gain.shape[1]):
            gain_smoothed[:, i] = medfilt(gain[:, i], kernel_size=5)
        
        return gain_smoothed
    
    def denoise(
        self,
        audio: np.ndarray,
        vad_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply spectral gating denoising.
        
        Args:
            audio: Input audio
            vad_mask: Optional per-sample VAD mask
        
        Returns:
            Denoised audio
        """
        if len(audio) < self.n_fft:
            log.warning("Audio too short for denoising")
            return audio
        
        try:
            # STFT
            f, t, stft_matrix = stft(
                audio,
                fs=self.sample_rate,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
            )
            
            magnitude = np.abs(stft_matrix)
            phase = np.angle(stft_matrix)
            
            # Convert per-sample VAD mask to per-frame mask
            speech_mask_frames = None
            if vad_mask is not None:
                num_frames = t.shape[0]
                speech_mask_frames = np.zeros(num_frames, dtype=bool)
                
                for i in range(num_frames):
                    start = i * self.hop_length
                    end = min(start + self.hop_length, len(vad_mask))
                    if end > start:
                        speech_mask_frames[i] = np.mean(vad_mask[start:end]) > 0.5
            
            # Estimate noise
            noise_spectrum = self.estimate_noise_spectrum(magnitude, speech_mask_frames)
            
            # Compute gain
            gain = self.compute_gain(magnitude, noise_spectrum)
            
            # Apply gain
            magnitude_denoised = magnitude * gain
            
            # Reconstruct
            stft_denoised = magnitude_denoised * np.exp(1j * phase)
            
            # ISTFT
            _, audio_denoised = istft(
                stft_denoised,
                fs=self.sample_rate,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
            )
            
            # Trim to original length
            audio_denoised = audio_denoised[:len(audio)]
            
            # Preserve original RMS
            orig_rms = compute_rms(audio)
            denoised_rms = compute_rms(audio_denoised) + 1e-12
            if denoised_rms > 0:
                audio_denoised = audio_denoised * (orig_rms / denoised_rms)
            
            return audio_denoised.astype(np.float32)
        
        except Exception as e:
            log.error(f"Denoising failed: {e}", exc_info=True)
            return audio


# ============================================================================
# Main Preprocessor
# ============================================================================

class Preprocessor:
    """
    Complete audio preprocessing pipeline.
    
    Features:
    - Format conversion (mono, float32)
    - DC offset removal
    - Bandpass filtering
    - Voice activity detection
    - Spectral gating denoising
    - Trimming and noise gating
    - RMS normalization
    """
    
    def __init__(
        self,
        sample_rate: int = PreprocessingConfig.DEFAULT_SAMPLE_RATE,
        lowcut: int = PreprocessingConfig.LOWCUT_HZ,
        highcut: int = PreprocessingConfig.HIGHCUT_HZ,
        vad_mode: int = PreprocessingConfig.VAD_MODE,
        vad_method: VADMethod = VADMethod.AUTO,
        denoise_enabled: bool = PreprocessingConfig.DENOISE_ENABLED,
        denoise_method: str = PreprocessingConfig.DENOISE_METHOD,
        n_fft: int = PreprocessingConfig.N_FFT,
        hop_length: int = PreprocessingConfig.HOP_LENGTH,
        noise_reduction_factor: float = PreprocessingConfig.NOISE_REDUCTION_FACTOR,
        target_dbfs: float = PreprocessingConfig.TARGET_DBFS,
        trim_enabled: bool = PreprocessingConfig.TRIM_ENABLED,
        padding_ms: int = PreprocessingConfig.PADDING_MS,
        noise_gate_threshold: float = PreprocessingConfig.NOISE_GATE_THRESHOLD,
    ):
        """
        Initialize preprocessor.
        
        Args:
            sample_rate: Audio sample rate
            lowcut: Bandpass low cutoff (Hz)
            highcut: Bandpass high cutoff (Hz)
            vad_mode: VAD aggressiveness (0-3)
            vad_method: VAD method (auto, webrtc, energy)
            denoise_enabled: Enable denoising
            denoise_method: Denoising method (spectral_gating)
            n_fft: FFT size for denoising
            hop_length: Hop length for denoising
            noise_reduction_factor: Denoising strength
            target_dbfs: Target RMS level
            trim_enabled: Enable trimming
            padding_ms: Padding around speech
            noise_gate_threshold: Noise gate threshold
        """
        self.sample_rate = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.target_dbfs = target_dbfs
        self.trim_enabled = trim_enabled
        self.padding_ms = padding_ms
        self.noise_gate_threshold = noise_gate_threshold
        self.denoise_enabled = denoise_enabled
        self.denoise_method = denoise_method
        
        # Initialize components
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            mode=vad_mode,
            method=vad_method,
        )
        
        self.signal_processor = SignalProcessor()
        
        self.denoiser = None
        if denoise_enabled and denoise_method == "spectral_gating":
            self.denoiser = SpectralGatingDenoiser(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                reduction_factor=noise_reduction_factor,
            )
        
        log.info(
            f"Preprocessor initialized: "
            f"sr={sample_rate}, vad={self.vad.method.value}, "
            f"denoise={denoise_enabled}"
        )
    
    def process(
        self,
        audio: np.ndarray,
        return_metrics: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, PreprocessingMetrics]:
        """
        Run full preprocessing pipeline.
        
        Args:
            audio: Input audio
            return_metrics: Whether to return metrics
        
        Returns:
            Preprocessed audio (and metrics if requested)
        
        Raises:
            ValueError: If audio is invalid
        """
        start_time = time.perf_counter()
        
        # Validate input
        audio = validate_audio(audio, self.sample_rate)
        original_length = len(audio)
        original_rms = compute_rms(audio)
        
        # Convert to mono float32
        audio = to_mono(audio)
        audio = to_float32(audio)
        
        # Remove DC offset
        audio = self.signal_processor.remove_dc_offset(audio)
        
        # Bandpass filter
        audio = self.signal_processor.bandpass_filter(
            audio,
            self.sample_rate,
            self.lowcut,
            self.highcut,
        )
        
        # VAD
        vad_mask = self.vad.get_speech_mask(audio)
        speech_ratio = np.mean(vad_mask) if vad_mask.size > 0 else 0.0
        
        # Trim silence (first pass)
        if self.trim_enabled:
            audio = self.signal_processor.trim_silence(
                audio,
                mask=vad_mask,
                padding_ms=self.padding_ms,
                sample_rate=self.sample_rate,
            )
            
            # Update VAD mask after trimming
            if len(audio) != len(vad_mask):
                vad_mask = self.vad.get_speech_mask(audio)
        
        # Denoise
        denoise_applied = False
        if self.denoise_enabled and self.denoiser is not None:
            try:
                audio = self.denoiser.denoise(audio, vad_mask)
                denoise_applied = True
            except Exception as e
          log.warning(f"Denoising failed: {e}, continuing without denoising")
                denoise_applied = False
        
        # Noise gate
        audio = self.signal_processor.apply_noise_gate(
            audio,
            threshold=self.noise_gate_threshold,
            sample_rate=self.sample_rate,
        )
        
        # Normalize RMS
        audio = self.signal_processor.normalize_rms(audio, self.target_dbfs)
        
        # Final trim (after denoising may have changed energy distribution)
        trim_applied = False
        if self.trim_enabled:
            try:
                vad_mask_final = self.vad.get_speech_mask(audio)
                audio = self.signal_processor.trim_silence(
                    audio,
                    mask=vad_mask_final,
                    padding_ms=self.padding_ms,
                    sample_rate=self.sample_rate,
                )
                trim_applied = True
            except Exception as e:
                log.warning(f"Final trimming failed: {e}")
        
        # Ensure float32
        audio = to_float32(audio)
        
        # Final validation
        if audio.size == 0:
            log.warning("Preprocessing resulted in empty audio")
            raise ValueError("Preprocessing produced empty audio")
        
        # Compute metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        processed_length = len(audio)
        processed_rms = compute_rms(audio)
        
        metrics = PreprocessingMetrics(
            duration_ms=duration_ms,
            original_length=original_length,
            processed_length=processed_length,
            original_rms=original_rms,
            processed_rms=processed_rms,
            vad_speech_ratio=speech_ratio,
            denoise_applied=denoise_applied,
            trim_applied=trim_applied,
        )
        
        log.debug(
            f"Preprocessing complete: {duration_ms:.1f}ms, "
            f"length: {original_length} -> {processed_length}, "
            f"speech_ratio: {speech_ratio:.2f}"
        )
        
        if return_metrics:
            return audio, metrics
        return audio
    
    def process_batch(
        self,
        audio_list: list[np.ndarray],
        return_metrics: bool = False,
    ) -> list[np.ndarray] | list[Tuple[np.ndarray, PreprocessingMetrics]]:
        """
        Process multiple audio clips.
        
        Args:
            audio_list: List of audio arrays
            return_metrics: Whether to return metrics
        
        Returns:
            List of preprocessed audio (and metrics if requested)
        """
        results = []
        for i, audio in enumerate(audio_list):
            try:
                result = self.process(audio, return_metrics=return_metrics)
                results.append(result)
                log.debug(f"Batch item {i+1}/{len(audio_list)} processed")
            except Exception as e:
                log.error(f"Batch item {i+1} failed: {e}")
                if return_metrics:
                    results.append((np.array([]), None))
                else:
                    results.append(np.array([]))
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "sample_rate": self.sample_rate,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "vad_method": self.vad.method.value,
            "vad_mode": self.vad.mode,
            "denoise_enabled": self.denoise_enabled,
            "denoise_method": self.denoise_method,
            "target_dbfs": self.target_dbfs,
            "trim_enabled": self.trim_enabled,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

_default_preprocessor: Optional[Preprocessor] = None


def clean_audio(
    audio: np.ndarray,
    sample_rate: int = PreprocessingConfig.DEFAULT_SAMPLE_RATE,
    **kwargs
) -> np.ndarray:
    """
    Convenience function for simple preprocessing.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        **kwargs: Additional arguments for Preprocessor
    
    Returns:
        Preprocessed audio
    
    Examples:
        >>> import numpy as np
        >>> audio = np.random.randn(16000).astype(np.float32)
        >>> cleaned = clean_audio(audio, sample_rate=16000)
    """
    global _default_preprocessor
    
    # Reuse preprocessor if sample rate matches
    if _default_preprocessor is None or _default_preprocessor.sample_rate != sample_rate:
        _default_preprocessor = Preprocessor(sample_rate=sample_rate, **kwargs)
    
    return _default_preprocessor.process(audio)


def get_speech_segments(
    audio: np.ndarray,
    sample_rate: int = PreprocessingConfig.DEFAULT_SAMPLE_RATE,
    min_duration_ms: int = 300,
) -> list[Tuple[int, int]]:
    """
    Detect speech segments in audio.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        min_duration_ms: Minimum segment duration
    
    Returns:
        List of (start_sample, end_sample) tuples
    
    Examples:
        >>> audio = load_audio("speech.wav")
        >>> segments = get_speech_segments(audio, sample_rate=16000)
        >>> for start, end in segments:
        ...     segment = audio[start:end]
    """
    vad = VoiceActivityDetector(sample_rate=sample_rate)
    mask = vad.get_speech_mask(audio)
    
    # Find contiguous speech regions
    segments = []
    in_speech = False
    start = 0
    
    min_samples = int(sample_rate * min_duration_ms / 1000)
    
    for i, is_speech in enumerate(mask):
        if is_speech and not in_speech:
            # Start of speech
            start = i
            in_speech = True
        elif not is_speech and in_speech:
            # End of speech
            if i - start >= min_samples:
                segments.append((start, i))
            in_speech = False
    
    # Handle final segment
    if in_speech and len(mask) - start >= min_samples:
        segments.append((start, len(mask)))
    
    return segments


# ============================================================================
# Analysis & Diagnostics
# ============================================================================

@dataclass
class AudioAnalysis:
    """Comprehensive audio analysis results."""
    duration_seconds: float
    sample_rate: int
    num_samples: int
    rms: float
    peak: float
    dynamic_range_db: float
    dc_offset: float
    zero_crossings: int
    speech_ratio: float
    clipping_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds, 3),
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "rms": round(self.rms, 6),
            "rms_dbfs": round(linear_to_db(self.rms), 2),
            "peak": round(self.peak, 6),
            "peak_dbfs": round(linear_to_db(self.peak), 2),
            "dynamic_range_db": round(self.dynamic_range_db, 2),
            "dc_offset": round(self.dc_offset, 6),
            "zero_crossings": self.zero_crossings,
            "speech_ratio": round(self.speech_ratio, 3),
            "clipping_ratio": round(self.clipping_ratio, 4),
        }


def analyze_audio(
    audio: np.ndarray,
    sample_rate: int = PreprocessingConfig.DEFAULT_SAMPLE_RATE,
    run_vad: bool = True,
) -> AudioAnalysis:
    """
    Comprehensive audio analysis.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        run_vad: Whether to run VAD analysis
    
    Returns:
        AudioAnalysis with detailed metrics
    """
    audio = to_float32(to_mono(audio))
    
    # Basic metrics
    duration = len(audio) / sample_rate
    rms = compute_rms(audio)
    peak = float(np.max(np.abs(audio)))
    
    # Dynamic range
    min_val = float(np.min(np.abs(audio[audio != 0]))) if np.any(audio != 0) else 1e-12
    dynamic_range_db = linear_to_db(peak) - linear_to_db(min_val)
    
    # DC offset
    dc_offset = float(np.mean(audio))
    
    # Zero crossings
    zero_crossings = int(np.sum(np.diff(np.sign(audio)) != 0))
    
    # Speech ratio
    speech_ratio = 0.0
    if run_vad:
        vad = VoiceActivityDetector(sample_rate=sample_rate)
        mask = vad.get_speech_mask(audio)
        speech_ratio = float(np.mean(mask))
    
    # Clipping detection
    clipping_threshold = 0.99
    clipping_ratio = float(np.mean(np.abs(audio) >= clipping_threshold))
    
    return AudioAnalysis(
        duration_seconds=duration,
        sample_rate=sample_rate,
        num_samples=len(audio),
        rms=rms,
        peak=peak,
        dynamic_range_db=dynamic_range_db,
        dc_offset=dc_offset,
        zero_crossings=zero_crossings,
        speech_ratio=speech_ratio,
        clipping_ratio=clipping_ratio,
    )


def compare_preprocessing(
    audio: np.ndarray,
    sample_rate: int = PreprocessingConfig.DEFAULT_SAMPLE_RATE,
) -> Dict[str, AudioAnalysis]:
    """
    Compare audio before and after preprocessing.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
    
    Returns:
        Dict with 'before' and 'after' AudioAnalysis
    """
    # Analyze original
    before = analyze_audio(audio, sample_rate, run_vad=True)
    
    # Process
    preprocessor = Preprocessor(sample_rate=sample_rate)
    processed = preprocessor.process(audio)
    
    # Analyze processed
    after = analyze_audio(processed, sample_rate, run_vad=True)
    
    return {
        "before": before,
        "after": after,
    }


# ============================================================================
# Health Check
# ============================================================================

def get_health() -> Dict[str, Any]:
    """Get preprocessing module health status."""
    return {
        "webrtc_available": _HAS_WEBRTC,
        "default_sample_rate": PreprocessingConfig.DEFAULT_SAMPLE_RATE,
        "denoise_enabled": PreprocessingConfig.DENOISE_ENABLED,
        "vad_methods": ["webrtc", "energy"] if _HAS_WEBRTC else ["energy"],
    }


# ============================================================================
# Testing & Examples
# ============================================================================

if __name__ == "__main__":
    import json
    
    print("=" * 80)
    print("AUDIO PREPROCESSING - COMPREHENSIVE DEMO")
    print("=" * 80)
    
    # Generate test audio
    print("\n1. Generating test audio...")
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Speech-like signal (modulated sine waves)
    speech = np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    
    # Add noise
    noise = np.random.randn(len(t)) * 0.1
    audio = speech + noise
    
    # Add silence at beginning and end
    silence_samples = int(0.5 * sample_rate)
    audio = np.concatenate([
        np.zeros(silence_samples),
        audio,
        np.zeros(silence_samples)
    ])
    
    print(f"Test audio: {len(audio)/sample_rate:.1f}s @ {sample_rate}Hz")
    
    # ========================================================================
    # 2. Audio Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. AUDIO ANALYSIS (BEFORE PREPROCESSING)")
    print("=" * 80)
    
    analysis_before = analyze_audio(audio, sample_rate)
    print(json.dumps(analysis_before.to_dict(), indent=2))
    
    # ========================================================================
    # 3. Simple Preprocessing
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. SIMPLE PREPROCESSING")
    print("=" * 80)
    
    cleaned = clean_audio(audio, sample_rate=sample_rate)
    print(f"Original length: {len(audio)} samples")
    print(f"Cleaned length: {len(cleaned)} samples")
    print(f"Reduction: {100*(1-len(cleaned)/len(audio)):.1f}%")
    
    # ========================================================================
    # 4. Advanced Preprocessing with Metrics
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. ADVANCED PREPROCESSING WITH METRICS")
    print("=" * 80)
    
    preprocessor = Preprocessor(
        sample_rate=sample_rate,
        denoise_enabled=True,
        trim_enabled=True,
    )
    
    processed, metrics = preprocessor.process(audio, return_metrics=True)
    print("\nMetrics:")
    print(json.dumps(metrics.to_dict(), indent=2))
    
    # ========================================================================
    # 5. Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. BEFORE/AFTER COMPARISON")
    print("=" * 80)
    
    comparison = compare_preprocessing(audio, sample_rate)
    
    print("\nBEFORE:")
    print(json.dumps(comparison["before"].to_dict(), indent=2))
    
    print("\nAFTER:")
    print(json.dumps(comparison["after"].to_dict(), indent=2))
    
    # ========================================================================
    # 6. Speech Segmentation
    # ========================================================================
    print("\n" + "=" * 80)
    print("6. SPEECH SEGMENTATION")
    print("=" * 80)
    
    segments = get_speech_segments(audio, sample_rate, min_duration_ms=300)
    print(f"Found {len(segments)} speech segments:")
    for i, (start, end) in enumerate(segments, 1):
        duration_ms = (end - start) / sample_rate * 1000
        print(f"  Segment {i}: {start}-{end} ({duration_ms:.0f}ms)")
    
    # ========================================================================
    # 7. Batch Processing
    # ========================================================================
    print("\n" + "=" * 80)
    print("7. BATCH PROCESSING")
    print("=" * 80)
    
    # Create batch of audio clips
    batch = [
        audio,
        audio[:len(audio)//2],  # Shorter clip
        audio + np.random.randn(len(audio)) * 0.05,  # Noisier clip
    ]
    
    results = preprocessor.process_batch(batch, return_metrics=True)
    
    for i, (processed_audio, metrics) in enumerate(results, 1):
        print(f"\nBatch item {i}:")
        print(f"  Duration: {metrics.duration_ms:.1f}ms")
        print(f"  Length: {metrics.original_length} -> {metrics.processed_length}")
        print(f"  Speech ratio: {metrics.vad_speech_ratio:.2f}")
    
    # ========================================================================
    # 8. Configuration
    # ========================================================================
    print("\n" + "=" * 80)
    print("8. PREPROCESSOR CONFIGURATION")
    print("=" * 80)
    
    config = preprocessor.get_config()
    print(json.dumps(config, indent=2))
    
    # ========================================================================
    # 9. Health Check
    # ========================================================================
    print("\n" + "=" * 80)
    print("9. HEALTH CHECK")
    print("=" * 80)
    
    health = get_health()
    print(json.dumps(health, indent=2))
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
