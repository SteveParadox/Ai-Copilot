# preprocessing.py
import numpy as np
import logging

log = logging.getLogger("Preprocessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===================== AUDIO CONVERSION =====================
def to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert audio to float32 in [-1, 1]."""
    if audio.dtype == np.float32:
        return audio
    elif audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    else:
        log.warning(f"Unexpected audio dtype {audio.dtype}, casting to float32")
        return audio.astype(np.float32)

# ===================== AUDIO NORMALIZATION =====================
def normalize_audio(audio: np.ndarray, target_dBFS: float = -20.0, eps: float = 1e-9) -> np.ndarray:
    """Normalize audio to a target dBFS."""
    audio = to_float32(audio)
    rms_val = rms(audio)
    if rms_val > 0:
        scalar = 10 ** (target_dBFS / 20) / max(rms_val, eps)
        audio = np.clip(audio * scalar, -1.0, 1.0)
    return audio

# ===================== DC OFFSET REMOVAL =====================
def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset by centering audio around zero."""
    return audio - np.mean(audio)

# ===================== MONO & FLOAT32 =====================
def prep_audio_float32(audio: np.ndarray) -> np.ndarray:
    """Ensure audio is float32 and mono."""
    audio = to_float32(audio)
    if audio.ndim == 2:
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, dtype=np.float32)
        else:
            audio = audio[:, 0]
    return audio

# ===================== SILENCE TRIMMING =====================
def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Remove leading and trailing silence below threshold.
    threshold: amplitude below which audio is considered silent
    """
    audio = to_float32(audio)
    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio  # silence only
    return audio[np.where(mask)[0][0]: np.where(mask)[0][-1] + 1]

# ===================== NOISE GATE =====================
def apply_noise_gate(audio: np.ndarray, gate_threshold: float = 0.02) -> np.ndarray:
    """
    Suppress very low amplitude noise.
    """
    audio = to_float32(audio)
    audio[np.abs(audio) < gate_threshold] = 0.0
    return audio

# ===================== FULL AUDIO CLEANING =====================
def clean_audio(audio: np.ndarray, target_dBFS: float = -20.0,
                silence_thresh: float = 0.01, gate_thresh: float = 0.02) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Convert to float32
    2. Remove DC offset
    3. Normalize to target dBFS
    4. Apply noise gate
    5. Trim silence
    6. Ensure mono float32
    """
    audio = to_float32(audio)
    audio = remove_dc_offset(audio)
    audio = normalize_audio(audio, target_dBFS=target_dBFS)
    audio = apply_noise_gate(audio, gate_threshold=gate_thresh)
    audio = trim_silence(audio, threshold=silence_thresh)
    audio = prep_audio_float32(audio)
    return audio

# ===================== UTILITY =====================
def rms(audio: np.ndarray) -> float:
    """Compute root mean square of audio."""
    return np.sqrt(np.mean(audio ** 2))
