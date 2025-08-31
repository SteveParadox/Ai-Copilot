# model.py
import os
import threading
import time
import logging
import re
import numpy as np
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from Asr.utils import majority_vote
from Asr._preprocessing import clean_audio

# ===================== ENV =====================
load_dotenv()
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "tiny")

# ===================== LOGGING =====================
log = logging.getLogger("ASR")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===================== GLOBALS =====================
model_lock = threading.Lock()
model: WhisperModel | None = None

# ===================== STABILITY FILTERS =====================
NO_SPEECH_MAX = 0.50
AVG_LOGPROB_MIN = -1.10
COMPRESSION_RATIO_MAX = 2.4
MIN_CHARS = 6
BANNED_SHORT = {"bye", "thanks", "thank you", "okay", "ok"}

# ===================== DEVICE/COMPUTE SELECTION =====================
def select_device_compute(device: str = "auto", compute: str = "auto") -> tuple[str, str]:
    """
    Choose optimal device & compute_type for Whisper.
    """
    try:
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        log.info(f"Selected device={device}, compute={compute}")
        return device, compute
    except Exception as e:
        log.warning(f"Failed to auto-select device/compute, falling back to CPU/int8: {e}")
        return "cpu", "int8"

# ===================== MODEL LOAD/RELOAD =====================
def load_whisper(model_size: str = DEFAULT_MODEL, device: str = "auto", compute: str = "auto") -> WhisperModel:
    """
    Load or reload Whisper model thread-safely.
    """
    global model
    try:
        device, compute = select_device_compute(device, compute)
        loaded_model = WhisperModel(model_size, device=device, compute_type=compute)
        with model_lock:
            model = loaded_model
        log.info(f"Whisper model loaded: {model_size} ({device}/{compute})")
        return loaded_model
    except Exception as e:
        log.error(f"Failed to load Whisper model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load Whisper model: {e}")

# ===================== TRANSCRIPTION HELPERS =====================
def _keep_segment(seg) -> bool:
    """Check if segment is reliable based on confidence thresholds."""
    txt = (seg.text or "").strip().lower()
    if getattr(seg, "no_speech_prob", 0) > NO_SPEECH_MAX:
        return False
    if getattr(seg, "avg_logprob", 0) < AVG_LOGPROB_MIN:
        return False
    if getattr(seg, "compression_ratio", 1.0) > COMPRESSION_RATIO_MAX:
        return False
    if len(txt) < MIN_CHARS:
        return False
    if txt in BANNED_SHORT:
        return False
    return True

def clean_text(text: str) -> str:
    """Basic cleanup for ASR output: repeated words, fillers, whitespace."""
    if not text:
        return ""
    fillers = ["um", "uh", "like", "you know", "i mean"]
    for f in fillers:
        text = re.sub(rf'\b{re.escape(f)}\b', '', text, flags=re.IGNORECASE)
    # collapse repeated words
    text = re.sub(r'\b(\w+)( \1){1,}\b', r'\1', text, flags=re.IGNORECASE)
    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===================== TRANSCRIPTION =====================

def transcribe(audio: np.ndarray) -> str:
    """
    Transcribe a numpy audio array using Whisper.
    Returns cleaned and reliable text.
    """
    global model
    with model_lock:
        m = model
    if m is None:
        raise RuntimeError("Whisper model not loaded.")

    try:
        audio = clean_audio(audio)  # <- use preprocessing pipeline

        t0 = time.perf_counter()
        segments, info = m.transcribe(
            audio,
            beam_size=5,
            best_of=5,
            patience=1.0,
            condition_on_previous_text=False,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        t1 = time.perf_counter()
        log.info(f"Transcription latency: {(t1-t0)*1000:.1f} ms")

        texts = [clean_text(seg.text) for seg in segments if _keep_segment(seg)]
        final_text = majority_vote(texts[-3:]) if texts else ""
        final_text = final_text.strip()
        log.info(f"Transcribed text: {final_text}")
        return final_text
    except Exception as e:
        log.error(f"Transcription error: {e}", exc_info=True)
        return ""

import asyncio

async def transcribe_async(audio: np.ndarray) -> str:
    """
    Async transcription of a numpy audio array using Whisper.
    Returns cleaned and reliable text.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, transcribe, audio)
