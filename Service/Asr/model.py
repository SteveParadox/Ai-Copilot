import os
import re
import time
import asyncio
import logging
import threading
from typing import Optional, Tuple, List, Sequence

import numpy as np
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from Asr.utils import weighted_vote
from Asr._preprocessing import clean_audio

# ===================== ENV =====================
load_dotenv()
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "tiny")

# ===================== LOGGING =====================
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ===================== GLOBALS =====================
model_lock = threading.Lock()
model: Optional[WhisperModel] = None

# ===================== CONFIG =====================
NO_SPEECH_MAX: float = float(os.getenv("NO_SPEECH_MAX", 0.50))
AVG_LOGPROB_MIN: float = float(os.getenv("AVG_LOGPROB_MIN", -1.10))
COMPRESSION_RATIO_MAX: float = float(os.getenv("COMPRESSION_RATIO_MAX", 2.4))
MIN_CHARS: int = int(os.getenv("MIN_CHARS", 6))
BANNED_SHORT: set[str] = {
    x.strip().lower()
    for x in os.getenv("BANNED_SHORT", "bye,thanks,thank you,okay,ok").split(",")
}

# ===================== DEVICE/COMPUTE =====================
def select_device_compute(device: str = "auto", compute: str = "auto") -> Tuple[str, str]:
    """
    Select optimal device and compute type for Whisper.
    """
    try:
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        log.info("Selected device=%s, compute=%s", device, compute)
        return device, compute
    except Exception as e:
        log.warning("Fallback to CPU/int8 due to error in device selection: %s", e)
        return "cpu", "int8"

# ===================== MODEL LOAD =====================
def load_whisper(model_size: str = DEFAULT_MODEL,
                 device: str = "auto",
                 compute: str = "auto",
                 retries: int = 3,
                 backoff: float = 2.0) -> WhisperModel:
    """
    Load Whisper model thread-safely with retry & backoff.
    """
    global model
    device, compute = select_device_compute(device, compute)

    for attempt in range(1, retries + 1):
        try:
            loaded_model = WhisperModel(model_size, device=device, compute_type=compute)
            with model_lock:
                model = loaded_model
            log.info("Whisper model loaded: %s (%s/%s)", model_size, device, compute)
            return loaded_model
        except Exception as e:
            log.error("Attempt %d to load Whisper failed: %s", attempt, e, exc_info=True)
            if attempt < retries:
                sleep_time = backoff ** attempt
                log.info("Retrying in %.1f seconds...", sleep_time)
                time.sleep(sleep_time)
            else:
                raise RuntimeError(f"Failed to load Whisper model after {retries} attempts") from e

def unload_whisper() -> None:
    """Unload and cleanup Whisper model."""
    global model
    with model_lock:
        if model is not None:
            try:
                model = None
                log.info("Whisper model unloaded successfully")
            except Exception as e:
                log.warning("Error unloading Whisper model: %s", e)

# ===================== SEGMENT FILTER =====================
def _keep_segment(seg) -> bool:
    """Check if segment passes reliability thresholds."""
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

# ===================== TEXT CLEANING =====================
_ngram_re = re.compile(r'\b(\w+(?:\s+\w+){1,5})\b(?:\s+\1\b)+', flags=re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalize ASR output by removing fillers, duplicates, extra spaces."""
    if not text:
        return ""
    fillers = ["okay", "um", "uh", "like", "you know", "i mean", "just got", "let's go"]
    for f in fillers:
        text = re.sub(rf"\b{re.escape(f)}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+)( \1){1,}\b", r"\1", text, flags=re.IGNORECASE)
    # --- Collapse repeated n-grams up to 6 words ---
    # e.g. "there are no hits on any hits on any hits"
    def _dedupe_ngrams(match: re.Match) -> str:
        return match.group(1)
    text = _ngram_re.sub(_dedupe_ngrams, text)

    # --- Collapse repeated sentences ---
    sentences = []
    seen = set()
    for s in re.split(r'(?<=[.!?])\s+', text.strip()):
        s_norm = s.lower().strip()
        if s_norm and s_norm not in seen:
            sentences.append(s.strip())
            seen.add(s_norm)
    text = " ".join(sentences)

    # --- Normalize spaces ---
    text = re.sub(r'\s+', ' ', text).strip()

    return text
# ===================== TRANSCRIPTION =====================
def _transcribe(audio: np.ndarray, timeout: float = 30.0) -> str:
    """
    Blocking transcription with timeout safeguard.
    """
    global model
    with model_lock:
        m = model
    if m is None:
        raise RuntimeError("Whisper model not loaded")

    try:
        #audio = clean_audio(audio)

        start = time.perf_counter()
        segments_gen, _ = m.transcribe(
            audio,
            beam_size=8,
            best_of=8,
            patience=1.2,
            condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4],
            vad_filter=False,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        elapsed = (time.perf_counter() - start) * 1000
        log.info("Transcription latency: %.1f ms", elapsed)
        texts = []
        for seg in segments_gen:
            if _keep_segment(seg):
                t = (seg.text or "").strip()
                if t:
                    t = clean_text(t)
                    texts.append(t)

        return " ".join(texts).strip()
    except Exception as e:
        log.error(f"[TRANSCRIBE] Error during transcription: {e}")
        return ""

    """    segments = []
        for seg in segments_gen:
            if (time.perf_counter() - start) > timeout:
                log.warning("Transcription aborted due to timeout (%.1fs)", timeout)
                break
            segments.append(seg)

        elapsed = (time.perf_counter() - start) * 1000
        log.info("Transcription latency: %.1f ms", elapsed)

        texts, weights = [], []
        for seg in segments[-3:]:
            if not getattr(seg, "text", "").strip():
                continue
            if not _keep_segment(seg):
                continue
            texts.append(seg.text.strip())
            weights.append(1 - getattr(seg, "no_speech_prob", 0))

        if not texts:
            return ""

        final_text = weighted_vote(texts, weights)
        return clean_text(final_text or "")
    except Exception as e:
        log.error("Transcription error: %s", e, exc_info=True)
        return """
    

async def transcribe_async(audio: np.ndarray, timeout: float = 30.0) -> str:
    """
    Async transcription of numpy audio array with Whisper.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _transcribe, audio, timeout)

# ===================== BATCH TRANSCRIPTION =====================
def transcribe_batch(audio_list: Sequence[np.ndarray], timeout: float = 30.0) -> List[str]:
    """
    Transcribe a batch of audio clips sequentially.
    Suitable for offline jobs or multi-utterance processing.
    """
    results: List[str] = []
    for i, audio in enumerate(audio_list, 1):
        try:
            text = _transcribe(audio, timeout=timeout)
            results.append(text)
            log.info("Batch item %d/%d -> '%s'", i, len(audio_list), text[:50])
        except Exception as e:
            log.error("Batch item %d failed: %s", i, e, exc_info=True)
            results.append("")
    return results

async def transcribe_batch_async(audio_list: Sequence[np.ndarray], timeout: float = 30.0) -> List[str]:
    """
    Async transcription of multiple audio clips concurrently.
    Useful for high-throughput servers or streaming pipelines.
    """
    tasks = [transcribe_async(audio, timeout=timeout) for audio in audio_list]
    return await asyncio.gather(*tasks, return_exceptions=False)
