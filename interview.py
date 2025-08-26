import asyncio
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import re
from typing import List
import numpy as np
import sounddevice as sd
from openai import OpenAI
from faster_whisper import WhisperModel
from dotenv import load_dotenv

from collections import deque

import logging
import sys
from pathlib import Path

# ===================== LOGGING SETUP =====================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "interview_copilot.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)  # keep console output
    ]
)

def log_debug(msg: str):
    logging.debug(msg)

def log_info(msg: str):
    logging.info(msg)

def log_warn(msg: str):
    logging.warning(msg)

def log_error(msg: str):
    logging.error(msg, exc_info=True)  # include traceback


# ===================== WINDOW PROTECTION =====================
import ctypes
import sys

def protect_window_from_sharing(root: tk.Tk):
    """
    On Windows: prevent Tkinter window from being captured in screen share or screenshots.
    On other OS: does nothing (safe no-op).
    """
    if sys.platform == "win32":
        WDA_NONE = 0x0
        WDA_MONITOR = 0x1
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        res = ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_MONITOR)
        if res == 0:
            log_warn("[GUI] Failed to set display affinity (maybe unsupported GPU/driver).")
        else:
            log_info("[GUI] Window display affinity set → hidden from sharing.")


# ===================== ENV & CONFIG =====================
print("[INIT] Loading environment variables...")
load_dotenv()

ENV_MODEL = os.getenv("WHISPER_MODEL", "tiny")
ENV_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # "auto" | "cpu" | "cuda"
ENV_COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")  # "auto" | "int8" | "float16"

SAMPLE_RATE = 16000  # Whisper-friendly
CHANNELS = 1

AUDIO_DEVICE_INDEX = int(os.getenv("AUDIO_DEVICE_INDEX", -1))

OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    log_error("OPENAI_API_KEY not set. Create a .env with OPENAI_API_KEY=...")
    raise RuntimeError("OPENAI_API_KEY not set. Create a .env with OPENAI_API_KEY=...")

log_info("Initializing OpenAI client...")
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    log_error(f"Failed to initialize OpenAI client: {e}")
    raise

# ===================== GLOBALS =====================
# UI state must remain on main thread
state_lock = threading.Lock()
is_recording = False
shutdown = False

# Async stuff
loop: asyncio.AbstractEventLoop | None = None
audio_q: asyncio.Queue[tuple[np.ndarray, float]] = asyncio.Queue(maxsize=5)


# Keep a simple UI queue for Tk (main thread only)
import queue as _q
ui_q: "_q.Queue[tuple]" = _q.Queue()


# Whisper model (shared); access under model_lock
model_lock = threading.Lock()
model: WhisperModel | None = None

# ===================== STREAMING PARAMS =====================
# Stable near-real-time windows
AUDIO_BLOCK_SECONDS = 5   # send audio every 5s for low latency

# ===================== STREAMING PARAMS =====================
# Larger window + longer hop gives Whisper more context (fewer fragments)
STREAM_WINDOW_SEC = 4.0     # was 2.0
STREAM_HOP_SEC    = 1.0     # was 0.5
STREAM_WINDOW_SAMPLES = int(STREAM_WINDOW_SEC * SAMPLE_RATE)
STREAM_HOP_SAMPLES    = int(STREAM_HOP_SEC * SAMPLE_RATE)

MIN_INFER_SAMPLES = int(0.8 * SAMPLE_RATE)  # allow inference after ~0.8s

# --- mic diagnostics ---
mic_last_level = 0.0
mic_last_chunk_ts = 0.0
MIC_LEVEL_THROTTLE_SEC = 0.25


# ===================== STABILITY FILTER TUNING =====================
# Voice Activity Detection parameters (faster-whisper built-in VAD)
VAD_PARAMS = {
    "vad_filter": True,
    "vad_parameters": {
        "threshold": 0.5,                 # sensitivity (lower = more sensitive)
        "min_speech_duration_ms": 200,    # ignore ultra-short blips
        "max_speech_duration_s": 30,      # split overly long speech
        "min_silence_duration_ms": 350,   # debounce end of speech
        "speech_pad_ms": 150,             # pad around speech
    },
}

# Segment/decoder confidence gates
NO_SPEECH_MAX = 0.50          # discard if model thinks it's likely "no speech"
AVG_LOGPROB_MIN = -1.10        # too low => likely garbage
COMPRESSION_RATIO_MAX = 2.4    # high = repetitive text (hallucinations)
MIN_CHARS = 6                  # very short snippets are noisy
BANNED_SHORT = {"bye", "thanks", "thank you", "okay", "ok"}

# UI debouncing
STABLE_VOTES = 2               # need N identical hypotheses across recent windows
VOTE_WINDOW = 3                # look back over last N windows

# Endpointing (finalize question on extended silence)
ENDPOINT_SILENCE_MS = 1100

# ===================== UTIL =====================
def _select_device_and_compute():
    """
    Decide device & compute_type intelligently.
    - If ENV_DEVICE/ENV_COMPUTE set explicitly, honor them.
    - Else prefer CUDA + float16. Fallbacks: CPU + int8.
    """
    device = ENV_DEVICE
    compute = ENV_COMPUTE

    try:
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"

        log_info(f"Selected device={device}, compute={compute}")
        return device, compute

    except Exception as e:
        log_warn(f"Failed to auto-select device/compute, falling back to CPU/int8: {e}")
        return "cpu", "int8"


def _prep_audio_float32(x: np.ndarray) -> np.ndarray:
    """Ensure mono float32 in [-1, 1]."""
    try:
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32)

        if x.ndim == 2 and x.shape[1] > 1:
            x = x.mean(axis=1, dtype=np.float32)
        elif x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]

        return x

    except Exception as e:
        log_error(f"Audio preprocessing failed: {e}, input shape={x.shape}, dtype={x.dtype}")
        raise


def _pick_input_device(explicit_index: int) -> tuple[int, int]:
    """
    Return (device_index, channels) for input.
    - If explicit_index is valid, use it.
    - Else use default input device.
    - Ensure at least 1 channel is available.
    """
    try:
        devices = sd.query_devices()
        if explicit_index >= 0:
            info = sd.query_devices(explicit_index)
            max_ch = int(info.get("max_input_channels", 0))
            if max_ch <= 0:
                log_warn(f"Explicit device {explicit_index} has no input channels, falling back to default")
                explicit_index = -1  # fallback to default
            else:
                log_info(f"Using explicit input device {explicit_index} with {max_ch} channels")
                return explicit_index, min(CHANNELS, max_ch) or 1

        # default input device
        cur_dev = sd.default.device
        default_in = cur_dev[0] if isinstance(cur_dev, (list, tuple)) else None

        if default_in is None or default_in < 0:
            # pick the first device that has input channels
            for i, d in enumerate(devices):
                if int(d.get("max_input_channels", 0)) > 0:
                    out_dev = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None
                    sd.default.device = (i, out_dev)
                    log_info(f"Auto-selected input device {i} with {d['max_input_channels']} channels")
                    return i, min(CHANNELS, int(d["max_input_channels"])) or 1
                
            raise RuntimeError("No input devices with channels > 0 were found.")

        info = sd.query_devices(default_in)
        max_ch = int(info.get("max_input_channels", 0))
        if max_ch <= 0:
            raise RuntimeError(f"Default input device {default_in} has no input channels")
        log_info(f"Using default input device {default_in} with {max_ch} channels")

        return default_in, min(CHANNELS, max_ch) or 1
    except Exception as e:
        log_error(f"Audio device selection failed: {e}")

        raise RuntimeError(f"Audio device selection failed: {e}")

# ===================== MODEL LOAD/RELOAD =====================
def load_whisper(selected_size: str):
    """(Re)load the Whisper model with the chosen size and best device/compute settings."""
    global model
    try:
        device, compute = _select_device_and_compute()
        log_info(f"Loading Whisper model: size={selected_size}, device={device}, compute={compute}")

        m = WhisperModel(selected_size, device=device, compute_type=compute)
        with model_lock:
            model = m
        log_info("Whisper model loaded and ready.")
    except Exception as e:
        log_error(f"Failed to load Whisper model: {e}")
        raise RuntimeError(f"Failed to load Whisper model: {e}")

# ===================== GPT =====================
def draft_answer(question: str, style: str, company: str) -> str:
    if not question:
        log_warn("[GPT] No question detected.")

        return "I couldn't hear a question clearly. Please try again."

    hint = f" The company is {company}." if company else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI interview assistant. "
                f"Give a {style} interview answer. Be structured, specific, and concise, in layman terms"
            ),
        },
        {"role": "user", "content": question + hint},
    ]
    try:
        log_info(f"[GPT] Sending request to OpenAI (streaming)...")
        ui_q.put(("answer_stream_start", None))

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=500,
            stream=True
        )

        full_answer = []
        for chunk in resp:
            if not getattr(chunk, "choices", None):
                continue
            delta_obj = getattr(chunk.choices[0], "delta", None)
            delta = getattr(delta_obj, "content", "") or ""
            if delta:
                ui_q.put(("answer_stream", delta))
                full_answer.append(delta)

        final_answer = "".join(full_answer).strip()
        ui_q.put(("answer_stream_end", None))
        log_info("[GPT] Streaming complete.")
        return final_answer

    except Exception as e:
        log_error(f"[GPT] Streaming failed: {e}")
        ui_q.put(("answer_stream_end", None))
        return f"OpenAI error: {e}"
    
# ===================== TRANSCRIBE (STREAMING) =====================
# ===================== ASR (RUN IN THREADPOOL VIA ASYNCIO) =====================
def _transcribe_window_sync(window_audio: np.ndarray) -> str:
    # Read the global model safely
    with model_lock:
        m = model
    if m is None:
        log_warn("[TRANSCRIBE] Model not loaded.")
        return ""
    
    try:
        t0 = time.perf_counter()

        # Request transcription on the full window. Disable faster-whisper's internal VAD so
        # segments aren't chopped into tiny fragments which cause fragmented "words" later.
        segments, info = m.transcribe(
            window_audio,
            beam_size=1,
            condition_on_previous_text=False,
            temperature=0.0,
            vad_filter=False,                # <-- disabled model VAD to reduce fragmentation
            vad_parameters=None,
        )

        t1 = time.perf_counter()
        log_info(f"[TRANSCRIBE] Latency: {(t1 - t0)*1000:.1f} ms")

        def _keep(seg) -> bool:
            no_speech_prob = getattr(seg, "no_speech_prob", None)
            avg_logprob    = getattr(seg, "avg_logprob", None)
            comp_ratio     = getattr(seg, "compression_ratio", 1.0)
            txt = (seg.text or "").strip().lower()

            if no_speech_prob is not None and no_speech_prob > NO_SPEECH_MAX:
                return False
            if avg_logprob is not None and avg_logprob < AVG_LOGPROB_MIN:
                return False
            if comp_ratio is not None and comp_ratio > COMPRESSION_RATIO_MAX:
                return False
            if len(txt) < MIN_CHARS:
                return False
            if txt in BANNED_SHORT:
                return False
            return True

        texts = []
        for seg in segments:
            if _keep(seg):
                t = (seg.text or "").strip()
                if t:
                    texts.append(t)

        return " ".join(texts).strip()
    except Exception as e:
        log_error(f"[TRANSCRIBE] Error during transcription: {e}")
        return ""


_normalize_re = re.compile(r"[^\w\s]")

def _normalize_text(s: str) -> str:
    """Lower, remove punctuation, collapse whitespace for voting/compare."""
    if not s:
        return ""
    s = s.lower().strip()
    s = _normalize_re.sub("", s)
    s = " ".join(s.split())
    return s

def _majority_vote(candidates: List[str], min_votes: int) -> str | None:
    """
    Return the candidate that has >= min_votes occurrences after normalization.
    If none reaches threshold, return the longest common prefix (if it's long enough).
    """
    if not candidates:
        return None
    norm_map = {}
    for c in candidates:
        n = _normalize_text(c)
        if not n:
            continue
        norm_map.setdefault(n, 0)
        norm_map[n] += 1

    # pick candidate with highest count that meets min_votes
    for cand, cnt in norm_map.items():
        if cnt >= min_votes:
            return cand

    # fallback: longest common prefix across normalized candidates
    norms = list(norm_map.keys())
    if not norms:
        return None
    # compute LCP
    prefix = norms[0]
    for s in norms[1:]:
        # shrink prefix while not a prefix of s
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                break
    if prefix and len(prefix) >= MIN_CHARS:
        return prefix
    return None

# ===================== WORKER (STREAMING) =====================
# ===================== ASYNC RECORDING WORKER =====================
async def recording_worker(style_var: tk.StringVar, company_var: tk.StringVar):
    """
    Refactored streaming worker:
    - Larger window & hop (set above) for better context.
    - Immediate partial append for responsiveness.
    - Faster consensus commit (commits and clears recent_hyps).
    """
    log_info(f"[WORKER] Streaming recording started")
    ui_q.put(("status", "Recording… Click Stop to finish"))

    # ring buffer for the rolling window (avoid repeated concatenation)
    window_size = STREAM_WINDOW_SAMPLES
    ring = np.zeros(window_size, dtype=np.float32)
    write_pos = 0
    filled = 0

    samples_since_last_hop = 0

    recent_hyps = deque(maxlen=VOTE_WINDOW)
    last_partial_norm = ""
    last_committed_norm = ""
    last_voiced_ts = time.time()
    last_ui_update_ts = 0.0
    UI_UPDATE_THROTTLE = 0.25

    try:
        while True:
            with state_lock:
                if not is_recording:
                    break

            try:
                chunk = await asyncio.wait_for(audio_q.get(), timeout=0.5)
                f32 = _prep_audio_float32(chunk)
                if f32.size == 0:
                    continue
                n = f32.shape[0]

                # write into ring buffer (wrap)
                end_pos = write_pos + n
                if end_pos <= window_size:
                    ring[write_pos:end_pos] = f32
                else:
                    first_len = window_size - write_pos
                    ring[write_pos:window_size] = f32[:first_len]
                    ring[0:end_pos % window_size] = f32[first_len:]
                write_pos = end_pos % window_size
                filled = min(window_size, filled + n)
                samples_since_last_hop += n

                # simple energy VAD to update last voiced time
                if np.max(np.abs(f32)) > 1e-3:
                    last_voiced_ts = time.time()

                # endpoint: final capture after silence
                if (time.time() - last_voiced_ts) * 1000 > ENDPOINT_SILENCE_MS and filled >= MIN_INFER_SAMPLES:
                    if filled < window_size:
                        window = ring[:filled].astype(np.float32)
                    else:
                        if write_pos == 0:
                            window = ring.copy()
                        else:
                            window = np.concatenate((ring[write_pos:], ring[:write_pos])).astype(np.float32)

                    hyp = await asyncio.to_thread(_transcribe_window_sync, window)
                    # endpoint: final capture after silence
                    if hyp:
                        norm_h = _normalize_text(hyp)
                        if norm_h and norm_h != last_partial_norm:
                            ui_q.put(("preview_overwrite", hyp))
                            last_partial_norm = norm_h

                        recent_hyps.append(hyp)

                        winner = _majority_vote(list(recent_hyps), STABLE_VOTES)
                        now_ts = time.time()
                        if winner and (now_ts - last_ui_update_ts) >= UI_UPDATE_THROTTLE:
                            display_choice = winner
                            for orig in reversed(recent_hyps):
                                if _normalize_text(orig) == winner:
                                    display_choice = orig
                                    break
                            if _normalize_text(display_choice) != last_committed_norm:
                                ui_q.put(("commit_text", display_choice))
                                last_committed_norm = _normalize_text(display_choice)
                                recent_hyps.clear()
                                last_ui_update_ts = now_ts

                    break

                if filled < MIN_INFER_SAMPLES:
                    continue

                # hop => periodic transcription
                if samples_since_last_hop >= STREAM_HOP_SAMPLES:
                    samples_since_last_hop = 0

                    if filled < window_size:
                        window = ring[:filled].astype(np.float32)
                    else:
                        if write_pos == 0:
                            window = ring.copy()
                        else:
                            window = np.concatenate((ring[write_pos:], ring[:write_pos])).astype(np.float32)

                    hyp = await asyncio.to_thread(_transcribe_window_sync, window)
                    if not hyp:
                        continue

                    # immediate responsive partial (append, but only when changed)
                    # hop => periodic transcription
                    norm_h = _normalize_text(hyp)
                    if norm_h and norm_h != last_partial_norm:
                        ui_q.put(("preview_overwrite", hyp))
                        last_partial_norm = norm_h

                    recent_hyps.append(hyp)

                    winner = _majority_vote(list(recent_hyps), STABLE_VOTES)
                    now_ts = time.time()
                    if winner and (now_ts - last_ui_update_ts) >= UI_UPDATE_THROTTLE:
                        display_choice = winner
                        for orig in reversed(recent_hyps):
                            if _normalize_text(orig) == winner:
                                display_choice = orig
                                break
                        if _normalize_text(display_choice) != last_committed_norm:
                            ui_q.put(("commit_text", display_choice))
                            last_committed_norm = _normalize_text(display_choice)
                            recent_hyps.clear()
                            last_ui_update_ts = now_ts

            except asyncio.TimeoutError:
                # regular loop; allows checking is_recording
                pass

    except Exception as e:
        log_info(f"[WORKER] Audio/ASR error: {e}")
        ui_q.put(("status", f"Audio/ASR error: {e}"))
        return

    # finalize -> send to GPT
    final_question = ""
    if last_committed_norm:
        final_question = last_committed_norm
    elif recent_hyps:
        final_question = _normalize_text(recent_hyps[-1])
    final_question = (final_question or "").strip()
    if not final_question:
        log_error("[WORKER] No speech recognized.")
        ui_q.put(("status", "No speech recognized. Try again."))
        return

    log_info("[WORKER] Finalizing → GPT…")
    ui_q.put(("status", "Generating answer…"))
    answer = await asyncio.to_thread(draft_answer, final_question, style_var.get(), company_var.get())
    ui_q.put(("answer", answer))
    ui_q.put(("status", "Ready. Click Record to start again."))
    log_info("[WORKER] Done. Answer sent to UI.")


# ===================== AUDIO CALLBACK =====================
# We need to feed frames into an asyncio.Queue from a non-async thread.
async def _audio_enqueue_with_drop(frame: np.ndarray):
    if audio_q is None:
        return
    try:
        if audio_q.full():
            try:
                audio_q.get_nowait()
            except asyncio.QueueEmpty:
                log_warn("Audio queue full, dropping oldest frame")
        await asyncio.sleep(0)   # yield to loop; helps on some backends
        audio_q.put_nowait(frame)
    except asyncio.QueueFull:
        log_error(f"Audio queue is unexpectedly full (put_nowait failed)")


def audio_callback(indata, frames, time_info, status):
    global mic_last_level, mic_last_chunk_ts
    if status:
        log_warn(f"[AUDIO] Status warning: {status}")
        ui_q.put(("status", f"Audio status: {status}"))

    # safety: ensure we actually have samples
    if indata is None or frames <= 0:
        log_debug("Empty audio frame received (ignored)")
        return

    # level meter (works even when not recording)
    try:
        # indata is float32 already; robustly handle shape
        f32 = indata if indata.dtype == np.float32 else indata.astype(np.float32)
        if f32.ndim == 2 and f32.shape[1] > 1:
            f32 = f32.mean(axis=1)
        elif f32.ndim == 2:
            f32 = f32[:, 0]
        peak = float(np.max(np.abs(f32))) if f32.size else 0.0
        now = time.monotonic()
        if now - mic_last_chunk_ts > MIC_LEVEL_THROTTLE_SEC:
            mic_last_chunk_ts = now
            mic_last_level = peak
            ui_q.put(("mic_level", peak))
    except Exception:
        log_error(f"Mic level computation failed: {e}")

    with state_lock:
        rec = is_recording
    if not rec:
        return

    # enqueue to asyncio queue
    try:
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                _audio_enqueue_with_drop(indata.copy()), loop
            )
        else:
            # loop missing: surface loudly to UI
            ui_q.put(("status", "Audio loop not running; restarting may help."))
            log_error("Audio loop not running; restarting may help.")
    except Exception as e:
        log_error(f"[AUDIO] Enqueue error: {e}")

# ===================== GUI =====================
class InterviewCopilot(tk.Tk):
    def __init__(self):
        super().__init__()
        print("[GUI] Starting Tkinter UI...")
        self.title("AI Interview Copilot")
        self.geometry("780x520+60+60")
        self.attributes("-topmost", True)

        protect_window_from_sharing(self)

        # ----- Top controls -----
        top = tk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        # Company hint
        tk.Label(top, text="Company Hint:").grid(row=0, column=0, sticky="w")
        self.company_hint = tk.StringVar(value="")
        tk.Entry(top, textvariable=self.company_hint).grid(row=0, column=1, sticky="ew", padx=6)

        # Style
        tk.Label(top, text="Answer Style:").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.current_style = tk.StringVar(value="Concise")
        ttk.Combobox(
            top,
            textvariable=self.current_style,
            values=["Concise", "STAR", "Deep Technical"],
            state="readonly",
            width=18,
        ).grid(row=0, column=3, sticky="w", padx=6)

        # Model size selector (latency/accuracy trade-off)
        tk.Label(top, text="Model Size:").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.model_size = tk.StringVar(value=ENV_MODEL)
        ttk.Combobox(
            top,
            textvariable=self.model_size,
            values=["tiny", "base", "small", "medium", "large-v3"],
            state="readonly",
            width=18,
        ).grid(row=1, column=1, sticky="w", padx=6, pady=(6,0))

        # Status
        self.status_var = tk.StringVar(value="Ready. Click Record to start.")
        tk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", padx=8, pady=(4, 0))

        # Textbox
        self.textbox = tk.Text(self, wrap="word", height=16)
        self.textbox.pack(fill="both", expand=True, padx=8, pady=8)
        self.textbox.mark_set("preview_start", "end")
        self.textbox.mark_set("preview_end", "end")


        # Buttons row
        frm_btns = tk.Frame(self)
        frm_btns.pack(pady=6)
        self.record_btn = tk.Button(frm_btns, text="🎤 Record", command=self.toggle_record)
        self.record_btn.pack(side="left", padx=6)
        tk.Button(frm_btns, text="📋 Copy", command=self.copy_answer).pack(side="left", padx=6)
        tk.Button(frm_btns, text="❌ Clear", command=self.clear_text).pack(side="left", padx=6)
        tk.Button(frm_btns, text="↻ Reload Model", command=self.reload_model).pack(side="left", padx=6)

        # Grid sizing
        top.grid_columnconfigure(1, weight=1)

        # UI queue polling
        self.after(100, self._drain_ui_queue)

        # Cleanup hook
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.focus_force()
        print("[GUI] Tkinter ready.")

    # ----- UI helpers -----
    def set_status(self, msg: str):
        print(f"[GUI] STATUS: {msg}")
        self.status_var.set(msg)

    def show_answer(self, answer: str):
        print(f"[GUI] Displaying answer: {answer[:50]}...")
        self.textbox.delete("1.0", tk.END)
        self.textbox.insert(tk.END, answer)

    def copy_answer(self):
        text = self.textbox.get("1.0", tk.END).strip()
        if not text:
            self.set_status("Nothing to copy.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.set_status("Copied to clipboard.")
        print("[GUI] Answer copied to clipboard.")

    def clear_text(self):
        self.show_answer("")
        self.set_status("Cleared.")
        print("[GUI] Cleared textbox.")

    # ----- Model control -----
    def reload_model(self):
        # Avoid reloading while a session is active
        with state_lock:
            if is_recording:
                messagebox.showinfo("Busy", "Stop recording before reloading the model.")
                return
        size = self.model_size.get().strip()
        self.set_status(f"Loading model '{size}'…")
        # Load off the UI thread
        threading.Thread(target=lambda: (load_whisper(size), ui_q.put(("status", f"Model '{size}' loaded."))), daemon=True).start()

    # ----- Record button logic -----
    def toggle_record(self):
        global is_recording
        with state_lock:
            starting = not is_recording
            is_recording = not is_recording

        if starting:
            print("[GUI] Button pressed: Starting recording...")
            self.set_status("Recording… speaking is transcribed live")
            self.record_btn.config(text="⏹ Stop")
            # Flush audio queue before start
            if loop and audio_q:
                def _flush():
                    try:
                        while True:
                            audio_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                loop.call_soon_threadsafe(_flush)
            # Kick off async worker
            if loop:
                asyncio.run_coroutine_threadsafe(
                    recording_worker(self.current_style, self.company_hint), loop
                )
        else:
            print("[GUI] Button pressed: Stopping recording...")
            self.set_status("Stopping…")
            self.record_btn.config(text="🎤 Record")

    # ----- Queue drain -----
    def _drain_ui_queue(self):
        try:
            while True:
                action, payload = ui_q.get_nowait()

                if action == "status":
                    self.set_status(str(payload))

                elif action == "answer":
                    self.show_answer(str(payload))

                elif action == "partial_append":
                    self.textbox.insert(tk.END, str(payload))
                    self.textbox.see(tk.END)

                elif action == "partial_overwrite":
                    # Instead of clearing, just append the new recognized chunk
                    self.textbox.insert(tk.END, str(payload) + " ")
                    self.textbox.see(tk.END)

                elif action == "answer_stream_start":
                    self.textbox.delete("1.0", tk.END)

                elif action == "answer_stream":
                    self.textbox.insert(tk.END, str(payload))
                    self.textbox.see(tk.END)

                elif action == "preview_overwrite":
                    # Replace the preview line (live hypothesis)
                    self.textbox.delete("preview_start", "preview_end")
                    self.textbox.insert("end", str(payload) + " ", ("preview",))
                    self.textbox.mark_set("preview_start", "end-1c linestart")
                    self.textbox.mark_set("preview_end", "end")
                    self.textbox.see("end")

                elif action == "commit_text":
                    # Finalized, stable text — remove preview, append commit
                    self.textbox.delete("preview_start", "preview_end")
                    self.textbox.insert("end", str(payload) + "\n")
                    self.textbox.see("end")

                elif action == "answer_stream_end":
                    self.textbox.insert(tk.END, "\n")
                    self.textbox.see(tk.END)
                elif action == "mic_level":
                    # simple textual meter; replace with a progressbar if you like
                    lvl = float(payload)
                    # Map ~[-1..1] to a 0..100 % (rough)
                    pct = min(100, int(lvl * 100))
                    self.status_var.set(f"🎙 Mic level: {pct}%")  # keeps recording status intact

        except _q.Empty:
            pass

        if not shutdown:
            self.after(100, self._drain_ui_queue)

    def on_close(self):
        global shutdown, is_recording
        print("[GUI] Closing app...")
        with state_lock:
            is_recording = False
        shutdown = True
        self.destroy()
        print("[GUI] Closed successfully.")

# ===================== ASYNCIO LOOP BOOTSTRAP =====================
def start_asyncio_runtime():
    """Run a dedicated asyncio loop in a background thread."""
    global loop, audio_q
    loop = asyncio.new_event_loop()
    audio_q = asyncio.Queue(maxsize=400)   # bounded queue to avoid runaway growth

    def _runner():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_runner, name="AsyncioLoop", daemon=True)
    t.start()

# ===================== SOUND DEVICE STREAM =====================
def start_input_stream():
    """Create and validate the sounddevice stream; calls our callback continuously."""
    dev_idx, ch = _pick_input_device(AUDIO_DEVICE_INDEX)
    print(f"[AUDIO] Using input device {dev_idx} with {ch} channel(s) @ {SAMPLE_RATE} Hz")

    try:
        sd.check_input_settings(device=dev_idx, samplerate=SAMPLE_RATE, channels=ch, dtype="float32")
    except Exception as e:
        raise RuntimeError(f"Input settings invalid: {e}")

    try:
        cur = sd.default.device
        out_dev = cur[1] if isinstance(cur, (list, tuple)) else None
        if out_dev is not None:
            sd.default.device = (dev_idx, out_dev)
    except Exception:
        pass

    # medium blocksize gives a good balance of throughput & latency across platforms
    return sd.InputStream(
        device=dev_idx,
        samplerate=SAMPLE_RATE,
        channels=ch,
        dtype="float32",
        callback=audio_callback,
        blocksize=4096,         # ↑ increased from defaults but small enough for responsive stream
        latency="low",
    )

# ===================== MAIN =====================
def main():
    try:
        print("[MAIN] Booting runtime…")
        start_asyncio_runtime()
        try:
            print("[AUDIO] Available devices:")
            for i, d in enumerate(sd.query_devices()):
                print(f"  [{i}] in={d.get('max_input_channels',0)} out={d.get('max_output_channels',0)} :: {d.get('name')}")
        except Exception as e:
            print(f"[AUDIO] Device list failed: {e}")

        # Load initial model (honors CUDA+float16 if available)
        load_whisper(ENV_MODEL)

        # Start audio stream
        stream = start_input_stream()
        stream.start()

        print("[MAIN] Launching InterviewCopilot UI…")
        app = InterviewCopilot()
        app.mainloop()

        print("[MAIN] Stopping audio stream…")
        stream.stop(); stream.close()

        # Stop asyncio loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

    except Exception as e:
        print(f"[MAIN] Fatal error: {e}")
        try:
            messagebox.showerror("Fatal error", str(e))
        except Exception:
            pass

if __name__ == "__main__":
    main()