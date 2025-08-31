# streaming.py
import os
import threading
import time
import queue
from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
import collections
import logging
from Asr._preprocessing import clean_audio
from device import validate_device, list_devices, _pick_input_device
from Asr.model import transcribe


log = logging.getLogger("Streaming")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv()


# ===================== CONFIG =====================
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = os.getenv("BLOCK_SIZE", 1024)        # frames per callback
AUDIO_DEVICE_INDEX = int(os.getenv("AUDIO_DEVICE_INDEX", -1))
BUFFER_MAX_SECONDS = 10  # ring buffer size
VAD_THRESHOLD = 0.01     # simple energy-based VAD
SILENCE_TIMEOUT = 10.2    # seconds to trigger endpoint

# ===================== RING BUFFER =====================
class RingBuffer:
    """Thread-safe fixed-size ring buffer for audio frames"""
    def __init__(self, maxlen: int):
        self.buffer = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, data: np.ndarray):
        with self.lock:
            self.buffer.append(data.copy())

    def get_all(self) -> np.ndarray:
        with self.lock:
            if not self.buffer:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(list(self.buffer))

    def clear(self):
        with self.lock:
            self.buffer.clear()

# ===================== SIMPLE ENERGY-BASED VAD =====================
def is_speech(frame: np.ndarray, threshold: float = VAD_THRESHOLD) -> bool:
    energy = np.mean(frame**2)
    return energy > threshold

# ===================== STREAMING WORKER =====================
class AudioStreamer:
    def __init__(self, callback, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, device_index=None):
        """
        callback: function(audio_chunk: np.ndarray) -> None
        """
        self.callback = callback
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.ring_buffer = RingBuffer(maxlen=int(BUFFER_MAX_SECONDS * sample_rate / block_size))
        self.silence_timer = 0
        self.running = False
        self.thread = None

        if device_index is not None:
            validate_device(device_index, input=True, channels=CHANNELS, samplerate=sample_rate)
            log.info(f"Using device {device_index}: {list_devices()[device_index]}")
            self.device_index = device_index

    def _audio_callback(self, indata, frames, time_info, status):
        if indata is None or frames <= 0:
            log.debug("Empty audio frame received (ignored)")
            return
        try:
            audio = indata[:, 0] if indata.ndim > 1 else indata
            audio = clean_audio(audio)

            #audio = audio.astype(np.float32)
            self.ring_buffer.append(audio)

            if is_speech(audio):
                self.silence_timer = 0
            else:
                self.silence_timer += frames / self.sample_rate

            # Endpoint detected
            if self.silence_timer >= SILENCE_TIMEOUT:
                chunk = self.ring_buffer.get_all()
                if len(chunk) > 0:
                    try:
                        self.callback(chunk)
                    except Exception as e:
                        log.error(f"Callback error: {e}")
                self.ring_buffer.clear()
                self.silence_timer = 0
        except Exception as e:
            log.error(f"Mic level computation failed: {e}", exc_info=True)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_stream, daemon=True)
        self.thread.start()
        log.info("AudioStreamer started.")

    def _run_stream(self):
        device_index, ch = _pick_input_device(AUDIO_DEVICE_INDEX)
        try:
            with sd.InputStream(
                device=device_index,
                samplerate=self.sample_rate,
                channels=CHANNELS,
                blocksize=self.block_size,
                dtype="float32",
                callback=self._audio_callback,
                latency="low",
            ):
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            log.error(f"Audio stream error: {e}")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        log.info("AudioStreamer stopped.")
