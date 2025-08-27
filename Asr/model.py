import os
import threading
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel


class ASRModel:
    def __init__(self):
        # Load configs from environment or defaults
        self.model_name = os.getenv("WHISPER_MODEL", "base")
        self.device = os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

        self.model = None
        self.model_lock = threading.Lock()

    def load_model(self):
        """Load Whisper model once at startup."""
        print(f"[ASR] Loading Whisper model: {self.model_name} on {self.device}")
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type
        )
        print("[ASR] Whisper model loaded successfully.")

    def transcribe(self, audio: np.ndarray, sr: int = 16000, partial: bool = False) -> str:
        """
        Transcribe audio from a NumPy array or from a file path.

        Args:
            audio: np.ndarray (raw waveform) or str (path to audio file)
            sr: sample rate of the array input (ignored if audio is a file)
            partial: if True, disables VAD for faster partial transcription
        """
        if self.model is None:
            raise RuntimeError("ASR model is not loaded")

        start_time = time.time()

        # Handle file path OR numpy array
        if isinstance(audio, str):
            input_audio = audio  # file path
        else:
            audio = self.normalize_audio(audio, sr)  # ensure float32 + 16kHz
            input_audio = audio

        with self.model_lock:
            segments, _ = self.model.transcribe(
                input_audio,
                beam_size=5,
                vad_filter=not partial
            )
            text = " ".join([seg.text for seg in segments])

        latency = time.time() - start_time
        mode = "partial" if partial else "final"
        print(f"[ASR] {mode} transcription finished in {latency:.2f}s")
        return text.strip()

    @staticmethod
    def normalize_audio(audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Normalize audio to float32 PCM and resample to 16kHz if needed.

        Args:
            audio: np.ndarray waveform
            sr: original sample rate
        """
        # Convert dtype if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != 16000:
            gcd = np.gcd(sr, 16000)
            up = 16000 // gcd
            down = sr // gcd
            audio = resample_poly(audio, up, down)

        return audio
