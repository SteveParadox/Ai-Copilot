"""
Audio Streaming Module - Extended Edition
========================================
Real-time and simulated audio streaming with advanced features.

Features:
- Multi-channel support (stereo processing, channel selection, mixing)
- Audio file streaming (simulate real-time from files)
- Silence suppression (skip silent portions)
- Continuous mode (fixed chunk sizes)
- Endpoint detection (simple VAD + silence timeout)
"""

from __future__ import annotations

import os
import time
import logging
import threading
import collections
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

from Asr.preprocessing import Preprocessor
from Asr.device import validate_device, _pick_input_device
from Asr.model import transcribe

load_dotenv()

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

# ============================================================================
# Enums
# ============================================================================

class ChannelMode(Enum):
    """Channel processing modes."""
    MONO = "mono"
    STEREO = "stereo"        # returns (left, right)
    LEFT_ONLY = "left"
    RIGHT_ONLY = "right"
    MIX = "mix"              # average all channels


class StreamMode(Enum):
    """Streaming modes."""
    ENDPOINT = "endpoint"
    CONTINUOUS = "continuous"
    SILENCE_SUPPRESSED = "silence_suppressed"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StreamConfig:
    # Audio
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
    CHANNELS: int = int(os.getenv("CHANNELS", "1"))
    BLOCK_SIZE: int = int(os.getenv("BLOCK_SIZE", "1024"))
    DTYPE: str = "float32"

    # Device
    AUDIO_DEVICE_INDEX: int = int(os.getenv("AUDIO_DEVICE_INDEX", "-1"))

    # Ring buffer
    BUFFER_MAX_SECONDS: float = float(os.getenv("BUFFER_MAX_SECONDS", "30.0"))
    MIN_AUDIO_SECONDS: float = 0.5

    # VAD
    VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.01"))
    VAD_SMOOTHING: float = 0.3

    # Endpointing
    SILENCE_TIMEOUT: float = float(os.getenv("SILENCE_TIMEOUT", "2.0"))
    MIN_SPEECH_DURATION: float = 0.3

    # Continuous mode
    CONTINUOUS_CHUNK_SECONDS: float = 5.0

    # Silence suppression
    SILENCE_THRESHOLD: float = 0.005
    MIN_SEGMENT_DURATION: float = 0.1

    # Preprocessing
    PREPROCESSING_ENABLED: bool = os.getenv("PREPROCESSING_ENABLED", "true").lower() == "true"

    # Performance
    LATENCY: str = "low"

    # File streaming
    FILE_STREAM_REALTIME_FACTOR: float = 1.0


StreamConfig.CHANNELS = max(1, StreamConfig.CHANNELS)


# ============================================================================
# Multi-Channel Audio Processing
# ============================================================================

StereoOut = Tuple[np.ndarray, np.ndarray]
AudioOut = Union[np.ndarray, StereoOut]

class MultiChannelProcessor:
    def __init__(self, mode: ChannelMode = ChannelMode.MONO, sample_rate: int = StreamConfig.SAMPLE_RATE):
        self.mode = mode
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray) -> AudioOut:
        # audio: (samples,) or (samples, channels)

        if audio.ndim == 1:
            if self.mode == ChannelMode.STEREO:
                return audio, audio
            return audio

        # Multi-channel
        if self.mode in (ChannelMode.MONO, ChannelMode.MIX):
            return np.mean(audio, axis=1, dtype=np.float32)

        if self.mode == ChannelMode.LEFT_ONLY:
            return audio[:, 0]

        if self.mode == ChannelMode.RIGHT_ONLY:
            return audio[:, min(1, audio.shape[1] - 1)]

        if self.mode == ChannelMode.STEREO:
            left = audio[:, 0]
            right = audio[:, min(1, audio.shape[1] - 1)]
            return left, right

        raise ValueError(f"Unknown channel mode: {self.mode}")


# ============================================================================
# Silence Suppression
# ============================================================================

class SilenceSuppressor:
    def __init__(
        self,
        threshold: float = StreamConfig.SILENCE_THRESHOLD,
        min_segment_duration: float = StreamConfig.MIN_SEGMENT_DURATION,
        sample_rate: int = StreamConfig.SAMPLE_RATE,
    ):
        self.threshold = threshold
        self.min_segment_duration = min_segment_duration
        self.sample_rate = sample_rate
        self.min_segment_samples = int(min_segment_duration * sample_rate)

    def is_silence(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return True
        energy = float(np.sqrt(np.mean(audio ** 2)))
        return energy < self.threshold

    def suppress(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio

        frame_size = 512
        frames = []

        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            if not self.is_silence(frame):
                frames.append(frame)

        if not frames:
            return np.array([], dtype=audio.dtype)

        result = np.concatenate(frames)
        if len(result) < self.min_segment_samples:
            return np.array([], dtype=audio.dtype)

        return result


# ============================================================================
# Ring Buffer
# ============================================================================

class RingBuffer:
    """Thread-safe fixed-size ring buffer for audio frames."""
    def __init__(self, maxlen: int):
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self.buffer = collections.deque(maxlen=maxlen)
        self.lock = threading.RLock()
        self.total_frames_added = 0
        self.total_frames_dropped = 0

    def append(self, data: np.ndarray):
        with self.lock:
            if len(self.buffer) >= self.buffer.maxlen:
                self.total_frames_dropped += 1
            self.buffer.append(data.copy())
            self.total_frames_added += 1

    def get_all(self) -> np.ndarray:
        with self.lock:
            if not self.buffer:
                return np.array([], dtype=np.float32)
            try:
                return np.concatenate(list(self.buffer))
            except ValueError as e:
                log.error(f"Concatenation error: {e}")
                return np.array([], dtype=np.float32)

    def get_duration(self, sample_rate: int) -> float:
        with self.lock:
            if not self.buffer:
                return 0.0
            total_samples = sum(len(frame) for frame in self.buffer)
            return total_samples / sample_rate

    def clear(self):
        with self.lock:
            self.buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "current_frames": len(self.buffer),
                "max_frames": self.buffer.maxlen,
                "total_added": self.total_frames_added,
                "total_dropped": self.total_frames_dropped,
                "drop_rate": (
                    self.total_frames_dropped / self.total_frames_added
                    if self.total_frames_added > 0 else 0.0
                ),
            }


# ============================================================================
# VAD
# ============================================================================

class SimpleVAD:
    """Simple energy-based Voice Activity Detection."""
    def __init__(self, threshold: float = StreamConfig.VAD_THRESHOLD, smoothing: float = StreamConfig.VAD_SMOOTHING):
        self.threshold = threshold
        self.smoothing = smoothing
        self.current_energy = 0.0
        self.lock = threading.Lock()

    def is_speech(self, frame: np.ndarray) -> bool:
        if frame.size == 0:
            return False

        energy = float(np.sqrt(np.mean(frame ** 2)))

        with self.lock:
            self.current_energy = (self.smoothing * energy) + ((1 - self.smoothing) * self.current_energy)
            return self.current_energy > self.threshold

    def get_energy(self) -> float:
        with self.lock:
            return self.current_energy

    def reset(self):
        with self.lock:
            self.current_energy = 0.0


# ============================================================================
# Metrics
# ============================================================================

@dataclass
class StreamMetrics:
    chunks_processed: int = 0
    chunks_dropped: int = 0
    total_audio_seconds: float = 0.0
    total_speech_seconds: float = 0.0
    total_silence_seconds: float = 0.0
    endpoints_detected: int = 0
    errors: int = 0
    start_time: float = 0.0

    left_channel_rms: float = 0.0
    right_channel_rms: float = 0.0

    def get_uptime(self) -> float:
        return time.time() - self.start_time if self.start_time > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks_processed": self.chunks_processed,
            "chunks_dropped": self.chunks_dropped,
            "total_audio_seconds": round(self.total_audio_seconds, 2),
            "total_speech_seconds": round(self.total_speech_seconds, 2),
            "total_silence_seconds": round(self.total_silence_seconds, 2),
            "endpoints_detected": self.endpoints_detected,
            "errors": self.errors,
            "uptime_seconds": round(self.get_uptime(), 2),
            "speech_ratio": (
                self.total_speech_seconds / self.total_audio_seconds
                if self.total_audio_seconds > 0 else 0.0
            ),
            "left_channel_rms": round(self.left_channel_rms, 4),
            "right_channel_rms": round(self.right_channel_rms, 4),
        }


# ============================================================================
# Enhanced Audio Streamer (Live)
# ============================================================================

class AudioStreamer:
    def __init__(
        self,
        callback: Callable[[np.ndarray], None],
        sample_rate: int = StreamConfig.SAMPLE_RATE,
        channels: int = StreamConfig.CHANNELS,
        block_size: int = StreamConfig.BLOCK_SIZE,
        device_index: Optional[int] = None,
        buffer_max_seconds: float = StreamConfig.BUFFER_MAX_SECONDS,
        stream_mode: StreamMode = StreamMode.ENDPOINT,
        channel_mode: ChannelMode = ChannelMode.MONO,
        silence_timeout: float = StreamConfig.SILENCE_TIMEOUT,
        continuous_chunk_seconds: float = StreamConfig.CONTINUOUS_CHUNK_SECONDS,
        vad_threshold: float = StreamConfig.VAD_THRESHOLD,
        silence_suppression: bool = False,
        preprocessing_enabled: bool = StreamConfig.PREPROCESSING_ENABLED,
    ):
        self.callback = callback
        self.sample_rate = sample_rate
        self.channels = max(1, channels)
        self.block_size = block_size
        self.stream_mode = stream_mode
        self.silence_timeout = silence_timeout
        self.continuous_chunk_seconds = continuous_chunk_seconds

        self.running = False
        self.stream: Optional[sd.InputStream] = None
        self.thread: Optional[threading.Thread] = None

        self.channel_processor = MultiChannelProcessor(mode=channel_mode, sample_rate=sample_rate)

        max_frames = int(buffer_max_seconds * sample_rate / block_size)
        self.ring_buffer = RingBuffer(maxlen=max_frames)

        self.vad = SimpleVAD(threshold=vad_threshold)
        self.silence_timer = 0.0
        self.in_speech = False

        self.silence_suppressor: Optional[SilenceSuppressor] = SilenceSuppressor(sample_rate=sample_rate) if silence_suppression else None

        self.continuous_buffer_samples = 0
        self.continuous_chunk_samples = int(continuous_chunk_seconds * sample_rate)

        self.preprocessor: Optional[Preprocessor] = None
        if preprocessing_enabled:
            try:
                self.preprocessor = Preprocessor(sample_rate=sample_rate)
                log.info("Preprocessing enabled")
            except Exception as e:
                log.warning(f"Failed to initialize preprocessor: {e}")

        self.metrics = StreamMetrics()

        if device_index is not None:
            validate_device(device_index, input=True, channels=self.channels, samplerate=sample_rate)
            self.device_index = device_index
            log.info(f"Using device {device_index}")
        else:
            self.device_index, _ = _pick_input_device(StreamConfig.AUDIO_DEVICE_INDEX)
            log.info(f"Auto-selected device {self.device_index}")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning(f"Audio callback status: {status}")
        if indata is None or frames <= 0:
            return

        try:
            audio = indata.astype(np.float32)

            # channel processing
            if self.channel_processor.mode == ChannelMode.STEREO:
                left, right = self.channel_processor.process(audio)  # tuple
                self.metrics.left_channel_rms = float(np.sqrt(np.mean(left ** 2)))
                self.metrics.right_channel_rms = float(np.sqrt(np.mean(right ** 2)))
                audio_mono = left  # choose one for ASR
            else:
                audio_mono = self.channel_processor.process(audio)  # mono array

            if self.preprocessor:
                try:
                    audio_mono = self.preprocessor.process(audio_mono)
                except Exception as e:
                    log.warning(f"Preprocessing failed: {e}")

            if self.stream_mode == StreamMode.ENDPOINT:
                self._handle_endpoint_mode(audio_mono, frames)
            elif self.stream_mode == StreamMode.CONTINUOUS:
                self._handle_continuous_mode(audio_mono, frames)
            elif self.stream_mode == StreamMode.SILENCE_SUPPRESSED:
                self._handle_silence_suppressed_mode(audio_mono, frames)

        except Exception as e:
            log.error(f"Audio callback error: {e}", exc_info=True)
            self.metrics.errors += 1

    def _handle_endpoint_mode(self, audio: np.ndarray, frames: int):
        is_speech_detected = self.vad.is_speech(audio)
        frame_duration = frames / self.sample_rate

        self.metrics.chunks_processed += 1
        self.metrics.total_audio_seconds += frame_duration
        if is_speech_detected:
            self.metrics.total_speech_seconds += frame_duration
        else:
            self.metrics.total_silence_seconds += frame_duration

        self.ring_buffer.append(audio)

        if is_speech_detected:
            self.silence_timer = 0.0
            self.in_speech = True
        else:
            if self.in_speech:
                self.silence_timer += frame_duration

        if self.in_speech and self.silence_timer >= self.silence_timeout:
            self._trigger_callback()

    def _handle_continuous_mode(self, audio: np.ndarray, frames: int):
        frame_duration = frames / self.sample_rate
        self.metrics.chunks_processed += 1
        self.metrics.total_audio_seconds += frame_duration

        self.ring_buffer.append(audio)
        self.continuous_buffer_samples += len(audio)

        if self.continuous_buffer_samples >= self.continuous_chunk_samples:
            chunk = self.ring_buffer.get_all()[:self.continuous_chunk_samples]
            if chunk.size > 0:
                self._process_chunk(chunk)
            self.ring_buffer.clear()
            self.continuous_buffer_samples = 0

    def _handle_silence_suppressed_mode(self, audio: np.ndarray, frames: int):
        frame_duration = frames / self.sample_rate
        self.metrics.chunks_processed += 1
        self.metrics.total_audio_seconds += frame_duration

        is_silence = self.silence_suppressor.is_silence(audio) if self.silence_suppressor else (not self.vad.is_speech(audio))

        if is_silence:
            self.metrics.total_silence_seconds += frame_duration
            return

        self.metrics.total_speech_seconds += frame_duration
        self.ring_buffer.append(audio)

        self.in_speech = True
        self.silence_timer = 0.0

        if self.ring_buffer.get_duration(self.sample_rate) >= self.continuous_chunk_seconds:
            self._trigger_callback()

    def _trigger_callback(self):
        chunk = self.ring_buffer.get_all()
        duration = len(chunk) / self.sample_rate

        if duration < StreamConfig.MIN_SPEECH_DURATION:
            self.ring_buffer.clear()
            self.silence_timer = 0.0
            self.in_speech = False
            return

        self._process_chunk(chunk)

        self.ring_buffer.clear()
        self.silence_timer = 0.0
        self.in_speech = False
        self.vad.reset()

    def _process_chunk(self, chunk: np.ndarray):
        try:
            if self.silence_suppressor and self.stream_mode != StreamMode.SILENCE_SUPPRESSED:
                chunk = self.silence_suppressor.suppress(chunk)
                if chunk.size == 0:
                    return

            self.metrics.endpoints_detected += 1
            self.callback(chunk)

        except Exception as e:
            log.error(f"Chunk processing/callback error: {e}", exc_info=True)
            self.metrics.errors += 1

    def _run_stream(self):
        try:
            log.info(f"Starting audio stream (mode={self.stream_mode.value}, channels={self.channels})...")
            with sd.InputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                dtype=StreamConfig.DTYPE,
                callback=self._audio_callback,
                latency=StreamConfig.LATENCY,
            ) as stream:
                self.stream = stream
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            log.error(f"Audio stream error: {e}", exc_info=True)
            self.metrics.errors += 1
        finally:
            self.stream = None
            log.info("Audio stream stopped")

    def start(self):
        if self.running:
            log.warning("Streamer already running")
            return

        self.running = True
        self.ring_buffer.clear()
        self.silence_timer = 0.0
        self.in_speech = False
        self.continuous_buffer_samples = 0
        self.vad.reset()
        self.metrics = StreamMetrics(start_time=time.time())

        self.thread = threading.Thread(target=self._run_stream, daemon=True, name="AudioStreamThread")
        self.thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        # flush final speech in endpoint mode
        if self.stream_mode == StreamMode.ENDPOINT and self.in_speech:
            chunk = self.ring_buffer.get_all()
            duration = len(chunk) / self.sample_rate
            if duration >= StreamConfig.MIN_SPEECH_DURATION:
                self._process_chunk(chunk)

        log.info("AudioStreamer stopped")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self.metrics.to_dict(),
            "buffer_stats": self.ring_buffer.get_stats(),
            "vad_energy": self.vad.get_energy(),
            "in_speech": self.in_speech,
            "silence_duration": self.silence_timer,
            "stream_mode": self.stream_mode.value,
            "channel_mode": self.channel_processor.mode.value,
        }

    def is_running(self) -> bool:
        return self.running


# ============================================================================
# File Streamer
# ============================================================================

class FileStreamer:
    def __init__(
        self,
        filename: Union[str, Path],
        callback: Callable[[np.ndarray], None],
        realtime_factor: float = StreamConfig.FILE_STREAM_REALTIME_FACTOR,
        chunk_seconds: float = 1.0,
        stream_mode: StreamMode = StreamMode.CONTINUOUS,
        channel_mode: ChannelMode = ChannelMode.MONO,
        silence_suppression: bool = False,
        preprocessing_enabled: bool = False,
    ):
        self.filename = Path(filename)
        self.callback = callback
        self.realtime_factor = max(0.1, realtime_factor)
        self.chunk_seconds = max(0.01, chunk_seconds)
        self.stream_mode = stream_mode

        if not self.filename.exists():
            raise FileNotFoundError(f"Audio file not found: {self.filename}")

        try:
            self.audio, self.sample_rate = sf.read(str(self.filename), dtype="float32")
            log.info(f"Loaded {self.filename}: {len(self.audio)/self.sample_rate:.1f}s @ {self.sample_rate}Hz")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")

        self.channel_processor = MultiChannelProcessor(mode=channel_mode, sample_rate=self.sample_rate)

        # if not stereo mode, force mono array output
        if self.channel_processor.mode != ChannelMode.STEREO:
            self.audio = self.channel_processor.process(self.audio)  # mono np.ndarray
        else:
            # if stereo mode but file is mono, duplicate
            if self.audio.ndim == 1:
                self.audio = np.stack([self.audio, self.audio], axis=1)

        self.silence_suppressor: Optional[SilenceSuppressor] = SilenceSuppressor(sample_rate=self.sample_rate) if silence_suppression else None

        self.preprocessor: Optional[Preprocessor] = None
        if preprocessing_enabled:
            try:
                self.preprocessor = Preprocessor(sample_rate=self.sample_rate)
                if isinstance(self.audio, np.ndarray) and self.audio.ndim == 1:
                    self.audio = self.preprocessor.process(self.audio)
                log.info("File preprocessed")
            except Exception as e:
                log.warning(f"Preprocessing failed: {e}")

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.metrics = StreamMetrics()
        self.chunk_samples = int(self.chunk_seconds * self.sample_rate)

    def _stream_loop(self):
        try:
            position = 0
            chunk_count = 0

            # file audio expected mono array for ASR path
            audio_mono = self.audio
            if isinstance(audio_mono, tuple):
                audio_mono = audio_mono[0]

            while self.running and position < len(audio_mono):
                t0 = time.time()

                end_position = min(position + self.chunk_samples, len(audio_mono))
                chunk = audio_mono[position:end_position].copy()

                if self.silence_suppressor and self.stream_mode == StreamMode.SILENCE_SUPPRESSED:
                    chunk = self.silence_suppressor.suppress(chunk)
                    if chunk.size == 0:
                        position = end_position
                        continue

                chunk_duration = len(chunk) / self.sample_rate
                self.metrics.chunks_processed += 1
                self.metrics.total_audio_seconds += chunk_duration
                chunk_count += 1

                try:
                    self.callback(chunk)
                except Exception as e:
                    log.error(f"Callback error: {e}", exc_info=True)
                    self.metrics.errors += 1

                position = end_position

                elapsed = time.time() - t0
                sleep_time = (chunk_duration / self.realtime_factor) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            log.info(f"File streaming complete: {chunk_count} chunks")

        except Exception as e:
            log.error(f"Stream loop error: {e}", exc_info=True)
            self.metrics.errors += 1

    def start(self):
        if self.running:
            log.warning("File streamer already running")
            return

        self.running = True
        self.metrics = StreamMetrics(start_time=time.time())

        self.thread = threading.Thread(target=self._stream_loop, daemon=True, name="FileStreamThread")
        self.thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def wait_until_done(self, timeout: Optional[float] = None):
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def get_metrics(self) -> Dict[str, Any]:
        file_duration = (len(self.audio) / self.sample_rate) if isinstance(self.audio, np.ndarray) else 0.0
        return {
            **self.metrics.to_dict(),
            "filename": str(self.filename),
            "file_duration": file_duration,
            "realtime_factor": self.realtime_factor,
            "progress": (self.metrics.total_audio_seconds / file_duration) if file_duration > 0 else 0.0,
        }

    def is_running(self) -> bool:
        return self.running


# ============================================================================
# Convenience Functions
# ============================================================================

def stream_to_console(
    duration: Optional[float] = None,
    device_index: Optional[int] = None,
    mode: StreamMode = StreamMode.ENDPOINT,
    channels: int = 1,
    silence_suppression: bool = False,
):
    def on_transcript(audio_chunk: np.ndarray):
        result = transcribe(audio_chunk)
        timestamp = time.strftime("%H:%M:%S")
        d = len(audio_chunk) / StreamConfig.SAMPLE_RATE
        print(f"[{timestamp}] ({d:.1f}s) {result.text}")

    streamer = AudioStreamer(
        callback=on_transcript,
        device_index=device_index,
        stream_mode=mode,
        channels=channels,
        silence_suppression=silence_suppression,
    )

    try:
        streamer.start()
        if duration:
            print(f"Streaming for {duration}s ({mode.value})... (Ctrl+C to stop)")
            time.sleep(duration)
        else:
            print(f"Streaming ({mode.value})... (Ctrl+C to stop)")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        streamer.stop()
        m = streamer.get_metrics()
        print("\n=== Streaming Metrics ===")
        print(f"Uptime: {m['uptime_seconds']:.1f}s")
        print(f"Audio processed: {m['total_audio_seconds']:.1f}s")
        print(f"Speech: {m['total_speech_seconds']:.1f}s")
        print(f"Silence: {m['total_silence_seconds']:.1f}s")
        print(f"Speech ratio: {m['speech_ratio']:.1%}")
        print(f"Endpoints: {m['endpoints_detected']}")
        print(f"Errors: {m['errors']}")


def stream_file_to_console(
    filename: str,
    realtime_factor: float = 1.0,
    silence_suppression: bool = False,
):
    def on_transcript(audio_chunk: np.ndarray):
        result = transcribe(audio_chunk)
        timestamp = time.strftime("%H:%M:%S")
        d = len(audio_chunk) / StreamConfig.SAMPLE_RATE
        print(f"[{timestamp}] ({d:.1f}s) {result.text}")

    streamer = FileStreamer(
        filename=filename,
        callback=on_transcript,
        realtime_factor=realtime_factor,
        silence_suppression=silence_suppression,
    )

    try:
        streamer.start()
        print(f"Streaming file: {filename} ({realtime_factor}x speed)")
        streamer.wait_until_done()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        streamer.stop()
        m = streamer.get_metrics()
        print("\n=== File Streaming Metrics ===")
        print(f"File: {m['filename']}")
        print(f"Duration: {m['file_duration']:.1f}s")
        print(f"Processed: {m['total_audio_seconds']:.1f}s")
        print(f"Progress: {m['progress']:.1%}")
        print(f"Chunks: {m['chunks_processed']}")
        print(f"Errors: {m['errors']}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    # If first arg looks like a file, do file streaming
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith((".wav", ".flac", ".mp3")):
        filename = sys.argv[1]
        realtime_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        silence_suppression = (sys.argv[3].lower() == "true") if len(sys.argv) > 3 else False
        stream_file_to_console(filename, realtime_factor, silence_suppression)
    else:
        duration = float(sys.argv[1]) if len(sys.argv) > 1 else None
        device = int(sys.argv[2]) if len(sys.argv) > 2 else None
        mode_str = sys.argv[3] if len(sys.argv) > 3 else "endpoint"
        mode = {
            "endpoint": StreamMode.ENDPOINT,
            "continuous": StreamMode.CONTINUOUS,
            "silence": StreamMode.SILENCE_SUPPRESSED,
        }.get(mode_str, StreamMode.ENDPOINT)

        stream_to_console(duration=duration, device_index=device, mode=mode)
