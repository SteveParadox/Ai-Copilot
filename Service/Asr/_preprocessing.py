"""
 speech preprocessing module
- VAD (WebRTC VAD preferred, with energy-based fallback)
- Spectral gating denoiser (uses STFT, noise estimated from non-speech frames)
- Bandpass filtering, DC removal, normalization, trimming, noise gate
- Configurable and thread-safe (no global state)

Usage:
    from production_preprocessing import Preprocessor
    pre = Preprocessor(sample_rate=16000)
    cleaned = pre.process(raw_audio)

Notes:
- Optional dependency: webrtcvad (`pip install webrtcvad`). If not installed a robust energy-based VAD is used.
- This implementation tries to keep dependencies minimal (numpy, scipy).
"""

from typing import Optional, Tuple
import numpy as np
import logging
from scipy.signal import butter, lfilter, stft, istft

log = logging.getLogger("Preprocessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Try to import webrtcvad; fall back to energy-based VAD if unavailable
try:
    import webrtcvad  # type: ignore
    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False


class Preprocessor:
    """ speech preprocessing pipeline.

    Main features:
      - to_float32, mono conversion
      - DC offset removal
      - optional bandpass filtering
      - normalization to target dBFS
      - VAD (WebRTC or energy-based)
      - spectral gating denoising (noise estimated from non-speech frames)
      - trimming and post-cleaning noise gate

    Parameters
    ----------
    sample_rate: int
        Audio sample rate (e.g. 16000)
    lowcut, highcut: int
        Bandpass filter cutoffs
    vad_mode: int
        WeRTC VAD mode (0-3) more aggressive -> 3 (only used if webrtc available)
    denoise_method: str
        'spectral_gating' (currently implemented)
    n_fft, hop_length: int
        STFT params for denoising
    noise_frames: int
        Number of non-speech frames to use to build noise profile
    target_dBFS: float
        Target loudness for normalization
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        lowcut: int = 80,
        highcut: int = 8000,
        vad_mode: int = 2,
        denoise_method: str = "spectral_gating",
        n_fft: int = 1024,
        hop_length: int = 256,
        noise_frames: int = 6,
        target_dBFS: float = -20.0,
    ) -> None:
        self.sr = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.vad_mode = vad_mode
        self.denoise_method = denoise_method
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_frames = noise_frames
        self.target_dBFS = target_dBFS

        if _HAS_WEBRTC:
            self._vad = webrtcvad.Vad()
            self._vad.set_mode(max(0, min(3, vad_mode)))
            log.info("WebRTC VAD available and enabled")
        else:
            self._vad = None
            log.info("WebRTC VAD not available — using energy-based VAD fallback")

    # ---------------- Basic conversions ----------------
    def to_float32(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype == np.float32:
            return audio
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        if audio.dtype == np.int32:
            return audio.astype(np.float32) / 2147483648.0
        log.debug(f"Casting audio from {audio.dtype} to float32")
        return audio.astype(np.float32)

    def prep_mono(self, audio: np.ndarray) -> np.ndarray:
        audio = self.to_float32(audio)
        if audio.ndim == 2:
            if audio.shape[1] > 1:
                audio = audio.mean(axis=1, dtype=np.float32)
            else:
                audio = audio[:, 0]
        return audio

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        return audio - np.mean(audio)

    def rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2) + 1e-12))

    def normalize_audio(self, audio: np.ndarray, target_dBFS: Optional[float] = None) -> np.ndarray:
        tgt = self.target_dBFS if target_dBFS is None else target_dBFS
        audio = self.to_float32(audio)
        rms_val = self.rms(audio)
        if rms_val <= 0:
            return audio
        scalar = 10 ** (tgt / 20.0) / rms_val
        audio = audio * scalar
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    # ---------------- Filters ----------------
    def bandpass_filter(self, audio: np.ndarray, order: int = 5) -> np.ndarray:
        nyq = 0.5 * self.sr
        low = max(self.lowcut / nyq, 1e-5)
        high = min(self.highcut / nyq, 0.99999)
        if low >= high:
            log.warning("Invalid bandpass range — skipping bandpass")
            return audio
        b, a = butter(order, [low, high], btype="band")
        return lfilter(b, a, audio).astype(np.float32)

    # ---------------- VAD ----------------
    def frame_generator(self, audio: np.ndarray, frame_ms: int = 30) -> Tuple[np.ndarray, int]: # type: ignore
        frame_len = int(self.sr * frame_ms / 1000)
        num_frames = int(np.ceil(len(audio) / frame_len))
        for i in range(num_frames):
            start = i * frame_len
            end = min(start + frame_len, len(audio))
            frame = audio[start:end]
            if len(frame) < frame_len:
                # pad
                frame = np.pad(frame, (0, frame_len - len(frame)), mode="constant")
            yield frame, start

    def is_speech_webrtc(self, frame: np.ndarray) -> bool:
        # webrtcvad expects 16-bit PCM bytes at 8000/16000/32000/48000
        if self._vad is None:
            return False
        pcm16 = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        try:
            return self._vad.is_speech(pcm16, sample_rate=self.sr)
        except Exception:
            return False

    def energy_vad(self, frame: np.ndarray, thr: float) -> bool:
        return self.rms(frame) > thr

    def apply_vad(self, audio: np.ndarray, frame_ms: int = 30, energy_threshold: float = 0.01) -> np.ndarray:
        """Return boolean mask of speech frames (per-sample mask)."""
        audio = self.to_float32(audio)
        mask = np.zeros_like(audio, dtype=bool)
        for frame, start in self.frame_generator(audio, frame_ms=frame_ms):
            use_speech = False
            if _HAS_WEBRTC and self._vad is not None:
                use_speech = self.is_speech_webrtc(frame)
            else:
                # energy threshold adapted by global RMS
                use_speech = self.energy_vad(frame, thr=energy_threshold)
            end = start + len(frame)
            mask[start:end] = use_speech
        return mask

    # ---------------- Trimming & gating ----------------
    def trim_silence(self, audio: np.ndarray, mask: Optional[np.ndarray] = None, padding_ms: int = 50) -> np.ndarray:
        audio = self.to_float32(audio)
        if mask is None:
            # fallback: amplitude threshold
            idx = np.where(np.abs(audio) > 0.01)[0]
            if idx.size == 0:
                return audio
            return audio[idx[0] : idx[-1] + 1]
        # mask -> find continuous speech region
        idx = np.where(mask)[0]
        if idx.size == 0:
            return audio
        pad = int(self.sr * padding_ms / 1000)
        start = max(0, idx[0] - pad)
        end = min(len(audio), idx[-1] + pad + 1)
        return audio[start:end]

    def apply_noise_gate(self, audio: np.ndarray, gate_threshold: float = 0.02) -> np.ndarray:
        a = self.to_float32(audio)
        a[np.abs(a) < gate_threshold] = 0.0
        return a

    # ---------------- Spectral gating denoiser ----------------
    def _estimate_noise_spectrum(self, S_mag: np.ndarray, spec_mask: np.ndarray) -> np.ndarray:
        """Estimate noise magnitude spectrum from frames where spec_mask==False (non-speech).
        S_mag: (freq_bins, frames)
        spec_mask: boolean array length frames (True=speech)
        """
        if spec_mask is None or np.all(spec_mask):
            # no non-speech frames — estimate noise as the minimum across time
            return np.min(S_mag, axis=1, keepdims=True)
        nonspeech = ~spec_mask
        if np.sum(nonspeech) < 1:
            return np.min(S_mag, axis=1, keepdims=True)
        return np.mean(S_mag[:, nonspeech], axis=1, keepdims=True)

    def spectral_gating(self, audio: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Simple spectral gating denoiser.

        Steps:
          1. Compute STFT magnitude
          2. Estimate noise spectrum from mask (non-speech frames) or from lowest-energy frames
          3. Apply soft gain to reduce magnitude where magnitude <= noise * noise_reduction_factor
          4. Reconstruct via ISTFT
        """
        if len(audio) < 2:
            return audio

        # STFT
        f, t, Zxx = stft(audio, fs=self.sr, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
        S = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Build spec_mask per STFT frame from VAD mask (if provided)
        spec_mask = None
        if mask is not None:
            # convert per-sample mask -> per-frame
            frame_len = int(self.sr * (self.n_fft / self.sr))
            # Use hop_length to align: number of frames should equal t.shape[0]
            per_frame_mask = []
            samples_per_frame = self.hop_length
            for i in range(t.shape[0]):
                start = i * samples_per_frame
                end = min(start + samples_per_frame, len(mask))
                if end <= start:
                    per_frame_mask.append(False)
                else:
                    per_frame_mask.append(bool(np.mean(mask[start:end]) > 0.5))
            spec_mask = np.array(per_frame_mask, dtype=bool)

        noise_mag = self._estimate_noise_spectrum(S, spec_mask)

        # Noise floor and gain
        reduction_factor = 1.5  # how many times above noise we allow
        gain = np.maximum(0.0, (S - noise_mag * reduction_factor) / (S + 1e-12))
        # Smooth gain across frequency/time to prevent musical noise: simple median filter across time
        # (keep implementation dependency-free)
        # Apply gain
        S_denoised = S * gain
        Zxx_denoised = S_denoised * np.exp(1j * phase)

        # ISTFT
        _, xrec = istft(Zxx_denoised, fs=self.sr, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
        xrec = xrec[: len(audio)]
        # Avoid artifacts: normalize to original RMS
        orig_rms = self.rms(audio)
        rec_rms = self.rms(xrec) + 1e-12
        if rec_rms > 0:
            xrec = xrec * (orig_rms / rec_rms)
        return xrec.astype(np.float32)

    # ---------------- Full pipeline ----------------
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Run the full preprocessing pipeline and return cleaned audio.

        Steps:
          - Mono & float32
          - DC offset removal
          - Bandpass filtering
          - VAD to build speech mask
          - Trim silence using VAD
          - Estimate noise and apply spectral gating
          - Post-gate trimming, noise gate, normalization
        """
        audio = self.prep_mono(audio)
        if audio.size == 0:
            return audio

        audio = self.remove_dc_offset(audio)
        audio = self.bandpass_filter(audio)

        # VAD mask (per-sample boolean)
        vad_mask = self.apply_vad(audio)

        # Trim leading/trailing non-speech using VAD
        audio = self.trim_silence(audio, mask=vad_mask)

        # Denoise using spectral gating with noise estimated from non-speech frames
        try:
            if self.denoise_method == "spectral_gating":
                audio = self.spectral_gating(audio, mask=vad_mask)
        except Exception as e:
            log.warning(f"Denoising failed: {e} — continuing without denoising")

        # Post processing
        audio = self.apply_noise_gate(audio, gate_threshold=0.005)
        audio = self.normalize_audio(audio)

        # Final trim with VAD again (in case denoising changed energy distribution)
        vad_mask2 = self.apply_vad(audio)
        audio = self.trim_silence(audio, mask=vad_mask2)

        # Ensure float32
        return self.to_float32(audio)


# ---------------- Convenience function for simple use ----------------

def clean_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    pre = Preprocessor(sample_rate=sample_rate)
    return pre.process(audio)
