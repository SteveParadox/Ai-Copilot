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
