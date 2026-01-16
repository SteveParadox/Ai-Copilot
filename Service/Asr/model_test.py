import pytest
from Asr.model import (
    load_whisper, 
    transcribe, 
    validate_audio,
    clean_text,
    reset_metrics
)

@pytest.fixture(scope="session")
def model():
    """Load model once for all tests."""
    load_whisper(model_size="tiny")
    yield
    unload_whisper()

def test_audio_validation():
    # Valid audio
    audio = np.random.randn(16000).astype(np.float32)
    validated = validate_audio(audio)
    assert validated.dtype == np.float32
    
    # Invalid audio
    with pytest.raises(ValueError):
        validate_audio(np.array([]))

def test_text_cleaning():
    dirty = "um like okay the the the answer is is clear"
    clean = clean_text(dirty)
    assert "um" not in clean
    assert clean.count("the") == 1
    assert clean.count("is") == 1

def test_transcription(model):
    reset_metrics()
    
    # Generate test audio
    audio = np.random.randn(16000).astype(np.float32) * 0.1
    
    result = transcribe(audio, timeout=10.0)
    
    assert isinstance(result.text, str)
    assert result.latency_ms > 0
    assert result.segments_total >= 0
    
    # Check metrics updated
    metrics = get_metrics()
    assert metrics['transcriptions_total'] == 1
