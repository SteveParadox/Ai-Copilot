import os
import uuid
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf

from model import ASRModel

app = FastAPI(title="ASR Service", version="1.0")

# Initialize ASR wrapper
asr = ASRModel()


@app.on_event("startup")
def load_asr():
    asr.load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model": asr.model_name}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe a full uploaded audio file (blocking).
    """
    try:
        # Save uploaded file to temp
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        content = await file.read()
        tmp.write(content)
        tmp.close()

        # Normalize audio
        file_path = ASRModel.normalize_audio(tmp.name)

        # Run transcription
        text = asr.transcribe(file_path)
        os.unlink(tmp.name)  # cleanup temp file

        return JSONResponse(content={"request_id": str(uuid.uuid4()), "text": text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/stream")
async def stream_transcription(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())

    # Rolling buffer (last 5 chunks)
    max_chunks = 5
    buffer = []

    try:
        while True:
            data = await ws.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16)
            buffer.append(chunk)

            # keep only last N chunks
            if len(buffer) > max_chunks:
                buffer.pop(0)

            # concat buffer → write temp wav
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio = np.concatenate(buffer, axis=0)

            # Write with detected sample rate (fallback 16000)
            sample_rate = asr.sample_rate if hasattr(asr, "sample_rate") else 16000
            sf.write(tmp.name, audio, sample_rate, subtype="PCM_16")

            file_path = ASRModel.normalize_audio(tmp.name)
            text = asr.transcribe(file_path, partial=True)
            os.unlink(tmp.name)

            await ws.send_json({
                "session_id": session_id,
                "partial_text": text
            })

    except Exception as e:
        # Proper error handling
        await ws.close(code=1011, reason=str(e))
