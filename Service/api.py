# api.py
import asyncio
import logging
import os
from typing import List
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from Asr.utils import majority_vote, weighted_majority_vote
from Asr.model import load_whisper, _transcribe, transcribe_batch, transcribe_batch_async
from Asr._preprocessing import clean_audio


log = logging.getLogger("Preprocessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


load_dotenv()
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")

app = FastAPI(title="ASR Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Initialize Model ----------------
whisper_model = load_whisper(model_size=WHISPER_MODEL_SIZE)

def transcribe_audio(audio_array: np.ndarray) -> str:
    """Wrapper to decouple model from transcription."""
    return _transcribe(audio_array)


# ---------------- REST ENDPOINT ----------------
@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse({"error": "Empty audio file"}, status_code=400)

        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array = clean_audio(audio_array)

        transcript = await asyncio.to_thread(transcribe_audio, audio_array)

        return {"transcript": transcript or ""}
    except Exception as e:
        log.error(f"Error in /transcribe: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/transcribe_batch")
async def transcribe_batch_files(files: List[UploadFile]):
    try:
        audio_arrays = []
        for file in files:
            audio_bytes = await file.read()
            if not audio_bytes:
                return JSONResponse(
                    {"error": f"Empty file: {file.filename}"}, status_code=400
                )
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_array = clean_audio(audio_array)
            audio_arrays.append(audio_array)

        transcripts = await transcribe_batch_async(audio_arrays)

        return {"results": [
            {"filename": f.filename, "transcript": t or ""}
            for f, t in zip(files, transcripts)
        ]}
    except Exception as e:
        log.error(f"Error in /transcribe_batch: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

    
# ---------------- WEBSOCKET STREAMING ----------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def send_text(self, websocket: WebSocket, message: str):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for conn in self.active_connections:
            await conn.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await manager.connect(websocket)
    history: list[str] = []
    buffer = b""

    try:
        while True:
            data = await websocket.receive_bytes()
            buffer += data

            chunk_size = 16000 * 2  # ~1 sec @ 16kHz int16
            while len(buffer) >= chunk_size:
                audio_array = np.frombuffer(buffer[:chunk_size], dtype=np.int16)
                audio_array = clean_audio(audio_array)

                try:
                    transcript = await asyncio.to_thread(transcribe_audio, audio_array)
                    if transcript.strip():
                        history.append(transcript)

                        # Keep sliding window
                        window = history[-5:]
                        stable = majority_vote(window, tie_strategy="first")

                        await manager.send_text(websocket, stable or transcript)
                        log.info(f"Transcript (stable): {stable}")
                except Exception as e:
                    log.error(f"Error during streaming: {e}", exc_info=True)
                    await manager.send_text(websocket, f"ERROR: {str(e)}")

                buffer = buffer[chunk_size:]
    except WebSocketDisconnect:
        log.info("WebSocket disconnected.")
    except Exception as e:
        log.error(f"Unhandled error in /ws/stream: {e}", exc_info=True)
        try:
            await manager.send_text(websocket, f"ERROR: {str(e)}")
        except Exception:
            pass
    finally:
        manager.disconnect(websocket)