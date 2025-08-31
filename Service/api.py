# api.py
import asyncio
import os
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from Asr.utils import majority_vote
from Asr.model import load_whisper, transcribe
from Asr._preprocessing import clean_audio, prep_audio_float32

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
    return transcribe(audio_array)


# ---------------- REST ENDPOINT ----------------
@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            return JSONResponse({"error": "Empty audio file"}, status_code=400)

        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array = prep_audio_float32(clean_audio(audio_array))
        transcript = await asyncio.to_thread(transcribe_audio, audio_array)

        if transcript:
            transcript = majority_vote([transcript])

        return {"transcript": transcript}

    except Exception as e:
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
    results = []
    buffer = b""

    try:
        while True:
            data = await websocket.receive_bytes()
            buffer += data

            # Process ~1 second chunks (16kHz int16)
            chunk_size = 16000 * 2
            while len(buffer) >= chunk_size:
                audio_array = np.frombuffer(buffer[:chunk_size], dtype=np.int16)
                audio_array = prep_audio_float32(clean_audio(audio_array))

                try:
                    transcript = await asyncio.to_thread(transcribe_audio, audio_array)
                    if transcript.strip():
                        results.append(transcript)
                        stable = majority_vote(results[-3:])
                        await manager.send_text(websocket, stable)
                        print("Transcript:", stable, flush=True)
                except Exception as e:
                    await manager.send_text(websocket, f"ERROR: {str(e)}")

                # Remove processed bytes
                buffer = buffer[chunk_size:]

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_text(websocket, f"ERROR: {str(e)}")
