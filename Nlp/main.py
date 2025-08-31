import asyncio
import json
import os
import uuid
from typing import Optional, List

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from pipeline import pipeline, cache_get, cache_set, client


# ===================== CONFIG =====================
SERVICE_NAME = os.getenv("SERVICE_NAME", "NLP Service")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0")

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== WS CONNECTION MANAGER =====================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WS] Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"[WS] Broadcast error: {e}")
                to_remove.append(connection)
        # cleanup dead connections
        for conn in to_remove:
            self.disconnect(conn)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # keep connection alive (ignore client messages)
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=60)
            except asyncio.TimeoutError:
                # send ping to keep alive
                await websocket.send_text("[WS] heartbeat")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ===================== MODELS =====================
class AnswerRequest(BaseModel):
    question: str
    style: Optional[str] = "Concise"
    company: Optional[str] = ""


# ===================== ROUTES =====================
@app.get("/health")
async def health():
    """ Health check endpoint """
    return {"status": "ok", "model": pipeline.model, "service": SERVICE_NAME}


@app.post("/answer")
async def answer_json(req: AnswerRequest, x_request_id: Optional[str] = Header(None)):
    """ Non-streaming NLP answer endpoint (returns pure JSON) """
    request_id = x_request_id or str(uuid.uuid4())
    prompt = req.question.strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="question empty")

    print(f"[NLP:{request_id}] Incoming request → Q='{prompt}' | style={req.style} | company={req.company}")

    # ---- Check cache ----
    cache_key = f"{prompt}|{req.style}|{req.company}"
    cached = cache_get(cache_key)
    if cached:
        print(f"[NLP:{request_id}] Returning cached response")
        return JSONResponse({"cached": True, "answer": cached})

    # ---- Run pipeline in non-streaming mode ----
    try:
        messages = [
            {"role": "system", "content": f"You are an AI assistant for interviews. Provide answers in a {req.style} manner."},
            {"role": "user", "content": prompt + (f" The company is {req.company}." if req.company else "")},
        ]

        resp = client.chat.completions.create(
            model=pipeline.model,
            messages=messages,
            temperature=0.4,
            max_tokens=800,
        )

        answer = resp.choices[0].message.content.strip()
        cache_set(cache_key, answer)

        # Broadcast to WS clients
        await manager.broadcast(answer)
        print(f"[NLP:{request_id}] Completed JSON answer")

        return JSONResponse({"answer": answer, "cached": False})

    except Exception as e:
        print(f"[NLP:{request_id}] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
