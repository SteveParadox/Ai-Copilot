import asyncio
import json
import os
import uuid
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from pipeline import pipeline, cache_get

# ===================== CONFIG =====================
SERVICE_NAME = os.getenv("SERVICE_NAME", "NLP Service")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0")

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


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
async def answer_stream(req: AnswerRequest, x_request_id: Optional[str] = Header(None)):
    """ Streaming NLP answer endpoint """
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

    # ---- Run pipeline ----
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Kick off background streaming
    asyncio.create_task(
        pipeline.run_stream(prompt, req.style, req.company, queue, loop, cache_key)
    )

    async def event_generator():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break

                # Error event
                if isinstance(item, dict) and item.get("error"):
                    yield f"event: error\ndata: {json.dumps({'message': item['error']})}\n\n"
                    print(f"[NLP:{request_id}] Error: {item['error']}")
                    break

                # Normal streaming delta
                yield f"data: {json.dumps({'delta': item['delta']})}\n\n"

            yield f"event: done\ndata: {json.dumps({'status': 'done'})}\n\n"
            print(f"[NLP:{request_id}] Completed stream")
        except Exception as e:
            print(f"[NLP:{request_id}] Exception in generator: {e}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")
