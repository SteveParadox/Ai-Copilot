"""
NLP Service API
FastAPI server for interview answer generation with WebSocket broadcasting.
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from pipeline import (
    pipeline,
    get_health as pipeline_health,
    get_metrics as pipeline_metrics,
    reset_metrics,
    cache_clear,
    Config,
    OpenAIError,
)

# ============================================================================
# Configuration
# ============================================================================

class ServiceConfig:
    """Service-level configuration."""
    NAME: str = os.getenv("SERVICE_NAME", "NLP Service")
    VERSION: str = os.getenv("SERVICE_VERSION", "1.0.0")
    HOST: str = os.getenv("SERVICE_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("SERVICE_PORT", "8001"))
    
    # CORS
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:3000,http://localhost:8000"
    ).split(",")
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = int(os.getenv("WS_HEARTBEAT_INTERVAL", "30"))
    WS_MAX_CONNECTIONS: int = int(os.getenv("WS_MAX_CONNECTIONS", "100"))
    
    # API
    MAX_QUESTION_LENGTH: int = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "60"))
    
    # Observability
    LOG_REQUESTS: bool = os.getenv("LOG_REQUESTS", "true").lower() == "true"
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"


# ============================================================================
# Lifecycle Management
# ============================================================================

class ServiceMetrics:
    """Track service-level metrics."""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.ws_connections = 0
        self.start_time = time.time()
    
    def get_uptime(self) -> float:
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests_total": self.request_count,
            "errors_total": self.error_count,
            "ws_connections_active": self.ws_connections,
            "uptime_seconds": self.get_uptime(),
        }


service_metrics = ServiceMetrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # Startup
    print(f"[{ServiceConfig.NAME}] Starting up...")
    print(f"  Version: {ServiceConfig.VERSION}")
    print(f"  Model: {Config.OPENAI_MODEL}")
    print(f"  Cache: {pipeline_health()['cache']['backend']}")
    
    # Validate dependencies
    health = pipeline_health()
    if not health['cache']['enabled']:
        print("  WARNING: Cache is disabled")
    
    yield
    
    # Shutdown
    print(f"[{ServiceConfig.NAME}] Shutting down...")
    print(f"  Total requests: {service_metrics.request_count}")
    print(f"  Total errors: {service_metrics.error_count}")
    await manager.disconnect_all()


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title=ServiceConfig.NAME,
    version=ServiceConfig.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ServiceConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# ============================================================================
# Request/Response Models
# ============================================================================

class AnswerRequest(BaseModel):
    """Request model for answer generation."""
    question: str = Field(..., min_length=1, max_length=ServiceConfig.MAX_QUESTION_LENGTH)
    style: str = Field(default="Concise", pattern="^(Concise|STAR|Deep Technical)$")
    company: str = Field(default="", max_length=100)
    
    @validator('question')
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or whitespace')
        return v.strip()
    
    @validator('company')
    def clean_company(cls, v):
        return v.strip()


class AnswerResponse(BaseModel):
    """Response model for answer generation."""
    answer: str
    cached: bool = False
    metadata: Optional[Dict[str, Any]] = None
    request_id: str


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str
    detail: Optional[str] = None
    request_id: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
    model: str
    cache_backend: str
    uptime_seconds: float


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections with heartbeat and cleanup."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept and register a new WebSocket connection."""
        if len(self.active_connections) >= ServiceConfig.WS_MAX_CONNECTIONS:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise HTTPException(
                status_code=503,
                detail="Maximum WebSocket connections reached"
            )
        
        await websocket.accept()
        self.active_connections[client_id] = websocket
        service_metrics.ws_connections = len(self.active_connections)
        
        print(f"[WS] Client {client_id} connected. Total: {len(self.active_connections)}")
        
        # Start heartbeat task if not running
        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        async with self._cleanup_lock:
            if client_id in self.active_connections:
                ws = self.active_connections[client_id]
                try:
                    await ws.close()
                except Exception:
                    pass
                del self.active_connections[client_id]
                service_metrics.ws_connections = len(self.active_connections)
                print(f"[WS] Client {client_id} disconnected. Total: {len(self.active_connections)}")
    
    async def disconnect_all(self):
        """Close all WebSocket connections (for shutdown)."""
        print(f"[WS] Disconnecting all {len(self.active_connections)} clients...")
        
        # Cancel heartbeat
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for client_id in list(self.active_connections.keys()):
            await self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast JSON message to all connected clients."""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message)
        dead_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message_str)
            except Exception as e:
                print(f"[WS] Broadcast error to {client_id}: {e}")
                dead_clients.append(client_id)
        
        # Clean up dead connections
        for client_id in dead_clients:
            await self.disconnect(client_id)
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client."""
        if client_id not in self.active_connections:
            return False
        
        try:
            await self.active_connections[client_id].send_text(json.dumps(message))
            return True
        except Exception as e:
            print(f"[WS] Send error to {client_id}: {e}")
            await self.disconnect(client_id)
            return False
    
    async def _heartbeat_loop(self):
        """Periodic heartbeat to detect dead connections."""
        while self.active_connections:
            try:
                await asyncio.sleep(ServiceConfig.WS_HEARTBEAT_INTERVAL)
                
                # Send ping to all clients
                dead_clients = []
                for client_id, ws in self.active_connections.items():
                    try:
                        await ws.send_json({"type": "ping", "timestamp": time.time()})
                    except Exception:
                        dead_clients.append(client_id)
                
                # Cleanup
                for client_id in dead_clients:
                    await self.disconnect(client_id)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[WS] Heartbeat error: {e}")


manager = ConnectionManager()


# ============================================================================
# API Routes
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect."""
    return {
        "service": ServiceConfig.NAME,
        "version": ServiceConfig.VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns service status, dependencies, and basic metrics.
    """
    try:
        health = pipeline_health()
        
        return HealthResponse(
            status="healthy",
            service=ServiceConfig.NAME,
            version=ServiceConfig.VERSION,
            model=health['model'],
            cache_backend=health['cache']['backend'],
            uptime_seconds=service_metrics.get_uptime(),
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "service": ServiceConfig.NAME,
            }
        )


@app.get("/metrics")
async def metrics():
    """
    Expose service and pipeline metrics.
    Can be scraped by Prometheus or similar.
    """
    if not ServiceConfig.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    return {
        "service": service_metrics.to_dict(),
        "pipeline": pipeline_metrics(),
    }


@app.post("/answer", response_model=AnswerResponse)
async def generate_answer(
    req: AnswerRequest,
    x_request_id: Optional[str] = Header(None),
):
    """
    Generate interview answer (non-streaming).
    
    - Validates input
    - Checks cache
    - Calls OpenAI API
    - Broadcasts to WebSocket clients
    - Returns JSON response
    """
    request_id = x_request_id or str(uuid.uuid4())
    service_metrics.request_count += 1
    
    if ServiceConfig.LOG_REQUESTS:
        print(
            f"[API:{request_id}] POST /answer | "
            f"question='{req.question[:50]}...' | "
            f"style={req.style} | "
            f"company={req.company or 'none'}"
        )
    
    start_time = time.perf_counter()
    
    try:
        # Generate answer
        answer, metadata = pipeline.answer(
            prompt=req.question,
            style=req.style,
            company=req.company,
            use_cache=True,
        )
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Broadcast to WebSocket clients
        asyncio.create_task(manager.broadcast({
            "type": "answer",
            "request_id": request_id,
            "question": req.question,
            "answer": answer,
            "cached": metadata.cached,
        }))
        
        if ServiceConfig.LOG_REQUESTS:
            print(
                f"[API:{request_id}] Completed in {duration_ms:.0f}ms | "
                f"cached={metadata.cached} | "
                f"tokens={metadata.total_tokens}"
            )
        
        return AnswerResponse(
            answer=answer,
            cached=metadata.cached,
            metadata={
                "model": metadata.model,
                "tokens": metadata.total_tokens,
                "duration_ms": metadata.duration_ms,
            } if not metadata.cached else None,
            request_id=request_id,
        )
    
    except OpenAIError as e:
        service_metrics.error_count += 1
        print(f"[API:{request_id}] OpenAI error: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error: {str(e)}"
        )
    
    except Exception as e:
        service_metrics.error_count += 1
        print(f"[API:{request_id}] Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/answer/stream")
async def generate_answer_stream(
    req: AnswerRequest,
    x_request_id: Optional[str] = Header(None),
):
    """
    Generate interview answer with streaming response.
    Returns Server-Sent Events (SSE) stream.
    """
    request_id = x_request_id or str(uuid.uuid4())
    service_metrics.request_count += 1
    
    if ServiceConfig.LOG_REQUESTS:
        print(f"[API:{request_id}] POST /answer/stream | question='{req.question[:50]}...'")
    
    async def event_generator():
        """Generate SSE events."""
        try:
            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'request_id': request_id})}\n\n"
            
            # Stream tokens
            full_answer = []
            for token in pipeline.stream_answer(req.question, req.style, req.company):
                full_answer.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run
            
            # Send complete event
            complete_answer = "".join(full_answer)
            yield f"data: {json.dumps({'type': 'done', 'answer': complete_answer})}\n\n"
            
            # Broadcast to WebSocket clients
            asyncio.create_task(manager.broadcast({
                "type": "answer",
                "request_id": request_id,
                "question": req.question,
                "answer": complete_answer,
                "streamed": True,
            }))
            
            if ServiceConfig.LOG_REQUESTS:
                print(f"[API:{request_id}] Stream completed")
        
        except Exception as e:
            service_metrics.error_count += 1
            print(f"[API:{request_id}] Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Request-ID": request_id,
        }
    )


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time answer broadcasting.
    
    Protocol:
    - Server sends: {"type": "ping", "timestamp": float}
    - Server sends: {"type": "answer", "request_id": str, "answer": str, ...}
    - Client can send: {"type": "pong"} or ignored
    """
    client_id = str(uuid.uuid4())
    
    try:
        await manager.connect(websocket, client_id)
        
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "message": "WebSocket connection established",
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages (or timeout)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=ServiceConfig.WS_HEARTBEAT_INTERVAL * 2
                )
                
                # Parse and handle client messages
                try:
                    message = json.loads(data)
                    msg_type = message.get("type")
                    
                    if msg_type == "pong":
                        # Acknowledge pong
                        pass
                    elif msg_type == "subscribe":
                        # Future: topic-based subscription
                        pass
                    else:
                        print(f"[WS] Unknown message type from {client_id}: {msg_type}")
                
                except json.JSONDecodeError:
                    print(f"[WS] Invalid JSON from {client_id}")
            
            except asyncio.TimeoutError:
                # No message received, connection might be dead
                print(f"[WS] Client {client_id} timeout, checking connection...")
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break  # Connection is dead
    
    except WebSocketDisconnect:
        print(f"[WS] Client {client_id} disconnected normally")
    
    except Exception as e:
        print(f"[WS] Error with client {client_id}: {e}")
    
    finally:
        await manager.disconnect(client_id)


# ============================================================================
# Admin Endpoints
# ============================================================================

@app.post("/admin/cache/clear")
async def clear_cache(x_admin_token: Optional[str] = Header(None)):
    """Clear the entire cache (requires admin token)."""
    admin_token = os.getenv("ADMIN_TOKEN")
    if admin_token and x_admin_token != admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    
    cache_clear()
    return {"status": "success", "message": "Cache cleared"}


@app.post("/admin/metrics/reset")
async def reset_service_metrics(x_admin_token: Optional[str] = Header(None)):
    """Reset all metrics (requires admin token)."""
    admin_token = os.getenv("ADMIN_TOKEN")
    if admin_token and x_admin_token != admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    
    reset_metrics()
    service_metrics.request_count = 0
    service_metrics.error_count = 0
    return {"status": "success", "message": "Metrics reset"}


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Standardized HTTP exception responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Catch-all exception handler."""
    service_metrics.error_count += 1
    print(f"[ERROR] Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else None,
            "path": str(request.url),
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=ServiceConfig.HOST,
        port=ServiceConfig.PORT,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
    )
