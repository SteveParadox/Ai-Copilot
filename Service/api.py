"""
ASR API Service
===============
FastAPI service for audio transcription with REST and WebSocket endpoints.

Features:
- Single file transcription (POST /transcribe)
- Batch transcription (POST /transcribe_batch)
- Streaming transcription (WebSocket /ws/stream)
- Health monitoring and metrics
- Comprehensive error handling
- Request validation and rate limiting

Dependencies:
- fastapi, uvicorn
- numpy
- python-multipart (for file uploads)
"""

import asyncio
import logging
import os
import time
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Header,
    status,
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator

from Asr.utils import majority_vote, weighted_vote, TieStrategy, VotingResult
from Asr.model import (
    load_whisper,
    transcribe,
    transcribe_async,
    transcribe_batch_async,
    get_health as model_health,
    get_metrics as model_metrics,
    TranscriptionResult,
)
from Asr._preprocessing import (
    Preprocessor,
    get_health as preprocessing_health,
    analyze_audio,
)

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

class ServiceConfig:
    """Service-level configuration."""
    NAME: str = os.getenv("SERVICE_NAME", "ASR Service")
    VERSION: str = os.getenv("SERVICE_VERSION", "1.0.0")
    HOST: str = os.getenv("SERVICE_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("SERVICE_PORT", "8000"))
    
    # Model
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
    
    # CORS
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8001"
    ).split(",")
    
    # File upload limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "10"))
    
    # WebSocket
    WS_CHUNK_SIZE: int = int(os.getenv("WS_CHUNK_SIZE", "32000"))  # 1s @ 16kHz int16
    WS_HISTORY_SIZE: int = int(os.getenv("WS_HISTORY_SIZE", "5"))
    WS_HEARTBEAT_INTERVAL: int = int(os.getenv("WS_HEARTBEAT_INTERVAL", "30"))
    
    # Timeouts
    TRANSCRIPTION_TIMEOUT: int = int(os.getenv("TRANSCRIPTION_TIMEOUT", "30"))
    BATCH_TIMEOUT: int = int(os.getenv("BATCH_TIMEOUT", "300"))
    
    # Features
    PREPROCESSING_ENABLED: bool = os.getenv("PREPROCESSING_ENABLED", "true").lower() == "true"
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

# ============================================================================
# Metrics
# ============================================================================

class ServiceMetrics:
    """Track service-level metrics."""
    
    def __init__(self):
        self.requests_total = 0
        self.requests_success = 0
        self.requests_error = 0
        self.ws_connections_total = 0
        self.ws_connections_active = 0
        self.total_audio_seconds = 0.0
        self.start_time = time.time()
    
    def record_request(self, success: bool, audio_duration: float = 0.0):
        self.requests_total += 1
        if success:
            self.requests_success += 1
        else:
            self.requests_error += 1
        self.total_audio_seconds += audio_duration
    
    def get_uptime(self) -> float:
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests_total": self.requests_total,
            "requests_success": self.requests_success,
            "requests_error": self.requests_error,
            "success_rate": (
                self.requests_success / self.requests_total
                if self.requests_total > 0
                else 0.0
            ),
            "ws_connections_total": self.ws_connections_total,
            "ws_connections_active": self.ws_connections_active,
            "total_audio_seconds": round(self.total_audio_seconds, 2),
            "uptime_seconds": round(self.get_uptime(), 2),
        }


service_metrics = ServiceMetrics()

# ============================================================================
# Preprocessor
# ============================================================================

preprocessor: Optional[Preprocessor] = None

# ============================================================================
# Lifecycle Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global preprocessor
    
    # Startup
    log.info(f"[{ServiceConfig.NAME}] Starting up...")
    log.info(f"  Version: {ServiceConfig.VERSION}")
    log.info(f"  Model: {ServiceConfig.WHISPER_MODEL}")
    log.info(f"  Sample rate: {ServiceConfig.SAMPLE_RATE}Hz")
    
    # Load Whisper model
    try:
        log.info("Loading Whisper model...")
        load_whisper(model_size=ServiceConfig.WHISPER_MODEL)
        log.info("Whisper model loaded successfully")
    except Exception as e:
        log.error(f"Failed to load Whisper model: {e}")
        raise
    
    # Initialize preprocessor
    if ServiceConfig.PREPROCESSING_ENABLED:
        try:
            preprocessor = Preprocessor(sample_rate=ServiceConfig.SAMPLE_RATE)
            log.info("Preprocessor initialized")
        except Exception as e:
            log.error(f"Failed to initialize preprocessor: {e}")
            raise
    
    log.info("Service ready")
    
    yield
    
    # Shutdown
    log.info(f"[{ServiceConfig.NAME}] Shutting down...")
    log.info(f"  Total requests: {service_metrics.requests_total}")
    log.info(f"  Success rate: {service_metrics.requests_success/max(service_metrics.requests_total, 1):.2%}")
    log.info(f"  Total audio: {service_metrics.total_audio_seconds:.1f}s")


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

class TranscribeResponse(BaseModel):
    """Response model for single transcription."""
    transcript: str
    request_id: str
    audio_duration_seconds: Optional[float] = None
    processing_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchTranscribeResponse(BaseModel):
    """Response model for batch transcription."""
    results: List[Dict[str, Any]]
    request_id: str
    total_files: int
    successful: int
    failed: int
    total_processing_time_ms: Optional[float] = None


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
    sample_rate: int
    preprocessing_enabled: bool
    uptime_seconds: float


# ============================================================================
# Audio Validation
# ============================================================================

def validate_audio_file(
    file: UploadFile,
    max_size_mb: int = ServiceConfig.MAX_FILE_SIZE_MB,
) -> bytes:
    """
    Validate uploaded audio file.
    
    Args:
        file: Uploaded file
        max_size_mb: Maximum file size in MB
    
    Returns:
        Audio bytes
    
    Raises:
        HTTPException: If validation fails
    """
    # Check file exists
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Read file
    try:
        audio_bytes = file.file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(e)}"
        )
    
    # Check size
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
        )
    
    # Check not empty
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file"
        )
    
    return audio_bytes


def bytes_to_audio_array(
    audio_bytes: bytes,
    dtype: np.dtype = np.int16,
) -> np.ndarray:
    """
    Convert bytes to audio array.
    
    Args:
        audio_bytes: Raw audio bytes
        dtype: Expected audio dtype
    
    Returns:
        Audio array
    
    Raises:
        HTTPException: If conversion fails
    """
    try:
        audio_array = np.frombuffer(audio_bytes, dtype=dtype)
        
        if audio_array.size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio array is empty after conversion"
            )
        
        return audio_array
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to convert audio bytes: {str(e)}"
        )


def preprocess_audio(audio: np.ndarray) -> np.ndarray:
    """
    Preprocess audio if enabled.
    
    Args:
        audio: Raw audio array
    
    Returns:
        Preprocessed audio
    """
    if not ServiceConfig.PREPROCESSING_ENABLED or preprocessor is None:
        return audio
    
    try:
        return preprocessor.process(audio)
    except Exception as e:
        log.warning(f"Preprocessing failed: {e}, using raw audio")
        return audio


# ============================================================================
# REST Endpoints
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
    """
    try:
        model_health_status = model_health()
        
        return HealthResponse(
            status="healthy",
            service=ServiceConfig.NAME,
            version=ServiceConfig.VERSION,
            model=model_health_status['model']['model_info']['size']
            if model_health_status['model']['loaded'] else "not_loaded",
            sample_rate=ServiceConfig.SAMPLE_RATE,
            preprocessing_enabled=ServiceConfig.PREPROCESSING_ENABLED,
            uptime_seconds=service_metrics.get_uptime(),
        )
    except Exception as e:
        log.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "service": ServiceConfig.NAME,
            }
        )


@app.get("/metrics")
async def metrics():
    """
    Expose service and model metrics.
    """
    if not ServiceConfig.METRICS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics disabled"
        )
    
    return {
        "service": service_metrics.to_dict(),
        "model": model_metrics(),
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    x_request_id: Optional[str] = Header(None),
    language: Optional[str] = None,
):
    """
    Transcribe a single audio file.
    
    Args:
        file: Audio file (PCM int16, 16kHz recommended)
        x_request_id: Optional request ID for tracking
        language: Optional language code (e.g., 'en')
    
    Returns:
        TranscribeResponse with transcript and metadata
    """
    request_id = x_request_id or str(uuid.uuid4())
    start_time = time.perf_counter()
    
    log.info(f"[{request_id}] POST /transcribe | file={file.filename}")
    
    try:
        # Validate and read file
        audio_bytes = validate_audio_file(file)
        
        # Convert to audio array
        audio_array = bytes_to_audio_array(audio_bytes)
        
        # Calculate duration
        audio_duration = len(audio_array) / ServiceConfig.SAMPLE_RATE
        
        # Preprocess
        audio_array = preprocess_audio(audio_array)
        
        # Transcribe with timeout
        try:
            result = await asyncio.wait_for(
                transcribe_async(audio_array, language=language),
                timeout=ServiceConfig.TRANSCRIPTION_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=f"Transcription timeout ({ServiceConfig.TRANSCRIPTION_TIMEOUT}s)"
            )
        
        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Record metrics
        service_metrics.record_request(success=True, audio_duration=audio_duration)
        
        log.info(
            f"[{request_id}] Completed in {processing_time_ms:.0f}ms | "
            f"audio={audio_duration:.1f}s | "
            f"text_len={len(result.text)}"
        )
        
        return TranscribeResponse(
            transcript=result.text,
            request_id=request_id,
            audio_duration_seconds=audio_duration,
            processing_time_ms=processing_time_ms,
            metadata=result.to_dict() if result else None,
        )
    
    except HTTPException:
        service_metrics.record_request(success=False)
        raise
    
    except Exception as e:
        service_metrics.record_request(success=False)
        log.error(f"[{request_id}] Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


@app.post("/transcribe_batch", response_model=BatchTranscribeResponse)
async def transcribe_batch_files(
    files: List[UploadFile] = File(...),
    x_request_id: Optional[str] = Header(None),
    language: Optional[str] = None,
):
    """
    Transcribe multiple audio files in batch.
    
    Args:
        files: List of audio files
        x_request_id: Optional request ID
        language: Optional language code
    
    Returns:
        BatchTranscribeResponse with results for each file
    """
    request_id = x_request_id or str(uuid.uuid4())
    start_time = time.perf_counter()
    
    log.info(f"[{request_id}] POST /transcribe_batch | files={len(files)}")
    
    # Validate batch size
    if len(files) > ServiceConfig.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(files)} exceeds maximum {ServiceConfig.MAX_BATCH_SIZE}"
        )
    
    try:
        # Validate and convert all files
        audio_arrays = []
        filenames = []
        
        for file in files:
            try:
                audio_bytes = validate_audio_file(file)
                audio_array = bytes_to_audio_array(audio_bytes)
                audio_array = preprocess_audio(audio_array)
                
                audio_arrays.append(audio_array)
                filenames.append(file.filename or "unknown")
            
            except HTTPException as e:
                # Record failed file but continue with others
                log.warning(f"[{request_id}] File {file.filename} failed validation: {e.detail}")
                audio_arrays.append(np.array([]))  # Empty array as placeholder
                filenames.append(file.filename or "unknown")
        
        # Transcribe batch with timeout
        try:
            results = await asyncio.wait_for(
                transcribe_batch_async(audio_arrays, language=language),
                timeout=ServiceConfig.BATCH_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=f"Batch transcription timeout ({ServiceConfig.BATCH_TIMEOUT}s)"
            )
        
        # Build response
        response_results = []
        successful = 0
        failed = 0
        
        for filename, audio, result in zip(filenames, audio_arrays, results):
            if audio.size == 0 or not result.text:
                response_results.append({
                    "filename": filename,
                    "transcript": "",
                    "success": False,
                    "error": "Validation or transcription failed",
                })
                failed += 1
            else:
                response_results.append({
                    "filename": filename,
                    "transcript": result.text,
                    "success": True,
                    "audio_duration_seconds": len(audio) / ServiceConfig.SAMPLE_RATE,
                    "metadata": result.to_dict(),
                })
                successful += 1
        
        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Record metrics
        total_audio_duration = sum(
            len(a) / ServiceConfig.SAMPLE_RATE
            for a in audio_arrays if a.size > 0
        )
        service_metrics.record_request(
            success=(failed == 0),
            audio_duration=total_audio_duration
        )
        
        log.info(
            f"[{request_id}] Batch completed in {processing_time_ms:.0f}ms | "
            f"successful={successful}/{len(files)}"
        )
        
        return BatchTranscribeResponse(
            results=response_results,
            request_id=request_id,
            total_files=len(files),
            successful=successful,
            failed=failed,
            total_processing_time_ms=processing_time_ms,
        )
    
    except HTTPException:
        service_metrics.record_request(success=False)
        raise
    
    except Exception as e:
        service_metrics.record_request(success=False)
        log.error(f"[{request_id}] Batch error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch transcription failed: {str(e)}"
        )


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections with heartbeat."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept and register WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        service_metrics.ws_connections_total += 1
        service_metrics.ws_connections_active = len(self.active_connections)
        
        log.info(f"[WS] Client {client_id} connected. Total: {len(self.active_connections)}")
        
        # Start heartbeat if not running
        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def disconnect(self, client_id: str):
        """Remove WebSocket connection."""
        if client_id in self.active_connections:
            ws = self.active_connections[client_id]
            try:
                await ws.close()
            except Exception:
                pass
            del self.active_connections[client_id]
            service_metrics.ws_connections_active = len(self.active_connections)
            log.info(f"[WS] Client {client_id} disconnected. Total: {len(self.active_connections)}")
    
    async def send_text(self, client_id: str, message: str):
        """Send message to specific client."""
        if client_id not in self.active_connections:
            return False
        
        try:
            await self.active_connections[client_id].send_text(message)
            return True
        except Exception as e:
            log.error(f"[WS] Send error to {client_id}: {e}")
            await self.disconnect(client_id)
            return False
    
    async def send_json(self, client_id: str, data: Dict[str, Any]):
        """Send JSON message to specific client."""
        if client_id not in self.active_connections:
            return False
        
        try:
            await self.active_connections[client_id].send_json(data)
            return True
        except Exception as e:
            log.error(f"[WS] Send error to {client_id}: {e}")
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
                log.error(f"[WS] Heartbeat error: {e}")


manager = ConnectionManager()


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Streaming transcription via WebSocket.
    
    Protocol:
    - Client sends: raw audio bytes (PCM int16, 16kHz)
    - Server sends: {"type": "transcript", "text": "...", "stable": bool}
    - Server sends: {"type": "ping", "timestamp": float}
    - Server sends: {"type": "error", "error": "..."}
    """
    client_id = str(uuid.uuid4())
    
    try:
        await manager.connect(websocket, client_id)
        
        # Send welcome message
        await manager.send_json(client_id, {
            "type": "connected",
            "client_id": client_id,
            "chunk_size": ServiceConfig.WS_CHUNK_SIZE,
            "sample_rate": ServiceConfig.SAMPLE_RATE,
        })
        
        # Streaming state
        history: List[str] = []
        buffer = b""
        chunk_count = 0
        
        while True:
            # Receive audio chunk
            try:
                data = await websocket.receive_bytes()
            except WebSocketDisconnect:
                log.info(f"[WS:{client_id}] Client disconnected")
                break
            
            buffer += data
            
            # Process complete chunks
            while len(buffer) >= ServiceConfig.WS_CHUNK_SIZE:
                chunk_bytes = buffer[:ServiceConfig.WS_CHUNK_SIZE]
                buffer = buffer[ServiceConfig.WS_CHUNK_SIZE:]
                
                chunk_count += 1
                
                try:
                    # Convert to audio array
                    audio_array = np.frombuffer(chunk_bytes, dtype=np.int16)
                    
                    # Preprocess
                    audio_array = preprocess_audio(audio_array)
                    
                    # Transcribe
                    result = await transcribe_async(audio_array)
                    
                    transcript = result.text.strip()
                    
                    if transcript:
                        history.append(transcript)
                        
                        # Keep sliding window
                        window = history[-ServiceConfig.WS_HISTORY_SIZE:]
                        
                        # Voting for stability
                        vote_result = majority_vote(
                            window,
                            tie_strategy=TieStrategy.FIRST,
                        )
                        
                        stable_text = vote_result.winner or transcript
                        is_stable = vote_result.confidence > 0.6
                        
                        # Send result
                        await manager.send_json(client_id, {
                            "type": "transcript",
                            "text": stable_text,
                            "raw_text": transcript,
                            "stable": is_stable,
                            "confidence": vote_result.confidence,
                            "chunk_number": chunk_count,
                        })
                        
                        log.info(
                            f"[WS:{client_id}] Chunk {chunk_count}: "
                            f"'{stable_text[:50]}...' (stable={is_stable})"
                        )
                
                except Exception as e:
                    log.error(f"[WS:{client_id}] Processing error: {e}", exc_info=True)
                    await manager.send_json(client_id, {
                        "type": "error",
                        "error": str(e),
                        "chunk_number": chunk_count,
                    })
    
    except Exception as e:
        log.error(f"[WS:{client_id}] Unhandled error: {e}", exc_info=True)
        try:
            await manager.send_json(client_id, {
                "type": "error",
                "error": f"Server error: {str(e)}",
            })
        except Exception:
            pass
    
    finally:
        await manager.disconnect(client_id)


# ============================================================================
# Admin Endpoints
# ============================================================================

@app.post("/admin/reset_metrics")
async def reset_metrics(x_admin_token: Optional[str] = Header(None)):
    """Reset service metrics (requires admin token)."""
    admin_token = os.getenv("ADMIN_TOKEN")
    if admin_token and x_admin_token != admin_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token"
        )
    
    global service_metrics
    service_metrics = ServiceMetrics()
    
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
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=ServiceConfig.HOST,
        port=ServiceConfig.PORT,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
    )

