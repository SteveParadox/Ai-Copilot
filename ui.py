"""
AI Interview Assistant GUI
===========================
Desktop application for real-time interview assistance with ASR and NLP.

Features:
- Real-time audio transcription via WebSocket
- Streaming GPT answers via Server-Sent Events
- Robust error handling and reconnection
- Audio level monitoring
- Transcript history and export
- Configurable settings

Dependencies:
- tkinter (standard library)
- sounddevice
- websockets
- httpx
- numpy
"""

import asyncio
import json
import queue
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import tkinter as tk
from tkinter import (
    scrolledtext,
    messagebox,
    filedialog,
)
import sounddevice as sd
import websockets
import httpx
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AppConfig:
    """Application configuration."""
    # Service endpoints
    ASR_WS_URL: str = "ws://localhost:8000/ws/stream"
    NLP_HTTP_URL: str = "http://localhost:8001/answer/stream"

    # Audio settings
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    DTYPE: str = "int16"
    CHUNK_SIZE: int = 1024
    BUFFER_SIZE: int = 100

    # UI settings
    WINDOW_WIDTH: int = 900
    WINDOW_HEIGHT: int = 700
    FONT_FAMILY: str = "Consolas"
    FONT_SIZE: int = 10

    # Features
    AUTO_SCROLL: bool = True
    SAVE_TRANSCRIPTS: bool = True
    TRANSCRIPT_DIR: Path = Path("transcripts")

    # Network
    WS_RECONNECT_DELAY: float = 2.0
    WS_MAX_RETRIES: int = 3
    HTTP_TIMEOUT: float = 120.0

    # Audio monitoring
    LEVEL_UPDATE_INTERVAL_MS: int = 100
    LEVEL_SMOOTHING: float = 0.3

    def __post_init__(self):
        if self.SAVE_TRANSCRIPTS:
            self.TRANSCRIPT_DIR.mkdir(exist_ok=True)


config = AppConfig()

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

# ============================================================================
# Application State
# ============================================================================

class AppState(Enum):
    """Application states."""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class TranscriptEntry:
    """Single transcript entry."""
    timestamp: datetime
    question: str
    answer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "question": self.question,
            "answer": self.answer,
        }

    def to_markdown(self) -> str:
        return f"""
## {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

**Question:** {self.question}

**Answer:** {self.answer}

---
"""


# ============================================================================
# Audio Level Monitor
# ============================================================================

class AudioLevelMonitor:
    """Monitor audio input levels."""

    def __init__(self, smoothing: float = 0.3):
        self.smoothing = smoothing
        self.current_level = 0.0
        self.peak_level = 0.0
        self.lock = threading.Lock()

    def update(self, audio_chunk: np.ndarray):
        """Update levels from audio chunk."""
        if audio_chunk.size == 0:
            return

        # Calculate RMS
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

        # Normalize to 0-1 range (assuming int16 input)
        level = float(rms / 32768.0)

        with self.lock:
            # Exponential smoothing
            self.current_level = (
                self.smoothing * level +
                (1 - self.smoothing) * self.current_level
            )
            self.peak_level = max(self.peak_level, level)

    def get_level(self) -> float:
        """Get current smoothed level."""
        with self.lock:
            return self.current_level

    def get_peak(self) -> float:
        """Get peak level."""
        with self.lock:
            return self.peak_level

    def reset_peak(self):
        """Reset peak level."""
        with self.lock:
            self.peak_level = 0.0


# ============================================================================
# WebSocket Client
# ============================================================================

class ASRWebSocketClient:
    """WebSocket client for ASR streaming."""

    def __init__(
        self,
        url: str,
        audio_queue: queue.Queue,
        callback: Callable[[str, Any], None],
        max_retries: int = config.WS_MAX_RETRIES,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.url = url
        self.audio_queue = audio_queue
        self.callback = callback
        self.max_retries = max_retries
        self.running = False
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.loop = loop

    async def connect(self):
        """Connect to WebSocket with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                log.info(f"Connecting to ASR service (attempt {attempt}/{self.max_retries})...")
                self.ws = await websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=10,
                )
                log.info("Connected to ASR service")
                return True

            except Exception as e:
                log.error(f"Connection attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(config.WS_RECONNECT_DELAY)
                else:
                    self.callback("error", f"Failed to connect after {self.max_retries} attempts")
                    return False

        return False

    async def run(self):
        """Run WebSocket client."""
        self.running = True

        if not await self.connect():
            return

        try:
            send_task = asyncio.create_task(self._send_loop())
            recv_task = asyncio.create_task(self._receive_loop())

            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            log.error(f"WebSocket client error: {e}", exc_info=True)
            self.callback("error", str(e))

        finally:
            try:
                if self.ws:
                    await self.ws.close()
            except Exception:
                pass
            self.running = False

    async def _send_loop(self):
        """Send audio chunks to server."""
        while self.running:
            try:
                try:
                    chunk = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            self.audio_queue.get,
                            True,
                            0.1
                        ),
                        timeout=0.2
                    )
                except (asyncio.TimeoutError, queue.Empty):
                    continue

                if self.ws:
                    await self.ws.send(chunk.tobytes())

            except Exception as e:
                log.error(f"Send error: {e}")
                break

    async def _receive_loop(self):
        """Receive transcripts from server."""
        while self.running and self.ws:
            try:
                message = await self.ws.recv()

                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "transcript")

                    if msg_type == "transcript":
                        text = data.get("text", "")
                        stable = data.get("stable", False)
                        confidence = data.get("confidence", 0.0)

                        if text:
                            self.callback("transcript", {
                                "text": text,
                                "stable": stable,
                                "confidence": confidence,
                            })

                    elif msg_type == "error":
                        error = data.get("error", "Unknown error")
                        self.callback("error", error)

                    elif msg_type == "ping":
                        pass

                except json.JSONDecodeError:
                    self.callback("transcript", {
                        "text": message,
                        "stable": False,
                        "confidence": 0.0,
                    })

            except websockets.exceptions.ConnectionClosed:
                log.warning("WebSocket connection closed")
                break

            except Exception as e:
                log.error(f"Receive error: {e}")
                break

    def stop(self):
        """Stop WebSocket client."""
        self.running = False
        # Try to close socket ASAP (best effort)
        if self.ws and self.loop and self.loop.is_running():
            async def _close():
                try:
                    await self.ws.close()
                except Exception:
                    pass
            asyncio.run_coroutine_threadsafe(_close(), self.loop)


# ============================================================================
# Main Application
# ============================================================================

class InterviewAssistant:
    """Main application class."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Interview Assistant")
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")

        self.state = AppState.IDLE
        self.audio_queue: queue.Queue = queue.Queue(maxsize=config.BUFFER_SIZE)
        self.audio_stream: Optional[sd.InputStream] = None
        self.level_monitor = AudioLevelMonitor()
        self.transcript_history: list[TranscriptEntry] = []

        self.current_question = ""
        self.current_answer = ""

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever,
            daemon=True,
            name="AsyncioLoop"
        )
        self.loop_thread.start()

        self.ws_client: Optional[ASRWebSocketClient] = None

        self._build_ui()
        self._update_audio_level()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        """Build user interface."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Transcript", command=self._export_transcript)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)

        status_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            anchor=tk.W,
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.level_canvas = tk.Canvas(
            status_frame,
            width=100,
            height=20,
            bg="white",
            highlightthickness=1,
        )
        self.level_canvas.pack(side=tk.RIGHT, padx=5)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(
            main_frame,
            text="Live Transcript (Partial):",
            font=(config.FONT_FAMILY, config.FONT_SIZE, "bold")
        ).pack(anchor=tk.W)

        self.partial_box = scrolledtext.ScrolledText(
            main_frame,
            height=4,
            font=(config.FONT_FAMILY, config.FONT_SIZE),
            wrap=tk.WORD,
        )
        self.partial_box.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        tk.Label(
            main_frame,
            text="Question:",
            font=(config.FONT_FAMILY, config.FONT_SIZE, "bold")
        ).pack(anchor=tk.W)

        self.question_box = scrolledtext.ScrolledText(
            main_frame,
            height=4,
            font=(config.FONT_FAMILY, config.FONT_SIZE),
            wrap=tk.WORD,
        )
        self.question_box.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        tk.Label(
            main_frame,
            text="Answer:",
            font=(config.FONT_FAMILY, config.FONT_SIZE, "bold")
        ).pack(anchor=tk.W)

        self.answer_box = scrolledtext.ScrolledText(
            main_frame,
            height=12,
            font=(config.FONT_FAMILY, config.FONT_SIZE),
            wrap=tk.WORD,
        )
        self.answer_box.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X)

        self.start_btn = tk.Button(
            control_frame,
            text="🎤 Start Recording",
            command=self._start_recording,
            width=15,
            font=(config.FONT_FAMILY, config.FONT_SIZE),
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(
            control_frame,
            text="⏹ Stop Recording",
            command=self._stop_recording,
            width=15,
            font=(config.FONT_FAMILY, config.FONT_SIZE),
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.send_btn = tk.Button(
            control_frame,
            text="📤 Send Question",
            command=self._send_question,
            width=15,
            font=(config.FONT_FAMILY, config.FONT_SIZE),
        )
        self.send_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(
            control_frame,
            text="🗑 Clear",
            command=self._clear_all,
            width=10,
            font=(config.FONT_FAMILY, config.FONT_SIZE),
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

    # ========================================================================
    # Audio Recording
    # ========================================================================

    def _start_recording(self):
        """Start audio recording and ASR streaming."""
        try:
            self.partial_box.delete("1.0", tk.END)

            self.state = AppState.RECORDING
            self._update_status("Recording...")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.send_btn.config(state=tk.DISABLED)

            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            def audio_callback(indata, frames, time_info, status):
                if status:
                    log.warning(f"Audio callback status: {status}")

                if self.state == AppState.RECORDING:
                    self.level_monitor.update(indata[:, 0])

                    try:
                        self.audio_queue.put_nowait(indata.copy())
                    except queue.Full:
                        log.warning("Audio queue full, dropping frame")

            self.audio_stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype=config.DTYPE,
                callback=audio_callback,
                blocksize=config.CHUNK_SIZE,
            )
            self.audio_stream.start()
            log.info("Audio stream started")

            self.ws_client = ASRWebSocketClient(
                url=config.ASR_WS_URL,
                audio_queue=self.audio_queue,
                callback=self._handle_asr_message,
                loop=self.loop,
            )

            asyncio.run_coroutine_threadsafe(
                self.ws_client.run(),
                self.loop
            )

        except Exception as e:
            log.error(f"Failed to start recording: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to start recording:\n{str(e)}")
            self._stop_recording()

    def _stop_recording(self):
        """Stop audio recording and ASR streaming."""
        try:
            self.state = AppState.IDLE
            self._update_status("Stopped")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.send_btn.config(state=tk.NORMAL)

            if self.audio_stream:
                try:
                    self.audio_stream.stop()
                except Exception:
                    pass
                try:
                    self.audio_stream.close()
                except Exception:
                    pass
                self.audio_stream = None
                log.info("Audio stream stopped")

            if self.ws_client:
                self.ws_client.stop()
                self.ws_client = None

            self.level_monitor.reset_peak()

        except Exception as e:
            log.error(f"Error stopping recording: {e}", exc_info=True)

    # ========================================================================
    # ASR Message Handling
    # ========================================================================

    def _handle_asr_message(self, msg_type: str, data: Any):
        """Handle messages from ASR WebSocket."""
        try:
            if msg_type == "transcript":
                text = data.get("text", "")
                stable = data.get("stable", False)
                confidence = data.get("confidence", 0.0)
                self.root.after(0, self._update_partial_transcript, text, stable, confidence)

            elif msg_type == "error":
                error_msg = str(data)
                log.error(f"ASR error: {error_msg}")
                self.root.after(0, self._show_error, f"ASR Error: {error_msg}")

        except Exception as e:
            log.error(f"Error handling ASR message: {e}", exc_info=True)

    def _update_partial_transcript(self, text: str, stable: bool, confidence: float):
        """Update partial transcript display."""
        self.partial_box.delete("1.0", tk.END)
        self.partial_box.insert(tk.END, text)

        if stable:
            self.partial_box.tag_add("stable", "1.0", tk.END)
            self.partial_box.tag_config("stable", background="#e8f5e9")
        else:
            self.partial_box.tag_add("unstable", "1.0", tk.END)
            self.partial_box.tag_config("unstable", background="#fff9c4")

        if config.AUTO_SCROLL:
            self.partial_box.see(tk.END)

    # ========================================================================
    # NLP Interaction
    # ========================================================================

    def _send_question(self):
        """Send question to NLP service."""
        question = self.partial_box.get("1.0", tk.END).strip()

        if not question:
            messagebox.showwarning("Warning", "No question detected. Start recording first.")
            return

        self.question_box.delete("1.0", tk.END)
        self.question_box.insert(tk.END, question)

        self.partial_box.delete("1.0", tk.END)
        self.answer_box.delete("1.0", tk.END)

        self.current_question = question
        self.current_answer = ""
        self.state = AppState.PROCESSING
        self._update_status("Processing...")
        self.send_btn.config(state=tk.DISABLED)

        asyncio.run_coroutine_threadsafe(
            self._stream_nlp_answer(question),
            self.loop
        )

    async def _stream_nlp_answer(self, question: str):
        """Stream answer from NLP service via SSE."""
        try:
            async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    config.NLP_HTTP_URL,
                    json={
                        "question": question,
                        "style": "Concise",
                        "company": "",
                    }
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        line = line.strip()

                        if not line or line.startswith(":"):
                            continue

                        if line.startswith("data:"):
                            try:
                                payload = json.loads(line.replace("data:", "").strip())
                                msg_type = payload.get("type")

                                if msg_type == "token":
                                    token = payload.get("content", "")
                                    if token:
                                        self.current_answer += token
                                        self.root.after(0, self._append_answer, token)

                                elif msg_type == "done":
                                    full_answer = payload.get("answer", self.current_answer)
                                    self.current_answer = full_answer
                                    self.root.after(0, self._finalize_answer)
                                    return

                                elif msg_type == "error":
                                    error = payload.get("error", "Unknown error")
                                    self.root.after(0, self._show_error, f"NLP Error: {error}")
                                    return

                            except json.JSONDecodeError:
                                continue

            self.root.after(0, self._finalize_answer)

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            log.error(f"NLP HTTP error: {error_msg}")
            self.root.after(0, self._show_error, f"NLP Error: {error_msg}")

        except Exception as e:
            log.error(f"NLP streaming error: {e}", exc_info=True)
            self.root.after(0, self._show_error, f"NLP Error: {str(e)}")

    def _append_answer(self, token: str):
        """Append token to answer box."""
        self.answer_box.insert(tk.END, token)
        if config.AUTO_SCROLL:
            self.answer_box.see(tk.END)

    def _finalize_answer(self):
        """Finalize answer and save to history."""
        self.state = AppState.IDLE
        self._update_status("Ready")
        self.send_btn.config(state=tk.NORMAL)

        if self.current_question and self.current_answer:
            entry = TranscriptEntry(
                timestamp=datetime.now(),
                question=self.current_question,
                answer=self.current_answer,
            )
            self.transcript_history.append(entry)

            if config.SAVE_TRANSCRIPTS:
                self._save_transcript_entry(entry)

        log.info("Answer finalized")

    # ========================================================================
    # UI Helpers
    # ========================================================================

    def _update_status(self, message: str):
        """Update status bar."""
        self.status_label.config(text=message)

    def _show_error(self, error: str):
        """Show error message and safely reset state."""
        prev_state = self.state
        self.state = AppState.ERROR
        self._update_status(f"Error: {error}")
        messagebox.showerror("Error", error)

        if prev_state == AppState.RECORDING:
            self._stop_recording()

        self.state = AppState.IDLE
        self.send_btn.config(state=tk.NORMAL)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _update_audio_level(self):
        """Update audio level indicator."""
        if self.state == AppState.RECORDING:
            level = self.level_monitor.get_level()
            peak = self.level_monitor.get_peak()

            self.level_canvas.delete("all")

            self.level_canvas.create_rectangle(
                0, 0, 100, 20,
                fill="white",
                outline="gray",
            )

            level_width = int(level * 100)
            if level_width > 0:
                color = "#4caf50" if level < 0.8 else "#ff9800"
                self.level_canvas.create_rectangle(
                    0, 0, level_width, 20,
                    fill=color,
                    outline="",
                )

            peak_x = int(peak * 100)
            if peak_x > 0:
                self.level_canvas.create_line(
                    peak_x, 0, peak_x, 20,
                    fill="red",
                    width=2,
                )
        else:
            self.level_canvas.delete("all")
            self.level_canvas.create_rectangle(
                0, 0, 100, 20,
                fill="white",
                outline="gray",
            )

        self.root.after(config.LEVEL_UPDATE_INTERVAL_MS, self._update_audio_level)

    def _clear_all(self):
        """Clear all text boxes."""
        self.partial_box.delete("1.0", tk.END)
        self.question_box.delete("1.0", tk.END)
        self.answer_box.delete("1.0", tk.END)
        self._update_status("Cleared")

    # ========================================================================
    # Transcript Management
    # ========================================================================

    def _save_transcript_entry(self, entry: TranscriptEntry):
        """Save transcript entry to file."""
        try:
            filename = config.TRANSCRIPT_DIR / f"transcript_{entry.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)
            log.info(f"Saved transcript to {filename}")
        except Exception as e:
            log.error(f"Failed to save transcript: {e}")

    def _export_transcript(self):
        """Export all transcripts to a file."""
        if not self.transcript_history:
            messagebox.showinfo("Info", "No transcripts to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[
                ("Markdown", "*.md"),
                ("JSON", "*.json"),
                ("Text", "*.txt"),
            ]
        )

        if not filename:
            return

        try:
            ext = Path(filename).suffix.lower()

            if ext == ".json":
                with open(filename, "w", encoding="utf-8") as f:
                    data = [entry.to_dict() for entry in self.transcript_history]
                    json.dump(data, f, indent=2, ensure_ascii=False)

            elif ext == ".md":
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("# Interview Transcript\n\n")
                    for entry in self.transcript_history:
                        f.write(entry.to_markdown())

            else:
                with open(filename, "w", encoding="utf-8") as f:
                    for entry in self.transcript_history:
                        f.write(f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}]\n")
                        f.write(f"Q: {entry.question}\n")
                        f.write(f"A: {entry.answer}\n\n")
                        f.write("-" * 80 + "\n\n")

            messagebox.showinfo("Success", f"Exported {len(self.transcript_history)} transcripts to:\n{filename}")
            log.info(f"Exported transcripts to {filename}")

        except Exception as e:
            log.error(f"Export failed: {e}", exc_info=True)
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")

    # ========================================================================
    # Cleanup
    # ========================================================================

    def _on_close(self):
        """Handle application close."""
        try:
            if self.state == AppState.RECORDING:
                self._stop_recording()

            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception:
                pass

            self.root.destroy()
            log.info("Application closed")

        except Exception as e:
            log.error(f"Error during close: {e}", exc_info=True)
            self.root.destroy()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    root = tk.Tk()
    _ = InterviewAssistant(root)
    root.mainloop()


if __name__ == "__main__":
    main()
