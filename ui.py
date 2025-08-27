import asyncio
import json
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
import sounddevice as sd
import websockets
import httpx
import numpy as np

# ---------------- CONFIG ----------------
ASR_WS_URL = "ws://localhost:8000/stream"     # ASR service WebSocket
NLP_SSE_URL = "http://localhost:8001/answer"  # NLP SSE endpoint

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # frames per chunk

# ---------------- UI ----------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("AI Interview Assistant")

        # Transcript displays
        tk.Label(root, text="ASR Partial Transcript").pack()
        self.partial_box = scrolledtext.ScrolledText(root, height=5)
        self.partial_box.pack(fill=tk.BOTH, expand=True)

        tk.Label(root, text="ASR Final Transcript").pack()
        self.final_box = scrolledtext.ScrolledText(root, height=5)
        self.final_box.pack(fill=tk.BOTH, expand=True)

        tk.Label(root, text="GPT Streaming Answer").pack()
        self.answer_box = scrolledtext.ScrolledText(root, height=10)
        self.answer_box.pack(fill=tk.BOTH, expand=True)

        self.start_btn = tk.Button(root, text="Start Mic", command=self.start_mic)
        self.start_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.stop_btn = tk.Button(root, text="Stop Mic", command=self.stop_mic, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.send_btn = tk.Button(root, text="Send Question", command=self.send_question)
        self.send_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Audio buffer / control
        self.recording = False
        self.audio_queue = queue.Queue()

        # Start asyncio loop in background
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    # ---------------- Mic ----------------
    def start_mic(self):
        self.recording = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.partial_box.delete("1.0", tk.END)
        self.final_box.delete("1.0", tk.END)
        threading.Thread(target=self.capture_audio, daemon=True).start()
        threading.Thread(target=self.asr_stream_worker, daemon=True).start()

    def stop_mic(self):
        self.recording = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def capture_audio(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            if self.recording:
                self.audio_queue.put(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                            callback=callback, blocksize=CHUNK_SIZE):
            while self.recording:
                sd.sleep(100)

    # ---------------- ASR Worker ----------------
    def asr_stream_worker(self):
        async def _worker():
            try:
                async with websockets.connect(ASR_WS_URL) as ws:
                    while self.recording or not self.audio_queue.empty():
                        try:
                            chunk = self.audio_queue.get(timeout=0.1)
                        except queue.Empty:
                            await asyncio.sleep(0.05)
                            continue
                        await ws.send(chunk.tobytes())
                        resp = await ws.recv()
                        data = json.loads(resp)
                        partial = data.get("partial_text", "")
                        if partial:
                            self.partial_box.delete("1.0", tk.END)
                            self.partial_box.insert(tk.END, partial)
            except Exception as e:
                print(f"ASR stream error: {e}")

        asyncio.run_coroutine_threadsafe(_worker(), self.loop)

    # ---------------- Send Question to NLP (SSE) ----------------
    def send_question(self):
        question = self.partial_box.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("Warning", "No question detected.")
            return

        self.final_box.insert(tk.END, question + "\n")
        self.partial_box.delete("1.0", tk.END)
        self.answer_box.delete("1.0", tk.END)

        # Start SSE worker
        asyncio.run_coroutine_threadsafe(self.gpt_sse_worker(question), self.loop)

    async def gpt_sse_worker(self, question):
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", NLP_SSE_URL, json={"question": question, "style": "Concise", "company": ""}) as resp:
                    async for line_bytes in resp.aiter_lines():
                        line = line_bytes.strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data:"):
                            try:
                                chunk = json.loads(line.replace("data:", ""))
                                delta = chunk.get("delta", "")
                                if delta:
                                    self.answer_box.insert(tk.END, delta)
                                    self.answer_box.see(tk.END)
                            except Exception:
                                continue
            except Exception as e:
                print(f"NLP SSE error: {e}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
