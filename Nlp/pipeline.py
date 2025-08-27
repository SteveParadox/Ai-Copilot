# nlp_service/pipeline.py
import os
import time
import threading
import asyncio
import logging
from openai import OpenAI

# ---------------- config ----------------
log = logging.getLogger("nlp_pipeline")
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY required for nlp_service")

OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# allow configurable temp / tokens / ttl
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.4))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 800))
CACHE_TTL = int(os.getenv("NLP_CACHE_TTL", 0))  # 0 = no expiry

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- cache ----------------
_CACHE = {}


def _cache_get(key: str):
    v = _CACHE.get(key)
    if not v:
        return None
    if CACHE_TTL and time.time() - v["ts"] > CACHE_TTL:
        del _CACHE[key]
        return None
    return v["val"]


def _cache_set(key: str, value: str):
    _CACHE[key] = {"ts": time.time(), "val": value}


# ---------------- pipeline ----------------
class NLPPipeline:
    def __init__(self):
        self.model = OPENAI_MODEL

    def stream_answer(self, prompt: str, style: str = "Concise", company: str = ""):
        """
        Generator that yields answer deltas from OpenAI streaming.
        """
        messages = [
            {"role": "system", "content": f"You are an AI assistant for interviews. Provide answers in a {style} manner."},
            {"role": "user", "content": prompt + (f" The company is {company}." if company else "")},
        ]
        log.info(f"Calling OpenAI {self.model} (stream=True)")
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
            stream=True,
        )
        for chunk in resp:
            try:
                delta_obj = getattr(chunk.choices[0], "delta", None)
                delta = getattr(delta_obj, "content", "") or ""
            except Exception:
                delta = ""
            if delta:
                yield delta

    async def run_stream(
        self,
        prompt: str,
        style: str,
        company: str,
        queue: asyncio.Queue,
        loop,
        cache_key: str,
    ):
        """
        Runs OpenAI stream in a background thread and pushes deltas to asyncio queue.
        """
        def _worker():
            full = []
            try:
                for delta in self.stream_answer(prompt, style, company):
                    asyncio.run_coroutine_threadsafe(queue.put({"delta": delta}), loop)
                    full.append(delta)
                _cache_set(cache_key, "".join(full).strip())
            except Exception as e:
                log.exception("OpenAI streaming error")
                asyncio.run_coroutine_threadsafe(queue.put({"error": str(e)}), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_worker, daemon=True).start()


pipeline = NLPPipeline()
cache_get = _cache_get
cache_set = _cache_set
