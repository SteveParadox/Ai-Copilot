# nlp_service/pipeline.py
import os
import time
import threading
import asyncio
import logging
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

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

    def _build_messages(self, prompt: str, style: str = "Concise", company: str = ""):
        """
        Internal helper to standardize messages.
        """
        return [
            {
                "role": "system",
                "content": f"You are an AI assistant for interviews. Provide answers in a {style} manner.",
            },
            {
                "role": "user",
                "content": prompt + (f" The company is {company}." if company else ""),
            },
        ]

    def non_stream_answer(self, prompt: str, style: str = "Concise", company: str = "") -> str:
        """
        Synchronous non-streaming answer (full JSON response).
        """
        cache_key = f"{prompt}|{style}|{company}"
        cached = _cache_get(cache_key)
        if cached:
            log.info(f"Cache hit for {cache_key}")
            return cached

        messages = self._build_messages(prompt, style, company)
        log.info(f"Calling OpenAI {self.model} (stream=False)")
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
        )

        answer = resp.choices[0].message.content.strip()
        _cache_set(cache_key, answer)
        return answer

    def stream_answer(self, prompt: str, style: str = "Concise", company: str = ""):
        """
        Generator that yields answer deltas from OpenAI streaming.
        """
        messages = self._build_messages(prompt, style, company)
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


pipeline = NLPPipeline()
cache_get = _cache_get
cache_set = _cache_set
