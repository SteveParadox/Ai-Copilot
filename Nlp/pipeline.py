import os
import time
import logging
import redis
import json
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

OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.4))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 800))
CACHE_TTL = int(os.getenv("NLP_CACHE_TTL", 3600))  # default: 1 hour

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- cache (Redis preferred) ----------------
try:
    redis_client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
    )
    redis_client.ping()
    log.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    log.warning(f"Redis not available, falling back to in-memory cache: {e}")
    redis_client = None
    _CACHE = {}


def _cache_get(key: str):
    try:
        if redis_client:
            val = redis_client.get(key)
            return json.loads(val) if val else None
        else:
            v = _CACHE.get(key)
            if not v:
                return None
            if CACHE_TTL and time.time() - v["ts"] > CACHE_TTL:
                del _CACHE[key]
                return None
            return v["val"]
    except Exception as e:
        log.error(f"Cache get error: {e}")
        return None


def _cache_set(key: str, value: str):
    try:
        if redis_client:
            redis_client.set(key, json.dumps(value), ex=CACHE_TTL or None)
        else:
            _CACHE[key] = {"ts": time.time(), "val": value}
    except Exception as e:
        log.error(f"Cache set error: {e}")


# ---------------- pipeline ----------------
class NLPPipeline:
    def __init__(self):
        self.model = OPENAI_MODEL

    def _build_messages(self, prompt: str, style: str = "Concise", company: str = ""):
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
