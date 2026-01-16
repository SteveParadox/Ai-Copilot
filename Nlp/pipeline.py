"""
NLP Pipeline Service
Handles OpenAI chat completions with caching, retry logic, and observability.
"""

import os
import time
import logging
import hashlib
import json
from typing import Generator, Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import redis
from openai import OpenAI, OpenAIError, RateLimitError, APITimeoutError
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

load_dotenv()

# ============================================================================
# Configuration & Setup
# ============================================================================

class Config:
    """Centralized configuration with validation."""
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.4"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "800"))
    OPENAI_TIMEOUT: int = int(os.getenv("OPENAI_TIMEOUT", "30"))
    OPENAI_MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_SOCKET_TIMEOUT: int = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
    REDIS_SOCKET_CONNECT_TIMEOUT: int = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
    
    # Cache
    CACHE_TTL: int = int(os.getenv("NLP_CACHE_TTL", "3600"))
    CACHE_ENABLED: bool = os.getenv("NLP_CACHE_ENABLED", "true").lower() == "true"
    CACHE_KEY_PREFIX: str = os.getenv("CACHE_KEY_PREFIX", "nlp:v1:")
    
    # Feature flags
    STREAM_ENABLED: bool = os.getenv("STREAM_ENABLED", "true").lower() == "true"
    FALLBACK_TO_MEMORY: bool = os.getenv("FALLBACK_TO_MEMORY", "true").lower() == "true"
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls):
        """Validate critical configuration."""
        if not cls.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required")
        
        if cls.OPENAI_TEMPERATURE < 0 or cls.OPENAI_TEMPERATURE > 2:
            raise ValueError("OPENAI_TEMPERATURE must be between 0 and 2")
        
        if cls.OPENAI_MAX_TOKENS < 1:
            raise ValueError("OPENAI_MAX_TOKENS must be positive")
        
        if cls.CACHE_TTL < 0:
            raise ValueError("NLP_CACHE_TTL must be non-negative")


# Initialize and validate config
Config.validate()

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("nlp_pipeline")


# ============================================================================
# Metrics & Monitoring
# ============================================================================

class Metrics:
    """Simple metrics collector. Replace with Prometheus/Datadog in production."""
    
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, list] = {}
    
    def increment(self, metric: str, value: int = 1):
        self.counters[metric] = self.counters.get(metric, 0) + value
    
    def record_time(self, metric: str, duration: float):
        if metric not in self.timers:
            self.timers[metric] = []
        self.timers[metric].append(duration)
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {"counters": self.counters}
        for metric, times in self.timers.items():
            if times:
                stats[f"{metric}_avg"] = sum(times) / len(times)
                stats[f"{metric}_max"] = max(times)
                stats[f"{metric}_min"] = min(times)
        return stats
    
    def reset(self):
        self.counters.clear()
        self.timers.clear()


metrics = Metrics()


# ============================================================================
# Cache Layer
# ============================================================================

class CacheBackend(Enum):
    REDIS = "redis"
    MEMORY = "memory"
    DISABLED = "disabled"


class CacheManager:
    """Unified cache interface with Redis primary, in-memory fallback."""
    
    def __init__(self):
        self.backend = CacheBackend.DISABLED
        self.redis_client: Optional[redis.Redis] = None
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        if not Config.CACHE_ENABLED:
            log.warning("Cache is disabled via config")
            return
        
        # Try Redis first
        if self._init_redis():
            self.backend = CacheBackend.REDIS
            log.info(f"Using Redis cache at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        elif Config.FALLBACK_TO_MEMORY:
            self.backend = CacheBackend.MEMORY
            log.warning("Redis unavailable, using in-memory cache")
        else:
            log.warning("Cache disabled: Redis unavailable and memory fallback disabled")
    
    def _init_redis(self) -> bool:
        """Initialize Redis connection with health check."""
        try:
            self.redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                password=Config.REDIS_PASSWORD,
                socket_timeout=Config.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=Config.REDIS_SOCKET_CONNECT_TIMEOUT,
                decode_responses=True,
                health_check_interval=30,  # Auto-reconnect
            )
            self.redis_client.ping()
            return True
        except Exception as e:
            log.error(f"Redis initialization failed: {e}")
            self.redis_client = None
            return False
    
    def _make_key(self, raw_key: str) -> str:
        """Generate versioned, namespaced cache key."""
        # Hash for consistent length + collision resistance
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()[:16]
        return f"{Config.CACHE_KEY_PREFIX}{key_hash}"
    
    def get(self, key: str) -> Optional[str]:
        """Retrieve from cache with metrics."""
        if self.backend == CacheBackend.DISABLED:
            return None
        
        full_key = self._make_key(key)
        start = time.perf_counter()
        
        try:
            if self.backend == CacheBackend.REDIS:
                result = self._redis_get(full_key)
            else:
                result = self._memory_get(full_key)
            
            metrics.record_time("cache_get_ms", (time.perf_counter() - start) * 1000)
            
            if result:
                metrics.increment("cache_hit")
                log.debug(f"Cache HIT: {key[:50]}...")
            else:
                metrics.increment("cache_miss")
                log.debug(f"Cache MISS: {key[:50]}...")
            
            return result
        
        except Exception as e:
            log.error(f"Cache get error for key {key}: {e}")
            metrics.increment("cache_error")
            return None
    
    def set(self, key: str, value: str):
        """Store in cache with metrics."""
        if self.backend == CacheBackend.DISABLED:
            return
        
        full_key = self._make_key(key)
        start = time.perf_counter()
        
        try:
            if self.backend == CacheBackend.REDIS:
                self._redis_set(full_key, value)
            else:
                self._memory_set(full_key, value)
            
            metrics.record_time("cache_set_ms", (time.perf_counter() - start) * 1000)
            metrics.increment("cache_set")
            log.debug(f"Cache SET: {key[:50]}...")
        
        except Exception as e:
            log.error(f"Cache set error for key {key}: {e}")
            metrics.increment("cache_error")
    
    def _redis_get(self, key: str) -> Optional[str]:
        if not self.redis_client:
            return None
        val = self.redis_client.get(key)
        return json.loads(val) if val else None
    
    def _redis_set(self, key: str, value: str):
        if not self.redis_client:
            return
        ttl = Config.CACHE_TTL if Config.CACHE_TTL > 0 else None
        self.redis_client.set(key, json.dumps(value), ex=ttl)
    
    def _memory_get(self, key: str) -> Optional[str]:
        entry = self._memory_cache.get(key)
        if not entry:
            return None
        
        # Check TTL
        if Config.CACHE_TTL > 0:
            age = time.time() - entry["ts"]
            if age > Config.CACHE_TTL:
                del self._memory_cache[key]
                return None
        
        return entry["val"]
    
    def _memory_set(self, key: str, value: str):
        self._memory_cache[key] = {
            "ts": time.time(),
            "val": value
        }
        
        # Basic eviction: clear if too large
        if len(self._memory_cache) > 10000:
            log.warning("Memory cache exceeds 10k entries, clearing oldest 50%")
            sorted_keys = sorted(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k]["ts"]
            )
            for k in sorted_keys[:5000]:
                del self._memory_cache[k]
    
    def health_check(self) -> Dict[str, Any]:
        """Return cache health status."""
        health = {
            "backend": self.backend.value,
            "enabled": self.backend != CacheBackend.DISABLED,
        }
        
        if self.backend == CacheBackend.REDIS and self.redis_client:
            try:
                self.redis_client.ping()
                health["redis_connected"] = True
            except Exception as e:
                health["redis_connected"] = False
                health["redis_error"] = str(e)
        
        if self.backend == CacheBackend.MEMORY:
            health["memory_entries"] = len(self._memory_cache)
        
        return health
    
    def clear(self):
        """Clear all cache (useful for testing)."""
        if self.backend == CacheBackend.REDIS and self.redis_client:
            # Only clear keys with our prefix
            cursor = 0
            while True:
                cursor, keys = self.redis_client.scan(
                    cursor, match=f"{Config.CACHE_KEY_PREFIX}*", count=100
                )
                if keys:
                    self.redis_client.delete(*keys)
                if cursor == 0:
                    break
            log.info("Redis cache cleared")
        
        elif self.backend == CacheBackend.MEMORY:
            self._memory_cache.clear()
            log.info("Memory cache cleared")


cache = CacheManager()


# ============================================================================
# OpenAI Client with Retry Logic
# ============================================================================

@dataclass
class ResponseMetadata:
    """Track response metadata for observability."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: float
    cached: bool


class OpenAIClient:
    """Wrapper around OpenAI client with retry, timeout, and error handling."""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            timeout=Config.OPENAI_TIMEOUT,
            max_retries=0,  # We handle retries with tenacity
        )
    
    @retry(
        stop=stop_after_attempt(Config.OPENAI_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    def _call_with_retry(self, **kwargs) -> Any:
        """Make OpenAI call with exponential backoff on rate limits."""
        return self.client.chat.completions.create(**kwargs)
    
    def complete(
        self,
        messages: list,
        stream: bool = False,
        **kwargs
    ) -> tuple[Any, Optional[ResponseMetadata]]:
        """
        Call OpenAI API with error handling and metrics.
        
        Returns:
            (response, metadata) for non-streaming
            (generator, None) for streaming
        """
        start = time.perf_counter()
        
        try:
            response = self._call_with_retry(
                model=Config.OPENAI_MODEL,
                messages=messages,
                temperature=Config.OPENAI_TEMPERATURE,
                max_tokens=Config.OPENAI_MAX_TOKENS,
                stream=stream,
                **kwargs
            )
            
            if stream:
                metrics.increment("openai_stream_request")
                return response, None
            
            # Non-streaming: extract metadata
            duration_ms = (time.perf_counter() - start) * 1000
            usage = response.usage
            
            metadata = ResponseMetadata(
                model=response.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                duration_ms=duration_ms,
                cached=False,
            )
            
            metrics.increment("openai_completion_request")
            metrics.increment("openai_tokens_total", metadata.total_tokens)
            metrics.record_time("openai_latency_ms", duration_ms)
            
            log.info(
                f"OpenAI completion: {metadata.total_tokens} tokens "
                f"in {duration_ms:.0f}ms"
            )
            
            return response, metadata
        
        except RateLimitError as e:
            log.error(f"Rate limit exceeded after retries: {e}")
            metrics.increment("openai_rate_limit")
            raise
        
        except APITimeoutError as e:
            log.error(f"Request timeout after {Config.OPENAI_TIMEOUT}s: {e}")
            metrics.increment("openai_timeout")
            raise
        
        except OpenAIError as e:
            log.error(f"OpenAI API error: {e}")
            metrics.increment("openai_error")
            raise
        
        except Exception as e:
            log.error(f"Unexpected error calling OpenAI: {e}")
            metrics.increment("openai_unexpected_error")
            raise


openai_client = OpenAIClient()


# ============================================================================
# NLP Pipeline
# ============================================================================

class NLPPipeline:
    """
    Main pipeline for generating interview answers with OpenAI.
    Includes caching, streaming, and comprehensive error handling.
    """
    
    def __init__(self):
        self.model = Config.OPENAI_MODEL
    
    def _build_messages(
        self,
        prompt: str,
        style: str = "Concise",
        company: str = ""
    ) -> list:
        """Construct chat messages with system prompt."""
        system_content = (
            f"You are an AI interview assistant. "
            f"Provide answers in a {style} manner. "
            f"Be structured, specific, and use layman terms."
        )
        
        user_content = prompt
        if company:
            user_content += f" The company is {company}."
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
    
    def _make_cache_key(self, prompt: str, style: str, company: str) -> str:
        """Generate deterministic cache key."""
        return f"{prompt}|{style}|{company}|{self.model}"
    
    def answer(
        self,
        prompt: str,
        style: str = "Concise",
        company: str = "",
        use_cache: bool = True,
    ) -> tuple[str, ResponseMetadata]:
        """
        Generate non-streaming answer with caching.
        
        Args:
            prompt: The interview question
            style: Answer style (Concise, STAR, Deep Technical)
            company: Company context hint
            use_cache: Whether to use cache (disable for testing)
        
        Returns:
            (answer_text, metadata)
        
        Raises:
            OpenAIError: On API failures
            ValueError: On invalid inputs
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Check cache
        cache_key = self._make_cache_key(prompt, style, company)
        if use_cache:
            cached = cache.get(cache_key)
            if cached:
                log.info(f"Cache hit for: {prompt[:50]}...")
                metadata = ResponseMetadata(
                    model=self.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    duration_ms=0,
                    cached=True,
                )
                return cached, metadata
        
        # Call OpenAI
        messages = self._build_messages(prompt, style, company)
        response, metadata = openai_client.complete(messages, stream=False)
        
        answer = response.choices[0].message.content.strip()
        
        # Cache successful response
        if use_cache and answer:
            cache.set(cache_key, answer)
        
        return answer, metadata
    
    def stream_answer(
        self,
        prompt: str,
        style: str = "Concise",
        company: str = ""
    ) -> Generator[str, None, None]:
        """
        Generate streaming answer (no caching).
        
        Args:
            prompt: The interview question
            style: Answer style
            company: Company context hint
        
        Yields:
            Token strings as they arrive
        
        Raises:
            OpenAIError: On API failures
            ValueError: On invalid inputs
        """
        if not Config.STREAM_ENABLED:
            log.warning("Streaming disabled, falling back to non-streaming")
            answer, _ = self.answer(prompt, style, company, use_cache=False)
            yield answer
            return
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        messages = self._build_messages(prompt, style, company)
        response, _ = openai_client.complete(messages, stream=True)
        
        total_tokens = 0
        start = time.perf_counter()
        
        try:
            for chunk in response:
                try:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        total_tokens += len(delta.split())  # Rough approximation
                        yield delta
                except (AttributeError, IndexError):
                    continue
            
            duration_ms = (time.perf_counter() - start) * 1000
            log.info(
                f"Stream completed: ~{total_tokens} tokens in {duration_ms:.0f}ms"
            )
            metrics.increment("stream_completion_success")
        
        except Exception as e:
            log.error(f"Streaming error: {e}")
            metrics.increment("stream_completion_error")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Return pipeline health status."""
        return {
            "model": self.model,
            "cache": cache.health_check(),
            "metrics": metrics.get_stats(),
            "config": {
                "temperature": Config.OPENAI_TEMPERATURE,
                "max_tokens": Config.OPENAI_MAX_TOKENS,
                "timeout": Config.OPENAI_TIMEOUT,
            }
        }


# ============================================================================
# Public API
# ============================================================================

pipeline = NLPPipeline()


# Backward-compatible aliases for existing code
def non_stream_answer(prompt: str, style: str = "Concise", company: str = "") -> str:
    """Legacy wrapper (returns only text, no metadata)."""
    answer, _ = pipeline.answer(prompt, style, company)
    return answer


def stream_answer(prompt: str, style: str = "Concise", company: str = "") -> Generator[str, None, None]:
    """Legacy wrapper."""
    yield from pipeline.stream_answer(prompt, style, company)


# Expose cache operations for manual control
cache_get = cache.get
cache_set = cache.set
cache_clear = cache.clear


# ============================================================================
# Health & Diagnostics
# ============================================================================

def get_health() -> Dict[str, Any]:
    """Get comprehensive health status."""
    return pipeline.health_check()


def get_metrics() -> Dict[str, Any]:
    """Get current metrics snapshot."""
    return metrics.get_stats()


def reset_metrics():
    """Reset metrics (useful for testing)."""
    metrics.reset()


# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <question> [style] [company]")
        sys.exit(1)
    
    question = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 else "Concise"
    company = sys.argv[3] if len(sys.argv) > 3 else ""
    
    print(f"\n=== Health Check ===")
    print(json.dumps(get_health(), indent=2))
    
    print(f"\n=== Non-Streaming Answer ===")
    answer, metadata = pipeline.answer(question, style, company)
    print(f"Answer: {answer}")
    print(f"Metadata: {metadata}")
    
    print(f"\n=== Streaming Answer ===")
    for token in pipeline.stream_answer(question, style, company):
        print(token, end="", flush=True)
    print("\n")
    
    print(f"\n=== Metrics ===")
    print(json.dumps(get_metrics(), indent=2))
