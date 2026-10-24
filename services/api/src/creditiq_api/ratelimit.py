"""In-process token-bucket rate limiting per API key (no Redis needed).

Limits are per worker process; with N workers the effective limit is N times the setting. That is
acceptable for this deployment (one worker by default) and is documented in the runbook.
"""
from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status

from creditiq_api.security import Principal, analyst

_UNITS = {"second": 1, "minute": 60, "hour": 3600}


def parse_rate(text: str) -> tuple[int, float]:
    """'600/minute' → (600 requests, 60 seconds)."""
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(second|minute|hour)\s*", text)
    if not m:
        raise ValueError(f"Invalid rate limit '{text}'; use e.g. '600/minute'.")
    return int(m.group(1)), float(_UNITS[m.group(2)])


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self, rate: str):
        self.capacity, self.period = parse_rate(rate)
        self.refill = self.capacity / self.period
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> float:
        """Consume one token. Returns 0 if allowed, else the seconds to wait."""
        now = time.monotonic()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = _Bucket(float(self.capacity), now)
            b.tokens = min(self.capacity, b.tokens + (now - b.updated) * self.refill)
            b.updated = now
            if b.tokens >= 1:
                b.tokens -= 1
                return 0.0
            return (1 - b.tokens) / self.refill


def score_rate_limit(request: Request, principal: Principal = Depends(analyst)) -> Principal:
    limiter: RateLimiter = request.app.state.score_limiter
    wait = limiter.hit(principal.key_id)
    if wait > 0:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f"Rate limit of {limiter.capacity} requests per {int(limiter.period)} s exceeded.",
                            headers={"Retry-After": str(math.ceil(wait))})
    return principal
