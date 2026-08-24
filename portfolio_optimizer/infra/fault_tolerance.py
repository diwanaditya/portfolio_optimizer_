"""
Fault Tolerance: Retry with Exponential Backoff + Circuit Breaker.

Real market-data and broker API calls fail transiently (rate limits,
network blips, momentary vendor outages). This module provides the two
standard, battle-tested patterns for handling that without either (a)
crashing the whole process on a single blip or (b) hammering a struggling
upstream service into a worse outage.
"""
from __future__ import annotations
import time
import random
import functools
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, timezone


def retry_with_backoff(max_attempts: int = 5, base_delay: float = 0.5, max_delay: float = 30.0,
                        exponential_base: float = 2.0, jitter: bool = True,
                        retryable_exceptions: tuple = (Exception,)):
    """Decorator: retries the wrapped function on failure with exponential
    backoff (delay doubles each attempt, capped at max_delay), plus
    optional random jitter to avoid a "thundering herd" of synchronized
    retries across multiple processes hitting the same upstream service.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay *= (0.5 + random.random())
                    time.sleep(delay)
            raise last_exception  # pragma: no cover (unreachable, satisfies type checkers)
        return wrapper
    return decorator


class CircuitState(Enum):
    CLOSED = "closed"        # normal operation, calls pass through
    OPEN = "open"             # tripped, calls fail fast without hitting upstream
    HALF_OPEN = "half_open"   # trial period, allows one call through to test recovery


class CircuitBreakerOpenError(Exception):
    pass


@dataclass
class CircuitBreaker:
    """Trips ('opens') after `failure_threshold` consecutive failures,
    fails fast (without even attempting the call) while open, then after
    `recovery_timeout` seconds moves to HALF_OPEN and allows one trial
    call through — closing again on success, or re-opening on failure.

    This is what stops a struggling upstream data feed or broker API from
    being hammered by retries while it's down, while still automatically
    recovering once it's healthy again.
    """
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time: datetime | None = field(default=None)

    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def call(self, fn, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN (failed {self.failure_count} times); "
                    f"failing fast without calling upstream."
                )
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def _record_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def reset(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
