"""Token bucket rate limiter for LLM endpoint rate control.

This module provides a token bucket rate limiter that controls the rate
of requests to LLM endpoints. The algorithm allows for bursting up to
a maximum capacity while maintaining a steady long-term rate.

Token Bucket Algorithm:
    - Tokens are added to the bucket at a constant rate
    - Each request consumes one token
    - If no tokens are available, the caller waits until refill
    - Burst capacity allows temporary spikes above the steady rate

Example:
    >>> import asyncio
    >>> from inf_engine.resources.rate_limit import RateLimiter
    >>>
    >>> async def make_requests():
    ...     limiter = RateLimiter(rate=10.0, max_tokens=10.0)
    ...     for i in range(5):
    ...         await limiter.acquire()
    ...         print(f"Request {i} sent")
    >>>
    >>> asyncio.run(make_requests())
"""

import asyncio
import time


class RateLimiter:
    """Token bucket rate limiter for controlling request rates.

    Implements a token bucket algorithm where tokens are continuously
    added to a bucket at a fixed rate. Each request consumes one token.
    When the bucket is empty, callers wait until tokens are available.

    The bucket has a maximum capacity (max_tokens) that limits burst size.
    Tokens accumulate when not in use, up to this maximum, allowing short
    bursts of requests above the steady-state rate.

    Args:
        rate: Token refill rate in tokens per second. This is the
            long-term average request rate the limiter will allow.
            Must be positive.
        max_tokens: Maximum bucket capacity. Controls burst size - how
            many requests can be made instantly before rate limiting
            kicks in. Must be at least 1.0. Defaults to same value as rate.

    Raises:
        ValueError: If rate is not positive or max_tokens is less than 1.0.

    Attributes:
        rate: Current token refill rate (tokens per second).
        max_tokens: Maximum bucket capacity.
        tokens: Current number of tokens in the bucket.

    Example:
        >>> limiter = RateLimiter(rate=10.0, max_tokens=5.0)
        >>> limiter.rate
        10.0
        >>> limiter.max_tokens
        5.0
        >>> limiter.tokens
        5.0

        >>> # Tokens start at max capacity
        >>> limiter = RateLimiter(rate=1.0)
        >>> limiter.tokens == limiter.max_tokens
        True
    """

    def __init__(
        self,
        rate: float = 10.0,
        max_tokens: float | None = None,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            rate: Token refill rate in tokens per second.
            max_tokens: Maximum bucket capacity. Defaults to rate value
                if not specified.
        """
        if rate <= 0:
            raise ValueError("rate must be positive")

        effective_max = max_tokens if max_tokens is not None else rate
        if effective_max < 1.0:
            raise ValueError("max_tokens must be at least 1.0")

        self.rate = rate
        self.max_tokens = effective_max
        self.tokens = self.max_tokens
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary.

        Consumes one token from the bucket. If no tokens are available,
        waits until enough time has passed for at least one token to be
        refilled.

        This method is async and will yield control to the event loop
        while waiting for tokens, allowing other coroutines to run.

        Example:
            >>> import asyncio
            >>> async def example():
            ...     limiter = RateLimiter(rate=10.0, max_tokens=2.0)
            ...     # First two calls are instant (burst capacity)
            ...     await limiter.acquire()
            ...     await limiter.acquire()
            ...     # Third call waits for refill
            ...     await limiter.acquire()
            >>> asyncio.run(example())

        Note:
            This method is thread-safe and uses asyncio.Lock to prevent
            race conditions when called concurrently from multiple tasks.
        """
        async with self._lock:
            self._refill()

            while self.tokens < 1:
                # Calculate wait time for 1 token
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self._refill()

            self.tokens -= 1

    def _refill(self) -> None:
        """Refill tokens based on elapsed time.

        Calculates how many tokens should have been added since the
        last refill and updates the bucket, capping at max_tokens.

        Note:
            This is a private method called by acquire(). It should
            only be called while holding the lock.
        """
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        # Add tokens based on elapsed time, cap at max
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
