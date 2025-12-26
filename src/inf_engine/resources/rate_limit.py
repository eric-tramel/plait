"""Token bucket rate limiter with adaptive backpressure for LLM endpoints.

This module provides a token bucket rate limiter that controls the rate
of requests to LLM endpoints. The algorithm allows for bursting up to
a maximum capacity while maintaining a steady long-term rate.

The limiter supports adaptive backpressure:
    - When rate limits are hit, `backoff()` reduces the request rate
    - After successful requests, `recover()` gradually restores the rate
    - This allows automatic adjustment to API rate limits

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
    ...         limiter.recover()  # Gradually restore rate after success
    >>>
    >>> asyncio.run(make_requests())
"""

import asyncio
import time


class RateLimiter:
    """Token bucket rate limiter with adaptive backpressure.

    Implements a token bucket algorithm where tokens are continuously
    added to a bucket at a fixed rate. Each request consumes one token.
    When the bucket is empty, callers wait until tokens are available.

    The bucket has a maximum capacity (max_tokens) that limits burst size.
    Tokens accumulate when not in use, up to this maximum, allowing short
    bursts of requests above the steady-state rate.

    Supports adaptive rate adjustment:
        - `backoff()`: Reduce rate when hitting API rate limits
        - `recover()`: Gradually restore rate after successful requests

    Args:
        rate: Initial token refill rate in tokens per second. This is the
            long-term average request rate the limiter will allow.
            Must be positive. Also used as max_rate for recovery.
        max_tokens: Maximum bucket capacity. Controls burst size - how
            many requests can be made instantly before rate limiting
            kicks in. Must be at least 1.0. Defaults to same value as rate.
        min_rate: Minimum rate after backoff. Rate will not go below this
            value regardless of how many backoffs occur. Defaults to 0.1.
        recovery_factor: Multiplier for rate on each recover() call.
            Should be > 1.0 to gradually increase rate. Defaults to 1.1.
        backoff_factor: Multiplier for rate on each backoff() call.
            Should be < 1.0 to reduce rate. Defaults to 0.5.

    Raises:
        ValueError: If rate is not positive, max_tokens < 1.0, or
            min_rate is not positive.

    Attributes:
        rate: Current token refill rate (tokens per second). Changes
            with backoff() and recover() calls.
        max_rate: Maximum rate (set from initial rate). Recovery cannot
            exceed this value.
        min_rate: Minimum rate. Backoff cannot go below this value.
        max_tokens: Maximum bucket capacity.
        tokens: Current number of tokens in the bucket.
        recovery_factor: Multiplier applied on recover().
        backoff_factor: Multiplier applied on backoff().

    Example:
        >>> limiter = RateLimiter(rate=10.0, max_tokens=5.0)
        >>> limiter.rate
        10.0
        >>> limiter.max_tokens
        5.0

        >>> # Backoff reduces rate
        >>> limiter.backoff()
        >>> limiter.rate
        5.0

        >>> # Recover gradually restores rate
        >>> limiter.recover()
        >>> limiter.rate
        5.5
    """

    def __init__(
        self,
        rate: float = 10.0,
        max_tokens: float | None = None,
        min_rate: float = 0.1,
        recovery_factor: float = 1.1,
        backoff_factor: float = 0.5,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            rate: Initial token refill rate in tokens per second.
            max_tokens: Maximum bucket capacity. Defaults to rate value
                if not specified.
            min_rate: Minimum rate after backoff. Defaults to 0.1.
            recovery_factor: Multiplier for rate recovery. Defaults to 1.1.
            backoff_factor: Multiplier for rate backoff. Defaults to 0.5.
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        if min_rate <= 0:
            raise ValueError("min_rate must be positive")

        effective_max = max_tokens if max_tokens is not None else rate
        if effective_max < 1.0:
            raise ValueError("max_tokens must be at least 1.0")

        self.rate = rate
        self.max_rate = rate
        self.min_rate = min_rate
        self.max_tokens = effective_max
        self.tokens = self.max_tokens
        self.recovery_factor = recovery_factor
        self.backoff_factor = backoff_factor
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

    def backoff(self, retry_after: float | None = None) -> None:
        """Reduce rate after hitting API backpressure.

        Call this method when an API returns a rate limit error (e.g., HTTP 429).
        The rate is reduced to slow down subsequent requests.

        If `retry_after` is provided (typically from the API's Retry-After header),
        the rate is set to allow approximately one request per retry_after seconds.
        Otherwise, the rate is multiplied by the backoff_factor.

        The rate will never go below min_rate.

        Args:
            retry_after: Optional number of seconds to wait before retrying.
                If provided, rate is set to min(current_rate, 1.0/retry_after).
                If None, rate is multiplied by backoff_factor.

        Example:
            >>> limiter = RateLimiter(rate=10.0)
            >>> limiter.backoff()
            >>> limiter.rate
            5.0
            >>> limiter.backoff()
            >>> limiter.rate
            2.5

            >>> # With retry_after hint
            >>> limiter = RateLimiter(rate=10.0)
            >>> limiter.backoff(retry_after=2.0)
            >>> limiter.rate
            0.5
        """
        if retry_after is not None and retry_after > 0:
            # Use server-provided retry time to estimate safe rate
            suggested_rate = 1.0 / retry_after
            self.rate = max(self.min_rate, min(self.rate, suggested_rate))
        else:
            # Apply multiplicative backoff
            self.rate = max(self.min_rate, self.rate * self.backoff_factor)

    def recover(self) -> None:
        """Gradually increase rate after successful requests.

        Call this method after a successful API request to slowly restore
        the rate toward max_rate. The rate is multiplied by recovery_factor
        but will never exceed max_rate.

        Typical usage is to call recover() after each successful request
        to gradually undo the effects of previous backoff() calls.

        Example:
            >>> limiter = RateLimiter(rate=10.0, recovery_factor=1.5)
            >>> limiter.backoff()  # rate = 5.0
            >>> limiter.recover()  # rate = 7.5
            >>> limiter.recover()  # rate = 10.0 (capped at max_rate)
            >>> limiter.recover()  # rate = 10.0 (still capped)
        """
        self.rate = min(self.max_rate, self.rate * self.recovery_factor)
