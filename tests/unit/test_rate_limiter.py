"""Tests for RateLimiter token bucket implementation.

Tests the token bucket rate limiting algorithm including creation,
token consumption, refill behavior, and waiting mechanics.
"""

import asyncio
import time

import pytest

from inf_engine.resources.rate_limit import RateLimiter


class TestRateLimiterCreation:
    """Tests for RateLimiter initialization."""

    def test_creation_with_defaults(self) -> None:
        """RateLimiter can be created with default arguments."""
        limiter = RateLimiter()
        assert limiter.rate == 10.0
        assert limiter.max_tokens == 10.0
        assert limiter.tokens == 10.0

    def test_creation_with_custom_rate(self) -> None:
        """RateLimiter accepts custom rate."""
        limiter = RateLimiter(rate=5.0)
        assert limiter.rate == 5.0
        assert limiter.max_tokens == 5.0  # Defaults to rate
        assert limiter.tokens == 5.0

    def test_creation_with_custom_max_tokens(self) -> None:
        """RateLimiter accepts custom max_tokens."""
        limiter = RateLimiter(rate=10.0, max_tokens=20.0)
        assert limiter.rate == 10.0
        assert limiter.max_tokens == 20.0
        assert limiter.tokens == 20.0

    def test_creation_with_small_burst(self) -> None:
        """RateLimiter can have smaller burst than rate."""
        limiter = RateLimiter(rate=100.0, max_tokens=5.0)
        assert limiter.rate == 100.0
        assert limiter.max_tokens == 5.0
        assert limiter.tokens == 5.0

    def test_tokens_start_at_max(self) -> None:
        """Tokens start at maximum capacity."""
        limiter = RateLimiter(rate=10.0, max_tokens=7.5)
        assert limiter.tokens == limiter.max_tokens

    def test_invalid_rate_zero(self) -> None:
        """RateLimiter raises ValueError for zero rate."""
        with pytest.raises(ValueError, match="rate must be positive"):
            RateLimiter(rate=0.0)

    def test_invalid_rate_negative(self) -> None:
        """RateLimiter raises ValueError for negative rate."""
        with pytest.raises(ValueError, match="rate must be positive"):
            RateLimiter(rate=-5.0)


class TestRateLimiterAcquire:
    """Tests for token acquisition."""

    async def test_acquire_consumes_token(self) -> None:
        """acquire() consumes one token."""
        limiter = RateLimiter(rate=10.0, max_tokens=5.0)
        initial_tokens = limiter.tokens
        await limiter.acquire()
        assert limiter.tokens == initial_tokens - 1

    async def test_acquire_multiple_times(self) -> None:
        """Multiple acquire() calls consume multiple tokens."""
        limiter = RateLimiter(rate=10.0, max_tokens=5.0)
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()
        # Started with 5, consumed 3
        assert limiter.tokens < 3  # Slightly less due to refill during calls

    async def test_acquire_is_async(self) -> None:
        """acquire() is an async method."""
        limiter = RateLimiter(rate=10.0)
        result = limiter.acquire()
        assert asyncio.iscoroutine(result)
        await result  # Clean up the coroutine

    async def test_acquire_burst(self) -> None:
        """Can acquire tokens up to burst capacity instantly."""
        limiter = RateLimiter(rate=1.0, max_tokens=5.0)

        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should be nearly instant (all within burst capacity)
        assert elapsed < 0.5

    async def test_acquire_waits_when_empty(self) -> None:
        """acquire() waits when tokens are exhausted."""
        limiter = RateLimiter(rate=100.0, max_tokens=1.0)  # High rate, low burst

        # Consume the only token
        await limiter.acquire()

        # Next acquire should wait for refill
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should have waited ~0.01 seconds (1 token / 100 rate)
        assert elapsed >= 0.005  # Allow some tolerance


class TestRateLimiterRefill:
    """Tests for token refill behavior."""

    async def test_tokens_refill_over_time(self) -> None:
        """Tokens refill based on elapsed time."""
        limiter = RateLimiter(rate=100.0, max_tokens=10.0)

        # Consume all tokens
        for _ in range(10):
            await limiter.acquire()

        # Wait for some refill
        await asyncio.sleep(0.05)  # 50ms = 5 tokens at 100/s

        # Force refill calculation by accessing with lock
        async with limiter._lock:
            limiter._refill()

        # Should have ~5 tokens (100 * 0.05)
        assert limiter.tokens >= 4.0
        assert limiter.tokens <= 6.0

    async def test_tokens_cap_at_max(self) -> None:
        """Tokens don't exceed max_tokens."""
        limiter = RateLimiter(rate=1000.0, max_tokens=5.0)

        # Already at max
        assert limiter.tokens == 5.0

        # Wait (would add many tokens at this rate)
        await asyncio.sleep(0.01)

        # Force refill
        async with limiter._lock:
            limiter._refill()

        # Still capped at max
        assert limiter.tokens == 5.0


class TestRateLimiterConcurrency:
    """Tests for concurrent access."""

    async def test_concurrent_acquire(self) -> None:
        """Multiple concurrent acquire() calls are safe."""
        limiter = RateLimiter(rate=100.0, max_tokens=10.0)

        async def worker() -> None:
            await limiter.acquire()

        # Launch multiple concurrent workers
        tasks = [asyncio.create_task(worker()) for _ in range(5)]
        await asyncio.gather(*tasks)

        # Should have consumed 5 tokens (approximately, with refill)
        assert limiter.tokens < 6.0

    async def test_serialized_under_lock(self) -> None:
        """Concurrent acquires are serialized by the lock."""
        limiter = RateLimiter(rate=10.0, max_tokens=2.0)
        acquire_order: list[int] = []
        start_time = time.monotonic()

        async def worker(idx: int) -> None:
            await limiter.acquire()
            acquire_order.append(idx)

        # Start 4 workers - only 2 can proceed immediately
        tasks = [asyncio.create_task(worker(i)) for i in range(4)]
        await asyncio.gather(*tasks)

        # All 4 should have acquired
        assert len(acquire_order) == 4
        elapsed = time.monotonic() - start_time

        # With rate=10 and burst=2, should take ~0.2s for 4 acquires
        # (2 instant + 2 waiting)
        assert elapsed >= 0.1


class TestRateLimiterEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_very_high_rate(self) -> None:
        """Works with very high rates."""
        limiter = RateLimiter(rate=10000.0, max_tokens=100.0)
        for _ in range(50):
            await limiter.acquire()
        # Should complete nearly instantly
        assert limiter.tokens < 100.0

    async def test_very_low_rate(self) -> None:
        """Works with very low rates."""
        limiter = RateLimiter(rate=0.1, max_tokens=1.0)  # 1 per 10 seconds

        start = time.monotonic()
        await limiter.acquire()  # Instant (have 1 token)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1  # First one is instant

    def test_invalid_max_tokens_below_one(self) -> None:
        """RateLimiter raises ValueError for max_tokens < 1.0."""
        with pytest.raises(ValueError, match="max_tokens must be at least 1.0"):
            RateLimiter(rate=10.0, max_tokens=0.5)

    async def test_fractional_max_tokens(self) -> None:
        """Handles fractional max_tokens values >= 1.0."""
        limiter = RateLimiter(rate=10.0, max_tokens=1.5)
        assert limiter.tokens == 1.5

        # Consume 1 token, should have 0.5 left
        await limiter.acquire()
        # Tokens will be slightly more than 0.5 due to refill during call
        assert limiter.tokens < 1.0
        assert limiter.tokens >= 0.0

    async def test_single_token_bucket(self) -> None:
        """Works with single-token bucket."""
        limiter = RateLimiter(rate=1000.0, max_tokens=1.0)

        # Can acquire one instantly
        await limiter.acquire()
        assert limiter.tokens < 1.0

        # Second acquire waits briefly
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed >= 0.0005  # At least 0.5ms wait
