"""Tests for RateLimiter with adaptive backpressure.

Tests the token bucket rate limiting algorithm including creation,
token consumption, refill behavior, waiting mechanics, and adaptive
backoff/recovery mechanisms. All rates are in RPM (requests per minute).
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
        assert limiter.rpm == 600.0
        assert limiter.max_tokens == 10.0  # 600 RPM / 60 = 10 tokens
        assert limiter.tokens == 10.0

    def test_creation_with_custom_rpm(self) -> None:
        """RateLimiter accepts custom rpm."""
        limiter = RateLimiter(rpm=300.0)
        assert limiter.rpm == 300.0
        assert limiter.max_tokens == 5.0  # Defaults to rpm/60
        assert limiter.tokens == 5.0

    def test_creation_with_custom_max_tokens(self) -> None:
        """RateLimiter accepts custom max_tokens."""
        limiter = RateLimiter(rpm=600.0, max_tokens=20.0)
        assert limiter.rpm == 600.0
        assert limiter.max_tokens == 20.0
        assert limiter.tokens == 20.0

    def test_creation_with_small_burst(self) -> None:
        """RateLimiter can have smaller burst than default."""
        limiter = RateLimiter(rpm=6000.0, max_tokens=5.0)
        assert limiter.rpm == 6000.0
        assert limiter.max_tokens == 5.0
        assert limiter.tokens == 5.0

    def test_tokens_start_at_max(self) -> None:
        """Tokens start at maximum capacity."""
        limiter = RateLimiter(rpm=600.0, max_tokens=7.5)
        assert limiter.tokens == limiter.max_tokens

    def test_invalid_rpm_zero(self) -> None:
        """RateLimiter raises ValueError for zero rpm."""
        with pytest.raises(ValueError, match="rpm must be positive"):
            RateLimiter(rpm=0.0)

    def test_invalid_rpm_negative(self) -> None:
        """RateLimiter raises ValueError for negative rpm."""
        with pytest.raises(ValueError, match="rpm must be positive"):
            RateLimiter(rpm=-300.0)


class TestRateLimiterAcquire:
    """Tests for token acquisition."""

    async def test_acquire_consumes_token(self) -> None:
        """acquire() consumes one token."""
        limiter = RateLimiter(rpm=600.0, max_tokens=5.0)
        initial_tokens = limiter.tokens
        await limiter.acquire()
        assert limiter.tokens == initial_tokens - 1

    async def test_acquire_multiple_times(self) -> None:
        """Multiple acquire() calls consume multiple tokens."""
        limiter = RateLimiter(rpm=600.0, max_tokens=5.0)
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()
        # Started with 5, consumed 3
        assert limiter.tokens < 3  # Slightly less due to refill during calls

    async def test_acquire_is_async(self) -> None:
        """acquire() is an async method."""
        limiter = RateLimiter(rpm=600.0)
        result = limiter.acquire()
        assert asyncio.iscoroutine(result)
        await result  # Clean up the coroutine

    async def test_acquire_burst(self) -> None:
        """Can acquire tokens up to burst capacity instantly."""
        limiter = RateLimiter(rpm=60.0, max_tokens=5.0)  # 1 per second

        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should be nearly instant (all within burst capacity)
        assert elapsed < 0.5

    async def test_acquire_waits_when_empty(self) -> None:
        """acquire() waits when tokens are exhausted."""
        limiter = RateLimiter(rpm=6000.0, max_tokens=1.0)  # High rate, low burst

        # Consume the only token
        await limiter.acquire()

        # Next acquire should wait for refill
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should have waited ~0.01 seconds (1 token / 100 per second)
        assert elapsed >= 0.005  # Allow some tolerance


class TestRateLimiterRefill:
    """Tests for token refill behavior."""

    async def test_tokens_refill_over_time(self) -> None:
        """Tokens refill based on elapsed time."""
        limiter = RateLimiter(rpm=6000.0, max_tokens=10.0)  # 100 per second

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
        limiter = RateLimiter(rpm=60000.0, max_tokens=5.0)  # 1000 per second

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
        limiter = RateLimiter(rpm=6000.0, max_tokens=10.0)

        async def worker() -> None:
            await limiter.acquire()

        # Launch multiple concurrent workers
        tasks = [asyncio.create_task(worker()) for _ in range(5)]
        await asyncio.gather(*tasks)

        # Should have consumed 5 tokens (approximately, with refill)
        assert limiter.tokens < 6.0

    async def test_serialized_under_lock(self) -> None:
        """Concurrent acquires are serialized by the lock."""
        limiter = RateLimiter(rpm=600.0, max_tokens=2.0)  # 10 per second
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

        # With rpm=600 (10/s) and burst=2, should take ~0.2s for 4 acquires
        # (2 instant + 2 waiting)
        assert elapsed >= 0.1


class TestRateLimiterEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_very_high_rpm(self) -> None:
        """Works with very high RPM."""
        limiter = RateLimiter(rpm=600000.0, max_tokens=100.0)  # 10000 per second
        for _ in range(50):
            await limiter.acquire()
        # Should complete nearly instantly
        assert limiter.tokens < 100.0

    async def test_very_low_rpm(self) -> None:
        """Works with very low RPM."""
        limiter = RateLimiter(rpm=6.0, max_tokens=1.0)  # 0.1 per second

        start = time.monotonic()
        await limiter.acquire()  # Instant (have 1 token)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1  # First one is instant

    def test_invalid_max_tokens_below_one(self) -> None:
        """RateLimiter raises ValueError for max_tokens < 1.0."""
        with pytest.raises(ValueError, match="max_tokens must be at least 1.0"):
            RateLimiter(rpm=600.0, max_tokens=0.5)

    async def test_fractional_max_tokens(self) -> None:
        """Handles fractional max_tokens values >= 1.0."""
        limiter = RateLimiter(rpm=600.0, max_tokens=1.5)
        assert limiter.tokens == 1.5

        # Consume 1 token, should have 0.5 left
        await limiter.acquire()
        # Tokens will be slightly more than 0.5 due to refill during call
        assert limiter.tokens < 1.0
        assert limiter.tokens >= 0.0

    async def test_single_token_bucket(self) -> None:
        """Works with single-token bucket."""
        limiter = RateLimiter(rpm=60000.0, max_tokens=1.0)  # 1000 per second

        # Can acquire one instantly
        await limiter.acquire()
        assert limiter.tokens < 1.0

        # Second acquire waits briefly
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed >= 0.0005  # At least 0.5ms wait


class TestRateLimiterAdaptiveParameters:
    """Tests for adaptive rate limiting parameters."""

    def test_creation_with_adaptive_defaults(self) -> None:
        """RateLimiter has correct adaptive defaults."""
        limiter = RateLimiter(rpm=600.0)
        assert limiter.max_rpm == 600.0
        assert limiter.min_rpm == 6.0
        assert limiter.recovery_factor == 1.1
        assert limiter.backoff_factor == 0.5

    def test_creation_with_custom_min_rpm(self) -> None:
        """RateLimiter accepts custom min_rpm."""
        limiter = RateLimiter(rpm=600.0, min_rpm=60.0)
        assert limiter.min_rpm == 60.0

    def test_creation_with_custom_recovery_factor(self) -> None:
        """RateLimiter accepts custom recovery_factor."""
        limiter = RateLimiter(rpm=600.0, recovery_factor=1.5)
        assert limiter.recovery_factor == 1.5

    def test_creation_with_custom_backoff_factor(self) -> None:
        """RateLimiter accepts custom backoff_factor."""
        limiter = RateLimiter(rpm=600.0, backoff_factor=0.25)
        assert limiter.backoff_factor == 0.25

    def test_invalid_min_rpm_zero(self) -> None:
        """RateLimiter raises ValueError for zero min_rpm."""
        with pytest.raises(ValueError, match="min_rpm must be positive"):
            RateLimiter(rpm=600.0, min_rpm=0.0)

    def test_invalid_min_rpm_negative(self) -> None:
        """RateLimiter raises ValueError for negative min_rpm."""
        with pytest.raises(ValueError, match="min_rpm must be positive"):
            RateLimiter(rpm=600.0, min_rpm=-60.0)


class TestRateLimiterBackoff:
    """Tests for rate backoff behavior."""

    def test_backoff_reduces_rpm(self) -> None:
        """backoff() reduces rpm by backoff_factor."""
        limiter = RateLimiter(rpm=600.0, backoff_factor=0.5)
        limiter.backoff()
        assert limiter.rpm == 300.0

    def test_backoff_multiple_times(self) -> None:
        """Multiple backoffs compound."""
        limiter = RateLimiter(rpm=600.0, backoff_factor=0.5)
        limiter.backoff()
        assert limiter.rpm == 300.0
        limiter.backoff()
        assert limiter.rpm == 150.0
        limiter.backoff()
        assert limiter.rpm == 75.0

    def test_backoff_respects_min_rpm(self) -> None:
        """Backoff doesn't go below min_rpm."""
        limiter = RateLimiter(rpm=60.0, min_rpm=30.0, backoff_factor=0.1)
        limiter.backoff()
        assert limiter.rpm == 30.0  # Capped at min_rpm
        limiter.backoff()
        assert limiter.rpm == 30.0  # Still at min_rpm

    def test_backoff_with_retry_after(self) -> None:
        """backoff() uses retry_after to set rpm."""
        limiter = RateLimiter(rpm=600.0)
        limiter.backoff(retry_after=2.0)  # 1 request per 2 seconds = 30 RPM
        assert limiter.rpm == 30.0

    def test_backoff_with_retry_after_lower_than_current(self) -> None:
        """backoff() with retry_after only reduces rpm."""
        limiter = RateLimiter(rpm=600.0)
        limiter.backoff(retry_after=0.05)  # Would suggest 1200 RPM, higher than current
        assert limiter.rpm == 600.0  # RPM unchanged (already lower)

    def test_backoff_with_retry_after_respects_min_rpm(self) -> None:
        """backoff() with retry_after respects min_rpm."""
        limiter = RateLimiter(rpm=600.0, min_rpm=30.0)
        limiter.backoff(retry_after=10.0)  # Would suggest 6 RPM
        assert limiter.rpm == 30.0  # Capped at min_rpm

    def test_backoff_with_retry_after_zero(self) -> None:
        """backoff() with zero retry_after uses backoff_factor."""
        limiter = RateLimiter(rpm=600.0, backoff_factor=0.5)
        limiter.backoff(retry_after=0.0)
        assert limiter.rpm == 300.0  # Used backoff_factor

    def test_backoff_with_retry_after_negative(self) -> None:
        """backoff() with negative retry_after uses backoff_factor."""
        limiter = RateLimiter(rpm=600.0, backoff_factor=0.5)
        limiter.backoff(retry_after=-1.0)
        assert limiter.rpm == 300.0  # Used backoff_factor


class TestRateLimiterRecover:
    """Tests for rate recovery behavior."""

    def test_recover_increases_rpm(self) -> None:
        """recover() increases rpm by recovery_factor."""
        limiter = RateLimiter(rpm=600.0, recovery_factor=1.5)
        limiter.rpm = 300.0  # Simulate backoff
        limiter.recover()
        assert limiter.rpm == 450.0

    def test_recover_respects_max_rpm(self) -> None:
        """Recover doesn't exceed max_rpm."""
        limiter = RateLimiter(rpm=600.0, recovery_factor=1.5)
        limiter.rpm = 540.0  # Close to max
        limiter.recover()
        assert limiter.rpm == 600.0  # Capped at max_rpm

    def test_recover_at_max_rpm(self) -> None:
        """Recover at max_rpm stays at max_rpm."""
        limiter = RateLimiter(rpm=600.0, recovery_factor=1.5)
        # Already at max rpm
        limiter.recover()
        assert limiter.rpm == 600.0

    def test_recover_multiple_times(self) -> None:
        """Multiple recovers compound up to max."""
        limiter = RateLimiter(rpm=600.0, recovery_factor=2.0)
        limiter.rpm = 60.0  # Heavily backed off
        limiter.recover()
        assert limiter.rpm == 120.0
        limiter.recover()
        assert limiter.rpm == 240.0
        limiter.recover()
        assert limiter.rpm == 480.0
        limiter.recover()
        assert limiter.rpm == 600.0  # Capped at max


class TestRateLimiterBackoffRecoverCycle:
    """Tests for combined backoff and recovery cycles."""

    def test_backoff_then_recover(self) -> None:
        """Can recover after backoff."""
        limiter = RateLimiter(rpm=600.0, backoff_factor=0.5, recovery_factor=1.1)
        original_rpm = limiter.rpm

        limiter.backoff()
        assert limiter.rpm == 300.0

        # Multiple recovers to get back
        for _ in range(20):  # Enough iterations
            limiter.recover()

        assert limiter.rpm == original_rpm

    def test_gradual_recovery_after_multiple_backoffs(self) -> None:
        """Recovery is gradual after multiple backoffs."""
        limiter = RateLimiter(rpm=600.0, backoff_factor=0.5, recovery_factor=1.1)

        # Multiple backoffs
        limiter.backoff()
        limiter.backoff()
        limiter.backoff()
        assert limiter.rpm == 75.0  # 600 * 0.5^3

        # First recovery
        limiter.recover()
        assert limiter.rpm == pytest.approx(82.5, rel=0.01)  # 75 * 1.1

    def test_rpm_oscillation(self) -> None:
        """RPM can oscillate between backoff and recovery."""
        limiter = RateLimiter(rpm=600.0, backoff_factor=0.5, recovery_factor=2.0)

        limiter.backoff()  # 600 -> 300
        assert limiter.rpm == 300.0
        limiter.recover()  # 300 -> 600
        assert limiter.rpm == 600.0
        limiter.backoff()  # 600 -> 300
        assert limiter.rpm == 300.0
        limiter.recover()  # 300 -> 600
        assert limiter.rpm == 600.0
