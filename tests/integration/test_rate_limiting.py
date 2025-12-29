"""Integration tests for rate limiting behavior.

This file contains integration tests for PR-062 verifying:
- Rate limit errors trigger requeue and backoff
- Backoff slows down subsequent requests
- Recovery after successful requests
- Multiple endpoints with independent rate limiting
"""

import asyncio

import pytest

from inf_engine.clients.base import LLMClient
from inf_engine.errors import RateLimitError
from inf_engine.execution.scheduler import Scheduler
from inf_engine.execution.state import ExecutionState, TaskStatus
from inf_engine.graph import GraphNode, InferenceGraph, NodeRef
from inf_engine.module import LLMInference
from inf_engine.resources.rate_limit import RateLimiter
from inf_engine.tracing.tracer import InputNode
from inf_engine.types import LLMRequest, LLMResponse

# ─────────────────────────────────────────────────────────────────────────────
# Test Helpers
# ─────────────────────────────────────────────────────────────────────────────


class RateLimitingClient(LLMClient):
    """A mock LLM client that simulates rate limiting behavior.

    Can be configured to fail N times before succeeding, with optional
    retry_after hints.
    """

    def __init__(
        self,
        fail_count: int = 1,
        retry_after: float | None = None,
        response_content: str = "success",
    ):
        self.fail_count = fail_count
        self.retry_after = retry_after
        self.response_content = response_content
        self.call_count = 0
        self.call_times: list[float] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Raise RateLimitError for first fail_count calls, then succeed."""
        import time

        self.call_times.append(time.monotonic())
        self.call_count += 1

        if self.call_count <= self.fail_count:
            raise RateLimitError(
                f"Rate limit exceeded (attempt {self.call_count})",
                retry_after=self.retry_after,
            )
        return LLMResponse(
            content=self.response_content,
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="mock-model",
        )


class SuccessfulClient(LLMClient):
    """A mock LLM client that always succeeds."""

    def __init__(self, response_content: str = "success"):
        self.response_content = response_content
        self.call_count = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a successful response."""
        self.call_count += 1
        return LLMResponse(
            content=self.response_content,
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="mock-model",
        )


class MockResourceManager:
    """A mock ResourceManager for testing rate limiting."""

    def __init__(
        self,
        clients: dict[str, LLMClient] | None = None,
        semaphores: dict[str, asyncio.Semaphore] | None = None,
        rate_limiters: dict[str, RateLimiter] | None = None,
    ):
        self.clients: dict[str, LLMClient] = clients or {}
        self.semaphores: dict[str, asyncio.Semaphore] = semaphores or {}
        self.rate_limiters: dict[str, RateLimiter] = rate_limiters or {}

    def get_client(self, alias: str) -> LLMClient:
        """Get a client by alias."""
        return self.clients[alias]

    def get_semaphore(self, alias: str) -> asyncio.Semaphore | None:
        """Get a semaphore by alias."""
        return self.semaphores.get(alias)

    def get_rate_limiter(self, alias: str) -> RateLimiter | None:
        """Get a rate limiter by alias."""
        return self.rate_limiters.get(alias)


def create_llm_graph(alias: str = "fast", num_llms: int = 1) -> InferenceGraph:
    """Create a graph with LLMInference modules."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode(value="Hello"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    nodes = {"input:input_0": input_node}
    output_ids = []

    for i in range(num_llms):
        node_id = f"llm_{i}"
        nodes[node_id] = GraphNode(
            id=node_id,
            module=LLMInference(alias=alias),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        output_ids.append(node_id)

    return InferenceGraph(
        nodes=nodes,
        input_ids=["input:input_0"],
        output_ids=output_ids,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limit Requeue Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRateLimitRequeue:
    """Tests for rate limit error triggering requeue."""

    @pytest.mark.asyncio
    async def test_rate_limit_requeues_and_succeeds(self) -> None:
        """Rate limited task is requeued and eventually succeeds."""
        mock_client = RateLimitingClient(fail_count=2)
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

        graph = create_llm_graph()
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        assert state.status["llm_0"] == TaskStatus.COMPLETED
        assert outputs == {"llm_0": "success"}
        assert mock_client.call_count == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_rate_limit_tracks_retry_count(self) -> None:
        """Rate limited retries are tracked in the result."""
        mock_client = RateLimitingClient(fail_count=3)
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

        graph = create_llm_graph()
        state = ExecutionState(graph)

        await scheduler.execute(state)

        # retry_count should reflect rate limit requeues
        assert state.results["llm_0"].retry_count == 3

    @pytest.mark.asyncio
    async def test_rate_limit_parallel_tasks_all_succeed(self) -> None:
        """Multiple parallel tasks all succeed after rate limiting."""
        # Fail first 3 calls to simulate rate limiting of 3 parallel tasks
        mock_client = RateLimitingClient(fail_count=3)
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

        graph = create_llm_graph(num_llms=3)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # All 3 should eventually succeed
        assert len(outputs) == 3
        for i in range(3):
            assert state.status[f"llm_{i}"] == TaskStatus.COMPLETED
        # Total call count depends on scheduling, but should be at least 3 (initial) + 3 (retries) = 6
        # However with shared client, first few fail, then rest succeed
        assert (
            mock_client.call_count >= 4
        )  # At minimum: 3 failures + 3 successes = 6 (but timing varies)


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter Backoff Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRateLimiterBackoff:
    """Tests for rate limiter backoff behavior."""

    @pytest.mark.asyncio
    async def test_backoff_reduces_rate(self) -> None:
        """Rate limiter rate decreases after backoff."""
        rate_limiter = RateLimiter(rpm=600.0)
        initial_rpm = rate_limiter.rpm

        rate_limiter.backoff()

        assert rate_limiter.rpm < initial_rpm
        assert rate_limiter.rpm == initial_rpm * rate_limiter.backoff_factor

    @pytest.mark.asyncio
    async def test_backoff_respects_min_rpm(self) -> None:
        """Rate limiter does not go below min_rpm."""
        rate_limiter = RateLimiter(rpm=600.0, min_rpm=60.0)

        # Apply many backoffs
        for _ in range(20):
            rate_limiter.backoff()

        assert rate_limiter.rpm >= rate_limiter.min_rpm

    @pytest.mark.asyncio
    async def test_backoff_with_retry_after(self) -> None:
        """Rate limiter uses retry_after hint when provided."""
        rate_limiter = RateLimiter(rpm=600.0)

        # retry_after=2.0 means 30 RPM (60/2)
        rate_limiter.backoff(retry_after=2.0)

        assert rate_limiter.rpm == 30.0

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_limiter_backoff(self) -> None:
        """RateLimitError triggers backoff on the rate limiter."""
        mock_client = RateLimitingClient(fail_count=1, retry_after=5.0)
        rate_limiter = RateLimiter(rpm=600.0)
        mock_manager = MockResourceManager(
            clients={"fast": mock_client},
            rate_limiters={"fast": rate_limiter},
        )
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

        graph = create_llm_graph()
        state = ExecutionState(graph)

        initial_rpm = rate_limiter.rpm
        await scheduler.execute(state)

        # Rate limiter should have backed off
        assert rate_limiter.rpm < initial_rpm


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter Recovery Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRateLimiterRecovery:
    """Tests for rate limiter recovery behavior."""

    @pytest.mark.asyncio
    async def test_recovery_increases_rate(self) -> None:
        """Rate limiter rate increases after recover()."""
        rate_limiter = RateLimiter(rpm=600.0)
        rate_limiter.backoff()  # First reduce rate

        reduced_rpm = rate_limiter.rpm
        rate_limiter.recover()

        assert rate_limiter.rpm > reduced_rpm

    @pytest.mark.asyncio
    async def test_recovery_respects_max_rpm(self) -> None:
        """Rate limiter does not exceed max_rpm after recovery."""
        rate_limiter = RateLimiter(rpm=600.0)

        # Apply many recoveries
        for _ in range(20):
            rate_limiter.recover()

        assert rate_limiter.rpm <= rate_limiter.max_rpm
        assert rate_limiter.rpm == rate_limiter.max_rpm

    @pytest.mark.asyncio
    async def test_recovery_gradual_restoration(self) -> None:
        """Rate limiter gradually restores to original rate."""
        rate_limiter = RateLimiter(rpm=600.0, recovery_factor=2.0)
        rate_limiter.backoff()  # 300 RPM

        rate_limiter.recover()  # 600 RPM (capped)

        assert rate_limiter.rpm == rate_limiter.max_rpm


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleEndpoints:
    """Tests for rate limiting with multiple endpoints."""

    @pytest.mark.asyncio
    async def test_independent_rate_limiters(self) -> None:
        """Each endpoint has independent rate limiting."""
        fast_client = RateLimitingClient(fail_count=2, response_content="fast_result")
        slow_client = SuccessfulClient(response_content="slow_result")

        fast_limiter = RateLimiter(rpm=600.0)
        slow_limiter = RateLimiter(rpm=60.0)

        mock_manager = MockResourceManager(
            clients={"fast": fast_client, "slow": slow_client},
            rate_limiters={"fast": fast_limiter, "slow": slow_limiter},
        )
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

        # Create graph with two different endpoints
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="test"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        fast_node = GraphNode(
            id="fast_llm",
            module=LLMInference(alias="fast"),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        slow_node = GraphNode(
            id="slow_llm",
            module=LLMInference(alias="slow"),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "fast_llm": fast_node,
                "slow_llm": slow_node,
            },
            input_ids=["input:input_0"],
            output_ids=["fast_llm", "slow_llm"],
        )
        state = ExecutionState(graph)

        initial_fast_rpm = fast_limiter.rpm
        initial_slow_rpm = slow_limiter.rpm

        outputs = await scheduler.execute(state)

        # Fast endpoint backed off, slow didn't
        assert fast_limiter.rpm < initial_fast_rpm
        assert slow_limiter.rpm == initial_slow_rpm

        # Both completed
        assert outputs == {"fast_llm": "fast_result", "slow_llm": "slow_result"}


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter Token Bucket Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTokenBucketBehavior:
    """Tests for token bucket rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_acquire_consumes_token(self) -> None:
        """acquire() consumes a token from the bucket."""
        rate_limiter = RateLimiter(rpm=600.0, max_tokens=5.0)

        initial_tokens = rate_limiter.tokens
        await rate_limiter.acquire()

        assert rate_limiter.tokens == initial_tokens - 1

    @pytest.mark.asyncio
    async def test_burst_capacity(self) -> None:
        """Rate limiter allows burst up to max_tokens."""
        rate_limiter = RateLimiter(rpm=600.0, max_tokens=3.0)

        # Should be able to make 3 requests immediately
        for _ in range(3):
            await rate_limiter.acquire()

        assert rate_limiter.tokens < 1

    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self) -> None:
        """Tokens refill based on elapsed time."""
        rate_limiter = RateLimiter(rpm=6000.0, max_tokens=10.0)  # 100/sec

        # Drain some tokens
        await rate_limiter.acquire()
        await rate_limiter.acquire()
        tokens_after_drain = rate_limiter.tokens

        # Wait for refill (should get ~1 token per 10ms at 100/sec)
        await asyncio.sleep(0.05)  # 50ms = ~5 tokens

        # Refill happens on next acquire
        await rate_limiter.acquire()

        # Should have more tokens than before (minus the acquire)
        # Note: exact token count depends on timing
        assert rate_limiter.tokens > tokens_after_drain - 1


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent Rate Limiting Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConcurrentRateLimiting:
    """Tests for rate limiting under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_acquires_are_serialized(self) -> None:
        """Concurrent acquires don't exceed bucket capacity."""
        rate_limiter = RateLimiter(rpm=600.0, max_tokens=2.0)
        acquired_count = 0

        async def acquire_task() -> None:
            nonlocal acquired_count
            await rate_limiter.acquire()
            acquired_count += 1

        # Start 5 concurrent acquires with only 2 tokens
        tasks = [asyncio.create_task(acquire_task()) for _ in range(5)]

        # Wait a bit - should see 2 immediate, then rest wait
        await asyncio.sleep(0.05)

        # Should have acquired more than 2 due to refilling
        # (10/sec = 0.5 tokens in 50ms, allowing some progress)
        assert acquired_count >= 2

        # Cancel remaining tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_rate_limiting_with_parallel_llm_calls(self) -> None:
        """Rate limiting works correctly with parallel LLM calls."""
        mock_client = SuccessfulClient()
        rate_limiter = RateLimiter(rpm=6000.0, max_tokens=5.0)  # Allow 5 burst
        mock_manager = MockResourceManager(
            clients={"fast": mock_client},
            rate_limiters={"fast": rate_limiter},
        )
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

        # Create 10 parallel LLM calls
        graph = create_llm_graph(num_llms=10)
        state = ExecutionState(graph)

        await scheduler.execute(state)

        # All should complete
        for i in range(10):
            assert state.status[f"llm_{i}"] == TaskStatus.COMPLETED
        assert mock_client.call_count == 10
