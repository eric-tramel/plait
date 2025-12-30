"""Integration tests for reliability features: timeout, retry, and cancellation.

This file contains integration tests for PR-062 verifying:
- Task timeout behavior and cascade cancellation
- Retry logic for transient failures
- Exponential backoff between retries
- Cancellation of dependent tasks on failure
- Combined timeout and retry scenarios
"""

import asyncio
import time

import pytest

from inf_engine.clients.base import LLMClient
from inf_engine.errors import TransientError
from inf_engine.execution.scheduler import RateLimiterProtocol, Scheduler
from inf_engine.execution.state import ExecutionState, TaskStatus
from inf_engine.graph import GraphNode, InferenceGraph, NodeRef
from inf_engine.module import InferenceModule, LLMInference
from inf_engine.tracing.tracer import InputNode
from inf_engine.types import LLMRequest, LLMResponse

# ─────────────────────────────────────────────────────────────────────────────
# Test Helpers
# ─────────────────────────────────────────────────────────────────────────────


class EchoModule(InferenceModule):
    """Simple test module that echoes input."""

    def forward(self, text: str) -> str:
        return f"{text}_echo"


class TimeoutModule(InferenceModule):
    """A module that takes too long, designed to trigger timeout."""

    async def forward(self, text: str) -> str:
        await asyncio.sleep(10.0)  # Will be cancelled by timeout
        return f"{text}_done"


class SlowModule(InferenceModule):
    """A module that takes a controllable amount of time."""

    def __init__(self, delay: float = 0.1) -> None:
        super().__init__()
        self.delay = delay

    async def forward(self, text: str) -> str:
        await asyncio.sleep(self.delay)
        return f"{text}_slow"


class TransientFailingModule(InferenceModule):
    """A module that fails transiently N times before succeeding."""

    def __init__(self, fail_count: int = 1) -> None:
        super().__init__()
        self.fail_count = fail_count
        self.call_count = 0
        self.call_times: list[float] = []

    async def forward(self, text: str) -> str:
        self.call_times.append(time.monotonic())
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise TransientError(f"Transient failure (attempt {self.call_count})")
        return f"{text}_success"


class PermanentFailingModule(InferenceModule):
    """A module that always fails with a non-transient error."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(self, text: str) -> str:
        self.call_count += 1
        raise ValueError("Permanent failure")


class TransientLLMClient(LLMClient):
    """A mock LLM client that fails transiently."""

    def __init__(self, fail_count: int = 1):
        self.fail_count = fail_count
        self.call_count = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise TransientError(f"Connection failed (attempt {self.call_count})")
        return LLMResponse(
            content="success",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="mock-model",
        )


class TimeoutLLMClient(LLMClient):
    """A mock LLM client that takes too long."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(10.0)  # Will timeout
        return LLMResponse(
            content="never reached",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="mock-model",
        )


class MockRateLimiter:
    """A mock rate limiter for testing."""

    def backoff(self, retry_after: float | None = None) -> None:
        """Record backoff call."""
        pass


class MockResourceManager:
    """A mock ResourceManager for testing."""

    def __init__(self, clients: dict[str, LLMClient] | None = None):
        self.clients: dict[str, LLMClient] = clients or {}
        self.semaphores: dict[str, asyncio.Semaphore] = {}
        self.rate_limiters: dict[str, MockRateLimiter] = {}

    def get_client(self, alias: str) -> LLMClient:
        return self.clients[alias]

    def get_semaphore(self, alias: str) -> asyncio.Semaphore | None:
        return self.semaphores.get(alias)

    def get_rate_limiter(self, alias: str) -> RateLimiterProtocol | None:
        return self.rate_limiters.get(alias)


def create_graph_with_module(
    module: InferenceModule, input_value: str = "test"
) -> InferenceGraph:
    """Create a simple graph with one module."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode(value=input_value),
        args=(),
        kwargs={},
        dependencies=[],
    )
    process_node = GraphNode(
        id="process_1",
        module=module,
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    return InferenceGraph(
        nodes={"input:input_0": input_node, "process_1": process_node},
        input_ids=["input:input_0"],
        output_ids=["process_1"],
    )


def create_linear_graph_with_modules(
    modules: list[InferenceModule], input_value: str = "test"
) -> InferenceGraph:
    """Create a linear graph: input -> m1 -> m2 -> m3..."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode(value=input_value),
        args=(),
        kwargs={},
        dependencies=[],
    )
    nodes = {"input:input_0": input_node}
    prev_id = "input:input_0"

    for i, module in enumerate(modules):
        node_id = f"module_{i}"
        nodes[node_id] = GraphNode(
            id=node_id,
            module=module,
            args=(NodeRef(prev_id),),
            kwargs={},
            dependencies=[prev_id],
        )
        prev_id = node_id

    return InferenceGraph(
        nodes=nodes,
        input_ids=["input:input_0"],
        output_ids=[prev_id],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task Timeout Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskTimeout:
    """Tests for task timeout behavior."""

    @pytest.mark.asyncio
    async def test_task_timeout_triggers_failure(self) -> None:
        """Task exceeding timeout is marked as failed."""
        scheduler = Scheduler(max_concurrent=10, task_timeout=0.01)
        graph = create_graph_with_module(TimeoutModule())
        state = ExecutionState(graph)

        await scheduler.execute(state)

        assert state.status["process_1"] == TaskStatus.FAILED
        assert "process_1" in state.errors
        assert isinstance(state.errors["process_1"], TimeoutError)

    @pytest.mark.asyncio
    async def test_task_within_timeout_succeeds(self) -> None:
        """Task completing within timeout succeeds."""
        scheduler = Scheduler(max_concurrent=10, task_timeout=1.0)
        graph = create_graph_with_module(SlowModule(delay=0.01))
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        assert state.status["process_1"] == TaskStatus.COMPLETED
        assert outputs == {"process_1": "test_slow"}

    @pytest.mark.asyncio
    async def test_timeout_cancels_dependent_tasks(self) -> None:
        """Timeout causes dependent tasks to be cancelled."""
        scheduler = Scheduler(max_concurrent=10, task_timeout=0.01)

        timeout_module = TimeoutModule()
        dependent_module = EchoModule()

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="test"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        slow_node = GraphNode(
            id="slow_1",
            module=timeout_module,
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        dependent_node = GraphNode(
            id="dependent_2",
            module=dependent_module,
            args=(NodeRef("slow_1"),),
            kwargs={},
            dependencies=["slow_1"],
        )

        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "slow_1": slow_node,
                "dependent_2": dependent_node,
            },
            input_ids=["input:input_0"],
            output_ids=["dependent_2"],
        )
        state = ExecutionState(graph)

        await scheduler.execute(state)

        assert state.status["slow_1"] == TaskStatus.FAILED
        assert state.status["dependent_2"] == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_timeout_invokes_on_error_callback(self) -> None:
        """Timeout invokes the on_error callback."""
        scheduler = Scheduler(max_concurrent=10, task_timeout=0.01)
        graph = create_graph_with_module(TimeoutModule())
        state = ExecutionState(graph)

        errors: list[tuple[str, Exception]] = []

        def on_error(node_id: str, error: Exception) -> None:
            errors.append((node_id, error))

        await scheduler.execute(state, on_error=on_error)

        assert len(errors) == 1
        assert errors[0][0] == "process_1"
        assert isinstance(errors[0][1], TimeoutError)

    @pytest.mark.asyncio
    async def test_no_timeout_allows_long_tasks(self) -> None:
        """Without timeout, long tasks complete normally."""
        scheduler = Scheduler(max_concurrent=10)  # No timeout
        graph = create_graph_with_module(SlowModule(delay=0.1))
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        assert state.status["process_1"] == TaskStatus.COMPLETED
        assert outputs == {"process_1": "test_slow"}


# ─────────────────────────────────────────────────────────────────────────────
# Transient Error Retry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTransientErrorRetry:
    """Tests for retry behavior on transient errors."""

    @pytest.mark.asyncio
    async def test_transient_error_retries_and_succeeds(self) -> None:
        """Transient error triggers retry and eventually succeeds."""
        module = TransientFailingModule(fail_count=2)
        scheduler = Scheduler(
            max_concurrent=10, max_task_retries=3, task_retry_delay=0.001
        )
        graph = create_graph_with_module(module)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        assert state.status["process_1"] == TaskStatus.COMPLETED
        assert outputs == {"process_1": "test_success"}
        assert module.call_count == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_transient_error_exhausts_retries(self) -> None:
        """Task fails after max retries exhausted."""
        module = TransientFailingModule(fail_count=10)  # More than retries
        scheduler = Scheduler(
            max_concurrent=10, max_task_retries=2, task_retry_delay=0.001
        )
        graph = create_graph_with_module(module)
        state = ExecutionState(graph)

        await scheduler.execute(state)

        assert state.status["process_1"] == TaskStatus.FAILED
        assert isinstance(state.errors["process_1"], TransientError)
        # Initial call + 2 retries = 3 total calls
        assert module.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_count_tracked_in_result(self) -> None:
        """Successful retry tracks retry count in result."""
        module = TransientFailingModule(fail_count=2)
        scheduler = Scheduler(
            max_concurrent=10, max_task_retries=5, task_retry_delay=0.001
        )
        graph = create_graph_with_module(module)
        state = ExecutionState(graph)

        await scheduler.execute(state)

        assert state.results["process_1"].retry_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_when_disabled(self) -> None:
        """Transient error fails immediately when retries disabled."""
        module = TransientFailingModule(fail_count=1)
        scheduler = Scheduler(max_concurrent=10, max_task_retries=0)
        graph = create_graph_with_module(module)
        state = ExecutionState(graph)

        await scheduler.execute(state)

        assert state.status["process_1"] == TaskStatus.FAILED
        assert module.call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_permanent_error_not_retried(self) -> None:
        """Non-transient errors are not retried."""
        module = PermanentFailingModule()
        scheduler = Scheduler(
            max_concurrent=10, max_task_retries=5, task_retry_delay=0.001
        )
        graph = create_graph_with_module(module)
        state = ExecutionState(graph)

        await scheduler.execute(state)

        assert state.status["process_1"] == TaskStatus.FAILED
        assert isinstance(state.errors["process_1"], ValueError)
        assert module.call_count == 1  # Not retried


# ─────────────────────────────────────────────────────────────────────────────
# Exponential Backoff Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExponentialBackoff:
    """Tests for exponential backoff between retries."""

    @pytest.mark.asyncio
    async def test_retry_uses_exponential_backoff(self) -> None:
        """Retries use exponential backoff delays."""
        module = TransientFailingModule(fail_count=2)
        base_delay = 0.02  # 20ms
        scheduler = Scheduler(
            max_concurrent=10, max_task_retries=5, task_retry_delay=base_delay
        )
        graph = create_graph_with_module(module)
        state = ExecutionState(graph)

        await scheduler.execute(state)

        # Check delay between calls increases
        assert len(module.call_times) == 3
        delay1 = module.call_times[1] - module.call_times[0]
        delay2 = module.call_times[2] - module.call_times[1]

        # First retry delay should be ~base_delay, second ~2*base_delay
        # Allow some timing slack
        assert delay1 >= base_delay * 0.8
        assert delay2 >= delay1 * 1.5  # Should be roughly double

    @pytest.mark.asyncio
    async def test_backoff_timing_accumulates(self) -> None:
        """Total time increases with more retries."""
        module = TransientFailingModule(fail_count=3)
        scheduler = Scheduler(
            max_concurrent=10, max_task_retries=5, task_retry_delay=0.01
        )
        graph = create_graph_with_module(module)
        state = ExecutionState(graph)

        start = time.perf_counter()
        await scheduler.execute(state)
        elapsed = time.perf_counter() - start

        # Total delay: 10ms + 20ms + 40ms = 70ms minimum
        assert elapsed >= 0.06


# ─────────────────────────────────────────────────────────────────────────────
# Cancellation Cascade Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCancellationCascade:
    """Tests for cancellation cascading to dependent tasks."""

    @pytest.mark.asyncio
    async def test_failure_cancels_all_descendants(self) -> None:
        """Failure cancels all dependent tasks in the chain."""
        failing_module = TransientFailingModule(fail_count=10)
        scheduler = Scheduler(
            max_concurrent=10, max_task_retries=1, task_retry_delay=0.001
        )

        modules = [failing_module, EchoModule(), EchoModule()]
        graph = create_linear_graph_with_modules(modules)
        state = ExecutionState(graph)

        await scheduler.execute(state)

        assert state.status["module_0"] == TaskStatus.FAILED
        assert state.status["module_1"] == TaskStatus.CANCELLED
        assert state.status["module_2"] == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_parallel_branch_not_affected_by_failure(self) -> None:
        """Parallel branches are not cancelled by failure in sibling."""
        failing_module = TransientFailingModule(fail_count=10)
        success_module = EchoModule()

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="test"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        failing_node = GraphNode(
            id="failing_branch",
            module=failing_module,
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        success_node = GraphNode(
            id="success_branch",
            module=success_module,
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )

        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "failing_branch": failing_node,
                "success_branch": success_node,
            },
            input_ids=["input:input_0"],
            output_ids=["failing_branch", "success_branch"],
        )
        state = ExecutionState(graph)
        scheduler = Scheduler(
            max_concurrent=10, max_task_retries=0, task_retry_delay=0.001
        )

        outputs = await scheduler.execute(state)

        # Failing branch failed, success branch succeeded
        assert state.status["failing_branch"] == TaskStatus.FAILED
        assert state.status["success_branch"] == TaskStatus.COMPLETED
        assert outputs == {"success_branch": "test_echo"}


# ─────────────────────────────────────────────────────────────────────────────
# Combined Timeout and Retry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCombinedTimeoutRetry:
    """Tests for combined timeout and retry behavior."""

    @pytest.mark.asyncio
    async def test_timeout_does_not_trigger_retry(self) -> None:
        """TimeoutError is not a TransientError and should not retry."""
        scheduler = Scheduler(
            max_concurrent=10,
            task_timeout=0.01,
            max_task_retries=5,
            task_retry_delay=0.001,
        )
        graph = create_graph_with_module(TimeoutModule())
        state = ExecutionState(graph)

        await scheduler.execute(state)

        # Timeout fails immediately without retry
        assert state.status["process_1"] == TaskStatus.FAILED
        assert isinstance(state.errors["process_1"], TimeoutError)

    @pytest.mark.asyncio
    async def test_retry_within_timeout(self) -> None:
        """Retries complete within overall timeout allowance."""
        module = TransientFailingModule(fail_count=2)
        scheduler = Scheduler(
            max_concurrent=10,
            task_timeout=1.0,  # Generous timeout
            max_task_retries=5,
            task_retry_delay=0.001,
        )
        graph = create_graph_with_module(module)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        assert state.status["process_1"] == TaskStatus.COMPLETED
        assert outputs == {"process_1": "test_success"}


# ─────────────────────────────────────────────────────────────────────────────
# LLM Module Reliability Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMModuleReliability:
    """Tests for reliability features with LLM modules."""

    @pytest.mark.asyncio
    async def test_llm_transient_error_retries(self) -> None:
        """LLM client transient errors trigger retry."""
        mock_client = TransientLLMClient(fail_count=2)
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(
            resource_manager=mock_manager,
            max_concurrent=10,
            max_task_retries=5,
            task_retry_delay=0.001,
        )

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="Hello"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        llm_node = GraphNode(
            id="llm_1",
            module=LLMInference(alias="fast"),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={"input:input_0": input_node, "llm_1": llm_node},
            input_ids=["input:input_0"],
            output_ids=["llm_1"],
        )
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        assert state.status["llm_1"] == TaskStatus.COMPLETED
        assert outputs == {"llm_1": "success"}
        assert mock_client.call_count == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_llm_timeout(self) -> None:
        """LLM call respects task timeout."""
        mock_client = TimeoutLLMClient()
        mock_manager = MockResourceManager(clients={"slow": mock_client})
        scheduler = Scheduler(
            resource_manager=mock_manager,
            max_concurrent=10,
            task_timeout=0.01,
        )

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="Hello"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        llm_node = GraphNode(
            id="llm_1",
            module=LLMInference(alias="slow"),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={"input:input_0": input_node, "llm_1": llm_node},
            input_ids=["input:input_0"],
            output_ids=["llm_1"],
        )
        state = ExecutionState(graph)

        await scheduler.execute(state)

        assert state.status["llm_1"] == TaskStatus.FAILED
        assert isinstance(state.errors["llm_1"], TimeoutError)


# ─────────────────────────────────────────────────────────────────────────────
# Error Callback Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorCallbacks:
    """Tests for error callback behavior."""

    @pytest.mark.asyncio
    async def test_on_error_called_after_retries_exhausted(self) -> None:
        """on_error is only called after all retries exhausted."""
        module = TransientFailingModule(fail_count=10)
        scheduler = Scheduler(
            max_concurrent=10, max_task_retries=2, task_retry_delay=0.001
        )
        graph = create_graph_with_module(module)
        state = ExecutionState(graph)

        errors: list[tuple[str, Exception]] = []

        def on_error(node_id: str, error: Exception) -> None:
            errors.append((node_id, error))

        await scheduler.execute(state, on_error=on_error)

        # Only one error callback (after all retries)
        assert len(errors) == 1
        assert errors[0][0] == "process_1"

    @pytest.mark.asyncio
    async def test_on_error_not_called_on_success(self) -> None:
        """on_error is not called when task eventually succeeds."""
        module = TransientFailingModule(fail_count=1)
        scheduler = Scheduler(
            max_concurrent=10, max_task_retries=5, task_retry_delay=0.001
        )
        graph = create_graph_with_module(module)
        state = ExecutionState(graph)

        errors: list[tuple[str, Exception]] = []

        def on_error(node_id: str, error: Exception) -> None:
            errors.append((node_id, error))

        await scheduler.execute(state, on_error=on_error)

        # No errors - task succeeded after retry
        assert len(errors) == 0
        assert state.status["process_1"] == TaskStatus.COMPLETED
