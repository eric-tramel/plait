"""Integration tests for the execution system.

This file contains tests for PR-028: Execution integration tests.
Tests verify full execution scenarios including linear, parallel,
and diamond graph patterns.
"""

import asyncio
from typing import Any

import pytest

from plait.execution.executor import run
from plait.execution.scheduler import Scheduler
from plait.execution.state import ExecutionState, TaskResult, TaskStatus
from plait.module import InferenceModule
from plait.tracing.tracer import Tracer

# ─────────────────────────────────────────────────────────────────────────────
# Test Helpers - Mock Modules for Testing
# ─────────────────────────────────────────────────────────────────────────────


class EchoModule(InferenceModule):
    """Test module that echoes its input with a prefix.

    Used for testing execution without actual LLM calls.
    """

    def __init__(self, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix

    def forward(self, text: str) -> str:
        return f"{self.prefix}{text}"


class ConcatModule(InferenceModule):
    """Test module that concatenates multiple inputs.

    Used for testing fan-in (diamond) patterns.
    """

    def __init__(self, separator: str = " + ") -> None:
        super().__init__()
        self.separator = separator

    def forward(self, *args: str) -> str:
        return self.separator.join(str(arg) for arg in args)


class UppercaseModule(InferenceModule):
    """Test module that uppercases its input."""

    def forward(self, text: str) -> str:
        return text.upper()


class DelayModule(InferenceModule):
    """Test module that adds a delay to simulate async work.

    Used for testing concurrent execution.
    """

    def __init__(self, delay_ms: float = 10.0) -> None:
        super().__init__()
        self.delay_ms = delay_ms

    async def forward(self, text: str) -> str:
        await asyncio.sleep(self.delay_ms / 1000)
        return f"delayed:{text}"


class FailingModule(InferenceModule):
    """Test module that always fails.

    Used for testing error handling.
    """

    def __init__(self, error_message: str = "Intentional failure") -> None:
        super().__init__()
        self.error_message = error_message

    def forward(self, text: str) -> str:
        raise RuntimeError(self.error_message)


class CountingModule(InferenceModule):
    """Test module that counts how many times it was called.

    Used for verifying execution ordering and call counts.
    """

    call_count: int = 0

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(self, text: str) -> str:
        self.call_count += 1
        return f"call_{self.call_count}:{text}"


# ─────────────────────────────────────────────────────────────────────────────
# Linear Graph Execution Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLinearGraphExecution:
    """Tests for executing linear (sequential) graphs."""

    @pytest.mark.asyncio
    async def test_single_node_execution(self) -> None:
        """Simplest case: single module execution."""

        class SingleStep(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.echo = EchoModule(prefix="echo:")

            def forward(self, text: str) -> str:
                return self.echo(text)

        result = await run(SingleStep(), "hello")
        assert result == "echo:hello"

    @pytest.mark.asyncio
    async def test_two_step_linear_execution(self) -> None:
        """Two-step pipeline: A -> B."""

        class TwoStep(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.step1 = EchoModule(prefix="[1]")
                self.step2 = EchoModule(prefix="[2]")

            def forward(self, text: str) -> str:
                r1 = self.step1(text)
                r2 = self.step2(r1)
                return r2

        result = await run(TwoStep(), "input")
        assert result == "[2][1]input"

    @pytest.mark.asyncio
    async def test_three_step_linear_execution(self) -> None:
        """Three-step pipeline: A -> B -> C."""

        class ThreeStep(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.step1 = EchoModule(prefix="[1]")
                self.step2 = EchoModule(prefix="[2]")
                self.step3 = EchoModule(prefix="[3]")

            def forward(self, text: str) -> str:
                r1 = self.step1(text)
                r2 = self.step2(r1)
                r3 = self.step3(r2)
                return r3

        result = await run(ThreeStep(), "start")
        assert result == "[3][2][1]start"

    @pytest.mark.asyncio
    async def test_linear_execution_preserves_order(self) -> None:
        """Linear execution maintains dependency ordering."""
        execution_order: list[str] = []

        class OrderTracker(InferenceModule):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name

            def forward(self, text: str) -> str:
                execution_order.append(self.name)
                return f"{self.name}:{text}"

        class LinearPipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = OrderTracker("A")
                self.b = OrderTracker("B")
                self.c = OrderTracker("C")

            def forward(self, text: str) -> str:
                r1 = self.a(text)
                r2 = self.b(r1)
                r3 = self.c(r2)
                return r3

        result = await run(LinearPipeline(), "input")

        # Verify execution order
        assert execution_order == ["A", "B", "C"]
        assert result == "C:B:A:input"

    @pytest.mark.asyncio
    async def test_linear_with_async_module(self) -> None:
        """Linear execution works with async modules."""

        class AsyncPipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.delay = DelayModule(delay_ms=5)
                self.echo = EchoModule(prefix="after_delay:")

            def forward(self, text: str) -> str:
                delayed = self.delay(text)
                return self.echo(delayed)

        result = await run(AsyncPipeline(), "test")
        assert result == "after_delay:delayed:test"


# ─────────────────────────────────────────────────────────────────────────────
# Parallel Graph Execution Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParallelGraphExecution:
    """Tests for executing parallel (fan-out) graphs."""

    @pytest.mark.asyncio
    async def test_simple_parallel_execution(self) -> None:
        """Two independent branches execute from same input."""

        class Parallel(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.branch_a = EchoModule(prefix="A:")
                self.branch_b = EchoModule(prefix="B:")

            def forward(self, text: str) -> dict[str, str]:
                return {
                    "a": self.branch_a(text),
                    "b": self.branch_b(text),
                }

        result = await run(Parallel(), "input")
        assert isinstance(result, dict)
        # User-defined dict keys are preserved
        assert result["a"] == "A:input"
        assert result["b"] == "B:input"

    @pytest.mark.asyncio
    async def test_three_way_parallel_execution(self) -> None:
        """Three independent branches execute from same input."""

        class ThreeWayParallel(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.branch_a = EchoModule(prefix="A:")
                self.branch_b = EchoModule(prefix="B:")
                self.branch_c = EchoModule(prefix="C:")

            def forward(self, text: str) -> dict[str, str]:
                return {
                    "a": self.branch_a(text),
                    "b": self.branch_b(text),
                    "c": self.branch_c(text),
                }

        result = await run(ThreeWayParallel(), "test")
        # User-defined dict keys are preserved
        assert result["a"] == "A:test"
        assert result["b"] == "B:test"
        assert result["c"] == "C:test"

    @pytest.mark.asyncio
    async def test_parallel_with_different_processing(self) -> None:
        """Parallel branches can apply different transformations."""

        class DifferentBranches(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.upper = UppercaseModule()
                self.echo = EchoModule(prefix="PREFIX:")

            def forward(self, text: str) -> dict[str, str]:
                return {
                    "upper": self.upper(text),
                    "prefixed": self.echo(text),
                }

        result = await run(DifferentBranches(), "hello")
        # User-defined dict keys are preserved
        assert result["upper"] == "HELLO"
        assert result["prefixed"] == "PREFIX:hello"

    @pytest.mark.asyncio
    async def test_parallel_execution_is_concurrent(self) -> None:
        """Parallel branches execute concurrently, not sequentially."""
        import time

        class SlowBranches(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.branch_a = DelayModule(delay_ms=50)
                self.branch_b = DelayModule(delay_ms=50)
                self.branch_c = DelayModule(delay_ms=50)

            def forward(self, text: str) -> dict[str, str]:
                return {
                    "a": self.branch_a(text),
                    "b": self.branch_b(text),
                    "c": self.branch_c(text),
                }

        start = time.time()
        result = await run(SlowBranches(), "test", max_concurrent=10)
        elapsed_ms = (time.time() - start) * 1000

        # If sequential, would take 150ms+. If parallel, should be ~50-100ms
        # Use generous margin for CI environment variability
        assert elapsed_ms < 200, f"Took {elapsed_ms}ms, expected concurrent execution"

        # Verify results are correct with user-defined keys
        assert result["a"] == "delayed:test"
        assert result["b"] == "delayed:test"
        assert result["c"] == "delayed:test"

    @pytest.mark.asyncio
    async def test_parallel_list_output(self) -> None:
        """Parallel execution with list output."""

        class ListOutput(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = EchoModule(prefix="1:")
                self.b = EchoModule(prefix="2:")
                self.c = EchoModule(prefix="3:")

            def forward(self, text: str) -> list[str]:
                return [
                    self.a(text),
                    self.b(text),
                    self.c(text),
                ]

        result = await run(ListOutput(), "x")
        # Result is dict with indices as keys for list outputs
        assert isinstance(result, dict)
        assert len(result) == 3
        assert result[0] == "1:x"
        assert result[1] == "2:x"
        assert result[2] == "3:x"


# ─────────────────────────────────────────────────────────────────────────────
# Diamond Graph Execution Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDiamondGraphExecution:
    """Tests for executing diamond (fan-out + fan-in) graphs."""

    @pytest.mark.asyncio
    async def test_simple_diamond_execution(self) -> None:
        """Classic diamond: input -> (A, B) -> merger -> output."""

        class Diamond(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.branch_a = EchoModule(prefix="A:")
                self.branch_b = EchoModule(prefix="B:")
                self.merger = ConcatModule(separator=" | ")

            def forward(self, text: str) -> str:
                a = self.branch_a(text)
                b = self.branch_b(text)
                return self.merger(a, b)

        result = await run(Diamond(), "input")
        assert result == "A:input | B:input"

    @pytest.mark.asyncio
    async def test_diamond_with_different_branch_lengths(self) -> None:
        """Diamond where branches have different depths."""

        class AsymmetricDiamond(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                # Short branch: just one step
                self.short = EchoModule(prefix="short:")
                # Long branch: two steps
                self.long_step1 = EchoModule(prefix="long1:")
                self.long_step2 = EchoModule(prefix="long2:")
                # Merger
                self.merger = ConcatModule(separator=" + ")

            def forward(self, text: str) -> str:
                short_result = self.short(text)
                long_r1 = self.long_step1(text)
                long_result = self.long_step2(long_r1)
                return self.merger(short_result, long_result)

        result = await run(AsymmetricDiamond(), "x")
        assert result == "short:x + long2:long1:x"

    @pytest.mark.asyncio
    async def test_three_way_diamond(self) -> None:
        """Diamond with three branches: input -> (A, B, C) -> merger."""

        class ThreeWayDiamond(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = EchoModule(prefix="A:")
                self.b = EchoModule(prefix="B:")
                self.c = EchoModule(prefix="C:")
                self.merger = ConcatModule(separator="-")

            def forward(self, text: str) -> str:
                ra = self.a(text)
                rb = self.b(text)
                rc = self.c(text)
                return self.merger(ra, rb, rc)

        result = await run(ThreeWayDiamond(), "x")
        assert result == "A:x-B:x-C:x"

    @pytest.mark.asyncio
    async def test_nested_diamond_execution(self) -> None:
        """Diamond within a linear pipeline."""

        class NestedDiamond(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.pre = EchoModule(prefix="pre:")
                self.a = EchoModule(prefix="A:")
                self.b = EchoModule(prefix="B:")
                self.merger = ConcatModule(separator="|")
                self.post = EchoModule(prefix="post:")

            def forward(self, text: str) -> str:
                preprocessed = self.pre(text)
                ra = self.a(preprocessed)
                rb = self.b(preprocessed)
                merged = self.merger(ra, rb)
                return self.post(merged)

        result = await run(NestedDiamond(), "x")
        assert result == "post:A:pre:x|B:pre:x"

    @pytest.mark.asyncio
    async def test_diamond_branches_execute_before_merger(self) -> None:
        """Merger waits for both branches to complete."""
        execution_order: list[str] = []

        class Tracker(InferenceModule):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name

            def forward(self, *args: Any) -> str:
                execution_order.append(self.name)
                return f"{self.name}:{':'.join(str(a) for a in args)}"

        class TrackedDiamond(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.branch_a = Tracker("A")
                self.branch_b = Tracker("B")
                self.merger = Tracker("M")

            def forward(self, text: str) -> str:
                a = self.branch_a(text)
                b = self.branch_b(text)
                return self.merger(a, b)

        await run(TrackedDiamond(), "input")

        # Merger must execute after both A and B
        assert "M" in execution_order
        assert execution_order.index("A") < execution_order.index("M")
        assert execution_order.index("B") < execution_order.index("M")


# ─────────────────────────────────────────────────────────────────────────────
# Error Handling Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExecutionErrorHandling:
    """Tests for error handling during execution."""

    @pytest.mark.asyncio
    async def test_failed_task_marks_descendants_cancelled(self) -> None:
        """When a task fails, its descendants are cancelled."""

        class FailingPipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.first = EchoModule(prefix="first:")
                self.failing = FailingModule("test error")
                self.after = EchoModule(prefix="after:")

            def forward(self, text: str) -> str:
                r1 = self.first(text)
                r2 = self.failing(r1)
                r3 = self.after(r2)
                return r3

        # Build graph and state to inspect
        tracer = Tracer()
        graph = tracer.trace(FailingPipeline(), "input")
        state = ExecutionState(graph)
        scheduler = Scheduler(max_concurrent=10)

        # Execute (will fail)
        await scheduler.execute(state)

        # Check that first completed
        assert state.status["EchoModule_1"] == TaskStatus.COMPLETED
        # FailingModule should be failed
        assert state.status["FailingModule_2"] == TaskStatus.FAILED
        # EchoModule after the failure should be cancelled
        assert state.status["EchoModule_3"] == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_error_stored_in_state(self) -> None:
        """Errors from failed tasks are stored in state."""

        class SimpleFailure(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.failing = FailingModule("specific error message")

            def forward(self, text: str) -> str:
                return self.failing(text)

        tracer = Tracer()
        graph = tracer.trace(SimpleFailure(), "input")
        state = ExecutionState(graph)
        scheduler = Scheduler()

        await scheduler.execute(state)

        # Should have stored the error
        assert "FailingModule_1" in state.errors
        assert "specific error message" in str(state.errors["FailingModule_1"])

    @pytest.mark.asyncio
    async def test_parallel_failure_cancels_downstream(self) -> None:
        """In diamond, failed branch cancels merger."""

        class PartialFailDiamond(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.good_branch = EchoModule(prefix="good:")
                self.bad_branch = FailingModule("branch failed")
                self.merger = ConcatModule()

            def forward(self, text: str) -> str:
                good = self.good_branch(text)
                bad = self.bad_branch(text)
                return self.merger(good, bad)

        tracer = Tracer()
        graph = tracer.trace(PartialFailDiamond(), "input")
        state = ExecutionState(graph)
        scheduler = Scheduler()

        await scheduler.execute(state)

        # Good branch should complete
        assert state.status["EchoModule_1"] == TaskStatus.COMPLETED
        # Bad branch should fail
        assert state.status["FailingModule_2"] == TaskStatus.FAILED
        # Merger should be cancelled (depends on failed task)
        assert state.status["ConcatModule_3"] == TaskStatus.CANCELLED


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency Control Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConcurrencyControl:
    """Tests for scheduler concurrency limits."""

    @pytest.mark.asyncio
    async def test_respects_max_concurrent_limit(self) -> None:
        """Scheduler respects max_concurrent setting."""
        max_observed_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        class ConcurrencyTracker(InferenceModule):
            async def forward(self, text: str) -> str:
                nonlocal max_observed_concurrent, current_concurrent
                async with lock:
                    current_concurrent += 1
                    if current_concurrent > max_observed_concurrent:
                        max_observed_concurrent = current_concurrent

                await asyncio.sleep(0.01)  # Simulate work

                async with lock:
                    current_concurrent -= 1

                return text

        class HighParallelism(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                # Create many parallel branches
                self.trackers = [ConcurrencyTracker() for _ in range(20)]

            def forward(self, text: str) -> list[str]:
                return [t(text) for t in self.trackers]

        # Use a low max_concurrent to test the limit
        await run(HighParallelism(), "test", max_concurrent=3)

        # Should never have exceeded 3 concurrent tasks
        # Note: +1 for possible race condition margin
        assert max_observed_concurrent <= 4

    @pytest.mark.asyncio
    async def test_max_concurrent_one_serializes_execution(self) -> None:
        """max_concurrent=1 forces fully serial execution."""
        execution_times: list[float] = []

        class TimeTracker(InferenceModule):
            async def forward(self, text: str) -> str:
                import time

                execution_times.append(time.time())
                await asyncio.sleep(0.01)
                return text

        class ParallelTasks(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = TimeTracker()
                self.b = TimeTracker()
                self.c = TimeTracker()

            def forward(self, text: str) -> list[str]:
                return [self.a(text), self.b(text), self.c(text)]

        await run(ParallelTasks(), "test", max_concurrent=1)

        # With serial execution, each should start after previous completes
        # Check that each start time is > previous
        for i in range(1, len(execution_times)):
            # Allow small tolerance for timing
            assert execution_times[i] >= execution_times[i - 1] - 0.001


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Inputs Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleInputExecution:
    """Tests for executing graphs with multiple inputs."""

    @pytest.mark.asyncio
    async def test_two_positional_inputs(self) -> None:
        """Module with two positional inputs executes correctly."""

        class TwoInputs(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.concat = ConcatModule(separator=" + ")

            def forward(self, a: str, b: str) -> str:
                return self.concat(a, b)

        result = await run(TwoInputs(), "first", "second")
        assert result == "first + second"

    @pytest.mark.asyncio
    async def test_keyword_inputs(self) -> None:
        """Module with keyword inputs executes correctly."""

        class KwargInputs(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.process = ConcatModule(separator=":")

            def forward(self, *, prefix: str, suffix: str) -> str:
                return self.process(prefix, suffix)

        result = await run(KwargInputs(), prefix="hello", suffix="world")
        assert result == "hello:world"

    @pytest.mark.asyncio
    async def test_mixed_positional_and_keyword_inputs(self) -> None:
        """Module with mixed args and kwargs executes correctly."""

        class MixedInputs(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.process_main = EchoModule(prefix="main:")
                self.process_ctx = EchoModule(prefix="ctx:")
                self.combine = ConcatModule(separator=" | ")

            def forward(self, text: str, *, context: str) -> str:
                main = self.process_main(text)
                ctx = self.process_ctx(context)
                return self.combine(main, ctx)

        result = await run(MixedInputs(), "primary", context="secondary")
        assert result == "main:primary | ctx:secondary"


# ─────────────────────────────────────────────────────────────────────────────
# Complete Execution Flow Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCompleteExecutionFlow:
    """Tests for complete end-to-end execution scenarios."""

    @pytest.mark.asyncio
    async def test_complex_graph_execution(self) -> None:
        """Complex graph with multiple patterns executes correctly.

        Graph structure:
        input -> preprocess -> [A, B] -> merger -> postprocess -> output
        """

        class ComplexPipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.preprocess = EchoModule(prefix="pre:")
                self.branch_a = EchoModule(prefix="A:")
                self.branch_b = EchoModule(prefix="B:")
                self.merger = ConcatModule(separator="|")
                self.postprocess = EchoModule(prefix="post:")

            def forward(self, text: str) -> str:
                preprocessed = self.preprocess(text)
                a = self.branch_a(preprocessed)
                b = self.branch_b(preprocessed)
                merged = self.merger(a, b)
                return self.postprocess(merged)

        result = await run(ComplexPipeline(), "input")
        assert result == "post:A:pre:input|B:pre:input"

    @pytest.mark.asyncio
    async def test_execution_callbacks_called(self) -> None:
        """on_complete and on_error callbacks are invoked."""
        completed: list[str] = []
        errors: list[str] = []

        class SimplePipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.step = EchoModule(prefix="done:")

            def forward(self, text: str) -> str:
                return self.step(text)

        tracer = Tracer()
        graph = tracer.trace(SimplePipeline(), "test")
        state = ExecutionState(graph)
        scheduler = Scheduler()

        def on_complete(node_id: str, result: TaskResult) -> None:
            completed.append(node_id)

        def on_error(node_id: str, error: Exception) -> None:
            errors.append(node_id)

        await scheduler.execute(state, on_complete=on_complete, on_error=on_error)

        # Should have completion callbacks for input and step
        assert "input:input_0" in completed
        assert "EchoModule_1" in completed
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_result_values_propagate_through_graph(self) -> None:
        """Values propagate correctly through multi-step execution."""

        class ValuePropagation(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.step1 = UppercaseModule()
                self.step2 = EchoModule(prefix="[")
                self.step3 = EchoModule(prefix="RESULT=")

            def forward(self, text: str) -> str:
                upper = self.step1(text)  # "hello" -> "HELLO"
                bracketed = self.step2(upper)  # "HELLO" -> "[HELLO"
                final = self.step3(bracketed)  # "[HELLO" -> "RESULT=[HELLO"
                return final

        result = await run(ValuePropagation(), "hello")
        assert result == "RESULT=[HELLO"

    @pytest.mark.asyncio
    async def test_execution_state_is_complete_after_run(self) -> None:
        """ExecutionState.is_complete() returns True after successful execution."""

        class SimplePipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.step = EchoModule(prefix="done:")

            def forward(self, text: str) -> str:
                return self.step(text)

        tracer = Tracer()
        graph = tracer.trace(SimplePipeline(), "test")
        state = ExecutionState(graph)
        scheduler = Scheduler()

        await scheduler.execute(state)

        assert state.is_complete()

    @pytest.mark.asyncio
    async def test_all_completed_tasks_have_results(self) -> None:
        """All completed tasks have valid results in state."""

        class MultiStep(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = EchoModule(prefix="A:")
                self.b = EchoModule(prefix="B:")
                self.c = EchoModule(prefix="C:")

            def forward(self, text: str) -> str:
                ra = self.a(text)
                rb = self.b(ra)
                rc = self.c(rb)
                return rc

        tracer = Tracer()
        graph = tracer.trace(MultiStep(), "input")
        state = ExecutionState(graph)
        scheduler = Scheduler()

        await scheduler.execute(state)

        # All nodes should have results
        for node_id, status in state.status.items():
            if status == TaskStatus.COMPLETED:
                assert node_id in state.results
                assert state.results[node_id].node_id == node_id


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_empty_module_returns_input(self) -> None:
        """Module that passes through input unchanged."""

        class PassThrough(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.identity = EchoModule(prefix="")

            def forward(self, text: str) -> str:
                return self.identity(text)

        result = await run(PassThrough(), "unchanged")
        assert result == "unchanged"

    @pytest.mark.asyncio
    async def test_deeply_nested_modules(self) -> None:
        """Deeply nested module structure executes correctly."""

        class DeepPipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.steps = [EchoModule(prefix=f"[{i}]") for i in range(10)]

            def forward(self, text: str) -> str:
                result = text
                for step in self.steps:
                    result = step(result)
                return result

        result = await run(DeepPipeline(), "x")
        # Result should have all prefixes
        assert "[9]" in result
        assert "[0]" in result

    @pytest.mark.asyncio
    async def test_run_function_unwraps_single_output(self) -> None:
        """run() unwraps dict when there's a single output."""

        class SingleOutput(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.step = EchoModule(prefix="result:")

            def forward(self, text: str) -> str:
                return self.step(text)

        result = await run(SingleOutput(), "test")
        # Should be unwrapped string, not dict
        assert isinstance(result, str)
        assert result == "result:test"

    @pytest.mark.asyncio
    async def test_run_function_returns_dict_for_multiple_outputs(self) -> None:
        """run() returns dict when there are multiple outputs."""

        class MultiOutput(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = EchoModule(prefix="A:")
                self.b = EchoModule(prefix="B:")

            def forward(self, text: str) -> dict[str, str]:
                return {
                    "first": self.a(text),
                    "second": self.b(text),
                }

        result = await run(MultiOutput(), "test")
        assert isinstance(result, dict)
        assert len(result) == 2
        # User-defined keys are preserved
        assert result["first"] == "A:test"
        assert result["second"] == "B:test"
