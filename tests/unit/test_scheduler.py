"""Unit tests for the Scheduler class."""

import asyncio

import pytest

from inf_engine.clients.base import LLMClient
from inf_engine.errors import RateLimitError
from inf_engine.execution.scheduler import Scheduler
from inf_engine.execution.state import ExecutionState, TaskStatus
from inf_engine.graph import GraphNode, InferenceGraph, NodeRef
from inf_engine.module import InferenceModule
from inf_engine.tracing.tracer import InputNode
from inf_engine.types import LLMRequest, LLMResponse

# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Initialization Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerInit:
    """Tests for Scheduler initialization."""

    def test_init_default_max_concurrent(self) -> None:
        """Scheduler uses default max_concurrent of 100."""
        scheduler = Scheduler()

        assert scheduler.max_concurrent == 100

    def test_init_custom_max_concurrent(self) -> None:
        """Scheduler accepts custom max_concurrent value."""
        scheduler = Scheduler(max_concurrent=50)

        assert scheduler.max_concurrent == 50

    def test_init_max_concurrent_of_one(self) -> None:
        """Scheduler accepts max_concurrent of 1 (serial execution)."""
        scheduler = Scheduler(max_concurrent=1)

        assert scheduler.max_concurrent == 1

    def test_init_large_max_concurrent(self) -> None:
        """Scheduler accepts large max_concurrent values."""
        scheduler = Scheduler(max_concurrent=10000)

        assert scheduler.max_concurrent == 10000

    def test_init_zero_max_concurrent_raises(self) -> None:
        """Scheduler raises ValueError for max_concurrent of 0."""
        with pytest.raises(ValueError, match="max_concurrent must be at least 1"):
            Scheduler(max_concurrent=0)

    def test_init_negative_max_concurrent_raises(self) -> None:
        """Scheduler raises ValueError for negative max_concurrent."""
        with pytest.raises(ValueError, match="max_concurrent must be at least 1"):
            Scheduler(max_concurrent=-1)

    def test_init_creates_semaphore(self) -> None:
        """Scheduler creates internal semaphore."""
        scheduler = Scheduler(max_concurrent=10)

        assert hasattr(scheduler, "_semaphore")
        assert isinstance(scheduler._semaphore, asyncio.Semaphore)

    def test_init_active_count_is_zero(self) -> None:
        """Scheduler starts with zero active tasks."""
        scheduler = Scheduler()

        assert scheduler.active_count == 0

    def test_init_available_slots_equals_max_concurrent(self) -> None:
        """Scheduler starts with all slots available."""
        scheduler = Scheduler(max_concurrent=25)

        assert scheduler.available_slots == 25


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Acquire/Release Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerAcquireRelease:
    """Tests for Scheduler acquire/release methods."""

    @pytest.mark.asyncio
    async def test_acquire_increments_active_count(self) -> None:
        """acquire() increments active_count."""
        scheduler = Scheduler(max_concurrent=10)

        await scheduler.acquire()

        assert scheduler.active_count == 1
        scheduler.release()  # cleanup

    @pytest.mark.asyncio
    async def test_release_decrements_active_count(self) -> None:
        """release() decrements active_count."""
        scheduler = Scheduler(max_concurrent=10)
        await scheduler.acquire()
        assert scheduler.active_count == 1

        scheduler.release()

        assert scheduler.active_count == 0

    @pytest.mark.asyncio
    async def test_multiple_acquires(self) -> None:
        """Multiple acquires increment active_count correctly."""
        scheduler = Scheduler(max_concurrent=10)

        await scheduler.acquire()
        await scheduler.acquire()
        await scheduler.acquire()

        assert scheduler.active_count == 3
        assert scheduler.available_slots == 7

        # cleanup
        scheduler.release()
        scheduler.release()
        scheduler.release()

    @pytest.mark.asyncio
    async def test_release_without_acquire_raises(self) -> None:
        """release() without acquire raises ValueError."""
        scheduler = Scheduler(max_concurrent=10)

        with pytest.raises(ValueError, match="Cannot release: no active tasks"):
            scheduler.release()

    @pytest.mark.asyncio
    async def test_acquire_respects_concurrency_limit(self) -> None:
        """acquire() blocks when at max concurrent tasks."""
        scheduler = Scheduler(max_concurrent=2)

        # Acquire both slots
        await scheduler.acquire()
        await scheduler.acquire()
        assert scheduler.active_count == 2
        assert scheduler.available_slots == 0

        # Third acquire should block - test with timeout
        acquired = False

        async def try_acquire() -> None:
            nonlocal acquired
            await scheduler.acquire()
            acquired = True

        # Start the blocking acquire
        task = asyncio.create_task(try_acquire())

        # Give it a moment - it should NOT complete
        await asyncio.sleep(0.01)
        assert not acquired, "acquire() should block when at capacity"

        # Release a slot - now it should complete
        scheduler.release()
        await asyncio.sleep(0.01)
        assert acquired, "acquire() should complete after release"

        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        scheduler.release()
        scheduler.release()

    @pytest.mark.asyncio
    async def test_acquire_release_cycle(self) -> None:
        """Multiple acquire/release cycles work correctly."""
        scheduler = Scheduler(max_concurrent=5)

        for _ in range(10):
            await scheduler.acquire()
            assert scheduler.active_count == 1
            scheduler.release()
            assert scheduler.active_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Context Manager Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerContextManager:
    """Tests for Scheduler async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_acquires_on_enter(self) -> None:
        """async with scheduler acquires a slot on entry."""
        scheduler = Scheduler(max_concurrent=10)

        async with scheduler:
            assert scheduler.active_count == 1

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exit(self) -> None:
        """async with scheduler releases slot on exit."""
        scheduler = Scheduler(max_concurrent=10)

        async with scheduler:
            assert scheduler.active_count == 1

        assert scheduler.active_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exception(self) -> None:
        """async with scheduler releases slot even on exception."""
        scheduler = Scheduler(max_concurrent=10)

        with pytest.raises(ValueError):
            async with scheduler:
                assert scheduler.active_count == 1
                raise ValueError("Test error")

        assert scheduler.active_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_returns_scheduler(self) -> None:
        """async with scheduler as s returns the scheduler."""
        scheduler = Scheduler(max_concurrent=10)

        async with scheduler as s:
            assert s is scheduler

    @pytest.mark.asyncio
    async def test_nested_context_managers(self) -> None:
        """Multiple nested context managers work correctly."""
        scheduler = Scheduler(max_concurrent=10)

        async with scheduler:
            assert scheduler.active_count == 1
            async with scheduler:
                assert scheduler.active_count == 2
                async with scheduler:
                    assert scheduler.active_count == 3
                assert scheduler.active_count == 2
            assert scheduler.active_count == 1
        assert scheduler.active_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Semaphore Behavior Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerSemaphoreBehavior:
    """Tests for Scheduler semaphore concurrency limiting behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_tasks_respect_limit(self) -> None:
        """Concurrent tasks are limited by max_concurrent."""
        scheduler = Scheduler(max_concurrent=3)
        max_observed = 0
        completed = 0

        async def task() -> None:
            nonlocal max_observed, completed
            async with scheduler:
                max_observed = max(max_observed, scheduler.active_count)
                await asyncio.sleep(0.01)  # Simulate work
                completed += 1

        # Run 10 concurrent tasks with limit of 3
        tasks = [asyncio.create_task(task()) for _ in range(10)]
        await asyncio.gather(*tasks)

        assert completed == 10
        assert max_observed <= 3

    @pytest.mark.asyncio
    async def test_serial_execution_with_max_one(self) -> None:
        """max_concurrent=1 forces serial execution."""
        scheduler = Scheduler(max_concurrent=1)
        execution_order: list[int] = []
        in_critical_section = False

        async def task(task_id: int) -> None:
            nonlocal in_critical_section
            async with scheduler:
                # Check no other task is in critical section
                assert not in_critical_section, "Tasks should not overlap"
                in_critical_section = True
                execution_order.append(task_id)
                await asyncio.sleep(0.001)  # Simulate work
                in_critical_section = False

        tasks = [asyncio.create_task(task(i)) for i in range(5)]
        await asyncio.gather(*tasks)

        assert len(execution_order) == 5

    @pytest.mark.asyncio
    async def test_all_tasks_eventually_complete(self) -> None:
        """All tasks complete even when exceeding concurrency limit."""
        scheduler = Scheduler(max_concurrent=2)
        completed_tasks: list[int] = []

        async def task(task_id: int) -> None:
            async with scheduler:
                await asyncio.sleep(0.001)
                completed_tasks.append(task_id)

        # Run more tasks than the concurrency limit
        tasks = [asyncio.create_task(task(i)) for i in range(20)]
        await asyncio.gather(*tasks)

        assert len(completed_tasks) == 20
        assert set(completed_tasks) == set(range(20))

    @pytest.mark.asyncio
    async def test_available_slots_updates_correctly(self) -> None:
        """available_slots updates as tasks acquire and release."""
        scheduler = Scheduler(max_concurrent=5)

        assert scheduler.available_slots == 5

        await scheduler.acquire()
        assert scheduler.available_slots == 4

        await scheduler.acquire()
        assert scheduler.available_slots == 3

        scheduler.release()
        assert scheduler.available_slots == 4

        scheduler.release()
        assert scheduler.available_slots == 5

    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self) -> None:
        """Scheduler handles high concurrency correctly."""
        scheduler = Scheduler(max_concurrent=50)
        counter = 0
        max_concurrent_observed = 0

        async def increment() -> None:
            nonlocal counter, max_concurrent_observed
            async with scheduler:
                max_concurrent_observed = max(
                    max_concurrent_observed, scheduler.active_count
                )
                counter += 1

        # Run many tasks concurrently
        tasks = [asyncio.create_task(increment()) for _ in range(200)]
        await asyncio.gather(*tasks)

        assert counter == 200
        assert max_concurrent_observed <= 50


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Property Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerProperties:
    """Tests for Scheduler properties."""

    def test_max_concurrent_is_readonly_attribute(self) -> None:
        """max_concurrent is set at init and accessible."""
        scheduler = Scheduler(max_concurrent=42)

        assert scheduler.max_concurrent == 42

    @pytest.mark.asyncio
    async def test_active_count_reflects_current_state(self) -> None:
        """active_count accurately reflects current active tasks."""
        scheduler = Scheduler(max_concurrent=10)

        assert scheduler.active_count == 0

        await scheduler.acquire()
        assert scheduler.active_count == 1

        await scheduler.acquire()
        assert scheduler.active_count == 2

        scheduler.release()
        assert scheduler.active_count == 1

        scheduler.release()
        assert scheduler.active_count == 0

    @pytest.mark.asyncio
    async def test_available_slots_is_computed_correctly(self) -> None:
        """available_slots = max_concurrent - active_count."""
        scheduler = Scheduler(max_concurrent=10)

        for i in range(10):
            assert scheduler.available_slots == 10 - i
            await scheduler.acquire()

        assert scheduler.available_slots == 0

        for i in range(10):
            scheduler.release()
            assert scheduler.available_slots == i + 1


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Import Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerImports:
    """Tests for Scheduler module imports."""

    def test_import_from_execution_package(self) -> None:
        """Scheduler can be imported from execution package."""
        from inf_engine.execution import Scheduler as SchedulerFromPackage

        assert SchedulerFromPackage is Scheduler

    def test_import_from_scheduler_module(self) -> None:
        """Scheduler can be imported from scheduler module."""
        from inf_engine.execution.scheduler import Scheduler as SchedulerFromModule

        assert SchedulerFromModule is Scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Execute Method Tests
# ─────────────────────────────────────────────────────────────────────────────


class SimpleModule(InferenceModule):
    """A simple test module that transforms input."""

    def forward(self, x: str) -> str:
        """Return input with a suffix."""
        return f"{x}_processed"


class AsyncModule(InferenceModule):
    """An async test module."""

    async def forward(self, x: str) -> str:
        """Return input with suffix, using async."""
        await asyncio.sleep(0.001)  # Simulate async work
        return f"{x}_async"


class FailingModule(InferenceModule):
    """A module that always fails."""

    def forward(self, x: str) -> str:
        """Raise an error."""
        raise ValueError("Test failure")


class SlowModule(InferenceModule):
    """A module that takes some time."""

    async def forward(self, x: str) -> str:
        """Simulate slow processing."""
        await asyncio.sleep(0.01)
        return f"{x}_slow"


def create_simple_graph() -> InferenceGraph:
    """Create a simple graph: input -> process."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode(value="hello"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    process_node = GraphNode(
        id="process_1",
        module=SimpleModule(),
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    return InferenceGraph(
        nodes={"input:input_0": input_node, "process_1": process_node},
        input_ids=["input:input_0"],
        output_ids=["process_1"],
    )


def create_linear_graph() -> InferenceGraph:
    """Create a linear graph: input -> a -> b -> c."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode(value="start"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    a_node = GraphNode(
        id="a_1",
        module=SimpleModule(),
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    b_node = GraphNode(
        id="b_2",
        module=SimpleModule(),
        args=(NodeRef("a_1"),),
        kwargs={},
        dependencies=["a_1"],
    )
    c_node = GraphNode(
        id="c_3",
        module=SimpleModule(),
        args=(NodeRef("b_2"),),
        kwargs={},
        dependencies=["b_2"],
    )
    return InferenceGraph(
        nodes={
            "input:input_0": input_node,
            "a_1": a_node,
            "b_2": b_node,
            "c_3": c_node,
        },
        input_ids=["input:input_0"],
        output_ids=["c_3"],
    )


def create_parallel_graph() -> InferenceGraph:
    """Create a parallel graph: input -> [a, b] (independent)."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode(value="parallel"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    a_node = GraphNode(
        id="a_1",
        module=SimpleModule(),
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    b_node = GraphNode(
        id="b_2",
        module=SimpleModule(),
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    return InferenceGraph(
        nodes={"input:input_0": input_node, "a_1": a_node, "b_2": b_node},
        input_ids=["input:input_0"],
        output_ids=["a_1", "b_2"],
    )


def create_diamond_graph() -> InferenceGraph:
    """Create a diamond graph: input -> [a, b] -> merge."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode(value="diamond"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    a_node = GraphNode(
        id="a_1",
        module=SimpleModule(),
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    b_node = GraphNode(
        id="b_2",
        module=SimpleModule(),
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    # Merge node depends on both a and b
    merge_module = SimpleModule()
    merge_node = GraphNode(
        id="merge_3",
        module=merge_module,
        args=(NodeRef("a_1"),),  # In real usage, this would combine both
        kwargs={},
        dependencies=["a_1", "b_2"],
    )
    return InferenceGraph(
        nodes={
            "input:input_0": input_node,
            "a_1": a_node,
            "b_2": b_node,
            "merge_3": merge_node,
        },
        input_ids=["input:input_0"],
        output_ids=["merge_3"],
    )


class TestSchedulerExecute:
    """Tests for Scheduler.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_simple_graph(self) -> None:
        """execute() processes a simple graph correctly."""
        scheduler = Scheduler(max_concurrent=10)
        graph = create_simple_graph()
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        assert outputs == {"process_1": "hello_processed"}

    @pytest.mark.asyncio
    async def test_execute_linear_graph(self) -> None:
        """execute() processes a linear graph in dependency order."""
        scheduler = Scheduler(max_concurrent=10)
        graph = create_linear_graph()
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # Each step adds "_processed"
        assert outputs == {"c_3": "start_processed_processed_processed"}

    @pytest.mark.asyncio
    async def test_execute_parallel_graph(self) -> None:
        """execute() can run parallel tasks concurrently."""
        scheduler = Scheduler(max_concurrent=10)
        graph = create_parallel_graph()
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        assert outputs == {
            "a_1": "parallel_processed",
            "b_2": "parallel_processed",
        }

    @pytest.mark.asyncio
    async def test_execute_diamond_graph(self) -> None:
        """execute() handles diamond dependencies correctly."""
        scheduler = Scheduler(max_concurrent=10)
        graph = create_diamond_graph()
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # Merge node gets the result of a_1
        assert outputs == {"merge_3": "diamond_processed_processed"}

    @pytest.mark.asyncio
    async def test_execute_respects_concurrency_limit(self) -> None:
        """execute() never exceeds max_concurrent tasks."""
        scheduler = Scheduler(max_concurrent=2)
        max_concurrent_observed = 0

        # Create a graph with many slow parallel tasks
        nodes = {
            "input:input_0": GraphNode(
                id="input:input_0",
                module=InputNode(value="test"),
                args=(),
                kwargs={},
                dependencies=[],
            )
        }
        output_ids = []

        for i in range(10):
            node_id = f"slow_{i}"
            nodes[node_id] = GraphNode(
                id=node_id,
                module=SlowModule(),
                args=(NodeRef("input:input_0"),),
                kwargs={},
                dependencies=["input:input_0"],
            )
            output_ids.append(node_id)

        graph = InferenceGraph(
            nodes=nodes,
            input_ids=["input:input_0"],
            output_ids=output_ids,
        )
        state = ExecutionState(graph)

        def track_concurrency(node_id: str, result: object) -> None:
            nonlocal max_concurrent_observed
            max_concurrent_observed = max(
                max_concurrent_observed, scheduler.active_count
            )

        await scheduler.execute(state, on_complete=track_concurrency)

        # Should never exceed the limit (may be at limit when callback fires)
        assert max_concurrent_observed <= 2

    @pytest.mark.asyncio
    async def test_execute_all_tasks_complete(self) -> None:
        """execute() marks all tasks as completed."""
        scheduler = Scheduler(max_concurrent=10)
        graph = create_linear_graph()
        state = ExecutionState(graph)

        await scheduler.execute(state)

        assert state.is_complete()
        for node_id in graph.nodes:
            assert state.status[node_id] == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_with_on_complete_callback(self) -> None:
        """execute() invokes on_complete callback for each task."""
        scheduler = Scheduler(max_concurrent=10)
        graph = create_simple_graph()
        state = ExecutionState(graph)

        completed_nodes: list[str] = []

        def on_complete(node_id: str, result: object) -> None:
            completed_nodes.append(node_id)

        await scheduler.execute(state, on_complete=on_complete)

        assert len(completed_nodes) == 2
        assert "input:input_0" in completed_nodes
        assert "process_1" in completed_nodes

    @pytest.mark.asyncio
    async def test_execute_returns_outputs(self) -> None:
        """execute() returns correct output values."""
        scheduler = Scheduler(max_concurrent=10)
        graph = create_simple_graph()
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        assert isinstance(outputs, dict)
        assert "process_1" in outputs
        assert outputs["process_1"] == "hello_processed"


class TestSchedulerExecuteWithAsync:
    """Tests for Scheduler.execute() with async modules."""

    @pytest.mark.asyncio
    async def test_execute_async_module(self) -> None:
        """execute() handles async forward methods."""
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="async_test"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        async_node = GraphNode(
            id="async_1",
            module=AsyncModule(),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={"input:input_0": input_node, "async_1": async_node},
            input_ids=["input:input_0"],
            output_ids=["async_1"],
        )
        state = ExecutionState(graph)
        scheduler = Scheduler(max_concurrent=10)

        outputs = await scheduler.execute(state)

        assert outputs == {"async_1": "async_test_async"}


class TestSchedulerExecuteFailure:
    """Tests for Scheduler.execute() failure handling."""

    @pytest.mark.asyncio
    async def test_execute_handles_task_failure(self) -> None:
        """execute() handles task failures gracefully."""
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="fail_test"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        failing_node = GraphNode(
            id="failing_1",
            module=FailingModule(),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={"input:input_0": input_node, "failing_1": failing_node},
            input_ids=["input:input_0"],
            output_ids=["failing_1"],
        )
        state = ExecutionState(graph)
        scheduler = Scheduler(max_concurrent=10)

        outputs = await scheduler.execute(state)

        # Task should be marked as failed
        assert state.status["failing_1"] == TaskStatus.FAILED
        assert "failing_1" in state.errors
        assert isinstance(state.errors["failing_1"], ValueError)
        # No outputs from failed task
        assert outputs == {}

    @pytest.mark.asyncio
    async def test_execute_with_on_error_callback(self) -> None:
        """execute() invokes on_error callback for failed tasks."""
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="error_test"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        failing_node = GraphNode(
            id="failing_1",
            module=FailingModule(),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={"input:input_0": input_node, "failing_1": failing_node},
            input_ids=["input:input_0"],
            output_ids=["failing_1"],
        )
        state = ExecutionState(graph)
        scheduler = Scheduler(max_concurrent=10)

        errors: list[tuple[str, Exception]] = []

        def on_error(node_id: str, error: Exception) -> None:
            errors.append((node_id, error))

        await scheduler.execute(state, on_error=on_error)

        assert len(errors) == 1
        assert errors[0][0] == "failing_1"
        assert isinstance(errors[0][1], ValueError)

    @pytest.mark.asyncio
    async def test_execute_cascades_failure(self) -> None:
        """execute() cancels descendants when a task fails."""
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="cascade_test"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        failing_node = GraphNode(
            id="failing_1",
            module=FailingModule(),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        dependent_node = GraphNode(
            id="dependent_2",
            module=SimpleModule(),
            args=(NodeRef("failing_1"),),
            kwargs={},
            dependencies=["failing_1"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "failing_1": failing_node,
                "dependent_2": dependent_node,
            },
            input_ids=["input:input_0"],
            output_ids=["dependent_2"],
        )
        state = ExecutionState(graph)
        scheduler = Scheduler(max_concurrent=10)

        await scheduler.execute(state)

        assert state.status["failing_1"] == TaskStatus.FAILED
        assert state.status["dependent_2"] == TaskStatus.CANCELLED
        assert state.is_complete()


class TestSchedulerExecuteDependencies:
    """Tests for Scheduler.execute() dependency handling."""

    @pytest.mark.asyncio
    async def test_execute_respects_dependencies(self) -> None:
        """execute() executes tasks only after dependencies complete."""
        execution_order: list[str] = []

        class TrackingModule(InferenceModule):
            def __init__(self, name: str):
                super().__init__()
                self.name = name

            def forward(self, x: str) -> str:
                execution_order.append(self.name)
                return f"{x}_{self.name}"

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="dep_test"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        a_node = GraphNode(
            id="a_1",
            module=TrackingModule("a"),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        b_node = GraphNode(
            id="b_2",
            module=TrackingModule("b"),
            args=(NodeRef("a_1"),),
            kwargs={},
            dependencies=["a_1"],
        )
        c_node = GraphNode(
            id="c_3",
            module=TrackingModule("c"),
            args=(NodeRef("b_2"),),
            kwargs={},
            dependencies=["b_2"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "a_1": a_node,
                "b_2": b_node,
                "c_3": c_node,
            },
            input_ids=["input:input_0"],
            output_ids=["c_3"],
        )
        state = ExecutionState(graph)
        scheduler = Scheduler(max_concurrent=10)

        await scheduler.execute(state)

        # a must come before b, b must come before c
        assert execution_order.index("a") < execution_order.index("b")
        assert execution_order.index("b") < execution_order.index("c")

    @pytest.mark.asyncio
    async def test_execute_input_node_provides_value(self) -> None:
        """execute() correctly provides input node values to dependents."""
        scheduler = Scheduler(max_concurrent=10)
        graph = create_simple_graph()
        state = ExecutionState(graph)

        await scheduler.execute(state)

        # The input node value "hello" should be passed to process_1
        # which adds "_processed"
        assert state.results["process_1"].value == "hello_processed"

    @pytest.mark.asyncio
    async def test_execute_result_propagation(self) -> None:
        """execute() propagates results through the graph."""
        scheduler = Scheduler(max_concurrent=10)
        graph = create_linear_graph()
        state = ExecutionState(graph)

        await scheduler.execute(state)

        # Check intermediate results
        assert state.results["input:input_0"].value == "start"
        assert state.results["a_1"].value == "start_processed"
        assert state.results["b_2"].value == "start_processed_processed"
        assert state.results["c_3"].value == "start_processed_processed_processed"


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Event Signaling Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerEventSignaling:
    """Tests for Scheduler event-driven waiting behavior."""

    @pytest.mark.asyncio
    async def test_execute_uses_event_for_waiting(self) -> None:
        """execute() uses event signaling instead of busy-wait."""
        # This test verifies the scheduler doesn't busy-wait by checking
        # that execution completes efficiently even with delays
        scheduler = Scheduler(max_concurrent=10)
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Should complete without hanging due to event signaling
        outputs = await scheduler.execute(state)

        assert outputs == {"c_3": "start_processed_processed_processed"}

    @pytest.mark.asyncio
    async def test_execute_no_busy_wait_latency(self) -> None:
        """execute() doesn't add artificial latency from busy-waiting."""
        import time

        scheduler = Scheduler(max_concurrent=10)
        graph = create_simple_graph()
        state = ExecutionState(graph)

        start = time.perf_counter()
        await scheduler.execute(state)
        elapsed = time.perf_counter() - start

        # Without busy-wait, execution should be very fast
        # Old busy-wait with 0.001s sleep would add noticeable latency
        # Event signaling should complete in under 10ms for simple graph
        assert elapsed < 0.1, f"Execution took {elapsed}s, may be busy-waiting"

    @pytest.mark.asyncio
    async def test_scheduler_wakes_on_task_completion(self) -> None:
        """Scheduler wakes up when task completes and makes new tasks ready."""

        class DelayedModule(InferenceModule):
            async def forward(self, x: str) -> str:
                await asyncio.sleep(0.01)  # Small delay
                return f"{x}_delayed"

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="test"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        delayed_node = GraphNode(
            id="delayed_1",
            module=DelayedModule(),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        final_node = GraphNode(
            id="final_2",
            module=SimpleModule(),
            args=(NodeRef("delayed_1"),),
            kwargs={},
            dependencies=["delayed_1"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "delayed_1": delayed_node,
                "final_2": final_node,
            },
            input_ids=["input:input_0"],
            output_ids=["final_2"],
        )
        state = ExecutionState(graph)
        scheduler = Scheduler(max_concurrent=10)

        outputs = await scheduler.execute(state)

        # Should complete correctly with event signaling
        assert outputs == {"final_2": "test_delayed_processed"}

    @pytest.mark.asyncio
    async def test_scheduler_handles_concurrent_completions(self) -> None:
        """Scheduler handles multiple concurrent task completions correctly."""

        class SlowishModule(InferenceModule):
            async def forward(self, x: str) -> str:
                await asyncio.sleep(0.005)  # Small delay
                return f"{x}_done"

        # Create graph with many parallel nodes
        nodes = {
            "input:input_0": GraphNode(
                id="input:input_0",
                module=InputNode(value="start"),
                args=(),
                kwargs={},
                dependencies=[],
            )
        }
        output_ids = []

        for i in range(10):
            node_id = f"parallel_{i}"
            nodes[node_id] = GraphNode(
                id=node_id,
                module=SlowishModule(),
                args=(NodeRef("input:input_0"),),
                kwargs={},
                dependencies=["input:input_0"],
            )
            output_ids.append(node_id)

        graph = InferenceGraph(
            nodes=nodes,
            input_ids=["input:input_0"],
            output_ids=output_ids,
        )
        state = ExecutionState(graph)
        scheduler = Scheduler(max_concurrent=5)  # Limit concurrency

        outputs = await scheduler.execute(state)

        # All tasks should complete
        assert len(outputs) == 10
        for node_id in output_ids:
            assert outputs[node_id] == "start_done"

    @pytest.mark.asyncio
    async def test_scheduler_event_wakes_on_failure(self) -> None:
        """Scheduler wakes up on task failure to handle completion."""
        scheduler = Scheduler(max_concurrent=10)

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="test"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        failing_node = GraphNode(
            id="failing_1",
            module=FailingModule(),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={"input:input_0": input_node, "failing_1": failing_node},
            input_ids=["input:input_0"],
            output_ids=["failing_1"],
        )
        state = ExecutionState(graph)

        # Should complete (with failure) without hanging
        await scheduler.execute(state)

        assert state.is_complete()
        assert state.status["failing_1"] == TaskStatus.FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler ResourceManager Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class MockLLMClient(LLMClient):
    """A mock LLM client for testing that implements the real interface."""

    def __init__(self, response_content: str = "mock response"):
        self.response_content = response_content
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a mock response."""
        self.requests.append(request)
        return LLMResponse(
            content=self.response_content,
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="mock-model",
        )


class MockRateLimiter:
    """A mock rate limiter for testing.

    Tracks backoff calls to verify rate limit handling behavior.
    """

    def __init__(self) -> None:
        self.backoff_calls: list[float | None] = []

    def backoff(self, retry_after: float | None = None) -> None:
        """Record backoff call."""
        self.backoff_calls.append(retry_after)


class MockResourceManager:
    """A mock ResourceManager for testing.

    Implements the ResourceManagerProtocol interface so it can be used
    with Scheduler for testing without real LLM endpoints.
    """

    def __init__(
        self,
        clients: dict[str, LLMClient] | None = None,
        semaphores: dict[str, asyncio.Semaphore] | None = None,
        rate_limiters: dict[str, MockRateLimiter] | None = None,
    ):
        self.clients: dict[str, LLMClient] = clients or {}
        self.semaphores: dict[str, asyncio.Semaphore] = semaphores or {}
        self.rate_limiters: dict[str, MockRateLimiter] = rate_limiters or {}

    def get_client(self, alias: str) -> LLMClient:
        """Get a client by alias."""
        return self.clients[alias]

    def get_semaphore(self, alias: str) -> asyncio.Semaphore | None:
        """Get a semaphore by alias."""
        return self.semaphores.get(alias)

    def get_rate_limiter(self, alias: str) -> MockRateLimiter | None:
        """Get a rate limiter by alias."""
        return self.rate_limiters.get(alias)


class TestSchedulerResourceManagerInit:
    """Tests for Scheduler initialization with ResourceManager."""

    def test_init_without_resource_manager(self) -> None:
        """Scheduler can be initialized without a ResourceManager."""
        scheduler = Scheduler()

        assert scheduler.resource_manager is None

    def test_init_with_resource_manager(self) -> None:
        """Scheduler accepts a ResourceManager at initialization."""
        mock_manager = MockResourceManager()
        scheduler = Scheduler(resource_manager=mock_manager)

        assert scheduler.resource_manager is mock_manager

    def test_init_with_resource_manager_and_max_concurrent(self) -> None:
        """Scheduler accepts both ResourceManager and max_concurrent."""
        mock_manager = MockResourceManager()
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=50)

        assert scheduler.resource_manager is mock_manager
        assert scheduler.max_concurrent == 50


class TestSchedulerBuildLLMRequest:
    """Tests for Scheduler._build_llm_request()."""

    def test_build_request_from_positional_args(self) -> None:
        """_build_llm_request extracts prompt from positional args."""
        from inf_engine.module import LLMInference

        scheduler = Scheduler()
        module = LLMInference(alias="test", temperature=0.7)

        request = scheduler._build_llm_request(module, ("Hello, world!",), {})

        assert request.prompt == "Hello, world!"
        assert request.temperature == 0.7

    def test_build_request_from_kwargs(self) -> None:
        """_build_llm_request extracts prompt from kwargs."""
        from inf_engine.module import LLMInference

        scheduler = Scheduler()
        module = LLMInference(alias="test")

        request = scheduler._build_llm_request(module, (), {"prompt": "Hi there"})

        assert request.prompt == "Hi there"

    def test_build_request_with_system_prompt(self) -> None:
        """_build_llm_request includes system prompt from module."""
        from inf_engine.module import LLMInference

        scheduler = Scheduler()
        module = LLMInference(
            alias="test", system_prompt="You are helpful.", temperature=0.5
        )

        request = scheduler._build_llm_request(module, ("Hello",), {})

        assert request.prompt == "Hello"
        assert request.system_prompt == "You are helpful."
        assert request.temperature == 0.5

    def test_build_request_with_max_tokens(self) -> None:
        """_build_llm_request includes max_tokens from module."""
        from inf_engine.module import LLMInference

        scheduler = Scheduler()
        module = LLMInference(alias="test", max_tokens=100)

        request = scheduler._build_llm_request(module, ("Hello",), {})

        assert request.max_tokens == 100

    def test_build_request_no_prompt_raises(self) -> None:
        """_build_llm_request raises ValueError when no prompt provided."""
        from inf_engine.module import LLMInference

        scheduler = Scheduler()
        module = LLMInference(alias="test")

        with pytest.raises(ValueError, match="LLMInference requires a prompt"):
            scheduler._build_llm_request(module, (), {})

    def test_build_request_with_response_format(self) -> None:
        """_build_llm_request includes response_format from module."""
        from inf_engine.module import LLMInference

        class MyFormat:
            pass

        scheduler = Scheduler()
        module = LLMInference(alias="test", response_format=MyFormat)

        request = scheduler._build_llm_request(module, ("Hello",), {})

        assert request.response_format is MyFormat


class TestSchedulerExecuteLLM:
    """Tests for Scheduler._execute_llm()."""

    @pytest.mark.asyncio
    async def test_execute_llm_calls_client(self) -> None:
        """_execute_llm calls the client's complete method."""
        from inf_engine.module import LLMInference

        mock_client = MockLLMClient(response_content="Hello back!")
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager)
        module = LLMInference(alias="fast", temperature=0.5)

        result = await scheduler._execute_llm(module, ("Hello",), {})

        assert result == "Hello back!"
        assert len(mock_client.requests) == 1
        assert mock_client.requests[0].prompt == "Hello"

    @pytest.mark.asyncio
    async def test_execute_llm_uses_semaphore(self) -> None:
        """_execute_llm respects per-endpoint semaphore."""
        from inf_engine.module import LLMInference

        mock_client = MockLLMClient()
        semaphore = asyncio.Semaphore(2)
        mock_manager = MockResourceManager(
            clients={"fast": mock_client},
            semaphores={"fast": semaphore},
        )
        scheduler = Scheduler(resource_manager=mock_manager)
        module = LLMInference(alias="fast")

        # Should complete successfully with semaphore
        result = await scheduler._execute_llm(module, ("Hello",), {})

        assert result == "mock response"

    @pytest.mark.asyncio
    async def test_execute_llm_without_semaphore(self) -> None:
        """_execute_llm works when no semaphore is configured."""
        from inf_engine.module import LLMInference

        mock_client = MockLLMClient()
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager)
        module = LLMInference(alias="fast")

        # Should work without semaphore
        result = await scheduler._execute_llm(module, ("Hello",), {})

        assert result == "mock response"

    @pytest.mark.asyncio
    async def test_execute_llm_no_resource_manager_raises(self) -> None:
        """_execute_llm raises RuntimeError when no ResourceManager."""
        from inf_engine.module import LLMInference

        scheduler = Scheduler()  # No resource manager
        module = LLMInference(alias="fast")

        with pytest.raises(RuntimeError, match="no ResourceManager provided"):
            await scheduler._execute_llm(module, ("Hello",), {})

    @pytest.mark.asyncio
    async def test_execute_llm_unknown_alias_raises(self) -> None:
        """_execute_llm raises KeyError for unknown alias."""
        from inf_engine.module import LLMInference

        mock_manager = MockResourceManager(clients={})
        scheduler = Scheduler(resource_manager=mock_manager)
        module = LLMInference(alias="unknown")

        with pytest.raises(KeyError):
            await scheduler._execute_llm(module, ("Hello",), {})


class TestSchedulerExecuteWithLLM:
    """Tests for Scheduler.execute() with LLMInference modules."""

    @pytest.mark.asyncio
    async def test_execute_graph_with_llm_module(self) -> None:
        """execute() handles graphs containing LLMInference modules."""
        from inf_engine.module import LLMInference

        mock_client = MockLLMClient(response_content="LLM output")
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

        # Create graph with LLMInference
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

        assert outputs == {"llm_1": "LLM output"}
        assert len(mock_client.requests) == 1

    @pytest.mark.asyncio
    async def test_execute_mixed_graph(self) -> None:
        """execute() handles graphs with both LLM and non-LLM modules."""
        from inf_engine.module import LLMInference

        mock_client = MockLLMClient(response_content="LLM says hello")
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

        # Create graph: input -> LLM -> SimpleModule
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="Query"),
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
        process_node = GraphNode(
            id="process_2",
            module=SimpleModule(),
            args=(NodeRef("llm_1"),),
            kwargs={},
            dependencies=["llm_1"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "llm_1": llm_node,
                "process_2": process_node,
            },
            input_ids=["input:input_0"],
            output_ids=["process_2"],
        )
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # LLM output goes through SimpleModule which adds "_processed"
        assert outputs == {"process_2": "LLM says hello_processed"}

    @pytest.mark.asyncio
    async def test_execute_llm_without_resource_manager_fails(self) -> None:
        """execute() fails gracefully when LLM module has no ResourceManager."""
        from inf_engine.module import LLMInference

        scheduler = Scheduler(max_concurrent=10)  # No resource manager

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

        await scheduler.execute(state)

        # LLM task should fail
        assert state.status["llm_1"] == TaskStatus.FAILED
        assert isinstance(state.errors["llm_1"], RuntimeError)

    @pytest.mark.asyncio
    async def test_execute_parallel_llm_modules(self) -> None:
        """execute() runs parallel LLM modules concurrently."""
        from inf_engine.module import LLMInference

        mock_client = MockLLMClient(response_content="parallel result")
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(value="Question"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        llm_a = GraphNode(
            id="llm_a",
            module=LLMInference(alias="fast"),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        llm_b = GraphNode(
            id="llm_b",
            module=LLMInference(alias="fast"),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={"input:input_0": input_node, "llm_a": llm_a, "llm_b": llm_b},
            input_ids=["input:input_0"],
            output_ids=["llm_a", "llm_b"],
        )
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        assert outputs == {"llm_a": "parallel result", "llm_b": "parallel result"}
        # Both LLM modules should have been called
        assert len(mock_client.requests) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler RateLimitError Handling Tests
# ─────────────────────────────────────────────────────────────────────────────


class RateLimitingMockClient(LLMClient):
    """A mock LLM client that raises RateLimitError on first attempts.

    Can be configured to fail a certain number of times before succeeding.
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

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Raise RateLimitError for first fail_count calls, then succeed."""
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


class TestSchedulerRateLimitHandling:
    """Tests for Scheduler RateLimitError handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_requeue(self) -> None:
        """RateLimitError causes task to be requeued, not failed."""
        from inf_engine.module import LLMInference

        # Client fails once, then succeeds
        mock_client = RateLimitingMockClient(fail_count=1)
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

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

        # Task should eventually succeed after retry
        assert outputs == {"llm_1": "success"}
        assert state.status["llm_1"] == TaskStatus.COMPLETED
        # Client should have been called twice (once failed, once succeeded)
        assert mock_client.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_backoff(self) -> None:
        """RateLimitError triggers backoff on the rate limiter."""
        from inf_engine.module import LLMInference

        mock_client = RateLimitingMockClient(fail_count=1)
        mock_limiter = MockRateLimiter()
        mock_manager = MockResourceManager(
            clients={"fast": mock_client},
            rate_limiters={"fast": mock_limiter},
        )
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

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

        await scheduler.execute(state)

        # Backoff should have been called once
        assert len(mock_limiter.backoff_calls) == 1
        assert mock_limiter.backoff_calls[0] is None  # No retry_after

    @pytest.mark.asyncio
    async def test_rate_limit_passes_retry_after_to_backoff(self) -> None:
        """RateLimitError passes retry_after value to backoff."""
        from inf_engine.module import LLMInference

        mock_client = RateLimitingMockClient(fail_count=1, retry_after=30.0)
        mock_limiter = MockRateLimiter()
        mock_manager = MockResourceManager(
            clients={"fast": mock_client},
            rate_limiters={"fast": mock_limiter},
        )
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

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

        await scheduler.execute(state)

        # Backoff should have been called with retry_after value
        assert len(mock_limiter.backoff_calls) == 1
        assert mock_limiter.backoff_calls[0] == 30.0

    @pytest.mark.asyncio
    async def test_rate_limit_increments_retry_count(self) -> None:
        """RateLimitError increments the task's retry_count."""
        from inf_engine.module import LLMInference

        mock_client = RateLimitingMockClient(fail_count=2)  # Fail twice
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

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

        await scheduler.execute(state)

        # Task should complete with retry_count of 2 (two rate limits before success)
        assert state.results["llm_1"].retry_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_without_rate_limiter_still_requeues(self) -> None:
        """RateLimitError causes requeue even without rate limiter configured."""
        from inf_engine.module import LLMInference

        mock_client = RateLimitingMockClient(fail_count=1)
        # No rate_limiters configured
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

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

        # Task should still succeed after retry
        assert outputs == {"llm_1": "success"}
        assert state.status["llm_1"] == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_rate_limit_does_not_call_on_error(self) -> None:
        """RateLimitError does not invoke on_error callback."""
        from inf_engine.module import LLMInference

        mock_client = RateLimitingMockClient(fail_count=1)
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

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

        errors: list[tuple[str, Exception]] = []

        def on_error(node_id: str, error: Exception) -> None:
            errors.append((node_id, error))

        await scheduler.execute(state, on_error=on_error)

        # No errors should be recorded (rate limits are not failures)
        assert len(errors) == 0
        assert state.status["llm_1"] == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_rate_limit_cancels_descendants_before_retry(self) -> None:
        """RateLimitError causes descendants to be blocked until retry completes."""
        from inf_engine.module import LLMInference

        mock_client = RateLimitingMockClient(fail_count=1)
        mock_manager = MockResourceManager(clients={"fast": mock_client})
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

        # Graph: input -> llm -> process
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
        process_node = GraphNode(
            id="process_2",
            module=SimpleModule(),
            args=(NodeRef("llm_1"),),
            kwargs={},
            dependencies=["llm_1"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "llm_1": llm_node,
                "process_2": process_node,
            },
            input_ids=["input:input_0"],
            output_ids=["process_2"],
        )
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # All tasks should complete after retry
        assert state.status["llm_1"] == TaskStatus.COMPLETED
        assert state.status["process_2"] == TaskStatus.COMPLETED
        assert outputs == {"process_2": "success_processed"}

    @pytest.mark.asyncio
    async def test_multiple_rate_limits_multiple_backoffs(self) -> None:
        """Multiple RateLimitErrors cause multiple backoff calls."""
        from inf_engine.module import LLMInference

        mock_client = RateLimitingMockClient(fail_count=3)  # Fail three times
        mock_limiter = MockRateLimiter()
        mock_manager = MockResourceManager(
            clients={"fast": mock_client},
            rate_limiters={"fast": mock_limiter},
        )
        scheduler = Scheduler(resource_manager=mock_manager, max_concurrent=10)

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

        await scheduler.execute(state)

        # Backoff should have been called three times
        assert len(mock_limiter.backoff_calls) == 3
        # Task should complete successfully
        assert state.status["llm_1"] == TaskStatus.COMPLETED
        assert mock_client.call_count == 4  # 3 failures + 1 success
