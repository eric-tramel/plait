"""Unit tests for the Scheduler class."""

import asyncio

import pytest

from inf_engine.execution.scheduler import Scheduler
from inf_engine.execution.state import ExecutionState, TaskStatus
from inf_engine.graph import GraphNode, InferenceGraph, NodeRef
from inf_engine.module import InferenceModule
from inf_engine.tracing.tracer import InputNode

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
