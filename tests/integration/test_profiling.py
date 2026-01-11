"""Integration tests for profiling infrastructure.

Tests the full profiling workflow including trace file generation,
event correctness, counter events, and multi-endpoint traces.
"""

import asyncio
import json
from pathlib import Path

import pytest

from plait.clients.base import LLMClient
from plait.execution.context import ExecutionSettings, get_execution_settings
from plait.execution.scheduler import RateLimiterProtocol, Scheduler
from plait.execution.state import ExecutionState, TaskStatus
from plait.graph import GraphNode, InferenceGraph, NodeRef
from plait.module import LLMInference
from plait.profiling import TraceProfiler
from plait.tracing.tracer import InputNode
from plait.types import LLMRequest, LLMResponse


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.call_count = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return LLMResponse(
            content=f"Response to: {request.prompt}",
            input_tokens=len(request.prompt),
            output_tokens=10,
            finish_reason="stop",
            model="mock-model",
        )


class MockResourceManager:
    """Mock resource manager for testing."""

    def __init__(self, endpoints: dict[str, float] | None = None) -> None:
        # endpoints is a dict of alias -> delay in seconds
        self.endpoints = endpoints or {"fast": 0.0}
        self.clients: dict[str, MockLLMClient] = {}

    def get_client(self, alias: str) -> LLMClient:
        if alias not in self.clients:
            delay = self.endpoints.get(alias, 0.0)
            self.clients[alias] = MockLLMClient(delay=delay)
        return self.clients[alias]

    def get_semaphore(self, alias: str) -> asyncio.Semaphore | None:
        return None

    def get_rate_limiter(self, alias: str) -> RateLimiterProtocol | None:
        return None


class TestTraceFileGeneration:
    """Tests for trace file generation."""

    @pytest.mark.asyncio
    async def test_trace_file_created_on_context_exit(self, tmp_path: Path) -> None:
        """Trace file is created when context exits."""
        output_path = tmp_path / "trace.json"
        settings = ExecutionSettings(
            profile=True,
            profile_path=output_path,
        )

        async with settings:
            profiler = settings.profiler
            assert profiler is not None
            profiler.add_instant_event("test_event")

        assert output_path.exists()
        content = json.loads(output_path.read_text())
        assert "traceEvents" in content

    @pytest.mark.asyncio
    async def test_trace_file_contains_metadata(self, tmp_path: Path) -> None:
        """Trace file includes metadata section."""
        output_path = tmp_path / "trace.json"
        settings = ExecutionSettings(
            profile=True,
            profile_path=output_path,
        )

        async with settings:
            pass

        content = json.loads(output_path.read_text())
        metadata = content["metadata"]
        assert "plait_version" in metadata
        assert "start_time" in metadata
        assert "total_events" in metadata

    @pytest.mark.asyncio
    async def test_auto_generated_trace_path(self, tmp_path: Path) -> None:
        """Trace path is auto-generated when not specified."""
        import os

        orig_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            settings = ExecutionSettings(profile=True)

            async with settings:
                profiler = settings.profiler
                assert profiler is not None
                profiler.add_instant_event("test")

            traces_dir = tmp_path / "traces"
            assert traces_dir.exists()
            trace_files = list(traces_dir.glob("trace_*.json"))
            assert len(trace_files) == 1
        finally:
            os.chdir(orig_dir)


class TestEventCorrectness:
    """Tests for correct event generation."""

    @pytest.mark.asyncio
    async def test_task_events_have_correct_structure(self, tmp_path: Path) -> None:
        """Task events have all required fields."""
        output_path = tmp_path / "trace.json"
        manager = MockResourceManager()
        profiler = TraceProfiler()

        scheduler = Scheduler(
            resource_manager=manager,
            max_concurrent=10,
            profiler=profiler,
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

        await scheduler.execute(state)
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        events = content["traceEvents"]

        # Find task events (begin/end)
        task_events = [e for e in events if e["cat"] == "llm_call"]
        assert len(task_events) >= 2  # At least begin and end

        for event in task_events:
            assert "name" in event
            assert "cat" in event
            assert "ph" in event  # phase
            assert "ts" in event  # timestamp
            assert "pid" in event  # process ID
            assert "tid" in event  # thread ID

    @pytest.mark.asyncio
    async def test_begin_end_event_pairing(self, tmp_path: Path) -> None:
        """Each begin event has a matching end event."""
        output_path = tmp_path / "trace.json"
        manager = MockResourceManager()
        profiler = TraceProfiler(include_counters=False)

        scheduler = Scheduler(
            resource_manager=manager,
            max_concurrent=10,
            profiler=profiler,
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

        await scheduler.execute(state)
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        events = content["traceEvents"]

        # Find task events
        task_events = [e for e in events if e["cat"] == "llm_call"]

        # Count begin and end events
        begins = [e for e in task_events if e["ph"] == "B"]
        ends = [e for e in task_events if e["ph"] == "E"]

        assert len(begins) == len(ends)
        assert len(begins) == 1  # One task

    @pytest.mark.asyncio
    async def test_timestamp_ordering(self, tmp_path: Path) -> None:
        """Event timestamps are properly ordered."""
        output_path = tmp_path / "trace.json"
        manager = MockResourceManager(endpoints={"fast": 0.01})  # Small delay
        profiler = TraceProfiler(include_counters=False)

        scheduler = Scheduler(
            resource_manager=manager,
            max_concurrent=10,
            profiler=profiler,
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

        await scheduler.execute(state)
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        events = content["traceEvents"]

        # Find task events
        task_events = [e for e in events if e["cat"] == "llm_call"]

        # Find begin and end for our task
        begin_events = [e for e in task_events if e["ph"] == "B"]
        end_events = [e for e in task_events if e["ph"] == "E"]

        if begin_events and end_events:
            begin = begin_events[0]
            end = end_events[0]
            # End should be after begin
            assert end["ts"] >= begin["ts"]


class TestCounterEvents:
    """Tests for counter events."""

    @pytest.mark.asyncio
    async def test_counter_events_included_when_enabled(self, tmp_path: Path) -> None:
        """Counter events are included when profile_counters=True."""
        output_path = tmp_path / "trace.json"
        manager = MockResourceManager()
        profiler = TraceProfiler(include_counters=True)

        scheduler = Scheduler(
            resource_manager=manager,
            max_concurrent=10,
            profiler=profiler,
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

        await scheduler.execute(state)
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        events = content["traceEvents"]

        # Find counter events
        counter_events = [e for e in events if e["ph"] == "C"]
        assert len(counter_events) >= 1

    @pytest.mark.asyncio
    async def test_counter_events_excluded_when_disabled(self, tmp_path: Path) -> None:
        """Counter events are excluded when profile_counters=False."""
        output_path = tmp_path / "trace.json"
        manager = MockResourceManager()
        profiler = TraceProfiler(include_counters=False)

        scheduler = Scheduler(
            resource_manager=manager,
            max_concurrent=10,
            profiler=profiler,
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

        await scheduler.execute(state)
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        events = content["traceEvents"]

        # Find counter events
        counter_events = [e for e in events if e["ph"] == "C"]
        assert len(counter_events) == 0

    @pytest.mark.asyncio
    async def test_counter_events_track_active_tasks(self, tmp_path: Path) -> None:
        """Counter events track active task counts."""
        output_path = tmp_path / "trace.json"
        manager = MockResourceManager(endpoints={"fast": 0.05})
        profiler = TraceProfiler(include_counters=True)

        scheduler = Scheduler(
            resource_manager=manager,
            max_concurrent=10,
            profiler=profiler,
        )

        # Create multiple concurrent tasks
        nodes: dict[str, GraphNode] = {}
        input_ids: list[str] = []
        output_ids: list[str] = []

        for i in range(3):
            input_id = f"input:input_{i}"
            llm_id = f"llm_{i}"

            nodes[input_id] = GraphNode(
                id=input_id,
                module=InputNode(value=f"Hello {i}"),
                args=(),
                kwargs={},
                dependencies=[],
            )
            nodes[llm_id] = GraphNode(
                id=llm_id,
                module=LLMInference(alias="fast"),
                args=(NodeRef(input_id),),
                kwargs={},
                dependencies=[input_id],
            )
            input_ids.append(input_id)
            output_ids.append(llm_id)

        graph = InferenceGraph(
            nodes=nodes,
            input_ids=input_ids,
            output_ids=output_ids,
        )
        state = ExecutionState(graph)

        await scheduler.execute(state)
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        events = content["traceEvents"]

        # Find counter events
        counter_events = [e for e in events if e["ph"] == "C"]
        assert len(counter_events) >= 1

        # Check that counter events have active counts
        for event in counter_events:
            assert "args" in event
            # Should have at least one metric
            assert len(event["args"]) >= 1


class TestMultiEndpointTraces:
    """Tests for traces with multiple endpoints."""

    @pytest.mark.asyncio
    async def test_multiple_endpoints_different_pids(self, tmp_path: Path) -> None:
        """Different endpoints get different process IDs."""
        output_path = tmp_path / "trace.json"
        manager = MockResourceManager(endpoints={"fast": 0.01, "slow": 0.02})
        profiler = TraceProfiler(include_counters=False)

        scheduler = Scheduler(
            resource_manager=manager,
            max_concurrent=10,
            profiler=profiler,
        )

        # Create tasks for different endpoints
        input_node1 = GraphNode(
            id="input:input_0",
            module=InputNode(value="Hello fast"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        input_node2 = GraphNode(
            id="input:input_1",
            module=InputNode(value="Hello slow"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        llm_node1 = GraphNode(
            id="llm_1",
            module=LLMInference(alias="fast"),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        llm_node2 = GraphNode(
            id="llm_2",
            module=LLMInference(alias="slow"),
            args=(NodeRef("input:input_1"),),
            kwargs={},
            dependencies=["input:input_1"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node1,
                "input:input_1": input_node2,
                "llm_1": llm_node1,
                "llm_2": llm_node2,
            },
            input_ids=["input:input_0", "input:input_1"],
            output_ids=["llm_1", "llm_2"],
        )
        state = ExecutionState(graph)

        await scheduler.execute(state)
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        events = content["traceEvents"]

        # Find task events
        task_events = [e for e in events if e["cat"] == "llm_call"]

        # Get unique PIDs from task events
        pids = {e["pid"] for e in task_events}
        assert len(pids) == 2, "fast and slow should have different PIDs"

    @pytest.mark.asyncio
    async def test_process_name_metadata_for_each_endpoint(
        self, tmp_path: Path
    ) -> None:
        """Each endpoint has process name metadata."""
        output_path = tmp_path / "trace.json"
        manager = MockResourceManager(endpoints={"fast": 0.01, "slow": 0.02})
        profiler = TraceProfiler(include_counters=False)

        scheduler = Scheduler(
            resource_manager=manager,
            max_concurrent=10,
            profiler=profiler,
        )

        # Create tasks for different endpoints
        input_node1 = GraphNode(
            id="input:input_0",
            module=InputNode(value="Hello fast"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        input_node2 = GraphNode(
            id="input:input_1",
            module=InputNode(value="Hello slow"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        llm_node1 = GraphNode(
            id="llm_1",
            module=LLMInference(alias="fast"),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        llm_node2 = GraphNode(
            id="llm_2",
            module=LLMInference(alias="slow"),
            args=(NodeRef("input:input_1"),),
            kwargs={},
            dependencies=["input:input_1"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node1,
                "input:input_1": input_node2,
                "llm_1": llm_node1,
                "llm_2": llm_node2,
            },
            input_ids=["input:input_0", "input:input_1"],
            output_ids=["llm_1", "llm_2"],
        )
        state = ExecutionState(graph)

        await scheduler.execute(state)
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        events = content["traceEvents"]

        # Find metadata events
        metadata_events = [e for e in events if e["ph"] == "M"]

        # Check for process name metadata for both endpoints
        endpoint_names = set()
        for event in metadata_events:
            if event["name"] == "process_name":
                endpoint_names.add(event["args"]["name"])

        assert "fast" in endpoint_names
        assert "slow" in endpoint_names


class TestExecutionSettingsIntegration:
    """Tests for profiler integration with ExecutionSettings."""

    @pytest.mark.asyncio
    async def test_profiler_available_in_context(self) -> None:
        """Profiler is accessible via ExecutionSettings."""
        settings = ExecutionSettings(profile=True)

        async with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.profiler is not None

    @pytest.mark.asyncio
    async def test_profiler_settings_propagate(self, tmp_path: Path) -> None:
        """Profiler settings are correctly applied."""
        output_path = tmp_path / "trace.json"
        settings = ExecutionSettings(
            profile=True,
            profile_path=output_path,
            profile_counters=True,
            profile_include_args=True,
        )

        async with settings:
            profiler = settings.profiler
            assert profiler is not None
            assert profiler.include_counters is True
            assert profiler.include_args is True

            # Record some events
            profiler.task_start("test", "endpoint", "Module", {"arg": "value"})
            profiler.task_end("test", "endpoint", duration_ms=100.0)

        # Check output
        content = json.loads(output_path.read_text())
        events = content["traceEvents"]

        # Find task begin event
        begin_events = [e for e in events if e["cat"] == "llm_call" and e["ph"] == "B"]
        assert len(begin_events) == 1

        # Check that args were included
        assert "args" in begin_events[0]
        assert "input" in begin_events[0]["args"]

    @pytest.mark.asyncio
    async def test_full_execution_with_profiling(self, tmp_path: Path) -> None:
        """Full execution with profiling enabled via ExecutionSettings."""
        output_path = tmp_path / "trace.json"
        manager = MockResourceManager(endpoints={"fast": 0.01})

        settings = ExecutionSettings(
            profile=True,
            profile_path=output_path,
        )

        async with settings:
            profiler = settings.profiler
            assert profiler is not None

            scheduler = Scheduler(
                resource_manager=manager,
                max_concurrent=10,
                profiler=profiler,
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

            await scheduler.execute(state)
            assert state.status["llm_1"] == TaskStatus.COMPLETED

        # Trace should be exported
        assert output_path.exists()
        content = json.loads(output_path.read_text())

        # Should have task events
        task_events = [e for e in content["traceEvents"] if e["cat"] == "llm_call"]
        assert len(task_events) >= 2  # Begin and end


class TestStatistics:
    """Tests for profiler statistics."""

    @pytest.mark.asyncio
    async def test_statistics_after_execution(self) -> None:
        """Statistics are correctly computed after execution."""
        manager = MockResourceManager(endpoints={"fast": 0.01})
        profiler = TraceProfiler()

        scheduler = Scheduler(
            resource_manager=manager,
            max_concurrent=10,
            profiler=profiler,
        )

        # Create multiple tasks
        nodes: dict[str, GraphNode] = {}
        input_ids: list[str] = []
        output_ids: list[str] = []

        for i in range(3):
            input_id = f"input:input_{i}"
            llm_id = f"llm_{i}"

            nodes[input_id] = GraphNode(
                id=input_id,
                module=InputNode(value=f"Hello {i}"),
                args=(),
                kwargs={},
                dependencies=[],
            )
            nodes[llm_id] = GraphNode(
                id=llm_id,
                module=LLMInference(alias="fast"),
                args=(NodeRef(input_id),),
                kwargs={},
                dependencies=[input_id],
            )
            input_ids.append(input_id)
            output_ids.append(llm_id)

        graph = InferenceGraph(
            nodes=nodes,
            input_ids=input_ids,
            output_ids=output_ids,
        )
        state = ExecutionState(graph)

        await scheduler.execute(state)

        stats = profiler.get_statistics()
        assert stats.total_tasks == 3
        assert stats.completed_tasks == 3
        assert stats.failed_tasks == 0
        assert "fast" in stats.endpoints
        assert stats.endpoints["fast"].task_count == 3
