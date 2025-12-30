"""Unit tests for the TraceProfiler and related profiling infrastructure."""

import json
from pathlib import Path

from inf_engine.profiling import (
    EndpointStats,
    ProfilerStatistics,
    TraceEvent,
    TraceProfiler,
)


class TestTraceEvent:
    """Tests for TraceEvent dataclass."""

    def test_create_minimal_event(self) -> None:
        """TraceEvent can be created with minimal required fields."""
        event = TraceEvent(
            name="test_event",
            category="test",
            phase="B",
            timestamp_us=1000,
            process_id=1,
            thread_id=1,
        )

        assert event.name == "test_event"
        assert event.category == "test"
        assert event.phase == "B"
        assert event.timestamp_us == 1000
        assert event.process_id == 1
        assert event.thread_id == 1
        assert event.duration_us is None
        assert event.args is None
        assert event.scope is None
        assert event.id is None

    def test_create_complete_event(self) -> None:
        """TraceEvent can be created with all fields."""
        event = TraceEvent(
            name="test_event",
            category="test",
            phase="X",
            timestamp_us=1000,
            process_id=1,
            thread_id=2,
            duration_us=500,
            args={"key": "value"},
            scope="p",
            id="event_123",
        )

        assert event.name == "test_event"
        assert event.category == "test"
        assert event.phase == "X"
        assert event.timestamp_us == 1000
        assert event.process_id == 1
        assert event.thread_id == 2
        assert event.duration_us == 500
        assert event.args == {"key": "value"}
        assert event.scope == "p"
        assert event.id == "event_123"

    def test_phase_types(self) -> None:
        """TraceEvent supports various phase types."""
        phases = ["B", "E", "X", "b", "e", "i", "C", "M"]
        for phase in phases:
            event = TraceEvent(
                name="test",
                category="test",
                phase=phase,
                timestamp_us=0,
                process_id=1,
                thread_id=1,
            )
            assert event.phase == phase


class TestEndpointStats:
    """Tests for EndpointStats dataclass."""

    def test_create_stats(self) -> None:
        """EndpointStats can be created with all fields."""
        stats = EndpointStats(
            task_count=100,
            avg_duration_ms=150.5,
            max_duration_ms=500.0,
            min_duration_ms=50.0,
            max_concurrency_observed=5,
            rate_limit_events=2,
            total_wait_ms=1000.0,
        )

        assert stats.task_count == 100
        assert stats.avg_duration_ms == 150.5
        assert stats.max_duration_ms == 500.0
        assert stats.min_duration_ms == 50.0
        assert stats.max_concurrency_observed == 5
        assert stats.rate_limit_events == 2
        assert stats.total_wait_ms == 1000.0


class TestProfilerStatistics:
    """Tests for ProfilerStatistics dataclass."""

    def test_create_statistics(self) -> None:
        """ProfilerStatistics can be created with all fields."""
        stats = ProfilerStatistics(
            total_tasks=200,
            completed_tasks=180,
            failed_tasks=20,
            total_duration_ms=5000.0,
            avg_duration_ms=25.0,
            max_duration_ms=100.0,
            min_duration_ms=10.0,
            total_bubble_ms=500.0,
            endpoints={},
        )

        assert stats.total_tasks == 200
        assert stats.completed_tasks == 180
        assert stats.failed_tasks == 20
        assert stats.total_duration_ms == 5000.0
        assert stats.avg_duration_ms == 25.0
        assert stats.max_duration_ms == 100.0
        assert stats.min_duration_ms == 10.0
        assert stats.total_bubble_ms == 500.0
        assert stats.endpoints == {}


class TestTraceProfilerCreation:
    """Tests for TraceProfiler instantiation."""

    def test_default_creation(self) -> None:
        """TraceProfiler can be created with defaults."""
        profiler = TraceProfiler()

        assert profiler.include_counters is True
        assert profiler.include_args is True
        assert profiler.event_count == 0

    def test_custom_creation(self) -> None:
        """TraceProfiler can be created with custom settings."""
        profiler = TraceProfiler(
            include_counters=False,
            include_args=False,
        )

        assert profiler.include_counters is False
        assert profiler.include_args is False

    def test_event_count_starts_at_zero(self) -> None:
        """New profiler has no events."""
        profiler = TraceProfiler()
        assert profiler.event_count == 0


class TestTraceProfilerTaskLifecycle:
    """Tests for task start/end/fail events."""

    def test_task_start_creates_event(self) -> None:
        """task_start creates a begin event."""
        profiler = TraceProfiler()

        profiler.task_start("task_1", "gpt-4", "MyModule")

        assert profiler.event_count >= 1

    def test_task_start_with_args(self) -> None:
        """task_start includes args when include_args is True."""
        profiler = TraceProfiler(include_args=True)

        profiler.task_start(
            "task_1",
            "gpt-4",
            "MyModule",
            {"input": "Hello, world!"},
        )

        assert profiler.event_count >= 1

    def test_task_start_without_args(self) -> None:
        """task_start excludes args when include_args is False."""
        profiler = TraceProfiler(include_args=False)

        profiler.task_start(
            "task_1",
            "gpt-4",
            "MyModule",
            {"input": "Hello, world!"},
        )

        assert profiler.event_count >= 1

    def test_task_end_creates_event(self) -> None:
        """task_end creates an end event."""
        profiler = TraceProfiler()

        profiler.task_start("task_1", "gpt-4", "MyModule")
        initial_count = profiler.event_count

        profiler.task_end("task_1", "gpt-4", duration_ms=150.0)

        assert profiler.event_count > initial_count

    def test_task_end_with_result_summary(self) -> None:
        """task_end includes result summary when provided."""
        profiler = TraceProfiler(include_args=True)

        profiler.task_start("task_1", "gpt-4", "MyModule")
        profiler.task_end(
            "task_1",
            "gpt-4",
            duration_ms=150.0,
            result_summary={"tokens": 100},
        )

        assert profiler.event_count >= 2

    def test_task_failed_creates_event(self) -> None:
        """task_failed creates an end event with error."""
        profiler = TraceProfiler()

        profiler.task_start("task_1", "gpt-4", "MyModule")
        initial_count = profiler.event_count

        profiler.task_failed("task_1", "gpt-4", "Connection timeout")

        assert profiler.event_count > initial_count

    def test_multiple_tasks(self) -> None:
        """Multiple tasks can be tracked."""
        profiler = TraceProfiler()

        # Start multiple tasks
        profiler.task_start("task_1", "gpt-4", "ModuleA")
        profiler.task_start("task_2", "gpt-4", "ModuleB")
        profiler.task_start("task_3", "claude", "ModuleC")

        # End tasks
        profiler.task_end("task_1", "gpt-4", duration_ms=100.0)
        profiler.task_end("task_2", "gpt-4", duration_ms=150.0)
        profiler.task_end("task_3", "claude", duration_ms=200.0)

        assert profiler.event_count >= 6


class TestTraceProfilerRateLimiting:
    """Tests for rate limiting events."""

    def test_rate_limit_hit(self) -> None:
        """rate_limit_hit creates an instant event."""
        profiler = TraceProfiler()

        profiler.rate_limit_hit("gpt-4", retry_after=30.0)

        assert profiler.event_count >= 1

    def test_rate_limit_hit_without_retry_after(self) -> None:
        """rate_limit_hit works without retry_after."""
        profiler = TraceProfiler()

        profiler.rate_limit_hit("gpt-4")

        assert profiler.event_count >= 1

    def test_rate_limit_wait(self) -> None:
        """rate_limit_wait_start and _end create async events."""
        profiler = TraceProfiler()

        profiler.rate_limit_wait_start("gpt-4", "task_1")
        initial_count = profiler.event_count

        profiler.rate_limit_wait_end("gpt-4", "task_1", wait_ms=5000.0)

        assert profiler.event_count > initial_count


class TestTraceProfilerQueueEvents:
    """Tests for queue depth tracking."""

    def test_task_queued(self) -> None:
        """task_queued updates queue depth."""
        profiler = TraceProfiler(include_counters=True)

        profiler.task_queued("task_1")

        assert profiler.event_count >= 1

    def test_task_dequeued(self) -> None:
        """task_dequeued updates queue depth."""
        profiler = TraceProfiler(include_counters=True)

        profiler.task_queued("task_1")
        initial_count = profiler.event_count

        profiler.task_dequeued("task_1")

        assert profiler.event_count > initial_count


class TestTraceProfilerCounters:
    """Tests for counter events."""

    def test_counters_emitted_on_task_start(self) -> None:
        """Counter events are emitted when counters are enabled."""
        profiler = TraceProfiler(include_counters=True)

        profiler.task_start("task_1", "gpt-4", "MyModule")

        # Should have metadata event + begin event + counter event
        assert profiler.event_count >= 2

    def test_counters_not_emitted_when_disabled(self) -> None:
        """No counter events when counters are disabled."""
        profiler = TraceProfiler(include_counters=False)

        profiler.task_start("task_1", "gpt-4", "MyModule")

        # Should have metadata event + begin event, but no counter
        assert profiler.event_count == 2


class TestTraceProfilerCustomEvents:
    """Tests for custom event markers."""

    def test_add_instant_event(self) -> None:
        """add_instant_event creates a marker."""
        profiler = TraceProfiler()

        profiler.add_instant_event("checkpoint")

        assert profiler.event_count == 1

    def test_add_instant_event_with_args(self) -> None:
        """add_instant_event includes args."""
        profiler = TraceProfiler()

        profiler.add_instant_event(
            "checkpoint",
            {"note": "batch complete"},
        )

        assert profiler.event_count == 1

    def test_add_instant_event_scopes(self) -> None:
        """add_instant_event supports different scopes."""
        profiler = TraceProfiler()

        profiler.add_instant_event("global", scope="g")
        profiler.add_instant_event("process", scope="p")
        profiler.add_instant_event("thread", scope="t")

        assert profiler.event_count == 3


class TestTraceProfilerExport:
    """Tests for trace export."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        """export() creates a JSON file."""
        profiler = TraceProfiler()
        output_path = tmp_path / "trace.json"

        profiler.export(output_path)

        assert output_path.exists()

    def test_export_valid_json(self, tmp_path: Path) -> None:
        """Exported file is valid JSON."""
        profiler = TraceProfiler()
        profiler.task_start("task_1", "gpt-4", "MyModule")
        profiler.task_end("task_1", "gpt-4", duration_ms=100.0)

        output_path = tmp_path / "trace.json"
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        assert "traceEvents" in content
        assert "metadata" in content

    def test_export_contains_events(self, tmp_path: Path) -> None:
        """Exported file contains recorded events."""
        profiler = TraceProfiler()
        profiler.task_start("task_1", "gpt-4", "MyModule")
        profiler.task_end("task_1", "gpt-4", duration_ms=100.0)

        output_path = tmp_path / "trace.json"
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        # Should have metadata event + begin + end + counters
        assert len(content["traceEvents"]) >= 2

    def test_export_event_structure(self, tmp_path: Path) -> None:
        """Exported events have correct structure."""
        profiler = TraceProfiler(include_counters=False)
        profiler.add_instant_event("test_event", {"key": "value"})

        output_path = tmp_path / "trace.json"
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        events = content["traceEvents"]
        assert len(events) == 1

        event = events[0]
        assert event["name"] == "test_event"
        assert event["cat"] == "custom"
        assert event["ph"] == "i"
        assert "ts" in event
        assert "pid" in event
        assert "tid" in event
        assert event["args"] == {"key": "value"}

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        """export() creates parent directories if needed."""
        profiler = TraceProfiler()
        output_path = tmp_path / "nested" / "dirs" / "trace.json"

        profiler.export(output_path)

        assert output_path.exists()

    def test_export_includes_metadata(self, tmp_path: Path) -> None:
        """Exported file includes metadata."""
        profiler = TraceProfiler()
        output_path = tmp_path / "trace.json"

        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        metadata = content["metadata"]
        assert "inf_engine_version" in metadata
        assert "start_time" in metadata
        assert "total_events" in metadata


class TestTraceProfilerStatistics:
    """Tests for statistics computation."""

    def test_empty_statistics(self) -> None:
        """get_statistics works with no events."""
        profiler = TraceProfiler()

        stats = profiler.get_statistics()

        assert stats.total_tasks == 0
        assert stats.completed_tasks == 0
        assert stats.failed_tasks == 0
        assert stats.avg_duration_ms == 0.0
        assert stats.max_duration_ms == 0.0
        assert stats.min_duration_ms == 0.0
        assert stats.endpoints == {}

    def test_statistics_with_completed_tasks(self) -> None:
        """get_statistics tracks completed tasks."""
        profiler = TraceProfiler()

        profiler.task_start("task_1", "gpt-4", "MyModule")
        profiler.task_end("task_1", "gpt-4", duration_ms=100.0)

        profiler.task_start("task_2", "gpt-4", "MyModule")
        profiler.task_end("task_2", "gpt-4", duration_ms=200.0)

        stats = profiler.get_statistics()

        assert stats.total_tasks == 2
        assert stats.completed_tasks == 2
        assert stats.failed_tasks == 0
        assert stats.avg_duration_ms == 150.0
        assert stats.max_duration_ms == 200.0
        assert stats.min_duration_ms == 100.0

    def test_statistics_with_failed_tasks(self) -> None:
        """get_statistics tracks failed tasks."""
        profiler = TraceProfiler()

        profiler.task_start("task_1", "gpt-4", "MyModule")
        profiler.task_failed("task_1", "gpt-4", "error")

        stats = profiler.get_statistics()

        assert stats.total_tasks == 1
        assert stats.completed_tasks == 0
        assert stats.failed_tasks == 1

    def test_statistics_per_endpoint(self) -> None:
        """get_statistics provides per-endpoint breakdown."""
        profiler = TraceProfiler()

        # Tasks on gpt-4
        profiler.task_start("task_1", "gpt-4", "MyModule")
        profiler.task_end("task_1", "gpt-4", duration_ms=100.0)
        profiler.task_start("task_2", "gpt-4", "MyModule")
        profiler.task_end("task_2", "gpt-4", duration_ms=200.0)

        # Tasks on claude
        profiler.task_start("task_3", "claude", "MyModule")
        profiler.task_end("task_3", "claude", duration_ms=150.0)

        stats = profiler.get_statistics()

        assert "gpt-4" in stats.endpoints
        assert "claude" in stats.endpoints

        gpt4_stats = stats.endpoints["gpt-4"]
        assert gpt4_stats.task_count == 2
        assert gpt4_stats.avg_duration_ms == 150.0

        claude_stats = stats.endpoints["claude"]
        assert claude_stats.task_count == 1
        assert claude_stats.avg_duration_ms == 150.0

    def test_statistics_max_concurrency(self) -> None:
        """get_statistics tracks max concurrency."""
        profiler = TraceProfiler()

        # Start 3 concurrent tasks
        profiler.task_start("task_1", "gpt-4", "MyModule")
        profiler.task_start("task_2", "gpt-4", "MyModule")
        profiler.task_start("task_3", "gpt-4", "MyModule")

        # End them
        profiler.task_end("task_1", "gpt-4", duration_ms=100.0)
        profiler.task_end("task_2", "gpt-4", duration_ms=100.0)
        profiler.task_end("task_3", "gpt-4", duration_ms=100.0)

        stats = profiler.get_statistics()

        assert stats.endpoints["gpt-4"].max_concurrency_observed == 3

    def test_statistics_rate_limit_tracking(self) -> None:
        """get_statistics tracks rate limit events."""
        profiler = TraceProfiler()

        profiler.task_start("task_1", "gpt-4", "MyModule")
        profiler.rate_limit_hit("gpt-4", retry_after=30.0)
        profiler.rate_limit_hit("gpt-4", retry_after=15.0)

        profiler.rate_limit_wait_start("gpt-4", "task_1")
        profiler.rate_limit_wait_end("gpt-4", "task_1", wait_ms=5000.0)

        profiler.task_end("task_1", "gpt-4", duration_ms=100.0)

        stats = profiler.get_statistics()

        assert stats.endpoints["gpt-4"].rate_limit_events == 2
        assert stats.endpoints["gpt-4"].total_wait_ms == 5000.0


class TestTraceProfilerThreadSafety:
    """Tests for thread-safety of profiler operations."""

    def test_concurrent_task_operations(self) -> None:
        """Concurrent task operations don't corrupt state."""
        import threading

        profiler = TraceProfiler()
        errors: list[Exception] = []

        def run_tasks(task_prefix: str, count: int) -> None:
            try:
                for i in range(count):
                    task_id = f"{task_prefix}_{i}"
                    profiler.task_start(task_id, "gpt-4", "MyModule")
                    profiler.task_end(task_id, "gpt-4", duration_ms=10.0)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=run_tasks, args=(f"thread_{t}", 100))
            for t in range(10)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0
        stats = profiler.get_statistics()
        assert stats.total_tasks == 1000
        assert stats.completed_tasks == 1000


class TestTraceProfilerEndpointMapping:
    """Tests for endpoint -> process ID mapping."""

    def test_unique_pid_per_endpoint(self, tmp_path: Path) -> None:
        """Each endpoint gets a unique process ID."""
        profiler = TraceProfiler()

        profiler.task_start("task_1", "gpt-4", "MyModule")
        profiler.task_end("task_1", "gpt-4", duration_ms=100.0)

        profiler.task_start("task_2", "claude", "MyModule")
        profiler.task_end("task_2", "claude", duration_ms=100.0)

        output_path = tmp_path / "trace.json"
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        events = content["traceEvents"]

        # Find task events (not metadata)
        task_events = [e for e in events if e["cat"] == "llm_call"]

        # Get unique PIDs
        pids = {e["pid"] for e in task_events}
        assert len(pids) == 2, "gpt-4 and claude should have different PIDs"

    def test_process_name_metadata(self, tmp_path: Path) -> None:
        """Process name metadata is included for each endpoint."""
        profiler = TraceProfiler()

        profiler.task_start("task_1", "gpt-4", "MyModule")
        profiler.task_end("task_1", "gpt-4", duration_ms=100.0)

        output_path = tmp_path / "trace.json"
        profiler.export(output_path)

        content = json.loads(output_path.read_text())
        events = content["traceEvents"]

        # Find metadata events
        metadata_events = [e for e in events if e["ph"] == "M"]

        assert len(metadata_events) >= 1
        assert any(
            e["name"] == "process_name" and e["args"]["name"] == "gpt-4"
            for e in metadata_events
        )
