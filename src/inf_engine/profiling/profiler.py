"""Trace profiler for execution visualization.

This module implements the TraceProfiler class that collects execution traces
in Chrome Trace Event Format. The traces can be visualized in Perfetto
(ui.perfetto.dev) or Chrome DevTools (chrome://tracing).

The profiler maps:
- Process = Endpoint (e.g., "gpt-4", "claude-3")
- Thread = Concurrent slot within that endpoint

Example:
    >>> from inf_engine.profiling import TraceProfiler
    >>>
    >>> # Create profiler
    >>> profiler = TraceProfiler(include_counters=True, include_args=True)
    >>>
    >>> # Record task lifecycle
    >>> profiler.task_start("task_1", "gpt-4", "MyModule", {"input": "hello"})
    >>> # ... task executes ...
    >>> profiler.task_end("task_1", "gpt-4", duration_ms=150.0)
    >>>
    >>> # Export to file
    >>> profiler.export("./trace.json")
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    """A single trace event in Chrome Trace Format.

    Represents an event that occurred during execution. Events can be of
    different types indicated by the phase field.

    Attributes:
        name: Display name for the event.
        category: Category string for filtering (e.g., "llm_call", "rate_limit").
        phase: Event type - "X" (complete), "B" (begin), "E" (end),
            "b" (async begin), "e" (async end), "i" (instant), "C" (counter),
            "M" (metadata).
        timestamp_us: Timestamp in microseconds since profiler start.
        process_id: Process ID (maps to endpoint alias).
        thread_id: Thread ID (maps to concurrent slot within endpoint).
        duration_us: Duration in microseconds (for complete events).
        args: Additional event arguments for display in the viewer.
        scope: Scope for instant events - "g" (global), "p" (process), "t" (thread).
        id: Unique identifier for async events.

    Example:
        >>> event = TraceEvent(
        ...     name="task_123",
        ...     category="llm_call",
        ...     phase="B",
        ...     timestamp_us=1000000,
        ...     process_id=1,
        ...     thread_id=1,
        ...     args={"endpoint": "gpt-4"},
        ... )
    """

    name: str
    category: str
    phase: str
    timestamp_us: int
    process_id: int
    thread_id: int
    duration_us: int | None = None
    args: dict[str, Any] | None = None
    scope: str | None = None
    id: str | None = None


@dataclass
class EndpointStats:
    """Per-endpoint statistics from a profiling run.

    Provides detailed metrics for a single endpoint/alias including
    task counts, timing, and rate limit information.

    Attributes:
        task_count: Total number of tasks executed on this endpoint.
        avg_duration_ms: Average task duration in milliseconds.
        max_duration_ms: Maximum task duration in milliseconds.
        min_duration_ms: Minimum task duration in milliseconds.
        max_concurrency_observed: Maximum number of concurrent tasks observed.
        rate_limit_events: Number of rate limit events recorded.
        total_wait_ms: Total time spent waiting for rate limits.

    Example:
        >>> stats = EndpointStats(
        ...     task_count=100,
        ...     avg_duration_ms=150.0,
        ...     max_duration_ms=500.0,
        ...     min_duration_ms=50.0,
        ...     max_concurrency_observed=5,
        ...     rate_limit_events=2,
        ...     total_wait_ms=1000.0,
        ... )
    """

    task_count: int
    avg_duration_ms: float
    max_duration_ms: float
    min_duration_ms: float
    max_concurrency_observed: int
    rate_limit_events: int
    total_wait_ms: float


@dataclass
class ProfilerStatistics:
    """Summary statistics from a profiling run.

    Provides aggregate metrics across all endpoints including task counts,
    timing information, and bubble (idle) time analysis.

    Attributes:
        total_tasks: Total number of tasks recorded.
        completed_tasks: Number of tasks that completed successfully.
        failed_tasks: Number of tasks that failed.
        total_duration_ms: Total wall-clock duration of the profiling run.
        avg_duration_ms: Average task duration across all tasks.
        max_duration_ms: Maximum task duration across all tasks.
        min_duration_ms: Minimum task duration across all tasks.
        total_bubble_ms: Total idle time (bubbles) observed.
        endpoints: Per-endpoint statistics dictionary.

    Example:
        >>> stats = profiler.get_statistics()
        >>> print(f"Total tasks: {stats.total_tasks}")
        >>> print(f"Avg latency: {stats.avg_duration_ms:.1f}ms")
    """

    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_duration_ms: float
    avg_duration_ms: float
    max_duration_ms: float
    min_duration_ms: float
    total_bubble_ms: float
    endpoints: dict[str, EndpointStats]


class TraceProfiler:
    """Collects execution traces in Chrome Trace Event Format.

    Captures task execution, rate limiting events, and concurrency metrics
    for visualization in Perfetto or Chrome DevTools. The profiler is
    thread-safe and can be used from multiple concurrent tasks.

    Attributes:
        include_counters: Whether to emit counter events for metrics.
        include_args: Whether to include task arguments in events.

    Example:
        >>> profiler = TraceProfiler(include_counters=True)
        >>>
        >>> # Record task lifecycle
        >>> profiler.task_start("task_1", "gpt-4", "MyModule")
        >>> # ... task executes ...
        >>> profiler.task_end("task_1", "gpt-4", duration_ms=150.0)
        >>>
        >>> # Add custom markers
        >>> profiler.add_instant_event("checkpoint", {"note": "batch complete"})
        >>>
        >>> # Export and analyze
        >>> profiler.export("./trace.json")
        >>> stats = profiler.get_statistics()
    """

    def __init__(
        self,
        include_counters: bool = True,
        include_args: bool = True,
    ) -> None:
        """Initialize the trace profiler.

        Args:
            include_counters: Whether to include counter events for queue depth
                and concurrency metrics. Defaults to True.
            include_args: Whether to include task arguments in trace events.
                May increase file size significantly. Defaults to True.

        Example:
            >>> profiler = TraceProfiler()
            >>> profiler.include_counters
            True
            >>>
            >>> # Disable counters for smaller traces
            >>> profiler = TraceProfiler(include_counters=False)
        """
        self.include_counters = include_counters
        self.include_args = include_args

        self._events: list[TraceEvent] = []
        self._start_time_ns: int = time.perf_counter_ns()
        self._start_wall_time: str = datetime.now(UTC).isoformat()

        # Track endpoint -> process ID mapping
        self._endpoint_pids: dict[str, int] = {}
        self._next_pid: int = 1

        # Track active slots per endpoint for thread ID assignment
        self._endpoint_slots: dict[str, set[int]] = defaultdict(set)
        self._task_slots: dict[str, tuple[int, int]] = {}  # task_id -> (pid, tid)

        # Counter state
        self._active_per_endpoint: dict[str, int] = defaultdict(int)
        self._queue_depth: int = 0

        # Task tracking for statistics
        self._task_start_times: dict[str, int] = {}  # task_id -> start timestamp_us
        self._task_durations: dict[str, float] = {}  # task_id -> duration_ms
        self._task_endpoints: dict[str, str] = {}  # task_id -> endpoint
        self._failed_tasks: set[str] = set()
        self._completed_tasks: set[str] = set()

        # Rate limit tracking for statistics
        self._rate_limit_counts: dict[str, int] = defaultdict(int)
        self._rate_limit_waits: dict[str, float] = defaultdict(float)

        # Max concurrency tracking
        self._max_concurrency: dict[str, int] = defaultdict(int)

        self._lock = threading.Lock()

    def _ts(self) -> int:
        """Get current timestamp in microseconds since profiler start.

        Returns:
            Timestamp in microseconds relative to profiler creation time.
        """
        return (time.perf_counter_ns() - self._start_time_ns) // 1000

    def _get_pid(self, endpoint: str) -> int:
        """Get or create process ID for an endpoint.

        Each unique endpoint gets a unique process ID, which maps to a
        separate row/lane in the trace visualization.

        Args:
            endpoint: The endpoint alias.

        Returns:
            Process ID for this endpoint.
        """
        if endpoint not in self._endpoint_pids:
            pid = self._next_pid
            self._endpoint_pids[endpoint] = pid
            self._next_pid += 1
            # Add metadata event for process name
            self._events.append(
                TraceEvent(
                    name="process_name",
                    category="__metadata",
                    phase="M",
                    timestamp_us=0,
                    process_id=pid,
                    thread_id=0,
                    args={"name": endpoint},
                )
            )
        return self._endpoint_pids[endpoint]

    def _acquire_slot(self, endpoint: str) -> int:
        """Get an available thread slot for an endpoint.

        Each concurrent task within an endpoint gets a unique thread ID,
        allowing visualization of parallel execution.

        Args:
            endpoint: The endpoint alias.

        Returns:
            Thread ID for this task.
        """
        slots = self._endpoint_slots[endpoint]
        # Find first available slot (1-indexed)
        tid = 1
        while tid in slots:
            tid += 1
        slots.add(tid)
        return tid

    def _release_slot(self, endpoint: str, tid: int) -> None:
        """Release a thread slot.

        Args:
            endpoint: The endpoint alias.
            tid: The thread ID to release.
        """
        self._endpoint_slots[endpoint].discard(tid)

    # ─────────────────────────────────────────────────────────────
    # Task Lifecycle Events
    # ─────────────────────────────────────────────────────────────

    def task_start(
        self,
        task_id: str,
        endpoint: str,
        module_name: str,
        args: dict[str, Any] | None = None,
    ) -> None:
        """Record task execution start.

        Creates a duration begin event ("B") for the task. Must be paired
        with a task_end() or task_failed() call.

        Args:
            task_id: Unique identifier for this task.
            endpoint: The endpoint alias (e.g., "gpt-4", "fast").
            module_name: The module class name (e.g., "LLMInference").
            args: Optional task arguments to include in the trace.

        Example:
            >>> profiler.task_start(
            ...     "task_123",
            ...     "gpt-4",
            ...     "MyModule",
            ...     {"input": "Hello, world!"},
            ... )
        """
        with self._lock:
            pid = self._get_pid(endpoint)
            tid = self._acquire_slot(endpoint)
            self._task_slots[task_id] = (pid, tid)

            # Track for statistics
            self._task_start_times[task_id] = self._ts()
            self._task_endpoints[task_id] = endpoint

            self._active_per_endpoint[endpoint] += 1

            # Update max concurrency
            current_active = self._active_per_endpoint[endpoint]
            if current_active > self._max_concurrency[endpoint]:
                self._max_concurrency[endpoint] = current_active

            event_args = {"endpoint": endpoint, "module": module_name}
            if self.include_args and args:
                event_args["input"] = args

            self._events.append(
                TraceEvent(
                    name=task_id,
                    category="llm_call",
                    phase="B",
                    timestamp_us=self._ts(),
                    process_id=pid,
                    thread_id=tid,
                    args=event_args,
                )
            )

            if self.include_counters:
                self._emit_counter_event()

    def task_end(
        self,
        task_id: str,
        endpoint: str,
        duration_ms: float,
        result_summary: dict[str, Any] | None = None,
    ) -> None:
        """Record task execution completion.

        Creates a duration end event ("E") for the task.

        Args:
            task_id: Unique identifier for this task.
            endpoint: The endpoint alias.
            duration_ms: Task duration in milliseconds.
            result_summary: Optional result summary to include in the trace.

        Example:
            >>> profiler.task_end(
            ...     "task_123",
            ...     "gpt-4",
            ...     duration_ms=150.5,
            ...     {"tokens": 100},
            ... )
        """
        with self._lock:
            pid, tid = self._task_slots.pop(task_id, (1, 1))
            self._release_slot(endpoint, tid)

            self._active_per_endpoint[endpoint] = max(
                0, self._active_per_endpoint[endpoint] - 1
            )

            # Track for statistics
            self._task_durations[task_id] = duration_ms
            self._completed_tasks.add(task_id)

            event_args: dict[str, Any] = {"duration_ms": duration_ms}
            if self.include_args and result_summary:
                event_args["result"] = result_summary

            self._events.append(
                TraceEvent(
                    name=task_id,
                    category="llm_call",
                    phase="E",
                    timestamp_us=self._ts(),
                    process_id=pid,
                    thread_id=tid,
                    args=event_args,
                )
            )

            if self.include_counters:
                self._emit_counter_event()

    def task_failed(
        self,
        task_id: str,
        endpoint: str,
        error: str,
    ) -> None:
        """Record task failure.

        Creates a duration end event ("E") with error information.

        Args:
            task_id: Unique identifier for this task.
            endpoint: The endpoint alias.
            error: Error message or description.

        Example:
            >>> profiler.task_failed("task_123", "gpt-4", "Connection timeout")
        """
        with self._lock:
            pid, tid = self._task_slots.pop(task_id, (1, 1))
            self._release_slot(endpoint, tid)

            self._active_per_endpoint[endpoint] = max(
                0, self._active_per_endpoint[endpoint] - 1
            )

            # Track for statistics
            self._failed_tasks.add(task_id)

            # Calculate duration from start time if available
            if task_id in self._task_start_times:
                start_us = self._task_start_times[task_id]
                duration_ms = (self._ts() - start_us) / 1000.0
                self._task_durations[task_id] = duration_ms

            self._events.append(
                TraceEvent(
                    name=task_id,
                    category="llm_call",
                    phase="E",
                    timestamp_us=self._ts(),
                    process_id=pid,
                    thread_id=tid,
                    args={"error": error, "status": "failed"},
                )
            )

            if self.include_counters:
                self._emit_counter_event()

    # ─────────────────────────────────────────────────────────────
    # Rate Limiting Events
    # ─────────────────────────────────────────────────────────────

    def rate_limit_hit(
        self,
        endpoint: str,
        retry_after: float | None = None,
    ) -> None:
        """Record a rate limit event.

        Creates an instant event ("i") marking when rate limiting occurred.

        Args:
            endpoint: The endpoint alias that hit the rate limit.
            retry_after: Optional seconds until retry is allowed.

        Example:
            >>> profiler.rate_limit_hit("gpt-4", retry_after=30.0)
        """
        with self._lock:
            pid = self._get_pid(endpoint)

            # Track for statistics
            self._rate_limit_counts[endpoint] += 1

            self._events.append(
                TraceEvent(
                    name="rate_limit",
                    category="rate_limit",
                    phase="i",
                    timestamp_us=self._ts(),
                    process_id=pid,
                    thread_id=0,
                    scope="p",  # Process-scoped instant event
                    args={"retry_after": retry_after},
                )
            )

    def rate_limit_wait_start(
        self,
        endpoint: str,
        task_id: str,
    ) -> None:
        """Record start of rate limit wait.

        Creates an async begin event ("b") for rate limit waiting.

        Args:
            endpoint: The endpoint alias.
            task_id: Task that is waiting.

        Example:
            >>> profiler.rate_limit_wait_start("gpt-4", "task_123")
        """
        with self._lock:
            pid = self._get_pid(endpoint)

            self._events.append(
                TraceEvent(
                    name="rate_limit_wait",
                    category="rate_limit",
                    phase="b",
                    timestamp_us=self._ts(),
                    process_id=pid,
                    thread_id=0,
                    id=f"wait_{task_id}",
                )
            )

    def rate_limit_wait_end(
        self,
        endpoint: str,
        task_id: str,
        wait_ms: float,
    ) -> None:
        """Record end of rate limit wait.

        Creates an async end event ("e") for rate limit waiting.

        Args:
            endpoint: The endpoint alias.
            task_id: Task that was waiting.
            wait_ms: Time spent waiting in milliseconds.

        Example:
            >>> profiler.rate_limit_wait_end("gpt-4", "task_123", wait_ms=5000.0)
        """
        with self._lock:
            pid = self._get_pid(endpoint)

            # Track for statistics
            self._rate_limit_waits[endpoint] += wait_ms

            self._events.append(
                TraceEvent(
                    name="rate_limit_wait",
                    category="rate_limit",
                    phase="e",
                    timestamp_us=self._ts(),
                    process_id=pid,
                    thread_id=0,
                    id=f"wait_{task_id}",
                    args={"wait_ms": wait_ms},
                )
            )

    # ─────────────────────────────────────────────────────────────
    # Queue Events
    # ─────────────────────────────────────────────────────────────

    def task_queued(self, task_id: str) -> None:
        """Record task added to queue.

        Updates the queue depth counter.

        Args:
            task_id: The task that was queued.

        Example:
            >>> profiler.task_queued("task_123")
        """
        with self._lock:
            self._queue_depth += 1
            if self.include_counters:
                self._emit_counter_event()

    def task_dequeued(self, task_id: str) -> None:
        """Record task removed from queue.

        Updates the queue depth counter.

        Args:
            task_id: The task that was dequeued.

        Example:
            >>> profiler.task_dequeued("task_123")
        """
        with self._lock:
            self._queue_depth = max(0, self._queue_depth - 1)
            if self.include_counters:
                self._emit_counter_event()

    # ─────────────────────────────────────────────────────────────
    # Counter Events
    # ─────────────────────────────────────────────────────────────

    def _emit_counter_event(self) -> None:
        """Emit a counter event with current state.

        Called internally when task state changes. Creates a counter
        event ("C") with queue depth and per-endpoint concurrency.
        """
        # Global counters (pid=0)
        args: dict[str, Any] = {"queue_depth": self._queue_depth}

        # Per-endpoint concurrency
        for endpoint, count in self._active_per_endpoint.items():
            args[f"{endpoint}_active"] = count

        self._events.append(
            TraceEvent(
                name="counters",
                category="metrics",
                phase="C",
                timestamp_us=self._ts(),
                process_id=0,
                thread_id=0,
                args=args,
            )
        )

    # ─────────────────────────────────────────────────────────────
    # Custom Events
    # ─────────────────────────────────────────────────────────────

    def add_instant_event(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        scope: str = "g",
    ) -> None:
        """Add a custom instant event marker.

        Creates an instant event ("i") that appears as a marker in the
        trace visualization.

        Args:
            name: Display name for the event.
            args: Optional arguments to include.
            scope: Event scope - "g" (global), "p" (process), "t" (thread).
                Defaults to "g" (global).

        Example:
            >>> profiler.add_instant_event(
            ...     "checkpoint",
            ...     {"note": "batch 1 complete"},
            ... )
        """
        with self._lock:
            self._events.append(
                TraceEvent(
                    name=name,
                    category="custom",
                    phase="i",
                    timestamp_us=self._ts(),
                    process_id=0,
                    thread_id=0,
                    scope=scope,
                    args=args,
                )
            )

    # ─────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────

    def export(self, path: Path | str) -> None:
        """Export trace to Chrome Trace Format JSON file.

        Writes all collected events to a JSON file that can be opened
        in Perfetto (ui.perfetto.dev) or Chrome DevTools (chrome://tracing).

        Args:
            path: Output file path. Parent directories are created if needed.

        Example:
            >>> profiler.export("./traces/run_001.json")
            >>> # Open the file at https://ui.perfetto.dev
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            trace_events = []
            for event in self._events:
                trace_event: dict[str, Any] = {
                    "name": event.name,
                    "cat": event.category,
                    "ph": event.phase,
                    "ts": event.timestamp_us,
                    "pid": event.process_id,
                    "tid": event.thread_id,
                }

                if event.duration_us is not None:
                    trace_event["dur"] = event.duration_us
                if event.args:
                    trace_event["args"] = event.args
                if event.scope:
                    trace_event["s"] = event.scope
                if event.id:
                    trace_event["id"] = event.id

                trace_events.append(trace_event)

            output = {
                "traceEvents": trace_events,
                "metadata": {
                    "inf_engine_version": "0.1.0",
                    "start_time": self._start_wall_time,
                    "total_events": len(trace_events),
                },
            }

            path.write_text(json.dumps(output, indent=2))

    def get_statistics(self) -> ProfilerStatistics:
        """Compute summary statistics from collected events.

        Analyzes recorded events to produce aggregate statistics including
        task counts, durations, and per-endpoint breakdowns.

        Returns:
            ProfilerStatistics with aggregate metrics.

        Example:
            >>> stats = profiler.get_statistics()
            >>> print(f"Total tasks: {stats.total_tasks}")
            >>> print(f"Avg latency: {stats.avg_duration_ms:.1f}ms")
            >>> for name, ep in stats.endpoints.items():
            ...     print(f"{name}: {ep.task_count} tasks")
        """
        with self._lock:
            total_tasks = len(self._task_start_times)
            completed = len(self._completed_tasks)
            failed = len(self._failed_tasks)

            # Calculate overall duration stats
            durations = list(self._task_durations.values())
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                min_duration = min(durations)
            else:
                avg_duration = 0.0
                max_duration = 0.0
                min_duration = 0.0

            # Calculate total duration from first to last event
            total_duration_ms = self._ts() / 1000.0

            # Calculate bubble time (rough estimate: total time - sum of active time)
            # This is a simplified estimate
            total_active_time = sum(durations)
            total_bubble_ms = max(0.0, total_duration_ms - total_active_time)

            # Per-endpoint statistics
            endpoints: dict[str, EndpointStats] = {}

            # Group tasks by endpoint
            endpoint_tasks: dict[str, list[str]] = defaultdict(list)
            for task_id, endpoint in self._task_endpoints.items():
                endpoint_tasks[endpoint].append(task_id)

            for endpoint, task_ids in endpoint_tasks.items():
                ep_durations = [
                    self._task_durations[tid]
                    for tid in task_ids
                    if tid in self._task_durations
                ]

                if ep_durations:
                    ep_avg = sum(ep_durations) / len(ep_durations)
                    ep_max = max(ep_durations)
                    ep_min = min(ep_durations)
                else:
                    ep_avg = 0.0
                    ep_max = 0.0
                    ep_min = 0.0

                endpoints[endpoint] = EndpointStats(
                    task_count=len(task_ids),
                    avg_duration_ms=ep_avg,
                    max_duration_ms=ep_max,
                    min_duration_ms=ep_min,
                    max_concurrency_observed=self._max_concurrency.get(endpoint, 0),
                    rate_limit_events=self._rate_limit_counts.get(endpoint, 0),
                    total_wait_ms=self._rate_limit_waits.get(endpoint, 0.0),
                )

            return ProfilerStatistics(
                total_tasks=total_tasks,
                completed_tasks=completed,
                failed_tasks=failed,
                total_duration_ms=total_duration_ms,
                avg_duration_ms=avg_duration,
                max_duration_ms=max_duration,
                min_duration_ms=min_duration,
                total_bubble_ms=total_bubble_ms,
                endpoints=endpoints,
            )

    @property
    def event_count(self) -> int:
        """Get the number of recorded events.

        Returns:
            Total number of trace events recorded.

        Example:
            >>> profiler = TraceProfiler()
            >>> profiler.event_count
            0
        """
        with self._lock:
            return len(self._events)
