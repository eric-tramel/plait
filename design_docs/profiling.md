# Profiling and Performance

The profiling system captures detailed timing and execution data, enabling visualization of pipeline performance, identification of bottlenecks, and detection of execution "bubbles" (idle periods).

## Design Goals

1. **Zero Overhead When Disabled**: No performance impact when profiling is off
2. **Standard Format**: Use Chrome Trace Event Format for compatibility with existing tools
3. **Rich Context**: Capture enough data to understand concurrency, rate limiting, and dependencies
4. **Simple Integration**: Enable via ExecutionSettings with minimal configuration
5. **Actionable Insights**: Surface bubbles, bottlenecks, and underutilization

## Key Design Decisions

- **Chrome Trace Event Format**: Industry standard supported by Perfetto, Chrome DevTools, and many other tools. Avoids building custom visualization.
- **Process = Endpoint**: Each endpoint alias maps to a "process" in the trace, making per-endpoint concurrency visible.
- **Thread = Concurrent Slot**: Each concurrent execution slot within an endpoint maps to a "thread", showing actual parallelism.
- **Lazy File Writing**: Events are buffered in memory and written on context exit to avoid I/O during execution.
- **Configuration via ExecutionSettings**: Profiling is enabled through the execution context manager, keeping module definitions clean.

## Chrome Trace Event Format

The output is a JSON file compatible with:
- **Perfetto** ([ui.perfetto.dev](https://ui.perfetto.dev)) - Recommended, most powerful
- **Chrome DevTools** (`chrome://tracing`)
- **Speedscope** ([speedscope.app](https://www.speedscope.app))

### Format Structure

```json
{
  "traceEvents": [
    {
      "name": "task_abc123",
      "cat": "llm_call",
      "ph": "X",
      "ts": 1000000,
      "dur": 500000,
      "pid": 1,
      "tid": 1,
      "args": {"endpoint": "gpt-4", "tokens": 150, "retry_count": 0}
    }
  ],
  "metadata": {
    "inf_engine_version": "0.1.0",
    "start_time": "2024-01-15T10:30:00Z"
  }
}
```

### Event Types

| Event Type | `ph` | Use Case |
|------------|------|----------|
| Complete | `X` | Task execution (has duration) |
| Duration Begin | `B` | Long operation start |
| Duration End | `E` | Long operation end |
| Async Begin | `b` | Overlapping async operations |
| Async End | `e` | Async operation completion |
| Instant | `i` | Point events (rate limit, retry) |
| Counter | `C` | Metrics over time (queue depth, concurrency) |
| Metadata | `M` | Name processes/threads |

### Visualization Concept

```
Process = Endpoint (e.g., "gpt-4", "claude-3")
Thread = Concurrent slot within that endpoint

Timeline:
[gpt-4 slot 1]  |████ task_1 ████|        |███ task_5 ███|
[gpt-4 slot 2]  |██ task_2 ██|  |████ task_4 ████|
[claude slot 1]     |████████ task_3 ████████|
                          ↑
                    "bubble" visible here
```

## Configuration via ExecutionSettings

Profiling is enabled through the `ExecutionSettings` context manager:

```python
from inf_engine import ExecutionSettings, ResourceConfig

resources = ResourceConfig({
    "fast": {"model": "gpt-4o-mini", "max_concurrent": 20},
    "smart": {"model": "gpt-4o", "max_concurrent": 5},
})

# Enable profiling with trace output
async with ExecutionSettings(
    resources=resources,
    profile=True,
    profile_path="./traces/run_001.json",
) as settings:
    result = await pipeline(input_data)

# Trace file written automatically on context exit
# Open with: https://ui.perfetto.dev
```

### ExecutionSettings Profiling Options

```python
@dataclass
class ExecutionSettings:
    """Context manager for shared execution configuration."""

    # ... existing fields ...

    # Profiling configuration
    profile: bool = False
    """Enable execution profiling."""

    profile_path: Path | str | None = None
    """Output path for trace file. If None with profile=True, uses
    './traces/trace_{timestamp}.json'."""

    profile_counters: bool = True
    """Include counter events for queue depth and concurrency metrics."""

    profile_include_args: bool = True
    """Include task arguments in trace events (may increase file size)."""
```

### Programmatic Access

For advanced use cases, access the profiler directly:

```python
async with ExecutionSettings(
    resources=resources,
    profile=True,
) as settings:
    result = await pipeline(input_data)

    # Access profiler for custom events or analysis
    profiler = settings.profiler
    profiler.add_instant_event("custom_marker", {"note": "checkpoint reached"})

    # Get summary statistics
    stats = profiler.get_statistics()
    print(f"Total tasks: {stats.total_tasks}")
    print(f"Avg latency: {stats.avg_duration_ms:.1f}ms")
    print(f"Bubble time: {stats.total_bubble_ms:.1f}ms")
```

## TraceProfiler

The core profiling implementation:

```python
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from typing import Any


@dataclass
class TraceEvent:
    """A single trace event in Chrome Trace Format."""

    name: str
    category: str
    phase: str  # "X", "B", "E", "b", "e", "i", "C", "M"
    timestamp_us: int
    process_id: int
    thread_id: int
    duration_us: int | None = None
    args: dict[str, Any] | None = None
    scope: str | None = None  # For instant events: "g", "p", "t"
    id: str | None = None  # For async events


@dataclass
class ProfilerStatistics:
    """Summary statistics from a profiling run."""

    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_duration_ms: float
    avg_duration_ms: float
    max_duration_ms: float
    min_duration_ms: float
    total_bubble_ms: float
    endpoints: dict[str, EndpointStats]


@dataclass
class EndpointStats:
    """Per-endpoint statistics."""

    task_count: int
    avg_duration_ms: float
    max_concurrency_observed: int
    rate_limit_events: int
    total_wait_ms: float


class TraceProfiler:
    """
    Collects execution traces in Chrome Trace Event Format.

    Captures task execution, rate limiting events, and concurrency
    metrics for visualization in Perfetto or Chrome DevTools.
    """

    def __init__(
        self,
        include_counters: bool = True,
        include_args: bool = True,
    ):
        self.include_counters = include_counters
        self.include_args = include_args

        self._events: list[TraceEvent] = []
        self._start_time_ns: int = time.perf_counter_ns()
        self._start_wall_time: str = datetime.utcnow().isoformat() + "Z"

        # Track endpoint -> process ID mapping
        self._endpoint_pids: dict[str, int] = {}
        self._next_pid: int = 1

        # Track active slots per endpoint for thread ID assignment
        self._endpoint_slots: dict[str, set[int]] = defaultdict(set)
        self._task_slots: dict[str, tuple[int, int]] = {}  # task_id -> (pid, tid)

        # Counter state
        self._active_per_endpoint: dict[str, int] = defaultdict(int)
        self._queue_depth: int = 0

        self._lock = threading.Lock()

    def _ts(self) -> int:
        """Get current timestamp in microseconds since profiler start."""
        return (time.perf_counter_ns() - self._start_time_ns) // 1000

    def _get_pid(self, endpoint: str) -> int:
        """Get or create process ID for an endpoint."""
        if endpoint not in self._endpoint_pids:
            self._endpoint_pids[endpoint] = self._next_pid
            self._next_pid += 1
            # Add metadata event for process name
            self._events.append(TraceEvent(
                name="process_name",
                category="__metadata",
                phase="M",
                timestamp_us=0,
                process_id=self._endpoint_pids[endpoint],
                thread_id=0,
                args={"name": endpoint},
            ))
        return self._endpoint_pids[endpoint]

    def _acquire_slot(self, endpoint: str) -> int:
        """Get an available thread slot for an endpoint."""
        slots = self._endpoint_slots[endpoint]
        # Find first available slot (1-indexed)
        tid = 1
        while tid in slots:
            tid += 1
        slots.add(tid)
        return tid

    def _release_slot(self, endpoint: str, tid: int) -> None:
        """Release a thread slot."""
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
        """Record task execution start."""
        with self._lock:
            pid = self._get_pid(endpoint)
            tid = self._acquire_slot(endpoint)
            self._task_slots[task_id] = (pid, tid)

            self._active_per_endpoint[endpoint] += 1

            event_args = {"endpoint": endpoint, "module": module_name}
            if self.include_args and args:
                event_args["input"] = args

            self._events.append(TraceEvent(
                name=task_id,
                category="llm_call",
                phase="B",
                timestamp_us=self._ts(),
                process_id=pid,
                thread_id=tid,
                args=event_args,
            ))

            if self.include_counters:
                self._emit_counter_event()

    def task_end(
        self,
        task_id: str,
        endpoint: str,
        duration_ms: float,
        result_summary: dict[str, Any] | None = None,
    ) -> None:
        """Record task execution completion."""
        with self._lock:
            pid, tid = self._task_slots.pop(task_id, (1, 1))
            self._release_slot(endpoint, tid)

            self._active_per_endpoint[endpoint] -= 1

            event_args = {"duration_ms": duration_ms}
            if self.include_args and result_summary:
                event_args["result"] = result_summary

            self._events.append(TraceEvent(
                name=task_id,
                category="llm_call",
                phase="E",
                timestamp_us=self._ts(),
                process_id=pid,
                thread_id=tid,
                args=event_args,
            ))

            if self.include_counters:
                self._emit_counter_event()

    def task_failed(
        self,
        task_id: str,
        endpoint: str,
        error: str,
    ) -> None:
        """Record task failure."""
        with self._lock:
            pid, tid = self._task_slots.pop(task_id, (1, 1))
            self._release_slot(endpoint, tid)

            self._active_per_endpoint[endpoint] -= 1

            self._events.append(TraceEvent(
                name=task_id,
                category="llm_call",
                phase="E",
                timestamp_us=self._ts(),
                process_id=pid,
                thread_id=tid,
                args={"error": error, "status": "failed"},
            ))

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
        """Record a rate limit event."""
        with self._lock:
            pid = self._get_pid(endpoint)

            self._events.append(TraceEvent(
                name="rate_limit",
                category="rate_limit",
                phase="i",
                timestamp_us=self._ts(),
                process_id=pid,
                thread_id=0,
                scope="p",  # Process-scoped instant event
                args={"retry_after": retry_after},
            ))

    def rate_limit_wait_start(
        self,
        endpoint: str,
        task_id: str,
    ) -> None:
        """Record start of rate limit wait."""
        with self._lock:
            pid = self._get_pid(endpoint)

            self._events.append(TraceEvent(
                name="rate_limit_wait",
                category="rate_limit",
                phase="b",
                timestamp_us=self._ts(),
                process_id=pid,
                thread_id=0,
                id=f"wait_{task_id}",
            ))

    def rate_limit_wait_end(
        self,
        endpoint: str,
        task_id: str,
        wait_ms: float,
    ) -> None:
        """Record end of rate limit wait."""
        with self._lock:
            pid = self._get_pid(endpoint)

            self._events.append(TraceEvent(
                name="rate_limit_wait",
                category="rate_limit",
                phase="e",
                timestamp_us=self._ts(),
                process_id=pid,
                thread_id=0,
                id=f"wait_{task_id}",
                args={"wait_ms": wait_ms},
            ))

    # ─────────────────────────────────────────────────────────────
    # Queue Events
    # ─────────────────────────────────────────────────────────────

    def task_queued(self, task_id: str) -> None:
        """Record task added to queue."""
        with self._lock:
            self._queue_depth += 1
            if self.include_counters:
                self._emit_counter_event()

    def task_dequeued(self, task_id: str) -> None:
        """Record task removed from queue."""
        with self._lock:
            self._queue_depth = max(0, self._queue_depth - 1)
            if self.include_counters:
                self._emit_counter_event()

    # ─────────────────────────────────────────────────────────────
    # Counter Events
    # ─────────────────────────────────────────────────────────────

    def _emit_counter_event(self) -> None:
        """Emit a counter event with current state."""
        # Global counters (pid=0)
        args = {"queue_depth": self._queue_depth}

        # Per-endpoint concurrency
        for endpoint, count in self._active_per_endpoint.items():
            args[f"{endpoint}_active"] = count

        self._events.append(TraceEvent(
            name="counters",
            category="metrics",
            phase="C",
            timestamp_us=self._ts(),
            process_id=0,
            thread_id=0,
            args=args,
        ))

    # ─────────────────────────────────────────────────────────────
    # Custom Events
    # ─────────────────────────────────────────────────────────────

    def add_instant_event(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        scope: str = "g",
    ) -> None:
        """Add a custom instant event marker."""
        with self._lock:
            self._events.append(TraceEvent(
                name=name,
                category="custom",
                phase="i",
                timestamp_us=self._ts(),
                process_id=0,
                thread_id=0,
                scope=scope,
                args=args,
            ))

    # ─────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────

    def export(self, path: Path | str) -> None:
        """Export trace to Chrome Trace Format JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        trace_events = []
        for event in self._events:
            trace_event = {
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
        """Compute summary statistics from collected events."""
        # Implementation computes stats from self._events
        ...
```

## Integration Points

### Scheduler Integration

The Scheduler calls profiler methods during task execution:

```python
class Scheduler:
    def __init__(
        self,
        resource_manager: ResourceManager,
        max_concurrent: int = 100,
        profiler: TraceProfiler | None = None,
    ):
        self.profiler = profiler
        # ...

    async def _execute_task(self, state: ExecutionState, task: Task) -> None:
        alias = getattr(task.module, "alias", None)

        # Record task start
        if self.profiler and alias:
            self.profiler.task_start(
                task_id=task.node_id,
                endpoint=alias,
                module_name=task.module.__class__.__name__,
                args={"input": str(task.args)[:100]} if task.args else None,
            )

        start_time = time.time()

        try:
            result = await self._do_execute(task)
            duration_ms = (time.time() - start_time) * 1000

            # Record task completion
            if self.profiler and alias:
                self.profiler.task_end(
                    task_id=task.node_id,
                    endpoint=alias,
                    duration_ms=duration_ms,
                )

            # ... rest of completion logic

        except RateLimitError as e:
            if self.profiler and alias:
                self.profiler.rate_limit_hit(alias, e.retry_after)
                self.profiler.task_failed(task.node_id, alias, "rate_limited")
            # ... requeue logic

        except Exception as e:
            if self.profiler and alias:
                self.profiler.task_failed(task.node_id, alias, str(e))
            # ... error handling
```

### ResourceManager Integration

Rate limiter wait times are captured:

```python
class ResourceManager:
    async def acquire_rate_limit(
        self,
        alias: str,
        profiler: TraceProfiler | None = None,
        task_id: str | None = None,
    ) -> None:
        """Acquire rate limit token, recording wait time if profiling."""
        rate_limiter = self._rate_limiters.get(alias)
        if not rate_limiter:
            return

        if profiler and task_id:
            profiler.rate_limit_wait_start(alias, task_id)

        start = time.time()
        await rate_limiter.acquire()
        wait_ms = (time.time() - start) * 1000

        if profiler and task_id and wait_ms > 1:  # Only record meaningful waits
            profiler.rate_limit_wait_end(alias, task_id, wait_ms)
```

### ExecutionSettings Integration

The context manager creates and manages the profiler:

```python
class ExecutionSettings:
    def __init__(
        self,
        # ... existing params ...
        profile: bool = False,
        profile_path: Path | str | None = None,
        profile_counters: bool = True,
        profile_include_args: bool = True,
    ):
        self.profile = profile
        self.profile_path = profile_path
        self.profile_counters = profile_counters
        self.profile_include_args = profile_include_args
        self.profiler: TraceProfiler | None = None

    async def __aenter__(self) -> Self:
        self._token = _execution_settings.set(self)

        if self.profile:
            self.profiler = TraceProfiler(
                include_counters=self.profile_counters,
                include_args=self.profile_include_args,
            )

        # ... rest of setup ...
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Export trace file on exit
        if self.profiler:
            path = self.profile_path
            if path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"./traces/trace_{timestamp}.json"
            self.profiler.export(path)

        # ... rest of cleanup ...
        _execution_settings.reset(self._token)
```

## Viewing Traces

### Perfetto (Recommended)

1. Navigate to [ui.perfetto.dev](https://ui.perfetto.dev)
2. Drag and drop the JSON trace file
3. Use keyboard shortcuts:
   - `W/S`: Zoom in/out
   - `A/D`: Pan left/right
   - `F`: Fit to view
   - `/`: Search

### Chrome DevTools

1. Open Chrome
2. Navigate to `chrome://tracing`
3. Click "Load" and select the JSON file

### Programmatic Analysis

```python
from inf_engine.profiling import TraceAnalyzer

# Load and analyze a trace
analyzer = TraceAnalyzer.from_file("./traces/run_001.json")

# Find bottlenecks
bottlenecks = analyzer.find_bottlenecks()
for b in bottlenecks:
    print(f"{b.endpoint}: {b.avg_wait_ms:.1f}ms avg wait, {b.utilization:.1%} utilization")

# Find bubbles (idle periods)
bubbles = analyzer.find_bubbles(min_duration_ms=100)
for bubble in bubbles:
    print(f"Bubble at {bubble.start_ms:.0f}ms: {bubble.duration_ms:.0f}ms idle")

# Get per-endpoint breakdown
for endpoint, stats in analyzer.endpoint_stats().items():
    print(f"{endpoint}: {stats.task_count} tasks, {stats.avg_duration_ms:.0f}ms avg")
```

## Example: Complete Profiling Flow

```python
from inf_engine import InferenceModule, LLMInference, ExecutionSettings, ResourceConfig

class AnalysisPipeline(InferenceModule):
    def __init__(self):
        super().__init__()
        self.extract = LLMInference(alias="fast")
        self.analyze = LLMInference(alias="smart")
        self.summarize = LLMInference(alias="fast")

    def forward(self, doc: str) -> str:
        entities = self.extract(f"Extract entities from: {doc}")
        analysis = self.analyze(f"Analyze: {entities}")
        return self.summarize(f"Summarize: {analysis}")


# Configure resources
resources = ResourceConfig({
    "fast": {"model": "gpt-4o-mini", "max_concurrent": 20, "rpm": 1000},
    "smart": {"model": "gpt-4o", "max_concurrent": 5, "rpm": 100},
})

# Run with profiling enabled
async with ExecutionSettings(
    resources=resources,
    profile=True,
    profile_path="./traces/analysis_run.json",
) as settings:
    pipeline = AnalysisPipeline()

    # Process batch of documents
    documents = ["Doc 1...", "Doc 2...", "Doc 3...", ...]
    results = await pipeline(documents)

    # Print quick stats
    stats = settings.profiler.get_statistics()
    print(f"Completed {stats.completed_tasks}/{stats.total_tasks} tasks")
    print(f"Bubble time: {stats.total_bubble_ms:.0f}ms ({stats.total_bubble_ms/stats.total_duration_ms:.1%})")

# Trace file automatically written to ./traces/analysis_run.json
# Open with: https://ui.perfetto.dev
```

## Performance Considerations

- **Memory**: Events are kept in memory until export. For very long runs (>100k tasks), consider periodic flushing.
- **Lock Contention**: A single lock protects event list. For extreme concurrency, consider lock-free approaches.
- **File Size**: With `include_args=True`, trace files can grow large. Disable for production profiling of large batches.
- **Timestamp Resolution**: Microsecond resolution (sufficient for LLM latencies in milliseconds range).

## Future Extensions

- **Streaming Export**: Write events incrementally for long-running pipelines
- **Sampling**: Probabilistic event capture for very high throughput
- **Memory Profiling**: Track intermediate result sizes
- **Cost Tracking**: Include token counts and estimated costs in events
- **Diff Analysis**: Compare two traces to identify regressions
