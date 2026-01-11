# Execution Engine

The execution engine transforms traced graphs into efficient async execution, managing task scheduling, state tracking, error handling, and checkpointing.

## Design Goals

1. **Maximum Throughput**: Parallelize independent operations, minimize idle GPU time
2. **State Tracking**: Know exactly what's pending, running, and completed
3. **Graceful Recovery**: Re-enqueue failed tasks with proper cascade handling
4. **Memory Bounded**: Limit inflight operations to prevent OOM
5. **Persistence**: Checkpoint progress for long-running pipelines

## Key Design Decisions

- **LLM execution always via ResourceManager**: All `LLMInference` modules execute through the `ResourceManager`, never directly. This ensures consistent rate limiting, metrics, and endpoint management. Non-LLM modules can execute directly.
- **Inputs bound at execution**: Tracing assigns input refs, but actual input
  values are provided at execution time via `valueify()` and resolved through
  `ValueRef` placeholders.
- **Priority ordering**: Lower priority values indicate higher precedence (matches heapq semantics).

## Execution State

The `ExecutionState` tracks all aspects of a graph execution:

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any
import asyncio
from collections import defaultdict

class TaskStatus(Enum):
    PENDING = auto()      # Ready to execute
    BLOCKED = auto()      # Waiting on dependencies
    IN_PROGRESS = auto()  # Currently executing
    COMPLETED = auto()    # Finished successfully
    FAILED = auto()       # Finished with error
    CANCELLED = auto()    # Dropped due to parent failure


@dataclass
class Task:
    """A single executable unit."""

    node_id: str
    module: Module
    args: tuple
    kwargs: dict
    dependencies: list[str]
    priority: int = 0
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other: Task) -> bool:
        """For priority queue ordering (lower priority = higher precedence)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


@dataclass
class TaskResult:
    """Result of a completed task."""

    node_id: str
    value: Value
    duration_ms: float
    retry_count: int


@dataclass
class ValueRef:
    """Placeholder for a dependency Value produced by another node."""

    ref: str

# See values.md for the ValueRef spec.


class ExecutionState:
    """
    Tracks the complete state of a graph execution.

    Maintains parent/child relationships for cascading operations.
    """

    def __init__(self, graph: InferenceGraph):
        self.graph = graph
        self.status: dict[str, TaskStatus] = {}
        self.results: dict[str, TaskResult] = {}
        self.errors: dict[str, Exception] = {}

        # Task management
        self.pending: asyncio.PriorityQueue[Task] = asyncio.PriorityQueue()
        self.in_progress: dict[str, Task] = {}

        # Dependency tracking
        self.waiting_on: dict[str, set[str]] = defaultdict(set)  # node -> deps not done
        self.dependents: dict[str, set[str]] = defaultdict(set)  # node -> nodes waiting

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Set up initial state from graph."""
        for node_id, node in self.graph.nodes.items():
            self.status[node_id] = TaskStatus.BLOCKED

            # Track dependencies
            for dep_id in node.dependencies:
                self.waiting_on[node_id].add(dep_id)
                self.dependents[dep_id].add(node_id)

            # Nodes with no dependencies are ready
            if not node.dependencies:
                self._make_ready(node_id)

    def _make_ready(self, node_id: str) -> None:
        """Move a task to the pending queue."""
        node = self.graph.nodes[node_id]
        self.status[node_id] = TaskStatus.PENDING

        task = Task(
            node_id=node_id,
            module=node.module,
            args=self._resolve_args(node.args),
            kwargs=self._resolve_kwargs(node.kwargs),
            dependencies=node.dependencies,
            priority=node.priority,
        )

        self.pending.put_nowait(task)

    def _resolve_args(self, args: tuple) -> tuple:
        """Resolve ValueRef placeholders to actual Values."""
        resolved = []
        for arg in args:
            if isinstance(arg, ValueRef) and arg.ref in self.results:
                resolved.append(self.results[arg.ref].value)
            else:
                resolved.append(arg)
        return tuple(resolved)

    def _resolve_kwargs(self, kwargs: dict) -> dict:
        """Resolve ValueRef placeholders to actual Values."""
        resolved = {}
        for key, value in kwargs.items():
            if isinstance(value, ValueRef) and value.ref in self.results:
                resolved[key] = self.results[value.ref].value
            else:
                resolved[key] = value
        return resolved

    async def get_next_task(self) -> Task | None:
        """Get the next task to execute."""
        if self.pending.empty():
            return None

        task = await self.pending.get()
        self.status[task.node_id] = TaskStatus.IN_PROGRESS
        self.in_progress[task.node_id] = task
        return task

    def mark_complete(self, node_id: str, result: TaskResult) -> list[str]:
        """
        Mark a task as complete and return newly-ready node IDs.
        """
        self.status[node_id] = TaskStatus.COMPLETED
        self.results[node_id] = result
        self.in_progress.pop(node_id, None)

        # Find newly-ready dependents
        newly_ready = []
        for dependent_id in self.dependents[node_id]:
            self.waiting_on[dependent_id].discard(node_id)

            if not self.waiting_on[dependent_id]:
                if self.status[dependent_id] == TaskStatus.BLOCKED:
                    self._make_ready(dependent_id)
                    newly_ready.append(dependent_id)

        return newly_ready

    def mark_failed(self, node_id: str, error: Exception) -> None:
        """Mark a task as failed and cancel all descendants."""
        self.status[node_id] = TaskStatus.FAILED
        self.errors[node_id] = error
        self.in_progress.pop(node_id, None)

        # Cancel all descendants
        descendants = self.graph.descendants(node_id)
        for desc_id in descendants:
            self.status[desc_id] = TaskStatus.CANCELLED

    def requeue(self, node_id: str) -> None:
        """
        Re-enqueue a task and drop all its descendants.

        Used when a task hits rate limiting and needs to retry.
        """
        # Remove from in-progress
        task = self.in_progress.pop(node_id, None)
        if task is None:
            return

        # Drop all descendants from pending
        descendants = self.graph.descendants(node_id)
        for desc_id in descendants:
            self.status[desc_id] = TaskStatus.BLOCKED
            # Re-add dependencies
            node = self.graph.nodes[desc_id]
            self.waiting_on[desc_id] = set(node.dependencies)

        # Re-queue the task with incremented retry count
        task.retry_count += 1
        self.status[node_id] = TaskStatus.PENDING
        self.pending.put_nowait(task)

    def is_complete(self) -> bool:
        """Check if all tasks are done."""
        for status in self.status.values():
            if status in (TaskStatus.PENDING, TaskStatus.BLOCKED, TaskStatus.IN_PROGRESS):
                return False
        return True

    def get_outputs(self) -> dict[str, Any]:
        """Get the final output values."""
        return {
            output_id: self.results[output_id].value
            for output_id in self.graph.output_ids
            if output_id in self.results
        }
```

## Scheduler

The scheduler manages task dispatch with resource awareness:

```python
class Scheduler:
    """
    Manages task scheduling with priority and resource awareness.
    """

    def __init__(
        self,
        resource_manager: ResourceManager,
        max_concurrent: int = 100,
    ):
        self.resource_manager = resource_manager
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0

    async def execute(
        self,
        state: ExecutionState,
        on_complete: Callable[[str, TaskResult], None] | None = None,
        on_error: Callable[[str, Exception], None] | None = None,
    ) -> dict[str, Any]:
        """
        Execute all tasks in the graph.
        """
        async with asyncio.TaskGroup() as tg:
            while not state.is_complete():
                # Wait for a slot
                await self._semaphore.acquire()

                # Get next task
                task = await state.get_next_task()
                if task is None:
                    self._semaphore.release()
                    # Wait for in-progress tasks to complete
                    await asyncio.sleep(0.01)
                    continue

                # Spawn task execution
                tg.create_task(
                    self._execute_task(state, task, on_complete, on_error)
                )

        return state.get_outputs()

    async def _execute_task(
        self,
        state: ExecutionState,
        task: Task,
        on_complete: Callable[[str, TaskResult], None] | None,
        on_error: Callable[[str, Exception], None] | None,
    ) -> None:
        """Execute a single task with error handling."""
        start_time = time.time()

        try:
            # Short-circuit if any dependency is a Value(ERROR)
            # (functional ops propagate errors as values, not exceptions)
            # if has_error_value(task.args, task.kwargs):
            #     result = first_error_value(task.args, task.kwargs)
            #     ...

            # Get resource alias (if LLMInference)
            alias = getattr(task.module, "alias", None)

            if alias:
                # Execute via resource manager
                result = await self.resource_manager.execute(
                    alias=alias,
                    module=task.module,
                    args=task.args,
                    kwargs=task.kwargs,
                )
            else:
                # Direct execution (non-LLM modules)
                result = await self._direct_execute(task)

            # Optional: interpret Value(ERROR) with HTTP 429 as backpressure
            # if isinstance(result, Value) and result.kind == ValueKind.ERROR:
            #     if result.meta.get("http_status") == 429:
            #         raise RateLimitError(retry_after=result.meta.get("retry_after"))

            # Create result
            duration_ms = (time.time() - start_time) * 1000
            task_result = TaskResult(
                node_id=task.node_id,
                value=result,
                duration_ms=duration_ms,
                retry_count=task.retry_count,
            )

            # Mark complete
            newly_ready = state.mark_complete(task.node_id, task_result)

            if on_complete:
                on_complete(task.node_id, task_result)

        except RateLimitError as e:
            # Backpressure - requeue
            self.resource_manager.handle_rate_limit(
                alias=getattr(task.module, "alias", None),
                retry_after=e.retry_after,
            )
            state.requeue(task.node_id)

        except Exception as e:
            # Task failed
            state.mark_failed(task.node_id, e)
            if on_error:
                on_error(task.node_id, e)

        finally:
            self._semaphore.release()

    async def _direct_execute(self, task: Task) -> Any:
        """Execute a non-LLM module directly."""
        # Unwrap Value payloads for user-defined forward() implementations
        args = unwrap(task.args)
        kwargs = unwrap(task.kwargs)
        if asyncio.iscoroutinefunction(task.module.forward):
            return await task.module.forward(*args, **kwargs)
        else:
            return task.module.forward(*args, **kwargs)
```

## Adaptive Rate Limiting

Handle backpressure from LLM endpoints:

```python
class RateLimiter:
    """
    Token bucket rate limiter with adaptive backoff.
    """

    def __init__(
        self,
        initial_rate: float = 10.0,    # Requests per second
        max_tokens: float = 10.0,       # Burst capacity
        min_rate: float = 0.1,          # Minimum rate after backoff
        recovery_factor: float = 1.1,   # Rate multiplier on success
        backoff_factor: float = 0.5,    # Rate multiplier on failure
    ):
        self.rate = initial_rate
        self.max_rate = initial_rate
        self.min_rate = min_rate
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.recovery_factor = recovery_factor
        self.backoff_factor = backoff_factor
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available."""
        async with self._lock:
            await self._refill()

            while self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                await self._refill()

            self.tokens -= 1

    async def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        self.tokens = min(
            self.max_tokens,
            self.tokens + elapsed * self.rate
        )

    def backoff(self, retry_after: float | None = None) -> None:
        """Reduce rate after hitting backpressure."""
        if retry_after:
            # Use server-provided retry time to estimate rate
            self.rate = min(self.rate, 1.0 / retry_after)
        else:
            self.rate = max(self.min_rate, self.rate * self.backoff_factor)

    def recover(self) -> None:
        """Gradually increase rate after successful requests."""
        self.rate = min(self.max_rate, self.rate * self.recovery_factor)
```

## Execution Manager

Manage multiple concurrent graph executions with memory limits:

```python
class ExecutionManager:
    """
    Manages multiple concurrent graph executions.

    Enforces memory limits by queuing executions when at capacity.
    """

    def __init__(
        self,
        resource_manager: ResourceManager,
        max_inflight_graphs: int = 10,
        max_inflight_tasks: int = 100,
        checkpoint_manager: CheckpointManager | None = None,
    ):
        self.resource_manager = resource_manager
        self.max_inflight_graphs = max_inflight_graphs
        self.max_inflight_tasks = max_inflight_tasks
        self.checkpoint_manager = checkpoint_manager

        self._active: dict[str, ExecutionState] = {}
        self._pending: asyncio.Queue[tuple[str, InferenceGraph, Any, asyncio.Future]] = asyncio.Queue()
        self._graph_semaphore = asyncio.Semaphore(max_inflight_graphs)
        self._scheduler = Scheduler(resource_manager, max_inflight_tasks)

    async def submit(
        self,
        graph: InferenceGraph,
        inputs: dict[str, Any],
    ) -> Any:
        """
        Submit a graph for execution.

        Returns immediately with a future if at capacity.
        """
        execution_id = str(uuid.uuid4())
        future: asyncio.Future[Any] = asyncio.Future()

        # Try to acquire slot
        acquired = self._graph_semaphore.locked()
        if acquired:
            # Queue for later
            await self._pending.put((execution_id, graph, inputs, future))
        else:
            await self._graph_semaphore.acquire()
            asyncio.create_task(
                self._execute(execution_id, graph, inputs, future)
            )

        return await future

    async def _execute(
        self,
        execution_id: str,
        graph: InferenceGraph,
        inputs: dict[str, Any],
        future: asyncio.Future,
    ) -> None:
        """Execute a graph and resolve its future.

        Note: Input values are provided at execution time and resolved via
        ValueRef placeholders created during tracing.
        """
        try:
            # Create execution state
            state = ExecutionState(graph)
            self._active[execution_id] = state

            # Execute with checkpointing
            def on_complete(node_id: str, result: TaskResult):
                if self.checkpoint_manager:
                    self.checkpoint_manager.record_completion(
                        execution_id, node_id, result
                    )

            outputs = await self._scheduler.execute(
                state,
                on_complete=on_complete,
            )

            future.set_result(outputs)

        except Exception as e:
            future.set_exception(e)

        finally:
            self._active.pop(execution_id, None)
            self._graph_semaphore.release()

            # Process pending queue
            await self._process_pending()

    async def _process_pending(self) -> None:
        """Process the next pending execution if capacity available."""
        if self._pending.empty():
            return

        await self._graph_semaphore.acquire()
        execution_id, graph, inputs, future = await self._pending.get()
        asyncio.create_task(
            self._execute(execution_id, graph, inputs, future)
        )
```

## Checkpointing

Persist progress for recovery and analysis:

```python
@dataclass
class Checkpoint:
    """A saved execution checkpoint."""

    execution_id: str
    timestamp: float
    completed_nodes: dict[str, TaskResult]
    failed_nodes: dict[str, str]  # node_id -> error message
    pending_nodes: list[str]

    def save(self, path: Path) -> None:
        """Save checkpoint to disk."""
        data = {
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "completed_nodes": {
                node_id: {
                    "value": result.value,
                    "duration_ms": result.duration_ms,
                    "retry_count": result.retry_count,
                }
                for node_id, result in self.completed_nodes.items()
            },
            "failed_nodes": self.failed_nodes,
            "pending_nodes": self.pending_nodes,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """Load checkpoint from disk."""
        data = json.loads(path.read_text())
        return cls(
            execution_id=data["execution_id"],
            timestamp=data["timestamp"],
            completed_nodes={
                node_id: TaskResult(
                    node_id=node_id,
                    value=result["value"],
                    duration_ms=result["duration_ms"],
                    retry_count=result["retry_count"],
                )
                for node_id, result in data["completed_nodes"].items()
            },
            failed_nodes=data["failed_nodes"],
            pending_nodes=data["pending_nodes"],
        )


class CheckpointManager:
    """
    Manages checkpointing for executions.

    Writes checkpoints periodically based on buffer size or time.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        buffer_size: int = 10,
        flush_interval: float = 60.0,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self._buffers: dict[str, list[tuple[str, TaskResult]]] = defaultdict(list)
        self._last_flush: dict[str, float] = {}
        self._lock = asyncio.Lock()

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def record_completion(
        self,
        execution_id: str,
        node_id: str,
        result: TaskResult,
    ) -> None:
        """Record a completed task."""
        self._buffers[execution_id].append((node_id, result))

        # Check if flush needed
        if len(self._buffers[execution_id]) >= self.buffer_size:
            asyncio.create_task(self.flush(execution_id))

    async def flush(self, execution_id: str) -> None:
        """Flush buffer to disk."""
        async with self._lock:
            buffer = self._buffers.pop(execution_id, [])
            if not buffer:
                return

            # Load existing checkpoint or create new
            checkpoint_path = self.checkpoint_dir / f"{execution_id}.json"
            if checkpoint_path.exists():
                checkpoint = Checkpoint.load(checkpoint_path)
            else:
                checkpoint = Checkpoint(
                    execution_id=execution_id,
                    timestamp=time.time(),
                    completed_nodes={},
                    failed_nodes={},
                    pending_nodes=[],
                )

            # Update with new completions
            for node_id, result in buffer:
                checkpoint.completed_nodes[node_id] = result

            checkpoint.timestamp = time.time()
            checkpoint.save(checkpoint_path)

            self._last_flush[execution_id] = time.time()

    async def flush_all(self) -> None:
        """Flush all buffers."""
        for execution_id in list(self._buffers.keys()):
            await self.flush(execution_id)
```

## Module Execution

plait provides two execution APIs: bound execution (recommended) and explicit `run()`.

### Bound Execution (Recommended)

The simplest way to execute modules is to bind resources and call directly:

```python
from plait import ResourceConfig

# Configure resources
resources = ResourceConfig({
    "fast": {"model": "gpt-4o-mini", "max_concurrent": 20},
    "smart": {"model": "gpt-4o", "max_concurrent": 5},
})

# Bind resources to the module
pipeline = AnalysisPipeline().bind(resources=resources)

# Call directly - traces and executes under the hood
result = await pipeline("Long document text...")

# Batch execution - process multiple documents
results = await pipeline([
    "Document 1...",
    "Document 2...",
    "Document 3...",
])
```

This approach:
- Mirrors PyTorch's intuitive `model(x) → y` pattern
- Configures resources once, uses them for all calls
- Handles batching transparently

### ExecutionSettings Context Manager

For advanced scenarios (checkpointing, custom schedulers, shared settings across multiple modules), use the `ExecutionSettings` context manager:

```python
from plait import ExecutionSettings, ResourceConfig

resources = ResourceConfig({...})

# All executions within this context share the same settings
with ExecutionSettings(
    resources=resources,
    checkpoint_dir="/checkpoints/run_001",
    max_concurrent=50,
):
    # Multiple pipelines execute with shared checkpointing
    results_1 = await pipeline_1(large_batch)
    results_2 = await pipeline_2(results_1)
    results_3 = await pipeline_3(other_data)
    # All progress is checkpointed to the same directory
```

This approach:
- Provides shared execution settings for multiple module calls
- Enables checkpointing across an entire workflow
- Allows custom scheduler configuration
- Settings can be nested (inner context overrides outer)

#### ExecutionSettings Class

```python
@dataclass
class ExecutionSettings:
    """Context manager for shared execution configuration.

    Provides default settings for all module executions within the context.
    Bound module settings take precedence over context settings.
    """

    resources: ResourceConfig | ResourceManager | None = None
    checkpoint_dir: Path | str | None = None
    max_concurrent: int = 100
    task_timeout: float | None = None
    max_task_retries: int = 0
    task_retry_delay: float = 1.0
    scheduler: Scheduler | None = None
    on_task_complete: Callable[[str, TaskResult], None] | None = None
    on_task_failed: Callable[[str, Exception], None] | None = None

    # Profiling configuration (see profiling.md for details)
    profile: bool = False
    profile_path: Path | str | None = None
    profile_counters: bool = True
    profile_include_args: bool = True

    def __enter__(self) -> Self:
        """Activate this settings context."""
        self._token = _execution_settings.set(self)
        self._checkpoint_manager = None
        if self.checkpoint_dir:
            self._checkpoint_manager = CheckpointManager(Path(self.checkpoint_dir))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Deactivate this settings context and flush checkpoints."""
        if self._checkpoint_manager:
            # Flush is async, so we schedule it
            asyncio.get_event_loop().run_until_complete(
                self._checkpoint_manager.flush_all()
            )
        _execution_settings.reset(self._token)
        return None

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        self._token = _execution_settings.set(self)
        self._checkpoint_manager = None
        if self.checkpoint_dir:
            self._checkpoint_manager = CheckpointManager(Path(self.checkpoint_dir))

        # Initialize profiler if enabled
        self.profiler = None
        if self.profile:
            self.profiler = TraceProfiler(
                include_counters=self.profile_counters,
                include_args=self.profile_include_args,
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with checkpoint flush and trace export."""
        if self._checkpoint_manager:
            await self._checkpoint_manager.flush_all()

        # Export trace file on exit
        if self.profiler:
            path = self.profile_path
            if path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"./traces/trace_{timestamp}.json"
            self.profiler.export(path)

        _execution_settings.reset(self._token)
        return None
```

#### Context Variable for Settings

```python
from contextvars import ContextVar

_execution_settings: ContextVar[ExecutionSettings | None] = ContextVar(
    "execution_settings", default=None
)

def get_execution_settings() -> ExecutionSettings | None:
    """Get the current execution settings from context."""
    return _execution_settings.get()
```

#### Priority Order

When executing a bound module, settings are resolved in this order (highest priority first):

1. **Call-time kwargs**: `await pipeline(input, max_concurrent=10)`
2. **Bound settings**: `pipeline.bind(max_concurrent=50)`
3. **Context settings**: `with ExecutionSettings(max_concurrent=100):`
4. **Defaults**: Built-in default values

Example:

```python
pipeline = MyPipeline().bind(resources=config, max_concurrent=50)

with ExecutionSettings(checkpoint_dir="/ckpt", max_concurrent=100):
    # Uses max_concurrent=50 (from bind), checkpoint_dir="/ckpt" (from context)
    result = await pipeline(input)

    # Call-time override: uses max_concurrent=10
    result = await pipeline(input, max_concurrent=10)
```

#### Nested Contexts

Contexts can be nested, with inner contexts overriding outer ones:

```python
with ExecutionSettings(checkpoint_dir="/outer", max_concurrent=100):
    result1 = await pipeline(input)  # Uses /outer, max=100

    with ExecutionSettings(max_concurrent=10):
        # Uses /outer (inherited), max=10 (overridden)
        result2 = await pipeline(input)

    result3 = await pipeline(input)  # Back to /outer, max=100
```

### The `run()` Function

For advanced control (custom per-call options, state inspection), use `run()` directly:

```python
async def run(
    module: Module,
    *args: Any,
    resources: ResourceConfig | ResourceManager,
    max_concurrent: int = 100,
    checkpoint_dir: Path | None = None,
    **kwargs: Any,
) -> Any:
    """
    Trace and execute an inference module.

    Args:
        module: The inference module to execute
        *args: Positional arguments to pass to forward()
        resources: Resource configuration or manager
        max_concurrent: Maximum concurrent tasks
        checkpoint_dir: Optional directory for checkpoints
        **kwargs: Keyword arguments to pass to forward()

    Returns:
        The output of the module's forward() method, with `Value` wrappers
        unwrapped for user-facing APIs. If module.training is True, returns
        TracedOutput with record attached.
    """
    # Create resource manager if needed
    if isinstance(resources, ResourceConfig):
        resource_manager = ResourceManager(resources)
    else:
        resource_manager = resources

    # Trace the module (inputs are bound at execution via valueify + ValueRef)
    tracer = Tracer()
    graph = tracer.trace(module, *args, **kwargs)

    # Create checkpoint manager if requested
    checkpoint_manager = None
    if checkpoint_dir:
        checkpoint_manager = CheckpointManager(checkpoint_dir)

    # Create scheduler and execute
    scheduler = Scheduler(resource_manager, max_concurrent)
    state = ExecutionState(graph)

    def on_complete(node_id: str, result: TaskResult):
        if checkpoint_manager:
            checkpoint_manager.record_completion("main", node_id, result)

    outputs = await scheduler.execute(state, on_complete=on_complete)

    # Flush any remaining checkpoints
    if checkpoint_manager:
        await checkpoint_manager.flush_all()

    # Return outputs (unwrap if single output)
    if len(outputs) == 1:
        return list(outputs.values())[0]
    return outputs
```

## Example: Complete Execution Flow

```python
# Define a pipeline
class AnalysisPipeline(Module):
    def __init__(self):
        super().__init__()
        self.extract = LLMInference(alias="fast")
        self.analyze = LLMInference(alias="smart")
        self.summarize = LLMInference(alias="fast")

    def forward(self, doc: str) -> str:
        entities = self.extract(doc)      # Task 1
        analysis = self.analyze(entities)  # Task 2 (waits for 1)
        summary = self.summarize(analysis) # Task 3 (waits for 2)
        return summary

# Configure resources
resources = ResourceConfig({
    "fast": {"model": "gpt-4o-mini", "max_concurrent": 20},
    "smart": {"model": "gpt-4o", "max_concurrent": 5},
})

# ─────────────────────────────────────────────────────────────
# Option 1: Bound Execution (Recommended)
# ─────────────────────────────────────────────────────────────

# Bind resources once
pipeline = AnalysisPipeline().bind(resources=resources)

# Call directly like a function
result = await pipeline("Long document text...")
print(result)

# Process multiple documents
documents = ["Doc 1...", "Doc 2...", "Doc 3..."]
results = await pipeline(documents)

# ─────────────────────────────────────────────────────────────
# Option 2: Explicit run() for Advanced Control
# ─────────────────────────────────────────────────────────────

# Use run() when you need per-call configuration
result = await run(
    AnalysisPipeline(),
    "Long document text...",
    resources=resources,
    checkpoint_dir=Path("./checkpoints"),
)

print(result)
```

## Error Handling

plait handles errors at two distinct levels:

### Error Handling Levels

| Level | Scope | What fails | How errors surface |
|-------|-------|------------|-------------------|
| **Intra-graph** | Single input through a DAG | One node in the graph | Dependent nodes cancelled, error propagates |
| **Inter-batch** | Multiple inputs in a batch | One input's entire graph | Other inputs continue, `BatchResult.error` set |

**Intra-graph errors**: When a node fails within a graph execution, all dependent nodes are automatically cancelled (via `mark_failed()`). Independent nodes are not affected. This is the only sensible behavior—downstream nodes can't execute without their inputs.

**Value(ERROR) outputs**: Functional ops and selectors return `Value(ERROR)` instead
of raising. These are treated as terminal values: downstream nodes will receive
the error `Value` and are expected to short-circuit per functional API rules.
Only exceptions (RateLimitError, TimeoutError, etc.) change scheduler state.

**Inter-batch errors**: When processing multiple inputs (e.g., `await pipeline([doc1, doc2, doc3])`), each input runs its own graph independently. If one input fails, others continue. In streaming mode, failures are reported via `BatchResult.error`. In non-streaming mode, the entire batch fails if any input fails (use streaming for partial failure tolerance).

### Task Timeout

Individual tasks (LLM calls) can hang indefinitely. The `task_timeout` setting ensures tasks fail after a maximum duration:

```python
async with ExecutionSettings(resources=config, task_timeout=60.0):
    # Each LLM call times out after 60 seconds
    result = await pipeline(document)
```

When a task times out:
1. The task is cancelled via `asyncio.timeout()`
2. A `TimeoutError` is recorded for that node
3. Dependent nodes are cancelled (standard intra-graph failure handling)

### Task Retry

Transient failures (network errors, temporary API issues) can be retried automatically. This is distinct from rate-limit handling (which uses `requeue()` with backoff).

```python
async with ExecutionSettings(
    resources=config,
    max_task_retries=3,      # Retry up to 3 times
    task_retry_delay=1.0,    # Wait 1 second between retries
):
    result = await pipeline(document)
```

Retry behavior:
- Only retries on `TransientError` (connection errors, 5xx responses)
- Does not retry on permanent errors (4xx responses, validation errors)
- Exponential backoff: delay doubles each retry (1s, 2s, 4s)
- After max retries exhausted, task fails normally

`TransientError` is raised by LLM clients for retryable failures:

```python
class TransientError(InfEngineError):
    """Error for transient failures that may succeed on retry.

    Raised for connection errors, server errors (5xx), and other
    temporary failures. The scheduler will retry these if max_task_retries > 0.
    """
    pass
```

### Scheduler Error Handling

The scheduler handles errors in `_execute_task()`:

```python
async def _execute_task(self, state: ExecutionState, task: Task, ...) -> None:
    try:
        async with asyncio.timeout(self.task_timeout):
            result = await self._run_task(task)

        # Success
        state.mark_complete(task.node_id, result)

    except TimeoutError:
        # Task timed out
        state.mark_failed(task.node_id, TimeoutError(f"Task timed out after {self.task_timeout}s"))

    except RateLimitError as e:
        # Backpressure - requeue (existing behavior)
        self.resource_manager.handle_rate_limit(...)
        state.requeue(task.node_id)

    except TransientError as e:
        # Retryable error
        if task.retry_count < self.max_task_retries:
            await asyncio.sleep(self.task_retry_delay * (2 ** task.retry_count))
            state.requeue(task.node_id)
        else:
            state.mark_failed(task.node_id, e)

    except Exception as e:
        # Permanent failure
        state.mark_failed(task.node_id, e)
```

## Execution Patterns

plait provides multiple execution patterns optimized for different use cases: synchronous scripts, async applications, and streaming servers.

### Pattern Overview

| Pattern | Syntax | Returns | Use Case |
|---------|--------|---------|----------|
| Async single | `await module("x")` | `T` | Standard async code |
| Async batch | `await module([...])` | `list[T]` | Process multiple inputs |
| Sync single | `module.run_sync("x")` | `T` | Scripts, notebooks |
| Sync batch | `module.run_sync([...])` | `list[T]` | Batch scripts |
| Streaming | `async for r in module([...])` | `BatchResult` | Servers, progress |

### Synchronous Execution

For scripts, notebooks, and contexts where async isn't needed, use `run_sync()`:

```python
# Bind resources to module
pipeline = AnalysisPipeline().bind(resources=config)

# Single input - blocks and returns result
result = pipeline.run_sync("Hello, world!")

# Batch input - blocks until all complete, returns list
results = pipeline.run_sync(["doc1", "doc2", "doc3"])
```

`run_sync()` also works with `ExecutionSettings` context (without binding):

```python
with ExecutionSettings(resources=config):
    result = pipeline.run_sync("Hello")
```

**Note**: `run_sync()` cannot be called from within an async context (it would block the event loop). Use `await` in async code.

### Async Execution

Standard async execution returns results when all processing completes:

```python
async with ExecutionSettings(resources=config):
    # Single input
    result = await pipeline("Hello")

    # Batch input - runs concurrently, returns list when all done
    results = await pipeline(["doc1", "doc2", "doc3"])
```

Batch execution runs all inputs concurrently (up to `max_concurrent`), not sequentially.

### Batch Execution for Training

For training workflows, enable training mode to capture `ForwardRecord` via `TracedOutput`:

```python
# Enable training mode - outputs carry records implicitly
pipeline.train()

# Single input - returns TracedOutput
output = await pipeline(input)  # TracedOutput[str]
output.value                     # The actual string
output._record                   # ForwardRecord for backward()

# Batch inputs - returns list[TracedOutput]
outputs = await pipeline(batch_inputs)  # list[TracedOutput]

# Disable training mode for inference
pipeline.eval()
output = await pipeline(input)  # str (raw value, no record)
```

Example training loop:

```python
async with ExecutionSettings(resources=config):
    pipeline.train()

    # Batch forward (returns TracedOutput with records)
    outputs = await pipeline(batch_inputs)

    # Batch loss (extracts records from TracedOutput automatically)
    feedbacks = await loss_fn.batch(outputs, targets=targets)

    # Batch backward (concurrent)
    await Feedback.backward_batch(feedbacks, optimizer=optimizer)

    # Optimizer step
    await optimizer.step()

    pipeline.eval()
```

See `optimization.md` → "Batch Training API" for complete details.

### Streaming Execution

For servers and progress tracking, streaming yields results as they complete:

```python
async with ExecutionSettings(resources=config, streaming=True):
    async for result in pipeline(["doc1", "doc2", "doc3"]):
        if result.ok:
            await send_to_client(result.output)
        else:
            logger.error(f"Input {result.index} failed: {result.error}")
```

Streaming requires `streaming=True` in the `ExecutionSettings` context. Single-input calls still return raw results (not wrapped in `BatchResult`).

#### BatchResult

When streaming batch inputs, each result is wrapped in `BatchResult`:

```python
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class BatchResult(Generic[T]):
    """Result wrapper for streaming batch execution.

    Provides full context about each result including the original
    input, output (if successful), and error (if failed).

    Attributes:
        index: Position in the original input list (0-based).
        input: The original input value that produced this result.
        output: The result value if successful, None if failed.
        error: The exception if failed, None if successful.
    """

    index: int
    input: Any
    output: T | None
    error: Exception | None

    @property
    def ok(self) -> bool:
        """Check if this result is successful."""
        return self.error is None
```

Example handling mixed results:

```python
async with ExecutionSettings(resources=config, streaming=True):
    succeeded = []
    failed = []

    async for r in pipeline(documents):
        if r.ok:
            succeeded.append((r.index, r.output))
        else:
            failed.append((r.index, r.input, r.error))

    print(f"Completed: {len(succeeded)} succeeded, {len(failed)} failed")
```

#### Result Ordering

By default, streaming yields results as they complete (fastest first). This maximizes throughput but means results may arrive out of order.

To preserve input order (yielding in sequence, potentially waiting on slow items):

```python
async with ExecutionSettings(
    resources=config,
    streaming=True,
    preserve_order=True,  # Yield in input order
):
    async for r in pipeline(batch):
        # Results arrive in same order as inputs
        process_in_order(r)
```

When `preserve_order=False` (default), use `result.index` to correlate with inputs.

### Progress Tracking

For long-running batches, track progress with the `on_progress` callback:

```python
def show_progress(done: int, total: int) -> None:
    percent = (done / total) * 100
    print(f"Progress: {done}/{total} ({percent:.1f}%)")


async with ExecutionSettings(
    resources=config,
    on_progress=show_progress,
):
    # Progress callback fires as each input completes
    results = await pipeline(large_batch)
```

The callback receives `(completed_count, total_count)` after each input finishes.

### Cancellation

When streaming, breaking out of the loop cancels all pending work:

```python
async with ExecutionSettings(resources=config, streaming=True):
    async for result in pipeline(huge_batch):
        if result.ok:
            yield result.output

        if should_stop():
            break  # All pending tasks are cancelled immediately
```

In-flight API calls are cancelled via `asyncio.Task.cancel()`. This ensures resources are freed promptly when early termination is needed.

### Updated ExecutionSettings

The complete `ExecutionSettings` with all execution pattern options:

```python
@dataclass
class ExecutionSettings:
    """Context manager for shared execution configuration.

    Controls execution behavior for all module calls within the context.
    """

    # ─────────────────────────────────────────────────────────────
    # Resources
    # ─────────────────────────────────────────────────────────────
    resources: ResourceConfig | ResourceManager | None = None

    # ─────────────────────────────────────────────────────────────
    # Execution Mode
    # ─────────────────────────────────────────────────────────────
    streaming: bool = False
    """Enable streaming mode for batch inputs.

    When True, batch calls (module([list])) return an async iterator
    that yields BatchResult objects as they complete. When False,
    batch calls return a list of all results.
    """

    preserve_order: bool = False
    """Yield streaming results in input order.

    When True, results are yielded in the same order as inputs,
    potentially waiting on slower items. When False (default),
    results yield as soon as they complete (fastest throughput).
    Only applies when streaming=True.
    """

    # ─────────────────────────────────────────────────────────────
    # Concurrency and Timeouts
    # ─────────────────────────────────────────────────────────────
    max_concurrent: int = 100
    """Maximum number of concurrent tasks across all batches."""

    task_timeout: float | None = None
    """Maximum seconds for a single task (LLM call) before timeout.

    When set, tasks exceeding this duration raise TimeoutError and
    dependent nodes are cancelled. None means no timeout (default).
    Recommended: 60-300 seconds depending on model and prompt length.
    """

    # ─────────────────────────────────────────────────────────────
    # Retry Behavior
    # ─────────────────────────────────────────────────────────────
    max_task_retries: int = 0
    """Maximum retry attempts for transient failures.

    Retries apply to connection errors and 5xx responses, not to
    permanent errors (4xx) or rate limits (handled separately).
    Default 0 means no retries.
    """

    task_retry_delay: float = 1.0
    """Base delay in seconds between retry attempts.

    Uses exponential backoff: delay doubles each retry.
    E.g., with delay=1.0: retries at 1s, 2s, 4s, 8s...
    """

    # ─────────────────────────────────────────────────────────────
    # Progress and Callbacks
    # ─────────────────────────────────────────────────────────────
    on_progress: Callable[[int, int], None] | None = None
    """Callback for batch progress updates.

    Called with (completed_count, total_count) after each input
    in a batch completes. Useful for progress bars and logging.
    """

    on_task_complete: Callable[[str, TaskResult], None] | None = None
    """Low-level callback for individual graph node completions."""

    on_task_failed: Callable[[str, Exception], None] | None = None
    """Low-level callback for individual graph node failures."""

    # ─────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────
    checkpoint_dir: Path | str | None = None
    """Directory for saving execution checkpoints."""

    # ─────────────────────────────────────────────────────────────
    # Profiling (see profiling.md)
    # ─────────────────────────────────────────────────────────────
    profile: bool = False
    profile_path: Path | str | None = None
    profile_counters: bool = True
    profile_include_args: bool = True

    # ─────────────────────────────────────────────────────────────
    # Advanced
    # ─────────────────────────────────────────────────────────────
    scheduler: Scheduler | None = None
    """Custom scheduler instance for advanced use cases."""
```

### Example: Complete Server Pattern

A typical server using streaming for real-time results:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
pipeline = AnalysisPipeline().bind(resources=config)


@app.post("/analyze")
async def analyze(documents: list[str]):
    """Stream analysis results as they complete."""

    async def generate():
        async with ExecutionSettings(streaming=True):
            async for result in pipeline(documents):
                if result.ok:
                    yield json.dumps({
                        "index": result.index,
                        "status": "success",
                        "output": result.output,
                    }) + "\n"
                else:
                    yield json.dumps({
                        "index": result.index,
                        "status": "error",
                        "error": str(result.error),
                    }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
```

### Example: Script with Progress

A batch processing script with progress tracking:

```python
from tqdm import tqdm

# Load documents
documents = load_documents("./corpus/")

# Set up progress bar
pbar = tqdm(total=len(documents), desc="Processing")


def update_progress(done: int, total: int) -> None:
    pbar.n = done
    pbar.refresh()


# Run synchronously with progress
with ExecutionSettings(
    resources=config,
    on_progress=update_progress,
):
    results = pipeline.run_sync(documents)

pbar.close()
print(f"Processed {len(results)} documents")
```
