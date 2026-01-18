# plait Architecture

## Overview

plait is a PyTorch-inspired framework for building, executing, and optimizing complex LLM inference pipelines. It provides:

- **Familiar API**: PyTorch-like `Module` with `forward()` and `backward()` methods
- **Automatic DAG Capture**: Trace-based graph construction from eager-mode code
- **Maximum Throughput**: Async execution with adaptive backpressure and resource pooling
- **LLM-Based Optimization**: Backward passes that propagate feedback through the graph

## Design Principles

1. **Separation of Concerns**: Module definitions are independent of resource configuration
2. **Composability**: Modules nest arbitrarily; the framework flattens them for execution
3. **Familiarity**: Mirror PyTorch conventions wherever possible
4. **Async Under the Hood**: Users write synchronous-looking code; execution is fully async
5. **Strong Typing**: Leverage Python's type system for correctness and IDE support
6. **Observability**: Built-in profiling hooks compatible with standard tools

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Code                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Module │  │ LLMInference    │  │ Parameter           │  │
│  │ (forward/back)  │  │ (atomic ops)    │  │ (learnable values)  │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │
└───────────┼─────────────────────┼─────────────────────┼─────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Tracing Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Tracer          │  │ Value           │  │ InferenceGraph      │  │
│  │ (DAG capture)   │  │ (data + refs)   │  │ (nodes + edges)     │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │
└───────────┼─────────────────────┼─────────────────────┼─────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Execution Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Scheduler       │  │ ExecutionState  │  │ ResourceManager     │  │
│  │ (priority queue)│  │ (task tracking) │  │ (endpoint pools)    │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │
└───────────┼─────────────────────┼─────────────────────┼─────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Infrastructure Layer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ LLM Clients     │  │ Persistence     │  │ Profiler            │  │
│  │ (OpenAI, etc)   │  │ (checkpoints)   │  │ (tracing hooks)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Module

The fundamental building block, analogous to `torch.nn.Module`:

```python
class Module:
    """Base class for all inference operations."""

    def __init__(self):
        self._children: dict[str, Module] = {}
        self._parameters: dict[str, Parameter] = {}

    def forward(self, *args, **kwargs) -> Any:
        """Define the inference computation."""
        raise NotImplementedError

    def backward(self, feedback: Value, ctx: BackwardContext) -> BackwardResult:
        """Propagate feedback to inputs and parameters."""
        raise NotImplementedError
```

Key properties:
- **Child Discovery**: Modules assigned as attributes are automatically registered
- **Parameter Tracking**: `Parameter` objects are collected for optimization
- **Dual Execution Modes**: Direct execution or traced for graph capture

### 2. LLMInference (Atomic Module)

The base operation for LLM calls:

```python
class LLMInference(Module):
    """Atomic module for LLM API calls."""

    def __init__(
        self,
        alias: str,                          # Resource binding key
        system_prompt: str | Parameter = "", # Can be optimizable
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ):
        super().__init__()
        self.alias = alias
        self.system_prompt = self._wrap_parameter(system_prompt)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def forward(self, prompt: str) -> str:
        # Actual execution handled by runtime
        ...
```

The `alias` decouples the module from specific endpoints, allowing the same module to run against different models/endpoints based on configuration.

### 3. Parameter

Learnable values that can be updated during optimization:

```python
class Parameter:
    """A value that can be optimized via backward passes."""

    def __init__(
        self,
        value: str,
        description: str | None = None,
        requires_grad: bool = True,
    ):
        self.value = value
        self.description = description  # Required when requires_grad=True
        self.requires_grad = requires_grad
        self._feedback_buffer: list[str] = []

    def accumulate_feedback(self, feedback: str) -> None:
        """Collect feedback from backward pass."""
        self._feedback_buffer.append(feedback)

    def apply_update(self, new_value: str) -> None:
        """Apply optimizer-computed update."""
        self.value = new_value
        self._feedback_buffer.clear()
```

When `requires_grad=True`, `description` should explain the parameter’s role so
optimizers can generate coherent updates.

### 4. Value (Data Container)

`Value` is the uniform container for data flowing through the system. It carries
payloads (text, structured responses, errors) and an optional `ref` pointing to
the producing node. This replaces ad-hoc dependency encoding in args/kwargs and
enables safe nested structures. See `values.md` for details.

### 5. InferenceGraph

The traced execution graph:

```python
@dataclass
class GraphNode:
    id: str
    module: Module
    dependencies: list[str]      # Input node IDs
    dependents: list[str]        # Output node IDs

class InferenceGraph:
    nodes: dict[str, GraphNode]
    input_ids: list[str]
    output_ids: list[str]

    def topological_order(self) -> list[str]:
        """Return nodes in valid execution order."""

    def descendants(self, node_id: str) -> set[str]:
        """Get all nodes that depend on this node."""
```

### 5. ExecutionState

Tracks the state of a graph execution:

```python
class ExecutionState:
    """Maintains state for a single graph execution."""

    graph: InferenceGraph
    pending: PriorityQueue[Task]           # Ready to execute
    in_progress: dict[str, Task]           # Currently executing
    completed: dict[str, Result]           # Finished with results
    failed: dict[str, Exception]           # Failed tasks

    def requeue(self, task_id: str) -> None:
        """Re-enqueue a task, dropping all its descendants."""
        descendants = self.graph.descendants(task_id)

        # Remove descendants from all states
        for desc_id in descendants:
            self.pending.remove(desc_id)
            self.in_progress.pop(desc_id, None)
            self.completed.pop(desc_id, None)

        # Re-add the task
        self.pending.put(self.graph.nodes[task_id])
```

### 6. ResourceManager

Separates resource configuration from module definitions:

```python
class ResourceManager:
    """Manages LLM endpoints and connection pools."""

    endpoints: dict[str, Endpoint]           # alias -> endpoint
    semaphores: dict[str, asyncio.Semaphore] # Concurrency limits
    rate_limiters: dict[str, RateLimiter]    # Per-endpoint rate control

    def __init__(self, config: ResourceConfig):
        for alias, endpoint_config in config.items():
            self.endpoints[alias] = self._create_endpoint(endpoint_config)
            self.semaphores[alias] = asyncio.Semaphore(endpoint_config.max_concurrent)
            self.rate_limiters[alias] = RateLimiter(endpoint_config.rate_limit)

    async def execute(self, alias: str, request: LLMRequest) -> LLMResponse:
        """Execute request with resource management."""
        async with self.semaphores[alias]:
            await self.rate_limiters[alias].acquire()
            return await self.endpoints[alias].call(request)
```

## Execution Flow

### Module Execution API

plait provides multiple execution patterns optimized for different use cases.

#### Execution Patterns Overview

| Pattern | Syntax | Returns | Use Case |
|---------|--------|---------|----------|
| Async single | `await module("x")` | `T` | Standard async code |
| Async batch | `await module([...])` | `list[T]` | Process multiple inputs |
| Sync single | `module.run_sync("x")` | `T` | Scripts, notebooks |
| Sync batch | `module.run_sync([...])` | `list[T]` | Batch scripts |
| Streaming | `async for r in module([...])` | `BatchResult` | Servers, progress |

See `execution.md` → "Execution Patterns" for complete details.

#### 1. Bound Execution (Recommended)

Bind resources to a module, then call it directly like a function:

```python
# Bind resources once
pipeline = MyPipeline().bind(resources=config)

# Async execution
result = await pipeline("input")

# Batch execution - runs concurrently, returns list
results = await pipeline(["input_a", "input_b", "input_c"])

# Sync execution for scripts (no await needed)
result = pipeline.run_sync("input")
results = pipeline.run_sync(["input_a", "input_b", "input_c"])
```

This is the preferred API because:
- Mirrors PyTorch's intuitive `model(x) → y` pattern
- Resources are configured once, used many times
- Batch execution runs concurrently (not sequentially)

#### 2. ExecutionSettings Context Manager

For shared settings across multiple module calls (checkpointing, streaming, custom schedulers):

```python
async with ExecutionSettings(
    resources=config,
    checkpoint_dir="/checkpoints/run_001",
    max_concurrent=50,
):
    # All modules share the same checkpointing and settings
    result_1 = await pipeline_1(large_batch)
    result_2 = await pipeline_2(result_1)
    result_3 = await other_pipeline(data)
```

For streaming results as they complete (useful for servers):

```python
async with ExecutionSettings(resources=config, streaming=True):
    async for result in pipeline(large_batch):
        if result.ok:
            await send_to_client(result.output)
        else:
            logger.error(f"Input {result.index} failed: {result.error}")
```

This is useful for:
- Checkpointing across an entire workflow
- Streaming results for server applications
- Progress tracking with `on_progress` callback
- Providing default settings for unbound modules

#### 3. Explicit run() Function

For per-call control (custom options per execution, state inspection):

```python
result = await run(
    module,
    "input",
    resources=config,
    max_concurrent=50,
    checkpoint_dir=Path("./checkpoints"),
)
```

### Forward Pass (Inference)

Both APIs follow the same internal flow:

```
1. User calls: result = await pipeline("input")  # or run(...)

2. Tracing Phase:
   - Create Tracer with input proxies
   - Execute module.forward() symbolically
   - Each module call records a GraphNode
   - Value.ref dependencies create edges
   - Result: InferenceGraph

3. Scheduling Phase:
   - Create ExecutionState from graph
   - Enqueue root nodes (no dependencies)
   - Create Scheduler with ResourceManager

4. Execution Phase:
   - While pending tasks exist:
     - Dequeue highest-priority ready task
     - Acquire resource (semaphore + rate limit)
     - Execute module.forward() with real values
     - On success: mark complete, enqueue dependents
     - On rate limit: requeue with backoff
     - On error: handle per error policy

5. Completion:
   - Gather outputs from output nodes
   - Return results to user
```

### Backward Pass (Optimization)

```
1. User provides feedback: optimizer.accumulate(output, loss)

2. Value Propagation:
   - Start from output nodes
   - For each node in reverse topological order:
     - Call module.backward(feedback, context)
     - Distribute feedback to input nodes
     - Collect parameter feedback

3. Aggregation (on optimizer.step()):
   - Optimizer LLM aggregates all parameter feedback
   - Produces meta-feedback per parameter
   - Parameters apply updates

4. State Update:
   - Parameters hold new values
   - Ready for next forward pass
```

## Memory and Persistence

### Inflight Limits

```python
class ExecutionManager:
    """Manages multiple concurrent graph executions."""

    max_inflight: int                         # Memory limit
    active: dict[str, ExecutionState]         # Currently running
    pending: Queue[tuple[InferenceGraph, Any]] # Waiting to start

    async def submit(self, graph: InferenceGraph, inputs: Any) -> Future[Any]:
        if len(self.active) >= self.max_inflight:
            # Queue for later execution
            future = Future()
            self.pending.put((graph, inputs, future))
            return future
        else:
            return await self._execute(graph, inputs)
```

### Checkpointing

```python
class CheckpointManager:
    """Periodically persists execution state."""

    buffer_size: int              # Completions before flush
    checkpoint_dir: Path
    buffer: list[CompletedTask]

    def record_completion(self, task: Task, result: Result) -> None:
        self.buffer.append(CompletedTask(task, result))
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        checkpoint = Checkpoint(
            timestamp=time.time(),
            completions=self.buffer,
        )
        checkpoint.save(self.checkpoint_dir / f"{checkpoint.timestamp}.json")
        self.buffer.clear()
```

## Integration Points

### Profiling

Uses standard Python profiling tools:

```python
class ProfiledExecutor:
    """Executor with profiling hooks."""

    def __init__(self, executor: Executor, profiler: Profiler):
        self.executor = executor
        self.profiler = profiler

    async def execute(self, task: Task) -> Result:
        with self.profiler.span(
            name=task.module.__class__.__name__,
            attributes={
                "task_id": task.id,
                "alias": getattr(task.module, "alias", None),
            }
        ):
            return await self.executor.execute(task)
```

Compatible with:
- OpenTelemetry
- Python's `cProfile`
- Custom profiling backends

### Testing

Makefile targets for different test levels:

```makefile
test:           # Unit tests
	uv run pytest tests/unit

test-integration:  # Integration tests (requires LLM endpoints)
	uv run pytest tests/integration

test-all:       # All tests
	uv run pytest
```

## File Structure

```
plait/
├── src/
│   └── plait/
│       ├── __init__.py
│       ├── module.py           # Module, LLMInference
│       ├── parameter.py        # Parameter
│       ├── graph.py            # InferenceGraph, GraphNode
│       ├── tracing/
│       │   ├── __init__.py
│       │   ├── tracer.py       # Tracer
│       │   ├── values.py       # Value, ValueKind, helpers
│       │   └── context.py      # TraceContext
│       ├── execution/
│       │   ├── __init__.py
│       │   ├── scheduler.py    # Scheduler, PriorityQueue
│       │   ├── state.py        # ExecutionState
│       │   ├── executor.py     # Executor
│       │   └── checkpoint.py   # CheckpointManager
│       ├── resources/
│       │   ├── __init__.py
│       │   ├── manager.py      # ResourceManager
│       │   ├── config.py       # ResourceConfig
│       │   ├── endpoint.py     # Endpoint base class
│       │   └── rate_limit.py   # RateLimiter
│       ├── optimization/
│       │   ├── __init__.py
│       │   ├── loss.py         # Loss functions
│       │   ├── optimizer.py    # Optimizer
│       │   └── backward.py     # BackwardContext, BackwardResult
│       └── clients/
│           ├── __init__.py
│           └── openai.py       # OpenAI client wrapper
├── tests/
│   ├── unit/
│   └── integration/
├── design_docs/
└── main.py
```
