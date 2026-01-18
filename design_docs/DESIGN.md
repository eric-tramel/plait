# plait Design

An async DAG execution engine optimized for LLM workloads with:
- **Priority Queue** as the central scheduler
- **Adaptive backpressure** - re-queue tasks and adjust timing when rate-limited
- **Async-first** design for efficient I/O-bound LLM operations

## Architecture

### 1. Task Definition

A Task is the fundamental unit of work:

```python
@dataclass
class Task:
    id: str
    operation: Callable[[dict[str, Any]], Awaitable[Any]]
    priority: int = 0           # Lower = higher priority
    retry_count: int = 0        # Incremented on backpressure re-queue
    created_at: float = field(default_factory=time.time)
```

### 2. DAG Definition

A DAG defines tasks and their dependencies:

```python
@dataclass
class Node:
    id: str
    task: Task
    dependencies: list[str]     # Node IDs this node depends on

class DAG:
    nodes: dict[str, Node]
    results: dict[str, Any]     # Completed node results

    def get_ready_nodes(self) -> list[Node]:
        """Return nodes whose dependencies are all satisfied."""

    def mark_complete(self, node_id: str, result: Any) -> list[Node]:
        """Mark node complete, return newly-unblocked nodes."""
```

### 3. Priority Queue Scheduler

```python
class Scheduler:
    queue: asyncio.PriorityQueue[tuple[int, float, Task]]
    # Priority tuple: (priority, created_at, task) for stable ordering

    rate_limiter: RateLimiter   # Adaptive rate control

    async def enqueue(self, task: Task) -> None:
        await self.queue.put((task.priority, task.created_at, task))

    async def dequeue(self) -> Task:
        await self.rate_limiter.acquire()  # Wait if rate-limited
        _, _, task = await self.queue.get()
        return task
```

### 4. Adaptive Backpressure

When an LLM returns a rate-limit error (429), we:
1. Re-queue the task with incremented retry_count
2. Adjust the rate limiter to slow down requests

```python
class RateLimiter:
    tokens: float               # Current tokens available
    rate: float                 # Tokens per second (adaptive)
    max_tokens: float           # Bucket size

    async def acquire(self) -> None:
        """Wait until a token is available."""

    def backoff(self, retry_after: float | None = None) -> None:
        """Reduce rate after hitting backpressure."""

    def recover(self) -> None:
        """Gradually increase rate after successful requests."""
```

**Backpressure flow**:
1. Task execution hits 429 → catch exception
2. Call `rate_limiter.backoff(retry_after)`
3. Increment `task.retry_count`, re-enqueue with same priority
4. On successful requests, call `rate_limiter.recover()` to gradually restore rate

### 5. Execution Loop

```python
async def run(dag: DAG) -> dict[str, Any]:
    scheduler = Scheduler()

    # Seed queue with nodes that have no dependencies
    for node in dag.get_ready_nodes():
        await scheduler.enqueue(node.task)

    pending: set[str] = {n.id for n in dag.nodes.values()}

    async with asyncio.TaskGroup() as tg:
        while pending:
            task = await scheduler.dequeue()
            tg.create_task(execute_node(task, dag, scheduler, pending))

    return dag.results

async def execute_node(
    task: Task,
    dag: DAG,
    scheduler: Scheduler,
    pending: set[str]
) -> None:
    node = dag.nodes[task.id]

    # Gather inputs from dependencies
    inputs = {dep: dag.results[dep] for dep in node.dependencies}

    try:
        result = await task.operation(inputs)
        scheduler.rate_limiter.recover()

        # Mark complete and enqueue newly-ready nodes
        newly_ready = dag.mark_complete(node.id, result)
        pending.discard(node.id)

        for ready_node in newly_ready:
            await scheduler.enqueue(ready_node.task)

    except RateLimitError as e:
        # Backpressure: re-queue and slow down
        scheduler.rate_limiter.backoff(e.retry_after)
        task.retry_count += 1
        await scheduler.enqueue(task)
```

## Task Definition API

A PyTorch-like API for defining task compositions with tracing-based DAG capture.

### Op Base Class

```python
class Op:
    """Base class for all operations (like nn.Module)."""

    def __init__(self):
        self._children: dict[str, Op] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Op):
            self._children[name] = value
        super().__setattr__(name, value)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Override to define operation logic."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ctx = get_trace_context()
        if ctx is not None:
            return ctx.record(self, args, kwargs)
        return self.forward(*args, **kwargs)
```

### Value Objects

```python
@dataclass
class Value:
    """Container for payload + provenance."""
    kind: ValueKind
    payload: Any
    ref: str | None = None
```

When a Value with `ref` is passed to another Op during tracing, it creates a dependency edge.

### Tracer

```python
class Tracer:
    """Records DAG by tracing forward() calls."""

    def trace(self, op: Op, *args: Any, **kwargs: Any) -> DAG:
        """Trace op.forward() and return the captured DAG."""
        with trace_context(self):
            inputs = valueify(args)
            kw_inputs = valueify(kwargs)
            inputs = self.bind_inputs(inputs)
            kw_inputs = self.bind_inputs(kw_inputs)
            output = op.forward(*inputs, **kw_inputs)
            output_ids = collect_refs(output)
        return DAG(nodes=self.nodes, output_ids=output_ids)

    def record(self, op: Op, args: tuple, kwargs: dict) -> Value:
        """Called when an Op is invoked during tracing."""
        node_id = self._generate_id(op)
        dependencies = collect_refs(args, kwargs)
        self.nodes[node_id] = Node(
            id=node_id,
            op=op,
            dependencies=dependencies,
            args=replace_values_with_refs(args),
            kwargs=replace_values_with_refs(kwargs),
        )
        return Value(kind=ValueKind.RESPONSE, payload=None, ref=node_id)
```

### Example Usage

```python
class Summarize(Op):
    async def forward(self, text: str) -> str:
        return await llm.complete(f"Summarize: {text}")

class ExtractKeywords(Op):
    async def forward(self, text: str) -> list[str]:
        return await llm.complete(f"Extract keywords: {text}")

class Sentiment(Op):
    async def forward(self, text: str) -> str:
        return await llm.complete(f"Sentiment: {text}")

class Analyze(Op):
    def __init__(self):
        super().__init__()
        self.summarize = Summarize()
        self.keywords = ExtractKeywords()
        self.sentiment = Sentiment()

    def forward(self, text):
        summary = self.summarize(text)        # Node A
        keywords = self.keywords(summary)      # Node B (depends on A)
        sentiment = self.sentiment(text)       # Node C (parallel with A)
        return {
            "summary": summary,
            "keywords": keywords,
            "sentiment": sentiment,
        }

# Usage
analyzer = Analyze()
results = await run(analyzer, "Some long article text...")
```

**Resulting DAG:**
```
Input
  ├──▶ A (summarize) ──▶ B (keywords)
  └──▶ C (sentiment)
```

A and C run in parallel; B waits for A.

### Nested Composition

```python
class DeepAnalyzer(Op):
    def __init__(self):
        super().__init__()
        self.analyze = Analyze()      # Nested composite
        self.format = FormatOutput()

    def forward(self, text):
        analysis = self.analyze(text)  # Expands to A, B, C internally
        return self.format(analysis)   # Node D, depends on B and C
```

Tracing flattens the hierarchy into a single DAG.

### Fan-in Pattern

```python
class FullPipeline(Op):
    def __init__(self):
        super().__init__()
        self.summarize = Summarize()
        self.keywords = ExtractKeywords()
        self.sentiment = Sentiment()
        self.combine = Combine()

    def forward(self, text):
        s = self.summarize(text)
        k = self.keywords(s)
        sent = self.sentiment(text)
        return self.combine(s, k, sent)  # Fan-in: depends on s, k, sent
```

### Integration with Executor

```python
async def run(op: Op, *args: Any, **kwargs: Any) -> Any:
    """Trace an Op and execute the resulting DAG.

    Input values are bound at execution time and resolved via ValueRef placeholders.
    """
    tracer = Tracer()
    dag = tracer.trace(op, *args, **kwargs)
    scheduler = Scheduler()
    return await execute(dag, scheduler)
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backpressure | Re-queue + adaptive rate | Graceful handling of rate limits without dropping work |
| Priority | User-defined | Keep it simple; can add depth-based boosting later |
| Rate limiting | Token bucket | Standard algorithm, easy to tune |
| DAG capture | Tracing (like torch.fx) | Automatic dependency inference from forward() |
| Topology | Full DAG | Support fan-out, fan-in, and arbitrary graphs |

## File Structure

```
plait/
├── src/
│   ├── op.py           # Op base class
│   ├── values.py       # Value, ValueKind, helpers
│   ├── tracer.py       # Tracer implementation
│   ├── dag.py          # DAG and Node definitions
│   ├── scheduler.py    # Priority queue + rate limiter
│   ├── executor.py     # Main execution loop
│   └── errors.py       # RateLimitError and other exceptions
├── tests/
└── main.py
```
