# Tracing and DAG Capture

The tracing system captures the execution graph from eager-mode Python code,
similar to `torch.fx`. With the Value system, tracing is **data-driven**: graph
structure is inferred from `Value.ref` dependencies rather than string IDs
embedded in args/kwargs.

## Design Goals

1. **Transparent Capture**: Users write normal Python; tracing happens automatically
2. **Value-Driven Dependencies**: Dependencies are discovered via `Value.ref`
3. **Profiling Ready**: Capture metadata for performance analysis
4. **Minimal Overhead**: Tracing should be fast for complex graphs

## Tracing Modes

The Tracer supports two modes for backwards compatibility:

### Value-Driven Mode (Recommended)

Use `Tracer.trace_values()` for the canonical Value-driven tracing approach:

- Inputs are wrapped as Values with refs pointing to input nodes
- Module calls return Values with refs pointing to graph nodes
- Dependencies are collected via `Value.ref` using `collect_refs()`
- Arguments are stored with `ValueRef` placeholders

```python
tracer = Tracer()
graph = tracer.trace_values(module, "input text")
```

### Proxy-Based Mode (Legacy)

Use `Tracer.trace()` for backwards-compatible Proxy-based tracing:

- Inputs are wrapped as Proxy objects with node IDs
- Module calls return Proxy objects
- Dependencies are tracked via `Proxy.node_id`
- Arguments are stored with `NodeRef` placeholders

```python
tracer = Tracer()
graph = tracer.trace(module, "input text")
```

Both modes produce equivalent `InferenceGraph` structures. Value-driven mode
is preferred for new code as it aligns with the broader Value-based data model.

## Core Concepts

### Value-Driven Capture

During tracing, module calls return `Value` objects with `ref` pointing to the
node that produced them. Dependencies are collected by traversing inputs and
extracting `Value.ref`:

```python
# Pseudocode
inputs = valueify(user_inputs)
inputs = tracer.bind_inputs(inputs)  # assign input refs
outputs = module.forward(*inputs, **kwargs)
output_ids = collect_refs(outputs)
```

### Graph Nodes

Each module invocation creates a node in the graph:

```python
@dataclass
class GraphNode:
    """A single operation in the execution graph."""

    id: str                                  # Unique identifier
    module: Module                  # The module to execute
    args: tuple[Any, ...]                    # Raw payloads or ValueRef
    kwargs: dict[str, Any]                   # Raw payloads or ValueRef
    dependencies: list[str]                  # Input refs (Value.ref)
    priority: int = 0                        # Execution priority (lower = higher precedence)

    # Profiling metadata
    module_name: str = ""                    # Human-readable name
    module_path: str = ""                    # Full path in module tree

    def __post_init__(self):
        if not self.module_name:
            self.module_name = self.module.__class__.__name__
```

### Inference Graph

The complete execution graph:

```python
@dataclass
class InferenceGraph:
    """Complete execution graph captured from tracing."""

    nodes: dict[str, GraphNode]              # All nodes
    input_ids: list[str]                     # Entry point nodes
    output_ids: list[str]                    # Exit point nodes
    parameters: dict[str, Parameter]         # All parameters in graph

    def topological_order(self) -> list[str]: ...
    def descendants(self, node_id: str) -> set[str]: ...
    def ancestors(self, node_id: str) -> set[str]: ...
```

## ValueRef (Argument Placeholder)

To preserve where dependency values are needed at runtime, tracing stores
`ValueRef` placeholders inside `args`/`kwargs` when a `Value` input is used.

```python
@dataclass
class ValueRef:
    ref: str
```

At execution time, `ValueRef` is resolved by looking up the producing node’s
result. This avoids ambiguous string IDs while keeping argument structure intact.
See `values.md` for the ValueRef spec.

```python
def replace_values_with_refs(x: Any) -> Any:
    \"\"\"Replace Value leaves with ValueRef(ref=...).\"\"\"
    # Walk pytree; for Value leaves return ValueRef(value.ref) else literal
    ...
```

## The Tracer

The tracer captures the graph during forward execution:

```python
class Tracer:
    """Records an InferenceGraph by tracing forward() execution."""

    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.input_ids: list[str] = []
        self.output_ids: list[str] = []
        self._node_counter: int = 0
        self._module_stack: list[str] = []

    def trace(self, module: Module, *args: Any, **kwargs: Any) -> InferenceGraph:
        with trace_context(self):
            inputs = valueify(args)
            kw_inputs = valueify(kwargs)
            inputs = self.bind_inputs(inputs)
            kw_inputs = self.bind_inputs(kw_inputs)

            output = module.forward(*inputs, **kw_inputs)
            self.output_ids = collect_refs(output)

        return InferenceGraph(
            nodes=self.nodes,
            input_ids=self.input_ids,
            output_ids=self.output_ids,
            parameters=dict(module.named_parameters()),
        )

    def bind_inputs(self, inputs: Any) -> Any:
        """Assign input refs and create input nodes."""
        # Walk pytree, wrap leaves as Value with ref like input:0, input:kw
        ...

    def record_call(self, module: Module, args: tuple, kwargs: dict) -> Value:
        """Record a module invocation during tracing."""
        node_id = self._generate_id(module)
        dependencies = collect_refs(args, kwargs)
        node = GraphNode(
            id=node_id,
            module=module,
            args=replace_values_with_refs(args),
            kwargs=replace_values_with_refs(kwargs),
            dependencies=dependencies,
            module_path=".".join(self._module_stack),
        )
        self.nodes[node_id] = node
        return Value(kind=ValueKind.RESPONSE, payload=None, ref=node_id)

    def _generate_id(self, module: Module) -> str:
        self._node_counter += 1
        return f"{module.__class__.__name__}_{self._node_counter}"
```

## Parameters in Tracing

Parameters are lifted into `Value` when used as inputs. Their refs are stable
(e.g., `param:module.path.name`) and are collected as dependencies like any
other `Value.ref`.

## Structured Access (getitem)

`Value.__getitem__` delegates to `F.select` for structured data access. When
tracing is active, a select operation can optionally record a graph node:

```python
data = valueify({"user": {"name": "Ada"}})
name = data["user"]["name"]  # each [] calls F.select
```

Note: In the current implementation, `F.select` is pure and does not consult
the trace context. Structured access during tracing operates on Value payloads
directly without creating graph nodes. Future versions may add trace-aware
select operations for more granular dependency tracking.

## Trace Context

Thread-local context for tracing:

```python
from contextvars import ContextVar

_trace_context: ContextVar[Tracer | None] = ContextVar("trace_context", default=None)


def get_trace_context() -> Tracer | None:
    return _trace_context.get()


@contextmanager
def trace_context(tracer: Tracer):
    token = _trace_context.set(tracer)
    try:
        yield tracer
    finally:
        _trace_context.reset(token)
```

## Graph Visualization

Visualization is unchanged; edges come from `dependencies` and outputs from
`Value.ref` collection:

```python
def visualize_graph(graph: InferenceGraph) -> str:
    lines = ["digraph InferenceGraph {"]
    lines.append("  rankdir=TB;")

    for node_id, node in graph.nodes.items():
        label = f"{node.module_name}"
        shape = "ellipse"
        if node_id in graph.input_ids:
            shape = "box"
        elif node_id in graph.output_ids:
            shape = "doubleoctagon"
        lines.append(f'  "{node_id}" [label="{label}", shape={shape}];')

    for node_id, node in graph.nodes.items():
        for dep_id in node.dependencies:
            lines.append(f'  "{dep_id}" -> "{node_id}";')

    lines.append("}")
    return "\n".join(lines)
```

## Profiling Integration

Profiling remains the same; the tracer attaches profiling metadata to each
recorded node.

## Example: Complete Tracing Flow

```python
class AnalysisPipeline(Module):
    def __init__(self):
        super().__init__()
        self.summarize = LLMInference(alias="fast")
        self.analyze = LLMInference(alias="smart")
        self.format = LLMInference(alias="fast")

    def forward(self, text: str) -> Value:
        summary = self.summarize(text)
        analysis = self.analyze(summary)
        formatted = self.format(analysis)
        return formatted

tracer = Tracer()
graph = tracer.trace(AnalysisPipeline(), "Some long document...")

print(f"Nodes: {len(graph.nodes)}")
print(f"Inputs: {graph.input_ids}")
print(f"Outputs: {graph.output_ids}")

for node_id in graph.topological_order():
    node = graph.nodes[node_id]
    print(f"  {node_id}: {node.module_name} <- {node.dependencies}")
```

## Best Practices

### 1. Keep Forward Methods Pure

```python
# Good: Deterministic, no side effects
+def forward(self, text: str) -> Value:
    return self.llm(text)

# Bad: Side effects during tracing
 def forward(self, text: str) -> Value:
     print(f"Processing: {text}")  # Runs during tracing
     self.counter += 1
     return self.llm(text)
```

### 2. Avoid Dynamic Module Creation

```python
# Good: Modules created in __init__
 def __init__(self):
     self.handlers = [LLMInference(alias="llm") for _ in range(3)]

# Bad: Modules created in forward
 def forward(self, text: str):
     handler = LLMInference(alias="llm")
     return handler(text)
```
