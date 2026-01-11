# Values and Data Containers

This document defines the `Value` container type used to represent data flowing
through inference graphs. `Value` replaces ad-hoc dependency encoding (string
node IDs in args/kwargs) with a uniform, type-aware container that carries both
payload and provenance.

## Goals

1. **Uniform data model**: Strings, f-strings, structured LLM responses, errors,
   structured objects, tool results, and binary payloads are represented consistently.
2. **Explicit provenance**: Dependencies are carried by `Value.ref`, not by
   parsing positional/keyword arguments.
3. **Safe composition**: Nested containers (lists, dicts, tuples) are supported
   without collisions or special casing.
4. **Minimal user friction**: Users can pass plain Python values; the runtime
   wraps/unwraps them automatically.

## Core Data Model

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValueKind(str, Enum):
    TEXT = "text"
    FSTRING = "fstring"
    RESPONSE = "response"
    STRUCTURED = "structured"
    INT = "int"
    FLOAT = "float"
    ERROR = "error"
    TOOL_RESULT = "tool_result"
    BINARY = "binary"
    OTHER = "other"


@dataclass
class Value:
    """Container for payload + provenance."""

    kind: ValueKind
    payload: Any
    ref: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str | int) -> "Value":
        """Graph-aware structured access (delegates to F.select)."""
        from plait import functional as F
        return F.select(self, key)

    def get(self, key: str | int, default: Any | None = None) -> "Value":
        """Graph-aware structured access with default."""
        from plait import functional as F
        return F.select(self, key, default=default)
```

### Semantics

- **payload**: The raw value (string, dict, response object, exception, bytes).
- **kind**: A discriminant for downstream handling and formatting.
- **ref**: Optional graph node ID that produced this value.
- **meta**: Optional metadata (model alias, tokens, schema, source, cost).

## Parameters vs Values

`Parameter` is **state**, while `Value` is **data**. Parameters are *not* a
subclass of `Value`. Instead, parameters are **lifted into values** when used
in computation:

```python
valueify(param) -> Value(
    kind=TEXT | STRUCTURED,
    payload=param.value,
    ref="param:module.path.name",
    meta={"param_name": "...", "param_id": "..."}
)
```

This keeps optimization state separate while still making parameter usage
visible to the graph via stable `ref` identifiers.

See `parameters.md` for the full Parameter specification.

## Construction Helpers

The runtime provides helpers to normalize inputs and outputs:

```python
def valueify(x: Any, *, kind: ValueKind | None = None) -> Value:
    """Wrap raw values into Value with optional kind override."""

def unwrap(x: Any) -> Any:
    """Return payload if x is Value, otherwise x unchanged."""

def collect_refs(*args: Any, **kwargs: Any) -> list[str]:
    """Recursively collect .ref from Values in nested structures."""
```

## ValueRef (Execution Placeholder)

During tracing, `Value` inputs are replaced with `ValueRef` placeholders inside
node args/kwargs. Execution resolves these references to the producing node’s
result:

```python
@dataclass
class ValueRef:
    ref: str
```

`ValueRef` avoids ambiguous string IDs and preserves argument structure while
still allowing dependency resolution.

## Interaction With Tracing

### Data-Driven Trace via Value Flow

Tracing can be driven by Value flow alone: send `Value` (or nested structures
of `Value`) through the module DAG and collect the output `Value.ref`s. The
graph is inferred from the refs captured during module calls.

Conceptually:

```python
inputs = valueify(user_inputs)  # Value or pytree of Value
inputs = tracer.bind_inputs(inputs)  # assign input-node refs
outputs = module.forward(inputs)  # returns Value(s) with ref set
output_ids = collect_refs(outputs)
```

### Module Call Recording Still Required

`Value.ref` is assigned when a module is invoked. This still requires the
tracing context to intercept module calls and assign node IDs. The simplification
is that dependencies are discovered from `Value.ref` rather than by parsing
string IDs in args/kwargs.

During tracing, modules return `Value` with `ref` pointing to the graph node.
Dependencies are discovered by scanning inputs for `Value.ref`:

```python
# Pseudocode
deps = collect_refs(args, kwargs)
node_id = tracer.record_call(module, args, kwargs, dependencies=deps)
return Value(kind=ValueKind.RESPONSE, payload=None, ref=node_id)
```

This eliminates ambiguity from string literals and supports nested structures.

## Interaction With Execution

During execution:

- Inputs are `valueify()`-wrapped so they can carry metadata/provenance.
- When invoking a module, `unwrap()` is used to pass raw payloads into
  `forward()` implementations or LLM clients.
- Outputs are re-wrapped as `Value` with an appropriate `kind` and `ref`.

## Batches and Collections (First-Class)

Batches, iterables, and mappings are first-class citizens. The runtime treats
lists/tuples/dicts as structured containers of `Value` objects:

- `valueify()` recurses into containers to wrap each element.
- `collect_refs()` traverses containers and collects all `Value.ref`s.
- `unwrap()` maps containers back to raw payloads for execution.

Example (batch of inputs):

```python
inputs = ["a", "b", "c"]
values = valueify(inputs)  # -> list[Value(TEXT)]
```

Example (structured output):

```python
output = {
    "summary": Value(ValueKind.TEXT, "...", ref="n1"),
    "entities": [Value(ValueKind.STRUCTURED, {...}, ref="n2")],
}
refs = collect_refs(output)  # -> ["n1", "n2"]
```

Optional extension for explicit batch semantics:

```python
@dataclass
class ValueBatch:
    items: list[Value]
    meta: dict[str, Any] = field(default_factory=dict)  # batch_id, size, etc.
```

`ValueBatch` is useful when the batch itself has semantics (batch_id, ordering
constraints, aggregation), but plain lists/tuples are sufficient for most cases.

## Structured Access (getitem)

Structured access is graph-aware. `Value.__getitem__` delegates to `F.select`,
so key access records a node and returns a new `Value` with its own `ref`:

```python
data = valueify({"user": {"name": "Ada"}})
name = data["user"]["name"]  # each [] is a select op
```

If the selected payload is structured, `F.select` returns `Value(STRUCTURED)`,
so chaining continues to work without losing provenance.

## Functional Ops (Graph-Aware Functions)

In addition to stateful modules, the library can provide a functional API
similar to `torch.nn.functional` for stateless, graph-aware operations on
`Value`s:

```python
import plait.functional as F

template = valueify("Summarize: {text}", kind=ValueKind.FSTRING)
vars = valueify({"text": "long doc"})

prompt = F.render(template, vars)          # -> Value(TEXT, ref=...)
summary = llm(prompt)                      # -> Value(RESPONSE, ref=...)
```

### Design Notes

- Functional ops accept raw values or `Value`; inputs are `valueify()`-normalized.
- When tracing is active, each op records a graph node and returns a `Value`
  with `ref` pointing to that node.
- During execution, ops operate on `payload` and return `Value` with `ref`.
- Typical ops: `render`, `concat`, `format`, `parse_structured`, `select`,
  `coerce`, `unwrap_or`, `merge`.

This provides a consistent, lightweight API for pure transforms without
forcing users to define a custom `Module`.

## Mapping Common Payloads

| Payload Type | ValueKind | Notes |
|-------------|-----------|-------|
| `str` | `TEXT` | Default for user text |
| f-string template | `FSTRING` | Format with `{name}` slots |
| LLM response object | `RESPONSE` | Includes tokens, model in `meta` |
| `dict` / structured | `STRUCTURED` | Use for structured results |
| `int` | `INT` | Integer scalar values |
| `float` | `FLOAT` | Floating-point scalar values |
| `Exception` | `ERROR` | Preserve traceback in `meta` |
| tool result | `TOOL_RESULT` | Standardized tool outputs |
| `bytes` | `BINARY` | Use for files/images |

## Example Use Cases

- **F-string rendering**: `Value(FSTRING)` with variables in `meta`, producing
  a `Value(TEXT)` for the rendered prompt.
- **LLM response normalization**: `Value(RESPONSE)` wrapping provider-specific
  response objects, exposing a normalized `payload` and `meta` (tokens, model).
- **Tool invocation results**: `Value(TOOL_RESULT)` to standardize outputs from
  external tools; downstream modules consume the same shape.
- **Structured outputs**: `Value(STRUCTURED)` for schema-validated responses.
- **Error-as-data**: `Value(ERROR)` propagates failures with provenance for
  debugging and retry logic.
 
## Error Handling

Errors are also `Value`s:

```python
Value(kind=ValueKind.ERROR, payload=err, meta={"node_id": node_id})
```

This allows error propagation without losing provenance.

## Relationship to PyTorch

`Value` plays the role of a Tensor-like carrier of both data and graph
provenance. Like `Tensor`, it allows most graph recording to be data-driven,
with Tracer fallback for full program capture when needed.

## Migration Plan (Design-Level)

1. Introduce `Value` and helper functions in a new module (e.g. `values.py`).
2. Update tracing to discover dependencies via `Value.ref`.
3. Update execution to `valueify()` inputs and `unwrap()` before calling
   `forward()`/LLM clients.
4. Deprecate string-based node references in args/kwargs.
