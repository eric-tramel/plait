# Parameters

This document defines the `Parameter` abstraction used for learnable, persistent
state (e.g., prompts, templates, configuration strings) and its interaction with
`Value` and optimization.

## Goals

1. **State, not data**: Parameters are persistent state, not transient `Value`s.
2. **Optimization-ready**: Parameters collect feedback and are updated by
   optimizers.
3. **Graph-visible usage**: When used in computation, parameters are lifted into
   `Value` with stable `ref` for dependency tracking.
4. **Minimal surface**: Simple API mirroring `torch.nn.Parameter` semantics.

## Core Data Model

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Parameter:
    """A learnable value optimized via backward passes."""

    value: Any
    description: str | None = None  # Required when requires_grad=True
    requires_grad: bool = True

    # Internal state (managed by framework)
    _name: str | None = field(default=None, repr=False)
    _feedback_buffer: list[str] = field(default_factory=list, repr=False)

    def accumulate_feedback(self, feedback: str) -> None: ...
    def apply_update(self, new_value: Any) -> None: ...
    def zero_feedback(self) -> None: ...
```

### Semantics

- **value**: The current parameter payload (commonly `str`, but can be structured).
- **description**: Required when `requires_grad=True`. Explains what the parameter
  does so optimizers can make coherent updates.
- **requires_grad**: Whether the parameter participates in optimization.
- **_name**: Set by the parent module for hierarchical naming.
- **_feedback_buffer**: Accumulates feedback for optimizer steps.

## Parameter vs Value

`Parameter` is **state**. `Value` is **data**. Parameters are not `Value`s.
When a parameter is used in computation, it is lifted to a `Value` via
`valueify(param)`.

### Lifting Parameters into Values

```python
valueify(param) -> Value(
    kind=TEXT | STRUCTURED,
    payload=param.value,
    ref="param:module.path.name",
    meta={
        "param_name": "module.path.name",
        "param_id": "<stable-id>",
        "module_state_version": "<int>",
        "requires_grad": param.requires_grad,
    }
)
```

This makes parameter usage visible to the graph while preserving
separate optimization state.

## Identity and Naming

- Parameter refs are stable and hierarchical: `param:<module.path.name>`.
- `_name` is assigned by `Module.__setattr__` when a parameter is set.
- `named_parameters()` yields fully qualified names used in `ref` generation.

## Module State Versioning

Parameters belong to modules. Each module maintains a `module_state_version`
that increments whenever any parameter in that module updates. This version is
included in `valueify(param).meta` to attribute outputs to a specific module
state.

## Optimization Lifecycle

1. **Forward**: Parameters are read; values are lifted into `Value` with stable refs.
2. **Backward**: Feedback is accumulated into `_feedback_buffer`.
3. **Step**: Optimizer computes updates and calls `apply_update()`.
4. **Reset**: `zero_feedback()` clears feedback buffers between batches.

## Interop Rules

- If `requires_grad=False`, parameter is lifted into a `Value` but does not
  accumulate feedback.
- Parameters may appear in structured values; lifting applies recursively.
- Parameter values should be treated as **immutable** during a single forward
  pass to avoid inconsistent graphs.

## Error Handling

Errors in parameter operations should not surface as runtime exceptions during
normal execution. Errors should be encoded as `Value(ERROR)` when interacting
through functional ops.

## Design Notes

- Keep `Parameter` minimal. Do not embed execution/runtime concerns here.
- Prefer using `Parameter` for text-like or structured configuration that needs
  optimization; for ephemeral data, use `Value`.
- If a parameter value is structured (dict/list), use `ValueKind.STRUCTURED` when
  lifting.
