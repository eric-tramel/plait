# plait Design Documentation

plait is a PyTorch-inspired framework for building, executing, and
optimizing **compound AI systems**: multi-step LLM pipelines that need
concurrency, backpressure, observability, and (optionally) feedback-driven
parameter updates.

This directory is the source of truth for the system’s design. The goal is to
mirror Torch ergonomics (modules, parameters, forward/backward, optimizers) while
making async DAG execution and resource management first-class.

## Motivation (Why This Exists)

Most useful LLM applications are systems, not single calls: they chain multiple
LLM invocations, pass structured data between steps, run verifiers/judges, and
need consistent handling for retries and rate limits.

In plain Python, that complexity tends to leak everywhere:
- Concurrency is manual and brittle.
- Backpressure/retries get re-implemented per call site.
- It’s hard to see where time is going (critical path vs. bubbles).
- “Learning” (updating prompts/instructions from feedback) is difficult without a
  stable record of what happened in the forward pass.

plait’s core idea is to keep authoring simple and familiar, and move the
complexity into a shared runtime:
1) write normal module composition in `forward()`, 2) trace it into a DAG,
3) execute it with a scheduler + resource manager, and 4) optionally propagate
feedback backward to update `Parameter`s.

### Quick Anchor Example

```python
from plait import Module, LLMInference, Parameter, ResourceConfig, run


class SummarizeThenAnalyze(Module):
    def __init__(self):
        super().__init__()
        self.instructions = Parameter(
            value="Be concise and concrete.",
            description="Controls the style and structure of analysis output.",
        )
        self.summarize = LLMInference(alias="fast", system_prompt="Summarize the input.")
        self.analyze = LLMInference(alias="smart", system_prompt=self.instructions)

    def forward(self, text: str) -> str:
        summary = self.summarize(text)
        return self.analyze(f"Summary:\n{summary}\n\nAnalysis:")


resources = ResourceConfig({
    "fast": {"model": "gpt-4o-mini", "max_concurrent": 20},
    "smart": {"model": "gpt-4o", "max_concurrent": 5},
})

result = await run(SummarizeThenAnalyze(), "input text", resources=resources)
```

You author a pipeline as plain Python composition. When executed through the
runtime, the pipeline is traced into an `InferenceGraph` and run with
resource-aware scheduling.

## What This Enables

### Torch-like authoring for compound AI systems

- `Module` provides a familiar module boundary (like `nn.Module`).
- `LLMInference` is the atomic unit for model calls (executed via resources).
- `Parameter` represents learnable, persistent state (often prompts/config).

**Why it matters**: you can build larger systems without giving up readability or
composability, and you can keep “what the system does” separate from “how it is
served/executed”.

### Automatic DAG capture from eager-mode code

Tracing (similar to `torch.fx`) records each module invocation as a node and
discovers dependencies via `Value.ref`. Argument structure is preserved using
`ValueRef` placeholders so execution can later resolve inputs by node output.

**Why it matters**: once you have a DAG, scheduling, profiling, and backward
propagation become graph problems instead of scattered conventions.

### Async execution with resource-aware scheduling and backpressure

Execution is built around:
- `ExecutionState` to track pending/in-progress/completed nodes
- a `Scheduler` (priority queue) to run ready nodes
- `ResourceManager` to execute `LLMInference` through alias-bound endpoints with
  concurrency limits, rate limiting, retries, and metrics

**Why it matters**: independent work runs concurrently without users wiring
`asyncio.gather(...)` everywhere, and backpressure policies are applied
consistently across the system.

### A uniform `Value` container for payload + provenance (including errors)

`Value` is the shared data model for text, f-strings, structured data, responses,
errors, numeric values, tool results, and more. Values optionally carry a `ref`
to the producing node, which makes dependency tracking unambiguous even in nested
containers.

Functional ops prefer **error-as-data**: failures are returned as `Value(ERROR)`
instead of raising runtime exceptions, enabling deterministic propagation.

**Why it matters**: it’s the glue that makes tracing/execution reliable with
structured inputs/outputs and partial failures.

### A functional API for stateless, graph-aware operations

`plait.functional` is the `torch.nn.functional` analogue: a catalog of
stateless, graph-aware operations on `Value`s (rendering, parsing, selection,
coercion, numeric and collection ops, response metadata extractors, and an atomic
LLM transport op). These functions define consistent coercion and error
propagation rules.

**Why it matters**: it reduces special casing in modules and provides one place
to specify semantics (especially around structured access and error behavior).

### Observability that matches the DAG runtime

Profiling emits Chrome Trace Event JSON compatible with Perfetto / Chrome
DevTools, exposing concurrency, endpoint utilization, and idle “bubbles”.

**Why it matters**: async performance problems are difficult to diagnose without
a timeline keyed to graph nodes and resource slots.

### Feedback-driven optimization

Optimization mirrors PyTorch’s loop:
- losses produce `Feedback`
- `feedback.backward()` propagates through captured forward records
- parameters accumulate feedback
- `optimizer.step()` updates parameters

The design includes ordered parameter updates (topological ordering with upstream
visibility) and module-level cohesion to avoid semantic mismatches between
dependent parameters.

**Why it matters**: you can iterate on prompts/instructions systematically, using
the same execution stack and resource configuration as the forward system.

## Architecture (How The Pieces Fit Together)

At a high level:

1. **User code**: `Module.forward()` expresses composition in Python.
2. **Tracing**: the `Tracer` runs `forward()` in a trace context; module calls
   are recorded as nodes; dependencies come from `Value.ref`; args/kwargs store
   `ValueRef` placeholders.
3. **Execution**: an `InferenceGraph` is executed by `ExecutionState` and a
   priority-queue `Scheduler`. `LLMInference` calls go through `ResourceManager`
   using alias-bound endpoints.
4. **(Optional) Optimization**: training-mode execution attaches forward records;
   losses produce `Feedback`; `feedback.backward()` propagates feedback through
   the graph; `optimizer.step()` applies ordered updates.

See **[Architecture](./architecture.md)** for the full overview.

## Reading Order

If you’re new to the design, read these in order:

1. **[Architecture](./architecture.md)** - High-level system overview
2. **[Values](./values.md)** - Data container + provenance (`Value`, `ValueRef`)
3. **[Parameters](./parameters.md)** - Learnable state and lifting into `Value`
4. **[Functional API](./functional_api.md)** - Stateless, graph-aware ops
5. **[Module](./inference_module.md)** - Core module system
6. **[Tracing](./tracing.md)** - How DAGs are captured from code
7. **[Execution](./execution.md)** - Scheduler, state, error handling, checkpointing
8. **[Resources](./resources.md)** - Endpoint configuration and pooling
9. **[Optimization](./optimization.md)** - Feedback/backward/optimizers
10. **[Profiling](./profiling.md)** - Performance visualization and analysis

## PyTorch Parallels

| PyTorch | plait | Purpose |
|---------|------------|---------|
| `nn.Module` | `Module` | Base class for operations |
| `nn.Parameter` | `Parameter` | Learnable values (state) |
| `forward()` | `forward()` | Define computation |
| `backward()` | `backward()` | Propagate gradients/feedback |
| `torch.fx.Tracer` | `Tracer` | Capture computation graph |
| `torch.optim.*` | `Optimizer` | Update parameters |

## Code Map (High-Level)

```
plait/
├── src/
│   └── plait/
│       ├── module.py           # Module, LLMInference
│       ├── parameter.py        # Parameter
│       ├── values.py           # Value, ValueKind, helpers
│       ├── functional.py       # plait.functional
│       ├── graph.py            # InferenceGraph, GraphNode
│       ├── tracing/            # Tracer + trace context (+ legacy proxy)
│       ├── execution/          # Scheduler, ExecutionState, checkpoints
│       ├── resources/          # ResourceConfig/ResourceManager, rate limiting
│       ├── optimization/       # Loss/Feedback/Optimizer, backward propagation
│       ├── profiling/          # Chrome Trace output
│       └── clients/            # Provider clients (e.g., OpenAI)
├── tests/
├── design_docs/                # You are here
└── main.py
```

## Contributing

When evolving the system:
1. Update the relevant design document(s)
2. Keep terminology consistent (`Value`, `Parameter`, tracing/execution semantics)
3. Add a small usage example when introducing new surface area
4. Prefer error-as-data (`Value(ERROR)`) for functional ops and value-level failures
