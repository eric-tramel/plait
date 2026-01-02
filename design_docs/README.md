# inf-engine Design Documentation

This directory contains the comprehensive design documentation for inf-engine, a PyTorch-inspired framework for building, executing, and optimizing LLM inference pipelines.

## Quick Start

If you're new to inf-engine, read the documents in this order:

1. **[Architecture](./architecture.md)** - High-level system overview
2. **[Values](./values.md)** - Data container and provenance model
3. **[Parameters](./parameters.md)** - Learnable state and optimization contract
4. **[Functional API](./functional_api.md)** - Stateless graph-aware ops
5. **[InferenceModule](./inference_module.md)** - Core module system (like `nn.Module`)
6. **[Tracing](./tracing.md)** - How DAGs are captured from code
7. **[Execution](./execution.md)** - Scheduler and state management
8. **[Resources](./resources.md)** - Endpoint configuration
9. **[Optimization](./optimization.md)** - Backward passes and learning
10. **[Profiling](./profiling.md)** - Performance visualization and analysis

## Document Overview

### [Architecture](./architecture.md)
The foundational document explaining how all components fit together:
- System layers (user code → tracing → execution → infrastructure)
- Component interactions
- Execution flow (forward and backward)
- File structure

### [Values](./values.md)
The container type for data and provenance:
- `Value` and `ValueKind`
- Payload + graph reference semantics
- Interaction with tracing and execution

### [Parameters](./parameters.md)
Learnable state and how it participates in graphs:
- `Parameter` data model
- Lifting parameters into `Value`
- Optimization lifecycle

### [Functional API](./functional_api.md)
Stateless, graph-aware operations on `Value`:
- `inf_engine.functional` namespace
- Function catalog (render, concat, parse_structured, etc.)

### [InferenceModule](./inference_module.md)
The core abstraction for defining inference operations:
- `InferenceModule` base class (like `nn.Module`)
- `LLMInference` atomic operation
- `Parameter` for learnable values
- Module composition patterns (sequential, parallel, nested)

### [Tracing](./tracing.md)
How execution graphs are captured from eager-mode code:
- Value-driven dependency capture (`Value.ref`)
- `Tracer` implementation
- Graph visualization

### [Execution](./execution.md)
Runtime execution of traced graphs:
- `ExecutionState` for tracking tasks
- `Scheduler` with priority queue
- Adaptive rate limiting
- Checkpointing and recovery
- The `run()` function
- Execution patterns: sync, async, streaming
- `BatchResult` for streaming error handling

### [Resources](./resources.md)
Infrastructure configuration separate from module definitions:
- `ResourceConfig` structure
- `ResourceManager` for coordination
- LLM client implementations
- Connection pooling and load balancing
- Metrics and cost tracking

### [Optimization](./optimization.md)
LLM-based learning through backward passes:
- `Feedback` and `Loss` functions
- `BackwardContext` and `BackwardResult`
- `Optimizer` for parameter updates
- Training loops

### [Profiling](./profiling.md)
Performance visualization and bottleneck analysis:
- Chrome Trace Event Format for tool compatibility
- Configuration via `ExecutionSettings`
- `TraceProfiler` for capturing execution data
- Integration with Perfetto and Chrome DevTools
- Bubble detection and endpoint utilization metrics

### [DESIGN.md](./DESIGN.md)
The original concise design document with quick reference pseudocode.

## Key Concepts

### PyTorch Parallels

| PyTorch | inf-engine | Purpose |
|---------|------------|---------|
| `nn.Module` | `InferenceModule` | Base class for operations |
| `nn.Parameter` | `Parameter` | Learnable values |
| `forward()` | `forward()` | Define computation |
| `backward()` | `backward()` | Propagate gradients/feedback |
| `torch.fx.Tracer` | `Tracer` | Capture computation graph |
| `torch.optim.SGD` | `Optimizer` | Update parameters |

### Core Workflow

```python
# 1. Define modules
class MyPipeline(InferenceModule):
    def __init__(self):
        self.summarize = LLMInference(alias="fast")
        self.analyze = LLMInference(alias="smart")

    def forward(self, text):
        summary = self.summarize(text)
        return self.analyze(summary)

# 2. Configure resources
resources = ResourceConfig({
    "fast": {"model": "gpt-4o-mini", "max_concurrent": 10},
    "smart": {"model": "gpt-4o", "max_concurrent": 5},
})

# 3. Execute
result = await run(MyPipeline(), "input text", resources=resources)

# 4. Optionally, optimize
optimizer = Optimizer(pipeline.parameters(), aggregator_llm)
feedback = await loss_fn(result)
await backward(pipeline, feedback)
await optimizer.step()
```

## Design Principles

1. **Familiarity**: Mirror PyTorch conventions
2. **Separation**: Modules don't know about endpoints
3. **Async**: Users write sync code; framework handles async
4. **Composability**: Arbitrary nesting and composition
5. **Observability**: Built-in profiling and metrics
6. **Efficiency**: Maximum parallelism, adaptive rate limiting

## File Structure

```
inf-engine/
├── src/
│   └── inf_engine/
│       ├── module.py           # InferenceModule, LLMInference
│       ├── parameter.py        # Parameter
│       ├── values.py           # Value, ValueKind, helpers
│       ├── graph.py            # InferenceGraph, GraphNode
│       ├── tracing/
│       │   ├── tracer.py       # Tracer
│       │   ├── proxy.py        # (legacy) Proxy helpers, if needed
│       │   └── context.py      # TraceContext
│       ├── execution/
│       │   ├── scheduler.py    # Scheduler
│       │   ├── state.py        # ExecutionState
│       │   ├── executor.py     # Executor, run()
│       │   └── checkpoint.py   # CheckpointManager
│       ├── resources/
│       │   ├── manager.py      # ResourceManager
│       │   ├── config.py       # ResourceConfig
│       │   └── rate_limit.py   # RateLimiter
│       ├── optimization/
│       │   ├── loss.py         # Loss functions
│       │   ├── optimizer.py    # Optimizer
│       │   └── backward.py     # BackwardContext
│       └── clients/
│           └── openai.py       # LLM clients
├── tests/
│   ├── unit/
│   └── integration/
├── design_docs/                 # You are here
└── main.py
```

## Contributing

When adding new features:
1. Update the relevant design document
2. Ensure consistency with PyTorch conventions
3. Add examples showing usage
4. Consider backward compatibility
