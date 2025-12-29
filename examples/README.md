# inf-engine Examples

This directory contains practical examples demonstrating the inf-engine API.

## Examples

### [01_basic_modules.py](01_basic_modules.py)
Basic module creation and composition. Shows how to:
- Create custom `InferenceModule` subclasses
- Implement `forward()` methods
- Compose modules together
- Return multiple outputs

```bash
uv run python examples/01_basic_modules.py
```

### [02_parameters.py](02_parameters.py)
Working with learnable parameters. Shows how to:
- Use `Parameter` for learnable values
- Mix fixed and learnable parameters
- Discover parameters in nested modules
- Simulate parameter updates (preview of optimization)

```bash
uv run python examples/02_parameters.py
```

### [03_llm_pipelines.py](03_llm_pipelines.py)
LLM pipeline definitions. Shows how to:
- Define `LLMInference` modules with aliases
- Build sequential pipelines
- Create parallel (fan-out) analysis
- Compose fan-in synthesis patterns
- Use learnable system prompts

**Note:** To execute these pipelines with real LLMs, configure resources
using `ResourceConfig` and bind them to modules. See examples 05-07 for
execution patterns.

```bash
uv run python examples/03_llm_pipelines.py
```

### [04_tracing.py](04_tracing.py)
Tracing and DAG capture. Shows how to:
- Use `Tracer` to capture execution graphs
- Inspect graph nodes and dependencies
- Trace parallel (fan-out) patterns
- Trace diamond (fan-out + fan-in) patterns
- Use graph traversal methods (topological order, ancestors, descendants)
- Inspect captured parameters and module details

```bash
uv run python examples/04_tracing.py
```

### [05_execution.py](05_execution.py)
Execution with `run()`, `bind()`, and `ExecutionSettings`. Shows how to:
- Execute modules with `run()`
- Use `bind()` for direct `await module(input)` pattern
- Share resources with `ExecutionSettings` context
- Process batches with `await module([a, b, c])`
- Control concurrency with `max_concurrent`
- Handle errors and failure cascading

```bash
uv run python examples/05_execution.py
```

### [06_checkpointing.py](06_checkpointing.py)
Progress checkpointing for long-running pipelines. Shows how to:
- Save execution progress to disk
- Resume from checkpoints after interruption
- Use `CheckpointManager` for buffered writes
- Configure checkpointing via `ExecutionSettings`

```bash
uv run python examples/06_checkpointing.py
```

### [07_execution_settings.py](07_execution_settings.py)
ExecutionSettings and module binding patterns. Shows how to:
- Use `bind()` for the clean `await module(input)` pattern
- Share resources across modules with `ExecutionSettings`
- Understand configuration priority (kwargs > bound > context)
- Process batches concurrently
- Use nested contexts and method chaining

```bash
uv run python examples/07_execution_settings.py
```

## Best Practices

### Recommended Execution Pattern

For production code, use `bind()` or `ExecutionSettings`:

```python
from inf_engine.execution.context import ExecutionSettings
from inf_engine.resources.config import ResourceConfig, EndpointConfig

# Configure resources
resources = ResourceConfig(
    endpoints={
        "fast": EndpointConfig(
            provider_api="openai",
            model="gpt-4o-mini",
            max_concurrent=10,
        ),
        "smart": EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            max_concurrent=5,
        ),
    }
)

# Option 1: bind() for single module
pipeline = MyPipeline().bind(resources=resources)
result = await pipeline("input")
results = await pipeline(["a", "b", "c"])  # Batch

# Option 2: ExecutionSettings for multiple modules
async with ExecutionSettings(resources=resources, max_concurrent=50):
    result1 = await pipeline1("input")
    result2 = await pipeline2("input")
```

### Configuration Priority

Settings are applied with this priority (highest first):
1. **Call-time kwargs**: `await module("x", max_concurrent=10)`
2. **Bound settings**: `module.bind(max_concurrent=50)`
3. **Context settings**: `ExecutionSettings(max_concurrent=100)`
4. **Defaults**

## Running Examples

From the repository root:

```bash
# Run a specific example
uv run python examples/01_basic_modules.py

# Or run all examples
for f in examples/*.py; do uv run python "$f"; echo; done
```

## What's Working Now

| Feature | Status | Example |
|---------|--------|---------|
| Module creation | ✅ | 01_basic_modules.py |
| Module composition | ✅ | 01_basic_modules.py |
| Parameters | ✅ | 02_parameters.py |
| LLMInference definition | ✅ | 03_llm_pipelines.py |
| Tracing/DAG capture | ✅ | 04_tracing.py |
| Execution with run() | ✅ | 05_execution.py |
| bind() and direct execution | ✅ | 05_execution.py, 07_execution_settings.py |
| ExecutionSettings context | ✅ | 07_execution_settings.py |
| Batch execution | ✅ | 05_execution.py, 07_execution_settings.py |
| Streaming batch results | ✅ | 07_execution_settings.py |
| Checkpointing | ✅ | 06_checkpointing.py |
| Resource configuration | ✅ | (see Best Practices above) |
| Rate limiting | ✅ | (automatic with ResourceManager) |

## Coming Soon

After Phase 6 (Optimization):
- Training loops with `train()`
- Backward pass and feedback propagation
- Parameter optimization with LLM-based updates
- Complete end-to-end learning examples
