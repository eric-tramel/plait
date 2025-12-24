# inf-engine Examples

This directory contains practical examples demonstrating the inf-engine API.

## Current Examples (Phase 1-3)

These examples work with the current implementation:

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
LLM pipeline definitions (structure preview). Shows how to:
- Define `LLMInference` modules
- Build sequential pipelines
- Create parallel (fan-out) analysis
- Compose fan-in synthesis patterns
- Use learnable system prompts

**Note:** These pipelines cannot execute with real LLMs yet (requires
Phase 4 resources). This example demonstrates the API and inspects
module structure.

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
**NEW!** Execution with the `run()` function. Shows how to:
- Execute modules with `run()`
- Run linear pipelines with dependency ordering
- Execute parallel (fan-out) patterns concurrently
- Handle diamond patterns (fan-out + fan-in)
- Control concurrency with `max_concurrent`
- Inspect execution state (PENDING, COMPLETED, FAILED, CANCELLED)
- Handle errors and failure cascading
- Work with multiple inputs (positional and keyword)

```bash
uv run python examples/05_execution.py
```

## Running Examples

From the repository root:

```bash
# Run a specific example
uv run python examples/01_basic_modules.py

# Or run all examples
for f in examples/*.py; do uv run python "$f"; echo; done
```

## What's Working Now (Phase 1-3)

| Feature | Status | Example |
|---------|--------|---------|
| Module creation | ✅ | 01_basic_modules.py |
| Module composition | ✅ | 01_basic_modules.py |
| Parameters | ✅ | 02_parameters.py |
| LLMInference definition | ✅ | 03_llm_pipelines.py |
| Tracing/DAG capture | ✅ | 04_tracing.py |
| Execution with run() | ✅ | 05_execution.py |
| Concurrent execution | ✅ | 05_execution.py |
| Dependency ordering | ✅ | 05_execution.py |
| Error handling | ✅ | 05_execution.py |

## Coming Soon

After Phase 4 (Resources):
- Real LLM execution with OpenAI/compatible endpoints
- Resource configuration and management
- Rate limiting and endpoint pooling

After Phase 7 (Optimization):
- Training loops with `train()`
- Backward pass and feedback propagation
- Parameter optimization with LLM-based updates
- Complete end-to-end learning examples
