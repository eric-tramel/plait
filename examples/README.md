# inf-engine Examples

This directory contains practical examples demonstrating the inf-engine API.

## Current Examples (Phase 1)

These examples work with the current Phase 1 implementation:

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

**Note:** These pipelines cannot be executed yet. Execution requires
the tracing and scheduling infrastructure from Phase 2-3. This example
demonstrates the API and inspects module structure.

```bash
uv run python examples/03_llm_pipelines.py
```

## Running Examples

From the repository root:

```bash
# Run a specific example
uv run python examples/01_basic_modules.py

# Or run all examples
for f in examples/*.py; do uv run python "$f"; echo; done
```

## Coming Soon

After Phase 2-3 implementation:
- Full LLM pipeline execution with `run()`
- Tracing and DAG visualization
- Async parallel execution

After Phase 7 implementation:
- Training loops with `train()`
- Backward pass and optimization
- Complete end-to-end examples
