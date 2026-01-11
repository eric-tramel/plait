# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**plait** is a PyTorch-inspired framework for building, executing, and optimizing LLM inference pipelines. Key features:

- **PyTorch-like API**: `Module` with `forward()` and `backward()` methods
- **Automatic DAG capture**: Trace-based graph construction from eager-mode code
- **Async execution**: Maximum throughput with adaptive backpressure
- **LLM-based optimization**: Backward passes that propagate feedback to improve prompts

See `design_docs/` for comprehensive architecture documentation.

## Development Environment

- **Python Version**: 3.13 (specified in `.python-version`)
- **Package Manager**: `uv` for dependency management
- **Virtual Environment**: `.venv` directory (excluded from git)

## Common Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Activate virtual environment (if not using uv run)
source .venv/bin/activate
```

### Quality Checks
```bash
# Run all CI checks (lint, types, test) - ALWAYS run after making changes
make ci

# Individual targets
make lint            # Format and lint with ruff
make types           # Type check with ty
make test            # Run all pytest tests
make test-unit       # Run unit tests only
make test-integration # Run integration tests only
```

### Development Tasks
```bash
# Install new dependencies
uv add <package-name>

# Install development dependencies
uv add --dev <package-name>
```

## Development Workflow

### Git Workflow

**Branch Strategy**: We use a **rebase strategy** for integrating changes into `main`. Never merge—always rebase.

**Commit Style**: Before integrating a feature branch, **squash commits** into a single, well-documented commit that represents the complete feature implementation.

#### Commit Message Format

Final squashed commits should use PR-style documentation:

```
feat(module): add Parameter class for learnable values

## Summary
Implement the Parameter class that holds learnable string values
which can be optimized via backward passes.
```

#### Commit Prefixes

- `feat(scope):` - New feature
- `fix(scope):` - Bug fix
- `refactor(scope):` - Code restructuring without behavior change
- `test(scope):` - Adding or updating tests
- `docs(scope):` - Documentation only
- `chore(scope):` - Build, tooling, or maintenance

### After Making Changes

**IMPORTANT**: Always run `make ci` after implementing new features or making code changes. This ensures all linting, type checking, and tests pass before committing.

## Project Structure

```
plait/
├── src/
│   └── plait/          # Main package
│       ├── __init__.py      # Package exports
│       ├── module.py        # Module, LLMInference
│       ├── parameter.py     # Parameter class
│       ├── values.py        # Value, ValueRef for semantic values
│       ├── functional.py    # Functional API (llm_call, structured_llm_call)
│       ├── graph.py         # InferenceGraph, GraphNode
│       ├── types.py         # Core type definitions
│       ├── errors.py        # Exception hierarchy
│       ├── tracing/         # Tracer, Proxy, context
│       ├── execution/       # Scheduler, Executor, ExecutionState
│       ├── resources/       # ResourceManager, config, rate limiting
│       ├── optimization/    # Loss, Optimizer, backward, feedback
│       ├── profiling/       # Profiler for execution analysis
│       └── clients/         # LLM client implementations
├── tests/
│   ├── unit/                    # Fast, isolated unit tests
│   └── integration/             # Component interaction tests
├── design_docs/                 # Architecture documentation
│   ├── README.md                    # Documentation index
│   ├── DESIGN.md                    # High-level design overview
│   ├── REVISIONS.md                 # Design revision history
│   ├── architecture.md              # System architecture
│   ├── inference_module.md          # Core module system
│   ├── tracing.md                   # DAG capture
│   ├── execution.md                 # Scheduler and state
│   ├── resources.md                 # Endpoint configuration
│   ├── optimization.md              # Backward pass and learning
│   ├── profiling.md                 # Execution profiling
│   ├── parameters.md                # Parameter system
│   ├── values.md                    # Values and references
│   ├── ordered_parameter_updates.md # Update ordering
│   └── functional_api.md            # Functional API design
├── examples/                    # Usage examples (01-09)
├── cookbooks/                   # Tutorials and recipes
│   └── hallucination_detection.py
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI workflow
├── CHANGELOG.md                 # Version history
├── README.md                    # Project overview
├── Makefile                     # Build targets
├── pyproject.toml               # Project configuration
├── uv.lock                      # Dependency lock file
└── main.py                      # Entry point
```

## Architecture Overview

The system has four main layers:

1. **User Code**: `Module`, `LLMInference`, `Parameter`
2. **Tracing**: `Tracer`, `Proxy`, `InferenceGraph` - captures DAG from forward()
3. **Execution**: `Scheduler`, `ExecutionState` - async execution with priority queue
4. **Infrastructure**: LLM clients, rate limiting, checkpointing

Key design principle: **Separation of concerns** - module definitions are independent of resource configuration. Modules use aliases; ResourceManager binds to actual endpoints.

## Key Files Reference

| File | Purpose |
|------|---------|
| `CHANGELOG.md` | Version history and changes |
| `design_docs/architecture.md` | System architecture |
| `Makefile` | Build and test targets |
| `pyproject.toml` | Dependencies and tool config |

## Code Style

- **Formatting**: ruff (line length 88)
- **Type hints**: Required on all functions, methods, and class attributes; checked with ty
- **Tests**: pytest with strict markers

### Documentation Requirements

All public functions, methods, and classes **must** have Google-style docstrings that include:

1. **One-line summary**: Brief description of what it does
2. **Args**: All parameters with types and descriptions
3. **Returns**: Return type and description (if not None)
4. **Raises**: Any exceptions that may be raised
5. **Usage/side effects**: Note any important behavior, state changes, or side effects

**Avoid** heavy inline comments. Code should be self-documenting through clear naming. Use inline comments only for non-obvious logic.

#### Docstring Example

```python
def record_call(
    self,
    module: Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Proxy:
    """Record a module invocation and return a proxy for its output.

    Creates a new graph node representing this call and tracks dependencies
    based on any Proxy objects in the arguments.

    Args:
        module: The module being called.
        args: Positional arguments passed to the module.
        kwargs: Keyword arguments passed to the module.

    Returns:
        A Proxy representing the eventual output of this call.

    Raises:
        TracingError: If called outside of an active trace context.

    Note:
        This method mutates the tracer's internal node registry.
    """
```

#### Type Annotation Requirements

- All function parameters must have type annotations
- All return types must be annotated (use `-> None` explicitly)
- Use `typing` module constructs where needed (`Any`, `TypeVar`, `Generic`, etc.)
- Prefer concrete types over `Any` when possible
- Use `| None` instead of `Optional` (Python 3.10+ style)

## Testing Strategy

| Category | Location | Purpose |
|----------|----------|---------|
| Unit | `tests/unit/` | Fast, isolated tests |
| Integration | `tests/integration/` | Component interaction |

Mock LLM responses in tests rather than making real API calls.
