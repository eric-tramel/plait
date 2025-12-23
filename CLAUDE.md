# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**inf-engine** is a PyTorch-inspired framework for building, executing, and optimizing LLM inference pipelines. Key features:

- **PyTorch-like API**: `InferenceModule` with `forward()` and `backward()` methods
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

### Local PR-Style Development

We develop locally, simulating a PR process. All development follows the task breakdown in `TASKS.md`. Each task (PR-XXX) represents a single, tested, reviewable increment:

1. **Find the next PR** in `TASKS.md`
2. **Create a feature branch** from `main`:
   ```bash
   git checkout main
   git checkout -b feat/feature-name
   ```
3. **Implement** the feature with tests
4. **Run CI** before committing:
   ```bash
   make ci
   ```
5. **Update CHANGELOG.md** under `[Unreleased]`
6. **Commit** with PR-style message (see Git Workflow below)
7. **Request review**: Present the branch for GO/NOGO approval
8. **On approval**: Rebase onto main and fast-forward merge

### Git Workflow

**Branch Strategy**: We use a **rebase strategy** for integrating changes into `main`. Never merge—always rebase.

**Commit Style**: Before integrating a feature branch, **squash commits** into a single, well-documented commit that represents the complete feature implementation.

#### Feature Branch Workflow

```bash
# 1. Work on feature with as many commits as needed
git add .
git commit -m "wip: initial implementation"
git commit -m "wip: add tests"
git commit -m "wip: fix type errors"

# 2. Before requesting review: squash all commits into one
git rebase -i main
# In the editor: mark all commits except first as 'squash' or 'fixup'

# 3. Write a proper PR-style commit message (see format below)

# 4. Request GO/NOGO approval from reviewer

# 5. On approval: rebase onto main and fast-forward merge
git checkout main
git merge --ff-only feat/feature-name

# 6. Clean up feature branch
git branch -d feat/feature-name
```

#### Commit Message Format

Final squashed commits should use PR-style documentation:

```
feat(module): add Parameter class for learnable values

## Summary
Implement the Parameter class that holds learnable string values
which can be optimized via backward passes.

## Changes
- Add Parameter dataclass with value, requires_grad, and feedback buffer
- Implement accumulate_feedback() for collecting backward pass feedback
- Implement apply_update() for optimizer-driven value updates
- Add zero_feedback() to clear buffer without updating

## Testing
- test_parameter_creation: basic instantiation
- test_parameter_accumulate_feedback: feedback buffer works
- test_parameter_apply_update: value updates, buffer clears

## References
- Design: architecture.md → "Core Components" → "Parameter"
- Task: PR-002 in TASKS.md
```

#### Commit Prefixes

- `feat(scope):` - New feature
- `fix(scope):` - Bug fix
- `refactor(scope):` - Code restructuring without behavior change
- `test(scope):` - Adding or updating tests
- `docs(scope):` - Documentation only
- `chore(scope):` - Build, tooling, or maintenance

### Review Requirements

Every feature branch must meet these criteria before GO approval:
- [ ] Include implementation code
- [ ] Include unit tests (100% coverage of new code)
- [ ] Include integration tests where applicable
- [ ] Update `CHANGELOG.md`
- [ ] Pass `make ci`
- [ ] Include usage examples in docstrings or tests
- [ ] Single squashed commit with PR-style message
- [ ] Clean rebase onto current main

### After Making Changes

**IMPORTANT**: Always run `make ci` after implementing new features or making changes. This ensures all linting, type checking, and tests pass before committing.

## Project Structure

```
inf-engine/
├── src/
│   └── inf_engine/          # Main package (to be implemented)
│       ├── module.py        # InferenceModule, LLMInference
│       ├── parameter.py     # Parameter class
│       ├── graph.py         # InferenceGraph, GraphNode
│       ├── tracing/         # Tracer, Proxy, context
│       ├── execution/       # Scheduler, ExecutionState
│       ├── resources/       # ResourceManager, config
│       ├── optimization/    # Loss, Optimizer, backward
│       └── clients/         # LLM client implementations
├── tests/
│   ├── unit/                # Fast, isolated unit tests
│   └── integration/         # Component interaction tests
├── design_docs/             # Architecture documentation
│   ├── README.md            # Documentation index
│   ├── architecture.md      # High-level system design
│   ├── inference_module.md  # Core module system
│   ├── tracing.md           # DAG capture
│   ├── execution.md         # Scheduler and state
│   ├── resources.md         # Endpoint configuration
│   ├── optimization.md      # Backward pass and learning
│   └── development_plan.md  # Implementation phases
├── TASKS.md                 # PR-by-PR implementation breakdown
├── CHANGELOG.md             # Version history
├── Makefile                 # Build targets
├── pyproject.toml           # Project configuration
└── main.py                  # Entry point
```

## Architecture Overview

The system has four main layers:

1. **User Code**: `InferenceModule`, `LLMInference`, `Parameter`
2. **Tracing**: `Tracer`, `Proxy`, `InferenceGraph` - captures DAG from forward()
3. **Execution**: `Scheduler`, `ExecutionState` - async execution with priority queue
4. **Infrastructure**: LLM clients, rate limiting, checkpointing

Key design principle: **Separation of concerns** - module definitions are independent of resource configuration. Modules use aliases; ResourceManager binds to actual endpoints.

## Key Files Reference

| File | Purpose |
|------|---------|
| `TASKS.md` | PR-by-PR implementation tasks |
| `CHANGELOG.md` | Version history and changes |
| `design_docs/architecture.md` | System architecture |
| `design_docs/development_plan.md` | Implementation phases |
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
    module: InferenceModule,
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
