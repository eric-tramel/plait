# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Formatting Rules

1. **Single-line entries only**: Each changelog entry must be a single line. No multi-line descriptions or nested bullet points.
2. **PR links required**: All entries must begin with a GitHub PR link in the format `[#123](https://github.com/eric-tramel/plait/pull/123)`.

## [Unreleased]

### Added
- Add `AnthropicClient` for Anthropic Claude models with full support for tools, extended thinking, and rate limit handling

### Changed
- [#13](https://github.com/eric-tramel/plait/pull/13) Consolidate unit tests from 2187 to 1908 tests (12.7% reduction) while maintaining 93% coverage

### Fixed
- [#13](https://github.com/eric-tramel/plait/pull/13) Fix scheduler race condition that caused 5-second delays when tasks completed while waiting for semaphore (10x test speedup: 33s → 3.4s)

### Added
- [#14](https://github.com/eric-tramel/plait/pull/14) Consolidate examples from 9 files to 5 focused single-concept files with `make example` runner
- [#12](https://github.com/eric-tramel/plait/pull/12) Export core classes (`Module`, `LLMInference`, `Parameter`, `ExecutionSettings`) from package root for cleaner imports
- [#11](https://github.com/eric-tramel/plait/pull/11) Add comprehensive README with project overview, quick start, examples, and development setup
- Integrate `Value.ref` system with loss functions and backward pass for seamless value unwrapping
- Add `OptimizationError` exception with retry logic for `SFAOptimizer` parameter updates
- Execute graphs with `ValueRef` and error-as-value semantics for automatic error propagation
- Refactor tracing to Value-driven capture with `ValueRef` placeholders and dependency discovery
- Add functional API for Value operations with `valueify()`, `unwrap()`, and `collect_refs()` helpers
- Add `Value` container and `ValueRef` for typed data flow with provenance tracking
- Add `Parameter` class for learnable values with stable refs and optional descriptions
- Add topologically-ordered parameter updates with upstream visibility in `SFAOptimizer.step()`
- Add `TracedOutput` for implicit record flow and `train()`/`eval()` mode switching
- Add `Optimizer` abstract base class following torch.optim patterns
- Add `SFAOptimizer` (Stochastic Feedback Ascent) with configurable conservatism
- Add backward pass infrastructure with `BackwardContext`, `BackwardResult`, and `_propagate_backward()`
- Add `Module.backward()` and `LLMInference.backward()` for feedback propagation
- Add `Loss` base class, `VerifierLoss`, `LLMJudge`, and `CompositeLoss` for output evaluation
- Add `Feedback` dataclass with `FeedbackType` enum for representing evaluation results
- Add `ForwardRecord` dataclass and `record` parameter to `run()` for backward pass support
- Add `Module` base class with child/parameter registration and introspection methods
- Add `LLMInference` atomic module for LLM API calls
- Add tracing infrastructure with `Tracer`, `Proxy`, `GraphNode`, and `InferenceGraph`
- Add `ExecutionState` for task management with dependency tracking and failure handling
- Add `Scheduler` with concurrency control and `run()` function for end-to-end execution
- Add data access operations (`__getitem__`, `__iter__`) and `NodeRef` for type-safe references
- Add cycle detection, event-driven scheduling, and `visualize_graph()` for DOT output
- Add `state_dict()` and `load_state_dict()` to `Module` for parameter serialization
- Add `EndpointConfig`, `ResourceConfig`, and `ResourceManager` for LLM endpoint coordination
- Add `LLMRequest`, `LLMResponse` types and `LLMClient` abstract base class
- Add `OpenAIClient` and `OpenAICompatibleClient` for OpenAI API and self-hosted models
- Add custom error types: `InfEngineError`, `RateLimitError`, `ExecutionError`, `TransientError`
- Add `RateLimiter` with token bucket algorithm and adaptive backoff for rate control
- Add `Checkpoint`, `CheckpointManager`, and `InferenceGraph.compute_hash()` for state persistence
- Add `ExecutionSettings` context manager with resource binding and batch execution support
- Add `run_sync()` method for synchronous blocking execution
- Add streaming execution with `BatchResult`, progress tracking, and cancellation support
- Add `ResourceMetrics` for endpoint observability with cost estimation
- Add task timeout and retry handling with exponential backoff
- Add `TraceProfiler` with Chrome Trace Format export for execution profiling
- Add rubric, preference, and ranking loss functions for human and LLM evaluation
- Add project scaffolding, design documentation, and example cookbooks

### Changed
- [#12](https://github.com/eric-tramel/plait/pull/12) Update README examples to use `bind()` and `forward` pattern instead of `run()`, with `OpenAIEndpointConfig` for endpoint configuration
- **BREAKING**: Rename `InferenceModule` to `Module` for simpler PyTorch-like naming (update imports: `from plait.module import Module`)
- Replace scheduler busy-wait polling with `asyncio.Event` signaling
- Standardize rate limiting units to RPM (requests per minute) across all APIs
- Refactor batch loss API to match PyTorch semantics with auto-detection and mean reduction

### Removed
- `Loss.batch()` method (use `loss(outputs_list)` directly)
- `Feedback.backward_batch()` method (use `feedback.backward()` on aggregated feedback)

### Fixed
- Standardize priority ordering convention to "lower value = higher precedence"

---

## Version History

_No releases yet._

---

## Release Process

1. Update version in `pyproject.toml`
2. Move items from `[Unreleased]` to new version section
3. Add release date
4. Create git tag: `git tag -a v0.X.0 -m "Release v0.X.0"`
5. Push tag: `git push origin v0.X.0`
