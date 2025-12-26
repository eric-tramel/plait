# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding and development tooling
- Design documentation in `design_docs/`
- Development plan and task breakdown
- Initial package structure with `py.typed` marker (PEP 561)
- Parameter class for learnable string values that can be optimized via backward passes
- InferenceModule base class with automatic child/parameter registration
- InferenceModule introspection methods: `children()`, `modules()`, `parameters()`, and `named_*` variants
- InferenceModule `forward()` abstract method and `__call__()` delegation
- LLMInference atomic module for LLM API calls
- Module composition integration tests
- Practical examples in `examples/` demonstrating modules, parameters, and LLM pipelines
- Trace context infrastructure for DAG capture using ContextVar
- Proxy class for symbolic tracing
- GraphNode and InferenceGraph data structures for representing traced execution graphs
- Topological ordering method on InferenceGraph for valid execution order
- Graph traversal methods (ancestors, descendants) on InferenceGraph
- Tracer class foundation with node storage and ID generation
- Input node creation in Tracer for capturing graph entry points
- `record_call()` method to Tracer for capturing module invocations
- `trace()` method to Tracer for executing forward() and returning InferenceGraph
- InferenceModule integration with tracing system via `__call__` method
- Tracing integration tests for nested modules, shared inputs, and dict outputs
- Task and TaskResult types for execution state management
- ExecutionState initialization with graph analysis and dependency tracking
- ExecutionState task management methods: `get_next_task()`, `mark_complete()`, `is_complete()`
- ExecutionState failure handling with `mark_failed()` and descendant cancellation
- ExecutionState `requeue()` method for retrying tasks with descendant dropping
- ExecutionState `get_outputs()` method for retrieving final output values
- Scheduler class with concurrency control via semaphore
- Scheduler `execute()` method for running tasks with TaskGroup and dependency management
- `run()` function for tracing and executing modules end-to-end
- Execution integration tests for linear, parallel, and diamond graph patterns
- Proxy data access operations: `__getitem__`, `__iter__`, `keys()`, `values()`, `items()`
- Tracer methods for data access: `record_getitem()`, `record_iter()`, `record_method()`
- Operation classes: `GetItemOp`, `IterOp`, `MethodOp` for representing data access in graphs
- Preserve user-defined output keys when forward() returns a dict, list, or nested structure
- Add `NodeRef` type for type-safe node references in args and kwargs
- Add cycle detection to `InferenceGraph.topological_order()` with clear error message
- Add `task_ready_event` to ExecutionState for event-driven scheduler coordination
- Add `state_dict()` and `load_state_dict()` to InferenceModule for parameter serialization
- Add `visualize_graph()` function for DOT format graph output
- Add `EndpointConfig` dataclass for LLM endpoint configuration
- Add `ResourceConfig` dataclass for managing multiple LLM endpoints
- Add shared `inf_engine.types` module for core types used across packages
  - `LLMRequest`: prompt, system_prompt, temperature, max_tokens, tools, tool_choice, extra_body
  - `LLMResponse`: content, tokens, finish_reason, reasoning, tool_calls, timing metrics
- Add `LLMClient` abstract base class for LLM provider clients
  - Defines async `complete(request: LLMRequest) -> LLMResponse` interface
  - Provides unified contract for OpenAI, Anthropic, vLLM, and other providers
- Add `OpenAIClient` implementation for OpenAI API
  - Async completion with message formatting and tool call support
  - Rate limit handling with `RateLimitError` and retry-after extraction
  - Configurable base URL, API key, and timeout
- Add `OpenAICompatibleClient` for self-hosted models (vLLM, TGI, Ollama)
  - Inherits from `OpenAIClient` with simplified configuration
  - Required `base_url` parameter for custom endpoints
  - Default `api_key="not-needed"` for local deployments
- Add `ResourceManager` initialization for endpoint coordination
  - Creates LLM clients based on provider_api configuration
  - Creates per-endpoint semaphores for concurrency control
  - Supports openai and vllm providers
- Add ResourceManager integration to Scheduler for LLM execution
  - Scheduler accepts optional `resource_manager` parameter
  - LLMInference modules are executed through ResourceManager's clients
  - Per-endpoint semaphores provide concurrency control for LLM calls
  - `_build_llm_request()` helper builds LLMRequest from module config and args
- Add custom error types for error handling and recovery
  - `InfEngineError`: Base exception for all inf-engine errors
  - `RateLimitError`: Raised when API rate limits are hit, with optional `retry_after`
  - `ExecutionError`: Raised on task execution failures, with optional `node_id` and `cause`
- Add `RateLimiter` with token bucket algorithm for endpoint rate control
  - Configurable RPM (requests per minute) and max_tokens (burst capacity)
  - Async `acquire()` method waits for available tokens
  - Thread-safe implementation using asyncio.Lock
- Add adaptive backoff to `RateLimiter` for automatic rate adjustment
  - `backoff()` method reduces rate when hitting API limits
  - `recover()` method gradually restores rate after successful requests
  - Configurable `min_rpm`, `recovery_factor`, and `backoff_factor`
  - Supports server-provided `retry_after` hints for optimal rate adjustment
- Add RateLimiter integration to ResourceManager
  - `rate_limiters` dict holds per-endpoint rate limiters
  - `get_rate_limiter()` method retrieves rate limiter for an alias
  - Rate limiters created automatically when `EndpointConfig.rate_limit` is set

### Changed
- Replace scheduler busy-wait polling with `asyncio.Event` signaling for efficient task-ready notifications
- Standardize rate limiting units to RPM (requests per minute) across all APIs
  - `RateLimiter` now accepts `rpm` parameter instead of `rate`
  - `EndpointConfig.rate_limit` is now documented as requests per minute
  - Aligns with LLM API provider conventions (OpenAI, Anthropic, etc.)

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- Standardize priority ordering convention to "lower value = higher precedence" across GraphNode and Task

### Security
- N/A

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
