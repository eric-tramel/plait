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
- Add RateLimitError handling to Scheduler
  - Catch `RateLimitError` in `_execute_task` and trigger task requeue
  - Call `backoff()` on the endpoint's rate limiter when rate limits are hit
  - Tasks are automatically retried without being marked as failed
  - `retry_after` value from the error is passed to the rate limiter for optimal backoff
- Add `Checkpoint` dataclass for execution state persistence
  - `execution_id`, `timestamp`, `completed_nodes`, `failed_nodes`, `pending_nodes` attributes
  - `graph_hash` field for detecting incompatible checkpoints when pipeline changes
  - `save(path)` method for JSON serialization to disk
  - `load(path)` classmethod for deserializing from disk
  - `is_compatible_with(hash)` method for verifying checkpoint compatibility
  - Enables progress recovery for long-running pipelines
- Add `InferenceGraph.compute_hash()` for deterministic graph fingerprinting
  - Merkle-tree style hash based on module types, configurations, and dependencies
  - Independent of node IDs - same logical structure produces same hash
  - Enables checkpoint validation across different Python sessions
- Add `CheckpointManager` for buffered checkpoint writes during execution
  - Configurable `buffer_size` and `flush_interval` for efficient disk I/O
  - Per-execution buffers for parallel run tracking
  - `record_completion()` for buffering task results
  - `flush()` and `flush_all()` for persisting buffered completions
  - `get_checkpoint()` for retrieving existing checkpoints
  - `set_graph_hash()` for storing graph hash in checkpoints
- Add `checkpoint_dir` and `execution_id` parameters to `run()` function
  - Enables automatic progress checkpointing during execution
  - Auto-generates UUID execution_id if not provided
  - Creates checkpoint directory if it doesn't exist
  - Stores graph hash for checkpoint compatibility checking
- Add checkpointing example in `examples/06_checkpointing.py`
- Add `ExecutionSettings` context manager for shared execution configuration
  - Dataclass with `resources`, `checkpoint_dir`, `max_concurrent`, `scheduler`, and callback fields
  - Both sync (`with`) and async (`async with`) context manager protocols
  - `get_execution_settings()` function for accessing current settings
  - Nested context support with proper restoration and value inheritance
  - Automatic `CheckpointManager` creation when `checkpoint_dir` is set
  - Profile configuration fields reserved for future profiling integration
- Integrate ExecutionSettings with InferenceModule for direct module execution
  - Add `bind(resources, max_concurrent, **kwargs)` method for binding resources to modules
  - Update `__call__` to check for resources from both bound config and ExecutionSettings context
  - Add `_execute_bound()` async method for tracing and executing bound modules
  - Configuration priority order: call-time kwargs > bound settings > context settings > defaults
  - Add `resources` parameter to `run()` function for passing ResourceConfig/ResourceManager
  - Support batch execution: `await module([input1, input2, ...])` returns list of results
- Concurrent batch execution: batch inputs now run in parallel using `asyncio.gather()` for maximum throughput
- Add `run_sync()` method for synchronous blocking execution in scripts and notebooks
  - Blocks until execution completes and returns the result
  - Supports both single and batch inputs
  - Raises `RuntimeError` if called from async context or without bound resources
- Add streaming execution with BatchResult, progress tracking, and cancellation
  - New `BatchResult` type wraps streaming results with index, input, output, and error fields
  - `BatchResult.ok` property for easy success/failure checking
  - `ExecutionSettings.streaming=True` enables async iteration over batch results
  - `ExecutionSettings.preserve_order` controls result ordering (completion vs input order)
  - `ExecutionSettings.on_progress` callback tracks batch progress (completed, total)
  - `InferenceModule._stream_batch()` method for internal streaming implementation
  - Cancellation support: breaking from streaming loop cancels all pending tasks
  - Progress callbacks work in both streaming and non-streaming batch modes
- Add resource management integration tests (`tests/integration/test_resources.py`)
  - Multiple endpoint alias tests with different configurations
  - Concurrent request handling with semaphore limits
  - ResourceManager and Scheduler integration tests
  - ExecutionSettings with resource context tests
  - Batch execution across multiple endpoints
  - End-to-end pipeline tests with mocked LLM endpoints
  - Configuration example validation from design docs
- Update examples documentation with best practices for resource configuration
- Add `ResourceMetrics` for endpoint observability
  - `EndpointMetrics` dataclass tracks per-endpoint request counts, latency, and tokens
  - Thread-safe `ResourceMetrics` class aggregates metrics across endpoints
  - `record_success()`, `record_error()`, `record_rate_limit()` for recording outcomes
  - `get_alias_stats()` and `get_all_stats()` for retrieving metrics
  - `estimate_cost()` for calculating costs based on token usage and endpoint pricing
  - `reset()` and `get_endpoint_metrics()` for testing and detailed access
- Integrate `ResourceMetrics` with `ResourceManager`
  - `metrics` attribute automatically created during initialization
  - `get_stats()` method returns availability and metrics for all endpoints
- Add task timeout and retry handling with TransientError
  - `TransientError` for retryable failures (connection timeouts, server errors)
  - `ExecutionSettings.task_timeout` sets maximum seconds per task
  - `ExecutionSettings.max_task_retries` controls retry attempts for transient failures
  - `ExecutionSettings.task_retry_delay` sets base delay with exponential backoff
  - Scheduler uses `asyncio.timeout()` for task timeout enforcement
  - Timed-out tasks are marked as failed with descriptive error message
  - TransientError triggers automatic retry with exponential backoff (delay doubles each retry)
  - Getter methods on ExecutionSettings: `get_task_timeout()`, `get_max_task_retries()`, `get_task_retry_delay()`

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
