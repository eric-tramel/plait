# TASKS.md

A line-by-line breakdown of PRs to implement inf-engine, in order.

Each PR represents a single, tested, reviewable increment of functionality.

## Progress

- [x] **Phase 1: Foundation** (7/7)
- [ ] **Phase 2: Tracing** (3/11)
- [ ] **Phase 3: Execution** (0/10)
- [ ] **Phase 4: Resources** (0/10)
- [ ] **Phase 5: Production Features** (0/11)
- [ ] **Phase 6: Branching** (0/8)
- [ ] **Phase 7: Optimization** (0/14)
- [ ] **Post-Implementation** (0/3)

**Total: 10/74 PRs completed**

---

## PR Requirements

Every PR must:
- [ ] Create a feature branch from `main`
- [ ] Include implementation code
- [ ] Include unit tests (100% coverage of new code)
- [ ] Include integration tests where applicable
- [ ] Update `CHANGELOG.md`
- [ ] Pass `make ci`
- [ ] Include usage examples in docstrings or tests

---

## Phase 1: Foundation

### - [x] PR-001: Project scaffolding and package structure
- **Branch**: `feat/project-structure`
- **Description**: Set up the `src/inf_engine/` package structure with `__init__.py` files
- **Design Docs**:
  - `architecture.md` → "File Structure"
  - `development_plan.md` → "1.1.4 Package Structure"
- **Files**:
  - `src/inf_engine/__init__.py`
  - `src/inf_engine/py.typed` (PEP 561 marker)
- **Tests**: Verify package imports correctly
- **CHANGELOG**: "Initial package structure"

### - [x] PR-002: Parameter class
- **Branch**: `feat/parameter`
- **Description**: Implement the `Parameter` class for learnable string values
- **Design Docs**:
  - `architecture.md` → "Core Components" → "3. Parameter"
  - `inference_module.md` → "Parameter Class"
  - `development_plan.md` → "1.1.1 Parameter Class"
- **Files**: `src/inf_engine/parameter.py`
- **Tests**:
  - `tests/unit/test_parameter.py` (creation, str, accumulate, apply, zero)
- **CHANGELOG**: "Add Parameter class for learnable values"

### - [x] PR-003: InferenceModule base class - core
- **Branch**: `feat/inference-module-core`
- **Description**: Implement `InferenceModule` with `__init__`, `__setattr__`, child/parameter registration
- **Design Docs**:
  - `architecture.md` → "Core Components" → "1. InferenceModule"
  - `inference_module.md` → "InferenceModule Base Class"
  - `development_plan.md` → "1.1.2 InferenceModule Base Class"
- **Files**: `src/inf_engine/module.py`
- **Tests**:
  - `tests/unit/test_module.py` (registration, basic instantiation)
- **CHANGELOG**: "Add InferenceModule base class with auto-registration"

### - [x] PR-004: InferenceModule introspection methods
- **Branch**: `feat/inference-module-introspection`
- **Description**: Add `children()`, `modules()`, `parameters()`, `named_*` iterators
- **Design Docs**:
  - `inference_module.md` → "InferenceModule Base Class" (introspection section)
  - `DESIGN.md` → "Task Definition API" → "Op Base Class"
- **Files**: `src/inf_engine/module.py`
- **Tests**:
  - `tests/unit/test_module.py` (all iterator methods)
- **CHANGELOG**: "Add InferenceModule introspection methods"

### - [x] PR-005: InferenceModule forward and call
- **Branch**: `feat/inference-module-forward`
- **Description**: Add `forward()` abstract method and `__call__` delegation
- **Design Docs**:
  - `inference_module.md` → "InferenceModule Base Class" (forward/call)
  - `architecture.md` → "Execution Flow" → "Forward Pass"
- **Files**: `src/inf_engine/module.py`
- **Tests**:
  - `tests/unit/test_module.py` (forward raises, call delegates)
- **CHANGELOG**: "Add InferenceModule forward() and __call__()"

### - [x] PR-006: LLMInference class
- **Branch**: `feat/llm-inference`
- **Description**: Implement `LLMInference` with alias, system_prompt, temperature, max_tokens
- **Design Docs**:
  - `architecture.md` → "Core Components" → "2. LLMInference (Atomic Module)"
  - `inference_module.md` → "LLMInference (Atomic Module)"
  - `development_plan.md` → "1.1.3 LLMInference Stub"
- **Files**: `src/inf_engine/module.py`
- **Tests**:
  - `tests/unit/test_llm_inference.py` (all constructor variations)
- **CHANGELOG**: "Add LLMInference atomic module"

### - [x] PR-007: Module composition integration tests
- **Branch**: `feat/module-composition-tests`
- **Description**: Add integration tests for composing modules together
- **Design Docs**:
  - `inference_module.md` → "Composing Modules" (Sequential, Parallel, Nested)
  - `DESIGN.md` → "Example Usage", "Nested Composition"
- **Files**: `tests/integration/test_module_composition.py`
- **Tests**: Simple, nested, multi-child composition
- **CHANGELOG**: "Add module composition integration tests"

---

## Phase 2: Tracing

### - [x] PR-008: Trace context infrastructure
- **Branch**: `feat/trace-context`
- **Description**: Implement `ContextVar`-based trace context with getter and context manager
- **Design Docs**:
  - `tracing.md` → "Trace Context"
  - `development_plan.md` → "2.1.3 Trace Context"
- **Files**: `src/inf_engine/tracing/context.py`
- **Tests**:
  - `tests/unit/test_trace_context.py` (default, set/get, cleared after)
- **CHANGELOG**: "Add trace context infrastructure"

### - [x] PR-009: Proxy class
- **Branch**: `feat/proxy`
- **Description**: Implement `Proxy` dataclass for symbolic values during tracing
- **Design Docs**:
  - `tracing.md` → "Core Concepts" → "Proxy Objects"
  - `DESIGN.md` → "Task Definition API" → "Proxy Objects"
  - `development_plan.md` → "2.1.1 Proxy Class"
- **Files**: `src/inf_engine/tracing/proxy.py`
- **Tests**:
  - `tests/unit/test_proxy.py` (creation, repr)
- **CHANGELOG**: "Add Proxy class for symbolic tracing"

### - [x] PR-010: GraphNode and InferenceGraph structures
- **Branch**: `feat/graph-structures`
- **Description**: Implement `GraphNode` and `InferenceGraph` dataclasses
- **Design Docs**:
  - `architecture.md` → "Core Components" → "4. InferenceGraph"
  - `tracing.md` → "Core Concepts" → "Graph Nodes", "Inference Graph"
  - `development_plan.md` → "2.1.2 Graph Structures"
- **Files**: `src/inf_engine/graph.py`
- **Tests**:
  - `tests/unit/test_graph.py` (creation, basic operations)
- **CHANGELOG**: "Add GraphNode and InferenceGraph data structures"

### - [ ] PR-011: InferenceGraph topological ordering
- **Branch**: `feat/graph-toposort`
- **Description**: Implement `topological_order()` method on InferenceGraph
- **Design Docs**:
  - `tracing.md` → "Inference Graph" (topological_order method)
  - `execution.md` → "Execution State" (uses toposort for scheduling)
- **Files**: `src/inf_engine/graph.py`
- **Tests**:
  - `tests/unit/test_graph.py` (linear, diamond, complex graphs)
- **CHANGELOG**: "Add topological ordering to InferenceGraph"

### - [ ] PR-012: InferenceGraph ancestors and descendants
- **Branch**: `feat/graph-traversal`
- **Description**: Implement `ancestors()` and `descendants()` methods
- **Design Docs**:
  - `tracing.md` → "Inference Graph" (ancestors, descendants methods)
  - `execution.md` → "ExecutionState" (uses descendants for failure cascading)
- **Files**: `src/inf_engine/graph.py`
- **Tests**:
  - `tests/unit/test_graph.py` (ancestors, descendants for various graphs)
- **CHANGELOG**: "Add graph traversal methods (ancestors, descendants)"

### - [ ] PR-013: Tracer - basic structure
- **Branch**: `feat/tracer-basic`
- **Description**: Implement `Tracer` class with node storage and ID generation
- **Design Docs**:
  - `tracing.md` → "The Tracer"
  - `DESIGN.md` → "Task Definition API" → "Tracer"
  - `development_plan.md` → "2.1.4 Tracer"
- **Files**: `src/inf_engine/tracing/tracer.py`
- **Tests**:
  - `tests/unit/test_tracer.py` (instantiation, ID generation)
- **CHANGELOG**: "Add Tracer class foundation"

### - [ ] PR-014: Tracer - input node creation
- **Branch**: `feat/tracer-inputs`
- **Description**: Implement input proxy creation in Tracer
- **Design Docs**:
  - `tracing.md` → "The Tracer" (_create_input_node method)
- **Files**: `src/inf_engine/tracing/tracer.py`
- **Tests**:
  - `tests/unit/test_tracer.py` (input nodes created correctly)
- **CHANGELOG**: "Add input node creation to Tracer"

### - [ ] PR-015: Tracer - record_call method
- **Branch**: `feat/tracer-record-call`
- **Description**: Implement `record_call()` to capture module invocations
- **Design Docs**:
  - `tracing.md` → "The Tracer" (record_call method)
  - `DESIGN.md` → "Tracer" (record method)
- **Files**: `src/inf_engine/tracing/tracer.py`
- **Tests**:
  - `tests/unit/test_tracer.py` (nodes created, dependencies captured)
- **CHANGELOG**: "Add record_call() to Tracer"

### - [ ] PR-016: Tracer - trace method
- **Branch**: `feat/tracer-trace`
- **Description**: Implement `trace()` method that runs forward() and returns InferenceGraph
- **Design Docs**:
  - `tracing.md` → "The Tracer" (trace method)
  - `tracing.md` → "Example: Complete Tracing Flow"
- **Files**: `src/inf_engine/tracing/tracer.py`
- **Tests**:
  - `tests/unit/test_tracer.py` (full trace of simple module)
- **CHANGELOG**: "Add trace() method to Tracer"

### - [ ] PR-017: Connect InferenceModule.__call__ to Tracer
- **Branch**: `feat/module-trace-integration`
- **Description**: Update `InferenceModule.__call__` to check trace context and record calls
- **Design Docs**:
  - `inference_module.md` → "InferenceModule Base Class" (__call__ method)
  - `tracing.md` → "Trace Context"
  - `development_plan.md` → "2.1.5 Update InferenceModule.__call__"
- **Files**: `src/inf_engine/module.py`
- **Tests**:
  - `tests/unit/test_module.py` (call behavior with/without trace context)
- **CHANGELOG**: "Integrate InferenceModule with tracing system"

### - [ ] PR-018: Tracing integration tests
- **Branch**: `feat/tracing-integration-tests`
- **Description**: Add integration tests for full tracing scenarios
- **Design Docs**:
  - `tracing.md` → "Example: Complete Tracing Flow"
  - `tracing.md` → "Best Practices"
- **Files**: `tests/integration/test_tracing.py`
- **Tests**: Nested modules, shared inputs, dict outputs
- **CHANGELOG**: "Add tracing integration tests"

---

## Phase 3: Execution

### - [ ] PR-019: Task and TaskResult dataclasses
- **Branch**: `feat/task-types`
- **Description**: Implement `Task`, `TaskResult`, and `TaskStatus` enum
- **Design Docs**:
  - `execution.md` → "Execution State" (Task, TaskResult, TaskStatus)
  - `development_plan.md` → "3.1.1 Execution State"
- **Files**: `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_execution_types.py` (creation, comparison)
- **CHANGELOG**: "Add Task and TaskResult types"

### - [ ] PR-020: ExecutionState - initialization
- **Branch**: `feat/execution-state-init`
- **Description**: Implement `ExecutionState.__init__` with graph analysis
- **Design Docs**:
  - `architecture.md` → "Core Components" → "5. ExecutionState"
  - `execution.md` → "Execution State" (ExecutionState class)
- **Files**: `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_execution_state.py` (init, ready nodes identified)
- **CHANGELOG**: "Add ExecutionState initialization"

### - [ ] PR-021: ExecutionState - task management
- **Branch**: `feat/execution-state-tasks`
- **Description**: Implement `get_next_task()`, `mark_complete()`, `is_complete()`
- **Design Docs**:
  - `execution.md` → "Execution State" (task management methods)
  - `architecture.md` → "Execution Flow" → "Forward Pass"
- **Files**: `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_execution_state.py` (task lifecycle)
- **CHANGELOG**: "Add ExecutionState task management methods"

### - [ ] PR-022: ExecutionState - failure handling
- **Branch**: `feat/execution-state-failure`
- **Description**: Implement `mark_failed()` with descendant cancellation
- **Design Docs**:
  - `execution.md` → "Execution State" (mark_failed method)
  - `execution.md` → "Error Handling Policies"
- **Files**: `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_execution_state.py` (failure cascades to descendants)
- **CHANGELOG**: "Add ExecutionState failure handling"

### - [ ] PR-023: ExecutionState - requeue
- **Branch**: `feat/execution-state-requeue`
- **Description**: Implement `requeue()` for retrying tasks
- **Design Docs**:
  - `execution.md` → "Execution State" (requeue method)
  - `DESIGN.md` → "Adaptive Backpressure" (requeue on rate limit)
- **Files**: `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_execution_state.py` (requeue drops descendants)
- **CHANGELOG**: "Add ExecutionState requeue functionality"

### - [ ] PR-024: ExecutionState - get_outputs
- **Branch**: `feat/execution-state-outputs`
- **Description**: Implement `get_outputs()` to retrieve final results
- **Design Docs**:
  - `execution.md` → "Execution State" (get_outputs method)
- **Files**: `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_execution_state.py` (outputs collected correctly)
- **CHANGELOG**: "Add ExecutionState output retrieval"

### - [ ] PR-025: Scheduler - basic implementation
- **Branch**: `feat/scheduler-basic`
- **Description**: Implement `Scheduler` with concurrency limiting
- **Design Docs**:
  - `execution.md` → "Scheduler"
  - `DESIGN.md` → "Architecture" → "3. Priority Queue Scheduler"
  - `development_plan.md` → "3.1.2 Scheduler"
- **Files**: `src/inf_engine/execution/scheduler.py`
- **Tests**:
  - `tests/unit/test_scheduler.py` (instantiation, semaphore behavior)
- **CHANGELOG**: "Add Scheduler with concurrency control"

### - [ ] PR-026: Scheduler - execute method
- **Branch**: `feat/scheduler-execute`
- **Description**: Implement `Scheduler.execute()` with TaskGroup
- **Design Docs**:
  - `execution.md` → "Scheduler" (execute method)
  - `DESIGN.md` → "Architecture" → "5. Execution Loop"
- **Files**: `src/inf_engine/execution/scheduler.py`
- **Tests**:
  - `tests/unit/test_scheduler.py` (executes all tasks, respects dependencies)
- **CHANGELOG**: "Add Scheduler.execute() method"

### - [ ] PR-027: Basic executor and run() function
- **Branch**: `feat/run-function`
- **Description**: Implement `run()` function that traces and executes
- **Design Docs**:
  - `execution.md` → "The run() Function"
  - `DESIGN.md` → "Integration with Executor"
  - `development_plan.md` → "3.1.3 Basic Executor"
- **Files**: `src/inf_engine/execution/executor.py`
- **Tests**:
  - `tests/unit/test_executor.py` (run with mock modules)
- **CHANGELOG**: "Add run() function for module execution"

### - [ ] PR-028: Execution integration tests
- **Branch**: `feat/execution-integration-tests`
- **Description**: Add integration tests for full execution scenarios
- **Design Docs**:
  - `execution.md` → "Example: Complete Execution Flow"
  - `architecture.md` → "Execution Flow" → "Forward Pass"
- **Files**: `tests/integration/test_execution.py`
- **Tests**: Linear, parallel, diamond graphs
- **CHANGELOG**: "Add execution integration tests"

---

## Phase 4: Resources

### - [ ] PR-029: EndpointConfig dataclass
- **Branch**: `feat/endpoint-config`
- **Description**: Implement `EndpointConfig` with all endpoint settings
- **Design Docs**:
  - `resources.md` → "Resource Configuration" → "Configuration Structure"
  - `development_plan.md` → "4.1.1 Resource Configuration"
- **Files**: `src/inf_engine/resources/config.py`
- **Tests**:
  - `tests/unit/test_resource_config.py` (creation, defaults)
- **CHANGELOG**: "Add EndpointConfig dataclass"

### - [ ] PR-030: ResourceConfig dataclass
- **Branch**: `feat/resource-config`
- **Description**: Implement `ResourceConfig` container for multiple endpoints
- **Design Docs**:
  - `resources.md` → "Resource Configuration" → "Configuration Structure"
  - `resources.md` → "Resource Configuration" → "Configuration Examples"
- **Files**: `src/inf_engine/resources/config.py`
- **Tests**:
  - `tests/unit/test_resource_config.py` (multiple endpoints, access)
- **CHANGELOG**: "Add ResourceConfig dataclass"

### - [ ] PR-031: LLMRequest and LLMResponse types
- **Branch**: `feat/llm-types`
- **Description**: Implement request/response dataclasses for LLM calls
- **Design Docs**:
  - `resources.md` → "LLM Clients" (LLMRequest, LLMResponse)
  - `development_plan.md` → "4.1.2 LLM Request/Response"
- **Files**: `src/inf_engine/resources/types.py`
- **Tests**:
  - `tests/unit/test_llm_types.py` (creation, serialization)
- **CHANGELOG**: "Add LLMRequest and LLMResponse types"

### - [ ] PR-032: LLMClient abstract base class
- **Branch**: `feat/llm-client-base`
- **Description**: Implement abstract `LLMClient` interface
- **Design Docs**:
  - `resources.md` → "LLM Clients" (LLMClient ABC)
  - `development_plan.md` → "4.1.3 LLM Clients"
- **Files**: `src/inf_engine/clients/base.py`
- **Tests**:
  - `tests/unit/test_llm_client.py` (interface validation)
- **CHANGELOG**: "Add LLMClient abstract base class"

### - [ ] PR-033: OpenAI client implementation
- **Branch**: `feat/openai-client`
- **Description**: Implement `OpenAIClient` with async completion
- **Design Docs**:
  - `resources.md` → "LLM Clients" (OpenAIClient)
- **Files**: `src/inf_engine/clients/openai.py`
- **Tests**:
  - `tests/unit/test_openai_client.py` (mocked API calls)
- **CHANGELOG**: "Add OpenAI client implementation"

### - [ ] PR-034: OpenAI-compatible client
- **Branch**: `feat/openai-compatible-client`
- **Description**: Implement client for vLLM/TGI-style endpoints
- **Design Docs**:
  - `resources.md` → "LLM Clients" (OpenAICompatibleClient)
  - `DESIGN.md` → "Architecture" → "4. Pipeline Parallelism for Self-Hosted LLMs"
- **Files**: `src/inf_engine/clients/openai.py`
- **Tests**:
  - `tests/unit/test_openai_client.py` (custom base_url handling)
- **CHANGELOG**: "Add OpenAI-compatible client for self-hosted models"

### - [ ] PR-035: ResourceManager - initialization
- **Branch**: `feat/resource-manager-init`
- **Description**: Implement `ResourceManager.__init__` with client/semaphore creation
- **Design Docs**:
  - `architecture.md` → "Core Components" → "6. ResourceManager"
  - `resources.md` → "Resource Manager"
  - `development_plan.md` → "4.1.4 Resource Manager"
- **Files**: `src/inf_engine/resources/manager.py`
- **Tests**:
  - `tests/unit/test_resource_manager.py` (init, clients created)
- **CHANGELOG**: "Add ResourceManager initialization"

### - [ ] PR-036: ResourceManager - execute method
- **Branch**: `feat/resource-manager-execute`
- **Description**: Implement `ResourceManager.execute()` with semaphore handling
- **Design Docs**:
  - `resources.md` → "Resource Manager" (execute method)
  - `resources.md` → "Module-Resource Binding"
- **Files**: `src/inf_engine/resources/manager.py`
- **Tests**:
  - `tests/unit/test_resource_manager.py` (execute with mocked client)
- **CHANGELOG**: "Add ResourceManager.execute() method"

### - [ ] PR-037: Update run() with resources parameter
- **Branch**: `feat/run-with-resources`
- **Description**: Add `resources` parameter to `run()` function
- **Design Docs**:
  - `execution.md` → "The run() Function" (resources parameter)
  - `development_plan.md` → "4.1.5 Update run() Function"
- **Files**: `src/inf_engine/execution/executor.py`
- **Tests**:
  - `tests/unit/test_executor.py` (run with ResourceManager)
- **CHANGELOG**: "Add resources parameter to run()"

### - [ ] PR-038: Resource integration tests
- **Branch**: `feat/resource-integration-tests`
- **Description**: Add integration tests with mocked LLM endpoints
- **Design Docs**:
  - `resources.md` → "Resource Configuration" → "Configuration Examples"
  - `development_plan.md` → "4.3 Integration Tests"
- **Files**: `tests/integration/test_resources.py`
- **Tests**: Multiple aliases, concurrent requests
- **CHANGELOG**: "Add resource management integration tests"

---

## Phase 5: Production Features

### - [ ] PR-039: Error types
- **Branch**: `feat/error-types`
- **Description**: Implement `InfEngineError`, `RateLimitError`, `ExecutionError`
- **Design Docs**:
  - `execution.md` → "Error Handling Policies"
  - `development_plan.md` → "5.1.2 Error Types"
- **Files**: `src/inf_engine/errors.py`
- **Tests**:
  - `tests/unit/test_errors.py` (error creation, attributes)
- **CHANGELOG**: "Add custom error types"

### - [ ] PR-040: RateLimiter - token bucket
- **Branch**: `feat/rate-limiter-basic`
- **Description**: Implement token bucket `RateLimiter` with `acquire()`
- **Design Docs**:
  - `execution.md` → "Adaptive Rate Limiting"
  - `DESIGN.md` → "Architecture" → "4. Adaptive Backpressure"
  - `development_plan.md` → "5.1.1 Rate Limiter"
- **Files**: `src/inf_engine/resources/rate_limit.py`
- **Tests**:
  - `tests/unit/test_rate_limiter.py` (acquire, token consumption)
- **CHANGELOG**: "Add RateLimiter with token bucket algorithm"

### - [ ] PR-041: RateLimiter - adaptive backoff
- **Branch**: `feat/rate-limiter-adaptive`
- **Description**: Add `backoff()` and `recover()` methods
- **Design Docs**:
  - `execution.md` → "Adaptive Rate Limiting" (backoff, recover)
  - `DESIGN.md` → "Adaptive Backpressure" (backoff flow)
- **Files**: `src/inf_engine/resources/rate_limit.py`
- **Tests**:
  - `tests/unit/test_rate_limiter.py` (backoff reduces rate, recover increases)
- **CHANGELOG**: "Add adaptive backoff to RateLimiter"

### - [ ] PR-042: Integrate RateLimiter with ResourceManager
- **Branch**: `feat/resource-manager-rate-limit`
- **Description**: Add rate limiting to `ResourceManager.execute()`
- **Design Docs**:
  - `resources.md` → "Resource Manager" (rate_limiters)
  - `development_plan.md` → "5.1.5 Integrate Rate Limiting"
- **Files**: `src/inf_engine/resources/manager.py`
- **Tests**:
  - `tests/unit/test_resource_manager.py` (rate limiter used)
- **CHANGELOG**: "Integrate RateLimiter with ResourceManager"

### - [ ] PR-043: Handle RateLimitError in Scheduler
- **Branch**: `feat/scheduler-rate-limit-handling`
- **Description**: Catch `RateLimitError` and trigger requeue
- **Design Docs**:
  - `execution.md` → "Scheduler" (_execute_task with RateLimitError)
  - `DESIGN.md` → "Execution Loop" (RateLimitError handling)
- **Files**: `src/inf_engine/execution/scheduler.py`
- **Tests**:
  - `tests/unit/test_scheduler.py` (rate limit triggers requeue)
- **CHANGELOG**: "Add RateLimitError handling to Scheduler"

### - [ ] PR-044: Checkpoint dataclass
- **Branch**: `feat/checkpoint-type`
- **Description**: Implement `Checkpoint` with save/load methods
- **Design Docs**:
  - `execution.md` → "Checkpointing" (Checkpoint class)
  - `architecture.md` → "Memory and Persistence" → "Checkpointing"
  - `development_plan.md` → "5.1.3 Checkpoint Manager"
- **Files**: `src/inf_engine/execution/checkpoint.py`
- **Tests**:
  - `tests/unit/test_checkpoint.py` (save, load, round-trip)
- **CHANGELOG**: "Add Checkpoint dataclass"

### - [ ] PR-045: CheckpointManager
- **Branch**: `feat/checkpoint-manager`
- **Description**: Implement `CheckpointManager` with buffered writes
- **Design Docs**:
  - `execution.md` → "Checkpointing" (CheckpointManager)
  - `architecture.md` → "Memory and Persistence" → "Checkpointing"
- **Files**: `src/inf_engine/execution/checkpoint.py`
- **Tests**:
  - `tests/unit/test_checkpoint.py` (buffer, flush, record_completion)
- **CHANGELOG**: "Add CheckpointManager for progress persistence"

### - [ ] PR-046: Integrate checkpointing with run()
- **Branch**: `feat/run-with-checkpointing`
- **Description**: Add `checkpoint_dir` parameter to `run()`
- **Design Docs**:
  - `execution.md` → "The run() Function" (checkpoint_dir parameter)
  - `development_plan.md` → "5.1.6 Update run() with Checkpointing"
- **Files**: `src/inf_engine/execution/executor.py`
- **Tests**:
  - `tests/unit/test_executor.py` (checkpoints created)
- **CHANGELOG**: "Add checkpointing support to run()"

### - [ ] PR-047: ResourceMetrics
- **Branch**: `feat/resource-metrics`
- **Description**: Implement metrics collection for endpoints
- **Design Docs**:
  - `resources.md` → "Metrics and Monitoring"
- **Files**: `src/inf_engine/resources/metrics.py`
- **Tests**:
  - `tests/unit/test_metrics.py` (record success, failure, stats)
- **CHANGELOG**: "Add ResourceMetrics for observability"

### - [ ] PR-048: Integrate metrics with ResourceManager
- **Branch**: `feat/resource-manager-metrics`
- **Description**: Add metrics recording to `ResourceManager`
- **Design Docs**:
  - `resources.md` → "Resource Manager" (metrics integration)
  - `resources.md` → "Metrics and Monitoring" (ResourceMetrics usage)
- **Files**: `src/inf_engine/resources/manager.py`
- **Tests**:
  - `tests/unit/test_resource_manager.py` (metrics recorded)
- **CHANGELOG**: "Integrate metrics with ResourceManager"

### - [ ] PR-049: Production features integration tests
- **Branch**: `feat/production-integration-tests`
- **Description**: Add integration tests for rate limiting and checkpointing
- **Design Docs**:
  - `development_plan.md` → "5.3 Integration Tests"
- **Files**:
  - `tests/integration/test_rate_limiting.py`
  - `tests/integration/test_checkpointing.py`
- **Tests**: Rate limit handling, checkpoint persistence
- **CHANGELOG**: "Add production features integration tests"

---

## Phase 6: Branching

### - [ ] PR-050: ConditionalProxy class
- **Branch**: `feat/conditional-proxy`
- **Description**: Implement `ConditionalProxy` with `resolve()` method
- **Design Docs**:
  - `tracing.md` → "Handling Branches" → "Solution: Branch Decorator"
  - `development_plan.md` → "6.1.1 Conditional Proxy"
- **Files**: `src/inf_engine/tracing/proxy.py`
- **Tests**:
  - `tests/unit/test_conditional_proxy.py` (creation, resolve true/false)
- **CHANGELOG**: "Add ConditionalProxy for branching"

### - [ ] PR-051: BranchContext
- **Branch**: `feat/branch-context`
- **Description**: Implement `BranchContext` context manager
- **Design Docs**:
  - `tracing.md` → "Handling Branches" → "Branch Implementation"
  - `development_plan.md` → "6.1.2 Branch Decorator"
- **Files**: `src/inf_engine/tracing/branch.py`
- **Tests**:
  - `tests/unit/test_branch.py` (context manager behavior)
- **CHANGELOG**: "Add BranchContext for conditional tracing"

### - [ ] PR-052: branch() decorator
- **Branch**: `feat/branch-decorator`
- **Description**: Implement `@branch` decorator for marking conditionals
- **Design Docs**:
  - `tracing.md` → "Handling Branches" → "Solution: Branch Decorator"
  - `architecture.md` → "Branching and Conditional Logic"
- **Files**: `src/inf_engine/tracing/branch.py`
- **Tests**:
  - `tests/unit/test_branch.py` (decorator behavior)
- **CHANGELOG**: "Add @branch decorator"

### - [ ] PR-053: match_branch() function
- **Branch**: `feat/match-branch`
- **Description**: Implement `match_branch()` for multi-way conditionals
- **Design Docs**:
  - `tracing.md` → "Handling Branches" → "Alternative: Match Statement Support"
  - `development_plan.md` → "6.1.3 match_branch() function"
- **Files**: `src/inf_engine/tracing/branch.py`
- **Tests**:
  - `tests/unit/test_branch.py` (multiple cases, default)
- **CHANGELOG**: "Add match_branch() for pattern matching"

### - [ ] PR-054: Update Tracer for branches
- **Branch**: `feat/tracer-branch-support`
- **Description**: Add branch stack and conditional node handling to Tracer
- **Design Docs**:
  - `tracing.md` → "Handling Branches" (Tracer modifications)
  - `development_plan.md` → "6.1.4 Update Tracer for branches"
- **Files**: `src/inf_engine/tracing/tracer.py`
- **Tests**:
  - `tests/unit/test_tracer.py` (branch metadata captured)
- **CHANGELOG**: "Add branch support to Tracer"

### - [ ] PR-055: Update GraphNode for branches
- **Branch**: `feat/graph-node-branches`
- **Description**: Add `branch_condition` and `branch_value` to GraphNode
- **Design Docs**:
  - `tracing.md` → "Core Concepts" → "Graph Nodes" (branch fields)
  - `development_plan.md` → "6.1.5 Update GraphNode for branches"
- **Files**: `src/inf_engine/graph.py`
- **Tests**:
  - `tests/unit/test_graph.py` (branch fields)
- **CHANGELOG**: "Add branch fields to GraphNode"

### - [ ] PR-056: Update Executor for branches
- **Branch**: `feat/executor-branches`
- **Description**: Evaluate conditions and select branch at runtime
- **Design Docs**:
  - `architecture.md` → "Branching and Conditional Logic"
  - `development_plan.md` → "6.1.6 Update Executor for branches"
- **Files**: `src/inf_engine/execution/executor.py`
- **Tests**:
  - `tests/unit/test_executor.py` (correct branch executed)
- **CHANGELOG**: "Add branch execution support"

### - [ ] PR-057: Branching integration tests
- **Branch**: `feat/branching-integration-tests`
- **Description**: Add integration tests for conditional execution
- **Design Docs**:
  - `architecture.md` → "Branching and Conditional Logic" (example)
  - `development_plan.md` → "6.3 Integration Tests"
- **Files**: `tests/integration/test_branching.py`
- **Tests**: True branch, false branch, nested branches
- **CHANGELOG**: "Add branching integration tests"

---

## Phase 7: Optimization

### - [ ] PR-058: FeedbackType enum and Feedback dataclass
- **Branch**: `feat/feedback-types`
- **Description**: Implement feedback data structures
- **Design Docs**:
  - `optimization.md` → "Core Components" → "Feedback"
  - `development_plan.md` → "7.1.1 Feedback Types"
- **Files**: `src/inf_engine/optimization/feedback.py`
- **Tests**:
  - `tests/unit/test_feedback.py` (creation, str, with score)
- **CHANGELOG**: "Add Feedback types"

### - [ ] PR-059: Loss abstract base class
- **Branch**: `feat/loss-base`
- **Description**: Implement abstract `Loss` class
- **Design Docs**:
  - `optimization.md` → "Core Components" → "Loss Functions"
  - `development_plan.md` → "7.1.2 Loss Functions"
- **Files**: `src/inf_engine/optimization/loss.py`
- **Tests**:
  - `tests/unit/test_loss.py` (interface validation)
- **CHANGELOG**: "Add Loss abstract base class"

### - [ ] PR-060: VerifierLoss implementation
- **Branch**: `feat/verifier-loss`
- **Description**: Implement `VerifierLoss` for programmatic evaluation
- **Design Docs**:
  - `optimization.md` → "Core Components" → "Loss Functions" (VerifierLoss)
- **Files**: `src/inf_engine/optimization/loss.py`
- **Tests**:
  - `tests/unit/test_loss.py` (pass/fail scenarios)
- **CHANGELOG**: "Add VerifierLoss for programmatic evaluation"

### - [ ] PR-061: LLMJudge loss implementation
- **Branch**: `feat/llm-judge-loss`
- **Description**: Implement `LLMJudge` for LLM-based evaluation
- **Design Docs**:
  - `optimization.md` → "Core Components" → "Loss Functions" (LLMJudge)
- **Files**: `src/inf_engine/optimization/loss.py`
- **Tests**:
  - `tests/unit/test_loss.py` (mocked judge responses)
- **CHANGELOG**: "Add LLMJudge for LLM-as-a-judge evaluation"

### - [ ] PR-062: CompositeLoss implementation
- **Branch**: `feat/composite-loss`
- **Description**: Implement `CompositeLoss` for combining losses
- **Design Docs**:
  - `optimization.md` → "Core Components" → "Loss Functions" (CompositeLoss)
- **Files**: `src/inf_engine/optimization/loss.py`
- **Tests**:
  - `tests/unit/test_loss.py` (aggregation, weighting)
- **CHANGELOG**: "Add CompositeLoss for combining evaluators"

### - [ ] PR-063: BackwardContext and BackwardResult
- **Branch**: `feat/backward-types`
- **Description**: Implement backward pass data structures
- **Design Docs**:
  - `optimization.md` → "Core Components" → "Backward Context"
  - `development_plan.md` → "7.1.3 Backward Context"
- **Files**: `src/inf_engine/optimization/backward.py`
- **Tests**:
  - `tests/unit/test_backward.py` (creation, fields)
- **CHANGELOG**: "Add BackwardContext and BackwardResult types"

### - [ ] PR-064: InferenceModule.backward() method
- **Branch**: `feat/module-backward`
- **Description**: Add default `backward()` implementation to InferenceModule
- **Design Docs**:
  - `optimization.md` → "Module Backward Implementation"
  - `inference_module.md` → "InferenceModule Base Class" (backward method)
  - `architecture.md` → "Execution Flow" → "Backward Pass"
- **Files**: `src/inf_engine/module.py`
- **Tests**:
  - `tests/unit/test_module.py` (default backward behavior)
- **CHANGELOG**: "Add backward() method to InferenceModule"

### - [ ] PR-065: LLMInference.backward() method
- **Branch**: `feat/llm-inference-backward`
- **Description**: Implement `backward()` for LLMInference with parameter feedback
- **Design Docs**:
  - `optimization.md` → "Module Backward Implementation" (LLMInference.backward)
  - `optimization.md` → "Custom Backward Example"
- **Files**: `src/inf_engine/module.py`
- **Tests**:
  - `tests/unit/test_llm_inference.py` (backward generates feedback)
- **CHANGELOG**: "Add backward() to LLMInference"

### - [ ] PR-066: backward() function
- **Branch**: `feat/backward-function`
- **Description**: Implement `backward()` function for graph-wide propagation
- **Design Docs**:
  - `optimization.md` → "Training Loop" (backward function)
  - `architecture.md` → "Execution Flow" → "Backward Pass"
  - `development_plan.md` → "7.1.6 Backward Function"
- **Files**: `src/inf_engine/optimization/backward.py`
- **Tests**:
  - `tests/unit/test_backward.py` (feedback propagates through graph)
- **CHANGELOG**: "Add backward() function for feedback propagation"

### - [ ] PR-067: Optimizer - initialization
- **Branch**: `feat/optimizer-init`
- **Description**: Implement `Optimizer.__init__` with parameter collection
- **Design Docs**:
  - `optimization.md` → "Optimizer"
  - `development_plan.md` → "7.1.7 Optimizer"
- **Files**: `src/inf_engine/optimization/optimizer.py`
- **Tests**:
  - `tests/unit/test_optimizer.py` (init, parameters stored)
- **CHANGELOG**: "Add Optimizer class initialization"

### - [ ] PR-068: Optimizer - accumulate and zero_feedback
- **Branch**: `feat/optimizer-accumulate`
- **Description**: Implement `accumulate()` and `zero_feedback()` methods
- **Design Docs**:
  - `optimization.md` → "Optimizer" (accumulate, zero_feedback methods)
- **Files**: `src/inf_engine/optimization/optimizer.py`
- **Tests**:
  - `tests/unit/test_optimizer.py` (accumulation, clearing)
- **CHANGELOG**: "Add Optimizer feedback accumulation"

### - [ ] PR-069: Optimizer - step method
- **Branch**: `feat/optimizer-step`
- **Description**: Implement `step()` with LLM-based aggregation and update
- **Design Docs**:
  - `optimization.md` → "Optimizer" (step method)
  - `optimization.md` → "Optimizer" (_aggregate_feedback, _generate_update)
- **Files**: `src/inf_engine/optimization/optimizer.py`
- **Tests**:
  - `tests/unit/test_optimizer.py` (step updates parameters)
- **CHANGELOG**: "Add Optimizer.step() for parameter updates"

### - [ ] PR-070: train() function
- **Branch**: `feat/train-function`
- **Description**: Implement `train()` for training loop execution
- **Design Docs**:
  - `optimization.md` → "Training Loop" (train function)
  - `optimization.md` → "Complete Example"
  - `development_plan.md` → "7.1.8 Training Loop"
- **Files**: `src/inf_engine/optimization/train.py`
- **Tests**:
  - `tests/unit/test_train.py` (single epoch, multiple epochs)
- **CHANGELOG**: "Add train() function for training loops"

### - [ ] PR-071: Optimization integration tests
- **Branch**: `feat/optimization-integration-tests`
- **Description**: Add integration tests for backward pass and training
- **Design Docs**:
  - `optimization.md` → "Complete Example"
  - `development_plan.md` → "7.3 Integration Tests"
- **Files**:
  - `tests/integration/test_backward_pass.py`
  - `tests/integration/test_training.py`
- **Tests**: Full backward propagation, parameter updates
- **CHANGELOG**: "Add optimization integration tests"

---

## Post-Implementation

### - [ ] PR-072: Public API exports
- **Branch**: `feat/public-api`
- **Description**: Finalize `__init__.py` exports for clean public API
- **Design Docs**:
  - `architecture.md` → "File Structure"
  - `README.md` → "Core Workflow" (import examples)
- **Files**: `src/inf_engine/__init__.py`
- **Tests**: Import tests for all public symbols
- **CHANGELOG**: "Finalize public API"

### - [ ] PR-073: End-to-end example
- **Branch**: `feat/e2e-example`
- **Description**: Add complete example in `examples/` directory
- **Design Docs**:
  - `DESIGN.md` → "Example Usage"
  - `optimization.md` → "Complete Example"
  - `README.md` → "Core Workflow"
- **Files**: `examples/getting_started.py`
- **Tests**: Example runs successfully
- **CHANGELOG**: "Add getting started example"

### - [ ] PR-074: Documentation
- **Branch**: `docs/api-documentation`
- **Description**: Add API documentation and usage guide
- **Design Docs**:
  - All design docs (reference for documentation content)
- **Files**: `docs/` directory
- **Tests**: N/A
- **CHANGELOG**: "Add API documentation"

---

## Summary

| Phase | PRs | Range |
|-------|-----|-------|
| Foundation | 7 | PR-001 to PR-007 |
| Tracing | 11 | PR-008 to PR-018 |
| Execution | 10 | PR-019 to PR-028 |
| Resources | 10 | PR-029 to PR-038 |
| Production | 11 | PR-039 to PR-049 |
| Branching | 8 | PR-050 to PR-057 |
| Optimization | 14 | PR-058 to PR-071 |
| Post-Implementation | 3 | PR-072 to PR-074 |
| **Total** | **74** | |

---

## Workflow

For each PR:

```bash
# 1. Create branch
git checkout main
git pull
git checkout -b feat/feature-name

# 2. Read referenced design docs
# Review the Design Docs sections listed in the PR

# 3. Implement
# ... write code ...

# 4. Test
make ci

# 5. Update CHANGELOG
# Add entry under [Unreleased]

# 6. Commit
git add .
git commit -m "feat: description"

# 7. Push and create PR
git push -u origin feat/feature-name
gh pr create --title "PR-XXX: Description" --body "..."

# 8. After review and merge
git checkout main
git pull
```
