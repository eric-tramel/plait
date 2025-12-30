# TASKS.md

A line-by-line breakdown of PRs to implement inf-engine, in order.

Each PR represents a single, tested, reviewable increment of functionality.

## Progress

- [x] **Phase 1: Foundation** (7/7)
- [x] **Phase 2: Tracing** (11/11)
- [x] **Phase 3: Execution** (10/10)
- [x] **Phase 3.5: Hardening** (8/8)
- [ ] **Phase 4: Resources** (11/12)
- [ ] **Phase 5: Production Features** (12/13)
- [ ] **Phase 5.5: Profiling** (0/1)
- [ ] **Phase 6: Optimization** (3/7)
- [ ] **Phase 7: Branching** (0/4)
- [ ] **Post-Implementation** (0/3)

**Total: 62/76 PRs completed**

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

### - [x] PR-011: InferenceGraph topological ordering
- **Branch**: `feat/graph-toposort`
- **Description**: Implement `topological_order()` method on InferenceGraph
- **Design Docs**:
  - `tracing.md` → "Inference Graph" (topological_order method)
  - `execution.md` → "Execution State" (uses toposort for scheduling)
- **Files**: `src/inf_engine/graph.py`
- **Tests**:
  - `tests/unit/test_graph.py` (linear, diamond, complex graphs)
- **CHANGELOG**: "Add topological ordering to InferenceGraph"

### - [x] PR-012: InferenceGraph ancestors and descendants
- **Branch**: `feat/graph-traversal`
- **Description**: Implement `ancestors()` and `descendants()` methods
- **Design Docs**:
  - `tracing.md` → "Inference Graph" (ancestors, descendants methods)
  - `execution.md` → "ExecutionState" (uses descendants for failure cascading)
- **Files**: `src/inf_engine/graph.py`
- **Tests**:
  - `tests/unit/test_graph.py` (ancestors, descendants for various graphs)
- **CHANGELOG**: "Add graph traversal methods (ancestors, descendants)"

### - [x] PR-013: Tracer - basic structure
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

### - [x] PR-014: Tracer - input node creation
- **Branch**: `feat/tracer-inputs`
- **Description**: Implement input proxy creation in Tracer
- **Design Docs**:
  - `tracing.md` → "The Tracer" (_create_input_node method)
- **Files**: `src/inf_engine/tracing/tracer.py`
- **Tests**:
  - `tests/unit/test_tracer.py` (input nodes created correctly)
- **CHANGELOG**: "Add input node creation to Tracer"

### - [x] PR-015: Tracer - record_call method
- **Branch**: `feat/tracer-record-call`
- **Description**: Implement `record_call()` to capture module invocations
- **Design Docs**:
  - `tracing.md` → "The Tracer" (record_call method)
  - `DESIGN.md` → "Tracer" (record method)
- **Files**: `src/inf_engine/tracing/tracer.py`
- **Tests**:
  - `tests/unit/test_tracer.py` (nodes created, dependencies captured)
- **CHANGELOG**: "Add record_call() to Tracer"

### - [x] PR-016: Tracer - trace method
- **Branch**: `feat/tracer-trace`
- **Description**: Implement `trace()` method that runs forward() and returns InferenceGraph
- **Design Docs**:
  - `tracing.md` → "The Tracer" (trace method)
  - `tracing.md` → "Example: Complete Tracing Flow"
- **Files**: `src/inf_engine/tracing/tracer.py`
- **Tests**:
  - `tests/unit/test_tracer.py` (full trace of simple module)
- **CHANGELOG**: "Add trace() method to Tracer"

### - [x] PR-017: Connect InferenceModule.__call__ to Tracer
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

### - [x] PR-018: Tracing integration tests
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

### - [x] PR-019: Task and TaskResult dataclasses
- **Branch**: `feat/task-types`
- **Description**: Implement `Task`, `TaskResult`, and `TaskStatus` enum
- **Design Docs**:
  - `execution.md` → "Execution State" (Task, TaskResult, TaskStatus)
  - `development_plan.md` → "3.1.1 Execution State"
- **Files**: `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_execution_types.py` (creation, comparison)
- **CHANGELOG**: "Add Task and TaskResult types"

### - [x] PR-020: ExecutionState - initialization
- **Branch**: `feat/execution-state-init`
- **Description**: Implement `ExecutionState.__init__` with graph analysis
- **Design Docs**:
  - `architecture.md` → "Core Components" → "5. ExecutionState"
  - `execution.md` → "Execution State" (ExecutionState class)
- **Files**: `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_execution_state.py` (init, ready nodes identified)
- **CHANGELOG**: "Add ExecutionState initialization"

### - [x] PR-021: ExecutionState - task management
- **Branch**: `feat/execution-state-tasks`
- **Description**: Implement `get_next_task()`, `mark_complete()`, `is_complete()`
- **Design Docs**:
  - `execution.md` → "Execution State" (task management methods)
  - `architecture.md` → "Execution Flow" → "Forward Pass"
- **Files**: `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_execution_state.py` (task lifecycle)
- **CHANGELOG**: "Add ExecutionState task management methods"

### - [x] PR-022: ExecutionState - failure handling
- **Branch**: `feat/execution-state-failure`
- **Description**: Implement `mark_failed()` with descendant cancellation
- **Design Docs**:
  - `execution.md` → "Execution State" (mark_failed method)
  - `execution.md` → "Error Handling Policies"
- **Files**: `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_execution_state.py` (failure cascades to descendants)
- **CHANGELOG**: "Add ExecutionState failure handling"

### - [x] PR-023: ExecutionState - requeue
- **Branch**: `feat/execution-state-requeue`
- **Description**: Implement `requeue()` for retrying tasks
- **Design Docs**:
  - `execution.md` → "Execution State" (requeue method)
  - `DESIGN.md` → "Adaptive Backpressure" (requeue on rate limit)
- **Files**: `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_execution_state.py` (requeue drops descendants)
- **CHANGELOG**: "Add ExecutionState requeue functionality"

### - [x] PR-024: ExecutionState - get_outputs
- **Branch**: `feat/execution-state-outputs`
- **Description**: Implement `get_outputs()` to retrieve final results
- **Design Docs**:
  - `execution.md` → "Execution State" (get_outputs method)
- **Files**: `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_execution_state.py` (outputs collected correctly)
- **CHANGELOG**: "Add ExecutionState output retrieval"

### - [x] PR-025: Scheduler - basic implementation
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

### - [x] PR-026: Scheduler - execute method
- **Branch**: `feat/scheduler-execute`
- **Description**: Implement `Scheduler.execute()` with TaskGroup
- **Design Docs**:
  - `execution.md` → "Scheduler" (execute method)
  - `DESIGN.md` → "Architecture" → "5. Execution Loop"
- **Files**: `src/inf_engine/execution/scheduler.py`
- **Tests**:
  - `tests/unit/test_scheduler.py` (executes all tasks, respects dependencies)
- **CHANGELOG**: "Add Scheduler.execute() method"

### - [x] PR-027: Basic executor and run() function
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

### - [x] PR-028: Execution integration tests
- **Branch**: `feat/execution-integration-tests`
- **Description**: Add integration tests for full execution scenarios
- **Design Docs**:
  - `execution.md` → "Example: Complete Execution Flow"
  - `architecture.md` → "Execution Flow" → "Forward Pass"
- **Files**: `tests/integration/test_execution.py`
- **Tests**: Linear, parallel, diamond graphs
- **CHANGELOG**: "Add execution integration tests"

---

## Phase 3.5: Hardening

Fixes, improvements, and consistency updates identified during implementation review.

### - [x] PR-029: Fix priority ordering convention
- **Branch**: `fix/priority-ordering`
- **Description**: Standardize priority to "lower value = higher priority" convention across GraphNode and Task
- **Design Docs**:
  - `tracing.md` → "Graph Nodes" (priority field)
  - `execution.md` → "Execution State" (Task priority)
- **Files**:
  - `src/inf_engine/graph.py`
  - `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_graph.py` (priority ordering)
  - `tests/unit/test_execution_state.py` (task ordering)
- **CHANGELOG**: "Standardize priority ordering convention"

### - [x] PR-030: Implement Proxy data access operations
- **Branch**: `feat/proxy-operations`
- **Description**: Implement `__getitem__`, `__iter__`, `keys()`, `values()`, `items()` on Proxy with corresponding Tracer methods
- **Design Docs**:
  - `tracing.md` → "Proxy Objects" (Proxy Operations)
- **Files**:
  - `src/inf_engine/tracing/proxy.py`
  - `src/inf_engine/tracing/tracer.py`
- **Tests**:
  - `tests/unit/test_proxy.py` (all operations)
  - `tests/unit/test_tracer.py` (record_getitem, record_iter, record_method)
- **CHANGELOG**: "Add Proxy data access operations"

### - [x] PR-031: Preserve user dict keys in outputs
- **Branch**: `feat/preserve-output-keys`
- **Description**: Track original dict structure during tracing so results use user-defined keys instead of node IDs
- **Design Docs**:
  - `tracing.md` → "Inference Graph" (output_structure field)
  - `execution.md` → "Execution State" (get_outputs with structure)
- **Files**:
  - `src/inf_engine/tracing/tracer.py`
  - `src/inf_engine/graph.py`
  - `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_tracer.py` (dict key preservation)
  - `tests/integration/test_execution.py` (output keys match forward return)
- **CHANGELOG**: "Preserve user-defined output keys"

### - [x] PR-032: Introduce NodeRef type for node ID references
- **Branch**: `feat/node-ref-type`
- **Description**: Replace raw string node IDs in args/kwargs with typed `NodeRef` to prevent collision with literal strings
- **Design Docs**:
  - `tracing.md` → "Graph Nodes" (NodeRef type)
  - `execution.md` → "Execution State" (argument resolution)
- **Files**:
  - `src/inf_engine/graph.py`
  - `src/inf_engine/tracing/tracer.py`
  - `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_graph.py` (NodeRef usage)
  - `tests/unit/test_execution_state.py` (resolution with NodeRef)
- **CHANGELOG**: "Add NodeRef type for type-safe node references"

### - [x] PR-033: Add cycle detection to topological ordering
- **Branch**: `feat/cycle-detection`
- **Description**: Add cycle detection to `InferenceGraph.topological_order()` with clear error message
- **Design Docs**:
  - `tracing.md` → "Inference Graph" (topological_order)
- **Files**:
  - `src/inf_engine/graph.py`
- **Tests**:
  - `tests/unit/test_graph.py` (cycle detection, error message)
- **CHANGELOG**: "Add cycle detection to topological ordering"

### - [x] PR-034: Replace scheduler busy-wait with Event signaling
- **Branch**: `feat/scheduler-event-signaling`
- **Description**: Replace `sleep(0.001)` polling with `asyncio.Event` for efficient task-ready signaling
- **Design Docs**:
  - `execution.md` → "Scheduler" (event-driven scheduling)
- **Files**:
  - `src/inf_engine/execution/scheduler.py`
  - `src/inf_engine/execution/state.py`
- **Tests**:
  - `tests/unit/test_scheduler.py` (event signaling behavior)
- **CHANGELOG**: "Use event signaling for scheduler efficiency"

### - [x] PR-035: Add state_dict and load_state_dict for parameter serialization
- **Branch**: `feat/state-dict`
- **Description**: Add `state_dict()` and `load_state_dict()` to InferenceModule for saving/loading learned parameters
- **Design Docs**:
  - `inference_module.md` → "InferenceModule Base Class" (serialization methods)
- **Files**:
  - `src/inf_engine/module.py`
- **Tests**:
  - `tests/unit/test_module.py` (state_dict, load_state_dict, round-trip)
- **CHANGELOG**: "Add parameter serialization methods"

### - [x] PR-036: Add graph visualization utility
- **Branch**: `feat/graph-visualization`
- **Description**: Implement `visualize_graph()` function for DOT format output
- **Design Docs**:
  - `tracing.md` → "Graph Visualization"
- **Files**:
  - `src/inf_engine/graph.py`
- **Tests**:
  - `tests/unit/test_graph.py` (DOT output format)
- **CHANGELOG**: "Add graph visualization utility"

---

## Phase 4: Resources

### - [x] PR-037: EndpointConfig dataclass
- **Branch**: `feat/endpoint-config`
- **Description**: Implement `EndpointConfig` with all endpoint settings
- **Design Docs**:
  - `resources.md` → "Resource Configuration" → "Configuration Structure"
  - `development_plan.md` → "4.1.1 Resource Configuration"
- **Files**: `src/inf_engine/resources/config.py`
- **Tests**:
  - `tests/unit/test_resource_config.py` (creation, defaults)
- **CHANGELOG**: "Add EndpointConfig dataclass"

### - [x] PR-038: ResourceConfig dataclass
- **Branch**: `feat/resource-config`
- **Description**: Implement `ResourceConfig` container for multiple endpoints
- **Design Docs**:
  - `resources.md` → "Resource Configuration" → "Configuration Structure"
  - `resources.md` → "Resource Configuration" → "Configuration Examples"
- **Files**: `src/inf_engine/resources/config.py`
- **Tests**:
  - `tests/unit/test_resource_config.py` (multiple endpoints, access)
- **CHANGELOG**: "Add ResourceConfig dataclass"

### - [x] PR-039: LLMRequest and LLMResponse types
- **Branch**: `feat/llm-types`
- **Description**: Implement request/response dataclasses for LLM calls
- **Design Docs**:
  - `resources.md` → "LLM Clients" (LLMRequest, LLMResponse)
  - `development_plan.md` → "4.1.2 LLM Request/Response"
- **Files**: `src/inf_engine/resources/types.py`
- **Tests**:
  - `tests/unit/test_llm_types.py` (creation, serialization)
- **CHANGELOG**: "Add LLMRequest and LLMResponse types"

### - [x] PR-040: LLMClient abstract base class
- **Branch**: `feat/llm-client-base`
- **Description**: Implement abstract `LLMClient` interface
- **Design Docs**:
  - `resources.md` → "LLM Clients" (LLMClient ABC)
  - `development_plan.md` → "4.1.3 LLM Clients"
- **Files**: `src/inf_engine/clients/base.py`
- **Tests**:
  - `tests/unit/test_llm_client.py` (interface validation)
- **CHANGELOG**: "Add LLMClient abstract base class"

### - [x] PR-041: OpenAI client implementation
- **Branch**: `feat/openai-client`
- **Description**: Implement `OpenAIClient` with async completion
- **Design Docs**:
  - `resources.md` → "LLM Clients" (OpenAIClient)
- **Files**: `src/inf_engine/clients/openai.py`
- **Tests**:
  - `tests/unit/test_openai_client.py` (mocked API calls)
- **CHANGELOG**: "Add OpenAI client implementation"

### - [x] PR-042: OpenAI-compatible client
- **Branch**: `feat/openai-compatible-client`
- **Description**: Implement client for vLLM/OpenAI-style endpoints
- **Design Docs**:
  - `resources.md` → "LLM Clients" (OpenAICompatibleClient)
  - `DESIGN.md` → "Architecture" → "4. Pipeline Parallelism for Self-Hosted LLMs"
- **Files**: `src/inf_engine/clients/openai.py`
- **Tests**:
  - `tests/unit/test_openai_client.py` (custom base_url handling)
- **CHANGELOG**: "Add OpenAI-compatible client for self-hosted models"

### - [x] PR-043: ResourceManager - initialization
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

### - [x] PR-044: Scheduler ResourceManager integration
- **Branch**: `feat/resource-manager-execute`
- **Description**: Update Scheduler to use ResourceManager for LLM module execution. ResourceManager stays a pure registry; Scheduler handles execution logic.
- **Design Docs**:
  - `resources.md` → "Resource Manager"
  - `resources.md` → "Module-Resource Binding"
  - `execution.md` → "Scheduler"
- **Files**:
  - `src/inf_engine/execution/scheduler.py`
- **Tests**:
  - `tests/unit/test_scheduler.py` (execute with mocked ResourceManager/client)
- **CHANGELOG**: "Add ResourceManager integration to Scheduler for LLM execution"

### - [x] PR-045: Module binding and resources parameter
- **Branch**: `feat/module-binding`
- **Description**: Add `resources` parameter to `run()`, `bind()` method to InferenceModule, and enable `await module(input)` for bound modules
- **Design Docs**:
  - `execution.md` → "The run() Function" (resources parameter)
  - `inference_module.md` → "InferenceModule Base Class" (bind method, __call__ method)
  - `execution.md` → "Bound Execution (Recommended)"
  - `architecture.md` → "Module Execution API"
  - `development_plan.md` → "4.1.5 Update run() Function", "4.1.6 Add bind() Method", "4.1.7 Update __call__ for Bound Execution"
- **Files**:
  - `src/inf_engine/execution/executor.py`
  - `src/inf_engine/module.py`
- **Tests**:
  - `tests/unit/test_executor.py` (run with ResourceManager)
  - `tests/unit/test_module_binding.py` (bind returns self, stores resources, bound call is async)
- **CHANGELOG**: "Add module binding and resources parameter to run()"

### - [x] PR-046: Batch execution and run_sync
- **Branch**: `feat/batch-execution`
- **Description**: Support list inputs for concurrent batch execution (`await module([a, b, c])`) and add `run_sync()` for synchronous blocking execution
- **Design Docs**:
  - `inference_module.md` → "Module Execution" (batch execution, run_sync method)
  - `architecture.md` → "Module Execution API"
  - `execution.md` → "Execution Patterns" → "Async Execution", "Synchronous Execution"
  - `development_plan.md` → "4.1.8 Add Batch Execution Support"
- **Files**: `src/inf_engine/module.py`
- **Tests**:
  - `tests/unit/test_module_binding.py` (list input returns list output, concurrent execution, run_sync blocks)
  - `tests/integration/test_binding.py` (batch processing runs concurrently)
- **CHANGELOG**: "Add concurrent batch execution and run_sync()"

### - [x] PR-047: Streaming execution
- **Branch**: `feat/streaming-execution`
- **Description**: Add `BatchResult` type, `streaming` and `preserve_order` flags to ExecutionSettings, `on_progress` callback, and cancellation support for streaming batch execution
- **Design Docs**:
  - `execution.md` → "Execution Patterns" → "BatchResult", "Streaming Execution", "Progress Tracking", "Cancellation"
  - `execution.md` → "Updated ExecutionSettings"
  - `inference_module.md` → "InferenceModule Base Class" (_stream_batch method)
- **Files**:
  - `src/inf_engine/execution/types.py`
  - `src/inf_engine/execution/context.py`
  - `src/inf_engine/module.py`
- **Tests**:
  - `tests/unit/test_batch_result.py` (creation, ok property, error handling)
  - `tests/unit/test_execution_settings.py` (streaming flag, preserve_order, on_progress callback)
  - `tests/unit/test_module_binding.py` (streaming returns async iterator, progress callback invoked, break cancels pending)
  - `tests/integration/test_streaming.py` (full streaming flow, cancellation cleanup)
- **CHANGELOG**: "Add streaming execution with BatchResult, progress tracking, and cancellation"

### - [x] PR-048: Resource integration tests
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

### - [x] PR-050: Error types
- **Branch**: `feat/error-types`
- **Description**: Implement `InfEngineError`, `RateLimitError`, `ExecutionError`
- **Design Docs**:
  - `execution.md` → "Error Handling Policies"
  - `development_plan.md` → "5.1.2 Error Types"
- **Files**: `src/inf_engine/errors.py`
- **Tests**:
  - `tests/unit/test_errors.py` (error creation, attributes)
- **CHANGELOG**: "Add custom error types"

### - [x] PR-051: RateLimiter - token bucket
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

### - [x] PR-052: RateLimiter - adaptive backoff
- **Branch**: `feat/rate-limiter-adaptive`
- **Description**: Add `backoff()` and `recover()` methods
- **Design Docs**:
  - `execution.md` → "Adaptive Rate Limiting" (backoff, recover)
  - `DESIGN.md` → "Adaptive Backpressure" (backoff flow)
- **Files**: `src/inf_engine/resources/rate_limit.py`
- **Tests**:
  - `tests/unit/test_rate_limiter.py` (backoff reduces rate, recover increases)
- **CHANGELOG**: "Add adaptive backoff to RateLimiter"

### - [x] PR-053: Integrate RateLimiter with ResourceManager
- **Branch**: `feat/resource-manager-rate-limit`
- **Description**: Add rate limiting to `ResourceManager.execute()`
- **Design Docs**:
  - `resources.md` → "Resource Manager" (rate_limiters)
  - `development_plan.md` → "5.1.5 Integrate Rate Limiting"
- **Files**: `src/inf_engine/resources/manager.py`
- **Tests**:
  - `tests/unit/test_resource_manager.py` (rate limiter used)
- **CHANGELOG**: "Integrate RateLimiter with ResourceManager"

### - [x] PR-054: Handle RateLimitError in Scheduler
- **Branch**: `feat/scheduler-rate-limit-handling`
- **Description**: Catch `RateLimitError` and trigger requeue
- **Design Docs**:
  - `execution.md` → "Scheduler" (_execute_task with RateLimitError)
  - `DESIGN.md` → "Execution Loop" (RateLimitError handling)
- **Files**: `src/inf_engine/execution/scheduler.py`
- **Tests**:
  - `tests/unit/test_scheduler.py` (rate limit triggers requeue)
- **CHANGELOG**: "Add RateLimitError handling to Scheduler"

### - [x] PR-055: Checkpoint dataclass
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

### - [x] PR-056: CheckpointManager
- **Branch**: `feat/checkpoint-manager`
- **Description**: Implement `CheckpointManager` with buffered writes
- **Design Docs**:
  - `execution.md` → "Checkpointing" (CheckpointManager)
  - `architecture.md` → "Memory and Persistence" → "Checkpointing"
- **Files**: `src/inf_engine/execution/checkpoint.py`
- **Tests**:
  - `tests/unit/test_checkpoint.py` (buffer, flush, record_completion)
- **CHANGELOG**: "Add CheckpointManager for progress persistence"

### - [x] PR-057: Integrate checkpointing with run()
- **Branch**: `feat/run-with-checkpointing`
- **Description**: Add `checkpoint_dir` parameter to `run()`
- **Design Docs**:
  - `execution.md` → "The run() Function" (checkpoint_dir parameter)
  - `development_plan.md` → "5.1.6 Update run() with Checkpointing"
- **Files**: `src/inf_engine/execution/executor.py`
- **Tests**:
  - `tests/unit/test_executor.py` (checkpoints created)
- **CHANGELOG**: "Add checkpointing support to run()"

### - [x] PR-058: ExecutionSettings context manager
- **Branch**: `feat/execution-settings`
- **Description**: Add context manager for shared execution configuration
- **Design Docs**:
  - `execution.md` → "ExecutionSettings Context Manager"
  - `architecture.md` → "Module Execution API"
  - `development_plan.md` → "5.1.7 ExecutionSettings Context Manager"
- **Files**: `src/inf_engine/execution/context.py`
- **Tests**:
  - `tests/unit/test_execution_settings.py` (context manager, nested contexts)
- **CHANGELOG**: "Add ExecutionSettings context manager"

### - [x] PR-059: Update InferenceModule for ExecutionSettings
- **Branch**: `feat/module-execution-settings`
- **Description**: Update `__call__` and `_execute_bound` to use ExecutionSettings context
- **Design Docs**:
  - `inference_module.md` → "InferenceModule Base Class" (__call__, _execute_bound)
  - `execution.md` → "Priority Order"
  - `development_plan.md` → "5.1.8 Update InferenceModule for ExecutionSettings"
- **Files**: `src/inf_engine/module.py`
- **Tests**:
  - `tests/unit/test_module_binding.py` (context settings used)
  - `tests/integration/test_execution_settings.py`
- **CHANGELOG**: "Integrate ExecutionSettings with InferenceModule"

### - [x] PR-060: Resource metrics
- **Branch**: `feat/resource-metrics`
- **Description**: Implement `ResourceMetrics` for endpoint observability and integrate with ResourceManager
- **Design Docs**:
  - `resources.md` → "Metrics and Monitoring"
  - `resources.md` → "Resource Manager" (metrics integration)
- **Files**:
  - `src/inf_engine/resources/metrics.py`
  - `src/inf_engine/resources/manager.py`
- **Tests**:
  - `tests/unit/test_metrics.py` (record success, failure, stats)
  - `tests/unit/test_resource_manager.py` (metrics recorded)
- **CHANGELOG**: "Add ResourceMetrics for observability"

### - [x] PR-061: Task timeout and retry handling
- **Branch**: `feat/task-timeout-retry`
- **Description**: Add per-task timeout handling and configurable retry logic for transient failures. Add `TransientError` type for retryable errors. Update Scheduler to use `asyncio.timeout()` and implement exponential backoff retry.
- **Design Docs**:
  - `execution.md` → "Error Handling" (Task Timeout, Task Retry, Scheduler Error Handling)
- **Files**:
  - `src/inf_engine/errors.py` (add TransientError)
  - `src/inf_engine/execution/scheduler.py` (timeout and retry logic)
  - `src/inf_engine/execution/context.py` (task_timeout, max_task_retries, task_retry_delay in ExecutionSettings)
- **Tests**:
  - `tests/unit/test_errors.py` (TransientError creation)
  - `tests/unit/test_scheduler.py` (timeout triggers failure, retry on transient error, exponential backoff, max retries exhausted)
  - `tests/unit/test_execution_settings.py` (timeout/retry settings)
- **CHANGELOG**: "Add task timeout and retry handling with TransientError"

### - [x] PR-062: Production features integration tests
- **Branch**: `feat/production-integration-tests`
- **Description**: Add integration tests for ExecutionSettings, rate limiting, checkpointing, timeouts, and cancellation
- **Design Docs**:
  - `development_plan.md` → "5.3 Integration Tests"
- **Files**:
  - `tests/integration/test_execution_settings.py`
  - `tests/integration/test_rate_limiting.py`
  - `tests/integration/test_checkpointing.py`
  - `tests/integration/test_reliability.py`
- **Tests**: Shared checkpointing, priority order, multiple pipelines, rate limit handling, checkpoint persistence, timeout behavior, cancellation
- **CHANGELOG**: "Add production features integration tests"

---

## Phase 5.5: Profiling

Performance visualization and bottleneck analysis using Chrome Trace Event Format.

### - [x] PR-063: Profiling infrastructure
- **Branch**: `feat/profiling`
- **Description**: Implement `TraceEvent` dataclass, `TraceProfiler` for collecting execution traces, integrate with Scheduler, and add profiling options to ExecutionSettings
- **Design Docs**:
  - `profiling.md` → "Chrome Trace Event Format"
  - `profiling.md` → "TraceProfiler"
  - `profiling.md` → "Integration Points" → "Scheduler Integration"
  - `profiling.md` → "Configuration via ExecutionSettings"
  - `execution.md` → "Scheduler"
  - `execution.md` → "ExecutionSettings Context Manager"
- **Files**:
  - `src/inf_engine/profiling/__init__.py`
  - `src/inf_engine/profiling/profiler.py`
  - `src/inf_engine/execution/scheduler.py`
  - `src/inf_engine/execution/context.py`
- **Tests**:
  - `tests/unit/test_profiler.py` (event creation, task lifecycle, export)
  - `tests/unit/test_scheduler.py` (profiler events emitted)
  - `tests/unit/test_execution_settings.py` (profiler created, trace exported on exit)
  - `tests/integration/test_profiling.py` (trace file generation, event correctness, counter events, multi-endpoint traces)
- **CHANGELOG**: "Add profiling infrastructure with TraceProfiler and ExecutionSettings integration"

---

## Phase 6: Optimization

### - [x] PR-064: Parameter description and ForwardRecord
- **Branch**: `feat/parameter-forward-record`
- **Description**: Update `Parameter` to require `description` field for self-documenting optimization. Implement `ForwardRecord` dataclass to capture forward pass state (graph, node inputs/outputs, module map) for backward propagation.
- **Design Docs**:
  - `optimization.md` → "Core Components" → "Parameter"
  - `optimization.md` → "Core Components" → "ForwardRecord"
- **Files**:
  - `src/inf_engine/parameter.py` (add required description field)
  - `src/inf_engine/optimization/__init__.py`
  - `src/inf_engine/optimization/record.py` (ForwardRecord)
  - `src/inf_engine/execution/executor.py` (update run() to support record=True)
- **Tests**:
  - `tests/unit/test_parameter.py` (description required, included in str)
  - `tests/unit/test_forward_record.py` (creation, fields, from execution)
  - `tests/unit/test_executor.py` (run with record=True returns ForwardRecord)
- **CHANGELOG**: "Add required Parameter.description and ForwardRecord for backward pass"

### - [x] PR-065: Feedback and Loss base types
- **Branch**: `feat/feedback-loss-types`
- **Description**: Implement `FeedbackType` enum, `Feedback` dataclass with `backward()` method (mirrors PyTorch's `loss.backward()`), and abstract `Loss` class that attaches ForwardRecord to Feedback.
- **Design Docs**:
  - `optimization.md` → "Core Components" → "Feedback"
  - `optimization.md` → "Loss Functions"
- **Files**:
  - `src/inf_engine/optimization/feedback.py`
  - `src/inf_engine/optimization/loss.py`
- **Tests**:
  - `tests/unit/test_feedback.py` (creation, str, score, backward raises without record)
  - `tests/unit/test_loss.py` (ABC interface, _attach_record helper)
- **CHANGELOG**: "Add Feedback with backward() method and Loss base class"

### - [x] PR-066: Loss implementations
- **Branch**: `feat/loss-implementations`
- **Description**: Implement `VerifierLoss` for programmatic evaluation, `LLMJudge` for LLM-based evaluation, and `CompositeLoss` for weighted multi-objective optimization. All loss functions accept `record=` parameter and attach it to returned Feedback.
- **Design Docs**:
  - `optimization.md` → "Loss Functions" (VerifierLoss, LLMJudge, CompositeLoss)
- **Files**: `src/inf_engine/optimization/loss.py`
- **Tests**:
  - `tests/unit/test_loss.py` (VerifierLoss pass/fail, LLMJudge mocked responses and parsing, CompositeLoss aggregation and weighting, record attachment)
- **CHANGELOG**: "Add VerifierLoss, LLMJudge, and CompositeLoss"

### - [ ] PR-067: Backward pass infrastructure
- **Branch**: `feat/backward-pass`
- **Description**: Implement `BackwardContext` (with `reason()` method for optimizer-provided LLM), `BackwardResult`, async `InferenceModule.backward()` default implementation, `LLMInference.backward()` that generates parameter feedback using description, and `_propagate_backward()` for graph traversal in reverse topological order.
- **Design Docs**:
  - `optimization.md` → "Core Components" → "BackwardContext"
  - `optimization.md` → "Module Backward Implementation"
  - `optimization.md` → "Backward Propagation"
- **Files**:
  - `src/inf_engine/optimization/backward.py` (BackwardContext, BackwardResult, _propagate_backward)
  - `src/inf_engine/optimization/feedback.py` (Feedback.backward() implementation)
  - `src/inf_engine/module.py` (async backward() method)
- **Tests**:
  - `tests/unit/test_backward_context.py` (creation, reason() with/without LLM)
  - `tests/unit/test_backward.py` (propagation through linear graph, fan-out aggregation)
  - `tests/unit/test_module.py` (default backward passes feedback to inputs)
  - `tests/unit/test_llm_inference.py` (backward generates parameter feedback with description)
- **CHANGELOG**: "Add backward pass with BackwardContext, module backward methods, and graph propagation"

### - [ ] PR-068: Optimizer base and SFAOptimizer
- **Branch**: `feat/optimizer`
- **Description**: Implement `Optimizer` ABC following torch.optim pattern: initialized with `module.parameters()`, has `zero_feedback()`, `bind()`, and async `step()`. Internal LLMs use fixed aliases (`optimizer/aggregator`, `optimizer/updater`, `optimizer/reasoning`). Implement `SFAOptimizer` (Stochastic Feedback Ascent) with `conservatism` hyperparameter for incremental updates.
- **Design Docs**:
  - `optimization.md` → "Optimizer"
  - `optimization.md` → "SFAOptimizer"
- **Files**: `src/inf_engine/optimization/optimizer.py`
- **Tests**:
  - `tests/unit/test_optimizer.py` (init with parameters, zero_feedback clears buffers, bind required before step, step aggregates and updates, conservatism affects prompts)
- **CHANGELOG**: "Add Optimizer base class and SFAOptimizer with torch.optim-style API"

### - [ ] PR-069: Training utilities
- **Branch**: `feat/training-utilities`
- **Description**: Implement `train()` function with mini-batch support (batch_size parameter), epoch iteration, and shuffle option. Add `TrainingHistory` for tracking losses and parameter snapshots. Add `eval()` and `train()` mode methods to InferenceModule, and `requires_grad_()` for freezing/unfreezing parameters.
- **Design Docs**:
  - `optimization.md` → "Mini-Batch Training"
  - `optimization.md` → "Complete Example"
- **Files**:
  - `src/inf_engine/optimization/train.py` (train function, TrainingHistory)
  - `src/inf_engine/module.py` (eval/train modes, requires_grad_)
- **Tests**:
  - `tests/unit/test_train.py` (single epoch, multiple epochs, batch accumulation, shuffle)
  - `tests/unit/test_training_history.py` (record_step, record_epoch, snapshots)
  - `tests/unit/test_module.py` (eval/train mode switching, requires_grad_ propagation)
- **CHANGELOG**: "Add train() function with mini-batch support and training mode utilities"

### - [ ] PR-070: Optimization integration tests
- **Branch**: `feat/optimization-integration-tests`
- **Description**: Add comprehensive integration tests for the full optimization workflow: forward with recording, loss computation, feedback.backward(), mini-batch accumulation, optimizer.step(), and parameter updates across epochs.
- **Design Docs**:
  - `optimization.md` → "Complete Example"
  - `optimization.md` → "Feedback Accumulation"
- **Files**:
  - `tests/integration/test_backward_pass.py` (feedback propagation through complex graphs)
  - `tests/integration/test_training.py` (full training loop with mocked LLMs)
  - `tests/integration/test_mini_batch.py` (batch accumulation, fan-out + mini-batch combined)
- **Tests**:
  - Linear graph backward propagation
  - Diamond graph with fan-out feedback aggregation
  - Mini-batch accumulation across 4-8 samples
  - Full training loop with parameter updates
  - Optimizer with different conservatism levels
- **CHANGELOG**: "Add optimization integration tests"

---

## Phase 7: Branching

### - [ ] PR-071: Core branching primitives
- **Branch**: `feat/branching-primitives`
- **Description**: Implement `ConditionalProxy` with `resolve()` method, `BranchContext` context manager, and `@branch` decorator
- **Design Docs**:
  - `tracing.md` → "Handling Branches" → "Solution: Branch Decorator"
  - `tracing.md` → "Handling Branches" → "Branch Implementation"
  - `architecture.md` → "Branching and Conditional Logic"
  - `development_plan.md` → "6.1.1 Conditional Proxy", "6.1.2 Branch Decorator"
- **Files**:
  - `src/inf_engine/tracing/proxy.py`
  - `src/inf_engine/tracing/branch.py`
- **Tests**:
  - `tests/unit/test_conditional_proxy.py` (creation, resolve true/false)
  - `tests/unit/test_branch.py` (context manager behavior, decorator behavior)
- **CHANGELOG**: "Add ConditionalProxy, BranchContext, and @branch decorator"

### - [ ] PR-072: match_branch function
- **Branch**: `feat/match-branch`
- **Description**: Implement `match_branch()` for multi-way conditionals
- **Design Docs**:
  - `tracing.md` → "Handling Branches" → "Alternative: Match Statement Support"
  - `development_plan.md` → "6.1.3 match_branch() function"
- **Files**: `src/inf_engine/tracing/branch.py`
- **Tests**:
  - `tests/unit/test_branch.py` (multiple cases, default)
- **CHANGELOG**: "Add match_branch() for pattern matching"

### - [ ] PR-073: Graph and tracer branch support
- **Branch**: `feat/graph-tracer-branches`
- **Description**: Add branch stack and conditional node handling to Tracer, add `branch_condition` and `branch_value` fields to GraphNode
- **Design Docs**:
  - `tracing.md` → "Handling Branches" (Tracer modifications)
  - `tracing.md` → "Core Concepts" → "Graph Nodes" (branch fields)
  - `development_plan.md` → "6.1.4 Update Tracer for branches", "6.1.5 Update GraphNode for branches"
- **Files**:
  - `src/inf_engine/tracing/tracer.py`
  - `src/inf_engine/graph.py`
- **Tests**:
  - `tests/unit/test_tracer.py` (branch metadata captured)
  - `tests/unit/test_graph.py` (branch fields)
- **CHANGELOG**: "Add branch support to Tracer and GraphNode"

### - [ ] PR-074: Branch execution and integration tests
- **Branch**: `feat/branch-execution`
- **Description**: Update Executor to evaluate conditions and select branch at runtime, add integration tests for conditional execution
- **Design Docs**:
  - `architecture.md` → "Branching and Conditional Logic"
  - `development_plan.md` → "6.1.6 Update Executor for branches", "6.3 Integration Tests"
- **Files**:
  - `src/inf_engine/execution/executor.py`
  - `tests/integration/test_branching.py`
- **Tests**:
  - `tests/unit/test_executor.py` (correct branch executed)
  - Integration tests for true branch, false branch, nested branches
- **CHANGELOG**: "Add branch execution support and branching integration tests"

---

## Post-Implementation

### - [ ] PR-075: Public API exports
- **Branch**: `feat/public-api`
- **Description**: Finalize `__init__.py` exports for clean public API
- **Design Docs**:
  - `architecture.md` → "File Structure"
  - `README.md` → "Core Workflow" (import examples)
- **Files**: `src/inf_engine/__init__.py`
- **Tests**: Import tests for all public symbols
- **CHANGELOG**: "Finalize public API"

### - [ ] PR-076: End-to-end example
- **Branch**: `feat/e2e-example`
- **Description**: Add complete example in `examples/` directory
- **Design Docs**:
  - `DESIGN.md` → "Example Usage"
  - `optimization.md` → "Complete Example"
  - `README.md` → "Core Workflow"
- **Files**: `examples/getting_started.py`
- **Tests**: Example runs successfully
- **CHANGELOG**: "Add getting started example"

### - [ ] PR-077: Documentation
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
| Hardening | 8 | PR-029 to PR-036 |
| Resources | 12 | PR-037 to PR-048 |
| Production | 13 | PR-050 to PR-062 |
| Profiling | 1 | PR-063 |
| Optimization | 7 | PR-064 to PR-070 |
| Branching | 4 | PR-071 to PR-074 |
| Post-Implementation | 3 | PR-075 to PR-077 |
| **Total** | **76** | |

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
