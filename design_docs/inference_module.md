# InferenceModule System

The `InferenceModule` is the core abstraction in plait, directly inspired by PyTorch's `nn.Module`. It provides a familiar interface for defining composable inference pipelines.

## Design Goals

1. **PyTorch Familiarity**: Users of PyTorch should feel at home
2. **Automatic Discovery**: Child modules and parameters are auto-registered
3. **Sync API, Async Execution**: Users write sync code; framework handles async
4. **Dual-Mode Execution**: Support both direct execution and tracing

## InferenceModule Base Class

```python
from __future__ import annotations
from typing import Any, TypeVar, Generic
from dataclasses import dataclass, field

T = TypeVar("T")

class InferenceModule:
    """
    Base class for all inference operations.

    Analogous to torch.nn.Module. Subclass this to define custom
    inference logic by implementing the forward() method.
    """

    def __init__(self) -> None:
        # Internal registries (like PyTorch's _modules and _parameters)
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_name", None)
        object.__setattr__(self, "_bound_resources", None)
        object.__setattr__(self, "_bound_config", {})

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-register child modules and parameters."""
        if isinstance(value, InferenceModule):
            self._children[name] = value
            value._name = name
        elif isinstance(value, Parameter):
            self._parameters[name] = value
            value._name = name
        object.__setattr__(self, name, value)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Define the inference computation.

        Override this method to implement your module's logic.
        During tracing, this receives Value objects (with refs).
        During execution, this receives actual values.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()"
        )

    def backward(
        self,
        feedback: Feedback,
        ctx: BackwardContext
    ) -> BackwardResult:
        """
        Propagate feedback to inputs and parameters.

        Override to customize how feedback flows backward through
        this module. The default implementation distributes feedback
        equally to all inputs.
        """
        return BackwardResult(
            input_feedback={inp: feedback for inp in ctx.inputs},
            parameter_feedback={},
        )

    def bind(
        self,
        resources: ResourceConfig | ResourceManager,
        max_concurrent: int = 100,
        **kwargs: Any,
    ) -> Self:
        """
        Bind resources to this module for direct execution.

        After binding, the module can be called directly with await:
            pipeline = MyPipeline().bind(resources=config)
            result = await pipeline("input")

        Args:
            resources: Resource configuration or manager for LLM endpoints.
            max_concurrent: Maximum concurrent tasks during execution.
            **kwargs: Additional execution options (checkpoint_dir, etc.).

        Returns:
            Self, for method chaining.

        Note:
            Binding is propagated to all child modules automatically.
        """
        object.__setattr__(self, "_bound_resources", resources)
        object.__setattr__(self, "_bound_config", {"max_concurrent": max_concurrent, **kwargs})
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the module.

        Behavior depends on context:
        1. If a trace context is active: records the call and returns a Value
        2. If resources are bound OR ExecutionSettings is active: traces and executes
        3. Otherwise: executes forward() directly (for non-LLM modules)

        When bound or in an ExecutionSettings context, this method is async
        and should be awaited. Supports batch execution when the first
        argument is a list.
        """
        ctx = get_trace_context()
        if ctx is not None:
            return ctx.record_call(self, args, kwargs)

        # Check if we have resources (bound or from context)
        settings = get_execution_settings()
        has_resources = (
            self._bound_resources is not None or
            (settings is not None and settings.resources is not None)
        )

        if has_resources:
            # Bound or context execution: trace and execute
            return self._execute_bound(*args, **kwargs)

        return self.forward(*args, **kwargs)

    def run_sync(self, *args: Any, **kwargs: Any) -> Any:
        """Execute synchronously (blocking).

        Convenience method for scripts and notebooks where async isn't needed.
        Blocks until execution completes and returns the result.

        Args:
            *args: Positional arguments passed to forward().
            **kwargs: Keyword arguments passed to forward().

        Returns:
            Single result for single input, list for batch input.

        Raises:
            RuntimeError: If called from within an async context.

        Example:
            >>> pipeline = MyPipeline().bind(resources=config)
            >>> result = pipeline.run_sync("Hello")
            >>> results = pipeline.run_sync(["a", "b", "c"])
        """
        import asyncio
        return asyncio.run(self._execute_bound(*args, **kwargs))

    async def _execute_bound(self, *args: Any, **kwargs: Any) -> Any:
        """Execute with bound or context resources.

        Settings are merged with this priority (highest first):
        1. Call-time kwargs
        2. Bound settings (from .bind())
        3. Context settings (from ExecutionSettings)
        4. Defaults

        When self.training is True:
        - Single input: returns TracedOutput[T] with record attached
        - Batch input: returns list[TracedOutput[T]]

        When self.training is False:
        - Single input: returns T (raw output)
        - Batch input: returns list[T]
        """
        import asyncio
        from plait.execution import run
        from plait.execution.context import get_execution_settings

        # Get context settings
        settings = get_execution_settings()

        # Build effective config: context < bound < kwargs
        effective_config = {}
        if settings is not None:
            effective_config.update({
                "max_concurrent": settings.max_concurrent,
                "checkpoint_dir": settings.checkpoint_dir,
                "scheduler": settings.scheduler,
            })
        effective_config.update(self._bound_config)
        effective_config.update(kwargs)

        # Determine resources: bound takes precedence over context
        resources = self._bound_resources
        if resources is None and settings is not None:
            resources = settings.resources

        # Handle batch execution
        if args and isinstance(args[0], list):
            inputs = args[0]

            # Check if streaming mode is enabled
            if settings is not None and settings.streaming:
                return self._stream_batch(inputs, args[1:], resources, effective_config)

            # Concurrent batch execution (not sequential)
            tasks = [
                asyncio.create_task(
                    run(self, inp, *args[1:], resources=resources, **effective_config)
                )
                for inp in inputs
            ]
            return await asyncio.gather(*tasks)

        return await run(
            self, *args,
            resources=resources,
            **effective_config,
        )

    async def _stream_batch(
        self,
        inputs: list[Any],
        extra_args: tuple[Any, ...],
        resources: Any,
        config: dict[str, Any],
    ) -> AsyncIterator[BatchResult]:
        """Stream batch results as they complete.

        Yields BatchResult objects as each input completes processing.
        Results are yielded in completion order (fastest first) unless
        preserve_order=True in ExecutionSettings.

        Args:
            inputs: List of inputs to process.
            extra_args: Additional positional arguments.
            resources: Resource configuration.
            config: Effective execution config.

        Yields:
            BatchResult for each completed input.
        """
        import asyncio
        from plait.execution import run
        from plait.execution.context import get_execution_settings

        settings = get_execution_settings()
        preserve_order = settings.preserve_order if settings else False

        # Create all tasks with their indices
        tasks = [
            asyncio.create_task(
                run(self, inp, *extra_args, resources=resources, **config)
            )
            for inp in inputs
        ]

        if preserve_order:
            # Yield in input order
            for i, task in enumerate(tasks):
                try:
                    output = await task
                    yield BatchResult(index=i, input=inputs[i], output=output, error=None)
                except Exception as e:
                    yield BatchResult(index=i, input=inputs[i], output=None, error=e)
        else:
            # Yield as completed (fastest first)
            pending = {task: i for i, task in enumerate(tasks)}
            for coro in asyncio.as_completed(tasks):
                task = None
                for t in pending:
                    if t.done() or t == coro:
                        task = t
                        break
                i = pending.pop(task, 0)
                try:
                    output = await coro
                    yield BatchResult(index=i, input=inputs[i], output=output, error=None)
                except Exception as e:
                    yield BatchResult(index=i, input=inputs[i], output=None, error=e)

    # ─────────────────────────────────────────────────────────────
    # Module Introspection (PyTorch-like API)
    # ─────────────────────────────────────────────────────────────

    def children(self) -> Iterator[InferenceModule]:
        """Iterate over immediate child modules."""
        yield from self._children.values()

    def named_children(self) -> Iterator[tuple[str, InferenceModule]]:
        """Iterate over immediate child modules with names."""
        yield from self._children.items()

    def modules(self) -> Iterator[InferenceModule]:
        """Iterate over all modules in the tree (including self)."""
        yield self
        for child in self.children():
            yield from child.modules()

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, InferenceModule]]:
        """Iterate over all modules with hierarchical names."""
        yield prefix, self
        for name, child in self.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(child_prefix)

    def parameters(self) -> Iterator[Parameter]:
        """Iterate over all parameters in the tree."""
        yield from self._parameters.values()
        for child in self.children():
            yield from child.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Iterate over all parameters with hierarchical names."""
        for name, param in self._parameters.items():
            param_name = f"{prefix}.{name}" if prefix else name
            yield param_name, param
        for name, child in self.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_parameters(child_prefix)

    # ─────────────────────────────────────────────────────────────
    # State Serialization (PyTorch-like API)
    # ─────────────────────────────────────────────────────────────

    def state_dict(self) -> dict[str, str]:
        """
        Return a dictionary of all parameter values.

        Used for saving learned prompts/instructions after optimization.
        Keys are hierarchical parameter names (e.g., "summarizer.system_prompt").
        """
        return {name: param.value for name, param in self.named_parameters()}

    def load_state_dict(self, state_dict: dict[str, str]) -> None:
        """
        Load parameter values from a dictionary.

        Used for restoring learned prompts/instructions.
        Missing keys are ignored; extra keys raise KeyError.
        """
        params = dict(self.named_parameters())
        for name, value in state_dict.items():
            if name not in params:
                raise KeyError(f"Unknown parameter: {name}")
            params[name].value = value

    # ─────────────────────────────────────────────────────────────
    # Training Mode Control (PyTorch-like API)
    # ─────────────────────────────────────────────────────────────

    def train(self, mode: bool = True) -> InferenceModule:
        """
        Set the module to training mode.

        In training mode, the backward pass collects feedback for optimization.
        Propagates to all child modules.

        Returns self for method chaining.
        """
        object.__setattr__(self, "_training", mode)
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> InferenceModule:
        """
        Set the module to evaluation mode.

        In evaluation mode, the backward pass is disabled.
        Equivalent to train(False). Propagates to all child modules.

        Returns self for method chaining.
        """
        return self.train(False)

    @property
    def training(self) -> bool:
        """Return whether the module is in training mode."""
        return getattr(self, "_training", True)

    def requires_grad_(self, requires_grad: bool = True) -> InferenceModule:
        """
        Set requires_grad for all parameters in this module tree.

        Used to freeze/unfreeze parameters during optimization.
        When frozen (requires_grad=False), parameters don't accumulate feedback.

        Returns self for method chaining.
        """
        for param in self.parameters():
            param.requires_grad = requires_grad
        return self
```

## Parameter Class

Parameters are values that can be optimized through backward passes.
When `requires_grad=True`, `description` should explain the parameter’s role
so optimizers can reason about updates.

Parameters are **not** `Value` objects. When parameters participate in
computation, they are lifted into `Value` via `valueify(param)` so their usage
is tracked in the graph with a stable `ref` (e.g., `param:module.path.name`).

See `parameters.md` for the full Parameter specification.

```python
@dataclass
class Parameter:
    """
    A learnable value that can be optimized via backward passes.

    Similar to torch.nn.Parameter, but for string values (prompts,
    instructions, etc.) that are optimized via LLM feedback rather
    than gradient descent.
    """

    value: str
    description: str | None = None  # Required when requires_grad=True
    requires_grad: bool = True
    _name: str | None = field(default=None, repr=False)
    _feedback_buffer: list[str] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        """Return the current value when used as a string."""
        return self.value

    def accumulate_feedback(self, feedback: str) -> None:
        """Collect feedback from backward passes."""
        if self.requires_grad:
            self._feedback_buffer.append(feedback)

    def get_accumulated_feedback(self) -> list[str]:
        """Get all accumulated feedback."""
        return list(self._feedback_buffer)

    def apply_update(self, new_value: str) -> None:
        """Apply an optimizer-computed update."""
        self.value = new_value
        self._feedback_buffer.clear()

    def zero_feedback(self) -> None:
        """Clear accumulated feedback (like zero_grad)."""
        self._feedback_buffer.clear()
```

## LLMInference (Atomic Module)

The fundamental building block for LLM operations:

```python
class LLMInference(InferenceModule):
    """
    Atomic module for LLM API calls.

    This is the base operation that actually makes LLM requests.
    All other modules ultimately compose these.
    """

    def __init__(
        self,
        alias: str,
        system_prompt: str | Parameter = "",
        temperature: float = 1.0,
        max_tokens: int | None = None,
        response_format: type | None = None,
    ) -> None:
        super().__init__()
        self.alias = alias
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format

        # Wrap string as Parameter if needed
        if isinstance(system_prompt, str):
            if system_prompt:
                self.system_prompt = Parameter(
                    system_prompt,
                    description=None,
                    requires_grad=False,
                )
            else:
                self.system_prompt = None
        else:
            self.system_prompt = system_prompt

    def forward(self, prompt: str) -> str:
        """
        Execute the LLM call.

        During tracing, this is not called directly.
        During execution, the runtime handles the actual API call.
        """
        # This is a placeholder - actual execution is handled by the runtime
        # which has access to the ResourceManager
        raise RuntimeError(
            "LLMInference.forward() should not be called directly. "
            "Use run() to execute the module."
        )

## Functional Delegate (Atomic LLM Op)

`LLMInference` is the public module interface, but its execution path should
delegate to the functional API’s atomic LLM op (e.g., `F.chat_complete`). This
keeps alias-based resource management, training hooks, and profiling identity
while standardizing on a single structured-input → response `Value(RESPONSE)`
implementation.

    def backward(
        self,
        feedback: Feedback,
        ctx: BackwardContext
    ) -> BackwardResult:
        """Propagate feedback to input prompt and system prompt."""
        result = BackwardResult()

        # Feedback for the input prompt
        result.input_feedback["prompt"] = feedback.content

        # Feedback for system prompt parameter (if learnable)
        if self.system_prompt is not None and self.system_prompt.requires_grad:
            result.parameter_feedback["system_prompt"] = (
                f"Given output feedback: {feedback.content}\n"
                f"Suggest improvements to the system prompt."
            )

        return result
```

## Composing Modules

### Sequential Composition

```python
class SummarizeAndAnalyze(InferenceModule):
    """A simple sequential pipeline."""

    def __init__(self):
        super().__init__()
        self.summarizer = LLMInference(
            alias="fast_llm",
            system_prompt="You are a concise summarizer.",
        )
        self.analyzer = LLMInference(
            alias="smart_llm",
            system_prompt="You are a thorough analyst.",
        )

    def forward(self, text: str) -> str:
        summary = self.summarizer(text)
        analysis = self.analyzer(summary)
        return analysis
```

### Parallel Composition (Fan-out)

```python
class MultiPerspectiveAnalysis(InferenceModule):
    """Analyze from multiple perspectives in parallel."""

    def __init__(self):
        super().__init__()
        self.technical = LLMInference(
            alias="llm",
            system_prompt="Analyze from a technical perspective.",
        )
        self.business = LLMInference(
            alias="llm",
            system_prompt="Analyze from a business perspective.",
        )
        self.user = LLMInference(
            alias="llm",
            system_prompt="Analyze from a user perspective.",
        )

    def forward(self, text: str) -> dict[str, str]:
        # These can execute in parallel (same input, no dependencies)
        return {
            "technical": self.technical(text),
            "business": self.business(text),
            "user": self.user(text),
        }
```

### Fan-in Composition

```python
class Synthesizer(InferenceModule):
    """Combine multiple analyses into a final report."""

    def __init__(self):
        super().__init__()
        self.analyzer = MultiPerspectiveAnalysis()
        self.synthesizer = LLMInference(
            alias="smart_llm",
            system_prompt="Synthesize multiple perspectives into a cohesive report.",
        )

    def forward(self, text: str) -> str:
        perspectives = self.analyzer(text)

        # Format for synthesis
        combined = "\n\n".join(
            f"## {name.title()} Perspective\n{analysis}"
            for name, analysis in perspectives.items()
        )

        return self.synthesizer(combined)
```

### Nested Composition

```python
class DeepPipeline(InferenceModule):
    """Deeply nested module composition."""

    def __init__(self):
        super().__init__()
        self.stage1 = SummarizeAndAnalyze()    # Contains 2 LLMInference
        self.stage2 = MultiPerspectiveAnalysis() # Contains 3 LLMInference
        self.stage3 = Synthesizer()              # Contains nested modules

    def forward(self, text: str) -> str:
        result1 = self.stage1(text)
        result2 = self.stage2(result1)
        return self.stage3(str(result2))
```

The framework flattens all nested modules into a single execution graph.

## Parameterized Modules

Modules with learnable parameters:

```python
class AssistantGeneration(InferenceModule):
    """
    An assistant with optimizable instructions.

    The system prompt can be improved through backward passes.
    """

    def __init__(
        self,
        assistant_instructions: str,
        temperature: float = 1.0
    ):
        super().__init__()
        # This parameter can be optimized
        self.assistant_instructions = Parameter(
            assistant_instructions,
            description=(
                "Defines the assistant’s core behavior and tone. "
                "Should be stable but improvable based on feedback."
            ),
        )

        # LLM for generating responses
        self.llm = LLMInference(
            alias="assistant_llm",
            temperature=temperature,
        )

        # LLM for backward pass (prompt optimization)
        self.optimizer_llm = LLMInference(
            alias="optimizer_llm",
            temperature=0.7,
            system_prompt="You help improve prompts based on feedback.",
        )

    def forward(self, request: str) -> str:
        # Combine instructions with request
        full_prompt = f"{self.assistant_instructions}\n\nUser: {request}"
        return self.llm(full_prompt)

    def backward(
        self,
        feedback: Feedback,
        ctx: BackwardContext
    ) -> BackwardResult:
        """Generate feedback for the instructions parameter."""
        result = BackwardResult()

        # Feedback for the input request
        result.input_feedback["request"] = feedback.content

        # Use LLM to generate parameter feedback
        if self.assistant_instructions.requires_grad:
            improvement_prompt = f"""
Current instructions: {self.assistant_instructions.value}
Output that was produced: {ctx.output}
Feedback on output: {feedback.content}

What specific changes to the instructions would improve the output?
"""
            # Note: This would be traced and executed as part of backward graph
            improvement = self.optimizer_llm(improvement_prompt)
            result.parameter_feedback["assistant_instructions"] = improvement

        return result
```

## Module Execution

plait provides multiple execution patterns. See `execution.md` → "Execution Patterns" for complete details.

### Execution Patterns Overview

| Pattern | Syntax | Returns | Use Case |
|---------|--------|---------|----------|
| Async single | `await module("x")` | `T` | Standard async code |
| Async batch | `await module([...])` | `list[T]` | Process multiple inputs |
| Sync single | `module.run_sync("x")` | `T` | Scripts, notebooks |
| Sync batch | `module.run_sync([...])` | `list[T]` | Batch scripts |
| Streaming | `async for r in module([...])` | `BatchResult` | Servers, progress |
| Training single | `module.train(); await module("x")` | `TracedOutput[T]` | Training with backward |
| Training batch | `module.train(); await module([...])` | `list[TracedOutput[T]]` | Batch training |

### Bound Execution (Recommended)

Bind resources to a module once, then call it directly:

```python
from plait import ResourceConfig

# Configure resources
resources = ResourceConfig({
    "fast_llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "max_concurrent": 10,
    },
    "smart_llm": {
        "provider": "openai",
        "model": "gpt-4o",
        "max_concurrent": 5,
    },
})

# Bind resources to the module
pipeline = DeepPipeline().bind(resources=resources)

# Async execution
result = await pipeline("Analyze this document...")

# Batch execution - runs CONCURRENTLY, returns list
results = await pipeline([
    "Document 1 text...",
    "Document 2 text...",
    "Document 3 text...",
])

# Sync execution for scripts (no await needed)
result = pipeline.run_sync("Analyze this document...")
results = pipeline.run_sync(["Doc 1...", "Doc 2...", "Doc 3..."])
```

### Streaming Execution

For servers and progress tracking, use streaming mode:

```python
async with ExecutionSettings(resources=config, streaming=True):
    async for result in pipeline(large_batch):
        if result.ok:
            await send_to_client(result.output)
        else:
            logger.error(f"Input {result.index} failed: {result.error}")
```

### Training Execution

For training workflows, enable training mode to capture `ForwardRecord` via `TracedOutput`:

```python
# Enable training mode - outputs carry records implicitly
pipeline.train()

# Single input - returns TracedOutput
output = await pipeline(input)  # TracedOutput[str]
output.value                     # The actual string output
output._record                   # ForwardRecord for backward()

# Batch inputs - returns list[TracedOutput]
outputs = await pipeline(batch_inputs)  # list[TracedOutput]

# Use in training loop (loss extracts records automatically)
feedbacks = await loss_fn.batch(outputs, targets=targets)
await Feedback.backward_batch(feedbacks, optimizer=optimizer)
await optimizer.step()

# Disable training mode for inference
pipeline.eval()
output = await pipeline(input)  # str (raw value, no overhead)
```

See `optimization.md` → "Batch Training API" for complete training documentation.

### Using run() for Advanced Control

For cases requiring custom configuration per-call:

```python
from plait import run, ResourceConfig

# Configure resources
resources = ResourceConfig({...})

# Create module (not bound)
pipeline = DeepPipeline()

# Use run() with per-call options
result = await run(
    pipeline,
    "Analyze this document...",
    resources=resources,
    max_concurrent=50,
    checkpoint_dir=Path("./checkpoints"),
)
```

### Direct Execution (for testing non-LLM modules)

```python
# For modules that don't use LLMInference, direct calls work
class TextProcessor(InferenceModule):
    def forward(self, text: str) -> str:
        return text.upper()

processor = TextProcessor()
result = processor("hello")  # Returns "HELLO" directly
```

## Type Hints and Generics

For type-safe modules:

```python
from typing import TypeVar, Generic

Input = TypeVar("Input")
Output = TypeVar("Output")

class TypedModule(InferenceModule, Generic[Input, Output]):
    """Base class for type-annotated modules."""

    def forward(self, input: Input) -> Output:
        raise NotImplementedError

class Summarizer(TypedModule[str, str]):
    """Strongly typed summarizer."""

    def __init__(self):
        super().__init__()
        self.llm = LLMInference(alias="llm")

    def forward(self, text: str) -> str:
        return self.llm(f"Summarize: {text}")

class Classifier(TypedModule[str, list[str]]):
    """Strongly typed classifier."""

    def __init__(self):
        super().__init__()
        self.llm = LLMInference(
            alias="llm",
            response_format=list[str],  # Structured output
        )

    def forward(self, text: str) -> list[str]:
        return self.llm(f"Classify into categories: {text}")
```

## Best Practices

### 1. Keep Modules Focused

```python
# Good: Single responsibility
class Summarizer(InferenceModule):
    def forward(self, text: str) -> str:
        return self.llm(f"Summarize: {text}")

# Bad: Multiple responsibilities
class DoEverything(InferenceModule):
    def forward(self, text: str) -> dict:
        summary = self.summarizer(text)
        sentiment = self.sentiment(text)
        keywords = self.keywords(text)
        translation = self.translator(summary)
        # ... too much in one module
```

### 2. Use Descriptive Aliases

```python
# Good: Clear purpose
self.llm = LLMInference(alias="customer_support_llm")

# Bad: Generic
self.llm = LLMInference(alias="llm1")
```

### 3. Document Parameters

```python
class CustomAssistant(InferenceModule):
    def __init__(self, instructions: str):
        super().__init__()
        # Document what this parameter controls
        self.instructions = Parameter(
            instructions,
            description="Defines the assistant’s response policy and tone.",
        )
```

### 4. Consider Testability

```python
class TestableModule(InferenceModule):
    def __init__(self, llm: LLMInference | None = None):
        super().__init__()
        # Allow dependency injection for testing
        self.llm = llm or LLMInference(alias="default")
```
