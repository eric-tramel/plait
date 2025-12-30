"""InferenceModule base class for inf-engine.

This module provides the core abstraction for building composable
inference pipelines, inspired by PyTorch's nn.Module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Self

from inf_engine.tracing.context import get_trace_context

if TYPE_CHECKING:
    from inf_engine.execution.types import BatchResult
    from inf_engine.parameter import Parameter
    from inf_engine.resources.config import ResourceConfig
    from inf_engine.resources.manager import ResourceManager


class InferenceModule:
    """Base class for all inference operations.

    Analogous to torch.nn.Module. Subclass this to define custom
    inference logic by implementing the forward() method.

    Child modules and parameters assigned as attributes are automatically
    registered, enabling recursive traversal and parameter collection.

    Args:
        None

    Example:
        >>> from inf_engine.parameter import Parameter
        >>> class MyModule(InferenceModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.prompt = Parameter("You are helpful.")
        ...
        >>> module = MyModule()
        >>> "prompt" in module._parameters
        True

    Note:
        Always call super().__init__() in subclass __init__ methods
        to ensure proper registration of children and parameters.
    """

    _children: dict[str, InferenceModule]
    _parameters: dict[str, Parameter]
    _name: str | None
    _bound_resources: ResourceConfig | ResourceManager | None
    _bound_config: dict[str, Any]

    def __init__(self) -> None:
        """Initialize the module with empty registries.

        Sets up internal dictionaries for tracking child modules and
        parameters. Uses object.__setattr__ to avoid triggering the
        custom __setattr__ during initialization.
        """
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_name", None)
        object.__setattr__(self, "_bound_resources", None)
        object.__setattr__(self, "_bound_config", {})

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute with automatic registration of modules and parameters.

        When a value is assigned to an attribute:
        - If it's an InferenceModule, it's registered as a child module
        - If it's a Parameter, it's registered in the parameters dict
        - The value's _name is set to the attribute name for introspection

        Args:
            name: The attribute name.
            value: The value to assign.

        Note:
            This method is called for all attribute assignments, including
            those in __init__. Internal attributes (starting with '_') that
            are not modules or parameters are set directly.
        """
        # Import here to avoid circular imports at module load time
        from inf_engine.parameter import Parameter

        if isinstance(value, InferenceModule):
            self._children[name] = value
            value._name = name
        elif isinstance(value, Parameter):
            self._parameters[name] = value
            value._name = name

        object.__setattr__(self, name, value)

    # ─────────────────────────────────────────────────────────────
    # Module Introspection (PyTorch-like API)
    # ─────────────────────────────────────────────────────────────

    def children(self) -> Iterator[InferenceModule]:
        """Iterate over immediate child modules.

        Yields child modules in the order they were registered.
        Does not recurse into nested modules.

        Yields:
            Each immediate child InferenceModule.

        Example:
            >>> class Parent(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.child1 = InferenceModule()
            ...         self.child2 = InferenceModule()
            ...
            >>> parent = Parent()
            >>> list(parent.children())  # doctest: +ELLIPSIS
            [<...InferenceModule...>, <...InferenceModule...>]
        """
        yield from self._children.values()

    def named_children(self) -> Iterator[tuple[str, InferenceModule]]:
        """Iterate over immediate child modules with their names.

        Yields (name, module) pairs for each immediate child.
        Does not recurse into nested modules.

        Yields:
            Tuples of (attribute_name, child_module).

        Example:
            >>> class Parent(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.child1 = InferenceModule()
            ...
            >>> parent = Parent()
            >>> [(name, type(m).__name__) for name, m in parent.named_children()]
            [('child1', 'InferenceModule')]
        """
        yield from self._children.items()

    def modules(self) -> Iterator[InferenceModule]:
        """Iterate over all modules in the tree, including self.

        Performs a depth-first traversal starting from this module.
        Includes this module as the first item yielded.

        Yields:
            All InferenceModules in the subtree rooted at this module.

        Example:
            >>> class Nested(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.inner = InferenceModule()
            ...
            >>> class Outer(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.nested = Nested()
            ...
            >>> outer = Outer()
            >>> len(list(outer.modules()))
            3
        """
        yield self
        for child in self.children():
            yield from child.modules()

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, InferenceModule]]:
        """Iterate over all modules with hierarchical dot-separated names.

        Performs a depth-first traversal, yielding (name, module) pairs.
        Names are hierarchical, e.g., "layer1.sublayer.module".

        Args:
            prefix: Prefix to prepend to all names. Used internally
                for recursive calls to build hierarchical names.

        Yields:
            Tuples of (hierarchical_name, module). The root module
            has an empty string name (or the prefix if provided).

        Example:
            >>> class Inner(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...
            >>> class Outer(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.inner = Inner()
            ...
            >>> outer = Outer()
            >>> [(name, type(m).__name__) for name, m in outer.named_modules()]
            [('', 'Outer'), ('inner', 'Inner')]
        """
        yield prefix, self
        for name, child in self.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(child_prefix)

    def parameters(self) -> Iterator[Parameter]:
        """Iterate over all parameters in the module tree.

        Recursively yields parameters from this module and all
        descendant modules in depth-first order.

        Yields:
            All Parameter objects in the subtree.

        Example:
            >>> from inf_engine.parameter import Parameter
            >>> class MyModule(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.prompt = Parameter("test")
            ...
            >>> module = MyModule()
            >>> list(module.parameters())  # doctest: +ELLIPSIS
            [Parameter(value='test', ...)]
        """
        yield from self._parameters.values()
        for child in self.children():
            yield from child.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Iterate over all parameters with hierarchical dot-separated names.

        Recursively yields (name, parameter) pairs from this module
        and all descendants. Names reflect the module hierarchy.

        Args:
            prefix: Prefix to prepend to parameter names. Used internally
                for recursive calls to build hierarchical names.

        Yields:
            Tuples of (hierarchical_name, parameter).

        Example:
            >>> from inf_engine.parameter import Parameter
            >>> class Inner(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.weight = Parameter("w")
            ...
            >>> class Outer(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.bias = Parameter("b")
            ...         self.inner = Inner()
            ...
            >>> outer = Outer()
            >>> [(name, p.value) for name, p in outer.named_parameters()]
            [('bias', 'b'), ('inner.weight', 'w')]
        """
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
        """Return a dictionary of all parameter values.

        Used for saving learned prompts/instructions after optimization.
        Keys are hierarchical parameter names (e.g., "summarizer.system_prompt"),
        matching the output of named_parameters().

        Returns:
            A dictionary mapping parameter names to their string values.

        Example:
            >>> from inf_engine.parameter import Parameter
            >>> class Inner(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.weight = Parameter("w")
            ...
            >>> class Outer(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.bias = Parameter("b")
            ...         self.inner = Inner()
            ...
            >>> outer = Outer()
            >>> outer.state_dict()
            {'bias': 'b', 'inner.weight': 'w'}

        Note:
            The returned dict can be serialized to JSON/pickle and later
            restored with load_state_dict().
        """
        return {name: param.value for name, param in self.named_parameters()}

    def load_state_dict(self, state_dict: dict[str, str]) -> None:
        """Load parameter values from a dictionary.

        Used for restoring learned prompts/instructions from a saved state.
        The keys in state_dict must match the hierarchical parameter names
        from this module's named_parameters().

        Args:
            state_dict: Dictionary mapping parameter names to their values.

        Raises:
            KeyError: If a key in state_dict does not match any parameter
                in this module. Missing keys in state_dict are silently
                ignored (partial loads are allowed).

        Example:
            >>> from inf_engine.parameter import Parameter
            >>> class MyModule(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.prompt = Parameter("original")
            ...
            >>> module = MyModule()
            >>> module.load_state_dict({"prompt": "updated"})
            >>> module.prompt.value
            'updated'

        Example with unknown key:
            >>> from inf_engine.parameter import Parameter
            >>> class MyModule(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.prompt = Parameter("test")
            ...
            >>> module = MyModule()
            >>> module.load_state_dict({"unknown": "value"})
            Traceback (most recent call last):
                ...
            KeyError: 'Unknown parameter: unknown'

        Note:
            This method modifies the parameter values in-place. If you need
            to preserve the original values, use state_dict() first to save
            them.
        """
        params = dict(self.named_parameters())
        for name, value in state_dict.items():
            if name not in params:
                raise KeyError(f"Unknown parameter: {name}")
            params[name].value = value

    # ─────────────────────────────────────────────────────────────
    # Resource Binding (Direct Execution API)
    # ─────────────────────────────────────────────────────────────

    def bind(
        self,
        resources: ResourceConfig | ResourceManager,
        max_concurrent: int = 100,
        **kwargs: Any,
    ) -> Self:
        """Bind resources to this module for direct execution.

        After binding, the module can be called directly with await:
            pipeline = MyPipeline().bind(resources=config)
            result = await pipeline("input")

        Args:
            resources: Resource configuration or manager for LLM endpoints.
            max_concurrent: Maximum concurrent tasks during execution.
            **kwargs: Additional execution options (checkpoint_dir, etc.).

        Returns:
            Self, for method chaining.

        Example:
            >>> from inf_engine.resources.config import ResourceConfig, EndpointConfig
            >>> config = ResourceConfig(endpoints={
            ...     "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
            ... })
            >>> pipeline = MyPipeline().bind(resources=config)
            >>> result = await pipeline("Hello!")

        Example with additional options:
            >>> pipeline = MyPipeline().bind(
            ...     resources=config,
            ...     max_concurrent=50,
            ...     checkpoint_dir="/data/checkpoints",
            ... )

        Note:
            Bound resources and config can be overridden per-call by passing
            keyword arguments to __call__, or by using ExecutionSettings context.
        """
        object.__setattr__(self, "_bound_resources", resources)
        object.__setattr__(
            self, "_bound_config", {"max_concurrent": max_concurrent, **kwargs}
        )
        return self

    def run_sync(self, *args: Any, **kwargs: Any) -> Any:
        """Execute synchronously (blocking).

        Convenience method for scripts and notebooks where async isn't needed.
        Blocks until execution completes and returns the result.

        This method requires that resources are available either through:
        - Prior `bind()` call on this module, or
        - An active `ExecutionSettings` context

        Args:
            *args: Positional arguments passed to forward().
            **kwargs: Keyword arguments passed to forward().

        Returns:
            Single result for single input, list for batch input.

        Raises:
            RuntimeError: If called from within an async context (would block
                the event loop), or if no resources are available.

        Example:
            >>> pipeline = MyPipeline().bind(resources=config)
            >>> result = pipeline.run_sync("Hello")
            >>> results = pipeline.run_sync(["a", "b", "c"])

        Example with ExecutionSettings:
            >>> with ExecutionSettings(resources=config):
            ...     result = pipeline.run_sync("Hello")

        Note:
            Use `await module(...)` in async code instead. This method is
            intended for synchronous scripts and REPL environments only.
        """
        import asyncio

        from inf_engine.execution.context import get_execution_settings

        # Check if we have resources (bound or from context)
        settings = get_execution_settings()
        has_resources = self._bound_resources is not None or (
            settings is not None and settings.resources is not None
        )

        if not has_resources:
            raise RuntimeError(
                "run_sync() requires bound resources. "
                "Use module.bind(resources=...) or ExecutionSettings context."
            )

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # If we get here, there's a running loop - can't use run_sync()
            raise RuntimeError(
                "run_sync() cannot be called from within an async context. "
                "Use 'await module(...)' instead."
            )
        except RuntimeError as e:
            # Re-raise our error, but catch the "no running loop" error
            if "cannot be called" in str(e):
                raise
            # No running loop - this is what we want, proceed with asyncio.run

        return asyncio.run(self._execute_bound(*args, **kwargs))

    # ─────────────────────────────────────────────────────────────
    # Forward and Call (Core Execution Interface)
    # ─────────────────────────────────────────────────────────────

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Define the inference computation.

        Override this method to implement your module's logic.
        During tracing, this receives Proxy objects representing
        symbolic values. During execution, this receives actual values.

        Args:
            *args: Positional arguments for the computation.
            **kwargs: Keyword arguments for the computation.

        Returns:
            The result of the inference computation.

        Raises:
            NotImplementedError: If not overridden in a subclass.

        Example:
            >>> class Greeter(InferenceModule):
            ...     def forward(self, name: str) -> str:
            ...         return f"Hello, {name}!"
            ...
            >>> greeter = Greeter()
            >>> greeter("World")
            'Hello, World!'
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")

    async def backward(
        self,
        feedback: Any,
        ctx: Any,
    ) -> Any:
        """Propagate feedback backward through this module.

        Default implementation passes feedback unchanged to all inputs.
        Override for custom backward logic that generates more targeted
        feedback for specific inputs or parameters.

        This method is called during the backward pass initiated by
        `feedback.backward()`. The ctx parameter provides access to
        the forward pass context including inputs, outputs, and the
        computation graph.

        Args:
            feedback: Combined feedback from downstream nodes. This is
                a Feedback object containing content, score, and metadata.
            ctx: BackwardContext with inputs, output, graph structure,
                and optional reasoning LLM for generating feedback.

        Returns:
            BackwardResult with input_feedback and parameter_feedback
            dictionaries specifying how feedback should be distributed.

        Example:
            >>> async def backward(self, feedback, ctx):
            ...     from inf_engine.optimization.backward import BackwardResult
            ...     result = BackwardResult()
            ...
            ...     # Pass feedback to all inputs unchanged
            ...     for input_name in ctx.inputs:
            ...         result.input_feedback[input_name] = feedback
            ...
            ...     return result

        Note:
            The default implementation passes feedback unchanged to all
            inputs. Override this method to implement custom feedback
            propagation logic, such as:
            - Generating parameter-specific feedback
            - Filtering feedback based on input relevance
            - Using ctx.reason() for LLM-powered feedback generation
        """
        from inf_engine.optimization.backward import BackwardResult

        result = BackwardResult()

        # Pass feedback to all inputs unchanged
        for input_name in ctx.inputs:
            result.input_feedback[input_name] = feedback

        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the module.

        Behavior depends on context:
        1. If a trace context is active: records the call and returns a Proxy
        2. If resources are bound OR ExecutionSettings is active: traces and executes
        3. Otherwise: executes forward() directly (for non-LLM modules)

        When bound or in an ExecutionSettings context, this method is async
        and should be awaited. Supports batch execution when the first
        argument is a list.

        Args:
            *args: Positional arguments passed to forward().
            **kwargs: Keyword arguments passed to forward().

        Returns:
            If tracing: A Proxy representing the eventual output of this call.
            If bound/context: A coroutine that yields the execution result.
            Otherwise: The result from forward().

        Example:
            >>> class Doubler(InferenceModule):
            ...     def forward(self, x: int) -> int:
            ...         return x * 2
            ...
            >>> doubler = Doubler()
            >>> doubler(5)  # Without trace context, calls forward() directly
            10

        Example with bound resources:
            >>> pipeline = MyPipeline().bind(resources=config)
            >>> result = await pipeline("input")  # Async execution

        Example with ExecutionSettings:
            >>> async with ExecutionSettings(resources=config):
            ...     result = await pipeline("input")

        Note:
            During tracing, the tracer records this call as a node in the
            execution graph. The forward() method is not called; instead,
            dependencies are tracked based on Proxy arguments.
        """
        from inf_engine.execution.context import get_execution_settings

        tracer = get_trace_context()
        if tracer is not None:
            return tracer.record_call(self, args, kwargs)

        # Check if we have resources (bound or from context)
        settings = get_execution_settings()
        has_resources = self._bound_resources is not None or (
            settings is not None and settings.resources is not None
        )

        if has_resources:
            # Bound or context execution: trace and execute
            return self._execute_bound(*args, **kwargs)

        return self.forward(*args, **kwargs)

    async def _stream_batch(
        self,
        inputs: list[Any],
        extra_args: tuple[Any, ...],
        resources: Any,
        forward_kwargs: dict[str, Any],
        effective_config: dict[str, Any],
        preserve_order: bool = False,
        on_progress: Any = None,
    ) -> AsyncIterator[BatchResult[Any]]:
        """Stream batch results as they complete.

        Creates tasks for all inputs and yields BatchResult objects
        as they complete. Supports both completion order (fastest)
        and preserve_order (input order) modes.

        Args:
            inputs: List of inputs to process.
            extra_args: Additional positional arguments after the input.
            resources: Resource configuration for execution.
            forward_kwargs: Keyword arguments for forward().
            effective_config: Execution configuration options.
            preserve_order: If True, yield in input order. If False,
                yield as soon as each result completes.
            on_progress: Optional callback(completed, total) for progress.

        Yields:
            BatchResult objects containing index, input, output, and error.

        Example:
            >>> async for result in module._stream_batch(inputs, ...):
            ...     if result.ok:
            ...         process(result.output)
            ...     else:
            ...         log_error(result.error)

        Note:
            When the consumer breaks out of the loop, pending tasks are
            cancelled automatically by Python's async generator cleanup.
        """
        import asyncio

        from inf_engine.execution.executor import run
        from inf_engine.execution.types import BatchResult

        total = len(inputs)
        completed = 0

        # Create all tasks upfront with their indices
        async def run_with_index(
            idx: int, inp: Any
        ) -> tuple[int, Any, Any, Exception | None]:
            """Run a single input and return (index, input, output, error)."""
            try:
                result = await run(
                    self,
                    inp,
                    *extra_args,
                    resources=resources,
                    **forward_kwargs,
                    **effective_config,
                )
                return (idx, inp, result, None)
            except Exception as e:
                return (idx, inp, None, e)

        tasks = [
            asyncio.create_task(run_with_index(i, inp)) for i, inp in enumerate(inputs)
        ]

        try:
            if preserve_order:
                # Yield results in input order (may wait on slower items)
                for task in tasks:
                    idx, inp, output, error = await task
                    completed += 1
                    if on_progress is not None:
                        on_progress(completed, total)
                    yield BatchResult(index=idx, input=inp, output=output, error=error)
            else:
                # Yield results as they complete (fastest throughput)
                # Map tasks to their indices for lookup
                pending = {task: i for i, task in enumerate(tasks)}

                while pending:
                    done, _ = await asyncio.wait(
                        pending.keys(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        del pending[task]
                        idx, inp, output, error = task.result()
                        completed += 1
                        if on_progress is not None:
                            on_progress(completed, total)
                        yield BatchResult(
                            index=idx, input=inp, output=output, error=error
                        )
        finally:
            # Cancel any remaining tasks if consumer breaks early
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    async def _execute_bound(
        self, *args: Any, **kwargs: Any
    ) -> Any | AsyncIterator[BatchResult[Any]]:
        """Execute with bound or context resources.

        Traces the module and executes it using the run() function.
        Settings are merged with this priority (highest first):
        1. Call-time kwargs
        2. Bound settings (from .bind())
        3. Context settings (from ExecutionSettings)
        4. Defaults

        Args:
            *args: Positional arguments passed to forward().
            **kwargs: Keyword arguments passed to forward().

        Returns:
            For single input: The output of the module's forward() method.
            For batch input (list): Returns a list of outputs.
            For streaming batch: Returns an async iterator of BatchResult.

        Example:
            >>> pipeline = MyPipeline().bind(resources=config)
            >>> result = await pipeline("Hello!")

        Example with batch execution:
            >>> results = await pipeline(["input1", "input2", "input3"])

        Example with streaming:
            >>> async with ExecutionSettings(streaming=True):
            ...     async for result in pipeline(["a", "b", "c"]):
            ...         print(result.output)

        Note:
            This method is called internally by __call__ when resources
            are available. Users should not call it directly.
        """
        import asyncio

        from inf_engine.execution.context import get_execution_settings
        from inf_engine.execution.executor import run

        # Get context settings
        settings = get_execution_settings()

        # Build effective config: context < bound < kwargs
        # Start with defaults
        effective_config: dict[str, Any] = {}

        # Layer 1: Context settings (lowest priority)
        if settings is not None:
            if settings.max_concurrent is not None:
                effective_config["max_concurrent"] = settings.max_concurrent
            checkpoint_dir = settings.get_checkpoint_dir()
            if checkpoint_dir is not None:
                effective_config["checkpoint_dir"] = checkpoint_dir

        # Layer 2: Bound settings (medium priority)
        effective_config.update(self._bound_config)

        # Layer 3: Call-time kwargs (highest priority)
        # Extract execution-related kwargs from user kwargs
        execution_keys = {"max_concurrent", "checkpoint_dir", "execution_id"}
        user_execution_kwargs = {k: v for k, v in kwargs.items() if k in execution_keys}
        forward_kwargs = {k: v for k, v in kwargs.items() if k not in execution_keys}
        effective_config.update(user_execution_kwargs)

        # Determine resources: bound takes precedence over context
        resources = self._bound_resources
        if resources is None and settings is not None:
            resources = settings.resources

        # Get streaming configuration from settings
        streaming = settings.get_streaming() if settings is not None else False
        preserve_order = (
            settings.get_preserve_order() if settings is not None else False
        )
        on_progress = settings.get_on_progress() if settings is not None else None

        # Handle batch execution - run all inputs concurrently
        if args and isinstance(args[0], list):
            inputs = args[0]
            if not inputs:
                if streaming:
                    # Return an empty async iterator
                    async def empty_iterator() -> AsyncIterator[BatchResult[Any]]:
                        return
                        yield  # type: ignore[misc]  # Make this a generator

                    return empty_iterator()
                return []

            # Check if streaming mode is enabled
            if streaming:
                return self._stream_batch(
                    inputs=inputs,
                    extra_args=args[1:],
                    resources=resources,
                    forward_kwargs=forward_kwargs,
                    effective_config=effective_config,
                    preserve_order=preserve_order,
                    on_progress=on_progress,
                )

            # Non-streaming batch execution with optional progress callback
            total = len(inputs)
            completed = 0

            async def run_with_progress(inp: Any) -> Any:
                """Run a single input and update progress."""
                nonlocal completed
                result = await run(
                    self,
                    inp,
                    *args[1:],
                    resources=resources,
                    **forward_kwargs,
                    **effective_config,
                )
                completed += 1
                if on_progress is not None:
                    on_progress(completed, total)
                return result

            # Create tasks for concurrent execution
            tasks = [asyncio.create_task(run_with_progress(inp)) for inp in inputs]
            return await asyncio.gather(*tasks)

        return await run(
            self,
            *args,
            resources=resources,
            **forward_kwargs,
            **effective_config,
        )


class LLMInference(InferenceModule):
    """Atomic module for LLM API calls.

    This is the fundamental building block for LLM operations. All other
    modules ultimately compose LLMInference instances to build complex
    inference pipelines.

    The alias parameter decouples the module from specific endpoints,
    allowing the same module to run against different models/endpoints
    based on resource configuration at runtime.

    Args:
        alias: Resource binding key that maps to an endpoint configuration.
            This allows the same module to use different LLM providers
            depending on the ResourceConfig passed to run().
        system_prompt: System prompt for the LLM. Can be a string (converted
            to a non-learnable Parameter) or a Parameter instance (for
            learnable prompts). Empty string results in no system prompt.
        temperature: Sampling temperature for the LLM. Higher values produce
            more random outputs. Defaults to 1.0.
        max_tokens: Maximum number of tokens to generate. None means no limit
            (use model default).
        response_format: Expected response format type for structured output.
            None means plain text response.

    Example:
        >>> llm = LLMInference(alias="fast_llm", temperature=0.7)
        >>> llm.alias
        'fast_llm'
        >>> llm.temperature
        0.7

    Example with system prompt:
        >>> llm = LLMInference(
        ...     alias="assistant",
        ...     system_prompt="You are a helpful assistant.",
        ...     temperature=0.5,
        ... )
        >>> llm.system_prompt.value
        'You are a helpful assistant.'
        >>> llm.system_prompt.requires_grad
        False

    Note:
        LLMInference.forward() should not be called directly. Use the run()
        function to execute modules, which handles tracing and resource
        management.
    """

    alias: str
    system_prompt: Parameter | None
    temperature: float
    max_tokens: int | None
    response_format: type | None

    def __init__(
        self,
        alias: str,
        system_prompt: str | Parameter = "",
        temperature: float = 1.0,
        max_tokens: int | None = None,
        response_format: type | None = None,
    ) -> None:
        """Initialize the LLMInference module.

        Args:
            alias: Resource binding key for endpoint resolution.
            system_prompt: System prompt string or Parameter.
            temperature: Sampling temperature (0.0 to 2.0 typical).
            max_tokens: Maximum tokens to generate.
            response_format: Type for structured output parsing.
        """
        super().__init__()
        self.alias = alias
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format

        # Handle system_prompt: wrap strings as Parameters, pass through Parameters
        from inf_engine.parameter import Parameter

        if isinstance(system_prompt, str):
            if system_prompt:
                # Non-empty string: wrap as non-learnable Parameter
                self.system_prompt = Parameter(
                    system_prompt,
                    description="System prompt for LLM",
                    requires_grad=False,
                )
            else:
                # Empty string: no system prompt
                self.system_prompt = None
        else:
            # Already a Parameter: use as-is (may be learnable)
            self.system_prompt = system_prompt

    def forward(self, prompt: str) -> str:
        """Execute the LLM call.

        This method should not be called directly. During tracing, the tracer
        intercepts calls and records them in the graph. During execution, the
        runtime handles the actual API call through the ResourceManager.

        Args:
            prompt: The user prompt to send to the LLM.

        Returns:
            The LLM's response text.

        Raises:
            RuntimeError: Always raised because direct execution is not
                supported. Use run() to execute modules.

        Note:
            The runtime replaces this with actual LLM calls. This placeholder
            exists to define the expected signature and to catch accidental
            direct invocations.
        """
        raise RuntimeError(
            "LLMInference.forward() should not be called directly. "
            "Use run() to execute the module."
        )

    async def backward(
        self,
        feedback: Any,
        ctx: Any,
    ) -> Any:
        """Backward pass for LLM inference.

        Generates feedback for both the input prompt and any learnable
        parameters (like system_prompt). The parameter feedback includes
        context about what the LLM received and produced to help the
        optimizer understand how to improve the parameter.

        Args:
            feedback: Combined feedback from downstream nodes.
            ctx: BackwardContext with inputs, output, and graph structure.

        Returns:
            BackwardResult with:
            - input_feedback["prompt"]: Feedback for the input prompt
            - parameter_feedback["system_prompt"]: Feedback for the system
              prompt if it's a learnable Parameter

        Example:
            >>> # LLMInference backward is called automatically during
            >>> # feedback.backward() when the module is in the graph
            >>> output, record = await run(llm_module, "Hello", record=True)
            >>> feedback = await loss_fn(output, target, record=record)
            >>> await feedback.backward()  # Calls llm_module.backward()

        Note:
            The parameter feedback includes:
            - The current system prompt value
            - The parameter description
            - A sample of the input and output
            - The feedback received
            This context helps the optimizer generate targeted improvements.
        """
        from inf_engine.optimization.backward import BackwardResult
        from inf_engine.optimization.feedback import Feedback
        from inf_engine.parameter import Parameter

        result = BackwardResult()

        # Feedback for the input prompt
        result.input_feedback["prompt"] = Feedback(
            content=f"The LLM output received this feedback: {feedback.content}",
            score=feedback.score,
            feedback_type=feedback.feedback_type,
        )

        # Feedback for learnable parameters
        if (
            isinstance(self.system_prompt, Parameter)
            and self.system_prompt.requires_grad
        ):
            # Get input and output for context, truncating if too long
            input_text = str(ctx.inputs.get("prompt", ""))[:500]
            output_text = str(ctx.output)[:500]

            # Build detailed feedback for the system prompt parameter
            score_info = (
                f"Score: {feedback.score}" if feedback.score is not None else ""
            )

            result.parameter_feedback["system_prompt"] = f"""
The LLM module with system prompt:
"{self.system_prompt.value}"

Parameter description: {self.system_prompt.description}

Received input: {input_text}{"..." if len(str(ctx.inputs.get("prompt", ""))) > 500 else ""}

Produced output: {output_text}{"..." if len(str(ctx.output)) > 500 else ""}

Received this feedback: {feedback.content}
{score_info}

Suggest specific improvements to the system prompt that would address this feedback.
""".strip()

        return result
