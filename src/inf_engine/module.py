"""InferenceModule base class for inf-engine.

This module provides the core abstraction for building composable
inference pipelines, inspired by PyTorch's nn.Module.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Self

from inf_engine.tracing.context import get_trace_context

if TYPE_CHECKING:
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

    async def _execute_bound(self, *args: Any, **kwargs: Any) -> Any:
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
            The output of the module's forward() method. For batch execution
            (list input), returns a list of outputs.

        Example:
            >>> pipeline = MyPipeline().bind(resources=config)
            >>> result = await pipeline("Hello!")

        Example with batch execution:
            >>> results = await pipeline(["input1", "input2", "input3"])

        Note:
            This method is called internally by __call__ when resources
            are available. Users should not call it directly.
        """
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

        # Handle batch execution
        if args and isinstance(args[0], list):
            inputs = args[0]
            results = []
            for inp in inputs:
                result = await run(
                    self,
                    inp,
                    *args[1:],
                    resources=resources,
                    **forward_kwargs,
                    **effective_config,
                )
                results.append(result)
            return results

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
                self.system_prompt = Parameter(system_prompt, requires_grad=False)
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
