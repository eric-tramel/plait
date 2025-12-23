"""InferenceModule base class for inf-engine.

This module provides the core abstraction for building composable
inference pipelines, inspired by PyTorch's nn.Module.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inf_engine.parameter import Parameter


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

    def __init__(self) -> None:
        """Initialize the module with empty registries.

        Sets up internal dictionaries for tracking child modules and
        parameters. Uses object.__setattr__ to avoid triggering the
        custom __setattr__ during initialization.
        """
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_name", None)

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

        Delegates to forward() to perform the actual computation.
        In the future, this will also handle trace context for
        automatic DAG capture during tracing.

        Args:
            *args: Positional arguments passed to forward().
            **kwargs: Keyword arguments passed to forward().

        Returns:
            The result from forward().

        Example:
            >>> class Doubler(InferenceModule):
            ...     def forward(self, x: int) -> int:
            ...         return x * 2
            ...
            >>> doubler = Doubler()
            >>> doubler(5)
            10
        """
        return self.forward(*args, **kwargs)


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
