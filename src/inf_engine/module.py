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
