"""InferenceModule base class for inf-engine.

This module provides the core abstraction for building composable
inference pipelines, inspired by PyTorch's nn.Module.
"""

from __future__ import annotations

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
