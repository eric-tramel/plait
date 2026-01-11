"""Container classes for organizing Parameters and Modules.

This module provides PyTorch-like container classes that enable proper
parameter and module collection when storing multiple items in list or
dict structures.

The containers are:
- ParameterList: Holds a list of Parameters
- ParameterDict: Holds a dict of Parameters
- ModuleList: Holds a list of Modules
- ModuleDict: Holds a dict of Modules

These containers integrate with Module's introspection methods (parameters(),
named_parameters(), children(), named_children(), modules()) to ensure
all contained items are properly discovered during traversal.

Example:
    >>> class MultiPrompt(Module):
    ...     def __init__(self, n_prompts: int):
    ...         super().__init__()
    ...         self.prompts = ParameterList([
    ...             Parameter(f"Prompt {i}", description=f"Prompt #{i}")
    ...             for i in range(n_prompts)
    ...         ])
    ...
    >>> module = MultiPrompt(3)
    >>> len(list(module.parameters()))  # All 3 parameters are collected
    3
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableMapping, MutableSequence
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from plait.module import Module
    from plait.parameter import Parameter


class ParameterList(MutableSequence["Parameter"]):
    """A list-like container for Parameters.

    Holds a list of Parameters that will be properly collected by
    Module.parameters() and Module.named_parameters(). This is analogous
    to torch.nn.ParameterList.

    Parameters are named by their index in the list (e.g., "0", "1", "2").

    Args:
        parameters: Optional iterable of Parameter objects to initialize with.

    Example:
        >>> class MultiPrompt(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.prompts = ParameterList([
        ...             Parameter("Be concise", description="Style prompt"),
        ...             Parameter("Be helpful", description="Tone prompt"),
        ...         ])
        ...
        >>> m = MultiPrompt()
        >>> list(m.named_parameters())
        [('prompts.0', Parameter(...)), ('prompts.1', Parameter(...))]

    Note:
        The container itself is not a Parameter, but it provides iteration
        methods that Module uses to collect the contained Parameters.
    """

    _parameters: list[Parameter]
    _name: str | None
    _parent: Module | None

    def __init__(self, parameters: Iterable[Parameter] | None = None) -> None:
        """Initialize the ParameterList.

        Args:
            parameters: Optional iterable of Parameter objects.
        """
        self._parameters = []
        self._name = None
        self._parent = None

        if parameters is not None:
            for param in parameters:
                self.append(param)

    def _set_param_name(self, idx: int, param: Parameter) -> None:
        """Set the parameter's name based on its index.

        Args:
            idx: The index of the parameter in the list.
            param: The parameter to name.
        """
        object.__setattr__(param, "_name", str(idx))
        object.__setattr__(param, "_parent", self._parent)

    # MutableSequence abstract methods

    @overload
    def __getitem__(self, index: int) -> Parameter: ...

    @overload
    def __getitem__(self, index: slice) -> list[Parameter]: ...

    def __getitem__(self, index: int | slice) -> Parameter | list[Parameter]:
        """Get parameter(s) by index.

        Args:
            index: Integer index or slice.

        Returns:
            Single Parameter for int index, list for slice.
        """
        return self._parameters[index]

    @overload
    def __setitem__(self, index: int, value: Parameter) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Parameter]) -> None: ...

    def __setitem__(
        self, index: int | slice, value: Parameter | Iterable[Parameter]
    ) -> None:
        """Set parameter(s) by index.

        Args:
            index: Integer index or slice.
            value: Parameter or iterable of Parameters.
        """
        from plait.parameter import Parameter as ParameterClass

        if isinstance(index, int):
            if not isinstance(value, ParameterClass):
                raise TypeError(
                    f"ParameterList only accepts Parameter objects, got {type(value)}"
                )
            self._parameters[index] = value
            self._set_param_name(index, value)
        else:
            # Handle slice assignment
            if not isinstance(value, Iterable):
                raise TypeError(
                    f"Expected iterable for slice assignment, got {type(value)}"
                )
            params = list(value)
            for p in params:
                if not isinstance(p, ParameterClass):
                    raise TypeError(
                        f"ParameterList only accepts Parameter objects, got {type(p)}"
                    )
            self._parameters[index] = params
            # Re-index all parameters after slice assignment
            for i, p in enumerate(self._parameters):
                self._set_param_name(i, p)

    @overload
    def __delitem__(self, index: int) -> None: ...

    @overload
    def __delitem__(self, index: slice) -> None: ...

    def __delitem__(self, index: int | slice) -> None:
        """Delete parameter(s) by index.

        Args:
            index: Integer index or slice.
        """
        del self._parameters[index]
        # Re-index remaining parameters
        for i, p in enumerate(self._parameters):
            self._set_param_name(i, p)

    def __len__(self) -> int:
        """Return the number of parameters.

        Returns:
            Number of parameters in the list.
        """
        return len(self._parameters)

    def insert(self, index: int, value: Parameter) -> None:
        """Insert a parameter at the given index.

        Args:
            index: Index to insert at.
            value: Parameter to insert.

        Raises:
            TypeError: If value is not a Parameter.
        """
        from plait.parameter import Parameter as ParameterClass

        if not isinstance(value, ParameterClass):
            raise TypeError(
                f"ParameterList only accepts Parameter objects, got {type(value)}"
            )
        self._parameters.insert(index, value)
        # Re-index all parameters from index onwards
        for i in range(index, len(self._parameters)):
            self._set_param_name(i, self._parameters[i])

    # Parameter iteration for Module integration

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Iterate over parameters with their names.

        Args:
            prefix: Prefix to prepend to parameter names.

        Yields:
            Tuples of (name, parameter).
        """
        for i, param in enumerate(self._parameters):
            name = f"{prefix}.{i}" if prefix else str(i)
            yield name, param

    def parameters(self) -> Iterator[Parameter]:
        """Iterate over all parameters.

        Yields:
            Each Parameter in the list.
        """
        yield from self._parameters

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the ParameterList.
        """
        return f"ParameterList({self._parameters!r})"


class ParameterDict(MutableMapping[str, "Parameter"]):
    """A dict-like container for Parameters.

    Holds a dictionary of Parameters that will be properly collected by
    Module.parameters() and Module.named_parameters(). This is analogous
    to torch.nn.ParameterDict.

    Args:
        parameters: Optional mapping or iterable of (key, Parameter) pairs.

    Example:
        >>> class MultiTask(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.prompts = ParameterDict({
        ...             "summarize": Parameter("Summarize:", description="Summary prompt"),
        ...             "translate": Parameter("Translate:", description="Translation prompt"),
        ...         })
        ...
        >>> m = MultiTask()
        >>> list(m.named_parameters())
        [('prompts.summarize', Parameter(...)), ('prompts.translate', Parameter(...))]

    Note:
        The container itself is not a Parameter, but it provides iteration
        methods that Module uses to collect the contained Parameters.
    """

    _parameters: dict[str, Parameter]
    _name: str | None
    _parent: Module | None

    def __init__(
        self,
        parameters: dict[str, Parameter]
        | Iterable[tuple[str, Parameter]]
        | None = None,
    ) -> None:
        """Initialize the ParameterDict.

        Args:
            parameters: Optional dict or iterable of (key, Parameter) pairs.
        """
        self._parameters: dict[str, Parameter] = {}
        self._name = None
        self._parent = None

        if parameters is not None:
            # Convert to list of tuples for uniform handling
            items: list[tuple[str, Parameter]]
            if isinstance(parameters, dict):
                # Type narrowing doesn't work through isinstance for generic dict
                items = list(parameters.items())  # type: ignore[assignment]
            else:
                items = list(parameters)
            for key, param in items:
                self._parameters[key] = param
                self._set_param_name(key, param)

    def _set_param_name(self, key: str, param: Parameter) -> None:
        """Set the parameter's name based on its key.

        Args:
            key: The key of the parameter in the dict.
            param: The parameter to name.
        """
        object.__setattr__(param, "_name", key)
        object.__setattr__(param, "_parent", self._parent)

    # MutableMapping abstract methods

    def __getitem__(self, key: str) -> Parameter:
        """Get a parameter by key.

        Args:
            key: The parameter's key.

        Returns:
            The Parameter at the given key.
        """
        return self._parameters[key]

    def __setitem__(self, key: str, value: Parameter) -> None:
        """Set a parameter by key.

        Args:
            key: The parameter's key.
            value: The Parameter to store.

        Raises:
            TypeError: If value is not a Parameter.
        """
        from plait.parameter import Parameter as ParameterClass

        if not isinstance(value, ParameterClass):
            raise TypeError(
                f"ParameterDict only accepts Parameter objects, got {type(value)}"
            )
        self._parameters[key] = value
        self._set_param_name(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete a parameter by key.

        Args:
            key: The parameter's key.
        """
        del self._parameters[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter keys.

        Yields:
            Each key in the dict.
        """
        return iter(self._parameters)

    def __len__(self) -> int:
        """Return the number of parameters.

        Returns:
            Number of parameters in the dict.
        """
        return len(self._parameters)

    # Parameter iteration for Module integration

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Iterate over parameters with their names.

        Args:
            prefix: Prefix to prepend to parameter names.

        Yields:
            Tuples of (name, parameter).
        """
        for key, param in self._parameters.items():
            name = f"{prefix}.{key}" if prefix else key
            yield name, param

    def parameters(self) -> Iterator[Parameter]:
        """Iterate over all parameters.

        Yields:
            Each Parameter in the dict.
        """
        yield from self._parameters.values()

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the ParameterDict.
        """
        return f"ParameterDict({self._parameters!r})"


class ModuleList(MutableSequence["Module"]):
    """A list-like container for Modules.

    Holds a list of Modules that will be properly collected by
    Module.children(), Module.modules(), etc. This is analogous
    to torch.nn.ModuleList.

    Modules are named by their index in the list (e.g., "0", "1", "2").

    Args:
        modules: Optional iterable of Module objects to initialize with.

    Example:
        >>> class Pipeline(Module):
        ...     def __init__(self, n_stages: int):
        ...         super().__init__()
        ...         self.stages = ModuleList([
        ...             LLMInference(alias=f"stage_{i}")
        ...             for i in range(n_stages)
        ...         ])
        ...
        >>> p = Pipeline(3)
        >>> len(list(p.modules()))  # self + 3 stages
        4

    Note:
        The container itself is a Module, allowing it to participate
        in the module tree structure.
    """

    _modules: list[Module]
    _name: str | None
    _parent: Module | None

    def __init__(self, modules: Iterable[Module] | None = None) -> None:
        """Initialize the ModuleList.

        Args:
            modules: Optional iterable of Module objects.
        """
        # Import here to avoid circular import
        from plait.module import Module as ModuleClass

        self._modules = []
        self._name = None
        self._parent = None
        # We need _children and _parameters for Module compatibility
        self._children: dict[str, Module] = {}
        self._parameters: dict[str, Parameter] = {}

        if modules is not None:
            for mod in modules:
                if not isinstance(mod, ModuleClass):
                    raise TypeError(
                        f"ModuleList only accepts Module objects, got {type(mod)}"
                    )
                self.append(mod)

    def _set_module_name(self, idx: int, mod: Module) -> None:
        """Set the module's name based on its index.

        Args:
            idx: The index of the module in the list.
            mod: The module to name.
        """
        object.__setattr__(mod, "_name", str(idx))
        object.__setattr__(mod, "_parent", self._parent)

    # MutableSequence abstract methods

    @overload
    def __getitem__(self, index: int) -> Module: ...

    @overload
    def __getitem__(self, index: slice) -> list[Module]: ...

    def __getitem__(self, index: int | slice) -> Module | list[Module]:
        """Get module(s) by index.

        Args:
            index: Integer index or slice.

        Returns:
            Single Module for int index, list for slice.
        """
        return self._modules[index]

    @overload
    def __setitem__(self, index: int, value: Module) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Module]) -> None: ...

    def __setitem__(self, index: int | slice, value: Module | Iterable[Module]) -> None:
        """Set module(s) by index.

        Args:
            index: Integer index or slice.
            value: Module or iterable of Modules.
        """
        from plait.module import Module as ModuleClass

        if isinstance(index, int):
            if not isinstance(value, ModuleClass):
                raise TypeError(
                    f"ModuleList only accepts Module objects, got {type(value)}"
                )
            self._modules[index] = value
            self._set_module_name(index, value)
        else:
            # Handle slice assignment
            if not isinstance(value, Iterable):
                raise TypeError(
                    f"Expected iterable for slice assignment, got {type(value)}"
                )
            mods = list(value)
            for m in mods:
                if not isinstance(m, ModuleClass):
                    raise TypeError(
                        f"ModuleList only accepts Module objects, got {type(m)}"
                    )
            self._modules[index] = mods
            # Re-index all modules after slice assignment
            for i, m in enumerate(self._modules):
                self._set_module_name(i, m)

    @overload
    def __delitem__(self, index: int) -> None: ...

    @overload
    def __delitem__(self, index: slice) -> None: ...

    def __delitem__(self, index: int | slice) -> None:
        """Delete module(s) by index.

        Args:
            index: Integer index or slice.
        """
        del self._modules[index]
        # Re-index remaining modules
        for i, m in enumerate(self._modules):
            self._set_module_name(i, m)

    def __len__(self) -> int:
        """Return the number of modules.

        Returns:
            Number of modules in the list.
        """
        return len(self._modules)

    def insert(self, index: int, value: Module) -> None:
        """Insert a module at the given index.

        Args:
            index: Index to insert at.
            value: Module to insert.

        Raises:
            TypeError: If value is not a Module.
        """
        from plait.module import Module as ModuleClass

        if not isinstance(value, ModuleClass):
            raise TypeError(
                f"ModuleList only accepts Module objects, got {type(value)}"
            )
        self._modules.insert(index, value)
        # Re-index all modules from index onwards
        for i in range(index, len(self._modules)):
            self._set_module_name(i, self._modules[i])

    # Module iteration for Module integration

    def children(self) -> Iterator[Module]:
        """Iterate over child modules.

        Yields:
            Each Module in the list.
        """
        yield from self._modules

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Iterate over child modules with names.

        Yields:
            Tuples of (name, module).
        """
        for i, mod in enumerate(self._modules):
            yield str(i), mod

    def modules(self) -> Iterator[Module]:
        """Iterate over all modules (including nested).

        Yields:
            Each Module in the list and their descendants.
        """
        for mod in self._modules:
            yield from mod.modules()

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        """Iterate over all modules with hierarchical names.

        Args:
            prefix: Prefix to prepend to module names.

        Yields:
            Tuples of (hierarchical_name, module).
        """
        for i, mod in enumerate(self._modules):
            child_prefix = f"{prefix}.{i}" if prefix else str(i)
            yield from mod.named_modules(child_prefix)

    def parameters(self) -> Iterator[Parameter]:
        """Iterate over all parameters in child modules.

        Yields:
            Parameters from all child modules.
        """
        for mod in self._modules:
            yield from mod.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Iterate over all parameters with hierarchical names.

        Args:
            prefix: Prefix to prepend to parameter names.

        Yields:
            Tuples of (hierarchical_name, parameter).
        """
        for i, mod in enumerate(self._modules):
            child_prefix = f"{prefix}.{i}" if prefix else str(i)
            yield from mod.named_parameters(child_prefix)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the ModuleList.
        """
        return f"ModuleList({self._modules!r})"


class ModuleDict(MutableMapping[str, "Module"]):
    """A dict-like container for Modules.

    Holds a dictionary of Modules that will be properly collected by
    Module.children(), Module.modules(), etc. This is analogous
    to torch.nn.ModuleDict.

    Args:
        modules: Optional mapping or iterable of (key, Module) pairs.

    Example:
        >>> class MultiTask(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.tasks = ModuleDict({
        ...             "summarize": LLMInference(alias="summarizer"),
        ...             "translate": LLMInference(alias="translator"),
        ...         })
        ...
        >>> m = MultiTask()
        >>> len(list(m.children()))  # The ModuleDict's children
        2

    Note:
        The container itself is a Module, allowing it to participate
        in the module tree structure.
    """

    _modules: dict[str, Module]
    _name: str | None
    _parent: Module | None

    def __init__(
        self,
        modules: dict[str, Module] | Iterable[tuple[str, Module]] | None = None,
    ) -> None:
        """Initialize the ModuleDict.

        Args:
            modules: Optional dict or iterable of (key, Module) pairs.
        """
        from plait.module import Module as ModuleClass

        self._modules: dict[str, Module] = {}
        self._name = None
        self._parent = None
        # We need _children and _parameters for Module compatibility
        self._children: dict[str, Module] = {}
        self._parameters: dict[str, Parameter] = {}

        if modules is not None:
            # Convert to list of tuples for uniform handling
            items: list[tuple[str, Module]]
            if isinstance(modules, dict):
                # Type narrowing doesn't work through isinstance for generic dict
                items = list(modules.items())  # type: ignore[assignment]
            else:
                items = list(modules)
            for key, mod in items:
                if not isinstance(mod, ModuleClass):
                    raise TypeError(
                        f"ModuleDict only accepts Module objects, got {type(mod)}"
                    )
                self._modules[key] = mod
                self._set_module_name(key, mod)

    def _set_module_name(self, key: str, mod: Module) -> None:
        """Set the module's name based on its key.

        Args:
            key: The key of the module in the dict.
            mod: The module to name.
        """
        object.__setattr__(mod, "_name", key)
        object.__setattr__(mod, "_parent", self._parent)

    # MutableMapping abstract methods

    def __getitem__(self, key: str) -> Module:
        """Get a module by key.

        Args:
            key: The module's key.

        Returns:
            The Module at the given key.
        """
        return self._modules[key]

    def __setitem__(self, key: str, value: Module) -> None:
        """Set a module by key.

        Args:
            key: The module's key.
            value: The Module to store.

        Raises:
            TypeError: If value is not a Module.
        """
        from plait.module import Module as ModuleClass

        if not isinstance(value, ModuleClass):
            raise TypeError(
                f"ModuleDict only accepts Module objects, got {type(value)}"
            )
        self._modules[key] = value
        self._set_module_name(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete a module by key.

        Args:
            key: The module's key.
        """
        del self._modules[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over module keys.

        Yields:
            Each key in the dict.
        """
        return iter(self._modules)

    def __len__(self) -> int:
        """Return the number of modules.

        Returns:
            Number of modules in the dict.
        """
        return len(self._modules)

    # Module iteration for Module integration

    def children(self) -> Iterator[Module]:
        """Iterate over child modules.

        Yields:
            Each Module in the dict.
        """
        yield from self._modules.values()

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Iterate over child modules with names.

        Yields:
            Tuples of (name, module).
        """
        yield from self._modules.items()

    def modules(self) -> Iterator[Module]:
        """Iterate over all modules (including nested).

        Yields:
            Each Module in the dict and their descendants.
        """
        for mod in self._modules.values():
            yield from mod.modules()

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        """Iterate over all modules with hierarchical names.

        Args:
            prefix: Prefix to prepend to module names.

        Yields:
            Tuples of (hierarchical_name, module).
        """
        for key, mod in self._modules.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            yield from mod.named_modules(child_prefix)

    def parameters(self) -> Iterator[Parameter]:
        """Iterate over all parameters in child modules.

        Yields:
            Parameters from all child modules.
        """
        for mod in self._modules.values():
            yield from mod.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Iterate over all parameters with hierarchical names.

        Args:
            prefix: Prefix to prepend to parameter names.

        Yields:
            Tuples of (hierarchical_name, parameter).
        """
        for key, mod in self._modules.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            yield from mod.named_parameters(child_prefix)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the ModuleDict.
        """
        return f"ModuleDict({self._modules!r})"
