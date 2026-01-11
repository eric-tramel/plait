"""Container modules for composing and organizing modules.

This module provides PyTorch-style container modules for building
complex inference pipelines:

- Sequential: Chains modules together, passing output to input
- ModuleList: List-like container for dynamic module management
- ModuleDict: Dict-like container for named module access
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator
from typing import Any, cast, overload

from plait.module import Module


class Sequential(Module):
    """A sequential container that chains modules together.

    Modules are executed in order, with each module's output passed
    as input to the next module. This mirrors PyTorch's nn.Sequential.

    Supports two initialization styles:
    1. Positional arguments: Sequential(mod1, mod2, mod3)
    2. OrderedDict for named access: Sequential(OrderedDict([('name', mod)]))

    When using OrderedDict, modules can be accessed by name as attributes.

    Args:
        *args: Either positional Module instances, or a single OrderedDict
            mapping names to modules.

    Example:
        >>> # Positional args
        >>> pipeline = Sequential(
        ...     Preprocessor(),
        ...     Analyzer(),
        ...     Formatter()
        ... )
        >>> len(pipeline)
        3
        >>> pipeline[0]  # First module
        <Preprocessor>

    Example with OrderedDict:
        >>> from collections import OrderedDict
        >>> pipeline = Sequential(OrderedDict([
        ...     ('preprocess', Preprocessor()),
        ...     ('analyze', Analyzer()),
        ... ]))
        >>> pipeline.preprocess  # Named access
        <Preprocessor>

    Note:
        The forward() method passes the input through each module
        in sequence, so each module must accept a single argument
        matching the previous module's output type.
    """

    _modules: dict[str, Module]

    def __init__(
        self,
        *args: Module | OrderedDict[str, Module],
    ) -> None:
        """Initialize the Sequential container.

        Args:
            *args: Either positional Module instances, or a single OrderedDict
                mapping names to modules.

        Raises:
            TypeError: If a non-Module argument is provided (except OrderedDict).
            ValueError: If OrderedDict is provided with positional args.
        """
        super().__init__()
        # Use ordered dict to maintain insertion order
        object.__setattr__(self, "_modules", OrderedDict())

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            # OrderedDict initialization - cast to typed dict
            ordered_modules = cast(OrderedDict[str, Module], args[0])
            for name, module in ordered_modules.items():
                self._add_module(name, module)
        else:
            # Positional args initialization
            for idx, module in enumerate(args):
                if not isinstance(module, Module):
                    raise TypeError(
                        f"Sequential argument {idx} must be a Module, "
                        f"got {type(module).__name__}"
                    )
                self._add_module(str(idx), module)

    def _add_module(self, name: str, module: Module) -> None:
        """Add a module with the given name.

        Args:
            name: The name to register the module under.
            module: The module to add.

        Raises:
            TypeError: If module is not a Module instance.
        """
        if not isinstance(module, Module):
            raise TypeError(
                f"Sequential only accepts Module instances, got {type(module).__name__}"
            )
        self._modules[name] = module
        # Register as child for proper parameter collection
        setattr(self, name, module)

    def __getattr__(self, name: str) -> Module:
        """Get a module by name for named Sequential containers.

        Args:
            name: The name of the module to retrieve.

        Returns:
            The module with the given name.

        Raises:
            AttributeError: If no module with that name exists.
        """
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @overload
    def __getitem__(self, idx: int) -> Module: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequential: ...

    def __getitem__(self, idx: int | slice) -> Module | Sequential:
        """Get module(s) by index or slice.

        Args:
            idx: Integer index or slice.

        Returns:
            Single module for integer index, new Sequential for slice.

        Raises:
            IndexError: If index is out of range.
            TypeError: If idx is not int or slice.
        """
        if isinstance(idx, slice):
            keys = list(self._modules.keys())[idx]
            return Sequential(OrderedDict((k, self._modules[k]) for k in keys))
        elif isinstance(idx, int):
            if idx < 0:
                idx += len(self._modules)
            if idx < 0 or idx >= len(self._modules):
                raise IndexError(f"index {idx} out of range")
            return list(self._modules.values())[idx]
        else:
            raise TypeError(f"indices must be integers or slices, not {type(idx)}")

    def __len__(self) -> int:
        """Return the number of modules in the container.

        Returns:
            The number of modules.
        """
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        """Iterate over modules in order.

        Yields:
            Each module in the sequence.
        """
        return iter(self._modules.values())

    def forward(self, x: Any) -> Any:
        """Execute modules sequentially, chaining outputs to inputs.

        Args:
            x: Input to the first module.

        Returns:
            Output of the last module.

        Note:
            Each module's output becomes the next module's input.
        """
        for module in self._modules.values():
            x = module(x)
        return x

    def append(self, module: Module) -> Sequential:
        """Append a module to the end of the sequence.

        Args:
            module: The module to append.

        Returns:
            Self, for method chaining.

        Raises:
            TypeError: If module is not a Module instance.
        """
        self._add_module(str(len(self._modules)), module)
        return self


class ModuleList(Module):
    """A list-like container for modules.

    Provides list-like operations (append, extend, insert, etc.) while
    properly registering modules for parameter collection. This mirrors
    PyTorch's nn.ModuleList.

    Args:
        modules: Optional iterable of modules to initialize with.

    Example:
        >>> layers = ModuleList([Layer() for _ in range(3)])
        >>> for layer in layers:
        ...     x = layer(x)
        >>> layers.append(AnotherLayer())
        >>> len(layers)
        4

    Note:
        ModuleList does not define a forward() method since the
        iteration pattern depends on use case. Use iteration or
        indexing to access and execute modules.
    """

    _modules: dict[str, Module]

    def __init__(self, modules: Iterable[Module] | None = None) -> None:
        """Initialize the ModuleList container.

        Args:
            modules: Optional iterable of modules to add.

        Raises:
            TypeError: If any item in modules is not a Module.
        """
        super().__init__()
        object.__setattr__(self, "_modules", OrderedDict())
        if modules is not None:
            self.extend(modules)

    def _add_module(self, name: str, module: Module) -> None:
        """Add a module with the given name.

        Args:
            name: The name to register the module under.
            module: The module to add.

        Raises:
            TypeError: If module is not a Module instance.
        """
        if not isinstance(module, Module):
            raise TypeError(
                f"ModuleList only accepts Module instances, got {type(module).__name__}"
            )
        self._modules[name] = module
        setattr(self, name, module)

    def _reindex(self) -> None:
        """Reindex modules after insertion or deletion.

        Rebuilds the _modules dict with sequential integer keys.
        """
        modules = list(self._modules.values())
        # Clear children and modules
        self._children.clear()
        self._modules.clear()
        # Re-add with new indices
        for idx, module in enumerate(modules):
            self._add_module(str(idx), module)

    @overload
    def __getitem__(self, idx: int) -> Module: ...

    @overload
    def __getitem__(self, idx: slice) -> ModuleList: ...

    def __getitem__(self, idx: int | slice) -> Module | ModuleList:
        """Get module(s) by index or slice.

        Args:
            idx: Integer index or slice.

        Returns:
            Single module for integer index, new ModuleList for slice.

        Raises:
            IndexError: If index is out of range.
        """
        if isinstance(idx, slice):
            modules = list(self._modules.values())[idx]
            return ModuleList(modules)
        else:
            if idx < 0:
                idx += len(self._modules)
            if idx < 0 or idx >= len(self._modules):
                raise IndexError(f"index {idx} out of range")
            return list(self._modules.values())[idx]

    def __setitem__(self, idx: int, module: Module) -> None:
        """Set a module at the given index.

        Args:
            idx: The index to set.
            module: The module to set.

        Raises:
            IndexError: If index is out of range.
            TypeError: If module is not a Module instance.
        """
        if not isinstance(module, Module):
            raise TypeError(
                f"ModuleList only accepts Module instances, got {type(module).__name__}"
            )
        if idx < 0:
            idx += len(self._modules)
        if idx < 0 or idx >= len(self._modules):
            raise IndexError(f"index {idx} out of range")
        key = str(idx)
        self._modules[key] = module
        setattr(self, key, module)

    def __delitem__(self, idx: int) -> None:
        """Delete a module at the given index.

        Args:
            idx: The index to delete.

        Raises:
            IndexError: If index is out of range.
        """
        if idx < 0:
            idx += len(self._modules)
        if idx < 0 or idx >= len(self._modules):
            raise IndexError(f"index {idx} out of range")
        del self._modules[str(idx)]
        self._reindex()

    def __len__(self) -> int:
        """Return the number of modules in the list.

        Returns:
            The number of modules.
        """
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        """Iterate over modules in order.

        Yields:
            Each module in the list.
        """
        return iter(self._modules.values())

    def __contains__(self, module: Module) -> bool:
        """Check if a module is in the list.

        Args:
            module: The module to check for.

        Returns:
            True if module is in the list, False otherwise.
        """
        return module in self._modules.values()

    def append(self, module: Module) -> ModuleList:
        """Append a module to the end of the list.

        Args:
            module: The module to append.

        Returns:
            Self, for method chaining.

        Raises:
            TypeError: If module is not a Module instance.
        """
        self._add_module(str(len(self._modules)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> ModuleList:
        """Extend the list with modules from an iterable.

        Args:
            modules: Iterable of modules to add.

        Returns:
            Self, for method chaining.

        Raises:
            TypeError: If any item is not a Module instance.
        """
        for module in modules:
            self.append(module)
        return self

    def insert(self, idx: int, module: Module) -> None:
        """Insert a module at the given index.

        Args:
            idx: The index to insert at.
            module: The module to insert.

        Raises:
            TypeError: If module is not a Module instance.
        """
        if not isinstance(module, Module):
            raise TypeError(
                f"ModuleList only accepts Module instances, got {type(module).__name__}"
            )
        # Handle negative indices
        if idx < 0:
            idx = max(0, len(self._modules) + idx + 1)
        # Clamp to valid range
        idx = min(idx, len(self._modules))

        # Insert by rebuilding the list
        modules = list(self._modules.values())
        modules.insert(idx, module)

        # Clear and re-add
        self._children.clear()
        self._modules.clear()
        for i, mod in enumerate(modules):
            self._add_module(str(i), mod)

    def pop(self, idx: int = -1) -> Module:
        """Remove and return the module at the given index.

        Args:
            idx: The index to pop from. Defaults to -1 (last).

        Returns:
            The removed module.

        Raises:
            IndexError: If index is out of range or list is empty.
        """
        if len(self._modules) == 0:
            raise IndexError("pop from empty ModuleList")
        if idx < 0:
            idx += len(self._modules)
        if idx < 0 or idx >= len(self._modules):
            raise IndexError(f"index {idx} out of range")

        module = list(self._modules.values())[idx]
        del self._modules[str(idx)]
        self._reindex()
        return module

    def forward(self, x: Any) -> Any:
        """Forward is not implemented for ModuleList.

        ModuleList does not define a forward method since the
        iteration pattern depends on use case.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "ModuleList does not implement forward(). "
            "Iterate over modules manually in your forward() method."
        )


class ModuleDict(Module):
    """A dict-like container for named module access.

    Provides dict-like operations (keys, values, items, etc.) while
    properly registering modules for parameter collection. This mirrors
    PyTorch's nn.ModuleDict.

    Args:
        modules: Optional dict or iterable of (key, module) pairs.

    Example:
        >>> modules = ModuleDict({
        ...     'encoder': Encoder(),
        ...     'decoder': Decoder()
        ... })
        >>> output = modules['encoder'](input)
        >>> modules.keys()
        dict_keys(['encoder', 'decoder'])

    Note:
        ModuleDict does not define a forward() method since access
        patterns depend on use case. Access modules by key and
        call them directly.
    """

    _modules: dict[str, Module]

    def __init__(
        self,
        modules: dict[str, Module] | Iterable[tuple[str, Module]] | None = None,
    ) -> None:
        """Initialize the ModuleDict container.

        Args:
            modules: Optional dict or iterable of (key, module) pairs.

        Raises:
            TypeError: If any value is not a Module instance.
        """
        super().__init__()
        object.__setattr__(self, "_modules", OrderedDict())
        if modules is not None:
            self.update(modules)

    def _add_module(self, name: str, module: Module) -> None:
        """Add a module with the given name.

        Args:
            name: The name to register the module under.
            module: The module to add.

        Raises:
            TypeError: If module is not a Module instance.
        """
        if not isinstance(module, Module):
            raise TypeError(
                f"ModuleDict only accepts Module instances, got {type(module).__name__}"
            )
        self._modules[name] = module
        # Register as child - but only if name is a valid identifier
        if name.isidentifier():
            setattr(self, name, module)
        else:
            # For non-identifier keys, register directly in _children
            self._children[name] = module
            object.__setattr__(module, "_name", name)
            object.__setattr__(module, "_parent", self)

    def __getitem__(self, key: str) -> Module:
        """Get a module by key.

        Args:
            key: The key of the module to retrieve.

        Returns:
            The module with the given key.

        Raises:
            KeyError: If key is not in the dict.
        """
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        """Set a module at the given key.

        Args:
            key: The key to set.
            module: The module to set.

        Raises:
            TypeError: If module is not a Module instance.
        """
        self._add_module(key, module)

    def __delitem__(self, key: str) -> None:
        """Delete a module by key.

        Args:
            key: The key to delete.

        Raises:
            KeyError: If key is not in the dict.
        """
        if key not in self._modules:
            raise KeyError(key)
        del self._modules[key]
        if key in self._children:
            del self._children[key]

    def __len__(self) -> int:
        """Return the number of modules in the dict.

        Returns:
            The number of modules.
        """
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        """Iterate over module keys.

        Yields:
            Each key in the dict.
        """
        return iter(self._modules)

    def __contains__(self, key: object) -> bool:
        """Check if a key is in the dict.

        Args:
            key: The key to check for.

        Returns:
            True if key is in the dict, False otherwise.
        """
        return key in self._modules

    def __getattr__(self, name: str) -> Module:
        """Get a module by name as attribute access.

        Args:
            name: The name of the module to retrieve.

        Returns:
            The module with the given name.

        Raises:
            AttributeError: If no module with that name exists.
        """
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def keys(self) -> Any:
        """Return a view of the module keys.

        Returns:
            A dict_keys view of the keys.
        """
        return self._modules.keys()

    def values(self) -> Any:
        """Return a view of the modules.

        Returns:
            A dict_values view of the modules.
        """
        return self._modules.values()

    def items(self) -> Any:
        """Return a view of the (key, module) pairs.

        Returns:
            A dict_items view of the items.
        """
        return self._modules.items()

    def update(
        self,
        modules: dict[str, Module] | Iterable[tuple[str, Module]],
    ) -> None:
        """Update the dict with modules from another dict or iterable.

        Args:
            modules: Dict or iterable of (key, module) pairs.

        Raises:
            TypeError: If any value is not a Module instance.
        """
        if isinstance(modules, dict):
            typed_modules = cast(dict[str, Module], modules)
            for key, module in typed_modules.items():
                self._add_module(key, module)
        else:
            for key, module in modules:
                self._add_module(key, module)

    def pop(self, key: str, default: Module | None = None) -> Module | None:
        """Remove and return a module by key.

        Args:
            key: The key to pop.
            default: Value to return if key is not found.

        Returns:
            The removed module, or default if not found.
        """
        if key in self._modules:
            module = self._modules.pop(key)
            if key in self._children:
                del self._children[key]
            return module
        return default

    def clear(self) -> None:
        """Remove all modules from the dict."""
        self._modules.clear()
        self._children.clear()

    def forward(self, x: Any) -> Any:
        """Forward is not implemented for ModuleDict.

        ModuleDict does not define a forward method since access
        patterns depend on use case.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "ModuleDict does not implement forward(). "
            "Access modules by key in your forward() method."
        )
