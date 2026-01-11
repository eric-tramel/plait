"""Value container and helper functions for plait.

This module provides the Value type used to represent data flowing through
inference graphs, along with helper functions for wrapping, unwrapping, and
collecting references from nested structures.

The Value type carries both payload and provenance, enabling explicit dependency
tracking without parsing positional/keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ValueKind(str, Enum):
    """Discriminant for Value payloads.

    Each kind indicates the type of data contained in a Value's payload,
    enabling appropriate downstream handling and formatting.
    """

    TEXT = "text"
    """Plain text string."""

    FSTRING = "fstring"
    """Format string with {name} slots."""

    RESPONSE = "response"
    """LLM response object with tokens and model info in meta."""

    STRUCTURED = "structured"
    """Structured data (dict, list, or schema-validated response)."""

    INT = "int"
    """Integer scalar value."""

    FLOAT = "float"
    """Floating-point scalar value."""

    ERROR = "error"
    """Error/exception payload with traceback in meta."""

    TOOL_RESULT = "tool_result"
    """Result from a tool invocation."""

    BINARY = "binary"
    """Binary data (bytes) for files/images."""

    OTHER = "other"
    """Fallback for unrecognized payload types."""


@dataclass
class Value:
    """Container for payload + provenance.

    Value is the primary data type flowing through inference graphs. It wraps
    raw payloads with type information (kind), graph provenance (ref), and
    optional metadata.

    Args:
        kind: The type of payload contained in this Value.
        payload: The raw value (string, dict, response object, exception, bytes).
        ref: Optional graph node ID that produced this value.
        meta: Optional metadata (model alias, tokens, schema, source, cost).

    Example:
        >>> v = Value(ValueKind.TEXT, "Hello, world!")
        >>> v.payload
        'Hello, world!'
        >>> v.kind
        <ValueKind.TEXT: 'text'>

        >>> v = Value(ValueKind.STRUCTURED, {"name": "Ada"}, ref="node_1")
        >>> v.ref
        'node_1'
    """

    kind: ValueKind
    payload: Any
    ref: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str | int) -> Value:
        """Graph-aware structured access (delegates to F.select).

        Accesses a nested element of the payload. When tracing is active,
        this records a select operation in the graph.

        Args:
            key: The key (for dicts) or index (for lists) to access.

        Returns:
            A new Value containing the selected element.

        Raises:
            KeyError: If the key does not exist in a dict payload.
            IndexError: If the index is out of range for a list payload.
            TypeError: If the payload type doesn't support indexing.

        Note:
            This method delegates to plait.functional.select when
            the functional module is available.
        """
        try:
            F = import_module("plait.functional")
        except ModuleNotFoundError:
            F = None

        if F is not None and hasattr(F, "select"):
            return F.select(self, key)

        # Direct implementation for now; will delegate to F.select in PR-061
        if self.kind == ValueKind.ERROR:
            return self  # Error propagation

        try:
            selected = self.payload[key]
        except (KeyError, IndexError, TypeError) as e:
            return Value(
                kind=ValueKind.ERROR,
                payload=e,
                meta={"source_ref": self.ref, "key": key},
            )

        if isinstance(selected, Value):
            return selected
        return Value(kind=_infer_kind(selected), payload=selected)

    def get(self, key: str | int, default: Any = None) -> Value:
        """Graph-aware structured access with default.

        Like __getitem__, but returns a default Value instead of raising
        an error when the key is not found.

        Args:
            key: The key (for dicts) or index (for lists) to access.
            default: Value to return if key is not found.

        Returns:
            A new Value containing the selected element or the default.

        Note:
            This method delegates to plait.functional.select when
            the functional module is available.
        """
        try:
            F = import_module("plait.functional")
        except ModuleNotFoundError:
            F = None

        if F is not None and hasattr(F, "select"):
            return F.select(self, key, default=default)

        # Direct implementation for now; will delegate to F.select in PR-061
        if self.kind == ValueKind.ERROR:
            return self  # Error propagation

        try:
            selected = self.payload[key]
            if isinstance(selected, Value):
                return selected
            return Value(kind=_infer_kind(selected), payload=selected)
        except (KeyError, IndexError, TypeError):
            if isinstance(default, Value):
                return default
            return Value(kind=_infer_kind(default), payload=default)


@dataclass(frozen=True)
class ValueRef:
    """Placeholder reference to a Value.

    During tracing, Value inputs are replaced with ValueRef placeholders inside
    node args/kwargs. Execution resolves these references to the producing
    node's result.

    Args:
        ref: The graph node ID or parameter reference string.

    Example:
        >>> ref = ValueRef("node_1")
        >>> ref.ref
        'node_1'
    """

    ref: str


def _infer_kind(x: Any) -> ValueKind:
    """Infer the appropriate ValueKind for a Python value.

    Args:
        x: The value to infer a kind for.

    Returns:
        The inferred ValueKind.
    """
    if isinstance(x, str):
        return ValueKind.TEXT
    elif isinstance(x, bool):
        # bool must be checked before int since bool is a subclass of int
        return ValueKind.OTHER
    elif isinstance(x, int):
        return ValueKind.INT
    elif isinstance(x, float):
        return ValueKind.FLOAT
    elif isinstance(x, bytes):
        return ValueKind.BINARY
    elif isinstance(x, dict):
        return ValueKind.STRUCTURED
    elif isinstance(x, (list, tuple)):
        return ValueKind.STRUCTURED
    elif isinstance(x, BaseException):
        return ValueKind.ERROR
    else:
        return ValueKind.OTHER


def valueify(x: Any, *, kind: ValueKind | None = None) -> Any:
    """Wrap raw values into Value with optional kind override.

    Normalizes arbitrary Python values into the Value container type.
    Handles primitives, containers, Parameters, and existing Values.

    Args:
        x: The value to wrap. Can be a primitive, container, Parameter,
           or existing Value.
        kind: Optional kind to use instead of inferring from the value type.

    Returns:
        A Value containing the input, or a container of Values for structured inputs.

    Example:
        >>> v = valueify("hello")
        >>> v.kind
        <ValueKind.TEXT: 'text'>
        >>> v.payload
        'hello'

        >>> v = valueify(42)
        >>> v.kind
        <ValueKind.INT: 'int'>

        >>> v = valueify({"key": "value"})
        >>> isinstance(v, dict)
        True
        >>> v["key"].kind
        <ValueKind.TEXT: 'text'>

    Note:
        - Parameters are lifted to Values with stable refs (param:<name> or param:<id>)
        - Existing Values are returned as-is unless kind is overridden
        - Containers (list, tuple, dict) are recursively wrapped into
          containers of Value objects when no kind override is provided.
    """
    # Import here to avoid circular imports
    from plait.parameter import Parameter

    # Already a Value - return as-is or re-wrap with new kind
    if isinstance(x, Value):
        if kind is not None and kind != x.kind:
            return Value(kind=kind, payload=x.payload, ref=x.ref, meta=x.meta.copy())
        return x

    # Parameter - lift to Value with stable ref
    if isinstance(x, Parameter):
        param_name = x._get_hierarchical_name()
        if param_name is None:
            param_name = x._name
        param_id = x._id
        module_state_version = 0
        if x._parent is not None:
            module_state_version = getattr(x._parent, "_module_state_version", 0)
        inferred_kind = kind
        if inferred_kind is None:
            # Infer kind from parameter value
            if isinstance(x.value, str):
                inferred_kind = ValueKind.TEXT
            elif isinstance(x.value, dict):
                inferred_kind = ValueKind.STRUCTURED
            elif isinstance(x.value, (list, tuple)):
                inferred_kind = ValueKind.STRUCTURED
            else:
                inferred_kind = _infer_kind(x.value)

        ref_name = param_name if param_name else param_id
        return Value(
            kind=inferred_kind,
            payload=x.value,
            ref=f"param:{ref_name}",
            meta={
                "param_name": param_name,
                "param_id": param_id,
                "module_state_version": module_state_version,
                "requires_grad": x.requires_grad,
            },
        )

    if kind is not None:
        return Value(kind=kind, payload=x)

    if isinstance(x, dict):
        return {k: valueify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [valueify(item) for item in x]
    if isinstance(x, tuple):
        return tuple(valueify(item) for item in x)

    # Raw value - wrap with inferred or provided kind
    inferred_kind = _infer_kind(x)
    return Value(kind=inferred_kind, payload=x)


def unwrap(x: Any) -> Any:
    """Return payload if x is Value, otherwise x unchanged.

    Extracts raw payloads from Values and recursively unwraps containers
    of Values back into raw Python structures.

    Args:
        x: A Value or any other object.

    Returns:
        The payload if x is a Value, otherwise x unchanged.

    Example:
        >>> v = Value(ValueKind.TEXT, "hello")
        >>> unwrap(v)
        'hello'

        >>> unwrap("already raw")
        'already raw'

        >>> unwrap(42)
        42
    """
    if isinstance(x, Value):
        return unwrap(x.payload)
    if isinstance(x, dict):
        return {k: unwrap(v) for k, v in x.items()}
    if isinstance(x, list):
        return [unwrap(item) for item in x]
    if isinstance(x, tuple):
        return tuple(unwrap(item) for item in x)
    return x


def collect_refs(*args: Any, **kwargs: Any) -> list[str]:
    """Recursively collect .ref from Values in nested structures.

    Traverses args and kwargs, finding all Value objects and collecting
    their ref attributes (if non-None). Handles nested lists, tuples,
    and dicts.

    Args:
        *args: Positional arguments to search for Values.
        **kwargs: Keyword arguments to search for Values.

    Returns:
        A list of unique ref strings found in the input structures.
        Order is not guaranteed.

    Example:
        >>> v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        >>> v2 = Value(ValueKind.TEXT, "b", ref="node_2")
        >>> refs = collect_refs(v1, v2)
        >>> sorted(refs)
        ['node_1', 'node_2']

        >>> refs = collect_refs([v1, {"nested": v2}])
        >>> sorted(refs)
        ['node_1', 'node_2']

        >>> v3 = Value(ValueKind.TEXT, "no ref")  # ref is None
        >>> refs = collect_refs(v1, v3)
        >>> refs
        ['node_1']
    """
    refs: set[str] = set()

    def _collect(obj: Any) -> None:
        if isinstance(obj, Value):
            if obj.ref is not None:
                refs.add(obj.ref)
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _collect(item)

    for arg in args:
        _collect(arg)
    for v in kwargs.values():
        _collect(v)

    return list(refs)


def replace_values_with_refs(obj: Any) -> Any:
    """Replace Value objects with ValueRef placeholders.

    Traverses a nested structure and replaces each Value that has a ref
    with a ValueRef placeholder. Values without refs are left unchanged.
    This is used during tracing to store dependencies in args/kwargs.

    Args:
        obj: The object to process. Can be a Value, list, tuple, dict,
             or any other type.

    Returns:
        A new structure with Values replaced by ValueRefs where applicable.
        Non-Value objects are returned unchanged. Containers are copied.

    Example:
        >>> v = Value(ValueKind.TEXT, "hello", ref="node_1")
        >>> ref = replace_values_with_refs(v)
        >>> isinstance(ref, ValueRef)
        True
        >>> ref.ref
        'node_1'

        >>> v_no_ref = Value(ValueKind.TEXT, "no ref")
        >>> result = replace_values_with_refs(v_no_ref)
        >>> isinstance(result, Value)
        True

        >>> data = [v, {"key": v}]
        >>> result = replace_values_with_refs(data)
        >>> isinstance(result[0], ValueRef)
        True
        >>> isinstance(result[1]["key"], ValueRef)
        True
    """
    if isinstance(obj, Value):
        if obj.ref is not None:
            return ValueRef(ref=obj.ref)
        return obj
    elif isinstance(obj, dict):
        return {k: replace_values_with_refs(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_values_with_refs(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(replace_values_with_refs(item) for item in obj)
    else:
        return obj


def has_error_value(*args: Any, **kwargs: Any) -> bool:
    """Check if any Value in the args/kwargs is a Value(ERROR).

    Traverses args and kwargs, looking for any Value with kind=ERROR.
    Handles nested lists, tuples, and dicts.

    Args:
        *args: Positional arguments to search for error Values.
        **kwargs: Keyword arguments to search for error Values.

    Returns:
        True if any Value(ERROR) is found, False otherwise.

    Example:
        >>> error_val = Value(ValueKind.ERROR, ValueError("test"))
        >>> has_error_value(error_val)
        True

        >>> ok_val = Value(ValueKind.TEXT, "hello")
        >>> has_error_value(ok_val)
        False

        >>> has_error_value([ok_val, {"nested": error_val}])
        True
    """

    def _has_error(obj: Any) -> bool:
        if isinstance(obj, Value):
            return obj.kind == ValueKind.ERROR
        elif isinstance(obj, dict):
            return any(_has_error(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return any(_has_error(item) for item in obj)
        return False

    for arg in args:
        if _has_error(arg):
            return True
    for v in kwargs.values():
        if _has_error(v):
            return True
    return False


def first_error_value(*args: Any, **kwargs: Any) -> Value | None:
    """Find the first Value(ERROR) in args/kwargs.

    Traverses args and kwargs in order, returning the first Value with
    kind=ERROR found. Uses depth-first traversal for nested structures.

    Args:
        *args: Positional arguments to search for error Values.
        **kwargs: Keyword arguments to search for error Values.

    Returns:
        The first Value(ERROR) found, or None if no error Values exist.

    Example:
        >>> error1 = Value(ValueKind.ERROR, ValueError("first"))
        >>> error2 = Value(ValueKind.ERROR, ValueError("second"))
        >>> result = first_error_value(error1, error2)
        >>> result is error1
        True

        >>> ok_val = Value(ValueKind.TEXT, "hello")
        >>> first_error_value(ok_val) is None
        True

        >>> first_error_value([ok_val, {"nested": error1}]) is error1
        True
    """

    def _find_error(obj: Any) -> Value | None:
        if isinstance(obj, Value):
            if obj.kind == ValueKind.ERROR:
                return obj
            return None
        elif isinstance(obj, dict):
            for v in obj.values():
                error = _find_error(v)
                if error is not None:
                    return error
            return None
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                error = _find_error(item)
                if error is not None:
                    return error
            return None
        return None

    for arg in args:
        error = _find_error(arg)
        if error is not None:
            return error
    for v in kwargs.values():
        error = _find_error(v)
        if error is not None:
            return error
    return None
