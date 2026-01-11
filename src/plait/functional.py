"""Functional, graph-aware operations on Value containers.

This module provides stateless, composable functions for operating on Values
without defining custom Module subclasses. All operations are
Value-aware and return Value objects, propagating errors as values instead
of raising exceptions.
"""

from __future__ import annotations

import builtins
import json
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import Any, cast

from plait.values import Value, ValueKind, collect_refs, unwrap, valueify

type TextInput = str | Value
type StructuredPayload = dict[str, Any] | list[Any] | tuple[Any, ...]
type StructuredInput = StructuredPayload | Value
type Numeric = int | float
type NumericInput = Numeric | Value
type PathPart = str | int
type Path = PathPart | Sequence[PathPart]
type MessagesInput = Sequence[Mapping[str, str]] | Value

_MISSING = object()


def _infer_kind(x: object) -> ValueKind:
    """Infer the appropriate ValueKind for a Python value."""
    if isinstance(x, str):
        return ValueKind.TEXT
    if isinstance(x, bool):
        return ValueKind.OTHER
    if isinstance(x, int):
        return ValueKind.INT
    if isinstance(x, float):
        return ValueKind.FLOAT
    if isinstance(x, bytes):
        return ValueKind.BINARY
    if isinstance(x, dict):
        return ValueKind.STRUCTURED
    if isinstance(x, (list, tuple)):
        return ValueKind.STRUCTURED
    if isinstance(x, BaseException):
        return ValueKind.ERROR
    return ValueKind.OTHER


def _sorted_keys(mapping: Mapping[object, object]) -> list[object]:
    """Return keys sorted deterministically (by string repr)."""
    return sorted(mapping.keys(), key=lambda key: str(key))


def _iter_error_values(obj: object) -> list[Value]:
    """Collect Value(ERROR) instances in deterministic traversal order."""
    errors: list[Value] = []

    def visit(item: object) -> None:
        if isinstance(item, Value):
            if item.kind == ValueKind.ERROR:
                errors.append(item)
                return
            if item.kind == ValueKind.STRUCTURED:
                payload = item.payload
                if isinstance(payload, dict):
                    for key in _sorted_keys(cast(Mapping[object, object], payload)):
                        visit(payload[key])
                elif isinstance(payload, (list, tuple)):
                    for value in payload:
                        visit(value)
            return
        if isinstance(item, dict):
            for key in _sorted_keys(cast(Mapping[object, object], item)):
                visit(item[key])
            return
        if isinstance(item, (list, tuple)):
            for value in item:
                visit(value)

    visit(obj)
    return errors


def _resolve_errors(*args: object, **kwargs: object) -> Value | None:
    """Resolve errors deterministically from inputs."""
    errors: list[Value] = []
    for arg in args:
        errors.extend(_iter_error_values(arg))
    for key in sorted(kwargs.keys()):
        errors.extend(_iter_error_values(kwargs[key]))

    if not errors:
        return None
    if len(errors) == 1:
        return errors[0]

    primary = errors[0]
    meta = primary.meta.copy()
    causes = list(meta.get("causes", []))
    for err in errors[1:]:
        causes.append({"ref": err.ref, "message": str(err.payload)})
    meta["causes"] = causes

    return Value(
        kind=ValueKind.ERROR,
        payload=primary.payload,
        ref=primary.ref,
        meta=meta,
    )


def _error_value(
    op: str,
    exc: Exception,
    *inputs: object,
    extra_meta: dict[str, Any] | None = None,
) -> Value:
    """Create a Value(ERROR) with metadata about the failed op."""
    meta: dict[str, Any] = {"op": op, "source_refs": collect_refs(*inputs)}
    if extra_meta:
        meta.update(extra_meta)
    return Value(ValueKind.ERROR, exc, meta=meta)


def _coerce_to_text(part: TextInput) -> str | None:
    """Return a string payload if part is text-like, else None."""
    if isinstance(part, Value):
        if part.kind == ValueKind.ERROR:
            return None
        if isinstance(part.payload, str):
            return part.payload
        return None
    if isinstance(part, str):
        return part
    return None


def _as_number(value: NumericInput, op: str) -> tuple[Numeric, ValueKind] | Value:
    """Extract a numeric payload and its kind or return Value(ERROR)."""
    if isinstance(value, Value):
        if value.kind == ValueKind.ERROR:
            return value
        if value.kind in {ValueKind.INT, ValueKind.FLOAT}:
            payload = value.payload
            if isinstance(payload, bool):
                return _error_value(op, TypeError("Boolean is not numeric"), value)
            if isinstance(payload, (int, float)):
                return payload, value.kind
        return _error_value(op, TypeError("Value is not numeric"), value)

    if isinstance(value, bool):
        return _error_value(op, TypeError("Boolean is not numeric"), value)
    if isinstance(value, (int, float)):
        return value, ValueKind.FLOAT if isinstance(value, float) else ValueKind.INT

    return _error_value(op, TypeError("Value is not numeric"), value)


def _ensure_structured(value: StructuredInput, op: str) -> StructuredPayload | Value:
    """Extract structured payload or return Value(ERROR)."""
    if isinstance(value, Value):
        if value.kind == ValueKind.ERROR:
            return value
        if value.kind != ValueKind.STRUCTURED:
            return _error_value(op, TypeError("Value is not structured"), value)
        payload = value.payload
    else:
        payload = value

    if isinstance(payload, (dict, list, tuple)):
        return payload

    return _error_value(op, TypeError("Value is not structured"), value)


def _render_scalar(template: TextInput, vars: StructuredInput) -> Value:
    """Render an f-string template using structured variables."""
    error = _resolve_errors(template, vars)
    if error is not None:
        return error

    tmpl_value = valueify(template, kind=ValueKind.FSTRING)
    if not isinstance(tmpl_value, Value) or not isinstance(tmpl_value.payload, str):
        return _error_value(
            "render",
            TypeError("Template must be a string"),
            template,
            vars,
        )

    raw_vars = unwrap(valueify(vars))
    if not isinstance(raw_vars, dict):
        return _error_value(
            "render",
            TypeError("vars must be a structured mapping"),
            template,
            vars,
        )

    try:
        rendered = tmpl_value.payload.format_map(raw_vars)
    except Exception as exc:  # noqa: BLE001 - convert to Value(ERROR)
        return _error_value("render", exc, template, vars)

    return Value(ValueKind.TEXT, rendered)


def render(template: TextInput, vars: StructuredInput) -> Value:
    """Render an f-string template using structured variables."""
    return _render_scalar(template, vars)


def try_render(template: TextInput, vars: StructuredInput) -> Value:
    """Render a template and return errors as Value(ERROR)."""
    return render(template, vars)


def _format_scalar(fmt: TextInput, vars: StructuredInput) -> Value:
    """Format a string using structured variables."""
    error = _resolve_errors(fmt, vars)
    if error is not None:
        return error

    fmt_value = valueify(fmt, kind=ValueKind.TEXT)
    if not isinstance(fmt_value, Value) or not isinstance(fmt_value.payload, str):
        return _error_value(
            "format",
            TypeError("Format string must be a string"),
            fmt,
            vars,
        )

    raw_vars = unwrap(valueify(vars))
    if not isinstance(raw_vars, dict):
        return _error_value(
            "format",
            TypeError("vars must be a structured mapping"),
            fmt,
            vars,
        )

    try:
        rendered = fmt_value.payload.format_map(raw_vars)
    except Exception as exc:  # noqa: BLE001 - convert to Value(ERROR)
        return _error_value("format", exc, fmt, vars)

    return Value(ValueKind.TEXT, rendered)


def format(fmt: TextInput, vars: StructuredInput) -> Value:
    """Format a string using structured variables."""
    return _format_scalar(fmt, vars)


def _concat_scalar(*parts: TextInput, sep: TextInput = "") -> Value:
    """Join text-like inputs with an optional separator."""
    error = _resolve_errors(*parts, sep=sep)
    if error is not None:
        return error

    sep_text = _coerce_to_text(sep)
    if sep_text is None:
        return _error_value(
            "concat",
            TypeError("Separator must be a string"),
            parts,
            sep,
        )

    rendered_parts: list[str] = []
    for part in parts:
        text = _coerce_to_text(part)
        if text is None:
            return _error_value(
                "concat",
                TypeError("All parts must be text-like"),
                parts,
                sep,
            )
        rendered_parts.append(text)

    return Value(ValueKind.TEXT, sep_text.join(rendered_parts))


def concat(*parts: TextInput, sep: TextInput = "") -> Value:
    """Join text-like inputs with an optional separator."""
    return _concat_scalar(*parts, sep=sep)


def strip(text: TextInput, *, chars: str | None = None) -> Value:
    """Strip leading/trailing characters from text."""
    return _strip_scalar(text, chars=chars)


def _strip_scalar(text: TextInput, *, chars: str | None = None) -> Value:
    error = _resolve_errors(text)
    if error is not None:
        return error

    raw_text = _coerce_to_text(text)
    if raw_text is None:
        return _error_value("strip", TypeError("Input must be text"), text)

    return Value(ValueKind.TEXT, raw_text.strip(chars))


def lower(text: TextInput) -> Value:
    """Lowercase text."""
    return _lower_scalar(text)


def _lower_scalar(text: TextInput) -> Value:
    error = _resolve_errors(text)
    if error is not None:
        return error

    raw_text = _coerce_to_text(text)
    if raw_text is None:
        return _error_value("lower", TypeError("Input must be text"), text)

    return Value(ValueKind.TEXT, raw_text.lower())


def upper(text: TextInput) -> Value:
    """Uppercase text."""
    return _upper_scalar(text)


def _upper_scalar(text: TextInput) -> Value:
    error = _resolve_errors(text)
    if error is not None:
        return error

    raw_text = _coerce_to_text(text)
    if raw_text is None:
        return _error_value("upper", TypeError("Input must be text"), text)

    return Value(ValueKind.TEXT, raw_text.upper())


def _parse_structured_scalar(
    text: TextInput, schema: type | tuple[type, ...] | None = None
) -> Value:
    """Parse structured text into a structured Value."""
    error = _resolve_errors(text)
    if error is not None:
        return error

    raw_text = unwrap(text)
    if not isinstance(raw_text, str):
        return _error_value(
            "parse_structured",
            TypeError("Input must be a JSON string"),
            text,
        )

    try:
        parsed = json.loads(raw_text)
    except Exception as exc:  # noqa: BLE001 - convert to Value(ERROR)
        return _error_value("parse_structured", exc, text)

    if schema is not None:
        if isinstance(schema, type) or (
            isinstance(schema, tuple) and all(isinstance(t, type) for t in schema)
        ):
            if not isinstance(parsed, schema):
                return _error_value(
                    "parse_structured",
                    TypeError("Parsed value does not match schema"),
                    text,
                    extra_meta={"schema": schema},
                )
        else:
            return _error_value(
                "parse_structured",
                TypeError("Schema must be a type or tuple of types"),
                text,
                extra_meta={"schema": schema},
            )

    return Value(ValueKind.STRUCTURED, parsed)


def parse_structured(
    text: TextInput, schema: type | tuple[type, ...] | None = None
) -> Value:
    """Parse structured text into a structured Value."""
    return _parse_structured_scalar(text, schema=schema)


def _normalize_path(path: Path) -> list[PathPart]:
    """Normalize a selection path into a list of keys/indices."""
    if isinstance(path, Value):
        path = unwrap(path)

    if isinstance(path, str):
        if path == "":
            raise ValueError("Path cannot be empty")
        return cast(list[PathPart], path.split("."))
    if isinstance(path, int):
        return [path]
    if isinstance(path, (list, tuple)):
        if not path:
            raise ValueError("Path cannot be empty")
        normalized: list[PathPart] = []
        for part in path:
            if isinstance(part, (str, int)):
                normalized.append(part)
            else:
                raise TypeError("Path elements must be str or int")
        return normalized

    raise TypeError("Path must be str, int, or list/tuple of str|int")


def _wrap_selected(value: object) -> Value:
    """Wrap a selected value into a Value with appropriate kind."""
    if isinstance(value, Value):
        return value
    if isinstance(value, (dict, list, tuple)):
        return Value(ValueKind.STRUCTURED, value)
    return Value(_infer_kind(value), value)


def _wrap_default(default: object) -> Value:
    """Wrap a default value into a Value with structured support."""
    if isinstance(default, Value):
        return default
    if isinstance(default, (dict, list, tuple)):
        return Value(ValueKind.STRUCTURED, default)
    return Value(_infer_kind(default), default)


def select(
    struct: StructuredInput, key_or_path: Path, *, default: object = _MISSING
) -> Value:
    """Select a field by key or path from a structured value."""
    if default is _MISSING:
        error = _resolve_errors(struct, key_or_path)
    else:
        error = _resolve_errors(struct, key_or_path, default=default)
    if error is not None:
        return error

    try:
        path = _normalize_path(key_or_path)
    except Exception as exc:  # noqa: BLE001 - convert to Value(ERROR)
        return _error_value("select", exc, struct, key_or_path, default)

    structured = _ensure_structured(struct, "select")
    if isinstance(structured, Value):
        return structured
    current: object = structured

    for index, key in enumerate(path):
        if isinstance(current, Value):
            if current.kind == ValueKind.ERROR:
                return current
            if current.kind != ValueKind.STRUCTURED:
                return _error_value(
                    "select",
                    TypeError("Encountered non-structured value in path"),
                    struct,
                    key_or_path,
                    extra_meta={"path_index": index},
                )
            current = current.payload

        try:
            current = current[key]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            if default is not _MISSING:
                return _wrap_default(default)
            return _error_value(
                "select",
                exc,
                struct,
                key_or_path,
                extra_meta={"path": path, "missing": key},
            )

    return _wrap_selected(current)


def merge(*structs: StructuredInput) -> Value:
    """Merge structured objects left-to-right."""
    error = _resolve_errors(*structs)
    if error is not None:
        return error

    merged: dict[str, Any] = {}
    for struct in structs:
        structured = _ensure_structured(struct, "merge")
        if isinstance(structured, Value):
            return structured
        if not isinstance(structured, dict):
            return _error_value(
                "merge",
                TypeError("merge requires mapping payloads"),
                struct,
            )
        merged.update(structured)

    return Value(ValueKind.STRUCTURED, merged)


def _coerce_scalar(
    value: StructuredInput | Numeric | str | Value, kind: ValueKind
) -> Value:
    """Coerce a value to a target kind when safe."""
    error = _resolve_errors(value)
    if error is not None:
        return error

    if isinstance(value, Value):
        if value.kind == ValueKind.ERROR:
            return value
        if value.kind == kind:
            return value
        source_kind = value.kind
        payload = value.payload
        allow_lossy = bool(value.meta.get("allow_lossy"))
    else:
        payload = value
        source_kind = _infer_kind(value)
        allow_lossy = False

    if source_kind == kind:
        return Value(kind, payload)

    if source_kind == ValueKind.TEXT and kind in {ValueKind.INT, ValueKind.FLOAT}:
        if not isinstance(payload, str):
            return _error_value("coerce", TypeError("Input must be text"), value)
        try:
            parsed = int(payload) if kind == ValueKind.INT else float(payload)
        except Exception as exc:  # noqa: BLE001 - convert to Value(ERROR)
            return _error_value("coerce", exc, value)
        return Value(kind, parsed)

    if source_kind == ValueKind.INT and kind == ValueKind.FLOAT:
        if isinstance(payload, bool):
            return _error_value("coerce", TypeError("Boolean is not numeric"), value)
        if isinstance(payload, int):
            return Value(ValueKind.FLOAT, float(payload))

    if source_kind == ValueKind.FLOAT and kind == ValueKind.INT:
        if isinstance(payload, float):
            if payload.is_integer() or allow_lossy:
                return Value(ValueKind.INT, int(payload))
            return _error_value(
                "coerce",
                TypeError("Lossy float-to-int conversion"),
                value,
            )

    if source_kind == ValueKind.STRUCTURED and kind == ValueKind.TEXT:
        try:
            serialized = json.dumps(
                unwrap(payload),
                sort_keys=True,
                separators=(",", ":"),
            )
        except Exception as exc:  # noqa: BLE001 - convert to Value(ERROR)
            return _error_value("coerce", exc, value)
        return Value(ValueKind.TEXT, serialized)

    if source_kind == ValueKind.TEXT and kind == ValueKind.STRUCTURED:
        return _error_value(
            "coerce",
            TypeError("Use parse_structured for text-to-structured"),
            value,
        )

    if source_kind == ValueKind.ERROR:
        return Value(ValueKind.ERROR, payload)

    return _error_value(
        "coerce",
        TypeError(f"Unsupported coercion: {source_kind} -> {kind}"),
        value,
    )


def coerce(value: StructuredInput | Numeric | str | Value, kind: ValueKind) -> Value:
    """Coerce a value to a target kind when safe."""
    return _coerce_scalar(value, kind)


def _add_scalar(a: NumericInput, b: NumericInput) -> Value:
    error = _resolve_errors(a, b)
    if error is not None:
        return error

    left = _as_number(a, "add")
    if isinstance(left, Value):
        return left
    right = _as_number(b, "add")
    if isinstance(right, Value):
        return right

    left_val, left_kind = left
    right_val, right_kind = right
    result_kind = (
        ValueKind.FLOAT if ValueKind.FLOAT in {left_kind, right_kind} else ValueKind.INT
    )
    result = left_val + right_val
    if result_kind == ValueKind.INT:
        return Value(ValueKind.INT, int(result))
    return Value(ValueKind.FLOAT, float(result))


def add(a: NumericInput, b: NumericInput) -> Value:
    """Add two numeric values with numeric promotion."""
    return _add_scalar(a, b)


def _sub_scalar(a: NumericInput, b: NumericInput) -> Value:
    error = _resolve_errors(a, b)
    if error is not None:
        return error

    left = _as_number(a, "sub")
    if isinstance(left, Value):
        return left
    right = _as_number(b, "sub")
    if isinstance(right, Value):
        return right

    left_val, left_kind = left
    right_val, right_kind = right
    result_kind = (
        ValueKind.FLOAT if ValueKind.FLOAT in {left_kind, right_kind} else ValueKind.INT
    )
    result = left_val - right_val
    if result_kind == ValueKind.INT:
        return Value(ValueKind.INT, int(result))
    return Value(ValueKind.FLOAT, float(result))


def sub(a: NumericInput, b: NumericInput) -> Value:
    """Subtract two numeric values with numeric promotion."""
    return _sub_scalar(a, b)


def _mul_scalar(a: NumericInput, b: NumericInput) -> Value:
    error = _resolve_errors(a, b)
    if error is not None:
        return error

    left = _as_number(a, "mul")
    if isinstance(left, Value):
        return left
    right = _as_number(b, "mul")
    if isinstance(right, Value):
        return right

    left_val, left_kind = left
    right_val, right_kind = right
    result_kind = (
        ValueKind.FLOAT if ValueKind.FLOAT in {left_kind, right_kind} else ValueKind.INT
    )
    result = left_val * right_val
    if result_kind == ValueKind.INT:
        return Value(ValueKind.INT, int(result))
    return Value(ValueKind.FLOAT, float(result))


def mul(a: NumericInput, b: NumericInput) -> Value:
    """Multiply two numeric values with numeric promotion."""
    return _mul_scalar(a, b)


def _div_scalar(a: NumericInput, b: NumericInput) -> Value:
    error = _resolve_errors(a, b)
    if error is not None:
        return error

    left = _as_number(a, "div")
    if isinstance(left, Value):
        return left
    right = _as_number(b, "div")
    if isinstance(right, Value):
        return right

    left_val, _ = left
    right_val, _ = right
    if right_val == 0:
        return _error_value("div", ZeroDivisionError("Division by zero"), a, b)

    return Value(ValueKind.FLOAT, float(left_val) / float(right_val))


def div(a: NumericInput, b: NumericInput) -> Value:
    """Divide two numeric values (always returns FLOAT)."""
    return _div_scalar(a, b)


def _coerce_numeric_sequence(
    values: Sequence[NumericInput] | Value, op: str
) -> tuple[list[Numeric], ValueKind] | Value:
    """Convert input to numeric values or return Value(ERROR)."""
    if isinstance(values, Value):
        if values.kind == ValueKind.ERROR:
            return values
        if values.kind == ValueKind.STRUCTURED and isinstance(
            values.payload, (list, tuple)
        ):
            values = values.payload
        else:
            return _error_value(op, TypeError("Values must be a list"), values)

    if not isinstance(values, (list, tuple)):
        return _error_value(op, TypeError("Values must be a list"), values)

    error = _resolve_errors(values)
    if error is not None:
        return error

    numeric_values: list[Numeric] = []
    has_float = False
    for value in values:
        numeric = _as_number(value, op)
        if isinstance(numeric, Value):
            return numeric
        payload, kind = numeric
        numeric_values.append(payload)
        if kind == ValueKind.FLOAT:
            has_float = True

    result_kind = ValueKind.FLOAT if has_float else ValueKind.INT
    return numeric_values, result_kind


def sum(values: Sequence[NumericInput] | Value) -> Value:
    """Sum a list of numeric values with numeric promotion."""
    coerced = _coerce_numeric_sequence(values, "sum")
    if isinstance(coerced, Value):
        return coerced
    numeric_values, result_kind = coerced

    result = builtins.sum(numeric_values)
    if result_kind == ValueKind.FLOAT:
        return Value(ValueKind.FLOAT, float(result))
    return Value(ValueKind.INT, int(result))


def min(values: Sequence[NumericInput] | Value) -> Value:
    """Return the minimum of a list of numeric values."""
    coerced = _coerce_numeric_sequence(values, "min")
    if isinstance(coerced, Value):
        return coerced
    numeric_values, result_kind = coerced

    if not numeric_values:
        return _error_value("min", ValueError("Values list is empty"), values)

    result = builtins.min(numeric_values)
    if result_kind == ValueKind.FLOAT:
        return Value(ValueKind.FLOAT, float(result))
    return Value(ValueKind.INT, int(result))


def max(values: Sequence[NumericInput] | Value) -> Value:
    """Return the maximum of a list of numeric values."""
    coerced = _coerce_numeric_sequence(values, "max")
    if isinstance(coerced, Value):
        return coerced
    numeric_values, result_kind = coerced

    if not numeric_values:
        return _error_value("max", ValueError("Values list is empty"), values)

    result = builtins.max(numeric_values)
    if result_kind == ValueKind.FLOAT:
        return Value(ValueKind.FLOAT, float(result))
    return Value(ValueKind.INT, int(result))


def mean(values: Sequence[NumericInput] | Value) -> Value:
    """Return the mean of a list of numeric values (always FLOAT)."""
    coerced = _coerce_numeric_sequence(values, "mean")
    if isinstance(coerced, Value):
        return coerced
    numeric_values, _ = coerced

    if not numeric_values:
        return _error_value("mean", ValueError("Values list is empty"), values)

    result = builtins.sum(numeric_values) / len(numeric_values)
    return Value(ValueKind.FLOAT, float(result))


def _extract_attr(payload: object, key: str) -> object | None:
    """Extract attribute or dict key from a payload."""
    if isinstance(payload, dict):
        mapping = cast(dict[str, Any], payload)
        return mapping.get(key)
    return getattr(payload, key, None)


def extract_text(resp: Value | object) -> Value:
    """Extract text content from a response value."""
    error = _resolve_errors(resp)
    if error is not None:
        return error

    if isinstance(resp, Value):
        if resp.kind == ValueKind.ERROR:
            return resp
        payload = resp.payload
    else:
        payload = resp

    text = _extract_attr(payload, "content")
    if text is None and isinstance(payload, dict):
        mapping = cast(dict[str, Any], payload)
        choices = mapping.get("choices")
        if isinstance(choices, list) and choices:
            if isinstance(choices[0], dict):
                message = cast(dict[str, Any], choices[0]).get("message")
            else:
                message = None
            if isinstance(message, dict):
                text = cast(dict[str, Any], message).get("content")
    if text is None and hasattr(payload, "choices"):
        choices = payload.choices
        if isinstance(choices, list) and choices:
            message = getattr(choices[0], "message", None)
            text = getattr(message, "content", None) if message is not None else None

    if isinstance(text, str):
        return Value(ValueKind.TEXT, text)

    return _error_value("extract_text", TypeError("Missing response text"), resp)


def extract_meta(resp: Value | object) -> Value:
    """Extract metadata from a response value."""
    error = _resolve_errors(resp)
    if error is not None:
        return error

    if isinstance(resp, Value):
        if resp.kind == ValueKind.ERROR:
            return resp
        payload = resp.payload
    else:
        payload = resp

    meta: dict[str, Any] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "finish_reason",
        "model",
        "reasoning",
        "tool_calls",
        "time_to_first_token_ms",
        "completion_time_ms",
        "queue_time_ms",
    ):
        value = _extract_attr(payload, key)
        if value is not None:
            meta[key] = value

    usage = _extract_attr(payload, "usage")
    if isinstance(usage, dict):
        usage_mapping = cast(dict[str, Any], usage)
        if "prompt_tokens" in usage_mapping:
            meta["input_tokens"] = usage_mapping.get("prompt_tokens")
        if "completion_tokens" in usage_mapping:
            meta["output_tokens"] = usage_mapping.get("completion_tokens")
        if "total_tokens" in usage_mapping:
            meta["total_tokens"] = usage_mapping.get("total_tokens")

    model_name = _extract_attr(payload, "model")
    if model_name is not None:
        meta.setdefault("model", model_name)

    if not meta:
        return _error_value(
            "extract_meta", TypeError("Missing response metadata"), resp
        )

    return Value(ValueKind.STRUCTURED, meta)


def _extract_http_status(exc: Exception) -> int | None:
    for key in ("status_code", "status", "http_status"):
        value = getattr(exc, key, None)
        if isinstance(value, int):
            return value
    return None


def chat_complete(
    client: object,
    *,
    messages: MessagesInput,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> Coroutine[Any, Any, Value]:
    """Execute an async chat completion request and wrap the response."""

    async def _run() -> Value:
        error = _resolve_errors(messages)
        if error is not None:
            return error

        raw_messages = unwrap(messages)
        if not isinstance(raw_messages, list):
            return _error_value(
                "chat_complete",
                TypeError("messages must be a list of dicts"),
                messages,
            )

        payload = None
        try:
            chat = getattr(client, "chat", None)
            completions = getattr(chat, "completions", None) if chat else None
            create = getattr(completions, "create", None) if completions else None
            if create is None:
                return _error_value(
                    "chat_complete",
                    TypeError("client does not support chat completions"),
                    client,
                )

            request_kwargs: dict[str, Any] = {**kwargs}
            if model is not None:
                request_kwargs["model"] = model
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens

            payload = await create(messages=raw_messages, **request_kwargs)
        except Exception as exc:  # noqa: BLE001 - convert to Value(ERROR)
            meta: dict[str, Any] = {"op": "chat_complete"}
            http_status = _extract_http_status(exc)
            if http_status is not None:
                meta["http_status"] = http_status
            return Value(ValueKind.ERROR, exc, meta=meta)

        return Value(ValueKind.RESPONSE, payload)

    return _run()


def with_meta(value: Value | object, **meta: Any) -> Value:
    """Add or overwrite metadata on a value."""
    val = value if isinstance(value, Value) else valueify(value)
    if val.kind == ValueKind.ERROR:
        merged = val.meta.copy()
        merged.update(meta)
        return Value(ValueKind.ERROR, val.payload, ref=val.ref, meta=merged)

    merged = val.meta.copy()
    merged.update(meta)
    return Value(val.kind, val.payload, ref=val.ref, meta=merged)


def unwrap_or(value: Value | object, default: object) -> Value:
    """Return a fallback value when the input is an error."""
    if isinstance(value, Value) and value.kind == ValueKind.ERROR:
        recovered = _wrap_default(default)
        meta = recovered.meta.copy()
        meta["recovered_from"] = value.ref
        return Value(recovered.kind, recovered.payload, ref=recovered.ref, meta=meta)

    if isinstance(value, Value):
        return value
    return valueify(value)


def is_error(value: Value | object) -> Value:
    """Return a structured flag indicating whether input is an error."""
    if isinstance(value, Value):
        return Value(
            ValueKind.STRUCTURED,
            {"is_error": value.kind == ValueKind.ERROR, "ref": value.ref},
        )

    return Value(ValueKind.STRUCTURED, {"is_error": False, "ref": None})


def map_values(
    fn: Callable[[Value], Value | object],
    values: Sequence[Value | object] | Value,
) -> list[Value]:
    """Apply a function element-wise across a batch."""
    if isinstance(values, Value):
        if values.kind == ValueKind.ERROR:
            return [values]
        if values.kind == ValueKind.STRUCTURED and isinstance(
            values.payload, (list, tuple)
        ):
            values = values.payload
        else:
            return [_error_value("map_values", TypeError("values must be a list"))]

    output: list[Value] = []
    for item in values:
        val = item if isinstance(item, Value) else valueify(item)
        if val.kind == ValueKind.ERROR:
            output.append(val)
            continue
        try:
            result = fn(val)
        except Exception as exc:  # noqa: BLE001 - convert to Value(ERROR)
            output.append(_error_value("map_values", exc, val))
            continue
        output.append(result if isinstance(result, Value) else valueify(result))

    return output


def zip_values(*batches: Sequence[Value] | Value) -> list[tuple[Value, ...]] | Value:
    """Zip multiple batches into tuples, validating equal lengths."""
    prepared: list[Sequence[Value]] = []
    for batch in batches:
        if isinstance(batch, Value):
            if batch.kind == ValueKind.ERROR:
                return batch
            if batch.kind == ValueKind.STRUCTURED and isinstance(
                batch.payload, (list, tuple)
            ):
                batch = batch.payload
            else:
                return _error_value(
                    "zip_values",
                    TypeError("batches must be lists"),
                    batch,
                )

        prepared_batch: list[Value] = [
            item if isinstance(item, Value) else valueify(item) for item in batch
        ]
        prepared.append(prepared_batch)

    if not prepared:
        return []

    length = len(prepared[0])
    if any(len(batch) != length for batch in prepared[1:]):
        return _error_value(
            "zip_values",
            ValueError("Batches must have equal length"),
            batches,
        )

    return list(zip(*prepared, strict=True))


def stack_structured(values: Sequence[Value] | Value) -> Value:
    """Stack structured values into a single structured payload."""
    if isinstance(values, Value):
        if values.kind == ValueKind.ERROR:
            return values
        if values.kind == ValueKind.STRUCTURED and isinstance(
            values.payload, (list, tuple)
        ):
            values = values.payload
        else:
            return _error_value(
                "stack_structured",
                TypeError("values must be a list of structured values"),
                values,
            )

    error = _resolve_errors(values)
    if error is not None:
        return error

    stacked: list[StructuredPayload] = []
    for value in values:
        if isinstance(value, Value):
            if value.kind == ValueKind.ERROR:
                return value
            if value.kind != ValueKind.STRUCTURED:
                return _error_value(
                    "stack_structured",
                    TypeError("All values must be structured"),
                    value,
                )
            payload = value.payload
        else:
            payload = value
        if not isinstance(payload, (dict, list, tuple)):
            return _error_value(
                "stack_structured",
                TypeError("Structured payload must be dict/list/tuple"),
                value,
            )
        stacked.append(payload)

    return Value(ValueKind.STRUCTURED, stacked)


__all__ = [
    "render",
    "try_render",
    "format",
    "concat",
    "strip",
    "lower",
    "upper",
    "parse_structured",
    "select",
    "merge",
    "coerce",
    "add",
    "sub",
    "mul",
    "div",
    "sum",
    "min",
    "max",
    "mean",
    "extract_text",
    "extract_meta",
    "chat_complete",
    "with_meta",
    "unwrap_or",
    "is_error",
    "map_values",
    "zip_values",
    "stack_structured",
]
