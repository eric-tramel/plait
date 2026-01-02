"""Additional tests for structured functional operations."""

import inf_engine.functional as F
from inf_engine.values import Value, ValueKind, valueify


def test_merge_structured_dicts() -> None:
    """merge combines mappings left-to-right."""
    left = valueify({"a": 1, "b": 2})
    right = valueify({"b": 3, "c": 4})

    result = F.merge(left, right)

    assert result.kind == ValueKind.STRUCTURED
    assert isinstance(result.payload["a"], Value)
    assert isinstance(result.payload["b"], Value)
    assert isinstance(result.payload["c"], Value)
    assert result.payload["a"].payload == 1
    assert result.payload["b"].payload == 3
    assert result.payload["c"].payload == 4


def test_merge_non_mapping_returns_error() -> None:
    """merge returns Value(ERROR) for non-mapping structured payloads."""
    result = F.merge(valueify([1, 2]))

    assert result.kind == ValueKind.ERROR
    assert result.meta["op"] == "merge"


def test_coerce_text_to_int_float() -> None:
    """coerce parses numeric text when safe."""
    result_int = F.coerce(valueify("42"), ValueKind.INT)
    result_float = F.coerce(valueify("3.5"), ValueKind.FLOAT)

    assert result_int.kind == ValueKind.INT
    assert result_int.payload == 42
    assert result_float.kind == ValueKind.FLOAT
    assert result_float.payload == 3.5


def test_coerce_float_to_int_lossy() -> None:
    """coerce rejects lossy float to int unless allow_lossy meta is set."""
    value = Value(ValueKind.FLOAT, 3.2)
    result = F.coerce(value, ValueKind.INT)

    assert result.kind == ValueKind.ERROR

    lossy = Value(ValueKind.FLOAT, 3.2, meta={"allow_lossy": True})
    result_lossy = F.coerce(lossy, ValueKind.INT)

    assert result_lossy.kind == ValueKind.INT
    assert result_lossy.payload == 3


def test_coerce_structured_to_text() -> None:
    """coerce serializes structured values to JSON."""
    result = F.coerce(valueify({"b": 2, "a": 1}), ValueKind.TEXT)

    assert result.kind == ValueKind.TEXT
    assert result.payload == '{"a":1,"b":2}'


def test_coerce_text_to_structured_error() -> None:
    """coerce refuses text-to-structured conversions."""
    result = F.coerce(valueify("{}"), ValueKind.STRUCTURED)

    assert result.kind == ValueKind.ERROR
    assert result.meta["op"] == "coerce"
