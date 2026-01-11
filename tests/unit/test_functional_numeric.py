"""Unit tests for numeric functional operations."""

import plait.functional as F
from plait.values import Value, ValueKind, valueify


def test_add_sub_mul_div() -> None:
    """Basic numeric ops work with promotion."""
    assert F.add(1, 2).payload == 3
    assert F.sub(5, 2).payload == 3
    assert F.mul(2, 3).payload == 6
    assert F.div(7, 2).payload == 3.5


def test_numeric_promotion_to_float() -> None:
    """Ops promote to float when any input is float."""
    result = F.add(valueify(1), valueify(2.5))

    assert result.kind == ValueKind.FLOAT
    assert result.payload == 3.5


def test_division_by_zero_returns_error() -> None:
    """div returns Value(ERROR) on zero divisor."""
    result = F.div(1, 0)

    assert result.kind == ValueKind.ERROR
    assert result.meta["op"] == "div"


def test_sum_min_max_mean() -> None:
    """Aggregations work on numeric lists."""
    values = valueify([1, 2, 3])

    assert F.sum(values).payload == 6
    assert F.min(values).payload == 1
    assert F.max(values).payload == 3
    assert F.mean(values).payload == 2.0


def test_min_max_mean_empty_returns_error() -> None:
    """min/max/mean error on empty lists."""
    empty = valueify([])

    assert F.min(empty).kind == ValueKind.ERROR
    assert F.max(empty).kind == ValueKind.ERROR
    assert F.mean(empty).kind == ValueKind.ERROR


def test_numeric_non_numeric_returns_error() -> None:
    """Numeric ops return Value(ERROR) for non-numeric inputs."""
    result = F.add(1, valueify("a"))

    assert result.kind == ValueKind.ERROR
    assert result.meta["op"] == "add"


def test_sum_with_error_propagates() -> None:
    """Aggregations propagate Value(ERROR)."""
    err = Value(ValueKind.ERROR, ValueError("boom"), ref="err")

    result = F.sum([valueify(1), err])

    assert result is err
