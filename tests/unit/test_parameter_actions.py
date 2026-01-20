"""Tests for Parameter feedback extraction helpers."""

from plait.parameter import Parameter, extract_actions
from plait.values import Value, ValueKind


def test_extract_actions_variants() -> None:
    value = Value(ValueKind.STRUCTURED, [["a", "", 1], []])
    assert extract_actions(value) == ["a", "1"]

    value = Value(ValueKind.STRUCTURED, ["a", "", 2])
    assert extract_actions(value) == ["a", "2"]

    value = Value(ValueKind.TEXT, "hi")
    assert extract_actions(value) == ["hi"]

    value = Value(ValueKind.OTHER, None)
    assert extract_actions(value) == []

    value = Value(ValueKind.OTHER, 123)
    assert extract_actions(value) == ["123"]


def test_parameter_accumulates_value_feedback() -> None:
    param = Parameter("x", description="test")
    param.accumulate_feedback(Value(ValueKind.STRUCTURED, [["keep", ""]]))
    assert param.get_accumulated_feedback() == ["keep"]
