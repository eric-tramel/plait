"""Tests for TracedOutput wrapper."""

from __future__ import annotations

from typing import Any

import pytest

from inf_engine.optimization.record import ForwardRecord, TracedOutput


@pytest.fixture
def mock_forward_record() -> ForwardRecord:
    """Create a minimal mock ForwardRecord for testing."""

    # Create a minimal graph-like structure
    class MockGraph:
        nodes: dict[str, Any] = {}
        outputs: list[str] = []

    return ForwardRecord(
        graph=MockGraph(),  # type: ignore[arg-type]
        node_inputs={"node_1": {"prompt": "test input"}},
        node_outputs={"node_1": "test output"},
        module_map={},
        execution_order=["node_1"],
        timing={"node_1": 0.5},
    )


class TestTracedOutputCreation:
    """Tests for TracedOutput creation and basic attributes."""

    def test_create_with_string_value(self, mock_forward_record: ForwardRecord) -> None:
        """TracedOutput can wrap a string value."""
        traced = TracedOutput(value="hello world", _record=mock_forward_record)

        assert traced.value == "hello world"
        assert traced._record is mock_forward_record

    def test_create_with_dict_value(self, mock_forward_record: ForwardRecord) -> None:
        """TracedOutput can wrap a dictionary value."""
        data: dict[str, Any] = {"key": "value", "nested": {"inner": 42}}
        traced = TracedOutput(value=data, _record=mock_forward_record)

        assert traced.value == data
        # Access through data variable to satisfy type checker
        assert data["key"] == "value"
        nested = data["nested"]
        assert isinstance(nested, dict)
        assert nested["inner"] == 42

    def test_create_with_list_value(self, mock_forward_record: ForwardRecord) -> None:
        """TracedOutput can wrap a list value."""
        data = [1, 2, 3, "four"]
        traced = TracedOutput(value=data, _record=mock_forward_record)

        assert traced.value == data
        assert len(traced.value) == 4

    def test_create_with_none_value(self, mock_forward_record: ForwardRecord) -> None:
        """TracedOutput can wrap None."""
        traced = TracedOutput(value=None, _record=mock_forward_record)

        assert traced.value is None
        assert traced._record is mock_forward_record


class TestTracedOutputStr:
    """Tests for TracedOutput string representation."""

    def test_str_delegates_to_value(self, mock_forward_record: ForwardRecord) -> None:
        """str() on TracedOutput returns str() of the value."""
        traced = TracedOutput(value="hello world", _record=mock_forward_record)

        assert str(traced) == "hello world"

    def test_str_with_numeric_value(self, mock_forward_record: ForwardRecord) -> None:
        """str() works with numeric values."""
        traced = TracedOutput(value=42, _record=mock_forward_record)

        assert str(traced) == "42"

    def test_str_with_dict_value(self, mock_forward_record: ForwardRecord) -> None:
        """str() works with dict values."""
        traced = TracedOutput(value={"key": "value"}, _record=mock_forward_record)

        assert str(traced) == "{'key': 'value'}"


class TestTracedOutputRepr:
    """Tests for TracedOutput repr representation."""

    def test_repr_wraps_value_repr(self, mock_forward_record: ForwardRecord) -> None:
        """repr() shows TracedOutput wrapper with value repr."""
        traced = TracedOutput(value="hello", _record=mock_forward_record)

        assert repr(traced) == "TracedOutput('hello')"

    def test_repr_with_numeric_value(self, mock_forward_record: ForwardRecord) -> None:
        """repr() works with numeric values."""
        traced = TracedOutput(value=42, _record=mock_forward_record)

        assert repr(traced) == "TracedOutput(42)"


class TestTracedOutputValueAccess:
    """Tests for accessing the underlying value."""

    def test_value_is_accessible(self, mock_forward_record: ForwardRecord) -> None:
        """The underlying value can be accessed via .value."""
        original_value = {"result": "success", "data": [1, 2, 3]}
        traced = TracedOutput(value=original_value, _record=mock_forward_record)

        assert traced.value is original_value

    def test_record_is_accessible(self, mock_forward_record: ForwardRecord) -> None:
        """The ForwardRecord can be accessed via ._record."""
        traced = TracedOutput(value="test", _record=mock_forward_record)

        assert traced._record is mock_forward_record
        assert traced._record.node_outputs["node_1"] == "test output"


class TestTracedOutputEquality:
    """Tests for TracedOutput equality comparison."""

    def test_equal_traced_outputs(self, mock_forward_record: ForwardRecord) -> None:
        """TracedOutputs with same value and record are equal."""
        traced1 = TracedOutput(value="hello", _record=mock_forward_record)
        traced2 = TracedOutput(value="hello", _record=mock_forward_record)

        assert traced1 == traced2

    def test_unequal_values(self, mock_forward_record: ForwardRecord) -> None:
        """TracedOutputs with different values are not equal."""
        traced1 = TracedOutput(value="hello", _record=mock_forward_record)
        traced2 = TracedOutput(value="world", _record=mock_forward_record)

        assert traced1 != traced2
