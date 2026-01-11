"""Unit tests for BatchResult type."""

import pytest

from plait.execution.types import BatchResult


class TestBatchResultCreation:
    """Tests for BatchResult instantiation."""

    def test_create_successful_result(self) -> None:
        """BatchResult can be created for successful execution."""
        result = BatchResult(
            index=0,
            input="hello",
            output="HELLO",
            error=None,
        )

        assert result.index == 0
        assert result.input == "hello"
        assert result.output == "HELLO"
        assert result.error is None

    def test_create_failed_result(self) -> None:
        """BatchResult can be created for failed execution."""
        error = ValueError("Something went wrong")
        result = BatchResult(
            index=1,
            input="bad input",
            output=None,
            error=error,
        )

        assert result.index == 1
        assert result.input == "bad input"
        assert result.output is None
        assert result.error is error

    def test_create_with_complex_input_output(self) -> None:
        """BatchResult works with complex input and output types."""
        input_data = {"text": "hello", "count": 5}
        output_data = {"result": "processed", "items": [1, 2, 3]}

        result = BatchResult(
            index=0,
            input=input_data,
            output=output_data,
            error=None,
        )

        assert result.input == input_data
        assert result.output == output_data

    def test_create_with_none_input(self) -> None:
        """BatchResult can have None as input."""
        result = BatchResult(
            index=0,
            input=None,
            output="result",
            error=None,
        )

        assert result.input is None
        assert result.output == "result"


class TestBatchResultOkProperty:
    """Tests for the ok property."""

    def test_ok_true_for_success(self) -> None:
        """ok returns True when error is None."""
        result = BatchResult(
            index=0,
            input="test",
            output="RESULT",
            error=None,
        )

        assert result.ok is True

    def test_ok_false_for_failure(self) -> None:
        """ok returns False when error is not None."""
        result = BatchResult(
            index=0,
            input="test",
            output=None,
            error=ValueError("error"),
        )

        assert result.ok is False

    def test_ok_false_even_with_output(self) -> None:
        """ok returns False if error is set, regardless of output."""
        # Edge case: error is set but output is also present
        result = BatchResult(
            index=0,
            input="test",
            output="partial result",
            error=RuntimeError("partial failure"),
        )

        assert result.ok is False

    def test_ok_with_different_error_types(self) -> None:
        """ok works with different exception types."""
        errors = [
            ValueError("value error"),
            TypeError("type error"),
            RuntimeError("runtime error"),
            KeyError("key error"),
            Exception("generic error"),
        ]

        for error in errors:
            result = BatchResult(
                index=0,
                input="test",
                output=None,
                error=error,
            )
            assert result.ok is False, f"Expected ok=False for {type(error).__name__}"


class TestBatchResultImmutability:
    """Tests for BatchResult immutability (frozen=True)."""

    def test_cannot_modify_index(self) -> None:
        """Cannot modify index after creation."""
        result = BatchResult(index=0, input="x", output="X", error=None)

        with pytest.raises(AttributeError):
            result.index = 1  # type: ignore[misc]

    def test_cannot_modify_input(self) -> None:
        """Cannot modify input after creation."""
        result = BatchResult(index=0, input="x", output="X", error=None)

        with pytest.raises(AttributeError):
            result.input = "y"  # type: ignore[misc]

    def test_cannot_modify_output(self) -> None:
        """Cannot modify output after creation."""
        result = BatchResult(index=0, input="x", output="X", error=None)

        with pytest.raises(AttributeError):
            result.output = "Y"  # type: ignore[misc]

    def test_cannot_modify_error(self) -> None:
        """Cannot modify error after creation."""
        result = BatchResult(index=0, input="x", output="X", error=None)

        with pytest.raises(AttributeError):
            result.error = ValueError("new error")  # type: ignore[misc]


class TestBatchResultRepr:
    """Tests for string representation."""

    def test_repr_successful(self) -> None:
        """repr shows ok=True and output for successful results."""
        result = BatchResult(index=0, input="test", output="RESULT", error=None)
        repr_str = repr(result)

        assert "index=0" in repr_str
        assert "ok=True" in repr_str
        assert "output='RESULT'" in repr_str

    def test_repr_failed(self) -> None:
        """repr shows ok=False and error for failed results."""
        error = ValueError("test error")
        result = BatchResult(index=1, input="test", output=None, error=error)
        repr_str = repr(result)

        assert "index=1" in repr_str
        assert "ok=False" in repr_str
        assert "error=" in repr_str
        assert "ValueError" in repr_str


class TestBatchResultEquality:
    """Tests for BatchResult equality (dataclass default)."""

    def test_equal_results(self) -> None:
        """Two BatchResults with same values are equal."""
        result1 = BatchResult(index=0, input="x", output="X", error=None)
        result2 = BatchResult(index=0, input="x", output="X", error=None)

        assert result1 == result2

    def test_different_index(self) -> None:
        """BatchResults with different indices are not equal."""
        result1 = BatchResult(index=0, input="x", output="X", error=None)
        result2 = BatchResult(index=1, input="x", output="X", error=None)

        assert result1 != result2

    def test_different_input(self) -> None:
        """BatchResults with different inputs are not equal."""
        result1 = BatchResult(index=0, input="x", output="X", error=None)
        result2 = BatchResult(index=0, input="y", output="X", error=None)

        assert result1 != result2

    def test_different_output(self) -> None:
        """BatchResults with different outputs are not equal."""
        result1 = BatchResult(index=0, input="x", output="X", error=None)
        result2 = BatchResult(index=0, input="x", output="Y", error=None)

        assert result1 != result2

    def test_same_error_instance(self) -> None:
        """BatchResults with same error instance are equal."""
        error = ValueError("error")
        result1 = BatchResult(index=0, input="x", output=None, error=error)
        result2 = BatchResult(index=0, input="x", output=None, error=error)

        assert result1 == result2


class TestBatchResultHashability:
    """Tests for BatchResult hashability (frozen dataclass)."""

    def test_hashable_success(self) -> None:
        """Successful BatchResults are hashable."""
        result = BatchResult(index=0, input="test", output="RESULT", error=None)

        # Should not raise
        hash(result)

    def test_usable_in_set(self) -> None:
        """BatchResults can be used in a set."""
        result1 = BatchResult(index=0, input="a", output="A", error=None)
        result2 = BatchResult(index=1, input="b", output="B", error=None)
        result3 = BatchResult(
            index=0, input="a", output="A", error=None
        )  # Duplicate of 1

        result_set = {result1, result2, result3}

        assert len(result_set) == 2  # Duplicates removed


class TestBatchResultGenericType:
    """Tests for BatchResult generic type parameter."""

    def test_string_type(self) -> None:
        """BatchResult works with string type."""
        result: BatchResult[str] = BatchResult(
            index=0, input="hello", output="HELLO", error=None
        )
        assert result.ok

    def test_int_type(self) -> None:
        """BatchResult works with int type."""
        result: BatchResult[int] = BatchResult(index=0, input=5, output=10, error=None)
        assert result.output == 10

    def test_dict_type(self) -> None:
        """BatchResult works with dict type."""
        result: BatchResult[dict[str, int]] = BatchResult(
            index=0,
            input={"a": 1},
            output={"b": 2},
            error=None,
        )
        assert result.output == {"b": 2}

    def test_mixed_input_output_types(self) -> None:
        """BatchResult can have different input and output types conceptually."""
        # Note: The generic parameter determines both input and output types,
        # so we use Any for mixed type scenarios
        result: BatchResult[object] = BatchResult(
            index=0,
            input="hello",
            output={"processed": True},  # Different type - allowed with object
            error=None,
        )
        assert result.ok
