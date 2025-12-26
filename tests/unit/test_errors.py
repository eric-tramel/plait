"""Tests for inf-engine error types.

Tests the exception hierarchy and error attributes to ensure
proper error handling and recovery capabilities.
"""

import pytest

from inf_engine.errors import ExecutionError, InfEngineError, RateLimitError


class TestInfEngineError:
    """Tests for the base InfEngineError class."""

    def test_creation(self) -> None:
        """InfEngineError can be created with a message."""
        error = InfEngineError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"

    def test_inheritance(self) -> None:
        """InfEngineError inherits from Exception."""
        error = InfEngineError("test")
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """InfEngineError can be raised and caught."""
        with pytest.raises(InfEngineError) as exc_info:
            raise InfEngineError("raised error")
        assert "raised error" in str(exc_info.value)


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_creation_with_retry_after(self) -> None:
        """RateLimitError can be created with retry_after."""
        error = RateLimitError("Rate limit exceeded", retry_after=30.0)
        assert str(error) == "Rate limit exceeded"
        assert error.message == "Rate limit exceeded"
        assert error.retry_after == 30.0

    def test_creation_without_retry_after(self) -> None:
        """RateLimitError can be created without retry_after."""
        error = RateLimitError("Too many requests")
        assert str(error) == "Too many requests"
        assert error.retry_after is None

    def test_inheritance_from_inf_engine_error(self) -> None:
        """RateLimitError inherits from InfEngineError."""
        error = RateLimitError("test")
        assert isinstance(error, InfEngineError)
        assert isinstance(error, Exception)

    def test_can_catch_as_inf_engine_error(self) -> None:
        """RateLimitError can be caught as InfEngineError."""
        with pytest.raises(InfEngineError):
            raise RateLimitError("rate limited", retry_after=5.0)

    def test_retry_after_zero(self) -> None:
        """RateLimitError handles retry_after of zero."""
        error = RateLimitError("limit hit", retry_after=0.0)
        assert error.retry_after == 0.0

    def test_retry_after_float(self) -> None:
        """RateLimitError handles fractional retry_after values."""
        error = RateLimitError("limit hit", retry_after=1.5)
        assert error.retry_after == 1.5


class TestExecutionError:
    """Tests for ExecutionError."""

    def test_creation_basic(self) -> None:
        """ExecutionError can be created with just a message."""
        error = ExecutionError("Task failed")
        assert str(error) == "Task failed"
        assert error.message == "Task failed"
        assert error.node_id is None
        assert error.cause is None

    def test_creation_with_node_id(self) -> None:
        """ExecutionError can be created with a node_id."""
        error = ExecutionError("Task failed", node_id="node_123")
        assert error.node_id == "node_123"
        assert error.cause is None

    def test_creation_with_cause(self) -> None:
        """ExecutionError can be created with an underlying cause."""
        cause = ValueError("Invalid response")
        error = ExecutionError("LLM call failed", cause=cause)
        assert error.cause is cause
        assert isinstance(error.cause, ValueError)
        assert error.node_id is None

    def test_creation_with_all_attributes(self) -> None:
        """ExecutionError can be created with all attributes."""
        cause = RuntimeError("Connection timeout")
        error = ExecutionError(
            "Execution failed",
            node_id="llm_node_42",
            cause=cause,
        )
        assert error.message == "Execution failed"
        assert error.node_id == "llm_node_42"
        assert error.cause is cause

    def test_inheritance_from_inf_engine_error(self) -> None:
        """ExecutionError inherits from InfEngineError."""
        error = ExecutionError("test")
        assert isinstance(error, InfEngineError)
        assert isinstance(error, Exception)

    def test_can_catch_as_inf_engine_error(self) -> None:
        """ExecutionError can be caught as InfEngineError."""
        with pytest.raises(InfEngineError):
            raise ExecutionError("execution failed", node_id="n1")


class TestErrorHierarchy:
    """Tests for the exception hierarchy as a whole."""

    def test_catch_all_inf_engine_errors(self) -> None:
        """All error types can be caught with InfEngineError."""
        errors = [
            InfEngineError("base error"),
            RateLimitError("rate limit", retry_after=10.0),
            ExecutionError("execution failed", node_id="n1"),
        ]

        for error in errors:
            with pytest.raises(InfEngineError):
                raise error

    def test_specific_error_types_are_distinct(self) -> None:
        """Different error types can be distinguished."""
        rate_error = RateLimitError("rate limit")
        exec_error = ExecutionError("exec failed")

        # RateLimitError is not ExecutionError
        assert not isinstance(rate_error, ExecutionError)
        # ExecutionError is not RateLimitError
        assert not isinstance(exec_error, RateLimitError)
        # Both are InfEngineError
        assert isinstance(rate_error, InfEngineError)
        assert isinstance(exec_error, InfEngineError)
