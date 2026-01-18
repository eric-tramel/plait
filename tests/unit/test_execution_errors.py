"""Unit tests for Value(ERROR) short-circuit behavior during execution."""

import pytest

from plait.execution.scheduler import Scheduler
from plait.execution.state import ExecutionState
from plait.graph import GraphNode, InferenceGraph
from plait.module import Module
from plait.tracing.tracer import InputNode
from plait.values import (
    Value,
    ValueKind,
    ValueRef,
    first_error_value,
    has_error_value,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper modules for testing
# ─────────────────────────────────────────────────────────────────────────────


class TrackedModule(Module):
    """Module that tracks whether forward() was called."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_called = False

    def forward(self, *args: str, **kwargs: str) -> str:
        """Mark that forward was called and return a result."""
        self.forward_called = True
        return "processed"


class UppercaseModule(Module):
    """Simple module that converts input to uppercase."""

    def forward(self, text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()


# ─────────────────────────────────────────────────────────────────────────────
# has_error_value and first_error_value Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHasErrorValue:
    """Tests for has_error_value helper function."""

    def test_no_values_returns_false(self) -> None:
        """Returns False when there are no Values at all."""
        assert has_error_value("hello", 42, key="value") is False

    def test_ok_values_returns_false(self) -> None:
        """Returns False when all Values are non-error."""
        v1 = Value(ValueKind.TEXT, "hello")
        v2 = Value(ValueKind.INT, 42)
        assert has_error_value(v1, v2) is False

    def test_error_value_in_args_returns_true(self) -> None:
        """Returns True when an error Value is in args."""
        ok = Value(ValueKind.TEXT, "hello")
        error = Value(ValueKind.ERROR, ValueError("test"))
        assert has_error_value(ok, error) is True

    def test_error_value_in_kwargs_returns_true(self) -> None:
        """Returns True when an error Value is in kwargs."""
        ok = Value(ValueKind.TEXT, "hello")
        error = Value(ValueKind.ERROR, ValueError("test"))
        assert has_error_value(ok, key=error) is True

    def test_nested_error_value_in_list_returns_true(self) -> None:
        """Returns True when an error Value is nested in a list."""
        ok = Value(ValueKind.TEXT, "hello")
        error = Value(ValueKind.ERROR, ValueError("test"))
        assert has_error_value([ok, error]) is True

    def test_nested_error_value_in_dict_returns_true(self) -> None:
        """Returns True when an error Value is nested in a dict."""
        ok = Value(ValueKind.TEXT, "hello")
        error = Value(ValueKind.ERROR, ValueError("test"))
        assert has_error_value({"ok": ok, "err": error}) is True

    def test_deeply_nested_error_value_returns_true(self) -> None:
        """Returns True when an error Value is deeply nested."""
        error = Value(ValueKind.ERROR, ValueError("test"))
        nested = {"level1": [{"level2": error}]}
        assert has_error_value(nested) is True


class TestFirstErrorValue:
    """Tests for first_error_value helper function."""

    def test_no_errors_returns_none(self) -> None:
        """Returns None when there are no error Values."""
        ok = Value(ValueKind.TEXT, "hello")
        assert first_error_value(ok) is None

    def test_single_error_returns_it(self) -> None:
        """Returns the error Value when there is one."""
        error = Value(ValueKind.ERROR, ValueError("test"))
        result = first_error_value(error)
        assert result is error

    def test_first_of_multiple_errors(self) -> None:
        """Returns the first error when there are multiple."""
        error1 = Value(ValueKind.ERROR, ValueError("first"))
        error2 = Value(ValueKind.ERROR, ValueError("second"))
        result = first_error_value(error1, error2)
        assert result is error1

    def test_error_in_kwargs_found(self) -> None:
        """Finds error in kwargs after checking args."""
        ok = Value(ValueKind.TEXT, "hello")
        error = Value(ValueKind.ERROR, ValueError("in kwargs"))
        result = first_error_value(ok, key=error)
        assert result is error

    def test_nested_error_found(self) -> None:
        """Finds error nested in containers."""
        ok = Value(ValueKind.TEXT, "hello")
        error = Value(ValueKind.ERROR, ValueError("nested"))
        result = first_error_value([ok, {"inner": error}])
        assert result is error

    def test_nested_dict_without_error_returns_none(self) -> None:
        """Returns None when nested dict has no error Values."""
        ok = Value(ValueKind.TEXT, "hello")
        result = first_error_value({"inner": {"ok": ok}})
        assert result is None

    def test_nested_list_without_error_returns_none(self) -> None:
        """Returns None when nested list has no error Values."""
        ok = Value(ValueKind.TEXT, "hello")
        result = first_error_value([ok, [Value(ValueKind.INT, 1)]])
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Short-Circuit Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerErrorShortCircuit:
    """Tests for Value(ERROR) short-circuit behavior in Scheduler."""

    @pytest.mark.asyncio
    async def test_short_circuit_with_error_value_arg(self) -> None:
        """Task is not executed when arg contains Value(ERROR)."""
        tracked = TrackedModule()
        error_value = Value(ValueKind.ERROR, ValueError("upstream error"))

        # Create a graph where the input is already an error
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(error_value),
            args=(),
            kwargs={},
            dependencies=[],
        )
        process_node = GraphNode(
            id="TrackedModule_1",
            module=tracked,
            args=(ValueRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "TrackedModule_1": process_node,
            },
            input_ids=["input:input_0"],
            output_ids=["TrackedModule_1"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # forward() should NOT have been called
        assert tracked.forward_called is False

        # Output should be the error value
        result = outputs["TrackedModule_1"]
        assert isinstance(result, Value)
        assert result.kind == ValueKind.ERROR

    @pytest.mark.asyncio
    async def test_short_circuit_preserves_first_error(self) -> None:
        """When multiple error Values exist, the first one is propagated."""
        error1 = Value(
            ValueKind.ERROR, ValueError("first error"), meta={"order": "first"}
        )
        error2 = Value(
            ValueKind.ERROR, ValueError("second error"), meta={"order": "second"}
        )

        # Create inputs with two error values
        input_node1 = GraphNode(
            id="input:input_0",
            module=InputNode(error1),
            args=(),
            kwargs={},
            dependencies=[],
        )
        input_node2 = GraphNode(
            id="input:input_1",
            module=InputNode(error2),
            args=(),
            kwargs={},
            dependencies=[],
        )
        tracked = TrackedModule()
        process_node = GraphNode(
            id="TrackedModule_1",
            module=tracked,
            args=(ValueRef("input:input_0"), ValueRef("input:input_1")),
            kwargs={},
            dependencies=["input:input_0", "input:input_1"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node1,
                "input:input_1": input_node2,
                "TrackedModule_1": process_node,
            },
            input_ids=["input:input_0", "input:input_1"],
            output_ids=["TrackedModule_1"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # forward() should NOT have been called
        assert tracked.forward_called is False

        # Output should be the first error
        result = outputs["TrackedModule_1"]
        assert isinstance(result, Value)
        assert result.kind == ValueKind.ERROR
        assert result.meta.get("order") == "first"

    @pytest.mark.asyncio
    async def test_no_short_circuit_with_ok_values(self) -> None:
        """Task is executed normally when args contain no errors."""
        upper = UppercaseModule()

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(Value(ValueKind.TEXT, "hello")),
            args=(),
            kwargs={},
            dependencies=[],
        )
        process_node = GraphNode(
            id="UppercaseModule_1",
            module=upper,
            args=(ValueRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "UppercaseModule_1": process_node,
            },
            input_ids=["input:input_0"],
            output_ids=["UppercaseModule_1"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # forward() should have been called and produced uppercase
        assert outputs["UppercaseModule_1"] == "HELLO"

    @pytest.mark.asyncio
    async def test_short_circuit_in_chain(self) -> None:
        """Error propagates through a chain of modules."""
        error_value = Value(ValueKind.ERROR, ValueError("source error"))

        # Build a chain: input -> module1 -> module2
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(error_value),
            args=(),
            kwargs={},
            dependencies=[],
        )
        tracked1 = TrackedModule()
        module1_node = GraphNode(
            id="TrackedModule_1",
            module=tracked1,
            args=(ValueRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        tracked2 = TrackedModule()
        module2_node = GraphNode(
            id="TrackedModule_2",
            module=tracked2,
            args=(ValueRef("TrackedModule_1"),),
            kwargs={},
            dependencies=["TrackedModule_1"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "TrackedModule_1": module1_node,
                "TrackedModule_2": module2_node,
            },
            input_ids=["input:input_0"],
            output_ids=["TrackedModule_2"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # Neither forward() should have been called
        assert tracked1.forward_called is False
        assert tracked2.forward_called is False

        # Final output should be the error
        result = outputs["TrackedModule_2"]
        assert isinstance(result, Value)
        assert result.kind == ValueKind.ERROR

    @pytest.mark.asyncio
    async def test_error_in_one_branch_does_not_affect_other(self) -> None:
        """In parallel branches, error in one branch doesn't affect the other."""
        error_value = Value(ValueKind.ERROR, ValueError("branch error"))
        ok_value = Value(ValueKind.TEXT, "hello")

        # Build parallel branches: input_ok -> module1, input_err -> module2
        input_ok = GraphNode(
            id="input:input_0",
            module=InputNode(ok_value),
            args=(),
            kwargs={},
            dependencies=[],
        )
        input_err = GraphNode(
            id="input:input_1",
            module=InputNode(error_value),
            args=(),
            kwargs={},
            dependencies=[],
        )
        upper = UppercaseModule()
        module1_node = GraphNode(
            id="UppercaseModule_1",
            module=upper,
            args=(ValueRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        tracked = TrackedModule()
        module2_node = GraphNode(
            id="TrackedModule_1",
            module=tracked,
            args=(ValueRef("input:input_1"),),
            kwargs={},
            dependencies=["input:input_1"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_ok,
                "input:input_1": input_err,
                "UppercaseModule_1": module1_node,
                "TrackedModule_1": module2_node,
            },
            input_ids=["input:input_0", "input:input_1"],
            output_ids=["UppercaseModule_1", "TrackedModule_1"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # OK branch should have executed
        assert outputs["UppercaseModule_1"] == "HELLO"

        # Error branch should not have called forward
        assert tracked.forward_called is False

        # Error branch output should be the error
        result = outputs["TrackedModule_1"]
        assert isinstance(result, Value)
        assert result.kind == ValueKind.ERROR
