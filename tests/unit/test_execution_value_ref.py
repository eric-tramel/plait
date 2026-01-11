"""Unit tests for ValueRef resolution during execution."""

from plait.execution.state import ExecutionState, TaskResult
from plait.graph import GraphNode, InferenceGraph, NodeRef
from plait.module import Module
from plait.tracing.tracer import InputNode
from plait.values import Value, ValueKind, ValueRef

# ─────────────────────────────────────────────────────────────────────────────
# Helper modules for testing
# ─────────────────────────────────────────────────────────────────────────────


class ConcatModule(Module):
    """Simple module that concatenates string inputs."""

    def forward(self, *args: str) -> str:
        """Concatenate all string arguments."""
        return " ".join(str(arg) for arg in args)


class UppercaseModule(Module):
    """Simple module that converts input to uppercase."""

    def forward(self, text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions for creating test graphs
# ─────────────────────────────────────────────────────────────────────────────


def create_graph_with_value_ref() -> InferenceGraph:
    """Create a graph with ValueRef in args."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode("hello"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    # Use ValueRef instead of NodeRef
    upper_node = GraphNode(
        id="UppercaseModule_1",
        module=UppercaseModule(),
        args=(ValueRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    return InferenceGraph(
        nodes={
            "input:input_0": input_node,
            "UppercaseModule_1": upper_node,
        },
        input_ids=["input:input_0"],
        output_ids=["UppercaseModule_1"],
    )


def create_graph_with_nested_value_ref() -> InferenceGraph:
    """Create a graph with ValueRef nested in args."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode("first"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    input_node2 = GraphNode(
        id="input:input_1",
        module=InputNode("second"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    # ValueRefs in a list and dict
    concat_node = GraphNode(
        id="ConcatModule_1",
        module=ConcatModule(),
        args=([ValueRef("input:input_0"), ValueRef("input:input_1")],),
        kwargs={},
        dependencies=["input:input_0", "input:input_1"],
    )
    return InferenceGraph(
        nodes={
            "input:input_0": input_node,
            "input:input_1": input_node2,
            "ConcatModule_1": concat_node,
        },
        input_ids=["input:input_0", "input:input_1"],
        output_ids=["ConcatModule_1"],
    )


def create_graph_with_value_ref_in_kwargs() -> InferenceGraph:
    """Create a graph with ValueRef in kwargs."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode("test"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    upper_node = GraphNode(
        id="UppercaseModule_1",
        module=UppercaseModule(),
        args=(),
        kwargs={"text": ValueRef("input:input_0")},
        dependencies=["input:input_0"],
    )
    return InferenceGraph(
        nodes={
            "input:input_0": input_node,
            "UppercaseModule_1": upper_node,
        },
        input_ids=["input:input_0"],
        output_ids=["UppercaseModule_1"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ValueRef Resolution Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestValueRefResolution:
    """Tests for ValueRef resolution in ExecutionState."""

    def test_resolve_value_ref_basic(self) -> None:
        """ValueRef in args is resolved to the producing node's result."""
        graph = create_graph_with_value_ref()
        state = ExecutionState(graph)

        # Complete input node
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=1.0),
        )

        # Resolve args containing ValueRef
        node = graph.nodes["UppercaseModule_1"]
        resolved_args = state._resolve_args(node.args)
        assert resolved_args == ("hello",)

    def test_resolve_value_ref_in_kwargs(self) -> None:
        """ValueRef in kwargs is resolved to the producing node's result."""
        graph = create_graph_with_value_ref_in_kwargs()
        state = ExecutionState(graph)

        # Complete input node
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="test", duration_ms=1.0),
        )

        # Resolve kwargs containing ValueRef
        node = graph.nodes["UppercaseModule_1"]
        resolved_kwargs = state._resolve_kwargs(node.kwargs)
        assert resolved_kwargs["text"] == "test"

    def test_resolve_nested_value_ref(self) -> None:
        """ValueRef nested in lists/dicts is resolved correctly."""
        graph = create_graph_with_nested_value_ref()
        state = ExecutionState(graph)

        # Complete both input nodes
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="first", duration_ms=1.0),
        )
        state.mark_complete(
            "input:input_1",
            TaskResult(node_id="input:input_1", value="second", duration_ms=1.0),
        )

        # Resolve args containing nested ValueRefs
        node = graph.nodes["ConcatModule_1"]
        resolved_args = state._resolve_args(node.args)
        assert resolved_args == (["first", "second"],)

    def test_resolve_value_maintains_value_wrapper(self) -> None:
        """When result is a Value, the Value is preserved during resolution."""
        graph = create_graph_with_value_ref()
        state = ExecutionState(graph)

        # Complete input node with a Value result
        input_value = Value(ValueKind.TEXT, "hello", ref="input:input_0")
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value=input_value, duration_ms=1.0),
        )

        # Resolve args
        node = graph.nodes["UppercaseModule_1"]
        resolved_args = state._resolve_args(node.args)
        result = resolved_args[0]
        assert isinstance(result, Value)
        assert result.payload == "hello"


class TestValueRefWithNodeRef:
    """Tests for compatibility between ValueRef and NodeRef resolution."""

    def test_noderef_still_works(self) -> None:
        """NodeRef in args is still resolved correctly."""
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode("hello"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        upper_node = GraphNode(
            id="UppercaseModule_1",
            module=UppercaseModule(),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "UppercaseModule_1": upper_node,
            },
            input_ids=["input:input_0"],
            output_ids=["UppercaseModule_1"],
        )
        state = ExecutionState(graph)

        # Complete input node
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=1.0),
        )

        # Resolve args
        node = graph.nodes["UppercaseModule_1"]
        resolved_args = state._resolve_args(node.args)
        assert resolved_args == ("hello",)

    def test_mixed_noderef_and_valueref(self) -> None:
        """Both NodeRef and ValueRef can be used in the same args."""
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode("first"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        input_node2 = GraphNode(
            id="input:input_1",
            module=InputNode("second"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        # Mix NodeRef and ValueRef
        concat_node = GraphNode(
            id="ConcatModule_1",
            module=ConcatModule(),
            args=(NodeRef("input:input_0"), ValueRef("input:input_1")),
            kwargs={},
            dependencies=["input:input_0", "input:input_1"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "input:input_1": input_node2,
                "ConcatModule_1": concat_node,
            },
            input_ids=["input:input_0", "input:input_1"],
            output_ids=["ConcatModule_1"],
        )
        state = ExecutionState(graph)

        # Complete both input nodes
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="first", duration_ms=1.0),
        )
        state.mark_complete(
            "input:input_1",
            TaskResult(node_id="input:input_1", value="second", duration_ms=1.0),
        )

        # Resolve args
        node = graph.nodes["ConcatModule_1"]
        resolved_args = state._resolve_args(node.args)
        assert resolved_args == ("first", "second")
