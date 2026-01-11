"""Unit tests for backward propagation.

This file contains tests for PR-067: Backward pass infrastructure.
Tests cover _propagate_backward() and _combine_feedback() functions.
"""

import pytest

from plait.graph import GraphNode, InferenceGraph, NodeRef
from plait.module import InferenceModule
from plait.optimization.backward import (
    BackwardContext,
    BackwardResult,
    _combine_feedback,
    _propagate_backward,
)
from plait.optimization.feedback import Feedback, FeedbackType
from plait.optimization.record import ForwardRecord
from plait.parameter import Parameter


class MockModule(InferenceModule):
    """Mock module for testing backward pass."""

    def __init__(self, name: str = "mock") -> None:
        super().__init__()
        self.name = name
        self.backward_calls: list[tuple[Feedback, BackwardContext]] = []

    def forward(self, x: str) -> str:
        return f"processed_{x}"

    async def backward(
        self,
        feedback: Feedback,
        ctx: BackwardContext,
    ) -> BackwardResult:
        """Record backward calls for verification."""
        self.backward_calls.append((feedback, ctx))
        return await super().backward(feedback, ctx)


class MockModuleWithParams(InferenceModule):
    """Mock module with learnable parameter for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.instructions = Parameter(
            value="Be helpful",
            description="Instructions for response generation",
            requires_grad=True,
        )
        self.backward_calls: list[tuple[Feedback, BackwardContext]] = []

    def forward(self, x: str) -> str:
        return f"{self.instructions.value}: {x}"

    async def backward(
        self,
        feedback: Feedback,
        ctx: BackwardContext,
    ) -> BackwardResult:
        """Generate feedback for parameter."""
        self.backward_calls.append((feedback, ctx))
        result = BackwardResult()

        # Pass feedback to input
        for input_name in ctx.inputs:
            result.input_feedback[input_name] = feedback

        # Generate parameter feedback
        result.parameter_feedback["instructions"] = (
            f"Received feedback: {feedback.content}"
        )

        return result


class TestCombineFeedback:
    """Tests for _combine_feedback function."""

    def test_combine_single_feedback(self) -> None:
        """Single feedback is returned unchanged."""
        feedback = Feedback(content="Single", score=0.8)
        result = _combine_feedback([feedback])

        assert result is feedback

    def test_combine_two_feedbacks(self) -> None:
        """Two feedbacks are combined with averaged score."""
        fb1 = Feedback(content="First", score=0.6)
        fb2 = Feedback(content="Second", score=0.8)

        result = _combine_feedback([fb1, fb2])

        assert "[Downstream 1] First" in result.content
        assert "[Downstream 2] Second" in result.content
        assert result.score == 0.7  # (0.6 + 0.8) / 2
        assert result.feedback_type == FeedbackType.COMPOSITE

    def test_combine_three_feedbacks(self) -> None:
        """Three feedbacks are combined correctly."""
        fb1 = Feedback(content="A", score=0.6)
        fb2 = Feedback(content="B", score=0.7)
        fb3 = Feedback(content="C", score=0.8)

        result = _combine_feedback([fb1, fb2, fb3])

        assert "[Downstream 1] A" in result.content
        assert "[Downstream 2] B" in result.content
        assert "[Downstream 3] C" in result.content
        assert result.score == pytest.approx(0.7)  # (0.6 + 0.7 + 0.8) / 3

    def test_combine_feedbacks_no_scores(self) -> None:
        """Feedbacks without scores result in None score."""
        fb1 = Feedback(content="First")
        fb2 = Feedback(content="Second")

        result = _combine_feedback([fb1, fb2])

        assert result.score is None

    def test_combine_feedbacks_mixed_scores(self) -> None:
        """Only non-None scores are averaged."""
        fb1 = Feedback(content="First", score=0.6)
        fb2 = Feedback(content="Second", score=None)
        fb3 = Feedback(content="Third", score=0.8)

        result = _combine_feedback([fb1, fb2, fb3])

        assert result.score == 0.7  # (0.6 + 0.8) / 2


class TestPropagateBackwardLinearGraph:
    """Tests for _propagate_backward with linear graphs."""

    @pytest.mark.asyncio
    async def test_propagate_single_node(self) -> None:
        """Backward propagates through single node."""
        module = MockModule("single")

        # Create graph: input -> module -> output
        input_node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        module_node = GraphNode(
            id="Module_1",
            module=module,
            args=(NodeRef("input:x"),),
            kwargs={},
            dependencies=["input:x"],
        )

        graph = InferenceGraph(
            nodes={"input:x": input_node, "Module_1": module_node},
            input_ids=["input:x"],
            output_ids=["Module_1"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={"Module_1": {"0": "hello"}},
            node_outputs={"input:x": "hello", "Module_1": "processed_hello"},
            module_map={"Module_1": module},
        )

        feedback = Feedback(content="Good output", score=0.9)
        await _propagate_backward(feedback, record)

        # Verify backward was called
        assert len(module.backward_calls) == 1
        fb, ctx = module.backward_calls[0]
        assert fb.content == "Good output"
        assert ctx.node_id == "Module_1"

    @pytest.mark.asyncio
    async def test_propagate_two_nodes_linear(self) -> None:
        """Backward propagates through two nodes in sequence."""
        module1 = MockModule("first")
        module2 = MockModule("second")

        # Create graph: input -> module1 -> module2 -> output
        input_node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        node1 = GraphNode(
            id="Module_1",
            module=module1,
            args=(NodeRef("input:x"),),
            kwargs={},
            dependencies=["input:x"],
        )
        node2 = GraphNode(
            id="Module_2",
            module=module2,
            args=(NodeRef("Module_1"),),
            kwargs={},
            dependencies=["Module_1"],
        )

        graph = InferenceGraph(
            nodes={"input:x": input_node, "Module_1": node1, "Module_2": node2},
            input_ids=["input:x"],
            output_ids=["Module_2"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={
                "Module_1": {"0": "hello"},
                "Module_2": {"0": "processed_hello"},
            },
            node_outputs={
                "input:x": "hello",
                "Module_1": "processed_hello",
                "Module_2": "processed_processed_hello",
            },
            module_map={"Module_1": module1, "Module_2": module2},
        )

        feedback = Feedback(content="Good", score=0.8)
        await _propagate_backward(feedback, record)

        # Module2 should be called first (reverse topo order)
        assert len(module2.backward_calls) == 1
        # Module1 should be called with feedback from module2
        assert len(module1.backward_calls) == 1


class TestPropagateBackwardFanOut:
    """Tests for _propagate_backward with fan-out graphs."""

    @pytest.mark.asyncio
    async def test_propagate_fan_out_aggregation(self) -> None:
        """Feedback from multiple downstream nodes is aggregated."""
        module = MockModule("shared")
        downstream1 = MockModule("down1")
        downstream2 = MockModule("down2")

        # Create graph: input -> module -> [downstream1, downstream2]
        input_node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        shared_node = GraphNode(
            id="Shared",
            module=module,
            args=(NodeRef("input:x"),),
            kwargs={},
            dependencies=["input:x"],
        )
        down1_node = GraphNode(
            id="Down1",
            module=downstream1,
            args=(NodeRef("Shared"),),
            kwargs={},
            dependencies=["Shared"],
        )
        down2_node = GraphNode(
            id="Down2",
            module=downstream2,
            args=(NodeRef("Shared"),),
            kwargs={},
            dependencies=["Shared"],
        )

        graph = InferenceGraph(
            nodes={
                "input:x": input_node,
                "Shared": shared_node,
                "Down1": down1_node,
                "Down2": down2_node,
            },
            input_ids=["input:x"],
            output_ids=["Down1", "Down2"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={
                "Shared": {"0": "hello"},
                "Down1": {"0": "processed"},
                "Down2": {"0": "processed"},
            },
            node_outputs={
                "input:x": "hello",
                "Shared": "processed",
                "Down1": "result1",
                "Down2": "result2",
            },
            module_map={
                "Shared": module,
                "Down1": downstream1,
                "Down2": downstream2,
            },
        )

        feedback = Feedback(content="Overall good", score=0.8)
        await _propagate_backward(feedback, record)

        # Shared module should receive combined feedback from both downstreams
        assert len(module.backward_calls) == 1
        fb, ctx = module.backward_calls[0]
        # Combined feedback should have COMPOSITE type
        assert (
            fb.feedback_type == FeedbackType.COMPOSITE
            or len(ctx.downstream_feedback) > 1
        )


class TestPropagateBackwardWithParameters:
    """Tests for _propagate_backward with parameter feedback accumulation."""

    @pytest.mark.asyncio
    async def test_parameter_feedback_accumulated(self) -> None:
        """Parameter feedback is accumulated into Parameters."""
        module = MockModuleWithParams()

        # Create graph: input -> module -> output
        input_node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        module_node = GraphNode(
            id="Module_1",
            module=module,
            args=(NodeRef("input:x"),),
            kwargs={},
            dependencies=["input:x"],
        )

        graph = InferenceGraph(
            nodes={"input:x": input_node, "Module_1": module_node},
            input_ids=["input:x"],
            output_ids=["Module_1"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={"Module_1": {"0": "test input"}},
            node_outputs={
                "input:x": "test input",
                "Module_1": "Be helpful: test input",
            },
            module_map={"Module_1": module},
        )

        # Clear any existing feedback
        module.instructions.zero_feedback()

        feedback = Feedback(content="Response was too brief", score=0.5)
        await _propagate_backward(feedback, record)

        # Parameter should have accumulated feedback
        assert len(module.instructions._feedback_buffer) == 1
        assert "Response was too brief" in module.instructions._feedback_buffer[0]


class TestPropagateBackwardEdgeCases:
    """Edge case tests for _propagate_backward."""

    @pytest.mark.asyncio
    async def test_empty_graph_no_crash(self) -> None:
        """Backward pass on empty graph doesn't crash."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        feedback = Feedback(content="Test")
        # Should not raise
        await _propagate_backward(feedback, record)

    @pytest.mark.asyncio
    async def test_input_only_graph(self) -> None:
        """Graph with only input nodes doesn't call backward on anything."""
        input_node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )

        graph = InferenceGraph(
            nodes={"input:x": input_node},
            input_ids=["input:x"],
            output_ids=["input:x"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={"input:x": "hello"},
            module_map={},
        )

        feedback = Feedback(content="Test")
        # Should not raise - input nodes don't have backward
        await _propagate_backward(feedback, record)


class TestResolveInputNode:
    """Tests for _resolve_input_node function."""

    def test_resolve_node_ref_from_args(self) -> None:
        """Resolves NodeRef from positional arguments."""
        from plait.optimization.backward import _resolve_input_node

        # Create a graph with node that has NodeRef in args
        input_node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        module_node = GraphNode(
            id="module_1",
            module=None,
            args=(NodeRef("input:x"),),
            kwargs={},
            dependencies=["input:x"],
        )

        graph = InferenceGraph(
            nodes={"input:x": input_node, "module_1": module_node},
            input_ids=["input:x"],
            output_ids=["module_1"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        # Resolve by positional index
        result = _resolve_input_node("module_1", "0", record)
        assert result == "input:x"

    def test_resolve_node_ref_from_kwargs(self) -> None:
        """Resolves NodeRef from keyword arguments."""
        from plait.optimization.backward import _resolve_input_node

        input_node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        module_node = GraphNode(
            id="module_1",
            module=None,
            args=(),
            kwargs={"prompt": NodeRef("input:x")},
            dependencies=["input:x"],
        )

        graph = InferenceGraph(
            nodes={"input:x": input_node, "module_1": module_node},
            input_ids=["input:x"],
            output_ids=["module_1"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        # Resolve by kwarg name
        result = _resolve_input_node("module_1", "prompt", record)
        assert result == "input:x"

    def test_resolve_value_ref_from_args(self) -> None:
        """Resolves ValueRef from positional arguments."""
        from plait.optimization.backward import _resolve_input_node
        from plait.values import ValueRef

        input_node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        module_node = GraphNode(
            id="module_1",
            module=None,
            args=(ValueRef("input:x"),),
            kwargs={},
            dependencies=["input:x"],
        )

        graph = InferenceGraph(
            nodes={"input:x": input_node, "module_1": module_node},
            input_ids=["input:x"],
            output_ids=["module_1"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        # Resolve by positional index - ValueRef should work just like NodeRef
        result = _resolve_input_node("module_1", "0", record)
        assert result == "input:x"

    def test_resolve_value_ref_from_kwargs(self) -> None:
        """Resolves ValueRef from keyword arguments."""
        from plait.optimization.backward import _resolve_input_node
        from plait.values import ValueRef

        input_node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        module_node = GraphNode(
            id="module_1",
            module=None,
            args=(),
            kwargs={"context": ValueRef("input:x")},
            dependencies=["input:x"],
        )

        graph = InferenceGraph(
            nodes={"input:x": input_node, "module_1": module_node},
            input_ids=["input:x"],
            output_ids=["module_1"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        # Resolve by kwarg name - ValueRef should work just like NodeRef
        result = _resolve_input_node("module_1", "context", record)
        assert result == "input:x"

    def test_resolve_single_value_ref_arg(self) -> None:
        """Single ValueRef arg resolves by name even if not index."""
        from plait.optimization.backward import _resolve_input_node
        from plait.values import ValueRef

        input_node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        module_node = GraphNode(
            id="module_1",
            module=None,
            args=(ValueRef("input:x"),),  # Single arg
            kwargs={},
            dependencies=["input:x"],
        )

        graph = InferenceGraph(
            nodes={"input:x": input_node, "module_1": module_node},
            input_ids=["input:x"],
            output_ids=["module_1"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        # With single arg, name lookup also works
        result = _resolve_input_node("module_1", "prompt", record)
        assert result == "input:x"

    def test_resolve_returns_none_for_missing(self) -> None:
        """Returns None when input cannot be resolved."""
        from plait.optimization.backward import _resolve_input_node

        module_node = GraphNode(
            id="module_1",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )

        graph = InferenceGraph(
            nodes={"module_1": module_node},
            input_ids=[],
            output_ids=["module_1"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        result = _resolve_input_node("module_1", "nonexistent", record)
        assert result is None
