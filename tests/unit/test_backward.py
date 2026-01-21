"""Unit tests for backward propagation with Value feedback."""

import pytest

from plait.graph import GraphNode, InferenceGraph, NodeRef
from plait.module import Module
from plait.optimization.backward import (
    BackwardContext,
    BackwardResult,
    _combine_feedback,
    _propagate_backward_value,
)
from plait.optimization.record import ForwardRecord
from plait.parameter import Parameter
from plait.values import Value, ValueKind


class MockModule(Module):
    """Mock module for testing backward pass."""

    def __init__(self, name: str = "mock") -> None:
        super().__init__()
        self.name = name
        self.backward_calls: list[tuple[Value, BackwardContext]] = []

    def forward(self, x: str) -> str:
        return f"processed_{x}"

    async def backward(self, feedback: Value, ctx: BackwardContext) -> BackwardResult:
        self.backward_calls.append((feedback, ctx))
        return await super().backward(feedback, ctx)


class MockModuleWithParams(Module):
    """Mock module with learnable parameter for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.instructions = Parameter(
            value="Be helpful",
            description="Instructions for response generation",
            requires_grad=True,
        )
        self.backward_calls: list[tuple[Value, BackwardContext]] = []

    def forward(self, x: str) -> str:
        return f"{self.instructions.value}: {x}"

    async def backward(self, feedback: Value, ctx: BackwardContext) -> BackwardResult:
        self.backward_calls.append((feedback, ctx))
        result = BackwardResult()

        for input_name in ctx.inputs:
            result.input_feedback[input_name] = feedback

        result.parameter_feedback["instructions"] = Value(
            kind=ValueKind.STRUCTURED,
            payload=[["Update instructions"]],
        )

        return result


class TestCombineFeedback:
    def test_combine_feedback_concatenates_payloads(self) -> None:
        fb1 = Value(ValueKind.STRUCTURED, [["First"]], meta={"score": 0.6})
        fb2 = Value(ValueKind.STRUCTURED, [["Second"]], meta={"score": 0.8})

        combined = _combine_feedback([fb1, fb2])

        assert combined.payload == [["First"], ["Second"]]
        assert combined.meta["scores"] == [0.6, 0.8]


class TestPropagateBackward:
    @pytest.mark.asyncio
    async def test_propagate_single_node(self) -> None:
        module = MockModule("single")

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

        feedback = Value(ValueKind.STRUCTURED, [["Good output"]], meta={"score": 0.9})
        await _propagate_backward_value(feedback, None, record)

        assert len(module.backward_calls) == 1
        fb, ctx = module.backward_calls[0]
        assert fb.payload == [["Good output"]]
        assert ctx.node_id == "Module_1"

    @pytest.mark.asyncio
    async def test_propagate_fan_out_combines_feedback(self) -> None:
        upstream = MockModule("upstream")
        downstream_a = MockModule("downstream_a")
        downstream_b = MockModule("downstream_b")

        input_node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        upstream_node = GraphNode(
            id="Upstream",
            module=upstream,
            args=(NodeRef("input:x"),),
            kwargs={},
            dependencies=["input:x"],
        )
        node_a = GraphNode(
            id="DownA",
            module=downstream_a,
            args=(NodeRef("Upstream"),),
            kwargs={},
            dependencies=["Upstream"],
        )
        node_b = GraphNode(
            id="DownB",
            module=downstream_b,
            args=(NodeRef("Upstream"),),
            kwargs={},
            dependencies=["Upstream"],
        )

        graph = InferenceGraph(
            nodes={
                "input:x": input_node,
                "Upstream": upstream_node,
                "DownA": node_a,
                "DownB": node_b,
            },
            input_ids=["input:x"],
            output_ids=["DownA", "DownB"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={
                "Upstream": {"0": "hello"},
                "DownA": {"0": "up_hello"},
                "DownB": {"0": "up_hello"},
            },
            node_outputs={
                "input:x": "hello",
                "Upstream": "up_hello",
                "DownA": "down_a",
                "DownB": "down_b",
            },
            module_map={
                "Upstream": upstream,
                "DownA": downstream_a,
                "DownB": downstream_b,
            },
        )

        feedback = Value(ValueKind.STRUCTURED, [["Root"]])
        await _propagate_backward_value(feedback, None, record)

        # Upstream should receive combined feedback from both downstream nodes
        assert len(upstream.backward_calls) == 1
        combined_payload = upstream.backward_calls[0][0].payload
        assert combined_payload == [["Root"], ["Root"]]

    @pytest.mark.asyncio
    async def test_parameter_feedback_accumulation(self) -> None:
        module = MockModuleWithParams()

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

        feedback = Value(ValueKind.STRUCTURED, [["Improve output"]])
        await _propagate_backward_value(feedback, None, record)

        assert module.instructions.get_accumulated_feedback() == ["Update instructions"]
