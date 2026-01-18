"""MWE demonstrating Value-based feedback propagation through backward traversal."""

import pytest

from plait.graph import GraphNode, InferenceGraph, NodeRef
from plait.module import Module
from plait.optimization.backward import (
    BackwardContext,
    BackwardResult,
    _propagate_backward_value,
)
from plait.optimization.record import ForwardRecord
from plait.values import Value, ValueKind


class UpstreamRecorder(Module):
    """Records the feedback payload it receives during backward."""

    def __init__(self) -> None:
        super().__init__()
        self.backward_payloads: list[list[list[str]]] = []

    def forward(self, x: str) -> str:
        return f"up_{x}"

    async def backward(self, feedback: Value, ctx: BackwardContext) -> BackwardResult:
        self.backward_payloads.append(feedback.payload)
        return await super().backward(feedback, ctx)


class DownstreamTransformer(Module):
    """Transforms feedback before propagating to its input."""

    def forward(self, x: str) -> str:
        return f"down_{x}"

    async def backward(self, feedback: Value, ctx: BackwardContext) -> BackwardResult:
        result = BackwardResult()
        result.input_feedback["0"] = Value(
            kind=ValueKind.STRUCTURED,
            payload=[["transformed-by-downstream"]],
            meta=dict(feedback.meta),
        )
        return result


@pytest.mark.asyncio
async def test_mwe_transformed_feedback_reaches_upstream() -> None:
    """Upstream node should receive downstream-transformed feedback."""
    upstream = UpstreamRecorder()
    downstream = DownstreamTransformer()

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
    downstream_node = GraphNode(
        id="Downstream",
        module=downstream,
        args=(NodeRef("Upstream"),),
        kwargs={},
        dependencies=["Upstream"],
    )

    graph = InferenceGraph(
        nodes={
            "input:x": input_node,
            "Upstream": upstream_node,
            "Downstream": downstream_node,
        },
        input_ids=["input:x"],
        output_ids=["Downstream"],
    )

    record = ForwardRecord(
        graph=graph,
        node_inputs={
            "Upstream": {"0": "hello"},
            "Downstream": {"0": "up_hello"},
        },
        node_outputs={
            "input:x": "hello",
            "Upstream": "up_hello",
            "Downstream": "down_up_hello",
        },
        module_map={
            "Upstream": upstream,
            "Downstream": downstream,
        },
    )

    loss_value = Value(ValueKind.STRUCTURED, [["original"]])
    await _propagate_backward_value(loss_value, None, record)

    assert upstream.backward_payloads == [[["transformed-by-downstream"]]]
