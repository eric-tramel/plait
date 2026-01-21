"""Tests for Value.backward and tape registry behavior."""

import pytest

from plait.graph import GraphNode, InferenceGraph, NodeRef
from plait.module import Module
from plait.optimization.backward import BackwardResult
from plait.optimization.record import ForwardRecord, get_record
from plait.values import Value, ValueKind, attach_tape, collect_tape_ids


class DummyModule(Module):
    def forward(self, x: str) -> str:
        return x


class CaptureModule(Module):
    def __init__(self) -> None:
        super().__init__()
        self.feedback: Value | None = None

    def forward(self, x: str) -> str:
        return x

    async def backward(self, feedback: Value, ctx: object) -> BackwardResult:  # noqa: ARG002
        self.feedback = feedback
        return BackwardResult()


@pytest.mark.asyncio
async def test_value_backward_raises_without_tape() -> None:
    value = Value(ValueKind.STRUCTURED, [["no tape"]])
    with pytest.raises(RuntimeError):
        await value.backward()


@pytest.mark.asyncio
async def test_value_backward_releases_tape() -> None:
    module = DummyModule()

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
        node_outputs={"input:x": "hello", "Module_1": "hello"},
        module_map={"Module_1": module},
    )

    loss_value = Value(ValueKind.STRUCTURED, [["feedback"]])
    attach_tape(loss_value, record)
    tape_ids = collect_tape_ids(loss_value)
    assert tape_ids

    await loss_value.backward()

    # Tape ids should be released and removed
    assert "_tape_ids" not in loss_value.meta
    for tape_id in tape_ids:
        with pytest.raises(KeyError):
            get_record(tape_id)


@pytest.mark.asyncio
async def test_value_backward_uses_per_record_grads() -> None:
    module_a = CaptureModule()
    module_b = CaptureModule()

    def make_record(node_id: str, module: Module) -> ForwardRecord:
        input_id = f"input:{node_id}"
        input_node = GraphNode(
            id=input_id,
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        module_node = GraphNode(
            id=node_id,
            module=module,
            args=(NodeRef(input_id),),
            kwargs={},
            dependencies=[input_id],
        )
        graph = InferenceGraph(
            nodes={input_id: input_node, node_id: module_node},
            input_ids=[input_id],
            output_ids=[node_id],
        )
        return ForwardRecord(
            graph=graph,
            node_inputs={node_id: {"0": "hello"}},
            node_outputs={input_id: "hello", node_id: "hello"},
            module_map={node_id: module},
        )

    record_a = make_record("Module_A", module_a)
    record_b = make_record("Module_B", module_b)

    loss_a = Value(ValueKind.STRUCTURED, [["loss-a"]])
    loss_b = Value(ValueKind.STRUCTURED, [["loss-b"]])
    attach_tape(loss_a, record_a)
    attach_tape(loss_b, record_b)

    grad_a = Value(ValueKind.STRUCTURED, [["grad-a"]], meta={"score": 0.1})
    grad_b = Value(ValueKind.STRUCTURED, [["grad-b"]], meta={"score": 0.9})

    await Value.backward([loss_a, loss_b], grad=[grad_a, grad_b])

    assert module_a.feedback is not None
    assert module_b.feedback is not None
    assert module_a.feedback.payload == [["grad-a"]]
    assert module_b.feedback.payload == [["grad-b"]]
    assert module_a.feedback.meta.get("score") == 0.1
    assert module_b.feedback.meta.get("score") == 0.9
