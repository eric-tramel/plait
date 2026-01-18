"""Tests for Value.backward and tape registry behavior."""

import pytest

from plait.graph import GraphNode, InferenceGraph, NodeRef
from plait.module import Module
from plait.optimization.record import ForwardRecord, get_record
from plait.values import Value, ValueKind, attach_tape, collect_tape_ids


class DummyModule(Module):
    def forward(self, x: str) -> str:
        return x


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
