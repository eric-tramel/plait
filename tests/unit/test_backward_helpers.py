"""Tests for backward helper utilities."""

from __future__ import annotations

from typing import Any

import pytest

from plait.graph import GraphNode, InferenceGraph, NodeRef
from plait.module import Module
from plait.optimization.backward import (
    BackwardContext,
    _build_dependents_map,
    _combine_feedback,
    _propagate_backward_value,
    _resolve_input_node,
)
from plait.optimization.optimizer import Optimizer
from plait.optimization.record import ForwardRecord
from plait.values import Value, ValueKind, ValueRef


class DummyModule(Module):
    def forward(self, x: str) -> str:
        return x


def _make_record(module: Module) -> ForwardRecord:
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

    return ForwardRecord(
        graph=graph,
        node_inputs={"Module_1": {"0": "hello"}},
        node_outputs={"input:x": "hello", "Module_1": "hello"},
        module_map={"Module_1": module},
    )


@pytest.mark.asyncio
async def test_backward_context_reason_uses_llm() -> None:
    class Reasoner:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def __call__(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "analysis"

    graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
    reasoner = Reasoner()
    ctx = BackwardContext(
        node_id="node",
        inputs={},
        output="out",
        graph=graph,
        all_results={},
        downstream_feedback=[],
        reasoning_llm=reasoner,
    )

    result = await ctx.reason("why")
    assert result == "analysis"
    assert reasoner.prompts == ["why"]


def test_combine_feedback_merges_scores() -> None:
    fb1 = Value(ValueKind.STRUCTURED, [["a"]], meta={"score": 0.5})
    fb2 = Value(ValueKind.STRUCTURED, [["b"]])
    combined = _combine_feedback([fb1, fb2])

    assert combined.payload == [["a"], ["b"]]
    assert combined.meta["scores"] == [0.5]

    single = _combine_feedback([Value(ValueKind.TEXT, "ok")])
    assert single.payload == [["ok"]]


def test_resolve_input_node_positional_and_kwargs() -> None:
    module = DummyModule()
    input_a = GraphNode(
        id="input:a",
        module=None,
        args=(),
        kwargs={},
        dependencies=[],
    )
    input_b = GraphNode(
        id="input:b",
        module=None,
        args=(),
        kwargs={},
        dependencies=[],
    )
    node = GraphNode(
        id="Module_1",
        module=module,
        args=(NodeRef("input:a"),),
        kwargs={"kw": ValueRef("input:b")},
        dependencies=["input:a", "input:b"],
    )
    graph = InferenceGraph(
        nodes={"input:a": input_a, "input:b": input_b, "Module_1": node},
        input_ids=["input:a", "input:b"],
        output_ids=["Module_1"],
    )
    record = ForwardRecord(
        graph=graph,
        node_inputs={"Module_1": {"0": "a", "kw": "b"}},
        node_outputs={"input:a": "a", "input:b": "b", "Module_1": "a"},
        module_map={"Module_1": module},
    )

    assert _resolve_input_node("Module_1", "0", record) == "input:a"
    assert _resolve_input_node("Module_1", "kw", record) == "input:b"
    assert _resolve_input_node("Module_1", "param", record) == "input:a"

    bad_node = GraphNode(
        id="Module_bad",
        module=module,
        args=(123,),
        kwargs={},
        dependencies=[],
    )
    bad_graph = InferenceGraph(
        nodes={
            "input:a": input_a,
            "input:b": input_b,
            "Module_1": node,
            "Module_bad": bad_node,
        },
        input_ids=["input:a", "input:b"],
        output_ids=["Module_1"],
    )
    bad_record = ForwardRecord(
        graph=bad_graph,
        node_inputs={},
        node_outputs={},
        module_map={},
    )
    assert _resolve_input_node("Module_bad", "0", bad_record) is None


def test_build_dependents_map() -> None:
    node_a = GraphNode(
        id="A",
        module=None,
        args=(),
        kwargs={},
        dependencies=[],
    )
    node_b = GraphNode(
        id="B",
        module=None,
        args=(),
        kwargs={},
        dependencies=["A"],
    )
    node_c = GraphNode(
        id="C",
        module=None,
        args=(),
        kwargs={},
        dependencies=["A", "B"],
    )
    graph = InferenceGraph(
        nodes={"A": node_a, "B": node_b, "C": node_c},
        input_ids=["A"],
        output_ids=["C"],
    )

    dependents = _build_dependents_map(graph)
    assert dependents["A"] == ["B", "C"]
    assert dependents["B"] == ["C"]
    assert dependents["C"] == []


@pytest.mark.asyncio
async def test_propagate_backward_captures_record_and_skips_unused() -> None:
    class NoFeedbackModule(Module):
        def forward(self, x: str) -> str:
            return x

        async def backward(self, feedback: Any, ctx: Any) -> Any:
            from plait.optimization.backward import BackwardResult

            return BackwardResult()

    module = DummyModule()
    no_feedback = NoFeedbackModule()
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
    top_node = GraphNode(
        id="Module_top",
        module=no_feedback,
        args=(NodeRef("Module_1"),),
        kwargs={},
        dependencies=["Module_1"],
    )
    graph = InferenceGraph(
        nodes={
            "input:x": input_node,
            "Module_1": module_node,
            "Module_top": top_node,
        },
        input_ids=["input:x"],
        output_ids=["Module_top"],
    )
    record = ForwardRecord(
        graph=graph,
        node_inputs={"Module_1": {"0": "hello"}, "Module_top": {"0": "hello"}},
        node_outputs={
            "input:x": "hello",
            "Module_1": "hello",
            "Module_top": "hello",
        },
        module_map={"Module_1": module, "Module_top": no_feedback},
    )

    class DummyOptimizer(Optimizer):
        def __init__(self) -> None:
            super().__init__([])

        async def step(self) -> dict[str, str]:
            return {}

    optimizer = DummyOptimizer()
    root = Value(ValueKind.STRUCTURED, [["feedback"]])
    await _propagate_backward_value(
        root=root, grad=None, record=record, optimizer=optimizer
    )

    assert optimizer._records == [record]
