"""Tests for LossModule behavior and routing."""

from typing import Any, cast

import pytest

from plait.optimization.loss import LossModule
from plait.tracing.context import trace_context
from plait.tracing.tracer import Tracer
from plait.values import Value, ValueKind


@pytest.mark.asyncio
async def test_loss_module_attaches_tape_ids_and_traces() -> None:
    async def loss_fn(output: object, **kwargs: object) -> Value:
        return Value(ValueKind.STRUCTURED, [["ok"]])

    loss = LossModule(loss_fn)

    input_value = Value(ValueKind.TEXT, "hi", meta={"_tape_ids": ["t1"]})
    result = await loss.forward(input_value)
    assert "t1" in result.meta.get("_tape_ids", [])

    tracer = Tracer()
    with trace_context(tracer):
        traced = loss("x")
    assert isinstance(traced, Value)
    assert traced.ref in tracer.nodes


@pytest.mark.asyncio
async def test_loss_module_backward_routes_to_input() -> None:
    async def loss_fn(output: object, **kwargs: object) -> Value:
        return Value(ValueKind.STRUCTURED, [["ok"]])

    loss = LossModule(loss_fn)

    class Ctx:
        inputs = {"arg_0": "x"}

    feedback = Value(ValueKind.STRUCTURED, [["fb"]])
    result = await loss.backward(feedback, Ctx())
    assert result.input_feedback["0"].payload == [["fb"]]


@pytest.mark.asyncio
async def test_loss_module_backward_respects_feedback_input() -> None:
    async def loss_fn(output: object, **kwargs: object) -> Value:
        return Value(ValueKind.STRUCTURED, [["ok"]])

    loss = LossModule(loss_fn, feedback_input="output")

    class Ctx:
        inputs = {"output": "x"}

    result = await loss.backward("fb", Ctx())
    assert result.input_feedback["output"].payload == [["fb"]]


def test_loss_module_attribute_passthrough() -> None:
    async def loss_fn(output: object, **kwargs: object) -> Value:
        return Value(ValueKind.STRUCTURED, [["ok"]])

    loss_fn_any = cast(Any, loss_fn)
    loss_fn_any.custom = "initial"
    loss = LossModule(loss_fn)
    assert loss.custom == "initial"

    loss.custom = "updated"
    assert loss_fn_any.custom == "updated"
