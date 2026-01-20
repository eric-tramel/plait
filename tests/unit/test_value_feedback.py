"""Tests for Value feedback helpers and tape behavior."""

import pytest

from plait.graph import GraphNode, InferenceGraph, NodeRef
from plait.module import Module
from plait.optimization.record import ForwardRecord, get_record
from plait.values import (
    Value,
    ValueKind,
    ValueRef,
    attach_tape,
    collect_records,
    collect_tape_ids,
    no_grad,
    normalize_feedback_payload,
    normalize_feedback_value,
)


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


def test_normalize_feedback_payload_variants() -> None:
    assert normalize_feedback_payload(None) == []
    assert normalize_feedback_payload(Value(ValueKind.TEXT, "hi")) == [["hi"]]
    assert normalize_feedback_payload([]) == []
    assert normalize_feedback_payload("") == []
    assert normalize_feedback_payload("ok") == [["ok"]]
    assert normalize_feedback_payload([["a", "", 1], []]) == [["a", "1"]]
    assert normalize_feedback_payload(["a", "", 2]) == [["a", "2"]]
    assert normalize_feedback_payload(()) == []
    assert normalize_feedback_payload(("x",)) == [["x"]]
    assert normalize_feedback_payload(3) == [["3"]]


def test_normalize_feedback_value_strips_tape_meta() -> None:
    value = Value(ValueKind.TEXT, "hi", meta={"_tape_ids": ["t1"], "score": 0.25})
    normalized = normalize_feedback_value(value)
    assert normalized.kind == ValueKind.STRUCTURED
    assert normalized.payload == [["hi"]]
    assert normalized.meta == {"score": 0.25}

    raw = normalize_feedback_value("raw")
    assert raw.payload == [["raw"]]


def test_value_getitem_fallback_without_functional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plait.values as values

    def raise_missing(_: str) -> None:
        raise ModuleNotFoundError("blocked")

    monkeypatch.setattr(values, "import_module", raise_missing)

    error_value = Value(ValueKind.ERROR, ValueError("boom"))
    assert error_value["x"] is error_value
    assert error_value.get("x") is error_value

    value = Value(ValueKind.STRUCTURED, {"a": Value(ValueKind.TEXT, "hi")})
    assert value.get("a") is value.payload["a"]

    raw_value = Value(ValueKind.STRUCTURED, {"b": "text"})
    raw_selected = raw_value.get("b")
    assert isinstance(raw_selected, Value)
    assert raw_selected.payload == "text"

    default = Value(ValueKind.TEXT, "fallback")
    assert value.get("missing", default=default) is default


def test_value_methods_without_tracing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins
    import types
    from collections.abc import Mapping, Sequence

    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> types.ModuleType:
        if name == "plait.tracing.context":
            raise ModuleNotFoundError("blocked")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    value = Value(ValueKind.STRUCTURED, {"a": "hi"})
    _ = value["a"]
    _ = list(iter(value))
    _ = list(value.keys())
    _ = list(value.values())
    _ = list(value.items())


def test_value_methods_use_tracing_context(monkeypatch: pytest.MonkeyPatch) -> None:
    import plait.tracing.context as trace_context

    class DummyTracer:
        def record_getitem(self, value: Value, key: str | int) -> Value:
            return Value(ValueKind.TEXT, f"getitem:{key}")

        def record_iter(self, value: Value) -> Value:
            return Value(ValueKind.TEXT, "iter")

        def record_method(self, value: Value, name: str) -> str:
            return f"{name}-called"

    tracer = DummyTracer()
    monkeypatch.setattr(trace_context, "get_trace_context", lambda: tracer)

    value = Value(ValueKind.STRUCTURED, {"a": "hi"}, ref="node:1")
    assert value["a"].payload == "getitem:a"
    assert [item.payload for item in value] == ["iter"]
    assert value.keys() == "keys-called"
    assert value.values() == "values-called"
    assert value.items() == "items-called"


def test_no_grad_skips_tape_attachment() -> None:
    value = Value(ValueKind.TEXT, "hello")
    record = _make_record(DummyModule())
    with no_grad():
        attach_tape(value, record)
    assert "_tape_ids" not in value.meta


def test_attach_collect_and_detach_tape_nested() -> None:
    record = _make_record(DummyModule())
    nested = Value(
        ValueKind.STRUCTURED,
        {"inner": Value(ValueKind.TEXT, "hi"), "refs": [ValueRef("node:x")]},
    )
    attach_tape(nested, record)
    tape_ids = collect_tape_ids(nested)
    assert tape_ids
    records = collect_records(nested)
    assert records == [record]

    nested.detach_tape()
    assert "_tape_ids" not in nested.meta
    assert "_tape_ids" not in nested.payload["inner"].meta
    for tape_id in tape_ids:
        with pytest.raises(KeyError):
            get_record(tape_id)


def test_value_get_and_getitem_defaults() -> None:
    inner = Value(ValueKind.TEXT, "hi")
    value = Value(ValueKind.STRUCTURED, {"a": inner})
    assert value.get("a") is inner

    default = Value(ValueKind.TEXT, "fallback")
    assert value.get("missing", default=default) is default

    error_value = Value(ValueKind.ERROR, ValueError("boom"))
    assert error_value.get("x") is error_value
    assert error_value["x"] is error_value

    missing = Value(ValueKind.STRUCTURED, {})["nope"]
    assert missing.kind == ValueKind.ERROR


@pytest.mark.asyncio
async def test_value_backward_unbound_and_retain() -> None:
    module = DummyModule()
    record = _make_record(module)
    loss_value = Value(ValueKind.STRUCTURED, [["feedback"]])
    attach_tape(loss_value, record)

    with pytest.raises(RuntimeError):
        await Value.backward([loss_value])

    await Value.backward([loss_value], grad=loss_value, retain_graph=True)

    assert "_tape_ids" in loss_value.meta
    for tape_id in collect_tape_ids(loss_value):
        assert get_record(tape_id) is record

    loss_value.detach_tape()
