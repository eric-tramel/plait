"""Unit tests for error/control and collection ops."""

import asyncio
from types import SimpleNamespace

import inf_engine.functional as F
from inf_engine.values import Value, ValueKind, valueify


def test_with_meta_merges() -> None:
    """with_meta adds metadata to a value."""
    value = valueify("hi")

    result = F.with_meta(value, source="test")

    assert result.meta["source"] == "test"


def test_unwrap_or_recovers_from_error() -> None:
    """unwrap_or returns default and marks recovery."""
    err = Value(ValueKind.ERROR, ValueError("boom"), ref="err")

    result = F.unwrap_or(err, "fallback")

    assert result.kind == ValueKind.TEXT
    assert result.payload == "fallback"
    assert result.meta["recovered_from"] == "err"


def test_is_error_flag() -> None:
    """is_error returns structured flag data."""
    err = Value(ValueKind.ERROR, ValueError("boom"), ref="err")

    result = F.is_error(err)

    assert result.kind == ValueKind.STRUCTURED
    assert result.payload == {"is_error": True, "ref": "err"}


def test_map_values_preserves_errors() -> None:
    """map_values applies fn and preserves Value(ERROR)."""
    err = Value(ValueKind.ERROR, ValueError("boom"), ref="err")
    values = [valueify(" a "), err]

    result = F.map_values(F.strip, values)

    assert result[0].payload == "a"
    assert result[1] is err


def test_zip_values_length_mismatch_returns_error() -> None:
    """zip_values returns Value(ERROR) on length mismatch."""
    result = F.zip_values([valueify("a")], [valueify("b"), valueify("c")])

    assert isinstance(result, Value)
    assert result.kind == ValueKind.ERROR


def test_zip_values_preserves_errors() -> None:
    """zip_values keeps error elements in output."""
    err = Value(ValueKind.ERROR, ValueError("boom"), ref="err")
    result = F.zip_values([valueify("a"), err], [valueify("b"), valueify("c")])

    assert isinstance(result, list)
    assert result[1][0] is err


def test_stack_structured() -> None:
    """stack_structured stacks structured payloads."""
    values = [valueify({"a": 1}), valueify({"a": 2})]

    result = F.stack_structured(values)

    assert result.kind == ValueKind.STRUCTURED
    assert isinstance(result.payload[0]["a"], Value)
    assert isinstance(result.payload[1]["a"], Value)
    assert result.payload[0]["a"].payload == 1
    assert result.payload[1]["a"].payload == 2


def test_stack_structured_non_structured_error() -> None:
    """stack_structured returns Value(ERROR) for non-structured inputs."""
    result = F.stack_structured([valueify("hi")])

    assert result.kind == ValueKind.ERROR


def test_chat_complete_success() -> None:
    """chat_complete wraps response in Value(RESPONSE)."""

    class DummyCompletions:
        async def create(self, **kwargs):
            return {"id": "resp", "choices": [{"message": {"content": "hi"}}]}

    dummy = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))

    result = asyncio.run(
        F.chat_complete(
            dummy,
            messages=[{"role": "user", "content": "hi"}],
            model="test",
        )
    )

    assert result.kind == ValueKind.RESPONSE
    assert result.payload["id"] == "resp"


def test_chat_complete_error_status() -> None:
    """chat_complete captures http status on error."""

    class DummyError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__("boom")
            self.status_code = status_code

    class DummyCompletions:
        async def create(self, **kwargs):
            raise DummyError(429)

    dummy = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))

    result = asyncio.run(
        F.chat_complete(
            dummy,
            messages=[{"role": "user", "content": "hi"}],
            model="test",
        )
    )

    assert result.kind == ValueKind.ERROR
    assert result.meta["http_status"] == 429
