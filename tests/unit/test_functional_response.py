"""Unit tests for response and metadata functional ops."""

from dataclasses import dataclass

import inf_engine.functional as F
from inf_engine.values import Value, ValueKind


@dataclass
class DummyResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int


def test_extract_text_from_response() -> None:
    """extract_text pulls content from response payloads."""
    resp = Value(ValueKind.RESPONSE, DummyResponse("hello", "model", 1, 2))

    result = F.extract_text(resp)

    assert result.kind == ValueKind.TEXT
    assert result.payload == "hello"


def test_extract_meta_from_response() -> None:
    """extract_meta returns structured metadata when available."""
    resp = Value(ValueKind.RESPONSE, DummyResponse("hello", "model", 1, 2))

    result = F.extract_meta(resp)

    assert result.kind == ValueKind.STRUCTURED
    assert result.payload["model"] == "model"
    assert result.payload["input_tokens"] == 1
    assert result.payload["output_tokens"] == 2


def test_extract_text_missing_returns_error() -> None:
    """extract_text returns Value(ERROR) when content is unavailable."""
    resp = Value(ValueKind.RESPONSE, {"message": "no content"})

    result = F.extract_text(resp)

    assert result.kind == ValueKind.ERROR
    assert result.meta["op"] == "extract_text"
