"""Additional tests for functional text operations."""

import inf_engine.functional as F
from inf_engine.values import Value, ValueKind, valueify


def test_strip_lower_upper() -> None:
    """strip/lower/upper operate on text inputs."""
    text = valueify("  Hi  ")

    stripped = F.strip(text)
    lowered = F.lower(stripped)
    uppered = F.upper(lowered)

    assert stripped.payload == "Hi"
    assert lowered.payload == "hi"
    assert uppered.payload == "HI"


def test_strip_non_text_returns_error() -> None:
    """strip returns Value(ERROR) for non-text input."""
    result = F.strip(valueify(123))

    assert result.kind == ValueKind.ERROR
    assert result.meta["op"] == "strip"


def test_try_render_matches_render() -> None:
    """try_render matches render behavior on error."""
    template = valueify("{missing}", kind=ValueKind.FSTRING)
    vars_value = valueify({"name": "Ada"})

    result = F.try_render(template, vars_value)

    assert result.kind == ValueKind.ERROR


def test_concat_with_error_propagates() -> None:
    """concat short-circuits on input errors."""
    err = Value(ValueKind.ERROR, ValueError("boom"), ref="err")

    result = F.concat("a", err)

    assert result is err
