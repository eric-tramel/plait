"""Unit tests for functional text operations."""

import plait.functional as F
from plait.values import Value, ValueKind, valueify


class TestRender:
    """Tests for F.render."""

    def test_render_success(self) -> None:
        """render formats a template with structured vars."""
        template = valueify("Hello {name}", kind=ValueKind.FSTRING)
        vars_value = valueify({"name": "Ada"})

        result = F.render(template, vars_value)

        assert result.kind == ValueKind.TEXT
        assert result.payload == "Hello Ada"

    def test_render_propagates_error(self) -> None:
        """render short-circuits on input errors."""
        err = Value(ValueKind.ERROR, ValueError("boom"), ref="err")
        result = F.render(err, {"name": "Ada"})

        assert result is err

    def test_render_missing_key_returns_error(self) -> None:
        """render returns Value(ERROR) on format failure."""
        template = valueify("Hello {missing}", kind=ValueKind.FSTRING)
        vars_value = valueify({"name": "Ada"})

        result = F.render(template, vars_value)

        assert result.kind == ValueKind.ERROR
        assert isinstance(result.payload, KeyError)
        assert result.meta["op"] == "render"


class TestFormat:
    """Tests for F.format."""

    def test_format_success(self) -> None:
        """format applies a format string to structured vars."""
        fmt = valueify("Name: {name}", kind=ValueKind.TEXT)
        vars_value = valueify({"name": "Ada"})

        result = F.format(fmt, vars_value)

        assert result.kind == ValueKind.TEXT
        assert result.payload == "Name: Ada"


class TestConcat:
    """Tests for F.concat."""

    def test_concat_success(self) -> None:
        """concat joins text parts with a separator."""
        result = F.concat("a", valueify("b"), sep="-")

        assert result.kind == ValueKind.TEXT
        assert result.payload == "a-b"

    def test_concat_non_text_returns_error(self) -> None:
        """concat returns Value(ERROR) on non-text inputs."""
        result = F.concat("a", valueify(1))

        assert result.kind == ValueKind.ERROR
        assert isinstance(result.payload, TypeError)
        assert result.meta["op"] == "concat"
