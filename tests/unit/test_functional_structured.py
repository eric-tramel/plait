"""Unit tests for structured functional operations."""

import plait.functional as F
from plait.values import ValueKind, valueify


class TestSelect:
    """Tests for F.select."""

    def test_select_string_path(self) -> None:
        """select supports dot-separated string paths."""
        struct = valueify({"user": {"name": "Ada"}})

        result = F.select(struct, "user.name")

        assert result.kind == ValueKind.TEXT
        assert result.payload == "Ada"

    def test_select_list_path(self) -> None:
        """select supports list paths with indices."""
        struct = valueify({"items": ["a", "b"]})

        result = F.select(struct, ["items", 1])

        assert result.kind == ValueKind.TEXT
        assert result.payload == "b"

    def test_select_default_on_missing(self) -> None:
        """select returns default when path is missing."""
        struct = valueify({"a": 1})

        result = F.select(struct, "missing", default="fallback")

        assert result.kind == ValueKind.TEXT
        assert result.payload == "fallback"

    def test_select_chaining_structured(self) -> None:
        """select preserves structured kind for chaining."""
        struct = valueify({"a": {"b": {"c": 2}}})

        level = F.select(struct, "a")
        result = F.select(level, "b.c")

        assert level.kind == ValueKind.STRUCTURED
        assert result.kind == ValueKind.INT
        assert result.payload == 2

    def test_select_missing_without_default_returns_error(self) -> None:
        """select returns Value(ERROR) when missing and no default."""
        struct = valueify({"a": 1})

        result = F.select(struct, "missing")

        assert result.kind == ValueKind.ERROR
        assert result.meta["op"] == "select"


class TestParseStructured:
    """Tests for F.parse_structured."""

    def test_parse_structured_success(self) -> None:
        """parse_structured parses JSON into structured Value."""
        result = F.parse_structured('{"a": 1}')

        assert result.kind == ValueKind.STRUCTURED
        assert result.payload == {"a": 1}

    def test_parse_structured_invalid_json(self) -> None:
        """parse_structured returns Value(ERROR) on parse failure."""
        result = F.parse_structured("{bad json}")

        assert result.kind == ValueKind.ERROR
        assert result.meta["op"] == "parse_structured"

    def test_parse_structured_schema_mismatch(self) -> None:
        """parse_structured validates basic schema types."""
        result = F.parse_structured("[]", schema=dict)

        assert result.kind == ValueKind.ERROR
        assert result.meta["op"] == "parse_structured"
