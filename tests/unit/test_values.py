"""Unit tests for the Value container and helper functions."""

import pytest

import plait.values as values_module
from plait.values import (
    Value,
    ValueKind,
    ValueRef,
    collect_refs,
    replace_values_with_refs,
    unwrap,
    valueify,
)


def _disable_functional(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(_: str) -> None:
        raise ModuleNotFoundError

    monkeypatch.setattr(values_module, "import_module", _missing)


class TestValueKind:
    """Tests for ValueKind enum."""

    def test_valuekind_is_string_enum(self) -> None:
        """ValueKind values are strings."""
        assert ValueKind.TEXT == "text"
        assert ValueKind.FSTRING == "fstring"
        assert ValueKind.RESPONSE == "response"

    def test_valuekind_all_members(self) -> None:
        """All expected ValueKind members exist."""
        expected = {
            "TEXT",
            "FSTRING",
            "RESPONSE",
            "STRUCTURED",
            "INT",
            "FLOAT",
            "ERROR",
            "TOOL_RESULT",
            "BINARY",
            "OTHER",
        }
        actual = {member.name for member in ValueKind}
        assert actual == expected


class TestValueCreation:
    """Tests for Value instantiation."""

    def test_value_with_minimal_args(self) -> None:
        """Value can be created with just kind and payload."""
        v = Value(ValueKind.TEXT, "hello")
        assert v.kind == ValueKind.TEXT
        assert v.payload == "hello"
        assert v.ref is None
        assert v.meta == {}

    def test_value_with_ref(self) -> None:
        """Value can be created with a ref."""
        v = Value(ValueKind.TEXT, "hello", ref="node_1")
        assert v.ref == "node_1"

    def test_value_with_meta(self) -> None:
        """Value can be created with metadata."""
        v = Value(ValueKind.RESPONSE, "content", meta={"tokens": 100})
        assert v.meta == {"tokens": 100}

    def test_value_with_all_args(self) -> None:
        """Value can be created with all arguments."""
        v = Value(
            kind=ValueKind.STRUCTURED,
            payload={"key": "value"},
            ref="node_2",
            meta={"schema": "user"},
        )
        assert v.kind == ValueKind.STRUCTURED
        assert v.payload == {"key": "value"}
        assert v.ref == "node_2"
        assert v.meta == {"schema": "user"}


class TestValueGetitem:
    """Tests for Value.__getitem__ structured access."""

    def test_getitem_dict_payload(self) -> None:
        """__getitem__ accesses dict keys."""
        v = Value(ValueKind.STRUCTURED, {"name": "Ada", "age": 30})
        result = v["name"]
        assert isinstance(result, Value)
        assert result.payload == "Ada"

    def test_getitem_list_payload(self) -> None:
        """__getitem__ accesses list indices."""
        v = Value(ValueKind.STRUCTURED, ["a", "b", "c"])
        result = v[1]
        assert isinstance(result, Value)
        assert result.payload == "b"

    def test_getitem_nested_access(self) -> None:
        """__getitem__ supports chained access."""
        v = Value(ValueKind.STRUCTURED, {"user": {"name": "Ada"}})
        result = v["user"]["name"]
        assert isinstance(result, Value)
        assert result.payload == "Ada"

    def test_getitem_missing_key_returns_error(self) -> None:
        """__getitem__ returns ERROR Value for missing key."""
        v = Value(ValueKind.STRUCTURED, {"name": "Ada"})
        result = v["missing"]
        assert result.kind == ValueKind.ERROR
        assert isinstance(result.payload, KeyError)

    def test_getitem_index_out_of_range_returns_error(self) -> None:
        """__getitem__ returns ERROR Value for out of range index."""
        v = Value(ValueKind.STRUCTURED, ["a", "b"])
        result = v[10]
        assert result.kind == ValueKind.ERROR
        assert isinstance(result.payload, IndexError)

    def test_getitem_on_non_indexable_returns_error(self) -> None:
        """__getitem__ returns ERROR Value for non-indexable payload."""
        v = Value(ValueKind.INT, 42)
        result = v[0]
        assert result.kind == ValueKind.ERROR
        assert isinstance(result.payload, TypeError)

    def test_getitem_propagates_error(self) -> None:
        """__getitem__ propagates ERROR Values unchanged."""
        err = Value(ValueKind.ERROR, Exception("failed"))
        result = err["key"]
        assert result is err


class TestValueGet:
    """Tests for Value.get with default."""

    def test_get_existing_key(self) -> None:
        """get returns value for existing key."""
        v = Value(ValueKind.STRUCTURED, {"name": "Ada"})
        result = v.get("name")
        assert result.payload == "Ada"

    def test_get_missing_key_returns_default(self) -> None:
        """get returns default for missing key."""
        v = Value(ValueKind.STRUCTURED, {"name": "Ada"})
        result = v.get("missing", "default_value")
        assert result.kind == ValueKind.TEXT
        assert result.payload == "default_value"

    def test_get_missing_key_default_none(self) -> None:
        """get returns None Value when default is None."""
        v = Value(ValueKind.STRUCTURED, {"name": "Ada"})
        result = v.get("missing")
        assert result.payload is None

    def test_get_propagates_error(self) -> None:
        """get propagates ERROR Values unchanged."""
        err = Value(ValueKind.ERROR, Exception("failed"))
        result = err.get("key", "default")
        assert result is err


class TestValueContainerMethods:
    """Tests for container-like methods on Value."""

    def test_iter_over_list_payload(self) -> None:
        """__iter__ yields payload items for iterable payloads."""
        v = Value(ValueKind.STRUCTURED, ["a", "b"])
        assert list(iter(v)) == ["a", "b"]

    def test_iter_over_error_returns_empty(self) -> None:
        """__iter__ returns empty iterator for error Values."""
        v = Value(ValueKind.ERROR, ValueError("boom"))
        assert list(iter(v)) == []

    def test_iter_over_non_iterable_returns_empty(self) -> None:
        """__iter__ returns empty iterator for non-iterable payloads."""
        v = Value(ValueKind.INT, 123)
        assert list(iter(v)) == []

    def test_keys_values_items_on_dict(self) -> None:
        """keys/values/items delegate to dict payload."""
        payload = {"a": 1, "b": 2}
        v = Value(ValueKind.STRUCTURED, payload)

        assert list(v.keys()) == list(payload.keys())
        assert list(v.values()) == list(payload.values())
        assert list(v.items()) == list(payload.items())

    def test_keys_values_items_on_non_dict_raise(self) -> None:
        """keys/values/items raise for non-dict payloads."""
        v = Value(ValueKind.TEXT, "hello")

        with pytest.raises(AttributeError, match="keys\\(\\)"):
            v.keys()
        with pytest.raises(AttributeError, match="values\\(\\)"):
            v.values()
        with pytest.raises(AttributeError, match="items\\(\\)"):
            v.items()


class TestValueFallbackWithoutFunctional:
    """Tests for Value fallbacks when functional is unavailable."""

    def test_getitem_fallback_structured_and_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """__getitem__ uses direct implementation when functional is missing."""
        _disable_functional(monkeypatch)
        inner = Value(ValueKind.TEXT, "hi")
        v = Value(
            ValueKind.STRUCTURED,
            {"nested": {"a": 1}, "items": [1, 2], "value": inner},
        )

        nested = v["nested"]
        assert nested.kind == ValueKind.STRUCTURED
        assert nested.payload == {"a": 1}

        items = v["items"]
        assert items.kind == ValueKind.STRUCTURED
        assert items.payload == [1, 2]

        picked = v["value"]
        assert picked is inner

    def test_getitem_fallback_missing_key_returns_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """__getitem__ returns ERROR Value for missing key in fallback path."""
        _disable_functional(monkeypatch)
        v = Value(ValueKind.STRUCTURED, {"a": 1}, ref="input:0")

        result = v["missing"]
        assert result.kind == ValueKind.ERROR
        assert isinstance(result.payload, KeyError)
        assert result.meta["source_ref"] == "input:0"
        assert result.meta["key"] == "missing"

    def test_get_fallback_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get returns default Value when missing key in fallback path."""
        _disable_functional(monkeypatch)
        v = Value(ValueKind.STRUCTURED, {"a": 1})
        default = Value(ValueKind.TEXT, "fallback")

        result = v.get("missing", default)
        assert result is default

    def test_get_fallback_default_raw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get wraps raw default when missing key in fallback path."""
        _disable_functional(monkeypatch)
        v = Value(ValueKind.STRUCTURED, {"a": 1})

        result = v.get("missing", ["x"])
        assert result.kind == ValueKind.STRUCTURED
        assert result.payload == ["x"]


class TestValueRef:
    """Tests for ValueRef placeholder."""

    def test_valueref_creation(self) -> None:
        """ValueRef can be created with a ref string."""
        ref = ValueRef("node_1")
        assert ref.ref == "node_1"

    def test_valueref_is_frozen(self) -> None:
        """ValueRef is immutable."""
        ref = ValueRef("node_1")
        with pytest.raises(AttributeError):
            ref.ref = "node_2"  # type: ignore[misc]

    def test_valueref_equality(self) -> None:
        """ValueRef equality is based on ref value."""
        ref1 = ValueRef("node_1")
        ref2 = ValueRef("node_1")
        ref3 = ValueRef("node_2")
        assert ref1 == ref2
        assert ref1 != ref3

    def test_valueref_hashable(self) -> None:
        """ValueRef can be used as dict key or in sets."""
        ref1 = ValueRef("node_1")
        ref2 = ValueRef("node_1")
        refs = {ref1, ref2}
        assert len(refs) == 1


class TestValueify:
    """Tests for valueify helper function."""

    def test_valueify_string(self) -> None:
        """valueify wraps string as TEXT."""
        v = valueify("hello")
        assert v.kind == ValueKind.TEXT
        assert v.payload == "hello"
        assert v.ref is None

    def test_valueify_int(self) -> None:
        """valueify wraps int as INT."""
        v = valueify(42)
        assert v.kind == ValueKind.INT
        assert v.payload == 42

    def test_valueify_float(self) -> None:
        """valueify wraps float as FLOAT."""
        v = valueify(3.14)
        assert v.kind == ValueKind.FLOAT
        assert v.payload == 3.14

    def test_valueify_bytes(self) -> None:
        """valueify wraps bytes as BINARY."""
        v = valueify(b"data")
        assert v.kind == ValueKind.BINARY
        assert v.payload == b"data"

    def test_valueify_dict(self) -> None:
        """valueify wraps dict values into Value objects."""
        v = valueify({"key": "value"})
        assert isinstance(v, dict)
        assert isinstance(v["key"], Value)
        assert v["key"].kind == ValueKind.TEXT
        assert v["key"].payload == "value"

    def test_valueify_list(self) -> None:
        """valueify wraps list items into Value objects."""
        v = valueify([1, 2, 3])
        assert isinstance(v, list)
        assert [item.payload for item in v] == [1, 2, 3]
        assert all(isinstance(item, Value) for item in v)

    def test_valueify_tuple(self) -> None:
        """valueify wraps tuple items into Value objects."""
        v = valueify((1, 2, 3))
        assert isinstance(v, tuple)
        assert [item.payload for item in v] == [1, 2, 3]
        assert all(isinstance(item, Value) for item in v)

    def test_valueify_exception(self) -> None:
        """valueify wraps exception as ERROR."""
        err = ValueError("something went wrong")
        v = valueify(err)
        assert v.kind == ValueKind.ERROR
        assert v.payload is err

    def test_valueify_bool(self) -> None:
        """valueify wraps bool as OTHER (not INT)."""
        v = valueify(True)
        assert v.kind == ValueKind.OTHER
        assert v.payload is True

    def test_valueify_none(self) -> None:
        """valueify wraps None as OTHER."""
        v = valueify(None)
        assert v.kind == ValueKind.OTHER
        assert v.payload is None

    def test_valueify_custom_object(self) -> None:
        """valueify wraps custom objects as OTHER."""

        class Custom:
            pass

        obj = Custom()
        v = valueify(obj)
        assert v.kind == ValueKind.OTHER
        assert v.payload is obj

    def test_valueify_with_kind_override(self) -> None:
        """valueify uses provided kind instead of inferring."""
        v = valueify("template {x}", kind=ValueKind.FSTRING)
        assert v.kind == ValueKind.FSTRING
        assert v.payload == "template {x}"

    def test_valueify_existing_value_returns_same(self) -> None:
        """valueify returns existing Value unchanged."""
        original = Value(ValueKind.TEXT, "hello", ref="node_1")
        result = valueify(original)
        assert result is original

    def test_valueify_existing_value_with_kind_override(self) -> None:
        """valueify re-wraps Value when kind differs."""
        original = Value(ValueKind.TEXT, "template", ref="node_1", meta={"a": 1})
        result = valueify(original, kind=ValueKind.FSTRING)
        assert result is not original
        assert result.kind == ValueKind.FSTRING
        assert result.payload == "template"
        assert result.ref == "node_1"
        assert result.meta == {"a": 1}

    def test_valueify_existing_value_with_same_kind(self) -> None:
        """valueify returns same Value when kind matches."""
        original = Value(ValueKind.TEXT, "hello")
        result = valueify(original, kind=ValueKind.TEXT)
        assert result is original


class TestUnwrap:
    """Tests for unwrap helper function."""

    def test_unwrap_value_returns_payload(self) -> None:
        """unwrap extracts payload from Value."""
        v = Value(ValueKind.TEXT, "hello")
        assert unwrap(v) == "hello"

    def test_unwrap_non_value_returns_unchanged(self) -> None:
        """unwrap returns non-Value input unchanged."""
        assert unwrap("hello") == "hello"
        assert unwrap(42) == 42
        assert unwrap(None) is None

    def test_unwrap_value_with_dict_payload(self) -> None:
        """unwrap extracts dict payload."""
        v = Value(ValueKind.STRUCTURED, {"key": "value"})
        assert unwrap(v) == {"key": "value"}

    def test_unwrap_list_of_values(self) -> None:
        """unwrap recursively extracts payloads from list of Values."""
        values = [Value(ValueKind.TEXT, "a"), Value(ValueKind.INT, 2)]
        assert unwrap(values) == ["a", 2]

    def test_unwrap_dict_of_values(self) -> None:
        """unwrap recursively extracts payloads from dict of Values."""
        values = {"a": Value(ValueKind.TEXT, "x"), "b": Value(ValueKind.INT, 3)}
        assert unwrap(values) == {"a": "x", "b": 3}

    def test_unwrap_value_with_none_payload(self) -> None:
        """unwrap extracts None payload."""
        v = Value(ValueKind.OTHER, None)
        assert unwrap(v) is None


class TestCollectRefs:
    """Tests for collect_refs helper function."""

    def test_collect_refs_single_value(self) -> None:
        """collect_refs finds ref from single Value."""
        v = Value(ValueKind.TEXT, "hello", ref="node_1")
        refs = collect_refs(v)
        assert refs == ["node_1"]

    def test_collect_refs_multiple_values(self) -> None:
        """collect_refs finds refs from multiple Values."""
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")
        refs = collect_refs(v1, v2)
        assert sorted(refs) == ["node_1", "node_2"]

    def test_collect_refs_skips_none_refs(self) -> None:
        """collect_refs skips Values without refs."""
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b")  # ref is None
        refs = collect_refs(v1, v2)
        assert refs == ["node_1"]

    def test_collect_refs_nested_list(self) -> None:
        """collect_refs traverses nested lists."""
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")
        refs = collect_refs([v1, [v2]])
        assert sorted(refs) == ["node_1", "node_2"]

    def test_collect_refs_nested_dict(self) -> None:
        """collect_refs traverses nested dicts."""
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")
        refs = collect_refs({"outer": {"inner": v1}, "other": v2})
        assert sorted(refs) == ["node_1", "node_2"]

    def test_collect_refs_nested_tuple(self) -> None:
        """collect_refs traverses tuples."""
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")
        refs = collect_refs((v1, v2))
        assert sorted(refs) == ["node_1", "node_2"]

    def test_collect_refs_mixed_nested_structure(self) -> None:
        """collect_refs traverses complex nested structures."""
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")
        v3 = Value(ValueKind.TEXT, "c", ref="node_3")
        structure = {
            "list": [v1, {"nested": v2}],
            "tuple": (v3,),
        }
        refs = collect_refs(structure)
        assert sorted(refs) == ["node_1", "node_2", "node_3"]

    def test_collect_refs_kwargs(self) -> None:
        """collect_refs accepts keyword arguments."""
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")
        refs = collect_refs(arg1=v1, arg2=v2)
        assert sorted(refs) == ["node_1", "node_2"]

    def test_collect_refs_args_and_kwargs(self) -> None:
        """collect_refs combines args and kwargs."""
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")
        refs = collect_refs(v1, kwarg=v2)
        assert sorted(refs) == ["node_1", "node_2"]

    def test_collect_refs_deduplicates(self) -> None:
        """collect_refs returns unique refs."""
        v = Value(ValueKind.TEXT, "a", ref="node_1")
        refs = collect_refs(v, v, [v])
        assert refs == ["node_1"]

    def test_collect_refs_empty_input(self) -> None:
        """collect_refs returns empty list for no input."""
        refs = collect_refs()
        assert refs == []

    def test_collect_refs_non_value_input(self) -> None:
        """collect_refs ignores non-Value inputs."""
        refs = collect_refs("string", 42, {"key": "value"})
        assert refs == []


class TestReplaceValuesWithRefs:
    """Tests for replace_values_with_refs helper function."""

    def test_replace_value_with_ref(self) -> None:
        """replace_values_with_refs converts Value to ValueRef."""
        v = Value(ValueKind.TEXT, "hello", ref="node_1")
        result = replace_values_with_refs(v)
        assert isinstance(result, ValueRef)
        assert result.ref == "node_1"

    def test_replace_value_without_ref_unchanged(self) -> None:
        """replace_values_with_refs keeps Value without ref unchanged."""
        v = Value(ValueKind.TEXT, "hello")
        result = replace_values_with_refs(v)
        assert isinstance(result, Value)
        assert result is v

    def test_replace_non_value_unchanged(self) -> None:
        """replace_values_with_refs keeps non-Value unchanged."""
        assert replace_values_with_refs("string") == "string"
        assert replace_values_with_refs(42) == 42
        assert replace_values_with_refs(None) is None

    def test_replace_in_list(self) -> None:
        """replace_values_with_refs traverses lists."""
        v = Value(ValueKind.TEXT, "hello", ref="node_1")
        result = replace_values_with_refs([v, "other"])
        assert isinstance(result, list)
        assert isinstance(result[0], ValueRef)
        assert result[0].ref == "node_1"
        assert result[1] == "other"

    def test_replace_in_tuple(self) -> None:
        """replace_values_with_refs traverses tuples."""
        v = Value(ValueKind.TEXT, "hello", ref="node_1")
        result = replace_values_with_refs((v, "other"))
        assert isinstance(result, tuple)
        assert isinstance(result[0], ValueRef)
        assert result[0].ref == "node_1"
        assert result[1] == "other"

    def test_replace_in_dict(self) -> None:
        """replace_values_with_refs traverses dicts."""
        v = Value(ValueKind.TEXT, "hello", ref="node_1")
        result = replace_values_with_refs({"value": v, "other": "raw"})
        assert isinstance(result, dict)
        assert isinstance(result["value"], ValueRef)
        assert result["value"].ref == "node_1"
        assert result["other"] == "raw"

    def test_replace_nested_structure(self) -> None:
        """replace_values_with_refs traverses complex nested structures."""
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")
        v3 = Value(ValueKind.TEXT, "c")  # No ref

        structure = {
            "list": [v1, {"nested": v2}],
            "tuple": (v3, "raw"),
        }
        result = replace_values_with_refs(structure)

        assert isinstance(result["list"][0], ValueRef)
        assert result["list"][0].ref == "node_1"
        assert isinstance(result["list"][1]["nested"], ValueRef)
        assert result["list"][1]["nested"].ref == "node_2"
        assert isinstance(result["tuple"][0], Value)  # No ref, kept as Value
        assert result["tuple"][1] == "raw"

    def test_replace_creates_new_containers(self) -> None:
        """replace_values_with_refs creates new containers, not modifying originals."""
        v = Value(ValueKind.TEXT, "hello", ref="node_1")
        original_list = [v]
        original_dict = {"v": v}

        result_list = replace_values_with_refs(original_list)
        result_dict = replace_values_with_refs(original_dict)

        # Originals unchanged
        assert original_list[0] is v
        assert original_dict["v"] is v

        # Results are new containers
        assert result_list is not original_list
        assert result_dict is not original_dict
