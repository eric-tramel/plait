"""Unit tests for container modules and parameter containers.

Tests cover:
- Basic instantiation and module registration
- Indexing, slicing, and iteration
- Method operations (append, extend, insert, etc.)
- Parameter collection through nested containers
- Named access patterns
- ParameterList and ParameterDict containers
"""

from collections import OrderedDict
from typing import cast

import pytest

from plait.containers import (
    ModuleDict,
    ModuleList,
    ParameterDict,
    ParameterList,
    Sequential,
)
from plait.module import Module
from plait.parameter import Parameter


# Helper modules for testing
class DummyModule(Module):
    """Simple module for testing container behavior."""

    def __init__(self, value: str = "default") -> None:
        super().__init__()
        self.value = value

    def forward(self, x: str) -> str:
        return f"{self.value}:{x}"


class ModuleWithParam(Module):
    """Module with a parameter for testing parameter collection."""

    def __init__(self, prompt: str = "test") -> None:
        super().__init__()
        self.prompt = Parameter(prompt, description="Test prompt")

    def forward(self, x: str) -> str:
        return f"{self.prompt}:{x}"


# ============================================================================
# Sequential Tests
# ============================================================================


class TestSequentialInstantiation:
    """Tests for Sequential instantiation."""

    def test_empty_sequential(self) -> None:
        """Empty Sequential can be created."""
        seq = Sequential()
        assert len(seq) == 0
        assert list(seq) == []

    def test_positional_args(self) -> None:
        """Sequential accepts positional Module arguments."""
        m1, m2, m3 = DummyModule("a"), DummyModule("b"), DummyModule("c")
        seq = Sequential(m1, m2, m3)

        assert len(seq) == 3
        assert seq[0] is m1
        assert seq[1] is m2
        assert seq[2] is m3

    def test_ordered_dict_initialization(self) -> None:
        """Sequential accepts OrderedDict with named modules."""
        m1, m2 = DummyModule("a"), DummyModule("b")
        seq = Sequential(
            OrderedDict(
                [
                    ("first", m1),
                    ("second", m2),
                ]
            )
        )

        assert len(seq) == 2
        assert seq.first is m1
        assert seq.second is m2

    def test_non_module_raises_type_error(self) -> None:
        """Sequential raises TypeError for non-Module arguments."""
        with pytest.raises(TypeError, match="must be a Module"):
            Sequential(DummyModule(), "not a module")  # type: ignore[arg-type]

    def test_is_module_subclass(self) -> None:
        """Sequential is a Module subclass."""
        seq = Sequential()
        assert isinstance(seq, Module)


class TestSequentialAccess:
    """Tests for Sequential indexing and access patterns."""

    def test_integer_indexing(self) -> None:
        """Sequential supports integer indexing."""
        modules = [DummyModule(str(i)) for i in range(3)]
        seq = Sequential(*modules)

        assert seq[0] is modules[0]
        assert seq[1] is modules[1]
        assert seq[2] is modules[2]

    def test_negative_indexing(self) -> None:
        """Sequential supports negative indexing."""
        modules = [DummyModule(str(i)) for i in range(3)]
        seq = Sequential(*modules)

        assert seq[-1] is modules[2]
        assert seq[-2] is modules[1]
        assert seq[-3] is modules[0]

    def test_index_out_of_range(self) -> None:
        """Sequential raises IndexError for out-of-range index."""
        seq = Sequential(DummyModule())

        with pytest.raises(IndexError):
            _ = seq[5]
        with pytest.raises(IndexError):
            _ = seq[-5]

    def test_slicing_returns_sequential(self) -> None:
        """Slicing Sequential returns a new Sequential."""
        modules = [DummyModule(str(i)) for i in range(5)]
        seq = Sequential(*modules)

        sliced = seq[1:4]
        assert isinstance(sliced, Sequential)
        assert len(sliced) == 3
        assert list(sliced) == modules[1:4]

    def test_named_attribute_access(self) -> None:
        """Sequential with OrderedDict supports attribute access."""
        seq = Sequential(
            OrderedDict(
                [
                    ("encoder", DummyModule("enc")),
                    ("decoder", DummyModule("dec")),
                ]
            )
        )

        assert cast(DummyModule, seq.encoder).value == "enc"
        assert cast(DummyModule, seq.decoder).value == "dec"

    def test_attribute_error_for_unknown_name(self) -> None:
        """Sequential raises AttributeError for unknown attribute."""
        seq = Sequential(DummyModule())

        with pytest.raises(AttributeError):
            _ = seq.nonexistent


class TestSequentialIteration:
    """Tests for Sequential iteration."""

    def test_iteration(self) -> None:
        """Sequential supports iteration."""
        modules = [DummyModule(str(i)) for i in range(3)]
        seq = Sequential(*modules)

        assert list(seq) == modules

    def test_len(self) -> None:
        """len() returns number of modules."""
        seq = Sequential(DummyModule(), DummyModule())
        assert len(seq) == 2


class TestSequentialForward:
    """Tests for Sequential forward execution."""

    def test_forward_chains_output_to_input(self) -> None:
        """Sequential.forward() chains module outputs to inputs."""

        class Doubler(Module):
            def forward(self, x: int) -> int:
                return x * 2

        class AddOne(Module):
            def forward(self, x: int) -> int:
                return x + 1

        seq = Sequential(Doubler(), AddOne(), Doubler())
        result = seq(5)

        # (5 * 2 + 1) * 2 = 22
        assert result == 22

    def test_forward_empty_sequential(self) -> None:
        """Empty Sequential.forward() returns input unchanged."""
        seq = Sequential()
        assert seq("input") == "input"


class TestSequentialAppend:
    """Tests for Sequential.append()."""

    def test_append_adds_module(self) -> None:
        """append() adds a module to the end."""
        seq = Sequential(DummyModule("a"))
        m2 = DummyModule("b")

        result = seq.append(m2)

        assert len(seq) == 2
        assert seq[-1] is m2
        assert result is seq  # Method chaining

    def test_append_type_error(self) -> None:
        """append() raises TypeError for non-Module."""
        seq = Sequential()
        with pytest.raises(TypeError, match="only accepts Module"):
            seq.append("not a module")  # type: ignore[arg-type]


class TestSequentialParameterCollection:
    """Tests for parameter collection in Sequential."""

    def test_parameters_collected_from_children(self) -> None:
        """parameters() recursively collects from Sequential children."""
        seq = Sequential(
            ModuleWithParam("prompt1"),
            ModuleWithParam("prompt2"),
        )

        params = list(seq.parameters())
        assert len(params) == 2
        assert params[0].value == "prompt1"
        assert params[1].value == "prompt2"

    def test_named_parameters_with_hierarchy(self) -> None:
        """named_parameters() returns hierarchical names."""
        seq = Sequential(
            OrderedDict(
                [
                    ("first", ModuleWithParam("p1")),
                    ("second", ModuleWithParam("p2")),
                ]
            )
        )

        named = dict(seq.named_parameters())
        assert "first.prompt" in named
        assert "second.prompt" in named


# ============================================================================
# ModuleList Tests
# ============================================================================


class TestModuleListInstantiation:
    """Tests for ModuleList instantiation."""

    def test_empty_module_list(self) -> None:
        """Empty ModuleList can be created."""
        ml = ModuleList()
        assert len(ml) == 0
        assert list(ml) == []

    def test_initialization_from_list(self) -> None:
        """ModuleList can be initialized from a list."""
        modules = [DummyModule(str(i)) for i in range(3)]
        ml = ModuleList(modules)

        assert len(ml) == 3
        for i, m in enumerate(ml):
            assert m is modules[i]

    def test_non_module_raises_type_error(self) -> None:
        """ModuleList raises TypeError for non-Module items."""
        with pytest.raises(TypeError, match="only accepts Module"):
            ModuleList([DummyModule(), "not a module"])  # type: ignore[list-item]

    def test_is_module_subclass(self) -> None:
        """ModuleList is a Module subclass."""
        ml = ModuleList()
        assert isinstance(ml, Module)


class TestModuleListAccess:
    """Tests for ModuleList indexing and access."""

    def test_integer_indexing(self) -> None:
        """ModuleList supports integer indexing."""
        modules = [DummyModule(str(i)) for i in range(3)]
        ml = ModuleList(modules)

        assert ml[0] is modules[0]
        assert ml[1] is modules[1]
        assert ml[2] is modules[2]

    def test_negative_indexing(self) -> None:
        """ModuleList supports negative indexing."""
        modules = [DummyModule(str(i)) for i in range(3)]
        ml = ModuleList(modules)

        assert ml[-1] is modules[2]
        assert ml[-2] is modules[1]

    def test_setitem(self) -> None:
        """ModuleList supports item assignment."""
        ml = ModuleList([DummyModule("a"), DummyModule("b")])
        new_module = DummyModule("new")

        ml[1] = new_module
        assert ml[1] is new_module

    def test_setitem_type_error(self) -> None:
        """ModuleList raises TypeError on non-Module assignment."""
        ml = ModuleList([DummyModule()])

        with pytest.raises(TypeError, match="only accepts Module"):
            ml[0] = "not a module"  # type: ignore[assignment]

    def test_delitem(self) -> None:
        """ModuleList supports item deletion."""
        modules = [DummyModule(str(i)) for i in range(3)]
        ml = ModuleList(modules)

        del ml[1]
        assert len(ml) == 2
        assert cast(DummyModule, ml[0]).value == "0"
        assert cast(DummyModule, ml[1]).value == "2"

    def test_slicing_returns_module_list(self) -> None:
        """Slicing ModuleList returns a new ModuleList."""
        modules = [DummyModule(str(i)) for i in range(5)]
        ml = ModuleList(modules)

        sliced = ml[1:4]
        assert isinstance(sliced, ModuleList)
        assert len(sliced) == 3

    def test_contains(self) -> None:
        """ModuleList supports 'in' operator."""
        m1 = DummyModule("a")
        m2 = DummyModule("b")
        ml = ModuleList([m1])

        assert m1 in ml
        assert m2 not in ml


class TestModuleListMutations:
    """Tests for ModuleList mutation methods."""

    def test_append(self) -> None:
        """append() adds a module to the end."""
        ml = ModuleList()
        m1 = DummyModule("a")

        result = ml.append(m1)

        assert len(ml) == 1
        assert ml[0] is m1
        assert result is ml

    def test_extend(self) -> None:
        """extend() adds multiple modules."""
        ml = ModuleList([DummyModule("a")])
        new_modules = [DummyModule("b"), DummyModule("c")]

        result = ml.extend(new_modules)

        assert len(ml) == 3
        assert result is ml

    def test_insert(self) -> None:
        """insert() adds module at specified index."""
        ml = ModuleList([DummyModule("a"), DummyModule("c")])
        m_new = DummyModule("b")

        ml.insert(1, m_new)

        assert len(ml) == 3
        assert ml[1] is m_new

    def test_insert_at_beginning(self) -> None:
        """insert(0, ...) adds module at beginning."""
        ml = ModuleList([DummyModule("b")])
        m_new = DummyModule("a")

        ml.insert(0, m_new)

        assert ml[0] is m_new

    def test_insert_at_end(self) -> None:
        """insert() at end works like append."""
        ml = ModuleList([DummyModule("a")])
        m_new = DummyModule("b")

        ml.insert(10, m_new)  # Index beyond length

        assert ml[-1] is m_new

    def test_pop_default_last(self) -> None:
        """pop() removes and returns last module by default."""
        modules = [DummyModule(str(i)) for i in range(3)]
        ml = ModuleList(modules)

        popped = ml.pop()

        assert popped is modules[2]
        assert len(ml) == 2

    def test_pop_with_index(self) -> None:
        """pop(idx) removes and returns module at index."""
        modules = [DummyModule(str(i)) for i in range(3)]
        ml = ModuleList(modules)

        popped = ml.pop(1)

        assert popped is modules[1]
        assert len(ml) == 2

    def test_pop_empty_raises(self) -> None:
        """pop() on empty list raises IndexError."""
        ml = ModuleList()

        with pytest.raises(IndexError, match="empty"):
            ml.pop()


class TestModuleListForward:
    """Tests for ModuleList.forward()."""

    def test_forward_not_implemented(self) -> None:
        """ModuleList.forward() raises NotImplementedError."""
        ml = ModuleList([DummyModule()])

        with pytest.raises(NotImplementedError, match="does not implement forward"):
            ml.forward("input")


class TestModuleListParameterCollection:
    """Tests for parameter collection in ModuleList."""

    def test_parameters_collected_from_children(self) -> None:
        """parameters() recursively collects from ModuleList children."""
        ml = ModuleList(
            [
                ModuleWithParam("prompt1"),
                ModuleWithParam("prompt2"),
            ]
        )

        params = list(ml.parameters())
        assert len(params) == 2

    def test_named_parameters_with_indices(self) -> None:
        """named_parameters() uses indices as names."""
        ml = ModuleList(
            [
                ModuleWithParam("p1"),
                ModuleWithParam("p2"),
            ]
        )

        named = dict(ml.named_parameters())
        assert "0.prompt" in named
        assert "1.prompt" in named


# ============================================================================
# ModuleDict Tests
# ============================================================================


class TestModuleDictInstantiation:
    """Tests for ModuleDict instantiation."""

    def test_empty_module_dict(self) -> None:
        """Empty ModuleDict can be created."""
        md = ModuleDict()
        assert len(md) == 0

    def test_initialization_from_dict(self) -> None:
        """ModuleDict can be initialized from a dict."""
        modules = {"a": DummyModule("a"), "b": DummyModule("b")}
        md = ModuleDict(modules)

        assert len(md) == 2
        assert cast(DummyModule, md["a"]).value == "a"
        assert cast(DummyModule, md["b"]).value == "b"

    def test_initialization_from_pairs(self) -> None:
        """ModuleDict can be initialized from (key, value) pairs."""
        pairs = [("x", DummyModule("x")), ("y", DummyModule("y"))]
        md = ModuleDict(pairs)

        assert len(md) == 2
        assert cast(DummyModule, md["x"]).value == "x"

    def test_non_module_raises_type_error(self) -> None:
        """ModuleDict raises TypeError for non-Module values."""
        with pytest.raises(TypeError, match="only accepts Module"):
            ModuleDict({"key": "not a module"})  # type: ignore[dict-item]

    def test_is_module_subclass(self) -> None:
        """ModuleDict is a Module subclass."""
        md = ModuleDict()
        assert isinstance(md, Module)


class TestModuleDictAccess:
    """Tests for ModuleDict access patterns."""

    def test_getitem(self) -> None:
        """ModuleDict supports [] access."""
        m = DummyModule("test")
        md = ModuleDict({"key": m})

        assert md["key"] is m

    def test_getitem_missing_key_raises(self) -> None:
        """ModuleDict raises KeyError for missing key."""
        md = ModuleDict()

        with pytest.raises(KeyError):
            _ = md["nonexistent"]

    def test_setitem(self) -> None:
        """ModuleDict supports [] assignment."""
        md = ModuleDict()
        m = DummyModule("test")

        md["key"] = m

        assert md["key"] is m

    def test_setitem_type_error(self) -> None:
        """ModuleDict raises TypeError on non-Module assignment."""
        md = ModuleDict()

        with pytest.raises(TypeError, match="only accepts Module"):
            md["key"] = "not a module"  # type: ignore[assignment]

    def test_delitem(self) -> None:
        """ModuleDict supports del []."""
        md = ModuleDict({"key": DummyModule()})

        del md["key"]

        assert "key" not in md

    def test_delitem_missing_key_raises(self) -> None:
        """del [] raises KeyError for missing key."""
        md = ModuleDict()

        with pytest.raises(KeyError):
            del md["nonexistent"]

    def test_attribute_access(self) -> None:
        """ModuleDict supports attribute access for identifier keys."""
        md = ModuleDict({"encoder": DummyModule("enc")})

        assert cast(DummyModule, md.encoder).value == "enc"

    def test_attribute_error_for_unknown(self) -> None:
        """ModuleDict raises AttributeError for unknown attribute."""
        md = ModuleDict()

        with pytest.raises(AttributeError):
            _ = md.nonexistent

    def test_contains(self) -> None:
        """ModuleDict supports 'in' operator."""
        md = ModuleDict({"key": DummyModule()})

        assert "key" in md
        assert "other" not in md


class TestModuleDictDictMethods:
    """Tests for ModuleDict dict-like methods."""

    def test_keys(self) -> None:
        """keys() returns dict_keys view."""
        md = ModuleDict({"a": DummyModule(), "b": DummyModule()})

        keys = md.keys()
        assert set(keys) == {"a", "b"}

    def test_values(self) -> None:
        """values() returns dict_values view."""
        m1, m2 = DummyModule("1"), DummyModule("2")
        md = ModuleDict({"a": m1, "b": m2})

        values = list(md.values())
        assert m1 in values
        assert m2 in values

    def test_items(self) -> None:
        """items() returns dict_items view."""
        m = DummyModule()
        md = ModuleDict({"key": m})

        items = list(md.items())
        assert ("key", m) in items

    def test_update_from_dict(self) -> None:
        """update() adds modules from dict."""
        md = ModuleDict({"a": DummyModule("a")})
        md.update({"b": DummyModule("b"), "c": DummyModule("c")})

        assert len(md) == 3

    def test_update_from_pairs(self) -> None:
        """update() adds modules from pairs."""
        md = ModuleDict()
        md.update([("x", DummyModule("x"))])

        assert "x" in md

    def test_pop(self) -> None:
        """pop() removes and returns module."""
        m = DummyModule()
        md = ModuleDict({"key": m})

        popped = md.pop("key")

        assert popped is m
        assert "key" not in md

    def test_pop_with_default(self) -> None:
        """pop() returns default for missing key."""
        md = ModuleDict()
        default = DummyModule()

        result = md.pop("missing", default)

        assert result is default

    def test_pop_missing_no_default(self) -> None:
        """pop() returns None for missing key with no default."""
        md = ModuleDict()
        result = md.pop("missing")
        assert result is None

    def test_clear(self) -> None:
        """clear() removes all modules."""
        md = ModuleDict({"a": DummyModule(), "b": DummyModule()})

        md.clear()

        assert len(md) == 0


class TestModuleDictIteration:
    """Tests for ModuleDict iteration."""

    def test_iteration_yields_keys(self) -> None:
        """Iteration yields keys."""
        md = ModuleDict({"a": DummyModule(), "b": DummyModule()})

        keys = list(md)
        assert set(keys) == {"a", "b"}

    def test_len(self) -> None:
        """len() returns number of modules."""
        md = ModuleDict({"a": DummyModule(), "b": DummyModule()})
        assert len(md) == 2


class TestModuleDictForward:
    """Tests for ModuleDict.forward()."""

    def test_forward_not_implemented(self) -> None:
        """ModuleDict.forward() raises NotImplementedError."""
        md = ModuleDict({"key": DummyModule()})

        with pytest.raises(NotImplementedError, match="does not implement forward"):
            md.forward("input")


class TestModuleDictParameterCollection:
    """Tests for parameter collection in ModuleDict."""

    def test_parameters_collected_from_children(self) -> None:
        """parameters() recursively collects from ModuleDict children."""
        md = ModuleDict(
            {
                "first": ModuleWithParam("prompt1"),
                "second": ModuleWithParam("prompt2"),
            }
        )

        params = list(md.parameters())
        assert len(params) == 2

    def test_named_parameters_with_keys(self) -> None:
        """named_parameters() uses dict keys as names."""
        md = ModuleDict(
            {
                "encoder": ModuleWithParam("p1"),
                "decoder": ModuleWithParam("p2"),
            }
        )

        named = dict(md.named_parameters())
        assert "encoder.prompt" in named
        assert "decoder.prompt" in named


# ============================================================================
# Integration Tests - Nested Containers
# ============================================================================


class TestNestedContainers:
    """Tests for nested container modules."""

    def test_sequential_containing_sequential(self) -> None:
        """Sequential can contain other Sequential modules."""
        inner = Sequential(DummyModule("a"), DummyModule("b"))
        outer = Sequential(inner, DummyModule("c"))

        assert len(outer) == 2
        assert outer[0] is inner

        # Parameters should be collected through nesting
        inner_with_params = Sequential(ModuleWithParam("p1"), ModuleWithParam("p2"))
        outer_with_params = Sequential(inner_with_params, ModuleWithParam("p3"))

        params = list(outer_with_params.parameters())
        assert len(params) == 3

    def test_module_list_containing_module_dict(self) -> None:
        """ModuleList can contain ModuleDict instances."""
        md = ModuleDict({"a": ModuleWithParam("p1")})
        ml = ModuleList([md, ModuleWithParam("p2")])

        params = list(ml.parameters())
        assert len(params) == 2

    def test_module_dict_containing_sequential(self) -> None:
        """ModuleDict can contain Sequential instances."""
        seq = Sequential(ModuleWithParam("p1"), ModuleWithParam("p2"))
        md = ModuleDict({"pipeline": seq})

        params = list(md.parameters())
        assert len(params) == 2

        named = dict(md.named_parameters())
        assert "pipeline.0.prompt" in named
        assert "pipeline.1.prompt" in named

    def test_deeply_nested_parameter_collection(self) -> None:
        """Parameters are collected through deeply nested structures."""
        # Build: ModuleDict -> ModuleList -> Sequential -> ModuleWithParam
        inner_seq = Sequential(ModuleWithParam("deep"))
        ml = ModuleList([inner_seq])
        md = ModuleDict({"nested": ml})

        params = list(md.parameters())
        assert len(params) == 1
        assert params[0].value == "deep"

        named = dict(md.named_parameters())
        assert "nested.0.0.prompt" in named

    def test_state_dict_with_nested_containers(self) -> None:
        """state_dict() captures parameters in nested containers."""
        seq = Sequential(
            OrderedDict(
                [
                    ("first", ModuleWithParam("value1")),
                    ("second", ModuleWithParam("value2")),
                ]
            )
        )

        state = seq.state_dict()
        assert state == {
            "first.prompt": "value1",
            "second.prompt": "value2",
        }

    def test_load_state_dict_with_nested_containers(self) -> None:
        """load_state_dict() restores parameters in nested containers."""
        seq = Sequential(
            OrderedDict(
                [
                    ("first", ModuleWithParam("original1")),
                    ("second", ModuleWithParam("original2")),
                ]
            )
        )

        seq.load_state_dict(
            {
                "first.prompt": "updated1",
                "second.prompt": "updated2",
            }
        )

        assert cast(ModuleWithParam, seq.first).prompt.value == "updated1"
        assert cast(ModuleWithParam, seq.second).prompt.value == "updated2"


# ============================================================================
# ParameterList Tests
# ============================================================================


def _make_param(value: str) -> Parameter:
    """Helper to create a test Parameter with required description."""
    return Parameter(value, description=f"Test param: {value}")


class TestParameterListInstantiation:
    """Tests for ParameterList instantiation."""

    def test_empty_initialization(self) -> None:
        """ParameterList can be created empty."""
        pl = ParameterList()
        assert len(pl) == 0

    def test_initialization_with_parameters(self) -> None:
        """ParameterList initializes with given parameters."""
        params = [_make_param(f"value{i}") for i in range(3)]
        pl = ParameterList(params)
        assert len(pl) == 3
        assert pl[0].value == "value0"
        assert pl[1].value == "value1"
        assert pl[2].value == "value2"


class TestParameterListAccess:
    """Tests for ParameterList access patterns."""

    def test_getitem_int(self) -> None:
        """ParameterList supports integer indexing."""
        pl = ParameterList([_make_param("a"), _make_param("b"), _make_param("c")])
        assert pl[0].value == "a"
        assert pl[1].value == "b"
        assert pl[-1].value == "c"

    def test_getitem_slice(self) -> None:
        """ParameterList supports slicing."""
        pl = ParameterList([_make_param(f"v{i}") for i in range(5)])
        sliced = pl[1:3]
        assert len(sliced) == 2
        assert sliced[0].value == "v1"

    def test_setitem(self) -> None:
        """ParameterList supports item assignment."""
        pl = ParameterList([_make_param("old")])
        pl[0] = _make_param("new")
        assert pl[0].value == "new"

    def test_setitem_type_error(self) -> None:
        """ParameterList rejects non-Parameter values."""
        pl = ParameterList([_make_param("test")])
        with pytest.raises(TypeError, match="ParameterList only accepts Parameter"):
            pl[0] = "not a parameter"  # type: ignore[assignment]

    def test_delitem(self) -> None:
        """ParameterList supports item deletion."""
        pl = ParameterList([_make_param("a"), _make_param("b"), _make_param("c")])
        del pl[1]
        assert len(pl) == 2
        assert pl[0].value == "a"
        assert pl[1].value == "c"


class TestParameterListMutations:
    """Tests for ParameterList mutation operations."""

    def test_append(self) -> None:
        """append() adds parameter to end."""
        pl = ParameterList()
        pl.append(_make_param("first"))
        pl.append(_make_param("second"))
        assert len(pl) == 2
        assert pl[0].value == "first"
        assert pl[1].value == "second"

    def test_insert(self) -> None:
        """insert() adds parameter at specified position."""
        pl = ParameterList([_make_param("a"), _make_param("c")])
        pl.insert(1, _make_param("b"))
        assert len(pl) == 3
        assert [p.value for p in pl] == ["a", "b", "c"]

    def test_insert_type_error(self) -> None:
        """insert() rejects non-Parameter values."""
        pl = ParameterList()
        with pytest.raises(TypeError, match="ParameterList only accepts Parameter"):
            pl.insert(0, "not a parameter")  # type: ignore[arg-type]


class TestParameterListIteration:
    """Tests for ParameterList iteration methods."""

    def test_iter(self) -> None:
        """ParameterList is iterable."""
        params = [_make_param(f"v{i}") for i in range(3)]
        pl = ParameterList(params)
        values = [p.value for p in pl]
        assert values == ["v0", "v1", "v2"]

    def test_parameters(self) -> None:
        """parameters() yields all contained parameters."""
        pl = ParameterList([_make_param("a"), _make_param("b")])
        params = list(pl.parameters())
        assert len(params) == 2
        assert params[0].value == "a"
        assert params[1].value == "b"

    def test_named_parameters(self) -> None:
        """named_parameters() yields (name, param) tuples."""
        pl = ParameterList([_make_param("a"), _make_param("b")])
        named = list(pl.named_parameters())
        assert len(named) == 2
        assert named[0][0] == "0"
        assert named[0][1].value == "a"
        assert named[1][0] == "1"
        assert named[1][1].value == "b"

    def test_named_parameters_with_prefix(self) -> None:
        """named_parameters() respects prefix."""
        pl = ParameterList([_make_param("x")])
        named = list(pl.named_parameters("prompts"))
        assert named[0][0] == "prompts.0"


# ============================================================================
# ParameterDict Tests
# ============================================================================


class TestParameterDictInstantiation:
    """Tests for ParameterDict instantiation."""

    def test_empty_initialization(self) -> None:
        """ParameterDict can be created empty."""
        pd = ParameterDict()
        assert len(pd) == 0

    def test_initialization_with_dict(self) -> None:
        """ParameterDict initializes from dict."""
        pd = ParameterDict({"a": _make_param("val_a"), "b": _make_param("val_b")})
        assert len(pd) == 2
        assert pd["a"].value == "val_a"
        assert pd["b"].value == "val_b"

    def test_initialization_with_tuples(self) -> None:
        """ParameterDict initializes from iterable of tuples."""
        pd = ParameterDict([("x", _make_param("vx")), ("y", _make_param("vy"))])
        assert len(pd) == 2
        assert pd["x"].value == "vx"


class TestParameterDictAccess:
    """Tests for ParameterDict access patterns."""

    def test_getitem(self) -> None:
        """ParameterDict supports key access."""
        pd = ParameterDict({"key": _make_param("value")})
        assert pd["key"].value == "value"

    def test_setitem(self) -> None:
        """ParameterDict supports item assignment."""
        pd = ParameterDict()
        pd["new"] = _make_param("new_value")
        assert pd["new"].value == "new_value"

    def test_setitem_type_error(self) -> None:
        """ParameterDict rejects non-Parameter values."""
        pd = ParameterDict()
        with pytest.raises(TypeError, match="ParameterDict only accepts Parameter"):
            pd["key"] = "not a parameter"  # type: ignore[assignment]

    def test_delitem(self) -> None:
        """ParameterDict supports item deletion."""
        pd = ParameterDict({"a": _make_param("va"), "b": _make_param("vb")})
        del pd["a"]
        assert len(pd) == 1
        assert "a" not in pd
        assert "b" in pd


class TestParameterDictIteration:
    """Tests for ParameterDict iteration methods."""

    def test_iter(self) -> None:
        """ParameterDict iterates over keys."""
        pd = ParameterDict({"x": _make_param("vx"), "y": _make_param("vy")})
        keys = list(pd)
        assert "x" in keys
        assert "y" in keys

    def test_parameters(self) -> None:
        """parameters() yields all contained parameters."""
        pd = ParameterDict({"a": _make_param("va"), "b": _make_param("vb")})
        params = list(pd.parameters())
        assert len(params) == 2

    def test_named_parameters(self) -> None:
        """named_parameters() yields (name, param) tuples."""
        pd = ParameterDict({"foo": _make_param("bar")})
        named = list(pd.named_parameters())
        assert len(named) == 1
        assert named[0][0] == "foo"
        assert named[0][1].value == "bar"

    def test_named_parameters_with_prefix(self) -> None:
        """named_parameters() respects prefix."""
        pd = ParameterDict({"task": _make_param("prompt")})
        named = list(pd.named_parameters("tasks"))
        assert named[0][0] == "tasks.task"


# ============================================================================
# Module Integration with Parameter Containers
# ============================================================================


class TestModuleWithParameterContainers:
    """Tests for Module integration with ParameterList and ParameterDict."""

    def test_module_registers_parameter_list(self) -> None:
        """Module registers ParameterList for parameter collection."""

        class MultiPrompt(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [_make_param("p1"), _make_param("p2"), _make_param("p3")]
                )

            def forward(self, x: str) -> str:
                return x

        m = MultiPrompt()
        params = list(m.parameters())
        assert len(params) == 3

    def test_module_registers_parameter_dict(self) -> None:
        """Module registers ParameterDict for parameter collection."""

        class TaskPrompts(Module):
            def __init__(self) -> None:
                super().__init__()
                self.tasks = ParameterDict(
                    {"summarize": _make_param("sum"), "translate": _make_param("trans")}
                )

            def forward(self, x: str) -> str:
                return x

        m = TaskPrompts()
        params = list(m.parameters())
        assert len(params) == 2

    def test_named_parameters_with_parameter_list(self) -> None:
        """named_parameters() includes ParameterList contents with hierarchy."""

        class MultiPrompt(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList([_make_param("a"), _make_param("b")])

            def forward(self, x: str) -> str:
                return x

        m = MultiPrompt()
        named = dict(m.named_parameters())
        assert "prompts.0" in named
        assert "prompts.1" in named
        assert named["prompts.0"].value == "a"

    def test_named_parameters_with_parameter_dict(self) -> None:
        """named_parameters() includes ParameterDict contents with hierarchy."""

        class TaskPrompts(Module):
            def __init__(self) -> None:
                super().__init__()
                self.tasks = ParameterDict({"task1": _make_param("v1")})

            def forward(self, x: str) -> str:
                return x

        m = TaskPrompts()
        named = dict(m.named_parameters())
        assert "tasks.task1" in named
        assert named["tasks.task1"].value == "v1"

    def test_mixed_parameters_and_containers(self) -> None:
        """Module collects both direct parameters and container parameters."""

        class MixedModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.direct = _make_param("direct_value")
                self.list_params = ParameterList([_make_param("list1")])
                self.dict_params = ParameterDict({"key": _make_param("dict1")})

            def forward(self, x: str) -> str:
                return x

        m = MixedModule()
        named = dict(m.named_parameters())
        assert len(named) == 3
        assert "direct" in named
        assert "list_params.0" in named
        assert "dict_params.key" in named


# ============================================================================
# Reviewer Feedback Tests
# ============================================================================


class TestModuleDictStaleAttributeCleanup:
    """Tests for ModuleDict attribute cleanup on deletion (PR #16 review)."""

    def test_delitem_removes_attribute(self) -> None:
        """Deleting from ModuleDict removes the attribute."""
        md = ModuleDict({"encoder": DummyModule("enc")})
        # Verify attribute exists
        assert hasattr(md, "encoder")
        # Delete the module
        del md["encoder"]
        # Verify attribute is removed
        assert "encoder" not in md
        with pytest.raises(AttributeError):
            _ = md.encoder

    def test_pop_removes_attribute(self) -> None:
        """pop() from ModuleDict removes the attribute."""
        md = ModuleDict({"decoder": DummyModule("dec")})
        assert hasattr(md, "decoder")
        md.pop("decoder")
        assert "decoder" not in md
        with pytest.raises(AttributeError):
            _ = md.decoder

    def test_clear_removes_all_attributes(self) -> None:
        """clear() from ModuleDict removes all attributes."""
        md = ModuleDict(
            {"enc": DummyModule("e"), "dec": DummyModule("d"), "cls": DummyModule("c")}
        )
        assert hasattr(md, "enc")
        assert hasattr(md, "dec")
        md.clear()
        assert len(md) == 0
        with pytest.raises(AttributeError):
            _ = md.enc
        with pytest.raises(AttributeError):
            _ = md.dec


class TestSlicingNoReparenting:
    """Tests for slicing not reparenting modules (PR #16 review)."""

    def test_sequential_slice_does_not_reparent(self) -> None:
        """Slicing Sequential does not change module's parent."""
        original = Sequential(
            OrderedDict(
                [
                    ("a", DummyModule("va")),
                    ("b", DummyModule("vb")),
                    ("c", DummyModule("vc")),
                ]
            )
        )
        # Get original parent reference
        mod_b = original._modules["b"]
        original_parent = mod_b._parent

        # Slice the sequential
        sliced = original[0:2]

        # Module's parent should still be the original
        assert mod_b._parent is original_parent
        assert mod_b._parent is original

        # Sliced container should not have the modules as children
        assert len(sliced._children) == 0

    def test_module_list_slice_does_not_reparent(self) -> None:
        """Slicing ModuleList does not change module's parent."""
        original = ModuleList([DummyModule(f"v{i}") for i in range(5)])
        mod_1 = original[1]
        original_parent = mod_1._parent

        sliced = original[1:4]

        assert mod_1._parent is original_parent
        assert mod_1._parent is original
        assert len(sliced._children) == 0


class TestParameterContainerReparenting:
    """Tests for parameter reparenting when containers attach to Module (PR #16 review)."""

    def test_parameter_list_keeps_container_in_parent_chain(self) -> None:
        """Parameters in ParameterList are parented to the container, not the module.

        This preserves the hierarchical name (e.g., 'prompts.0' not just '0')
        which is important for valueify() to produce correct refs.
        """
        from plait.parameter import Parameter

        # Create parameters and put them in a list BEFORE assigning to module
        p1 = Parameter("prompt1", description="first")
        p2 = Parameter("prompt2", description="second")
        param_list = ParameterList([p1, p2])

        # When parameters are added to a container, they immediately get
        # the container as their parent (not None)
        assert p1._parent is param_list
        assert p2._parent is param_list

        # Create a module and assign the list
        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = param_list

        module = TestModule()

        # After assignment, parameters should still have the container as parent
        # (not the module directly) to preserve hierarchical naming
        assert p1._parent is param_list
        assert p2._parent is param_list
        # The container itself should now have the module as parent
        assert param_list._parent is module

    def test_parameter_dict_keeps_container_in_parent_chain(self) -> None:
        """Parameters in ParameterDict are parented to the container, not the module.

        This preserves the hierarchical name (e.g., 'tasks.summarize' not just 'summarize')
        which is important for valueify() to produce correct refs.
        """
        from plait.parameter import Parameter

        # Create parameters and put them in a dict BEFORE assigning to module
        p1 = Parameter("summarize this", description="summary prompt")
        p2 = Parameter("translate this", description="translation prompt")
        param_dict = ParameterDict({"summarize": p1, "translate": p2})

        # When parameters are added to a container, they immediately get
        # the container as their parent (not None)
        assert p1._parent is param_dict
        assert p2._parent is param_dict

        # Create a module and assign the dict
        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.tasks = param_dict

        module = TestModule()

        # After assignment, parameters should still have the container as parent
        # (not the module directly) to preserve hierarchical naming
        assert p1._parent is param_dict
        assert p2._parent is param_dict
        # The container itself should now have the module as parent
        assert param_dict._parent is module


class TestModuleDictUpdateMapping:
    """Tests for ModuleDict.update() accepting Mapping inputs (PR #16 review)."""

    def test_update_from_another_module_dict(self) -> None:
        """ModuleDict.update() should accept another ModuleDict."""
        first = ModuleDict({"a": DummyModule("v1"), "b": DummyModule("v2")})
        second = ModuleDict({"c": DummyModule("v3"), "d": DummyModule("v4")})

        first.update(second)

        assert len(first) == 4
        assert "a" in first
        assert "b" in first
        assert "c" in first
        assert "d" in first

    def test_init_from_another_module_dict(self) -> None:
        """ModuleDict() should accept another ModuleDict in constructor."""
        original = ModuleDict({"x": DummyModule("vx"), "y": DummyModule("vy")})
        copy = ModuleDict(original)

        assert len(copy) == 2
        assert "x" in copy
        assert "y" in copy
        # They should reference the same module objects
        assert copy["x"] is original["x"]
        assert copy["y"] is original["y"]


class TestParameterContainerHierarchicalNaming:
    """Tests for hierarchical naming through parameter containers (PR #16 review).

    This addresses the review comment about keeping container names in the
    parameter parent chain so that _get_hierarchical_name() and valueify()
    produce correct refs like 'param:prompts.0' instead of 'param:0'.
    """

    def test_parameter_list_hierarchical_name(self) -> None:
        """Parameter in ParameterList gets full hierarchical name including container."""
        from plait.parameter import Parameter

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("first prompt", description="first"),
                        Parameter("second prompt", description="second"),
                    ]
                )

            def forward(self, x: str) -> str:
                return x

        m = TestModule()

        # Get parameters and check their hierarchical names
        p0 = m.prompts[0]
        p1 = m.prompts[1]

        assert p0._get_hierarchical_name() == "prompts.0"
        assert p1._get_hierarchical_name() == "prompts.1"

    def test_parameter_dict_hierarchical_name(self) -> None:
        """Parameter in ParameterDict gets full hierarchical name including container."""
        from plait.parameter import Parameter

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.tasks = ParameterDict(
                    {
                        "summarize": Parameter("summarize this", description="summary"),
                        "translate": Parameter(
                            "translate this", description="translation"
                        ),
                    }
                )

            def forward(self, x: str) -> str:
                return x

        m = TestModule()

        # Get parameters and check their hierarchical names
        p_sum = m.tasks["summarize"]
        p_trans = m.tasks["translate"]

        assert p_sum._get_hierarchical_name() == "tasks.summarize"
        assert p_trans._get_hierarchical_name() == "tasks.translate"

    def test_valueify_produces_correct_refs_for_parameter_list(self) -> None:
        """valueify() includes container name in param ref for ParameterList."""
        from plait.parameter import Parameter
        from plait.values import valueify

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("first prompt", description="first"),
                        Parameter("second prompt", description="second"),
                    ]
                )

            def forward(self, x: str) -> str:
                return x

        m = TestModule()

        # Valueify the parameters and check refs
        v0 = valueify(m.prompts[0])
        v1 = valueify(m.prompts[1])

        assert v0.ref == "param:prompts.0"
        assert v1.ref == "param:prompts.1"

    def test_valueify_produces_correct_refs_for_parameter_dict(self) -> None:
        """valueify() includes container name in param ref for ParameterDict."""
        from plait.parameter import Parameter
        from plait.values import valueify

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.tasks = ParameterDict(
                    {
                        "summarize": Parameter("summarize this", description="summary"),
                        "translate": Parameter(
                            "translate this", description="translation"
                        ),
                    }
                )

            def forward(self, x: str) -> str:
                return x

        m = TestModule()

        # Valueify the parameters and check refs
        v_sum = valueify(m.tasks["summarize"])
        v_trans = valueify(m.tasks["translate"])

        assert v_sum.ref == "param:tasks.summarize"
        assert v_trans.ref == "param:tasks.translate"

    def test_multiple_containers_produce_unique_refs(self) -> None:
        """Two different containers in the same module produce distinct refs.

        This was the core issue in the review comment - without container names
        in the ref, two containers would produce ambiguous refs like 'param:0'.
        """
        from plait.parameter import Parameter
        from plait.values import valueify

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList([Parameter("prompt", description="p")])
                self.styles = ParameterList([Parameter("style", description="s")])

            def forward(self, x: str) -> str:
                return x

        m = TestModule()

        # These should be different refs
        v_prompt = valueify(m.prompts[0])
        v_style = valueify(m.styles[0])

        assert v_prompt.ref == "param:prompts.0"
        assert v_style.ref == "param:styles.0"
        # Most importantly, they should NOT be the same
        assert v_prompt.ref != v_style.ref


class TestParameterContainerIncrementStateVersion:
    """Tests for _increment_state_version on container-held parameters (PR #16 P1 review).

    This addresses the critical P1 review comment: Parameters in ParameterList/ParameterDict
    call _parent._increment_state_version() in apply_update(), but containers don't
    implement this method. This would break training when using the new containers.
    """

    def test_parameter_list_apply_update_does_not_crash(self) -> None:
        """apply_update() on parameter in ParameterList should not raise AttributeError.

        When a parameter's _parent is a container (ParameterList), calling apply_update()
        should work without raising AttributeError for missing _increment_state_version.
        """
        from plait.parameter import Parameter

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("initial value", description="test prompt"),
                    ]
                )

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        param = m.prompts[0]

        # This should NOT raise AttributeError
        param.apply_update("updated value")

        assert param.value == "updated value"

    def test_parameter_dict_apply_update_does_not_crash(self) -> None:
        """apply_update() on parameter in ParameterDict should not raise AttributeError.

        When a parameter's _parent is a container (ParameterDict), calling apply_update()
        should work without raising AttributeError for missing _increment_state_version.
        """
        from plait.parameter import Parameter

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.tasks = ParameterDict(
                    {
                        "summarize": Parameter("summarize this", description="summary"),
                    }
                )

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        param = m.tasks["summarize"]

        # This should NOT raise AttributeError
        param.apply_update("new summary prompt")

        assert param.value == "new summary prompt"

    def test_apply_update_increments_module_state_version(self) -> None:
        """apply_update() on container parameter should increment the owning module's state version.

        The state version should be incremented on the actual Module, not just the container.
        """
        from plait.parameter import Parameter

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList([Parameter("initial", description="test")])

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        initial_version = m._module_state_version

        # Apply update should increment the module's state version
        m.prompts[0].apply_update("updated")

        assert m._module_state_version > initial_version


class TestParameterListMutationsKeepContainerPrefix:
    """Tests for ParameterList mutations keeping container prefixes (PR #16 review).

    This addresses the review comment about keeping container prefixes after mutations.
    Operations like append, insert, and slice should not drop the container prefix
    from hierarchical names.
    """

    def test_append_preserves_hierarchical_name(self) -> None:
        """Appending to ParameterList preserves container prefix in hierarchical names."""
        from plait.parameter import Parameter

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList([Parameter("first", description="first")])

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        new_param = Parameter("second", description="second")
        m.prompts.append(new_param)

        # The new parameter should have the full hierarchical name
        assert new_param._get_hierarchical_name() == "prompts.1"

    def test_insert_preserves_hierarchical_name(self) -> None:
        """Inserting into ParameterList preserves container prefix in hierarchical names."""
        from plait.parameter import Parameter

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("first", description="first"),
                        Parameter("third", description="third"),
                    ]
                )

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        new_param = Parameter("second", description="second")
        m.prompts.insert(1, new_param)

        # All parameters should have proper hierarchical names
        assert m.prompts[0]._get_hierarchical_name() == "prompts.0"
        assert m.prompts[1]._get_hierarchical_name() == "prompts.1"
        assert m.prompts[2]._get_hierarchical_name() == "prompts.2"

    def test_setitem_preserves_hierarchical_name(self) -> None:
        """Setting an item in ParameterList preserves container prefix."""
        from plait.parameter import Parameter

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList([Parameter("old", description="old")])

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        new_param = Parameter("new", description="new")
        m.prompts[0] = new_param

        # The new parameter should have the full hierarchical name
        assert new_param._get_hierarchical_name() == "prompts.0"

    def test_delitem_reindexes_with_container_prefix(self) -> None:
        """Deleting from ParameterList preserves container prefix after reindexing."""
        from plait.parameter import Parameter

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("first", description="first"),
                        Parameter("second", description="second"),
                        Parameter("third", description="third"),
                    ]
                )

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        del m.prompts[1]  # Delete "second"

        # Remaining parameters should have proper hierarchical names
        assert m.prompts[0]._get_hierarchical_name() == "prompts.0"
        assert m.prompts[1]._get_hierarchical_name() == "prompts.1"
        assert m.prompts[0].value == "first"
        assert m.prompts[1].value == "third"


# ============================================================================
# Negative Indexing Tests (PR #16 review)
# ============================================================================


class TestModuleListNegativeIndexInsert:
    """Tests for negative index handling in ModuleList.insert (PR #16 review).

    The review identified that insert(-1, ...) was appending instead of
    inserting before the last element, due to using `len + idx + 1` instead
    of `len + idx` for normalization.
    """

    def test_insert_negative_one_before_last(self) -> None:
        """insert(-1, ...) inserts before the last element, not at the end.

        Python list semantics: list.insert(-1, x) inserts x before the last element.
        """
        ml = ModuleList([DummyModule("a"), DummyModule("b"), DummyModule("c")])
        new_mod = DummyModule("new")

        ml.insert(-1, new_mod)

        # After insert(-1, new), the order should be: [a, b, new, c]
        # NOT [a, b, c, new] (which would be append behavior)
        assert len(ml) == 4
        assert cast(DummyModule, ml[0]).value == "a"
        assert cast(DummyModule, ml[1]).value == "b"
        assert cast(DummyModule, ml[2]).value == "new"
        assert cast(DummyModule, ml[3]).value == "c"

    def test_insert_negative_two_before_second_to_last(self) -> None:
        """insert(-2, ...) inserts before the second-to-last element."""
        ml = ModuleList([DummyModule("a"), DummyModule("b"), DummyModule("c")])
        new_mod = DummyModule("new")

        ml.insert(-2, new_mod)

        # After insert(-2, new), the order should be: [a, new, b, c]
        assert len(ml) == 4
        assert cast(DummyModule, ml[0]).value == "a"
        assert cast(DummyModule, ml[1]).value == "new"
        assert cast(DummyModule, ml[2]).value == "b"
        assert cast(DummyModule, ml[3]).value == "c"

    def test_insert_negative_beyond_length_inserts_at_beginning(self) -> None:
        """insert(-N, ...) where N > len inserts at the beginning."""
        ml = ModuleList([DummyModule("a"), DummyModule("b")])
        new_mod = DummyModule("new")

        ml.insert(-10, new_mod)

        # Should insert at beginning when index is too negative
        assert len(ml) == 3
        assert cast(DummyModule, ml[0]).value == "new"
        assert cast(DummyModule, ml[1]).value == "a"
        assert cast(DummyModule, ml[2]).value == "b"

    def test_insert_negative_matches_python_list_behavior(self) -> None:
        """ModuleList.insert negative index behavior matches Python list.

        This is a comprehensive test that verifies parity with Python list semantics.
        """
        # Test with Python list for reference
        py_list = ["a", "b", "c"]
        py_list.insert(-1, "new")
        expected = py_list.copy()

        # Now test ModuleList
        ml = ModuleList([DummyModule("a"), DummyModule("b"), DummyModule("c")])
        ml.insert(-1, DummyModule("new"))

        actual = [cast(DummyModule, m).value for m in ml]
        assert actual == expected


class TestParameterListNegativeIndexSetItem:
    """Tests for negative index handling in ParameterList.__setitem__ (PR #16 review).

    The review identified that negative indices were passed directly to
    _set_param_name without normalization, creating names like '-1' instead
    of the correct positive index.
    """

    def test_setitem_negative_one_normalizes_name(self) -> None:
        """Setting pl[-1] = param should give param a positive index name.

        The parameter name should be the normalized positive index (e.g., '2'),
        not the negative index (e.g., '-1').
        """
        from plait.parameter import Parameter

        pl = ParameterList([_make_param("a"), _make_param("b"), _make_param("c")])
        new_param = Parameter("new_value", description="replacement")

        pl[-1] = new_param

        # The new parameter should have name '2', not '-1'
        assert new_param._name == "2"
        # And it should be at the last position
        assert pl[2] is new_param
        assert pl[-1] is new_param

    def test_setitem_negative_two_normalizes_name(self) -> None:
        """Setting pl[-2] = param should give param the correct positive index name."""
        from plait.parameter import Parameter

        pl = ParameterList([_make_param("a"), _make_param("b"), _make_param("c")])
        new_param = Parameter("new_value", description="replacement")

        pl[-2] = new_param

        # The new parameter should have name '1', not '-2'
        assert new_param._name == "1"
        assert pl[1] is new_param
        assert pl[-2] is new_param

    def test_setitem_negative_index_in_module_preserves_hierarchy(self) -> None:
        """Setting pl[-1] on a module-attached list preserves full hierarchical name."""
        from plait.parameter import Parameter

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [_make_param("a"), _make_param("b"), _make_param("c")]
                )

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        new_param = Parameter("new", description="replacement")

        m.prompts[-1] = new_param

        # Should have proper hierarchical name with positive index
        assert new_param._get_hierarchical_name() == "prompts.2"
        # NOT 'prompts.-1'

    def test_setitem_negative_index_valueify_produces_correct_ref(self) -> None:
        """valueify() on parameter set via negative index produces correct ref."""
        from plait.parameter import Parameter
        from plait.values import valueify

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList([_make_param("a"), _make_param("b")])

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        new_param = Parameter("new", description="replacement")
        m.prompts[-1] = new_param

        # valueify should produce 'param:prompts.1', not 'param:prompts.-1'
        v = valueify(new_param)
        assert v.ref == "param:prompts.1"


# ============================================================================
# Edge Case Tests - Empty Containers
# ============================================================================


class TestEmptyContainerEdgeCases:
    """Tests for edge cases with empty containers."""

    def test_empty_sequential_forward_returns_input(self) -> None:
        """Empty Sequential returns input unchanged."""
        seq = Sequential()
        assert seq("test") == "test"
        assert seq(123) == 123
        assert seq(None) is None

    def test_empty_sequential_slicing(self) -> None:
        """Slicing empty Sequential works correctly."""
        seq = Sequential()
        sliced = seq[:]
        assert isinstance(sliced, Sequential)
        assert len(sliced) == 0

    def test_empty_module_list_iteration(self) -> None:
        """Empty ModuleList iteration yields nothing."""
        ml = ModuleList()
        assert list(ml) == []
        assert len(ml) == 0

    def test_empty_module_list_slicing(self) -> None:
        """Slicing empty ModuleList works correctly."""
        ml = ModuleList()
        sliced = ml[:]
        assert isinstance(sliced, ModuleList)
        assert len(sliced) == 0

    def test_empty_module_list_contains(self) -> None:
        """Empty ModuleList contains check returns False."""
        ml = ModuleList()
        assert DummyModule() not in ml

    def test_empty_module_dict_iteration(self) -> None:
        """Empty ModuleDict iteration yields nothing."""
        md = ModuleDict()
        assert list(md) == []
        assert list(md.keys()) == []
        assert list(md.values()) == []
        assert list(md.items()) == []

    def test_empty_module_dict_contains(self) -> None:
        """Empty ModuleDict contains check returns False."""
        md = ModuleDict()
        assert "key" not in md

    def test_empty_module_dict_clear(self) -> None:
        """Clearing an already empty ModuleDict is safe."""
        md = ModuleDict()
        md.clear()  # Should not raise
        assert len(md) == 0

    def test_empty_parameter_list_iteration(self) -> None:
        """Empty ParameterList iteration yields nothing."""
        pl = ParameterList()
        assert list(pl) == []
        assert list(pl.parameters()) == []
        assert list(pl.named_parameters()) == []

    def test_empty_parameter_dict_iteration(self) -> None:
        """Empty ParameterDict iteration yields nothing."""
        pd = ParameterDict()
        assert list(pd) == []
        assert list(pd.parameters()) == []
        assert list(pd.named_parameters()) == []

    def test_empty_containers_in_module_yield_no_parameters(self) -> None:
        """Module with empty containers yields no parameters."""

        class EmptyContainers(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList()
                self.tasks = ParameterDict()
                self.layers = ModuleList()
                self.modules_dict = ModuleDict()

            def forward(self, x: str) -> str:
                return x

        m = EmptyContainers()
        assert list(m.parameters()) == []
        assert list(m.named_parameters()) == []
        assert m.state_dict() == {}


# ============================================================================
# Edge Case Tests - Deeply Nested Containers
# ============================================================================


class TestDeeplyNestedContainers:
    """Tests for deeply nested container structures."""

    def test_triple_nested_sequential(self) -> None:
        """Sequential containing Sequential containing Sequential works."""
        inner = Sequential(ModuleWithParam("deep"))
        middle = Sequential(inner)
        outer = Sequential(middle)

        params = list(outer.parameters())
        assert len(params) == 1
        assert params[0].value == "deep"

        named = dict(outer.named_parameters())
        # Path: 0.0.0.prompt
        assert "0.0.0.prompt" in named

    def test_four_level_mixed_nesting(self) -> None:
        """Four levels of mixed container nesting."""
        # Level 4: ModuleWithParam
        mod = ModuleWithParam("level4")
        # Level 3: ModuleList containing the module
        ml = ModuleList([mod])
        # Level 2: ModuleDict containing the list
        md = ModuleDict({"nested": ml})
        # Level 1: Sequential containing the dict
        seq = Sequential(md)

        params = list(seq.parameters())
        assert len(params) == 1
        assert params[0].value == "level4"

        named = dict(seq.named_parameters())
        assert "0.nested.0.prompt" in named

    def test_parameter_container_in_nested_module(self) -> None:
        """ParameterList inside nested module containers."""

        class Inner(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [_make_param("inner0"), _make_param("inner1")]
                )

            def forward(self, x: str) -> str:
                return x

        class Middle(Module):
            def __init__(self) -> None:
                super().__init__()
                self.child = Inner()

            def forward(self, x: str) -> str:
                return x

        class Outer(Module):
            def __init__(self) -> None:
                super().__init__()
                self.middle = Middle()

            def forward(self, x: str) -> str:
                return x

        m = Outer()
        named = dict(m.named_parameters())
        assert "middle.child.prompts.0" in named
        assert "middle.child.prompts.1" in named
        assert len(named) == 2

    def test_deeply_nested_state_dict_round_trip(self) -> None:
        """state_dict and load_state_dict work with deep nesting."""

        class DeepNested(Module):
            def __init__(self) -> None:
                super().__init__()
                inner = ModuleWithParam("deep_value")
                ml = ModuleList([inner])
                self.container = ModuleDict({"layers": ml})

            def forward(self, x: str) -> str:
                return x

        m = DeepNested()
        state = m.state_dict()
        assert "container.layers.0.prompt" in state
        assert state["container.layers.0.prompt"] == "deep_value"

        # Load new state
        m.load_state_dict({"container.layers.0.prompt": "updated_deep"})
        new_state = m.state_dict()
        assert new_state["container.layers.0.prompt"] == "updated_deep"


# ============================================================================
# Edge Case Tests - Type Validation Edge Cases
# ============================================================================


class TestTypeValidationEdgeCases:
    """Tests for type validation edge cases."""

    def test_sequential_rejects_none(self) -> None:
        """Sequential raises TypeError for None."""
        with pytest.raises(TypeError, match="must be a Module"):
            Sequential(DummyModule(), None)  # type: ignore[arg-type]

    def test_module_list_extend_rejects_mixed_types(self) -> None:
        """ModuleList.extend rejects list with non-Module items."""
        ml = ModuleList([DummyModule()])
        with pytest.raises(TypeError, match="only accepts Module"):
            ml.extend([DummyModule(), "not a module"])  # type: ignore[list-item]

    def test_module_dict_update_rejects_non_modules(self) -> None:
        """ModuleDict.update rejects non-Module values."""
        md = ModuleDict()
        with pytest.raises(TypeError, match="only accepts Module"):
            md.update({"key": "not a module"})  # type: ignore[dict-item]

    def test_parameter_list_slice_assignment_validates_types(self) -> None:
        """ParameterList slice assignment validates all items."""
        pl = ParameterList([_make_param("a"), _make_param("b"), _make_param("c")])
        with pytest.raises(TypeError, match="ParameterList only accepts"):
            pl[0:2] = [_make_param("x"), "not a param"]  # type: ignore[list-item]

    def test_sequential_append_validates_module_type(self) -> None:
        """Sequential.append validates module type."""
        seq = Sequential()
        with pytest.raises(TypeError, match="only accepts Module"):
            seq.append("string")  # type: ignore[arg-type]

    def test_module_list_insert_validates_module_type(self) -> None:
        """ModuleList.insert validates module type."""
        ml = ModuleList()
        with pytest.raises(TypeError, match="only accepts Module"):
            ml.insert(0, "string")  # type: ignore[arg-type]


# ============================================================================
# Edge Case Tests - Slicing Edge Cases
# ============================================================================


class TestSlicingEdgeCases:
    """Tests for slicing edge cases in containers."""

    def test_sequential_step_slicing(self) -> None:
        """Sequential supports step in slicing."""
        modules = [DummyModule(str(i)) for i in range(5)]
        seq = Sequential(*modules)

        # Every other module
        sliced = seq[::2]
        assert len(sliced) == 3
        assert cast(DummyModule, sliced[0]).value == "0"
        assert cast(DummyModule, sliced[1]).value == "2"
        assert cast(DummyModule, sliced[2]).value == "4"

    def test_sequential_reverse_slicing(self) -> None:
        """Sequential supports reverse slicing."""
        modules = [DummyModule(str(i)) for i in range(3)]
        seq = Sequential(*modules)

        sliced = seq[::-1]
        assert len(sliced) == 3
        assert cast(DummyModule, sliced[0]).value == "2"
        assert cast(DummyModule, sliced[1]).value == "1"
        assert cast(DummyModule, sliced[2]).value == "0"

    def test_module_list_step_slicing(self) -> None:
        """ModuleList supports step in slicing."""
        modules = [DummyModule(str(i)) for i in range(6)]
        ml = ModuleList(modules)

        sliced = ml[1:5:2]  # indices 1 and 3
        assert len(sliced) == 2
        assert cast(DummyModule, sliced[0]).value == "1"
        assert cast(DummyModule, sliced[1]).value == "3"

    def test_module_list_negative_step_slicing(self) -> None:
        """ModuleList supports negative step slicing."""
        modules = [DummyModule(str(i)) for i in range(4)]
        ml = ModuleList(modules)

        sliced = ml[::-1]
        assert len(sliced) == 4
        values = [cast(DummyModule, m).value for m in sliced]
        assert values == ["3", "2", "1", "0"]

    def test_parameter_list_slice_assignment_shrinks(self) -> None:
        """ParameterList slice assignment can shrink the list."""
        pl = ParameterList([_make_param(f"p{i}") for i in range(5)])
        pl[1:4] = [_make_param("new")]

        assert len(pl) == 3
        assert pl[0].value == "p0"
        assert pl[1].value == "new"
        assert pl[2].value == "p4"

    def test_parameter_list_slice_assignment_grows(self) -> None:
        """ParameterList slice assignment can grow the list."""
        pl = ParameterList([_make_param("a"), _make_param("b")])
        pl[1:1] = [_make_param("x"), _make_param("y"), _make_param("z")]

        assert len(pl) == 5
        assert [p.value for p in pl] == ["a", "x", "y", "z", "b"]

    def test_parameter_list_slice_delete(self) -> None:
        """ParameterList supports slice deletion."""
        pl = ParameterList([_make_param(f"p{i}") for i in range(5)])
        del pl[1:4]

        assert len(pl) == 2
        assert pl[0].value == "p0"
        assert pl[1].value == "p4"

    def test_sliced_sequential_is_independent(self) -> None:
        """Modifications to sliced Sequential don't affect original."""
        modules = [DummyModule(str(i)) for i in range(5)]
        original = Sequential(*modules)

        sliced = original[1:4]
        sliced.append(DummyModule("new"))

        assert len(original) == 5  # Original unchanged
        assert len(sliced) == 4  # Sliced has the new module

    def test_sliced_module_list_is_independent(self) -> None:
        """Modifications to sliced ModuleList don't affect original."""
        modules = [DummyModule(str(i)) for i in range(5)]
        original = ModuleList(modules)

        sliced = original[1:4]
        sliced.append(DummyModule("new"))

        assert len(original) == 5  # Original unchanged
        assert len(sliced) == 4  # Sliced has the new module


# ============================================================================
# Edge Case Tests - ParameterList Insert with Negative Indices
# ============================================================================


class TestParameterListInsertNegativeIndex:
    """Tests for ParameterList.insert with negative indices (PR #16 review).

    The review identified that insert could fail with very negative indices
    due to incorrect reindexing logic.
    """

    def test_insert_negative_one_before_last(self) -> None:
        """insert(-1, ...) inserts before the last element."""
        pl = ParameterList([_make_param("a"), _make_param("b"), _make_param("c")])
        new_param = _make_param("new")

        pl.insert(-1, new_param)

        # Should match Python list behavior: [a, b, new, c]
        assert len(pl) == 4
        assert [p.value for p in pl] == ["a", "b", "new", "c"]

    def test_insert_negative_two_before_second_last(self) -> None:
        """insert(-2, ...) inserts before the second-to-last element."""
        pl = ParameterList([_make_param("a"), _make_param("b"), _make_param("c")])
        new_param = _make_param("new")

        pl.insert(-2, new_param)

        # Should match Python list behavior: [a, new, b, c]
        assert len(pl) == 4
        assert [p.value for p in pl] == ["a", "new", "b", "c"]

    def test_insert_very_negative_index(self) -> None:
        """insert() with very negative index inserts at beginning."""
        pl = ParameterList([_make_param("a"), _make_param("b")])
        new_param = _make_param("new")

        pl.insert(-100, new_param)

        # Should insert at beginning like Python list
        assert len(pl) == 3
        assert [p.value for p in pl] == ["new", "a", "b"]

    def test_insert_matches_python_list_behavior(self) -> None:
        """ParameterList.insert matches Python list semantics for negative indices."""
        # Test against Python list behavior
        for negative_idx in [-1, -2, -3, -5, -10]:
            py_list = ["a", "b", "c"]
            py_list.insert(negative_idx, "new")

            pl = ParameterList([_make_param("a"), _make_param("b"), _make_param("c")])
            pl.insert(negative_idx, _make_param("new"))

            actual = [p.value for p in pl]
            assert actual == py_list, f"Mismatch for insert({negative_idx})"

    def test_insert_preserves_hierarchical_names_after_negative_insert(self) -> None:
        """Hierarchical names are correct after negative index insert."""

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList([_make_param("a"), _make_param("b")])

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        m.prompts.insert(-1, _make_param("new"))

        # All parameters should have correct hierarchical names
        for i, p in enumerate(m.prompts):
            assert p._get_hierarchical_name() == f"prompts.{i}"


# ============================================================================
# Edge Case Tests - Non-Identifier Keys in ModuleDict
# ============================================================================


class TestModuleDictNonIdentifierKeys:
    """Tests for ModuleDict with non-identifier keys."""

    def test_numeric_string_key(self) -> None:
        """ModuleDict handles numeric string keys."""
        md = ModuleDict({"123": DummyModule("num")})
        assert "123" in md
        assert cast(DummyModule, md["123"]).value == "num"

    def test_key_with_spaces(self) -> None:
        """ModuleDict handles keys with spaces."""
        md = ModuleDict({"key with spaces": DummyModule("spacy")})
        assert "key with spaces" in md
        assert cast(DummyModule, md["key with spaces"]).value == "spacy"

    def test_key_with_special_chars(self) -> None:
        """ModuleDict handles keys with special characters."""
        md = ModuleDict({"key-with-dashes": DummyModule("dashes")})
        assert "key-with-dashes" in md
        assert cast(DummyModule, md["key-with-dashes"]).value == "dashes"

    def test_non_identifier_keys_no_attribute_access(self) -> None:
        """Non-identifier keys are not accessible as attributes."""
        md = ModuleDict({"123": DummyModule("num")})

        # Should be accessible via []
        assert md["123"] is not None

        # Should not be accessible as attribute (would cause syntax error anyway)
        with pytest.raises(AttributeError):
            _ = md.nonexistent

    def test_non_identifier_key_still_registered_as_child(self) -> None:
        """Non-identifier keys are still registered as children."""
        md = ModuleDict({"123": ModuleWithParam("num_param")})

        # Parameters should be collected
        params = list(md.parameters())
        assert len(params) == 1
        assert params[0].value == "num_param"

        # Named parameters should use the key
        named = dict(md.named_parameters())
        assert "123.prompt" in named

    def test_delete_non_identifier_key(self) -> None:
        """Deleting non-identifier key cleans up properly."""
        md = ModuleDict({"123": DummyModule("num"), "abc": DummyModule("alpha")})

        del md["123"]

        assert "123" not in md
        assert "abc" in md
        assert len(md) == 1


# ============================================================================
# Edge Case Tests - Module Replacement in Containers
# ============================================================================


class TestModuleReplacementInContainers:
    """Tests for replacing modules in containers."""

    def test_sequential_replace_via_setattr(self) -> None:
        """Sequential module replacement via setattr."""
        seq = Sequential(
            OrderedDict(
                [
                    ("first", DummyModule("a")),
                    ("second", DummyModule("b")),
                ]
            )
        )

        new_module = DummyModule("new")
        seq.first = new_module

        # The new module should be set
        assert cast(DummyModule, seq.first).value == "new"

    def test_module_list_setitem_reparents(self) -> None:
        """ModuleList setitem properly reparents the new module."""
        ml = ModuleList([DummyModule("old")])
        new_module = DummyModule("new")

        ml[0] = new_module

        assert new_module._parent is ml
        assert new_module._name == "0"

    def test_module_dict_setitem_reparents(self) -> None:
        """ModuleDict setitem properly reparents the new module."""
        md = ModuleDict({"key": DummyModule("old")})
        new_module = DummyModule("new")

        md["key"] = new_module

        assert new_module._parent is md
        assert new_module._name == "key"

    def test_parameter_list_setitem_reparents(self) -> None:
        """ParameterList setitem properly reparents the parameter."""
        pl = ParameterList([_make_param("old")])
        new_param = _make_param("new")

        pl[0] = new_param

        assert new_param._parent is pl
        assert new_param._name == "0"


# ============================================================================
# Edge Case Tests - Index Bounds
# ============================================================================


class TestIndexBoundsEdgeCases:
    """Tests for index bounds edge cases."""

    def test_sequential_single_element_negative_index(self) -> None:
        """Sequential with single element handles negative index."""
        seq = Sequential(DummyModule("only"))
        assert cast(DummyModule, seq[-1]).value == "only"
        assert cast(DummyModule, seq[0]).value == "only"

    def test_module_list_single_element_negative_index(self) -> None:
        """ModuleList with single element handles negative index."""
        ml = ModuleList([DummyModule("only")])
        assert cast(DummyModule, ml[-1]).value == "only"
        assert cast(DummyModule, ml[0]).value == "only"

    def test_module_list_setitem_negative_index_bounds(self) -> None:
        """ModuleList setitem with out-of-bounds negative index raises."""
        ml = ModuleList([DummyModule("a")])

        with pytest.raises(IndexError):
            ml[-5] = DummyModule("fail")

    def test_module_list_getitem_negative_index_bounds(self) -> None:
        """ModuleList getitem with out-of-bounds negative index raises."""
        ml = ModuleList([DummyModule("a"), DummyModule("b")])

        with pytest.raises(IndexError):
            _ = ml[-10]

    def test_module_list_delitem_negative_index_bounds(self) -> None:
        """ModuleList delitem with out-of-bounds negative index raises."""
        ml = ModuleList([DummyModule("a")])

        with pytest.raises(IndexError):
            del ml[-5]

    def test_parameter_list_index_out_of_bounds(self) -> None:
        """ParameterList raises IndexError for out-of-bounds index."""
        pl = ParameterList([_make_param("a")])

        with pytest.raises(IndexError):
            _ = pl[10]

        with pytest.raises(IndexError):
            _ = pl[-10]


# ============================================================================
# Edge Case Tests - Repr and String Representations
# ============================================================================


class TestContainerRepresentations:
    """Tests for container string representations."""

    def test_parameter_list_repr(self) -> None:
        """ParameterList has useful repr."""
        pl = ParameterList([_make_param("a"), _make_param("b")])
        repr_str = repr(pl)

        assert "ParameterList" in repr_str
        assert "a" in repr_str or "Parameter" in repr_str

    def test_parameter_dict_repr(self) -> None:
        """ParameterDict has useful repr."""
        pd = ParameterDict({"key": _make_param("value")})
        repr_str = repr(pd)

        assert "ParameterDict" in repr_str
        assert "key" in repr_str or "Parameter" in repr_str

    def test_empty_parameter_list_repr(self) -> None:
        """Empty ParameterList repr works."""
        pl = ParameterList()
        repr_str = repr(pl)
        assert "ParameterList" in repr_str
        assert "[]" in repr_str

    def test_empty_parameter_dict_repr(self) -> None:
        """Empty ParameterDict repr works."""
        pd = ParameterDict()
        repr_str = repr(pd)
        assert "ParameterDict" in repr_str
        assert "{}" in repr_str


# ============================================================================
# Edge Case Tests - Container Module Equality and Identity
# ============================================================================


class TestContainerModuleIdentity:
    """Tests for module identity within containers."""

    def test_same_module_in_multiple_positions(self) -> None:
        """Same module instance can be referenced multiple times."""
        shared = DummyModule("shared")
        ml = ModuleList([shared, DummyModule("other")])

        # The shared module is at index 0
        assert ml[0] is shared
        # But if we access the module, it's the same instance
        assert shared in ml

    def test_module_identity_preserved_in_slice(self) -> None:
        """Module identity is preserved in sliced containers."""
        modules = [DummyModule(str(i)) for i in range(5)]
        seq = Sequential(*modules)

        sliced = seq[1:4]

        # The sliced modules should be the same instances
        assert sliced[0] is modules[1]
        assert sliced[1] is modules[2]
        assert sliced[2] is modules[3]


# ============================================================================
# Edge Case Tests - Parameter Update Propagation
# ============================================================================


class TestParameterUpdatePropagation:
    """Tests for parameter update propagation through containers."""

    def test_nested_container_state_version_increments_owning_module(self) -> None:
        """State version updates increment the parameter's owning module's version.

        When a parameter inside a nested module (within a container) is updated,
        the state version of the module that directly owns that parameter is
        incremented. The parent container's state version is not affected.
        """

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                inner = ModuleWithParam("nested")
                self.layers = ModuleList([inner])

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        inner = cast(ModuleWithParam, m.layers[0])
        initial_inner_version = inner._module_state_version

        # Get the nested parameter and update it
        param = inner.prompt
        param.apply_update("updated")

        # The inner module's state version should have increased
        assert inner._module_state_version > initial_inner_version
        # Note: The root module's state version is NOT affected because
        # each Module tracks its own parameter state independently

    def test_parameter_container_update_propagates_to_module(self) -> None:
        """Parameter in container updates the owning module's state version."""
        from plait.parameter import Parameter

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList([Parameter("test", description="test")])

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        initial = m._module_state_version

        m.prompts[0].apply_update("new value")

        assert m._module_state_version > initial

    def test_parameter_dict_update_propagates_to_module(self) -> None:
        """Parameter in ParameterDict updates the owning module's state version."""
        from plait.parameter import Parameter

        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.tasks = ParameterDict({"a": Parameter("val", description="d")})

            def forward(self, x: str) -> str:
                return x

        m = TestModule()
        initial = m._module_state_version

        m.tasks["a"].apply_update("new")

        assert m._module_state_version > initial


# ============================================================================
# Edge Case Tests - Clearing and Re-populating Containers
# ============================================================================


class TestClearAndRepopulate:
    """Tests for clearing and re-populating containers."""

    def test_module_dict_clear_and_repopulate(self) -> None:
        """ModuleDict can be cleared and repopulated."""
        md = ModuleDict({"a": DummyModule("1"), "b": DummyModule("2")})

        md.clear()
        assert len(md) == 0

        md["c"] = DummyModule("3")
        md["d"] = DummyModule("4")

        assert len(md) == 2
        assert "c" in md
        assert "d" in md
        assert "a" not in md

    def test_module_list_pop_all_and_repopulate(self) -> None:
        """ModuleList can be emptied via pop and repopulated."""
        ml = ModuleList([DummyModule("a"), DummyModule("b")])

        # Pop all
        while len(ml) > 0:
            ml.pop()

        assert len(ml) == 0

        # Repopulate
        ml.append(DummyModule("new"))
        assert len(ml) == 1

    def test_parameter_list_clear_via_slice_and_repopulate(self) -> None:
        """ParameterList can be cleared via slice deletion and repopulated."""
        pl = ParameterList([_make_param("a"), _make_param("b"), _make_param("c")])

        del pl[:]
        assert len(pl) == 0

        pl.append(_make_param("new"))
        assert len(pl) == 1
        assert pl[0].value == "new"


# ============================================================================
# Edge Case Tests - Concurrent-like Scenarios (Sequential Operations)
# ============================================================================


class TestSequentialModificationPatterns:
    """Tests for patterns that might occur in complex modification scenarios."""

    def test_module_list_reindex_after_multiple_deletes(self) -> None:
        """ModuleList maintains correct indices after multiple deletes."""
        ml = ModuleList([DummyModule(str(i)) for i in range(10)])

        # Delete from various positions
        del ml[0]  # Delete first
        del ml[-1]  # Delete last (now at index 7)
        del ml[3]  # Delete from middle

        assert len(ml) == 7
        # Verify indices are sequential
        for i in range(len(ml)):
            assert str(i) in ml._modules

    def test_parameter_list_reindex_consistency(self) -> None:
        """ParameterList maintains consistent indices after mutations."""
        pl = ParameterList([_make_param(f"p{i}") for i in range(5)])

        # Insert, delete, and replace
        pl.insert(2, _make_param("inserted"))
        del pl[0]
        pl[-1] = _make_param("replaced")

        # Verify all parameters have correct sequential indices
        for i, p in enumerate(pl):
            assert p._name == str(i)


# ============================================================================
# Edge Case Tests - ModuleDict Type Index Access
# ============================================================================


class TestModuleDictTypeEdgeCases:
    """Tests for ModuleDict access with edge case key types."""

    def test_module_dict_integer_type_index_fails(self) -> None:
        """ModuleDict with integer key requires string access."""
        # Note: Keys must be strings, but users might try integer access
        md = ModuleDict({"0": DummyModule("zero")})

        # Access with string works
        assert md["0"] is not None

        # Access with int would raise KeyError (not TypeError)
        # because int is not in the dict
        with pytest.raises(KeyError):
            _ = md[0]  # type: ignore[index]


# ============================================================================
# Edge Case Tests - Large Container Operations
# ============================================================================


class TestLargeContainerOperations:
    """Tests for operations on larger containers."""

    def test_large_sequential_forward(self) -> None:
        """Sequential with many modules executes correctly."""

        class Counter(Module):
            def forward(self, x: int) -> int:
                return x + 1

        seq = Sequential(*[Counter() for _ in range(100)])
        result = seq(0)

        assert result == 100

    def test_large_module_list_iteration(self) -> None:
        """ModuleList with many modules iterates correctly."""
        count = 1000
        ml = ModuleList([DummyModule(str(i)) for i in range(count)])

        values = [cast(DummyModule, m).value for m in ml]
        assert len(values) == count
        assert values == [str(i) for i in range(count)]

    def test_large_parameter_list_named_parameters(self) -> None:
        """ParameterList with many parameters names correctly."""

        class LargeModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList([_make_param(f"p{i}") for i in range(100)])

            def forward(self, x: str) -> str:
                return x

        m = LargeModule()
        named = dict(m.named_parameters())

        assert len(named) == 100
        for i in range(100):
            assert f"prompts.{i}" in named
            assert named[f"prompts.{i}"].value == f"p{i}"
