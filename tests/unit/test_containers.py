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

        # Initially, parameters have no parent (None)
        assert p1._parent is None
        assert p2._parent is None

        # Create a module and assign the list
        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = param_list

        module = TestModule()

        # After assignment, parameters should have the container as parent
        # (not the module directly) to preserve hierarchical naming
        assert p1._parent is param_list
        assert p2._parent is param_list
        # The container itself should have the module as parent
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

        # Initially, parameters have no parent (None)
        assert p1._parent is None
        assert p2._parent is None

        # Create a module and assign the dict
        class TestModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.tasks = param_dict

        module = TestModule()

        # After assignment, parameters should have the container as parent
        # (not the module directly) to preserve hierarchical naming
        assert p1._parent is param_dict
        assert p2._parent is param_dict
        # The container itself should have the module as parent
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
