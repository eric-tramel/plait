"""Unit tests for container modules (Sequential, ModuleList, ModuleDict).

Tests cover:
- Basic instantiation and module registration
- Indexing, slicing, and iteration
- Method operations (append, extend, insert, etc.)
- Parameter collection through nested containers
- Named access patterns
"""

from collections import OrderedDict
from typing import cast

import pytest

from plait.containers import ModuleDict, ModuleList, Sequential
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
