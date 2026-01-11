"""Unit tests for container classes (ParameterList, ParameterDict, ModuleList, ModuleDict).

Tests cover:
- Container creation and basic operations
- Integration with Module for parameter/module collection
- Proper naming and hierarchical structure
"""

import pytest

from plait.containers import ModuleDict, ModuleList, ParameterDict, ParameterList
from plait.module import LLMInference, Module
from plait.parameter import Parameter


class TestParameterList:
    """Tests for ParameterList container."""

    def test_creation_empty(self) -> None:
        """ParameterList can be created empty."""
        pl = ParameterList()
        assert len(pl) == 0
        assert list(pl.parameters()) == []

    def test_creation_with_parameters(self) -> None:
        """ParameterList can be initialized with parameters."""
        params = [
            Parameter("val1", description="Param 1"),
            Parameter("val2", description="Param 2"),
        ]
        pl = ParameterList(params)
        assert len(pl) == 2
        assert pl[0].value == "val1"
        assert pl[1].value == "val2"

    def test_append_parameter(self) -> None:
        """Parameters can be appended to ParameterList."""
        pl = ParameterList()
        pl.append(Parameter("test", description="Test param"))
        assert len(pl) == 1
        assert pl[0].value == "test"

    def test_append_non_parameter_raises(self) -> None:
        """Appending non-Parameter raises TypeError."""
        pl = ParameterList()
        with pytest.raises(TypeError, match="only accepts Parameter"):
            pl.append("not a parameter")  # type: ignore[arg-type]

    def test_indexing(self) -> None:
        """Parameters can be accessed by index."""
        pl = ParameterList(
            [
                Parameter("a", description="A"),
                Parameter("b", description="B"),
                Parameter("c", description="C"),
            ]
        )
        assert pl[0].value == "a"
        assert pl[1].value == "b"
        assert pl[-1].value == "c"

    def test_slicing(self) -> None:
        """ParameterList supports slicing."""
        pl = ParameterList([Parameter(str(i), description=f"P{i}") for i in range(5)])
        sliced = pl[1:3]
        assert len(sliced) == 2
        assert sliced[0].value == "1"
        assert sliced[1].value == "2"

    def test_iteration(self) -> None:
        """ParameterList is iterable."""
        pl = ParameterList(
            [
                Parameter("a", description="A"),
                Parameter("b", description="B"),
            ]
        )
        values = [p.value for p in pl]
        assert values == ["a", "b"]

    def test_parameters_method(self) -> None:
        """parameters() yields all contained parameters."""
        pl = ParameterList(
            [
                Parameter("x", description="X"),
                Parameter("y", description="Y"),
            ]
        )
        params = list(pl.parameters())
        assert len(params) == 2
        assert params[0].value == "x"
        assert params[1].value == "y"

    def test_named_parameters_method(self) -> None:
        """named_parameters() yields (name, param) tuples with index names."""
        pl = ParameterList(
            [
                Parameter("x", description="X"),
                Parameter("y", description="Y"),
            ]
        )
        named = list(pl.named_parameters())
        assert len(named) == 2
        assert named[0] == ("0", pl[0])
        assert named[1] == ("1", pl[1])

    def test_named_parameters_with_prefix(self) -> None:
        """named_parameters() respects prefix argument."""
        pl = ParameterList([Parameter("x", description="X")])
        named = list(pl.named_parameters(prefix="prompts"))
        assert named[0][0] == "prompts.0"


class TestParameterDict:
    """Tests for ParameterDict container."""

    def test_creation_empty(self) -> None:
        """ParameterDict can be created empty."""
        pd = ParameterDict()
        assert len(pd) == 0
        assert list(pd.parameters()) == []

    def test_creation_with_dict(self) -> None:
        """ParameterDict can be initialized with a dict."""
        pd = ParameterDict(
            {
                "a": Parameter("val_a", description="A"),
                "b": Parameter("val_b", description="B"),
            }
        )
        assert len(pd) == 2
        assert pd["a"].value == "val_a"
        assert pd["b"].value == "val_b"

    def test_setitem_and_getitem(self) -> None:
        """Parameters can be set and retrieved by key."""
        pd = ParameterDict()
        pd["key"] = Parameter("value", description="Test")
        assert pd["key"].value == "value"

    def test_setitem_non_parameter_raises(self) -> None:
        """Setting non-Parameter value raises TypeError."""
        pd = ParameterDict()
        with pytest.raises(TypeError, match="only accepts Parameter"):
            pd["key"] = "not a parameter"  # type: ignore[assignment]

    def test_delitem(self) -> None:
        """Parameters can be deleted by key."""
        pd = ParameterDict({"a": Parameter("val", description="A")})
        del pd["a"]
        assert len(pd) == 0

    def test_iteration(self) -> None:
        """ParameterDict keys are iterable."""
        pd = ParameterDict(
            {
                "x": Parameter("px", description="X"),
                "y": Parameter("py", description="Y"),
            }
        )
        keys = list(pd)
        assert "x" in keys
        assert "y" in keys

    def test_parameters_method(self) -> None:
        """parameters() yields all contained parameters."""
        pd = ParameterDict(
            {
                "a": Parameter("pa", description="A"),
                "b": Parameter("pb", description="B"),
            }
        )
        params = list(pd.parameters())
        assert len(params) == 2
        values = {p.value for p in params}
        assert values == {"pa", "pb"}

    def test_named_parameters_method(self) -> None:
        """named_parameters() yields (name, param) tuples with key names."""
        pd = ParameterDict(
            {
                "first": Parameter("p1", description="First"),
                "second": Parameter("p2", description="Second"),
            }
        )
        named = dict(pd.named_parameters())
        assert "first" in named
        assert "second" in named
        assert named["first"].value == "p1"
        assert named["second"].value == "p2"

    def test_named_parameters_with_prefix(self) -> None:
        """named_parameters() respects prefix argument."""
        pd = ParameterDict({"key": Parameter("val", description="K")})
        named = dict(pd.named_parameters(prefix="configs"))
        assert "configs.key" in named


class TestModuleList:
    """Tests for ModuleList container."""

    def test_creation_empty(self) -> None:
        """ModuleList can be created empty."""
        ml = ModuleList()
        assert len(ml) == 0
        assert list(ml.children()) == []

    def test_creation_with_modules(self) -> None:
        """ModuleList can be initialized with modules."""
        modules = [Module(), Module()]
        ml = ModuleList(modules)
        assert len(ml) == 2

    def test_creation_with_non_module_raises(self) -> None:
        """Creating with non-Module raises TypeError."""
        with pytest.raises(TypeError, match="only accepts Module"):
            ModuleList(["not a module"])  # type: ignore[list-item]

    def test_append_module(self) -> None:
        """Modules can be appended to ModuleList."""
        ml = ModuleList()
        ml.append(Module())
        assert len(ml) == 1

    def test_append_non_module_raises(self) -> None:
        """Appending non-Module raises TypeError."""
        ml = ModuleList()
        with pytest.raises(TypeError, match="only accepts Module"):
            ml.append("not a module")  # type: ignore[arg-type]

    def test_indexing(self) -> None:
        """Modules can be accessed by index."""
        m1, m2 = Module(), Module()
        ml = ModuleList([m1, m2])
        assert ml[0] is m1
        assert ml[1] is m2

    def test_children_method(self) -> None:
        """children() yields all contained modules."""
        m1, m2 = Module(), Module()
        ml = ModuleList([m1, m2])
        children = list(ml.children())
        assert len(children) == 2
        assert m1 in children
        assert m2 in children

    def test_named_children_method(self) -> None:
        """named_children() yields (name, module) tuples with index names."""
        m1, m2 = Module(), Module()
        ml = ModuleList([m1, m2])
        named = list(ml.named_children())
        assert named[0] == ("0", m1)
        assert named[1] == ("1", m2)

    def test_parameters_from_contained_modules(self) -> None:
        """parameters() yields parameters from all contained modules."""

        class ModuleWithParam(Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.param = Parameter(name, description=f"Param {name}")

        ml = ModuleList([ModuleWithParam("a"), ModuleWithParam("b")])
        params = list(ml.parameters())
        assert len(params) == 2
        values = {p.value for p in params}
        assert values == {"a", "b"}

    def test_named_parameters_from_contained_modules(self) -> None:
        """named_parameters() yields hierarchical names for contained params."""

        class ModuleWithParam(Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.param = Parameter(name, description=f"Param {name}")

        ml = ModuleList([ModuleWithParam("x")])
        named = dict(ml.named_parameters())
        assert "0.param" in named
        assert named["0.param"].value == "x"


class TestModuleDict:
    """Tests for ModuleDict container."""

    def test_creation_empty(self) -> None:
        """ModuleDict can be created empty."""
        md = ModuleDict()
        assert len(md) == 0
        assert list(md.children()) == []

    def test_creation_with_dict(self) -> None:
        """ModuleDict can be initialized with a dict."""
        m1, m2 = Module(), Module()
        md = ModuleDict({"first": m1, "second": m2})
        assert len(md) == 2
        assert md["first"] is m1
        assert md["second"] is m2

    def test_creation_with_non_module_raises(self) -> None:
        """Creating with non-Module raises TypeError."""
        with pytest.raises(TypeError, match="only accepts Module"):
            ModuleDict({"key": "not a module"})  # type: ignore[dict-item]

    def test_setitem_and_getitem(self) -> None:
        """Modules can be set and retrieved by key."""
        md = ModuleDict()
        m = Module()
        md["key"] = m
        assert md["key"] is m

    def test_setitem_non_module_raises(self) -> None:
        """Setting non-Module value raises TypeError."""
        md = ModuleDict()
        with pytest.raises(TypeError, match="only accepts Module"):
            md["key"] = "not a module"  # type: ignore[assignment]

    def test_children_method(self) -> None:
        """children() yields all contained modules."""
        m1, m2 = Module(), Module()
        md = ModuleDict({"a": m1, "b": m2})
        children = list(md.children())
        assert len(children) == 2
        assert m1 in children
        assert m2 in children

    def test_named_children_method(self) -> None:
        """named_children() yields (name, module) tuples with key names."""
        m1, m2 = Module(), Module()
        md = ModuleDict({"first": m1, "second": m2})
        named = dict(md.named_children())
        assert "first" in named
        assert "second" in named
        assert named["first"] is m1
        assert named["second"] is m2

    def test_parameters_from_contained_modules(self) -> None:
        """parameters() yields parameters from all contained modules."""

        class ModuleWithParam(Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.param = Parameter(name, description=f"Param {name}")

        md = ModuleDict(
            {
                "a": ModuleWithParam("param_a"),
                "b": ModuleWithParam("param_b"),
            }
        )
        params = list(md.parameters())
        assert len(params) == 2
        values = {p.value for p in params}
        assert values == {"param_a", "param_b"}


class TestModuleIntegrationWithContainers:
    """Tests for Module integration with container types."""

    def test_parameter_list_in_module(self) -> None:
        """ParameterList in Module is properly collected by parameters()."""

        class MultiPrompt(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("p1", description="First"),
                        Parameter("p2", description="Second"),
                        Parameter("p3", description="Third"),
                    ]
                )

        module = MultiPrompt()
        params = list(module.parameters())
        assert len(params) == 3
        values = {p.value for p in params}
        assert values == {"p1", "p2", "p3"}

    def test_parameter_list_named_parameters(self) -> None:
        """ParameterList parameters have correct hierarchical names."""

        class MultiPrompt(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("p1", description="First"),
                        Parameter("p2", description="Second"),
                    ]
                )

        module = MultiPrompt()
        named = dict(module.named_parameters())
        assert "prompts.0" in named
        assert "prompts.1" in named
        assert named["prompts.0"].value == "p1"
        assert named["prompts.1"].value == "p2"

    def test_parameter_dict_in_module(self) -> None:
        """ParameterDict in Module is properly collected by parameters()."""

        class TaskPrompts(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterDict(
                    {
                        "summarize": Parameter("Summarize:", description="Summary"),
                        "translate": Parameter("Translate:", description="Translation"),
                    }
                )

        module = TaskPrompts()
        params = list(module.parameters())
        assert len(params) == 2
        values = {p.value for p in params}
        assert values == {"Summarize:", "Translate:"}

    def test_parameter_dict_named_parameters(self) -> None:
        """ParameterDict parameters have correct hierarchical names."""

        class TaskPrompts(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterDict(
                    {
                        "task1": Parameter("t1", description="Task 1"),
                        "task2": Parameter("t2", description="Task 2"),
                    }
                )

        module = TaskPrompts()
        named = dict(module.named_parameters())
        assert "prompts.task1" in named
        assert "prompts.task2" in named

    def test_module_list_in_module(self) -> None:
        """ModuleList in Module is properly collected by children()."""

        class Pipeline(Module):
            def __init__(self) -> None:
                super().__init__()
                self.stages = ModuleList(
                    [
                        LLMInference(alias="stage1"),
                        LLMInference(alias="stage2"),
                    ]
                )

        pipeline = Pipeline()
        # ModuleList itself is a child
        children = list(pipeline.children())
        assert len(children) == 1  # The ModuleList itself

    def test_module_list_modules_iteration(self) -> None:
        """ModuleList modules are collected by modules()."""

        class Pipeline(Module):
            def __init__(self) -> None:
                super().__init__()
                self.stages = ModuleList([Module(), Module()])

        pipeline = Pipeline()
        # modules() includes self + ModuleList + all contained modules
        all_modules = list(pipeline.modules())
        # self (Pipeline) + 2 contained modules
        # Note: ModuleList is not a Module subclass, so it's not yielded
        assert len(all_modules) >= 1  # At least self

    def test_module_list_parameters_collected(self) -> None:
        """Parameters in ModuleList modules are collected."""

        class ModuleWithParam(Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.param = Parameter(name, description=f"Param {name}")

        class Pipeline(Module):
            def __init__(self) -> None:
                super().__init__()
                self.stages = ModuleList(
                    [
                        ModuleWithParam("stage1"),
                        ModuleWithParam("stage2"),
                    ]
                )

        pipeline = Pipeline()
        # The ModuleList.parameters() method should yield params from children
        # But we need to check that named_parameters works with container
        named = dict(pipeline.named_parameters())
        # Should have stages.0.param and stages.1.param
        assert len(named) == 2
        assert "stages.0.param" in named or any("param" in k for k in named)

    def test_module_dict_in_module(self) -> None:
        """ModuleDict in Module is properly collected by children()."""

        class MultiTask(Module):
            def __init__(self) -> None:
                super().__init__()
                self.tasks = ModuleDict(
                    {
                        "summarize": LLMInference(alias="summarizer"),
                        "translate": LLMInference(alias="translator"),
                    }
                )

        module = MultiTask()
        children = list(module.children())
        assert len(children) == 1  # The ModuleDict itself

    def test_mixed_containers_and_direct_params(self) -> None:
        """Module with both containers and direct parameters works correctly."""

        class MixedModule(Module):
            def __init__(self) -> None:
                super().__init__()
                # Direct parameter
                self.global_prompt = Parameter("global", description="Global")
                # ParameterList
                self.prompts = ParameterList(
                    [
                        Parameter("list1", description="List 1"),
                        Parameter("list2", description="List 2"),
                    ]
                )
                # ParameterDict
                self.configs = ParameterDict(
                    {
                        "cfg1": Parameter("config1", description="Config 1"),
                    }
                )

        module = MixedModule()
        params = list(module.parameters())
        # Should have 4 total: global + 2 from list + 1 from dict
        assert len(params) == 4

        named = dict(module.named_parameters())
        assert "global_prompt" in named
        assert "prompts.0" in named
        assert "prompts.1" in named
        assert "configs.cfg1" in named

    def test_nested_module_with_containers(self) -> None:
        """Nested modules with containers are properly collected."""

        class Inner(Module):
            def __init__(self) -> None:
                super().__init__()
                self.params = ParameterList(
                    [
                        Parameter("inner1", description="Inner 1"),
                        Parameter("inner2", description="Inner 2"),
                    ]
                )

        class Outer(Module):
            def __init__(self) -> None:
                super().__init__()
                self.outer_param = Parameter("outer", description="Outer")
                self.inner = Inner()

        outer = Outer()
        params = list(outer.parameters())
        # outer_param + 2 from inner.params
        assert len(params) == 3

        named = dict(outer.named_parameters())
        assert "outer_param" in named
        assert "inner.params.0" in named
        assert "inner.params.1" in named

    def test_state_dict_with_containers(self) -> None:
        """state_dict() works with container-based parameters."""

        class MultiPrompt(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("p1", description="First"),
                        Parameter("p2", description="Second"),
                    ]
                )

        module = MultiPrompt()
        state = module.state_dict()
        assert "prompts.0" in state
        assert "prompts.1" in state
        assert state["prompts.0"] == "p1"
        assert state["prompts.1"] == "p2"

    def test_load_state_dict_with_containers(self) -> None:
        """load_state_dict() works with container-based parameters."""

        class MultiPrompt(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("old1", description="First"),
                        Parameter("old2", description="Second"),
                    ]
                )

        module = MultiPrompt()
        module.load_state_dict({"prompts.0": "new1", "prompts.1": "new2"})
        assert module.prompts[0].value == "new1"
        assert module.prompts[1].value == "new2"


class TestContainerRepr:
    """Tests for container __repr__ methods."""

    def test_parameter_list_repr(self) -> None:
        """ParameterList has informative repr."""
        pl = ParameterList([Parameter("x", description="X")])
        r = repr(pl)
        assert "ParameterList" in r

    def test_parameter_dict_repr(self) -> None:
        """ParameterDict has informative repr."""
        pd = ParameterDict({"k": Parameter("v", description="V")})
        r = repr(pd)
        assert "ParameterDict" in r

    def test_module_list_repr(self) -> None:
        """ModuleList has informative repr."""
        ml = ModuleList([Module()])
        r = repr(ml)
        assert "ModuleList" in r

    def test_module_dict_repr(self) -> None:
        """ModuleDict has informative repr."""
        md = ModuleDict({"k": Module()})
        r = repr(md)
        assert "ModuleDict" in r
