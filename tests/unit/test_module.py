"""Unit tests for the Module base class.

Tests cover:
- Basic instantiation and child/parameter registration
- Introspection methods: children(), modules(), parameters(), named_* iterators
Tests are consolidated using parametrize to reduce redundancy.
"""

import pytest

from plait.module import LLMInference, Module
from plait.parameter import Parameter


class TestModuleInstantiation:
    """Tests for Module basic instantiation."""

    def test_module_initial_state(self) -> None:
        """Module has correct initial state after init."""
        module = Module()

        assert isinstance(module, Module)
        assert hasattr(module, "_children")
        assert hasattr(module, "_parameters")
        assert hasattr(module, "_name")
        assert module._children == {}
        assert module._parameters == {}
        assert module._name is None


class TestChildModuleRegistration:
    """Tests for automatic child module registration."""

    def test_child_registered_with_name(self) -> None:
        """Assigning a Module registers it and sets its _name."""

        class Parent(Module):
            def __init__(self) -> None:
                super().__init__()
                self.my_child = Module()

        parent = Parent()

        assert "my_child" in parent._children
        assert parent._children["my_child"] is parent.my_child
        assert parent.my_child._name == "my_child"

    def test_multiple_children_registered(self) -> None:
        """Multiple child modules are all registered."""

        class Parent(Module):
            def __init__(self) -> None:
                super().__init__()
                self.child1 = Module()
                self.child2 = Module()
                self.child3 = Module()

        parent = Parent()

        assert len(parent._children) == 3
        for name in ["child1", "child2", "child3"]:
            assert name in parent._children

    def test_nested_child_registration(self) -> None:
        """Nested modules are registered at each level."""

        class Inner(Module):
            pass

        class Middle(Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

        class Outer(Module):
            def __init__(self) -> None:
                super().__init__()
                self.middle = Middle()

        outer = Outer()

        assert "middle" in outer._children
        assert "inner" in outer.middle._children

    def test_reassigning_child_updates_registration(self) -> None:
        """Reassigning a child updates the registration."""

        class Parent(Module):
            def __init__(self) -> None:
                super().__init__()
                self.child = Module()

        parent = Parent()
        new_child = Module()
        parent.child = new_child

        assert parent._children["child"] is new_child
        assert new_child._name == "child"


class TestParameterRegistration:
    """Tests for automatic parameter registration."""

    def test_parameter_registered_with_name(self) -> None:
        """Assigning a Parameter registers it and sets its _name."""

        class MyModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.my_param = Parameter("test", description="test")

        module = MyModule()

        assert "my_param" in module._parameters
        assert module._parameters["my_param"] is module.my_param
        assert module.my_param._name == "my_param"

    def test_multiple_parameters_registered(self) -> None:
        """Multiple parameters are all registered."""

        class MyModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.param1 = Parameter("v1", description="test")
                self.param2 = Parameter("v2", description="test")
                self.param3 = Parameter("v3", description="test")

        module = MyModule()

        assert len(module._parameters) == 3
        for name in ["param1", "param2", "param3"]:
            assert name in module._parameters

    def test_requires_grad_false_still_registered(self) -> None:
        """Parameters with requires_grad=False are still registered."""

        class MyModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.frozen = Parameter(
                    "frozen", description="test", requires_grad=False
                )

        module = MyModule()
        assert "frozen" in module._parameters


class TestMixedRegistration:
    """Tests for modules with both children and parameters."""

    def test_children_and_parameters_separate(self) -> None:
        """Children and parameters are tracked in separate dicts."""

        class MyModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.child = Module()
                self.param = Parameter("test", description="test")
                self.regular = "plain string"

        module = MyModule()

        # Check children
        assert "child" in module._children
        assert "child" not in module._parameters

        # Check parameters
        assert "param" in module._parameters
        assert "param" not in module._children

        # Check regular attributes
        assert module.regular == "plain string"
        assert "regular" not in module._children
        assert "regular" not in module._parameters


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_none_attribute_not_registered(self) -> None:
        """None values are not registered."""

        class MyModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.optional = None

        module = MyModule()
        assert module._children == {}
        assert module.optional is None

    def test_shared_module_gets_last_parent_name(self) -> None:
        """A module assigned to multiple parents gets last parent's name."""
        shared_child = Module()

        class Parent1(Module):
            def __init__(self) -> None:
                super().__init__()
                self.child_a = shared_child

        class Parent2(Module):
            def __init__(self) -> None:
                super().__init__()
                self.child_b = shared_child

        _parent1 = Parent1()
        assert shared_child._name == "child_a"

        _parent2 = Parent2()
        assert shared_child._name == "child_b"

    def test_subclass_without_super_init_raises(self) -> None:
        """Subclass that forgets super().__init__() raises AttributeError."""

        class BadModule(Module):
            def __init__(self) -> None:
                pass  # Forgot to call super().__init__()

        bad = BadModule()
        with pytest.raises(AttributeError):
            bad.child = Module()


class TestChildrenIterators:
    """Tests for children() and named_children() methods."""

    def test_children_empty_module(self) -> None:
        """children() yields nothing for module with no children."""
        module = Module()
        assert list(module.children()) == []
        assert list(module.named_children()) == []

    def test_children_yields_immediate_only(self) -> None:
        """children() yields only immediate children, not grandchildren."""

        class Inner(Module):
            def __init__(self) -> None:
                super().__init__()
                self.grandchild = Module()

        class Outer(Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()
                self.param = Parameter("test", description="test")

        outer = Outer()
        children_list = list(outer.children())
        named_list = list(outer.named_children())

        assert len(children_list) == 1
        assert children_list[0] is outer.inner
        assert len(named_list) == 1
        assert named_list[0] == ("inner", outer.inner)


class TestModulesIterators:
    """Tests for modules() and named_modules() methods."""

    def test_modules_includes_self(self) -> None:
        """modules() yields self as first item."""
        module = Module()
        modules_list = list(module.modules())
        named_list = list(module.named_modules())

        assert modules_list == [module]
        assert named_list == [("", module)]

    def test_modules_nested_structure(self) -> None:
        """modules() yields all modules in depth-first order."""

        class Level2(Module):
            pass

        class Level1(Module):
            def __init__(self) -> None:
                super().__init__()
                self.level2 = Level2()

        class Root(Module):
            def __init__(self) -> None:
                super().__init__()
                self.level1 = Level1()

        root = Root()
        modules_list = list(root.modules())
        named_dict = dict(root.named_modules())

        assert len(modules_list) == 3
        assert modules_list[0] is root
        assert modules_list[1] is root.level1
        assert modules_list[2] is root.level1.level2

        assert "" in named_dict
        assert "level1" in named_dict
        assert "level1.level2" in named_dict

    def test_named_modules_with_prefix(self) -> None:
        """named_modules(prefix) prepends prefix to all names."""

        class Child(Module):
            pass

        class Parent(Module):
            def __init__(self) -> None:
                super().__init__()
                self.child = Child()

        parent = Parent()
        named_dict = dict(parent.named_modules(prefix="base"))

        assert "base" in named_dict
        assert "base.child" in named_dict


class TestParametersIterators:
    """Tests for parameters() and named_parameters() methods."""

    def test_parameters_empty_module(self) -> None:
        """parameters() yields nothing for module with no parameters."""
        module = Module()
        assert list(module.parameters()) == []
        assert list(module.named_parameters()) == []

    def test_parameters_recurses_into_children(self) -> None:
        """parameters() recursively yields parameters from children."""

        class Child(Module):
            def __init__(self) -> None:
                super().__init__()
                self.child_param = Parameter("child", description="test")

        class Parent(Module):
            def __init__(self) -> None:
                super().__init__()
                self.parent_param = Parameter("parent", description="test")
                self.child = Child()

        parent = Parent()
        params_list = list(parent.parameters())
        named_dict = dict(parent.named_parameters())

        assert len(params_list) == 2
        assert parent.parent_param in params_list
        assert parent.child.child_param in params_list

        assert "parent_param" in named_dict
        assert "child.child_param" in named_dict


class TestLLMInference:
    """Tests for LLMInference module."""

    def test_creation_with_alias(self) -> None:
        """LLMInference can be created with an alias."""
        llm = LLMInference(alias="fast")
        assert llm.alias == "fast"

    @pytest.mark.parametrize(
        "field,default",
        [
            ("temperature", 1.0),
            ("max_tokens", None),
            ("response_format", None),
        ],
    )
    def test_defaults(self, field: str, default: object) -> None:
        """LLMInference has correct default values."""
        llm = LLMInference(alias="test")
        assert getattr(llm, field) == default

    def test_creation_with_all_options(self) -> None:
        """LLMInference stores all provided options."""
        llm = LLMInference(
            alias="test",
            system_prompt="You are helpful",
            temperature=0.7,
            max_tokens=1000,
        )

        assert llm.alias == "test"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 1000

    def test_is_module_subclass(self) -> None:
        """LLMInference is a Module subclass."""
        llm = LLMInference(alias="test")
        assert isinstance(llm, Module)

    def test_registerable_as_child(self) -> None:
        """LLMInference can be registered as a child module."""

        class Parent(Module):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="fast")

        parent = Parent()
        assert "llm" in parent._children
        assert parent.llm._name == "llm"

    def test_can_have_parameters(self) -> None:
        """LLMInference can have Parameter system_prompt."""

        class LLMWithParam(LLMInference):
            def __init__(self) -> None:
                super().__init__(alias="test")
                self.prompt = Parameter("You are helpful", description="System prompt")

        llm = LLMWithParam()
        assert "prompt" in llm._parameters
