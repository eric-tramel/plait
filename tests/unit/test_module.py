"""Unit tests for the InferenceModule base class.

This file contains tests for:
- PR-003 (core structure): Basic instantiation, child/parameter registration
- PR-004 (introspection): children(), modules(), parameters(), named_* iterators
"""

from inf_engine.module import InferenceModule
from inf_engine.parameter import Parameter


class TestInferenceModuleInstantiation:
    """Tests for InferenceModule basic instantiation."""

    def test_module_instantiation(self) -> None:
        """InferenceModule can be instantiated."""
        module = InferenceModule()
        assert module is not None

    def test_module_has_children_dict(self) -> None:
        """InferenceModule has _children dict after init."""
        module = InferenceModule()
        assert hasattr(module, "_children")
        assert isinstance(module._children, dict)
        assert module._children == {}

    def test_module_has_parameters_dict(self) -> None:
        """InferenceModule has _parameters dict after init."""
        module = InferenceModule()
        assert hasattr(module, "_parameters")
        assert isinstance(module._parameters, dict)
        assert module._parameters == {}

    def test_module_has_name_attribute(self) -> None:
        """InferenceModule has _name attribute after init."""
        module = InferenceModule()
        assert hasattr(module, "_name")
        assert module._name is None


class TestChildModuleRegistration:
    """Tests for automatic child module registration."""

    def test_child_module_registered(self) -> None:
        """Assigning an InferenceModule registers it as a child."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = InferenceModule()

        parent = Parent()
        assert "child" in parent._children
        assert parent._children["child"] is parent.child

    def test_child_module_name_set(self) -> None:
        """Child module's _name is set to the attribute name."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_child = InferenceModule()

        parent = Parent()
        assert parent.my_child._name == "my_child"

    def test_multiple_children_registered(self) -> None:
        """Multiple child modules are all registered."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child1 = InferenceModule()
                self.child2 = InferenceModule()
                self.child3 = InferenceModule()

        parent = Parent()
        assert len(parent._children) == 3
        assert "child1" in parent._children
        assert "child2" in parent._children
        assert "child3" in parent._children

    def test_nested_child_registration(self) -> None:
        """Nested modules are registered at each level."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

        class Middle(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.middle = Middle()

        outer = Outer()
        assert "middle" in outer._children
        assert "inner" in outer.middle._children

    def test_reassigning_child_updates_registration(self) -> None:
        """Reassigning a child updates the registration."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = InferenceModule()

        parent = Parent()
        original_child = parent.child
        new_child = InferenceModule()
        parent.child = new_child

        assert parent._children["child"] is new_child
        assert parent.child is new_child
        assert new_child._name == "child"
        # Original child's name is not cleared (by design, like PyTorch)
        assert original_child._name == "child"


class TestParameterRegistration:
    """Tests for automatic parameter registration."""

    def test_parameter_registered(self) -> None:
        """Assigning a Parameter registers it in _parameters."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("test prompt", description="test")

        module = MyModule()
        assert "prompt" in module._parameters
        assert module._parameters["prompt"] is module.prompt

    def test_parameter_name_set(self) -> None:
        """Parameter's _name is set to the attribute name."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_param = Parameter("test", description="test")

        module = MyModule()
        assert module.my_param._name == "my_param"

    def test_multiple_parameters_registered(self) -> None:
        """Multiple parameters are all registered."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.param1 = Parameter("value1", description="test")
                self.param2 = Parameter("value2", description="test")
                self.param3 = Parameter("value3", description="test")

        module = MyModule()
        assert len(module._parameters) == 3
        assert "param1" in module._parameters
        assert "param2" in module._parameters
        assert "param3" in module._parameters

    def test_parameter_requires_grad_false_still_registered(self) -> None:
        """Parameters with requires_grad=False are still registered."""

        class MyModule(InferenceModule):
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

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = InferenceModule()
                self.param = Parameter("test", description="test")

        module = MyModule()
        assert "child" in module._children
        assert "child" not in module._parameters
        assert "param" in module._parameters
        assert "param" not in module._children

    def test_complex_module_structure(self) -> None:
        """Complex module with multiple children and parameters."""

        class ComplexModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.system_prompt = Parameter("You are helpful.", description="test")
                self.temperature = 0.7  # Regular attribute
                self.sub_module = InferenceModule()
                self.response_format = Parameter("json", description="test")

        module = ComplexModule()

        # Check parameters
        assert len(module._parameters) == 2
        assert "system_prompt" in module._parameters
        assert "response_format" in module._parameters

        # Check children
        assert len(module._children) == 1
        assert "sub_module" in module._children

        # Check regular attribute is accessible but not registered
        assert module.temperature == 0.7
        assert "temperature" not in module._parameters
        assert "temperature" not in module._children


class TestRegularAttributes:
    """Tests for non-module, non-parameter attributes."""

    def test_regular_attributes_not_registered(self) -> None:
        """Regular attributes are not registered as children or parameters."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.name = "test"
                self.count = 42
                self.data = [1, 2, 3]

        module = MyModule()
        assert module._children == {}
        assert module._parameters == {}
        assert module.name == "test"
        assert module.count == 42
        assert module.data == [1, 2, 3]

    def test_none_attribute_not_registered(self) -> None:
        """None values are not registered."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.optional_child = None

        module = MyModule()
        assert module._children == {}
        assert module.optional_child is None


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_reassign_parameter_to_module(self) -> None:
        """Reassigning from Parameter to Module updates both registries."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.thing = Parameter("initial", description="test")

        module = MyModule()
        assert "thing" in module._parameters
        assert "thing" not in module._children

        # Reassign to a Module
        module.thing = InferenceModule()

        # Should now be in _children, but _parameters is NOT auto-cleaned
        # (this matches PyTorch behavior - old registrations are not removed)
        assert "thing" in module._children
        assert "thing" in module._parameters  # Stale entry remains

    def test_reassign_module_to_parameter(self) -> None:
        """Reassigning from Module to Parameter updates both registries."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.thing = InferenceModule()

        module = MyModule()
        assert "thing" in module._children
        assert "thing" not in module._parameters

        # Reassign to a Parameter
        module.thing = Parameter("now a param", description="test")

        # Should now be in _parameters, but _children is NOT auto-cleaned
        assert "thing" in module._parameters
        assert "thing" in module._children  # Stale entry remains

    def test_module_assigned_to_multiple_parents(self) -> None:
        """A module assigned to multiple parents gets last parent's name."""
        shared_child = InferenceModule()

        class Parent1(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child_a = shared_child

        class Parent2(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child_b = shared_child

        parent1 = Parent1()
        assert shared_child._name == "child_a"

        parent2 = Parent2()
        # Name is overwritten by second parent
        assert shared_child._name == "child_b"

        # Both parents still have it registered
        assert parent1._children["child_a"] is shared_child
        assert parent2._children["child_b"] is shared_child

    def test_parameter_assigned_to_multiple_modules(self) -> None:
        """A parameter assigned to multiple modules gets last module's name."""
        shared_param = Parameter("shared value", description="test")

        class Module1(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt_a = shared_param

        class Module2(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt_b = shared_param

        mod1 = Module1()
        assert shared_param._name == "prompt_a"

        mod2 = Module2()
        # Name is overwritten by second module
        assert shared_param._name == "prompt_b"

        # Both modules still have it registered
        assert mod1._parameters["prompt_a"] is shared_param
        assert mod2._parameters["prompt_b"] is shared_param

    def test_subclass_without_super_init_raises(self) -> None:
        """Subclass that forgets super().__init__() raises AttributeError."""
        import pytest

        class BadModule(InferenceModule):
            def __init__(self) -> None:
                # Forgot to call super().__init__()
                pass

        # Instantiation works, but assigning a child fails
        bad = BadModule()
        with pytest.raises(AttributeError):
            bad.child = InferenceModule()

    def test_empty_string_parameter_registered(self) -> None:
        """Empty string parameters are still registered."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.empty = Parameter("", description="test")

        module = MyModule()
        assert "empty" in module._parameters
        assert module.empty.value == ""

    def test_attribute_accessible_after_registration(self) -> None:
        """Verify attributes are accessible after registration (not just registered)."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = InferenceModule()
                self.param = Parameter("test", description="test")
                self.regular = "plain"

        module = MyModule()

        # All attributes should be directly accessible
        assert isinstance(module.child, InferenceModule)
        assert isinstance(module.param, Parameter)
        assert module.regular == "plain"

        # And should match what's in the registries
        assert module.child is module._children["child"]
        assert module.param is module._parameters["param"]

    def test_reassign_to_regular_value_leaves_stale_registration(self) -> None:
        """Reassigning a child/param to regular value leaves stale registration."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.thing = InferenceModule()

        module = MyModule()
        assert "thing" in module._children

        # Reassign to a regular value
        module.thing = "just a string"

        # Stale entry remains in _children (matches PyTorch behavior)
        assert "thing" in module._children
        # But the actual attribute is the string
        assert module.thing == "just a string"


# ─────────────────────────────────────────────────────────────────────────────
# PR-004: Introspection Methods Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestChildrenIterator:
    """Tests for children() method."""

    def test_children_empty_module(self) -> None:
        """children() yields nothing for module with no children."""
        module = InferenceModule()
        assert list(module.children()) == []

    def test_children_single_child(self) -> None:
        """children() yields the single child module."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = InferenceModule()

        parent = Parent()
        children_list = list(parent.children())

        assert len(children_list) == 1
        assert children_list[0] is parent.child

    def test_children_multiple_children(self) -> None:
        """children() yields all immediate children."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child1 = InferenceModule()
                self.child2 = InferenceModule()
                self.child3 = InferenceModule()

        parent = Parent()
        children_list = list(parent.children())

        assert len(children_list) == 3
        assert parent.child1 in children_list
        assert parent.child2 in children_list
        assert parent.child3 in children_list

    def test_children_does_not_recurse(self) -> None:
        """children() only yields immediate children, not grandchildren."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.grandchild = InferenceModule()

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

        outer = Outer()
        children_list = list(outer.children())

        # Only inner, not grandchild
        assert len(children_list) == 1
        assert children_list[0] is outer.inner

    def test_children_excludes_parameters(self) -> None:
        """children() does not yield parameters."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = InferenceModule()
                self.param = Parameter("test", description="test")

        module = MyModule()
        children_list = list(module.children())

        assert len(children_list) == 1
        assert children_list[0] is module.child


class TestNamedChildrenIterator:
    """Tests for named_children() method."""

    def test_named_children_empty_module(self) -> None:
        """named_children() yields nothing for module with no children."""
        module = InferenceModule()
        assert list(module.named_children()) == []

    def test_named_children_single_child(self) -> None:
        """named_children() yields (name, module) tuple."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_child = InferenceModule()

        parent = Parent()
        named_list = list(parent.named_children())

        assert len(named_list) == 1
        assert named_list[0] == ("my_child", parent.my_child)

    def test_named_children_multiple_children(self) -> None:
        """named_children() yields all (name, module) tuples."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.alpha = InferenceModule()
                self.beta = InferenceModule()

        parent = Parent()
        named_dict = dict(parent.named_children())

        assert len(named_dict) == 2
        assert named_dict["alpha"] is parent.alpha
        assert named_dict["beta"] is parent.beta

    def test_named_children_does_not_recurse(self) -> None:
        """named_children() only yields immediate children."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.deep = InferenceModule()

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

        outer = Outer()
        named_list = list(outer.named_children())

        assert len(named_list) == 1
        assert named_list[0][0] == "inner"


class TestModulesIterator:
    """Tests for modules() method."""

    def test_modules_includes_self(self) -> None:
        """modules() yields self as first item."""
        module = InferenceModule()
        modules_list = list(module.modules())

        assert len(modules_list) == 1
        assert modules_list[0] is module

    def test_modules_single_child(self) -> None:
        """modules() yields self and child."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = InferenceModule()

        parent = Parent()
        modules_list = list(parent.modules())

        assert len(modules_list) == 2
        assert modules_list[0] is parent
        assert modules_list[1] is parent.child

    def test_modules_nested_structure(self) -> None:
        """modules() yields all modules in depth-first order."""

        class Level2(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

        class Level1(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level2 = Level2()

        class Root(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level1 = Level1()

        root = Root()
        modules_list = list(root.modules())

        assert len(modules_list) == 3
        assert modules_list[0] is root
        assert modules_list[1] is root.level1
        assert modules_list[2] is root.level1.level2

    def test_modules_multiple_children(self) -> None:
        """modules() yields all modules from multiple children."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child1 = InferenceModule()
                self.child2 = InferenceModule()

        parent = Parent()
        modules_list = list(parent.modules())

        assert len(modules_list) == 3
        assert parent in modules_list
        assert parent.child1 in modules_list
        assert parent.child2 in modules_list

    def test_modules_complex_tree(self) -> None:
        """modules() traverses complex tree structure correctly."""

        class Leaf(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

        class Branch(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.leaf1 = Leaf()
                self.leaf2 = Leaf()

        class Root(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.branch1 = Branch()
                self.branch2 = Branch()

        root = Root()
        modules_list = list(root.modules())

        # root + 2 branches + 4 leaves = 7
        assert len(modules_list) == 7


class TestNamedModulesIterator:
    """Tests for named_modules() method."""

    def test_named_modules_self_has_empty_name(self) -> None:
        """named_modules() yields self with empty string name."""
        module = InferenceModule()
        named_list = list(module.named_modules())

        assert len(named_list) == 1
        assert named_list[0] == ("", module)

    def test_named_modules_with_prefix(self) -> None:
        """named_modules(prefix) prepends prefix to all names."""
        module = InferenceModule()
        named_list = list(module.named_modules(prefix="root"))

        assert named_list[0] == ("root", module)

    def test_named_modules_single_child(self) -> None:
        """named_modules() yields hierarchical names for children."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = InferenceModule()

        parent = Parent()
        named_dict = dict(parent.named_modules())

        assert "" in named_dict
        assert "child" in named_dict
        assert named_dict[""] is parent
        assert named_dict["child"] is parent.child

    def test_named_modules_nested_names(self) -> None:
        """named_modules() builds dot-separated hierarchical names."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

        class Middle(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.middle = Middle()

        outer = Outer()
        named_dict = dict(outer.named_modules())

        assert "" in named_dict
        assert "middle" in named_dict
        assert "middle.inner" in named_dict

    def test_named_modules_with_prefix_nested(self) -> None:
        """named_modules(prefix) works correctly with nested structure."""

        class Child(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = Child()

        parent = Parent()
        named_dict = dict(parent.named_modules(prefix="base"))

        assert "base" in named_dict
        assert "base.child" in named_dict


class TestParametersIterator:
    """Tests for parameters() method."""

    def test_parameters_empty_module(self) -> None:
        """parameters() yields nothing for module with no parameters."""
        module = InferenceModule()
        assert list(module.parameters()) == []

    def test_parameters_single_parameter(self) -> None:
        """parameters() yields the single parameter."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("test", description="test")

        module = MyModule()
        params_list = list(module.parameters())

        assert len(params_list) == 1
        assert params_list[0] is module.prompt

    def test_parameters_multiple_parameters(self) -> None:
        """parameters() yields all parameters from this module."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.param1 = Parameter("value1", description="test")
                self.param2 = Parameter("value2", description="test")

        module = MyModule()
        params_list = list(module.parameters())

        assert len(params_list) == 2
        assert module.param1 in params_list
        assert module.param2 in params_list

    def test_parameters_recurses_into_children(self) -> None:
        """parameters() recursively yields parameters from children."""

        class Child(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child_param = Parameter("child value", description="test")

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.parent_param = Parameter("parent value", description="test")
                self.child = Child()

        parent = Parent()
        params_list = list(parent.parameters())

        assert len(params_list) == 2
        assert parent.parent_param in params_list
        assert parent.child.child_param in params_list

    def test_parameters_deep_nesting(self) -> None:
        """parameters() finds parameters in deeply nested modules."""

        class Level3(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.deep_param = Parameter("deep", description="test")

        class Level2(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level3 = Level3()

        class Level1(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level2 = Level2()

        root = Level1()
        params_list = list(root.parameters())

        assert len(params_list) == 1
        assert params_list[0] is root.level2.level3.deep_param

    def test_parameters_excludes_children_modules(self) -> None:
        """parameters() does not yield child modules themselves."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = InferenceModule()
                self.param = Parameter("test", description="test")

        module = MyModule()
        params_list = list(module.parameters())

        assert len(params_list) == 1
        assert all(isinstance(p, Parameter) for p in params_list)


class TestNamedParametersIterator:
    """Tests for named_parameters() method."""

    def test_named_parameters_empty_module(self) -> None:
        """named_parameters() yields nothing for module with no parameters."""
        module = InferenceModule()
        assert list(module.named_parameters()) == []

    def test_named_parameters_single_parameter(self) -> None:
        """named_parameters() yields (name, param) tuple."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("test", description="test")

        module = MyModule()
        named_list = list(module.named_parameters())

        assert len(named_list) == 1
        assert named_list[0] == ("prompt", module.prompt)

    def test_named_parameters_multiple_parameters(self) -> None:
        """named_parameters() yields all (name, param) tuples."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.alpha = Parameter("a", description="test")
                self.beta = Parameter("b", description="test")

        module = MyModule()
        named_dict = dict(module.named_parameters())

        assert len(named_dict) == 2
        assert named_dict["alpha"] is module.alpha
        assert named_dict["beta"] is module.beta

    def test_named_parameters_hierarchical_names(self) -> None:
        """named_parameters() builds dot-separated names for nested params."""

        class Child(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child_param = Parameter("child", description="test")

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.parent_param = Parameter("parent", description="test")
                self.child = Child()

        parent = Parent()
        named_dict = dict(parent.named_parameters())

        assert "parent_param" in named_dict
        assert "child.child_param" in named_dict
        assert named_dict["parent_param"] is parent.parent_param
        assert named_dict["child.child_param"] is parent.child.child_param

    def test_named_parameters_with_prefix(self) -> None:
        """named_parameters(prefix) prepends prefix to all names."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.param = Parameter("test", description="test")

        module = MyModule()
        named_dict = dict(module.named_parameters(prefix="base"))

        assert "base.param" in named_dict

    def test_named_parameters_complex_hierarchy(self) -> None:
        """named_parameters() handles complex module hierarchies."""

        class Leaf(InferenceModule):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.leaf_param = Parameter(name, description="test")

        class Branch(InferenceModule):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.branch_param = Parameter(name, description="test")
                self.left = Leaf(f"{name}_left")
                self.right = Leaf(f"{name}_right")

        class Root(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.root_param = Parameter("root", description="test")
                self.branch = Branch("branch")

        root = Root()
        named_dict = dict(root.named_parameters())

        expected_names = [
            "root_param",
            "branch.branch_param",
            "branch.left.leaf_param",
            "branch.right.leaf_param",
        ]
        assert len(named_dict) == 4
        for name in expected_names:
            assert name in named_dict


class TestIntrospectionIntegration:
    """Integration tests for introspection methods."""

    def test_all_iterators_are_lazy(self) -> None:
        """Verify that iterators are generators (lazy evaluation)."""
        module = InferenceModule()

        # All should return iterators/generators, not lists
        from collections.abc import Iterator

        assert isinstance(module.children(), Iterator)
        assert isinstance(module.named_children(), Iterator)
        assert isinstance(module.modules(), Iterator)
        assert isinstance(module.named_modules(), Iterator)
        assert isinstance(module.parameters(), Iterator)
        assert isinstance(module.named_parameters(), Iterator)

    def test_can_iterate_multiple_times(self) -> None:
        """Verify iterators can be created multiple times."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = InferenceModule()
                self.param = Parameter("test", description="test")

        module = MyModule()

        # First iteration
        children1 = list(module.children())
        params1 = list(module.parameters())

        # Second iteration
        children2 = list(module.children())
        params2 = list(module.parameters())

        assert children1 == children2
        assert params1 == params2

    def test_comprehensive_module_traversal(self) -> None:
        """Test that modules() and parameters() are consistent."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner_param = Parameter("inner", description="test")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.outer_param = Parameter("outer", description="test")
                self.inner = Inner()

        outer = Outer()

        # Count modules
        all_modules = list(outer.modules())
        assert len(all_modules) == 2

        # Count parameters
        all_params = list(outer.parameters())
        assert len(all_params) == 2

        # Verify names match module hierarchy
        named_mods = dict(outer.named_modules())
        named_params = dict(outer.named_parameters())

        assert "" in named_mods  # Root
        assert "inner" in named_mods
        assert "outer_param" in named_params
        assert "inner.inner_param" in named_params


class TestIntrospectionEdgeCases:
    """Edge case tests for introspection methods."""

    def test_shared_module_visited_multiple_times(self) -> None:
        """modules() visits shared module multiple times through different paths.

        When the same module instance is registered under multiple parents,
        it will be yielded once for each path to it in the module tree.
        """
        shared = InferenceModule()

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child1 = InferenceModule()
                self.child2 = InferenceModule()

        parent = Parent()
        # Assign shared module to both children
        parent.child1.shared = shared
        parent.child2.shared = shared

        modules_list = list(parent.modules())

        # parent + child1 + child2 + shared (via child1) + shared (via child2) = 5
        assert len(modules_list) == 5
        # Shared appears twice in the list
        assert modules_list.count(shared) == 2

    def test_shared_module_named_modules_different_paths(self) -> None:
        """named_modules() yields shared module with different hierarchical names."""
        shared = InferenceModule()

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.left = InferenceModule()
                self.right = InferenceModule()

        parent = Parent()
        parent.left.common = shared
        parent.right.common = shared

        named_dict = dict(parent.named_modules())

        # Both paths should exist with different names
        assert "left.common" in named_dict
        assert "right.common" in named_dict
        # Both point to the same shared module
        assert named_dict["left.common"] is shared
        assert named_dict["right.common"] is shared

    def test_shared_parameter_visited_multiple_times(self) -> None:
        """parameters() visits shared parameter multiple times through different paths."""
        shared_param = Parameter("shared value", description="test")

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child1 = InferenceModule()
                self.child2 = InferenceModule()

        parent = Parent()
        parent.child1.param = shared_param
        parent.child2.param = shared_param

        params_list = list(parent.parameters())

        # Shared param appears twice
        assert len(params_list) == 2
        assert params_list.count(shared_param) == 2

    def test_diamond_structure_modules(self) -> None:
        """modules() handles diamond-shaped module graph.

        Structure: root -> left -> bottom
                   root -> right -> bottom (same bottom)
        """
        bottom = InferenceModule()

        class Left(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.bottom = bottom

        class Right(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.bottom = bottom

        class Root(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.left = Left()
                self.right = Right()

        root = Root()
        modules_list = list(root.modules())

        # root + left + bottom (via left) + right + bottom (via right) = 5
        assert len(modules_list) == 5
        assert modules_list.count(bottom) == 2

    def test_stale_children_entry_still_yielded(self) -> None:
        """children() yields stale entries when module is reassigned to non-module.

        When a child module is reassigned to a regular value, the stale
        entry in _children still points to the original module.
        """

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = InferenceModule()

        parent = Parent()
        original_child = parent.child

        # Reassign to a string - stale entry remains
        parent.child = "not a module anymore"

        children_list = list(parent.children())

        # Stale entry is still yielded
        assert len(children_list) == 1
        assert children_list[0] is original_child
        # But the actual attribute is the string
        assert parent.child == "not a module anymore"

    def test_stale_parameters_entry_still_yielded(self) -> None:
        """parameters() yields stale entries when parameter is reassigned."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.param = Parameter("original", description="test")

        module = MyModule()
        original_param = module.param

        # Reassign to a string
        module.param = "not a parameter"

        params_list = list(module.parameters())

        # Stale entry is still yielded
        assert len(params_list) == 1
        assert params_list[0] is original_param

    def test_partial_generator_consumption(self) -> None:
        """Generators work correctly when partially consumed."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child1 = InferenceModule()
                self.child2 = InferenceModule()
                self.child3 = InferenceModule()

        parent = Parent()
        gen = parent.children()

        # Consume first item
        first = next(gen)
        assert isinstance(first, InferenceModule)

        # Consume second item
        second = next(gen)
        assert isinstance(second, InferenceModule)
        assert first is not second

        # Remaining items can still be consumed
        remaining = list(gen)
        assert len(remaining) == 1

    def test_partial_modules_generator_consumption(self) -> None:
        """modules() generator works correctly when partially consumed."""

        class Child(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child1 = Child()
                self.child2 = Child()

        parent = Parent()
        gen = parent.modules()

        # First should be parent (self)
        first = next(gen)
        assert first is parent

        # Second should be child1
        second = next(gen)
        assert second is parent.child1

        # Rest can still be consumed
        remaining = list(gen)
        assert len(remaining) == 1
        assert remaining[0] is parent.child2

    def test_named_modules_no_leading_dot(self) -> None:
        """named_modules() never produces names with leading dots."""

        class Deep(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

        class Middle(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.deep = Deep()

        class Root(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.middle = Middle()

        root = Root()
        names = [name for name, _ in root.named_modules()]

        # No name should start with a dot
        for name in names:
            assert not name.startswith("."), f"Name '{name}' starts with dot"

        # Verify expected names
        assert "" in names  # Root has empty name
        assert "middle" in names
        assert "middle.deep" in names

    def test_named_parameters_no_leading_dot(self) -> None:
        """named_parameters() never produces names with leading dots."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner_param = Parameter("inner", description="test")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.outer_param = Parameter("outer", description="test")
                self.inner = Inner()

        outer = Outer()
        names = [name for name, _ in outer.named_parameters()]

        # No name should start with a dot
        for name in names:
            assert not name.startswith("."), f"Name '{name}' starts with dot"

        # Verify expected names
        assert "outer_param" in names
        assert "inner.inner_param" in names

    def test_named_modules_with_empty_prefix_deeply_nested(self) -> None:
        """named_modules() with empty prefix handles deep nesting correctly."""

        class Level4(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

        class Level3(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.l4 = Level4()

        class Level2(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.l3 = Level3()

        class Level1(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.l2 = Level2()

        root = Level1()
        named_dict = dict(root.named_modules())

        # All names should be properly formed
        assert "" in named_dict
        assert "l2" in named_dict
        assert "l2.l3" in named_dict
        assert "l2.l3.l4" in named_dict

        # No double dots
        for name in named_dict:
            assert ".." not in name, f"Name '{name}' contains double dot"


# ─────────────────────────────────────────────────────────────────────────────
# PR-005: Forward and Call Methods Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestForwardMethod:
    """Tests for forward() method."""

    def test_forward_raises_not_implemented_on_base_class(self) -> None:
        """forward() raises NotImplementedError on base InferenceModule."""
        import pytest

        module = InferenceModule()
        with pytest.raises(NotImplementedError) as exc_info:
            module.forward()

        assert "InferenceModule must implement forward()" in str(exc_info.value)

    def test_forward_error_includes_class_name(self) -> None:
        """forward() error message includes the actual class name."""
        import pytest

        class MyCustomModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

        module = MyCustomModule()
        with pytest.raises(NotImplementedError) as exc_info:
            module.forward()

        assert "MyCustomModule must implement forward()" in str(exc_info.value)

    def test_forward_can_be_overridden(self) -> None:
        """forward() can be overridden in subclasses."""

        class Greeter(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, name: str) -> str:
                return f"Hello, {name}!"

        greeter = Greeter()
        result = greeter.forward("World")

        assert result == "Hello, World!"

    def test_forward_accepts_args_and_kwargs(self) -> None:
        """forward() accepts arbitrary args and kwargs."""

        class Formatter(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, template: str, *args: str, **kwargs: str) -> str:
                return template.format(*args, **kwargs)

        formatter = Formatter()
        result = formatter.forward("{} says {greeting}", "Alice", greeting="hello")

        assert result == "Alice says hello"

    def test_forward_can_return_any_type(self) -> None:
        """forward() can return any type."""

        class DictReturner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: int) -> dict[str, int]:
                return {"value": x, "squared": x * x}

        module = DictReturner()
        result = module.forward(5)

        assert result == {"value": 5, "squared": 25}

    def test_forward_can_return_none(self) -> None:
        """forward() can return None."""

        class NoneReturner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self) -> None:
                pass

        module = NoneReturner()
        result = module.forward()

        assert result is None

    def test_forward_with_variadic_args_only(self) -> None:
        """forward() can accept only *args (no named parameters)."""

        class Summer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, *args: int) -> int:
                return sum(args)

        module = Summer()

        assert module.forward(1, 2, 3, 4, 5) == 15
        assert module.forward(10) == 10
        assert module.forward() == 0  # Empty sum

    def test_forward_with_kwargs_only(self) -> None:
        """forward() can accept only **kwargs (no positional parameters)."""

        class Collector(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, **kwargs: str) -> dict[str, str]:
                return kwargs

        module = Collector()

        assert module.forward(a="1", b="2") == {"a": "1", "b": "2"}
        assert module.forward() == {}  # Empty kwargs
        assert module.forward(single="value") == {"single": "value"}

    def test_forward_with_none_argument(self) -> None:
        """forward() can receive None as an argument."""

        class NoneHandler(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: str | None) -> str:
                return x if x is not None else "default"

        module = NoneHandler()

        assert module.forward(None) == "default"
        assert module.forward("value") == "value"

    def test_forward_with_default_parameter(self) -> None:
        """forward() can have default parameter values."""

        class Greeter(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, name: str = "World", punctuation: str = "!") -> str:
                return f"Hello, {name}{punctuation}"

        module = Greeter()

        # All defaults
        assert module.forward() == "Hello, World!"
        # Override first default
        assert module.forward("Alice") == "Hello, Alice!"
        # Override both defaults
        assert module.forward("Bob", "?") == "Hello, Bob?"
        # Override only second via keyword
        assert module.forward(punctuation="...") == "Hello, World..."

    def test_forward_with_empty_string_argument(self) -> None:
        """forward() handles empty string arguments correctly."""

        class Wrapper(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, text: str) -> str:
                return f"[{text}]"

        module = Wrapper()

        assert module.forward("") == "[]"
        assert module.forward("content") == "[content]"


class TestCallMethod:
    """Tests for __call__() method."""

    def test_call_delegates_to_forward(self) -> None:
        """__call__ delegates to forward()."""

        class Doubler(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: int) -> int:
                return x * 2

        doubler = Doubler()
        result = doubler(5)

        assert result == 10

    def test_call_passes_positional_args(self) -> None:
        """__call__ passes positional args to forward()."""

        class Adder(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a: int, b: int, c: int) -> int:
                return a + b + c

        adder = Adder()
        result = adder(1, 2, 3)

        assert result == 6

    def test_call_passes_keyword_args(self) -> None:
        """__call__ passes keyword args to forward()."""

        class Greeter(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, name: str, greeting: str = "Hello") -> str:
                return f"{greeting}, {name}!"

        greeter = Greeter()
        result = greeter("Alice", greeting="Hi")

        assert result == "Hi, Alice!"

    def test_call_passes_mixed_args_and_kwargs(self) -> None:
        """__call__ passes both positional and keyword args to forward()."""

        class Formatter(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self, template: str, *values: str, prefix: str = "", suffix: str = ""
            ) -> str:
                formatted = template.format(*values)
                return f"{prefix}{formatted}{suffix}"

        formatter = Formatter()
        result = formatter("{} + {}", "A", "B", prefix="[", suffix="]")

        assert result == "[A + B]"

    def test_call_raises_when_forward_not_implemented(self) -> None:
        """__call__ raises NotImplementedError when forward() is not implemented."""
        import pytest

        module = InferenceModule()
        with pytest.raises(NotImplementedError) as exc_info:
            module()

        assert "InferenceModule must implement forward()" in str(exc_info.value)

    def test_call_on_subclass_without_forward_raises(self) -> None:
        """__call__ raises NotImplementedError on subclass without forward()."""
        import pytest

        class EmptyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

        module = EmptyModule()
        with pytest.raises(NotImplementedError) as exc_info:
            module("test")

        assert "EmptyModule must implement forward()" in str(exc_info.value)

    def test_call_with_no_args(self) -> None:
        """__call__ works with no arguments."""

        class ConstantReturner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self) -> str:
                return "constant"

        module = ConstantReturner()
        result = module()

        assert result == "constant"

    def test_call_returns_forward_result(self) -> None:
        """__call__ returns exactly what forward() returns."""

        class ListReturner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, items: list[int]) -> list[int]:
                return [x * 2 for x in items]

        module = ListReturner()
        input_list = [1, 2, 3]
        result = module(input_list)

        assert result == [2, 4, 6]
        assert result is not input_list  # New list created

    def test_call_with_variadic_args_only(self) -> None:
        """__call__ works with forward() that accepts only *args."""

        class Multiplier(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, *args: int) -> int:
                result = 1
                for arg in args:
                    result *= arg
                return result

        module = Multiplier()

        assert module(2, 3, 4) == 24
        assert module(5) == 5
        assert module() == 1  # Empty product

    def test_call_with_kwargs_only(self) -> None:
        """__call__ works with forward() that accepts only **kwargs."""

        class ConfigBuilder(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, **kwargs: str) -> str:
                return ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))

        module = ConfigBuilder()

        assert module(host="localhost", port="8080") == "host=localhost, port=8080"
        assert module() == ""  # Empty config

    def test_call_with_none_argument(self) -> None:
        """__call__ passes None argument to forward() correctly."""

        class OptionalProcessor(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, value: int | None, default: int = 0) -> int:
                return value if value is not None else default

        module = OptionalProcessor()

        assert module(None) == 0
        assert module(None, default=42) == 42
        assert module(10) == 10

    def test_call_uses_default_parameter_values(self) -> None:
        """__call__ uses default values when arguments are omitted."""

        class Formatter(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self, value: str = "default", prefix: str = "[", suffix: str = "]"
            ) -> str:
                return f"{prefix}{value}{suffix}"

        module = Formatter()

        # All defaults
        assert module() == "[default]"
        # Override first
        assert module("custom") == "[custom]"
        # Override via keyword
        assert module(prefix="<", suffix=">") == "<default>"
        # Override all
        assert module("text", "{", "}") == "{text}"

    def test_call_and_forward_produce_identical_results(self) -> None:
        """__call__ and forward() produce identical results for same inputs."""

        class Calculator(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: int, y: int, operation: str = "add") -> int:
                if operation == "add":
                    return x + y
                elif operation == "multiply":
                    return x * y
                else:
                    return x - y

        module = Calculator()

        # Test various input combinations - verify equivalence explicitly
        assert module(1, 2) == module.forward(1, 2)
        assert module(5, 3, "add") == module.forward(5, 3, "add")
        assert module(4, 7, operation="multiply") == module.forward(
            4, 7, operation="multiply"
        )
        assert module(10, 3, "subtract") == module.forward(10, 3, "subtract")

        # Also verify with keyword-only style
        assert module(x=2, y=3) == module.forward(x=2, y=3)
        assert module(x=4, y=5, operation="multiply") == module.forward(
            x=4, y=5, operation="multiply"
        )

    def test_call_with_empty_string_argument(self) -> None:
        """__call__ handles empty string arguments correctly."""

        class Validator(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, text: str) -> bool:
                return len(text) > 0

        module = Validator()

        assert module("") is False
        assert module("content") is True

    def test_call_with_many_arguments(self) -> None:
        """__call__ handles many arguments correctly (stress test)."""

        class ArgCounter(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, *args: int, **kwargs: str) -> dict[str, int]:
                return {
                    "positional_count": len(args),
                    "keyword_count": len(kwargs),
                    "positional_sum": sum(args),
                }

        module = ArgCounter()

        # 10 positional args
        result = module(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        assert result["positional_count"] == 10
        assert result["positional_sum"] == 55

        # 10 keyword args
        kwargs = {f"key{i}": f"val{i}" for i in range(10)}
        result = module(**kwargs)
        assert result["keyword_count"] == 10

        # Mix of both
        result = module(1, 2, 3, a="x", b="y", c="z")
        assert result["positional_count"] == 3
        assert result["keyword_count"] == 3


class TestForwardCallIntegration:
    """Integration tests for forward() and __call__() methods."""

    def test_nested_module_calls(self) -> None:
        """Nested modules can call each other through __call__."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: int) -> int:
                return x + 1

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

            def forward(self, x: int) -> int:
                return self.inner(x) * 2

        outer = Outer()
        result = outer(5)

        # (5 + 1) * 2 = 12
        assert result == 12

    def test_deeply_nested_calls(self) -> None:
        """Deeply nested module calls work correctly."""

        class Level3(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: int) -> int:
                return x + 1

        class Level2(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level3 = Level3()

            def forward(self, x: int) -> int:
                return self.level3(x) * 2

        class Level1(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level2 = Level2()

            def forward(self, x: int) -> int:
                return self.level2(x) + 10

        root = Level1()
        result = root(5)

        # ((5 + 1) * 2) + 10 = 22
        assert result == 22

    def test_parallel_module_calls(self) -> None:
        """Module can call multiple child modules."""

        class Adder(InferenceModule):
            def __init__(self, amount: int) -> None:
                super().__init__()
                self.amount = amount

            def forward(self, x: int) -> int:
                return x + self.amount

        class ParallelProcessor(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.add_one = Adder(1)
                self.add_ten = Adder(10)
                self.add_hundred = Adder(100)

            def forward(self, x: int) -> dict[str, int]:
                return {
                    "plus_one": self.add_one(x),
                    "plus_ten": self.add_ten(x),
                    "plus_hundred": self.add_hundred(x),
                }

        processor = ParallelProcessor()
        result = processor(5)

        assert result == {
            "plus_one": 6,
            "plus_ten": 15,
            "plus_hundred": 105,
        }

    def test_module_with_parameters_in_forward(self) -> None:
        """Module can use parameters in forward()."""

        class Prefixer(InferenceModule):
            def __init__(self, prefix: str) -> None:
                super().__init__()
                self.prefix = Parameter(prefix, description="test")

            def forward(self, text: str) -> str:
                return f"{self.prefix.value}: {text}"

        prefixer = Prefixer("INFO")
        result = prefixer("hello")

        assert result == "INFO: hello"

    def test_sequential_module_composition(self) -> None:
        """Modules can be composed sequentially."""

        class Upper(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, text: str) -> str:
                return text.upper()

        class Reverse(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, text: str) -> str:
                return text[::-1]

        class Pipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.upper = Upper()
                self.reverse = Reverse()

            def forward(self, text: str) -> str:
                return self.reverse(self.upper(text))

        pipeline = Pipeline()
        result = pipeline("hello")

        # "hello" -> "HELLO" -> "OLLEH"
        assert result == "OLLEH"

    def test_forward_can_access_self_attributes(self) -> None:
        """forward() can access self attributes set in __init__."""

        class Multiplier(InferenceModule):
            def __init__(self, factor: int) -> None:
                super().__init__()
                self.factor = factor

            def forward(self, x: int) -> int:
                return x * self.factor

        multiplier = Multiplier(3)
        result = multiplier(7)

        assert result == 21

    def test_call_propagates_exceptions_from_forward(self) -> None:
        """__call__ propagates exceptions raised in forward()."""
        import pytest

        class RaisingModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: int) -> int:
                if x < 0:
                    raise ValueError("x must be non-negative")
                return x

        module = RaisingModule()

        # Positive value works
        assert module(5) == 5

        # Negative value raises
        with pytest.raises(ValueError, match="x must be non-negative"):
            module(-1)


# ─────────────────────────────────────────────────────────────────────────────
# PR-017: Trace Context Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallWithTraceContext:
    """Tests for __call__() behavior with trace context.

    PR-017: Connect InferenceModule.__call__ to Tracer
    """

    def test_call_without_trace_context_calls_forward(self) -> None:
        """Without trace context, __call__ delegates to forward()."""

        class Counter(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0

            def forward(self, x: int) -> int:
                self.call_count += 1
                return x * 2

        module = Counter()
        result = module(5)

        assert result == 10
        assert module.call_count == 1

    def test_call_with_trace_context_returns_value(self) -> None:
        """With trace context, __call__ returns a Value."""
        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.tracer import Tracer
        from inf_engine.values import Value

        class Doubler(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: int) -> int:
                return x * 2

        module = Doubler()
        tracer = Tracer()

        with trace_context(tracer):
            result = module(5)

        assert isinstance(result, Value)

    def test_call_with_trace_context_does_not_call_forward(self) -> None:
        """With trace context, forward() is not called."""
        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.tracer import Tracer

        class Counter(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0

            def forward(self, x: int) -> int:
                self.call_count += 1
                return x * 2

        module = Counter()
        tracer = Tracer()

        with trace_context(tracer):
            module(5)

        # forward() should not have been called
        assert module.call_count == 0

    def test_call_with_trace_context_records_node(self) -> None:
        """With trace context, __call__ records a node in the tracer."""
        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.tracer import Tracer

        class Doubler(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: int) -> int:
                return x * 2

        module = Doubler()
        tracer = Tracer()

        with trace_context(tracer):
            module(5)

        # Should have recorded one node
        assert len(tracer.nodes) == 1
        node = list(tracer.nodes.values())[0]
        assert node.module is module
        assert "Doubler" in node.id

    def test_call_with_proxy_input_creates_dependency(self) -> None:
        """When called with a Proxy input, creates a dependency edge."""
        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.tracer import Tracer

        class Processor(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: str) -> str:
                return x.upper()

        module = Processor()
        tracer = Tracer()

        with trace_context(tracer):
            # Create an input proxy
            input_proxy = tracer._create_input_node("text", "hello")
            # Call module with proxy
            output = module(input_proxy)

        # Should have 2 nodes: input and the module call
        assert len(tracer.nodes) == 2

        # The module call node should depend on the input
        assert output.ref is not None
        module_node = tracer.nodes[output.ref]
        assert input_proxy.node_id in module_node.dependencies

    def test_call_with_multiple_proxy_inputs(self) -> None:
        """Multiple Proxy inputs all create dependencies."""
        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.tracer import Tracer

        class Combiner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a: str, b: str) -> str:
                return f"{a} {b}"

        module = Combiner()
        tracer = Tracer()

        with trace_context(tracer):
            proxy_a = tracer._create_input_node("a", "hello")
            proxy_b = tracer._create_input_node("b", "world")
            output = module(proxy_a, proxy_b)

        # The module node should depend on both inputs
        assert output.ref is not None
        module_node = tracer.nodes[output.ref]
        assert proxy_a.node_id in module_node.dependencies
        assert proxy_b.node_id in module_node.dependencies

    def test_call_with_mixed_proxy_and_literal_args(self) -> None:
        """Mix of Proxy and literal arguments is handled correctly."""
        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.tracer import Tracer

        class Formatter(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, template: str, value: str) -> str:
                return template.format(value)

        module = Formatter()
        tracer = Tracer()

        with trace_context(tracer):
            proxy = tracer._create_input_node("value", "world")
            # First arg is literal, second is proxy
            output = module("Hello, {}!", proxy)

        assert output.ref is not None
        module_node = tracer.nodes[output.ref]
        # Should only depend on the proxy, not the literal
        assert len(module_node.dependencies) == 1
        assert proxy.node_id in module_node.dependencies
        # The literal should be stored in args
        assert "Hello, {}!" in module_node.args

    def test_call_with_kwarg_proxy_creates_dependency(self) -> None:
        """Proxy passed as keyword argument creates dependency."""
        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.tracer import Tracer

        class Greeter(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, name: str, greeting: str = "Hello") -> str:
                return f"{greeting}, {name}!"

        module = Greeter()
        tracer = Tracer()

        with trace_context(tracer):
            proxy = tracer._create_input_node("name", "Alice")
            output = module(name=proxy, greeting="Hi")

        assert output.ref is not None
        module_node = tracer.nodes[output.ref]
        assert proxy.node_id in module_node.dependencies
        assert module_node.kwargs["greeting"] == "Hi"

    def test_sequential_calls_create_chain(self) -> None:
        """Sequential module calls create a dependency chain."""
        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.tracer import Tracer

        class Step1(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: str) -> str:
                return f"Step1({x})"

        class Step2(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: str) -> str:
                return f"Step2({x})"

        step1 = Step1()
        step2 = Step2()
        tracer = Tracer()

        with trace_context(tracer):
            input_proxy = tracer._create_input_node("input", "data")
            intermediate = step1(input_proxy)
            final = step2(intermediate)

        # 3 nodes: input, step1, step2
        assert len(tracer.nodes) == 3

        # step1 depends on input
        assert intermediate.ref is not None
        step1_node = tracer.nodes[intermediate.ref]
        assert input_proxy.node_id in step1_node.dependencies

        # step2 depends on step1 (via Value.ref)
        assert final.ref is not None
        step2_node = tracer.nodes[final.ref]
        assert intermediate.ref in step2_node.dependencies

    def test_parallel_calls_from_same_input(self) -> None:
        """Parallel module calls from same input create fan-out."""
        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.tracer import Tracer

        class ModuleA(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: str) -> str:
                return f"A({x})"

        class ModuleB(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: str) -> str:
                return f"B({x})"

        mod_a = ModuleA()
        mod_b = ModuleB()
        tracer = Tracer()

        with trace_context(tracer):
            input_proxy = tracer._create_input_node("input", "data")
            output_a = mod_a(input_proxy)
            output_b = mod_b(input_proxy)

        # 3 nodes: input, module_a, module_b
        assert len(tracer.nodes) == 3

        # Both outputs depend on the same input
        assert output_a.ref is not None
        assert output_b.ref is not None
        node_a = tracer.nodes[output_a.ref]
        node_b = tracer.nodes[output_b.ref]
        assert input_proxy.node_id in node_a.dependencies
        assert input_proxy.node_id in node_b.dependencies

        # They don't depend on each other
        assert output_b.ref not in node_a.dependencies
        assert output_a.ref not in node_b.dependencies

    def test_nested_module_calls_during_tracing(self) -> None:
        """Nested module calls are all recorded during tracing."""
        from typing import Any

        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.tracer import Tracer

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Any) -> Any:
                return f"Inner({x})"

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

            def forward(self, x: Any) -> Any:
                # During tracing, this call is also recorded
                intermediate = self.inner(x)
                return f"Outer({intermediate})"

        outer = Outer()
        tracer = Tracer()

        with trace_context(tracer):
            input_proxy = tracer._create_input_node("input", "data")
            # Calling outer.forward directly (as trace() does)
            outer.forward(input_proxy)

        # Should record: input, inner call (from within outer.forward)
        assert len(tracer.nodes) == 2
        # The inner module call was recorded
        assert any("Inner" in node_id for node_id in tracer.nodes)

    def test_llm_inference_with_trace_context(self) -> None:
        """LLMInference works correctly with trace context."""
        from inf_engine.module import LLMInference
        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.tracer import Tracer
        from inf_engine.values import Value

        llm = LLMInference(alias="test_llm", system_prompt="Be helpful.")
        tracer = Tracer()

        with trace_context(tracer):
            input_proxy = tracer._create_input_node("prompt", "Hello!")
            output = llm(input_proxy)

        # Should return a Value
        assert isinstance(output, Value)

        # Should record the LLM call
        assert len(tracer.nodes) == 2  # input + llm call
        assert output.ref is not None
        llm_node = tracer.nodes[output.ref]
        assert llm_node.module is llm
        assert input_proxy.node_id in llm_node.dependencies

    def test_context_cleared_after_block(self) -> None:
        """Trace context is cleared after exiting the block."""
        from inf_engine.tracing.context import get_trace_context, trace_context
        from inf_engine.tracing.tracer import Tracer

        class Doubler(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0

            def forward(self, x: int) -> int:
                self.call_count += 1
                return x * 2

        module = Doubler()
        tracer = Tracer()

        # Inside context - returns proxy
        with trace_context(tracer):
            assert get_trace_context() is tracer

        # Outside context - no trace context
        assert get_trace_context() is None

        # Now call should delegate to forward
        result = module(5)
        assert result == 10
        assert module.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# State Dict Tests (PR-035)
# ─────────────────────────────────────────────────────────────────────────────


class TestStateDict:
    """Tests for InferenceModule.state_dict() serialization."""

    def test_state_dict_empty_module(self) -> None:
        """state_dict returns empty dict for module with no parameters."""
        module = InferenceModule()
        state = module.state_dict()
        assert state == {}

    def test_state_dict_single_parameter(self) -> None:
        """state_dict returns single parameter value."""

        class SingleParam(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("hello", description="test")

        module = SingleParam()
        state = module.state_dict()
        assert state == {"prompt": "hello"}

    def test_state_dict_multiple_parameters(self) -> None:
        """state_dict returns all parameters."""

        class MultiParam(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.system = Parameter("You are helpful.", description="test")
                self.template = Parameter("Answer: {}", description="test")

        module = MultiParam()
        state = module.state_dict()
        assert state == {"system": "You are helpful.", "template": "Answer: {}"}

    def test_state_dict_nested_parameters(self) -> None:
        """state_dict returns hierarchical names for nested parameters."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.weight = Parameter("w", description="test")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.bias = Parameter("b", description="test")
                self.inner = Inner()

        module = Outer()
        state = module.state_dict()
        assert state == {"bias": "b", "inner.weight": "w"}

    def test_state_dict_deeply_nested(self) -> None:
        """state_dict handles deeply nested module hierarchies."""

        class Level3(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.deep = Parameter("level3", description="test")

        class Level2(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level3 = Level3()

        class Level1(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level2 = Level2()

        module = Level1()
        state = module.state_dict()
        assert state == {"level2.level3.deep": "level3"}

    def test_state_dict_preserves_order(self) -> None:
        """state_dict preserves parameter order (depth-first)."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.c = Parameter("c", description="test")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = Parameter("a", description="test")
                self.inner = Inner()
                self.b = Parameter("b", description="test")

        module = Outer()
        state = module.state_dict()
        # Keys should be in depth-first order
        assert list(state.keys()) == ["a", "b", "inner.c"]


class TestLoadStateDict:
    """Tests for InferenceModule.load_state_dict() deserialization."""

    def test_load_state_dict_single_parameter(self) -> None:
        """load_state_dict updates single parameter value."""

        class SingleParam(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("original", description="test")

        module = SingleParam()
        module.load_state_dict({"prompt": "updated"})
        assert module.prompt.value == "updated"

    def test_load_state_dict_multiple_parameters(self) -> None:
        """load_state_dict updates multiple parameters."""

        class MultiParam(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = Parameter("a_original", description="test")
                self.b = Parameter("b_original", description="test")

        module = MultiParam()
        module.load_state_dict({"a": "a_updated", "b": "b_updated"})
        assert module.a.value == "a_updated"
        assert module.b.value == "b_updated"

    def test_load_state_dict_nested_parameters(self) -> None:
        """load_state_dict updates nested parameters correctly."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.weight = Parameter("original_weight", description="test")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.bias = Parameter("original_bias", description="test")
                self.inner = Inner()

        module = Outer()
        module.load_state_dict({"bias": "new_bias", "inner.weight": "new_weight"})
        assert module.bias.value == "new_bias"
        assert module.inner.weight.value == "new_weight"

    def test_load_state_dict_partial_load(self) -> None:
        """load_state_dict allows partial loads (missing keys ignored)."""

        class MultiParam(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = Parameter("a_original", description="test")
                self.b = Parameter("b_original", description="test")

        module = MultiParam()
        module.load_state_dict({"a": "a_updated"})  # Only update 'a'
        assert module.a.value == "a_updated"
        assert module.b.value == "b_original"  # Unchanged

    def test_load_state_dict_unknown_key_raises(self) -> None:
        """load_state_dict raises KeyError for unknown parameter names."""
        import pytest

        class SingleParam(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("original", description="test")

        module = SingleParam()
        with pytest.raises(KeyError) as exc_info:
            module.load_state_dict({"unknown": "value"})
        assert "Unknown parameter: unknown" in str(exc_info.value)

    def test_load_state_dict_unknown_nested_key_raises(self) -> None:
        """load_state_dict raises KeyError for unknown nested parameter."""
        import pytest

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.weight = Parameter("w", description="test")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

        module = Outer()
        with pytest.raises(KeyError) as exc_info:
            module.load_state_dict({"inner.unknown": "value"})
        assert "Unknown parameter: inner.unknown" in str(exc_info.value)

    def test_load_state_dict_empty_dict(self) -> None:
        """load_state_dict with empty dict does nothing."""

        class SingleParam(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("original", description="test")

        module = SingleParam()
        module.load_state_dict({})
        assert module.prompt.value == "original"

    def test_load_state_dict_on_empty_module(self) -> None:
        """load_state_dict on module with no parameters fails for any key."""
        import pytest

        module = InferenceModule()
        with pytest.raises(KeyError):
            module.load_state_dict({"anything": "value"})


class TestStateDictRoundTrip:
    """Tests for state_dict/load_state_dict round-trip serialization."""

    def test_round_trip_single_parameter(self) -> None:
        """state_dict -> load_state_dict preserves single parameter."""

        class SingleParam(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("hello world", description="test")

        module1 = SingleParam()
        state = module1.state_dict()

        module2 = SingleParam()
        module2.load_state_dict(state)

        assert module2.prompt.value == module1.prompt.value

    def test_round_trip_nested_parameters(self) -> None:
        """state_dict -> load_state_dict preserves nested parameters."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.weight = Parameter("inner_value", description="test")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.bias = Parameter("outer_value", description="test")
                self.inner = Inner()

        module1 = Outer()
        state = module1.state_dict()

        module2 = Outer()
        module2.load_state_dict(state)

        assert module2.bias.value == module1.bias.value
        assert module2.inner.weight.value == module1.inner.weight.value

    def test_round_trip_modified_values(self) -> None:
        """Round-trip preserves modified parameter values."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("default", description="test")

        module1 = MyModule()
        module1.prompt.value = "custom value after modification"
        state = module1.state_dict()

        module2 = MyModule()
        assert module2.prompt.value == "default"  # Still default
        module2.load_state_dict(state)
        assert module2.prompt.value == "custom value after modification"

    def test_round_trip_empty_module(self) -> None:
        """Round-trip works for module with no parameters."""
        module1 = InferenceModule()
        state = module1.state_dict()

        module2 = InferenceModule()
        module2.load_state_dict(state)

        assert state == {}

    def test_round_trip_complex_hierarchy(self) -> None:
        """Round-trip preserves complex nested hierarchies."""

        class Encoder(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.embed = Parameter("encoder_embed", description="test")

        class Decoder(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.output = Parameter("decoder_output", description="test")

        class Transformer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = Encoder()
                self.decoder = Decoder()
                self.norm = Parameter("layer_norm", description="test")

        module1 = Transformer()
        module1.encoder.embed.value = "modified_embed"
        module1.decoder.output.value = "modified_output"
        module1.norm.value = "modified_norm"

        state = module1.state_dict()

        module2 = Transformer()
        module2.load_state_dict(state)

        assert module2.encoder.embed.value == "modified_embed"
        assert module2.decoder.output.value == "modified_output"
        assert module2.norm.value == "modified_norm"

    def test_state_dict_json_serializable(self) -> None:
        """state_dict result can be serialized to JSON and back."""
        import json

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("test value", description="test")
                self.template = Parameter("template: {}", description="test")

        module1 = MyModule()
        state = module1.state_dict()

        # Serialize to JSON and back
        json_str = json.dumps(state)
        loaded_state = json.loads(json_str)

        module2 = MyModule()
        module2.load_state_dict(loaded_state)

        assert module2.prompt.value == module1.prompt.value
        assert module2.template.value == module1.template.value


# ─────────────────────────────────────────────────────────────
# Backward Pass Tests (PR-067)
# ─────────────────────────────────────────────────────────────


class TestInferenceModuleBackward:
    """Tests for InferenceModule.backward() default implementation."""

    @staticmethod
    def _make_context(inputs: dict) -> "BackwardContext":
        """Create a BackwardContext for testing."""
        from inf_engine.graph import InferenceGraph
        from inf_engine.optimization.backward import BackwardContext
        from inf_engine.optimization.feedback import Feedback

        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        return BackwardContext(
            node_id="test_node",
            inputs=inputs,
            output="test output",
            graph=graph,
            all_results={},
            downstream_feedback=[Feedback(content="test")],
        )

    @staticmethod
    async def _run_backward(module: InferenceModule, feedback, ctx) -> "BackwardResult":
        """Run backward on a module."""
        return await module.backward(feedback, ctx)

    def test_backward_is_async(self) -> None:
        """backward() is an async method."""
        import inspect

        module = InferenceModule()
        assert inspect.iscoroutinefunction(module.backward)

    def test_backward_default_passes_feedback_to_inputs(self) -> None:
        """Default backward() passes feedback unchanged to all inputs."""
        import asyncio

        from inf_engine.optimization.backward import BackwardResult
        from inf_engine.optimization.feedback import Feedback

        class TestModule(InferenceModule):
            def forward(self, x: str) -> str:
                return x.upper()

        module = TestModule()
        feedback = Feedback(content="Good output", score=0.8)
        ctx = self._make_context({"arg_0": "hello", "key1": "value1"})

        result = asyncio.run(self._run_backward(module, feedback, ctx))

        assert isinstance(result, BackwardResult)
        # Should have feedback for each input
        assert "arg_0" in result.input_feedback
        assert "key1" in result.input_feedback
        # Feedback should be the same object passed in
        assert result.input_feedback["arg_0"] is feedback
        assert result.input_feedback["key1"] is feedback

    def test_backward_default_no_parameter_feedback(self) -> None:
        """Default backward() produces no parameter feedback."""
        import asyncio

        from inf_engine.optimization.feedback import Feedback

        module = InferenceModule()
        feedback = Feedback(content="Test")
        ctx = self._make_context({"input": "value"})

        # Can't call backward directly on base class, need subclass
        class TestModule(InferenceModule):
            def forward(self, x: str) -> str:
                return x

        module = TestModule()
        result = asyncio.run(self._run_backward(module, feedback, ctx))

        assert result.parameter_feedback == {}

    def test_backward_empty_inputs(self) -> None:
        """Default backward() handles empty inputs dict."""
        import asyncio

        from inf_engine.optimization.feedback import Feedback

        class TestModule(InferenceModule):
            def forward(self) -> str:
                return "result"

        module = TestModule()
        feedback = Feedback(content="Test")
        ctx = self._make_context({})

        result = asyncio.run(self._run_backward(module, feedback, ctx))

        assert result.input_feedback == {}

    def test_backward_preserves_feedback_properties(self) -> None:
        """Default backward() preserves all feedback properties."""
        import asyncio

        from inf_engine.optimization.feedback import Feedback, FeedbackType

        class TestModule(InferenceModule):
            def forward(self, x: str) -> str:
                return x

        module = TestModule()
        feedback = Feedback(
            content="Detailed feedback",
            score=0.75,
            feedback_type=FeedbackType.LLM_JUDGE,
            metadata={"key": "value"},
        )
        ctx = self._make_context({"input": "test"})

        result = asyncio.run(self._run_backward(module, feedback, ctx))

        passed_feedback = result.input_feedback["input"]
        assert passed_feedback.content == "Detailed feedback"
        assert passed_feedback.score == 0.75
        assert passed_feedback.feedback_type == FeedbackType.LLM_JUDGE


# Import for type hints
if True:  # Avoid circular import at runtime
    from inf_engine.optimization.backward import BackwardContext, BackwardResult
