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
                self.prompt = Parameter("test prompt")

        module = MyModule()
        assert "prompt" in module._parameters
        assert module._parameters["prompt"] is module.prompt

    def test_parameter_name_set(self) -> None:
        """Parameter's _name is set to the attribute name."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_param = Parameter("test")

        module = MyModule()
        assert module.my_param._name == "my_param"

    def test_multiple_parameters_registered(self) -> None:
        """Multiple parameters are all registered."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.param1 = Parameter("value1")
                self.param2 = Parameter("value2")
                self.param3 = Parameter("value3")

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
                self.frozen = Parameter("frozen", requires_grad=False)

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
                self.param = Parameter("test")

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
                self.system_prompt = Parameter("You are helpful.")
                self.temperature = 0.7  # Regular attribute
                self.sub_module = InferenceModule()
                self.response_format = Parameter("json")

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
                self.thing = Parameter("initial")

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
        module.thing = Parameter("now a param")

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
        shared_param = Parameter("shared value")

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
                self.empty = Parameter("")

        module = MyModule()
        assert "empty" in module._parameters
        assert module.empty.value == ""

    def test_attribute_accessible_after_registration(self) -> None:
        """Verify attributes are accessible after registration (not just registered)."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = InferenceModule()
                self.param = Parameter("test")
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
                self.param = Parameter("test")

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
                self.prompt = Parameter("test")

        module = MyModule()
        params_list = list(module.parameters())

        assert len(params_list) == 1
        assert params_list[0] is module.prompt

    def test_parameters_multiple_parameters(self) -> None:
        """parameters() yields all parameters from this module."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.param1 = Parameter("value1")
                self.param2 = Parameter("value2")

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
                self.child_param = Parameter("child value")

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.parent_param = Parameter("parent value")
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
                self.deep_param = Parameter("deep")

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
                self.param = Parameter("test")

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
                self.prompt = Parameter("test")

        module = MyModule()
        named_list = list(module.named_parameters())

        assert len(named_list) == 1
        assert named_list[0] == ("prompt", module.prompt)

    def test_named_parameters_multiple_parameters(self) -> None:
        """named_parameters() yields all (name, param) tuples."""

        class MyModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.alpha = Parameter("a")
                self.beta = Parameter("b")

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
                self.child_param = Parameter("child")

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.parent_param = Parameter("parent")
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
                self.param = Parameter("test")

        module = MyModule()
        named_dict = dict(module.named_parameters(prefix="base"))

        assert "base.param" in named_dict

    def test_named_parameters_complex_hierarchy(self) -> None:
        """named_parameters() handles complex module hierarchies."""

        class Leaf(InferenceModule):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.leaf_param = Parameter(name)

        class Branch(InferenceModule):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.branch_param = Parameter(name)
                self.left = Leaf(f"{name}_left")
                self.right = Leaf(f"{name}_right")

        class Root(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.root_param = Parameter("root")
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
                self.param = Parameter("test")

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
                self.inner_param = Parameter("inner")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.outer_param = Parameter("outer")
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
        shared_param = Parameter("shared value")

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
                self.param = Parameter("original")

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
                self.inner_param = Parameter("inner")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.outer_param = Parameter("outer")
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
