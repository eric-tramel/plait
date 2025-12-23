"""Unit tests for the InferenceModule base class.

This file contains tests for PR-003 (core structure) covering:
- Basic instantiation
- Child module registration
- Parameter registration
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
