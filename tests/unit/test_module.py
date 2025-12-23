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
