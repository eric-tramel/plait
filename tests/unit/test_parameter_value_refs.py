"""Unit tests for Parameter value refs and structured kind inference.

Tests the stable ref format (param:<name>) and ValueKind inference
when lifting Parameters to Values via valueify().
"""

from plait.module import InferenceModule
from plait.parameter import Parameter
from plait.values import (
    ValueKind,
    ValueRef,
    collect_refs,
    replace_values_with_refs,
    valueify,
)


class TestParameterRefFormat:
    """Tests for stable param:name ref format."""

    def test_param_ref_prefix(self) -> None:
        """Parameter refs always start with 'param:' prefix."""
        param = Parameter("value", description="Test")
        v = valueify(param)
        assert v.ref.startswith("param:")

    def test_param_ref_with_name_from_module(self) -> None:
        """Parameters assigned to modules use param:<attr_name> format."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.system_prompt = Parameter("value", description="Test")

        module = TestModule()
        v = valueify(module.system_prompt)
        assert v.ref == "param:system_prompt"

    def test_param_ref_without_name_uses_id(self) -> None:
        """Unnamed parameters use param:<id> format."""
        param = Parameter("value", description="Test")
        v = valueify(param)
        assert v.ref == f"param:{param._id}"

    def test_param_ref_simple_attribute_name(self) -> None:
        """Parameter _name is set to attribute name when assigned to module."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("value", description="Test")

        module = TestModule()
        v = valueify(module.prompt)
        assert v.ref == "param:prompt"

    def test_param_ref_nested_module_uses_hierarchical_path(self) -> None:
        """Nested module parameters use hierarchical path in refs.

        When valueify() is called on a parameter owned by a nested module,
        the ref includes the full hierarchical path from the root module.
        """

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.weight = Parameter("w", description="Inner weight")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

        outer = Outer()
        # The ref uses the hierarchical path "inner.weight"
        v = valueify(outer.inner.weight)
        assert v.ref == "param:inner.weight"
        # The _name field still stores immediate attr name
        assert outer.inner.weight._name == "weight"

    def test_param_ref_with_underscores(self) -> None:
        """Parameter names with underscores are preserved."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_prompt = Parameter("value", description="Test")

        module = TestModule()
        v = valueify(module.my_prompt)
        assert v.ref == "param:my_prompt"

    def test_param_ref_numeric_suffix(self) -> None:
        """Parameter names with numeric suffixes are preserved."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt1 = Parameter("value", description="Test")

        module = TestModule()
        v = valueify(module.prompt1)
        assert v.ref == "param:prompt1"


class TestParameterStructuredKindInference:
    """Tests for ValueKind inference from Parameter values."""

    def test_string_value_infers_text(self) -> None:
        """String parameter values infer TEXT kind."""
        param = Parameter("hello world", description="Test")
        v = valueify(param)
        assert v.kind == ValueKind.TEXT

    def test_empty_string_infers_text(self) -> None:
        """Empty string parameter values infer TEXT kind."""
        param = Parameter("", description="Test")
        v = valueify(param)
        assert v.kind == ValueKind.TEXT

    def test_dict_value_infers_structured(self) -> None:
        """Dict parameter values infer STRUCTURED kind."""
        param = Parameter({"key": "value"}, description="Config")
        v = valueify(param)
        assert v.kind == ValueKind.STRUCTURED

    def test_empty_dict_infers_structured(self) -> None:
        """Empty dict parameter values infer STRUCTURED kind."""
        param = Parameter({}, description="Empty config")
        v = valueify(param)
        assert v.kind == ValueKind.STRUCTURED

    def test_list_value_infers_structured(self) -> None:
        """List parameter values infer STRUCTURED kind."""
        param = Parameter(["a", "b", "c"], description="Items")
        v = valueify(param)
        assert v.kind == ValueKind.STRUCTURED

    def test_empty_list_infers_structured(self) -> None:
        """Empty list parameter values infer STRUCTURED kind."""
        param = Parameter([], description="Empty list")
        v = valueify(param)
        assert v.kind == ValueKind.STRUCTURED

    def test_tuple_value_infers_structured(self) -> None:
        """Tuple parameter values infer STRUCTURED kind."""
        param = Parameter((1, 2, 3), description="Tuple")
        v = valueify(param)
        assert v.kind == ValueKind.STRUCTURED

    def test_nested_dict_infers_structured(self) -> None:
        """Nested dict parameter values infer STRUCTURED kind."""
        param = Parameter(
            {"outer": {"inner": "value"}},
            description="Nested config",
        )
        v = valueify(param)
        assert v.kind == ValueKind.STRUCTURED

    def test_list_of_dicts_infers_structured(self) -> None:
        """List of dicts parameter values infer STRUCTURED kind."""
        param = Parameter(
            [{"id": 1}, {"id": 2}],
            description="Items list",
        )
        v = valueify(param)
        assert v.kind == ValueKind.STRUCTURED

    def test_int_value_infers_int(self) -> None:
        """Integer parameter values infer INT kind."""
        param = Parameter(42, description="Count")
        v = valueify(param)
        assert v.kind == ValueKind.INT

    def test_float_value_infers_float(self) -> None:
        """Float parameter values infer FLOAT kind."""
        param = Parameter(3.14, description="Rate")
        v = valueify(param)
        assert v.kind == ValueKind.FLOAT

    def test_bytes_value_infers_binary(self) -> None:
        """Bytes parameter values infer BINARY kind."""
        param = Parameter(b"binary data", description="Binary")
        v = valueify(param)
        assert v.kind == ValueKind.BINARY


class TestParameterKindOverride:
    """Tests for explicit kind override when valueifying Parameters."""

    def test_override_text_to_fstring(self) -> None:
        """Can override TEXT inference to FSTRING."""
        param = Parameter("Hello {name}", description="Template")
        v = valueify(param, kind=ValueKind.FSTRING)
        assert v.kind == ValueKind.FSTRING
        assert v.payload == "Hello {name}"

    def test_override_preserves_ref(self) -> None:
        """Kind override preserves the parameter ref."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_param = Parameter("value", description="Test")

        module = TestModule()
        v = valueify(module.my_param, kind=ValueKind.OTHER)
        assert v.ref == "param:my_param"

    def test_override_preserves_metadata(self) -> None:
        """Kind override preserves all parameter metadata."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_param = Parameter(
                    "value", description="Test", requires_grad=True
                )

        module = TestModule()
        v = valueify(module.my_param, kind=ValueKind.OTHER)
        assert v.meta["param_name"] == "my_param"
        assert v.meta["param_id"] == module.my_param._id
        assert v.meta["requires_grad"] is True


class TestParameterValueRefInteraction:
    """Tests for Parameter-derived Values with ValueRef."""

    def test_parameter_value_in_collect_refs(self) -> None:
        """Parameter-derived Values work with collect_refs."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_param = Parameter("value", description="Test")

        module = TestModule()
        v = valueify(module.my_param)
        refs = collect_refs(v)
        assert refs == ["param:my_param"]

    def test_parameter_value_replace_with_ref(self) -> None:
        """Parameter-derived Values can be replaced with ValueRef."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_param = Parameter("value", description="Test")

        module = TestModule()
        v = valueify(module.my_param)
        ref = replace_values_with_refs(v)
        assert isinstance(ref, ValueRef)
        assert ref.ref == "param:my_param"

    def test_parameter_values_in_nested_structure(self) -> None:
        """Parameter Values work in nested structures."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.p1 = Parameter("first", description="First")
                self.p2 = Parameter("second", description="Second")

        module = TestModule()
        v1 = valueify(module.p1)
        v2 = valueify(module.p2)
        nested = {"a": v1, "b": [v2]}

        refs = collect_refs(nested)
        assert set(refs) == {"param:p1", "param:p2"}

    def test_parameter_value_ref_in_replace(self) -> None:
        """Nested parameter Values are replaced with ValueRefs."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.nested_param = Parameter("value", description="Test")

        module = TestModule()
        v = valueify(module.nested_param)
        structure = [v, {"key": v}]

        result = replace_values_with_refs(structure)
        assert isinstance(result[0], ValueRef)
        assert isinstance(result[1]["key"], ValueRef)
        assert result[0].ref == "param:nested_param"


class TestConstantParameterRefs:
    """Tests for Parameters with requires_grad=False."""

    def test_constant_parameter_has_ref_when_owned(self) -> None:
        """Constant parameters get refs from module ownership."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.config = Parameter({"model": "gpt-4"}, requires_grad=False)

        module = TestModule()
        v = valueify(module.config)
        assert v.ref == "param:config"

    def test_constant_parameter_has_ref_when_unowned(self) -> None:
        """Unowned constant parameters use id-based refs."""
        param = Parameter({"model": "gpt-4"}, requires_grad=False)
        v = valueify(param)
        assert v.ref == f"param:{param._id}"

    def test_constant_parameter_meta_requires_grad_false(self) -> None:
        """Constant parameters have requires_grad=False in meta."""
        param = Parameter("constant", requires_grad=False)
        v = valueify(param)
        assert v.meta["requires_grad"] is False

    def test_constant_parameter_structured_kind(self) -> None:
        """Constant structured parameters infer STRUCTURED kind."""
        param = Parameter({"key": "value"}, requires_grad=False)
        v = valueify(param)
        assert v.kind == ValueKind.STRUCTURED


class TestParameterModuleStateVersion:
    """Tests for module_state_version tracking in Parameter Values.

    The module_state_version tracks the version of the owning module's state.
    This is tracked at the module level, not the individual parameter level.
    """

    def test_initial_version_is_zero(self) -> None:
        """New parameters start with module_state_version=0."""
        param = Parameter("value", description="Test")
        v = valueify(param)
        assert v.meta["module_state_version"] == 0

    def test_owned_parameter_initial_version_is_zero(self) -> None:
        """Parameters owned by modules start at the module's version (0)."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("value", description="Test")

        module = TestModule()
        v = valueify(module.prompt)
        assert v.meta["module_state_version"] == 0

    def test_module_version_shared_across_parameters(self) -> None:
        """All parameters in a module share the same module_state_version."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt1 = Parameter("v1", description="First")
                self.prompt2 = Parameter("v2", description="Second")

        module = TestModule()
        v1 = valueify(module.prompt1)
        v2 = valueify(module.prompt2)
        assert v1.meta["module_state_version"] == v2.meta["module_state_version"]

    def test_module_version_increments_on_any_param_update(self) -> None:
        """Module version increments when any parameter is updated."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt1 = Parameter("v1", description="First")
                self.prompt2 = Parameter("v2", description="Second")

        module = TestModule()
        # Update first parameter
        module.prompt1.apply_update("new1")

        # Both parameters should reflect the module's incremented version
        v1 = valueify(module.prompt1)
        v2 = valueify(module.prompt2)
        assert v1.meta["module_state_version"] == 1
        assert v2.meta["module_state_version"] == 1

    def test_module_version_tracks_multiple_updates(self) -> None:
        """Module version increments with each parameter update."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("v0", description="Test")

        module = TestModule()
        module.prompt.apply_update("v1")
        module.prompt.apply_update("v2")
        module.prompt.apply_update("v3")
        v = valueify(module.prompt)
        assert v.meta["module_state_version"] == 3

    def test_nested_module_has_own_version(self) -> None:
        """Nested modules maintain their own module_state_version."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.weight = Parameter("w", description="Inner weight")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.bias = Parameter("b", description="Outer bias")
                self.inner = Inner()

        outer = Outer()
        # Update only inner's parameter
        outer.inner.weight.apply_update("new_w")

        # Inner's version should increment, outer's should not
        inner_v = valueify(outer.inner.weight)
        outer_v = valueify(outer.bias)
        assert inner_v.meta["module_state_version"] == 1
        assert outer_v.meta["module_state_version"] == 0


class TestParameterOwnership:
    """Tests for Parameter ownership/parenting when assigned to modules."""

    def test_parameter_name_set_on_assignment(self) -> None:
        """Parameter._name is set when assigned to a module."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("value", description="Test")

        module = TestModule()
        assert module.prompt._name == "prompt"

    def test_parameter_registered_in_module(self) -> None:
        """Parameter is registered in module's _parameters dict."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_param = Parameter("value", description="Test")

        module = TestModule()
        assert "my_param" in module._parameters
        assert module._parameters["my_param"] is module.my_param

    def test_nested_module_parameter_registration(self) -> None:
        """Parameters in nested modules are registered in their parent."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.weight = Parameter("w", description="Inner weight")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.bias = Parameter("b", description="Outer bias")
                self.inner = Inner()

        outer = Outer()
        # Outer's own parameter
        assert "bias" in outer._parameters
        # Inner's parameter is in Inner's _parameters, not Outer's
        assert "weight" not in outer._parameters
        assert "weight" in outer.inner._parameters

    def test_named_parameters_produces_hierarchical_paths(self) -> None:
        """named_parameters() produces hierarchical paths for nested params."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.weight = Parameter("w", description="Inner weight")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.bias = Parameter("b", description="Outer bias")
                self.inner = Inner()

        outer = Outer()
        named_params = dict(outer.named_parameters())

        # Direct parameter gets simple name
        assert "bias" in named_params
        # Nested parameter gets hierarchical name
        assert "inner.weight" in named_params

    def test_deeply_nested_named_parameters(self) -> None:
        """named_parameters() handles deeply nested modules."""

        class Level3(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.param = Parameter("l3", description="Level 3 param")

        class Level2(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level3 = Level3()

        class Level1(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level2 = Level2()

        root = Level1()
        named_params = dict(root.named_parameters())

        assert "level2.level3.param" in named_params

    def test_unowned_parameter_has_no_name(self) -> None:
        """Parameters not assigned to modules have _name=None."""
        param = Parameter("value", description="Test")
        assert param._name is None

    def test_parameter_name_is_immediate_but_refs_are_hierarchical(self) -> None:
        """Parameter._name stores immediate name; refs use hierarchical paths.

        The _name field stores the immediate attribute name for the module
        introspection API. The hierarchical path is computed for refs and
        named_parameters().
        """

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.weight = Parameter("w", description="Inner weight")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

        outer = Outer()
        # _name stores immediate attribute name
        assert outer.inner.weight._name == "weight"

        # named_parameters() gives hierarchical paths
        named_params = dict(outer.named_parameters())
        assert "inner.weight" in named_params

        # valueify() ref uses hierarchical path
        v = valueify(outer.inner.weight)
        assert v.ref == "param:inner.weight"
