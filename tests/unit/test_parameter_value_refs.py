"""Unit tests for Parameter value refs and structured kind inference.

Tests the stable ref format (param:<name>) and ValueKind inference
when lifting Parameters to Values via valueify().
"""

from inf_engine.parameter import Parameter
from inf_engine.values import (
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

    def test_param_ref_with_name(self) -> None:
        """Named parameters use param:<name> format."""
        param = Parameter("value", description="Test")
        param._name = "system_prompt"
        v = valueify(param)
        assert v.ref == "param:system_prompt"

    def test_param_ref_without_name_uses_id(self) -> None:
        """Unnamed parameters use param:<id> format."""
        param = Parameter("value", description="Test")
        v = valueify(param)
        assert v.ref == f"param:{param._id}"

    def test_param_ref_hierarchical_single_level(self) -> None:
        """Single-level module names are preserved."""
        param = Parameter("value", description="Test")
        param._name = "prompt"
        v = valueify(param)
        assert v.ref == "param:prompt"

    def test_param_ref_hierarchical_two_levels(self) -> None:
        """Two-level module.param names are preserved."""
        param = Parameter("value", description="Test")
        param._name = "module.prompt"
        v = valueify(param)
        assert v.ref == "param:module.prompt"

    def test_param_ref_hierarchical_deep_nesting(self) -> None:
        """Deeply nested names are fully preserved."""
        param = Parameter("value", description="Test")
        param._name = "root.child.grandchild.prompt"
        v = valueify(param)
        assert v.ref == "param:root.child.grandchild.prompt"

    def test_param_ref_with_underscores(self) -> None:
        """Parameter names with underscores are preserved."""
        param = Parameter("value", description="Test")
        param._name = "my_module.my_prompt"
        v = valueify(param)
        assert v.ref == "param:my_module.my_prompt"

    def test_param_ref_numeric_suffix(self) -> None:
        """Parameter names with numeric suffixes are preserved."""
        param = Parameter("value", description="Test")
        param._name = "layer0.prompt1"
        v = valueify(param)
        assert v.ref == "param:layer0.prompt1"


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
        param = Parameter("value", description="Test")
        param._name = "my_param"
        v = valueify(param, kind=ValueKind.OTHER)
        assert v.ref == "param:my_param"

    def test_override_preserves_metadata(self) -> None:
        """Kind override preserves all parameter metadata."""
        param = Parameter("value", description="Test", requires_grad=True)
        param._name = "my_param"
        v = valueify(param, kind=ValueKind.OTHER)
        assert v.meta["param_name"] == "my_param"
        assert v.meta["param_id"] == param._id
        assert v.meta["requires_grad"] is True


class TestParameterValueRefInteraction:
    """Tests for Parameter-derived Values with ValueRef."""

    def test_parameter_value_in_collect_refs(self) -> None:
        """Parameter-derived Values work with collect_refs."""
        param = Parameter("value", description="Test")
        param._name = "my_param"
        v = valueify(param)
        refs = collect_refs(v)
        assert refs == ["param:my_param"]

    def test_parameter_value_replace_with_ref(self) -> None:
        """Parameter-derived Values can be replaced with ValueRef."""
        param = Parameter("value", description="Test")
        param._name = "my_param"
        v = valueify(param)
        ref = replace_values_with_refs(v)
        assert isinstance(ref, ValueRef)
        assert ref.ref == "param:my_param"

    def test_parameter_values_in_nested_structure(self) -> None:
        """Parameter Values work in nested structures."""
        param1 = Parameter("first", description="First")
        param1._name = "p1"
        param2 = Parameter("second", description="Second")
        param2._name = "p2"

        v1 = valueify(param1)
        v2 = valueify(param2)
        nested = {"a": v1, "b": [v2]}

        refs = collect_refs(nested)
        assert set(refs) == {"param:p1", "param:p2"}

    def test_parameter_value_ref_in_replace(self) -> None:
        """Nested parameter Values are replaced with ValueRefs."""
        param = Parameter("value", description="Test")
        param._name = "nested_param"
        v = valueify(param)
        structure = [v, {"key": v}]

        result = replace_values_with_refs(structure)
        assert isinstance(result[0], ValueRef)
        assert isinstance(result[1]["key"], ValueRef)
        assert result[0].ref == "param:nested_param"


class TestConstantParameterRefs:
    """Tests for Parameters with requires_grad=False."""

    def test_constant_parameter_has_ref(self) -> None:
        """Constant parameters still get refs when valueified."""
        param = Parameter({"model": "gpt-4"}, requires_grad=False)
        v = valueify(param)
        assert v.ref.startswith("param:")

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
    """Tests for module_state_version tracking in Parameter Values."""

    def test_initial_version_is_zero(self) -> None:
        """New parameters start with module_state_version=0."""
        param = Parameter("value", description="Test")
        v = valueify(param)
        assert v.meta["module_state_version"] == 0

    def test_version_increments_after_update(self) -> None:
        """module_state_version increments after apply_update."""
        param = Parameter("old", description="Test")
        param.apply_update("new")
        v = valueify(param)
        assert v.meta["module_state_version"] == 1

    def test_version_tracks_multiple_updates(self) -> None:
        """module_state_version tracks multiple updates."""
        param = Parameter("v0", description="Test")
        param.apply_update("v1")
        param.apply_update("v2")
        param.apply_update("v3")
        v = valueify(param)
        assert v.meta["module_state_version"] == 3
