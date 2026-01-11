"""Unit tests for Parameter lifting to Value with stable refs."""

from plait.module import InferenceModule
from plait.parameter import Parameter
from plait.values import Value, ValueKind, valueify


class TestValueifyParameter:
    """Tests for valueify(Parameter) behavior."""

    def test_valueify_parameter_returns_value(self) -> None:
        """valueify lifts Parameter to Value."""
        param = Parameter("prompt text", description="Test prompt")
        v = valueify(param)
        assert isinstance(v, Value)

    def test_valueify_parameter_payload(self) -> None:
        """valueify extracts parameter value as payload."""
        param = Parameter("prompt text", description="Test prompt")
        v = valueify(param)
        assert v.payload == "prompt text"

    def test_valueify_parameter_text_kind(self) -> None:
        """valueify infers TEXT kind for string parameter."""
        param = Parameter("prompt text", description="Test prompt")
        v = valueify(param)
        assert v.kind == ValueKind.TEXT

    def test_valueify_parameter_structured_dict(self) -> None:
        """valueify infers STRUCTURED kind for dict parameter.

        Note: Parameter currently only supports string values,
        but the design allows for structured values in future.
        """
        param = Parameter("string value", description="Test")
        param.value = {"key": "value"}  # type: ignore[assignment]
        v = valueify(param)
        assert v.kind == ValueKind.STRUCTURED

    def test_valueify_parameter_structured_list(self) -> None:
        """valueify infers STRUCTURED kind for list parameter."""
        param = Parameter("string value", description="Test")
        param.value = ["a", "b", "c"]  # type: ignore[assignment]
        v = valueify(param)
        assert v.kind == ValueKind.STRUCTURED


class TestParameterStableRef:
    """Tests for stable param:name ref format."""

    def test_valueify_parameter_ref_format(self) -> None:
        """valueify creates ref with param: prefix."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.system_prompt = Parameter("value", description="Test")

        module = TestModule()
        v = valueify(module.system_prompt)
        assert v.ref == "param:system_prompt"

    def test_valueify_parameter_unnamed_ref(self) -> None:
        """valueify uses param_id for parameters without _name."""
        param = Parameter("value", description="Test")
        v = valueify(param)
        assert v.ref == f"param:{param._id}"

    def test_valueify_parameter_hierarchical_name(self) -> None:
        """valueify preserves hierarchical parameter names."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("value", description="Test")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

        outer = Outer()
        v = valueify(outer.inner.prompt)
        assert v.ref == "param:inner.prompt"


class TestParameterMetadata:
    """Tests for Parameter metadata in lifted Value."""

    def test_valueify_parameter_meta_param_name(self) -> None:
        """valueify includes param_name in meta."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_param = Parameter("value", description="Test")

        module = TestModule()
        v = valueify(module.my_param)
        assert v.meta["param_name"] == "my_param"

    def test_valueify_parameter_meta_param_id(self) -> None:
        """valueify includes param_id in meta."""
        param = Parameter("value", description="Test")
        v = valueify(param)
        assert v.meta["param_id"] == param._id

    def test_valueify_parameter_meta_module_state_version(self) -> None:
        """valueify includes module_state_version in meta."""
        param = Parameter("value", description="Test")
        v = valueify(param)
        assert v.meta["module_state_version"] == 0

    def test_valueify_parameter_meta_requires_grad_true(self) -> None:
        """valueify includes requires_grad=True in meta."""
        param = Parameter("value", description="Test", requires_grad=True)
        v = valueify(param)
        assert v.meta["requires_grad"] is True

    def test_valueify_parameter_meta_requires_grad_false(self) -> None:
        """valueify includes requires_grad=False in meta."""
        param = Parameter("value", description="Test", requires_grad=False)
        v = valueify(param)
        assert v.meta["requires_grad"] is False

    def test_valueify_parameter_meta_unnamed(self) -> None:
        """valueify uses None in meta for parameters without _name."""
        param = Parameter("value", description="Test")
        v = valueify(param)
        assert v.meta["param_name"] is None


class TestParameterKindOverride:
    """Tests for kind override when valueifying Parameters."""

    def test_valueify_parameter_kind_override(self) -> None:
        """valueify respects kind override for parameters."""
        param = Parameter("template {x}", description="Template")
        v = valueify(param, kind=ValueKind.FSTRING)
        assert v.kind == ValueKind.FSTRING
        assert v.payload == "template {x}"

    def test_valueify_parameter_kind_override_preserves_ref(self) -> None:
        """valueify kind override preserves parameter ref."""

        class TestModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_param = Parameter("value", description="Test")

        module = TestModule()
        v = valueify(module.my_param, kind=ValueKind.OTHER)
        assert v.ref == "param:my_param"

    def test_valueify_parameter_kind_override_preserves_meta(self) -> None:
        """valueify kind override preserves parameter metadata."""
        param = Parameter("value", description="Test", requires_grad=True)
        v = valueify(param, kind=ValueKind.OTHER)
        assert v.meta["requires_grad"] is True


class TestInputRefFormat:
    """Tests for input:name ref format (used by tracer.bind_inputs)."""

    def test_input_ref_format(self) -> None:
        """Input refs use input: prefix convention.

        While valueify doesn't create input: refs directly, this test
        documents the expected format for input refs that tracer.bind_inputs
        will create.
        """
        # Direct Value creation with input ref
        v = Value(ValueKind.TEXT, "user input", ref="input:query")
        assert v.ref == "input:query"
        assert v.ref.startswith("input:")

    def test_input_ref_in_collect_refs(self) -> None:
        """collect_refs works with input: refs."""
        from plait.values import collect_refs

        v = Value(ValueKind.TEXT, "user input", ref="input:query")
        refs = collect_refs(v)
        assert refs == ["input:query"]
