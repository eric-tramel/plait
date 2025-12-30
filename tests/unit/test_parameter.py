"""Unit tests for the Parameter class."""

import pytest

from inf_engine.parameter import Parameter


class TestParameterCreation:
    """Tests for Parameter instantiation."""

    def test_parameter_creation_with_value_and_description(self) -> None:
        """Parameter can be created with value and description."""
        param = Parameter("test value", description="Test description")
        assert param.value == "test value"
        assert param.description == "Test description"

    def test_parameter_description_required(self) -> None:
        """Parameter requires a description field."""
        with pytest.raises(TypeError, match="missing.*'description'"):
            Parameter("test value")  # type: ignore[call-arg]

    def test_parameter_creation_with_requires_grad_true(self) -> None:
        """Parameter defaults to requires_grad=True."""
        param = Parameter("test value", description="Test description")
        assert param.requires_grad is True

    def test_parameter_creation_with_requires_grad_false(self) -> None:
        """Parameter can be created with requires_grad=False."""
        param = Parameter(
            "test value", description="Test description", requires_grad=False
        )
        assert param.requires_grad is False

    def test_parameter_creation_empty_feedback_buffer(self) -> None:
        """Parameter starts with an empty feedback buffer."""
        param = Parameter("test value", description="Test description")
        assert param._feedback_buffer == []
        assert param.get_accumulated_feedback() == []

    def test_parameter_creation_name_defaults_none(self) -> None:
        """Parameter _name defaults to None."""
        param = Parameter("test value", description="Test description")
        assert param._name is None


class TestParameterStr:
    """Tests for Parameter string representation."""

    def test_parameter_str_returns_value(self) -> None:
        """str(param) returns the current value."""
        param = Parameter("hello world", description="Test description")
        assert str(param) == "hello world"

    def test_parameter_str_empty_value(self) -> None:
        """str(param) works with empty string value."""
        param = Parameter("", description="Empty value test")
        assert str(param) == ""

    def test_parameter_str_multiline_value(self) -> None:
        """str(param) works with multiline values."""
        value = "line 1\nline 2\nline 3"
        param = Parameter(value, description="Multiline test")
        assert str(param) == value


class TestParameterAccumulateFeedback:
    """Tests for Parameter.accumulate_feedback()."""

    def test_accumulate_single_feedback(self) -> None:
        """Single feedback is accumulated."""
        param = Parameter("value", description="Test description")
        param.accumulate_feedback("be more specific")
        assert param.get_accumulated_feedback() == ["be more specific"]

    def test_accumulate_multiple_feedback(self) -> None:
        """Multiple feedback strings are accumulated in order."""
        param = Parameter("value", description="Test description")
        param.accumulate_feedback("feedback 1")
        param.accumulate_feedback("feedback 2")
        param.accumulate_feedback("feedback 3")
        assert param.get_accumulated_feedback() == [
            "feedback 1",
            "feedback 2",
            "feedback 3",
        ]

    def test_accumulate_feedback_requires_grad_false(self) -> None:
        """Feedback is not accumulated when requires_grad=False."""
        param = Parameter("value", description="Test description", requires_grad=False)
        param.accumulate_feedback("should be ignored")
        assert param.get_accumulated_feedback() == []

    def test_accumulate_feedback_empty_string(self) -> None:
        """Empty string feedback is accumulated."""
        param = Parameter("value", description="Test description")
        param.accumulate_feedback("")
        assert param.get_accumulated_feedback() == [""]

    def test_get_accumulated_feedback_returns_copy(self) -> None:
        """get_accumulated_feedback returns a copy, not the original list."""
        param = Parameter("value", description="Test description")
        param.accumulate_feedback("test")
        feedback = param.get_accumulated_feedback()
        feedback.append("should not affect original")
        assert param.get_accumulated_feedback() == ["test"]


class TestParameterApplyUpdate:
    """Tests for Parameter.apply_update()."""

    def test_apply_update_changes_value(self) -> None:
        """apply_update changes the parameter value."""
        param = Parameter("old value", description="Test description")
        param.apply_update("new value")
        assert param.value == "new value"
        assert str(param) == "new value"

    def test_apply_update_clears_feedback_buffer(self) -> None:
        """apply_update clears the feedback buffer."""
        param = Parameter("value", description="Test description")
        param.accumulate_feedback("feedback 1")
        param.accumulate_feedback("feedback 2")
        assert len(param.get_accumulated_feedback()) == 2

        param.apply_update("new value")
        assert param.get_accumulated_feedback() == []

    def test_apply_update_with_empty_buffer(self) -> None:
        """apply_update works even with empty feedback buffer."""
        param = Parameter("old value", description="Test description")
        param.apply_update("new value")
        assert param.value == "new value"
        assert param.get_accumulated_feedback() == []

    def test_apply_update_does_not_change_requires_grad(self) -> None:
        """apply_update does not affect requires_grad setting."""
        param = Parameter("value", description="Test description", requires_grad=False)
        param.apply_update("new value")
        assert param.requires_grad is False


class TestParameterZeroFeedback:
    """Tests for Parameter.zero_feedback()."""

    def test_zero_feedback_clears_buffer(self) -> None:
        """zero_feedback clears the feedback buffer."""
        param = Parameter("value", description="Test description")
        param.accumulate_feedback("feedback 1")
        param.accumulate_feedback("feedback 2")
        assert len(param.get_accumulated_feedback()) == 2

        param.zero_feedback()
        assert param.get_accumulated_feedback() == []

    def test_zero_feedback_does_not_change_value(self) -> None:
        """zero_feedback does not change the parameter value."""
        param = Parameter("unchanged value", description="Test description")
        param.accumulate_feedback("feedback")
        param.zero_feedback()
        assert param.value == "unchanged value"

    def test_zero_feedback_on_empty_buffer(self) -> None:
        """zero_feedback works on already empty buffer."""
        param = Parameter("value", description="Test description")
        param.zero_feedback()  # Should not raise
        assert param.get_accumulated_feedback() == []


class TestParameterEquality:
    """Tests for Parameter equality comparison."""

    def test_parameters_equal_same_value_description_and_requires_grad(self) -> None:
        """Parameters with same value, description and requires_grad are equal."""
        param1 = Parameter("value", description="Test description", requires_grad=True)
        param2 = Parameter("value", description="Test description", requires_grad=True)
        assert param1 == param2

    def test_parameters_not_equal_different_value(self) -> None:
        """Parameters with different values are not equal."""
        param1 = Parameter("value1", description="Test description")
        param2 = Parameter("value2", description="Test description")
        assert param1 != param2

    def test_parameters_not_equal_different_description(self) -> None:
        """Parameters with different descriptions are not equal."""
        param1 = Parameter("value", description="Description 1")
        param2 = Parameter("value", description="Description 2")
        assert param1 != param2

    def test_parameters_not_equal_different_requires_grad(self) -> None:
        """Parameters with different requires_grad are not equal."""
        param1 = Parameter("value", description="Test description", requires_grad=True)
        param2 = Parameter("value", description="Test description", requires_grad=False)
        assert param1 != param2

    def test_parameters_equal_ignores_name(self) -> None:
        """Parameter equality ignores _name field."""
        param1 = Parameter("value", description="Test description")
        param2 = Parameter("value", description="Test description")
        param1._name = "name1"
        param2._name = "name2"
        assert param1 == param2

    def test_parameters_equal_ignores_feedback_buffer(self) -> None:
        """Parameter equality ignores feedback buffer."""
        param1 = Parameter("value", description="Test description")
        param2 = Parameter("value", description="Test description")
        param1.accumulate_feedback("feedback")
        assert param1 == param2


class TestParameterRepr:
    """Tests for Parameter repr representation."""

    def test_parameter_repr_includes_value(self) -> None:
        """repr includes the value."""
        param = Parameter("test value", description="Test description")
        assert "test value" in repr(param)

    def test_parameter_repr_includes_description(self) -> None:
        """repr includes the description."""
        param = Parameter("value", description="My description")
        assert "My description" in repr(param)

    def test_parameter_repr_includes_requires_grad(self) -> None:
        """repr includes requires_grad status."""
        param = Parameter("value", description="Test description", requires_grad=False)
        assert "requires_grad=False" in repr(param)

    def test_parameter_repr_excludes_name(self) -> None:
        """repr excludes _name field."""
        param = Parameter("value", description="Test description")
        param._name = "should_not_appear"
        assert "_name" not in repr(param)
        assert "should_not_appear" not in repr(param)

    def test_parameter_repr_excludes_feedback_buffer(self) -> None:
        """repr excludes _feedback_buffer field."""
        param = Parameter("value", description="Test description")
        param.accumulate_feedback("should_not_appear")
        assert "_feedback_buffer" not in repr(param)
