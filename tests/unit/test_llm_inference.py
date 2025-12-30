"""Unit tests for the LLMInference atomic module.

This file contains tests for PR-006: LLMInference class implementation.
Tests cover all constructor variations, system_prompt handling, and
the forward() method behavior.
"""

import pytest

from inf_engine.module import InferenceModule, LLMInference
from inf_engine.parameter import Parameter


class TestLLMInferenceCreation:
    """Tests for LLMInference basic instantiation."""

    def test_llm_inference_creation_minimal(self) -> None:
        """LLMInference can be created with just an alias."""
        llm = LLMInference(alias="test_llm")

        assert llm is not None
        assert isinstance(llm, LLMInference)

    def test_llm_inference_is_inference_module(self) -> None:
        """LLMInference is a subclass of InferenceModule."""
        llm = LLMInference(alias="test_llm")

        assert isinstance(llm, InferenceModule)

    def test_llm_inference_has_registries(self) -> None:
        """LLMInference has _children and _parameters registries from parent."""
        llm = LLMInference(alias="test_llm")

        assert hasattr(llm, "_children")
        assert hasattr(llm, "_parameters")
        assert isinstance(llm._children, dict)
        assert isinstance(llm._parameters, dict)


class TestLLMInferenceAlias:
    """Tests for alias attribute."""

    def test_alias_stored_correctly(self) -> None:
        """Alias is stored as provided."""
        llm = LLMInference(alias="my_custom_alias")

        assert llm.alias == "my_custom_alias"

    def test_alias_various_formats(self) -> None:
        """Alias can be any string format."""
        test_aliases = [
            "simple",
            "with_underscores",
            "with-dashes",
            "CamelCase",
            "with.dots",
            "complex_alias-v2.1",
        ]

        for alias in test_aliases:
            llm = LLMInference(alias=alias)
            assert llm.alias == alias


class TestLLMInferenceSystemPrompt:
    """Tests for system_prompt handling."""

    def test_system_prompt_string_wrapped_as_parameter(self) -> None:
        """Non-empty string system_prompt is wrapped as a Parameter."""
        llm = LLMInference(
            alias="test_llm",
            system_prompt="You are a helpful assistant.",
        )

        assert llm.system_prompt is not None
        assert isinstance(llm.system_prompt, Parameter)
        assert llm.system_prompt.value == "You are a helpful assistant."

    def test_system_prompt_string_requires_grad_false(self) -> None:
        """String system_prompt is wrapped with requires_grad=False."""
        llm = LLMInference(
            alias="test_llm",
            system_prompt="You are helpful.",
        )

        assert llm.system_prompt is not None
        assert llm.system_prompt.requires_grad is False

    def test_system_prompt_empty_string_is_none(self) -> None:
        """Empty string system_prompt results in None."""
        llm = LLMInference(alias="test_llm", system_prompt="")

        assert llm.system_prompt is None

    def test_system_prompt_default_is_none(self) -> None:
        """Default system_prompt (empty string) results in None."""
        llm = LLMInference(alias="test_llm")

        assert llm.system_prompt is None

    def test_system_prompt_parameter_passed_through(self) -> None:
        """Parameter system_prompt is used as-is."""
        param = Parameter("Learnable prompt", description="test", requires_grad=True)
        llm = LLMInference(alias="test_llm", system_prompt=param)

        assert llm.system_prompt is param
        assert llm.system_prompt.value == "Learnable prompt"
        assert llm.system_prompt.requires_grad is True

    def test_system_prompt_parameter_requires_grad_false(self) -> None:
        """Parameter with requires_grad=False is preserved."""
        param = Parameter("Fixed prompt", description="test", requires_grad=False)
        llm = LLMInference(alias="test_llm", system_prompt=param)

        assert llm.system_prompt is param
        assert llm.system_prompt.requires_grad is False

    def test_system_prompt_parameter_registered(self) -> None:
        """Parameter system_prompt is registered in _parameters."""
        param = Parameter("Learnable prompt", description="test", requires_grad=True)
        llm = LLMInference(alias="test_llm", system_prompt=param)

        assert "system_prompt" in llm._parameters
        assert llm._parameters["system_prompt"] is param

    def test_system_prompt_string_registered_as_parameter(self) -> None:
        """String system_prompt wrapped as Parameter is registered."""
        llm = LLMInference(alias="test_llm", system_prompt="Test prompt")

        assert "system_prompt" in llm._parameters
        assert llm._parameters["system_prompt"] is llm.system_prompt

    def test_system_prompt_none_not_registered(self) -> None:
        """None system_prompt is not registered in _parameters."""
        llm = LLMInference(alias="test_llm", system_prompt="")

        assert "system_prompt" not in llm._parameters

    def test_system_prompt_multiline(self) -> None:
        """Multiline system_prompt is handled correctly."""
        multiline_prompt = """You are a helpful assistant.
You always respond in a friendly manner.
You provide detailed explanations."""

        llm = LLMInference(alias="test_llm", system_prompt=multiline_prompt)

        assert llm.system_prompt is not None
        assert llm.system_prompt.value == multiline_prompt


class TestLLMInferenceTemperature:
    """Tests for temperature attribute."""

    def test_temperature_default(self) -> None:
        """Temperature defaults to 1.0."""
        llm = LLMInference(alias="test_llm")

        assert llm.temperature == 1.0

    def test_temperature_custom_value(self) -> None:
        """Temperature can be set to custom values."""
        llm = LLMInference(alias="test_llm", temperature=0.7)

        assert llm.temperature == 0.7

    def test_temperature_zero(self) -> None:
        """Temperature can be set to 0 (deterministic)."""
        llm = LLMInference(alias="test_llm", temperature=0.0)

        assert llm.temperature == 0.0

    def test_temperature_high(self) -> None:
        """Temperature can be set to high values (more random)."""
        llm = LLMInference(alias="test_llm", temperature=2.0)

        assert llm.temperature == 2.0


class TestLLMInferenceMaxTokens:
    """Tests for max_tokens attribute."""

    def test_max_tokens_default_none(self) -> None:
        """max_tokens defaults to None (no limit)."""
        llm = LLMInference(alias="test_llm")

        assert llm.max_tokens is None

    def test_max_tokens_custom_value(self) -> None:
        """max_tokens can be set to custom values."""
        llm = LLMInference(alias="test_llm", max_tokens=1000)

        assert llm.max_tokens == 1000

    def test_max_tokens_small(self) -> None:
        """max_tokens can be set to small values."""
        llm = LLMInference(alias="test_llm", max_tokens=10)

        assert llm.max_tokens == 10

    def test_max_tokens_large(self) -> None:
        """max_tokens can be set to large values."""
        llm = LLMInference(alias="test_llm", max_tokens=100000)

        assert llm.max_tokens == 100000


class TestLLMInferenceResponseFormat:
    """Tests for response_format attribute."""

    def test_response_format_default_none(self) -> None:
        """response_format defaults to None (plain text)."""
        llm = LLMInference(alias="test_llm")

        assert llm.response_format is None

    def test_response_format_dict(self) -> None:
        """response_format can be set to dict."""
        llm = LLMInference(alias="test_llm", response_format=dict)

        assert llm.response_format is dict

    def test_response_format_list(self) -> None:
        """response_format can be set to list."""
        llm = LLMInference(alias="test_llm", response_format=list)

        assert llm.response_format is list

    def test_response_format_custom_class(self) -> None:
        """response_format can be set to a custom class."""

        class CustomResponse:
            pass

        llm = LLMInference(alias="test_llm", response_format=CustomResponse)

        assert llm.response_format is CustomResponse


class TestLLMInferenceAllParameters:
    """Tests for LLMInference with all parameters specified."""

    def test_all_parameters_set(self) -> None:
        """All parameters can be set at once."""
        llm = LLMInference(
            alias="full_config",
            system_prompt="You are helpful.",
            temperature=0.5,
            max_tokens=2000,
            response_format=dict,
        )

        assert llm.alias == "full_config"
        assert llm.system_prompt is not None
        assert llm.system_prompt.value == "You are helpful."
        assert llm.temperature == 0.5
        assert llm.max_tokens == 2000
        assert llm.response_format is dict

    def test_all_parameters_with_parameter_system_prompt(self) -> None:
        """All parameters work with Parameter system_prompt."""
        param = Parameter("Learnable", description="test", requires_grad=True)
        llm = LLMInference(
            alias="learnable_config",
            system_prompt=param,
            temperature=0.3,
            max_tokens=500,
            response_format=list,
        )

        assert llm.alias == "learnable_config"
        assert llm.system_prompt is param
        assert llm.system_prompt.requires_grad is True
        assert llm.temperature == 0.3
        assert llm.max_tokens == 500
        assert llm.response_format is list


class TestLLMInferenceForward:
    """Tests for forward() method behavior."""

    def test_forward_raises_runtime_error(self) -> None:
        """forward() raises RuntimeError when called directly."""
        llm = LLMInference(alias="test_llm")

        with pytest.raises(RuntimeError) as exc_info:
            llm.forward("test prompt")

        assert "should not be called directly" in str(exc_info.value)
        assert "run()" in str(exc_info.value)

    def test_call_raises_runtime_error(self) -> None:
        """__call__() raises RuntimeError (delegates to forward)."""
        llm = LLMInference(alias="test_llm")

        with pytest.raises(RuntimeError) as exc_info:
            llm("test prompt")

        assert "should not be called directly" in str(exc_info.value)

    def test_forward_error_message_mentions_llm_inference(self) -> None:
        """Error message specifically mentions LLMInference."""
        llm = LLMInference(alias="test_llm")

        with pytest.raises(RuntimeError) as exc_info:
            llm.forward("test")

        assert "LLMInference" in str(exc_info.value)


class TestLLMInferenceModuleIntegration:
    """Tests for LLMInference integration with InferenceModule features."""

    def test_llm_inference_as_child_module(self) -> None:
        """LLMInference can be registered as a child of another module."""

        class ParentModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="child_llm")

        parent = ParentModule()

        assert "llm" in parent._children
        assert parent._children["llm"] is parent.llm
        assert isinstance(parent.llm, LLMInference)

    def test_llm_inference_name_set_when_child(self) -> None:
        """LLMInference._name is set when registered as child."""

        class ParentModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.my_llm = LLMInference(alias="test")

        parent = ParentModule()

        assert parent.my_llm._name == "my_llm"

    def test_multiple_llm_inferences_in_module(self) -> None:
        """Multiple LLMInference instances can be children of same parent."""

        class MultiLLMModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.fast_llm = LLMInference(alias="fast", temperature=0.3)
                self.smart_llm = LLMInference(alias="smart", temperature=0.7)
                self.creative_llm = LLMInference(alias="creative", temperature=1.5)

        module = MultiLLMModule()

        assert len(module._children) == 3
        assert "fast_llm" in module._children
        assert "smart_llm" in module._children
        assert "creative_llm" in module._children

    def test_llm_inference_found_by_modules_iterator(self) -> None:
        """LLMInference is yielded by modules() iterator."""

        class ParentModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="test")

        parent = ParentModule()
        all_modules = list(parent.modules())

        assert len(all_modules) == 2
        assert parent in all_modules
        assert parent.llm in all_modules

    def test_llm_inference_found_by_named_modules(self) -> None:
        """LLMInference has correct name in named_modules()."""

        class ParentModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="test")

        parent = ParentModule()
        named = dict(parent.named_modules())

        assert "llm" in named
        assert named["llm"] is parent.llm

    def test_system_prompt_parameter_found_by_parameters(self) -> None:
        """system_prompt Parameter is found by parameters() iterator."""

        class ParentModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="test", system_prompt="You are helpful.")

        parent = ParentModule()
        all_params = list(parent.parameters())

        assert len(all_params) == 1
        assert all_params[0] is parent.llm.system_prompt

    def test_system_prompt_parameter_found_by_named_parameters(self) -> None:
        """system_prompt has correct hierarchical name in named_parameters()."""

        class ParentModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="test", system_prompt="You are helpful.")

        parent = ParentModule()
        named_params = dict(parent.named_parameters())

        assert "llm.system_prompt" in named_params
        assert named_params["llm.system_prompt"] is parent.llm.system_prompt

    def test_learnable_system_prompt_requires_grad(self) -> None:
        """Learnable system_prompt found in parameters with requires_grad=True."""

        class ParentModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(
                    alias="test",
                    system_prompt=Parameter(
                        "Learnable", description="test", requires_grad=True
                    ),
                )

        parent = ParentModule()
        params = list(parent.parameters())

        assert len(params) == 1
        assert params[0].requires_grad is True


class TestLLMInferenceEdgeCases:
    """Edge case tests for LLMInference."""

    def test_whitespace_only_system_prompt(self) -> None:
        """Whitespace-only system_prompt is treated as non-empty."""
        llm = LLMInference(alias="test", system_prompt="   ")

        # Whitespace is a non-empty string, so it should be wrapped
        assert llm.system_prompt is not None
        assert llm.system_prompt.value == "   "

    def test_unicode_system_prompt(self) -> None:
        """Unicode characters in system_prompt are handled correctly."""
        unicode_prompt = "你是一个有帮助的助手。🤖"
        llm = LLMInference(alias="test", system_prompt=unicode_prompt)

        assert llm.system_prompt is not None
        assert llm.system_prompt.value == unicode_prompt

    def test_unicode_alias(self) -> None:
        """Unicode characters in alias are accepted."""
        llm = LLMInference(alias="模型_v1")

        assert llm.alias == "模型_v1"

    def test_very_long_system_prompt(self) -> None:
        """Very long system_prompt is handled correctly."""
        long_prompt = "You are helpful. " * 1000
        llm = LLMInference(alias="test", system_prompt=long_prompt)

        assert llm.system_prompt is not None
        assert llm.system_prompt.value == long_prompt

    def test_system_prompt_with_special_characters(self) -> None:
        """Special characters in system_prompt are preserved."""
        special_prompt = "Handle: <xml>, {json}, 'quotes', \"double\", \\backslash"
        llm = LLMInference(alias="test", system_prompt=special_prompt)

        assert llm.system_prompt is not None
        assert llm.system_prompt.value == special_prompt

    def test_empty_parameter_system_prompt(self) -> None:
        """Parameter with empty string value is still used as-is."""
        param = Parameter("", description="test", requires_grad=True)
        llm = LLMInference(alias="test", system_prompt=param)

        # Parameter is passed through even if its value is empty
        assert llm.system_prompt is param
        assert llm.system_prompt.value == ""

    def test_reassign_system_prompt(self) -> None:
        """system_prompt can be reassigned after creation."""
        llm = LLMInference(alias="test", system_prompt="Original")

        assert llm.system_prompt is not None
        assert llm.system_prompt.value == "Original"
        new_param = Parameter("New prompt", description="test", requires_grad=True)
        llm.system_prompt = new_param

        # New parameter is registered (stale entry remains per PyTorch behavior)
        assert llm.system_prompt is new_param
        assert "system_prompt" in llm._parameters

    def test_children_iterator_empty_for_llm_inference(self) -> None:
        """LLMInference has no children of its own."""
        llm = LLMInference(alias="test", system_prompt="You are helpful.")

        children = list(llm.children())

        assert children == []

    def test_temperature_float_precision(self) -> None:
        """Temperature preserves float precision."""
        llm = LLMInference(alias="test", temperature=0.123456789)

        assert llm.temperature == 0.123456789
