"""Unit tests for BackwardContext and BackwardResult classes.

This file contains tests for PR-067: Backward pass infrastructure.
Tests cover BackwardContext creation, reason() method, and BackwardResult.
"""

import pytest

from plait.graph import InferenceGraph
from plait.optimization.backward import BackwardContext, BackwardResult
from plait.optimization.feedback import Feedback


class TestBackwardContextCreation:
    """Tests for BackwardContext instantiation."""

    def test_backward_context_creation(self) -> None:
        """BackwardContext can be created with all required fields."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        feedback = Feedback(content="Good job", score=0.8)

        ctx = BackwardContext(
            node_id="LLMInference_1",
            inputs={"prompt": "Hello"},
            output="Hi there!",
            graph=graph,
            all_results={"LLMInference_1": "Hi there!"},
            downstream_feedback=[feedback],
        )

        assert ctx.node_id == "LLMInference_1"
        assert ctx.inputs == {"prompt": "Hello"}
        assert ctx.output == "Hi there!"
        assert ctx.graph is graph
        assert ctx.all_results == {"LLMInference_1": "Hi there!"}
        assert ctx.downstream_feedback == [feedback]
        assert ctx.reasoning_llm is None

    def test_backward_context_with_reasoning_llm(self) -> None:
        """BackwardContext can be created with optional reasoning_llm."""
        from plait.module import LLMInference

        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        feedback = Feedback(content="Needs work", score=0.4)
        reasoning_llm = LLMInference(alias="reasoning")

        ctx = BackwardContext(
            node_id="node_1",
            inputs={},
            output="result",
            graph=graph,
            all_results={},
            downstream_feedback=[feedback],
            reasoning_llm=reasoning_llm,
        )

        assert ctx.reasoning_llm is reasoning_llm

    def test_backward_context_multiple_downstream_feedback(self) -> None:
        """BackwardContext can hold multiple downstream feedback items."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        fb1 = Feedback(content="Feedback 1", score=0.6)
        fb2 = Feedback(content="Feedback 2", score=0.8)
        fb3 = Feedback(content="Feedback 3", score=0.7)

        ctx = BackwardContext(
            node_id="node_1",
            inputs={},
            output="result",
            graph=graph,
            all_results={},
            downstream_feedback=[fb1, fb2, fb3],
        )

        assert len(ctx.downstream_feedback) == 3
        assert ctx.downstream_feedback[0] is fb1
        assert ctx.downstream_feedback[1] is fb2
        assert ctx.downstream_feedback[2] is fb3

    def test_backward_context_complex_inputs(self) -> None:
        """BackwardContext can store complex input structures."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        feedback = Feedback(content="OK")

        ctx = BackwardContext(
            node_id="node_1",
            inputs={
                "prompt": "Hello",
                "context": ["item1", "item2"],
                "config": {"key": "value", "nested": {"a": 1}},
            },
            output={"response": "Hi", "metadata": {"tokens": 5}},
            graph=graph,
            all_results={},
            downstream_feedback=[feedback],
        )

        assert ctx.inputs["prompt"] == "Hello"
        assert ctx.inputs["context"] == ["item1", "item2"]
        assert ctx.inputs["config"]["nested"]["a"] == 1
        assert ctx.output["response"] == "Hi"


class TestBackwardContextReason:
    """Tests for BackwardContext.reason() method."""

    @pytest.mark.asyncio
    async def test_reason_without_llm_raises(self) -> None:
        """reason() raises RuntimeError when no reasoning_llm is available."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        feedback = Feedback(content="Test")

        ctx = BackwardContext(
            node_id="node_1",
            inputs={},
            output="result",
            graph=graph,
            all_results={},
            downstream_feedback=[feedback],
            reasoning_llm=None,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await ctx.reason("How should we improve?")

        assert "No reasoning LLM available" in str(exc_info.value)
        assert "optimizer" in str(exc_info.value)


class TestBackwardResultCreation:
    """Tests for BackwardResult instantiation."""

    def test_backward_result_creation_empty(self) -> None:
        """BackwardResult can be created with empty defaults."""
        result = BackwardResult()

        assert result.input_feedback == {}
        assert result.parameter_feedback == {}

    def test_backward_result_with_input_feedback(self) -> None:
        """BackwardResult can store input feedback."""
        result = BackwardResult()
        feedback = Feedback(content="Input was good", score=0.9)
        result.input_feedback["prompt"] = feedback

        assert "prompt" in result.input_feedback
        assert result.input_feedback["prompt"] is feedback

    def test_backward_result_with_parameter_feedback(self) -> None:
        """BackwardResult can store parameter feedback strings."""
        result = BackwardResult()
        result.parameter_feedback["system_prompt"] = "Be more concise"
        result.parameter_feedback["instructions"] = "Add more examples"

        assert result.parameter_feedback["system_prompt"] == "Be more concise"
        assert result.parameter_feedback["instructions"] == "Add more examples"

    def test_backward_result_multiple_inputs(self) -> None:
        """BackwardResult can store feedback for multiple inputs."""
        result = BackwardResult()
        fb1 = Feedback(content="Input 1 feedback")
        fb2 = Feedback(content="Input 2 feedback")

        result.input_feedback["arg_0"] = fb1
        result.input_feedback["context"] = fb2

        assert len(result.input_feedback) == 2
        assert result.input_feedback["arg_0"] is fb1
        assert result.input_feedback["context"] is fb2

    def test_backward_result_combined(self) -> None:
        """BackwardResult can store both input and parameter feedback."""
        result = BackwardResult()
        feedback = Feedback(content="Good", score=0.8)

        result.input_feedback["prompt"] = feedback
        result.parameter_feedback["system_prompt"] = "Improve clarity"

        assert len(result.input_feedback) == 1
        assert len(result.parameter_feedback) == 1
        assert result.input_feedback["prompt"].score == 0.8
        assert result.parameter_feedback["system_prompt"] == "Improve clarity"


class TestBackwardResultDataclass:
    """Tests for BackwardResult dataclass behavior."""

    def test_backward_result_default_factory_isolation(self) -> None:
        """Each BackwardResult instance has its own dictionaries."""
        result1 = BackwardResult()
        result2 = BackwardResult()

        result1.input_feedback["a"] = Feedback(content="Test")
        result1.parameter_feedback["b"] = "feedback"

        assert result2.input_feedback == {}
        assert result2.parameter_feedback == {}

    def test_backward_result_initialization_with_values(self) -> None:
        """BackwardResult can be initialized with values."""
        feedback = Feedback(content="Initialized", score=0.5)
        result = BackwardResult(
            input_feedback={"prompt": feedback},
            parameter_feedback={"system_prompt": "Initial feedback"},
        )

        assert result.input_feedback["prompt"] is feedback
        assert result.parameter_feedback["system_prompt"] == "Initial feedback"
