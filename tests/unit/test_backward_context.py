"""Unit tests for BackwardContext and BackwardResult classes."""

import pytest

from plait.graph import InferenceGraph
from plait.optimization.backward import BackwardContext, BackwardResult
from plait.values import Value, ValueKind


class TestBackwardContextCreation:
    def test_backward_context_creation(self) -> None:
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        feedback = Value(ValueKind.STRUCTURED, [["Good job"]], meta={"score": 0.8})

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
        from plait.module import LLMInference

        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        feedback = Value(ValueKind.STRUCTURED, [["Needs work"]])
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
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        fb1 = Value(ValueKind.STRUCTURED, [["Feedback 1"]])
        fb2 = Value(ValueKind.STRUCTURED, [["Feedback 2"]])
        fb3 = Value(ValueKind.STRUCTURED, [["Feedback 3"]])

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
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        feedback = Value(ValueKind.STRUCTURED, [["OK"]])

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
    @pytest.mark.asyncio
    async def test_reason_without_llm_raises(self) -> None:
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        feedback = Value(ValueKind.STRUCTURED, [["Test"]])

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
    def test_backward_result_creation_empty(self) -> None:
        result = BackwardResult()
        assert result.input_feedback == {}
        assert result.parameter_feedback == {}

    def test_backward_result_with_input_feedback(self) -> None:
        result = BackwardResult()
        feedback = Value(ValueKind.STRUCTURED, [["Input was good"]])
        result.input_feedback["prompt"] = feedback

        assert "prompt" in result.input_feedback
        assert result.input_feedback["prompt"] is feedback

    def test_backward_result_with_parameter_feedback(self) -> None:
        result = BackwardResult()
        result.parameter_feedback["system_prompt"] = "Be more concise"
        result.parameter_feedback["instructions"] = "Add more examples"

        assert result.parameter_feedback["system_prompt"] == "Be more concise"
        assert result.parameter_feedback["instructions"] == "Add more examples"

    def test_backward_result_multiple_inputs(self) -> None:
        result = BackwardResult()
        fb1 = Value(ValueKind.STRUCTURED, [["Input 1 feedback"]])
        fb2 = Value(ValueKind.STRUCTURED, [["Input 2 feedback"]])

        result.input_feedback["arg_0"] = fb1
        result.input_feedback["context"] = fb2

        assert len(result.input_feedback) == 2
        assert result.input_feedback["arg_0"] is fb1
        assert result.input_feedback["context"] is fb2

    def test_backward_result_combined(self) -> None:
        result = BackwardResult()
        feedback = Value(ValueKind.STRUCTURED, [["Good"]], meta={"score": 0.8})

        result.input_feedback["prompt"] = feedback
        result.parameter_feedback["system_prompt"] = "Improve clarity"

        assert len(result.input_feedback) == 1
        assert len(result.parameter_feedback) == 1
        assert result.input_feedback["prompt"].meta["score"] == 0.8
        assert result.parameter_feedback["system_prompt"] == "Improve clarity"


class TestBackwardResultDataclass:
    def test_backward_result_default_factory_isolation(self) -> None:
        result1 = BackwardResult()
        result2 = BackwardResult()

        result1.input_feedback["a"] = Value(ValueKind.STRUCTURED, [["Test"]])
        result1.parameter_feedback["b"] = "feedback"

        assert result2.input_feedback == {}
        assert result2.parameter_feedback == {}

    def test_backward_result_initialization_with_values(self) -> None:
        feedback = Value(ValueKind.STRUCTURED, [["Initialized"]], meta={"score": 0.5})
        result = BackwardResult(
            input_feedback={"prompt": feedback},
            parameter_feedback={"system_prompt": "Initial feedback"},
        )

        assert result.input_feedback["prompt"] is feedback
        assert result.parameter_feedback["system_prompt"] == "Initial feedback"
