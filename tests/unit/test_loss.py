"""Tests for Loss abstract base class and concrete implementations."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inf_engine.graph import InferenceGraph
from inf_engine.optimization.feedback import Feedback, FeedbackType
from inf_engine.optimization.loss import (
    CompositeLoss,
    ContrastiveLoss,
    HumanFeedbackLoss,
    HumanPreferenceLoss,
    HumanRankingLoss,
    HumanRubricLoss,
    LLMJudge,
    LLMPreferenceLoss,
    LLMRankingLoss,
    LLMRubricLoss,
    Loss,
    PreferenceResponse,
    RankingResponse,
    RubricLevel,
    RubricResponse,
    VerifierLoss,
)
from inf_engine.optimization.record import ForwardRecord


class TestLossABC:
    """Tests for Loss abstract base class interface."""

    def test_loss_is_abstract(self) -> None:
        """Loss cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            Loss()  # type: ignore[abstract]
        assert "abstract" in str(exc_info.value).lower()

    def test_loss_requires_call_method(self) -> None:
        """Subclass must implement __call__ method."""

        class IncompleteLoss(Loss):
            pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteLoss()  # type: ignore[abstract]
        assert (
            "__call__" in str(exc_info.value)
            or "abstract" in str(exc_info.value).lower()
        )


class SimpleLoss(Loss):
    """Simple Loss implementation for testing."""

    def __init__(self, always_score: float = 0.5) -> None:
        self.always_score = always_score

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        score = 1.0 if output == target else self.always_score
        content = f"Output: {output}, Target: {target}"
        feedback = Feedback(
            content=content,
            score=score,
            feedback_type=FeedbackType.VERIFIER,
        )
        return self._attach_record(feedback, record)


class ContextAwareLoss(Loss):
    """Loss that uses context in evaluation."""

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        context = context or {}
        criteria = context.get("criteria", "default")
        feedback = Feedback(
            content=f"Evaluated on: {criteria}",
            score=0.8,
            metadata={"criteria": criteria},
        )
        return self._attach_record(feedback, record)


class TestLossSubclass:
    """Tests for Loss subclass implementation."""

    @pytest.mark.asyncio
    async def test_simple_loss_call(self) -> None:
        """Simple loss can be called and returns feedback."""
        loss = SimpleLoss(always_score=0.7)
        feedback = await loss("hello", target="world")

        assert isinstance(feedback, Feedback)
        assert feedback.score == 0.7
        assert "hello" in feedback.content
        assert "world" in feedback.content
        assert feedback.feedback_type == FeedbackType.VERIFIER

    @pytest.mark.asyncio
    async def test_simple_loss_match(self) -> None:
        """Simple loss returns 1.0 when output matches target."""
        loss = SimpleLoss()
        feedback = await loss("same", target="same")

        assert feedback.score == 1.0

    @pytest.mark.asyncio
    async def test_loss_without_target(self) -> None:
        """Loss can be called without target argument."""
        loss = SimpleLoss(always_score=0.6)
        feedback = await loss("output only")

        assert feedback.score == 0.6
        assert "None" in feedback.content  # target=None in the output

    @pytest.mark.asyncio
    async def test_loss_with_context(self) -> None:
        """Loss can use context for evaluation."""
        loss = ContextAwareLoss()
        feedback = await loss(
            "some output",
            context={"criteria": "helpfulness"},
        )

        assert "helpfulness" in feedback.content
        assert feedback.metadata["criteria"] == "helpfulness"

    @pytest.mark.asyncio
    async def test_loss_without_context(self) -> None:
        """Loss handles missing context gracefully."""
        loss = ContextAwareLoss()
        feedback = await loss("some output")

        assert "default" in feedback.content


class TestAttachRecord:
    """Tests for Loss._attach_record helper method."""

    def _create_record(self) -> ForwardRecord:
        """Create a minimal ForwardRecord for testing."""
        graph = InferenceGraph(
            nodes={},
            input_ids=[],
            output_ids=[],
        )
        return ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

    @pytest.mark.asyncio
    async def test_attach_record_sets_record(self) -> None:
        """_attach_record attaches ForwardRecord to feedback."""
        loss = SimpleLoss()
        record = self._create_record()

        feedback = await loss("output", target="target", record=record)

        assert feedback._record is record

    @pytest.mark.asyncio
    async def test_attach_record_none(self) -> None:
        """_attach_record does nothing when record is None."""
        loss = SimpleLoss()

        feedback = await loss("output", record=None)

        assert feedback._record is None

    @pytest.mark.asyncio
    async def test_attach_record_enables_backward(self) -> None:
        """Feedback with attached record can call backward()."""
        loss = SimpleLoss()
        record = self._create_record()

        feedback = await loss("output", record=record)

        # Should not raise since record is attached
        await feedback.backward()

    @pytest.mark.asyncio
    async def test_no_record_prevents_backward(self) -> None:
        """Feedback without record cannot call backward()."""
        loss = SimpleLoss()

        feedback = await loss("output")

        with pytest.raises(RuntimeError):
            await feedback.backward()

    def test_attach_record_returns_same_object(self) -> None:
        """_attach_record returns the same feedback object."""
        loss = SimpleLoss()
        record = self._create_record()
        feedback = Feedback(content="Test")

        result = loss._attach_record(feedback, record)

        assert result is feedback

    def test_attach_record_mutates_in_place(self) -> None:
        """_attach_record modifies the feedback object in place."""
        loss = SimpleLoss()
        record = self._create_record()
        feedback = Feedback(content="Test")

        assert feedback._record is None
        loss._attach_record(feedback, record)
        assert feedback._record is record


class TestLossCallSignature:
    """Tests for Loss __call__ method signature."""

    @pytest.mark.asyncio
    async def test_call_positional_args(self) -> None:
        """Loss can be called with positional args."""
        loss = SimpleLoss()
        feedback = await loss("output", "target")
        assert feedback.score == 0.5  # Not equal

    @pytest.mark.asyncio
    async def test_call_keyword_args(self) -> None:
        """Loss can be called with keyword args."""
        loss = SimpleLoss()
        feedback = await loss(output="out", target="target")
        assert feedback.score == 0.5

    @pytest.mark.asyncio
    async def test_call_mixed_args(self) -> None:
        """Loss can be called with mixed positional and keyword args."""
        loss = SimpleLoss()
        record = ForwardRecord(
            graph=InferenceGraph(nodes={}, input_ids=[], output_ids=[]),
            node_inputs={},
            node_outputs={},
            module_map={},
        )
        feedback = await loss(
            "output",
            target="target",
            record=record,
            context={"key": "value"},
        )
        assert feedback._record is record

    @pytest.mark.asyncio
    async def test_record_and_context_are_keyword_only(self) -> None:
        """record and context must be keyword arguments."""
        loss = SimpleLoss()

        # These should work
        await loss("out")
        await loss("out", "target")
        await loss("out", target="target")
        await loss("out", record=None)
        await loss("out", context={})
        await loss("out", record=None, context={})

        # The signature enforces keyword-only after target
        # If someone tries to pass record positionally, it would fail
        # but we can't easily test this at runtime without introspection


class TestMultipleLossInstances:
    """Tests for multiple Loss instances."""

    @pytest.mark.asyncio
    async def test_different_configurations(self) -> None:
        """Different loss instances can have different configurations."""
        loss1 = SimpleLoss(always_score=0.3)
        loss2 = SimpleLoss(always_score=0.9)

        fb1 = await loss1("a", target="b")
        fb2 = await loss2("a", target="b")

        assert fb1.score == 0.3
        assert fb2.score == 0.9

    @pytest.mark.asyncio
    async def test_independent_records(self) -> None:
        """Each loss call can have independent records."""
        loss = SimpleLoss()
        record1 = ForwardRecord(
            graph=InferenceGraph(nodes={}, input_ids=["a"], output_ids=[]),
            node_inputs={},
            node_outputs={},
            module_map={},
        )
        record2 = ForwardRecord(
            graph=InferenceGraph(nodes={}, input_ids=["b"], output_ids=[]),
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        fb1 = await loss("output1", record=record1)
        fb2 = await loss("output2", record=record2)

        assert fb1._record is record1
        assert fb2._record is record2
        assert fb1._record is not fb2._record


# ═══════════════════════════════════════════════════════════════════════════
#  VerifierLoss Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVerifierLoss:
    """Tests for VerifierLoss programmatic evaluation."""

    def test_verifier_loss_creation(self) -> None:
        """VerifierLoss can be created with a verifier function."""

        def simple_verifier(output: Any) -> tuple[bool, str]:
            return True, "OK"

        loss = VerifierLoss(verifier=simple_verifier)
        assert loss.verifier is simple_verifier
        assert loss.success_feedback == "Output passed verification."

    def test_verifier_loss_custom_success_message(self) -> None:
        """VerifierLoss can have a custom success message."""
        loss = VerifierLoss(
            verifier=lambda x: (True, ""),
            success_feedback="Custom success!",
        )
        assert loss.success_feedback == "Custom success!"

    @pytest.mark.asyncio
    async def test_verifier_loss_pass(self) -> None:
        """VerifierLoss returns score 1.0 when verification passes."""

        def check_no_error(output: str) -> tuple[bool, str]:
            if "error" in output.lower():
                return False, "Output contains error"
            return True, "Output is valid"

        loss = VerifierLoss(verifier=check_no_error)
        feedback = await loss("Hello world")

        assert feedback.score == 1.0
        assert feedback.content == "Output passed verification."
        assert feedback.feedback_type == FeedbackType.VERIFIER

    @pytest.mark.asyncio
    async def test_verifier_loss_fail(self) -> None:
        """VerifierLoss returns score 0.0 when verification fails."""

        def check_no_error(output: str) -> tuple[bool, str]:
            if "error" in output.lower():
                return False, "Output contains error"
            return True, "Output is valid"

        loss = VerifierLoss(verifier=check_no_error)
        feedback = await loss("Error: something went wrong")

        assert feedback.score == 0.0
        assert feedback.content == "Output contains error"
        assert feedback.feedback_type == FeedbackType.VERIFIER

    @pytest.mark.asyncio
    async def test_verifier_loss_with_record(self) -> None:
        """VerifierLoss attaches record when provided."""
        loss = VerifierLoss(verifier=lambda x: (True, ""))
        record = ForwardRecord(
            graph=InferenceGraph(nodes={}, input_ids=[], output_ids=[]),
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        feedback = await loss("output", record=record)

        assert feedback._record is record

    @pytest.mark.asyncio
    async def test_verifier_loss_complex_check(self) -> None:
        """VerifierLoss works with complex verification logic."""

        def check_json_and_keys(output: str) -> tuple[bool, str]:
            import json

            try:
                data = json.loads(output)
                if "required_key" not in data:
                    return False, "Missing required_key"
                return True, "Valid JSON with required key"
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {e}"

        loss = VerifierLoss(verifier=check_json_and_keys)

        # Valid JSON with required key
        fb1 = await loss('{"required_key": "value"}')
        assert fb1.score == 1.0

        # Valid JSON but missing key
        fb2 = await loss('{"other_key": "value"}')
        assert fb2.score == 0.0
        assert "Missing required_key" in fb2.content

        # Invalid JSON
        fb3 = await loss("not json at all")
        assert fb3.score == 0.0
        assert "Invalid JSON" in fb3.content

    @pytest.mark.asyncio
    async def test_verifier_loss_ignores_target(self) -> None:
        """VerifierLoss ignores target parameter."""
        loss = VerifierLoss(verifier=lambda x: (True, ""))

        # Target is ignored, output determines result
        feedback = await loss("output", target="completely different")

        assert feedback.score == 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  LLMJudge Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLLMJudge:
    """Tests for LLMJudge LLM-based evaluation."""

    def test_llm_judge_creation(self) -> None:
        """LLMJudge can be created with alias and criteria."""
        judge = LLMJudge(alias="test-judge", criteria="helpfulness")

        assert judge.criteria == "helpfulness"
        assert judge.judge.alias == "test-judge"

    def test_llm_judge_default_alias(self) -> None:
        """LLMJudge has default alias 'judge'."""
        judge = LLMJudge()

        assert judge.judge.alias == "judge"
        assert judge.criteria is None

    def test_llm_judge_system_prompt(self) -> None:
        """LLMJudge has appropriate system prompt."""
        judge = LLMJudge()

        assert judge.judge.system_prompt is not None
        system_prompt = judge.judge.system_prompt.value
        assert "critical reviewer" in system_prompt
        assert "actionable" in system_prompt

    def test_llm_judge_bind(self) -> None:
        """LLMJudge.bind() configures resources."""
        judge = LLMJudge(alias="test-judge")
        mock_resources = MagicMock()

        result = judge.bind(mock_resources)

        # Returns self for chaining
        assert result is judge
        # judge module should have bind called
        assert judge.judge._bound_resources is mock_resources

    @pytest.mark.asyncio
    async def test_llm_judge_call(self) -> None:
        """LLMJudge calls internal LLM and returns feedback."""
        judge = LLMJudge(alias="test-judge", criteria="quality")

        # Mock the judge module's __call__
        judge.judge = AsyncMock(return_value="The output could be improved by...")

        feedback = await judge("test output", target="expected behavior")

        # Verify the judge was called
        judge.judge.assert_called_once()
        call_prompt = judge.judge.call_args[0][0]
        assert "test output" in call_prompt
        assert "expected behavior" in call_prompt
        assert "quality" in call_prompt

        # Verify feedback
        assert feedback.content == "The output could be improved by..."
        assert feedback.score is None  # Freeform feedback has no score
        assert feedback.feedback_type == FeedbackType.LLM_JUDGE

    @pytest.mark.asyncio
    async def test_llm_judge_with_context(self) -> None:
        """LLMJudge includes context in prompt."""
        judge = LLMJudge()
        judge.judge = AsyncMock(return_value="Feedback")

        await judge("output", context={"task": "summarization"})

        call_prompt = judge.judge.call_args[0][0]
        assert "summarization" in call_prompt

    @pytest.mark.asyncio
    async def test_llm_judge_with_record(self) -> None:
        """LLMJudge attaches record when provided."""
        judge = LLMJudge()
        judge.judge = AsyncMock(return_value="Feedback")
        record = ForwardRecord(
            graph=InferenceGraph(nodes={}, input_ids=[], output_ids=[]),
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        feedback = await judge("output", record=record)

        assert feedback._record is record

    @pytest.mark.asyncio
    async def test_llm_judge_prompt_construction(self) -> None:
        """LLMJudge builds prompt correctly with all components."""
        judge = LLMJudge(criteria="clarity and brevity")
        judge.judge = AsyncMock(return_value="Feedback")

        await judge(
            "The quick brown fox",
            target="A short sentence about animals",
            context={"source": "user query"},
        )

        call_prompt = judge.judge.call_args[0][0]
        # Check all components are included
        assert "Output to critique:" in call_prompt
        assert "The quick brown fox" in call_prompt
        assert "Expected behavior:" in call_prompt
        assert "A short sentence about animals" in call_prompt
        assert "Context:" in call_prompt
        assert "user query" in call_prompt
        assert "Focus areas:" in call_prompt
        assert "clarity and brevity" in call_prompt


# ═══════════════════════════════════════════════════════════════════════════
#  CompositeLoss Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositeLoss:
    """Tests for CompositeLoss weighted combination."""

    def test_composite_loss_creation(self) -> None:
        """CompositeLoss can be created with losses and weights."""
        loss1 = VerifierLoss(verifier=lambda x: (True, ""))
        loss2 = VerifierLoss(verifier=lambda x: (True, ""))

        composite = CompositeLoss(
            [
                (loss1, 0.3),
                (loss2, 0.7),
            ]
        )

        assert len(composite.losses) == 2
        assert composite.aggregator is None

    def test_composite_loss_with_aggregator(self) -> None:
        """CompositeLoss can have an LLM aggregator."""
        loss1 = VerifierLoss(verifier=lambda x: (True, ""))
        mock_aggregator = MagicMock()

        composite = CompositeLoss(
            losses=[(loss1, 1.0)],
            aggregator=mock_aggregator,
        )

        assert composite.aggregator is mock_aggregator

    @pytest.mark.asyncio
    async def test_composite_loss_simple_aggregate(self) -> None:
        """CompositeLoss concatenates feedback when no aggregator."""
        loss1 = VerifierLoss(
            verifier=lambda x: (True, ""),
            success_feedback="Format OK",
        )
        loss2 = VerifierLoss(
            verifier=lambda x: (False, "Missing keyword"),
        )

        composite = CompositeLoss(
            [
                (loss1, 0.3),
                (loss2, 0.7),
            ]
        )

        feedback = await composite("some output")

        # Content should contain both feedback messages
        assert "[Weight: 0.3]" in feedback.content
        assert "Format OK" in feedback.content
        assert "[Weight: 0.7]" in feedback.content
        assert "Missing keyword" in feedback.content
        assert feedback.feedback_type == FeedbackType.COMPOSITE

    @pytest.mark.asyncio
    async def test_composite_loss_weighted_score(self) -> None:
        """CompositeLoss computes weighted average score."""
        # loss1: score 1.0, weight 0.3 => contribution 0.3
        loss1 = VerifierLoss(verifier=lambda x: (True, ""))
        # loss2: score 0.0, weight 0.7 => contribution 0.0
        loss2 = VerifierLoss(verifier=lambda x: (False, "Failed"))

        composite = CompositeLoss(
            [
                (loss1, 0.3),
                (loss2, 0.7),
            ]
        )

        feedback = await composite("output")

        # (1.0 * 0.3 + 0.0 * 0.7) / (0.3 + 0.7) = 0.3
        assert feedback.score == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_composite_loss_all_pass(self) -> None:
        """CompositeLoss score is 1.0 when all pass."""
        loss1 = VerifierLoss(verifier=lambda x: (True, ""))
        loss2 = VerifierLoss(verifier=lambda x: (True, ""))

        composite = CompositeLoss(
            [
                (loss1, 0.5),
                (loss2, 0.5),
            ]
        )

        feedback = await composite("output")

        assert feedback.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_composite_loss_with_record(self) -> None:
        """CompositeLoss attaches record to final feedback."""
        loss = VerifierLoss(verifier=lambda x: (True, ""))
        composite = CompositeLoss([(loss, 1.0)])
        record = ForwardRecord(
            graph=InferenceGraph(nodes={}, input_ids=[], output_ids=[]),
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        feedback = await composite("output", record=record)

        assert feedback._record is record

    @pytest.mark.asyncio
    async def test_composite_loss_no_scores(self) -> None:
        """CompositeLoss returns None score when no components have scores."""
        # Create a loss that returns no score
        mock_loss = AsyncMock()
        mock_loss.return_value = Feedback(content="No score", score=None)

        composite = CompositeLoss([(mock_loss, 1.0)])

        feedback = await composite("output")

        assert feedback.score is None

    @pytest.mark.asyncio
    async def test_composite_loss_mixed_scores(self) -> None:
        """CompositeLoss handles mix of scored and unscored feedback."""
        # Scored loss
        loss1 = VerifierLoss(verifier=lambda x: (True, ""))  # score 1.0

        # Unscored loss (mock)
        mock_loss = AsyncMock()
        mock_loss.return_value = Feedback(content="Unscored", score=None)

        composite = CompositeLoss(
            [
                (loss1, 0.3),
                (mock_loss, 0.7),
            ]
        )

        feedback = await composite("output")

        # Only loss1 has a score, so average is just loss1's score
        # (1.0 * 0.3) / 0.3 = 1.0
        assert feedback.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_composite_loss_llm_aggregate(self) -> None:
        """CompositeLoss uses aggregator when provided."""
        loss1 = VerifierLoss(
            verifier=lambda x: (True, ""),
            success_feedback="Check 1 passed",
        )
        loss2 = VerifierLoss(
            verifier=lambda x: (True, ""),
            success_feedback="Check 2 passed",
        )

        # Mock aggregator
        mock_aggregator = AsyncMock(return_value="Synthesized feedback summary")

        composite = CompositeLoss(
            losses=[(loss1, 0.5), (loss2, 0.5)],
            aggregator=mock_aggregator,
        )

        feedback = await composite("output")

        # Aggregator should be called with prompt containing feedback
        mock_aggregator.assert_called_once()
        call_prompt = mock_aggregator.call_args[0][0]
        assert "Check 1 passed" in call_prompt
        assert "Check 2 passed" in call_prompt
        assert "weight: 0.5" in call_prompt

        # Content should be the synthesized feedback
        assert feedback.content == "Synthesized feedback summary"

    def test_composite_loss_bind(self) -> None:
        """CompositeLoss.bind() binds all components."""
        # Mock losses with bind methods
        mock_loss1 = MagicMock()
        mock_loss1.bind = MagicMock(return_value=mock_loss1)

        mock_loss2 = MagicMock()
        mock_loss2.bind = MagicMock(return_value=mock_loss2)

        mock_aggregator = MagicMock()
        mock_aggregator.bind = MagicMock(return_value=mock_aggregator)

        composite = CompositeLoss(
            losses=[(mock_loss1, 0.5), (mock_loss2, 0.5)],
            aggregator=mock_aggregator,
        )

        mock_resources = MagicMock()
        result = composite.bind(mock_resources)

        # Returns self for chaining
        assert result is composite

        # All components should have bind called
        mock_loss1.bind.assert_called_once_with(mock_resources)
        mock_loss2.bind.assert_called_once_with(mock_resources)
        mock_aggregator.bind.assert_called_once_with(mock_resources)

    def test_composite_loss_bind_no_aggregator(self) -> None:
        """CompositeLoss.bind() works without aggregator."""
        mock_loss = MagicMock()
        mock_loss.bind = MagicMock(return_value=mock_loss)

        composite = CompositeLoss(losses=[(mock_loss, 1.0)])

        mock_resources = MagicMock()
        composite.bind(mock_resources)

        mock_loss.bind.assert_called_once_with(mock_resources)

    def test_composite_loss_bind_loss_without_bind(self) -> None:
        """CompositeLoss.bind() handles losses without bind method."""
        # VerifierLoss doesn't have bind method
        loss = VerifierLoss(verifier=lambda x: (True, ""))
        composite = CompositeLoss([(loss, 1.0)])

        # Should not raise
        composite.bind(MagicMock())

    @pytest.mark.asyncio
    async def test_composite_loss_passes_target_and_context(self) -> None:
        """CompositeLoss passes target and context to sub-losses."""
        mock_loss = AsyncMock()
        mock_loss.return_value = Feedback(content="Test", score=1.0)

        composite = CompositeLoss([(mock_loss, 1.0)])

        await composite(
            "output",
            target="expected",
            context={"key": "value"},
        )

        # Verify target and context were passed
        mock_loss.assert_called_once()
        _, kwargs = mock_loss.call_args
        assert kwargs.get("target") is None or mock_loss.call_args[0][1] == "expected"
        # Check context was passed
        call_args = mock_loss.call_args
        assert call_args[1].get("context") == {"key": "value"}


# ═══════════════════════════════════════════════════════════════════════════
#  Structured Output Schema Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuredOutputSchemas:
    """Tests for structured output dataclasses."""

    def test_rubric_level_creation(self) -> None:
        """RubricLevel can be created with score, label, description."""
        level = RubricLevel(
            score=5,
            label="Excellent",
            description="Exceptionally clear and complete",
        )

        assert level.score == 5
        assert level.label == "Excellent"
        assert level.description == "Exceptionally clear and complete"

    def test_rubric_response_creation(self) -> None:
        """RubricResponse can be created with all fields."""
        response = RubricResponse(
            score=4,
            justification="The output was clear",
            feedback="Consider adding more examples",
        )

        assert response.score == 4
        assert response.justification == "The output was clear"
        assert response.feedback == "Consider adding more examples"

    def test_preference_response_creation(self) -> None:
        """PreferenceResponse can be created with all fields."""
        response = PreferenceResponse(
            winner="A",
            reason="Output A was more concise",
            a_strengths="Clear and concise",
            a_weaknesses="Could use more detail",
            b_strengths="Very detailed",
            b_weaknesses="Too verbose",
        )

        assert response.winner == "A"
        assert response.reason == "Output A was more concise"
        assert response.a_strengths == "Clear and concise"

    def test_ranking_response_creation(self) -> None:
        """RankingResponse can be created with all fields."""
        response = RankingResponse(
            ranking=[3, 1, 2],
            best_qualities="Most accurate and helpful",
            worst_issues="Too brief and missing context",
            comparison="Output 3 was best due to accuracy",
        )

        assert response.ranking == [3, 1, 2]
        assert response.best_qualities == "Most accurate and helpful"


# ═══════════════════════════════════════════════════════════════════════════
#  HumanFeedbackLoss Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHumanFeedbackLoss:
    """Tests for HumanFeedbackLoss human-in-the-loop evaluation."""

    def test_human_feedback_loss_creation(self) -> None:
        """HumanFeedbackLoss can be created with defaults."""
        loss = HumanFeedbackLoss()

        assert loss.prompt_template is None
        assert loss.show_context is True

    def test_human_feedback_loss_custom_template(self) -> None:
        """HumanFeedbackLoss accepts custom prompt template."""
        loss = HumanFeedbackLoss(
            prompt_template="Rate this: {output}",
            show_context=False,
        )

        assert loss.prompt_template == "Rate this: {output}"
        assert loss.show_context is False

    @pytest.mark.asyncio
    async def test_human_feedback_loss_collects_input(self) -> None:
        """HumanFeedbackLoss collects feedback from stdin."""
        loss = HumanFeedbackLoss()

        # Mock input() to return feedback then empty line to finish
        with patch("builtins.input", side_effect=["Great output!", ""]):
            with patch("builtins.print"):  # Suppress output
                feedback = await loss("test output")

        assert feedback.content == "Great output!"
        assert feedback.score is None
        assert feedback.feedback_type == FeedbackType.HUMAN

    @pytest.mark.asyncio
    async def test_human_feedback_loss_multiline_input(self) -> None:
        """HumanFeedbackLoss collects multiline feedback."""
        loss = HumanFeedbackLoss()

        with patch("builtins.input", side_effect=["Line 1", "Line 2", ""]):
            with patch("builtins.print"):
                feedback = await loss("test output")

        assert feedback.content == "Line 1\nLine 2"

    @pytest.mark.asyncio
    async def test_human_feedback_loss_empty_input(self) -> None:
        """HumanFeedbackLoss handles empty feedback."""
        loss = HumanFeedbackLoss()

        with patch("builtins.input", side_effect=[""]):
            with patch("builtins.print"):
                feedback = await loss("test output")

        assert feedback.content == "No feedback provided."

    @pytest.mark.asyncio
    async def test_human_feedback_loss_with_record(self) -> None:
        """HumanFeedbackLoss attaches record."""
        loss = HumanFeedbackLoss()
        record = ForwardRecord(
            graph=InferenceGraph(nodes={}, input_ids=[], output_ids=[]),
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        with patch("builtins.input", side_effect=["feedback", ""]):
            with patch("builtins.print"):
                feedback = await loss("output", record=record)

        assert feedback._record is record


# ═══════════════════════════════════════════════════════════════════════════
#  LLMRubricLoss Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLLMRubricLoss:
    """Tests for LLMRubricLoss rubric-based evaluation."""

    def _create_rubric(self) -> list[RubricLevel]:
        """Create a standard 5-point rubric for testing."""
        return [
            RubricLevel(1, "Poor", "Fails completely"),
            RubricLevel(2, "Below Average", "Partially addresses"),
            RubricLevel(3, "Average", "Adequately addresses"),
            RubricLevel(4, "Good", "Thoroughly addresses"),
            RubricLevel(5, "Excellent", "Exceptionally addresses"),
        ]

    def test_llm_rubric_loss_creation(self) -> None:
        """LLMRubricLoss can be created with criteria and rubric."""
        rubric = self._create_rubric()
        loss = LLMRubricLoss(
            criteria="helpfulness",
            rubric=rubric,
            alias="test-judge",
        )

        assert loss.criteria == "helpfulness"
        assert len(loss.rubric) == 5
        assert loss.judge.alias == "test-judge"

    def test_llm_rubric_loss_sorts_rubric(self) -> None:
        """LLMRubricLoss sorts rubric by score."""
        # Create rubric out of order
        rubric = [
            RubricLevel(3, "Medium", "Medium"),
            RubricLevel(1, "Low", "Low"),
            RubricLevel(5, "High", "High"),
        ]
        loss = LLMRubricLoss(criteria="test", rubric=rubric)

        # Should be sorted
        assert loss.rubric[0].score == 1
        assert loss.rubric[1].score == 3
        assert loss.rubric[2].score == 5

    def test_llm_rubric_loss_system_prompt(self) -> None:
        """LLMRubricLoss generates appropriate system prompt."""
        rubric = self._create_rubric()
        loss = LLMRubricLoss(criteria="clarity", rubric=rubric)

        prompt = loss._build_system_prompt()
        assert "clarity" in prompt
        assert "Rating Scale:" in prompt
        assert "1 - Poor" in prompt
        assert "5 - Excellent" in prompt

    def test_llm_rubric_loss_bind(self) -> None:
        """LLMRubricLoss.bind() configures resources."""
        loss = LLMRubricLoss(criteria="test", rubric=self._create_rubric())
        mock_resources = MagicMock()

        result = loss.bind(mock_resources)

        assert result is loss
        assert loss.judge._bound_resources is mock_resources

    @pytest.mark.asyncio
    async def test_llm_rubric_loss_call_with_dict_response(self) -> None:
        """LLMRubricLoss handles dict response from LLM."""
        loss = LLMRubricLoss(criteria="test", rubric=self._create_rubric())

        # Mock judge to return dict response
        loss.judge = AsyncMock(
            return_value={
                "score": 4,
                "justification": "Well done",
                "feedback": "Add more examples",
            }
        )

        feedback = await loss("test output")

        # Score 4 on 1-5 scale: (4-1)/(5-1) = 0.75
        assert feedback.score == pytest.approx(0.75)
        assert "Well done" in feedback.content
        assert "Add more examples" in feedback.content
        assert feedback.feedback_type == FeedbackType.LLM_JUDGE
        assert feedback.metadata["raw_score"] == 4

    @pytest.mark.asyncio
    async def test_llm_rubric_loss_call_with_object_response(self) -> None:
        """LLMRubricLoss handles object response from LLM."""
        loss = LLMRubricLoss(criteria="test", rubric=self._create_rubric())

        # Mock judge to return RubricResponse object
        loss.judge = AsyncMock(
            return_value=RubricResponse(
                score=5,
                justification="Perfect",
                feedback="No improvements needed",
            )
        )

        feedback = await loss("test output")

        # Score 5 on 1-5 scale: (5-1)/(5-1) = 1.0
        assert feedback.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_llm_rubric_loss_with_record(self) -> None:
        """LLMRubricLoss attaches record."""
        loss = LLMRubricLoss(criteria="test", rubric=self._create_rubric())
        loss.judge = AsyncMock(
            return_value={"score": 3, "justification": "OK", "feedback": "OK"}
        )
        record = ForwardRecord(
            graph=InferenceGraph(nodes={}, input_ids=[], output_ids=[]),
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        feedback = await loss("output", record=record)

        assert feedback._record is record


# ═══════════════════════════════════════════════════════════════════════════
#  HumanRubricLoss Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHumanRubricLoss:
    """Tests for HumanRubricLoss human rubric-based evaluation."""

    def _create_rubric(self) -> list[RubricLevel]:
        """Create a standard 5-point rubric for testing."""
        return [
            RubricLevel(1, "Poor", "Fails completely"),
            RubricLevel(2, "Below Average", "Partially addresses"),
            RubricLevel(3, "Average", "Adequately addresses"),
            RubricLevel(4, "Good", "Thoroughly addresses"),
            RubricLevel(5, "Excellent", "Exceptionally addresses"),
        ]

    def test_human_rubric_loss_creation(self) -> None:
        """HumanRubricLoss can be created with criteria and rubric."""
        rubric = self._create_rubric()
        loss = HumanRubricLoss(
            criteria="helpfulness",
            rubric=rubric,
            require_feedback=True,
        )

        assert loss.criteria == "helpfulness"
        assert len(loss.rubric) == 5
        assert loss.require_feedback is True

    @pytest.mark.asyncio
    async def test_human_rubric_loss_collects_score(self) -> None:
        """HumanRubricLoss collects score from user."""
        loss = HumanRubricLoss(
            criteria="test",
            rubric=self._create_rubric(),
            require_feedback=False,
        )

        with patch("builtins.input", side_effect=["4"]):
            with patch("builtins.print"):
                feedback = await loss("test output")

        # Score 4 on 1-5 scale: (4-1)/(5-1) = 0.75
        assert feedback.score == pytest.approx(0.75)
        assert feedback.feedback_type == FeedbackType.HUMAN
        assert feedback.metadata["raw_score"] == 4

    @pytest.mark.asyncio
    async def test_human_rubric_loss_collects_feedback(self) -> None:
        """HumanRubricLoss collects written feedback when required."""
        loss = HumanRubricLoss(
            criteria="test",
            rubric=self._create_rubric(),
            require_feedback=True,
        )

        # First input is score, then feedback lines, then empty to finish
        with patch("builtins.input", side_effect=["3", "Could be better", ""]):
            with patch("builtins.print"):
                feedback = await loss("test output")

        assert "Could be better" in feedback.content

    @pytest.mark.asyncio
    async def test_human_rubric_loss_validates_score(self) -> None:
        """HumanRubricLoss validates score is in range."""
        loss = HumanRubricLoss(
            criteria="test",
            rubric=self._create_rubric(),
            require_feedback=False,
        )

        # Invalid scores followed by valid score
        with patch("builtins.input", side_effect=["0", "6", "invalid", "3"]):
            with patch("builtins.print"):
                feedback = await loss("test output")

        assert feedback.score == pytest.approx(0.5)  # Score 3 normalized


# ═══════════════════════════════════════════════════════════════════════════
#  ContrastiveLoss Tests
# ═══════════════════════════════════════════════════════════════════════════


class ConcreteContrastiveLoss(ContrastiveLoss):
    """Concrete implementation for testing ContrastiveLoss base."""

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        return Feedback(content="Test", score=0.5)


class TestContrastiveLoss:
    """Tests for ContrastiveLoss base class."""

    def test_contrastive_loss_generate_feedback(self) -> None:
        """ContrastiveLoss generates contrastive feedback."""
        loss = ConcreteContrastiveLoss()

        feedback = loss._generate_contrastive_feedback(
            winner="Great output",
            loser="Poor output",
            reason="More concise",
        )

        assert "More concise" in feedback
        assert "Preferred output characteristics" in feedback
        assert "Rejected output weaknesses" in feedback

    def test_contrastive_loss_summarize_output_short(self) -> None:
        """ContrastiveLoss summarizes short output as-is."""
        loss = ConcreteContrastiveLoss()

        result = loss._summarize_output("Short text")

        assert result == "Short text"

    def test_contrastive_loss_summarize_output_long(self) -> None:
        """ContrastiveLoss truncates long output."""
        loss = ConcreteContrastiveLoss()
        long_text = "x" * 300

        result = loss._summarize_output(long_text)

        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")


# ═══════════════════════════════════════════════════════════════════════════
#  LLMPreferenceLoss Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLLMPreferenceLoss:
    """Tests for LLMPreferenceLoss pairwise comparison."""

    def test_llm_preference_loss_creation(self) -> None:
        """LLMPreferenceLoss can be created with criteria."""
        loss = LLMPreferenceLoss(criteria="quality", alias="test-judge")

        assert loss.criteria == "quality"
        assert loss.judge.alias == "test-judge"

    def test_llm_preference_loss_bind(self) -> None:
        """LLMPreferenceLoss.bind() configures resources."""
        loss = LLMPreferenceLoss(criteria="test")
        mock_resources = MagicMock()

        result = loss.bind(mock_resources)

        assert result is loss
        assert loss.judge._bound_resources is mock_resources

    @pytest.mark.asyncio
    async def test_llm_preference_loss_compare_a_wins(self) -> None:
        """LLMPreferenceLoss returns correct feedback when A wins."""
        loss = LLMPreferenceLoss(criteria="quality")

        loss.judge = AsyncMock(
            return_value={
                "winner": "A",
                "reason": "More concise",
                "a_strengths": "Clear and brief",
                "a_weaknesses": "Could be more detailed",
                "b_strengths": "Very detailed",
                "b_weaknesses": "Too verbose",
            }
        )

        fb_a, fb_b = await loss.compare("Output A", "Output B")

        assert fb_a.score == 1.0
        assert "Preferred" in fb_a.content
        assert fb_b.score == 0.0
        assert "better because" in fb_b.content

    @pytest.mark.asyncio
    async def test_llm_preference_loss_compare_b_wins(self) -> None:
        """LLMPreferenceLoss returns correct feedback when B wins."""
        loss = LLMPreferenceLoss(criteria="quality")

        loss.judge = AsyncMock(
            return_value=PreferenceResponse(
                winner="B",
                reason="More accurate",
                a_strengths="Concise",
                a_weaknesses="Inaccurate",
                b_strengths="Accurate",
                b_weaknesses="Verbose",
            )
        )

        fb_a, fb_b = await loss.compare("Output A", "Output B")

        assert fb_a.score == 0.0
        assert fb_b.score == 1.0
        assert "Preferred" in fb_b.content

    @pytest.mark.asyncio
    async def test_llm_preference_loss_call_requires_target(self) -> None:
        """LLMPreferenceLoss.__call__ requires target."""
        loss = LLMPreferenceLoss(criteria="test")

        with pytest.raises(ValueError) as exc_info:
            await loss("output")

        assert "requires target" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_llm_preference_loss_call_with_target(self) -> None:
        """LLMPreferenceLoss.__call__ compares output with target."""
        loss = LLMPreferenceLoss(criteria="test")
        loss.judge = AsyncMock(
            return_value={
                "winner": "A",
                "reason": "Better",
                "a_strengths": "Good",
                "a_weaknesses": "None",
                "b_strengths": "OK",
                "b_weaknesses": "Bad",
            }
        )

        feedback = await loss("output", target="baseline")

        assert feedback.score == 1.0

    @pytest.mark.asyncio
    async def test_llm_preference_loss_attaches_records(self) -> None:
        """LLMPreferenceLoss attaches records to feedback."""
        loss = LLMPreferenceLoss(criteria="test")
        loss.judge = AsyncMock(
            return_value={
                "winner": "A",
                "reason": "Better",
                "a_strengths": "Good",
                "a_weaknesses": "None",
                "b_strengths": "OK",
                "b_weaknesses": "Bad",
            }
        )

        record_a = ForwardRecord(
            graph=InferenceGraph(nodes={}, input_ids=["a"], output_ids=[]),
            node_inputs={},
            node_outputs={},
            module_map={},
        )
        record_b = ForwardRecord(
            graph=InferenceGraph(nodes={}, input_ids=["b"], output_ids=[]),
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        fb_a, fb_b = await loss.compare("A", "B", record_a=record_a, record_b=record_b)

        assert fb_a._record is record_a
        assert fb_b._record is record_b


# ═══════════════════════════════════════════════════════════════════════════
#  HumanPreferenceLoss Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHumanPreferenceLoss:
    """Tests for HumanPreferenceLoss pairwise comparison."""

    def test_human_preference_loss_creation(self) -> None:
        """HumanPreferenceLoss can be created with criteria."""
        loss = HumanPreferenceLoss(criteria="quality", require_reason=False)

        assert loss.criteria == "quality"
        assert loss.require_reason is False

    @pytest.mark.asyncio
    async def test_human_preference_loss_compare_a_wins(self) -> None:
        """HumanPreferenceLoss returns correct feedback when user picks A."""
        loss = HumanPreferenceLoss(criteria="quality", require_reason=True)

        with patch("builtins.input", side_effect=["A", "It's clearer", ""]):
            with patch("builtins.print"):
                fb_a, fb_b = await loss.compare("Output A", "Output B")

        assert fb_a.score == 1.0
        assert "Preferred by human" in fb_a.content
        assert fb_b.score == 0.0
        assert fb_a.feedback_type == FeedbackType.HUMAN

    @pytest.mark.asyncio
    async def test_human_preference_loss_compare_b_wins(self) -> None:
        """HumanPreferenceLoss returns correct feedback when user picks B."""
        loss = HumanPreferenceLoss(criteria="quality", require_reason=False)

        with patch("builtins.input", side_effect=["b"]):  # lowercase works
            with patch("builtins.print"):
                fb_a, fb_b = await loss.compare("Output A", "Output B")

        assert fb_a.score == 0.0
        assert fb_b.score == 1.0

    @pytest.mark.asyncio
    async def test_human_preference_loss_validates_input(self) -> None:
        """HumanPreferenceLoss validates A/B input."""
        loss = HumanPreferenceLoss(criteria="test", require_reason=False)

        # Invalid inputs followed by valid
        with patch("builtins.input", side_effect=["C", "invalid", "A"]):
            with patch("builtins.print"):
                fb_a, fb_b = await loss.compare("Output A", "Output B")

        assert fb_a.score == 1.0

    @pytest.mark.asyncio
    async def test_human_preference_loss_call_requires_target(self) -> None:
        """HumanPreferenceLoss.__call__ requires target."""
        loss = HumanPreferenceLoss(criteria="test")

        with pytest.raises(ValueError) as exc_info:
            await loss("output")

        assert "requires target" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════
#  LLMRankingLoss Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLLMRankingLoss:
    """Tests for LLMRankingLoss n-way ranking."""

    def test_llm_ranking_loss_creation(self) -> None:
        """LLMRankingLoss can be created with criteria and n."""
        loss = LLMRankingLoss(criteria="quality", n=4, alias="test-judge")

        assert loss.criteria == "quality"
        assert loss.n == 4
        assert loss.judge.alias == "test-judge"

    def test_llm_ranking_loss_bind(self) -> None:
        """LLMRankingLoss.bind() configures resources."""
        loss = LLMRankingLoss(criteria="test")
        mock_resources = MagicMock()

        result = loss.bind(mock_resources)

        assert result is loss
        assert loss.judge._bound_resources is mock_resources

    @pytest.mark.asyncio
    async def test_llm_ranking_loss_rank(self) -> None:
        """LLMRankingLoss ranks multiple outputs."""
        loss = LLMRankingLoss(criteria="quality")

        loss.judge = AsyncMock(
            return_value={
                "ranking": [2, 3, 1],  # 1-indexed: output 2 is best
                "best_qualities": "Most accurate",
                "worst_issues": "Too brief",
                "comparison": "Output 2 was most helpful",
            }
        )

        feedbacks = await loss.rank(["Output 1", "Output 2", "Output 3"])

        assert len(feedbacks) == 3
        # Output at index 1 (Output 2) is ranked #1
        assert feedbacks[1].score == 1.0
        assert "#1 (best)" in feedbacks[1].content
        # Output at index 2 (Output 3) is ranked #2
        assert feedbacks[2].score == 0.5
        # Output at index 0 (Output 1) is ranked #3 (worst)
        assert feedbacks[0].score == 0.0
        assert "(worst)" in feedbacks[0].content

    @pytest.mark.asyncio
    async def test_llm_ranking_loss_rank_with_object_response(self) -> None:
        """LLMRankingLoss handles RankingResponse object."""
        loss = LLMRankingLoss(criteria="quality")

        loss.judge = AsyncMock(
            return_value=RankingResponse(
                ranking=[1, 2],
                best_qualities="Clear",
                worst_issues="Verbose",
                comparison="First was better",
            )
        )

        feedbacks = await loss.rank(["A", "B"])

        assert feedbacks[0].score == 1.0
        assert feedbacks[1].score == 0.0

    @pytest.mark.asyncio
    async def test_llm_ranking_loss_requires_two_outputs(self) -> None:
        """LLMRankingLoss requires at least 2 outputs."""
        loss = LLMRankingLoss(criteria="test")

        with pytest.raises(ValueError) as exc_info:
            await loss.rank(["only one"])

        assert "at least 2" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_llm_ranking_loss_call_requires_target(self) -> None:
        """LLMRankingLoss.__call__ requires target."""
        loss = LLMRankingLoss(criteria="test")

        with pytest.raises(ValueError) as exc_info:
            await loss("output")

        assert "requires target" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_llm_ranking_loss_call_with_target(self) -> None:
        """LLMRankingLoss.__call__ ranks output against target."""
        loss = LLMRankingLoss(criteria="test")
        loss.judge = AsyncMock(
            return_value={
                "ranking": [1, 2],
                "best_qualities": "Good",
                "worst_issues": "Bad",
                "comparison": "First is better",
            }
        )

        feedback = await loss("output", target="baseline")

        assert feedback.score == 1.0

    @pytest.mark.asyncio
    async def test_llm_ranking_loss_call_with_list_target(self) -> None:
        """LLMRankingLoss.__call__ handles list of targets."""
        loss = LLMRankingLoss(criteria="test")
        loss.judge = AsyncMock(
            return_value={
                "ranking": [2, 1, 3],  # Second (first target) is best
                "best_qualities": "Accurate",
                "worst_issues": "Brief",
                "comparison": "Varied quality",
            }
        )

        feedback = await loss("output", target=["baseline1", "baseline2"])

        # Output is at index 0, ranked #2
        assert feedback.score == 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  HumanRankingLoss Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHumanRankingLoss:
    """Tests for HumanRankingLoss n-way ranking."""

    def test_human_ranking_loss_creation(self) -> None:
        """HumanRankingLoss can be created with criteria and n."""
        loss = HumanRankingLoss(criteria="quality", n=4, require_feedback=False)

        assert loss.criteria == "quality"
        assert loss.n == 4
        assert loss.require_feedback is False

    @pytest.mark.asyncio
    async def test_human_ranking_loss_rank(self) -> None:
        """HumanRankingLoss collects ranking from user."""
        loss = HumanRankingLoss(criteria="quality", require_feedback=False)

        # User ranks: 2 is best, then 3, then 1
        with patch("builtins.input", side_effect=["2,3,1"]):
            with patch("builtins.print"):
                feedbacks = await loss.rank(["Output 1", "Output 2", "Output 3"])

        assert len(feedbacks) == 3
        # Output 2 (index 1) is ranked #1
        assert feedbacks[1].score == 1.0
        assert "#1 (best)" in feedbacks[1].content
        # Output 1 (index 0) is ranked #3 (worst)
        assert feedbacks[0].score == 0.0
        assert "(worst)" in feedbacks[0].content

    @pytest.mark.asyncio
    async def test_human_ranking_loss_with_feedback(self) -> None:
        """HumanRankingLoss collects written feedback."""
        loss = HumanRankingLoss(criteria="test", require_feedback=True)

        with patch("builtins.input", side_effect=["1,2", "First was clearer", ""]):
            with patch("builtins.print"):
                feedbacks = await loss.rank(["A", "B"])

        assert "First was clearer" in feedbacks[0].content

    @pytest.mark.asyncio
    async def test_human_ranking_loss_validates_ranking(self) -> None:
        """HumanRankingLoss validates ranking input."""
        loss = HumanRankingLoss(criteria="test", require_feedback=False)

        # Invalid rankings followed by valid
        with patch("builtins.input", side_effect=["1", "1,1", "invalid", "1,2"]):
            with patch("builtins.print"):
                feedbacks = await loss.rank(["A", "B"])

        assert len(feedbacks) == 2

    @pytest.mark.asyncio
    async def test_human_ranking_loss_requires_two_outputs(self) -> None:
        """HumanRankingLoss requires at least 2 outputs."""
        loss = HumanRankingLoss(criteria="test")

        with pytest.raises(ValueError) as exc_info:
            await loss.rank(["only one"])

        assert "at least 2" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_human_ranking_loss_call_requires_target(self) -> None:
        """HumanRankingLoss.__call__ requires target."""
        loss = HumanRankingLoss(criteria="test")

        with pytest.raises(ValueError) as exc_info:
            await loss("output")

        assert "requires target" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════
#  Loss.batch() Tests (PR-068b)
# ═══════════════════════════════════════════════════════════════════════════


class TestLossBatch:
    """Tests for Loss.batch() concurrent batch evaluation."""

    @pytest.mark.asyncio
    async def test_batch_evaluates_all_outputs(self) -> None:
        """batch() evaluates all outputs and returns list of Feedback."""
        loss = SimpleLoss(always_score=0.5)

        feedbacks = await loss.batch(["out1", "out2", "out3"])

        assert len(feedbacks) == 3
        for fb in feedbacks:
            assert isinstance(fb, Feedback)
            assert fb.score == 0.5

    @pytest.mark.asyncio
    async def test_batch_with_targets(self) -> None:
        """batch() passes targets to each evaluation."""
        loss = SimpleLoss()

        # Matching outputs with targets should get score 1.0
        feedbacks = await loss.batch(
            outputs=["a", "b", "c"],
            targets=["a", "b", "wrong"],
        )

        assert feedbacks[0].score == 1.0  # a == a
        assert feedbacks[1].score == 1.0  # b == b
        assert feedbacks[2].score == 0.5  # c != wrong

    @pytest.mark.asyncio
    async def test_batch_with_records(self) -> None:
        """batch() attaches records to feedback."""
        loss = SimpleLoss()
        record1 = ForwardRecord(
            graph=InferenceGraph(nodes={}, input_ids=["a"], output_ids=[]),
            node_inputs={},
            node_outputs={},
            module_map={},
        )
        record2 = ForwardRecord(
            graph=InferenceGraph(nodes={}, input_ids=["b"], output_ids=[]),
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        feedbacks = await loss.batch(
            outputs=["out1", "out2"],
            records=[record1, record2],
        )

        assert feedbacks[0]._record is record1
        assert feedbacks[1]._record is record2

    @pytest.mark.asyncio
    async def test_batch_with_contexts(self) -> None:
        """batch() passes contexts to each evaluation."""
        loss = ContextAwareLoss()

        feedbacks = await loss.batch(
            outputs=["out1", "out2"],
            contexts=[
                {"criteria": "helpfulness"},
                {"criteria": "accuracy"},
            ],
        )

        assert "helpfulness" in feedbacks[0].content
        assert "accuracy" in feedbacks[1].content

    @pytest.mark.asyncio
    async def test_batch_runs_concurrently(self) -> None:
        """batch() runs evaluations concurrently."""
        import asyncio
        import time

        call_times: list[float] = []

        class SlowLoss(Loss):
            async def __call__(
                self,
                output: Any,
                target: Any | None = None,
                *,
                record: ForwardRecord | None = None,
                context: dict[str, Any] | None = None,
            ) -> Feedback:
                call_times.append(time.monotonic())
                await asyncio.sleep(0.05)  # 50ms delay
                return Feedback(content="Done", score=1.0)

        loss = SlowLoss()

        start = time.monotonic()
        feedbacks = await loss.batch(["a", "b", "c"])
        elapsed = time.monotonic() - start

        assert len(feedbacks) == 3

        # If sequential: 3 * 0.05 = 0.15s
        # If concurrent: ~0.05s
        assert elapsed < 0.12, f"Took {elapsed:.3f}s - batch not running concurrently"

        # All calls should start nearly simultaneously
        if len(call_times) == 3:
            time_spread = max(call_times) - min(call_times)
            assert time_spread < 0.03, "Tasks did not start concurrently"

    @pytest.mark.asyncio
    async def test_batch_empty_list(self) -> None:
        """batch() with empty list returns empty list."""
        loss = SimpleLoss()

        feedbacks = await loss.batch([])

        assert feedbacks == []

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self) -> None:
        """batch() returns results in same order as inputs."""
        import asyncio

        class OrderedLoss(Loss):
            async def __call__(
                self,
                output: Any,
                target: Any | None = None,
                *,
                record: ForwardRecord | None = None,
                context: dict[str, Any] | None = None,
            ) -> Feedback:
                # Different delays to ensure order is preserved
                delays = {"first": 0.05, "second": 0.02, "third": 0.01}
                await asyncio.sleep(delays.get(output, 0))
                return Feedback(content=f"Evaluated: {output}", score=1.0)

        loss = OrderedLoss()

        feedbacks = await loss.batch(["first", "second", "third"])

        assert "first" in feedbacks[0].content
        assert "second" in feedbacks[1].content
        assert "third" in feedbacks[2].content


# ═══════════════════════════════════════════════════════════════════════════
#  Loss._extract_value_and_record() Tests (PR-068b)
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractValueAndRecord:
    """Tests for Loss._extract_value_and_record() helper."""

    def _create_record(self) -> ForwardRecord:
        """Create a minimal ForwardRecord for testing."""
        graph = InferenceGraph(
            nodes={},
            input_ids=[],
            output_ids=[],
        )
        return ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

    def test_extract_from_raw_value(self) -> None:
        """Extract from raw value returns value and None record."""

        loss = SimpleLoss()

        value, record = loss._extract_value_and_record("raw string")

        assert value == "raw string"
        assert record is None

    def test_extract_from_traced_output(self) -> None:
        """Extract from TracedOutput returns value and record."""
        from inf_engine.optimization.record import TracedOutput

        loss = SimpleLoss()
        inner_record = self._create_record()
        traced = TracedOutput(value="wrapped value", _record=inner_record)

        value, record = loss._extract_value_and_record(traced)

        assert value == "wrapped value"
        assert record is inner_record

    def test_extract_explicit_record_takes_precedence(self) -> None:
        """Explicit record takes precedence over TracedOutput's record."""
        from inf_engine.optimization.record import TracedOutput

        loss = SimpleLoss()
        traced_record = self._create_record()
        explicit_record = self._create_record()
        traced = TracedOutput(value="value", _record=traced_record)

        value, record = loss._extract_value_and_record(traced, explicit_record)

        assert value == "value"
        assert record is explicit_record
        assert record is not traced_record

    def test_extract_explicit_record_with_raw_value(self) -> None:
        """Explicit record works with raw values."""
        loss = SimpleLoss()
        explicit_record = self._create_record()

        value, record = loss._extract_value_and_record("raw", explicit_record)

        assert value == "raw"
        assert record is explicit_record


# ═══════════════════════════════════════════════════════════════════════════
#  Loss.batch() with TracedOutput Tests (PR-068b)
# ═══════════════════════════════════════════════════════════════════════════


class TestLossBatchWithTracedOutput:
    """Tests for Loss.batch() auto-extracting from TracedOutput."""

    def _create_record(self) -> ForwardRecord:
        """Create a minimal ForwardRecord for testing."""
        graph = InferenceGraph(
            nodes={},
            input_ids=[],
            output_ids=[],
        )
        return ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

    @pytest.mark.asyncio
    async def test_batch_extracts_values_from_traced_outputs(self) -> None:
        """batch() extracts actual values from TracedOutput wrappers."""
        from inf_engine.optimization.record import TracedOutput

        loss = SimpleLoss()

        records = [self._create_record() for _ in range(3)]
        traced_outputs = [
            TracedOutput(value="value1", _record=records[0]),
            TracedOutput(value="value2", _record=records[1]),
            TracedOutput(value="value3", _record=records[2]),
        ]

        feedbacks = await loss.batch(traced_outputs)

        assert len(feedbacks) == 3
        # Check that actual values were evaluated
        assert "value1" in feedbacks[0].content
        assert "value2" in feedbacks[1].content
        assert "value3" in feedbacks[2].content

    @pytest.mark.asyncio
    async def test_batch_extracts_records_from_traced_outputs(self) -> None:
        """batch() auto-extracts records from TracedOutput and attaches to feedback."""
        from inf_engine.optimization.record import TracedOutput

        loss = SimpleLoss()

        records = [self._create_record() for _ in range(2)]
        traced_outputs = [
            TracedOutput(value="a", _record=records[0]),
            TracedOutput(value="b", _record=records[1]),
        ]

        feedbacks = await loss.batch(traced_outputs)

        # Records should be attached from TracedOutput
        assert feedbacks[0]._record is records[0]
        assert feedbacks[1]._record is records[1]

    @pytest.mark.asyncio
    async def test_batch_explicit_records_override_traced_output(self) -> None:
        """batch() explicit records override TracedOutput's records."""
        from inf_engine.optimization.record import TracedOutput

        loss = SimpleLoss()

        traced_record = self._create_record()
        explicit_records = [self._create_record(), self._create_record()]

        traced_outputs = [
            TracedOutput(value="a", _record=traced_record),
            TracedOutput(value="b", _record=traced_record),
        ]

        feedbacks = await loss.batch(traced_outputs, records=explicit_records)

        # Explicit records should take precedence
        assert feedbacks[0]._record is explicit_records[0]
        assert feedbacks[1]._record is explicit_records[1]
        assert feedbacks[0]._record is not traced_record

    @pytest.mark.asyncio
    async def test_batch_mixed_raw_and_traced_outputs(self) -> None:
        """batch() handles mix of raw values and TracedOutput."""
        from inf_engine.optimization.record import TracedOutput

        loss = SimpleLoss()

        traced_record = self._create_record()
        outputs = [
            "raw value",
            TracedOutput(value="wrapped value", _record=traced_record),
        ]

        feedbacks = await loss.batch(outputs)

        # Raw value has no record
        assert feedbacks[0]._record is None
        # TracedOutput has its record attached
        assert feedbacks[1]._record is traced_record

    @pytest.mark.asyncio
    async def test_batch_traced_outputs_with_targets(self) -> None:
        """batch() works with TracedOutput and targets."""
        from inf_engine.optimization.record import TracedOutput

        loss = SimpleLoss()

        records = [self._create_record(), self._create_record()]
        traced_outputs = [
            TracedOutput(value="a", _record=records[0]),
            TracedOutput(value="b", _record=records[1]),
        ]

        feedbacks = await loss.batch(traced_outputs, targets=["a", "wrong"])

        # "a" matches target "a"
        assert feedbacks[0].score == 1.0
        # "b" doesn't match target "wrong"
        assert feedbacks[1].score == 0.5
