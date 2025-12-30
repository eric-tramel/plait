"""Tests for Loss abstract base class and concrete implementations."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from inf_engine.graph import InferenceGraph
from inf_engine.optimization.feedback import Feedback, FeedbackType
from inf_engine.optimization.loss import CompositeLoss, LLMJudge, Loss, VerifierLoss
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
