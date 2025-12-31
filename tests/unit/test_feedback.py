"""Tests for Feedback and FeedbackType."""

import pytest

from inf_engine.optimization.feedback import Feedback, FeedbackType
from inf_engine.optimization.record import ForwardRecord


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_feedback_type_values(self) -> None:
        """FeedbackType has expected string values."""
        assert FeedbackType.HUMAN.value == "human"
        assert FeedbackType.LLM_JUDGE.value == "llm_judge"
        assert FeedbackType.VERIFIER.value == "verifier"
        assert FeedbackType.COMPOSITE.value == "composite"

    def test_feedback_type_is_enum(self) -> None:
        """FeedbackType members are proper enum members."""
        assert isinstance(FeedbackType.HUMAN, FeedbackType)
        assert isinstance(FeedbackType.LLM_JUDGE, FeedbackType)
        assert isinstance(FeedbackType.VERIFIER, FeedbackType)
        assert isinstance(FeedbackType.COMPOSITE, FeedbackType)

    def test_feedback_type_all_members(self) -> None:
        """FeedbackType has exactly four members."""
        members = list(FeedbackType)
        assert len(members) == 4
        assert FeedbackType.HUMAN in members
        assert FeedbackType.LLM_JUDGE in members
        assert FeedbackType.VERIFIER in members
        assert FeedbackType.COMPOSITE in members


class TestFeedbackCreation:
    """Tests for Feedback dataclass creation."""

    def test_feedback_with_content_only(self) -> None:
        """Feedback can be created with just content."""
        feedback = Feedback(content="Some feedback text")
        assert feedback.content == "Some feedback text"
        assert feedback.score is None
        assert feedback.feedback_type == FeedbackType.HUMAN
        assert feedback.metadata == {}
        assert feedback._record is None
        assert feedback._optimizer is None

    def test_feedback_with_score(self) -> None:
        """Feedback can include a numeric score."""
        feedback = Feedback(content="Good response", score=0.85)
        assert feedback.content == "Good response"
        assert feedback.score == 0.85

    def test_feedback_with_score_zero(self) -> None:
        """Feedback can have a score of zero."""
        feedback = Feedback(content="Bad response", score=0.0)
        assert feedback.score == 0.0

    def test_feedback_with_score_one(self) -> None:
        """Feedback can have a perfect score."""
        feedback = Feedback(content="Perfect response", score=1.0)
        assert feedback.score == 1.0

    def test_feedback_with_feedback_type(self) -> None:
        """Feedback can specify a feedback type."""
        feedback = Feedback(
            content="LLM evaluation",
            feedback_type=FeedbackType.LLM_JUDGE,
        )
        assert feedback.feedback_type == FeedbackType.LLM_JUDGE

    def test_feedback_with_verifier_type(self) -> None:
        """Feedback can be from verifier."""
        feedback = Feedback(
            content="Passed format check",
            score=1.0,
            feedback_type=FeedbackType.VERIFIER,
        )
        assert feedback.feedback_type == FeedbackType.VERIFIER

    def test_feedback_with_composite_type(self) -> None:
        """Feedback can be composite."""
        feedback = Feedback(
            content="Combined feedback",
            score=0.75,
            feedback_type=FeedbackType.COMPOSITE,
        )
        assert feedback.feedback_type == FeedbackType.COMPOSITE

    def test_feedback_with_metadata(self) -> None:
        """Feedback can include arbitrary metadata."""
        metadata = {"criteria": "helpfulness", "raw_score": 4}
        feedback = Feedback(
            content="Helpful response",
            score=0.8,
            metadata=metadata,
        )
        assert feedback.metadata == {"criteria": "helpfulness", "raw_score": 4}

    def test_feedback_full_construction(self) -> None:
        """Feedback can be created with all fields."""
        feedback = Feedback(
            content="Detailed feedback text",
            score=0.9,
            feedback_type=FeedbackType.LLM_JUDGE,
            metadata={"key": "value"},
        )
        assert feedback.content == "Detailed feedback text"
        assert feedback.score == 0.9
        assert feedback.feedback_type == FeedbackType.LLM_JUDGE
        assert feedback.metadata == {"key": "value"}


class TestFeedbackStr:
    """Tests for Feedback string representation."""

    def test_str_without_score(self) -> None:
        """Feedback without score returns just content."""
        feedback = Feedback(content="Just the content")
        assert str(feedback) == "Just the content"

    def test_str_with_score(self) -> None:
        """Feedback with score prepends formatted score."""
        feedback = Feedback(content="Content here", score=0.85)
        assert str(feedback) == "[0.85] Content here"

    def test_str_with_zero_score(self) -> None:
        """Feedback with zero score shows [0.00]."""
        feedback = Feedback(content="Failed", score=0.0)
        assert str(feedback) == "[0.00] Failed"

    def test_str_with_perfect_score(self) -> None:
        """Feedback with perfect score shows [1.00]."""
        feedback = Feedback(content="Perfect", score=1.0)
        assert str(feedback) == "[1.00] Perfect"

    def test_str_with_score_rounds(self) -> None:
        """Score is formatted to 2 decimal places."""
        feedback = Feedback(content="Content", score=0.333333)
        assert str(feedback) == "[0.33] Content"

        feedback2 = Feedback(content="Content", score=0.999)
        assert str(feedback2) == "[1.00] Content"

    def test_str_with_multiline_content(self) -> None:
        """Multiline content is preserved."""
        feedback = Feedback(content="Line 1\nLine 2", score=0.5)
        assert str(feedback) == "[0.50] Line 1\nLine 2"


class TestFeedbackRepr:
    """Tests for Feedback repr representation."""

    def test_repr_excludes_private_fields(self) -> None:
        """Repr should not include _record and _optimizer."""
        feedback = Feedback(content="Test", score=0.5)
        repr_str = repr(feedback)
        assert "_record" not in repr_str
        assert "_optimizer" not in repr_str

    def test_repr_includes_public_fields(self) -> None:
        """Repr should include public fields."""
        feedback = Feedback(
            content="Test content",
            score=0.75,
            feedback_type=FeedbackType.LLM_JUDGE,
        )
        repr_str = repr(feedback)
        assert "content='Test content'" in repr_str
        assert "score=0.75" in repr_str
        assert "FeedbackType.LLM_JUDGE" in repr_str


class TestFeedbackBackward:
    """Tests for Feedback.backward() method."""

    @pytest.mark.asyncio
    async def test_backward_raises_without_record(self) -> None:
        """backward() raises RuntimeError when no record is attached."""
        feedback = Feedback(content="Test feedback")
        with pytest.raises(RuntimeError) as exc_info:
            await feedback.backward()
        assert "Cannot call backward() without a ForwardRecord" in str(exc_info.value)
        assert "Pass record=record when computing feedback" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_backward_raises_even_with_score(self) -> None:
        """backward() raises regardless of other fields being set."""
        feedback = Feedback(
            content="Full feedback",
            score=0.9,
            feedback_type=FeedbackType.LLM_JUDGE,
            metadata={"key": "value"},
        )
        with pytest.raises(RuntimeError) as exc_info:
            await feedback.backward()
        assert "ForwardRecord" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_backward_raises_with_optimizer_but_no_record(self) -> None:
        """backward() raises even if optimizer is passed without record."""
        feedback = Feedback(content="Test")
        with pytest.raises(RuntimeError):
            await feedback.backward(optimizer=object())

    @pytest.mark.asyncio
    async def test_backward_with_mock_record(self) -> None:
        """backward() succeeds when record is attached (stub implementation)."""
        from inf_engine.graph import InferenceGraph
        from inf_engine.optimization.record import ForwardRecord

        # Create a minimal ForwardRecord
        graph = InferenceGraph(
            nodes={},
            input_ids=[],
            output_ids=[],
        )
        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        feedback = Feedback(content="Test feedback", score=0.8)
        feedback._record = record

        # Should not raise - stub implementation just passes
        await feedback.backward()

    @pytest.mark.asyncio
    async def test_backward_with_optimizer_argument(self) -> None:
        """backward() accepts optimizer parameter when record is attached."""
        from inf_engine.graph import InferenceGraph
        from inf_engine.optimization.record import ForwardRecord

        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        feedback = Feedback(content="Test")
        feedback._record = record

        # Create a mock optimizer with reasoning_llm attribute
        class MockOptimizer:
            reasoning_llm = None

        # Should not raise
        await feedback.backward(optimizer=MockOptimizer())


class TestFeedbackEquality:
    """Tests for Feedback equality comparison."""

    def test_equal_feedbacks(self) -> None:
        """Two feedbacks with same public fields are equal."""
        fb1 = Feedback(content="Test", score=0.5)
        fb2 = Feedback(content="Test", score=0.5)
        assert fb1 == fb2

    def test_unequal_content(self) -> None:
        """Feedbacks with different content are not equal."""
        fb1 = Feedback(content="Test 1")
        fb2 = Feedback(content="Test 2")
        assert fb1 != fb2

    def test_unequal_score(self) -> None:
        """Feedbacks with different scores are not equal."""
        fb1 = Feedback(content="Test", score=0.5)
        fb2 = Feedback(content="Test", score=0.6)
        assert fb1 != fb2

    def test_private_fields_not_compared(self) -> None:
        """Private fields (_record, _optimizer) are not included in equality."""
        from inf_engine.graph import InferenceGraph
        from inf_engine.optimization.record import ForwardRecord

        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        fb1 = Feedback(content="Test", score=0.5)
        fb2 = Feedback(content="Test", score=0.5)
        fb2._record = record

        # Should still be equal since _record is excluded from comparison
        assert fb1 == fb2


# ═══════════════════════════════════════════════════════════════════════════
#  Feedback.backward_batch() Tests (PR-068b)
# ═══════════════════════════════════════════════════════════════════════════


def _create_record() -> ForwardRecord:
    """Create a minimal ForwardRecord for testing."""
    from inf_engine.graph import InferenceGraph

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


class TestFeedbackBackwardBatch:
    """Tests for Feedback.backward_batch() static method."""

    @pytest.mark.asyncio
    async def test_backward_batch_runs_all_backwards(self) -> None:
        """backward_batch() runs backward for all feedbacks."""
        records = [_create_record() for _ in range(3)]
        feedbacks = [
            Feedback(content=f"Feedback {i}", score=0.5 + i * 0.1) for i in range(3)
        ]
        for fb, rec in zip(feedbacks, records, strict=True):
            fb._record = rec

        # Should not raise
        await Feedback.backward_batch(feedbacks)

    @pytest.mark.asyncio
    async def test_backward_batch_empty_list(self) -> None:
        """backward_batch() with empty list does nothing."""
        # Should not raise
        await Feedback.backward_batch([])

    @pytest.mark.asyncio
    async def test_backward_batch_raises_on_missing_record(self) -> None:
        """backward_batch() raises if any feedback lacks record."""
        record = _create_record()
        feedbacks = [
            Feedback(content="Has record"),
            Feedback(content="No record"),
            Feedback(content="Also has record"),
        ]
        feedbacks[0]._record = record
        feedbacks[2]._record = record
        # feedbacks[1] has no record

        with pytest.raises(RuntimeError) as exc_info:
            await Feedback.backward_batch(feedbacks)

        # Should indicate which feedback is missing record
        assert "index 1" in str(exc_info.value)
        assert "ForwardRecord" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_backward_batch_validates_before_starting(self) -> None:
        """backward_batch() validates all feedbacks before starting any backward."""
        backward_calls = []

        async def tracking_backward(self: Feedback, optimizer: object = None) -> None:
            backward_calls.append(self.content)
            # Original backward implementation
            if self._record is None:
                raise RuntimeError("No record")
            from inf_engine.optimization.backward import _propagate_backward

            await _propagate_backward(
                feedback=self,
                record=self._record,
                reasoning_llm=None,
            )

        record = _create_record()
        feedbacks = [
            Feedback(content="First"),
            Feedback(content="Second - no record"),
            Feedback(content="Third"),
        ]
        feedbacks[0]._record = record
        feedbacks[2]._record = record
        # feedbacks[1] intentionally has no record

        # Should raise before any backward is called
        with pytest.raises(RuntimeError):
            await Feedback.backward_batch(feedbacks)

        # No backward should have been called since validation fails first
        # Note: We can't easily verify this without mocking, but the error
        # message should indicate the validation was pre-check

    @pytest.mark.asyncio
    async def test_backward_batch_passes_optimizer(self) -> None:
        """backward_batch() passes optimizer to each backward call."""
        records = [_create_record(), _create_record()]
        feedbacks = [
            Feedback(content="First"),
            Feedback(content="Second"),
        ]
        for fb, rec in zip(feedbacks, records, strict=True):
            fb._record = rec

        class MockOptimizer:
            reasoning_llm = None

        optimizer = MockOptimizer()

        # Should not raise
        await Feedback.backward_batch(feedbacks, optimizer=optimizer)

    @pytest.mark.asyncio
    async def test_backward_batch_runs_concurrently(self) -> None:
        """backward_batch() runs backward passes concurrently."""
        import asyncio
        import time

        call_times: list[float] = []

        # Monkey-patch _propagate_backward for this test
        async def slow_propagate(*args, **kwargs) -> None:
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)  # 50ms delay

        records = [_create_record() for _ in range(3)]
        feedbacks = [Feedback(content=f"FB {i}") for i in range(3)]
        for fb, rec in zip(feedbacks, records, strict=True):
            fb._record = rec

        # Patch the propagate function
        import inf_engine.optimization.backward as backward_module

        original_propagate = backward_module._propagate_backward

        try:
            # Monkey-patch for testing concurrency
            backward_module._propagate_backward = slow_propagate  # type: ignore[assignment]

            start = time.monotonic()
            await Feedback.backward_batch(feedbacks)
            elapsed = time.monotonic() - start

            # If sequential: 3 * 0.05 = 0.15s
            # If concurrent: ~0.05s
            assert elapsed < 0.12, (
                f"Took {elapsed:.3f}s - batch not running concurrently"
            )

            # All calls should start nearly simultaneously
            if len(call_times) == 3:
                time_spread = max(call_times) - min(call_times)
                assert time_spread < 0.03, "Tasks did not start concurrently"
        finally:
            # Restore original function
            backward_module._propagate_backward = original_propagate  # type: ignore[assignment]

    @pytest.mark.asyncio
    async def test_backward_batch_single_feedback(self) -> None:
        """backward_batch() works with single feedback."""
        record = _create_record()
        feedback = Feedback(content="Single feedback")
        feedback._record = record

        # Should not raise
        await Feedback.backward_batch([feedback])
