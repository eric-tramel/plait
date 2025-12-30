"""Tests for Loss abstract base class."""

from typing import Any

import pytest

from inf_engine.graph import InferenceGraph
from inf_engine.optimization.feedback import Feedback, FeedbackType
from inf_engine.optimization.loss import Loss
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
