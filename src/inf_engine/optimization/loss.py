"""Loss functions for evaluating module outputs.

This module provides the abstract Loss base class that all loss functions
must inherit from. Loss functions evaluate outputs and produce Feedback
that can be propagated backward through the computation graph.

The Loss API is designed to mirror PyTorch's loss functions:
    >>> loss_fn = SomeLoss(criteria="helpfulness")
    >>> feedback = await loss_fn(output, target, record=record)
    >>> await feedback.backward()

Example:
    >>> from inf_engine.optimization.loss import Loss
    >>> from inf_engine.optimization.feedback import Feedback, FeedbackType
    >>>
    >>> class SimpleLoss(Loss):
    ...     async def __call__(
    ...         self,
    ...         output,
    ...         target=None,
    ...         *,
    ...         record=None,
    ...         context=None,
    ...     ):
    ...         score = 1.0 if output == target else 0.0
    ...         feedback = Feedback(
    ...             content="Match" if score == 1.0 else "Mismatch",
    ...             score=score,
    ...             feedback_type=FeedbackType.VERIFIER,
    ...         )
    ...         return self._attach_record(feedback, record)

Note:
    Concrete loss implementations (VerifierLoss, LLMJudge, etc.) will be
    added in PR-066 (Loss implementations).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inf_engine.optimization.feedback import Feedback
    from inf_engine.optimization.record import ForwardRecord


class Loss(ABC):
    """Abstract base class for loss functions.

    Loss functions evaluate outputs and produce Feedback that can be
    propagated backward through the computation graph via feedback.backward().

    All loss functions must implement the async __call__ method which takes
    an output to evaluate and optionally a target/expected value. The record
    parameter is used to attach the ForwardRecord to the returned Feedback,
    enabling backward propagation.

    Subclasses should use the _attach_record() helper method to properly
    attach the ForwardRecord to feedback before returning.

    Example:
        >>> class MyLoss(Loss):
        ...     async def __call__(
        ...         self,
        ...         output,
        ...         target=None,
        ...         *,
        ...         record=None,
        ...         context=None,
        ...     ):
        ...         # Evaluate output and create feedback
        ...         feedback = Feedback(content="Evaluation result", score=0.8)
        ...         # Always use _attach_record before returning
        ...         return self._attach_record(feedback, record)

    Note:
        LLM-based losses (like LLMJudge) use internal LLMInference modules
        with structured output (response_format) for reliable parsing. These
        modules are called through the normal __call__ interface.
    """

    @abstractmethod
    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        """Compute feedback for an output.

        This method must be implemented by all loss function subclasses.
        It evaluates the output and produces Feedback that describes
        what was good or bad about the output.

        Args:
            output: The module output to evaluate. Can be any type depending
                on what the module produces.
            target: Optional target/expected output for comparison. Some loss
                functions (like VerifierLoss) may not need this, while others
                (like contrastive losses) require it.
            record: ForwardRecord from run(..., record=True). Required for
                feedback.backward() to work. If provided, it will be attached
                to the returned Feedback via _attach_record().
            context: Optional additional context for evaluation. Can include
                things like the original input, metadata about the task, or
                any other information needed by the loss function.

        Returns:
            Feedback object containing evaluation results. If record was
            provided, feedback.backward() can be called to propagate the
            feedback through the computation graph.

        Example:
            >>> loss_fn = MyLoss()
            >>> feedback = await loss_fn(output, target=expected, record=record)
            >>> print(feedback.score)
            0.85
            >>> await feedback.backward()  # Propagates to Parameters
        """
        pass

    def _attach_record(
        self,
        feedback: Feedback,
        record: ForwardRecord | None,
    ) -> Feedback:
        """Attach ForwardRecord to feedback for backward propagation.

        This helper method should be called by all loss implementations
        before returning feedback. It attaches the ForwardRecord to the
        feedback object, enabling feedback.backward() to work.

        Args:
            feedback: The Feedback object to attach the record to.
            record: The ForwardRecord from the forward pass, or None if
                not recording.

        Returns:
            The same Feedback object with _record set (if record was provided).

        Example:
            >>> async def __call__(self, output, target=None, *, record=None, context=None):
            ...     feedback = Feedback(content="Evaluation", score=0.9)
            ...     return self._attach_record(feedback, record)

        Note:
            This method mutates the feedback object in place. The same object
            is returned for convenience in chaining.
        """
        if record is not None:
            feedback._record = record
        return feedback
