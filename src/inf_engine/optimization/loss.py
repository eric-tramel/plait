"""Loss functions for evaluating module outputs.

This module provides the abstract Loss base class and concrete loss
implementations for evaluating outputs and producing Feedback that can be
propagated backward through the computation graph.

The Loss API is designed to mirror PyTorch's loss functions:
    >>> loss_fn = VerifierLoss(verifier=my_verifier)
    >>> feedback = await loss_fn(output, target, record=record)
    >>> await feedback.backward()

Available loss functions:
    - VerifierLoss: Programmatic verification using code
    - LLMJudge: LLM-based evaluation with freeform feedback
    - CompositeLoss: Weighted combination of multiple loss functions

Example:
    >>> from inf_engine.optimization.loss import VerifierLoss
    >>>
    >>> def check_format(output):
    ...     if "error" in output.lower():
    ...         return False, "Output contains error message"
    ...     return True, "Output is valid"
    >>>
    >>> loss = VerifierLoss(verifier=check_format)
    >>> feedback = await loss("Hello world")
    >>> feedback.score
    1.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from inf_engine.module import LLMInference
    from inf_engine.optimization.feedback import Feedback
    from inf_engine.optimization.record import ForwardRecord
    from inf_engine.resources.config import ResourceConfig
    from inf_engine.resources.manager import ResourceManager


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


class VerifierLoss(Loss):
    """Loss from programmatic verification.

    Uses code to evaluate outputs deterministically. The verifier function
    receives the output and returns a tuple of (passed, message). This is
    useful for format checks, keyword presence, JSON validity, or any
    programmatic constraint.

    Attributes:
        verifier: Function that takes output and returns (passed, message).
        success_feedback: Feedback message when verification passes.

    Example:
        >>> def check_json(output):
        ...     import json
        ...     try:
        ...         json.loads(output)
        ...         return True, "Valid JSON"
        ...     except json.JSONDecodeError as e:
        ...         return False, f"Invalid JSON: {e}"
        >>>
        >>> loss = VerifierLoss(verifier=check_json)
        >>> feedback = await loss('{"key": "value"}')
        >>> feedback.score
        1.0

    Example with custom success message:
        >>> loss = VerifierLoss(
        ...     verifier=lambda x: ("error" not in x.lower(), "Contains error"),
        ...     success_feedback="Output is clean and error-free",
        ... )
    """

    def __init__(
        self,
        verifier: Callable[[Any], tuple[bool, str]],
        success_feedback: str = "Output passed verification.",
    ) -> None:
        """Initialize the VerifierLoss.

        Args:
            verifier: Function taking output and returning (passed, message).
                The first element is True if verification passed, False otherwise.
                The second element is a message explaining the result.
            success_feedback: Feedback message when verification passes.
                Defaults to "Output passed verification."
        """
        self.verifier = verifier
        self.success_feedback = success_feedback

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        """Evaluate output using the verifier function.

        The verifier function is called with the output and returns
        (passed, message). The score is 1.0 if passed, 0.0 otherwise.

        Args:
            output: The module output to verify.
            target: Ignored for VerifierLoss (programmatic checks don't
                typically need a target).
            record: ForwardRecord from run(..., record=True).
            context: Ignored for VerifierLoss.

        Returns:
            Feedback with score 1.0 (passed) or 0.0 (failed), and either
            the success_feedback message or the verifier's error message.
        """
        from inf_engine.optimization.feedback import Feedback, FeedbackType

        passed, message = self.verifier(output)

        feedback = Feedback(
            content=self.success_feedback if passed else message,
            score=1.0 if passed else 0.0,
            feedback_type=FeedbackType.VERIFIER,
        )
        return self._attach_record(feedback, record)


class LLMJudge(Loss):
    """Freeform LLM feedback without structured scoring.

    The LLM provides critical feedback on the output without being
    constrained to a specific rubric or scale. Useful for open-ended
    improvement suggestions and qualitative evaluation.

    The internal LLMInference module must be bound to resources before
    calling the loss function. Use the bind() method to configure resources.

    Attributes:
        criteria: Optional focus areas for feedback.
        judge: Internal LLMInference module for evaluation.

    Example:
        >>> judge = LLMJudge(
        ...     alias="judge",
        ...     criteria="clarity, completeness, and accuracy",
        ... )
        >>> judge.bind(resources)  # Must bind before use
        >>> feedback = await judge(output, target=expected)
        >>> print(feedback.content)  # Detailed LLM feedback

    Note:
        The returned Feedback has score=None since this is freeform
        evaluation without structured scoring.
    """

    def __init__(
        self,
        alias: str = "judge",
        criteria: str | None = None,
    ) -> None:
        """Initialize the LLMJudge.

        Args:
            alias: Resource alias for the judge LLM endpoint.
            criteria: Optional focus areas for evaluation. If provided,
                the LLM will focus its feedback on these aspects.
        """
        from inf_engine.module import LLMInference

        self.criteria = criteria
        self.judge: LLMInference = LLMInference(
            alias=alias,
            system_prompt=(
                "You are a critical reviewer. Provide specific, actionable "
                "feedback on how the output could be improved. Be constructive "
                "but thorough in identifying weaknesses."
            ),
        )

    def bind(self, resources: ResourceConfig | ResourceManager) -> Self:
        """Bind the internal judge module to resources.

        This must be called before using the loss function. The resources
        configuration must include the alias specified during initialization.

        Args:
            resources: ResourceConfig or ResourceManager containing the
                judge endpoint configuration.

        Returns:
            Self for method chaining.

        Example:
            >>> judge = LLMJudge(alias="judge").bind(resources)
            >>> feedback = await judge(output)
        """
        self.judge.bind(resources)
        return self

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        """Evaluate output using the LLM judge.

        Constructs a prompt from the output, optional target, context, and
        criteria, then sends it to the LLM for evaluation.

        Args:
            output: The module output to evaluate.
            target: Optional target/expected output for comparison.
            record: ForwardRecord from run(..., record=True).
            context: Optional additional context for evaluation.

        Returns:
            Feedback containing the LLM's freeform critique. The score
            is None since this is qualitative evaluation.

        Raises:
            RuntimeError: If the judge has not been bound to resources.
        """
        from inf_engine.optimization.feedback import Feedback, FeedbackType

        # Build the evaluation prompt
        prompt_parts = [f"Output to critique:\n{output}"]

        if context:
            prompt_parts.append(f"Context: {context}")
        if target:
            prompt_parts.append(f"Expected behavior: {target}")
        if self.criteria:
            prompt_parts.append(f"Focus areas: {self.criteria}")

        prompt_parts.append("\nProvide detailed, actionable feedback:")
        prompt = "\n\n".join(prompt_parts)

        # Get LLM feedback
        response = await self.judge(prompt)

        feedback = Feedback(
            content=response,
            score=None,  # Freeform feedback has no structured score
            feedback_type=FeedbackType.LLM_JUDGE,
        )
        return self._attach_record(feedback, record)


class CompositeLoss(Loss):
    """Combine multiple loss functions with weights.

    Useful for multi-objective optimization where multiple aspects need
    to be evaluated (e.g., helpfulness + safety, clarity + accuracy).

    The final score is a weighted average of all component scores (for
    components that return scores). The feedback content is either
    concatenated from all components (simple aggregation) or synthesized
    by an optional LLM aggregator.

    Attributes:
        losses: List of (loss_function, weight) pairs.
        aggregator: Optional LLMInference to synthesize feedback.

    Example:
        >>> format_check = VerifierLoss(verifier=check_format)
        >>> llm_quality = LLMJudge(alias="judge", criteria="quality")
        >>>
        >>> composite = CompositeLoss([
        ...     (format_check, 0.3),   # 30% weight
        ...     (llm_quality, 0.7),    # 70% weight
        ... ])
        >>> feedback = await composite(output, record=record)

    Example with LLM aggregator:
        >>> aggregator = LLMInference(alias="aggregator")
        >>> composite = CompositeLoss(
        ...     losses=[(loss1, 0.5), (loss2, 0.5)],
        ...     aggregator=aggregator,
        ... )
    """

    def __init__(
        self,
        losses: list[tuple[Loss, float]],
        aggregator: LLMInference | None = None,
    ) -> None:
        """Initialize the CompositeLoss.

        Args:
            losses: List of (loss_function, weight) pairs. Weights should
                typically sum to 1.0 but this is not enforced.
            aggregator: Optional LLM to synthesize feedback from all
                components into a coherent summary. If None, feedback
                is concatenated with weights shown.
        """
        self.losses = losses
        self.aggregator = aggregator

    def bind(self, resources: ResourceConfig | ResourceManager) -> Self:
        """Bind all component losses and optional aggregator to resources.

        Iterates through all component losses and calls bind() on those
        that have a bind method (e.g., LLMJudge). Also binds the aggregator
        if present.

        Args:
            resources: ResourceConfig or ResourceManager containing endpoint
                configurations for all components.

        Returns:
            Self for method chaining.
        """
        for loss, _ in self.losses:
            bind_method = getattr(loss, "bind", None)
            if callable(bind_method):
                bind_method(resources)
        if self.aggregator is not None:
            self.aggregator.bind(resources)
        return self

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        """Evaluate output using all component losses.

        Each component loss is called and their feedback is aggregated.
        The final score is a weighted average of component scores.

        Args:
            output: The module output to evaluate.
            target: Optional target/expected output for comparison.
            record: ForwardRecord from run(..., record=True). Note that
                the record is attached to the composite feedback, not
                passed to sub-losses.
            context: Optional additional context for evaluation.

        Returns:
            Feedback with aggregated content and weighted average score.
        """
        from inf_engine.optimization.feedback import Feedback, FeedbackType

        # Gather all feedback (don't pass record to sub-losses)
        feedbacks: list[tuple[Feedback, float]] = []
        weighted_score = 0.0
        total_weight = 0.0

        for loss, weight in self.losses:
            fb = await loss(output, target, context=context)
            feedbacks.append((fb, weight))
            if fb.score is not None:
                weighted_score += fb.score * weight
                total_weight += weight

        # Aggregate feedback text
        if self.aggregator:
            combined = await self._llm_aggregate(feedbacks)
        else:
            combined = self._simple_aggregate(feedbacks)

        feedback = Feedback(
            content=combined,
            score=weighted_score / total_weight if total_weight > 0 else None,
            feedback_type=FeedbackType.COMPOSITE,
        )
        return self._attach_record(feedback, record)

    def _simple_aggregate(self, feedbacks: list[tuple[Feedback, float]]) -> str:
        """Concatenate feedback with weights shown.

        Args:
            feedbacks: List of (feedback, weight) pairs.

        Returns:
            Concatenated feedback string with weights.
        """
        parts = []
        for fb, weight in feedbacks:
            parts.append(f"[Weight: {weight}] {fb.content}")
        return "\n\n".join(parts)

    async def _llm_aggregate(self, feedbacks: list[tuple[Feedback, float]]) -> str:
        """Use LLM to synthesize feedback into coherent summary.

        Args:
            feedbacks: List of (feedback, weight) pairs.

        Returns:
            Synthesized feedback from the aggregator LLM.

        Note:
            This method should only be called when self.aggregator is not None.
        """
        assert self.aggregator is not None  # Caller ensures this
        prompt = "Synthesize the following feedback into a coherent summary:\n\n"
        for fb, weight in feedbacks:
            prompt += f"--- Feedback (weight: {weight}) ---\n{fb.content}\n\n"
        prompt += "Provide a unified summary of the key points and suggestions."
        return await self.aggregator(prompt)
