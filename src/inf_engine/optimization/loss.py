"""Loss functions for evaluating module outputs.

This module provides the abstract Loss base class and concrete loss
implementations for evaluating outputs and producing Feedback that can be
propagated backward through the computation graph.

The Loss API is designed to mirror PyTorch's loss functions:
    >>> loss_fn = VerifierLoss(verifier=my_verifier)
    >>> feedback = await loss_fn(output, target, record=record)
    >>> await feedback.backward()

Loss Taxonomy:
    Single-Sample Losses (evaluate one output at a time):
        - VerifierLoss: Programmatic verification using code
        - LLMJudge: Freeform LLM critique
        - HumanFeedbackLoss: Freeform human critique via stdin
        - LLMRubricLoss: LLM evaluation against Likert scale
        - HumanRubricLoss: Human evaluation against Likert scale

    Contrastive Losses (compare multiple outputs):
        - LLMPreferenceLoss: LLM picks winner from pair
        - HumanPreferenceLoss: Human picks winner from pair
        - LLMRankingLoss: LLM ranks n outputs
        - HumanRankingLoss: Human ranks n outputs

    Composite:
        - CompositeLoss: Weighted combination of multiple losses

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self

if TYPE_CHECKING:
    from inf_engine.module import LLMInference
    from inf_engine.optimization.feedback import Feedback
    from inf_engine.optimization.record import ForwardRecord
    from inf_engine.resources.config import ResourceConfig
    from inf_engine.resources.manager import ResourceManager


# =============================================================================
# Structured Output Schemas for LLM-based Losses
# =============================================================================


@dataclass
class RubricLevel:
    """A single level in a Likert scale rubric.

    Used to define scoring scales for rubric-based evaluation losses.
    Each level has a numeric score, a short label, and a detailed description.

    Attributes:
        score: Numeric score for this level (e.g., 1-5).
        label: Short label (e.g., "Poor", "Excellent").
        description: Detailed description of what this level means.

    Example:
        >>> level = RubricLevel(
        ...     score=5,
        ...     label="Excellent",
        ...     description="Exceptionally clear and complete response",
        ... )
    """

    score: int
    label: str
    description: str


@dataclass
class RubricResponse:
    """Structured response for rubric-based LLM evaluation.

    This schema is used with LLMInference's response_format parameter
    to ensure reliable parsing of rubric-based evaluations.

    Attributes:
        score: The numeric score assigned (matching a RubricLevel.score).
        justification: Explanation of why this score was assigned.
        feedback: Actionable suggestions for improvement.
    """

    score: int
    justification: str
    feedback: str


@dataclass
class PreferenceResponse:
    """Structured response for pairwise preference comparison.

    This schema is used with LLMInference's response_format parameter
    to ensure reliable parsing of preference comparisons.

    Attributes:
        winner: Which output won ("A" or "B").
        reason: Why the winner was selected.
        a_strengths: Strengths of output A.
        a_weaknesses: Weaknesses of output A.
        b_strengths: Strengths of output B.
        b_weaknesses: Weaknesses of output B.
    """

    winner: Literal["A", "B"]
    reason: str
    a_strengths: str
    a_weaknesses: str
    b_strengths: str
    b_weaknesses: str


@dataclass
class RankingResponse:
    """Structured response for n-way ranking.

    This schema is used with LLMInference's response_format parameter
    to ensure reliable parsing of ranking evaluations.

    Attributes:
        ranking: List of indices in order from best to worst (1-indexed).
        best_qualities: What made the best output stand out.
        worst_issues: What problems the worst output had.
        comparison: Overall comparison of the outputs.
    """

    ranking: list[int]
    best_qualities: str
    worst_issues: str
    comparison: str


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


# =============================================================================
# Human Feedback Losses
# =============================================================================


class HumanFeedbackLoss(Loss):
    """Freeform human feedback collected via stdout/stdin.

    Prompts the user to provide critical feedback on each output.
    Useful for RLHF-style training with human-in-the-loop or for
    manual evaluation during development.

    The output is displayed to the user via print(), and feedback is
    collected via input() until an empty line is entered.

    Attributes:
        prompt_template: Custom prompt shown to user. Can use {output},
            {target}, {context} placeholders.
        show_context: Whether to display context/target to user.

    Example:
        >>> loss = HumanFeedbackLoss(show_context=True)
        >>> # When called, displays output and prompts for feedback
        >>> feedback = await loss("The AI response here", target="expected")
        # User sees output and types feedback interactively
    """

    def __init__(
        self,
        prompt_template: str | None = None,
        show_context: bool = True,
    ) -> None:
        """Initialize the HumanFeedbackLoss.

        Args:
            prompt_template: Custom prompt shown to user. Use {output},
                {target}, {context} placeholders for dynamic content.
            show_context: Whether to display context/target to user.
                Defaults to True.
        """
        self.prompt_template = prompt_template
        self.show_context = show_context

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        """Collect human feedback on the output.

        Displays the output to the user via stdout and collects
        feedback via stdin until an empty line is entered.

        Args:
            output: The module output to evaluate.
            target: Optional target/expected output for comparison.
            record: ForwardRecord from run(..., record=True).
            context: Optional additional context for evaluation.

        Returns:
            Feedback containing the human's critique. Score is None
            since this is freeform evaluation.
        """
        from inf_engine.optimization.feedback import Feedback, FeedbackType

        # Display output to user
        print("\n" + "=" * 60)
        print("OUTPUT TO EVALUATE:")
        print("-" * 60)
        print(output)

        if self.show_context:
            if target:
                print("-" * 60)
                print(f"Expected: {target}")
            if context:
                print(f"Context: {context}")

        print("=" * 60)

        # Collect feedback
        if self.prompt_template:
            prompt = self.prompt_template.format(
                output=output, target=target, context=context
            )
            print(prompt)
        else:
            print("Please provide feedback on this output.")
            print("What could be improved? (Enter empty line to finish)")

        lines = []
        while True:
            line = input("> ")
            if not line:
                break
            lines.append(line)

        content = "\n".join(lines) if lines else "No feedback provided."

        feedback = Feedback(
            content=content,
            score=None,
            feedback_type=FeedbackType.HUMAN,
        )
        return self._attach_record(feedback, record)


# =============================================================================
# Rubric-Based Losses
# =============================================================================


class LLMRubricLoss(Loss):
    """LLM evaluation against a structured Likert scale rubric.

    The LLM evaluates the output against specific criteria using
    a defined scale (e.g., 1-5 or 1-7), providing both a score
    and justification. Uses structured output for reliable parsing.

    This is useful when you want consistent, structured evaluation
    with numeric scores that can be compared across evaluations.

    Attributes:
        criteria: What aspect to evaluate.
        rubric: List of RubricLevel defining the scale.
        judge: Internal LLMInference module for evaluation.

    Example:
        >>> rubric = [
        ...     RubricLevel(1, "Poor", "Fails to address the query"),
        ...     RubricLevel(2, "Below Average", "Partially addresses query"),
        ...     RubricLevel(3, "Average", "Adequately addresses query"),
        ...     RubricLevel(4, "Good", "Thoroughly addresses query"),
        ...     RubricLevel(5, "Excellent", "Exceptionally addresses query"),
        ... ]
        >>> loss = LLMRubricLoss(
        ...     criteria="helpfulness",
        ...     rubric=rubric,
        ...     alias="judge",
        ... ).bind(resources)
        >>> feedback = await loss(output)
        >>> print(feedback.score)  # Normalized to 0-1
    """

    def __init__(
        self,
        criteria: str,
        rubric: list[RubricLevel],
        alias: str = "judge",
    ) -> None:
        """Initialize the LLMRubricLoss.

        Args:
            criteria: What aspect to evaluate (e.g., "helpfulness",
                "clarity", "accuracy").
            rubric: List of RubricLevel defining the scale. Will be
                sorted by score automatically.
            alias: Resource alias for the judge LLM endpoint.
        """
        from inf_engine.module import LLMInference

        self.criteria = criteria
        self.rubric = sorted(rubric, key=lambda r: r.score)
        self._max_score = max(r.score for r in rubric)
        self._min_score = min(r.score for r in rubric)

        # Internal LLM module with structured output
        self.judge: LLMInference = LLMInference(
            alias=alias,
            system_prompt=self._build_system_prompt(),
            response_format=RubricResponse,
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt including rubric definition."""
        rubric_text = "\n".join(
            f"  {level.score} - {level.label}: {level.description}"
            for level in self.rubric
        )
        return f"""You evaluate outputs against a rubric.

Criteria: {self.criteria}

Rating Scale:
{rubric_text}

Always provide a score, justification, and actionable feedback."""

    def bind(self, resources: ResourceConfig | ResourceManager) -> Self:
        """Bind the internal judge module to resources.

        This must be called before using the loss function.

        Args:
            resources: ResourceConfig or ResourceManager containing the
                judge endpoint configuration.

        Returns:
            Self for method chaining.
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
        """Evaluate output using the rubric.

        Args:
            output: The module output to evaluate.
            target: Optional target/expected output for comparison.
            record: ForwardRecord from run(..., record=True).
            context: Optional additional context for evaluation.

        Returns:
            Feedback with normalized score (0-1) and structured evaluation.
        """
        from inf_engine.optimization.feedback import Feedback, FeedbackType

        # Build prompt
        prompt_parts = [f"Output to evaluate:\n{output}"]
        if target:
            prompt_parts.append(f"Expected/Target: {target}")
        prompt = "\n\n".join(prompt_parts)

        # Get structured response
        response = await self.judge(prompt)

        # Handle both dict (parsed JSON) and object responses
        if isinstance(response, dict):
            raw_score = response["score"]
            justification = response["justification"]
            feedback_text = response["feedback"]
        else:
            raw_score = response.score
            justification = response.justification
            feedback_text = response.feedback

        # Normalize score to 0-1 range
        normalized_score = (raw_score - self._min_score) / (
            self._max_score - self._min_score
        )

        content = f"Justification: {justification}\n\nFeedback: {feedback_text}"

        feedback = Feedback(
            content=content,
            score=normalized_score,
            feedback_type=FeedbackType.LLM_JUDGE,
            metadata={"raw_score": raw_score, "criteria": self.criteria},
        )
        return self._attach_record(feedback, record)


class HumanRubricLoss(Loss):
    """Human evaluation against a structured Likert scale rubric.

    Displays the output and rubric to the user via stdout, then
    collects their score and optional feedback via stdin.

    Attributes:
        criteria: What aspect to evaluate.
        rubric: List of RubricLevel defining the scale.
        require_feedback: Whether to require written feedback.

    Example:
        >>> rubric = [
        ...     RubricLevel(1, "Poor", "Fails to address the query"),
        ...     RubricLevel(2, "Below Average", "Partially addresses query"),
        ...     RubricLevel(3, "Average", "Adequately addresses query"),
        ...     RubricLevel(4, "Good", "Thoroughly addresses query"),
        ...     RubricLevel(5, "Excellent", "Exceptionally addresses query"),
        ... ]
        >>> loss = HumanRubricLoss(
        ...     criteria="helpfulness",
        ...     rubric=rubric,
        ... )
        >>> feedback = await loss(output)
        # User sees rubric and scores interactively
    """

    def __init__(
        self,
        criteria: str,
        rubric: list[RubricLevel],
        require_feedback: bool = True,
    ) -> None:
        """Initialize the HumanRubricLoss.

        Args:
            criteria: What aspect to evaluate.
            rubric: List of RubricLevel defining the scale.
            require_feedback: Whether to require written feedback
                in addition to the score. Defaults to True.
        """
        self.criteria = criteria
        self.rubric = sorted(rubric, key=lambda r: r.score)
        self.require_feedback = require_feedback
        self._max_score = max(r.score for r in rubric)
        self._min_score = min(r.score for r in rubric)

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        """Collect human score and feedback against the rubric.

        Displays the output, target, and rubric, then collects
        the user's score and optional written feedback.

        Args:
            output: The module output to evaluate.
            target: Optional target/expected output for comparison.
            record: ForwardRecord from run(..., record=True).
            context: Optional additional context for evaluation.

        Returns:
            Feedback with normalized score (0-1) and human feedback.
        """
        from inf_engine.optimization.feedback import Feedback, FeedbackType

        # Display output and rubric
        print("\n" + "=" * 60)
        print(f"EVALUATE: {self.criteria}")
        print("=" * 60)
        print("\nOutput:")
        print("-" * 40)
        print(output)
        print("-" * 40)

        if target:
            print(f"\nExpected: {target}")

        print("\nRating Scale:")
        for level in self.rubric:
            print(f"  [{level.score}] {level.label}: {level.description}")

        # Collect score
        while True:
            try:
                score_input = input(
                    f"\nYour score ({self._min_score}-{self._max_score}): "
                )
                score = int(score_input)
                if self._min_score <= score <= self._max_score:
                    break
                print(
                    f"Please enter a number between "
                    f"{self._min_score} and {self._max_score}"
                )
            except ValueError:
                print("Please enter a valid number")

        # Collect feedback
        feedback_text = ""
        if self.require_feedback:
            print("\nProvide feedback (enter empty line to finish):")
            lines = []
            while True:
                line = input("> ")
                if not line:
                    break
                lines.append(line)
            feedback_text = "\n".join(lines)

        # Normalize score to 0-1
        normalized_score = (score - self._min_score) / (
            self._max_score - self._min_score
        )

        feedback = Feedback(
            content=feedback_text or f"Score: {score}/{self._max_score}",
            score=normalized_score,
            feedback_type=FeedbackType.HUMAN,
            metadata={"raw_score": score},
        )
        return self._attach_record(feedback, record)


# =============================================================================
# Contrastive Losses
# =============================================================================


class ContrastiveLoss(Loss):
    """Base class for contrastive losses that compare multiple outputs.

    Contrastive losses generate feedback by comparing outputs rather
    than evaluating them in isolation. This often produces more
    actionable feedback about what makes one output better than another.

    Subclasses should implement the comparison logic and use the
    helper methods for generating contrastive feedback.
    """

    def _generate_contrastive_feedback(
        self,
        winner: Any,
        loser: Any,
        reason: str,
    ) -> str:
        """Generate feedback explaining why winner is better than loser.

        Args:
            winner: The preferred output.
            loser: The rejected output.
            reason: Explanation of why winner was preferred.

        Returns:
            Detailed feedback for improving the loser to match the winner.
        """
        return f"""The preferred output was better because: {reason}

To improve, the output should:
- Emulate qualities of the preferred response
- Avoid weaknesses identified in the rejected response

Preferred output characteristics:
{self._summarize_output(winner)}

Rejected output weaknesses:
{self._summarize_output(loser)}"""

    def _summarize_output(self, output: Any) -> str:
        """Truncate output for feedback if too long.

        Args:
            output: The output to summarize.

        Returns:
            String representation, truncated to 200 chars if needed.
        """
        text = str(output)
        if len(text) > 200:
            return text[:200] + "..."
        return text


class LLMPreferenceLoss(ContrastiveLoss):
    """LLM pairwise preference comparison.

    Given two outputs (e.g., from current vs previous parameters),
    the LLM selects which is better and explains why. Feedback is
    generated from the contrast.

    Attributes:
        criteria: What aspect to compare on.
        judge: Internal LLMInference module for comparison.

    Example:
        >>> loss = LLMPreferenceLoss(
        ...     criteria="overall quality",
        ...     alias="judge",
        ... ).bind(resources)
        >>> # Compare two outputs
        >>> fb_a, fb_b = await loss.compare(output_a, output_b)
        >>> # Or use single-output interface with target as comparison
        >>> feedback = await loss(output, target=baseline)
    """

    def __init__(
        self,
        criteria: str,
        alias: str = "judge",
    ) -> None:
        """Initialize the LLMPreferenceLoss.

        Args:
            criteria: What aspect to compare on.
            alias: Resource alias for the judge LLM endpoint.
        """
        from inf_engine.module import LLMInference

        self.criteria = criteria

        # Internal LLM module with structured output
        self.judge: LLMInference = LLMInference(
            alias=alias,
            system_prompt=(
                f"You compare two outputs on: {criteria}. "
                "Determine which is better and explain why."
            ),
            response_format=PreferenceResponse,
        )

    def bind(self, resources: ResourceConfig | ResourceManager) -> Self:
        """Bind the internal judge module to resources.

        Args:
            resources: ResourceConfig or ResourceManager containing the
                judge endpoint configuration.

        Returns:
            Self for method chaining.
        """
        self.judge.bind(resources)
        return self

    async def compare(
        self,
        output_a: Any,
        output_b: Any,
        *,
        record_a: ForwardRecord | None = None,
        record_b: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[Feedback, Feedback]:
        """Compare two outputs and return feedback for each.

        Args:
            output_a: First output to compare.
            output_b: Second output to compare.
            record_a: Optional ForwardRecord for output_a.
            record_b: Optional ForwardRecord for output_b.
            context: Optional context for comparison.

        Returns:
            Tuple of (feedback_for_a, feedback_for_b). The preferred
            output gets positive feedback (score=1.0), the rejected gets
            improvement suggestions (score=0.0).
        """
        from inf_engine.optimization.feedback import Feedback, FeedbackType

        # Build comparison prompt
        prompt_parts = []
        if context:
            prompt_parts.append(f"Context: {context}")
        prompt_parts.append(f"Output A:\n{output_a}")
        prompt_parts.append(f"Output B:\n{output_b}")
        prompt_parts.append("Which output is better?")
        prompt = "\n\n".join(prompt_parts)

        # Get structured response
        response = await self.judge(prompt)

        # Handle both dict and object responses
        if isinstance(response, dict):
            winner = response["winner"]
            reason = response["reason"]
            a_strengths = response["a_strengths"]
            a_weaknesses = response["a_weaknesses"]
            b_strengths = response["b_strengths"]
            b_weaknesses = response["b_weaknesses"]
        else:
            winner = response.winner
            reason = response.reason
            a_strengths = response.a_strengths
            a_weaknesses = response.a_weaknesses
            b_strengths = response.b_strengths
            b_weaknesses = response.b_weaknesses

        # Generate contrastive feedback
        if winner == "A":
            fb_a = Feedback(
                content=f"Preferred. Strengths: {a_strengths}",
                score=1.0,
                feedback_type=FeedbackType.LLM_JUDGE,
            )
            fb_b = Feedback(
                content=self._generate_contrastive_feedback(
                    output_a,
                    output_b,
                    f"{reason}\n\nWeaknesses: {b_weaknesses}",
                ),
                score=0.0,
                feedback_type=FeedbackType.LLM_JUDGE,
            )
        else:
            fb_a = Feedback(
                content=self._generate_contrastive_feedback(
                    output_b,
                    output_a,
                    f"{reason}\n\nWeaknesses: {a_weaknesses}",
                ),
                score=0.0,
                feedback_type=FeedbackType.LLM_JUDGE,
            )
            fb_b = Feedback(
                content=f"Preferred. Strengths: {b_strengths}",
                score=1.0,
                feedback_type=FeedbackType.LLM_JUDGE,
            )

        # Attach records
        if record_a:
            fb_a._record = record_a
        if record_b:
            fb_b._record = record_b

        return fb_a, fb_b

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        """Single-output interface using target as comparison baseline.

        Args:
            output: The output to evaluate.
            target: Required baseline output to compare against.
            record: ForwardRecord from run(..., record=True).
            context: Optional additional context.

        Returns:
            Feedback for the output based on comparison with target.

        Raises:
            ValueError: If target is None (required for comparison).
        """
        if target is None:
            raise ValueError("LLMPreferenceLoss requires target for comparison")
        fb_output, _ = await self.compare(
            output, target, record_a=record, context=context
        )
        return fb_output


class HumanPreferenceLoss(ContrastiveLoss):
    """Human pairwise preference comparison via stdout.

    Displays two outputs side-by-side and asks the user to select
    which is better and explain why.

    Attributes:
        criteria: What aspect to compare on.
        require_reason: Whether to require explanation.

    Example:
        >>> loss = HumanPreferenceLoss(
        ...     criteria="overall quality",
        ...     require_reason=True,
        ... )
        >>> fb_a, fb_b = await loss.compare(output_a, output_b)
    """

    def __init__(
        self,
        criteria: str,
        require_reason: bool = True,
    ) -> None:
        """Initialize the HumanPreferenceLoss.

        Args:
            criteria: What aspect to compare on.
            require_reason: Whether to require explanation for choice.
                Defaults to True.
        """
        self.criteria = criteria
        self.require_reason = require_reason

    async def compare(
        self,
        output_a: Any,
        output_b: Any,
        *,
        record_a: ForwardRecord | None = None,
        record_b: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[Feedback, Feedback]:
        """Compare two outputs and return feedback for each.

        Displays both outputs to the user and collects their preference.

        Args:
            output_a: First output to compare.
            output_b: Second output to compare.
            record_a: Optional ForwardRecord for output_a.
            record_b: Optional ForwardRecord for output_b.
            context: Optional context for comparison.

        Returns:
            Tuple of (feedback_for_a, feedback_for_b).
        """
        from inf_engine.optimization.feedback import Feedback, FeedbackType

        print("\n" + "=" * 60)
        print(f"COMPARE: {self.criteria}")
        print("=" * 60)

        if context:
            print(f"\nContext: {context}")

        print("\n[A] Output A:")
        print("-" * 40)
        print(output_a)

        print("\n[B] Output B:")
        print("-" * 40)
        print(output_b)

        print("=" * 60)

        # Get preference
        while True:
            choice = input("\nWhich is better? (A/B): ").strip().upper()
            if choice in ("A", "B"):
                break
            print("Please enter A or B")

        # Get reason
        reason = ""
        if self.require_reason:
            print("\nWhy is it better? (enter empty line to finish):")
            lines = []
            while True:
                line = input("> ")
                if not line:
                    break
                lines.append(line)
            reason = "\n".join(lines)

        # Generate feedback
        winner, loser = (output_a, output_b) if choice == "A" else (output_b, output_a)

        if choice == "A":
            fb_a = Feedback(
                content=f"Preferred by human. Reason: {reason}",
                score=1.0,
                feedback_type=FeedbackType.HUMAN,
            )
            fb_b = Feedback(
                content=self._generate_contrastive_feedback(winner, loser, reason),
                score=0.0,
                feedback_type=FeedbackType.HUMAN,
            )
        else:
            fb_a = Feedback(
                content=self._generate_contrastive_feedback(winner, loser, reason),
                score=0.0,
                feedback_type=FeedbackType.HUMAN,
            )
            fb_b = Feedback(
                content=f"Preferred by human. Reason: {reason}",
                score=1.0,
                feedback_type=FeedbackType.HUMAN,
            )

        if record_a:
            fb_a._record = record_a
        if record_b:
            fb_b._record = record_b

        return fb_a, fb_b

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        """Single-output interface using target as comparison baseline.

        Args:
            output: The output to evaluate.
            target: Required baseline output to compare against.
            record: ForwardRecord from run(..., record=True).
            context: Optional additional context.

        Returns:
            Feedback for the output based on comparison with target.

        Raises:
            ValueError: If target is None (required for comparison).
        """
        if target is None:
            raise ValueError("HumanPreferenceLoss requires target for comparison")
        fb_output, _ = await self.compare(
            output, target, record_a=record, context=context
        )
        return fb_output


class LLMRankingLoss(ContrastiveLoss):
    """LLM ranking of multiple outputs.

    Given n outputs, the LLM ranks them from best to worst and
    explains the ranking. Feedback is generated based on relative
    position and comparison to better-ranked outputs.

    Attributes:
        criteria: What aspect to rank on.
        n: Expected number of outputs to compare.
        judge: Internal LLMInference module for ranking.

    Example:
        >>> loss = LLMRankingLoss(
        ...     criteria="response quality",
        ...     n=4,
        ...     alias="judge",
        ... ).bind(resources)
        >>> feedbacks = await loss.rank([out1, out2, out3, out4])
        >>> for i, fb in enumerate(feedbacks):
        ...     print(f"Output {i}: score={fb.score:.2f}")
    """

    def __init__(
        self,
        criteria: str,
        n: int = 4,
        alias: str = "judge",
    ) -> None:
        """Initialize the LLMRankingLoss.

        Args:
            criteria: What aspect to rank on.
            n: Expected number of outputs to compare. Defaults to 4.
            alias: Resource alias for the judge LLM endpoint.
        """
        from inf_engine.module import LLMInference

        self.criteria = criteria
        self.n = n

        # Internal LLM module with structured output
        self.judge: LLMInference = LLMInference(
            alias=alias,
            system_prompt=(
                f"You rank outputs from best to worst on: {criteria}. "
                "Provide the ranking as a list of indices and explain "
                "your reasoning."
            ),
            response_format=RankingResponse,
        )

    def bind(self, resources: ResourceConfig | ResourceManager) -> Self:
        """Bind the internal judge module to resources.

        Args:
            resources: ResourceConfig or ResourceManager containing the
                judge endpoint configuration.

        Returns:
            Self for method chaining.
        """
        self.judge.bind(resources)
        return self

    async def rank(
        self,
        outputs: list[Any],
        *,
        records: list[ForwardRecord | None] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Feedback]:
        """Rank multiple outputs and return feedback for each.

        Args:
            outputs: List of outputs to rank. Must have at least 2.
            records: Optional list of ForwardRecords (same length as outputs).
            context: Optional context for ranking.

        Returns:
            List of Feedback objects in same order as inputs.
            Scores are normalized by rank (best=1.0, worst=0.0).

        Raises:
            ValueError: If fewer than 2 outputs provided.
        """
        from inf_engine.optimization.feedback import Feedback, FeedbackType

        if len(outputs) < 2:
            raise ValueError("Need at least 2 outputs to rank")

        # Build prompt
        output_strs = [f"[{i + 1}] {out}" for i, out in enumerate(outputs)]

        prompt_parts = []
        if context:
            prompt_parts.append(f"Context: {context}")
        prompt_parts.append("Outputs to rank:")
        prompt_parts.append("\n".join(output_strs))
        prompt_parts.append(f"Rank these {len(outputs)} outputs from BEST to WORST.")
        prompt = "\n\n".join(prompt_parts)

        # Get structured response
        response = await self.judge(prompt)

        # Handle both dict and object responses
        if isinstance(response, dict):
            raw_ranking = response["ranking"]
            best_qualities = response["best_qualities"]
            worst_issues = response["worst_issues"]
            comparison = response["comparison"]
        else:
            raw_ranking = response.ranking
            best_qualities = response.best_qualities
            worst_issues = response.worst_issues
            comparison = response.comparison

        # Convert 1-indexed ranking to 0-indexed
        ranking = [r - 1 for r in raw_ranking]

        # Generate feedback based on rank
        feedbacks = []
        n = len(outputs)
        for i in range(n):
            rank = ranking.index(i) + 1  # 1-indexed rank for display
            score = (n - rank) / (n - 1) if n > 1 else 1.0  # Normalize to 0-1

            if rank == 1:
                content = f"Ranked #1 (best). {best_qualities}"
            elif rank == n:
                content = (
                    f"Ranked #{rank} (worst). {worst_issues}\n\n"
                    f"To improve, emulate the #1 output's qualities: "
                    f"{best_qualities}"
                )
            else:
                content = (
                    f"Ranked #{rank}/{n}. {comparison}\n\n"
                    f"To improve, move toward qualities of higher-ranked outputs."
                )

            fb = Feedback(
                content=content,
                score=score,
                feedback_type=FeedbackType.LLM_JUDGE,
                metadata={"rank": rank, "total": n},
            )

            if records and i < len(records) and records[i]:
                fb._record = records[i]

            feedbacks.append(fb)

        return feedbacks

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        """Single-output interface that ranks output against target(s).

        Args:
            output: The output to evaluate.
            target: Required baseline(s) to compare against. Can be a single
                output or a list of outputs.
            record: ForwardRecord from run(..., record=True).
            context: Optional additional context.

        Returns:
            Feedback for the output based on its ranking.

        Raises:
            ValueError: If target is None (required for comparison).
        """
        if target is None:
            raise ValueError("LLMRankingLoss requires target for comparison")
        targets = target if isinstance(target, list) else [target]
        outputs = [output] + targets
        records_list: list[ForwardRecord | None] = [record] + [None] * len(targets)
        feedbacks = await self.rank(outputs, records=records_list, context=context)
        return feedbacks[0]


class HumanRankingLoss(ContrastiveLoss):
    """Human ranking of multiple outputs via stdout.

    Displays n outputs and asks the user to rank them from best to
    worst, with optional feedback.

    Attributes:
        criteria: What aspect to rank on.
        n: Expected number of outputs to compare.
        require_feedback: Whether to require written feedback.

    Example:
        >>> loss = HumanRankingLoss(
        ...     criteria="response quality",
        ...     n=4,
        ...     require_feedback=True,
        ... )
        >>> feedbacks = await loss.rank([out1, out2, out3, out4])
    """

    def __init__(
        self,
        criteria: str,
        n: int = 4,
        require_feedback: bool = True,
    ) -> None:
        """Initialize the HumanRankingLoss.

        Args:
            criteria: What aspect to rank on.
            n: Expected number of outputs to compare. Defaults to 4.
            require_feedback: Whether to require written feedback.
                Defaults to True.
        """
        self.criteria = criteria
        self.n = n
        self.require_feedback = require_feedback

    async def rank(
        self,
        outputs: list[Any],
        *,
        records: list[ForwardRecord | None] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Feedback]:
        """Rank multiple outputs and return feedback for each.

        Displays all outputs to the user and collects their ranking.

        Args:
            outputs: List of outputs to rank. Must have at least 2.
            records: Optional list of ForwardRecords (same length as outputs).
            context: Optional context for ranking.

        Returns:
            List of Feedback objects in same order as inputs.

        Raises:
            ValueError: If fewer than 2 outputs provided.
        """
        from inf_engine.optimization.feedback import Feedback, FeedbackType

        if len(outputs) < 2:
            raise ValueError("Need at least 2 outputs to rank")

        # Display outputs
        print("\n" + "=" * 60)
        print(f"RANK THESE OUTPUTS: {self.criteria}")
        print("=" * 60)

        if context:
            print(f"\nContext: {context}")

        for i, out in enumerate(outputs):
            print(f"\n[{i + 1}] Output {i + 1}:")
            print("-" * 40)
            print(out)

        print("=" * 60)

        # Get ranking
        while True:
            try:
                ranking_input = input(
                    "\nRank from best to worst (e.g., '3,1,2' if 3 is best): "
                )
                ranking = [int(x.strip()) - 1 for x in ranking_input.split(",")]
                if len(ranking) == len(outputs) and set(ranking) == set(
                    range(len(outputs))
                ):
                    break
                print(f"Please rank all {len(outputs)} outputs exactly once")
            except ValueError:
                print("Please enter comma-separated numbers")

        # Get feedback
        feedback_text = ""
        if self.require_feedback:
            print("\nExplain your ranking (enter empty line to finish):")
            lines = []
            while True:
                line = input("> ")
                if not line:
                    break
                lines.append(line)
            feedback_text = "\n".join(lines)

        # Generate feedback based on rank
        feedbacks = []
        n = len(outputs)
        for i in range(n):
            rank = ranking.index(i) + 1
            score = (n - rank) / (n - 1) if n > 1 else 1.0

            if rank == 1:
                content = f"Ranked #1 (best) by human.\n\n{feedback_text}"
            elif rank == n:
                content = f"Ranked #{rank} (worst) by human.\n\n{feedback_text}"
            else:
                content = f"Ranked #{rank}/{n} by human.\n\n{feedback_text}"

            fb = Feedback(
                content=content,
                score=score,
                feedback_type=FeedbackType.HUMAN,
                metadata={"rank": rank, "total": n},
            )

            if records and i < len(records) and records[i]:
                fb._record = records[i]

            feedbacks.append(fb)

        return feedbacks

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Feedback:
        """Single-output interface that ranks output against target(s).

        Args:
            output: The output to evaluate.
            target: Required baseline(s) to compare against. Can be a single
                output or a list of outputs.
            record: ForwardRecord from run(..., record=True).
            context: Optional additional context.

        Returns:
            Feedback for the output based on its ranking.

        Raises:
            ValueError: If target is None (required for comparison).
        """
        if target is None:
            raise ValueError("HumanRankingLoss requires target for comparison")
        targets = target if isinstance(target, list) else [target]
        outputs = [output] + targets
        records_list: list[ForwardRecord | None] = [record] + [None] * len(targets)
        feedbacks = await self.rank(outputs, records=records_list, context=context)
        return feedbacks[0]
