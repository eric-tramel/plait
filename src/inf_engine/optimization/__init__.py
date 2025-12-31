"""Optimization system for LLM-based parameter learning.

This package provides the infrastructure for optimizing InferenceModule
parameters through backward passes. Instead of numerical gradients,
feedback is propagated through the computation graph to improve
Parameters (prompts, instructions, etc.).

The core workflow mirrors PyTorch:
    1. Forward pass with recording: `output, record = await run(module, input, record=True)`
    2. Compute feedback: `feedback = await loss_fn(output, target, record=record)`
    3. Backward pass: `await feedback.backward()`
    4. Update parameters: `await optimizer.step()`

Example:
    >>> from inf_engine import run
    >>> from inf_engine.optimization import ForwardRecord, Feedback, FeedbackType
    >>>
    >>> # Execute with recording to enable backward pass
    >>> output, record = await run(module, "input text", record=True)
    >>> isinstance(record, ForwardRecord)
    True
    >>>
    >>> # Feedback represents evaluation results
    >>> feedback = Feedback(
    ...     content="Response was helpful",
    ...     score=0.9,
    ...     feedback_type=FeedbackType.LLM_JUDGE,
    ... )
    >>> str(feedback)
    '[0.90] Response was helpful'
"""

from inf_engine.optimization.backward import BackwardContext, BackwardResult
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
from inf_engine.optimization.optimizer import Optimizer, SFAOptimizer
from inf_engine.optimization.record import ForwardRecord

__all__ = [
    # Backward pass
    "BackwardContext",
    "BackwardResult",
    # Feedback
    "Feedback",
    "FeedbackType",
    # Record
    "ForwardRecord",
    # Loss base class
    "Loss",
    # Single-sample losses
    "VerifierLoss",
    "LLMJudge",
    "HumanFeedbackLoss",
    "LLMRubricLoss",
    "HumanRubricLoss",
    # Contrastive losses
    "ContrastiveLoss",
    "LLMPreferenceLoss",
    "HumanPreferenceLoss",
    "LLMRankingLoss",
    "HumanRankingLoss",
    # Composite loss
    "CompositeLoss",
    # Structured output schemas
    "RubricLevel",
    "RubricResponse",
    "PreferenceResponse",
    "RankingResponse",
    # Optimizers
    "Optimizer",
    "SFAOptimizer",
]
