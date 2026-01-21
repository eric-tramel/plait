"""Optimization system for LLM-based parameter learning.

This package provides the infrastructure for optimizing Module
parameters through backward passes. Instead of numerical gradients,
feedback is propagated through the computation graph to improve
Parameters (prompts, instructions, etc.).

The core workflow mirrors PyTorch:
    1. Compose model + loss into a traced step (TrainingStep)
    2. Forward pass: `loss = await step(input, target)`
    3. Backward pass: `await loss.backward()`
    4. Update parameters: `await optimizer.step()`

Example:
    >>> from plait import run
    >>> from plait.optimization import ForwardRecord, TrainingStep
    >>>
    >>> # Execute with recording to enable backward pass
    >>> step = TrainingStep(module, loss_fn)
    >>> loss_val = await step("input text", target)
    >>> await loss_val.backward()
"""

from plait.optimization.backward import BackwardContext, BackwardResult
from plait.optimization.loss import (
    CompositeLoss,
    HumanFeedbackLoss,
    HumanPreferenceLoss,
    HumanRankingLoss,
    HumanRubricLoss,
    LLMJudge,
    LLMPreferenceLoss,
    LLMRankingLoss,
    LLMRubricLoss,
    LossModule,
    PreferenceResponse,
    RankingResponse,
    RubricLevel,
    RubricResponse,
    VerifierLoss,
)
from plait.optimization.optimizer import Optimizer, SFAOptimizer
from plait.optimization.record import ForwardRecord
from plait.optimization.training import TrainingStep

__all__ = [
    # Backward pass
    "BackwardContext",
    "BackwardResult",
    # Record
    "ForwardRecord",
    "TrainingStep",
    # Single-sample losses
    "VerifierLoss",
    "LLMJudge",
    "HumanFeedbackLoss",
    "LLMRubricLoss",
    "HumanRubricLoss",
    "LossModule",
    # Contrastive losses
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
