"""Optimization system for LLM-based parameter learning.

This package provides the infrastructure for optimizing Module
parameters through backward passes. Instead of numerical gradients,
feedback is propagated through the computation graph to improve
Parameters (prompts, instructions, etc.).

The core workflow mirrors PyTorch:
    1. Forward pass with recording: `output, record = await run(module, input, record=True)`
    2. Compute loss value: `loss_val = await loss_fn(output, target)`
    3. Backward pass: `await loss_val.backward()`
    4. Update parameters: `await optimizer.step()`

Example:
    >>> from plait import run
    >>> from plait.optimization import ForwardRecord
    >>>
    >>> # Execute with recording to enable backward pass
    >>> output, record = await run(module, "input text", record=True)
    >>> isinstance(record, ForwardRecord)
    True
    >>>
    >>> # Loss functions return Value objects
    >>> loss_val = await loss_fn(output, target)
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
    PreferenceResponse,
    RankingResponse,
    RubricLevel,
    RubricResponse,
    VerifierLoss,
)
from plait.optimization.optimizer import Optimizer, SFAOptimizer
from plait.optimization.record import ForwardRecord

__all__ = [
    # Backward pass
    "BackwardContext",
    "BackwardResult",
    # Record
    "ForwardRecord",
    # Single-sample losses
    "VerifierLoss",
    "LLMJudge",
    "HumanFeedbackLoss",
    "LLMRubricLoss",
    "HumanRubricLoss",
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
