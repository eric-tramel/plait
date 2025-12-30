"""Backward pass infrastructure for feedback propagation.

This module will provide the infrastructure for propagating feedback
backward through traced computation graphs. The full implementation
will be added in PR-067.

Note:
    This is a stub module. The full implementation including BackwardContext,
    BackwardResult, and _propagate_backward() will be added in PR-067.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inf_engine.optimization.feedback import Feedback
    from inf_engine.optimization.record import ForwardRecord


async def _propagate_backward(
    feedback: Feedback,
    record: ForwardRecord,
    reasoning_llm: Any = None,
) -> None:
    """Propagate feedback backward through a traced graph.

    Traverses nodes in reverse topological order, calling each module's
    backward() method and accumulating feedback into Parameters.

    Args:
        feedback: The feedback to propagate (from loss function).
        record: ForwardRecord from the forward pass.
        reasoning_llm: Optional LLM for backward reasoning.

    Note:
        This is a stub implementation. The full implementation will be
        added in PR-067 (Backward pass infrastructure).
    """
    # Stub implementation - full implementation in PR-067
    # For now, this function exists to allow Feedback.backward() to be called
    # with a ForwardRecord attached, even though it won't actually propagate yet.
    pass
