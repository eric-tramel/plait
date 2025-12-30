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
    >>> from inf_engine.optimization import ForwardRecord
    >>>
    >>> # Execute with recording to enable backward pass
    >>> output, record = await run(module, "input text", record=True)
    >>> isinstance(record, ForwardRecord)
    True
"""

from inf_engine.optimization.record import ForwardRecord

__all__ = ["ForwardRecord"]
