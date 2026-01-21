"""Training helpers for optimization loops."""

from __future__ import annotations

from typing import Any

from plait.module import Module
from plait.tracing.context import get_trace_context


class TrainingStep(Module):
    """Compose a model and loss module into a single traced step.

    The returned loss Value carries tape ids so `loss.backward()` works
    without passing an optimizer or grad explicitly.
    """

    model: Module
    loss: Module

    def __init__(
        self,
        model: Module,
        loss: Module,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(
        self,
        *args: Any,
        target: Any | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        tracer = get_trace_context()
        if tracer is not None and any(self.model.children()):
            output = self.model.forward(*args, **kwargs)
        else:
            output = self.model(*args, **kwargs)
        loss_value = self.loss(output, target=target, context=context)
        return loss_value
