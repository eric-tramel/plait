"""Tests for TrainingStep behavior."""

from plait.module import Module
from plait.optimization import TrainingStep
from plait.tracing.context import trace_context
from plait.tracing.tracer import Tracer
from plait.values import Value


class Child(Module):
    def forward(self, x: str) -> str:
        return f"{x}!"


class FlagModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.child = Child()
        self.called_via_call = False
        self.called_via_forward = False

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.called_via_call = True
        return super().__call__(*args, **kwargs)

    def forward(self, x: str) -> str:
        self.called_via_forward = True
        return self.child(x)


class CaptureLoss(Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: tuple[object, object, object] | None = None

    def forward(
        self, output: object, target: object | None = None, context: dict | None = None
    ) -> object:
        self.seen = (output, target, context)
        return {"loss": output}


def test_training_step_calls_model_without_trace() -> None:
    model = FlagModel()
    loss = CaptureLoss()
    step = TrainingStep(model, loss)

    result = step.forward("hi", target="t", context={"k": "v"})

    assert result == {"loss": "hi!"}
    assert model.called_via_call is True
    assert model.called_via_forward is True
    assert loss.seen == ("hi!", "t", {"k": "v"})


def test_training_step_uses_forward_in_trace_context() -> None:
    model = FlagModel()
    loss = CaptureLoss()
    step = TrainingStep(model, loss)

    tracer = Tracer()
    with trace_context(tracer):
        result = step.forward("hi")

    assert model.called_via_call is False
    assert model.called_via_forward is True
    assert isinstance(result, Value)
    assert result.ref in tracer.nodes
