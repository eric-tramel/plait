"""Integration tests for per-record tape semantics."""

import pytest

from plait.execution.executor import run
from plait.module import Module
from plait.parameter import Parameter
from plait.values import unwrap


class Branch(Module):
    """Simple branch with a direct parameter."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag
        self.prompt = Parameter(
            value=f"{tag}_prompt",
            description=f"{tag} branch prompt",
            requires_grad=True,
        )

    def forward(self, text: str) -> str:
        return f"{self.prompt.value}:{text}"


class Switch(Module):
    """Chooses a branch based on internal state."""

    def __init__(self) -> None:
        super().__init__()
        self.left = Branch("left")
        self.right = Branch("right")
        self.use_left = True

    def forward(self, text: str) -> str:
        if self.use_left:
            return self.left(text)
        return self.right(text)


@pytest.mark.asyncio
async def test_per_record_tape_tracks_runtime_branch_parameters() -> None:
    """Per-record tape reflects the branch executed at runtime."""
    module = Switch()

    # Record with left branch
    module.use_left = True
    left_output, left_record = await run(module, "hello", record=True)
    left_params = [p for params in left_record.node_parameters.values() for p in params]

    assert "left_prompt" in unwrap(left_output)
    assert any(p is module.left.prompt for p in left_params)
    assert not any(p is module.right.prompt for p in left_params)

    # Record with right branch
    module.use_left = False
    right_output, right_record = await run(module, "hello", record=True)
    right_params = [
        p for params in right_record.node_parameters.values() for p in params
    ]

    assert "right_prompt" in unwrap(right_output)
    assert any(p is module.right.prompt for p in right_params)
    assert not any(p is module.left.prompt for p in right_params)
