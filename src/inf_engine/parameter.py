"""Parameter class for learnable values in inf-engine.

Parameters hold string values that can be optimized via backward passes,
similar to torch.nn.Parameter but for prompt optimization rather than
gradient descent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Parameter:
    """A learnable value that can be optimized via backward passes.

    Similar to torch.nn.Parameter, but for string values (prompts,
    instructions, etc.) that are optimized via LLM feedback rather
    than gradient descent.

    The description field is REQUIRED and should explain what this parameter
    represents, enabling the optimizer to understand how to improve it.

    Args:
        value: The current string value of the parameter.
        description: A description of what this parameter does/represents.
            This is required to enable self-documenting optimization.
        requires_grad: If True, feedback will be accumulated during backward
            passes. If False, the parameter is treated as a constant.

    Example:
        >>> param = Parameter(
        ...     "You are a helpful assistant.",
        ...     description="Defines the agent's identity and baseline behavior."
        ... )
        >>> str(param)
        'You are a helpful assistant.'
        >>> param.description
        "Defines the agent's identity and baseline behavior."
        >>> param.accumulate_feedback("Be more concise")
        >>> param.get_accumulated_feedback()
        ['Be more concise']
        >>> param.apply_update("You are a concise, helpful assistant.")
        >>> str(param)
        'You are a concise, helpful assistant.'
    """

    value: str
    description: str
    requires_grad: bool = True
    _name: str | None = field(default=None, repr=False, compare=False)
    _feedback_buffer: list[str] = field(default_factory=list, repr=False, compare=False)

    def __str__(self) -> str:
        """Return the current value when used as a string.

        Returns:
            The current string value of the parameter.
        """
        return self.value

    def accumulate_feedback(self, feedback: str) -> None:
        """Collect feedback from backward passes.

        Feedback is only accumulated if requires_grad is True.

        Args:
            feedback: The feedback string to accumulate.
        """
        if self.requires_grad:
            self._feedback_buffer.append(feedback)

    def get_accumulated_feedback(self) -> list[str]:
        """Get all accumulated feedback.

        Returns:
            A copy of the list of accumulated feedback strings.
        """
        return list(self._feedback_buffer)

    def apply_update(self, new_value: str) -> None:
        """Apply an optimizer-computed update.

        Updates the parameter value and clears the feedback buffer.

        Args:
            new_value: The new string value to set.
        """
        self.value = new_value
        self._feedback_buffer.clear()

    def zero_feedback(self) -> None:
        """Clear accumulated feedback without updating the value.

        Similar to zero_grad() in PyTorch, this clears the feedback
        buffer to prepare for a new backward pass.
        """
        self._feedback_buffer.clear()
