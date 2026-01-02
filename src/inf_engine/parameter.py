"""Parameter class for learnable values in inf-engine.

Parameters hold values that can be optimized via backward passes,
similar to torch.nn.Parameter but for prompt optimization rather than
gradient descent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from inf_engine.module import InferenceModule


@dataclass
class Parameter:
    """A learnable value that can be optimized via backward passes.

    Similar to torch.nn.Parameter, but for values (prompts, instructions,
    structured configs, etc.) that are optimized via LLM feedback rather
    than gradient descent.

    The description field is required when requires_grad=True to enable
    the optimizer to understand how to improve the parameter.

    Args:
        value: The current value of the parameter (string, dict, list, etc.).
        description: A description of what this parameter does/represents.
            Required when requires_grad=True to enable self-documenting
            optimization. Can be None when requires_grad=False.
        requires_grad: If True, feedback will be accumulated during backward
            passes and description is required. If False, the parameter is
            treated as a constant.

    Raises:
        ValueError: If requires_grad=True but description is None.

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

        >>> # Constant parameter (no description required)
        >>> const = Parameter({"model": "gpt-4"}, requires_grad=False)
        >>> const.requires_grad
        False
    """

    value: Any
    description: str | None = None
    requires_grad: bool = True
    _name: str | None = field(default=None, repr=False, compare=False)
    _parent: InferenceModule | None = field(default=None, repr=False, compare=False)
    _id: str = field(
        default_factory=lambda: uuid4().hex, init=False, repr=False, compare=False
    )
    _feedback_buffer: list[str] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate that description is provided when requires_grad=True.

        Raises:
            ValueError: If requires_grad=True but description is None.
        """
        if self.requires_grad and self.description is None:
            raise ValueError(
                "Parameter description is required when requires_grad=True. "
                "Provide a description explaining what this parameter represents "
                "to enable meaningful optimization feedback."
            )

    def __str__(self) -> str:
        """Return the current value as a string.

        Returns:
            The string representation of the parameter value.
        """
        return str(self.value)

    def _get_hierarchical_name(self) -> str | None:
        """Return the full hierarchical name for this parameter.

        Builds a dot-separated path from the owning module chain. If the
        parameter is unowned, returns the local name.

        Returns:
            The hierarchical name (e.g., "child.prompt"), or None if unnamed.
        """
        if self._name is None:
            return None

        if self._parent is None:
            return self._name

        parts: list[str] = []
        module = self._parent
        while module is not None:
            module_name = getattr(module, "_name", None)
            if module_name:
                parts.append(module_name)
            module = getattr(module, "_parent", None)

        parts.reverse()
        parts.append(self._name)
        return ".".join(parts)

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

    def apply_update(self, new_value: Any) -> None:
        """Apply an optimizer-computed update.

        Updates the parameter value and clears the feedback buffer.

        Args:
            new_value: The new value to set.
        """
        self.value = new_value
        if self._parent is not None:
            self._parent._increment_state_version()
        self._feedback_buffer.clear()

    def zero_feedback(self) -> None:
        """Clear accumulated feedback without updating the value.

        Similar to zero_grad() in PyTorch, this clears the feedback
        buffer to prepare for a new backward pass.
        """
        self._feedback_buffer.clear()
