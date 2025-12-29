"""Execution types for inf-engine.

This module provides types used in execution patterns, particularly
for streaming batch execution where results are yielded as they complete.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchResult[T]:
    """Result wrapper for streaming batch execution.

    Wraps individual results from streaming batch execution with full context,
    enabling robust error handling and correlation with inputs.

    When using streaming batch execution (`streaming=True` in ExecutionSettings),
    each result is yielded as a `BatchResult` that contains:
    - The original index in the input list
    - The original input value
    - Either the successful output or the error

    Attributes:
        index: Position in the original input list (0-based).
        input: The original input value that produced this result.
        output: The result value if successful, None if failed.
        error: The exception if failed, None if successful.

    Example:
        >>> result = BatchResult(index=0, input="hello", output="HELLO", error=None)
        >>> result.ok
        True
        >>> result.output
        'HELLO'

    Example with error:
        >>> result = BatchResult(
        ...     index=1,
        ...     input="bad input",
        ...     output=None,
        ...     error=ValueError("Invalid input"),
        ... )
        >>> result.ok
        False
        >>> result.error
        ValueError('Invalid input')

    Example usage in streaming:
        >>> async with ExecutionSettings(resources=config, streaming=True):
        ...     async for result in pipeline(["doc1", "doc2"]):
        ...         if result.ok:
        ...             print(f"Input {result.index}: {result.output}")
        ...         else:
        ...             print(f"Input {result.index} failed: {result.error}")

    Note:
        BatchResult is immutable (frozen=True) to ensure result integrity
        after yielding to the consumer.
    """

    index: int
    input: T
    output: T | None
    error: Exception | None

    @property
    def ok(self) -> bool:
        """Check if the result is successful.

        A result is considered successful if no error occurred.

        Returns:
            True if error is None (successful), False otherwise.

        Example:
            >>> success = BatchResult(index=0, input="x", output="X", error=None)
            >>> success.ok
            True
            >>> failure = BatchResult(
            ...     index=0, input="x", output=None, error=ValueError("fail")
            ... )
            >>> failure.ok
            False
        """
        return self.error is None

    def __repr__(self) -> str:
        """Return a concise string representation.

        Returns:
            A string showing index, ok status, and output or error.
        """
        if self.ok:
            return f"BatchResult(index={self.index}, ok=True, output={self.output!r})"
        return f"BatchResult(index={self.index}, ok=False, error={self.error!r})"
