"""Custom exception types for plait.

This module defines the exception hierarchy used throughout plait
for error handling and recovery. All exceptions inherit from InfEngineError,
enabling catch-all handling when needed while allowing specific error types
to be caught individually.

Exception Hierarchy:
    InfEngineError (base)
    ├── RateLimitError - API rate limiting and backpressure
    ├── TransientError - Temporary failures that may succeed on retry
    ├── ExecutionError - Task execution failures
    └── OptimizationError - Parameter optimization failures

Example:
    >>> from plait.errors import InfEngineError, RateLimitError
    >>>
    >>> try:
    ...     # ... API call that might hit rate limits
    ...     pass
    ... except RateLimitError as e:
    ...     if e.retry_after:
    ...         print(f"Retry after {e.retry_after} seconds")
    ... except InfEngineError:
    ...     print("Some other plait error")
"""


class InfEngineError(Exception):
    """Base exception for all plait errors.

    This is the root of the plait exception hierarchy. All custom
    exceptions in plait inherit from this class, allowing users to
    catch all plait errors with a single except clause while still
    being able to handle specific error types individually.

    Args:
        message: A human-readable description of the error.

    Example:
        >>> try:
        ...     raise InfEngineError("Something went wrong")
        ... except InfEngineError as e:
        ...     print(f"Caught error: {e}")
        Caught error: Something went wrong
    """

    def __init__(self, message: str) -> None:
        """Initialize the error with a message.

        Args:
            message: A human-readable description of the error.
        """
        super().__init__(message)
        self.message = message


class RateLimitError(InfEngineError):
    """Error raised when an API rate limit is hit.

    Raised when an LLM endpoint returns a rate limit response (typically
    HTTP 429). Contains information about when to retry, enabling the
    scheduler to implement adaptive backpressure.

    Args:
        message: A human-readable description of the rate limit error.
        retry_after: Optional number of seconds to wait before retrying.
            If provided by the API (via Retry-After header), this gives
            a hint for how long to wait. If None, the caller should use
            exponential backoff.

    Attributes:
        message: The error message.
        retry_after: Seconds to wait before retrying, or None if not specified.

    Example:
        >>> error = RateLimitError("Rate limit exceeded", retry_after=30.0)
        >>> error.retry_after
        30.0
        >>> str(error)
        'Rate limit exceeded'

        >>> # Error without retry hint
        >>> error = RateLimitError("Too many requests")
        >>> error.retry_after is None
        True
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        """Initialize the rate limit error.

        Args:
            message: A human-readable description of the rate limit error.
            retry_after: Optional seconds to wait before retrying.
        """
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(InfEngineError):
    """Error for transient failures that may succeed on retry.

    Raised for connection errors, server errors (5xx), and other temporary
    failures that may resolve on retry. The scheduler will automatically
    retry these errors if max_task_retries > 0.

    Unlike RateLimitError (which triggers backoff on the rate limiter),
    TransientError uses the configured retry delay with exponential backoff.

    Args:
        message: A human-readable description of the transient failure.

    Example:
        >>> error = TransientError("Connection timeout to API server")
        >>> str(error)
        'Connection timeout to API server'

        >>> # Distinguishing from other error types
        >>> isinstance(TransientError("timeout"), RateLimitError)
        False
        >>> isinstance(TransientError("timeout"), InfEngineError)
        True
    """

    def __init__(self, message: str) -> None:
        """Initialize the transient error.

        Args:
            message: A human-readable description of the transient failure.
        """
        super().__init__(message)


class ExecutionError(InfEngineError):
    """Error raised when task execution fails.

    Raised when a task fails during execution due to an error in the
    module's forward() method or during LLM API calls. This error wraps
    the underlying exception and provides context about which node failed.

    Args:
        message: A human-readable description of the execution error.
        node_id: Optional identifier of the node that failed. Useful for
            debugging and for the scheduler to mark dependent nodes as
            cancelled.
        cause: Optional underlying exception that caused this error.
            Preserved for debugging and logging purposes.

    Attributes:
        message: The error message.
        node_id: The ID of the failed node, or None if not applicable.
        cause: The underlying exception, or None if not applicable.

    Example:
        >>> # Basic execution error
        >>> error = ExecutionError("Task failed")
        >>> str(error)
        'Task failed'

        >>> # Error with node context
        >>> error = ExecutionError(
        ...     "LLM call failed",
        ...     node_id="node_123",
        ...     cause=ValueError("Invalid response"),
        ... )
        >>> error.node_id
        'node_123'
        >>> isinstance(error.cause, ValueError)
        True
    """

    def __init__(
        self,
        message: str,
        node_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the execution error.

        Args:
            message: A human-readable description of the execution error.
            node_id: Optional identifier of the node that failed.
            cause: Optional underlying exception that caused this error.
        """
        super().__init__(message)
        self.node_id = node_id
        self.cause = cause


class OptimizationError(InfEngineError):
    """Error raised when parameter optimization fails.

    Raised when the optimizer cannot produce valid parameter updates
    after exhausting retry attempts. This typically occurs when the
    updater LLM returns empty or malformed responses.

    Args:
        message: A human-readable description of the optimization error.
        parameter_name: Optional name of the parameter that failed to update.
        attempts: Number of attempts made before failing.

    Attributes:
        message: The error message.
        parameter_name: The name of the failed parameter, or None if not applicable.
        attempts: The number of attempts made before failing.

    Example:
        >>> error = OptimizationError(
        ...     "Failed to generate update for parameter after 3 attempts",
        ...     parameter_name="system_prompt",
        ...     attempts=3,
        ... )
        >>> error.parameter_name
        'system_prompt'
        >>> error.attempts
        3
    """

    def __init__(
        self,
        message: str,
        parameter_name: str | None = None,
        attempts: int = 0,
    ) -> None:
        """Initialize the optimization error.

        Args:
            message: A human-readable description of the optimization error.
            parameter_name: Optional name of the parameter that failed to update.
            attempts: Number of attempts made before failing.
        """
        super().__init__(message)
        self.parameter_name = parameter_name
        self.attempts = attempts
