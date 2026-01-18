"""Thread-safe trace context management using ContextVar.

This module provides the mechanism for tracking the active tracer during
graph capture. It uses Python's contextvars for proper async/threading support.

Example:
    >>> from plait.tracing.context import get_trace_context, trace_context
    >>> from plait.tracing.tracer import Tracer
    >>>
    >>> # Outside trace context
    >>> get_trace_context()  # Returns None
    >>>
    >>> # Inside trace context
    >>> tracer = Tracer()
    >>> with trace_context(tracer):
    ...     ctx = get_trace_context()
    ...     assert ctx is tracer
    >>>
    >>> # After exiting, context is cleared
    >>> get_trace_context()  # Returns None
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plait.tracing.tracer import Tracer

# Thread-local/async-safe context variable for the active tracer.
# Default is None, indicating no active trace.
_trace_context: ContextVar[Tracer | None] = ContextVar("trace_context", default=None)


def get_trace_context() -> Tracer | None:
    """Get the current trace context, if any.

    Returns the active Tracer if called within a trace_context block,
    or None if no tracing is active.

    Returns:
        The active Tracer instance, or None if not currently tracing.

    Example:
        >>> tracer = get_trace_context()
        >>> if tracer is not None:
        ...     # Currently tracing - record the call
        ...     return tracer.record_call(module, args, kwargs)
        >>> else:
        ...     # Not tracing - execute normally
        ...     return module.forward(*args, **kwargs)
    """
    return _trace_context.get()


@contextmanager
def trace_context(tracer: Tracer) -> Generator[Tracer]:
    """Set the trace context for a block.

    This context manager establishes the given tracer as the active
    trace context. Module calls within this block will be recorded
    to the tracer instead of being executed directly.

    The context is properly cleaned up even if an exception occurs,
    and it's safe to use with async code due to ContextVar semantics.

    Args:
        tracer: The Tracer instance to set as the active context.

    Yields:
        The tracer that was set as the active context.

    Example:
        >>> tracer = Tracer()
        >>> with trace_context(tracer):
        ...     # All module calls here are recorded
        ...     output = module(input_value)
        >>> # tracer.nodes now contains the recorded call
    """
    token = _trace_context.set(tracer)
    try:
        yield tracer
    finally:
        _trace_context.reset(token)
