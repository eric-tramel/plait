"""Executor for running inference modules.

This module provides the `run()` function which traces and executes an
InferenceModule, handling the complete flow from tracing to execution.

Example:
    >>> from inf_engine.execution.executor import run
    >>> from inf_engine.module import InferenceModule, LLMInference
    >>>
    >>> class Pipeline(InferenceModule):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.llm = LLMInference(alias="fast")
    ...
    ...     def forward(self, text: str) -> str:
    ...         return self.llm(text)
    >>>
    >>> # Execute the pipeline
    >>> result = await run(Pipeline(), "Hello, world!")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inf_engine.execution.scheduler import Scheduler
from inf_engine.execution.state import ExecutionState
from inf_engine.tracing.tracer import Tracer

if TYPE_CHECKING:
    from inf_engine.module import InferenceModule


async def run(
    module: InferenceModule,
    *args: Any,
    max_concurrent: int = 100,
    **kwargs: Any,
) -> Any:
    """Trace and execute an inference module.

    This function traces the module's forward() method to capture the
    execution graph, then executes the graph asynchronously with
    proper dependency handling and concurrency control.

    Args:
        module: The inference module to execute.
        *args: Positional arguments to pass to forward().
        max_concurrent: Maximum number of concurrent tasks during
            execution. Defaults to 100.
        **kwargs: Keyword arguments to pass to forward().

    Returns:
        The output of the module's forward() method. If the module
        produces a single output, returns that value directly. If
        the module produces multiple outputs, returns a dictionary
        mapping output node IDs to their values.

    Raises:
        NotImplementedError: If the module's forward() method raises
            NotImplementedError.
        Exception: Any exception raised during task execution is
            stored in the ExecutionState. If all output nodes fail
            or are cancelled, an empty dict is returned.

    Note:
        This is the basic version of run() without resource management
        or checkpointing. These features will be added in later PRs.

    Example:
        >>> from inf_engine.module import InferenceModule
        >>>
        >>> class Echo(InferenceModule):
        ...     def forward(self, text: str) -> str:
        ...         return text.upper()
        >>>
        >>> result = await run(Echo(), "hello")
        >>> print(result)
        HELLO

    Example with max_concurrent:
        >>> result = await run(
        ...     pipeline,
        ...     "input text",
        ...     max_concurrent=10,
        ... )
    """
    # Trace the module to build the execution graph
    tracer = Tracer()
    graph = tracer.trace(module, *args, **kwargs)

    # Create execution state from the graph
    state = ExecutionState(graph)

    # Create scheduler and execute
    scheduler = Scheduler(max_concurrent=max_concurrent)
    outputs = await scheduler.execute(state)

    # Return outputs (unwrap if single output)
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    return outputs
