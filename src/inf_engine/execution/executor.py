"""Executor for running inference modules.

This module provides the `run()` function which traces and executes an
InferenceModule, handling the complete flow from tracing to execution.

Example:
    >>> from inf_engine.execution.executor import run
    >>> from inf_engine.module import InferenceModule, LLMInference
    >>> from inf_engine.resources.config import ResourceConfig, EndpointConfig
    >>>
    >>> class Pipeline(InferenceModule):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.llm = LLMInference(alias="fast")
    ...
    ...     def forward(self, text: str) -> str:
    ...         return self.llm(text)
    >>>
    >>> # Configure resources for LLM execution
    >>> resources = ResourceConfig(endpoints={
    ...     "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
    ... })
    >>>
    >>> # Execute the pipeline with resources
    >>> result = await run(Pipeline(), "Hello, world!", resources=resources)
    >>>
    >>> # Execute with checkpointing for long-running pipelines
    >>> result = await run(
    ...     Pipeline(),
    ...     "Hello, world!",
    ...     resources=resources,
    ...     checkpoint_dir="/data/checkpoints",
    ...     execution_id="run_001",
    ... )
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inf_engine.execution.checkpoint import CheckpointManager
from inf_engine.execution.scheduler import Scheduler
from inf_engine.execution.state import ExecutionState
from inf_engine.tracing.tracer import Tracer

if TYPE_CHECKING:
    from inf_engine.execution.state import TaskResult
    from inf_engine.module import InferenceModule
    from inf_engine.resources.config import ResourceConfig
    from inf_engine.resources.manager import ResourceManager


async def run(
    module: InferenceModule,
    *args: Any,
    resources: ResourceConfig | ResourceManager | None = None,
    max_concurrent: int = 100,
    checkpoint_dir: Path | str | None = None,
    execution_id: str | None = None,
    **kwargs: Any,
) -> Any:
    """Trace and execute an inference module.

    This function traces the module's forward() method to capture the
    execution graph, then executes the graph asynchronously with
    proper dependency handling and concurrency control.

    Args:
        module: The inference module to execute.
        *args: Positional arguments to pass to forward().
        resources: Optional resource configuration or manager for LLM endpoints.
            When provided (as ResourceConfig or ResourceManager), LLMInference
            modules will be executed through the appropriate LLM clients.
            When None, LLMInference modules will raise an error during execution.
        max_concurrent: Maximum number of concurrent tasks during
            execution. Defaults to 100.
        checkpoint_dir: Optional directory for saving execution checkpoints.
            When provided, task completions are periodically written to disk
            for progress tracking and potential recovery.
        execution_id: Optional identifier for this execution run. Used as
            the checkpoint filename. If not provided and checkpoint_dir is
            set, a UUID will be generated.
        **kwargs: Keyword arguments to pass to forward().

    Returns:
        The output of the module's forward() method. If the module
        produces a single output, returns that value directly. If
        the module produces multiple outputs, returns a dictionary
        mapping output node IDs to their values.

    Raises:
        NotImplementedError: If the module's forward() method raises
            NotImplementedError.
        RuntimeError: If an LLMInference module is executed without
            a ResourceManager configured.
        Exception: Any exception raised during task execution is
            stored in the ExecutionState. If all output nodes fail
            or are cancelled, an empty dict is returned.

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

    Example with resources:
        >>> from inf_engine.resources.config import ResourceConfig, EndpointConfig
        >>>
        >>> resources = ResourceConfig(endpoints={
        ...     "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
        ... })
        >>> result = await run(pipeline, "input text", resources=resources)

    Example with max_concurrent:
        >>> result = await run(
        ...     pipeline,
        ...     "input text",
        ...     resources=resources,
        ...     max_concurrent=10,
        ... )

    Example with checkpointing:
        >>> from pathlib import Path
        >>>
        >>> # Enable checkpointing for long-running pipelines
        >>> result = await run(
        ...     pipeline,
        ...     "input text",
        ...     resources=resources,
        ...     checkpoint_dir=Path("/data/checkpoints"),
        ...     execution_id="batch_001",
        ... )
        >>> # Checkpoint saved to /data/checkpoints/batch_001.json
    """
    # Trace the module to build the execution graph
    tracer = Tracer()
    graph = tracer.trace(module, *args, **kwargs)

    # Create execution state from the graph
    state = ExecutionState(graph)

    # Create ResourceManager if resources are provided
    resource_manager = None
    if resources is not None:
        from inf_engine.resources.config import ResourceConfig
        from inf_engine.resources.manager import ResourceManager

        if isinstance(resources, ResourceConfig):
            resource_manager = ResourceManager(resources)
        else:
            # Already a ResourceManager
            resource_manager = resources

    # Set up checkpointing if requested
    checkpoint_manager: CheckpointManager | None = None
    exec_id: str = execution_id or str(uuid.uuid4())

    if checkpoint_dir is not None:
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        # Store graph hash for checkpoint compatibility checking
        checkpoint_manager.set_graph_hash(exec_id, graph.compute_hash())

    # Create on_complete callback for checkpointing
    def on_complete(node_id: str, result: TaskResult) -> None:
        if checkpoint_manager is not None:
            should_flush = checkpoint_manager.record_completion(
                exec_id, node_id, result
            )
            if should_flush:
                # Schedule flush as a task (non-blocking)
                import asyncio

                asyncio.create_task(checkpoint_manager.flush(exec_id))

    # Create scheduler and execute
    scheduler = Scheduler(
        resource_manager=resource_manager,
        max_concurrent=max_concurrent,
    )
    outputs = await scheduler.execute(
        state, on_complete=on_complete if checkpoint_manager else None
    )

    # Flush any remaining checkpoints
    if checkpoint_manager is not None:
        await checkpoint_manager.flush_all()

    # Return outputs (unwrap if single output)
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    return outputs
