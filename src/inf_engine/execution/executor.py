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
    >>>
    >>> # Execute with recording for backward pass support
    >>> output, record = await run(Pipeline(), "Hello!", resources=resources, record=True)
    >>> # record is a ForwardRecord containing graph and execution data
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from inf_engine.execution.checkpoint import CheckpointManager
from inf_engine.execution.scheduler import Scheduler
from inf_engine.execution.state import ExecutionState
from inf_engine.optimization.record import ForwardRecord
from inf_engine.tracing.tracer import Tracer

if TYPE_CHECKING:
    from inf_engine.execution.state import TaskResult
    from inf_engine.module import InferenceModule
    from inf_engine.resources.config import ResourceConfig
    from inf_engine.resources.manager import ResourceManager


from typing import Literal


@overload
async def run(
    module: InferenceModule,
    *args: Any,
    resources: ResourceConfig | ResourceManager | None = None,
    max_concurrent: int = 100,
    checkpoint_dir: Path | str | None = None,
    execution_id: str | None = None,
    record: Literal[False] = False,
    **kwargs: Any,
) -> Any: ...


@overload
async def run(
    module: InferenceModule,
    *args: Any,
    resources: ResourceConfig | ResourceManager | None = None,
    max_concurrent: int = 100,
    checkpoint_dir: Path | str | None = None,
    execution_id: str | None = None,
    record: Literal[True] = ...,
    **kwargs: Any,
) -> tuple[Any, ForwardRecord]: ...


async def run(
    module: InferenceModule,
    *args: Any,
    resources: ResourceConfig | ResourceManager | None = None,
    max_concurrent: int = 100,
    checkpoint_dir: Path | str | None = None,
    execution_id: str | None = None,
    record: bool = False,
    **kwargs: Any,
) -> Any | tuple[Any, ForwardRecord]:
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
        record: If True, return a ForwardRecord along with the output for
            backward pass support. The ForwardRecord contains the execution
            graph, node inputs/outputs, and module references needed for
            optimization. Defaults to False.
        **kwargs: Keyword arguments to pass to forward().

    Returns:
        If record=False (default): The output of the module's forward()
            method. If the module produces a single output, returns that
            value directly. If multiple outputs, returns a dictionary
            mapping output node IDs to their values.
        If record=True: A tuple of (output, ForwardRecord) where output
            is as above and ForwardRecord contains execution data for
            backward propagation.

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

    Example with recording for backward pass:
        >>> # Execute with recording enabled for optimization
        >>> output, record = await run(pipeline, "input", resources=resources, record=True)
        >>> # record contains graph, inputs, outputs for backward()
        >>> feedback = await loss_fn(output, target, record=record)
        >>> await feedback.backward()
    """
    from inf_engine.execution.context import get_execution_settings

    # Trace the module to build the execution graph
    tracer = Tracer()
    graph = tracer.trace(module, *args, **kwargs)

    # Create execution state from the graph
    # Enable recording mode if we need to return a ForwardRecord
    state = ExecutionState(graph, record=record)

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

    # Get profiler from ExecutionSettings context if available
    profiler = None
    settings = get_execution_settings()
    if settings is not None:
        profiler = settings.get_profiler()

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
        profiler=profiler,
    )
    outputs = await scheduler.execute(
        state, on_complete=on_complete if checkpoint_manager else None
    )

    # Flush any remaining checkpoints
    if checkpoint_manager is not None:
        await checkpoint_manager.flush_all()

    # Unwrap outputs if single output
    if len(outputs) == 1:
        output = next(iter(outputs.values()))
    else:
        output = outputs

    # Build and return ForwardRecord if recording is enabled
    if record:
        forward_record = _build_forward_record(graph, state)
        return output, forward_record

    return output


def _build_forward_record(graph: Any, state: ExecutionState) -> ForwardRecord:
    """Build a ForwardRecord from execution state.

    Constructs a ForwardRecord containing all data needed for backward
    propagation: the graph structure, node inputs/outputs, module references,
    execution order, and timing information.

    Args:
        graph: The InferenceGraph that was executed.
        state: The ExecutionState with recorded execution data.

    Returns:
        A ForwardRecord containing execution data for backward pass.
    """
    from inf_engine.module import InferenceModule

    # Extract node outputs from state results
    node_outputs: dict[str, Any] = {
        node_id: result.value for node_id, result in state.results.items()
    }

    # Extract timing (convert ms to seconds)
    timing: dict[str, float] = {
        node_id: result.duration_ms / 1000 for node_id, result in state.results.items()
    }

    # Build module map from graph nodes
    module_map: dict[str, InferenceModule] = {}
    for node_id, node in graph.nodes.items():
        if isinstance(node.module, InferenceModule):
            module_map[node_id] = node.module

    return ForwardRecord(
        graph=graph,
        node_inputs=state.recorded_inputs,
        node_outputs=node_outputs,
        module_map=module_map,
        execution_order=state.execution_order,
        timing=timing,
    )
