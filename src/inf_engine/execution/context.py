"""Execution settings context manager.

This module provides the ExecutionSettings class for shared execution
configuration across multiple module calls. It uses Python's contextvars
for thread-safe and async-safe context propagation.

Example:
    >>> from inf_engine.execution.context import ExecutionSettings
    >>> from inf_engine.resources.config import ResourceConfig
    >>>
    >>> # Simple usage with context manager
    >>> with ExecutionSettings(
    ...     resources=ResourceConfig(...),
    ...     checkpoint_dir="/data/checkpoints",
    ... ):
    ...     result1 = await pipeline1(input1)
    ...     result2 = await pipeline2(input2)
    >>>
    >>> # Nested contexts override outer settings
    >>> with ExecutionSettings(max_concurrent=100):
    ...     result1 = await pipeline(input1)  # Uses max_concurrent=100
    ...     with ExecutionSettings(max_concurrent=10):
    ...         result2 = await pipeline(input2)  # Uses max_concurrent=10
    ...     result3 = await pipeline(input3)  # Back to max_concurrent=100

Example with streaming:
    >>> # Stream results as they complete (out of order)
    >>> async with ExecutionSettings(resources=config, streaming=True):
    ...     async for result in pipeline(["doc1", "doc2", "doc3"]):
    ...         if result.ok:
    ...             print(f"Input {result.index}: {result.output}")
    ...         else:
    ...             print(f"Input {result.index} failed: {result.error}")

Example with progress tracking:
    >>> def on_progress(done: int, total: int) -> None:
    ...     print(f"Progress: {done}/{total}")
    >>>
    >>> with ExecutionSettings(resources=config, on_progress=on_progress):
    ...     results = pipeline.run_sync(documents)
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from inf_engine.execution.checkpoint import CheckpointManager
    from inf_engine.execution.scheduler import Scheduler
    from inf_engine.execution.state import TaskResult
    from inf_engine.resources.config import ResourceConfig
    from inf_engine.resources.manager import ResourceManager

# Context variable for the active execution settings.
# Default is None, indicating no active settings context.
_execution_settings: ContextVar[ExecutionSettings | None] = ContextVar(
    "execution_settings", default=None
)


def get_execution_settings() -> ExecutionSettings | None:
    """Get the current execution settings from context.

    Returns the active ExecutionSettings if called within an ExecutionSettings
    context, or None if no settings context is active.

    Returns:
        The active ExecutionSettings instance, or None if not in context.

    Example:
        >>> settings = get_execution_settings()
        >>> if settings is not None:
        ...     resources = settings.resources
        ...     checkpoint_dir = settings.checkpoint_dir
    """
    return _execution_settings.get()


@dataclass
class ExecutionSettings:
    """Context manager for shared execution configuration.

    Provides default settings for all module executions within the context.
    Bound module settings take precedence over context settings. Settings
    can be nested, with inner contexts overriding specific fields from
    outer contexts.

    Attributes:
        resources: Optional ResourceConfig or ResourceManager for LLM endpoints.
            Modules will use these resources for LLM execution.
        checkpoint_dir: Optional directory for saving execution checkpoints.
            When set, a CheckpointManager is created automatically.
        max_concurrent: Maximum number of concurrent tasks. Defaults to 100.
        scheduler: Optional custom Scheduler instance. When None, a new
            Scheduler is created for each execution.
        on_task_complete: Optional callback invoked when a task completes.
            Receives the node_id and TaskResult.
        on_task_failed: Optional callback invoked when a task fails.
            Receives the node_id and the exception.
        streaming: When True, batch calls return an async iterator yielding
            BatchResult objects as they complete. When False (default),
            batch calls return a list of all results.
        preserve_order: When True, streaming results are yielded in input
            order (may wait on slower items). When False (default), results
            yield as soon as they complete for maximum throughput.
            Only applies when streaming=True.
        on_progress: Optional callback for batch progress updates. Called
            with (completed_count, total_count) after each input completes.
            Works with both streaming and non-streaming batch execution.
        profile: Whether to enable profiling. Defaults to False.
            Note: Profiling is not yet implemented (see PR-063).
        profile_path: Path for saving profile traces. Defaults to auto-generated.
        profile_counters: Whether to include counter events. Defaults to True.
        profile_include_args: Whether to include args in trace. Defaults to True.

    Example:
        >>> with ExecutionSettings(
        ...     resources=resource_config,
        ...     checkpoint_dir="/data/checkpoints",
        ...     max_concurrent=50,
        ... ):
        ...     # All module executions in this block share these settings
        ...     result1 = await pipeline1(input1)
        ...     result2 = await pipeline2(input2)

    Example with callbacks:
        >>> def on_complete(node_id: str, result):
        ...     print(f"Completed: {node_id}")
        >>>
        >>> with ExecutionSettings(on_task_complete=on_complete):
        ...     result = await pipeline(input)

    Example with streaming:
        >>> async with ExecutionSettings(resources=config, streaming=True):
        ...     async for result in pipeline(["doc1", "doc2"]):
        ...         if result.ok:
        ...             await send_to_client(result.output)
        ...         else:
        ...             logger.error(f"Input {result.index} failed")

    Example with progress tracking:
        >>> def on_progress(done: int, total: int) -> None:
        ...     print(f"Progress: {done}/{total}")
        >>>
        >>> with ExecutionSettings(resources=config, on_progress=on_progress):
        ...     results = pipeline.run_sync(documents)

    Note:
        ExecutionSettings supports both sync and async context managers.
        Use `with` for synchronous code and `async with` for async code.
        The async version properly awaits checkpoint flushing on exit.
    """

    resources: ResourceConfig | ResourceManager | None = None
    checkpoint_dir: Path | str | None = None
    max_concurrent: int = 100
    scheduler: Scheduler | None = None
    on_task_complete: Callable[[str, TaskResult], None] | None = None
    on_task_failed: Callable[[str, Exception], None] | None = None

    # Streaming and progress configuration
    streaming: bool = False
    preserve_order: bool = False
    on_progress: Callable[[int, int], None] | None = None

    # Profiling configuration (reserved for PR-063)
    profile: bool = False
    profile_path: Path | str | None = None
    profile_counters: bool = True
    profile_include_args: bool = True

    # Internal state (not part of public API)
    _token: Token[ExecutionSettings | None] | None = field(
        default=None, repr=False, compare=False
    )
    _checkpoint_manager: CheckpointManager | None = field(
        default=None, repr=False, compare=False
    )
    _parent: ExecutionSettings | None = field(default=None, repr=False, compare=False)

    def _get_effective_value(self, field_name: str, default: Any = None) -> Any:
        """Get the effective value for a field, checking parent contexts.

        For nested contexts, if a field is None in this context but set
        in a parent context, returns the parent's value.

        Args:
            field_name: The name of the field to get.
            default: Default value if not set in any context.

        Returns:
            The effective value for the field.
        """
        value = getattr(self, field_name)
        if value is not None:
            return value
        if self._parent is not None:
            return self._parent._get_effective_value(field_name, default)
        return default

    def get_resources(self) -> ResourceConfig | ResourceManager | None:
        """Get the effective resources configuration.

        Checks this context and parent contexts for resources.

        Returns:
            The ResourceConfig or ResourceManager, or None if not set.
        """
        return self._get_effective_value("resources")

    def get_checkpoint_dir(self) -> Path | None:
        """Get the effective checkpoint directory.

        Checks this context and parent contexts for checkpoint_dir.

        Returns:
            The checkpoint directory as a Path, or None if not set.
        """
        value = self._get_effective_value("checkpoint_dir")
        if value is not None:
            return Path(value) if isinstance(value, str) else value
        return None

    def get_max_concurrent(self) -> int:
        """Get the effective max_concurrent setting.

        Returns:
            The max_concurrent value, defaulting to 100.
        """
        # max_concurrent is never None in the dataclass, so just return it
        return self.max_concurrent

    def get_scheduler(self) -> Scheduler | None:
        """Get the effective scheduler.

        Checks this context and parent contexts for a custom scheduler.

        Returns:
            The Scheduler instance, or None if not set.
        """
        return self._get_effective_value("scheduler")

    def get_checkpoint_manager(self) -> CheckpointManager | None:
        """Get the checkpoint manager for this context.

        The checkpoint manager is created automatically when entering
        a context with checkpoint_dir set. For nested contexts without
        their own checkpoint_dir, returns the parent's manager.

        Returns:
            The CheckpointManager instance, or None if no checkpointing.
        """
        if self._checkpoint_manager is not None:
            return self._checkpoint_manager
        if self._parent is not None:
            return self._parent.get_checkpoint_manager()
        return None

    def get_streaming(self) -> bool:
        """Get the effective streaming setting.

        Returns:
            True if streaming mode is enabled, False otherwise.
        """
        return self.streaming

    def get_preserve_order(self) -> bool:
        """Get the effective preserve_order setting.

        Returns:
            True if results should be yielded in input order, False otherwise.
        """
        return self.preserve_order

    def get_on_progress(self) -> Callable[[int, int], None] | None:
        """Get the effective on_progress callback.

        Checks this context and parent contexts for the callback.

        Returns:
            The on_progress callback, or None if not set.
        """
        return self._get_effective_value("on_progress")

    def _enter(self) -> Self:
        """Common entry logic for both sync and async context managers.

        Sets up the context variable, creates CheckpointManager if needed,
        and links to any parent context.

        Returns:
            This ExecutionSettings instance.
        """
        # Link to parent context if one exists
        self._parent = _execution_settings.get()

        # Activate this context
        self._token = _execution_settings.set(self)

        # Create CheckpointManager if checkpoint_dir is set
        if self.checkpoint_dir is not None:
            from inf_engine.execution.checkpoint import CheckpointManager

            checkpoint_path = (
                Path(self.checkpoint_dir)
                if isinstance(self.checkpoint_dir, str)
                else self.checkpoint_dir
            )
            self._checkpoint_manager = CheckpointManager(checkpoint_path)

        return self

    def _exit(self) -> None:
        """Common exit logic - resets context variable.

        Note: Checkpoint flushing is handled separately in __exit__ and
        __aexit__ due to sync/async differences.
        """
        if self._token is not None:
            _execution_settings.reset(self._token)
            self._token = None

        self._parent = None

    def __enter__(self) -> Self:
        """Enter the execution settings context (synchronous).

        Activates this settings context, making it available via
        get_execution_settings(). Creates a CheckpointManager if
        checkpoint_dir is set.

        Returns:
            This ExecutionSettings instance.

        Example:
            >>> with ExecutionSettings(max_concurrent=10) as settings:
            ...     assert get_execution_settings() is settings
        """
        return self._enter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the execution settings context (synchronous).

        Resets the context variable to the previous state. If a
        CheckpointManager was created, it is NOT flushed in sync mode
        (use async context manager for proper cleanup).

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.

        Note:
            For proper checkpoint flushing, use the async context manager.
            The sync version is primarily for simple configurations without
            active checkpointing.
        """
        # In sync mode, we can't properly flush checkpoints since it's async.
        # Clear the checkpoint manager reference - caller should use async
        # context manager if checkpointing is needed.
        self._checkpoint_manager = None
        self._exit()

    async def __aenter__(self) -> Self:
        """Enter the execution settings context (asynchronous).

        Activates this settings context, making it available via
        get_execution_settings(). Creates a CheckpointManager if
        checkpoint_dir is set.

        Returns:
            This ExecutionSettings instance.

        Example:
            >>> async with ExecutionSettings(checkpoint_dir="/ckpt") as settings:
            ...     result = await pipeline(input)
            >>> # Checkpoints are flushed on exit
        """
        return self._enter()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the execution settings context (asynchronous).

        Flushes any pending checkpoints and resets the context variable
        to the previous state.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        # Flush checkpoints before exiting
        if self._checkpoint_manager is not None:
            await self._checkpoint_manager.flush_all()
            self._checkpoint_manager = None

        self._exit()
