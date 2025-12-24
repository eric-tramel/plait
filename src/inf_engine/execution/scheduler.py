"""Scheduler for executing inference graphs with concurrency control.

This module provides the Scheduler class which manages task dispatch with
priority and resource awareness. The scheduler enforces concurrency limits
and coordinates task execution across the graph.

Example:
    >>> from inf_engine.execution.scheduler import Scheduler
    >>>
    >>> # Create scheduler with default concurrency
    >>> scheduler = Scheduler()
    >>> scheduler.max_concurrent
    100
    >>>
    >>> # Create scheduler with custom concurrency limit
    >>> scheduler = Scheduler(max_concurrent=10)
    >>> scheduler.max_concurrent
    10
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from inf_engine.tracing.tracer import InputNode

if TYPE_CHECKING:
    from inf_engine.execution.state import ExecutionState, Task, TaskResult


class Scheduler:
    """Manages task scheduling with concurrency control.

    The Scheduler coordinates the execution of tasks from an ExecutionState,
    enforcing concurrency limits via a semaphore to prevent resource exhaustion.
    It dispatches tasks as they become ready and tracks active task count.

    Attributes:
        max_concurrent: Maximum number of tasks that can execute concurrently.

    Example:
        >>> scheduler = Scheduler(max_concurrent=50)
        >>> scheduler.max_concurrent
        50
        >>>
        >>> # Acquire a slot before executing a task
        >>> async def run_task():
        ...     async with scheduler:
        ...         # Execute task here
        ...         pass

    Note:
        The scheduler uses an asyncio.Semaphore internally to enforce the
        concurrency limit. Tasks should acquire a slot before starting and
        release it when complete.
    """

    def __init__(self, max_concurrent: int = 100) -> None:
        """Initialize the scheduler with a concurrency limit.

        Creates an asyncio.Semaphore to enforce the maximum number of
        concurrent task executions.

        Args:
            max_concurrent: Maximum number of tasks that can execute
                simultaneously. Must be positive. Defaults to 100.

        Raises:
            ValueError: If max_concurrent is less than 1.

        Example:
            >>> scheduler = Scheduler()
            >>> scheduler.max_concurrent
            100
            >>>
            >>> scheduler = Scheduler(max_concurrent=20)
            >>> scheduler.max_concurrent
            20
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")

        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0

    @property
    def active_count(self) -> int:
        """Get the number of currently active tasks.

        Returns:
            The number of tasks currently executing (holding semaphore slots).

        Example:
            >>> scheduler = Scheduler(max_concurrent=10)
            >>> scheduler.active_count
            0
        """
        return self._active_count

    @property
    def available_slots(self) -> int:
        """Get the number of available execution slots.

        Returns:
            The number of additional tasks that can start immediately
            without waiting.

        Example:
            >>> scheduler = Scheduler(max_concurrent=10)
            >>> scheduler.available_slots
            10
        """
        return self.max_concurrent - self._active_count

    async def acquire(self) -> None:
        """Acquire an execution slot.

        Waits until a slot is available if the scheduler is at capacity.
        Must be paired with a call to release() when the task completes.

        Example:
            >>> async def execute_task():
            ...     await scheduler.acquire()
            ...     try:
            ...         # Execute task
            ...         pass
            ...     finally:
            ...         scheduler.release()
        """
        await self._semaphore.acquire()
        self._active_count += 1

    def release(self) -> None:
        """Release an execution slot.

        Should be called when a task completes to allow other tasks to
        execute. Must be paired with a prior call to acquire().

        Raises:
            ValueError: If release is called more times than acquire.

        Example:
            >>> # After task completion
            >>> scheduler.release()
        """
        if self._active_count <= 0:
            raise ValueError("Cannot release: no active tasks")
        self._active_count -= 1
        self._semaphore.release()

    async def __aenter__(self) -> Scheduler:
        """Async context manager entry - acquires a slot.

        Returns:
            The scheduler instance.

        Example:
            >>> async with scheduler:
            ...     # Execute task with acquired slot
            ...     pass
        """
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit - releases the slot.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        self.release()

    async def execute(
        self,
        state: ExecutionState,
        on_complete: Callable[[str, TaskResult], None] | None = None,
        on_error: Callable[[str, Exception], None] | None = None,
    ) -> dict[str, Any]:
        """Execute all tasks in the graph.

        Runs tasks from the ExecutionState concurrently, respecting dependencies
        and the concurrency limit. Tasks are executed in priority order as they
        become ready. The method returns when all tasks have completed (either
        successfully, failed, or been cancelled).

        Args:
            state: The ExecutionState tracking task statuses and dependencies.
            on_complete: Optional callback invoked when a task completes successfully.
                Receives the node_id and TaskResult.
            on_error: Optional callback invoked when a task fails.
                Receives the node_id and the exception.

        Returns:
            Dictionary mapping output node IDs to their result values.

        Note:
            This method uses asyncio.TaskGroup for structured concurrency.
            If any task raises an unhandled exception (other than task-level
            failures which are caught and recorded), the entire execution
            may be cancelled.

        Example:
            >>> from inf_engine.execution.scheduler import Scheduler
            >>> from inf_engine.execution.state import ExecutionState
            >>>
            >>> scheduler = Scheduler(max_concurrent=10)
            >>> state = ExecutionState(graph)
            >>> outputs = await scheduler.execute(state)
            >>> print(outputs)
            {'LLMInference_1': 'result text'}

        Example with callbacks:
            >>> def on_done(node_id, result):
            ...     print(f"{node_id} completed in {result.duration_ms}ms")
            >>>
            >>> def on_fail(node_id, error):
            ...     print(f"{node_id} failed: {error}")
            >>>
            >>> outputs = await scheduler.execute(
            ...     state,
            ...     on_complete=on_done,
            ...     on_error=on_fail,
            ... )
        """
        # Import here to avoid circular imports at module load time
        from inf_engine.execution.state import TaskResult

        async with asyncio.TaskGroup() as tg:
            while not state.is_complete():
                # Wait for a slot to be available
                await self.acquire()

                # Get next task from the pending queue
                task = await state.get_next_task()
                if task is None:
                    # No task available but not complete - release slot and wait
                    self.release()
                    # Small sleep to allow in-progress tasks to complete
                    await asyncio.sleep(0.001)
                    continue

                # Spawn task execution
                tg.create_task(
                    self._execute_task(state, task, TaskResult, on_complete, on_error)
                )

        return state.get_outputs()

    async def _execute_task(
        self,
        state: ExecutionState,
        task: Task,
        task_result_class: type[TaskResult],
        on_complete: Callable[[str, TaskResult], None] | None,
        on_error: Callable[[str, Exception], None] | None,
    ) -> None:
        """Execute a single task with error handling.

        Runs the task's module with its resolved arguments, records the result
        or error in the ExecutionState, and invokes any callbacks.

        Args:
            state: The ExecutionState to update with results.
            task: The Task to execute.
            task_result_class: The TaskResult class for creating results.
            on_complete: Optional callback for successful completion.
            on_error: Optional callback for failures.

        Note:
            This method handles both InputNode tasks (which just return their
            stored value) and regular module tasks (which call forward()).
            The semaphore slot is always released, even if the task fails.
        """
        start_time = time.time()

        try:
            # Handle input nodes specially - just return their stored value
            if isinstance(task.module, InputNode):
                result = task.module.value
            else:
                # Execute the module's forward method
                result = await self._direct_execute(task)

            # Calculate duration and create result
            duration_ms = (time.time() - start_time) * 1000
            task_result = task_result_class(
                node_id=task.node_id,
                value=result,
                duration_ms=duration_ms,
                retry_count=task.retry_count,
            )

            # Mark complete in state
            state.mark_complete(task.node_id, task_result)

            # Invoke callback if provided
            if on_complete:
                on_complete(task.node_id, task_result)

        except Exception as e:
            # Task failed - mark failed and invoke callback
            state.mark_failed(task.node_id, e)
            if on_error:
                on_error(task.node_id, e)

        finally:
            # Always release the semaphore slot
            self.release()

    async def _direct_execute(self, task: Task) -> Any:
        """Execute a module directly without resource management.

        Calls the module's forward() method with the task's arguments.
        Handles both sync and async forward methods.

        Args:
            task: The Task containing the module and arguments to execute.

        Returns:
            The result of calling module.forward(*args, **kwargs).

        Note:
            This method will be enhanced in later PRs to integrate with
            ResourceManager for LLM calls with proper rate limiting and
            endpoint management.
        """
        from inf_engine.module import InferenceModule

        if task.module is None:
            return None

        # At this point, module must be an InferenceModule (InputNode is handled
        # in _execute_task before calling this method)
        assert isinstance(task.module, InferenceModule)

        if asyncio.iscoroutinefunction(task.module.forward):
            return await task.module.forward(*task.args, **task.kwargs)
        else:
            return task.module.forward(*task.args, **task.kwargs)
