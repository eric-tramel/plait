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
