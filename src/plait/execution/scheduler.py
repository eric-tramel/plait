"""Scheduler for executing inference graphs with concurrency control.

This module provides the Scheduler class which manages task dispatch with
priority and resource awareness. The scheduler enforces concurrency limits
and coordinates task execution across the graph.

Example:
    >>> from plait.execution.scheduler import Scheduler
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
    >>>
    >>> # Create scheduler with ResourceManager for LLM execution
    >>> from plait.resources.manager import ResourceManager
    >>> from plait.resources.config import ResourceConfig, EndpointConfig
    >>> config = ResourceConfig(endpoints={
    ...     "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini")
    ... })
    >>> manager = ResourceManager(config)
    >>> scheduler = Scheduler(resource_manager=manager)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from plait.errors import RateLimitError, TransientError
from plait.tracing.tracer import InputNode
from plait.types import LLMRequest

if TYPE_CHECKING:
    from plait.clients.base import LLMClient
    from plait.execution.state import ExecutionState, Task, TaskResult
    from plait.module import LLMInference
    from plait.profiling import TraceProfiler


class RateLimiterProtocol(Protocol):
    """Protocol defining the interface for rate limiters.

    Allows the Scheduler to interact with rate limiters without
    depending on the concrete RateLimiter implementation.
    """

    def backoff(self, retry_after: float | None = None) -> None:
        """Reduce rate after hitting API backpressure."""
        ...


class ResourceManagerProtocol(Protocol):
    """Protocol defining the interface required for resource management.

    Any object implementing get_client(), get_semaphore(), and get_rate_limiter()
    can be used as a resource manager, enabling easier testing with mock objects.
    """

    def get_client(self, alias: str) -> LLMClient:
        """Get the LLM client for an endpoint alias."""
        ...

    def get_semaphore(self, alias: str) -> asyncio.Semaphore | None:
        """Get the concurrency semaphore for an endpoint alias."""
        ...

    def get_rate_limiter(self, alias: str) -> RateLimiterProtocol | None:
        """Get the rate limiter for an endpoint alias."""
        ...


# Timeout for waiting on task_ready_event to prevent indefinite blocking
# if there's a logic error. In normal operation, the event is always
# signaled when state changes.
_EVENT_WAIT_TIMEOUT: float = 5.0


class Scheduler:
    """Manages task scheduling with concurrency control and LLM execution.

    The Scheduler coordinates the execution of tasks from an ExecutionState,
    enforcing concurrency limits via a semaphore to prevent resource exhaustion.
    It dispatches tasks as they become ready and tracks active task count.

    When a ResourceManager is provided, LLMInference modules are executed through
    the manager's clients with proper per-endpoint concurrency control. Non-LLM
    modules are executed directly via their forward() methods.

    Attributes:
        resource_manager: Optional resource manager for LLM execution.
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

    resource_manager: ResourceManagerProtocol | None
    max_concurrent: int
    task_timeout: float | None
    max_task_retries: int
    task_retry_delay: float
    profiler: TraceProfiler | None

    def __init__(
        self,
        resource_manager: ResourceManagerProtocol | None = None,
        max_concurrent: int = 100,
        task_timeout: float | None = None,
        max_task_retries: int = 0,
        task_retry_delay: float = 1.0,
        profiler: TraceProfiler | None = None,
    ) -> None:
        """Initialize the scheduler with optional resource manager and concurrency limit.

        Creates an asyncio.Semaphore to enforce the maximum number of
        concurrent task executions. If a ResourceManager is provided, LLM
        modules will be executed through it with proper endpoint management.

        Args:
            resource_manager: Optional resource manager for executing LLM modules.
                Must implement ResourceManagerProtocol (get_client, get_semaphore).
                When provided, LLMInference modules are executed through the
                manager's clients with per-endpoint concurrency control.
                When None, LLM modules will raise an error during execution.
            max_concurrent: Maximum number of tasks that can execute
                simultaneously. Must be positive. Defaults to 100.
            task_timeout: Maximum seconds for a single task before timeout.
                When set, tasks exceeding this duration raise TimeoutError.
                None means no timeout (default).
            max_task_retries: Maximum retry attempts for TransientError.
                Default 0 means no retries. Does not apply to rate limits
                (which use requeue) or permanent errors.
            task_retry_delay: Base delay in seconds between retry attempts.
                Uses exponential backoff: delay doubles each retry.
                Defaults to 1.0 second.
            profiler: Optional TraceProfiler for capturing execution traces.
                When provided, task start/end/fail events are recorded for
                visualization in Perfetto or Chrome DevTools.

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
            >>>
            >>> scheduler = Scheduler(task_timeout=60.0, max_task_retries=3)
            >>> scheduler.task_timeout
            60.0
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")

        self.resource_manager = resource_manager
        self.max_concurrent = max_concurrent
        self.task_timeout = task_timeout
        self.max_task_retries = max_task_retries
        self.task_retry_delay = task_retry_delay
        self.profiler = profiler
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
            >>> from plait.execution.scheduler import Scheduler
            >>> from plait.execution.state import ExecutionState
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
        from plait.execution.state import TaskResult

        async with asyncio.TaskGroup() as tg:
            while not state.is_complete():
                # Wait for a slot to be available
                await self.acquire()

                # Get next task from the pending queue
                task = await state.get_next_task()
                if task is None:
                    # No task available but not complete - release slot and wait
                    # for task_ready_event to be signaled (when a task completes
                    # or fails, the event is set to wake up waiting schedulers)
                    self.release()
                    state.task_ready_event.clear()
                    # Wait for event with timeout to prevent indefinite blocking
                    try:
                        await asyncio.wait_for(
                            state.task_ready_event.wait(),
                            timeout=_EVENT_WAIT_TIMEOUT,
                        )
                    except TimeoutError:
                        # Timeout is a safety mechanism - just retry the loop
                        # In normal operation this should not happen
                        pass
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

        Handles timeouts, transient error retries, and rate limits:
        - TimeoutError: Task is marked as failed
        - RateLimitError: Task is requeued with rate limiter backoff
        - TransientError: Task is retried up to max_task_retries times
          with exponential backoff
        - Other exceptions: Task is marked as failed

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

        # Get endpoint alias for profiling
        alias = getattr(task.module, "alias", None)
        module_name = task.module.__class__.__name__

        # Generate unique profiler task ID (node_id is reused across batch items)
        # Use id(task) to ensure uniqueness across concurrent executions
        profiler_task_id = f"{task.node_id}_{id(task):x}"

        # Record task start for profiling
        if self.profiler and alias:
            task_args = None
            if task.args:
                # Truncate args for trace display
                task_args = {"input": str(task.args[0])[:100]}
            self.profiler.task_start(
                task_id=profiler_task_id,
                endpoint=alias,
                module_name=module_name,
                args=task_args,
            )

        try:
            # Execute with optional timeout
            if self.task_timeout is not None:
                async with asyncio.timeout(self.task_timeout):
                    result = await self._run_task_inner(task)
            else:
                result = await self._run_task_inner(task)

            # Calculate duration and create result
            duration_ms = (time.time() - start_time) * 1000
            task_result = task_result_class(
                node_id=task.node_id,
                value=result,
                duration_ms=duration_ms,
                retry_count=task.retry_count,
            )

            # Record task completion for profiling
            if self.profiler and alias:
                self.profiler.task_end(
                    task_id=profiler_task_id,
                    endpoint=alias,
                    duration_ms=duration_ms,
                )

            # Mark complete in state
            state.mark_complete(task.node_id, task_result)

            # Invoke callback if provided
            if on_complete:
                on_complete(task.node_id, task_result)

        except TimeoutError:
            # Task timed out - mark as failed with descriptive error
            timeout_error = TimeoutError(
                f"Task '{task.node_id}' timed out after {self.task_timeout}s"
            )

            # Record failure for profiling
            if self.profiler and alias:
                self.profiler.task_failed(profiler_task_id, alias, "timeout")

            state.mark_failed(task.node_id, timeout_error)
            if on_error:
                on_error(task.node_id, timeout_error)

        except RateLimitError as e:
            # Record rate limit for profiling
            if self.profiler and alias:
                self.profiler.rate_limit_hit(alias, e.retry_after)
                self.profiler.task_failed(profiler_task_id, alias, "rate_limited")

            # Rate limit hit - trigger backoff and requeue
            self._handle_rate_limit(task, e)
            state.requeue(task.node_id)

        except TransientError as e:
            # Transient error - retry with exponential backoff if retries remain
            if task.retry_count < self.max_task_retries:
                # Record retry for profiling (task will restart with new event)
                if self.profiler and alias:
                    self.profiler.task_failed(
                        profiler_task_id, alias, f"transient_retry_{task.retry_count}"
                    )

                # Calculate delay with exponential backoff
                delay = self.task_retry_delay * (2**task.retry_count)
                await asyncio.sleep(delay)
                state.requeue(task.node_id)
            else:
                # Max retries exhausted - mark as failed
                if self.profiler and alias:
                    self.profiler.task_failed(profiler_task_id, alias, str(e))

                state.mark_failed(task.node_id, e)
                if on_error:
                    on_error(task.node_id, e)

        except Exception as e:
            # Record failure for profiling
            if self.profiler and alias:
                self.profiler.task_failed(profiler_task_id, alias, str(e))

            # Task failed - mark failed and invoke callback
            state.mark_failed(task.node_id, e)
            if on_error:
                on_error(task.node_id, e)

        finally:
            # Always release the semaphore slot
            self.release()

    async def _run_task_inner(self, task: Task) -> Any:
        """Execute the actual task logic.

        Separated from _execute_task to allow timeout wrapping.
        Implements Value(ERROR) short-circuit: if any dependency is a Value(ERROR),
        the error is propagated without executing the task.

        Args:
            task: The Task to execute.

        Returns:
            The result of the task execution, or the first Value(ERROR) if
            any dependency contains an error.
        """
        from plait.module import LLMInference
        from plait.values import first_error_value, has_error_value

        # Handle input nodes specially - just return their stored value
        if isinstance(task.module, InputNode):
            return task.module.value

        # Short-circuit if any dependency is a Value(ERROR)
        # (functional ops propagate errors as values, not exceptions)
        if has_error_value(*task.args, **task.kwargs):
            error_value = first_error_value(*task.args, **task.kwargs)
            return error_value

        if isinstance(task.module, LLMInference):
            # Execute LLM modules through ResourceManager
            return await self._execute_llm(task.module, task.args, task.kwargs)
        else:
            # Execute non-LLM modules directly
            return await self._direct_execute(task)

    def _handle_rate_limit(self, task: Task, error: RateLimitError) -> None:
        """Handle a rate limit error by triggering backoff on the rate limiter.

        When an LLM request hits a rate limit, this method looks up the
        rate limiter for the endpoint and triggers backoff to reduce the
        request rate. The task will be requeued for retry.

        Args:
            task: The Task that hit the rate limit.
            error: The RateLimitError containing retry information.

        Note:
            If no rate limiter is configured for the endpoint, the backoff
            is silently skipped. The task will still be requeued for retry.
        """
        if self.resource_manager is None:
            return

        # Get the alias from the LLM module
        alias = getattr(task.module, "alias", None)
        if alias is None:
            return

        # Get the rate limiter and trigger backoff
        rate_limiter = self.resource_manager.get_rate_limiter(alias)
        if rate_limiter is not None:
            rate_limiter.backoff(retry_after=error.retry_after)

    async def _direct_execute(self, task: Task) -> Any:
        """Execute a non-LLM module directly.

        Calls the module's forward() method with the task's arguments.
        Handles both sync and async forward methods. Values are unwrapped
        to raw payloads before calling forward().

        Args:
            task: The Task containing the module and arguments to execute.

        Returns:
            The result of calling module.forward(*args, **kwargs).

        Note:
            LLMInference modules are handled by _execute_llm(), not this method.
            Value payloads are unwrapped so user-defined forward() methods
            receive raw Python values, not Value containers.
        """
        from plait.module import InferenceModule
        from plait.values import unwrap

        if task.module is None:
            return None

        # At this point, module must be an InferenceModule (InputNode is handled
        # in _execute_task before calling this method)
        assert isinstance(task.module, InferenceModule)

        # Unwrap Value payloads for user-defined forward() implementations
        args = unwrap(task.args)
        kwargs = unwrap(task.kwargs)

        if asyncio.iscoroutinefunction(task.module.forward):
            return await task.module.forward(*args, **kwargs)
        else:
            return task.module.forward(*args, **kwargs)

    async def _execute_llm(
        self,
        module: LLMInference,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        """Execute an LLM module through the ResourceManager.

        Retrieves the appropriate client and semaphore from the ResourceManager,
        builds an LLMRequest from the module's configuration and arguments,
        and executes the completion request with per-endpoint concurrency control.

        Args:
            module: The LLMInference module to execute.
            args: Positional arguments passed to the module (prompt is first).
            kwargs: Keyword arguments passed to the module.

        Returns:
            The content string from the LLM response.

        Raises:
            RuntimeError: If no ResourceManager was provided to the Scheduler.
            KeyError: If the module's alias is not found in the ResourceManager.

        Example:
            >>> # Assuming scheduler has a resource_manager with "fast" endpoint
            >>> module = LLMInference(alias="fast", temperature=0.7)
            >>> result = await scheduler._execute_llm(module, ("Hello",), {})
        """
        if self.resource_manager is None:
            raise RuntimeError(
                f"Cannot execute LLMInference module '{module.alias}': "
                "no ResourceManager provided to Scheduler. "
                "Pass a ResourceManager to Scheduler() to enable LLM execution."
            )

        alias = module.alias

        # Get client and optional semaphore from ResourceManager
        client = self.resource_manager.get_client(alias)
        semaphore = self.resource_manager.get_semaphore(alias)

        # Build the request from module config and args
        request = self._build_llm_request(module, args, kwargs)

        # Execute with per-endpoint concurrency control if semaphore exists
        if semaphore is not None:
            async with semaphore:
                response = await client.complete(request)
        else:
            response = await client.complete(request)

        return response.content

    def _build_llm_request(
        self,
        module: LLMInference,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> LLMRequest:
        """Build an LLMRequest from a module and its arguments.

        Extracts the prompt from args/kwargs and combines it with the module's
        configuration (system_prompt, temperature, max_tokens, etc.) to create
        a complete LLMRequest. Values are unwrapped to raw payloads.

        Args:
            module: The LLMInference module containing configuration.
            args: Positional arguments (prompt expected as first arg).
            kwargs: Keyword arguments (prompt may be passed as kwarg).

        Returns:
            An LLMRequest ready to be sent to an LLM client.

        Raises:
            ValueError: If no prompt is provided in args or kwargs.

        Example:
            >>> module = LLMInference(
            ...     alias="fast",
            ...     system_prompt="You are helpful.",
            ...     temperature=0.5,
            ... )
            >>> request = scheduler._build_llm_request(module, ("Hello",), {})
            >>> request.prompt
            'Hello'
            >>> request.system_prompt
            'You are helpful.'
        """
        from plait.values import unwrap

        # Get prompt from args or kwargs
        if args:
            prompt = args[0]
        elif "prompt" in kwargs:
            prompt = kwargs["prompt"]
        else:
            raise ValueError(
                "LLMInference requires a prompt argument. "
                "Pass it as the first positional argument or as prompt=..."
            )

        # Unwrap Value to raw payload for LLM request
        prompt = unwrap(prompt)

        # Get system prompt from module's Parameter if present
        system_prompt: str | None = None
        if module.system_prompt is not None:
            system_prompt = str(module.system_prompt)

        return LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=module.temperature,
            max_tokens=module.max_tokens,
            response_format=module.response_format,
        )
