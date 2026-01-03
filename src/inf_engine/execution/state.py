"""Task types and execution state for inf-engine.

This module provides the core data types for tracking task execution:
- TaskStatus: Enum representing the current state of a task
- Task: A single executable unit in the graph
- TaskResult: The result of a completed task
- ExecutionState: Tracks complete state of a graph execution

Example:
    >>> from inf_engine.execution.state import Task, TaskResult, TaskStatus
    >>>
    >>> # Create a task
    >>> task = Task(
    ...     node_id="LLMInference_1",
    ...     module=some_module,
    ...     args=("input text",),
    ...     kwargs={},
    ...     dependencies=["input:input_0"],
    ... )
    >>>
    >>> # Check initial state
    >>> task.retry_count
    0
    >>>
    >>> # Create a result after completion
    >>> result = TaskResult(
    ...     node_id="LLMInference_1",
    ...     value="output text",
    ...     duration_ms=150.5,
    ...     retry_count=0,
    ... )
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from inf_engine.graph import NodeRef
from inf_engine.values import ValueRef

if TYPE_CHECKING:
    from inf_engine.graph import InferenceGraph
    from inf_engine.module import InferenceModule
    from inf_engine.tracing.tracer import GetItemOp, InputNode, IterOp, MethodOp


class TaskStatus(Enum):
    """Status of a task in the execution graph.

    Tasks progress through these states during execution:
    - PENDING: Ready to execute, waiting for scheduler
    - BLOCKED: Waiting on dependencies to complete
    - IN_PROGRESS: Currently being executed
    - COMPLETED: Finished successfully with a result
    - FAILED: Finished with an error
    - CANCELLED: Dropped because a parent task failed

    Example:
        >>> status = TaskStatus.PENDING
        >>> status.name
        'PENDING'
        >>> status == TaskStatus.PENDING
        True

    Note:
        The state transitions are:
        BLOCKED -> PENDING (when dependencies complete)
        PENDING -> IN_PROGRESS (when scheduler picks up)
        IN_PROGRESS -> COMPLETED | FAILED | PENDING (on retry)
        BLOCKED -> CANCELLED (when parent fails)
    """

    PENDING = auto()
    BLOCKED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class Task:
    """A single executable unit in the execution graph.

    Represents a module invocation that can be scheduled and executed.
    Tasks are created from GraphNodes and contain all information needed
    to execute the module with its resolved arguments.

    Tasks support priority ordering for scheduling. Lower priority values
    indicate higher precedence (priority 0 runs before priority 1).
    When priorities are equal, earlier-created tasks run first.

    Attributes:
        node_id: Unique identifier matching the GraphNode ID.
        module: The operation to execute. Can be an InferenceModule,
            InputNode, or a data access operation (GetItemOp, IterOp,
            MethodOp). May be None for special cases.
        args: Positional arguments for module.forward().
        kwargs: Keyword arguments for module.forward().
        dependencies: List of node IDs this task depends on.
        priority: Execution priority (lower = higher precedence).
        retry_count: Number of times this task has been retried.
        created_at: Unix timestamp when the task was created.

    Example:
        >>> from inf_engine.module import LLMInference
        >>> module = LLMInference(alias="test")
        >>> task = Task(
        ...     node_id="LLMInference_1",
        ...     module=module,
        ...     args=("hello",),
        ...     kwargs={},
        ...     dependencies=["input:input_0"],
        ... )
        >>> task.node_id
        'LLMInference_1'
        >>> task.retry_count
        0

    Example with priority ordering:
        >>> task1 = Task("node_1", module, (), {}, [], priority=0)
        >>> task2 = Task("node_2", module, (), {}, [], priority=1)
        >>> task1 < task2  # task1 has higher precedence
        True
    """

    node_id: str
    module: InferenceModule | InputNode | GetItemOp | IterOp | MethodOp | None
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    dependencies: list[str]
    priority: int = 0
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other: Task) -> bool:
        """Compare tasks for priority queue ordering.

        Lower priority values indicate higher precedence. When priorities
        are equal, earlier-created tasks have higher precedence.

        Args:
            other: Another Task to compare against.

        Returns:
            True if this task should be executed before the other.

        Example:
            >>> task1 = Task("a", module, (), {}, [], priority=0)
            >>> task2 = Task("b", module, (), {}, [], priority=1)
            >>> task1 < task2
            True
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at

    def __eq__(self, other: object) -> bool:
        """Check equality based on node_id.

        Two tasks are equal if they have the same node_id.

        Args:
            other: Another object to compare against.

        Returns:
            True if other is a Task with the same node_id.
        """
        if not isinstance(other, Task):
            return NotImplemented
        return self.node_id == other.node_id

    def __hash__(self) -> int:
        """Hash based on node_id.

        Returns:
            Hash of the node_id.
        """
        return hash(self.node_id)


@dataclass
class TaskResult:
    """Result of a completed task execution.

    Contains the output value from the module's forward() method
    along with execution metadata for observability and debugging.

    Attributes:
        node_id: Unique identifier matching the Task/GraphNode ID.
        value: The return value from module.forward().
        duration_ms: Execution time in milliseconds.
        retry_count: Number of retries before success.

    Example:
        >>> result = TaskResult(
        ...     node_id="LLMInference_1",
        ...     value="Generated response text",
        ...     duration_ms=245.8,
        ...     retry_count=0,
        ... )
        >>> result.value
        'Generated response text'
        >>> result.duration_ms
        245.8

    Note:
        The duration_ms includes only the actual execution time,
        not time spent waiting in the queue or for dependencies.
    """

    node_id: str
    value: Any
    duration_ms: float
    retry_count: int = 0

    def __eq__(self, other: object) -> bool:
        """Check equality based on node_id and value.

        Args:
            other: Another object to compare against.

        Returns:
            True if other is a TaskResult with same node_id and value.
        """
        if not isinstance(other, TaskResult):
            return NotImplemented
        return self.node_id == other.node_id and self.value == other.value


class ExecutionState:
    """Tracks the complete state of a graph execution.

    Maintains task statuses, results, errors, and dependency relationships.
    Provides methods for tracking which tasks are ready to execute based
    on their dependencies.

    The execution state is initialized from an InferenceGraph and tracks:
    - Status of each node (BLOCKED, PENDING, IN_PROGRESS, COMPLETED, FAILED, CANCELLED)
    - Results of completed tasks
    - Errors from failed tasks
    - Pending tasks in a priority queue (ready to execute)
    - In-progress tasks currently being executed
    - Dependency relationships for scheduling

    Attributes:
        graph: The InferenceGraph being executed.
        status: Dictionary mapping node IDs to their current TaskStatus.
        results: Dictionary mapping node IDs to their TaskResult.
        errors: Dictionary mapping node IDs to their exceptions.
        pending: Priority queue of tasks ready to execute.
        in_progress: Dictionary of tasks currently being executed.
        waiting_on: Maps each node to the set of dependencies not yet done.
        dependents: Maps each node to the set of nodes waiting on it.
        task_ready_event: Event signaled when new tasks become ready.

    Example:
        >>> from inf_engine.execution.state import ExecutionState
        >>> from inf_engine.graph import InferenceGraph, GraphNode
        >>> from inf_engine.tracing.tracer import InputNode
        >>>
        >>> # Create a simple graph
        >>> input_node = GraphNode(
        ...     id="input:input_0",
        ...     module=InputNode("hello"),
        ...     args=(),
        ...     kwargs={},
        ...     dependencies=[],
        ... )
        >>> graph = InferenceGraph(
        ...     nodes={"input:input_0": input_node},
        ...     input_ids=["input:input_0"],
        ...     output_ids=["input:input_0"],
        ... )
        >>>
        >>> # Create execution state
        >>> state = ExecutionState(graph)
        >>> state.status["input:input_0"]
        <TaskStatus.PENDING: 1>

    Note:
        Nodes with no dependencies are automatically marked as PENDING
        during initialization and added to the pending queue.
    """

    def __init__(self, graph: InferenceGraph, record: bool = False) -> None:
        """Initialize execution state from an inference graph.

        Analyzes the graph to determine initial task states and sets up
        dependency tracking. Nodes with no dependencies are immediately
        marked as PENDING and added to the priority queue.

        Args:
            graph: The InferenceGraph to execute.
            record: If True, track additional data (node inputs, execution
                order) needed for building a ForwardRecord. Default False.

        Example:
            >>> state = ExecutionState(graph)
            >>> len(state.status)  # One entry per node
            3

            >>> # Enable recording for backward pass support
            >>> state = ExecutionState(graph, record=True)
            >>> state.is_recording
            True
        """
        self.graph = graph

        # Task status tracking
        self.status: dict[str, TaskStatus] = {}
        self.results: dict[str, TaskResult] = {}
        self.errors: dict[str, Exception] = {}

        # Task management queues
        self.pending: asyncio.PriorityQueue[Task] = asyncio.PriorityQueue()
        self.in_progress: dict[str, Task] = {}

        # Dependency tracking
        # waiting_on[node_id] = set of node_ids this node is waiting for
        self.waiting_on: dict[str, set[str]] = defaultdict(set)
        # dependents[node_id] = set of node_ids waiting for this node
        self.dependents: dict[str, set[str]] = defaultdict(set)

        # Event signaling for scheduler efficiency
        # Set when new tasks become ready for execution
        self.task_ready_event: asyncio.Event = asyncio.Event()

        # Recording mode for ForwardRecord support
        self._record: bool = record
        # node_id -> dict of argument name to resolved value
        self.recorded_inputs: dict[str, dict[str, Any]] = {}
        # Order in which nodes completed execution
        self.execution_order: list[str] = []

        # Initialize state from graph
        self._initialize()

    @property
    def is_recording(self) -> bool:
        """Check if this state is tracking data for ForwardRecord.

        Returns:
            True if recording mode is enabled, False otherwise.
        """
        return self._record

    def _initialize(self) -> None:
        """Set up initial state from the graph.

        Iterates through all nodes in the graph to:
        1. Set initial status to BLOCKED
        2. Build dependency tracking structures (waiting_on, dependents)
        3. Mark nodes with no dependencies as ready via _make_ready()

        This method is called automatically during __init__.

        Note:
            Only dependencies that are actual graph nodes are tracked for
            scheduling purposes. Non-node dependencies (e.g., parameter refs
            like "param:...") are static values and don't need to be awaited.

        Example:
            >>> # After initialization, input nodes are PENDING
            >>> state.status["input:input_0"]
            <TaskStatus.PENDING: 1>
            >>> # Nodes with dependencies are BLOCKED
            >>> state.status["LLMInference_1"]
            <TaskStatus.BLOCKED: 2>
        """
        for node_id, node in self.graph.nodes.items():
            # All nodes start as BLOCKED
            self.status[node_id] = TaskStatus.BLOCKED

            # Track dependencies - only for actual graph nodes
            # Non-node dependencies (e.g., param refs) are static values
            for dep_id in node.dependencies:
                if dep_id in self.graph.nodes:
                    self.waiting_on[node_id].add(dep_id)
                    self.dependents[dep_id].add(node_id)

            # Nodes with no node dependencies are immediately ready
            if not self.waiting_on[node_id]:
                self._make_ready(node_id)

    def _make_ready(self, node_id: str) -> None:
        """Move a task to the pending queue.

        Creates a Task from the GraphNode and adds it to the priority queue.
        Updates the node status from BLOCKED to PENDING. Signals the
        task_ready_event to wake up any waiting scheduler.

        Args:
            node_id: The ID of the node to make ready.

        Note:
            This method resolves any argument references to completed results
            using _resolve_args() and _resolve_kwargs(). For initial nodes,
            arguments are passed through unchanged since no results exist yet.

        Example:
            >>> state._make_ready("input:input_0")
            >>> state.status["input:input_0"]
            <TaskStatus.PENDING: 1>
            >>> state.pending.qsize()
            1
        """
        node = self.graph.nodes[node_id]
        self.status[node_id] = TaskStatus.PENDING

        task = Task(
            node_id=node_id,
            module=node.module,
            args=self._resolve_args(node.args),
            kwargs=self._resolve_kwargs(node.kwargs),
            dependencies=list(node.dependencies),
            priority=node.priority,
        )

        self.pending.put_nowait(task)

        # Signal that a new task is ready for execution
        self.task_ready_event.set()

    def _resolve_value(self, value: Any) -> Any:
        """Recursively resolve NodeRef and ValueRef references to actual values.

        Traverses nested structures (dict, list, tuple) and replaces all
        NodeRef/ValueRef instances with their corresponding result values.

        Args:
            value: A value that may be a NodeRef, ValueRef, or a nested
                structure containing them.

        Returns:
            The value with all NodeRef/ValueRef references resolved to
            their actual result values.

        Example:
            >>> # After node "input:0" completes with value "hello"
            >>> state._resolve_value(NodeRef("input:0"))
            "hello"
            >>> state._resolve_value({"x": [ValueRef("input:0")]})
            {"x": ["hello"]}
        """
        if isinstance(value, NodeRef):
            return self.results[value.node_id].value
        elif isinstance(value, ValueRef):
            return self.results[value.ref].value
        elif isinstance(value, dict):
            return {k: self._resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_value(item) for item in value]
        elif isinstance(value, tuple):
            return tuple(self._resolve_value(item) for item in value)
        else:
            return value

    def _resolve_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """Resolve NodeRef and ValueRef references in args to actual result values.

        Recursively resolves all NodeRef/ValueRef references in the argument
        tuple, including those nested within dicts, lists, or tuples.

        Args:
            args: Tuple of arguments, which may contain NodeRef or ValueRef
                references at any nesting level.

        Returns:
            Tuple with all NodeRef/ValueRef references replaced by their
            result values, preserving the nested structure.

        Example:
            >>> # After node "input:input_0" completes with value "hello"
            >>> state._resolve_args((NodeRef("input:input_0"), "literal"))
            ("hello", "literal")
            >>> state._resolve_args(({"nested": ValueRef("input:0")},))
            ({"nested": "hello"},)
        """
        return tuple(self._resolve_value(arg) for arg in args)

    def _resolve_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve NodeRef and ValueRef references in kwargs to actual result values.

        Recursively resolves all NodeRef/ValueRef references in the kwargs
        dict, including those nested within dicts, lists, or tuples.

        Args:
            kwargs: Dictionary of keyword arguments, which may contain
                NodeRef or ValueRef references at any nesting level.

        Returns:
            Dictionary with all NodeRef/ValueRef references replaced by their
            result values, preserving the nested structure.

        Example:
            >>> # After node "input:context" completes with value "world"
            >>> state._resolve_kwargs({"context": NodeRef("input:context"), "temp": 0.7})
            {"context": "world", "temp": 0.7}
            >>> state._resolve_kwargs({"data": {"nested": ValueRef("input:0")}})
            {"data": {"nested": "hello"}}
        """
        return {key: self._resolve_value(value) for key, value in kwargs.items()}

    def get_ready_count(self) -> int:
        """Get the number of tasks ready to execute.

        Returns:
            The number of tasks in the pending queue.

        Example:
            >>> state.get_ready_count()
            2
        """
        return self.pending.qsize()

    def get_blocked_count(self) -> int:
        """Get the number of tasks waiting on dependencies.

        Returns:
            The number of nodes with BLOCKED status.

        Example:
            >>> state.get_blocked_count()
            3
        """
        return sum(1 for s in self.status.values() if s == TaskStatus.BLOCKED)

    async def get_next_task(self) -> Task | None:
        """Get the next task to execute from the pending queue.

        Retrieves the highest-priority task from the pending queue and
        transitions it to IN_PROGRESS status. Returns None if no tasks
        are currently pending.

        Returns:
            The next Task to execute, or None if the pending queue is empty.

        Note:
            This method is async because asyncio.PriorityQueue.get() is
            a coroutine. The task is automatically moved to the in_progress
            dict when retrieved.

        Example:
            >>> task = await state.get_next_task()
            >>> if task:
            ...     print(f"Executing {task.node_id}")
            ...     # ... execute task ...
            ...     state.mark_complete(task.node_id, result)
        """
        if self.pending.empty():
            return None

        task = await self.pending.get()
        self.status[task.node_id] = TaskStatus.IN_PROGRESS
        self.in_progress[task.node_id] = task
        return task

    def mark_complete(self, node_id: str, result: TaskResult) -> list[str]:
        """Mark a task as complete and return newly-ready node IDs.

        Updates the task status to COMPLETED, stores the result, and removes
        the task from in_progress. Then checks all dependent nodes to see if
        any have become ready (all their dependencies are now complete).

        If recording mode is enabled, also records the task's resolved inputs
        and adds the node to the execution order.

        Args:
            node_id: The ID of the node that completed.
            result: The TaskResult containing the output value and metadata.

        Returns:
            List of node IDs that became ready as a result of this completion.
            These nodes have been added to the pending queue and their status
            changed from BLOCKED to PENDING.

        Note:
            This method automatically triggers _make_ready() for any dependent
            nodes whose dependencies are now all satisfied.

        Example:
            >>> result = TaskResult(
            ...     node_id="input:input_0",
            ...     value="hello",
            ...     duration_ms=10.5,
            ... )
            >>> newly_ready = state.mark_complete("input:input_0", result)
            >>> print(newly_ready)
            ['LLMInference_1']
        """
        # Record inputs and execution order if in recording mode
        if self._record:
            task = self.in_progress.get(node_id)
            if task is not None:
                # Build input dict from args and kwargs
                inputs: dict[str, Any] = {}
                # Add positional args by index
                for i, arg in enumerate(task.args):
                    inputs[f"arg_{i}"] = arg
                # Add keyword args
                inputs.update(task.kwargs)
                self.recorded_inputs[node_id] = inputs
            self.execution_order.append(node_id)

        self.status[node_id] = TaskStatus.COMPLETED
        self.results[node_id] = result
        self.in_progress.pop(node_id, None)

        # Find newly-ready dependents
        newly_ready: list[str] = []
        for dependent_id in self.dependents[node_id]:
            self.waiting_on[dependent_id].discard(node_id)

            # Check if all dependencies are now satisfied
            if not self.waiting_on[dependent_id]:
                if self.status[dependent_id] == TaskStatus.BLOCKED:
                    self._make_ready(dependent_id)
                    newly_ready.append(dependent_id)

        # Signal scheduler even if no new tasks are ready - it needs to
        # re-check is_complete() and may exit the execution loop
        self.task_ready_event.set()

        return newly_ready

    def mark_failed(self, node_id: str, error: Exception) -> list[str]:
        """Mark a task as failed and cancel all its descendants.

        Updates the task status to FAILED, stores the error, and removes
        the task from in_progress. Then cancels all descendant nodes since
        they can no longer execute (their dependency has failed).

        Args:
            node_id: The ID of the node that failed.
            error: The exception that caused the failure.

        Returns:
            List of node IDs that were cancelled as a result of this failure.
            These are all descendants of the failed node.

        Note:
            Cancelled nodes are transitioned to CANCELLED status regardless
            of their current status (BLOCKED, PENDING, etc.). This cascading
            cancellation ensures that no work is wasted on tasks that cannot
            produce useful results.

        Example:
            >>> # If node "LLMInference_1" fails in a linear graph:
            >>> # input -> LLMInference_1 -> LLMInference_2 -> LLMInference_3
            >>> cancelled = state.mark_failed("LLMInference_1", ValueError("API error"))
            >>> cancelled
            ['LLMInference_2', 'LLMInference_3']
            >>> state.status["LLMInference_1"]
            <TaskStatus.FAILED: 5>
            >>> state.errors["LLMInference_1"]
            ValueError('API error')
        """
        self.status[node_id] = TaskStatus.FAILED
        self.errors[node_id] = error
        self.in_progress.pop(node_id, None)

        # Cancel all descendants
        cancelled: list[str] = []
        descendants = self.graph.descendants(node_id)
        for desc_id in descendants:
            self.status[desc_id] = TaskStatus.CANCELLED
            cancelled.append(desc_id)

        # Signal scheduler to re-check state (may now be complete)
        self.task_ready_event.set()

        return cancelled

    def requeue(self, node_id: str) -> list[str]:
        """Re-enqueue a task and drop all its descendants.

        Used when a task hits rate limiting and needs to retry. The task is
        removed from in_progress, its retry_count is incremented, and it's
        placed back in the pending queue. All descendant nodes are reset to
        BLOCKED status with their dependencies restored.

        Args:
            node_id: The ID of the node to requeue.

        Returns:
            List of node IDs that were dropped (reset to BLOCKED).
            Returns empty list if the node was not in progress.

        Note:
            This method is designed for handling transient failures like
            rate limits. For permanent failures, use mark_failed() instead.

            Descendants are dropped rather than cancelled because the requeued
            task may eventually succeed, at which point they'll become ready
            again through the normal dependency resolution process.

        Example:
            >>> # When a task hits a rate limit
            >>> dropped = state.requeue("LLMInference_1")
            >>> print(dropped)
            ['LLMInference_2', 'LLMInference_3']
            >>> state.status["LLMInference_1"]
            <TaskStatus.PENDING: 1>
            >>> state.status["LLMInference_2"]
            <TaskStatus.BLOCKED: 2>
        """
        # Remove from in-progress
        task = self.in_progress.pop(node_id, None)
        if task is None:
            return []

        # Drop all descendants back to BLOCKED
        dropped: list[str] = []
        descendants = self.graph.descendants(node_id)
        for desc_id in descendants:
            self.status[desc_id] = TaskStatus.BLOCKED
            # Restore dependencies
            node = self.graph.nodes[desc_id]
            self.waiting_on[desc_id] = set(node.dependencies)
            dropped.append(desc_id)

        # Re-queue the task with incremented retry count
        task.retry_count += 1
        self.status[node_id] = TaskStatus.PENDING
        self.pending.put_nowait(task)

        # Signal scheduler that a task is ready
        self.task_ready_event.set()

        return dropped

    def is_complete(self) -> bool:
        """Check if all tasks are done (completed, failed, or cancelled).

        A graph execution is complete when no tasks are PENDING, BLOCKED,
        or IN_PROGRESS. This means all tasks have reached a terminal state
        (COMPLETED, FAILED, or CANCELLED).

        Returns:
            True if all tasks have finished, False if any are still active.

        Example:
            >>> while not state.is_complete():
            ...     task = await state.get_next_task()
            ...     if task:
            ...         # execute task
            ...         state.mark_complete(task.node_id, result)
            >>> print("Execution finished!")
        """
        for status in self.status.values():
            if status in (
                TaskStatus.PENDING,
                TaskStatus.BLOCKED,
                TaskStatus.IN_PROGRESS,
            ):
                return False
        return True

    def get_outputs(self) -> dict[str, Any]:
        """Get the final output values from completed output nodes.

        Retrieves the result values for all output nodes in the graph that
        have completed successfully. If the graph has output_structure set
        (from tracing a forward() that returns a dict), the original dict
        keys are preserved in the output.

        Returns:
            A dictionary mapping keys to result values. If output_structure
            is a dict, the original user-defined keys are used. Otherwise,
            node IDs are used as keys. Only includes outputs that have
            completed successfully (have a result in self.results).

        Note:
            This method should typically be called after is_complete()
            returns True to ensure all possible outputs have been collected.
            However, it can be called at any time to get partial results
            from outputs that have completed so far.

        Example:
            >>> # forward() returned {"summary": proxy1, "analysis": proxy2}
            >>> outputs = state.get_outputs()
            >>> outputs
            {'summary': 'Summary text', 'analysis': 'Analysis text'}

            >>> # forward() returned a single proxy
            >>> outputs = state.get_outputs()
            >>> outputs
            {'LLMInference_1': 'result text'}

            >>> # forward() returned [proxy1, proxy2]
            >>> outputs = state.get_outputs()
            >>> outputs
            {0: 'result A', 1: 'result B'}
        """
        return self._resolve_output_structure(self.graph.output_structure)

    def _resolve_output_structure(
        self, structure: str | dict[str, Any] | list[Any] | None
    ) -> dict[str, Any]:
        """Resolve output structure to actual values from results.

        Recursively traverses the output structure, replacing node IDs
        with actual result values.

        Args:
            structure: The output structure from InferenceGraph, containing
                node IDs in place of Proxy objects.

        Returns:
            A dictionary with resolved values. Dict structures preserve
            their keys. List structures use indices as keys.
        """
        if structure is None:
            # Fall back to node IDs as keys
            return {
                output_id: self.results[output_id].value
                for output_id in self.graph.output_ids
                if output_id in self.results
            }
        elif isinstance(structure, str):
            # Single output - use node_id as key
            if structure in self.results:
                return {structure: self.results[structure].value}
            return {}
        elif isinstance(structure, dict):
            # Dict output - preserve user keys
            result: dict[str, Any] = {}
            for key, value in structure.items():
                resolved = self._resolve_single_output(value)
                if resolved is not None:
                    result[key] = resolved
            return result
        elif isinstance(structure, list):
            # List output - use indices as keys
            result = {}
            for i, value in enumerate(structure):
                resolved = self._resolve_single_output(value)
                if resolved is not None:
                    result[i] = resolved
            return result
        else:
            return {}

    def _resolve_single_output(self, value: Any) -> Any:
        """Resolve a single output value from the structure.

        Args:
            value: Either a node_id string, or a nested structure.

        Returns:
            The resolved value, or None if not available.
        """
        if isinstance(value, str):
            # It's a node_id
            if value in self.results:
                return self.results[value].value
            return None
        elif isinstance(value, dict):
            # Nested dict
            result: dict[str, Any] = {}
            for k, v in value.items():
                resolved = self._resolve_single_output(v)
                if resolved is not None:
                    result[k] = resolved
            return result if result else None
        elif isinstance(value, list):
            # Nested list
            result_list: list[Any] = []
            for item in value:
                resolved = self._resolve_single_output(item)
                if resolved is not None:
                    result_list.append(resolved)
            return result_list if result_list else None
        else:
            return None
