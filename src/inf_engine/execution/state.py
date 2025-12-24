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

if TYPE_CHECKING:
    from inf_engine.graph import InferenceGraph
    from inf_engine.module import InferenceModule
    from inf_engine.tracing.tracer import InputNode


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
        module: The module to execute (InferenceModule, InputNode, or None).
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
    module: InferenceModule | InputNode | None
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

    def __init__(self, graph: InferenceGraph) -> None:
        """Initialize execution state from an inference graph.

        Analyzes the graph to determine initial task states and sets up
        dependency tracking. Nodes with no dependencies are immediately
        marked as PENDING and added to the priority queue.

        Args:
            graph: The InferenceGraph to execute.

        Example:
            >>> state = ExecutionState(graph)
            >>> len(state.status)  # One entry per node
            3
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

        # Initialize state from graph
        self._initialize()

    def _initialize(self) -> None:
        """Set up initial state from the graph.

        Iterates through all nodes in the graph to:
        1. Set initial status to BLOCKED
        2. Build dependency tracking structures (waiting_on, dependents)
        3. Mark nodes with no dependencies as ready via _make_ready()

        This method is called automatically during __init__.

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

            # Track dependencies
            for dep_id in node.dependencies:
                self.waiting_on[node_id].add(dep_id)
                self.dependents[dep_id].add(node_id)

            # Nodes with no dependencies are immediately ready
            if not node.dependencies:
                self._make_ready(node_id)

    def _make_ready(self, node_id: str) -> None:
        """Move a task to the pending queue.

        Creates a Task from the GraphNode and adds it to the priority queue.
        Updates the node status from BLOCKED to PENDING.

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

    def _resolve_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """Resolve node ID references in args to actual result values.

        For each argument that is a string matching a completed node ID,
        replaces it with the corresponding result value. Non-string arguments
        and strings that don't match result IDs are passed through unchanged.

        Args:
            args: Tuple of arguments, which may contain node ID references.

        Returns:
            Tuple with node ID references replaced by their result values.

        Example:
            >>> # After node "input:input_0" completes with value "hello"
            >>> state._resolve_args(("input:input_0", "literal"))
            ("hello", "literal")
        """
        resolved: list[Any] = []
        for arg in args:
            if isinstance(arg, str) and arg in self.results:
                resolved.append(self.results[arg].value)
            else:
                resolved.append(arg)
        return tuple(resolved)

    def _resolve_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve node ID references in kwargs to actual result values.

        For each value that is a string matching a completed node ID,
        replaces it with the corresponding result value. Non-string values
        and strings that don't match result IDs are passed through unchanged.

        Args:
            kwargs: Dictionary of keyword arguments, which may contain
                node ID references as values.

        Returns:
            Dictionary with node ID references replaced by their result values.

        Example:
            >>> # After node "input:context" completes with value "world"
            >>> state._resolve_kwargs({"context": "input:context", "temp": 0.7})
            {"context": "world", "temp": 0.7}
        """
        resolved: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, str) and value in self.results:
                resolved[key] = self.results[value].value
            else:
                resolved[key] = value
        return resolved

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

        return cancelled

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
