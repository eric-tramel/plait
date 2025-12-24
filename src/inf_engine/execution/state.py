"""Task types and execution state for inf-engine.

This module provides the core data types for tracking task execution:
- TaskStatus: Enum representing the current state of a task
- Task: A single executable unit in the graph
- TaskResult: The result of a completed task

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

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inf_engine.module import InferenceModule


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
        module: The InferenceModule to execute.
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
    module: InferenceModule
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
