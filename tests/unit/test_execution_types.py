"""Unit tests for Task, TaskResult, and TaskStatus types."""

import time

from plait.execution.state import Task, TaskResult, TaskStatus
from plait.module import LLMInference

# ─────────────────────────────────────────────────────────────────────────────
# TaskStatus Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_task_status_pending_exists(self) -> None:
        """PENDING status exists for tasks ready to execute."""
        status = TaskStatus.PENDING
        assert status.name == "PENDING"

    def test_task_status_blocked_exists(self) -> None:
        """BLOCKED status exists for tasks waiting on dependencies."""
        status = TaskStatus.BLOCKED
        assert status.name == "BLOCKED"

    def test_task_status_in_progress_exists(self) -> None:
        """IN_PROGRESS status exists for executing tasks."""
        status = TaskStatus.IN_PROGRESS
        assert status.name == "IN_PROGRESS"

    def test_task_status_completed_exists(self) -> None:
        """COMPLETED status exists for successfully finished tasks."""
        status = TaskStatus.COMPLETED
        assert status.name == "COMPLETED"

    def test_task_status_failed_exists(self) -> None:
        """FAILED status exists for tasks that finished with errors."""
        status = TaskStatus.FAILED
        assert status.name == "FAILED"

    def test_task_status_cancelled_exists(self) -> None:
        """CANCELLED status exists for tasks dropped due to parent failure."""
        status = TaskStatus.CANCELLED
        assert status.name == "CANCELLED"

    def test_task_statuses_are_distinct(self) -> None:
        """All task statuses have distinct values."""
        statuses = [
            TaskStatus.PENDING,
            TaskStatus.BLOCKED,
            TaskStatus.IN_PROGRESS,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ]
        values = [s.value for s in statuses]
        assert len(values) == len(set(values)), "All status values should be unique"

    def test_task_status_equality(self) -> None:
        """Task statuses can be compared for equality."""
        assert TaskStatus.PENDING == TaskStatus.PENDING
        assert TaskStatus.PENDING != TaskStatus.COMPLETED

    def test_task_status_is_enum_member(self) -> None:
        """TaskStatus values are proper enum members."""
        assert isinstance(TaskStatus.PENDING, TaskStatus)
        assert isinstance(TaskStatus.COMPLETED, TaskStatus)


# ─────────────────────────────────────────────────────────────────────────────
# Task Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskCreation:
    """Tests for Task instantiation."""

    def test_task_creation_basic(self) -> None:
        """Task can be created with required fields."""
        module = LLMInference(alias="test")
        task = Task(
            node_id="LLMInference_1",
            module=module,
            args=("input text",),
            kwargs={},
            dependencies=["input:input_0"],
        )

        assert task.node_id == "LLMInference_1"
        assert task.module is module
        assert task.args == ("input text",)
        assert task.kwargs == {}
        assert task.dependencies == ["input:input_0"]

    def test_task_creation_with_all_fields(self) -> None:
        """Task can be created with all optional fields."""
        module = LLMInference(alias="test")
        created_at = time.time()

        task = Task(
            node_id="LLMInference_1",
            module=module,
            args=("input",),
            kwargs={"context": "extra"},
            dependencies=["dep_1", "dep_2"],
            priority=5,
            retry_count=2,
            created_at=created_at,
        )

        assert task.priority == 5
        assert task.retry_count == 2
        assert task.created_at == created_at

    def test_task_defaults(self) -> None:
        """Task has correct default values."""
        module = LLMInference(alias="test")
        before = time.time()

        task = Task(
            node_id="test",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
        )

        after = time.time()

        assert task.priority == 0
        assert task.retry_count == 0
        assert before <= task.created_at <= after

    def test_task_with_empty_dependencies(self) -> None:
        """Task can have no dependencies (input node)."""
        module = LLMInference(alias="test")
        task = Task(
            node_id="input:input_0",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
        )

        assert task.dependencies == []

    def test_task_with_multiple_dependencies(self) -> None:
        """Task can depend on multiple other nodes."""
        module = LLMInference(alias="test")
        task = Task(
            node_id="merger_1",
            module=module,
            args=(),
            kwargs={},
            dependencies=["branch_a", "branch_b", "branch_c"],
        )

        assert len(task.dependencies) == 3
        assert "branch_a" in task.dependencies
        assert "branch_b" in task.dependencies
        assert "branch_c" in task.dependencies

    def test_task_with_kwargs(self) -> None:
        """Task can have keyword arguments."""
        module = LLMInference(alias="test")
        task = Task(
            node_id="test",
            module=module,
            args=(),
            kwargs={"context": "value", "temperature": 0.5},
            dependencies=[],
        )

        assert task.kwargs == {"context": "value", "temperature": 0.5}


class TestTaskComparison:
    """Tests for Task comparison and ordering."""

    def test_task_lt_by_priority(self) -> None:
        """Tasks with lower priority values come first."""
        module = LLMInference(alias="test")

        high_priority = Task(
            node_id="high",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
            priority=0,
        )
        low_priority = Task(
            node_id="low",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
            priority=10,
        )

        assert high_priority < low_priority
        assert not low_priority < high_priority

    def test_task_lt_by_created_at_when_same_priority(self) -> None:
        """Tasks with same priority are ordered by creation time."""
        module = LLMInference(alias="test")

        earlier = Task(
            node_id="earlier",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
            priority=5,
            created_at=1000.0,
        )
        later = Task(
            node_id="later",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
            priority=5,
            created_at=2000.0,
        )

        assert earlier < later
        assert not later < earlier

    def test_task_lt_priority_takes_precedence(self) -> None:
        """Priority takes precedence over creation time."""
        module = LLMInference(alias="test")

        # Created later but higher priority
        high_priority_late = Task(
            node_id="high_late",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
            priority=0,
            created_at=2000.0,
        )
        # Created earlier but lower priority
        low_priority_early = Task(
            node_id="low_early",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
            priority=10,
            created_at=1000.0,
        )

        assert high_priority_late < low_priority_early

    def test_task_equality_by_node_id(self) -> None:
        """Tasks are equal if they have the same node_id."""
        module = LLMInference(alias="test")

        task1 = Task(
            node_id="same_id",
            module=module,
            args=("a",),
            kwargs={},
            dependencies=[],
        )
        task2 = Task(
            node_id="same_id",
            module=module,
            args=("b",),  # Different args
            kwargs={},
            dependencies=[],
        )

        assert task1 == task2

    def test_task_inequality_by_node_id(self) -> None:
        """Tasks are not equal if they have different node_ids."""
        module = LLMInference(alias="test")

        task1 = Task(
            node_id="id_1",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
        )
        task2 = Task(
            node_id="id_2",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
        )

        assert task1 != task2

    def test_task_hash_by_node_id(self) -> None:
        """Tasks can be hashed for use in sets/dicts."""
        module = LLMInference(alias="test")

        task1 = Task(
            node_id="same_id",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
        )
        task2 = Task(
            node_id="same_id",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
        )
        task3 = Task(
            node_id="different_id",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
        )

        assert hash(task1) == hash(task2)
        # Tasks can be used in sets
        task_set = {task1, task3}
        assert len(task_set) == 2

    def test_task_not_equal_to_non_task(self) -> None:
        """Task equality returns NotImplemented for non-Task objects."""
        module = LLMInference(alias="test")
        task = Task(
            node_id="test",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
        )

        assert task != "not a task"
        assert task != 123
        none_value = None
        assert task != none_value


# ─────────────────────────────────────────────────────────────────────────────
# TaskResult Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskResultCreation:
    """Tests for TaskResult instantiation."""

    def test_task_result_creation_basic(self) -> None:
        """TaskResult can be created with required fields."""
        result = TaskResult(
            node_id="LLMInference_1",
            value="output text",
            duration_ms=150.5,
        )

        assert result.node_id == "LLMInference_1"
        assert result.value == "output text"
        assert result.duration_ms == 150.5

    def test_task_result_creation_with_retry_count(self) -> None:
        """TaskResult can include retry count."""
        result = TaskResult(
            node_id="LLMInference_1",
            value="output",
            duration_ms=100.0,
            retry_count=3,
        )

        assert result.retry_count == 3

    def test_task_result_default_retry_count(self) -> None:
        """TaskResult defaults to 0 retry count."""
        result = TaskResult(
            node_id="test",
            value="value",
            duration_ms=50.0,
        )

        assert result.retry_count == 0

    def test_task_result_with_complex_value(self) -> None:
        """TaskResult can hold complex values like dicts or lists."""
        complex_value = {"analysis": "text", "scores": [0.8, 0.9, 0.7]}

        result = TaskResult(
            node_id="test",
            value=complex_value,
            duration_ms=200.0,
        )

        assert result.value == complex_value
        assert result.value["analysis"] == "text"

    def test_task_result_with_none_value(self) -> None:
        """TaskResult can hold None as a value."""
        result = TaskResult(
            node_id="test",
            value=None,
            duration_ms=10.0,
        )

        assert result.value is None

    def test_task_result_duration_precision(self) -> None:
        """TaskResult maintains float precision for duration."""
        result = TaskResult(
            node_id="test",
            value="output",
            duration_ms=123.456789,
        )

        assert result.duration_ms == 123.456789


class TestTaskResultComparison:
    """Tests for TaskResult comparison."""

    def test_task_result_equality(self) -> None:
        """TaskResults are equal if node_id and value match."""
        result1 = TaskResult(
            node_id="test",
            value="output",
            duration_ms=100.0,
            retry_count=0,
        )
        result2 = TaskResult(
            node_id="test",
            value="output",
            duration_ms=200.0,  # Different duration
            retry_count=1,  # Different retry count
        )

        assert result1 == result2

    def test_task_result_inequality_by_node_id(self) -> None:
        """TaskResults with different node_ids are not equal."""
        result1 = TaskResult(
            node_id="test_1",
            value="output",
            duration_ms=100.0,
        )
        result2 = TaskResult(
            node_id="test_2",
            value="output",
            duration_ms=100.0,
        )

        assert result1 != result2

    def test_task_result_inequality_by_value(self) -> None:
        """TaskResults with different values are not equal."""
        result1 = TaskResult(
            node_id="test",
            value="output_a",
            duration_ms=100.0,
        )
        result2 = TaskResult(
            node_id="test",
            value="output_b",
            duration_ms=100.0,
        )

        assert result1 != result2

    def test_task_result_not_equal_to_non_result(self) -> None:
        """TaskResult equality returns NotImplemented for non-TaskResult."""
        result = TaskResult(
            node_id="test",
            value="output",
            duration_ms=100.0,
        )

        assert result != "not a result"
        assert result != 123
        none_value = None
        assert result != none_value
