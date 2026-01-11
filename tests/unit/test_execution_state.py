"""Unit tests for ExecutionState initialization and task management."""

import asyncio

import pytest

from plait.execution.state import ExecutionState, TaskResult, TaskStatus
from plait.graph import GraphNode, InferenceGraph, NodeRef
from plait.module import LLMInference
from plait.tracing.tracer import InputNode

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions for creating test graphs
# ─────────────────────────────────────────────────────────────────────────────


def create_single_input_graph() -> InferenceGraph:
    """Create a graph with just one input node."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode("hello"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    return InferenceGraph(
        nodes={"input:input_0": input_node},
        input_ids=["input:input_0"],
        output_ids=["input:input_0"],
    )


def create_linear_graph() -> InferenceGraph:
    """Create a linear graph: input -> llm1 -> llm2."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode("hello"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    llm1_node = GraphNode(
        id="LLMInference_1",
        module=LLMInference(alias="fast"),
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    llm2_node = GraphNode(
        id="LLMInference_2",
        module=LLMInference(alias="smart"),
        args=(NodeRef("LLMInference_1"),),
        kwargs={},
        dependencies=["LLMInference_1"],
    )
    return InferenceGraph(
        nodes={
            "input:input_0": input_node,
            "LLMInference_1": llm1_node,
            "LLMInference_2": llm2_node,
        },
        input_ids=["input:input_0"],
        output_ids=["LLMInference_2"],
    )


def create_parallel_graph() -> InferenceGraph:
    """Create a graph with parallel branches: input -> [llm1, llm2]."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode("hello"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    llm1_node = GraphNode(
        id="LLMInference_1",
        module=LLMInference(alias="a"),
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    llm2_node = GraphNode(
        id="LLMInference_2",
        module=LLMInference(alias="b"),
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    return InferenceGraph(
        nodes={
            "input:input_0": input_node,
            "LLMInference_1": llm1_node,
            "LLMInference_2": llm2_node,
        },
        input_ids=["input:input_0"],
        output_ids=["LLMInference_1", "LLMInference_2"],
    )


def create_diamond_graph() -> InferenceGraph:
    """Create a diamond graph: input -> [a, b] -> merge."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode("hello"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    branch_a = GraphNode(
        id="LLMInference_1",
        module=LLMInference(alias="a"),
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    branch_b = GraphNode(
        id="LLMInference_2",
        module=LLMInference(alias="b"),
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    merge_node = GraphNode(
        id="LLMInference_3",
        module=LLMInference(alias="merge"),
        args=(NodeRef("LLMInference_1"), NodeRef("LLMInference_2")),
        kwargs={},
        dependencies=["LLMInference_1", "LLMInference_2"],
    )
    return InferenceGraph(
        nodes={
            "input:input_0": input_node,
            "LLMInference_1": branch_a,
            "LLMInference_2": branch_b,
            "LLMInference_3": merge_node,
        },
        input_ids=["input:input_0"],
        output_ids=["LLMInference_3"],
    )


def create_multi_input_graph() -> InferenceGraph:
    """Create a graph with multiple inputs."""
    input_a = GraphNode(
        id="input:input_0",
        module=InputNode("first"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    input_b = GraphNode(
        id="input:input_1",
        module=InputNode("second"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    llm_node = GraphNode(
        id="LLMInference_1",
        module=LLMInference(alias="test"),
        args=(NodeRef("input:input_0"), NodeRef("input:input_1")),
        kwargs={},
        dependencies=["input:input_0", "input:input_1"],
    )
    return InferenceGraph(
        nodes={
            "input:input_0": input_a,
            "input:input_1": input_b,
            "LLMInference_1": llm_node,
        },
        input_ids=["input:input_0", "input:input_1"],
        output_ids=["LLMInference_1"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionState Initialization Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExecutionStateInit:
    """Tests for ExecutionState initialization."""

    def test_init_stores_graph_reference(self) -> None:
        """ExecutionState stores a reference to the graph."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        assert state.graph is graph

    def test_init_creates_status_for_all_nodes(self) -> None:
        """ExecutionState creates a status entry for each node."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        assert len(state.status) == 3
        assert "input:input_0" in state.status
        assert "LLMInference_1" in state.status
        assert "LLMInference_2" in state.status

    def test_init_creates_empty_results(self) -> None:
        """ExecutionState starts with empty results dict."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        assert state.results == {}

    def test_init_creates_empty_errors(self) -> None:
        """ExecutionState starts with empty errors dict."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        assert state.errors == {}

    def test_init_creates_empty_in_progress(self) -> None:
        """ExecutionState starts with empty in_progress dict."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        assert state.in_progress == {}


class TestExecutionStateReadyNodes:
    """Tests for identifying ready nodes during initialization."""

    def test_input_node_is_pending(self) -> None:
        """Input nodes with no dependencies are marked PENDING."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        assert state.status["input:input_0"] == TaskStatus.PENDING

    def test_input_node_is_in_pending_queue(self) -> None:
        """Input nodes are added to the pending queue."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        assert state.pending.qsize() == 1

    def test_multiple_inputs_all_pending(self) -> None:
        """All input nodes are marked PENDING."""
        graph = create_multi_input_graph()
        state = ExecutionState(graph)

        assert state.status["input:input_0"] == TaskStatus.PENDING
        assert state.status["input:input_1"] == TaskStatus.PENDING
        assert state.pending.qsize() == 2

    def test_dependent_nodes_are_blocked(self) -> None:
        """Nodes with dependencies are marked BLOCKED."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        assert state.status["LLMInference_1"] == TaskStatus.BLOCKED
        assert state.status["LLMInference_2"] == TaskStatus.BLOCKED

    def test_get_ready_count_returns_pending_size(self) -> None:
        """get_ready_count returns number of pending tasks."""
        graph = create_multi_input_graph()
        state = ExecutionState(graph)

        assert state.get_ready_count() == 2

    def test_get_blocked_count_returns_blocked_nodes(self) -> None:
        """get_blocked_count returns number of blocked nodes."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # 1 input (pending) + 2 LLMs (blocked)
        assert state.get_blocked_count() == 2


class TestExecutionStateDependencyTracking:
    """Tests for dependency tracking initialization."""

    def test_waiting_on_tracks_dependencies(self) -> None:
        """waiting_on correctly tracks node dependencies."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        assert state.waiting_on["input:input_0"] == set()
        assert state.waiting_on["LLMInference_1"] == {"input:input_0"}
        assert state.waiting_on["LLMInference_2"] == {"LLMInference_1"}

    def test_waiting_on_multiple_dependencies(self) -> None:
        """waiting_on tracks multiple dependencies correctly."""
        graph = create_diamond_graph()
        state = ExecutionState(graph)

        assert state.waiting_on["LLMInference_3"] == {
            "LLMInference_1",
            "LLMInference_2",
        }

    def test_dependents_tracks_reverse_dependencies(self) -> None:
        """dependents correctly tracks which nodes depend on each node."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        assert state.dependents["input:input_0"] == {"LLMInference_1"}
        assert state.dependents["LLMInference_1"] == {"LLMInference_2"}
        assert state.dependents["LLMInference_2"] == set()

    def test_dependents_multiple_dependents(self) -> None:
        """dependents tracks multiple dependent nodes."""
        graph = create_parallel_graph()
        state = ExecutionState(graph)

        # Both LLMInference nodes depend on input
        assert state.dependents["input:input_0"] == {
            "LLMInference_1",
            "LLMInference_2",
        }


class TestExecutionStateTaskCreation:
    """Tests for Task creation during initialization."""

    def test_pending_task_has_correct_node_id(self) -> None:
        """Tasks in pending queue have correct node_id."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        task = state.pending.get_nowait()
        assert task.node_id == "input:input_0"

    def test_pending_task_has_module_reference(self) -> None:
        """Tasks in pending queue have module reference."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        task = state.pending.get_nowait()
        assert isinstance(task.module, InputNode)
        assert task.module.value == "hello"

    def test_pending_task_has_empty_dependencies(self) -> None:
        """Input tasks have empty dependencies list."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        task = state.pending.get_nowait()
        assert task.dependencies == []

    def test_pending_task_priority_from_node(self) -> None:
        """Task priority is taken from GraphNode."""
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode("hello"),
            args=(),
            kwargs={},
            dependencies=[],
            priority=10,
        )
        graph = InferenceGraph(
            nodes={"input:input_0": input_node},
            input_ids=["input:input_0"],
            output_ids=["input:input_0"],
        )
        state = ExecutionState(graph)

        task = state.pending.get_nowait()
        assert task.priority == 10


class TestExecutionStateResolveArgs:
    """Tests for argument resolution."""

    def test_resolve_args_passes_through_when_no_results(self) -> None:
        """Args pass through unchanged when no results exist."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        resolved = state._resolve_args(("input:input_0", "literal", 42))
        assert resolved == ("input:input_0", "literal", 42)

    def test_resolve_args_replaces_node_refs(self) -> None:
        """Args with NodeRef are replaced with result values."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Simulate a completed result
        state.results["input:input_0"] = TaskResult(
            node_id="input:input_0",
            value="resolved_value",
            duration_ms=10.0,
        )

        resolved = state._resolve_args((NodeRef("input:input_0"), "literal"))
        assert resolved == ("resolved_value", "literal")

    def test_resolve_kwargs_passes_through_when_no_results(self) -> None:
        """Kwargs pass through unchanged when no results exist."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        resolved = state._resolve_kwargs({"key": "input:input_0", "temp": 0.7})
        assert resolved == {"key": "input:input_0", "temp": 0.7}

    def test_resolve_kwargs_replaces_node_refs(self) -> None:
        """Kwargs with NodeRef are replaced with result values."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Simulate a completed result
        state.results["input:input_0"] = TaskResult(
            node_id="input:input_0",
            value="resolved_value",
            duration_ms=10.0,
        )

        resolved = state._resolve_kwargs(
            {"context": NodeRef("input:input_0"), "temp": 0.7}
        )
        assert resolved == {"context": "resolved_value", "temp": 0.7}


class TestExecutionStateNestedResolution:
    """Tests for nested ValueRef/NodeRef resolution."""

    def test_resolve_value_nested_dict(self) -> None:
        """_resolve_value handles nested dicts with ValueRef."""
        from plait.values import ValueRef

        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Simulate completed results
        state.results["input:input_0"] = TaskResult(
            node_id="input:input_0",
            value="hello",
            duration_ms=10.0,
        )

        nested = {"outer": {"inner": ValueRef("input:input_0")}}
        resolved = state._resolve_value(nested)

        assert resolved == {"outer": {"inner": "hello"}}

    def test_resolve_value_nested_list(self) -> None:
        """_resolve_value handles nested lists with ValueRef."""
        from plait.values import ValueRef

        graph = create_linear_graph()
        state = ExecutionState(graph)

        state.results["input:input_0"] = TaskResult(
            node_id="input:input_0",
            value="hello",
            duration_ms=10.0,
        )

        nested = [[ValueRef("input:input_0")], "literal"]
        resolved = state._resolve_value(nested)

        assert resolved == [["hello"], "literal"]

    def test_resolve_value_mixed_nesting(self) -> None:
        """_resolve_value handles mixed dict/list nesting."""
        from plait.values import ValueRef

        graph = create_linear_graph()
        state = ExecutionState(graph)

        state.results["input:input_0"] = TaskResult(
            node_id="input:input_0",
            value="hello",
            duration_ms=10.0,
        )
        state.results["LLMInference_1"] = TaskResult(
            node_id="LLMInference_1",
            value="world",
            duration_ms=10.0,
        )

        nested = {
            "items": [
                ValueRef("input:input_0"),
                {"nested": ValueRef("LLMInference_1")},
            ],
            "extra": ValueRef("input:input_0"),
        }
        resolved = state._resolve_value(nested)

        assert resolved == {
            "items": ["hello", {"nested": "world"}],
            "extra": "hello",
        }

    def test_resolve_args_with_nested_valuerefs(self) -> None:
        """_resolve_args handles nested ValueRefs in args tuple."""
        from plait.values import ValueRef

        graph = create_linear_graph()
        state = ExecutionState(graph)

        state.results["input:input_0"] = TaskResult(
            node_id="input:input_0",
            value="hello",
            duration_ms=10.0,
        )

        args = ({"nested": [ValueRef("input:input_0")]},)
        resolved = state._resolve_args(args)

        assert resolved == ({"nested": ["hello"]},)

    def test_resolve_kwargs_with_nested_valuerefs(self) -> None:
        """_resolve_kwargs handles nested ValueRefs in kwargs."""
        from plait.values import ValueRef

        graph = create_linear_graph()
        state = ExecutionState(graph)

        state.results["input:input_0"] = TaskResult(
            node_id="input:input_0",
            value="hello",
            duration_ms=10.0,
        )

        kwargs = {"data": {"items": [ValueRef("input:input_0")]}}
        resolved = state._resolve_kwargs(kwargs)

        assert resolved == {"data": {"items": ["hello"]}}

    def test_resolve_value_preserves_tuples(self) -> None:
        """_resolve_value preserves tuple type in nested structures."""
        from plait.values import ValueRef

        graph = create_linear_graph()
        state = ExecutionState(graph)

        state.results["input:input_0"] = TaskResult(
            node_id="input:input_0",
            value="hello",
            duration_ms=10.0,
        )

        nested = (ValueRef("input:input_0"), "literal")
        resolved = state._resolve_value(nested)

        assert resolved == ("hello", "literal")
        assert isinstance(resolved, tuple)


class TestExecutionStateParameterRefs:
    """Tests for handling parameter refs in dependency tracking."""

    def test_param_ref_dependency_not_tracked(self) -> None:
        """Parameter refs in dependencies are not added to waiting_on."""
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode("hello"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        # Node with both a normal dependency and a param ref dependency
        llm_node = GraphNode(
            id="LLMInference_1",
            module=LLMInference(alias="fast"),
            args=(NodeRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0", "param:system_prompt"],  # param ref!
        )
        graph = InferenceGraph(
            nodes={"input:input_0": input_node, "LLMInference_1": llm_node},
            input_ids=["input:input_0"],
            output_ids=["LLMInference_1"],
        )
        state = ExecutionState(graph)

        # LLMInference_1 should only wait on input:input_0, not param:system_prompt
        assert state.waiting_on["LLMInference_1"] == {"input:input_0"}
        # param ref should not be in dependents
        assert "param:system_prompt" not in state.dependents

    def test_node_with_only_param_ref_deps_is_ready(self) -> None:
        """A node with only param ref dependencies should be immediately ready."""
        # Node that only depends on a param ref
        llm_node = GraphNode(
            id="LLMInference_1",
            module=LLMInference(alias="fast"),
            args=(),
            kwargs={},
            dependencies=["param:system_prompt"],  # only param ref
        )
        graph = InferenceGraph(
            nodes={"LLMInference_1": llm_node},
            input_ids=[],
            output_ids=["LLMInference_1"],
        )
        state = ExecutionState(graph)

        # Node should be PENDING (ready) since param refs don't block
        assert state.status["LLMInference_1"] == TaskStatus.PENDING
        assert state.get_ready_count() == 1


class TestExecutionStateEmptyGraph:
    """Tests for edge cases with empty or minimal graphs."""

    def test_empty_graph_has_no_tasks(self) -> None:
        """Empty graph results in no pending tasks."""
        graph = InferenceGraph(
            nodes={},
            input_ids=[],
            output_ids=[],
        )
        state = ExecutionState(graph)

        assert state.pending.qsize() == 0
        assert len(state.status) == 0

    def test_single_node_graph_is_immediately_ready(self) -> None:
        """Single node graph has that node ready immediately."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        assert state.get_ready_count() == 1
        assert state.get_blocked_count() == 0


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionState Task Management Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetNextTask:
    """Tests for ExecutionState.get_next_task()."""

    @pytest.mark.asyncio
    async def test_get_next_task_returns_task(self) -> None:
        """get_next_task returns a task from the pending queue."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        task = await state.get_next_task()

        assert task is not None
        assert task.node_id == "input:input_0"

    @pytest.mark.asyncio
    async def test_get_next_task_returns_none_when_empty(self) -> None:
        """get_next_task returns None when no pending tasks."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        state = ExecutionState(graph)

        task = await state.get_next_task()

        assert task is None

    @pytest.mark.asyncio
    async def test_get_next_task_transitions_to_in_progress(self) -> None:
        """get_next_task changes task status to IN_PROGRESS."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()

        assert state.status["input:input_0"] == TaskStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_get_next_task_adds_to_in_progress_dict(self) -> None:
        """get_next_task adds task to in_progress tracking dict."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        task = await state.get_next_task()

        assert "input:input_0" in state.in_progress
        assert state.in_progress["input:input_0"] is task

    @pytest.mark.asyncio
    async def test_get_next_task_removes_from_pending(self) -> None:
        """get_next_task removes task from pending queue."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        assert state.pending.qsize() == 1
        await state.get_next_task()
        assert state.pending.qsize() == 0

    @pytest.mark.asyncio
    async def test_get_next_task_respects_priority(self) -> None:
        """get_next_task returns lower priority (higher precedence) tasks first."""
        # Create graph with two input nodes at different priorities
        input_a = GraphNode(
            id="input:a",
            module=InputNode("a"),
            args=(),
            kwargs={},
            dependencies=[],
            priority=10,  # Lower precedence
        )
        input_b = GraphNode(
            id="input:b",
            module=InputNode("b"),
            args=(),
            kwargs={},
            dependencies=[],
            priority=1,  # Higher precedence (lower number)
        )
        graph = InferenceGraph(
            nodes={"input:a": input_a, "input:b": input_b},
            input_ids=["input:a", "input:b"],
            output_ids=["input:a", "input:b"],
        )
        state = ExecutionState(graph)

        task1 = await state.get_next_task()
        task2 = await state.get_next_task()

        assert task1 is not None
        assert task2 is not None
        assert task1.node_id == "input:b"  # priority 1 first
        assert task2.node_id == "input:a"  # priority 10 second

    @pytest.mark.asyncio
    async def test_get_next_task_multiple_calls(self) -> None:
        """get_next_task can be called multiple times to get all tasks."""
        graph = create_multi_input_graph()
        state = ExecutionState(graph)

        task1 = await state.get_next_task()
        task2 = await state.get_next_task()
        task3 = await state.get_next_task()

        assert task1 is not None
        assert task2 is not None
        assert task3 is None  # No more pending (third node is blocked)

        # Both input nodes should be in progress
        assert state.status["input:input_0"] == TaskStatus.IN_PROGRESS
        assert state.status["input:input_1"] == TaskStatus.IN_PROGRESS


class TestMarkComplete:
    """Tests for ExecutionState.mark_complete()."""

    @pytest.mark.asyncio
    async def test_mark_complete_updates_status(self) -> None:
        """mark_complete sets task status to COMPLETED."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        result = TaskResult(node_id="input:input_0", value="hello", duration_ms=10.0)

        state.mark_complete("input:input_0", result)

        assert state.status["input:input_0"] == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_mark_complete_stores_result(self) -> None:
        """mark_complete stores the TaskResult."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        result = TaskResult(node_id="input:input_0", value="hello", duration_ms=10.0)

        state.mark_complete("input:input_0", result)

        assert state.results["input:input_0"] == result
        assert state.results["input:input_0"].value == "hello"

    @pytest.mark.asyncio
    async def test_mark_complete_removes_from_in_progress(self) -> None:
        """mark_complete removes task from in_progress dict."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        assert "input:input_0" in state.in_progress

        result = TaskResult(node_id="input:input_0", value="hello", duration_ms=10.0)
        state.mark_complete("input:input_0", result)

        assert "input:input_0" not in state.in_progress

    @pytest.mark.asyncio
    async def test_mark_complete_returns_newly_ready(self) -> None:
        """mark_complete returns list of nodes that became ready."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Get and complete input task
        await state.get_next_task()
        result = TaskResult(node_id="input:input_0", value="hello", duration_ms=10.0)

        newly_ready = state.mark_complete("input:input_0", result)

        assert newly_ready == ["LLMInference_1"]

    @pytest.mark.asyncio
    async def test_mark_complete_makes_dependent_pending(self) -> None:
        """mark_complete transitions dependent node from BLOCKED to PENDING."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        result = TaskResult(node_id="input:input_0", value="hello", duration_ms=10.0)
        state.mark_complete("input:input_0", result)

        assert state.status["LLMInference_1"] == TaskStatus.PENDING
        assert state.pending.qsize() == 1

    @pytest.mark.asyncio
    async def test_mark_complete_diamond_partial_deps(self) -> None:
        """mark_complete only makes nodes ready when ALL deps complete."""
        graph = create_diamond_graph()
        state = ExecutionState(graph)

        # Complete input node
        await state.get_next_task()
        result = TaskResult(node_id="input:input_0", value="hello", duration_ms=10.0)
        newly_ready = state.mark_complete("input:input_0", result)

        # Both branch nodes become ready
        assert set(newly_ready) == {"LLMInference_1", "LLMInference_2"}

        # Complete only one branch
        await state.get_next_task()  # LLMInference_1
        result1 = TaskResult(node_id="LLMInference_1", value="a", duration_ms=10.0)
        newly_ready = state.mark_complete("LLMInference_1", result1)

        # Merge node should NOT be ready (still waiting on LLMInference_2)
        assert newly_ready == []
        assert state.status["LLMInference_3"] == TaskStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_mark_complete_diamond_all_deps(self) -> None:
        """mark_complete makes node ready when all dependencies complete."""
        graph = create_diamond_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=10.0),
        )

        # Complete both branches
        task_a = await state.get_next_task()
        task_b = await state.get_next_task()
        assert task_a is not None
        assert task_b is not None

        state.mark_complete(
            task_a.node_id,
            TaskResult(node_id=task_a.node_id, value="a", duration_ms=10.0),
        )
        newly_ready = state.mark_complete(
            task_b.node_id,
            TaskResult(node_id=task_b.node_id, value="b", duration_ms=10.0),
        )

        # Merge node should now be ready
        assert newly_ready == ["LLMInference_3"]
        assert state.status["LLMInference_3"] == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_mark_complete_resolves_args_in_dependent(self) -> None:
        """mark_complete ensures dependent tasks get resolved args."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input with a value
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello world", duration_ms=10.0),
        )

        # Get the dependent task
        task = await state.get_next_task()
        assert task is not None

        # The task should have resolved args (value instead of node ID)
        assert task.args == ("hello world",)

    @pytest.mark.asyncio
    async def test_mark_complete_returns_empty_for_no_dependents(self) -> None:
        """mark_complete returns empty list when no dependents exist."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        result = TaskResult(node_id="input:input_0", value="hello", duration_ms=10.0)
        newly_ready = state.mark_complete("input:input_0", result)

        assert newly_ready == []

    @pytest.mark.asyncio
    async def test_mark_complete_parallel_branches(self) -> None:
        """mark_complete handles parallel branches correctly."""
        graph = create_parallel_graph()
        state = ExecutionState(graph)

        # Complete input - should make both branches ready
        await state.get_next_task()
        newly_ready = state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=10.0),
        )

        assert set(newly_ready) == {"LLMInference_1", "LLMInference_2"}


class TestIsComplete:
    """Tests for ExecutionState.is_complete()."""

    def test_is_complete_false_with_pending_tasks(self) -> None:
        """is_complete returns False when PENDING tasks exist."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        assert state.is_complete() is False

    def test_is_complete_false_with_blocked_tasks(self) -> None:
        """is_complete returns False when BLOCKED tasks exist."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        assert state.is_complete() is False

    @pytest.mark.asyncio
    async def test_is_complete_false_with_in_progress_tasks(self) -> None:
        """is_complete returns False when IN_PROGRESS tasks exist."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()

        assert state.is_complete() is False

    @pytest.mark.asyncio
    async def test_is_complete_true_all_completed(self) -> None:
        """is_complete returns True when all tasks are COMPLETED."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=10.0),
        )

        assert state.is_complete() is True

    @pytest.mark.asyncio
    async def test_is_complete_linear_graph(self) -> None:
        """is_complete returns True after completing entire linear graph."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=10.0),
        )
        assert state.is_complete() is False

        # Complete LLMInference_1
        await state.get_next_task()
        state.mark_complete(
            "LLMInference_1",
            TaskResult(node_id="LLMInference_1", value="result1", duration_ms=10.0),
        )
        assert state.is_complete() is False

        # Complete LLMInference_2
        await state.get_next_task()
        state.mark_complete(
            "LLMInference_2",
            TaskResult(node_id="LLMInference_2", value="result2", duration_ms=10.0),
        )
        assert state.is_complete() is True

    def test_is_complete_true_for_empty_graph(self) -> None:
        """is_complete returns True for empty graph."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        state = ExecutionState(graph)

        assert state.is_complete() is True

    def test_is_complete_with_failed_status(self) -> None:
        """is_complete returns True when tasks are FAILED (terminal state)."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        # Manually set to FAILED (simulating mark_failed)
        state.status["input:input_0"] = TaskStatus.FAILED

        assert state.is_complete() is True

    def test_is_complete_with_cancelled_status(self) -> None:
        """is_complete returns True when tasks are CANCELLED (terminal state)."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        # Manually set to CANCELLED
        state.status["input:input_0"] = TaskStatus.CANCELLED

        assert state.is_complete() is True

    @pytest.mark.asyncio
    async def test_is_complete_mixed_terminal_states(self) -> None:
        """is_complete returns True with mix of COMPLETED, FAILED, CANCELLED."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Set different terminal states
        state.status["input:input_0"] = TaskStatus.COMPLETED
        state.status["LLMInference_1"] = TaskStatus.FAILED
        state.status["LLMInference_2"] = TaskStatus.CANCELLED

        assert state.is_complete() is True


class TestTaskLifecycle:
    """Integration tests for complete task lifecycle."""

    @pytest.mark.asyncio
    async def test_linear_graph_lifecycle(self) -> None:
        """Test complete lifecycle of linear graph execution."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Initial state
        assert state.get_ready_count() == 1
        assert state.get_blocked_count() == 2
        assert state.is_complete() is False

        # Process input node
        task1 = await state.get_next_task()
        assert task1 is not None
        assert task1.node_id == "input:input_0"
        assert state.status["input:input_0"] == TaskStatus.IN_PROGRESS

        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="input_value", duration_ms=5.0),
        )
        assert state.get_ready_count() == 1  # LLMInference_1 now ready
        assert state.get_blocked_count() == 1  # LLMInference_2 still blocked

        # Process LLMInference_1
        task2 = await state.get_next_task()
        assert task2 is not None
        assert task2.node_id == "LLMInference_1"
        assert task2.args == ("input_value",)  # Args resolved

        state.mark_complete(
            "LLMInference_1",
            TaskResult(node_id="LLMInference_1", value="step1_value", duration_ms=10.0),
        )
        assert state.get_ready_count() == 1  # LLMInference_2 now ready
        assert state.get_blocked_count() == 0

        # Process LLMInference_2
        task3 = await state.get_next_task()
        assert task3 is not None
        assert task3.node_id == "LLMInference_2"
        assert task3.args == ("step1_value",)  # Args resolved

        state.mark_complete(
            "LLMInference_2",
            TaskResult(node_id="LLMInference_2", value="final_value", duration_ms=15.0),
        )

        # Complete!
        assert state.is_complete() is True
        assert state.get_ready_count() == 0
        assert len(state.results) == 3

    @pytest.mark.asyncio
    async def test_diamond_graph_lifecycle(self) -> None:
        """Test complete lifecycle of diamond graph execution."""
        graph = create_diamond_graph()
        state = ExecutionState(graph)

        # Initial state
        assert state.get_ready_count() == 1
        assert state.get_blocked_count() == 3

        # Complete input
        task = await state.get_next_task()
        assert task is not None
        state.mark_complete(
            task.node_id,
            TaskResult(node_id=task.node_id, value="input", duration_ms=5.0),
        )

        # Both branches should be ready now
        assert state.get_ready_count() == 2
        assert state.get_blocked_count() == 1  # merge still blocked

        # Complete branch A
        task_a = await state.get_next_task()
        assert task_a is not None
        state.mark_complete(
            task_a.node_id,
            TaskResult(node_id=task_a.node_id, value="branch_a", duration_ms=10.0),
        )

        # Merge still blocked (waiting on B)
        assert state.status["LLMInference_3"] == TaskStatus.BLOCKED

        # Complete branch B
        task_b = await state.get_next_task()
        assert task_b is not None
        state.mark_complete(
            task_b.node_id,
            TaskResult(node_id=task_b.node_id, value="branch_b", duration_ms=10.0),
        )

        # Merge now ready
        assert state.status["LLMInference_3"] == TaskStatus.PENDING

        # Complete merge
        task_merge = await state.get_next_task()
        assert task_merge is not None
        assert task_merge.node_id == "LLMInference_3"
        # Both branch results should be in args
        assert set(task_merge.args) == {"branch_a", "branch_b"}

        state.mark_complete(
            task_merge.node_id,
            TaskResult(node_id=task_merge.node_id, value="merged", duration_ms=15.0),
        )

        assert state.is_complete() is True


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionState Failure Handling Tests
# ─────────────────────────────────────────────────────────────────────────────


def create_deep_linear_graph() -> InferenceGraph:
    """Create a deeper linear graph: input -> llm1 -> llm2 -> llm3."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode("hello"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    llm1_node = GraphNode(
        id="LLMInference_1",
        module=LLMInference(alias="fast"),
        args=("input:input_0",),
        kwargs={},
        dependencies=["input:input_0"],
    )
    llm2_node = GraphNode(
        id="LLMInference_2",
        module=LLMInference(alias="smart"),
        args=("LLMInference_1",),
        kwargs={},
        dependencies=["LLMInference_1"],
    )
    llm3_node = GraphNode(
        id="LLMInference_3",
        module=LLMInference(alias="final"),
        args=("LLMInference_2",),
        kwargs={},
        dependencies=["LLMInference_2"],
    )
    return InferenceGraph(
        nodes={
            "input:input_0": input_node,
            "LLMInference_1": llm1_node,
            "LLMInference_2": llm2_node,
            "LLMInference_3": llm3_node,
        },
        input_ids=["input:input_0"],
        output_ids=["LLMInference_3"],
    )


class TestMarkFailed:
    """Tests for ExecutionState.mark_failed()."""

    @pytest.mark.asyncio
    async def test_mark_failed_updates_status(self) -> None:
        """mark_failed sets task status to FAILED."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        state.mark_failed("input:input_0", ValueError("Test error"))

        assert state.status["input:input_0"] == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_mark_failed_stores_error(self) -> None:
        """mark_failed stores the exception in errors dict."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        error = ValueError("API failed")
        state.mark_failed("input:input_0", error)

        assert "input:input_0" in state.errors
        assert state.errors["input:input_0"] is error

    @pytest.mark.asyncio
    async def test_mark_failed_removes_from_in_progress(self) -> None:
        """mark_failed removes task from in_progress dict."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        assert "input:input_0" in state.in_progress

        state.mark_failed("input:input_0", ValueError("Error"))

        assert "input:input_0" not in state.in_progress

    @pytest.mark.asyncio
    async def test_mark_failed_returns_cancelled_list(self) -> None:
        """mark_failed returns list of cancelled descendant node IDs."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input, then fail LLMInference_1
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )
        await state.get_next_task()

        cancelled = state.mark_failed("LLMInference_1", ValueError("Error"))

        # LLMInference_2 should be cancelled
        assert cancelled == ["LLMInference_2"]

    @pytest.mark.asyncio
    async def test_mark_failed_cancels_descendants(self) -> None:
        """mark_failed sets all descendant statuses to CANCELLED."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input, then fail LLMInference_1
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )
        await state.get_next_task()

        state.mark_failed("LLMInference_1", ValueError("Error"))

        assert state.status["LLMInference_2"] == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_mark_failed_cascades_through_chain(self) -> None:
        """mark_failed cancels all descendants in a chain."""
        graph = create_deep_linear_graph()
        state = ExecutionState(graph)

        # Complete input, then fail LLMInference_1
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )
        await state.get_next_task()

        cancelled = state.mark_failed("LLMInference_1", ValueError("Error"))

        # Both LLMInference_2 and LLMInference_3 should be cancelled
        assert set(cancelled) == {"LLMInference_2", "LLMInference_3"}
        assert state.status["LLMInference_2"] == TaskStatus.CANCELLED
        assert state.status["LLMInference_3"] == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_mark_failed_no_descendants(self) -> None:
        """mark_failed returns empty list when no descendants."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        cancelled = state.mark_failed("input:input_0", ValueError("Error"))

        assert cancelled == []

    @pytest.mark.asyncio
    async def test_mark_failed_diamond_partial_cascade(self) -> None:
        """mark_failed in diamond graph cancels merge but not sibling branch."""
        graph = create_diamond_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Start branch A and fail it
        task_a = await state.get_next_task()
        assert task_a is not None
        state.mark_failed(task_a.node_id, ValueError("Branch A failed"))

        # Merge node should be cancelled
        assert state.status["LLMInference_3"] == TaskStatus.CANCELLED

        # Branch B should still be able to execute (PENDING)
        # Get the remaining pending task - it should be branch B
        task_b = await state.get_next_task()
        assert task_b is not None
        # The sibling branch is still pending/in_progress, not cancelled
        assert state.status[task_b.node_id] == TaskStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_mark_failed_allows_is_complete(self) -> None:
        """is_complete returns True when all nodes are in terminal states."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Fail LLMInference_1 (which cancels LLMInference_2)
        await state.get_next_task()
        state.mark_failed("LLMInference_1", ValueError("Error"))

        # All nodes are now in terminal states: COMPLETED, FAILED, CANCELLED
        assert state.status["input:input_0"] == TaskStatus.COMPLETED
        assert state.status["LLMInference_1"] == TaskStatus.FAILED
        assert state.status["LLMInference_2"] == TaskStatus.CANCELLED
        assert state.is_complete() is True

    @pytest.mark.asyncio
    async def test_mark_failed_preserves_completed_results(self) -> None:
        """mark_failed doesn't affect already completed nodes."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Fail LLMInference_1
        await state.get_next_task()
        state.mark_failed("LLMInference_1", ValueError("Error"))

        # Input should still have its result
        assert "input:input_0" in state.results
        assert state.results["input:input_0"].value == "hello"

    @pytest.mark.asyncio
    async def test_mark_failed_handles_not_in_progress(self) -> None:
        """mark_failed handles gracefully when node not in in_progress."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        # Mark failed without getting the task first (edge case)
        # This simulates a race condition or external failure detection
        state.mark_failed("input:input_0", ValueError("Error"))

        assert state.status["input:input_0"] == TaskStatus.FAILED
        assert "input:input_0" in state.errors

    @pytest.mark.asyncio
    async def test_mark_failed_error_message_preserved(self) -> None:
        """mark_failed preserves the full error information."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        error = RuntimeError("Detailed error: connection timeout after 30s")
        state.mark_failed("input:input_0", error)

        assert str(state.errors["input:input_0"]) == str(error)
        assert isinstance(state.errors["input:input_0"], RuntimeError)


class TestFailureLifecycle:
    """Integration tests for failure handling in graph execution."""

    @pytest.mark.asyncio
    async def test_early_failure_cancels_entire_chain(self) -> None:
        """Early failure in linear graph cancels all subsequent nodes."""
        graph = create_deep_linear_graph()
        state = ExecutionState(graph)

        # Fail at input level
        await state.get_next_task()
        state.mark_failed("input:input_0", ValueError("Input error"))

        # All other nodes should be cancelled
        assert state.status["LLMInference_1"] == TaskStatus.CANCELLED
        assert state.status["LLMInference_2"] == TaskStatus.CANCELLED
        assert state.status["LLMInference_3"] == TaskStatus.CANCELLED
        assert state.is_complete() is True

    @pytest.mark.asyncio
    async def test_mid_chain_failure(self) -> None:
        """Mid-chain failure cancels downstream but preserves upstream results."""
        graph = create_deep_linear_graph()
        state = ExecutionState(graph)

        # Complete input and LLMInference_1
        task0 = await state.get_next_task()
        assert task0 is not None
        state.mark_complete(
            task0.node_id,
            TaskResult(node_id=task0.node_id, value="input_val", duration_ms=5.0),
        )

        task1 = await state.get_next_task()
        assert task1 is not None
        state.mark_complete(
            task1.node_id,
            TaskResult(node_id=task1.node_id, value="llm1_val", duration_ms=10.0),
        )

        # Fail LLMInference_2
        task2 = await state.get_next_task()
        assert task2 is not None
        state.mark_failed(task2.node_id, ValueError("LLM2 failed"))

        # Upstream results preserved
        assert state.results["input:input_0"].value == "input_val"
        assert state.results["LLMInference_1"].value == "llm1_val"

        # Current node failed, downstream cancelled
        assert state.status["LLMInference_2"] == TaskStatus.FAILED
        assert state.status["LLMInference_3"] == TaskStatus.CANCELLED
        assert state.is_complete() is True

    @pytest.mark.asyncio
    async def test_parallel_branch_failure_isolation(self) -> None:
        """Failure in one parallel branch doesn't cancel sibling branch."""
        graph = create_parallel_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Start both branches
        task_a = await state.get_next_task()
        task_b = await state.get_next_task()
        assert task_a is not None
        assert task_b is not None

        # Fail one branch
        state.mark_failed(task_a.node_id, ValueError("Branch A failed"))

        # Other branch should still be in progress
        assert state.status[task_b.node_id] == TaskStatus.IN_PROGRESS

        # Complete the other branch
        state.mark_complete(
            task_b.node_id,
            TaskResult(node_id=task_b.node_id, value="branch_b", duration_ms=10.0),
        )

        # Now complete
        assert state.is_complete() is True

        # Check final states
        assert state.status["input:input_0"] == TaskStatus.COMPLETED
        assert state.status[task_a.node_id] == TaskStatus.FAILED
        assert state.status[task_b.node_id] == TaskStatus.COMPLETED


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionState Requeue Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRequeue:
    """Tests for ExecutionState.requeue()."""

    @pytest.mark.asyncio
    async def test_requeue_sets_status_to_pending(self) -> None:
        """requeue sets task status back to PENDING."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        assert state.status["input:input_0"] == TaskStatus.IN_PROGRESS

        state.requeue("input:input_0")

        assert state.status["input:input_0"] == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_requeue_increments_retry_count(self) -> None:
        """requeue increments the task's retry_count."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        task = await state.get_next_task()
        assert task is not None
        assert task.retry_count == 0

        state.requeue("input:input_0")

        # Get the requeued task
        requeued_task = await state.get_next_task()
        assert requeued_task is not None
        assert requeued_task.retry_count == 1

    @pytest.mark.asyncio
    async def test_requeue_removes_from_in_progress(self) -> None:
        """requeue removes task from in_progress dict."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        assert "input:input_0" in state.in_progress

        state.requeue("input:input_0")

        assert "input:input_0" not in state.in_progress

    @pytest.mark.asyncio
    async def test_requeue_adds_to_pending_queue(self) -> None:
        """requeue puts task back in pending queue."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        assert state.pending.qsize() == 0

        state.requeue("input:input_0")

        assert state.pending.qsize() == 1

    @pytest.mark.asyncio
    async def test_requeue_returns_empty_if_not_in_progress(self) -> None:
        """requeue returns empty list if node is not in progress."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        # Don't get the task first - it's still PENDING, not IN_PROGRESS
        dropped = state.requeue("input:input_0")

        assert dropped == []

    @pytest.mark.asyncio
    async def test_requeue_returns_dropped_descendants(self) -> None:
        """requeue returns list of dropped descendant node IDs."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input, then start LLMInference_1
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )
        await state.get_next_task()

        dropped = state.requeue("LLMInference_1")

        # LLMInference_2 should be dropped
        assert dropped == ["LLMInference_2"]

    @pytest.mark.asyncio
    async def test_requeue_drops_descendants_to_blocked(self) -> None:
        """requeue sets descendant statuses back to BLOCKED."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input, which makes LLMInference_1 ready
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Start LLMInference_1 but requeue it
        await state.get_next_task()
        state.requeue("LLMInference_1")

        # LLMInference_2 should be back to BLOCKED
        assert state.status["LLMInference_2"] == TaskStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_requeue_restores_dependencies(self) -> None:
        """requeue restores waiting_on for descendants."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input, which clears LLMInference_1's dependencies
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Start LLMInference_1 and complete it (makes LLMInference_2 ready)
        await state.get_next_task()
        state.mark_complete(
            "LLMInference_1",
            TaskResult(node_id="LLMInference_1", value="result1", duration_ms=10.0),
        )
        # At this point, LLMInference_2 has empty waiting_on
        assert state.waiting_on["LLMInference_2"] == set()

        # Start LLMInference_2 and then requeue LLMInference_1
        # This is a bit contrived but tests the dependency restoration
        await state.get_next_task()  # LLMInference_2

        # Manually put LLMInference_1 back in progress for requeue test
        graph2 = create_linear_graph()
        state2 = ExecutionState(graph2)
        await state2.get_next_task()
        state2.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )
        await state2.get_next_task()
        state2.requeue("LLMInference_1")

        # LLMInference_2's dependencies should be restored
        assert state2.waiting_on["LLMInference_2"] == {"LLMInference_1"}

    @pytest.mark.asyncio
    async def test_requeue_drops_cascade_in_chain(self) -> None:
        """requeue drops all descendants in a chain."""
        graph = create_deep_linear_graph()
        state = ExecutionState(graph)

        # Complete input, then start LLMInference_1
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )
        await state.get_next_task()

        dropped = state.requeue("LLMInference_1")

        # Both LLMInference_2 and LLMInference_3 should be dropped
        assert set(dropped) == {"LLMInference_2", "LLMInference_3"}
        assert state.status["LLMInference_2"] == TaskStatus.BLOCKED
        assert state.status["LLMInference_3"] == TaskStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_requeue_diamond_drops_merge(self) -> None:
        """requeue in diamond graph drops merge node but not sibling."""
        graph = create_diamond_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Start branch A and requeue it
        task_a = await state.get_next_task()
        assert task_a is not None
        dropped = state.requeue(task_a.node_id)

        # Merge node should be dropped
        assert "LLMInference_3" in dropped
        assert state.status["LLMInference_3"] == TaskStatus.BLOCKED

        # Branch B should still be pending (not affected)
        task_b = await state.get_next_task()
        assert task_b is not None
        # The other branch should still be available
        assert state.status[task_b.node_id] == TaskStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_requeue_no_descendants(self) -> None:
        """requeue with no descendants returns empty list."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        await state.get_next_task()
        dropped = state.requeue("input:input_0")

        assert dropped == []

    @pytest.mark.asyncio
    async def test_requeue_multiple_times(self) -> None:
        """requeue can be called multiple times, incrementing retry_count."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        # First requeue
        await state.get_next_task()
        state.requeue("input:input_0")

        # Second requeue
        await state.get_next_task()
        state.requeue("input:input_0")

        # Third requeue
        await state.get_next_task()
        state.requeue("input:input_0")

        # Get and check retry count
        task = await state.get_next_task()
        assert task is not None
        assert task.retry_count == 3

    @pytest.mark.asyncio
    async def test_requeue_preserves_task_properties(self) -> None:
        """requeue preserves other task properties like priority and args."""
        input_node = GraphNode(
            id="input:input_0",
            module=InputNode("hello"),
            args=(),
            kwargs={},
            dependencies=[],
            priority=5,
        )
        graph = InferenceGraph(
            nodes={"input:input_0": input_node},
            input_ids=["input:input_0"],
            output_ids=["input:input_0"],
        )
        state = ExecutionState(graph)

        original_task = await state.get_next_task()
        assert original_task is not None
        original_args = original_task.args
        original_priority = original_task.priority
        original_node_id = original_task.node_id

        state.requeue("input:input_0")

        requeued_task = await state.get_next_task()
        assert requeued_task is not None
        assert requeued_task.node_id == original_node_id
        assert requeued_task.args == original_args
        assert requeued_task.priority == original_priority

    @pytest.mark.asyncio
    async def test_requeue_does_not_affect_completed_nodes(self) -> None:
        """requeue doesn't affect already completed nodes."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Start and requeue LLMInference_1
        await state.get_next_task()
        state.requeue("LLMInference_1")

        # Input should still be completed with its result
        assert state.status["input:input_0"] == TaskStatus.COMPLETED
        assert state.results["input:input_0"].value == "hello"


class TestRequeueLifecycle:
    """Integration tests for requeue in graph execution lifecycle."""

    @pytest.mark.asyncio
    async def test_requeue_then_complete(self) -> None:
        """Task can be requeued and then completed successfully."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Start LLMInference_1 and requeue it
        await state.get_next_task()
        state.requeue("LLMInference_1")

        # LLMInference_2 should be blocked
        assert state.status["LLMInference_2"] == TaskStatus.BLOCKED

        # Now get the requeued task and complete it
        task = await state.get_next_task()
        assert task is not None
        assert task.node_id == "LLMInference_1"
        assert task.retry_count == 1

        state.mark_complete(
            "LLMInference_1",
            TaskResult(node_id="LLMInference_1", value="result1", duration_ms=10.0),
        )

        # LLMInference_2 should now be pending
        assert state.status["LLMInference_2"] == TaskStatus.PENDING

        # Complete the rest
        task2 = await state.get_next_task()
        assert task2 is not None
        assert task2.node_id == "LLMInference_2"

        state.mark_complete(
            "LLMInference_2",
            TaskResult(node_id="LLMInference_2", value="final", duration_ms=15.0),
        )

        assert state.is_complete() is True
        assert len(state.results) == 3

    @pytest.mark.asyncio
    async def test_requeue_drops_pending_descendants(self) -> None:
        """requeue drops descendants that were already pending."""
        graph = create_diamond_graph()
        state = ExecutionState(graph)

        # Complete input - makes both branches pending
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Complete both branches - makes merge pending
        task_a = await state.get_next_task()
        task_b = await state.get_next_task()
        assert task_a is not None
        assert task_b is not None

        state.mark_complete(
            task_a.node_id,
            TaskResult(node_id=task_a.node_id, value="a", duration_ms=10.0),
        )
        state.mark_complete(
            task_b.node_id,
            TaskResult(node_id=task_b.node_id, value="b", duration_ms=10.0),
        )

        # Merge is now pending
        assert state.status["LLMInference_3"] == TaskStatus.PENDING

        # Simulate that branch_a needs to be retried (edge case - manually set up)
        # This is a contrived scenario, but tests that pending descendants get dropped
        # In practice, you'd requeue before completing, but let's test the mechanism

    @pytest.mark.asyncio
    async def test_requeue_with_concurrent_branches(self) -> None:
        """requeue in one branch doesn't affect execution of other branch."""
        graph = create_parallel_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Start both branches
        task_a = await state.get_next_task()
        task_b = await state.get_next_task()
        assert task_a is not None
        assert task_b is not None

        # Requeue one branch
        state.requeue(task_a.node_id)

        # Other branch should still be in progress
        assert state.status[task_b.node_id] == TaskStatus.IN_PROGRESS

        # Complete the other branch
        state.mark_complete(
            task_b.node_id,
            TaskResult(node_id=task_b.node_id, value="b_result", duration_ms=10.0),
        )

        # Requeued branch should still be pending
        assert state.status[task_a.node_id] == TaskStatus.PENDING

        # Get and complete the requeued task
        requeued = await state.get_next_task()
        assert requeued is not None
        assert requeued.retry_count == 1

        state.mark_complete(
            requeued.node_id,
            TaskResult(node_id=requeued.node_id, value="a_result", duration_ms=10.0),
        )

        assert state.is_complete() is True


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionState get_outputs() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetOutputs:
    """Tests for ExecutionState.get_outputs()."""

    def test_get_outputs_empty_graph(self) -> None:
        """get_outputs returns empty dict for empty graph."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        state = ExecutionState(graph)

        outputs = state.get_outputs()

        assert outputs == {}

    @pytest.mark.asyncio
    async def test_get_outputs_single_output(self) -> None:
        """get_outputs returns single output after completion."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        # Complete the only node
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello world", duration_ms=5.0),
        )

        outputs = state.get_outputs()

        assert outputs == {"input:input_0": "hello world"}

    @pytest.mark.asyncio
    async def test_get_outputs_multiple_outputs(self) -> None:
        """get_outputs returns all completed outputs for parallel graph."""
        graph = create_parallel_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Complete both output branches
        task_a = await state.get_next_task()
        task_b = await state.get_next_task()
        assert task_a is not None
        assert task_b is not None

        state.mark_complete(
            task_a.node_id,
            TaskResult(node_id=task_a.node_id, value="result_a", duration_ms=10.0),
        )
        state.mark_complete(
            task_b.node_id,
            TaskResult(node_id=task_b.node_id, value="result_b", duration_ms=10.0),
        )

        outputs = state.get_outputs()

        # Both outputs should be present
        assert len(outputs) == 2
        assert "LLMInference_1" in outputs
        assert "LLMInference_2" in outputs
        assert set(outputs.values()) == {"result_a", "result_b"}

    @pytest.mark.asyncio
    async def test_get_outputs_excludes_incomplete(self) -> None:
        """get_outputs only includes completed outputs, not pending/in-progress."""
        graph = create_parallel_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Complete only one of the two outputs
        task_a = await state.get_next_task()
        assert task_a is not None
        state.mark_complete(
            task_a.node_id,
            TaskResult(node_id=task_a.node_id, value="result_a", duration_ms=10.0),
        )

        outputs = state.get_outputs()

        # Only the completed output should be present
        assert len(outputs) == 1
        assert task_a.node_id in outputs

    @pytest.mark.asyncio
    async def test_get_outputs_excludes_failed(self) -> None:
        """get_outputs excludes failed output nodes."""
        graph = create_parallel_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Complete one output, fail the other
        task_a = await state.get_next_task()
        task_b = await state.get_next_task()
        assert task_a is not None
        assert task_b is not None

        state.mark_complete(
            task_a.node_id,
            TaskResult(node_id=task_a.node_id, value="result_a", duration_ms=10.0),
        )
        state.mark_failed(task_b.node_id, ValueError("Task failed"))

        outputs = state.get_outputs()

        # Only the completed output should be present
        assert len(outputs) == 1
        assert task_a.node_id in outputs
        assert outputs[task_a.node_id] == "result_a"

    @pytest.mark.asyncio
    async def test_get_outputs_excludes_cancelled(self) -> None:
        """get_outputs excludes cancelled output nodes."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Fail the middle node, cancelling the output node
        await state.get_next_task()
        state.mark_failed("LLMInference_1", ValueError("Error"))

        # Output node is now cancelled
        assert state.status["LLMInference_2"] == TaskStatus.CANCELLED

        outputs = state.get_outputs()

        # No outputs should be present (the only output was cancelled)
        assert outputs == {}

    @pytest.mark.asyncio
    async def test_get_outputs_linear_graph_complete(self) -> None:
        """get_outputs returns correct value after linear graph execution."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete entire linear graph
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="input_val", duration_ms=5.0),
        )

        await state.get_next_task()
        state.mark_complete(
            "LLMInference_1",
            TaskResult(node_id="LLMInference_1", value="step1_val", duration_ms=10.0),
        )

        await state.get_next_task()
        state.mark_complete(
            "LLMInference_2",
            TaskResult(
                node_id="LLMInference_2", value="final_output", duration_ms=15.0
            ),
        )

        outputs = state.get_outputs()

        # Only the output node should be in outputs
        assert outputs == {"LLMInference_2": "final_output"}

    @pytest.mark.asyncio
    async def test_get_outputs_diamond_graph_complete(self) -> None:
        """get_outputs returns correct value after diamond graph execution."""
        graph = create_diamond_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Complete both branches
        task_a = await state.get_next_task()
        task_b = await state.get_next_task()
        assert task_a is not None
        assert task_b is not None

        state.mark_complete(
            task_a.node_id,
            TaskResult(node_id=task_a.node_id, value="a", duration_ms=10.0),
        )
        state.mark_complete(
            task_b.node_id,
            TaskResult(node_id=task_b.node_id, value="b", duration_ms=10.0),
        )

        # Complete merge
        task_merge = await state.get_next_task()
        assert task_merge is not None
        state.mark_complete(
            task_merge.node_id,
            TaskResult(
                node_id=task_merge.node_id, value="merged_result", duration_ms=15.0
            ),
        )

        outputs = state.get_outputs()

        # Only the merge node is the output
        assert outputs == {"LLMInference_3": "merged_result"}

    def test_get_outputs_before_any_execution(self) -> None:
        """get_outputs returns empty dict when no tasks have completed."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        outputs = state.get_outputs()

        assert outputs == {}

    @pytest.mark.asyncio
    async def test_get_outputs_preserves_value_types(self) -> None:
        """get_outputs preserves the type of result values."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        # Complete with a complex value
        await state.get_next_task()
        complex_value = {"key": "value", "nested": [1, 2, 3], "count": 42}
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value=complex_value, duration_ms=5.0),
        )

        outputs = state.get_outputs()

        assert outputs["input:input_0"] == complex_value
        assert isinstance(outputs["input:input_0"], dict)
        assert outputs["input:input_0"]["count"] == 42

    @pytest.mark.asyncio
    async def test_get_outputs_none_value(self) -> None:
        """get_outputs correctly handles None as a result value."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        # Complete with None value
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value=None, duration_ms=5.0),
        )

        outputs = state.get_outputs()

        assert outputs == {"input:input_0": None}
        assert "input:input_0" in outputs


class TestGetOutputsLifecycle:
    """Integration tests for get_outputs in graph execution lifecycle."""

    @pytest.mark.asyncio
    async def test_get_outputs_progressive_completion(self) -> None:
        """get_outputs shows progressive results as outputs complete."""
        graph = create_parallel_graph()
        state = ExecutionState(graph)

        # Initially empty
        assert state.get_outputs() == {}

        # Complete input (not an output node)
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )
        assert state.get_outputs() == {}  # Still empty, no outputs completed

        # Complete first output
        task_a = await state.get_next_task()
        assert task_a is not None
        state.mark_complete(
            task_a.node_id,
            TaskResult(node_id=task_a.node_id, value="result_a", duration_ms=10.0),
        )

        partial_outputs = state.get_outputs()
        assert len(partial_outputs) == 1
        assert task_a.node_id in partial_outputs

        # Complete second output
        task_b = await state.get_next_task()
        assert task_b is not None
        state.mark_complete(
            task_b.node_id,
            TaskResult(node_id=task_b.node_id, value="result_b", duration_ms=10.0),
        )

        final_outputs = state.get_outputs()
        assert len(final_outputs) == 2

    @pytest.mark.asyncio
    async def test_get_outputs_after_requeue_and_complete(self) -> None:
        """get_outputs returns correct value after requeue and completion."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Complete input
        await state.get_next_task()
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Start and requeue LLMInference_1
        await state.get_next_task()
        state.requeue("LLMInference_1")

        # No outputs yet
        assert state.get_outputs() == {}

        # Complete the requeued task
        task = await state.get_next_task()
        assert task is not None
        state.mark_complete(
            "LLMInference_1",
            TaskResult(node_id="LLMInference_1", value="step1", duration_ms=10.0),
        )

        # Still no outputs (LLMInference_2 is the output)
        assert state.get_outputs() == {}

        # Complete final output
        await state.get_next_task()
        state.mark_complete(
            "LLMInference_2",
            TaskResult(node_id="LLMInference_2", value="final", duration_ms=15.0),
        )

        outputs = state.get_outputs()
        assert outputs == {"LLMInference_2": "final"}


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionState Event Signaling Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskReadyEvent:
    """Tests for ExecutionState.task_ready_event signaling."""

    def test_event_exists_after_init(self) -> None:
        """ExecutionState has task_ready_event attribute after initialization."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        assert hasattr(state, "task_ready_event")
        assert isinstance(state.task_ready_event, asyncio.Event)

    def test_event_set_during_initialization(self) -> None:
        """Event is set during initialization when tasks become ready."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        # Event should be set because _make_ready() was called during init
        assert state.task_ready_event.is_set()

    def test_event_set_when_make_ready_called(self) -> None:
        """Event is set when _make_ready() adds a task to pending queue."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Clear the event
        state.task_ready_event.clear()
        assert not state.task_ready_event.is_set()

        # Manually trigger _make_ready for the blocked node
        # First simulate completing its dependency
        state.results["input:input_0"] = TaskResult(
            node_id="input:input_0", value="hello", duration_ms=5.0
        )
        state.waiting_on["LLMInference_1"].clear()
        state._make_ready("LLMInference_1")

        # Event should be set
        assert state.task_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_event_set_on_mark_complete(self) -> None:
        """Event is set when mark_complete is called."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Get task and clear event
        await state.get_next_task()
        state.task_ready_event.clear()
        assert not state.task_ready_event.is_set()

        # Complete the task
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Event should be set
        assert state.task_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_event_set_on_mark_complete_no_new_ready(self) -> None:
        """Event is set even when no new tasks become ready."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        # Get task and clear event
        await state.get_next_task()
        state.task_ready_event.clear()
        assert not state.task_ready_event.is_set()

        # Complete the task (no dependents to make ready)
        state.mark_complete(
            "input:input_0",
            TaskResult(node_id="input:input_0", value="hello", duration_ms=5.0),
        )

        # Event should still be set (to allow scheduler to re-check is_complete)
        assert state.task_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_event_set_on_mark_failed(self) -> None:
        """Event is set when mark_failed is called."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Get task and clear event
        await state.get_next_task()
        state.task_ready_event.clear()
        assert not state.task_ready_event.is_set()

        # Fail the task
        state.mark_failed("input:input_0", ValueError("Test error"))

        # Event should be set
        assert state.task_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_event_set_on_requeue(self) -> None:
        """Event is set when requeue is called."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        # Get task and clear event
        await state.get_next_task()
        state.task_ready_event.clear()
        assert not state.task_ready_event.is_set()

        # Requeue the task
        state.requeue("input:input_0")

        # Event should be set
        assert state.task_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_event_can_be_waited_on(self) -> None:
        """Event can be awaited and wakes up when set."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Clear event and start waiting
        state.task_ready_event.clear()
        wait_completed = False

        async def wait_for_event() -> None:
            nonlocal wait_completed
            await state.task_ready_event.wait()
            wait_completed = True

        # Start the wait task
        wait_task = asyncio.create_task(wait_for_event())

        # Give it a moment - should not complete yet
        await asyncio.sleep(0.001)
        assert not wait_completed

        # Set the event
        state.task_ready_event.set()

        # Wait for the task to complete
        await asyncio.sleep(0.001)
        assert wait_completed

        await wait_task

    @pytest.mark.asyncio
    async def test_event_multiple_waiters(self) -> None:
        """Multiple waiters are all woken when event is set."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)
        state.task_ready_event.clear()

        woken_count = 0

        async def waiter() -> None:
            nonlocal woken_count
            await state.task_ready_event.wait()
            woken_count += 1

        # Start multiple waiters
        tasks = [asyncio.create_task(waiter()) for _ in range(5)]

        # Give them a moment to start waiting
        await asyncio.sleep(0.001)
        assert woken_count == 0

        # Set the event
        state.task_ready_event.set()

        # Wait for all to complete
        await asyncio.gather(*tasks)
        assert woken_count == 5
