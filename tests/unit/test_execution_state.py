"""Unit tests for ExecutionState initialization."""

from inf_engine.execution.state import ExecutionState, TaskResult, TaskStatus
from inf_engine.graph import GraphNode, InferenceGraph
from inf_engine.module import LLMInference
from inf_engine.tracing.tracer import InputNode

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
        args=("input:input_0",),
        kwargs={},
        dependencies=["input:input_0"],
    )
    llm2_node = GraphNode(
        id="LLMInference_2",
        module=LLMInference(alias="b"),
        args=("input:input_0",),
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
        args=("input:input_0",),
        kwargs={},
        dependencies=["input:input_0"],
    )
    branch_b = GraphNode(
        id="LLMInference_2",
        module=LLMInference(alias="b"),
        args=("input:input_0",),
        kwargs={},
        dependencies=["input:input_0"],
    )
    merge_node = GraphNode(
        id="LLMInference_3",
        module=LLMInference(alias="merge"),
        args=("LLMInference_1", "LLMInference_2"),
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
        args=("input:input_0", "input:input_1"),
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

    def test_resolve_args_replaces_completed_node_ids(self) -> None:
        """Args referencing completed nodes are replaced with values."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Simulate a completed result
        state.results["input:input_0"] = TaskResult(
            node_id="input:input_0",
            value="resolved_value",
            duration_ms=10.0,
        )

        resolved = state._resolve_args(("input:input_0", "literal"))
        assert resolved == ("resolved_value", "literal")

    def test_resolve_kwargs_passes_through_when_no_results(self) -> None:
        """Kwargs pass through unchanged when no results exist."""
        graph = create_single_input_graph()
        state = ExecutionState(graph)

        resolved = state._resolve_kwargs({"key": "input:input_0", "temp": 0.7})
        assert resolved == {"key": "input:input_0", "temp": 0.7}

    def test_resolve_kwargs_replaces_completed_node_ids(self) -> None:
        """Kwargs referencing completed nodes are replaced with values."""
        graph = create_linear_graph()
        state = ExecutionState(graph)

        # Simulate a completed result
        state.results["input:input_0"] = TaskResult(
            node_id="input:input_0",
            value="resolved_value",
            duration_ms=10.0,
        )

        resolved = state._resolve_kwargs({"context": "input:input_0", "temp": 0.7})
        assert resolved == {"context": "resolved_value", "temp": 0.7}


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
