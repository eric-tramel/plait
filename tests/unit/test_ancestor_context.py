"""Tests for AncestorContext and related graph methods."""

from unittest.mock import MagicMock

from plait.graph import GraphNode, InferenceGraph
from plait.module import LLMInference, Module
from plait.optimization.backward import AncestorContext, build_ancestor_context
from plait.optimization.record import ForwardRecord
from plait.parameter import Parameter


def create_mock_module(param_name: str | None = None) -> MagicMock:
    """Create a mock module with optional parameter.

    Args:
        param_name: If provided, the module will have a parameter with this name.

    Returns:
        A MagicMock configured as an LLMInference module.
    """
    mock = MagicMock(spec=LLMInference)
    if param_name:
        param = Parameter("test value", description=f"Description for {param_name}")
        param._name = param_name
        mock.named_parameters.return_value = [(param_name, param)]
        mock.parameters.return_value = [param]
    else:
        mock.named_parameters.return_value = []
        mock.parameters.return_value = []
    return mock


def create_linear_graph(
    node_count: int = 3,
    with_params: bool = True,
) -> tuple[InferenceGraph, dict[str, Module], dict[str, str]]:
    """Create a linear graph: n0 -> n1 -> ... -> n_last.

    Args:
        node_count: Number of nodes in the chain.
        with_params: Whether to add parameters to modules.

    Returns:
        Tuple of (graph, module_map, node_outputs).
    """
    node_ids = [f"node_{i}" for i in range(node_count)]
    module_map: dict[str, Module] = {}
    nodes: dict[str, GraphNode] = {}
    node_outputs: dict[str, str] = {}

    for i, node_id in enumerate(node_ids):
        deps = [node_ids[i - 1]] if i > 0 else []
        param_name = f"param_{i}" if with_params else None
        mock = create_mock_module(param_name)
        module_map[node_id] = mock
        nodes[node_id] = GraphNode(
            id=node_id,
            module=mock,
            args=(),
            kwargs={},
            dependencies=deps,
            module_name=f"Module_{i}",
        )
        node_outputs[node_id] = f"output_{i}"

    graph = InferenceGraph(
        nodes=nodes,
        input_ids=[node_ids[0]],
        output_ids=[node_ids[-1]],
    )

    return graph, module_map, node_outputs


def create_diamond_graph(
    with_params: bool = True,
) -> tuple[InferenceGraph, dict[str, Module], dict[str, str]]:
    """Create a diamond graph: input -> [a, b] -> merge.

    Args:
        with_params: Whether to add parameters to modules.

    Returns:
        Tuple of (graph, module_map, node_outputs).
    """
    module_map: dict[str, Module] = {}
    nodes: dict[str, GraphNode] = {}
    node_outputs: dict[str, str] = {}

    node_configs = [
        ("input", [], "input_param"),
        ("a", ["input"], "a_param"),
        ("b", ["input"], "b_param"),
        ("merge", ["a", "b"], "merge_param"),
    ]

    for node_id, deps, param_name in node_configs:
        param = param_name if with_params else None
        mock = create_mock_module(param)
        module_map[node_id] = mock
        nodes[node_id] = GraphNode(
            id=node_id,
            module=mock,
            args=(),
            kwargs={},
            dependencies=deps,
            module_name=f"Module({node_id})",
        )
        node_outputs[node_id] = f"output_{node_id}"

    graph = InferenceGraph(
        nodes=nodes,
        input_ids=["input"],
        output_ids=["merge"],
    )

    return graph, module_map, node_outputs


class TestAncestorContextDataclass:
    """Tests for AncestorContext dataclass."""

    def test_empty_context(self) -> None:
        """Empty AncestorContext has empty dicts."""
        ctx = AncestorContext()
        assert ctx.ancestor_values == {}
        assert ctx.ancestor_params == {}
        assert ctx.sibling_values == {}
        assert ctx.sibling_params == {}

    def test_has_ancestors_empty(self) -> None:
        """has_ancestors returns False when empty."""
        ctx = AncestorContext()
        assert ctx.has_ancestors() is False

    def test_has_ancestors_with_values(self) -> None:
        """has_ancestors returns True when ancestor_values populated."""
        ctx = AncestorContext(ancestor_values={"node": "value"})
        assert ctx.has_ancestors() is True

    def test_has_ancestors_with_params(self) -> None:
        """has_ancestors returns True when ancestor_params populated."""
        param = Parameter("test", description="test")
        ctx = AncestorContext(ancestor_params={"param": param})
        assert ctx.has_ancestors() is True

    def test_has_siblings_empty(self) -> None:
        """has_siblings returns False when empty."""
        ctx = AncestorContext()
        assert ctx.has_siblings() is False

    def test_has_siblings_with_values(self) -> None:
        """has_siblings returns True when sibling_values populated."""
        ctx = AncestorContext(sibling_values={"node": "value"})
        assert ctx.has_siblings() is True

    def test_has_siblings_with_params(self) -> None:
        """has_siblings returns True when sibling_params populated."""
        param = Parameter("test", description="test")
        ctx = AncestorContext(sibling_params={"param": param})
        assert ctx.has_siblings() is True

    def test_context_with_all_fields(self) -> None:
        """AncestorContext can be created with all fields."""
        param1 = Parameter("v1", description="d1")
        param2 = Parameter("v2", description="d2")

        ctx = AncestorContext(
            ancestor_values={"a": "value_a"},
            ancestor_params={"a.param": param1},
            sibling_values={"b": "value_b"},
            sibling_params={"b.param": param2},
        )

        assert ctx.ancestor_values == {"a": "value_a"}
        assert ctx.ancestor_params == {"a.param": param1}
        assert ctx.sibling_values == {"b": "value_b"}
        assert ctx.sibling_params == {"b.param": param2}
        assert ctx.has_ancestors() is True
        assert ctx.has_siblings() is True


class TestGraphSiblings:
    """Tests for InferenceGraph.siblings() method."""

    def test_linear_graph_no_siblings(self) -> None:
        """Linear graph nodes have no siblings."""
        graph, _, _ = create_linear_graph(3)
        assert graph.siblings("node_0") == set()
        assert graph.siblings("node_1") == set()
        assert graph.siblings("node_2") == set()

    def test_diamond_graph_parallel_branches_are_siblings(self) -> None:
        """Parallel branches in diamond graph are siblings."""
        graph, _, _ = create_diamond_graph()
        # a and b share ancestor "input", so they are siblings
        assert graph.siblings("a") == {"b"}
        assert graph.siblings("b") == {"a"}

    def test_diamond_graph_input_has_no_siblings(self) -> None:
        """Input node has no siblings (no shared ancestors)."""
        graph, _, _ = create_diamond_graph()
        assert graph.siblings("input") == set()

    def test_diamond_graph_merge_has_no_siblings(self) -> None:
        """Merge node has no siblings (a and b are ancestors)."""
        graph, _, _ = create_diamond_graph()
        # merge has a, b, input as ancestors - no siblings
        assert graph.siblings("merge") == set()


class TestGraphAncestorContext:
    """Tests for InferenceGraph.collect_ancestor_context()."""

    def test_input_node_no_ancestors(self) -> None:
        """Input node has no ancestors."""
        graph, module_map, node_outputs = create_linear_graph(3)
        values, params = graph.collect_ancestor_context(
            "node_0", node_outputs, module_map
        )
        assert values == {}
        assert params == {}

    def test_middle_node_has_one_ancestor(self) -> None:
        """Middle node collects ancestor from upstream."""
        graph, module_map, node_outputs = create_linear_graph(3)
        values, params = graph.collect_ancestor_context(
            "node_1", node_outputs, module_map
        )
        assert "node_0" in values
        assert values["node_0"] == "output_0"

    def test_output_node_has_all_ancestors(self) -> None:
        """Output node collects all upstream ancestors."""
        graph, module_map, node_outputs = create_linear_graph(3)
        values, params = graph.collect_ancestor_context(
            "node_2", node_outputs, module_map
        )
        assert set(values.keys()) == {"node_0", "node_1"}

    def test_diamond_merge_collects_all_branches(self) -> None:
        """Merge node in diamond collects from all branches."""
        graph, module_map, node_outputs = create_diamond_graph()
        values, params = graph.collect_ancestor_context(
            "merge", node_outputs, module_map
        )
        assert set(values.keys()) == {"input", "a", "b"}

    def test_collects_params_from_ancestors(self) -> None:
        """Ancestor parameters are collected with full names."""
        graph, module_map, node_outputs = create_linear_graph(3, with_params=True)
        values, params = graph.collect_ancestor_context(
            "node_2", node_outputs, module_map
        )
        # Should have params from node_0 and node_1
        assert "node_0.param_0" in params
        assert "node_1.param_1" in params


class TestGraphSiblingContext:
    """Tests for InferenceGraph.collect_sibling_context()."""

    def test_linear_graph_no_sibling_context(self) -> None:
        """Linear graph nodes have no sibling context."""
        graph, module_map, node_outputs = create_linear_graph(3)
        values, params = graph.collect_sibling_context(
            "node_1", node_outputs, module_map
        )
        assert values == {}
        assert params == {}

    def test_diamond_branch_collects_sibling(self) -> None:
        """Branch in diamond collects sibling branch values."""
        graph, module_map, node_outputs = create_diamond_graph()
        # From perspective of 'a', sibling is 'b'
        values, params = graph.collect_sibling_context("a", node_outputs, module_map)
        assert "b" in values
        assert values["b"] == "output_b"

    def test_diamond_branch_collects_sibling_params(self) -> None:
        """Branch in diamond collects sibling parameters."""
        graph, module_map, node_outputs = create_diamond_graph(with_params=True)
        values, params = graph.collect_sibling_context("a", node_outputs, module_map)
        assert "b.b_param" in params


class TestBuildAncestorContext:
    """Tests for build_ancestor_context() function."""

    def test_build_for_input_node(self) -> None:
        """Building context for input node returns empty context."""
        graph, module_map, node_outputs = create_linear_graph(3)
        record = ForwardRecord(
            graph=graph,
            node_inputs={nid: {} for nid in graph.nodes},
            node_outputs=node_outputs,
            module_map=module_map,
        )
        ctx = build_ancestor_context("node_0", record)
        assert not ctx.has_ancestors()
        assert not ctx.has_siblings()

    def test_build_for_output_node(self) -> None:
        """Building context for output node includes all ancestors."""
        graph, module_map, node_outputs = create_linear_graph(3)
        record = ForwardRecord(
            graph=graph,
            node_inputs={nid: {} for nid in graph.nodes},
            node_outputs=node_outputs,
            module_map=module_map,
        )
        ctx = build_ancestor_context("node_2", record)
        assert ctx.has_ancestors()
        assert "node_0" in ctx.ancestor_values
        assert "node_1" in ctx.ancestor_values

    def test_build_for_diamond_branch(self) -> None:
        """Building context for diamond branch includes siblings."""
        graph, module_map, node_outputs = create_diamond_graph()
        record = ForwardRecord(
            graph=graph,
            node_inputs={nid: {} for nid in graph.nodes},
            node_outputs=node_outputs,
            module_map=module_map,
        )
        ctx = build_ancestor_context("a", record)
        # 'a' has ancestor 'input'
        assert "input" in ctx.ancestor_values
        # 'a' has sibling 'b'
        assert "b" in ctx.sibling_values


class TestAncestorContextImport:
    """Tests for AncestorContext package exports."""

    def test_import_from_optimization(self) -> None:
        """AncestorContext is exported from optimization package."""
        from plait.optimization import AncestorContext, build_ancestor_context

        ctx = AncestorContext()
        assert ctx is not None
        assert build_ancestor_context is not None
