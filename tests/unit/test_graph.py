"""Unit tests for the GraphNode and InferenceGraph data structures.

This module tests graph construction, topological ordering, and visualization.
Tests are consolidated to reduce redundancy while maintaining coverage.
"""

import pytest

from plait.graph import GraphNode, InferenceGraph, NodeRef, visualize_graph
from plait.module import LLMInference, Module
from plait.parameter import Parameter


class TestNodeRef:
    """Tests for NodeRef type."""

    def test_creation_and_repr(self) -> None:
        """NodeRef stores node_id and has readable repr."""
        ref = NodeRef("input:text")
        assert ref.node_id == "input:text"
        assert repr(ref) == "NodeRef(input:text)"

    def test_equality_and_hashing(self) -> None:
        """NodeRefs with same node_id are equal and hashable."""
        ref1 = NodeRef("node_1")
        ref2 = NodeRef("node_1")
        ref3 = NodeRef("node_2")

        assert ref1 == ref2
        assert ref1 != ref3
        assert ref1 != "node_1"  # Distinguishes from strings

        # Hashable for use in sets and dicts
        ref_set = {ref1, ref2, ref3}
        assert len(ref_set) == 2

        ref_dict = {ref1: "value1"}
        assert ref_dict[ref2] == "value1"

    def test_immutability(self) -> None:
        """NodeRef is frozen (immutable)."""
        ref = NodeRef("test")
        with pytest.raises(AttributeError):
            ref.node_id = "modified"  # type: ignore


class TestGraphNode:
    """Tests for GraphNode creation and behavior."""

    def test_creation_with_all_fields(self) -> None:
        """GraphNode correctly stores all fields."""
        module = LLMInference(alias="test")
        node = GraphNode(
            id="LLMInference_1",
            module=module,
            args=("input:prompt",),
            kwargs={"temperature": 0.7},
            dependencies=["input:prompt"],
            priority=10,
            branch_condition="condition_1",
            branch_value=True,
            module_name="CustomName",
            module_path="root.layer1.llm",
        )

        assert node.id == "LLMInference_1"
        assert node.module is module
        assert node.args == ("input:prompt",)
        assert node.kwargs == {"temperature": 0.7}
        assert node.dependencies == ["input:prompt"]
        assert node.priority == 10
        assert node.branch_condition == "condition_1"
        assert node.branch_value is True
        assert node.module_name == "CustomName"
        assert node.module_path == "root.layer1.llm"

    @pytest.mark.parametrize(
        "field,default",
        [
            ("priority", 0),
            ("branch_condition", None),
            ("branch_value", None),
            ("module_name", ""),
            ("module_path", ""),
        ],
    )
    def test_defaults(self, field: str, default: object) -> None:
        """GraphNode has correct default values."""
        node = GraphNode(id="test", module=None, args=(), kwargs={}, dependencies=[])
        assert getattr(node, field) == default

    def test_module_name_auto_populated(self) -> None:
        """module_name is auto-populated from module's class name."""
        module = LLMInference(alias="test")
        node = GraphNode(id="test", module=module, args=(), kwargs={}, dependencies=[])
        assert node.module_name == "LLMInference"

        # Custom module class
        class MyCustomModule(Module):
            def forward(self, x: str) -> str:
                return x

        node2 = GraphNode(
            id="test2", module=MyCustomModule(), args=(), kwargs={}, dependencies=[]
        )
        assert node2.module_name == "MyCustomModule"

    def test_module_name_preserved_when_provided(self) -> None:
        """Explicitly provided module_name is not overwritten."""
        module = LLMInference(alias="test")
        node = GraphNode(
            id="test",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
            module_name="CustomModuleName",
        )
        assert node.module_name == "CustomModuleName"

    def test_equality(self) -> None:
        """GraphNodes with same fields are equal."""
        module = LLMInference(alias="test")
        node1 = GraphNode(
            id="test", module=module, args=("a",), kwargs={}, dependencies=["a"]
        )
        node2 = GraphNode(
            id="test", module=module, args=("a",), kwargs={}, dependencies=["a"]
        )
        node3 = GraphNode(
            id="other", module=module, args=(), kwargs={}, dependencies=[]
        )

        assert node1 == node2
        assert node1 != node3


class TestInferenceGraph:
    """Tests for InferenceGraph creation and access."""

    def test_creation_and_access(self) -> None:
        """InferenceGraph stores and provides access to nodes."""
        input_node = GraphNode(
            id="input:text",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
            module_name="Input(text)",
        )
        llm_node = GraphNode(
            id="LLMInference_1",
            module=LLMInference(alias="test"),
            args=("input:text",),
            kwargs={},
            dependencies=["input:text"],
        )
        graph = InferenceGraph(
            nodes={"input:text": input_node, "LLMInference_1": llm_node},
            input_ids=["input:text"],
            output_ids=["LLMInference_1"],
        )

        assert len(graph.nodes) == 2
        assert graph.input_ids == ["input:text"]
        assert graph.output_ids == ["LLMInference_1"]
        assert graph.nodes["input:text"] is input_node
        assert graph.parameters == {}  # default

    def test_with_parameters(self) -> None:
        """InferenceGraph can store parameters."""
        param1 = Parameter("value1", description="test")
        param2 = Parameter("value2", description="test")
        graph = InferenceGraph(
            nodes={},
            input_ids=[],
            output_ids=[],
            parameters={"param1": param1, "param2": param2},
        )

        assert len(graph.parameters) == 2
        assert graph.parameters["param1"] is param1

    def test_multiple_inputs_outputs(self) -> None:
        """InferenceGraph can have multiple inputs and outputs."""
        node1 = GraphNode(
            id="input:a", module=None, args=(), kwargs={}, dependencies=[]
        )
        node2 = GraphNode(
            id="input:b", module=None, args=(), kwargs={}, dependencies=[]
        )
        out1 = GraphNode(
            id="output_1",
            module=None,
            args=("input:a",),
            kwargs={},
            dependencies=["input:a"],
        )
        out2 = GraphNode(
            id="output_2",
            module=None,
            args=("input:b",),
            kwargs={},
            dependencies=["input:b"],
        )
        graph = InferenceGraph(
            nodes={
                "input:a": node1,
                "input:b": node2,
                "output_1": out1,
                "output_2": out2,
            },
            input_ids=["input:a", "input:b"],
            output_ids=["output_1", "output_2"],
        )

        assert set(graph.input_ids) == {"input:a", "input:b"}
        assert set(graph.output_ids) == {"output_1", "output_2"}

    def test_equality(self) -> None:
        """InferenceGraphs with same fields are equal."""
        node = GraphNode(id="n", module=None, args=(), kwargs={}, dependencies=[])
        graph1 = InferenceGraph(nodes={"n": node}, input_ids=["n"], output_ids=["n"])
        graph2 = InferenceGraph(nodes={"n": node}, input_ids=["n"], output_ids=["n"])

        assert graph1 == graph2


class TestTopologicalOrder:
    """Tests for InferenceGraph.topological_order() method."""

    def test_linear_graph(self) -> None:
        """Topological order of a linear graph (A -> B -> C)."""
        node_a = GraphNode(id="a", module=None, args=(), kwargs={}, dependencies=[])
        node_b = GraphNode(
            id="b", module=None, args=("a",), kwargs={}, dependencies=["a"]
        )
        node_c = GraphNode(
            id="c", module=None, args=("b",), kwargs={}, dependencies=["b"]
        )
        graph = InferenceGraph(
            nodes={"a": node_a, "b": node_b, "c": node_c},
            input_ids=["a"],
            output_ids=["c"],
        )

        order = graph.topological_order()
        assert order == ["a", "b", "c"]

    def test_diamond_graph(self) -> None:
        """Topological order of a diamond graph (A -> [B, C] -> D)."""
        node_a = GraphNode(id="a", module=None, args=(), kwargs={}, dependencies=[])
        node_b = GraphNode(
            id="b", module=None, args=("a",), kwargs={}, dependencies=["a"]
        )
        node_c = GraphNode(
            id="c", module=None, args=("a",), kwargs={}, dependencies=["a"]
        )
        node_d = GraphNode(
            id="d", module=None, args=("b", "c"), kwargs={}, dependencies=["b", "c"]
        )
        graph = InferenceGraph(
            nodes={"a": node_a, "b": node_b, "c": node_c, "d": node_d},
            input_ids=["a"],
            output_ids=["d"],
        )

        order = graph.topological_order()

        # A must come before B and C; B and C must come before D
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_empty_graph(self) -> None:
        """Topological order of an empty graph is empty."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        assert graph.topological_order() == []

    @pytest.mark.parametrize(
        "cycle_type,nodes_config",
        [
            (
                "self",
                {"a": ([], ["a"])},  # self-referencing
            ),
            (
                "two-node",
                {"a": ([], ["b"]), "b": ([], ["a"])},
            ),
            (
                "three-node",
                {"a": ([], ["c"]), "b": ([], ["a"]), "c": ([], ["b"])},
            ),
        ],
    )
    def test_cycle_detection(
        self, cycle_type: str, nodes_config: dict[str, tuple[list[str], list[str]]]
    ) -> None:
        """Cycles are detected and raise ValueError."""
        nodes = {
            node_id: GraphNode(
                id=node_id,
                module=None,
                args=(),
                kwargs={},
                dependencies=deps,
            )
            for node_id, (_, deps) in nodes_config.items()
        }
        graph = InferenceGraph(
            nodes=nodes,
            input_ids=[],
            output_ids=[list(nodes.keys())[0]],
        )

        with pytest.raises(ValueError, match="Cycle detected"):
            graph.topological_order()


class TestAncestorsDescendants:
    """Tests for InferenceGraph.ancestors() and descendants() methods."""

    @pytest.fixture
    def linear_graph(self) -> InferenceGraph:
        """Create a linear graph: a -> b -> c."""
        a = GraphNode(id="a", module=None, args=(), kwargs={}, dependencies=[])
        b = GraphNode(id="b", module=None, args=("a",), kwargs={}, dependencies=["a"])
        c = GraphNode(id="c", module=None, args=("b",), kwargs={}, dependencies=["b"])
        return InferenceGraph(
            nodes={"a": a, "b": b, "c": c},
            input_ids=["a"],
            output_ids=["c"],
        )

    @pytest.fixture
    def diamond_graph(self) -> InferenceGraph:
        """Create a diamond graph: a -> [b, c] -> d."""
        a = GraphNode(id="a", module=None, args=(), kwargs={}, dependencies=[])
        b = GraphNode(id="b", module=None, args=("a",), kwargs={}, dependencies=["a"])
        c = GraphNode(id="c", module=None, args=("a",), kwargs={}, dependencies=["a"])
        d = GraphNode(
            id="d", module=None, args=("b", "c"), kwargs={}, dependencies=["b", "c"]
        )
        return InferenceGraph(
            nodes={"a": a, "b": b, "c": c, "d": d},
            input_ids=["a"],
            output_ids=["d"],
        )

    def test_ancestors_linear(self, linear_graph: InferenceGraph) -> None:
        """Test ancestors in linear graph."""
        assert linear_graph.ancestors("a") == set()
        assert linear_graph.ancestors("b") == {"a"}
        assert linear_graph.ancestors("c") == {"a", "b"}

    def test_ancestors_diamond(self, diamond_graph: InferenceGraph) -> None:
        """Test ancestors in diamond graph."""
        assert diamond_graph.ancestors("d") == {"a", "b", "c"}
        assert diamond_graph.ancestors("b") == {"a"}

    def test_descendants_linear(self, linear_graph: InferenceGraph) -> None:
        """Test descendants in linear graph."""
        assert linear_graph.descendants("a") == {"b", "c"}
        assert linear_graph.descendants("b") == {"c"}
        assert linear_graph.descendants("c") == set()

    def test_descendants_diamond(self, diamond_graph: InferenceGraph) -> None:
        """Test descendants in diamond graph."""
        assert diamond_graph.descendants("a") == {"b", "c", "d"}
        assert diamond_graph.descendants("b") == {"d"}
        assert diamond_graph.descendants("d") == set()


class TestVisualizeGraph:
    """Tests for visualize_graph() function generating DOT format."""

    def test_empty_graph(self) -> None:
        """visualize_graph handles empty graph."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        dot = visualize_graph(graph)

        assert "digraph InferenceGraph" in dot
        assert "rankdir=TB" in dot
        assert dot.endswith("}")

    def test_node_shapes(self) -> None:
        """visualize_graph uses correct shapes for node types."""
        input_node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
            module_name="Input(x)",
        )
        middle_node = GraphNode(
            id="middle",
            module=None,
            args=(),
            kwargs={},
            dependencies=["input:x"],
            module_name="Middle",
        )
        output_node = GraphNode(
            id="output",
            module=None,
            args=(),
            kwargs={},
            dependencies=["middle"],
            module_name="Output",
        )
        graph = InferenceGraph(
            nodes={"input:x": input_node, "middle": middle_node, "output": output_node},
            input_ids=["input:x"],
            output_ids=["output"],
        )
        dot = visualize_graph(graph)

        assert '"input:x" [label="Input(x)", shape=box]' in dot  # input
        assert '"middle" [label="Middle", shape=ellipse]' in dot  # intermediate
        assert '"output" [label="Output", shape=doubleoctagon]' in dot  # output

    def test_edges(self) -> None:
        """visualize_graph renders edges for dependencies."""
        input_node = GraphNode(
            id="input", module=None, args=(), kwargs={}, dependencies=[]
        )
        branch_a = GraphNode(
            id="branch_a",
            module=None,
            args=(),
            kwargs={},
            dependencies=["input"],
        )
        branch_b = GraphNode(
            id="branch_b",
            module=None,
            args=(),
            kwargs={},
            dependencies=["input"],
        )
        merge = GraphNode(
            id="merge",
            module=None,
            args=(),
            kwargs={},
            dependencies=["branch_a", "branch_b"],
        )
        graph = InferenceGraph(
            nodes={
                "input": input_node,
                "branch_a": branch_a,
                "branch_b": branch_b,
                "merge": merge,
            },
            input_ids=["input"],
            output_ids=["merge"],
        )
        dot = visualize_graph(graph)

        assert '"input" -> "branch_a"' in dot
        assert '"input" -> "branch_b"' in dot
        assert '"branch_a" -> "merge"' in dot
        assert '"branch_b" -> "merge"' in dot

    def test_branch_condition_in_label(self) -> None:
        """visualize_graph shows branch condition in label."""
        node = GraphNode(
            id="conditional",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
            module_name="Conditional",
            branch_condition="condition_1",
            branch_value=True,
        )
        graph = InferenceGraph(
            nodes={"conditional": node},
            input_ids=["conditional"],
            output_ids=["conditional"],
        )
        dot = visualize_graph(graph)

        assert r"Conditional\n[True]" in dot

    def test_uses_node_id_when_no_module_name(self) -> None:
        """visualize_graph uses node_id as label when module_name is empty."""
        node = GraphNode(
            id="unnamed_node",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
            module_name="",
        )
        graph = InferenceGraph(
            nodes={"unnamed_node": node},
            input_ids=["unnamed_node"],
            output_ids=["unnamed_node"],
        )
        dot = visualize_graph(graph)

        assert 'label="unnamed_node"' in dot


class TestComputeHash:
    """Tests for InferenceGraph.compute_hash() method."""

    def test_basic_properties(self) -> None:
        """compute_hash returns a valid SHA256 hex string."""
        node = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
            module_name="InputNode",
        )
        graph = InferenceGraph(
            nodes={"input:x": node}, input_ids=["input:x"], output_ids=["input:x"]
        )

        hash_value = graph.compute_hash()
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64
        assert graph.compute_hash() == hash_value  # deterministic

    def test_different_structure_different_hash(self) -> None:
        """Different graph structures produce different hashes."""
        node1 = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
            module_name="InputNode",
        )
        graph1 = InferenceGraph(
            nodes={"input:x": node1}, input_ids=["input:x"], output_ids=["input:x"]
        )

        node2 = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
            module_name="InputNode",
        )
        llm2 = GraphNode(
            id="llm:1",
            module=None,
            args=(),
            kwargs={},
            dependencies=["input:x"],
            module_name="LLMInference",
        )
        graph2 = InferenceGraph(
            nodes={"input:x": node2, "llm:1": llm2},
            input_ids=["input:x"],
            output_ids=["llm:1"],
        )

        assert graph1.compute_hash() != graph2.compute_hash()

    def test_independent_of_node_ids(self) -> None:
        """Hash is independent of node ID naming."""
        llm1 = LLMInference(alias="fast")
        input1 = GraphNode(
            id="input:x",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
            module_name="InputNode",
        )
        llm_node1 = GraphNode(
            id="LLM_1", module=llm1, args=(), kwargs={}, dependencies=["input:x"]
        )
        graph1 = InferenceGraph(
            nodes={"input:x": input1, "LLM_1": llm_node1},
            input_ids=["input:x"],
            output_ids=["LLM_1"],
        )

        llm2 = LLMInference(alias="fast")
        input2 = GraphNode(
            id="input_0",  # Different ID
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
            module_name="InputNode",
        )
        llm_node2 = GraphNode(
            id="LLMInference_0",  # Different ID
            module=llm2,
            args=(),
            kwargs={},
            dependencies=["input_0"],
        )
        graph2 = InferenceGraph(
            nodes={"input_0": input2, "LLMInference_0": llm_node2},
            input_ids=["input_0"],
            output_ids=["LLMInference_0"],
        )

        assert graph1.compute_hash() == graph2.compute_hash()

    @pytest.mark.parametrize(
        "config1,config2,should_match",
        [
            (
                {"alias": "fast", "temperature": 0.5},
                {"alias": "fast", "temperature": 0.7},
                False,
            ),
            ({"alias": "fast"}, {"alias": "slow"}, False),
            (
                {"alias": "fast", "system_prompt": "A"},
                {"alias": "fast", "system_prompt": "B"},
                False,
            ),
            (
                {"alias": "fast", "system_prompt": "A"},
                {"alias": "fast", "system_prompt": "A"},
                True,
            ),
        ],
    )
    def test_hash_sensitivity_to_config(
        self,
        config1: dict[str, object],
        config2: dict[str, object],
        should_match: bool,
    ) -> None:
        """Hash correctly reflects module configuration differences."""

        def make_graph(config: dict[str, object]) -> InferenceGraph:
            llm = LLMInference(**config)  # type: ignore[arg-type]
            input_node = GraphNode(
                id="input:x",
                module=None,
                args=(),
                kwargs={},
                dependencies=[],
                module_name="InputNode",
            )
            llm_node = GraphNode(
                id="LLM_1", module=llm, args=(), kwargs={}, dependencies=["input:x"]
            )
            return InferenceGraph(
                nodes={"input:x": input_node, "LLM_1": llm_node},
                input_ids=["input:x"],
                output_ids=["LLM_1"],
            )

        graph1 = make_graph(config1)
        graph2 = make_graph(config2)

        if should_match:
            assert graph1.compute_hash() == graph2.compute_hash()
        else:
            assert graph1.compute_hash() != graph2.compute_hash()
