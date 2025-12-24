"""Unit tests for the GraphNode and InferenceGraph data structures."""

from inf_engine.graph import GraphNode, InferenceGraph
from inf_engine.module import InferenceModule, LLMInference
from inf_engine.parameter import Parameter


class TestGraphNodeCreation:
    """Tests for GraphNode instantiation."""

    def test_graph_node_creation_basic(self) -> None:
        """GraphNode can be created with required fields."""
        module = LLMInference(alias="test")
        node = GraphNode(
            id="LLMInference_1",
            module=module,
            args=("input:prompt",),
            kwargs={},
            dependencies=["input:prompt"],
        )

        assert node.id == "LLMInference_1"
        assert node.module is module
        assert node.args == ("input:prompt",)
        assert node.kwargs == {}
        assert node.dependencies == ["input:prompt"]

    def test_graph_node_creation_with_all_fields(self) -> None:
        """GraphNode can be created with all optional fields."""
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

        assert node.priority == 10
        assert node.branch_condition == "condition_1"
        assert node.branch_value is True
        assert node.module_name == "CustomName"
        assert node.module_path == "root.layer1.llm"

    def test_graph_node_defaults(self) -> None:
        """GraphNode has correct default values."""
        node = GraphNode(
            id="test",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )

        assert node.priority == 0
        assert node.branch_condition is None
        assert node.branch_value is None
        assert node.module_name == ""
        assert node.module_path == ""

    def test_graph_node_with_none_module(self) -> None:
        """GraphNode can have None module (for input nodes)."""
        node = GraphNode(
            id="input:text",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
            module_name="Input(text)",
        )

        assert node.module is None
        assert node.module_name == "Input(text)"

    def test_graph_node_args_as_tuple(self) -> None:
        """GraphNode args can contain mixed types (node IDs and literals)."""
        node = GraphNode(
            id="test",
            module=None,
            args=("input:text", "literal_value", 42),
            kwargs={},
            dependencies=["input:text"],
        )

        assert node.args == ("input:text", "literal_value", 42)

    def test_graph_node_kwargs_mixed_values(self) -> None:
        """GraphNode kwargs can contain mixed types (node IDs and literals)."""
        node = GraphNode(
            id="test",
            module=None,
            args=(),
            kwargs={"prompt_ref": "input:text", "temperature": 0.5, "count": 10},
            dependencies=["input:text"],
        )

        assert node.kwargs == {
            "prompt_ref": "input:text",
            "temperature": 0.5,
            "count": 10,
        }

    def test_graph_node_multiple_dependencies(self) -> None:
        """GraphNode can have multiple dependencies."""
        node = GraphNode(
            id="aggregator",
            module=None,
            args=("result_1", "result_2", "result_3"),
            kwargs={},
            dependencies=["result_1", "result_2", "result_3"],
        )

        assert len(node.dependencies) == 3
        assert node.dependencies == ["result_1", "result_2", "result_3"]


class TestGraphNodePostInit:
    """Tests for GraphNode.__post_init__() behavior."""

    def test_module_name_auto_populated_from_module(self) -> None:
        """module_name is auto-populated from module's class name if empty."""
        module = LLMInference(alias="test")
        node = GraphNode(
            id="test",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
            # module_name not provided
        )

        assert node.module_name == "LLMInference"

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

    def test_module_name_not_set_when_module_is_none(self) -> None:
        """module_name stays empty when module is None and not provided."""
        node = GraphNode(
            id="test",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )

        assert node.module_name == ""

    def test_module_name_with_custom_module_class(self) -> None:
        """module_name correctly reflects custom module class names."""

        class MyCustomModule(InferenceModule):
            def forward(self, x: str) -> str:
                return x

        module = MyCustomModule()
        node = GraphNode(
            id="test",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
        )

        assert node.module_name == "MyCustomModule"


class TestGraphNodeEquality:
    """Tests for GraphNode equality comparison."""

    def test_nodes_with_same_fields_are_equal(self) -> None:
        """GraphNodes with identical fields are equal."""
        module = LLMInference(alias="test")
        node1 = GraphNode(
            id="test",
            module=module,
            args=("a",),
            kwargs={"k": "v"},
            dependencies=["a"],
        )
        node2 = GraphNode(
            id="test",
            module=module,
            args=("a",),
            kwargs={"k": "v"},
            dependencies=["a"],
        )

        assert node1 == node2

    def test_nodes_with_different_ids_are_not_equal(self) -> None:
        """GraphNodes with different IDs are not equal."""
        module = LLMInference(alias="test")
        node1 = GraphNode(
            id="node1", module=module, args=(), kwargs={}, dependencies=[]
        )
        node2 = GraphNode(
            id="node2", module=module, args=(), kwargs={}, dependencies=[]
        )

        assert node1 != node2


class TestInferenceGraphCreation:
    """Tests for InferenceGraph instantiation."""

    def test_graph_creation_basic(self) -> None:
        """InferenceGraph can be created with required fields."""
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

    def test_graph_defaults(self) -> None:
        """InferenceGraph has correct default values."""
        graph = InferenceGraph(
            nodes={},
            input_ids=[],
            output_ids=[],
        )

        assert graph.parameters == {}

    def test_graph_with_parameters(self) -> None:
        """InferenceGraph can store parameters."""
        param1 = Parameter("value1")
        param2 = Parameter("value2")

        graph = InferenceGraph(
            nodes={},
            input_ids=[],
            output_ids=[],
            parameters={"param1": param1, "param2": param2},
        )

        assert len(graph.parameters) == 2
        assert graph.parameters["param1"] is param1
        assert graph.parameters["param2"] is param2

    def test_graph_access_nodes_by_id(self) -> None:
        """Graph nodes can be accessed by ID."""
        node = GraphNode(
            id="my_node",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )

        graph = InferenceGraph(
            nodes={"my_node": node},
            input_ids=["my_node"],
            output_ids=["my_node"],
        )

        assert graph.nodes["my_node"] is node

    def test_graph_multiple_inputs(self) -> None:
        """InferenceGraph can have multiple input nodes."""
        node1 = GraphNode(
            id="input:a", module=None, args=(), kwargs={}, dependencies=[]
        )
        node2 = GraphNode(
            id="input:b", module=None, args=(), kwargs={}, dependencies=[]
        )
        node3 = GraphNode(
            id="combine",
            module=None,
            args=("input:a", "input:b"),
            kwargs={},
            dependencies=["input:a", "input:b"],
        )

        graph = InferenceGraph(
            nodes={"input:a": node1, "input:b": node2, "combine": node3},
            input_ids=["input:a", "input:b"],
            output_ids=["combine"],
        )

        assert len(graph.input_ids) == 2
        assert "input:a" in graph.input_ids
        assert "input:b" in graph.input_ids

    def test_graph_multiple_outputs(self) -> None:
        """InferenceGraph can have multiple output nodes."""
        input_node = GraphNode(
            id="input:x", module=None, args=(), kwargs={}, dependencies=[]
        )
        out1 = GraphNode(
            id="output_1",
            module=None,
            args=("input:x",),
            kwargs={},
            dependencies=["input:x"],
        )
        out2 = GraphNode(
            id="output_2",
            module=None,
            args=("input:x",),
            kwargs={},
            dependencies=["input:x"],
        )

        graph = InferenceGraph(
            nodes={"input:x": input_node, "output_1": out1, "output_2": out2},
            input_ids=["input:x"],
            output_ids=["output_1", "output_2"],
        )

        assert len(graph.output_ids) == 2
        assert "output_1" in graph.output_ids
        assert "output_2" in graph.output_ids


class TestInferenceGraphEquality:
    """Tests for InferenceGraph equality comparison."""

    def test_graphs_with_same_fields_are_equal(self) -> None:
        """InferenceGraphs with identical fields are equal."""
        node = GraphNode(id="n", module=None, args=(), kwargs={}, dependencies=[])

        graph1 = InferenceGraph(
            nodes={"n": node},
            input_ids=["n"],
            output_ids=["n"],
        )
        graph2 = InferenceGraph(
            nodes={"n": node},
            input_ids=["n"],
            output_ids=["n"],
        )

        assert graph1 == graph2

    def test_graphs_with_different_nodes_are_not_equal(self) -> None:
        """InferenceGraphs with different nodes are not equal."""
        node1 = GraphNode(id="a", module=None, args=(), kwargs={}, dependencies=[])
        node2 = GraphNode(id="b", module=None, args=(), kwargs={}, dependencies=[])

        graph1 = InferenceGraph(nodes={"a": node1}, input_ids=["a"], output_ids=["a"])
        graph2 = InferenceGraph(nodes={"b": node2}, input_ids=["b"], output_ids=["b"])

        assert graph1 != graph2


class TestInferenceGraphEmptyGraph:
    """Tests for edge cases with empty graphs."""

    def test_empty_graph(self) -> None:
        """An empty graph is valid."""
        graph = InferenceGraph(
            nodes={},
            input_ids=[],
            output_ids=[],
        )

        assert len(graph.nodes) == 0
        assert len(graph.input_ids) == 0
        assert len(graph.output_ids) == 0


class TestGraphNodeBranching:
    """Tests for branch-related fields on GraphNode."""

    def test_branch_condition_and_value(self) -> None:
        """GraphNode correctly stores branch metadata."""
        node = GraphNode(
            id="conditional_call",
            module=LLMInference(alias="test"),
            args=(),
            kwargs={},
            dependencies=["condition_node"],
            branch_condition="condition_node",
            branch_value=True,
        )

        assert node.branch_condition == "condition_node"
        assert node.branch_value is True

    def test_false_branch_value(self) -> None:
        """GraphNode can represent false branch."""
        node = GraphNode(
            id="false_branch",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
            branch_condition="cond",
            branch_value=False,
        )

        assert node.branch_value is False


class TestInferenceGraphTopologicalOrder:
    """Tests for InferenceGraph.topological_order() method."""

    def test_single_node_graph(self) -> None:
        """Topological order of a single node graph."""
        node = GraphNode(
            id="only_node",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        graph = InferenceGraph(
            nodes={"only_node": node},
            input_ids=["only_node"],
            output_ids=["only_node"],
        )

        order = graph.topological_order()

        assert order == ["only_node"]

    def test_linear_graph(self) -> None:
        """Topological order of a linear graph (A -> B -> C)."""
        node_a = GraphNode(
            id="input:a",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        node_b = GraphNode(
            id="b",
            module=None,
            args=("input:a",),
            kwargs={},
            dependencies=["input:a"],
        )
        node_c = GraphNode(
            id="c",
            module=None,
            args=("b",),
            kwargs={},
            dependencies=["b"],
        )
        graph = InferenceGraph(
            nodes={"input:a": node_a, "b": node_b, "c": node_c},
            input_ids=["input:a"],
            output_ids=["c"],
        )

        order = graph.topological_order()

        # A must come before B, B must come before C
        assert order == ["input:a", "b", "c"]

    def test_diamond_graph(self) -> None:
        """Topological order of a diamond graph (A -> [B, C] -> D)."""
        node_a = GraphNode(
            id="input:a",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        node_b = GraphNode(
            id="b",
            module=None,
            args=("input:a",),
            kwargs={},
            dependencies=["input:a"],
        )
        node_c = GraphNode(
            id="c",
            module=None,
            args=("input:a",),
            kwargs={},
            dependencies=["input:a"],
        )
        node_d = GraphNode(
            id="d",
            module=None,
            args=("b", "c"),
            kwargs={},
            dependencies=["b", "c"],
        )
        graph = InferenceGraph(
            nodes={"input:a": node_a, "b": node_b, "c": node_c, "d": node_d},
            input_ids=["input:a"],
            output_ids=["d"],
        )

        order = graph.topological_order()

        # A must come before B and C; B and C must come before D
        assert order.index("input:a") < order.index("b")
        assert order.index("input:a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")
        assert len(order) == 4

    def test_complex_graph(self) -> None:
        """Topological order of a complex graph with multiple paths.

        Graph structure:
            input1 ─┬─> llm1 ──┬─> llm3 ─┬─> output
            input2 ─┴─> llm2 ──┘         │
                                  llm4 ──┘
        """
        input1 = GraphNode(
            id="input:1", module=None, args=(), kwargs={}, dependencies=[]
        )
        input2 = GraphNode(
            id="input:2", module=None, args=(), kwargs={}, dependencies=[]
        )
        llm1 = GraphNode(
            id="llm1",
            module=None,
            args=("input:1", "input:2"),
            kwargs={},
            dependencies=["input:1", "input:2"],
        )
        llm2 = GraphNode(
            id="llm2",
            module=None,
            args=("input:1", "input:2"),
            kwargs={},
            dependencies=["input:1", "input:2"],
        )
        llm3 = GraphNode(
            id="llm3",
            module=None,
            args=("llm1", "llm2"),
            kwargs={},
            dependencies=["llm1", "llm2"],
        )
        llm4 = GraphNode(
            id="llm4",
            module=None,
            args=(),
            kwargs={},
            dependencies=[],
        )
        output = GraphNode(
            id="output",
            module=None,
            args=("llm3", "llm4"),
            kwargs={},
            dependencies=["llm3", "llm4"],
        )
        graph = InferenceGraph(
            nodes={
                "input:1": input1,
                "input:2": input2,
                "llm1": llm1,
                "llm2": llm2,
                "llm3": llm3,
                "llm4": llm4,
                "output": output,
            },
            input_ids=["input:1", "input:2"],
            output_ids=["output"],
        )

        order = graph.topological_order()

        # Verify all dependencies are satisfied
        assert order.index("input:1") < order.index("llm1")
        assert order.index("input:2") < order.index("llm1")
        assert order.index("input:1") < order.index("llm2")
        assert order.index("input:2") < order.index("llm2")
        assert order.index("llm1") < order.index("llm3")
        assert order.index("llm2") < order.index("llm3")
        assert order.index("llm3") < order.index("output")
        assert order.index("llm4") < order.index("output")
        assert len(order) == 7

    def test_multiple_outputs(self) -> None:
        """Topological order includes all nodes reachable from outputs."""
        input_node = GraphNode(
            id="input", module=None, args=(), kwargs={}, dependencies=[]
        )
        branch1 = GraphNode(
            id="branch1",
            module=None,
            args=("input",),
            kwargs={},
            dependencies=["input"],
        )
        branch2 = GraphNode(
            id="branch2",
            module=None,
            args=("input",),
            kwargs={},
            dependencies=["input"],
        )
        graph = InferenceGraph(
            nodes={"input": input_node, "branch1": branch1, "branch2": branch2},
            input_ids=["input"],
            output_ids=["branch1", "branch2"],
        )

        order = graph.topological_order()

        # Input comes before both branches
        assert order.index("input") < order.index("branch1")
        assert order.index("input") < order.index("branch2")
        assert len(order) == 3

    def test_empty_graph(self) -> None:
        """Topological order of an empty graph is empty."""
        graph = InferenceGraph(
            nodes={},
            input_ids=[],
            output_ids=[],
        )

        order = graph.topological_order()

        assert order == []

    def test_parallel_independent_chains(self) -> None:
        """Topological order handles independent parallel chains.

        Graph structure:
            input1 -> a -> b ─┐
                              ├─> merge
            input2 -> c -> d ─┘
        """
        input1 = GraphNode(
            id="input:1", module=None, args=(), kwargs={}, dependencies=[]
        )
        input2 = GraphNode(
            id="input:2", module=None, args=(), kwargs={}, dependencies=[]
        )
        a = GraphNode(
            id="a", module=None, args=("input:1",), kwargs={}, dependencies=["input:1"]
        )
        b = GraphNode(id="b", module=None, args=("a",), kwargs={}, dependencies=["a"])
        c = GraphNode(
            id="c", module=None, args=("input:2",), kwargs={}, dependencies=["input:2"]
        )
        d = GraphNode(id="d", module=None, args=("c",), kwargs={}, dependencies=["c"])
        merge = GraphNode(
            id="merge",
            module=None,
            args=("b", "d"),
            kwargs={},
            dependencies=["b", "d"],
        )
        graph = InferenceGraph(
            nodes={
                "input:1": input1,
                "input:2": input2,
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "merge": merge,
            },
            input_ids=["input:1", "input:2"],
            output_ids=["merge"],
        )

        order = graph.topological_order()

        # Each chain must be in order
        assert order.index("input:1") < order.index("a")
        assert order.index("a") < order.index("b")
        assert order.index("input:2") < order.index("c")
        assert order.index("c") < order.index("d")
        assert order.index("b") < order.index("merge")
        assert order.index("d") < order.index("merge")
        assert len(order) == 7

    def test_dependencies_appear_before_dependents(self) -> None:
        """Every node appears after all its dependencies in the order."""
        # Create a graph where each node depends on all previous nodes
        nodes = {}
        prev_ids: list[str] = []
        for i in range(5):
            node_id = f"node_{i}"
            nodes[node_id] = GraphNode(
                id=node_id,
                module=None,
                args=tuple(prev_ids),
                kwargs={},
                dependencies=list(prev_ids),
            )
            prev_ids.append(node_id)

        graph = InferenceGraph(
            nodes=nodes,
            input_ids=["node_0"],
            output_ids=["node_4"],
        )

        order = graph.topological_order()

        # Verify the order is exactly what we expect
        assert order == ["node_0", "node_1", "node_2", "node_3", "node_4"]
