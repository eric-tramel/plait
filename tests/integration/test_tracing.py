"""Integration tests for the tracing system.

This file contains tests for PR-018: Tracing integration tests.
Tests verify full tracing scenarios including nested modules,
shared inputs, dict outputs, and complete tracing flows.
"""

from inf_engine.module import InferenceModule, LLMInference
from inf_engine.parameter import Parameter
from inf_engine.tracing.proxy import Proxy
from inf_engine.tracing.tracer import Tracer

# ─────────────────────────────────────────────────────────────────────────────
# Nested Module Tracing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNestedModuleTracing:
    """Tests for tracing nested module hierarchies."""

    def test_single_child_module_is_traced(self) -> None:
        """A module containing a child module is traced correctly."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = LLMInference(alias="test")

            def forward(self, text: str) -> Proxy:
                return self.child(text)

        tracer = Tracer()
        graph = tracer.trace(Parent(), "input text")

        # Should have: 1 input node + 1 LLMInference call
        assert len(graph.nodes) == 2
        assert "input:input_0" in graph.nodes
        assert "LLMInference_1" in graph.nodes

        # LLMInference should depend on the input
        llm_node = graph.nodes["LLMInference_1"]
        assert llm_node.dependencies == ["input:input_0"]

        # Output should be the LLMInference result
        assert graph.output_ids == ["LLMInference_1"]

    def test_two_level_nested_modules_are_traced(self) -> None:
        """Modules nested two levels deep are traced correctly.

        When a parent module calls a child via __call__, the child becomes
        an opaque node - its internal forward() is not executed during tracing.
        """

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="inner")

            def forward(self, text: str) -> Proxy:
                return self.llm(text)

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

            def forward(self, text: str) -> Proxy:
                # self.inner() goes through __call__ which records the call
                # and returns a Proxy, without executing Inner.forward()
                return self.inner(text)

        tracer = Tracer()
        graph = tracer.trace(Outer(), "input text")

        # Should have: 1 input + 1 Inner call (as opaque node)
        # The LLMInference inside Inner is not traced because
        # Inner.forward() is not called during tracing
        assert len(graph.nodes) == 2
        assert "input:input_0" in graph.nodes
        assert "Inner_1" in graph.nodes

        # Inner call depends on input
        assert graph.nodes["Inner_1"].dependencies == ["input:input_0"]
        assert graph.output_ids == ["Inner_1"]

    def test_three_level_nested_modules_are_traced(self) -> None:
        """Modules nested three levels deep are traced correctly.

        When modules call children via __call__, only the immediate child
        is captured as a node - deeper nesting becomes opaque.
        """

        class Level3(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="deep")

            def forward(self, text: str) -> Proxy:
                return self.llm(text)

        class Level2(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level3 = Level3()

            def forward(self, text: str) -> Proxy:
                return self.level3(text)

        class Level1(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level2 = Level2()

            def forward(self, text: str) -> Proxy:
                return self.level2(text)

        tracer = Tracer()
        graph = tracer.trace(Level1(), "input")

        # Level2 is the only captured node (Level3 and LLM are inside it)
        assert len(graph.nodes) == 2
        assert "Level2_1" in graph.nodes
        assert graph.output_ids == ["Level2_1"]

    def test_nested_module_parameters_are_collected(self) -> None:
        """Parameters from nested modules are collected in the graph."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="inner", system_prompt="Inner prompt")

            def forward(self, text: str) -> Proxy:
                return self.llm(text)

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prefix = Parameter("Outer prefix")
                self.inner = Inner()

            def forward(self, text: str) -> Proxy:
                return self.inner(text)

        tracer = Tracer()
        graph = tracer.trace(Outer(), "input")

        # Should collect parameters from both outer and inner modules
        assert "prefix" in graph.parameters
        assert graph.parameters["prefix"].value == "Outer prefix"
        assert "inner.llm.system_prompt" in graph.parameters
        assert graph.parameters["inner.llm.system_prompt"].value == "Inner prompt"

    def test_nested_module_with_multiple_children(self) -> None:
        """Module with multiple child modules at same level is traced."""

        class MultiChild(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.step1 = LLMInference(alias="step1")
                self.step2 = LLMInference(alias="step2")

            def forward(self, text: str) -> Proxy:
                result1 = self.step1(text)
                result2 = self.step2(result1)
                return result2

        tracer = Tracer()
        graph = tracer.trace(MultiChild(), "input")

        # Should have: 1 input + 2 LLM calls
        assert len(graph.nodes) == 3
        assert graph.nodes["LLMInference_1"].dependencies == ["input:input_0"]
        assert graph.nodes["LLMInference_2"].dependencies == ["LLMInference_1"]
        assert graph.output_ids == ["LLMInference_2"]


# ─────────────────────────────────────────────────────────────────────────────
# Shared Input Tracing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSharedInputTracing:
    """Tests for tracing modules with shared inputs (fan-out pattern)."""

    def test_multiple_modules_share_same_input(self) -> None:
        """Multiple modules depending on the same input are traced correctly."""

        class FanOut(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.branch_a = LLMInference(alias="a")
                self.branch_b = LLMInference(alias="b")
                self.branch_c = LLMInference(alias="c")

            def forward(self, text: str) -> dict[str, Proxy]:
                return {
                    "a": self.branch_a(text),
                    "b": self.branch_b(text),
                    "c": self.branch_c(text),
                }

        tracer = Tracer()
        graph = tracer.trace(FanOut(), "shared input")

        # Should have: 1 input + 3 LLM calls
        assert len(graph.nodes) == 4

        # All three branches depend on the same input
        assert graph.nodes["LLMInference_1"].dependencies == ["input:input_0"]
        assert graph.nodes["LLMInference_2"].dependencies == ["input:input_0"]
        assert graph.nodes["LLMInference_3"].dependencies == ["input:input_0"]

        # All three are outputs
        assert len(graph.output_ids) == 3
        assert "LLMInference_1" in graph.output_ids
        assert "LLMInference_2" in graph.output_ids
        assert "LLMInference_3" in graph.output_ids

    def test_shared_input_with_fan_in(self) -> None:
        """Fan-out followed by fan-in (diamond pattern) is traced correctly."""

        class Diamond(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.branch_a = LLMInference(alias="a")
                self.branch_b = LLMInference(alias="b")
                self.merger = LLMInference(alias="merge")

            def forward(self, text: str) -> Proxy:
                a_result = self.branch_a(text)
                b_result = self.branch_b(text)
                # In tracing, merger receives both proxy results
                return self.merger(a_result, b_result)

        tracer = Tracer()
        graph = tracer.trace(Diamond(), "input")

        # Should have: 1 input + 3 LLM calls
        assert len(graph.nodes) == 4

        # Both branches depend on input
        assert graph.nodes["LLMInference_1"].dependencies == ["input:input_0"]
        assert graph.nodes["LLMInference_2"].dependencies == ["input:input_0"]

        # Merger depends on both branches
        merge_deps = graph.nodes["LLMInference_3"].dependencies
        assert "LLMInference_1" in merge_deps
        assert "LLMInference_2" in merge_deps

        # Only the merger is the output
        assert graph.output_ids == ["LLMInference_3"]

    def test_shared_intermediate_result(self) -> None:
        """Shared intermediate results are traced with correct dependencies."""

        class SharedIntermediate(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.preprocess = LLMInference(alias="pre")
                self.analyze_a = LLMInference(alias="a")
                self.analyze_b = LLMInference(alias="b")

            def forward(self, text: str) -> dict[str, Proxy]:
                # Preprocess once, then use result in two branches
                preprocessed = self.preprocess(text)
                return {
                    "a": self.analyze_a(preprocessed),
                    "b": self.analyze_b(preprocessed),
                }

        tracer = Tracer()
        graph = tracer.trace(SharedIntermediate(), "input")

        # Should have: 1 input + 3 LLM calls
        assert len(graph.nodes) == 4

        # Preprocess depends on input
        assert graph.nodes["LLMInference_1"].dependencies == ["input:input_0"]

        # Both analyzers depend on preprocessed result
        assert graph.nodes["LLMInference_2"].dependencies == ["LLMInference_1"]
        assert graph.nodes["LLMInference_3"].dependencies == ["LLMInference_1"]

        # Both analyzer outputs are in output_ids
        assert len(graph.output_ids) == 2

    def test_multiple_inputs_shared_across_modules(self) -> None:
        """Multiple inputs can be shared across different modules."""

        class MultiInput(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.uses_first = LLMInference(alias="first")
                self.uses_second = LLMInference(alias="second")
                self.uses_both = LLMInference(alias="both")

            def forward(self, a: str, b: str) -> dict[str, Proxy]:
                return {
                    "first_only": self.uses_first(a),
                    "second_only": self.uses_second(b),
                    "combined": self.uses_both(a, b),
                }

        tracer = Tracer()
        graph = tracer.trace(MultiInput(), "first input", "second input")

        # Should have: 2 inputs + 3 LLM calls
        assert len(graph.nodes) == 5
        assert "input:input_0" in graph.nodes
        assert "input:input_1" in graph.nodes

        # Check dependencies
        assert graph.nodes["LLMInference_1"].dependencies == ["input:input_0"]
        assert graph.nodes["LLMInference_2"].dependencies == ["input:input_1"]

        uses_both_deps = graph.nodes["LLMInference_3"].dependencies
        assert "input:input_0" in uses_both_deps
        assert "input:input_1" in uses_both_deps


# ─────────────────────────────────────────────────────────────────────────────
# Dict Output Tracing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDictOutputTracing:
    """Tests for tracing modules with dictionary outputs."""

    def test_simple_dict_output(self) -> None:
        """Module returning a simple dict is traced correctly."""

        class DictOutput(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm_a = LLMInference(alias="a")
                self.llm_b = LLMInference(alias="b")

            def forward(self, text: str) -> dict[str, Proxy]:
                return {
                    "result_a": self.llm_a(text),
                    "result_b": self.llm_b(text),
                }

        tracer = Tracer()
        graph = tracer.trace(DictOutput(), "input")

        # Both LLM outputs should be in output_ids
        assert len(graph.output_ids) == 2
        assert "LLMInference_1" in graph.output_ids
        assert "LLMInference_2" in graph.output_ids

    def test_nested_dict_output(self) -> None:
        """Module returning nested dict is traced correctly."""

        class NestedDictOutput(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm_a = LLMInference(alias="a")
                self.llm_b = LLMInference(alias="b")
                self.llm_c = LLMInference(alias="c")

            def forward(self, text: str) -> dict:
                return {
                    "category1": {
                        "a": self.llm_a(text),
                    },
                    "category2": {
                        "b": self.llm_b(text),
                        "c": self.llm_c(text),
                    },
                }

        tracer = Tracer()
        graph = tracer.trace(NestedDictOutput(), "input")

        # All three LLM outputs should be collected
        assert len(graph.output_ids) == 3
        assert "LLMInference_1" in graph.output_ids
        assert "LLMInference_2" in graph.output_ids
        assert "LLMInference_3" in graph.output_ids

    def test_dict_with_list_values(self) -> None:
        """Module returning dict with list values is traced correctly."""

        class DictWithLists(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm_a = LLMInference(alias="a")
                self.llm_b = LLMInference(alias="b")
                self.llm_c = LLMInference(alias="c")

            def forward(self, text: str) -> dict:
                return {
                    "single": self.llm_a(text),
                    "multiple": [self.llm_b(text), self.llm_c(text)],
                }

        tracer = Tracer()
        graph = tracer.trace(DictWithLists(), "input")

        # All outputs should be collected
        assert len(graph.output_ids) == 3

    def test_empty_dict_output(self) -> None:
        """Module returning empty dict has no outputs."""

        class EmptyDictOutput(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, text: str) -> dict:
                return {}

        tracer = Tracer()
        graph = tracer.trace(EmptyDictOutput(), "input")

        assert graph.output_ids == []

    def test_dict_output_with_mixed_values(self) -> None:
        """Dict with both Proxy and literal values is traced correctly."""

        class MixedDict(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="test")

            def forward(self, text: str) -> dict:
                return {
                    "result": self.llm(text),
                    "literal": "not a proxy",
                    "number": 42,
                }

        tracer = Tracer()
        graph = tracer.trace(MixedDict(), "input")

        # Only the Proxy should be in output_ids
        assert graph.output_ids == ["LLMInference_1"]


# ─────────────────────────────────────────────────────────────────────────────
# Complete Tracing Flow Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCompleteTracingFlow:
    """Tests for complete end-to-end tracing scenarios."""

    def test_linear_pipeline_trace(self) -> None:
        """Linear pipeline (summarize -> analyze -> format) is traced correctly."""

        class AnalysisPipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.summarize = LLMInference(alias="fast")
                self.analyze = LLMInference(alias="smart")
                self.format = LLMInference(alias="fast")

            def forward(self, text: str) -> Proxy:
                summary = self.summarize(text)
                analysis = self.analyze(summary)
                formatted = self.format(analysis)
                return formatted

        tracer = Tracer()
        graph = tracer.trace(AnalysisPipeline(), "Some long document...")

        # Verify graph structure
        assert len(graph.nodes) == 4  # 1 input + 3 LLM calls
        assert graph.input_ids == ["input:input_0"]
        assert graph.output_ids == ["LLMInference_3"]

        # Verify topological order
        order = graph.topological_order()
        assert order.index("input:input_0") < order.index("LLMInference_1")
        assert order.index("LLMInference_1") < order.index("LLMInference_2")
        assert order.index("LLMInference_2") < order.index("LLMInference_3")

    def test_complex_pipeline_with_multiple_inputs(self) -> None:
        """Complex pipeline with multiple inputs and shared processing."""

        class ComplexPipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.encode_text = LLMInference(alias="encoder")
                self.encode_context = LLMInference(alias="encoder")
                self.combine = LLMInference(alias="combiner")
                self.analyze = LLMInference(alias="analyzer")

            def forward(self, text: str, context: str) -> Proxy:
                encoded_text = self.encode_text(text)
                encoded_context = self.encode_context(context)
                combined = self.combine(encoded_text, encoded_context)
                result = self.analyze(combined)
                return result

        tracer = Tracer()
        graph = tracer.trace(ComplexPipeline(), "main text", "context info")

        # Verify structure
        assert len(graph.nodes) == 6  # 2 inputs + 4 LLM calls
        assert len(graph.input_ids) == 2

        # Verify dependencies
        assert graph.nodes["LLMInference_1"].dependencies == ["input:input_0"]
        assert graph.nodes["LLMInference_2"].dependencies == ["input:input_1"]

        combine_deps = graph.nodes["LLMInference_3"].dependencies
        assert "LLMInference_1" in combine_deps
        assert "LLMInference_2" in combine_deps

        assert graph.nodes["LLMInference_4"].dependencies == ["LLMInference_3"]

    def test_graph_ancestors_and_descendants(self) -> None:
        """Graph traversal methods work correctly on traced graph."""

        class Pipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.step1 = LLMInference(alias="1")
                self.step2 = LLMInference(alias="2")
                self.step3 = LLMInference(alias="3")

            def forward(self, text: str) -> Proxy:
                r1 = self.step1(text)
                r2 = self.step2(r1)
                r3 = self.step3(r2)
                return r3

        tracer = Tracer()
        graph = tracer.trace(Pipeline(), "input")

        # Verify ancestors
        assert graph.ancestors("LLMInference_1") == {"input:input_0"}
        assert graph.ancestors("LLMInference_2") == {"input:input_0", "LLMInference_1"}
        assert graph.ancestors("LLMInference_3") == {
            "input:input_0",
            "LLMInference_1",
            "LLMInference_2",
        }

        # Verify descendants
        assert graph.descendants("input:input_0") == {
            "LLMInference_1",
            "LLMInference_2",
            "LLMInference_3",
        }
        assert graph.descendants("LLMInference_1") == {
            "LLMInference_2",
            "LLMInference_3",
        }
        assert graph.descendants("LLMInference_3") == set()

    def test_trace_preserves_module_references(self) -> None:
        """Traced nodes preserve references to original modules."""

        class Pipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(
                    alias="test", system_prompt="Test prompt", temperature=0.5
                )

            def forward(self, text: str) -> Proxy:
                return self.llm(text)

        pipeline = Pipeline()
        tracer = Tracer()
        graph = tracer.trace(pipeline, "input")

        # Get the node's module
        node = graph.nodes["LLMInference_1"]

        # Module should be the exact same instance
        assert node.module is pipeline.llm
        assert node.module.alias == "test"
        assert node.module.temperature == 0.5

    def test_trace_with_kwargs_input(self) -> None:
        """Tracing works correctly with keyword arguments."""

        class KwargModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="test")

            def forward(self, *, text: str, context: str) -> Proxy:
                return self.llm(text, context=context)

        tracer = Tracer()
        graph = tracer.trace(KwargModule(), text="hello", context="world")

        # Should have input nodes for kwargs
        assert "input:input_text" in graph.nodes
        assert "input:input_context" in graph.nodes
        assert "input:input_text" in graph.input_ids
        assert "input:input_context" in graph.input_ids

    def test_multiple_traces_are_independent(self) -> None:
        """Multiple traces from the same tracer are independent."""

        class Simple(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="test")

            def forward(self, text: str) -> Proxy:
                return self.llm(text)

        tracer = Tracer()

        graph1 = tracer.trace(Simple(), "first input")
        graph2 = tracer.trace(Simple(), "second input")

        # Graphs should be independent
        assert graph1 is not graph2
        assert graph1.nodes is not graph2.nodes

        # Input values should differ
        from inf_engine.tracing.tracer import InputNode

        input1 = graph1.nodes["input:input_0"].module
        input2 = graph2.nodes["input:input_0"].module
        assert isinstance(input1, InputNode)
        assert isinstance(input2, InputNode)
        assert input1.value == "first input"
        assert input2.value == "second input"


# ─────────────────────────────────────────────────────────────────────────────
# Best Practices Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTracingBestPractices:
    """Tests that verify best practices from tracing.md are properly supported."""

    def test_modules_defined_in_init_are_traced(self) -> None:
        """Modules created in __init__ are properly traced."""

        class ProperModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                # Good: modules created in __init__
                self.llm1 = LLMInference(alias="a")
                self.llm2 = LLMInference(alias="b")

            def forward(self, text: str) -> Proxy:
                r1 = self.llm1(text)
                return self.llm2(r1)

        tracer = Tracer()
        graph = tracer.trace(ProperModule(), "input")

        # Both modules should be traced
        assert len(graph.nodes) == 3  # 1 input + 2 LLM calls
        assert "LLMInference_1" in graph.nodes
        assert "LLMInference_2" in graph.nodes

    def test_pure_forward_produces_consistent_graph(self) -> None:
        """Pure forward() methods produce consistent graphs."""

        class PureModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.step1 = LLMInference(alias="1")
                self.step2 = LLMInference(alias="2")

            def forward(self, text: str) -> Proxy:
                # Pure: no side effects, deterministic
                r1 = self.step1(text)
                return self.step2(r1)

        tracer = Tracer()
        module = PureModule()

        # Trace twice with same input
        graph1 = tracer.trace(module, "same input")
        graph2 = tracer.trace(module, "same input")

        # Structure should be identical
        assert len(graph1.nodes) == len(graph2.nodes)
        assert graph1.input_ids == graph2.input_ids
        assert graph1.output_ids == graph2.output_ids

    def test_list_output_collection(self) -> None:
        """List outputs are properly collected."""

        class ListOutput(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm1 = LLMInference(alias="1")
                self.llm2 = LLMInference(alias="2")
                self.llm3 = LLMInference(alias="3")

            def forward(self, text: str) -> list[Proxy]:
                return [
                    self.llm1(text),
                    self.llm2(text),
                    self.llm3(text),
                ]

        tracer = Tracer()
        graph = tracer.trace(ListOutput(), "input")

        # All three outputs should be collected
        assert len(graph.output_ids) == 3
        assert "LLMInference_1" in graph.output_ids
        assert "LLMInference_2" in graph.output_ids
        assert "LLMInference_3" in graph.output_ids

    def test_tuple_output_collection(self) -> None:
        """Tuple outputs are properly collected."""

        class TupleOutput(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm1 = LLMInference(alias="1")
                self.llm2 = LLMInference(alias="2")

            def forward(self, text: str) -> tuple[Proxy, Proxy]:
                return (self.llm1(text), self.llm2(text))

        tracer = Tracer()
        graph = tracer.trace(TupleOutput(), "input")

        # Both outputs should be collected
        assert len(graph.output_ids) == 2

    def test_graph_has_valid_topological_order(self) -> None:
        """Traced graphs always have a valid topological order."""

        class ComplexGraph(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = LLMInference(alias="a")
                self.b = LLMInference(alias="b")
                self.c = LLMInference(alias="c")
                self.d = LLMInference(alias="d")

            def forward(self, text: str) -> Proxy:
                # Create a diamond-ish pattern
                r_a = self.a(text)
                r_b = self.b(text)
                r_c = self.c(r_a, r_b)
                r_d = self.d(r_c)
                return r_d

        tracer = Tracer()
        graph = tracer.trace(ComplexGraph(), "input")

        # Get topological order
        order = graph.topological_order()

        # Verify all nodes are in the order
        assert set(order) == set(graph.nodes.keys())

        # Verify dependencies are satisfied
        for node_id in order:
            node = graph.nodes[node_id]
            for dep_id in node.dependencies:
                assert order.index(dep_id) < order.index(node_id), (
                    f"Dependency {dep_id} should come before {node_id}"
                )
