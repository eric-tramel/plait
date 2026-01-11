"""Integration tests for Value-driven tracing."""

from plait.module import LLMInference, Module
from plait.parameter import Parameter
from plait.tracing.context import get_trace_context
from plait.tracing.tracer import InputNode, Tracer
from plait.values import Value, ValueRef


class TestSimpleModuleGraphWithValues:
    """Tests for simple module graphs using Value-driven tracing."""

    def test_single_llm_module(self) -> None:
        """Single LLM module produces correct graph with Values."""

        class SingleLLM(Module):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="model")

            def forward(self, prompt: Value) -> Value:
                return self.llm(prompt)

        tracer = Tracer()
        graph = tracer.trace_values(SingleLLM(), "Hello, world!")

        # Verify graph structure
        assert len(graph.input_ids) == 1
        assert len(graph.nodes) == 2  # 1 input + 1 LLM call

        # Verify input node
        input_id = graph.input_ids[0]
        assert input_id in graph.nodes
        input_node = graph.nodes[input_id]
        assert isinstance(input_node.module, InputNode)
        assert input_node.module.value == "Hello, world!"

        # Verify LLM node
        assert "LLMInference_1" in graph.nodes
        llm_node = graph.nodes["LLMInference_1"]
        assert input_id in llm_node.dependencies

        # Verify output
        assert graph.output_ids == ["LLMInference_1"]

    def test_two_stage_pipeline(self) -> None:
        """Two-stage LLM pipeline creates linear dependency chain."""

        class TwoStagePipeline(Module):
            def __init__(self) -> None:
                super().__init__()
                self.stage1 = LLMInference(alias="stage1")
                self.stage2 = LLMInference(alias="stage2")

            def forward(self, prompt: Value) -> Value:
                intermediate = self.stage1(prompt)
                result = self.stage2(intermediate)
                return result

        tracer = Tracer()
        graph = tracer.trace_values(TwoStagePipeline(), "Input prompt")

        # Verify structure: input -> stage1 -> stage2
        assert len(graph.nodes) == 3
        assert graph.nodes["LLMInference_2"].dependencies == ["LLMInference_1"]
        assert graph.output_ids == ["LLMInference_2"]

    def test_parallel_branches(self) -> None:
        """Parallel branches create independent dependency paths."""

        class ParallelBranches(Module):
            def __init__(self) -> None:
                super().__init__()
                self.branch_a = LLMInference(alias="branch_a")
                self.branch_b = LLMInference(alias="branch_b")

            def forward(self, prompt: Value) -> dict[str, Value]:
                result_a = self.branch_a(prompt)
                result_b = self.branch_b(prompt)
                return {"a": result_a, "b": result_b}

        tracer = Tracer()
        graph = tracer.trace_values(ParallelBranches(), "Input")

        # Both branches depend on same input
        input_id = graph.input_ids[0]
        assert input_id in graph.nodes["LLMInference_1"].dependencies
        assert input_id in graph.nodes["LLMInference_2"].dependencies

        # Output structure preserved
        assert graph.output_structure == {"a": "LLMInference_1", "b": "LLMInference_2"}

    def test_merge_after_parallel(self) -> None:
        """Merge node depends on both parallel branches."""

        class MergePattern(Module):
            def __init__(self) -> None:
                super().__init__()
                self.summarize = LLMInference(alias="summarize")
                self.analyze = LLMInference(alias="analyze")
                self.merge = LLMInference(alias="merge")

            def forward(self, text: Value) -> Value:
                summary = self.summarize(text)
                analysis = self.analyze(text)
                # Merge by passing both as tuple
                ctx = get_trace_context()
                if ctx:
                    return ctx.record_call(self.merge, (summary, analysis), {})
                return summary

        tracer = Tracer()
        graph = tracer.trace_values(MergePattern(), "Document text")

        # Merge node depends on both branches
        merge_deps = graph.nodes["LLMInference_3"].dependencies
        assert "LLMInference_1" in merge_deps
        assert "LLMInference_2" in merge_deps


class TestOutputIdsFromValues:
    """Tests for collecting output IDs from Value structures."""

    def test_single_value_output(self) -> None:
        """Single Value output produces single output ID."""

        class SingleOutput(Module):
            def forward(self, x: Value) -> Value:
                return x

        tracer = Tracer()
        graph = tracer.trace_values(SingleOutput(), "input")

        assert len(graph.output_ids) == 1

    def test_list_value_output(self) -> None:
        """List of Values produces multiple output IDs."""

        class ListOutput(Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = LLMInference(alias="a")
                self.b = LLMInference(alias="b")

            def forward(self, x: Value) -> list[Value]:
                return [self.a(x), self.b(x)]

        tracer = Tracer()
        graph = tracer.trace_values(ListOutput(), "input")

        assert len(graph.output_ids) == 2
        assert "LLMInference_1" in graph.output_ids
        assert "LLMInference_2" in graph.output_ids

    def test_dict_value_output(self) -> None:
        """Dict of Values produces output IDs for all values."""

        class DictOutput(Module):
            def __init__(self) -> None:
                super().__init__()
                self.summary = LLMInference(alias="summary")
                self.keywords = LLMInference(alias="keywords")

            def forward(self, text: Value) -> dict[str, Value]:
                return {
                    "summary": self.summary(text),
                    "keywords": self.keywords(text),
                }

        tracer = Tracer()
        graph = tracer.trace_values(DictOutput(), "Document")

        assert len(graph.output_ids) == 2
        assert graph.output_structure == {
            "summary": "LLMInference_1",
            "keywords": "LLMInference_2",
        }

    def test_nested_value_output(self) -> None:
        """Nested Value structure produces all output IDs."""

        class NestedOutput(Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = LLMInference(alias="a")
                self.b = LLMInference(alias="b")
                self.c = LLMInference(alias="c")

            def forward(self, x: Value) -> dict[str, list[Value] | Value]:
                return {
                    "items": [self.a(x), self.b(x)],
                    "final": self.c(x),
                }

        tracer = Tracer()
        graph = tracer.trace_values(NestedOutput(), "input")

        assert len(graph.output_ids) == 3


class TestValueRefPlaceholders:
    """Tests for ValueRef placeholders in graph nodes."""

    def test_value_ref_stored_in_args(self) -> None:
        """ValueRef placeholders are stored in node args."""

        class SimpleModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="llm")

            def forward(self, x: Value) -> Value:
                return self.llm(x)

        tracer = Tracer()
        graph = tracer.trace_values(SimpleModule(), "hello")

        llm_node = graph.nodes["LLMInference_1"]
        assert len(llm_node.args) == 1
        assert isinstance(llm_node.args[0], ValueRef)

    def test_value_ref_has_correct_ref(self) -> None:
        """ValueRef placeholder has correct ref pointing to dependency."""

        class SimpleModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="llm")

            def forward(self, x: Value) -> Value:
                return self.llm(x)

        tracer = Tracer()
        graph = tracer.trace_values(SimpleModule(), "hello")

        llm_node = graph.nodes["LLMInference_1"]
        value_ref = llm_node.args[0]
        assert isinstance(value_ref, ValueRef)
        assert value_ref.ref == graph.input_ids[0]

    def test_chained_value_refs(self) -> None:
        """Chained calls have ValueRefs pointing to previous outputs."""

        class ChainedModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.step1 = LLMInference(alias="step1")
                self.step2 = LLMInference(alias="step2")

            def forward(self, x: Value) -> Value:
                intermediate = self.step1(x)
                return self.step2(intermediate)

        tracer = Tracer()
        graph = tracer.trace_values(ChainedModule(), "input")

        step2_node = graph.nodes["LLMInference_2"]
        value_ref = step2_node.args[0]
        assert isinstance(value_ref, ValueRef)
        assert value_ref.ref == "LLMInference_1"


class TestParameterIntegration:
    """Tests for parameter handling with Value-driven tracing."""

    def test_parameters_collected(self) -> None:
        """Parameters are collected from module tree."""

        class ModuleWithParams(Module):
            def __init__(self) -> None:
                super().__init__()
                self.system_prompt = Parameter("You are helpful", description="system")
                self.llm = LLMInference(alias="llm")

            def forward(self, user_input: Value) -> Value:
                return self.llm(user_input)

        tracer = Tracer()
        graph = tracer.trace_values(ModuleWithParams(), "Hello")

        assert "system_prompt" in graph.parameters
        assert graph.parameters["system_prompt"].value == "You are helpful"

    def test_nested_parameters_collected(self) -> None:
        """Nested module parameters have dotted paths."""

        class Inner(Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner_prompt = Parameter("inner", description="inner prompt")

            def forward(self, x: Value) -> Value:
                return x

        class Outer(Module):
            def __init__(self) -> None:
                super().__init__()
                self.outer_prompt = Parameter("outer", description="outer prompt")
                self.inner = Inner()

            def forward(self, x: Value) -> Value:
                return self.inner(x)

        tracer = Tracer()
        graph = tracer.trace_values(Outer(), "input")

        assert "outer_prompt" in graph.parameters
        assert "inner.inner_prompt" in graph.parameters


class TestEdgeCases:
    """Tests for edge cases in Value-driven tracing."""

    def test_empty_input(self) -> None:
        """Module with no input arguments."""

        class NoInput(Module):
            def forward(self) -> str:
                return "constant"

        tracer = Tracer()
        graph = tracer.trace_values(NoInput())

        assert graph.input_ids == []
        assert graph.output_ids == []

    def test_literal_output(self) -> None:
        """Module returning literal (non-Value) output."""

        class LiteralOutput(Module):
            def forward(self, x: Value) -> str:
                return "literal"

        tracer = Tracer()
        graph = tracer.trace_values(LiteralOutput(), "input")

        # Literal output doesn't produce output IDs
        assert graph.output_ids == []

    def test_mixed_value_and_literal_output(self) -> None:
        """Module returning mix of Values and literals."""

        class MixedOutput(Module):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="llm")

            def forward(self, x: Value) -> dict[str, str | Value]:
                return {
                    "result": self.llm(x),
                    "status": "success",  # literal
                }

        tracer = Tracer()
        graph = tracer.trace_values(MixedOutput(), "input")

        # Only Value output tracked
        assert graph.output_ids == ["LLMInference_1"]
        assert graph.output_structure == {"result": "LLMInference_1"}

    def test_multiple_traces_independent(self) -> None:
        """Multiple trace_values calls produce independent graphs."""

        class SimpleModule(Module):
            def forward(self, x: Value) -> Value:
                return x

        tracer = Tracer()

        graph1 = tracer.trace_values(SimpleModule(), "first")
        graph2 = tracer.trace_values(SimpleModule(), "second")

        # Graphs should be independent
        assert graph1.nodes != graph2.nodes
        input1 = list(graph1.nodes.values())[0]
        input2 = list(graph2.nodes.values())[0]
        assert isinstance(input1.module, InputNode)
        assert isinstance(input2.module, InputNode)
        assert input1.module.value == "first"
        assert input2.module.value == "second"
