"""Unit tests for Value-driven tracing in the Tracer class."""

from plait.graph import GraphNode, InferenceGraph
from plait.module import LLMInference, Module
from plait.tracing.tracer import InputNode, Tracer
from plait.values import Value, ValueKind, ValueRef


class TestBindInputs:
    """Tests for Tracer.bind_inputs()."""

    def test_bind_single_value(self) -> None:
        """bind_inputs assigns ref to a single Value."""
        tracer = Tracer()
        v = Value(ValueKind.TEXT, "hello")

        bound = tracer.bind_inputs(v, prefix="input")

        assert isinstance(bound, Value)
        assert bound.ref == "input:input_0"
        assert bound.payload == "hello"
        assert bound.kind == ValueKind.TEXT

    def test_bind_creates_input_node(self) -> None:
        """bind_inputs creates input node in graph."""
        tracer = Tracer()
        v = Value(ValueKind.TEXT, "hello")

        bound = tracer.bind_inputs(v, prefix="input")

        assert bound.ref in tracer.nodes
        node = tracer.nodes[bound.ref]
        assert isinstance(node, GraphNode)
        assert isinstance(node.module, InputNode)
        assert node.module.value == "hello"

    def test_bind_adds_to_input_ids(self) -> None:
        """bind_inputs adds node ID to input_ids."""
        tracer = Tracer()
        v = Value(ValueKind.TEXT, "hello")

        bound = tracer.bind_inputs(v, prefix="input")

        assert tracer.input_ids == [bound.ref]

    def test_bind_list_of_values(self) -> None:
        """bind_inputs handles list of Values."""
        tracer = Tracer()
        values = [Value(ValueKind.TEXT, "a"), Value(ValueKind.TEXT, "b")]

        bound = tracer.bind_inputs(values, prefix="input")

        assert isinstance(bound, list)
        assert len(bound) == 2
        assert bound[0].ref == "input:input_0"
        assert bound[1].ref == "input:input_1"
        assert len(tracer.input_ids) == 2

    def test_bind_tuple_of_values(self) -> None:
        """bind_inputs handles tuple of Values."""
        tracer = Tracer()
        values = (Value(ValueKind.TEXT, "a"), Value(ValueKind.TEXT, "b"))

        bound = tracer.bind_inputs(values, prefix="input")

        assert isinstance(bound, tuple)
        assert len(bound) == 2
        assert bound[0].ref == "input:input_0"
        assert bound[1].ref == "input:input_1"

    def test_bind_dict_of_values(self) -> None:
        """bind_inputs handles dict of Values with key-based naming."""
        tracer = Tracer()
        values = {
            "text": Value(ValueKind.TEXT, "hello"),
            "context": Value(ValueKind.STRUCTURED, {"key": "value"}),
        }

        bound = tracer.bind_inputs(values, prefix="input")

        assert isinstance(bound, dict)
        assert "text" in bound
        assert "context" in bound
        # Dict keys are used in naming.
        assert bound["text"].ref == "input:input_text"
        assert bound["context"].ref == "input:input_context"

    def test_bind_nested_structure(self) -> None:
        """bind_inputs handles nested list/dict structures."""
        tracer = Tracer()
        values = {
            "items": [
                Value(ValueKind.TEXT, "first"),
                Value(ValueKind.TEXT, "second"),
            ]
        }

        bound = tracer.bind_inputs(values, prefix="input")

        assert isinstance(bound, dict)
        assert len(bound["items"]) == 2
        assert bound["items"][0].ref is not None
        assert bound["items"][1].ref is not None

    def test_bind_literal_converts_to_value(self) -> None:
        """bind_inputs converts literals to Values before binding."""
        tracer = Tracer()

        bound = tracer.bind_inputs("hello", prefix="input")

        assert isinstance(bound, Value)
        assert bound.ref is not None
        assert bound.payload == "hello"

    def test_bind_multiple_literals(self) -> None:
        """bind_inputs converts multiple literals in a list."""
        tracer = Tracer()

        bound = tracer.bind_inputs(["a", "b", "c"], prefix="input")

        assert isinstance(bound, list)
        assert all(isinstance(b, Value) for b in bound)
        assert all(b.ref is not None for b in bound)

    def test_bind_preserves_value_metadata(self) -> None:
        """bind_inputs preserves Value metadata."""
        tracer = Tracer()
        v = Value(ValueKind.TEXT, "hello", meta={"source": "test"})

        bound = tracer.bind_inputs(v, prefix="input")

        assert bound.meta.get("source") == "test"


class TestRecordCallWithValues:
    """Tests for record_call with Value arguments."""

    def test_record_call_returns_value(self) -> None:
        """record_call returns a Value."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        result = tracer.record_call(module, (), {})

        assert isinstance(result, Value)
        assert result.kind == ValueKind.RESPONSE
        assert result.ref == "LLMInference_1"

    def test_record_call_value_ref_is_node_id(self) -> None:
        """Returned Value has ref set to node ID."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        result = tracer.record_call(module, (), {})

        assert result.ref == "LLMInference_1"
        assert result.ref in tracer.nodes

    def test_record_call_extracts_value_dependencies(self) -> None:
        """record_call extracts dependencies from Value.ref."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        input_value = Value(ValueKind.TEXT, "hello", ref="input:0")

        result = tracer.record_call(module, (input_value,), {})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        assert "input:0" in node.dependencies

    def test_record_call_extracts_multiple_value_dependencies(self) -> None:
        """record_call extracts dependencies from multiple Values."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        v1 = Value(ValueKind.TEXT, "a", ref="input:0")
        v2 = Value(ValueKind.TEXT, "b", ref="input:1")

        result = tracer.record_call(module, (v1, v2), {})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        assert "input:0" in node.dependencies
        assert "input:1" in node.dependencies

    def test_record_call_extracts_kwargs_value_dependencies(self) -> None:
        """record_call extracts dependencies from Value kwargs."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        v = Value(ValueKind.TEXT, "hello", ref="input:text")

        result = tracer.record_call(module, (), {"prompt": v})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        assert "input:text" in node.dependencies

    def test_record_call_stores_value_refs(self) -> None:
        """record_call stores ValueRef placeholders in args/kwargs."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        v = Value(ValueKind.TEXT, "hello", ref="input:0")

        result = tracer.record_call(module, (v,), {})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        assert len(node.args) == 1
        assert isinstance(node.args[0], ValueRef)
        assert node.args[0].ref == "input:0"

    def test_record_call_stores_kwargs_value_refs(self) -> None:
        """record_call stores ValueRef placeholders in kwargs."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        v = Value(ValueKind.TEXT, "hello", ref="input:text")

        result = tracer.record_call(module, (), {"prompt": v})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        assert "prompt" in node.kwargs
        assert isinstance(node.kwargs["prompt"], ValueRef)
        assert node.kwargs["prompt"].ref == "input:text"

    def test_record_call_nested_value_dependencies(self) -> None:
        """record_call extracts dependencies from nested Value structures."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        v1 = Value(ValueKind.TEXT, "a", ref="input:0")
        v2 = Value(ValueKind.TEXT, "b", ref="input:1")
        nested = {"items": [v1, v2]}

        result = tracer.record_call(module, (nested,), {})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        # collect_refs should find both refs
        assert "input:0" in node.dependencies
        assert "input:1" in node.dependencies

    def test_record_call_chained_values(self) -> None:
        """record_call creates correct dependency chain with Values."""
        tracer = Tracer()
        step1 = LLMInference(alias="step1")
        step2 = LLMInference(alias="step2")

        # First call
        input_val = Value(ValueKind.TEXT, "hello", ref="input:0")
        out1 = tracer.record_call(step1, (input_val,), {})

        # Second call uses output of first
        tracer.record_call(step2, (out1,), {})

        # Verify dependency chain
        assert tracer.nodes["LLMInference_1"].dependencies == ["input:0"]
        assert tracer.nodes["LLMInference_2"].dependencies == ["LLMInference_1"]

    def test_record_call_nested_value_refs_in_args(self) -> None:
        """record_call stores ValueRef placeholders for nested Value structures."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        v1 = Value(ValueKind.TEXT, "a", ref="input:0")
        v2 = Value(ValueKind.TEXT, "b", ref="input:1")
        nested_arg = {"items": [v1, v2], "meta": {"extra": v1}}

        result = tracer.record_call(module, (nested_arg,), {})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        # Verify nested structure contains ValueRefs
        assert len(node.args) == 1
        arg_dict = node.args[0]
        assert isinstance(arg_dict, dict)
        assert "items" in arg_dict
        assert isinstance(arg_dict["items"], list)
        assert isinstance(arg_dict["items"][0], ValueRef)
        assert arg_dict["items"][0].ref == "input:0"
        assert isinstance(arg_dict["items"][1], ValueRef)
        assert arg_dict["items"][1].ref == "input:1"
        assert isinstance(arg_dict["meta"]["extra"], ValueRef)
        assert arg_dict["meta"]["extra"].ref == "input:0"

    def test_record_call_nested_value_refs_in_kwargs(self) -> None:
        """record_call stores ValueRef placeholders for nested kwargs structures."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        v = Value(ValueKind.TEXT, "hello", ref="input:text")
        nested_kwarg = {"config": {"prompt": v, "options": [v]}}

        result = tracer.record_call(module, (), {"data": nested_kwarg})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        # Verify nested structure in kwargs contains ValueRefs
        assert "data" in node.kwargs
        data = node.kwargs["data"]
        assert isinstance(data, dict)
        config = data["config"]
        assert isinstance(config, dict)
        assert isinstance(config["prompt"], ValueRef)
        assert config["prompt"].ref == "input:text"
        options = config["options"]
        assert isinstance(options, list)
        assert isinstance(options[0], ValueRef)
        assert options[0].ref == "input:text"


class TestCollectOutputIdsWithValues:
    """Tests for _collect_output_ids with Value outputs."""

    def test_collect_single_value(self) -> None:
        """_collect_output_ids extracts ref from single Value."""
        tracer = Tracer()
        v = Value(ValueKind.TEXT, "hello", ref="node_1")

        result = tracer._collect_output_ids(v)

        assert result == ["node_1"]

    def test_collect_value_without_ref(self) -> None:
        """_collect_output_ids returns empty for Value without ref."""
        tracer = Tracer()
        v = Value(ValueKind.TEXT, "hello")

        result = tracer._collect_output_ids(v)

        assert result == []

    def test_collect_list_of_values(self) -> None:
        """_collect_output_ids extracts refs from list of Values."""
        tracer = Tracer()
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")

        result = tracer._collect_output_ids([v1, v2])

        assert "node_1" in result
        assert "node_2" in result

    def test_collect_dict_of_values(self) -> None:
        """_collect_output_ids extracts refs from dict values."""
        tracer = Tracer()
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")

        result = tracer._collect_output_ids({"x": v1, "y": v2})

        assert "node_1" in result
        assert "node_2" in result

    def test_collect_nested_values(self) -> None:
        """_collect_output_ids handles nested Value structures."""
        tracer = Tracer()
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")
        nested = {"items": [v1, {"inner": v2}]}

        result = tracer._collect_output_ids(nested)

        assert "node_1" in result
        assert "node_2" in result


class TestCaptureOutputStructureWithValues:
    """Tests for _capture_output_structure with Value outputs."""

    def test_capture_single_value(self) -> None:
        """_capture_output_structure returns ref for single Value."""
        tracer = Tracer()
        v = Value(ValueKind.TEXT, "hello", ref="node_1")

        result = tracer._capture_output_structure(v)

        assert result == "node_1"

    def test_capture_value_without_ref(self) -> None:
        """_capture_output_structure returns None for Value without ref."""
        tracer = Tracer()
        v = Value(ValueKind.TEXT, "hello")

        result = tracer._capture_output_structure(v)

        assert result is None

    def test_capture_dict_of_values(self) -> None:
        """_capture_output_structure preserves dict structure with refs."""
        tracer = Tracer()
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")

        result = tracer._capture_output_structure({"summary": v1, "analysis": v2})

        assert result == {"summary": "node_1", "analysis": "node_2"}

    def test_capture_list_of_values(self) -> None:
        """_capture_output_structure preserves list structure with refs."""
        tracer = Tracer()
        v1 = Value(ValueKind.TEXT, "a", ref="node_1")
        v2 = Value(ValueKind.TEXT, "b", ref="node_2")

        result = tracer._capture_output_structure([v1, v2])

        assert result == ["node_1", "node_2"]


class TestTraceValues:
    """Tests for Tracer.trace_values()."""

    def test_trace_values_returns_inference_graph(self) -> None:
        """trace_values returns an InferenceGraph."""

        class PassThrough(Module):
            def forward(self, x: Value) -> Value:
                return x

        tracer = Tracer()
        graph = tracer.trace_values(PassThrough(), "input")

        assert isinstance(graph, InferenceGraph)

    def test_trace_values_creates_input_nodes(self) -> None:
        """trace_values creates input nodes for arguments."""

        class PassThrough(Module):
            def forward(self, x: Value) -> Value:
                return x

        tracer = Tracer()
        graph = tracer.trace_values(PassThrough(), "hello")

        assert len(graph.input_ids) == 1
        assert graph.input_ids[0].startswith("input:")

    def test_trace_values_binds_values_with_refs(self) -> None:
        """trace_values binds input Values with refs."""
        received_value: list[Value] = []

        class CaptureValue(Module):
            def forward(self, x: Value) -> Value:
                received_value.append(x)
                return x

        tracer = Tracer()
        tracer.trace_values(CaptureValue(), "hello")

        assert len(received_value) == 1
        assert isinstance(received_value[0], Value)
        assert received_value[0].ref is not None
        assert received_value[0].ref.startswith("input:")

    def test_trace_values_collects_output_ids(self) -> None:
        """trace_values collects output IDs from returned Values."""

        class PassThrough(Module):
            def forward(self, x: Value) -> Value:
                return x

        tracer = Tracer()
        graph = tracer.trace_values(PassThrough(), "input")

        assert len(graph.output_ids) == 1
        assert graph.output_ids[0].startswith("input:")

    def test_trace_values_with_module_call(self) -> None:
        """trace_values records module calls returning Values."""
        from plait.tracing.context import get_trace_context

        class WithCall(Module):
            def forward(self, x: Value) -> Value:
                ctx = get_trace_context()
                if ctx is not None:
                    child = LLMInference(alias="test")
                    return ctx.record_call(child, (x,), {})
                return x

        tracer = Tracer()
        graph = tracer.trace_values(WithCall(), "input")

        # Should have input node + LLMInference call
        assert len(graph.nodes) == 2
        assert "LLMInference_1" in graph.nodes
        assert graph.output_ids == ["LLMInference_1"]

    def test_trace_values_multiple_args(self) -> None:
        """trace_values handles multiple arguments."""

        class TwoInputs(Module):
            def forward(self, a: Value, b: Value) -> tuple[Value, Value]:
                return a, b

        tracer = Tracer()
        graph = tracer.trace_values(TwoInputs(), "first", "second")

        assert len(graph.input_ids) == 2
        assert len(graph.output_ids) == 2

    def test_trace_values_with_kwargs(self) -> None:
        """trace_values handles keyword arguments."""

        class KwargModule(Module):
            def forward(self, *, text: Value, context: Value) -> dict[str, Value]:
                return {"text": text, "context": context}

        tracer = Tracer()
        graph = tracer.trace_values(KwargModule(), text="hello", context="world")

        assert len(graph.input_ids) == 2
        assert graph.output_structure is not None

    def test_trace_values_dict_output_structure(self) -> None:
        """trace_values captures dict output structure."""

        class DictOutput(Module):
            def forward(self, x: Value) -> dict[str, Value]:
                return {"result": x}

        tracer = Tracer()
        graph = tracer.trace_values(DictOutput(), "input")

        assert isinstance(graph.output_structure, dict)
        assert "result" in graph.output_structure

    def test_trace_values_collects_parameters(self) -> None:
        """trace_values collects parameters from module."""
        from plait.parameter import Parameter

        class ModuleWithParam(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("test prompt", description="test")

            def forward(self, x: Value) -> Value:
                return x

        tracer = Tracer()
        graph = tracer.trace_values(ModuleWithParam(), "input")

        assert "prompt" in graph.parameters


class TestValueDependencyChains:
    """Tests for complex Value dependency patterns."""

    def test_linear_value_chain(self) -> None:
        """Linear chain of Values creates correct dependencies."""
        from plait.tracing.context import get_trace_context

        class LinearChain(Module):
            def forward(self, x: Value) -> Value:
                ctx = get_trace_context()
                if ctx is None:
                    return x

                step1 = LLMInference(alias="step1")
                step2 = LLMInference(alias="step2")
                step3 = LLMInference(alias="step3")

                out1 = ctx.record_call(step1, (x,), {})
                out2 = ctx.record_call(step2, (out1,), {})
                out3 = ctx.record_call(step3, (out2,), {})
                return out3

        tracer = Tracer()
        graph = tracer.trace_values(LinearChain(), "input")

        # Verify linear dependency chain
        assert graph.nodes["LLMInference_2"].dependencies == ["LLMInference_1"]
        assert graph.nodes["LLMInference_3"].dependencies == ["LLMInference_2"]
        assert graph.output_ids == ["LLMInference_3"]

    def test_diamond_value_pattern(self) -> None:
        """Diamond pattern with Values creates correct dependencies."""
        from plait.tracing.context import get_trace_context

        class DiamondPattern(Module):
            def forward(self, x: Value) -> Value:
                ctx = get_trace_context()
                if ctx is None:
                    return x

                branch_a = LLMInference(alias="a")
                branch_b = LLMInference(alias="b")
                merge = LLMInference(alias="merge")

                out_a = ctx.record_call(branch_a, (x,), {})
                out_b = ctx.record_call(branch_b, (x,), {})
                out_merge = ctx.record_call(merge, (out_a, out_b), {})
                return out_merge

        tracer = Tracer()
        graph = tracer.trace_values(DiamondPattern(), "input")

        # Verify diamond structure
        merge_deps = graph.nodes["LLMInference_3"].dependencies
        assert "LLMInference_1" in merge_deps
        assert "LLMInference_2" in merge_deps
        assert graph.output_ids == ["LLMInference_3"]

    def test_fan_out_value_pattern(self) -> None:
        """Fan-out pattern with Values tracks all branches."""
        from plait.tracing.context import get_trace_context

        class FanOut(Module):
            def forward(self, x: Value) -> list[Value]:
                ctx = get_trace_context()
                if ctx is None:
                    return [x]

                a = LLMInference(alias="a")
                b = LLMInference(alias="b")
                c = LLMInference(alias="c")

                out_a = ctx.record_call(a, (x,), {})
                out_b = ctx.record_call(b, (x,), {})
                out_c = ctx.record_call(c, (x,), {})
                return [out_a, out_b, out_c]

        tracer = Tracer()
        graph = tracer.trace_values(FanOut(), "input")

        # All branches depend on input
        input_id = graph.input_ids[0]
        assert input_id in graph.nodes["LLMInference_1"].dependencies
        assert input_id in graph.nodes["LLMInference_2"].dependencies
        assert input_id in graph.nodes["LLMInference_3"].dependencies

        # All outputs tracked
        assert len(graph.output_ids) == 3
