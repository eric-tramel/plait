"""Unit tests for Value-driven tracing."""

from plait.module import LLMInference, Module
from plait.tracing.context import get_trace_context, trace_context
from plait.tracing.tracer import GetItemOp, InputNode, IterOp, MethodOp, Tracer
from plait.values import Value, ValueKind, ValueRef


class TestTracerInstantiation:
    def test_creation_with_empty_initial_state(self) -> None:
        tracer = Tracer()
        assert tracer.nodes == {}
        assert tracer.input_ids == []
        assert tracer.output_ids == []
        assert tracer._node_counter == 0
        assert tracer._module_stack == []
        assert tracer._branch_stack == []


class TestTracerIdGeneration:
    def test_generate_id_format_and_increments(self) -> None:
        tracer = Tracer()
        module = LLMInference(alias="test")

        id1 = tracer._generate_id(module)
        id2 = tracer._generate_id(module)
        id3 = tracer._generate_id(module)

        assert id1 == "LLMInference_1"
        assert id2 == "LLMInference_2"
        assert id3 == "LLMInference_3"
        assert tracer._node_counter == 3

    def test_generate_id_uses_class_name(self) -> None:
        class CustomModule(Module):
            def forward(self, x: Value) -> Value:
                return x

        tracer = Tracer()
        assert tracer._generate_id(CustomModule()) == "CustomModule_1"


class TestTracerReset:
    def test_reset_clears_all_state(self) -> None:
        tracer = Tracer()
        tracer._node_counter = 5
        tracer.nodes["n"] = None  # type: ignore[assignment]
        tracer.input_ids.append("input:0")
        tracer.output_ids.append("out:0")
        tracer._module_stack.append("parent")
        tracer._branch_stack.append(("cond", True))

        tracer.reset()

        assert tracer.nodes == {}
        assert tracer.input_ids == []
        assert tracer.output_ids == []
        assert tracer._node_counter == 0
        assert tracer._module_stack == []
        assert tracer._branch_stack == []


class TestBindInputs:
    def test_bind_inputs_creates_input_nodes(self) -> None:
        tracer = Tracer()
        value = Value(ValueKind.TEXT, "hello")
        bound = tracer.bind_inputs(value, prefix="input")

        assert bound.ref == "input:input_0"
        assert tracer.input_ids == ["input:input_0"]
        assert isinstance(tracer.nodes["input:input_0"].module, InputNode)
        assert tracer.nodes["input:input_0"].module.value == "hello"

    def test_bind_inputs_structured(self) -> None:
        tracer = Tracer()
        bound = tracer.bind_inputs({"a": "x", "b": "y"}, prefix="input")

        assert bound["a"].ref == "input:input_a"
        assert bound["b"].ref == "input:input_b"
        assert set(tracer.input_ids) == {"input:input_a", "input:input_b"}


class TestRecordCall:
    def test_record_call_collects_value_dependencies(self) -> None:
        tracer = Tracer()
        module = LLMInference(alias="test")
        value = Value(ValueKind.TEXT, "hello", ref="input:0")

        output = tracer.record_call(module, (value,), {})

        assert output.ref == "LLMInference_1"
        node = tracer.nodes["LLMInference_1"]
        assert node.dependencies == ["input:0"]
        assert isinstance(node.args[0], ValueRef)
        assert node.args[0].ref == "input:0"


class TestValueOps:
    def test_record_getitem_creates_node(self) -> None:
        tracer = Tracer()
        value = Value(ValueKind.STRUCTURED, {"key": "value"}, ref="input:0")
        with trace_context(tracer):
            result = tracer.record_getitem(value, "key")

        node = tracer.nodes[result.ref]
        assert isinstance(node.module, GetItemOp)
        assert node.dependencies == ["input:0"]

    def test_record_iter_creates_node(self) -> None:
        tracer = Tracer()
        value = Value(ValueKind.STRUCTURED, [1, 2], ref="input:0")
        with trace_context(tracer):
            result = tracer.record_iter(value)

        node = tracer.nodes[result.ref]
        assert isinstance(node.module, IterOp)
        assert node.dependencies == ["input:0"]

    def test_record_method_creates_node(self) -> None:
        tracer = Tracer()
        value = Value(ValueKind.STRUCTURED, {"a": 1}, ref="input:0")
        with trace_context(tracer):
            result = tracer.record_method(value, "keys")

        node = tracer.nodes[result.ref]
        assert isinstance(node.module, MethodOp)
        assert node.dependencies == ["input:0"]


class TestOutputCapture:
    def test_collect_output_ids_from_value(self) -> None:
        tracer = Tracer()
        value = Value(ValueKind.TEXT, "hello", ref="node_1")
        assert tracer._collect_output_ids(value) == ["node_1"]

    def test_capture_output_structure_dict(self) -> None:
        tracer = Tracer()
        output = {
            "a": Value(ValueKind.TEXT, "x", ref="node_1"),
            "b": Value(ValueKind.TEXT, "y", ref="node_2"),
        }
        captured = tracer._capture_output_structure(output)
        assert captured == {"a": "node_1", "b": "node_2"}


class TestTraceValues:
    def test_trace_values_simple_pipeline(self) -> None:
        class Pipeline(Module):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="fast")

            def forward(self, text: Value) -> Value:
                return self.llm(text)

        tracer = Tracer()
        graph = tracer.trace_values(Pipeline(), "input text")

        assert len(graph.nodes) == 2
        assert "input:input_0" in graph.nodes
        assert "LLMInference_1" in graph.nodes
        assert graph.output_ids == ["LLMInference_1"]

    def test_trace_alias_uses_values(self) -> None:
        class Pipeline(Module):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(alias="fast")

            def forward(self, text: Value) -> Value:
                return self.llm(text)

        tracer = Tracer()
        graph = tracer.trace(Pipeline(), "input text")
        assert "input:input_0" in graph.nodes
        assert "LLMInference_1" in graph.nodes


class TestBranchContext:
    def test_branch_context_applied_to_nodes(self) -> None:
        tracer = Tracer()
        module = LLMInference(alias="test")
        value = Value(ValueKind.TEXT, "hello", ref="input:0")

        tracer._branch_stack.append(("condition_node", True))
        tracer.record_call(module, (value,), {})
        node = tracer.nodes["LLMInference_1"]

        assert node.branch_condition == "condition_node"
        assert node.branch_value is True

    def test_get_trace_context_roundtrip(self) -> None:
        tracer = Tracer()
        assert get_trace_context() is None
        with trace_context(tracer):
            assert get_trace_context() is tracer
        assert get_trace_context() is None
