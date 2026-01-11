"""Unit tests for the Tracer class.

This module tests tracer creation, input nodes, recording module calls,
dependency extraction, and graph construction via trace().
Tests are consolidated to reduce redundancy while maintaining coverage.
"""

from typing import Any

import pytest

from plait.graph import GraphNode, InferenceGraph, NodeRef
from plait.module import LLMInference, Module
from plait.parameter import Parameter
from plait.tracing.context import get_trace_context
from plait.tracing.proxy import Proxy
from plait.tracing.tracer import GetItemOp, InputNode, IterOp, MethodOp, Tracer
from plait.values import Value


class TestTracerInstantiation:
    """Tests for Tracer instantiation and initial state."""

    def test_creation_with_empty_initial_state(self) -> None:
        """New tracer has empty nodes, input_ids, output_ids, stacks, and counter."""
        tracer = Tracer()

        assert isinstance(tracer, Tracer)
        assert tracer.nodes == {}
        assert tracer.input_ids == []
        assert tracer.output_ids == []
        assert tracer._node_counter == 0
        assert tracer._module_stack == []
        assert tracer._branch_stack == []


class TestTracerIdGeneration:
    """Tests for Tracer._generate_id()."""

    def test_generate_id_format_and_increments(self) -> None:
        """Generated ID has format 'ClassName_N' and increments counter."""
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
        """Generated ID uses the module's class name."""

        class CustomModule(Module):
            def forward(self, x: str) -> str:
                return x

        tracer = Tracer()
        assert tracer._generate_id(CustomModule()) == "CustomModule_1"

    def test_different_modules_share_counter(self) -> None:
        """Different module types share a single counter."""

        class ModuleA(Module):
            def forward(self, x: str) -> str:
                return x

        class ModuleB(Module):
            def forward(self, x: str) -> str:
                return x

        tracer = Tracer()
        id1 = tracer._generate_id(ModuleA())
        id2 = tracer._generate_id(ModuleB())
        id3 = tracer._generate_id(LLMInference(alias="test"))

        assert id1 == "ModuleA_1"
        assert id2 == "ModuleB_2"
        assert id3 == "LLMInference_3"


class TestTracerReset:
    """Tests for Tracer.reset()."""

    def test_reset_clears_all_state(self) -> None:
        """Reset clears nodes, IDs, counter, and stacks."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        # Pollute tracer state
        tracer._generate_id(module)
        tracer._generate_id(module)
        tracer.nodes["test"] = None  # type: ignore
        tracer.input_ids.append("input_0")
        tracer.output_ids.append("output_0")
        tracer._module_stack.append("parent")
        tracer._branch_stack.append(("cond", True))

        tracer.reset()

        assert tracer.nodes == {}
        assert tracer.input_ids == []
        assert tracer.output_ids == []
        assert tracer._node_counter == 0
        assert tracer._module_stack == []
        assert tracer._branch_stack == []

    def test_reset_allows_fresh_tracing(self) -> None:
        """After reset, tracer can be used for a fresh trace."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        tracer._generate_id(module)
        tracer.reset()

        # Generate IDs again - should start from 1
        new_id = tracer._generate_id(module)
        assert new_id == "LLMInference_1"


class TestInputNode:
    """Tests for the InputNode class."""

    @pytest.mark.parametrize(
        "value",
        ["text", 42, {"key": "value"}, [1, 2, 3], None],
        ids=["string", "int", "dict", "list", "none"],
    )
    def test_stores_any_type(self, value: object) -> None:
        """InputNode can store values of any type."""
        node = InputNode(value=value)
        assert node.value == value

    def test_equality(self) -> None:
        """InputNodes with same value are equal."""
        assert InputNode(value="test") == InputNode(value="test")
        assert InputNode(value="test") != InputNode(value="other")


class TestCreateInputNode:
    """Tests for Tracer._create_input_node()."""

    def test_returns_proxy_with_correct_id(self) -> None:
        """_create_input_node returns a Proxy with format 'input:{name}'."""
        tracer = Tracer()

        proxy = tracer._create_input_node("text", "value")

        assert isinstance(proxy, Proxy)
        assert proxy.node_id == "input:text"
        assert proxy.tracer is tracer

    def test_adds_to_input_ids_and_nodes(self) -> None:
        """Created input nodes are added to input_ids and nodes dict."""
        tracer = Tracer()

        proxy = tracer._create_input_node("prompt", "user input")

        assert tracer.input_ids == ["input:prompt"]
        assert "input:prompt" in tracer.nodes

        node = tracer.nodes[proxy.node_id]
        assert isinstance(node, GraphNode)
        assert isinstance(node.module, InputNode)
        assert node.module.value == "user input"
        assert node.module_name == "Input(prompt)"
        assert node.dependencies == []

    def test_does_not_affect_node_counter(self) -> None:
        """Input nodes do not increment the _node_counter."""
        tracer = Tracer()

        tracer._create_input_node("a", "value1")
        tracer._create_input_node("b", "value2")

        assert tracer._node_counter == 0


class TestRecordCall:
    """Tests for Tracer.record_call()."""

    def test_returns_value_with_correct_ref(self) -> None:
        """record_call returns a Value with correct node ref."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        result = tracer.record_call(module, (), {})

        assert isinstance(result, Value)
        assert result.ref == "LLMInference_1"
        assert result.ref in tracer.nodes

    def test_creates_graph_node_with_correct_fields(self) -> None:
        """record_call creates a GraphNode with correct module and metadata."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        result = tracer.record_call(module, (), {})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        assert node.id == result.ref
        assert node.module is module
        assert node.module_name == "LLMInference"

    def test_extracts_dependencies_from_proxy_args_and_kwargs(self) -> None:
        """Dependencies are extracted from Proxy objects in args and kwargs."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        proxy1 = tracer._create_input_node("arg0", "value1")
        proxy2 = tracer._create_input_node("kwarg1", "value2")

        output = tracer.record_call(module, (proxy1,), {"extra": proxy2})
        assert output.ref is not None
        node = tracer.nodes[output.ref]

        assert "input:arg0" in node.dependencies
        assert "input:kwarg1" in node.dependencies

    def test_preserves_literal_values(self) -> None:
        """Literal values in args/kwargs are preserved as-is."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        output = tracer.record_call(module, ("literal_value", 42), {"temperature": 0.7})
        assert output.ref is not None
        node = tracer.nodes[output.ref]

        assert node.args == ("literal_value", 42)
        assert node.kwargs == {"temperature": 0.7}
        assert node.dependencies == []

    def test_replaces_proxy_with_node_refs(self) -> None:
        """Proxy objects are replaced with NodeRef wrappers."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        proxy = tracer._create_input_node("text", "hello")

        output = tracer.record_call(module, (proxy, "literal"), {"ctx": proxy})
        assert output.ref is not None
        node = tracer.nodes[output.ref]

        assert node.args == (NodeRef("input:text"), "literal")
        assert node.kwargs == {"ctx": NodeRef("input:text")}


class TestRecordCallChaining:
    """Tests for chaining multiple record_call invocations."""

    def test_linear_chain(self) -> None:
        """Chained module calls create a linear dependency chain."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("text", "input")
        output1 = tracer.record_call(LLMInference(alias="s1"), (input_proxy,), {})
        output2 = tracer.record_call(LLMInference(alias="s2"), (output1,), {})
        _output3 = tracer.record_call(LLMInference(alias="s3"), (output2,), {})

        assert tracer.nodes["LLMInference_1"].dependencies == ["input:text"]
        assert tracer.nodes["LLMInference_2"].dependencies == ["LLMInference_1"]
        assert tracer.nodes["LLMInference_3"].dependencies == ["LLMInference_2"]

    def test_fan_out_pattern(self) -> None:
        """Multiple modules depending on the same input (fan-out)."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("text", "input")
        out_a = tracer.record_call(LLMInference(alias="a"), (input_proxy,), {})
        out_b = tracer.record_call(LLMInference(alias="b"), (input_proxy,), {})

        assert out_a.ref is not None
        assert out_b.ref is not None
        assert tracer.nodes[out_a.ref].dependencies == ["input:text"]
        assert tracer.nodes[out_b.ref].dependencies == ["input:text"]

    def test_fan_in_pattern(self) -> None:
        """One module depending on multiple inputs (fan-in)."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("text", "input")
        out_a = tracer.record_call(LLMInference(alias="a"), (input_proxy,), {})
        out_b = tracer.record_call(LLMInference(alias="b"), (input_proxy,), {})
        merged = tracer.record_call(LLMInference(alias="merge"), (out_a, out_b), {})

        assert merged.ref is not None
        merge_deps = tracer.nodes[merged.ref].dependencies
        assert set(merge_deps) == {"LLMInference_1", "LLMInference_2"}


class TestRecordCallBranchAndPath:
    """Tests for record_call with branch context and module path."""

    def test_no_branch_context(self) -> None:
        """Node has no branch info when not in a branch context."""
        tracer = Tracer()
        result = tracer.record_call(LLMInference(alias="test"), (), {})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        assert node.branch_condition is None
        assert node.branch_value is None

    @pytest.mark.parametrize(
        "branch_value", [True, False], ids=["true_branch", "false_branch"]
    )
    def test_captures_branch_context(self, branch_value: bool) -> None:
        """Node captures branch info when in a branch context."""
        tracer = Tracer()
        tracer._branch_stack.append(("condition_node", branch_value))

        result = tracer.record_call(LLMInference(alias="test"), (), {})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        assert node.branch_condition == "condition_node"
        assert node.branch_value is branch_value

    def test_nested_branch_uses_innermost(self) -> None:
        """Nested branches use the innermost branch context."""
        tracer = Tracer()
        tracer._branch_stack.append(("outer", True))
        tracer._branch_stack.append(("inner", False))

        result = tracer.record_call(LLMInference(alias="test"), (), {})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        assert node.branch_condition == "inner"
        assert node.branch_value is False

    def test_module_path_from_stack(self) -> None:
        """Node has dot-separated module_path from stack."""
        tracer = Tracer()
        tracer._module_stack.extend(["encoder", "layer1", "attention"])

        result = tracer.record_call(LLMInference(alias="test"), (), {})
        assert result.ref is not None
        node = tracer.nodes[result.ref]

        assert node.module_path == "encoder.layer1.attention"


class TestRecordDataAccessOps:
    """Tests for record_getitem, record_iter, and record_method."""

    def test_getitem_creates_node_with_dependency(self) -> None:
        """record_getitem creates a node that depends on the source."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"key": "value"})

        result = tracer.record_getitem(input_proxy, "key")

        assert isinstance(result, Proxy)
        assert "getitem_1" in tracer.nodes
        node = tracer.nodes["getitem_1"]
        assert node.dependencies == ["input:data"]
        assert isinstance(node.module, GetItemOp)
        assert node.module.key == "key"
        assert node.module_name == "getitem['key']"

    def test_getitem_chaining(self) -> None:
        """Multiple getitem calls can be chained."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": {"b": "c"}})

        first = tracer.record_getitem(input_proxy, "a")
        second = tracer.record_getitem(first, "b")

        assert tracer.nodes["getitem_1"].dependencies == ["input:data"]
        assert tracer.nodes["getitem_2"].dependencies == ["getitem_1"]
        assert second.node_id == "getitem_2"

    def test_iter_creates_node_with_dependency(self) -> None:
        """record_iter creates a node that depends on the source."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", [1, 2, 3])

        result = tracer.record_iter(input_proxy)

        assert isinstance(result, Proxy)
        assert "iter_1" in tracer.nodes
        node = tracer.nodes["iter_1"]
        assert node.dependencies == ["input:data"]
        assert isinstance(node.module, IterOp)
        assert node.module_name == "iter"

    @pytest.mark.parametrize("method", ["keys", "values", "items"])
    def test_method_creates_node_with_dependency(self, method: str) -> None:
        """record_method creates a node for dict methods."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        result = tracer.record_method(input_proxy, method)

        assert isinstance(result, Proxy)
        assert "method_1" in tracer.nodes
        node = tracer.nodes["method_1"]
        assert node.dependencies == ["input:data"]
        assert isinstance(node.module, MethodOp)
        assert node.module.method == method
        assert node.module_name == f".{method}()"


class TestCollectAndCaptureOutputs:
    """Tests for _collect_output_ids and _capture_output_structure."""

    def test_collect_from_single_proxy(self) -> None:
        """Collects node ID from a single Proxy."""
        tracer = Tracer()
        proxy = tracer._create_input_node("text", "value")

        assert tracer._collect_output_ids(proxy) == ["input:text"]

    @pytest.mark.parametrize(
        "structure_factory,expected_ids",
        [
            (lambda p1, p2: {"x": p1, "y": p2}, {"input:a", "input:b"}),
            (lambda p1, p2: [p1, p2], ["input:a", "input:b"]),
            (lambda p1, p2: (p1, p2), ["input:a", "input:b"]),
        ],
        ids=["dict", "list", "tuple"],
    )
    def test_collect_from_containers(
        self, structure_factory: Any, expected_ids: Any
    ) -> None:
        """Collects node IDs from dict, list, and tuple."""
        tracer = Tracer()
        proxy1 = tracer._create_input_node("a", "val1")
        proxy2 = tracer._create_input_node("b", "val2")

        result = tracer._collect_output_ids(structure_factory(proxy1, proxy2))

        if isinstance(expected_ids, set):
            assert set(result) == expected_ids
        else:
            assert result == expected_ids

    def test_collect_ignores_literals(self) -> None:
        """Ignores literals when collecting from mixed structure."""
        tracer = Tracer()
        proxy = tracer._create_input_node("a", "val1")

        result = tracer._collect_output_ids([proxy, "literal", 42, None])
        assert result == ["input:a"]

    def test_capture_preserves_structure(self) -> None:
        """_capture_output_structure preserves dict keys and list order."""
        tracer = Tracer()
        p1 = tracer._create_input_node("a", "v1")
        p2 = tracer._create_input_node("b", "v2")

        dict_result = tracer._capture_output_structure({"summary": p1, "analysis": p2})
        list_result = tracer._capture_output_structure([p1, p2])

        assert dict_result == {"summary": "input:a", "analysis": "input:b"}
        assert list_result == ["input:a", "input:b"]


class TestTraceMethod:
    """Tests for Tracer.trace()."""

    def test_returns_inference_graph(self) -> None:
        """trace() returns an InferenceGraph."""

        class PassThrough(Module):
            def forward(self, x: str) -> Proxy:
                return x  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(PassThrough(), "input")

        assert isinstance(graph, InferenceGraph)

    def test_creates_input_nodes_for_args_and_kwargs(self) -> None:
        """trace() creates input nodes for positional and keyword arguments."""

        class TwoInputs(Module):
            def forward(self, a: str, *, text: str) -> tuple[Proxy, Proxy]:
                return a, text  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(TwoInputs(), "first", text="hello")

        assert "input:input_0" in graph.input_ids
        assert "input:input_text" in graph.input_ids
        assert graph.nodes["input:input_0"].module.value == "first"  # type: ignore
        assert graph.nodes["input:input_text"].module.value == "hello"  # type: ignore

    def test_collects_outputs(self) -> None:
        """trace() collects outputs from single, list, and dict returns."""

        class DictOutput(Module):
            def forward(self, a: str, b: str) -> dict[str, Proxy]:
                return {"x": a, "y": b}  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(DictOutput(), "first", "second")

        assert "input:input_0" in graph.output_ids
        assert "input:input_1" in graph.output_ids

    def test_captures_output_structure(self) -> None:
        """trace() captures output structure with user keys."""

        class DictOutput(Module):
            def forward(self, a: str, b: str) -> dict[str, Proxy]:
                return {"summary": a, "analysis": b}  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(DictOutput(), "first", "second")

        assert graph.output_structure == {
            "summary": "input:input_0",
            "analysis": "input:input_1",
        }

    def test_resets_state_before_tracing(self) -> None:
        """trace() resets tracer state before starting."""
        tracer = Tracer()
        tracer.nodes["garbage"] = None  # type: ignore
        tracer._node_counter = 99

        class PassThrough(Module):
            def forward(self, x: str) -> Proxy:
                return x  # type: ignore

        graph = tracer.trace(PassThrough(), "input")

        assert "garbage" not in graph.nodes

    def test_sets_and_clears_trace_context(self) -> None:
        """trace() sets trace context during forward and clears after."""
        captured_context: list[Tracer | None] = []

        class ContextCapture(Module):
            def forward(self, x: str) -> Proxy:
                captured_context.append(get_trace_context())
                return x  # type: ignore

        tracer = Tracer()
        tracer.trace(ContextCapture(), "input")

        assert captured_context[0] is tracer
        assert get_trace_context() is None

    def test_collects_parameters_from_module(self) -> None:
        """trace() collects parameters from the module tree."""

        class Inner(Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner_param = Parameter("inner value", description="test")

            def forward(self, x: str) -> str:
                return x

        class Outer(Module):
            def __init__(self) -> None:
                super().__init__()
                self.outer_param = Parameter("outer value", description="test")
                self.inner = Inner()

            def forward(self, x: str) -> Proxy:
                return x  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(Outer(), "input")

        assert "outer_param" in graph.parameters
        assert "inner.inner_param" in graph.parameters

    def test_can_be_called_multiple_times(self) -> None:
        """trace() can be called multiple times on same tracer."""

        class PassThrough(Module):
            def forward(self, x: str) -> Proxy:
                return x  # type: ignore

        tracer = Tracer()
        graph1 = tracer.trace(PassThrough(), "first")
        graph2 = tracer.trace(PassThrough(), "second")

        assert graph1.nodes["input:input_0"].module.value == "first"  # type: ignore
        assert graph2.nodes["input:input_0"].module.value == "second"  # type: ignore


class TestTraceGraphPatterns:
    """Tests for trace() capturing complex graph patterns."""

    def test_linear_chain(self) -> None:
        """trace() captures a linear chain of module calls."""

        class LinearChain(Module):
            def forward(self, x: Any) -> Any:
                ctx = get_trace_context()
                if ctx is None:
                    return x

                out1 = ctx.record_call(LLMInference(alias="s1"), (x,), {})
                out2 = ctx.record_call(LLMInference(alias="s2"), (out1,), {})
                out3 = ctx.record_call(LLMInference(alias="s3"), (out2,), {})
                return out3

        tracer = Tracer()
        graph = tracer.trace(LinearChain(), "input")

        assert len(graph.nodes) == 4  # 1 input + 3 LLM calls
        assert graph.nodes["LLMInference_1"].dependencies == ["input:input_0"]
        assert graph.nodes["LLMInference_2"].dependencies == ["LLMInference_1"]
        assert graph.nodes["LLMInference_3"].dependencies == ["LLMInference_2"]
        assert graph.output_ids == ["LLMInference_3"]

    def test_diamond_pattern(self) -> None:
        """trace() captures diamond dependency pattern."""

        class DiamondPattern(Module):
            def forward(self, x: Any) -> Any:
                ctx = get_trace_context()
                if ctx is None:
                    return x

                out_a = ctx.record_call(LLMInference(alias="a"), (x,), {})
                out_b = ctx.record_call(LLMInference(alias="b"), (x,), {})
                out_merge = ctx.record_call(
                    LLMInference(alias="merge"), (out_a, out_b), {}
                )
                return out_merge

        tracer = Tracer()
        graph = tracer.trace(DiamondPattern(), "input")

        assert len(graph.nodes) == 4
        assert graph.nodes["LLMInference_1"].dependencies == ["input:input_0"]
        assert graph.nodes["LLMInference_2"].dependencies == ["input:input_0"]
        merge_deps = graph.nodes["LLMInference_3"].dependencies
        assert set(merge_deps) == {"LLMInference_1", "LLMInference_2"}


class TestMultipleTracerInstances:
    """Tests for multiple Tracer instances."""

    def test_independent_state(self) -> None:
        """Different tracer instances have independent state."""
        tracer1 = Tracer()
        tracer2 = Tracer()

        tracer1.nodes["node1"] = None  # type: ignore
        tracer1._generate_id(LLMInference(alias="test"))

        assert "node1" not in tracer2.nodes
        assert tracer2._node_counter == 0
        assert tracer2._generate_id(LLMInference(alias="test")) == "LLMInference_1"
