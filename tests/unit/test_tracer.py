"""Unit tests for the Tracer class."""

from inf_engine.graph import GraphNode, InferenceGraph
from inf_engine.module import InferenceModule, LLMInference
from inf_engine.parameter import Parameter
from inf_engine.tracing.context import get_trace_context
from inf_engine.tracing.proxy import Proxy
from inf_engine.tracing.tracer import InputNode, Tracer


class TestTracerInstantiation:
    """Tests for Tracer instantiation."""

    def test_tracer_creation(self) -> None:
        """Tracer can be created with no arguments."""
        tracer = Tracer()

        assert tracer is not None
        assert isinstance(tracer, Tracer)

    def test_tracer_has_empty_nodes(self) -> None:
        """New tracer has empty nodes dictionary."""
        tracer = Tracer()

        assert tracer.nodes == {}
        assert len(tracer.nodes) == 0

    def test_tracer_has_empty_input_ids(self) -> None:
        """New tracer has empty input_ids list."""
        tracer = Tracer()

        assert tracer.input_ids == []

    def test_tracer_has_empty_output_ids(self) -> None:
        """New tracer has empty output_ids list."""
        tracer = Tracer()

        assert tracer.output_ids == []

    def test_tracer_node_counter_starts_at_zero(self) -> None:
        """Node counter starts at zero."""
        tracer = Tracer()

        assert tracer._node_counter == 0

    def test_tracer_has_empty_module_stack(self) -> None:
        """New tracer has empty module stack."""
        tracer = Tracer()

        assert tracer._module_stack == []

    def test_tracer_has_empty_branch_stack(self) -> None:
        """New tracer has empty branch stack."""
        tracer = Tracer()

        assert tracer._branch_stack == []


class TestTracerIdGeneration:
    """Tests for Tracer._generate_id()."""

    def test_generate_id_format(self) -> None:
        """Generated ID has format 'ClassName_N'."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        node_id = tracer._generate_id(module)

        assert node_id == "LLMInference_1"

    def test_generate_id_increments_counter(self) -> None:
        """Each call to _generate_id increments the counter."""
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
        tracer = Tracer()

        # Create a custom module subclass
        from inf_engine.module import InferenceModule

        class CustomModule(InferenceModule):
            def forward(self, x: str) -> str:
                return x

        module = CustomModule()
        node_id = tracer._generate_id(module)

        assert node_id == "CustomModule_1"

    def test_generate_id_different_modules(self) -> None:
        """Different module types get their own IDs with shared counter."""
        tracer = Tracer()

        from inf_engine.module import InferenceModule

        class ModuleA(InferenceModule):
            def forward(self, x: str) -> str:
                return x

        class ModuleB(InferenceModule):
            def forward(self, x: str) -> str:
                return x

        id1 = tracer._generate_id(ModuleA())
        id2 = tracer._generate_id(ModuleB())
        id3 = tracer._generate_id(LLMInference(alias="test"))

        assert id1 == "ModuleA_1"
        assert id2 == "ModuleB_2"
        assert id3 == "LLMInference_3"

    def test_generate_id_counter_persists(self) -> None:
        """Counter persists across multiple calls with different modules."""
        tracer = Tracer()
        module1 = LLMInference(alias="a")
        module2 = LLMInference(alias="b")

        tracer._generate_id(module1)
        tracer._generate_id(module2)
        tracer._generate_id(module1)  # Same module again

        assert tracer._node_counter == 3


class TestTracerReset:
    """Tests for Tracer.reset()."""

    def test_reset_clears_nodes(self) -> None:
        """Reset clears the nodes dictionary."""
        tracer = Tracer()
        tracer.nodes["test_node"] = None  # type: ignore

        tracer.reset()

        assert tracer.nodes == {}

    def test_reset_clears_input_ids(self) -> None:
        """Reset clears the input_ids list."""
        tracer = Tracer()
        tracer.input_ids.append("input_0")

        tracer.reset()

        assert tracer.input_ids == []

    def test_reset_clears_output_ids(self) -> None:
        """Reset clears the output_ids list."""
        tracer = Tracer()
        tracer.output_ids.append("output_0")

        tracer.reset()

        assert tracer.output_ids == []

    def test_reset_resets_node_counter(self) -> None:
        """Reset sets node counter back to zero."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        tracer._generate_id(module)
        tracer._generate_id(module)
        assert tracer._node_counter == 2

        tracer.reset()

        assert tracer._node_counter == 0

    def test_reset_clears_module_stack(self) -> None:
        """Reset clears the module stack."""
        tracer = Tracer()
        tracer._module_stack.append("parent")
        tracer._module_stack.append("child")

        tracer.reset()

        assert tracer._module_stack == []

    def test_reset_clears_branch_stack(self) -> None:
        """Reset clears the branch stack."""
        tracer = Tracer()
        tracer._branch_stack.append(("condition_1", True))

        tracer.reset()

        assert tracer._branch_stack == []

    def test_reset_allows_fresh_tracing(self) -> None:
        """After reset, tracer can be used for a fresh trace."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        # Simulate some tracing
        tracer._generate_id(module)
        tracer.nodes["LLMInference_1"] = None  # type: ignore
        tracer.input_ids.append("input_0")
        tracer.output_ids.append("LLMInference_1")

        # Reset and verify fresh state
        tracer.reset()

        # Generate IDs again - should start from 1
        new_id = tracer._generate_id(module)
        assert new_id == "LLMInference_1"
        assert tracer.nodes == {}
        assert tracer.input_ids == []
        assert tracer.output_ids == []


class TestTracerMultipleInstances:
    """Tests for multiple Tracer instances."""

    def test_tracers_have_independent_nodes(self) -> None:
        """Different tracer instances have independent node dictionaries."""
        tracer1 = Tracer()
        tracer2 = Tracer()

        tracer1.nodes["node1"] = None  # type: ignore

        assert "node1" in tracer1.nodes
        assert "node1" not in tracer2.nodes

    def test_tracers_have_independent_counters(self) -> None:
        """Different tracer instances have independent counters."""
        tracer1 = Tracer()
        tracer2 = Tracer()
        module = LLMInference(alias="test")

        tracer1._generate_id(module)
        tracer1._generate_id(module)

        # tracer2 counter should still be at 0
        id1 = tracer2._generate_id(module)
        assert id1 == "LLMInference_1"
        assert tracer2._node_counter == 1
        assert tracer1._node_counter == 2

    def test_tracers_have_independent_stacks(self) -> None:
        """Different tracer instances have independent stacks."""
        tracer1 = Tracer()
        tracer2 = Tracer()

        tracer1._module_stack.append("module1")
        tracer1._branch_stack.append(("cond", True))

        assert tracer2._module_stack == []
        assert tracer2._branch_stack == []


class TestInputNode:
    """Tests for the InputNode class."""

    def test_input_node_creation(self) -> None:
        """InputNode can be created with a value."""
        node = InputNode(value="test input")

        assert node.value == "test input"

    def test_input_node_stores_any_type(self) -> None:
        """InputNode can store values of any type."""
        string_node = InputNode(value="text")
        int_node = InputNode(value=42)
        dict_node = InputNode(value={"key": "value"})
        list_node = InputNode(value=[1, 2, 3])
        none_node = InputNode(value=None)

        assert string_node.value == "text"
        assert int_node.value == 42
        assert dict_node.value == {"key": "value"}
        assert list_node.value == [1, 2, 3]
        assert none_node.value is None

    def test_input_node_is_dataclass(self) -> None:
        """InputNode is a dataclass with expected behavior."""
        node1 = InputNode(value="test")
        node2 = InputNode(value="test")

        # Dataclasses support equality
        assert node1 == node2

        # Different values are not equal
        node3 = InputNode(value="other")
        assert node1 != node3


class TestCreateInputNode:
    """Tests for Tracer._create_input_node()."""

    def test_create_input_node_returns_proxy(self) -> None:
        """_create_input_node returns a Proxy object."""
        tracer = Tracer()

        proxy = tracer._create_input_node("text", "input value")

        assert isinstance(proxy, Proxy)

    def test_create_input_node_id_format(self) -> None:
        """Input node IDs have the format 'input:{name}'."""
        tracer = Tracer()

        proxy = tracer._create_input_node("text", "value")

        assert proxy.node_id == "input:text"

    def test_create_input_node_adds_to_input_ids(self) -> None:
        """Created input nodes are added to input_ids list."""
        tracer = Tracer()

        tracer._create_input_node("first", "value1")
        tracer._create_input_node("second", "value2")

        assert tracer.input_ids == ["input:first", "input:second"]

    def test_create_input_node_creates_graph_node(self) -> None:
        """_create_input_node creates a GraphNode in nodes dict."""
        tracer = Tracer()

        proxy = tracer._create_input_node("text", "hello")

        assert proxy.node_id in tracer.nodes
        node = tracer.nodes[proxy.node_id]
        assert isinstance(node, GraphNode)

    def test_create_input_node_graph_node_has_correct_fields(self) -> None:
        """Created GraphNode has correct field values."""
        tracer = Tracer()

        proxy = tracer._create_input_node("prompt", "user input")
        node = tracer.nodes[proxy.node_id]

        assert node.id == "input:prompt"
        assert isinstance(node.module, InputNode)
        assert node.args == ()
        assert node.kwargs == {}
        assert node.dependencies == []
        assert node.module_name == "Input(prompt)"

    def test_create_input_node_stores_value_in_input_node(self) -> None:
        """The input value is stored in the InputNode module."""
        tracer = Tracer()

        proxy = tracer._create_input_node("data", {"key": "value"})
        node = tracer.nodes[proxy.node_id]

        assert isinstance(node.module, InputNode)
        assert node.module.value == {"key": "value"}

    def test_create_input_node_proxy_references_tracer(self) -> None:
        """Returned proxy references the correct tracer."""
        tracer = Tracer()

        proxy = tracer._create_input_node("text", "value")

        assert proxy.tracer is tracer

    def test_create_input_node_with_positional_args_convention(self) -> None:
        """Input nodes for positional args follow 'input_N' naming convention."""
        tracer = Tracer()

        proxy0 = tracer._create_input_node("input_0", "first arg")
        proxy1 = tracer._create_input_node("input_1", "second arg")

        assert proxy0.node_id == "input:input_0"
        assert proxy1.node_id == "input:input_1"
        assert tracer.input_ids == ["input:input_0", "input:input_1"]

    def test_create_input_node_with_kwarg_convention(self) -> None:
        """Input nodes for kwargs use the kwarg name."""
        tracer = Tracer()

        proxy = tracer._create_input_node("input_temperature", 0.7)

        assert proxy.node_id == "input:input_temperature"
        module = tracer.nodes[proxy.node_id].module
        assert isinstance(module, InputNode)
        assert module.value == 0.7

    def test_create_multiple_input_nodes(self) -> None:
        """Multiple input nodes can be created and tracked."""
        tracer = Tracer()

        tracer._create_input_node("text", "Hello")
        tracer._create_input_node("context", {"user": "alice"})

        assert len(tracer.nodes) == 2
        assert len(tracer.input_ids) == 2
        text_module = tracer.nodes["input:text"].module
        context_module = tracer.nodes["input:context"].module
        assert isinstance(text_module, InputNode)
        assert isinstance(context_module, InputNode)
        assert text_module.value == "Hello"
        assert context_module.value == {"user": "alice"}

    def test_create_input_node_does_not_affect_node_counter(self) -> None:
        """Input nodes do not increment the _node_counter."""
        tracer = Tracer()

        tracer._create_input_node("text", "value")
        tracer._create_input_node("other", "value2")

        # Counter should still be 0 - only _generate_id increments it
        assert tracer._node_counter == 0

    def test_reset_clears_input_nodes(self) -> None:
        """Tracer.reset() clears created input nodes."""
        tracer = Tracer()

        tracer._create_input_node("text", "value")
        assert len(tracer.nodes) == 1
        assert len(tracer.input_ids) == 1

        tracer.reset()

        assert len(tracer.nodes) == 0
        assert len(tracer.input_ids) == 0


class TestRecordCall:
    """Tests for Tracer.record_call()."""

    def test_record_call_returns_proxy(self) -> None:
        """record_call returns a Proxy object."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        result = tracer.record_call(module, (), {})

        assert isinstance(result, Proxy)

    def test_record_call_proxy_has_correct_node_id(self) -> None:
        """Returned proxy has the correct node_id."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        proxy = tracer.record_call(module, (), {})

        assert proxy.node_id == "LLMInference_1"

    def test_record_call_proxy_references_tracer(self) -> None:
        """Returned proxy references the correct tracer."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        proxy = tracer.record_call(module, (), {})

        assert proxy.tracer is tracer

    def test_record_call_creates_graph_node(self) -> None:
        """record_call creates a GraphNode in the nodes dict."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        proxy = tracer.record_call(module, (), {})

        assert proxy.node_id in tracer.nodes
        node = tracer.nodes[proxy.node_id]
        assert isinstance(node, GraphNode)

    def test_record_call_node_has_correct_module(self) -> None:
        """Created GraphNode stores the correct module reference."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        proxy = tracer.record_call(module, (), {})
        node = tracer.nodes[proxy.node_id]

        assert node.module is module

    def test_record_call_node_has_correct_id(self) -> None:
        """Created GraphNode has the same ID as the proxy."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        proxy = tracer.record_call(module, (), {})
        node = tracer.nodes[proxy.node_id]

        assert node.id == proxy.node_id

    def test_record_call_extracts_dependencies_from_proxy_args(self) -> None:
        """Dependencies are extracted from Proxy objects in args."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        input_proxy = tracer._create_input_node("text", "hello")

        output_proxy = tracer.record_call(module, (input_proxy,), {})
        node = tracer.nodes[output_proxy.node_id]

        assert node.dependencies == ["input:text"]

    def test_record_call_extracts_dependencies_from_proxy_kwargs(self) -> None:
        """Dependencies are extracted from Proxy objects in kwargs."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        input_proxy = tracer._create_input_node("text", "hello")

        output_proxy = tracer.record_call(module, (), {"text": input_proxy})
        node = tracer.nodes[output_proxy.node_id]

        assert node.dependencies == ["input:text"]

    def test_record_call_extracts_dependencies_from_both_args_and_kwargs(self) -> None:
        """Dependencies are extracted from both args and kwargs."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        proxy1 = tracer._create_input_node("arg0", "value1")
        proxy2 = tracer._create_input_node("kwarg1", "value2")

        output_proxy = tracer.record_call(module, (proxy1,), {"extra": proxy2})
        node = tracer.nodes[output_proxy.node_id]

        assert "input:arg0" in node.dependencies
        assert "input:kwarg1" in node.dependencies
        assert len(node.dependencies) == 2

    def test_record_call_preserves_literal_args(self) -> None:
        """Literal values in args are preserved as-is."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        output_proxy = tracer.record_call(module, ("literal_value", 42), {})
        node = tracer.nodes[output_proxy.node_id]

        assert node.args == ("literal_value", 42)
        assert node.dependencies == []

    def test_record_call_preserves_literal_kwargs(self) -> None:
        """Literal values in kwargs are preserved as-is."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        output_proxy = tracer.record_call(
            module, (), {"temperature": 0.7, "max_tokens": 100}
        )
        node = tracer.nodes[output_proxy.node_id]

        assert node.kwargs == {"temperature": 0.7, "max_tokens": 100}
        assert node.dependencies == []

    def test_record_call_replaces_proxy_args_with_node_ids(self) -> None:
        """Proxy objects in args are replaced with their node_ids."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        input_proxy = tracer._create_input_node("text", "hello")

        output_proxy = tracer.record_call(module, (input_proxy, "literal"), {})
        node = tracer.nodes[output_proxy.node_id]

        assert node.args == ("input:text", "literal")

    def test_record_call_replaces_proxy_kwargs_with_node_ids(self) -> None:
        """Proxy objects in kwargs are replaced with their node_ids."""
        tracer = Tracer()
        module = LLMInference(alias="test")
        input_proxy = tracer._create_input_node("context", {"key": "value"})

        output_proxy = tracer.record_call(
            module, (), {"context": input_proxy, "temp": 0.5}
        )
        node = tracer.nodes[output_proxy.node_id]

        assert node.kwargs == {"context": "input:context", "temp": 0.5}

    def test_record_call_multiple_nodes_increments_id(self) -> None:
        """Multiple record_call creates nodes with incrementing IDs."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        proxy1 = tracer.record_call(module, (), {})
        proxy2 = tracer.record_call(module, (), {})
        proxy3 = tracer.record_call(module, (), {})

        assert proxy1.node_id == "LLMInference_1"
        assert proxy2.node_id == "LLMInference_2"
        assert proxy3.node_id == "LLMInference_3"
        assert len(tracer.nodes) == 3

    def test_record_call_node_has_empty_dependencies_without_proxies(self) -> None:
        """Node has empty dependencies list when no proxies are passed."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        proxy = tracer.record_call(module, ("literal",), {"key": "value"})
        node = tracer.nodes[proxy.node_id]

        assert node.dependencies == []

    def test_record_call_node_module_name_is_set(self) -> None:
        """Created GraphNode has correct module_name."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        proxy = tracer.record_call(module, (), {})
        node = tracer.nodes[proxy.node_id]

        assert node.module_name == "LLMInference"

    def test_record_call_with_custom_module_class(self) -> None:
        """record_call works with custom module subclasses."""
        from inf_engine.module import InferenceModule

        class CustomProcessor(InferenceModule):
            def forward(self, x: str) -> str:
                return x.upper()

        tracer = Tracer()
        module = CustomProcessor()

        proxy = tracer.record_call(module, (), {})
        node = tracer.nodes[proxy.node_id]

        assert proxy.node_id == "CustomProcessor_1"
        assert node.module_name == "CustomProcessor"
        assert node.module is module


class TestRecordCallChaining:
    """Tests for chaining multiple record_call invocations."""

    def test_chained_calls_create_dependency_chain(self) -> None:
        """Chained module calls create a dependency chain."""
        tracer = Tracer()
        module1 = LLMInference(alias="step1")
        module2 = LLMInference(alias="step2")
        module3 = LLMInference(alias="step3")

        input_proxy = tracer._create_input_node("text", "input")
        output1 = tracer.record_call(module1, (input_proxy,), {})
        output2 = tracer.record_call(module2, (output1,), {})
        _output3 = tracer.record_call(module3, (output2,), {})

        # Verify the dependency chain
        assert tracer.nodes["LLMInference_1"].dependencies == ["input:text"]
        assert tracer.nodes["LLMInference_2"].dependencies == ["LLMInference_1"]
        assert tracer.nodes["LLMInference_3"].dependencies == ["LLMInference_2"]

    def test_fan_out_pattern(self) -> None:
        """Multiple modules depending on the same input (fan-out)."""
        tracer = Tracer()
        module_a = LLMInference(alias="a")
        module_b = LLMInference(alias="b")
        module_c = LLMInference(alias="c")

        input_proxy = tracer._create_input_node("text", "input")
        output_a = tracer.record_call(module_a, (input_proxy,), {})
        output_b = tracer.record_call(module_b, (input_proxy,), {})
        output_c = tracer.record_call(module_c, (input_proxy,), {})

        # All three depend on the same input
        assert tracer.nodes[output_a.node_id].dependencies == ["input:text"]
        assert tracer.nodes[output_b.node_id].dependencies == ["input:text"]
        assert tracer.nodes[output_c.node_id].dependencies == ["input:text"]

    def test_fan_in_pattern(self) -> None:
        """One module depending on multiple inputs (fan-in)."""
        tracer = Tracer()
        module_a = LLMInference(alias="a")
        module_b = LLMInference(alias="b")
        module_merge = LLMInference(alias="merge")

        input_proxy = tracer._create_input_node("text", "input")
        output_a = tracer.record_call(module_a, (input_proxy,), {})
        output_b = tracer.record_call(module_b, (input_proxy,), {})
        merged = tracer.record_call(module_merge, (output_a, output_b), {})

        # Merge depends on both a and b
        merge_deps = tracer.nodes[merged.node_id].dependencies
        assert "LLMInference_1" in merge_deps
        assert "LLMInference_2" in merge_deps
        assert len(merge_deps) == 2

    def test_diamond_pattern(self) -> None:
        """Diamond dependency pattern: input -> [a, b] -> merge."""
        tracer = Tracer()

        input_proxy = tracer._create_input_node("text", "input")
        module_a = LLMInference(alias="a")
        module_b = LLMInference(alias="b")
        module_merge = LLMInference(alias="merge")

        output_a = tracer.record_call(module_a, (input_proxy,), {})
        output_b = tracer.record_call(module_b, (input_proxy,), {})
        _merged = tracer.record_call(module_merge, (output_a, output_b), {})

        # Total nodes: 1 input + 3 modules = 4
        assert len(tracer.nodes) == 4

        # Verify structure
        assert tracer.nodes["LLMInference_1"].dependencies == ["input:text"]
        assert tracer.nodes["LLMInference_2"].dependencies == ["input:text"]
        merge_deps = tracer.nodes["LLMInference_3"].dependencies
        assert set(merge_deps) == {"LLMInference_1", "LLMInference_2"}


class TestRecordCallBranchContext:
    """Tests for record_call with branch context."""

    def test_record_call_without_branch_context(self) -> None:
        """Node has no branch info when not in a branch context."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        proxy = tracer.record_call(module, (), {})
        node = tracer.nodes[proxy.node_id]

        assert node.branch_condition is None
        assert node.branch_value is None

    def test_record_call_with_branch_context(self) -> None:
        """Node captures branch info when in a branch context."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        # Simulate being in a branch context
        tracer._branch_stack.append(("condition_node_1", True))

        proxy = tracer.record_call(module, (), {})
        node = tracer.nodes[proxy.node_id]

        assert node.branch_condition == "condition_node_1"
        assert node.branch_value is True

    def test_record_call_with_false_branch(self) -> None:
        """Node captures False branch value correctly."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        tracer._branch_stack.append(("cond_proxy", False))

        proxy = tracer.record_call(module, (), {})
        node = tracer.nodes[proxy.node_id]

        assert node.branch_condition == "cond_proxy"
        assert node.branch_value is False

    def test_record_call_nested_branch_uses_innermost(self) -> None:
        """Nested branches use the innermost branch context."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        # Push two branch contexts
        tracer._branch_stack.append(("outer_cond", True))
        tracer._branch_stack.append(("inner_cond", False))

        proxy = tracer.record_call(module, (), {})
        node = tracer.nodes[proxy.node_id]

        # Should use the innermost (last) branch
        assert node.branch_condition == "inner_cond"
        assert node.branch_value is False


class TestRecordCallModulePath:
    """Tests for record_call with module path tracking."""

    def test_record_call_without_module_stack(self) -> None:
        """Node has empty module_path when module_stack is empty."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        proxy = tracer.record_call(module, (), {})
        node = tracer.nodes[proxy.node_id]

        assert node.module_path == ""

    def test_record_call_with_single_module_in_stack(self) -> None:
        """Node has correct module_path with one module in stack."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        tracer._module_stack.append("parent")

        proxy = tracer.record_call(module, (), {})
        node = tracer.nodes[proxy.node_id]

        assert node.module_path == "parent"

    def test_record_call_with_nested_module_stack(self) -> None:
        """Node has dot-separated module_path with nested stack."""
        tracer = Tracer()
        module = LLMInference(alias="test")

        tracer._module_stack.append("encoder")
        tracer._module_stack.append("layer1")
        tracer._module_stack.append("attention")

        proxy = tracer.record_call(module, (), {})
        node = tracer.nodes[proxy.node_id]

        assert node.module_path == "encoder.layer1.attention"


class TestRecordGetitem:
    """Tests for Tracer.record_getitem()."""

    def test_record_getitem_returns_proxy(self) -> None:
        """record_getitem returns a Proxy."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"key": "value"})

        result = tracer.record_getitem(input_proxy, "key")

        assert isinstance(result, Proxy)

    def test_record_getitem_creates_node(self) -> None:
        """record_getitem creates a new node in the graph."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"key": "value"})

        tracer.record_getitem(input_proxy, "key")

        assert "getitem_1" in tracer.nodes

    def test_record_getitem_node_has_correct_dependencies(self) -> None:
        """Getitem node depends on the source node."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"key": "value"})

        tracer.record_getitem(input_proxy, "key")

        node = tracer.nodes["getitem_1"]
        assert node.dependencies == ["input:data"]

    def test_record_getitem_stores_key(self) -> None:
        """Getitem node stores the key used."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"key": "value"})

        tracer.record_getitem(input_proxy, "my_key")

        from inf_engine.tracing.tracer import GetItemOp

        node = tracer.nodes["getitem_1"]
        assert isinstance(node.module, GetItemOp)
        assert node.module.key == "my_key"

    def test_record_getitem_with_integer_key(self) -> None:
        """record_getitem works with integer keys."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", [1, 2, 3])

        tracer.record_getitem(input_proxy, 0)

        from inf_engine.tracing.tracer import GetItemOp

        node = tracer.nodes["getitem_1"]
        assert isinstance(node.module, GetItemOp)
        assert node.module.key == 0

    def test_record_getitem_chaining(self) -> None:
        """Multiple getitem calls can be chained."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": {"b": "c"}})

        first = tracer.record_getitem(input_proxy, "a")
        second = tracer.record_getitem(first, "b")

        assert "getitem_1" in tracer.nodes
        assert "getitem_2" in tracer.nodes
        assert tracer.nodes["getitem_1"].dependencies == ["input:data"]
        assert tracer.nodes["getitem_2"].dependencies == ["getitem_1"]
        assert second.node_id == "getitem_2"

    def test_record_getitem_has_module_name(self) -> None:
        """Getitem node has descriptive module_name."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"key": "value"})

        tracer.record_getitem(input_proxy, "my_key")

        node = tracer.nodes["getitem_1"]
        assert node.module_name == "getitem['my_key']"

    def test_record_getitem_stores_source_in_args(self) -> None:
        """Getitem node stores source node_id in args."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"key": "value"})

        tracer.record_getitem(input_proxy, "key")

        node = tracer.nodes["getitem_1"]
        assert node.args == ("input:data",)


class TestRecordIter:
    """Tests for Tracer.record_iter()."""

    def test_record_iter_returns_proxy(self) -> None:
        """record_iter returns a Proxy."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", [1, 2, 3])

        result = tracer.record_iter(input_proxy)

        assert isinstance(result, Proxy)

    def test_record_iter_creates_node(self) -> None:
        """record_iter creates a new node in the graph."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", [1, 2, 3])

        tracer.record_iter(input_proxy)

        assert "iter_1" in tracer.nodes

    def test_record_iter_node_has_correct_dependencies(self) -> None:
        """Iter node depends on the source node."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", [1, 2, 3])

        tracer.record_iter(input_proxy)

        node = tracer.nodes["iter_1"]
        assert node.dependencies == ["input:data"]

    def test_record_iter_uses_iter_op(self) -> None:
        """Iter node uses IterOp module."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", [1, 2, 3])

        tracer.record_iter(input_proxy)

        from inf_engine.tracing.tracer import IterOp

        node = tracer.nodes["iter_1"]
        assert isinstance(node.module, IterOp)

    def test_record_iter_has_module_name(self) -> None:
        """Iter node has descriptive module_name."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", [1, 2, 3])

        tracer.record_iter(input_proxy)

        node = tracer.nodes["iter_1"]
        assert node.module_name == "iter"


class TestRecordMethod:
    """Tests for Tracer.record_method()."""

    def test_record_method_returns_proxy(self) -> None:
        """record_method returns a Proxy."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        result = tracer.record_method(input_proxy, "keys")

        assert isinstance(result, Proxy)

    def test_record_method_creates_node(self) -> None:
        """record_method creates a new node in the graph."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        tracer.record_method(input_proxy, "keys")

        assert "method_1" in tracer.nodes

    def test_record_method_node_has_correct_dependencies(self) -> None:
        """Method node depends on the source node."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        tracer.record_method(input_proxy, "values")

        node = tracer.nodes["method_1"]
        assert node.dependencies == ["input:data"]

    def test_record_method_stores_method_name(self) -> None:
        """Method node stores the method name."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        tracer.record_method(input_proxy, "items")

        from inf_engine.tracing.tracer import MethodOp

        node = tracer.nodes["method_1"]
        assert isinstance(node.module, MethodOp)
        assert node.module.method == "items"

    def test_record_method_keys(self) -> None:
        """record_method works with 'keys'."""
        from inf_engine.tracing.tracer import MethodOp

        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        tracer.record_method(input_proxy, "keys")

        node = tracer.nodes["method_1"]
        assert isinstance(node.module, MethodOp)
        assert node.module.method == "keys"
        assert node.module_name == ".keys()"

    def test_record_method_values(self) -> None:
        """record_method works with 'values'."""
        from inf_engine.tracing.tracer import MethodOp

        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        tracer.record_method(input_proxy, "values")

        node = tracer.nodes["method_1"]
        assert isinstance(node.module, MethodOp)
        assert node.module.method == "values"
        assert node.module_name == ".values()"

    def test_record_method_items(self) -> None:
        """record_method works with 'items'."""
        from inf_engine.tracing.tracer import MethodOp

        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        tracer.record_method(input_proxy, "items")

        node = tracer.nodes["method_1"]
        assert isinstance(node.module, MethodOp)
        assert node.module.method == "items"
        assert node.module_name == ".items()"

    def test_record_method_multiple_calls(self) -> None:
        """Multiple method calls create separate nodes."""
        tracer = Tracer()
        input_proxy = tracer._create_input_node("data", {"a": 1})

        tracer.record_method(input_proxy, "keys")
        tracer.record_method(input_proxy, "values")
        tracer.record_method(input_proxy, "items")

        assert "method_1" in tracer.nodes
        assert "method_2" in tracer.nodes
        assert "method_3" in tracer.nodes


class TestRecordOperationBranchContext:
    """Tests for data access operations with branch context."""

    def test_record_getitem_with_branch_context(self) -> None:
        """Getitem node captures branch info when in a branch context."""
        tracer = Tracer()
        tracer._branch_stack.append(("condition_1", True))
        input_proxy = tracer._create_input_node("data", {"key": "value"})

        tracer.record_getitem(input_proxy, "key")

        node = tracer.nodes["getitem_1"]
        assert node.branch_condition == "condition_1"
        assert node.branch_value is True

    def test_record_iter_with_branch_context(self) -> None:
        """Iter node captures branch info when in a branch context."""
        tracer = Tracer()
        tracer._branch_stack.append(("condition_2", False))
        input_proxy = tracer._create_input_node("data", [1, 2])

        tracer.record_iter(input_proxy)

        node = tracer.nodes["iter_1"]
        assert node.branch_condition == "condition_2"
        assert node.branch_value is False

    def test_record_method_with_branch_context(self) -> None:
        """Method node captures branch info when in a branch context."""
        tracer = Tracer()
        tracer._branch_stack.append(("condition_3", True))
        input_proxy = tracer._create_input_node("data", {"a": 1})

        tracer.record_method(input_proxy, "keys")

        node = tracer.nodes["method_1"]
        assert node.branch_condition == "condition_3"
        assert node.branch_value is True


class TestCollectOutputIds:
    """Tests for Tracer._collect_output_ids()."""

    def test_collect_single_proxy(self) -> None:
        """Collects node ID from a single Proxy."""
        tracer = Tracer()
        proxy = tracer._create_input_node("text", "value")

        result = tracer._collect_output_ids(proxy)

        assert result == ["input:text"]

    def test_collect_from_dict(self) -> None:
        """Collects node IDs from dict values."""
        tracer = Tracer()
        proxy1 = tracer._create_input_node("a", "val1")
        proxy2 = tracer._create_input_node("b", "val2")

        result = tracer._collect_output_ids({"x": proxy1, "y": proxy2})

        assert "input:a" in result
        assert "input:b" in result
        assert len(result) == 2

    def test_collect_from_list(self) -> None:
        """Collects node IDs from list items."""
        tracer = Tracer()
        proxy1 = tracer._create_input_node("a", "val1")
        proxy2 = tracer._create_input_node("b", "val2")

        result = tracer._collect_output_ids([proxy1, proxy2])

        assert result == ["input:a", "input:b"]

    def test_collect_from_tuple(self) -> None:
        """Collects node IDs from tuple items."""
        tracer = Tracer()
        proxy1 = tracer._create_input_node("a", "val1")
        proxy2 = tracer._create_input_node("b", "val2")

        result = tracer._collect_output_ids((proxy1, proxy2))

        assert result == ["input:a", "input:b"]

    def test_collect_from_nested_structure(self) -> None:
        """Collects node IDs from nested dict/list structure."""
        tracer = Tracer()
        proxy1 = tracer._create_input_node("a", "val1")
        proxy2 = tracer._create_input_node("b", "val2")
        proxy3 = tracer._create_input_node("c", "val3")

        nested = {"outer": [proxy1, {"inner": proxy2}], "single": proxy3}
        result = tracer._collect_output_ids(nested)

        assert "input:a" in result
        assert "input:b" in result
        assert "input:c" in result
        assert len(result) == 3

    def test_collect_from_literal_returns_empty(self) -> None:
        """Returns empty list for literal values."""
        tracer = Tracer()

        assert tracer._collect_output_ids("string") == []
        assert tracer._collect_output_ids(42) == []
        assert tracer._collect_output_ids(None) == []

    def test_collect_from_empty_dict(self) -> None:
        """Returns empty list for empty dict."""
        tracer = Tracer()

        result = tracer._collect_output_ids({})

        assert result == []

    def test_collect_from_empty_list(self) -> None:
        """Returns empty list for empty list."""
        tracer = Tracer()

        result = tracer._collect_output_ids([])

        assert result == []

    def test_collect_mixed_proxies_and_literals(self) -> None:
        """Ignores literals when collecting from mixed structure."""
        tracer = Tracer()
        proxy = tracer._create_input_node("a", "val1")

        result = tracer._collect_output_ids([proxy, "literal", 42, None])

        assert result == ["input:a"]


class TestTraceMethod:
    """Tests for Tracer.trace()."""

    def test_trace_returns_inference_graph(self) -> None:
        """trace() returns an InferenceGraph."""

        class PassThrough(InferenceModule):
            def forward(self, x: str) -> Proxy:
                return x  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(PassThrough(), "input")

        assert isinstance(graph, InferenceGraph)

    def test_trace_creates_input_node_for_positional_arg(self) -> None:
        """trace() creates input node for each positional argument."""

        class PassThrough(InferenceModule):
            def forward(self, x: str) -> Proxy:
                return x  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(PassThrough(), "input value")

        assert "input:input_0" in graph.input_ids
        assert "input:input_0" in graph.nodes
        node = graph.nodes["input:input_0"]
        assert isinstance(node.module, InputNode)
        assert node.module.value == "input value"

    def test_trace_creates_input_nodes_for_multiple_args(self) -> None:
        """trace() creates input nodes for multiple positional arguments."""

        class TwoInputs(InferenceModule):
            def forward(self, a: str, b: str) -> tuple[Proxy, Proxy]:
                return a, b  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(TwoInputs(), "first", "second")

        assert graph.input_ids == ["input:input_0", "input:input_1"]
        assert graph.nodes["input:input_0"].module.value == "first"  # type: ignore
        assert graph.nodes["input:input_1"].module.value == "second"  # type: ignore

    def test_trace_creates_input_nodes_for_kwargs(self) -> None:
        """trace() creates input nodes for keyword arguments."""

        class KwargModule(InferenceModule):
            def forward(self, *, text: str, context: str) -> tuple[Proxy, Proxy]:
                return text, context  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(KwargModule(), text="hello", context="world")

        assert "input:input_text" in graph.input_ids
        assert "input:input_context" in graph.input_ids
        assert graph.nodes["input:input_text"].module.value == "hello"  # type: ignore
        assert graph.nodes["input:input_context"].module.value == "world"  # type: ignore

    def test_trace_collects_single_proxy_output(self) -> None:
        """trace() collects single proxy output."""

        class PassThrough(InferenceModule):
            def forward(self, x: str) -> Proxy:
                return x  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(PassThrough(), "input")

        assert graph.output_ids == ["input:input_0"]

    def test_trace_collects_list_output(self) -> None:
        """trace() collects outputs from list."""

        class ListOutput(InferenceModule):
            def forward(self, a: str, b: str) -> list[Proxy]:
                return [a, b]  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(ListOutput(), "first", "second")

        assert graph.output_ids == ["input:input_0", "input:input_1"]

    def test_trace_collects_dict_output(self) -> None:
        """trace() collects outputs from dict values."""

        class DictOutput(InferenceModule):
            def forward(self, a: str, b: str) -> dict[str, Proxy]:
                return {"x": a, "y": b}  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(DictOutput(), "first", "second")

        assert "input:input_0" in graph.output_ids
        assert "input:input_1" in graph.output_ids

    def test_trace_resets_state_before_tracing(self) -> None:
        """trace() resets tracer state before starting."""
        tracer = Tracer()

        # Pollute tracer state
        tracer.nodes["garbage"] = None  # type: ignore
        tracer.input_ids.append("old_input")
        tracer.output_ids.append("old_output")
        tracer._node_counter = 99

        class PassThrough(InferenceModule):
            def forward(self, x: str) -> Proxy:
                return x  # type: ignore

        graph = tracer.trace(PassThrough(), "input")

        # State should be clean
        assert "garbage" not in graph.nodes
        assert "old_input" not in graph.input_ids
        assert "old_output" not in graph.output_ids

    def test_trace_sets_trace_context(self) -> None:
        """trace() sets trace context during forward execution."""
        captured_context: list[Tracer | None] = []

        class ContextCapture(InferenceModule):
            def forward(self, x: str) -> Proxy:
                captured_context.append(get_trace_context())
                return x  # type: ignore

        tracer = Tracer()
        tracer.trace(ContextCapture(), "input")

        assert captured_context[0] is tracer

    def test_trace_clears_context_after_tracing(self) -> None:
        """trace() clears trace context after completing."""

        class PassThrough(InferenceModule):
            def forward(self, x: str) -> Proxy:
                return x  # type: ignore

        tracer = Tracer()
        tracer.trace(PassThrough(), "input")

        assert get_trace_context() is None

    def test_trace_collects_parameters_from_module(self) -> None:
        """trace() collects parameters from the module tree."""

        class ModuleWithParam(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prompt = Parameter("test prompt")

            def forward(self, x: str) -> Proxy:
                return x  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(ModuleWithParam(), "input")

        assert "prompt" in graph.parameters
        assert graph.parameters["prompt"].value == "test prompt"

    def test_trace_collects_nested_parameters(self) -> None:
        """trace() collects parameters from nested modules."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner_param = Parameter("inner value")

            def forward(self, x: str) -> str:
                return x

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.outer_param = Parameter("outer value")
                self.inner = Inner()

            def forward(self, x: str) -> Proxy:
                return x  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(Outer(), "input")

        assert "outer_param" in graph.parameters
        assert "inner.inner_param" in graph.parameters

    def test_trace_with_module_calling_child(self) -> None:
        """trace() captures calls from parent to child modules."""

        class Child(InferenceModule):
            def forward(self, x: Proxy) -> Proxy:
                # Just return the input - in a real module this would
                # be recorded by record_call when trace context is active
                return x

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = Child()

            def forward(self, x: str) -> Proxy:
                # Forward should pass the proxy through
                return self.child.forward(x)  # type: ignore

        tracer = Tracer()
        graph = tracer.trace(Parent(), "input")

        # Should have the input node
        assert "input:input_0" in graph.nodes
        # The child.forward call returns the same proxy, so output should be the input
        assert graph.output_ids == ["input:input_0"]

    def test_trace_can_be_called_multiple_times(self) -> None:
        """trace() can be called multiple times on same tracer."""

        class PassThrough(InferenceModule):
            def forward(self, x: str) -> Proxy:
                return x  # type: ignore

        tracer = Tracer()

        graph1 = tracer.trace(PassThrough(), "first")
        graph2 = tracer.trace(PassThrough(), "second")

        # Each trace should be independent
        assert graph1.nodes["input:input_0"].module.value == "first"  # type: ignore
        assert graph2.nodes["input:input_0"].module.value == "second"  # type: ignore

    def test_trace_with_empty_inputs(self) -> None:
        """trace() works with no input arguments."""

        class NoInput(InferenceModule):
            def forward(self) -> str:
                return "constant"

        tracer = Tracer()
        graph = tracer.trace(NoInput())

        assert graph.input_ids == []
        assert graph.output_ids == []  # "constant" is not a Proxy

    def test_trace_records_call_via_record_call(self) -> None:
        """trace() records module calls when using record_call directly."""
        tracer = Tracer()

        class ManualRecord(InferenceModule):
            def forward(self, x: Proxy) -> Proxy:
                # Simulate what a module would do when trace context is active
                ctx = get_trace_context()
                if ctx is not None:
                    child = LLMInference(alias="test")
                    return ctx.record_call(child, (x,), {})
                return x

        graph = tracer.trace(ManualRecord(), "input")

        # Should have input node + recorded LLMInference call
        assert len(graph.nodes) == 2
        assert "input:input_0" in graph.nodes
        assert "LLMInference_1" in graph.nodes
        assert graph.nodes["LLMInference_1"].dependencies == ["input:input_0"]
        assert graph.output_ids == ["LLMInference_1"]

    def test_trace_captures_linear_chain(self) -> None:
        """trace() captures a linear chain of module calls."""
        tracer = Tracer()

        class LinearChain(InferenceModule):
            def forward(self, x: Proxy) -> Proxy:
                ctx = get_trace_context()
                if ctx is None:
                    return x

                # Simulate: step1 -> step2 -> step3
                step1 = LLMInference(alias="step1")
                step2 = LLMInference(alias="step2")
                step3 = LLMInference(alias="step3")

                out1 = ctx.record_call(step1, (x,), {})
                out2 = ctx.record_call(step2, (out1,), {})
                out3 = ctx.record_call(step3, (out2,), {})
                return out3

        graph = tracer.trace(LinearChain(), "input")

        # Verify structure
        assert len(graph.nodes) == 4  # 1 input + 3 LLM calls
        assert graph.nodes["LLMInference_1"].dependencies == ["input:input_0"]
        assert graph.nodes["LLMInference_2"].dependencies == ["LLMInference_1"]
        assert graph.nodes["LLMInference_3"].dependencies == ["LLMInference_2"]
        assert graph.output_ids == ["LLMInference_3"]

    def test_trace_captures_diamond_pattern(self) -> None:
        """trace() captures diamond dependency pattern."""
        tracer = Tracer()

        class DiamondPattern(InferenceModule):
            def forward(self, x: Proxy) -> Proxy:
                ctx = get_trace_context()
                if ctx is None:
                    return x

                # Diamond: input -> [branch_a, branch_b] -> merge
                branch_a = LLMInference(alias="a")
                branch_b = LLMInference(alias="b")
                merge = LLMInference(alias="merge")

                out_a = ctx.record_call(branch_a, (x,), {})
                out_b = ctx.record_call(branch_b, (x,), {})
                out_merge = ctx.record_call(merge, (out_a, out_b), {})
                return out_merge

        graph = tracer.trace(DiamondPattern(), "input")

        # Verify diamond structure
        assert len(graph.nodes) == 4  # 1 input + 3 LLM calls

        # Both branches depend on input
        assert graph.nodes["LLMInference_1"].dependencies == ["input:input_0"]
        assert graph.nodes["LLMInference_2"].dependencies == ["input:input_0"]

        # Merge depends on both branches
        merge_deps = graph.nodes["LLMInference_3"].dependencies
        assert "LLMInference_1" in merge_deps
        assert "LLMInference_2" in merge_deps

        assert graph.output_ids == ["LLMInference_3"]
