"""Unit tests for the Tracer class."""

from inf_engine.graph import GraphNode
from inf_engine.module import LLMInference
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
