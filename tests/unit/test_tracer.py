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
