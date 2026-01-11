"""Unit tests for the ForwardRecord class."""

import pytest

from plait.graph import GraphNode, InferenceGraph
from plait.module import LLMInference, Module
from plait.optimization.record import ForwardRecord
from plait.tracing.tracer import InputNode


class TestForwardRecordCreation:
    """Tests for ForwardRecord instantiation."""

    def test_forward_record_creation(self) -> None:
        """ForwardRecord can be created with all required fields."""
        # Create a simple graph
        input_node = GraphNode(
            id="input:text",
            module=InputNode("hello"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        graph = InferenceGraph(
            nodes={"input:text": input_node},
            input_ids=["input:text"],
            output_ids=["input:text"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={"input:text": {}},
            node_outputs={"input:text": "hello"},
            module_map={},
            execution_order=["input:text"],
            timing={"input:text": 0.001},
        )

        assert record.graph is graph
        assert record.node_inputs == {"input:text": {}}
        assert record.node_outputs == {"input:text": "hello"}
        assert record.module_map == {}
        assert record.execution_order == ["input:text"]
        assert record.timing == {"input:text": 0.001}

    def test_forward_record_default_execution_order(self) -> None:
        """ForwardRecord defaults execution_order to empty list."""
        graph = InferenceGraph(
            nodes={},
            input_ids=[],
            output_ids=[],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        assert record.execution_order == []

    def test_forward_record_default_timing(self) -> None:
        """ForwardRecord defaults timing to empty dict."""
        graph = InferenceGraph(
            nodes={},
            input_ids=[],
            output_ids=[],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        assert record.timing == {}


class TestForwardRecordAccessors:
    """Tests for ForwardRecord accessor methods."""

    def test_get_node_input(self) -> None:
        """get_node_input returns the correct input dict."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        record = ForwardRecord(
            graph=graph,
            node_inputs={
                "node_1": {"arg_0": "hello"},
                "node_2": {"arg_0": "world", "key": "value"},
            },
            node_outputs={},
            module_map={},
        )

        assert record.get_node_input("node_1") == {"arg_0": "hello"}
        assert record.get_node_input("node_2") == {"arg_0": "world", "key": "value"}

    def test_get_node_input_missing_raises(self) -> None:
        """get_node_input raises KeyError for missing node."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        with pytest.raises(KeyError):
            record.get_node_input("nonexistent")

    def test_get_node_output(self) -> None:
        """get_node_output returns the correct output value."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={
                "node_1": "result 1",
                "node_2": {"key": "value"},
            },
            module_map={},
        )

        assert record.get_node_output("node_1") == "result 1"
        assert record.get_node_output("node_2") == {"key": "value"}

    def test_get_node_output_missing_raises(self) -> None:
        """get_node_output raises KeyError for missing node."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        with pytest.raises(KeyError):
            record.get_node_output("nonexistent")

    def test_get_module(self) -> None:
        """get_module returns the correct module instance."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        module = LLMInference(alias="test")
        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={"LLMInference_1": module},
        )

        assert record.get_module("LLMInference_1") is module

    def test_get_module_missing_raises(self) -> None:
        """get_module raises KeyError for missing node."""
        graph = InferenceGraph(nodes={}, input_ids=[], output_ids=[])
        record = ForwardRecord(
            graph=graph,
            node_inputs={},
            node_outputs={},
            module_map={},
        )

        with pytest.raises(KeyError):
            record.get_module("nonexistent")


class TestForwardRecordWithModules:
    """Tests for ForwardRecord with real module instances."""

    def test_forward_record_with_llm_module(self) -> None:
        """ForwardRecord can store LLMInference modules."""
        module = LLMInference(alias="assistant")
        llm_node = GraphNode(
            id="LLMInference_1",
            module=module,
            args=(),
            kwargs={},
            dependencies=["input:text"],
        )
        input_node = GraphNode(
            id="input:text",
            module=InputNode("hello"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        graph = InferenceGraph(
            nodes={"input:text": input_node, "LLMInference_1": llm_node},
            input_ids=["input:text"],
            output_ids=["LLMInference_1"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={
                "input:text": {},
                "LLMInference_1": {"arg_0": "hello"},
            },
            node_outputs={
                "input:text": "hello",
                "LLMInference_1": "response",
            },
            module_map={"LLMInference_1": module},
            execution_order=["input:text", "LLMInference_1"],
            timing={"input:text": 0.0001, "LLMInference_1": 0.5},
        )

        assert record.get_module("LLMInference_1") is module
        assert record.get_node_output("LLMInference_1") == "response"
        assert record.execution_order == ["input:text", "LLMInference_1"]

    def test_forward_record_with_custom_module(self) -> None:
        """ForwardRecord can store custom Module subclasses."""

        class CustomModule(Module):
            def forward(self, x: str) -> str:
                return x.upper()

        module = CustomModule()
        node = GraphNode(
            id="CustomModule_1",
            module=module,
            args=(),
            kwargs={},
            dependencies=[],
        )
        graph = InferenceGraph(
            nodes={"CustomModule_1": node},
            input_ids=["CustomModule_1"],
            output_ids=["CustomModule_1"],
        )

        record = ForwardRecord(
            graph=graph,
            node_inputs={"CustomModule_1": {"arg_0": "hello"}},
            node_outputs={"CustomModule_1": "HELLO"},
            module_map={"CustomModule_1": module},
        )

        assert isinstance(record.get_module("CustomModule_1"), CustomModule)
        assert record.get_node_output("CustomModule_1") == "HELLO"
