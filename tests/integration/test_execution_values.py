"""Integration tests for execution with Values, ValueRef, and error propagation.

These tests verify end-to-end execution behavior with the Value system,
including structured select operations and error propagation through graphs.
"""

import pytest

import plait.functional as F
from plait.execution.scheduler import Scheduler
from plait.execution.state import ExecutionState
from plait.graph import GraphNode, InferenceGraph
from plait.module import InferenceModule
from plait.tracing.tracer import InputNode
from plait.values import Value, ValueKind, ValueRef

# ─────────────────────────────────────────────────────────────────────────────
# Helper modules for testing
# ─────────────────────────────────────────────────────────────────────────────


class SelectFieldModule(InferenceModule):
    """Module that selects a field from structured input."""

    def __init__(self, field: str) -> None:
        super().__init__()
        self.field = field

    def forward(self, data: dict) -> str:
        """Select a field from the input dict."""
        return data[self.field]


class ConcatModule(InferenceModule):
    """Module that concatenates string inputs."""

    def forward(self, *args: str) -> str:
        """Concatenate all string arguments."""
        return " ".join(str(arg) for arg in args)


class UppercaseModule(InferenceModule):
    """Module that converts input to uppercase."""

    def forward(self, text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()


class FormatModule(InferenceModule):
    """Module that formats a template with data."""

    def __init__(self, template: str) -> None:
        super().__init__()
        self.template = template

    def forward(self, **kwargs: str) -> str:
        """Format the template with kwargs."""
        return self.template.format(**kwargs)


class ParseJsonModule(InferenceModule):
    """Module that parses JSON and wraps errors as Values."""

    def forward(self, text: str) -> dict | Value:
        """Parse JSON, returning Value(ERROR) on failure."""
        import json

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            return Value(ValueKind.ERROR, e, meta={"op": "parse_json"})


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests: End-to-End Graph Execution with Values
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndValueExecution:
    """Integration tests for complete graph execution with Value system."""

    @pytest.mark.asyncio
    async def test_simple_value_flow(self) -> None:
        """Values flow correctly through a simple graph."""
        input_value = Value(ValueKind.TEXT, "hello world")

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(input_value),
            args=(),
            kwargs={},
            dependencies=[],
        )
        upper = UppercaseModule()
        process_node = GraphNode(
            id="UppercaseModule_1",
            module=upper,
            args=(ValueRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "UppercaseModule_1": process_node,
            },
            input_ids=["input:input_0"],
            output_ids=["UppercaseModule_1"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # Value payload is unwrapped for forward(), result is raw
        assert outputs["UppercaseModule_1"] == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_structured_data_flow(self) -> None:
        """Structured data (dicts) flows through and can be accessed."""
        input_data = Value(
            ValueKind.STRUCTURED,
            {"user": {"name": "Ada", "title": "Engineer"}},
        )

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(input_data),
            args=(),
            kwargs={},
            dependencies=[],
        )
        select = SelectFieldModule("user")
        select_node = GraphNode(
            id="SelectFieldModule_1",
            module=select,
            args=(ValueRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "SelectFieldModule_1": select_node,
            },
            input_ids=["input:input_0"],
            output_ids=["SelectFieldModule_1"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # Dict was passed through, field was selected
        assert outputs["SelectFieldModule_1"] == {"name": "Ada", "title": "Engineer"}

    @pytest.mark.asyncio
    async def test_chained_operations(self) -> None:
        """Operations can be chained with Values flowing through."""
        input_value = Value(ValueKind.STRUCTURED, {"greeting": "hello", "name": "ada"})

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(input_value),
            args=(),
            kwargs={},
            dependencies=[],
        )

        # First select "name"
        select_name = SelectFieldModule("name")
        select_node = GraphNode(
            id="SelectFieldModule_1",
            module=select_name,
            args=(ValueRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )

        # Then uppercase it
        upper = UppercaseModule()
        upper_node = GraphNode(
            id="UppercaseModule_1",
            module=upper,
            args=(ValueRef("SelectFieldModule_1"),),
            kwargs={},
            dependencies=["SelectFieldModule_1"],
        )

        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "SelectFieldModule_1": select_node,
                "UppercaseModule_1": upper_node,
            },
            input_ids=["input:input_0"],
            output_ids=["UppercaseModule_1"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        assert outputs["UppercaseModule_1"] == "ADA"

    @pytest.mark.asyncio
    async def test_parallel_value_processing(self) -> None:
        """Parallel branches process Values independently."""
        input_value = Value(ValueKind.STRUCTURED, {"a": "first", "b": "second"})

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(input_value),
            args=(),
            kwargs={},
            dependencies=[],
        )

        # Branch A: select "a" and uppercase
        select_a = SelectFieldModule("a")
        branch_a = GraphNode(
            id="SelectA_1",
            module=select_a,
            args=(ValueRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        upper_a = UppercaseModule()
        upper_a_node = GraphNode(
            id="UpperA_1",
            module=upper_a,
            args=(ValueRef("SelectA_1"),),
            kwargs={},
            dependencies=["SelectA_1"],
        )

        # Branch B: select "b" and uppercase
        select_b = SelectFieldModule("b")
        branch_b = GraphNode(
            id="SelectB_1",
            module=select_b,
            args=(ValueRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        upper_b = UppercaseModule()
        upper_b_node = GraphNode(
            id="UpperB_1",
            module=upper_b,
            args=(ValueRef("SelectB_1"),),
            kwargs={},
            dependencies=["SelectB_1"],
        )

        # Merge: concatenate results
        concat = ConcatModule()
        merge_node = GraphNode(
            id="Concat_1",
            module=concat,
            args=(ValueRef("UpperA_1"), ValueRef("UpperB_1")),
            kwargs={},
            dependencies=["UpperA_1", "UpperB_1"],
        )

        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "SelectA_1": branch_a,
                "UpperA_1": upper_a_node,
                "SelectB_1": branch_b,
                "UpperB_1": upper_b_node,
                "Concat_1": merge_node,
            },
            input_ids=["input:input_0"],
            output_ids=["Concat_1"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        assert outputs["Concat_1"] == "FIRST SECOND"


class TestErrorPropagationIntegration:
    """Integration tests for error-as-value propagation."""

    @pytest.mark.asyncio
    async def test_error_propagates_through_chain(self) -> None:
        """Value(ERROR) propagates through a chain without executing modules."""
        error_value = Value(
            ValueKind.ERROR, ValueError("source error"), meta={"source": "test"}
        )

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(error_value),
            args=(),
            kwargs={},
            dependencies=[],
        )
        upper1 = UppercaseModule()
        node1 = GraphNode(
            id="Upper_1",
            module=upper1,
            args=(ValueRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        upper2 = UppercaseModule()
        node2 = GraphNode(
            id="Upper_2",
            module=upper2,
            args=(ValueRef("Upper_1"),),
            kwargs={},
            dependencies=["Upper_1"],
        )

        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "Upper_1": node1,
                "Upper_2": node2,
            },
            input_ids=["input:input_0"],
            output_ids=["Upper_2"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # Error should propagate through to the end
        result = outputs["Upper_2"]
        assert isinstance(result, Value)
        assert result.kind == ValueKind.ERROR
        assert result.meta.get("source") == "test"

    @pytest.mark.asyncio
    async def test_error_in_one_branch_merges_correctly(self) -> None:
        """When one branch errors, the merge node receives the error."""
        error_value = Value(ValueKind.ERROR, ValueError("branch error"))
        ok_value = Value(ValueKind.TEXT, "hello")

        input_err = GraphNode(
            id="input:error",
            module=InputNode(error_value),
            args=(),
            kwargs={},
            dependencies=[],
        )
        input_ok = GraphNode(
            id="input:ok",
            module=InputNode(ok_value),
            args=(),
            kwargs={},
            dependencies=[],
        )

        upper_ok = UppercaseModule()
        upper_ok_node = GraphNode(
            id="UpperOK_1",
            module=upper_ok,
            args=(ValueRef("input:ok"),),
            kwargs={},
            dependencies=["input:ok"],
        )

        upper_err = UppercaseModule()
        upper_err_node = GraphNode(
            id="UpperErr_1",
            module=upper_err,
            args=(ValueRef("input:error"),),
            kwargs={},
            dependencies=["input:error"],
        )

        # Merge receives one OK and one ERROR
        concat = ConcatModule()
        merge_node = GraphNode(
            id="Concat_1",
            module=concat,
            args=(ValueRef("UpperOK_1"), ValueRef("UpperErr_1")),
            kwargs={},
            dependencies=["UpperOK_1", "UpperErr_1"],
        )

        graph = InferenceGraph(
            nodes={
                "input:error": input_err,
                "input:ok": input_ok,
                "UpperOK_1": upper_ok_node,
                "UpperErr_1": upper_err_node,
                "Concat_1": merge_node,
            },
            input_ids=["input:error", "input:ok"],
            output_ids=["Concat_1"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # Merge should short-circuit and output the error
        result = outputs["Concat_1"]
        assert isinstance(result, Value)
        assert result.kind == ValueKind.ERROR

    @pytest.mark.asyncio
    async def test_module_can_produce_error_value(self) -> None:
        """Modules can return Value(ERROR) which then propagates."""
        # Invalid JSON will produce an error Value
        input_value = Value(ValueKind.TEXT, "not valid json {{")

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(input_value),
            args=(),
            kwargs={},
            dependencies=[],
        )
        parse = ParseJsonModule()
        parse_node = GraphNode(
            id="ParseJson_1",
            module=parse,
            args=(ValueRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )
        # Try to select from the parse result
        select = SelectFieldModule("key")
        select_node = GraphNode(
            id="Select_1",
            module=select,
            args=(ValueRef("ParseJson_1"),),
            kwargs={},
            dependencies=["ParseJson_1"],
        )

        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "ParseJson_1": parse_node,
                "Select_1": select_node,
            },
            input_ids=["input:input_0"],
            output_ids=["Select_1"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        # The parse error should propagate to the select node
        result = outputs["Select_1"]
        assert isinstance(result, Value)
        assert result.kind == ValueKind.ERROR
        assert result.meta.get("op") == "parse_json"


class TestFunctionalAPIIntegration:
    """Integration tests using the functional API with execution."""

    @pytest.mark.asyncio
    async def test_functional_select_with_execution(self) -> None:
        """F.select can be used in a module and executes correctly."""

        class FunctionalSelectModule(InferenceModule):
            """Module that uses F.select internally."""

            def __init__(self, path: str) -> None:
                super().__init__()
                self.path = path

            def forward(self, data: dict) -> Value:
                """Select using functional API."""
                value = Value(ValueKind.STRUCTURED, data)
                return F.select(value, self.path)

        input_data = {"user": {"name": "Ada"}}

        input_node = GraphNode(
            id="input:input_0",
            module=InputNode(input_data),
            args=(),
            kwargs={},
            dependencies=[],
        )
        select = FunctionalSelectModule("user")
        select_node = GraphNode(
            id="FunctionalSelect_1",
            module=select,
            args=(ValueRef("input:input_0"),),
            kwargs={},
            dependencies=["input:input_0"],
        )

        graph = InferenceGraph(
            nodes={
                "input:input_0": input_node,
                "FunctionalSelect_1": select_node,
            },
            input_ids=["input:input_0"],
            output_ids=["FunctionalSelect_1"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        result = outputs["FunctionalSelect_1"]
        assert isinstance(result, Value)
        assert result.kind == ValueKind.STRUCTURED
        assert result.payload == {"name": "Ada"}

    @pytest.mark.asyncio
    async def test_functional_render_with_execution(self) -> None:
        """F.render can be used in a module and executes correctly."""

        class RenderModule(InferenceModule):
            """Module that uses F.render internally."""

            def __init__(self, template: str) -> None:
                super().__init__()
                self.template = template

            def forward(self, **kwargs: str) -> Value:
                """Render template using functional API."""
                return F.render(self.template, kwargs)

        input_node = GraphNode(
            id="input:name",
            module=InputNode("Ada"),
            args=(),
            kwargs={},
            dependencies=[],
        )
        render = RenderModule("Hello, {name}!")
        render_node = GraphNode(
            id="Render_1",
            module=render,
            args=(),
            kwargs={"name": ValueRef("input:name")},
            dependencies=["input:name"],
        )

        graph = InferenceGraph(
            nodes={
                "input:name": input_node,
                "Render_1": render_node,
            },
            input_ids=["input:name"],
            output_ids=["Render_1"],
        )

        scheduler = Scheduler(max_concurrent=10)
        state = ExecutionState(graph)

        outputs = await scheduler.execute(state)

        result = outputs["Render_1"]
        assert isinstance(result, Value)
        assert result.kind == ValueKind.TEXT
        assert result.payload == "Hello, Ada!"
