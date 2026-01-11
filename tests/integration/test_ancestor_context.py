"""Integration tests for ancestor context in optimization.

These tests verify that the optimizer correctly uses ancestor and sibling
context when updating parameters, enabling coherent updates across the
full pipeline.
"""

from unittest.mock import MagicMock

import pytest

from plait.graph import GraphNode, InferenceGraph
from plait.module import LLMInference, Module
from plait.optimization.backward import AncestorContext
from plait.optimization.optimizer import SFAOptimizer
from plait.optimization.record import ForwardRecord
from plait.parameter import Parameter


def create_pipeline_record(
    param_configs: list[tuple[Parameter, str]],
    dependencies: dict[str, list[str]] | None = None,
    outputs: dict[str, str] | None = None,
) -> ForwardRecord:
    """Create a ForwardRecord for a pipeline with specified dependencies.

    Args:
        param_configs: List of (Parameter, node_id) tuples.
        dependencies: Optional dict mapping node_id to list of dependency node_ids.
            If not provided, creates a linear chain.
        outputs: Optional dict mapping node_id to output value. If not provided,
            generates default outputs.

    Returns:
        A ForwardRecord with the specified graph structure.
    """
    node_ids = [node_id for _, node_id in param_configs]

    if dependencies is None:
        dependencies = {}
        for i, node_id in enumerate(node_ids):
            if i > 0:
                dependencies[node_id] = [node_ids[i - 1]]
            else:
                dependencies[node_id] = []

    if outputs is None:
        outputs = {nid: f"output_{nid}" for nid in node_ids}

    module_map: dict[str, Module] = {}
    for param, node_id in param_configs:
        mock_module = MagicMock(spec=LLMInference)
        mock_module.named_parameters.return_value = [(param._name, param)]
        mock_module.parameters.return_value = [param]
        module_map[node_id] = mock_module

    nodes: dict[str, GraphNode] = {}
    for _, node_id in param_configs:
        deps = dependencies.get(node_id, [])
        nodes[node_id] = GraphNode(
            id=node_id,
            module=module_map[node_id],
            args=(),
            kwargs={},
            dependencies=deps,
            module_name=f"Module({node_id})",
        )

    input_ids = [nid for nid in node_ids if not dependencies.get(nid)]
    all_deps = set()
    for deps in dependencies.values():
        all_deps.update(deps)
    output_ids = [nid for nid in node_ids if nid not in all_deps]

    graph = InferenceGraph(
        nodes=nodes,
        input_ids=input_ids,
        output_ids=output_ids,
    )

    return ForwardRecord(
        graph=graph,
        node_inputs={nid: {} for nid in node_ids},
        node_outputs=outputs,
        module_map=module_map,
    )


class TestOptimizerWithAncestorContext:
    """Tests for optimizer using ancestor context."""

    @pytest.mark.asyncio
    async def test_step_receives_ancestor_values(self) -> None:
        """step() provides ancestor values to downstream updates."""
        # Create upstream and downstream parameters
        upstream = Parameter("upstream value", description="Upstream param")
        upstream._name = "upstream"

        downstream = Parameter("downstream value", description="Downstream param")
        downstream._name = "downstream"

        # Both have feedback
        upstream.accumulate_feedback("feedback for upstream")
        downstream.accumulate_feedback("feedback for downstream")

        optimizer = SFAOptimizer([upstream, downstream])

        # Create record with outputs
        record = create_pipeline_record(
            [
                (upstream, "upstream_node"),
                (downstream, "downstream_node"),
            ],
            outputs={
                "upstream_node": "UPSTREAM OUTPUT TEXT",
                "downstream_node": "downstream output",
            },
        )
        optimizer.capture_record(record)

        # Track prompts sent to updater
        captured_prompts: list[str] = []

        async def mock_updater(prompt: str) -> str:
            captured_prompts.append(prompt)
            if '<parameter name="upstream">' in prompt:
                return "updated upstream"
            return "updated downstream"

        optimizer.updater = mock_updater
        optimizer._bound = True

        await optimizer.step()

        # The downstream prompt should contain ancestor output
        assert len(captured_prompts) == 2
        downstream_prompt = captured_prompts[1]

        # Check for ancestor values in the prompt
        assert "<pipeline-context>" in downstream_prompt
        assert "UPSTREAM OUTPUT TEXT" in downstream_prompt

    @pytest.mark.asyncio
    async def test_step_receives_sibling_values_in_diamond(self) -> None:
        """step() provides sibling values in diamond graph."""
        # Create diamond structure: input -> [a, b] -> merge
        input_param = Parameter("input", description="Input param")
        input_param._name = "input_param"

        a_param = Parameter("branch a", description="A param")
        a_param._name = "a_param"

        b_param = Parameter("branch b", description="B param")
        b_param._name = "b_param"

        # Give feedback to A branch only (to see sibling context)
        a_param.accumulate_feedback("feedback for a")

        optimizer = SFAOptimizer([input_param, a_param, b_param])

        # Create diamond record
        record = create_pipeline_record(
            [
                (input_param, "input"),
                (a_param, "a"),
                (b_param, "b"),
            ],
            dependencies={
                "input": [],
                "a": ["input"],
                "b": ["input"],
            },
            outputs={
                "input": "input output",
                "a": "A OUTPUT",
                "b": "B SIBLING OUTPUT",
            },
        )
        optimizer.capture_record(record)

        captured_prompt = ""

        async def mock_updater(prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return "updated a"

        optimizer.updater = mock_updater
        optimizer._bound = True

        await optimizer.step()

        # The 'a' branch should see sibling 'b' output
        assert "B SIBLING OUTPUT" in captured_prompt

    @pytest.mark.asyncio
    async def test_step_receives_ancestor_params(self) -> None:
        """step() provides ancestor parameter info in context."""
        upstream = Parameter(
            "You are a helpful assistant",
            description="System prompt for upstream",
        )
        upstream._name = "system_prompt"

        downstream = Parameter(
            "Format as JSON",
            description="Output format specification",
        )
        downstream._name = "format_spec"

        downstream.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([upstream, downstream])

        record = create_pipeline_record(
            [
                (upstream, "upstream_node"),
                (downstream, "downstream_node"),
            ],
        )
        optimizer.capture_record(record)

        captured_prompt = ""

        async def mock_updater(prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return "updated format"

        optimizer.updater = mock_updater
        optimizer._bound = True

        await optimizer.step()

        # The downstream prompt should have ancestor param context
        assert "<ancestor-parameters>" in captured_prompt
        assert "System prompt for upstream" in captured_prompt
        assert "You are a helpful assistant" in captured_prompt


class TestBuildAncestorContextStr:
    """Tests for _build_ancestor_context_str formatting."""

    def test_empty_context_returns_empty_string(self) -> None:
        """Empty ancestor context produces empty string."""
        optimizer = SFAOptimizer([])
        result = optimizer._build_ancestor_context_str(None)
        assert result == ""

    def test_empty_ancestor_context_returns_empty_string(self) -> None:
        """AncestorContext with no data produces empty string."""
        optimizer = SFAOptimizer([])
        ctx = AncestorContext()
        result = optimizer._build_ancestor_context_str(ctx)
        assert result == ""

    def test_ancestor_values_formatted(self) -> None:
        """Ancestor values are included in XML format."""
        optimizer = SFAOptimizer([])
        ctx = AncestorContext(
            ancestor_values={"node1": "value1", "node2": "value2"},
        )
        result = optimizer._build_ancestor_context_str(ctx)

        assert "<pipeline-context>" in result
        assert "<ancestor-values>" in result
        assert '<ancestor-output node="node1">' in result
        assert "value1" in result
        assert "value2" in result

    def test_ancestor_params_formatted(self) -> None:
        """Ancestor parameters are included with descriptions."""
        param = Parameter("param value", description="Param description")
        optimizer = SFAOptimizer([])
        ctx = AncestorContext(
            ancestor_params={"node1.param": param},
        )
        result = optimizer._build_ancestor_context_str(ctx)

        assert "<ancestor-parameters>" in result
        assert '<ancestor-param name="node1.param">' in result
        assert "Param description" in result
        assert "param value" in result

    def test_sibling_values_formatted(self) -> None:
        """Sibling values are included in XML format."""
        optimizer = SFAOptimizer([])
        ctx = AncestorContext(
            sibling_values={"sibling1": "sibling value"},
        )
        result = optimizer._build_ancestor_context_str(ctx)

        assert "<sibling-values>" in result
        assert '<sibling-output node="sibling1">' in result
        assert "sibling value" in result

    def test_sibling_params_formatted(self) -> None:
        """Sibling parameters are included with descriptions."""
        param = Parameter("sibling param", description="Sibling description")
        optimizer = SFAOptimizer([])
        ctx = AncestorContext(
            sibling_params={"sibling.param": param},
        )
        result = optimizer._build_ancestor_context_str(ctx)

        assert "<sibling-parameters>" in result
        assert '<sibling-param name="sibling.param">' in result
        assert "Sibling description" in result

    def test_long_values_truncated(self) -> None:
        """Long values are truncated to 500 chars."""
        optimizer = SFAOptimizer([])
        long_value = "x" * 600
        ctx = AncestorContext(
            ancestor_values={"node": long_value},
        )
        result = optimizer._build_ancestor_context_str(ctx)

        # Should be truncated with ellipsis
        assert "..." in result
        # Full value should not appear
        assert long_value not in result
        # Should have first 500 chars
        assert "x" * 100 in result
