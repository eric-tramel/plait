"""Integration tests for topologically-ordered parameter updates.

These tests verify that the optimizer correctly coordinates updates
between dependent parameters, ensuring downstream parameters see
upstream changes and can maintain consistency.
"""

from unittest.mock import MagicMock

import pytest

from plait.graph import GraphNode, InferenceGraph
from plait.module import LLMInference, Module
from plait.optimization.optimizer import SFAOptimizer
from plait.optimization.record import ForwardRecord
from plait.parameter import Parameter


def create_pipeline_record(
    param_configs: list[tuple[Parameter, str]],
    dependencies: dict[str, list[str]] | None = None,
) -> ForwardRecord:
    """Create a ForwardRecord for a pipeline with specified dependencies.

    Args:
        param_configs: List of (Parameter, node_id) tuples.
        dependencies: Optional dict mapping node_id to list of dependency node_ids.
            If not provided, creates a linear chain.

    Returns:
        A ForwardRecord with the specified graph structure.
    """
    node_ids = [node_id for _, node_id in param_configs]

    # Default to linear dependencies if not specified
    if dependencies is None:
        dependencies = {}
        for i, node_id in enumerate(node_ids):
            if i > 0:
                dependencies[node_id] = [node_ids[i - 1]]
            else:
                dependencies[node_id] = []

    module_map: dict[str, Module] = {}
    for param, node_id in param_configs:
        mock_module = MagicMock(spec=LLMInference)
        mock_module.named_parameters.return_value = [(param._name, param)]
        mock_module.parameters.return_value = [param]
        module_map[node_id] = mock_module

    # Create nodes with specified dependencies
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

    # Find input nodes (no dependencies) and output nodes (no dependents)
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
        node_outputs={nid: f"output_{nid}" for nid in node_ids},
        module_map=module_map,
    )


class TestCoordinatedUpdates:
    """Tests for coordinated updates between dependent parameters."""

    @pytest.mark.asyncio
    async def test_format_spec_and_validator_consistency(self) -> None:
        """Format spec and validator updates maintain consistency.

        This tests the canonical example from the design doc:
        format_spec -> validator

        When format_spec changes from JSON to YAML, validator should
        see this change and update accordingly.
        """
        format_spec = Parameter(
            "Output as JSON with keys: name, age, city",
            description="Specifies the output format for the LLM",
        )
        format_spec._name = "format_spec"

        validator_rules = Parameter(
            "Verify output is valid JSON with required keys",
            description="Validation rules that must match the format spec",
        )
        validator_rules._name = "validator_rules"

        format_spec.accumulate_feedback("Users prefer YAML format")
        validator_rules.accumulate_feedback("Validation is too strict")

        optimizer = SFAOptimizer([format_spec, validator_rules])

        # Create graph: format_spec -> validator_rules
        record = create_pipeline_record(
            [
                (format_spec, "format_node"),
                (validator_rules, "validator_node"),
            ]
        )
        optimizer.capture_record(record)

        captured_prompts: list[str] = []

        async def mock_updater(prompt: str) -> str:
            captured_prompts.append(prompt)
            if "format_spec" in prompt:
                return "Output as YAML with keys: name, age, city"
            else:
                # Validator should see that format changed to YAML
                return "Verify output is valid YAML with required keys"

        optimizer.updater = mock_updater
        optimizer._bound = True

        updates = await optimizer.step()

        # Verify both parameters were updated
        assert len(updates) == 2
        assert "YAML" in format_spec.value
        assert "YAML" in validator_rules.value

        # Verify validator prompt contained upstream context
        validator_prompt = captured_prompts[1]
        assert "<upstream-updates>" in validator_prompt
        assert "format_spec" in validator_prompt
        # The new YAML value should be visible to the validator update
        assert "YAML" in validator_prompt

    @pytest.mark.asyncio
    async def test_three_stage_pipeline(self) -> None:
        """Three-stage pipeline updates in correct order.

        input_format -> processor -> output_format

        Each stage should see the updates from previous stages.
        """
        input_format = Parameter("Accept JSON input", description="Input format spec")
        input_format._name = "input_format"

        processor = Parameter(
            "Process the data fields",
            description="Processing instructions",
        )
        processor._name = "processor"

        output_format = Parameter(
            "Output as JSON with result field",
            description="Output format spec",
        )
        output_format._name = "output_format"

        # All three need updates
        input_format.accumulate_feedback("Support CSV input")
        processor.accumulate_feedback("Add validation step")
        output_format.accumulate_feedback("Include metadata in output")

        optimizer = SFAOptimizer([input_format, processor, output_format])

        # Linear graph: input -> processor -> output
        record = create_pipeline_record(
            [
                (input_format, "input_node"),
                (processor, "processor_node"),
                (output_format, "output_node"),
            ]
        )
        optimizer.capture_record(record)

        update_order: list[str] = []

        async def mock_updater(prompt: str) -> str:
            # Check for the specific parameter being updated (in <parameter name="...">)
            if '<parameter name="input_format">' in prompt:
                update_order.append("input_format")
                return "Accept CSV input"
            elif '<parameter name="processor">' in prompt:
                update_order.append("processor")
                # Should see input_format changed to CSV in upstream context
                assert "CSV" in prompt, (
                    "Processor should see input_format change to CSV"
                )
                return "Process and validate CSV data fields"
            else:
                update_order.append("output_format")
                # Should see both previous changes
                return "Output as JSON with result field and metadata"

        optimizer.updater = mock_updater
        optimizer._bound = True

        await optimizer.step()

        # Verify update order
        assert update_order == ["input_format", "processor", "output_format"]

    @pytest.mark.asyncio
    async def test_diamond_graph_updates(self) -> None:
        """Diamond graph handles parallel and dependent updates.

        Graph structure:
              input
             /     \\
           left   right
             \\     /
              merge

        left and right can update in parallel, but merge must wait.
        """
        input_param = Parameter("Input spec", description="Input specification")
        input_param._name = "input_param"

        left_param = Parameter("Left processing", description="Left branch config")
        left_param._name = "left_param"

        right_param = Parameter("Right processing", description="Right branch config")
        right_param._name = "right_param"

        merge_param = Parameter("Merge results", description="Merge configuration")
        merge_param._name = "merge_param"

        # All need updates
        input_param.accumulate_feedback("Update input")
        left_param.accumulate_feedback("Update left")
        right_param.accumulate_feedback("Update right")
        merge_param.accumulate_feedback("Update merge")

        optimizer = SFAOptimizer([input_param, left_param, right_param, merge_param])

        # Diamond dependencies
        record = create_pipeline_record(
            [
                (input_param, "input"),
                (left_param, "left"),
                (right_param, "right"),
                (merge_param, "merge"),
            ],
            dependencies={
                "input": [],
                "left": ["input"],
                "right": ["input"],
                "merge": ["left", "right"],
            },
        )
        optimizer.capture_record(record)

        update_times: dict[str, int] = {}
        counter = 0

        async def mock_updater(prompt: str) -> str:
            nonlocal counter
            counter += 1

            # Check for the specific parameter being updated (in <parameter name="...">)
            if '<parameter name="input_param">' in prompt:
                update_times["input_param"] = counter
                return "Updated input"
            elif '<parameter name="left_param">' in prompt:
                update_times["left_param"] = counter
                return "Updated left"
            elif '<parameter name="right_param">' in prompt:
                update_times["right_param"] = counter
                return "Updated right"
            else:
                update_times["merge_param"] = counter
                return "Updated merge"

        optimizer.updater = mock_updater
        optimizer._bound = True

        await optimizer.step()

        # Input must be first
        assert update_times["input_param"] == 1

        # Left and right can be parallel (either order is fine)
        assert update_times["left_param"] in [2, 3]
        assert update_times["right_param"] in [2, 3]

        # Merge must be last
        assert update_times["merge_param"] == 4

    @pytest.mark.asyncio
    async def test_no_upstream_context_for_root_params(self) -> None:
        """Root parameters (no upstream deps) don't get upstream context."""
        root_param = Parameter("Root value", description="Root parameter")
        root_param._name = "root_param"
        root_param.accumulate_feedback("Update root")

        optimizer = SFAOptimizer([root_param])

        record = create_pipeline_record([(root_param, "root_node")])
        optimizer.capture_record(record)

        captured_prompt = ""

        async def mock_updater(prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Updated root"

        optimizer.updater = mock_updater
        optimizer._bound = True

        await optimizer.step()

        # Root should not have upstream context
        assert "<upstream-updates>" not in captured_prompt


class TestPartialUpdates:
    """Tests for partial update scenarios."""

    @pytest.mark.asyncio
    async def test_only_downstream_has_feedback(self) -> None:
        """Downstream param updates without upstream context if upstream has no feedback."""
        upstream = Parameter("Upstream value", description="Upstream param")
        upstream._name = "upstream"
        # No feedback for upstream

        downstream = Parameter("Downstream value", description="Downstream param")
        downstream._name = "downstream"
        downstream.accumulate_feedback("Update downstream")

        optimizer = SFAOptimizer([upstream, downstream])

        record = create_pipeline_record(
            [
                (upstream, "upstream_node"),
                (downstream, "downstream_node"),
            ]
        )
        optimizer.capture_record(record)

        captured_prompts: list[str] = []

        async def mock_updater(prompt: str) -> str:
            captured_prompts.append(prompt)
            return "Updated downstream"

        optimizer.updater = mock_updater
        optimizer._bound = True

        updates = await optimizer.step()

        # Only downstream should be updated
        assert len(updates) == 1
        assert "downstream" in updates

        # Should have exactly one call (downstream only)
        assert len(captured_prompts) == 1

        # No upstream context since upstream didn't change
        assert "<upstream-updates>" not in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_only_upstream_has_feedback(self) -> None:
        """Upstream param updates, downstream skipped if no feedback."""
        upstream = Parameter("Upstream value", description="Upstream param")
        upstream._name = "upstream"
        upstream.accumulate_feedback("Update upstream")

        downstream = Parameter("Downstream value", description="Downstream param")
        downstream._name = "downstream"
        # No feedback for downstream

        optimizer = SFAOptimizer([upstream, downstream])

        record = create_pipeline_record(
            [
                (upstream, "upstream_node"),
                (downstream, "downstream_node"),
            ]
        )
        optimizer.capture_record(record)

        async def mock_updater(prompt: str) -> str:
            return "Updated upstream"

        optimizer.updater = mock_updater
        optimizer._bound = True

        updates = await optimizer.step()

        # Only upstream should be updated
        assert len(updates) == 1
        assert "upstream" in updates
        assert upstream.value == "Updated upstream"
        assert downstream.value == "Downstream value"  # Unchanged
