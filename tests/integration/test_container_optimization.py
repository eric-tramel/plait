"""Integration tests for parameter collection and optimization with containers.

These tests verify that parameters in container classes (ParameterList,
ParameterDict, ModuleList, ModuleDict) are correctly:
1. Collected by parameters() and named_parameters()
2. Included in state_dict() and load_state_dict()
3. Updated during optimizer.step()
"""

from unittest.mock import MagicMock

import pytest

from plait.containers import ModuleDict, ModuleList, ParameterDict, ParameterList
from plait.graph import GraphNode, InferenceGraph
from plait.module import LLMInference, Module
from plait.optimization.optimizer import SFAOptimizer
from plait.optimization.record import ForwardRecord
from plait.parameter import Parameter


def create_module_record(
    module: Module,
    node_ids: list[str] | None = None,
) -> ForwardRecord:
    """Create a ForwardRecord for a module with all its parameters.

    Args:
        module: The module to create a record for.
        node_ids: Optional list of node IDs. If not provided, creates
            one node per parameter.

    Returns:
        A ForwardRecord with nodes for all parameters.
    """
    params = list(module.named_parameters())
    if node_ids is None:
        node_ids = [f"node_{i}" for i in range(len(params))]

    # Create a mock module for each node that returns its parameter
    module_map: dict[str, Module] = {}
    nodes: dict[str, GraphNode] = {}

    for (param_name, param), node_id in zip(params, node_ids, strict=False):
        mock_module = MagicMock(spec=LLMInference)
        mock_module.named_parameters.return_value = [(param_name.split(".")[-1], param)]
        mock_module.parameters.return_value = [param]
        module_map[node_id] = mock_module

        # Linear dependencies
        idx = node_ids.index(node_id)
        deps = [node_ids[idx - 1]] if idx > 0 else []

        nodes[node_id] = GraphNode(
            id=node_id,
            module=mock_module,
            args=(),
            kwargs={},
            dependencies=deps,
            module_name=f"Module({node_id})",
        )

    input_ids = [node_ids[0]] if node_ids else []
    output_ids = [node_ids[-1]] if node_ids else []

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


class TestParameterListOptimization:
    """Tests for ParameterList parameter collection and optimization."""

    def test_all_list_parameters_collected(self) -> None:
        """All parameters in ParameterList are collected by parameters()."""

        class MultiPromptModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("Prompt 1", description="First prompt"),
                        Parameter("Prompt 2", description="Second prompt"),
                        Parameter("Prompt 3", description="Third prompt"),
                    ]
                )

        module = MultiPromptModule()
        params = list(module.parameters())
        assert len(params) == 3
        values = {p.value for p in params}
        assert values == {"Prompt 1", "Prompt 2", "Prompt 3"}

    def test_list_parameters_in_state_dict(self) -> None:
        """ParameterList parameters appear in state_dict with correct names."""

        class MultiPromptModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("P1", description="First"),
                        Parameter("P2", description="Second"),
                    ]
                )

        module = MultiPromptModule()
        state = module.state_dict()

        assert "prompts.0" in state
        assert "prompts.1" in state
        assert state["prompts.0"] == "P1"
        assert state["prompts.1"] == "P2"

    @pytest.mark.asyncio
    async def test_all_list_parameters_updated(self) -> None:
        """All parameters in ParameterList receive updates during step()."""

        class MultiPromptModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("Old 1", description="First prompt"),
                        Parameter("Old 2", description="Second prompt"),
                        Parameter("Old 3", description="Third prompt"),
                    ]
                )

        module = MultiPromptModule()

        # Add feedback to all parameters
        for i, param in enumerate(module.prompts):
            param._name = str(i)  # Ensure proper naming
            param.accumulate_feedback(f"Update prompt {i}")

        optimizer = SFAOptimizer(module.parameters())
        record = create_module_record(module)
        optimizer.capture_record(record)

        update_count = 0

        async def mock_updater(prompt: str) -> str:
            nonlocal update_count
            update_count += 1
            return f"New {update_count}"

        optimizer.updater = mock_updater
        optimizer._bound = True

        updates = await optimizer.step()

        assert len(updates) == 3
        assert update_count == 3


class TestParameterDictOptimization:
    """Tests for ParameterDict parameter collection and optimization."""

    def test_all_dict_parameters_collected(self) -> None:
        """All parameters in ParameterDict are collected by parameters()."""

        class TaskPromptModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterDict(
                    {
                        "summarize": Parameter("Summarize:", description="Summary"),
                        "translate": Parameter("Translate:", description="Translation"),
                        "analyze": Parameter("Analyze:", description="Analysis"),
                    }
                )

        module = TaskPromptModule()
        params = list(module.parameters())
        assert len(params) == 3
        values = {p.value for p in params}
        assert values == {"Summarize:", "Translate:", "Analyze:"}

    def test_dict_parameters_in_state_dict(self) -> None:
        """ParameterDict parameters appear in state_dict with correct names."""

        class TaskPromptModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterDict(
                    {
                        "task1": Parameter("T1", description="Task 1"),
                        "task2": Parameter("T2", description="Task 2"),
                    }
                )

        module = TaskPromptModule()
        state = module.state_dict()

        assert "prompts.task1" in state
        assert "prompts.task2" in state
        assert state["prompts.task1"] == "T1"
        assert state["prompts.task2"] == "T2"

    @pytest.mark.asyncio
    async def test_all_dict_parameters_updated(self) -> None:
        """All parameters in ParameterDict receive updates during step()."""

        class TaskPromptModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterDict(
                    {
                        "task1": Parameter("Old task1", description="Task 1"),
                        "task2": Parameter("Old task2", description="Task 2"),
                    }
                )

        module = TaskPromptModule()

        # Add feedback to all parameters
        for key, param in module.prompts.items():
            param.accumulate_feedback(f"Update {key}")

        optimizer = SFAOptimizer(module.parameters())
        record = create_module_record(module)
        optimizer.capture_record(record)

        update_count = 0

        async def mock_updater(prompt: str) -> str:
            nonlocal update_count
            update_count += 1
            return f"New task{update_count}"

        optimizer.updater = mock_updater
        optimizer._bound = True

        updates = await optimizer.step()

        # Both parameters should be updated
        assert len(updates) == 2
        assert update_count == 2


class TestModuleContainerParameterCollection:
    """Tests for parameter collection from ModuleList and ModuleDict."""

    def test_module_list_parameters_collected(self) -> None:
        """Parameters in ModuleList children are collected."""

        class ModuleWithParam(Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.prompt = Parameter(name, description=f"Param {name}")

        class PipelineModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.stages = ModuleList(
                    [
                        ModuleWithParam("stage1"),
                        ModuleWithParam("stage2"),
                        ModuleWithParam("stage3"),
                    ]
                )

        pipeline = PipelineModule()
        params = list(pipeline.parameters())
        assert len(params) == 3
        values = {p.value for p in params}
        assert values == {"stage1", "stage2", "stage3"}

    def test_module_dict_parameters_collected(self) -> None:
        """Parameters in ModuleDict children are collected."""

        class ModuleWithParam(Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.prompt = Parameter(name, description=f"Param {name}")

        class MultiTaskModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.tasks = ModuleDict(
                    {
                        "summarize": ModuleWithParam("summarize_prompt"),
                        "translate": ModuleWithParam("translate_prompt"),
                    }
                )

        module = MultiTaskModule()
        params = list(module.parameters())
        assert len(params) == 2
        values = {p.value for p in params}
        assert values == {"summarize_prompt", "translate_prompt"}


class TestMixedContainerOptimization:
    """Tests for modules with mixed container types."""

    def test_mixed_containers_all_collected(self) -> None:
        """Parameters from all container types are collected."""

        class MixedModule(Module):
            def __init__(self) -> None:
                super().__init__()
                # Direct parameter
                self.global_prompt = Parameter("global", description="Global")

                # ParameterList
                self.list_prompts = ParameterList(
                    [
                        Parameter("list1", description="List 1"),
                        Parameter("list2", description="List 2"),
                    ]
                )

                # ParameterDict
                self.dict_prompts = ParameterDict(
                    {
                        "key1": Parameter("dict1", description="Dict 1"),
                        "key2": Parameter("dict2", description="Dict 2"),
                    }
                )

        module = MixedModule()
        params = list(module.parameters())
        # 1 direct + 2 list + 2 dict = 5 total
        assert len(params) == 5

        named = dict(module.named_parameters())
        assert "global_prompt" in named
        assert "list_prompts.0" in named
        assert "list_prompts.1" in named
        assert "dict_prompts.key1" in named
        assert "dict_prompts.key2" in named

    @pytest.mark.asyncio
    async def test_mixed_containers_all_updated(self) -> None:
        """All parameters from mixed containers receive updates."""

        class MixedModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.global_prompt = Parameter("global_old", description="Global")
                self.list_prompts = ParameterList(
                    [
                        Parameter("list_old", description="List 1"),
                    ]
                )
                self.dict_prompts = ParameterDict(
                    {
                        "key": Parameter("dict_old", description="Dict 1"),
                    }
                )

        module = MixedModule()

        # Add feedback to all parameters
        for param in module.parameters():
            param.accumulate_feedback("Update this parameter")

        optimizer = SFAOptimizer(module.parameters())
        record = create_module_record(module)
        optimizer.capture_record(record)

        update_count = 0

        async def mock_updater(prompt: str) -> str:
            nonlocal update_count
            update_count += 1
            return f"new_value_{update_count}"

        optimizer.updater = mock_updater
        optimizer._bound = True

        updates = await optimizer.step()

        # All 3 parameters should be updated
        assert len(updates) == 3
        assert update_count == 3


class TestNestedContainerOptimization:
    """Tests for nested modules with containers."""

    def test_nested_module_parameters_collected(self) -> None:
        """Parameters in nested modules with containers are all collected."""

        class InnerModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner_prompts = ParameterList(
                    [
                        Parameter("inner1", description="Inner 1"),
                        Parameter("inner2", description="Inner 2"),
                    ]
                )

        class OuterModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.outer_prompt = Parameter("outer", description="Outer")
                self.inner = InnerModule()

        module = OuterModule()
        params = list(module.parameters())
        # 1 outer + 2 inner = 3 total
        assert len(params) == 3

        named = dict(module.named_parameters())
        assert "outer_prompt" in named
        assert "inner.inner_prompts.0" in named
        assert "inner.inner_prompts.1" in named

    def test_deeply_nested_parameters_collected(self) -> None:
        """Parameters in deeply nested containers are collected."""

        class Level2(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterDict(
                    {
                        "deep": Parameter("deep_value", description="Deep param"),
                    }
                )

        class Level1(Module):
            def __init__(self) -> None:
                super().__init__()
                self.level2 = Level2()

        class Root(Module):
            def __init__(self) -> None:
                super().__init__()
                self.level1 = Level1()

        module = Root()
        params = list(module.parameters())
        assert len(params) == 1

        named = dict(module.named_parameters())
        # Should have hierarchical name through the module tree
        assert "level1.level2.prompts.deep" in named
        assert named["level1.level2.prompts.deep"].value == "deep_value"


class TestLoadStateDictWithContainers:
    """Tests for load_state_dict with container-based parameters."""

    def test_load_state_dict_updates_list_parameters(self) -> None:
        """load_state_dict correctly updates ParameterList parameters."""

        class MultiPromptModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterList(
                    [
                        Parameter("old1", description="First"),
                        Parameter("old2", description="Second"),
                    ]
                )

        module = MultiPromptModule()
        module.load_state_dict(
            {
                "prompts.0": "new1",
                "prompts.1": "new2",
            }
        )

        assert module.prompts[0].value == "new1"
        assert module.prompts[1].value == "new2"

    def test_load_state_dict_updates_dict_parameters(self) -> None:
        """load_state_dict correctly updates ParameterDict parameters."""

        class TaskPromptModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompts = ParameterDict(
                    {
                        "task1": Parameter("old1", description="Task 1"),
                        "task2": Parameter("old2", description="Task 2"),
                    }
                )

        module = TaskPromptModule()
        module.load_state_dict(
            {
                "prompts.task1": "new1",
                "prompts.task2": "new2",
            }
        )

        assert module.prompts["task1"].value == "new1"
        assert module.prompts["task2"].value == "new2"

    def test_round_trip_state_dict(self) -> None:
        """state_dict and load_state_dict are consistent for containers."""

        class MixedModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.global_prompt = Parameter("global", description="Global")
                self.list_prompts = ParameterList(
                    [
                        Parameter("list1", description="List 1"),
                        Parameter("list2", description="List 2"),
                    ]
                )
                self.dict_prompts = ParameterDict(
                    {
                        "key1": Parameter("dict1", description="Dict 1"),
                    }
                )

        original = MixedModule()
        state = original.state_dict()

        # Create a new module and load the state
        loaded = MixedModule()
        loaded.load_state_dict(state)

        # Verify all values match
        assert loaded.global_prompt.value == "global"
        assert loaded.list_prompts[0].value == "list1"
        assert loaded.list_prompts[1].value == "list2"
        assert loaded.dict_prompts["key1"].value == "dict1"
