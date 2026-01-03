"""Tests for Optimizer abstract base class and SFAOptimizer implementation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from inf_engine.graph import GraphNode, InferenceGraph
from inf_engine.module import InferenceModule, LLMInference
from inf_engine.optimization.optimizer import Optimizer, SFAOptimizer
from inf_engine.optimization.record import ForwardRecord
from inf_engine.parameter import Parameter


def create_mock_record(
    params: list[Parameter] | None = None,
    node_ids: list[str] | None = None,
) -> ForwardRecord:
    """Create a mock ForwardRecord for testing.

    Args:
        params: Parameters to include in the module_map.
        node_ids: Node IDs to create. Defaults to ["node_0", "node_1"].

    Returns:
        A mock ForwardRecord with proper structure.
    """
    if node_ids is None:
        node_ids = ["node_0", "node_1"]

    # Create mock modules for each node
    module_map: dict[str, InferenceModule] = {}
    for _i, node_id in enumerate(node_ids):
        mock_module = MagicMock(spec=LLMInference)
        mock_module.named_parameters.return_value = []
        mock_module.parameters.return_value = []
        module_map[node_id] = mock_module

    # If params provided, attach them to the first module
    if params and node_ids:
        first_mock = module_map[node_ids[0]]
        first_mock.named_parameters.return_value = [  # type: ignore[attr-defined]
            (p._name or f"param_{i}", p) for i, p in enumerate(params)
        ]
        first_mock.parameters.return_value = params  # type: ignore[attr-defined]

    # Create nodes with proper dependencies
    nodes: dict[str, GraphNode] = {}
    for i, node_id in enumerate(node_ids):
        deps = [node_ids[i - 1]] if i > 0 else []
        nodes[node_id] = GraphNode(
            id=node_id,
            module=module_map[node_id],
            args=(),
            kwargs={},
            dependencies=deps,
            module_name=f"Module_{i}",
        )

    graph = InferenceGraph(
        nodes=nodes,
        input_ids=[node_ids[0]] if node_ids else [],
        output_ids=[node_ids[-1]] if node_ids else [],
    )

    return ForwardRecord(
        graph=graph,
        node_inputs={nid: {} for nid in node_ids},
        node_outputs={nid: f"output_{i}" for i, nid in enumerate(node_ids)},
        module_map=module_map,
    )


class TestOptimizerABC:
    """Tests for Optimizer abstract base class interface."""

    def test_optimizer_is_abstract(self) -> None:
        """Optimizer cannot be instantiated directly."""
        params = [Parameter("test", description="test param")]
        with pytest.raises(TypeError) as exc_info:
            Optimizer(params)  # type: ignore[abstract]
        assert "abstract" in str(exc_info.value).lower()

    def test_optimizer_requires_step_method(self) -> None:
        """Subclass must implement step() method."""

        class IncompleteOptimizer(Optimizer):
            pass

        params = [Parameter("test", description="test param")]
        with pytest.raises(TypeError) as exc_info:
            IncompleteOptimizer(params)  # type: ignore[abstract]
        assert (
            "step" in str(exc_info.value) or "abstract" in str(exc_info.value).lower()
        )


class SimpleOptimizer(Optimizer):
    """Simple Optimizer implementation for testing."""

    async def step(self) -> dict[str, str]:
        """Simple step that just returns current values."""
        if not self._bound:
            raise RuntimeError("Optimizer not bound. Call bind(resources) first.")
        updates: dict[str, str] = {}
        for param in self.params:
            if param.requires_grad and param._feedback_buffer:
                # Just uppercase the value as a simple "update"
                new_value = param.value.upper()
                param.apply_update(new_value)
                updates[param._name or str(id(param))] = new_value
        self._step_count += 1
        return updates


class TestOptimizerInit:
    """Tests for Optimizer initialization."""

    def test_optimizer_init_with_params_list(self) -> None:
        """Optimizer can be initialized with a list of parameters."""
        params = [
            Parameter("value1", description="First param"),
            Parameter("value2", description="Second param"),
        ]
        optimizer = SimpleOptimizer(params)

        assert len(optimizer.params) == 2
        assert optimizer.params[0].value == "value1"
        assert optimizer.params[1].value == "value2"

    def test_optimizer_init_with_generator(self) -> None:
        """Optimizer can be initialized with a generator of parameters."""

        def gen_params():
            yield Parameter("a", description="param a")
            yield Parameter("b", description="param b")

        optimizer = SimpleOptimizer(gen_params())

        assert len(optimizer.params) == 2
        assert optimizer.params[0].value == "a"
        assert optimizer.params[1].value == "b"

    def test_optimizer_init_empty_params(self) -> None:
        """Optimizer can be initialized with empty parameters."""
        optimizer = SimpleOptimizer([])

        assert len(optimizer.params) == 0
        assert optimizer._step_count == 0

    def test_optimizer_init_creates_internal_llms(self) -> None:
        """Optimizer creates internal LLM wrappers for aggregation and updates."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)

        # Should have aggregator and updater wrappers
        assert optimizer.aggregator is not None
        assert optimizer.updater is not None
        # Wrappers contain internal modules with the correct aliases
        assert optimizer.aggregator._module.llm.alias == Optimizer.AGGREGATOR_ALIAS
        assert optimizer.updater._module.llm.alias == Optimizer.UPDATER_ALIAS

    def test_optimizer_init_without_reasoning_model(self) -> None:
        """Optimizer without reasoning_model has no reasoning_llm."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)

        assert optimizer.reasoning_llm is None

    def test_optimizer_init_with_reasoning_model(self) -> None:
        """Optimizer with reasoning_model creates reasoning_llm wrapper."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params, reasoning_model="gpt-4o")

        assert optimizer.reasoning_llm is not None
        assert optimizer.reasoning_llm._module.llm.alias == Optimizer.REASONING_ALIAS

    def test_optimizer_init_not_bound(self) -> None:
        """Optimizer starts unbound."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)

        assert optimizer._bound is False

    def test_optimizer_init_step_count_zero(self) -> None:
        """Optimizer starts with step count of zero."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)

        assert optimizer._step_count == 0

    def test_optimizer_init_records_empty(self) -> None:
        """Optimizer starts with empty records list."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)

        assert optimizer._records == []


class TestOptimizerAliases:
    """Tests for optimizer's fixed aliases."""

    def test_aggregator_alias_constant(self) -> None:
        """AGGREGATOR_ALIAS has expected value."""
        assert Optimizer.AGGREGATOR_ALIAS == "optimizer/aggregator"

    def test_updater_alias_constant(self) -> None:
        """UPDATER_ALIAS has expected value."""
        assert Optimizer.UPDATER_ALIAS == "optimizer/updater"

    def test_reasoning_alias_constant(self) -> None:
        """REASONING_ALIAS has expected value."""
        assert Optimizer.REASONING_ALIAS == "optimizer/reasoning"


class TestOptimizerBind:
    """Tests for Optimizer.bind() method."""

    def test_bind_sets_bound_flag(self) -> None:
        """bind() sets _bound to True."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)
        mock_resources = MagicMock()

        optimizer.bind(mock_resources)

        assert optimizer._bound is True

    def test_bind_returns_self(self) -> None:
        """bind() returns self for chaining."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)
        mock_resources = MagicMock()

        result = optimizer.bind(mock_resources)

        assert result is optimizer

    def test_bind_configures_aggregator(self) -> None:
        """bind() configures the aggregator wrapper."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)
        mock_resources = MagicMock()

        optimizer.bind(mock_resources)

        # The aggregator wrapper should be bound
        assert optimizer.aggregator._bound is True

    def test_bind_configures_updater(self) -> None:
        """bind() configures the updater wrapper."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)
        mock_resources = MagicMock()

        optimizer.bind(mock_resources)

        assert optimizer.updater._bound is True

    def test_bind_configures_reasoning_llm(self) -> None:
        """bind() configures reasoning_llm wrapper when present."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params, reasoning_model="gpt-4o")
        mock_resources = MagicMock()

        optimizer.bind(mock_resources)

        assert optimizer.reasoning_llm is not None
        assert optimizer.reasoning_llm._bound is True

    def test_bind_without_reasoning_llm(self) -> None:
        """bind() works when reasoning_llm is None."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)
        mock_resources = MagicMock()

        # Should not raise
        optimizer.bind(mock_resources)

        assert optimizer.reasoning_llm is None


class TestOptimizerCaptureRecord:
    """Tests for Optimizer.capture_record() method."""

    def test_capture_record_stores_record(self) -> None:
        """capture_record() stores ForwardRecord in _records list."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)

        record = create_mock_record(params)
        optimizer.capture_record(record)

        assert len(optimizer._records) == 1
        assert optimizer._records[0] is record

    def test_capture_record_accumulates_multiple(self) -> None:
        """capture_record() accumulates multiple records."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)

        record1 = create_mock_record(params)
        record2 = create_mock_record(params)
        record3 = create_mock_record(params)

        optimizer.capture_record(record1)
        optimizer.capture_record(record2)
        optimizer.capture_record(record3)

        assert len(optimizer._records) == 3
        assert optimizer._records[0] is record1
        assert optimizer._records[1] is record2
        assert optimizer._records[2] is record3


class TestOptimizerZeroFeedback:
    """Tests for Optimizer.zero_feedback() method."""

    def test_zero_feedback_clears_all_params(self) -> None:
        """zero_feedback() clears feedback buffer for all parameters."""
        param1 = Parameter("value1", description="First param")
        param2 = Parameter("value2", description="Second param")

        # Accumulate some feedback
        param1.accumulate_feedback("feedback 1")
        param1.accumulate_feedback("feedback 2")
        param2.accumulate_feedback("feedback 3")

        assert len(param1._feedback_buffer) == 2
        assert len(param2._feedback_buffer) == 1

        optimizer = SimpleOptimizer([param1, param2])
        optimizer.zero_feedback()

        assert len(param1._feedback_buffer) == 0
        assert len(param2._feedback_buffer) == 0

    def test_zero_feedback_on_empty_buffers(self) -> None:
        """zero_feedback() works on already empty buffers."""
        param = Parameter("test", description="test")
        optimizer = SimpleOptimizer([param])

        # Should not raise
        optimizer.zero_feedback()

        assert len(param._feedback_buffer) == 0

    def test_zero_feedback_only_affects_optimizer_params(self) -> None:
        """zero_feedback() only affects parameters in the optimizer."""
        param1 = Parameter("value1", description="In optimizer")
        param2 = Parameter("value2", description="Not in optimizer")

        param1.accumulate_feedback("feedback 1")
        param2.accumulate_feedback("feedback 2")

        optimizer = SimpleOptimizer([param1])  # Only param1
        optimizer.zero_feedback()

        assert len(param1._feedback_buffer) == 0
        assert len(param2._feedback_buffer) == 1  # Unchanged

    def test_zero_feedback_clears_records(self) -> None:
        """zero_feedback() clears accumulated ForwardRecords."""
        param = Parameter("test", description="test")
        optimizer = SimpleOptimizer([param])

        # Capture some records
        record1 = create_mock_record([param])
        record2 = create_mock_record([param])
        optimizer.capture_record(record1)
        optimizer.capture_record(record2)

        assert len(optimizer._records) == 2

        optimizer.zero_feedback()

        assert len(optimizer._records) == 0


class TestOptimizerStep:
    """Tests for Optimizer.step() abstract method."""

    @pytest.mark.asyncio
    async def test_step_requires_bind(self) -> None:
        """step() raises RuntimeError if not bound."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)

        with pytest.raises(RuntimeError) as exc_info:
            await optimizer.step()

        assert "not bound" in str(exc_info.value).lower()
        assert "bind" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_step_after_bind_succeeds(self) -> None:
        """step() succeeds after bind()."""
        param = Parameter("test", description="test")
        param.accumulate_feedback("some feedback")

        optimizer = SimpleOptimizer([param])
        optimizer.bind(MagicMock())

        # Should not raise
        updates = await optimizer.step()

        assert isinstance(updates, dict)

    @pytest.mark.asyncio
    async def test_step_increments_step_count(self) -> None:
        """step() increments the step counter."""
        param = Parameter("test", description="test")
        param.accumulate_feedback("feedback")

        optimizer = SimpleOptimizer([param])
        optimizer.bind(MagicMock())

        assert optimizer._step_count == 0

        await optimizer.step()
        assert optimizer._step_count == 1

        await optimizer.step()
        assert optimizer._step_count == 2


class TestOptimizerSystemPrompts:
    """Tests for optimizer's system prompts."""

    def test_aggregator_system_prompt(self) -> None:
        """Aggregator has appropriate system prompt."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)

        # Access the internal LLM's system prompt via the wrapper
        system_prompt = optimizer.aggregator._module.llm.system_prompt
        assert system_prompt is not None
        prompt_text = system_prompt.value

        # Should mention synthesizing/aggregating feedback
        assert "synthesize" in prompt_text.lower() or "aggregate" in prompt_text.lower()

    def test_updater_system_prompt(self) -> None:
        """Updater has appropriate system prompt."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params)

        system_prompt = optimizer.updater._module.llm.system_prompt
        assert system_prompt is not None
        prompt_text = system_prompt.value

        # Should mention improving/updating parameters
        assert "improve" in prompt_text.lower() or "update" in prompt_text.lower()

    def test_reasoning_system_prompt(self) -> None:
        """Reasoning LLM has appropriate system prompt."""
        params = [Parameter("test", description="test")]
        optimizer = SimpleOptimizer(params, reasoning_model="gpt-4o")

        assert optimizer.reasoning_llm is not None
        system_prompt = optimizer.reasoning_llm._module.llm.system_prompt
        assert system_prompt is not None
        prompt_text = system_prompt.value

        # Should mention analyzing feedback
        assert "analyze" in prompt_text.lower()


# ═══════════════════════════════════════════════════════════════════════════
#  SFAOptimizer Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSFAOptimizerInit:
    """Tests for SFAOptimizer initialization."""

    def test_sfa_optimizer_creation(self) -> None:
        """SFAOptimizer can be created with parameters."""
        params = [Parameter("test", description="test param")]
        optimizer = SFAOptimizer(params)

        assert len(optimizer.params) == 1
        assert optimizer.conservatism == 0.7  # Default

    def test_sfa_optimizer_custom_conservatism(self) -> None:
        """SFAOptimizer accepts custom conservatism value."""
        params = [Parameter("test", description="test")]
        optimizer = SFAOptimizer(params, conservatism=0.5)

        assert optimizer.conservatism == 0.5

    def test_sfa_optimizer_conservatism_bounds(self) -> None:
        """SFAOptimizer validates conservatism is in [0, 1]."""
        params = [Parameter("test", description="test")]

        # Valid boundary values
        SFAOptimizer(params, conservatism=0.0)
        SFAOptimizer(params, conservatism=1.0)

        # Invalid values
        with pytest.raises(ValueError):
            SFAOptimizer(params, conservatism=-0.1)

        with pytest.raises(ValueError):
            SFAOptimizer(params, conservatism=1.1)

    def test_sfa_optimizer_with_reasoning_model(self) -> None:
        """SFAOptimizer accepts reasoning_model parameter."""
        params = [Parameter("test", description="test")]
        optimizer = SFAOptimizer(params, reasoning_model="gpt-4o")

        assert optimizer.reasoning_llm is not None

    def test_sfa_optimizer_inherits_from_optimizer(self) -> None:
        """SFAOptimizer inherits from Optimizer."""
        params = [Parameter("test", description="test")]
        optimizer = SFAOptimizer(params)

        assert isinstance(optimizer, Optimizer)


class TestSFAOptimizerStep:
    """Tests for SFAOptimizer.step() method."""

    @pytest.mark.asyncio
    async def test_step_requires_bind(self) -> None:
        """step() raises RuntimeError if not bound."""
        params = [Parameter("test", description="test")]
        optimizer = SFAOptimizer(params)

        with pytest.raises(RuntimeError) as exc_info:
            await optimizer.step()

        assert "not bound" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_step_requires_records(self) -> None:
        """step() raises RuntimeError if no records captured."""
        param = Parameter("test", description="test")
        param._name = "param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param])
        optimizer._bound = True

        with pytest.raises(RuntimeError) as exc_info:
            await optimizer.step()

        assert "No ForwardRecords captured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_step_skips_params_without_feedback(self) -> None:
        """step() skips parameters with empty feedback buffer."""
        param1 = Parameter("value1", description="Has feedback")
        param1._name = "param1"
        param2 = Parameter("value2", description="No feedback")
        param2._name = "param2"

        param1.accumulate_feedback("some feedback")
        # param2 has no feedback

        optimizer = SFAOptimizer([param1, param2])

        # Provide a mock record
        record = create_mock_record([param1, param2])
        optimizer.capture_record(record)

        # Mock the internal LLMs
        optimizer.aggregator = AsyncMock(return_value="aggregated feedback")
        optimizer.updater = AsyncMock(return_value="updated value1")
        optimizer._bound = True

        updates = await optimizer.step()

        # Only param1 should be updated
        assert len(updates) == 1
        assert "param1" in updates

    @pytest.mark.asyncio
    async def test_step_skips_non_grad_params(self) -> None:
        """step() skips parameters with requires_grad=False."""
        param = Parameter("test", description="Frozen param", requires_grad=False)
        param._name = "param"
        param._feedback_buffer = ["feedback"]  # Bypass accumulate_feedback check

        optimizer = SFAOptimizer([param])
        record = create_mock_record([param])
        optimizer.capture_record(record)
        optimizer._bound = True

        updates = await optimizer.step()

        # Should not update frozen params
        assert len(updates) == 0

    @pytest.mark.asyncio
    async def test_step_aggregates_feedback(self) -> None:
        """step() calls aggregator when multiple feedback items."""
        param = Parameter("original", description="test param")
        param._name = "test_param"
        param.accumulate_feedback("feedback 1")
        param.accumulate_feedback("feedback 2")
        param.accumulate_feedback("feedback 3")

        optimizer = SFAOptimizer([param])
        record = create_mock_record([param])
        optimizer.capture_record(record)
        optimizer.aggregator = AsyncMock(return_value="aggregated: all three feedbacks")
        optimizer.updater = AsyncMock(return_value="updated value")
        optimizer._bound = True

        await optimizer.step()

        # Aggregator should be called
        optimizer.aggregator.assert_called_once()
        call_prompt = optimizer.aggregator.call_args[0][0]

        # Should include all feedback items (using XML format now)
        assert "feedback 1" in call_prompt
        assert "feedback 2" in call_prompt
        assert "feedback 3" in call_prompt
        assert 'count="3"' in call_prompt

    @pytest.mark.asyncio
    async def test_step_skips_aggregation_for_single_feedback(self) -> None:
        """step() skips aggregation when only one feedback item."""
        param = Parameter("original", description="test param")
        param._name = "test_param"
        param.accumulate_feedback("single feedback")

        optimizer = SFAOptimizer([param])
        record = create_mock_record([param])
        optimizer.capture_record(record)
        optimizer.aggregator = AsyncMock()
        optimizer.updater = AsyncMock(return_value="updated value")
        optimizer._bound = True

        await optimizer.step()

        # Aggregator should NOT be called
        optimizer.aggregator.assert_not_called()

        # Updater should receive the single feedback directly
        call_prompt = optimizer.updater.call_args[0][0]
        assert "single feedback" in call_prompt

    @pytest.mark.asyncio
    async def test_step_includes_param_description(self) -> None:
        """step() includes parameter description in prompts."""
        param = Parameter(
            "original value",
            description="This is the system prompt that defines agent behavior",
        )
        param._name = "system_prompt"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param])
        record = create_mock_record([param])
        optimizer.capture_record(record)
        optimizer.aggregator = AsyncMock()
        optimizer.updater = AsyncMock(return_value="updated")
        optimizer._bound = True

        await optimizer.step()

        call_prompt = optimizer.updater.call_args[0][0]
        assert "This is the system prompt that defines agent behavior" in call_prompt

    @pytest.mark.asyncio
    async def test_step_updates_param_value(self) -> None:
        """step() updates parameter value with LLM output."""
        param = Parameter("old value", description="test")
        param._name = "param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param])
        record = create_mock_record([param])
        optimizer.capture_record(record)
        optimizer.updater = AsyncMock(return_value="new improved value")
        optimizer._bound = True

        await optimizer.step()

        assert param.value == "new improved value"

    @pytest.mark.asyncio
    async def test_step_clears_feedback_buffer(self) -> None:
        """step() clears feedback buffer after update."""
        param = Parameter("value", description="test")
        param._name = "param"
        param.accumulate_feedback("feedback 1")
        param.accumulate_feedback("feedback 2")

        assert len(param._feedback_buffer) == 2

        optimizer = SFAOptimizer([param])
        record = create_mock_record([param])
        optimizer.capture_record(record)
        optimizer.aggregator = AsyncMock(return_value="aggregated feedback")
        optimizer.updater = AsyncMock(return_value="updated")
        optimizer._bound = True

        await optimizer.step()

        assert len(param._feedback_buffer) == 0

    @pytest.mark.asyncio
    async def test_step_returns_updates_dict(self) -> None:
        """step() returns dict of parameter name to new value."""
        param = Parameter("old", description="test")
        param._name = "my_param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param])
        record = create_mock_record([param])
        optimizer.capture_record(record)
        optimizer.updater = AsyncMock(return_value="new value")
        optimizer._bound = True

        updates = await optimizer.step()

        assert isinstance(updates, dict)
        assert "my_param" in updates
        assert updates["my_param"] == "new value"

    @pytest.mark.asyncio
    async def test_step_increments_counter(self) -> None:
        """step() increments step count."""
        param = Parameter("test", description="test")
        param._name = "param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param])
        record = create_mock_record([param])
        optimizer.capture_record(record)
        optimizer.updater = AsyncMock(return_value="updated")
        optimizer._bound = True

        assert optimizer._step_count == 0

        await optimizer.step()
        assert optimizer._step_count == 1

        # For next step, need to re-add record and feedback
        optimizer.capture_record(record)
        param.accumulate_feedback("more feedback")
        await optimizer.step()
        assert optimizer._step_count == 2


class TestSFAOptimizerConservatism:
    """Tests for SFAOptimizer conservatism affecting prompts."""

    @pytest.mark.asyncio
    async def test_conservatism_in_update_prompt(self) -> None:
        """Conservatism value is included in update prompt."""
        param = Parameter("test", description="test param")
        param._name = "param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param], conservatism=0.8)
        record = create_mock_record([param])
        optimizer.capture_record(record)
        optimizer.updater = AsyncMock(return_value="updated")
        optimizer._bound = True

        await optimizer.step()

        call_prompt = optimizer.updater.call_args[0][0]
        assert "0.8" in call_prompt
        assert "conservatism" in call_prompt.lower()

    @pytest.mark.asyncio
    async def test_low_conservatism_prompt(self) -> None:
        """Low conservatism indicates aggressive changes."""
        param = Parameter("test", description="test")
        param._name = "param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param], conservatism=0.2)
        record = create_mock_record([param])
        optimizer.capture_record(record)
        optimizer.updater = AsyncMock(return_value="updated")
        optimizer._bound = True

        await optimizer.step()

        call_prompt = optimizer.updater.call_args[0][0]
        assert "0.2" in call_prompt
        # Prompt should explain the scale (0=aggressive, 1=minimal)
        assert "aggressive" in call_prompt.lower() or "0 =" in call_prompt

    @pytest.mark.asyncio
    async def test_high_conservatism_prompt(self) -> None:
        """High conservatism indicates minimal changes."""
        param = Parameter("test", description="test")
        param._name = "param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param], conservatism=0.9)
        record = create_mock_record([param])
        optimizer.capture_record(record)
        optimizer.updater = AsyncMock(return_value="updated")
        optimizer._bound = True

        await optimizer.step()

        call_prompt = optimizer.updater.call_args[0][0]
        assert "0.9" in call_prompt
        # Prompt should explain the scale
        assert "minimal" in call_prompt.lower() or "1 =" in call_prompt


class TestSFAOptimizerMultipleParams:
    """Tests for SFAOptimizer with multiple parameters."""

    @pytest.mark.asyncio
    async def test_step_updates_all_params_with_feedback(self) -> None:
        """step() updates all parameters that have feedback."""
        param1 = Parameter("value1", description="First param")
        param1._name = "param1"
        param1.accumulate_feedback("feedback for 1")

        param2 = Parameter("value2", description="Second param")
        param2._name = "param2"
        param2.accumulate_feedback("feedback for 2")

        optimizer = SFAOptimizer([param1, param2])
        record = create_mock_record([param1, param2])
        optimizer.capture_record(record)
        optimizer.updater = AsyncMock(side_effect=["updated1", "updated2"])
        optimizer._bound = True

        updates = await optimizer.step()

        assert len(updates) == 2
        assert param1.value == "updated1"
        assert param2.value == "updated2"

    @pytest.mark.asyncio
    async def test_step_with_mixed_feedback(self) -> None:
        """step() handles mix of params with and without feedback."""
        param1 = Parameter("value1", description="Has feedback")
        param1._name = "param1"
        param1.accumulate_feedback("feedback")

        param2 = Parameter("value2", description="No feedback")
        param2._name = "param2"
        # No feedback for param2

        param3 = Parameter("value3", description="Also has feedback")
        param3._name = "param3"
        param3.accumulate_feedback("more feedback")

        optimizer = SFAOptimizer([param1, param2, param3])
        # Create a record that includes all params
        record = create_mock_record([param1, param2, param3])
        optimizer.capture_record(record)
        optimizer.updater = AsyncMock(side_effect=["new1", "new3"])
        optimizer._bound = True

        updates = await optimizer.step()

        assert len(updates) == 2
        assert "param1" in updates
        assert "param2" not in updates
        assert "param3" in updates

        assert param1.value == "new1"
        assert param2.value == "value2"  # Unchanged
        assert param3.value == "new3"


class TestSFAOptimizerIntegration:
    """Integration tests for SFAOptimizer workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self) -> None:
        """Test complete zero_feedback -> accumulate -> step workflow."""
        param = Parameter("initial prompt", description="System prompt")
        param._name = "system_prompt"

        optimizer = SFAOptimizer([param], conservatism=0.6)
        optimizer.aggregator = AsyncMock(return_value="combined feedback")
        optimizer.updater = AsyncMock(return_value="improved prompt")
        optimizer._bound = True

        # Simulate mini-batch training
        optimizer.zero_feedback()

        # Forward + backward for sample 1
        param.accumulate_feedback("Sample 1: too verbose")
        record1 = create_mock_record([param])
        optimizer.capture_record(record1)

        # Forward + backward for sample 2
        param.accumulate_feedback("Sample 2: good structure")
        record2 = create_mock_record([param])
        optimizer.capture_record(record2)

        # Forward + backward for sample 3
        param.accumulate_feedback("Sample 3: needs examples")
        record3 = create_mock_record([param])
        optimizer.capture_record(record3)

        # Optimizer step
        updates = await optimizer.step()

        # Verify aggregator was called with all feedback
        agg_prompt = optimizer.aggregator.call_args[0][0]
        assert "too verbose" in agg_prompt
        assert "good structure" in agg_prompt
        assert "needs examples" in agg_prompt

        # Verify updater received aggregated feedback
        update_prompt = optimizer.updater.call_args[0][0]
        assert "combined feedback" in update_prompt
        assert "0.6" in update_prompt

        # Verify parameter was updated
        assert param.value == "improved prompt"
        assert "system_prompt" in updates

    @pytest.mark.asyncio
    async def test_multiple_epochs(self) -> None:
        """Test optimizer across multiple training epochs."""
        param = Parameter("v1", description="Evolving param")
        param._name = "param"

        optimizer = SFAOptimizer([param])
        # Track what values the updater returns
        update_values = ["v2", "v3", "v4"]
        optimizer.updater = AsyncMock(side_effect=update_values)
        optimizer._bound = True

        # Epoch 1
        optimizer.zero_feedback()
        param.accumulate_feedback("epoch 1 feedback")
        record = create_mock_record([param])
        optimizer.capture_record(record)
        await optimizer.step()
        assert param.value == "v2"
        assert optimizer._step_count == 1

        # Epoch 2
        optimizer.zero_feedback()
        param.accumulate_feedback("epoch 2 feedback")
        optimizer.capture_record(record)
        await optimizer.step()
        assert param.value == "v3"
        assert optimizer._step_count == 2

        # Epoch 3
        optimizer.zero_feedback()
        param.accumulate_feedback("epoch 3 feedback")
        optimizer.capture_record(record)
        await optimizer.step()
        assert param.value == "v4"
        assert optimizer._step_count == 3


class TestSFAOptimizerTopologicalOrder:
    """Tests for topologically-ordered parameter updates."""

    def _create_linear_graph_record(
        self,
        params: list[Parameter],
    ) -> ForwardRecord:
        """Create a record with linear dependency: node0 -> node1 -> node2.

        Each param is placed in a separate node for proper topological order testing.
        """
        node_ids = [f"node_{i}" for i in range(len(params))]
        module_map: dict[str, InferenceModule] = {}

        for i, (node_id, param) in enumerate(zip(node_ids, params, strict=True)):
            mock_module = MagicMock(spec=LLMInference)
            mock_module.named_parameters.return_value = [
                (param._name or f"p{i}", param)
            ]
            mock_module.parameters.return_value = [param]
            module_map[node_id] = mock_module

        # Create nodes with linear dependencies
        nodes: dict[str, GraphNode] = {}
        for i, node_id in enumerate(node_ids):
            deps = [node_ids[i - 1]] if i > 0 else []
            nodes[node_id] = GraphNode(
                id=node_id,
                module=module_map[node_id],
                args=(),
                kwargs={},
                dependencies=deps,
                module_name=f"Module_{i}",
            )

        graph = InferenceGraph(
            nodes=nodes,
            input_ids=[node_ids[0]] if node_ids else [],
            output_ids=[node_ids[-1]] if node_ids else [],
        )

        return ForwardRecord(
            graph=graph,
            node_inputs={nid: {} for nid in node_ids},
            node_outputs={nid: f"output_{i}" for i, nid in enumerate(node_ids)},
            module_map=module_map,
        )

    @pytest.mark.asyncio
    async def test_step_processes_in_topological_order(self) -> None:
        """step() processes parameters in forward topological order."""
        param1 = Parameter("value1", description="First in pipeline")
        param1._name = "upstream_param"
        param2 = Parameter("value2", description="Second in pipeline")
        param2._name = "downstream_param"

        param1.accumulate_feedback("feedback 1")
        param2.accumulate_feedback("feedback 2")

        optimizer = SFAOptimizer([param1, param2])

        # Create a linear graph where param1 is upstream of param2
        record = self._create_linear_graph_record([param1, param2])
        optimizer.capture_record(record)

        # Track the order of updates
        update_order: list[str] = []

        async def mock_updater(prompt: str) -> str:
            # Check for the parameter being updated (in <parameter name="...">)
            if '<parameter name="upstream_param">' in prompt:
                update_order.append("upstream_param")
                return "updated_upstream"
            else:
                update_order.append("downstream_param")
                return "updated_downstream"

        optimizer.updater = mock_updater
        optimizer._bound = True

        await optimizer.step()

        # Upstream should be processed before downstream
        assert update_order == ["upstream_param", "downstream_param"]

    @pytest.mark.asyncio
    async def test_step_includes_upstream_context(self) -> None:
        """step() includes upstream changes in downstream update prompts."""
        param1 = Parameter("original_format", description="Format specification")
        param1._name = "format_spec"
        param2 = Parameter("original_validator", description="Validation rules")
        param2._name = "validator"

        param1.accumulate_feedback("change to YAML")
        param2.accumulate_feedback("update validation")

        optimizer = SFAOptimizer([param1, param2])

        # Create a linear graph where param1 is upstream of param2
        record = self._create_linear_graph_record([param1, param2])
        optimizer.capture_record(record)

        captured_prompts: list[str] = []

        async def mock_updater(prompt: str) -> str:
            captured_prompts.append(prompt)
            if "format_spec" in prompt:
                return "YAML format"
            else:
                return "updated validator"

        optimizer.updater = mock_updater
        optimizer._bound = True

        await optimizer.step()

        # The second prompt (downstream) should have upstream context
        assert len(captured_prompts) == 2

        # First prompt should NOT have upstream-updates
        assert "<upstream-updates>" not in captured_prompts[0]

        # Second prompt should have upstream context showing the format change
        assert "<upstream-updates>" in captured_prompts[1]
        assert "format_spec" in captured_prompts[1]
        assert "YAML format" in captured_prompts[1]

    @pytest.mark.asyncio
    async def test_step_uses_xml_formatted_prompts(self) -> None:
        """step() uses XML tags for content demarcation in prompts."""
        param = Parameter("test value", description="Test param")
        param._name = "param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param])
        record = create_mock_record([param])
        optimizer.capture_record(record)

        captured_prompt = ""

        async def mock_updater(prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return "updated"

        optimizer.updater = mock_updater
        optimizer._bound = True

        await optimizer.step()

        # Check for XML structure
        assert "<task>" in captured_prompt
        assert "<parameter" in captured_prompt
        assert "<description>" in captured_prompt
        assert "<current-value>" in captured_prompt
        assert "<aggregated-feedback>" in captured_prompt
        assert "<update-guidelines>" in captured_prompt
        assert "<output-format>" in captured_prompt


class TestOptimizerExportFromPackage:
    """Tests for optimizer exports from package."""

    def test_exports_from_optimization_package(self) -> None:
        """Optimizer and SFAOptimizer are exported from optimization package."""
        from inf_engine.optimization import Optimizer, SFAOptimizer

        params = [Parameter("test", description="test")]
        optimizer = SFAOptimizer(params)

        assert isinstance(optimizer, Optimizer)


class TestSFAOptimizerRetryBehavior:
    """Tests for SFAOptimizer retry behavior on empty/invalid updates."""

    def test_max_retries_default(self) -> None:
        """SFAOptimizer has default max_retries of 3."""
        params = [Parameter("test", description="test")]
        optimizer = SFAOptimizer(params)
        assert optimizer.max_retries == 3

    def test_max_retries_custom(self) -> None:
        """SFAOptimizer accepts custom max_retries value."""
        params = [Parameter("test", description="test")]
        optimizer = SFAOptimizer(params, max_retries=5)
        assert optimizer.max_retries == 5

    def test_max_retries_zero(self) -> None:
        """SFAOptimizer accepts max_retries=0 (no retries)."""
        params = [Parameter("test", description="test")]
        optimizer = SFAOptimizer(params, max_retries=0)
        assert optimizer.max_retries == 0

    def test_max_retries_negative_raises(self) -> None:
        """SFAOptimizer rejects negative max_retries."""
        params = [Parameter("test", description="test")]
        with pytest.raises(ValueError) as exc_info:
            SFAOptimizer(params, max_retries=-1)
        assert "non-negative" in str(exc_info.value).lower()

    def test_validate_update_empty_string(self) -> None:
        """_validate_update returns False for empty string."""
        params = [Parameter("test", description="test")]
        optimizer = SFAOptimizer(params)
        assert optimizer._validate_update("") is False

    def test_validate_update_whitespace_only(self) -> None:
        """_validate_update returns False for whitespace-only string."""
        params = [Parameter("test", description="test")]
        optimizer = SFAOptimizer(params)
        assert optimizer._validate_update("   ") is False
        assert optimizer._validate_update("\n\t") is False

    def test_validate_update_valid_string(self) -> None:
        """_validate_update returns True for valid non-empty string."""
        params = [Parameter("test", description="test")]
        optimizer = SFAOptimizer(params)
        assert optimizer._validate_update("valid update") is True
        assert optimizer._validate_update("  content  ") is True

    @pytest.mark.asyncio
    async def test_step_retries_on_empty_response(self) -> None:
        """step() retries when updater returns empty response."""

        param = Parameter("test", description="test param")
        param._name = "param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param], max_retries=2)
        record = create_mock_record([param])
        optimizer.capture_record(record)

        # Track call count
        call_count = 0

        async def mock_updater(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # First 2 calls return empty
                return ""
            return "valid update"  # Third call succeeds

        optimizer.updater = mock_updater
        optimizer._bound = True

        updates = await optimizer.step()

        # Should have retried and succeeded
        assert call_count == 3
        assert param.value == "valid update"
        assert "param" in updates

    @pytest.mark.asyncio
    async def test_step_raises_after_exhausting_retries(self) -> None:
        """step() raises OptimizationError after exhausting retries."""
        from inf_engine.errors import OptimizationError

        param = Parameter("test", description="test param")
        param._name = "my_param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param], max_retries=2)
        record = create_mock_record([param])
        optimizer.capture_record(record)

        # Always return empty
        optimizer.updater = AsyncMock(return_value="")
        optimizer._bound = True

        with pytest.raises(OptimizationError) as exc_info:
            await optimizer.step()

        error = exc_info.value
        assert error.parameter_name == "my_param"
        assert error.attempts == 3  # 1 initial + 2 retries
        assert "my_param" in str(error)

    @pytest.mark.asyncio
    async def test_step_no_retry_on_valid_response(self) -> None:
        """step() doesn't retry when updater returns valid response."""
        param = Parameter("test", description="test param")
        param._name = "param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param], max_retries=3)
        record = create_mock_record([param])
        optimizer.capture_record(record)

        call_count = 0

        async def mock_updater(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "valid update"

        optimizer.updater = mock_updater
        optimizer._bound = True

        await optimizer.step()

        # Should only be called once (no retries needed)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_step_with_zero_retries_fails_immediately(self) -> None:
        """step() fails immediately with max_retries=0 on empty response."""
        from inf_engine.errors import OptimizationError

        param = Parameter("test", description="test param")
        param._name = "param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param], max_retries=0)
        record = create_mock_record([param])
        optimizer.capture_record(record)

        call_count = 0

        async def mock_updater(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return ""

        optimizer.updater = mock_updater
        optimizer._bound = True

        with pytest.raises(OptimizationError) as exc_info:
            await optimizer.step()

        # Only one attempt (no retries)
        assert call_count == 1
        assert exc_info.value.attempts == 1

    @pytest.mark.asyncio
    async def test_step_retries_on_whitespace_response(self) -> None:
        """step() treats whitespace-only response as invalid and retries."""
        param = Parameter("test", description="test param")
        param._name = "param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param], max_retries=1)
        record = create_mock_record([param])
        optimizer.capture_record(record)

        call_count = 0

        async def mock_updater(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "   \n\t  "  # Whitespace only
            return "valid"

        optimizer.updater = mock_updater
        optimizer._bound = True

        await optimizer.step()

        assert call_count == 2
        assert param.value == "valid"


class TestOptimizationErrorExport:
    """Tests for OptimizationError exception class."""

    def test_optimization_error_import(self) -> None:
        """OptimizationError can be imported from errors module."""
        from inf_engine.errors import OptimizationError

        error = OptimizationError("test error")
        assert str(error) == "test error"

    def test_optimization_error_with_parameter_name(self) -> None:
        """OptimizationError stores parameter_name attribute."""
        from inf_engine.errors import OptimizationError

        error = OptimizationError(
            "Failed to update parameter",
            parameter_name="system_prompt",
        )
        assert error.parameter_name == "system_prompt"

    def test_optimization_error_with_attempts(self) -> None:
        """OptimizationError stores attempts attribute."""
        from inf_engine.errors import OptimizationError

        error = OptimizationError(
            "Failed after retries",
            attempts=3,
        )
        assert error.attempts == 3

    def test_optimization_error_inherits_from_inf_engine_error(self) -> None:
        """OptimizationError inherits from InfEngineError."""
        from inf_engine.errors import InfEngineError, OptimizationError

        error = OptimizationError("test")
        assert isinstance(error, InfEngineError)
