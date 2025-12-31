"""Tests for Optimizer abstract base class and SFAOptimizer implementation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from inf_engine.optimization.optimizer import Optimizer, SFAOptimizer
from inf_engine.parameter import Parameter


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
    async def test_step_skips_params_without_feedback(self) -> None:
        """step() skips parameters with empty feedback buffer."""
        param1 = Parameter("value1", description="Has feedback")
        param2 = Parameter("value2", description="No feedback")

        param1.accumulate_feedback("some feedback")
        # param2 has no feedback

        optimizer = SFAOptimizer([param1, param2])

        # Mock the internal LLMs
        optimizer.aggregator = AsyncMock(return_value="aggregated feedback")
        optimizer.updater = AsyncMock(return_value="updated value1")
        optimizer._bound = True

        updates = await optimizer.step()

        # Only param1 should be updated
        assert len(updates) == 1
        assert param1._name in updates or "value1" in str(updates)

    @pytest.mark.asyncio
    async def test_step_skips_non_grad_params(self) -> None:
        """step() skips parameters with requires_grad=False."""
        param = Parameter("test", description="Frozen param", requires_grad=False)
        param._feedback_buffer = ["feedback"]  # Bypass accumulate_feedback check

        optimizer = SFAOptimizer([param])
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
        optimizer.aggregator = AsyncMock(return_value="aggregated: all three feedbacks")
        optimizer.updater = AsyncMock(return_value="updated value")
        optimizer._bound = True

        await optimizer.step()

        # Aggregator should be called
        optimizer.aggregator.assert_called_once()
        call_prompt = optimizer.aggregator.call_args[0][0]

        # Should include all feedback items
        assert "feedback 1" in call_prompt
        assert "feedback 2" in call_prompt
        assert "feedback 3" in call_prompt
        assert "3 feedback items" in call_prompt

    @pytest.mark.asyncio
    async def test_step_skips_aggregation_for_single_feedback(self) -> None:
        """step() skips aggregation when only one feedback item."""
        param = Parameter("original", description="test param")
        param._name = "test_param"
        param.accumulate_feedback("single feedback")

        optimizer = SFAOptimizer([param])
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
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param])
        optimizer.updater = AsyncMock(return_value="updated")
        optimizer._bound = True

        assert optimizer._step_count == 0

        await optimizer.step()
        assert optimizer._step_count == 1

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
        optimizer.updater = AsyncMock(return_value="updated")
        optimizer._bound = True

        await optimizer.step()

        call_prompt = optimizer.updater.call_args[0][0]
        assert "0.2" in call_prompt
        # Prompt should explain the scale (0=aggressive, 1=minimal)
        assert "aggressive" in call_prompt.lower() or "0=" in call_prompt

    @pytest.mark.asyncio
    async def test_high_conservatism_prompt(self) -> None:
        """High conservatism indicates minimal changes."""
        param = Parameter("test", description="test")
        param._name = "param"
        param.accumulate_feedback("feedback")

        optimizer = SFAOptimizer([param], conservatism=0.9)
        optimizer.updater = AsyncMock(return_value="updated")
        optimizer._bound = True

        await optimizer.step()

        call_prompt = optimizer.updater.call_args[0][0]
        assert "0.9" in call_prompt
        # Prompt should explain the scale
        assert "minimal" in call_prompt.lower() or "1=" in call_prompt


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

        # Forward + backward for sample 2
        param.accumulate_feedback("Sample 2: good structure")

        # Forward + backward for sample 3
        param.accumulate_feedback("Sample 3: needs examples")

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
        await optimizer.step()
        assert param.value == "v2"
        assert optimizer._step_count == 1

        # Epoch 2
        optimizer.zero_feedback()
        param.accumulate_feedback("epoch 2 feedback")
        await optimizer.step()
        assert param.value == "v3"
        assert optimizer._step_count == 2

        # Epoch 3
        optimizer.zero_feedback()
        param.accumulate_feedback("epoch 3 feedback")
        await optimizer.step()
        assert param.value == "v4"
        assert optimizer._step_count == 3


class TestOptimizerExportFromPackage:
    """Tests for optimizer exports from package."""

    def test_exports_from_optimization_package(self) -> None:
        """Optimizer and SFAOptimizer are exported from optimization package."""
        from inf_engine.optimization import Optimizer, SFAOptimizer

        params = [Parameter("test", description="test")]
        optimizer = SFAOptimizer(params)

        assert isinstance(optimizer, Optimizer)
