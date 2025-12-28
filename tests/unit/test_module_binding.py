"""Unit tests for InferenceModule binding and ExecutionSettings integration.

This file contains tests for:
- PR-046: bind() method for direct module execution
- PR-047: __call__ behavior for bound modules
- PR-048: Batch execution support
- PR-059: ExecutionSettings context integration
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inf_engine.execution.context import ExecutionSettings, get_execution_settings
from inf_engine.module import InferenceModule, LLMInference


@pytest.fixture(autouse=True)
def clean_context() -> None:
    """Ensure execution settings context is clean before each test."""
    from inf_engine.execution.context import _execution_settings

    current = get_execution_settings()
    while current is not None:
        if current._token is not None:
            _execution_settings.reset(current._token)
            current._token = None
        current = get_execution_settings()


class SimpleModule(InferenceModule):
    """Simple test module that returns transformed input."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, text: str) -> str:
        return text.upper()


class NestedModule(InferenceModule):
    """Module with a nested child module."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = SimpleModule()

    def forward(self, text: str) -> str:
        return f"outer({self.inner(text)})"


# ─────────────────────────────────────────────────────────────────────────────
# bind() Method Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBindMethod:
    """Tests for InferenceModule.bind() method."""

    def test_bind_stores_resources(self) -> None:
        """bind() stores the provided resources."""
        module = SimpleModule()
        mock_resources = MagicMock()

        module.bind(resources=mock_resources)

        assert module._bound_resources is mock_resources

    def test_bind_stores_max_concurrent(self) -> None:
        """bind() stores max_concurrent in bound config."""
        module = SimpleModule()
        mock_resources = MagicMock()

        module.bind(resources=mock_resources, max_concurrent=50)

        assert module._bound_config["max_concurrent"] == 50

    def test_bind_default_max_concurrent(self) -> None:
        """bind() defaults max_concurrent to 100."""
        module = SimpleModule()
        mock_resources = MagicMock()

        module.bind(resources=mock_resources)

        assert module._bound_config["max_concurrent"] == 100

    def test_bind_stores_additional_kwargs(self) -> None:
        """bind() stores additional kwargs in bound config."""
        module = SimpleModule()
        mock_resources = MagicMock()

        module.bind(
            resources=mock_resources,
            checkpoint_dir="/data/checkpoints",
            execution_id="test_run",
        )

        assert module._bound_config["checkpoint_dir"] == "/data/checkpoints"
        assert module._bound_config["execution_id"] == "test_run"

    def test_bind_returns_self(self) -> None:
        """bind() returns self for method chaining."""
        module = SimpleModule()
        mock_resources = MagicMock()

        result = module.bind(resources=mock_resources)

        assert result is module

    def test_bind_method_chaining(self) -> None:
        """bind() supports method chaining."""
        mock_resources = MagicMock()

        # Can chain bind() with instantiation
        module = SimpleModule().bind(resources=mock_resources)

        assert module._bound_resources is mock_resources

    def test_bind_overwrites_previous_binding(self) -> None:
        """bind() overwrites any previous binding."""
        module = SimpleModule()
        resources1 = MagicMock()
        resources2 = MagicMock()

        module.bind(resources=resources1, max_concurrent=10)
        module.bind(resources=resources2, max_concurrent=20)

        assert module._bound_resources is resources2
        assert module._bound_config["max_concurrent"] == 20


# ─────────────────────────────────────────────────────────────────────────────
# __call__ with Resources Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallWithResources:
    """Tests for __call__ behavior when resources are available."""

    def test_call_without_resources_calls_forward_directly(self) -> None:
        """Without resources, __call__ delegates to forward()."""
        module = SimpleModule()

        result = module("hello")

        assert result == "HELLO"

    def test_call_with_bound_resources_returns_coroutine(self) -> None:
        """With bound resources, __call__ returns a coroutine."""
        import asyncio

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        result = module("hello")

        # Should be a coroutine
        assert asyncio.iscoroutine(result)
        # Clean up the coroutine
        result.close()

    def test_call_with_context_resources_returns_coroutine(self) -> None:
        """With ExecutionSettings resources, __call__ returns a coroutine."""
        import asyncio

        module = SimpleModule()
        mock_resources = MagicMock()

        with ExecutionSettings(resources=mock_resources):
            result = module("hello")

            # Should be a coroutine
            assert asyncio.iscoroutine(result)
            # Clean up the coroutine
            result.close()

    def test_call_outside_context_without_binding_calls_forward(self) -> None:
        """Outside context without binding, __call__ calls forward()."""
        module = SimpleModule()

        result = module("hello")

        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_call_with_resources_executes_module(self) -> None:
        """Module with resources is traced and executed."""
        module = SimpleModule()
        mock_resources = MagicMock()

        # Mock the run() function to return a predictable result
        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "EXECUTED_HELLO"

            module.bind(resources=mock_resources)
            result = await module("hello")

            assert result == "EXECUTED_HELLO"
            mock_run.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# _execute_bound Configuration Priority Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExecuteBoundConfigPriority:
    """Tests for _execute_bound configuration priority order."""

    @pytest.mark.asyncio
    async def test_call_time_kwargs_override_bound_config(self) -> None:
        """Call-time kwargs have highest priority."""
        module = SimpleModule()
        mock_resources = MagicMock()

        module.bind(resources=mock_resources, max_concurrent=10)

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            # Pass max_concurrent at call time
            await module("hello", max_concurrent=5)

            # Verify max_concurrent=5 was passed
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["max_concurrent"] == 5

    @pytest.mark.asyncio
    async def test_bound_config_overrides_context_config(self) -> None:
        """Bound config has higher priority than context config."""
        module = SimpleModule()
        mock_resources = MagicMock()

        module.bind(resources=mock_resources, max_concurrent=25)

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            with ExecutionSettings(max_concurrent=50):
                await module("hello")

            # Bound config should override context
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["max_concurrent"] == 25

    @pytest.mark.asyncio
    async def test_context_config_used_when_not_bound(self) -> None:
        """Context config is used when not overridden by binding."""
        module = SimpleModule()
        mock_resources = MagicMock()

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            with ExecutionSettings(resources=mock_resources, max_concurrent=75):
                await module("hello")

            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["max_concurrent"] == 75

    @pytest.mark.asyncio
    async def test_bound_resources_override_context_resources(self) -> None:
        """Bound resources have priority over context resources."""
        module = SimpleModule()
        bound_resources = MagicMock(name="bound")
        context_resources = MagicMock(name="context")

        module.bind(resources=bound_resources)

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            with ExecutionSettings(resources=context_resources):
                await module("hello")

            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["resources"] is bound_resources

    @pytest.mark.asyncio
    async def test_context_resources_used_when_not_bound(self) -> None:
        """Context resources are used when module has no bound resources."""
        module = SimpleModule()
        context_resources = MagicMock(name="context")

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            with ExecutionSettings(resources=context_resources):
                await module("hello")

            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["resources"] is context_resources

    @pytest.mark.asyncio
    async def test_checkpoint_dir_from_context(self, tmp_path: Path) -> None:
        """checkpoint_dir from context is passed to run()."""
        module = SimpleModule()
        mock_resources = MagicMock()
        checkpoint_dir = tmp_path / "checkpoints"

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            with ExecutionSettings(
                resources=mock_resources, checkpoint_dir=checkpoint_dir
            ):
                await module("hello")

            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["checkpoint_dir"] == checkpoint_dir


# ─────────────────────────────────────────────────────────────────────────────
# Batch Execution Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchExecution:
    """Tests for batch execution support."""

    @pytest.mark.asyncio
    async def test_batch_execution_with_list_input(self) -> None:
        """List input triggers batch execution."""
        module = SimpleModule()
        mock_resources = MagicMock()

        module.bind(resources=mock_resources)

        with patch("inf_engine.execution.executor.run") as mock_run:
            # Return different results for each call
            mock_run.side_effect = ["RESULT1", "RESULT2", "RESULT3"]

            results = await module(["input1", "input2", "input3"])

            # Should call run() once for each input
            assert mock_run.call_count == 3
            assert results == ["RESULT1", "RESULT2", "RESULT3"]

    @pytest.mark.asyncio
    async def test_batch_execution_passes_same_resources(self) -> None:
        """Batch execution passes same resources to each run()."""
        module = SimpleModule()
        mock_resources = MagicMock()

        module.bind(resources=mock_resources)

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.side_effect = ["R1", "R2"]

            await module(["input1", "input2"])

            # Both calls should have same resources
            for call in mock_run.call_args_list:
                assert call.kwargs["resources"] is mock_resources

    @pytest.mark.asyncio
    async def test_batch_execution_empty_list(self) -> None:
        """Empty list input returns empty list."""
        module = SimpleModule()
        mock_resources = MagicMock()

        module.bind(resources=mock_resources)

        with patch("inf_engine.execution.executor.run") as mock_run:
            results = await module([])

            assert results == []
            mock_run.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Tracing Context Behavior Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTracingContextBehavior:
    """Tests for __call__ behavior with trace context."""

    def test_trace_context_takes_precedence(self) -> None:
        """Trace context takes precedence over bound resources."""
        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.proxy import Proxy
        from inf_engine.tracing.tracer import Tracer

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        tracer = Tracer()
        with trace_context(tracer):
            result = module("hello")

        # Should return a Proxy, not execute
        assert isinstance(result, Proxy)

    def test_trace_context_takes_precedence_over_execution_settings(self) -> None:
        """Trace context takes precedence over ExecutionSettings."""
        from inf_engine.tracing.context import trace_context
        from inf_engine.tracing.proxy import Proxy
        from inf_engine.tracing.tracer import Tracer

        module = SimpleModule()
        mock_resources = MagicMock()

        tracer = Tracer()
        with ExecutionSettings(resources=mock_resources):
            with trace_context(tracer):
                result = module("hello")

        assert isinstance(result, Proxy)


# ─────────────────────────────────────────────────────────────────────────────
# Module Initialization Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModuleInitialization:
    """Tests for module initialization with binding attributes."""

    def test_new_module_has_no_bound_resources(self) -> None:
        """New module has _bound_resources as None."""
        module = SimpleModule()
        assert module._bound_resources is None

    def test_new_module_has_empty_bound_config(self) -> None:
        """New module has empty _bound_config."""
        module = SimpleModule()
        assert module._bound_config == {}

    def test_nested_module_children_have_no_binding(self) -> None:
        """Child modules don't inherit parent's binding."""
        parent = NestedModule()
        mock_resources = MagicMock()

        parent.bind(resources=mock_resources)

        # Parent is bound
        assert parent._bound_resources is mock_resources
        # Child is not bound
        assert parent.inner._bound_resources is None


# ─────────────────────────────────────────────────────────────────────────────
# LLMInference Binding Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMInferenceBinding:
    """Tests for LLMInference with binding."""

    def test_llm_inference_can_be_bound(self) -> None:
        """LLMInference supports binding."""
        llm = LLMInference(alias="fast", system_prompt="Be helpful.")
        mock_resources = MagicMock()

        result = llm.bind(resources=mock_resources)

        assert result is llm
        assert llm._bound_resources is mock_resources

    @pytest.mark.asyncio
    async def test_llm_inference_executes_when_bound(self) -> None:
        """Bound LLMInference executes through run()."""
        llm = LLMInference(alias="fast")
        mock_resources = MagicMock()

        llm.bind(resources=mock_resources)

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "LLM response"

            result = await llm("Hello!")

            assert result == "LLM response"
            mock_run.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBindingIntegration:
    """Integration tests for binding functionality."""

    @pytest.mark.asyncio
    async def test_full_binding_flow(self) -> None:
        """Test complete binding and execution flow."""
        module = SimpleModule()
        mock_resources = MagicMock()

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "EXECUTED"

            # Bind and execute
            result = await module.bind(resources=mock_resources)("hello")

            assert result == "EXECUTED"

    @pytest.mark.asyncio
    async def test_multiple_modules_different_bindings(self) -> None:
        """Multiple modules can have different bindings."""
        module1 = SimpleModule()
        module2 = SimpleModule()
        resources1 = MagicMock(name="r1")
        resources2 = MagicMock(name="r2")

        module1.bind(resources=resources1, max_concurrent=10)
        module2.bind(resources=resources2, max_concurrent=20)

        assert module1._bound_resources is resources1
        assert module2._bound_resources is resources2
        assert module1._bound_config["max_concurrent"] == 10
        assert module2._bound_config["max_concurrent"] == 20

    @pytest.mark.asyncio
    async def test_execution_settings_shared_across_modules(self) -> None:
        """ExecutionSettings is shared across multiple module calls."""
        module1 = SimpleModule()
        module2 = SimpleModule()
        shared_resources = MagicMock()

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.side_effect = ["RESULT1", "RESULT2"]

            with ExecutionSettings(resources=shared_resources, max_concurrent=30):
                await module1("input1")
                await module2("input2")

            # Both should use same settings
            assert mock_run.call_count == 2
            for call in mock_run.call_args_list:
                assert call.kwargs["resources"] is shared_resources
                assert call.kwargs["max_concurrent"] == 30
