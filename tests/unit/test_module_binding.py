"""Unit tests for Module binding and ExecutionSettings integration.

This file contains tests for:
- PR-045: bind() method for direct module execution
- PR-045: __call__ behavior for bound modules
- PR-046: Concurrent batch execution and run_sync()
- PR-059: ExecutionSettings context integration
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from plait.execution.context import ExecutionSettings, get_execution_settings
from plait.module import LLMInference, Module


@pytest.fixture(autouse=True)
def clean_context() -> None:
    """Ensure execution settings context is clean before each test."""
    from plait.execution.context import _execution_settings

    current = get_execution_settings()
    while current is not None:
        if current._token is not None:
            _execution_settings.reset(current._token)
            current._token = None
        current = get_execution_settings()


class SimpleModule(Module):
    """Simple test module that returns transformed input."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, text: str) -> str:
        return text.upper()


class NestedModule(Module):
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
    """Tests for Module.bind() method."""

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
        with patch("plait.execution.executor.run") as mock_run:
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

        with patch("plait.execution.executor.run") as mock_run:
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

        with patch("plait.execution.executor.run") as mock_run:
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

        with patch("plait.execution.executor.run") as mock_run:
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

        with patch("plait.execution.executor.run") as mock_run:
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

        with patch("plait.execution.executor.run") as mock_run:
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

        with patch("plait.execution.executor.run") as mock_run:
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

        with patch("plait.execution.executor.run") as mock_run:
            # Return different results for each call
            mock_run.side_effect = ["RESULT1", "RESULT2", "RESULT3"]

            results = await module(["input1", "input2", "input3"])

            # Should call run() once for each input
            assert mock_run.call_count == 3
            # Note: asyncio.gather returns results in order
            assert list(results) == ["RESULT1", "RESULT2", "RESULT3"]

    @pytest.mark.asyncio
    async def test_batch_execution_passes_same_resources(self) -> None:
        """Batch execution passes same resources to each run()."""
        module = SimpleModule()
        mock_resources = MagicMock()

        module.bind(resources=mock_resources)

        with patch("plait.execution.executor.run") as mock_run:
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

        with patch("plait.execution.executor.run") as mock_run:
            results = await module([])

            assert results == []
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_execution_runs_concurrently(self) -> None:
        """Batch execution runs tasks concurrently, not sequentially."""
        import asyncio
        import time

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        # Track timing to prove concurrency
        call_times: list[float] = []

        async def slow_run(*args: Any, **kwargs: Any) -> str:
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)  # 50ms simulated delay
            return "result"

        with patch("plait.execution.executor.run", side_effect=slow_run):
            start = time.monotonic()
            results = await module(["a", "b", "c"])
            elapsed = time.monotonic() - start

        # All three inputs should return results
        assert len(results) == 3

        # If sequential, would take 3 * 0.05 = 0.15s
        # If concurrent, should take ~0.05s (plus overhead)
        # Use a reasonable margin for test reliability
        assert elapsed < 0.12, f"Took {elapsed:.3f}s - batch not running concurrently"

        # All three calls should start nearly simultaneously
        if len(call_times) == 3:
            time_spread = max(call_times) - min(call_times)
            assert time_spread < 0.03, "Tasks did not start concurrently"


# ─────────────────────────────────────────────────────────────────────────────
# Tracing Context Behavior Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTracingContextBehavior:
    """Tests for __call__ behavior with trace context."""

    def test_trace_context_takes_precedence(self) -> None:
        """Trace context takes precedence over bound resources."""
        from plait.tracing.context import trace_context
        from plait.tracing.tracer import Tracer
        from plait.values import Value

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        tracer = Tracer()
        with trace_context(tracer):
            result = module("hello")

        # Should return a Value, not execute
        assert isinstance(result, Value)

    def test_trace_context_takes_precedence_over_execution_settings(self) -> None:
        """Trace context takes precedence over ExecutionSettings."""
        from plait.tracing.context import trace_context
        from plait.tracing.tracer import Tracer
        from plait.values import Value

        module = SimpleModule()
        mock_resources = MagicMock()

        tracer = Tracer()
        with ExecutionSettings(resources=mock_resources):
            with trace_context(tracer):
                result = module("hello")

        assert isinstance(result, Value)


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

        with patch("plait.execution.executor.run") as mock_run:
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

        with patch("plait.execution.executor.run") as mock_run:
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

        with patch("plait.execution.executor.run") as mock_run:
            mock_run.side_effect = ["RESULT1", "RESULT2"]

            with ExecutionSettings(resources=shared_resources, max_concurrent=30):
                await module1("input1")
                await module2("input2")

            # Both should use same settings
            assert mock_run.call_count == 2
            for call in mock_run.call_args_list:
                assert call.kwargs["resources"] is shared_resources
                assert call.kwargs["max_concurrent"] == 30


# ─────────────────────────────────────────────────────────────────────────────
# run_sync() Method Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunSync:
    """Tests for run_sync() synchronous execution method."""

    def test_run_sync_single_input(self) -> None:
        """run_sync() executes and returns single result."""
        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        with patch("plait.execution.executor.run") as mock_run:

            async def async_return(*args: Any, **kwargs: Any) -> str:
                return "SYNC_RESULT"

            mock_run.side_effect = async_return

            result = module.run_sync("hello")

            assert result == "SYNC_RESULT"
            mock_run.assert_called_once()

    def test_run_sync_batch_input(self) -> None:
        """run_sync() with list input returns list of results."""
        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        call_count = 0

        async def async_return(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            return f"RESULT_{call_count}"

        with patch("plait.execution.executor.run", side_effect=async_return):
            results = module.run_sync(["a", "b", "c"])

            assert list(results) == ["RESULT_1", "RESULT_2", "RESULT_3"]

    def test_run_sync_requires_resources(self) -> None:
        """run_sync() raises if no resources available."""
        module = SimpleModule()
        # Not bound, no ExecutionSettings context

        with pytest.raises(RuntimeError, match="requires bound resources"):
            module.run_sync("hello")

    def test_run_sync_with_execution_settings_context(self) -> None:
        """run_sync() works with ExecutionSettings context."""
        module = SimpleModule()
        mock_resources = MagicMock()

        async def async_return(*args: Any, **kwargs: Any) -> str:
            return "CONTEXT_RESULT"

        with patch("plait.execution.executor.run", side_effect=async_return):
            with ExecutionSettings(resources=mock_resources):
                result = module.run_sync("hello")

            assert result == "CONTEXT_RESULT"

    @pytest.mark.asyncio
    async def test_run_sync_raises_in_async_context(self) -> None:
        """run_sync() raises if called from async context."""
        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        with pytest.raises(RuntimeError, match="cannot be called from within an async"):
            module.run_sync("hello")

    def test_run_sync_passes_kwargs(self) -> None:
        """run_sync() passes execution kwargs correctly."""
        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        async def async_return(*args: Any, **kwargs: Any) -> str:
            return "RESULT"

        with patch("plait.execution.executor.run", side_effect=async_return) as m:
            module.run_sync("hello", max_concurrent=5)

            call_kwargs = m.call_args.kwargs
            assert call_kwargs["max_concurrent"] == 5

    def test_run_sync_empty_batch(self) -> None:
        """run_sync() with empty list returns empty list."""
        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        with patch("plait.execution.executor.run") as mock_run:
            result = module.run_sync([])

            assert result == []
            mock_run.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Execution Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStreamingExecution:
    """Tests for streaming batch execution."""

    @pytest.mark.asyncio
    async def test_streaming_returns_async_iterator(self) -> None:
        """Streaming mode returns an async iterator."""
        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        async def async_return(*args: Any, **kwargs: Any) -> str:
            return f"RESULT_{args[1]}"

        with patch("plait.execution.executor.run", side_effect=async_return):
            with ExecutionSettings(resources=mock_resources, streaming=True):
                result = await module._execute_bound(["a", "b", "c"])

                # Result should be an async iterator
                from collections.abc import AsyncIterator

                assert isinstance(result, AsyncIterator)

    @pytest.mark.asyncio
    async def test_streaming_yields_batch_results(self) -> None:
        """Streaming mode yields BatchResult objects."""
        from plait.execution.types import BatchResult

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        async def async_return(*args: Any, **kwargs: Any) -> str:
            return f"RESULT_{args[1]}"

        with patch("plait.execution.executor.run", side_effect=async_return):
            with ExecutionSettings(resources=mock_resources, streaming=True):
                result = await module._execute_bound(["a", "b", "c"])

                results = []
                async for batch_result in result:
                    results.append(batch_result)

                assert len(results) == 3
                for r in results:
                    assert isinstance(r, BatchResult)
                    assert r.ok is True

    @pytest.mark.asyncio
    async def test_streaming_results_have_correct_fields(self) -> None:
        """Streaming BatchResults have correct index, input, output, error."""

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        async def async_return(*args: Any, **kwargs: Any) -> str:
            inp = args[1]
            return inp.upper()

        with patch("plait.execution.executor.run", side_effect=async_return):
            with ExecutionSettings(resources=mock_resources, streaming=True):
                result = await module._execute_bound(["hello", "world"])

                results = []
                async for batch_result in result:
                    results.append(batch_result)

                # Sort by index to check correctness regardless of order
                results.sort(key=lambda r: r.index)

                assert results[0].index == 0
                assert results[0].input == "hello"
                assert results[0].output == "HELLO"
                assert results[0].error is None

                assert results[1].index == 1
                assert results[1].input == "world"
                assert results[1].output == "WORLD"
                assert results[1].error is None

    @pytest.mark.asyncio
    async def test_streaming_handles_errors(self) -> None:
        """Streaming captures errors in BatchResult."""

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        call_count = 0

        async def async_return(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Test error")
            return f"RESULT_{call_count}"

        with patch("plait.execution.executor.run", side_effect=async_return):
            with ExecutionSettings(resources=mock_resources, streaming=True):
                result = await module._execute_bound(["a", "b", "c"])

                results = []
                async for batch_result in result:
                    results.append(batch_result)

                # Sort by index for predictable checking
                results.sort(key=lambda r: r.index)

                # Result 0 should be success
                assert results[0].ok is True

                # Result 1 should be failure
                assert results[1].ok is False
                assert results[1].error is not None
                assert isinstance(results[1].error, ValueError)
                assert "Test error" in str(results[1].error)

                # Result 2 should be success
                assert results[2].ok is True

    @pytest.mark.asyncio
    async def test_streaming_preserve_order_true(self) -> None:
        """preserve_order=True yields results in input order."""
        import asyncio

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        # Make later inputs complete faster to test ordering
        async def async_return(*args: Any, **kwargs: Any) -> str:
            inp = args[1]
            delays = {"a": 0.05, "b": 0.02, "c": 0.01}
            await asyncio.sleep(delays.get(inp, 0))
            return inp.upper()

        with patch("plait.execution.executor.run", side_effect=async_return):
            with ExecutionSettings(
                resources=mock_resources, streaming=True, preserve_order=True
            ):
                result = await module._execute_bound(["a", "b", "c"])

                indices = []
                async for batch_result in result:
                    indices.append(batch_result.index)

                # Should be in input order: 0, 1, 2
                assert indices == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_streaming_preserve_order_false_fastest_first(self) -> None:
        """preserve_order=False may yield fastest-completing results first."""
        import asyncio

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        # Make later inputs complete faster
        async def async_return(*args: Any, **kwargs: Any) -> str:
            inp = args[1]
            delays = {"a": 0.08, "b": 0.05, "c": 0.01}
            await asyncio.sleep(delays.get(inp, 0))
            return inp.upper()

        with patch("plait.execution.executor.run", side_effect=async_return):
            with ExecutionSettings(
                resources=mock_resources, streaming=True, preserve_order=False
            ):
                result = await module._execute_bound(["a", "b", "c"])

                indices = []
                async for batch_result in result:
                    indices.append(batch_result.index)

                # "c" (index 2) should complete first, then "b" (index 1), then "a" (index 0)
                # This is probabilistic but with these delays should be consistent
                assert indices == [2, 1, 0]

    @pytest.mark.asyncio
    async def test_streaming_empty_list(self) -> None:
        """Streaming with empty list yields no results."""
        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        with ExecutionSettings(resources=mock_resources, streaming=True):
            result = await module._execute_bound([])

            results = []
            async for batch_result in result:
                results.append(batch_result)

            assert results == []

    @pytest.mark.asyncio
    async def test_streaming_cancellation_on_break(self) -> None:
        """Breaking from streaming loop cancels pending tasks."""
        import asyncio

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        completed = []

        async def async_return(*args: Any, **kwargs: Any) -> str:
            inp = args[1]
            # First completes quickly, others take longer
            delays = {"a": 0.01, "b": 0.5, "c": 0.5}
            await asyncio.sleep(delays.get(inp, 0))
            completed.append(inp)
            return inp.upper()

        with patch("plait.execution.executor.run", side_effect=async_return):
            with ExecutionSettings(
                resources=mock_resources, streaming=True, preserve_order=False
            ):
                result = await module._execute_bound(["a", "b", "c"])

                # Only consume first result, then break
                async for _batch_result in result:
                    break  # Stop after first result

                # Give some time for cancellation to propagate
                await asyncio.sleep(0.1)

                # Only the first (fastest) should have completed
                assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_on_progress_callback_streaming(self) -> None:
        """on_progress callback is called during streaming."""
        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        progress_calls: list[tuple[int, int]] = []

        def on_progress(done: int, total: int) -> None:
            progress_calls.append((done, total))

        async def async_return(*args: Any, **kwargs: Any) -> str:
            return "RESULT"

        with patch("plait.execution.executor.run", side_effect=async_return):
            with ExecutionSettings(
                resources=mock_resources, streaming=True, on_progress=on_progress
            ):
                result = await module._execute_bound(["a", "b", "c"])

                async for _ in result:
                    pass

        # Progress should be called 3 times
        assert len(progress_calls) == 3
        # All calls should have total=3
        for _done, total in progress_calls:
            assert total == 3
        # Done counts should range from 1 to 3 (order may vary)
        done_values = [d for d, t in progress_calls]
        assert sorted(done_values) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_on_progress_callback_non_streaming(self) -> None:
        """on_progress callback is called in non-streaming batch mode."""
        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        progress_calls: list[tuple[int, int]] = []

        def on_progress(done: int, total: int) -> None:
            progress_calls.append((done, total))

        async def async_return(*args: Any, **kwargs: Any) -> str:
            return "RESULT"

        with patch("plait.execution.executor.run", side_effect=async_return):
            with ExecutionSettings(
                resources=mock_resources, streaming=False, on_progress=on_progress
            ):
                results = await module._execute_bound(["a", "b", "c"])

        # Non-streaming returns list
        assert isinstance(results, list)
        assert len(results) == 3

        # Progress should be called 3 times
        assert len(progress_calls) == 3
        for _done, total in progress_calls:
            assert total == 3


class TestStreamBatchMethod:
    """Tests for _stream_batch method directly."""

    @pytest.mark.asyncio
    async def test_stream_batch_yields_all_inputs(self) -> None:
        """_stream_batch yields a result for each input."""
        from plait.execution.types import BatchResult

        module = SimpleModule()

        async def async_return(*args: Any, **kwargs: Any) -> str:
            return f"RESULT_{args[1]}"

        with patch("plait.execution.executor.run", side_effect=async_return):
            results = []
            async for r in module._stream_batch(
                inputs=["x", "y", "z"],
                extra_args=(),
                resources=MagicMock(),
                forward_kwargs={},
                effective_config={},
            ):
                results.append(r)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, BatchResult)

    @pytest.mark.asyncio
    async def test_stream_batch_with_extra_args(self) -> None:
        """_stream_batch passes extra_args correctly."""

        module = SimpleModule()

        received_args: list[tuple[Any, ...]] = []

        async def async_return(*args: Any, **kwargs: Any) -> str:
            received_args.append(args)
            return "RESULT"

        with patch("plait.execution.executor.run", side_effect=async_return):
            results = []
            async for r in module._stream_batch(
                inputs=["a"],
                extra_args=("extra1", "extra2"),
                resources=MagicMock(),
                forward_kwargs={},
                effective_config={},
            ):
                results.append(r)

        # Check that extra_args were passed
        assert len(received_args) == 1
        # args[0] is module, args[1] is input, args[2:] are extra_args
        assert received_args[0][2] == "extra1"
        assert received_args[0][3] == "extra2"


# ─────────────────────────────────────────────────────────────────────────────
# Training Mode Tests (PR-068b)
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainingModeProperty:
    """Tests for training mode property and control methods."""

    def test_training_property_default_false(self) -> None:
        """New modules have training=False by default."""
        module = SimpleModule()
        assert module.training is False

    def test_train_sets_training_true(self) -> None:
        """train() sets training to True."""
        module = SimpleModule()
        module.train()
        assert module.training is True

    def test_train_with_false_sets_training_false(self) -> None:
        """train(False) sets training to False."""
        module = SimpleModule()
        module.train()  # First enable
        module.train(False)  # Then disable
        assert module.training is False

    def test_eval_sets_training_false(self) -> None:
        """eval() sets training to False."""
        module = SimpleModule()
        module.train()  # First enable
        module.eval()
        assert module.training is False

    def test_train_returns_self(self) -> None:
        """train() returns self for method chaining."""
        module = SimpleModule()
        result = module.train()
        assert result is module

    def test_eval_returns_self(self) -> None:
        """eval() returns self for method chaining."""
        module = SimpleModule()
        result = module.eval()
        assert result is module

    def test_train_propagates_to_children(self) -> None:
        """train() propagates to child modules."""
        parent = NestedModule()
        parent.train()
        assert parent.training is True
        assert parent.inner.training is True

    def test_eval_propagates_to_children(self) -> None:
        """eval() propagates to child modules."""
        parent = NestedModule()
        parent.train()  # First enable
        parent.eval()  # Then disable
        assert parent.training is False
        assert parent.inner.training is False


class TestTrainingModeExecution:
    """Tests for training mode affecting execution output."""

    @pytest.mark.asyncio
    async def test_training_mode_returns_traced_output(self) -> None:
        """Training mode returns TracedOutput wrapper."""
        from plait.optimization.record import ForwardRecord, TracedOutput

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)
        module.train()

        # Create a mock ForwardRecord
        mock_record = MagicMock(spec=ForwardRecord)

        async def async_run_with_record(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("record"):
                return ("RESULT", mock_record)
            return "RESULT"

        with patch("plait.execution.executor.run", side_effect=async_run_with_record):
            result = await module("hello")

        # Should be TracedOutput
        assert isinstance(result, TracedOutput)
        assert result.value == "RESULT"
        assert result._record is mock_record

    @pytest.mark.asyncio
    async def test_eval_mode_returns_raw_value(self) -> None:
        """Eval mode returns raw value, not TracedOutput."""
        from plait.optimization.record import TracedOutput

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)
        module.train()  # First enable
        module.eval()  # Then disable

        async def async_run(*args: Any, **kwargs: Any) -> str:
            return "RESULT"

        with patch("plait.execution.executor.run", side_effect=async_run):
            result = await module("hello")

        # Should be raw value, not TracedOutput
        assert not isinstance(result, TracedOutput)
        assert result == "RESULT"

    @pytest.mark.asyncio
    async def test_training_mode_passes_record_true(self) -> None:
        """Training mode passes record=True to run()."""
        from plait.optimization.record import ForwardRecord

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)
        module.train()

        mock_record = MagicMock(spec=ForwardRecord)
        calls_kwargs: list[dict[str, Any]] = []

        async def async_run(*args: Any, **kwargs: Any) -> Any:
            calls_kwargs.append(kwargs)
            if kwargs.get("record"):
                return ("RESULT", mock_record)
            return "RESULT"

        with patch("plait.execution.executor.run", side_effect=async_run):
            await module("hello")

        # Should have passed record=True
        assert len(calls_kwargs) == 1
        assert calls_kwargs[0]["record"] is True

    @pytest.mark.asyncio
    async def test_training_mode_batch_returns_traced_outputs(self) -> None:
        """Training mode with batch input returns list of TracedOutput."""
        from plait.optimization.record import ForwardRecord, TracedOutput

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)
        module.train()

        call_count = 0

        async def async_run(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if kwargs.get("record"):
                mock_record = MagicMock(spec=ForwardRecord)
                return (f"RESULT_{call_count}", mock_record)
            return f"RESULT_{call_count}"

        with patch("plait.execution.executor.run", side_effect=async_run):
            results = await module(["a", "b", "c"])

        # Should be a list of TracedOutput
        assert len(results) == 3
        for i, result in enumerate(results, 1):
            assert isinstance(result, TracedOutput)
            assert result.value == f"RESULT_{i}"

    @pytest.mark.asyncio
    async def test_training_mode_streaming_returns_traced_outputs(self) -> None:
        """Training mode with streaming returns TracedOutput in BatchResults."""
        from plait.execution.types import BatchResult
        from plait.optimization.record import ForwardRecord, TracedOutput

        module = SimpleModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)
        module.train()

        call_count = 0

        async def async_run(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if kwargs.get("record"):
                mock_record = MagicMock(spec=ForwardRecord)
                return (f"RESULT_{call_count}", mock_record)
            return f"RESULT_{call_count}"

        with patch("plait.execution.executor.run", side_effect=async_run):
            with ExecutionSettings(resources=mock_resources, streaming=True):
                result = await module._execute_bound(["a", "b", "c"])

                results = []
                async for batch_result in result:
                    results.append(batch_result)

        # Should have 3 results, each with TracedOutput
        assert len(results) == 3
        for r in results:
            assert isinstance(r, BatchResult)
            assert isinstance(r.output, TracedOutput)


class TestTrainingModeMethodChaining:
    """Tests for method chaining with training mode."""

    def test_train_bind_chain(self) -> None:
        """Can chain train() with bind()."""
        mock_resources = MagicMock()
        module = SimpleModule().train().bind(resources=mock_resources)

        assert module.training is True
        assert module._bound_resources is mock_resources

    def test_bind_train_chain(self) -> None:
        """Can chain bind() with train()."""
        mock_resources = MagicMock()
        module = SimpleModule().bind(resources=mock_resources).train()

        assert module.training is True
        assert module._bound_resources is mock_resources

    @pytest.mark.asyncio
    async def test_fluent_training_workflow(self) -> None:
        """Test complete fluent API workflow."""
        from plait.optimization.record import ForwardRecord, TracedOutput

        mock_resources = MagicMock()
        mock_record = MagicMock(spec=ForwardRecord)

        async def async_run(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("record"):
                return ("TRAINED_RESULT", mock_record)
            return "EVAL_RESULT"

        with patch("plait.execution.executor.run", side_effect=async_run):
            # Fluent API: create, bind, train, execute
            module = SimpleModule().bind(resources=mock_resources).train()
            result = await module("input")

        assert isinstance(result, TracedOutput)
        assert result.value == "TRAINED_RESULT"
