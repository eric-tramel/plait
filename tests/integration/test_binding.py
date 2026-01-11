"""Integration tests for module binding and batch execution.

This file contains integration tests for:
- PR-046: Concurrent batch execution with real execution flow
"""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from plait.execution.context import ExecutionSettings
from plait.module import InferenceModule, LLMInference


class SlowModule(InferenceModule):
    """Test module that simulates slow processing."""

    def __init__(self, delay: float = 0.05) -> None:
        super().__init__()
        self.delay = delay
        self.call_times: list[float] = []

    def forward(self, text: str) -> str:
        return text.upper()


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent Batch Execution Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchExecutionConcurrency:
    """Integration tests for concurrent batch execution."""

    @pytest.mark.asyncio
    async def test_batch_processing_runs_concurrently(self) -> None:
        """Batch processing runs all items concurrently, not sequentially."""
        module = SlowModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        # Track when each call starts to prove concurrency
        call_start_times: list[float] = []
        delay_per_call = 0.05  # 50ms

        async def slow_run(*args: Any, **kwargs: Any) -> str:
            call_start_times.append(time.monotonic())
            await asyncio.sleep(delay_per_call)
            # Return the input uppercased
            return str(args[1]).upper() if len(args) > 1 else "RESULT"

        with patch("plait.execution.executor.run", side_effect=slow_run):
            start = time.monotonic()
            results = await module(["a", "b", "c", "d", "e"])
            total_time = time.monotonic() - start

        # Verify we got all results
        assert len(results) == 5

        # If sequential: 5 * 0.05 = 0.25 seconds
        # If concurrent: ~0.05 seconds (plus overhead)
        # We allow a reasonable margin for test stability
        sequential_time = 5 * delay_per_call
        assert total_time < sequential_time * 0.7, (
            f"Batch took {total_time:.3f}s, expected < {sequential_time * 0.7:.3f}s. "
            "Batch execution is not running concurrently."
        )

        # All calls should start within a small window (proving concurrency)
        if len(call_start_times) == 5:
            spread = max(call_start_times) - min(call_start_times)
            # All should start within 30ms of each other
            assert spread < 0.03, (
                f"Call start times spread: {spread:.3f}s. "
                "Tasks are not starting concurrently."
            )

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self) -> None:
        """Batch execution returns results in the same order as inputs."""
        module = SlowModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        # Different delays to ensure results could complete out of order
        delays = [0.04, 0.02, 0.05, 0.01, 0.03]
        call_index = 0

        async def varied_delay_run(*args: Any, **kwargs: Any) -> str:
            nonlocal call_index
            idx = call_index
            call_index += 1
            await asyncio.sleep(delays[idx % len(delays)])
            return f"RESULT_{idx}"

        with patch("plait.execution.executor.run", side_effect=varied_delay_run):
            results = await module(["a", "b", "c", "d", "e"])

        # Despite different completion times, order should be preserved
        # Note: asyncio.gather preserves order
        expected = ["RESULT_0", "RESULT_1", "RESULT_2", "RESULT_3", "RESULT_4"]
        assert list(results) == expected

    @pytest.mark.asyncio
    async def test_batch_with_large_input_list(self) -> None:
        """Batch execution handles larger input lists correctly."""
        module = SlowModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        num_inputs = 20

        async def quick_run(*args: Any, **kwargs: Any) -> str:
            return f"RESULT_{args[1]}" if len(args) > 1 else "RESULT"

        with patch("plait.execution.executor.run", side_effect=quick_run):
            inputs = [f"input_{i}" for i in range(num_inputs)]
            results = await module(inputs)

        assert len(results) == num_inputs
        for i, result in enumerate(results):
            assert result == f"RESULT_input_{i}"


# ─────────────────────────────────────────────────────────────────────────────
# run_sync() Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunSyncIntegration:
    """Integration tests for run_sync() method."""

    def test_run_sync_basic_flow(self) -> None:
        """run_sync() completes a basic execution flow."""
        module = SlowModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        async def simple_run(*args: Any, **kwargs: Any) -> str:
            return "SYNC_EXECUTED"

        with patch("plait.execution.executor.run", side_effect=simple_run):
            result = module.run_sync("hello")

        assert result == "SYNC_EXECUTED"

    def test_run_sync_batch_flow(self) -> None:
        """run_sync() handles batch inputs correctly."""
        module = SlowModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        call_count = 0

        async def counting_run(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            return f"RESULT_{call_count}"

        with patch("plait.execution.executor.run", side_effect=counting_run):
            results = module.run_sync(["a", "b", "c"])

        # All items processed
        assert len(results) == 3
        # Results are returned
        assert all(r.startswith("RESULT_") for r in results)

    def test_run_sync_with_context(self) -> None:
        """run_sync() works within ExecutionSettings context."""
        module = SlowModule()
        mock_resources = MagicMock()

        async def context_run(*args: Any, **kwargs: Any) -> str:
            return "CONTEXT_EXECUTED"

        with patch("plait.execution.executor.run", side_effect=context_run):
            with ExecutionSettings(resources=mock_resources):
                result = module.run_sync("hello")

        assert result == "CONTEXT_EXECUTED"


# ─────────────────────────────────────────────────────────────────────────────
# Bound Module Execution Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBoundModuleExecution:
    """Integration tests for bound module execution."""

    @pytest.mark.asyncio
    async def test_bound_llm_inference_executes(self) -> None:
        """Bound LLMInference module executes through the execution pipeline."""
        llm = LLMInference(alias="test", system_prompt="Be helpful.")
        mock_resources = MagicMock()
        llm.bind(resources=mock_resources)

        async def mock_run(*args: Any, **kwargs: Any) -> str:
            return "LLM response"

        with patch("plait.execution.executor.run", side_effect=mock_run):
            result = await llm("Hello!")

        assert result == "LLM response"

    @pytest.mark.asyncio
    async def test_bound_nested_module_executes(self) -> None:
        """Bound module with nested children executes correctly."""

        class NestedPipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.step1 = LLMInference(alias="fast")
                self.step2 = LLMInference(alias="smart")

            def forward(self, text: str) -> str:
                intermediate = self.step1(text)
                return self.step2(intermediate)

        pipeline = NestedPipeline()
        mock_resources = MagicMock()
        pipeline.bind(resources=mock_resources)

        async def mock_run(*args: Any, **kwargs: Any) -> str:
            return "PIPELINE_OUTPUT"

        with patch("plait.execution.executor.run", side_effect=mock_run):
            result = await pipeline("input")

        assert result == "PIPELINE_OUTPUT"

    @pytest.mark.asyncio
    async def test_multiple_bound_modules_independent(self) -> None:
        """Multiple independently bound modules work correctly."""
        module1 = SlowModule()
        module2 = SlowModule()

        resources1 = MagicMock(name="resources1")
        resources2 = MagicMock(name="resources2")

        module1.bind(resources=resources1)
        module2.bind(resources=resources2)

        async def run_check(*args: Any, **kwargs: Any) -> str:
            return "RESULT"

        with patch("plait.execution.executor.run", side_effect=run_check) as m:
            await module1("input1")
            await module2("input2")

        # Both should have been called with their respective resources
        assert m.call_count == 2
        call_resources = [c.kwargs["resources"] for c in m.call_args_list]
        assert resources1 in call_resources
        assert resources2 in call_resources
