"""Integration tests for streaming execution.

Tests full streaming flows with real module execution patterns,
including cancellation, cleanup, and multi-step pipelines.
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from plait.execution.context import ExecutionSettings
from plait.execution.types import BatchResult
from plait.module import InferenceModule


class ProcessingModule(InferenceModule):
    """Test module that processes text input."""

    def __init__(self, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix
        self.calls: list[str] = []

    def forward(self, text: str) -> str:
        self.calls.append(text)
        return f"{self.prefix}{text.upper()}"


class SlowModule(InferenceModule):
    """Test module with configurable delay per input."""

    def __init__(self) -> None:
        super().__init__()
        self.delays: dict[str, float] = {}

    def forward(self, text: str) -> str:
        return text.upper()


class FailingModule(InferenceModule):
    """Test module that fails on certain inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_on: set[str] = set()

    def forward(self, text: str) -> str:
        if text in self.fail_on:
            raise ValueError(f"Intentional failure on: {text}")
        return text.upper()


@pytest.fixture(autouse=True)
def clean_context() -> None:
    """Ensure execution settings context is clean."""
    from plait.execution.context import _execution_settings, get_execution_settings

    current = get_execution_settings()
    while current is not None:
        if current._token is not None:
            _execution_settings.reset(current._token)
            current._token = None
        current = get_execution_settings()


class TestStreamingFullFlow:
    """Full streaming flow integration tests."""

    @pytest.mark.asyncio
    async def test_streaming_end_to_end(self) -> None:
        """Complete streaming flow with module execution."""
        module = ProcessingModule(prefix="PROC_")
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        async def mock_run(*args: Any, **kwargs: Any) -> str:
            # args[0] is module, args[1] is input
            result = args[0].forward(args[1])
            return result

        with patch("plait.execution.executor.run", side_effect=mock_run):
            async with ExecutionSettings(resources=mock_resources, streaming=True):
                result = await module._execute_bound(["hello", "world", "test"])

                collected: list[BatchResult[Any]] = []
                async for batch_result in result:
                    collected.append(batch_result)

        # All results should be collected
        assert len(collected) == 3

        # All should be successful
        for r in collected:
            assert r.ok is True
            assert r.output is not None

        # Sort by index and verify outputs
        collected.sort(key=lambda x: x.index)
        assert collected[0].output == "PROC_HELLO"
        assert collected[1].output == "PROC_WORLD"
        assert collected[2].output == "PROC_TEST"

    @pytest.mark.asyncio
    async def test_streaming_mixed_success_failure(self) -> None:
        """Streaming with mixed success and failure results."""
        module = FailingModule()
        module.fail_on = {"bad1", "bad2"}
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        async def mock_run(*args: Any, **kwargs: Any) -> str:
            return args[0].forward(args[1])

        with patch("plait.execution.executor.run", side_effect=mock_run):
            async with ExecutionSettings(resources=mock_resources, streaming=True):
                inputs = ["good1", "bad1", "good2", "bad2", "good3"]
                result = await module._execute_bound(inputs)

                collected: list[BatchResult[Any]] = []
                async for batch_result in result:
                    collected.append(batch_result)

        assert len(collected) == 5

        # Sort by index
        collected.sort(key=lambda x: x.index)

        # Check success/failure pattern
        assert collected[0].ok is True  # good1
        assert collected[1].ok is False  # bad1
        assert collected[2].ok is True  # good2
        assert collected[3].ok is False  # bad2
        assert collected[4].ok is True  # good3

        # Verify error messages
        assert "bad1" in str(collected[1].error)
        assert "bad2" in str(collected[3].error)


class TestStreamingCancellation:
    """Tests for cancellation behavior during streaming."""

    @pytest.mark.asyncio
    async def test_cancellation_stops_pending_work(self) -> None:
        """Breaking from stream cancels pending tasks."""
        module = ProcessingModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        started: list[str] = []
        completed: list[str] = []

        async def slow_run(*args: Any, **kwargs: Any) -> str:
            inp = args[1]
            started.append(inp)
            # First item is fast, rest are slow
            delay = 0.01 if inp == "fast" else 0.5
            await asyncio.sleep(delay)
            completed.append(inp)
            return inp.upper()

        with patch("plait.execution.executor.run", side_effect=slow_run):
            async with ExecutionSettings(
                resources=mock_resources, streaming=True, preserve_order=False
            ):
                result = await module._execute_bound(["fast", "slow1", "slow2"])

                # Consume only first result then break
                count = 0
                async for _batch_result in result:
                    count += 1
                    if count >= 1:
                        break

                # Give cleanup time
                await asyncio.sleep(0.15)

        # Only the fast one should have completed
        assert "fast" in completed
        assert "slow1" not in completed
        assert "slow2" not in completed

    @pytest.mark.asyncio
    async def test_cancellation_cleanup_no_leaked_tasks(self) -> None:
        """Cancellation properly cleans up without leaking tasks."""
        module = ProcessingModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        async def very_slow_run(*args: Any, **kwargs: Any) -> str:
            inp = args[1]
            delay = 0.001 if inp == "fast" else 10.0  # 10 second delay for slow
            await asyncio.sleep(delay)
            return inp.upper()

        with patch("plait.execution.executor.run", side_effect=very_slow_run):
            async with ExecutionSettings(
                resources=mock_resources, streaming=True, preserve_order=False
            ):
                result = await module._execute_bound(["fast", "slow1", "slow2"])

                # Break immediately after first
                async for _batch_result in result:
                    break

        # Test should complete quickly - if tasks leaked, it would hang


class TestProgressTracking:
    """Tests for progress tracking integration."""

    @pytest.mark.asyncio
    async def test_progress_tracks_all_completions(self) -> None:
        """Progress callback receives all completion notifications."""
        module = ProcessingModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        progress_updates: list[tuple[int, int]] = []

        def on_progress(done: int, total: int) -> None:
            progress_updates.append((done, total))

        async def mock_run(*args: Any, **kwargs: Any) -> str:
            return args[1].upper()

        with patch("plait.execution.executor.run", side_effect=mock_run):
            async with ExecutionSettings(
                resources=mock_resources,
                streaming=True,
                on_progress=on_progress,
            ):
                result = await module._execute_bound(["a", "b", "c", "d", "e"])

                async for _ in result:
                    pass

        # Should have 5 progress updates
        assert len(progress_updates) == 5

        # All totals should be 5
        for _done, total in progress_updates:
            assert total == 5

        # Done values should span 1-5
        done_values = sorted([d for d, t in progress_updates])
        assert done_values == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_progress_with_failures(self) -> None:
        """Progress callback is called even for failed items."""
        module = FailingModule()
        module.fail_on = {"b", "d"}
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        progress_updates: list[tuple[int, int]] = []

        def on_progress(done: int, total: int) -> None:
            progress_updates.append((done, total))

        async def mock_run(*args: Any, **kwargs: Any) -> str:
            return args[0].forward(args[1])

        with patch("plait.execution.executor.run", side_effect=mock_run):
            async with ExecutionSettings(
                resources=mock_resources,
                streaming=True,
                on_progress=on_progress,
            ):
                result = await module._execute_bound(["a", "b", "c", "d", "e"])

                async for _ in result:
                    pass

        # All 5 items should trigger progress (even failures)
        assert len(progress_updates) == 5


class TestPreserveOrder:
    """Tests for preserve_order behavior."""

    @pytest.mark.asyncio
    async def test_preserve_order_waits_for_slow_items(self) -> None:
        """preserve_order=True waits for items even if later ones finish first."""
        module = ProcessingModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        order_yielded: list[int] = []

        async def delayed_run(*args: Any, **kwargs: Any) -> str:
            inp = args[1]
            # First item is slow, others are fast
            delays = {"first": 0.08, "second": 0.02, "third": 0.01}
            await asyncio.sleep(delays.get(inp, 0))
            return inp.upper()

        with patch("plait.execution.executor.run", side_effect=delayed_run):
            async with ExecutionSettings(
                resources=mock_resources,
                streaming=True,
                preserve_order=True,
            ):
                result = await module._execute_bound(["first", "second", "third"])

                async for batch_result in result:
                    order_yielded.append(batch_result.index)

        # Despite "first" being slowest, preserve_order ensures order
        assert order_yielded == [0, 1, 2]


class TestStreamingResourceIntegration:
    """Tests for streaming with resource settings."""

    @pytest.mark.asyncio
    async def test_streaming_uses_context_resources(self) -> None:
        """Streaming uses resources from ExecutionSettings context."""
        module = ProcessingModule()
        context_resources = MagicMock(name="context_resources")

        received_resources: list[Any] = []

        async def mock_run(*args: Any, **kwargs: Any) -> str:
            received_resources.append(kwargs.get("resources"))
            return args[1].upper()

        with patch("plait.execution.executor.run", side_effect=mock_run):
            async with ExecutionSettings(
                resources=context_resources,
                streaming=True,
            ):
                result = await module._execute_bound(["a", "b"])

                async for _ in result:
                    pass

        # All calls should use context resources
        assert len(received_resources) == 2
        for res in received_resources:
            assert res is context_resources

    @pytest.mark.asyncio
    async def test_streaming_bound_resources_override_context(self) -> None:
        """Bound resources take precedence over context in streaming."""
        module = ProcessingModule()
        bound_resources = MagicMock(name="bound_resources")
        context_resources = MagicMock(name="context_resources")

        module.bind(resources=bound_resources)

        received_resources: list[Any] = []

        async def mock_run(*args: Any, **kwargs: Any) -> str:
            received_resources.append(kwargs.get("resources"))
            return args[1].upper()

        with patch("plait.execution.executor.run", side_effect=mock_run):
            async with ExecutionSettings(
                resources=context_resources,
                streaming=True,
            ):
                result = await module._execute_bound(["a", "b"])

                async for _ in result:
                    pass

        # All calls should use bound resources (not context)
        assert len(received_resources) == 2
        for res in received_resources:
            assert res is bound_resources


class TestStreamingEdgeCases:
    """Edge case tests for streaming execution."""

    @pytest.mark.asyncio
    async def test_streaming_single_item_list(self) -> None:
        """Streaming works with single-item list."""
        module = ProcessingModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        async def mock_run(*args: Any, **kwargs: Any) -> str:
            return args[1].upper()

        with patch("plait.execution.executor.run", side_effect=mock_run):
            async with ExecutionSettings(resources=mock_resources, streaming=True):
                result = await module._execute_bound(["only"])

                collected: list[BatchResult[Any]] = []
                async for batch_result in result:
                    collected.append(batch_result)

        assert len(collected) == 1
        assert collected[0].index == 0
        assert collected[0].input == "only"
        assert collected[0].output == "ONLY"

    @pytest.mark.asyncio
    async def test_streaming_large_batch(self) -> None:
        """Streaming handles large batches."""
        module = ProcessingModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        async def mock_run(*args: Any, **kwargs: Any) -> str:
            return args[1].upper()

        batch_size = 100
        inputs = [f"item_{i}" for i in range(batch_size)]

        with patch("plait.execution.executor.run", side_effect=mock_run):
            async with ExecutionSettings(resources=mock_resources, streaming=True):
                result = await module._execute_bound(inputs)

                collected: list[BatchResult[Any]] = []
                async for batch_result in result:
                    collected.append(batch_result)

        assert len(collected) == batch_size

        # All should be successful
        assert all(r.ok for r in collected)

        # Verify all indices present
        indices = {r.index for r in collected}
        assert indices == set(range(batch_size))

    @pytest.mark.asyncio
    async def test_streaming_empty_list(self) -> None:
        """Streaming with empty list yields nothing."""
        module = ProcessingModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        async with ExecutionSettings(resources=mock_resources, streaming=True):
            result = await module._execute_bound([])

            collected: list[BatchResult[Any]] = []
            async for batch_result in result:
                collected.append(batch_result)

        assert collected == []

    @pytest.mark.asyncio
    async def test_non_streaming_batch_still_works(self) -> None:
        """Non-streaming batch execution still returns list."""
        module = ProcessingModule()
        mock_resources = MagicMock()
        module.bind(resources=mock_resources)

        async def mock_run(*args: Any, **kwargs: Any) -> str:
            return args[1].upper()

        with patch("plait.execution.executor.run", side_effect=mock_run):
            # streaming=False (default)
            async with ExecutionSettings(resources=mock_resources):
                results = await module._execute_bound(["a", "b", "c"])

        # Should be a list, not an async iterator
        assert isinstance(results, list)
        assert len(results) == 3
        assert results == ["A", "B", "C"]
