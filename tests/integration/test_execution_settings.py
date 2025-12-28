"""Integration tests for ExecutionSettings with module execution.

This file contains tests for PR-059: Update InferenceModule for ExecutionSettings.
Tests verify:
- Shared checkpointing across multiple pipelines
- Priority order for configuration (context < bound < kwargs)
- Multiple modules sharing ExecutionSettings
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inf_engine.execution.context import ExecutionSettings, get_execution_settings
from inf_engine.module import InferenceModule

# ─────────────────────────────────────────────────────────────────────────────
# Test Helpers
# ─────────────────────────────────────────────────────────────────────────────


class EchoModule(InferenceModule):
    """Simple test module that echoes input with prefix."""

    def __init__(self, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix

    def forward(self, text: str) -> str:
        return f"{self.prefix}{text}"


class UpperModule(InferenceModule):
    """Test module that uppercases input."""

    def forward(self, text: str) -> str:
        return text.upper()


class ChainModule(InferenceModule):
    """Module that chains two child modules."""

    def __init__(self) -> None:
        super().__init__()
        self.step1 = EchoModule(prefix="[1]")
        self.step2 = EchoModule(prefix="[2]")

    def forward(self, text: str) -> str:
        r1 = self.step1(text)
        r2 = self.step2(r1)
        return r2


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


# ─────────────────────────────────────────────────────────────────────────────
# Shared ExecutionSettings Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSharedExecutionSettings:
    """Tests for multiple modules sharing ExecutionSettings."""

    @pytest.mark.asyncio
    async def test_multiple_modules_share_context_resources(self) -> None:
        """Multiple modules within same context use same resources."""
        module1 = EchoModule(prefix="m1:")
        module2 = UpperModule()
        shared_resources = MagicMock(name="shared_resources")

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.side_effect = ["result1", "result2"]

            with ExecutionSettings(resources=shared_resources):
                await module1("input1")
                await module2("input2")

            # Both should use same resources
            assert mock_run.call_count == 2
            for call in mock_run.call_args_list:
                assert call.kwargs["resources"] is shared_resources

    @pytest.mark.asyncio
    async def test_multiple_modules_share_max_concurrent(self) -> None:
        """Multiple modules within same context use same max_concurrent."""
        module1 = EchoModule(prefix="m1:")
        module2 = UpperModule()
        shared_resources = MagicMock()

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.side_effect = ["result1", "result2"]

            with ExecutionSettings(resources=shared_resources, max_concurrent=42):
                await module1("input1")
                await module2("input2")

            for call in mock_run.call_args_list:
                assert call.kwargs["max_concurrent"] == 42


# ─────────────────────────────────────────────────────────────────────────────
# Priority Order Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigurationPriority:
    """Tests for configuration priority order: context < bound < kwargs."""

    @pytest.mark.asyncio
    async def test_priority_order_all_levels(self) -> None:
        """Test all priority levels: kwargs > bound > context."""
        module = EchoModule()

        context_resources = MagicMock(name="context")
        bound_resources = MagicMock(name="bound")

        module.bind(resources=bound_resources, max_concurrent=50)

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            with ExecutionSettings(resources=context_resources, max_concurrent=100):
                # Override max_concurrent at call time
                await module("test", max_concurrent=25)

            call_kwargs = mock_run.call_args.kwargs
            # Resources: bound > context
            assert call_kwargs["resources"] is bound_resources
            # max_concurrent: kwargs > bound > context
            assert call_kwargs["max_concurrent"] == 25

    @pytest.mark.asyncio
    async def test_context_provides_defaults(self) -> None:
        """Context provides defaults when module has no binding."""
        module = EchoModule()
        context_resources = MagicMock(name="context")

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            with ExecutionSettings(resources=context_resources, max_concurrent=75):
                await module("test")

            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["resources"] is context_resources
            assert call_kwargs["max_concurrent"] == 75

    @pytest.mark.asyncio
    async def test_binding_overrides_context(self) -> None:
        """Bound settings override context settings."""
        module = EchoModule()
        context_resources = MagicMock(name="context")
        bound_resources = MagicMock(name="bound")

        module.bind(resources=bound_resources, max_concurrent=30)

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            with ExecutionSettings(resources=context_resources, max_concurrent=100):
                await module("test")

            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["resources"] is bound_resources
            assert call_kwargs["max_concurrent"] == 30


# ─────────────────────────────────────────────────────────────────────────────
# Nested Context Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNestedContexts:
    """Tests for nested ExecutionSettings contexts."""

    @pytest.mark.asyncio
    async def test_inner_context_overrides_outer(self) -> None:
        """Inner context settings override outer context settings."""
        module = EchoModule()
        outer_resources = MagicMock(name="outer")
        inner_resources = MagicMock(name="inner")

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            with ExecutionSettings(resources=outer_resources, max_concurrent=100):
                with ExecutionSettings(resources=inner_resources, max_concurrent=10):
                    await module("test")

            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["resources"] is inner_resources
            assert call_kwargs["max_concurrent"] == 10

    @pytest.mark.asyncio
    async def test_outer_context_restored_after_inner(self) -> None:
        """Outer context is restored when inner context exits."""
        module = EchoModule()
        outer_resources = MagicMock(name="outer")
        inner_resources = MagicMock(name="inner")

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.side_effect = ["inner_result", "outer_result"]

            with ExecutionSettings(resources=outer_resources, max_concurrent=100):
                with ExecutionSettings(resources=inner_resources, max_concurrent=10):
                    await module("inner_call")

                await module("outer_call")

            # First call used inner, second call used outer
            assert mock_run.call_count == 2
            assert mock_run.call_args_list[0].kwargs["resources"] is inner_resources
            assert mock_run.call_args_list[1].kwargs["resources"] is outer_resources


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Directory Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckpointDirectory:
    """Tests for checkpoint_dir handling in ExecutionSettings."""

    @pytest.mark.asyncio
    async def test_checkpoint_dir_passed_to_run(self, tmp_path: Path) -> None:
        """checkpoint_dir from context is passed to run()."""
        module = EchoModule()
        mock_resources = MagicMock()
        checkpoint_dir = tmp_path / "checkpoints"

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            with ExecutionSettings(
                resources=mock_resources, checkpoint_dir=checkpoint_dir
            ):
                await module("test")

            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["checkpoint_dir"] == checkpoint_dir

    @pytest.mark.asyncio
    async def test_bound_checkpoint_dir_overrides_context(self, tmp_path: Path) -> None:
        """Bound checkpoint_dir overrides context checkpoint_dir."""
        module = EchoModule()
        mock_resources = MagicMock()
        context_dir = tmp_path / "context_checkpoints"
        bound_dir = tmp_path / "bound_checkpoints"

        module.bind(resources=mock_resources, checkpoint_dir=bound_dir)

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            with ExecutionSettings(
                resources=mock_resources, checkpoint_dir=context_dir
            ):
                await module("test")

            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["checkpoint_dir"] == bound_dir


# ─────────────────────────────────────────────────────────────────────────────
# Async Context Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAsyncContext:
    """Tests for async context manager behavior."""

    @pytest.mark.asyncio
    async def test_async_context_with_module_execution(self) -> None:
        """ExecutionSettings works as async context manager with module execution."""
        module = EchoModule()
        mock_resources = MagicMock()

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "async_result"

            async with ExecutionSettings(resources=mock_resources):
                result = await module("test")

            assert result == "async_result"

    @pytest.mark.asyncio
    async def test_async_context_cleared_after_exit(self) -> None:
        """Context is cleared after async context manager exit."""
        module = EchoModule()
        mock_resources = MagicMock()

        async with ExecutionSettings(resources=mock_resources):
            assert get_execution_settings() is not None

        # Context should be cleared
        assert get_execution_settings() is None

        # Module should call forward() directly now (no resources)
        result = module("test")
        assert result == "test"  # EchoModule with no prefix


# ─────────────────────────────────────────────────────────────────────────────
# Mixed Binding and Context Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMixedBindingAndContext:
    """Tests for modules with mixed bound and context settings."""

    @pytest.mark.asyncio
    async def test_bound_module_with_context(self) -> None:
        """Bound module can run within ExecutionSettings context."""
        module = EchoModule()
        bound_resources = MagicMock(name="bound")
        context_resources = MagicMock(name="context")

        module.bind(resources=bound_resources)

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.return_value = "result"

            with ExecutionSettings(resources=context_resources):
                await module("test")

            # Should use bound resources, not context resources
            assert mock_run.call_args.kwargs["resources"] is bound_resources

    @pytest.mark.asyncio
    async def test_unbound_and_bound_modules_in_context(self) -> None:
        """Unbound and bound modules can coexist in same context."""
        unbound = EchoModule(prefix="unbound:")
        bound = EchoModule(prefix="bound:")

        context_resources = MagicMock(name="context")
        bound_resources = MagicMock(name="bound")

        bound.bind(resources=bound_resources)

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.side_effect = ["unbound_result", "bound_result"]

            with ExecutionSettings(resources=context_resources):
                await unbound("input1")
                await bound("input2")

            # First call used context resources
            assert mock_run.call_args_list[0].kwargs["resources"] is context_resources
            # Second call used bound resources
            assert mock_run.call_args_list[1].kwargs["resources"] is bound_resources


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_context_with_no_resources_uses_forward(self) -> None:
        """ExecutionSettings without resources still allows module calls via forward()."""
        module = EchoModule(prefix="test:")

        with ExecutionSettings(max_concurrent=50):
            # No resources, so should call forward() directly
            result = module("input")

        assert result == "test:input"

    def test_module_without_binding_outside_context_uses_forward(self) -> None:
        """Module without binding outside context uses forward()."""
        module = EchoModule(prefix="direct:")

        result = module("input")

        assert result == "direct:input"

    @pytest.mark.asyncio
    async def test_rebinding_within_context(self) -> None:
        """Module can be rebound within ExecutionSettings context."""
        module = EchoModule()
        resources1 = MagicMock(name="r1")
        resources2 = MagicMock(name="r2")

        with patch("inf_engine.execution.executor.run") as mock_run:
            mock_run.side_effect = ["result1", "result2"]

            with ExecutionSettings(resources=resources1):
                # Initially uses context
                await module("test1")

                # Now bind to different resources
                module.bind(resources=resources2)
                await module("test2")

            # First call used context resources
            assert mock_run.call_args_list[0].kwargs["resources"] is resources1
            # Second call used bound resources
            assert mock_run.call_args_list[1].kwargs["resources"] is resources2
