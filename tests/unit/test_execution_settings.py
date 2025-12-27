"""Unit tests for the ExecutionSettings context manager."""

from pathlib import Path

import pytest

from inf_engine.execution.context import ExecutionSettings, get_execution_settings


@pytest.fixture(autouse=True)
def clean_context() -> None:
    """Ensure execution settings context is clean before each test.

    This fixture runs automatically before each test to prevent
    context leakage between tests.
    """
    from inf_engine.execution.context import _execution_settings

    # Reset context to None if any stale settings exist
    current = get_execution_settings()
    while current is not None:
        if current._token is not None:
            _execution_settings.reset(current._token)
            current._token = None
        current = get_execution_settings()


class TestExecutionSettingsCreation:
    """Tests for ExecutionSettings instantiation."""

    def test_default_values(self) -> None:
        """ExecutionSettings has sensible defaults."""
        settings = ExecutionSettings()

        assert settings.resources is None
        assert settings.checkpoint_dir is None
        assert settings.max_concurrent == 100
        assert settings.scheduler is None
        assert settings.on_task_complete is None
        assert settings.on_task_failed is None
        assert settings.profile is False
        assert settings.profile_path is None
        assert settings.profile_counters is True
        assert settings.profile_include_args is True

    def test_with_all_fields(self) -> None:
        """ExecutionSettings accepts all configuration fields."""

        def on_complete(node_id: str, result: object) -> None:
            pass

        def on_failed(node_id: str, error: Exception) -> None:
            pass

        settings = ExecutionSettings(
            resources=None,  # Would normally be ResourceConfig/ResourceManager
            checkpoint_dir="/data/checkpoints",
            max_concurrent=50,
            scheduler=None,  # Would normally be a Scheduler
            on_task_complete=on_complete,
            on_task_failed=on_failed,
            profile=True,
            profile_path="/traces/trace.json",
            profile_counters=False,
            profile_include_args=False,
        )

        assert settings.checkpoint_dir == "/data/checkpoints"
        assert settings.max_concurrent == 50
        assert settings.on_task_complete is on_complete
        assert settings.on_task_failed is on_failed
        assert settings.profile is True
        assert settings.profile_path == "/traces/trace.json"
        assert settings.profile_counters is False
        assert settings.profile_include_args is False

    def test_checkpoint_dir_as_path(self) -> None:
        """checkpoint_dir can be a Path object."""
        settings = ExecutionSettings(checkpoint_dir=Path("/data/checkpoints"))
        assert settings.checkpoint_dir == Path("/data/checkpoints")

    def test_checkpoint_dir_as_string(self) -> None:
        """checkpoint_dir can be a string."""
        settings = ExecutionSettings(checkpoint_dir="/data/checkpoints")
        assert settings.checkpoint_dir == "/data/checkpoints"


class TestGetExecutionSettings:
    """Tests for get_execution_settings()."""

    def test_default_none(self) -> None:
        """get_execution_settings returns None by default."""
        assert get_execution_settings() is None

    def test_returns_none_outside_context(self) -> None:
        """get_execution_settings returns None outside any context."""
        # Enter and exit a context to ensure cleanup
        with ExecutionSettings():
            pass
        assert get_execution_settings() is None


class TestSyncContextManager:
    """Tests for synchronous context manager behavior."""

    def test_enter_sets_context(self) -> None:
        """Entering context makes settings available."""
        settings = ExecutionSettings(max_concurrent=42)

        with settings:
            current = get_execution_settings()
            assert current is settings
            assert current.max_concurrent == 42

    def test_exit_clears_context(self) -> None:
        """Exiting context clears the settings."""
        with ExecutionSettings():
            assert get_execution_settings() is not None
        assert get_execution_settings() is None

    def test_returns_self(self) -> None:
        """Context manager returns the settings instance."""
        settings = ExecutionSettings(max_concurrent=10)

        with settings as ctx:
            assert ctx is settings

    def test_context_cleared_on_exception(self) -> None:
        """Context is cleared even when an exception occurs."""
        settings = ExecutionSettings()

        try:
            with settings:
                assert get_execution_settings() is settings
                raise ValueError("test exception")
        except ValueError:
            pass

        assert get_execution_settings() is None

    def test_nested_contexts(self) -> None:
        """Nested sync contexts work correctly with proper restoration."""
        outer = ExecutionSettings(max_concurrent=100)
        inner = ExecutionSettings(max_concurrent=10)

        assert get_execution_settings() is None

        with outer:
            current = get_execution_settings()
            assert current is outer
            assert current.max_concurrent == 100

            with inner:
                current = get_execution_settings()
                assert current is inner
                assert current.max_concurrent == 10

            # After inner exits, outer should be restored
            current = get_execution_settings()
            assert current is outer
            assert current.max_concurrent == 100

        assert get_execution_settings() is None

    def test_nested_context_restored_on_exception(self) -> None:
        """Nested contexts restore properly even with exceptions."""
        outer = ExecutionSettings(max_concurrent=100)
        inner = ExecutionSettings(max_concurrent=10)

        with outer:
            try:
                with inner:
                    assert get_execution_settings() is inner
                    raise ValueError("test exception")
            except ValueError:
                pass

            # Outer should be restored after inner raises
            assert get_execution_settings() is outer

        assert get_execution_settings() is None

    def test_multiple_sequential_contexts(self) -> None:
        """Multiple sequential context managers work correctly."""
        settings1 = ExecutionSettings(max_concurrent=10)
        settings2 = ExecutionSettings(max_concurrent=20)
        settings3 = ExecutionSettings(max_concurrent=30)

        with settings1:
            assert get_execution_settings() is settings1

        assert get_execution_settings() is None

        with settings2:
            assert get_execution_settings() is settings2

        assert get_execution_settings() is None

        with settings3:
            assert get_execution_settings() is settings3

        assert get_execution_settings() is None


class TestAsyncContextManager:
    """Tests for asynchronous context manager behavior."""

    @pytest.mark.asyncio
    async def test_aenter_sets_context(self) -> None:
        """Entering async context makes settings available."""
        settings = ExecutionSettings(max_concurrent=42)

        async with settings:
            current = get_execution_settings()
            assert current is settings
            assert current.max_concurrent == 42

    @pytest.mark.asyncio
    async def test_aexit_clears_context(self) -> None:
        """Exiting async context clears the settings."""
        async with ExecutionSettings():
            assert get_execution_settings() is not None
        assert get_execution_settings() is None

    @pytest.mark.asyncio
    async def test_returns_self(self) -> None:
        """Async context manager returns the settings instance."""
        settings = ExecutionSettings(max_concurrent=10)

        async with settings as ctx:
            assert ctx is settings

    @pytest.mark.asyncio
    async def test_context_cleared_on_exception(self) -> None:
        """Async context is cleared even when an exception occurs."""
        settings = ExecutionSettings()

        try:
            async with settings:
                assert get_execution_settings() is settings
                raise ValueError("test exception")
        except ValueError:
            pass

        assert get_execution_settings() is None

    @pytest.mark.asyncio
    async def test_nested_async_contexts(self) -> None:
        """Nested async contexts work correctly with proper restoration."""
        outer = ExecutionSettings(max_concurrent=100)
        inner = ExecutionSettings(max_concurrent=10)

        assert get_execution_settings() is None

        async with outer:
            assert get_execution_settings() is outer

            async with inner:
                assert get_execution_settings() is inner

            # After inner exits, outer should be restored
            assert get_execution_settings() is outer

        assert get_execution_settings() is None

    @pytest.mark.asyncio
    async def test_mixed_sync_async_contexts(self) -> None:
        """Sync and async contexts can be mixed (async outer, sync inner)."""
        outer = ExecutionSettings(max_concurrent=100)
        inner = ExecutionSettings(max_concurrent=10)

        async with outer:
            assert get_execution_settings() is outer

            with inner:
                assert get_execution_settings() is inner

            assert get_execution_settings() is outer

        assert get_execution_settings() is None


class TestNestedContextInheritance:
    """Tests for nested context value inheritance."""

    def test_inner_inherits_resources_from_outer(self) -> None:
        """Inner context can access outer's resources via _parent."""
        outer = ExecutionSettings(max_concurrent=100)
        inner = ExecutionSettings(max_concurrent=10)

        with outer:
            with inner:
                current = get_execution_settings()
                assert current is inner
                # Inner has its own max_concurrent
                assert current.max_concurrent == 10
                # But can still access the relationship
                assert current._parent is outer

    def test_get_effective_resources(self) -> None:
        """get_resources returns inherited value when not set locally."""
        # We can't easily test with real ResourceConfig/ResourceManager
        # without importing them, so we'll use a marker value
        outer = ExecutionSettings(max_concurrent=100)
        inner = ExecutionSettings(max_concurrent=10)

        with outer:
            with inner:
                current = get_execution_settings()
                assert current is not None
                # Resources are None in both, so returns None
                assert current.get_resources() is None

    def test_get_checkpoint_dir_inheritance(self, tmp_path: Path) -> None:
        """get_checkpoint_dir returns inherited value when not set locally."""
        outer_dir = tmp_path / "outer_checkpoints"
        outer = ExecutionSettings(checkpoint_dir=outer_dir)
        inner = ExecutionSettings()  # No checkpoint_dir

        with outer:
            # Outer has checkpoint_dir
            current = get_execution_settings()
            assert current is not None
            assert current.get_checkpoint_dir() == outer_dir

            with inner:
                # Inner inherits from outer
                current = get_execution_settings()
                assert current is not None
                assert current.get_checkpoint_dir() == outer_dir

    def test_get_checkpoint_dir_override(self, tmp_path: Path) -> None:
        """Inner checkpoint_dir overrides outer."""
        outer_dir = tmp_path / "outer_checkpoints"
        inner_dir = tmp_path / "inner_checkpoints"
        outer = ExecutionSettings(checkpoint_dir=outer_dir)
        inner = ExecutionSettings(checkpoint_dir=inner_dir)

        with outer:
            with inner:
                current = get_execution_settings()
                assert current is not None
                assert current.get_checkpoint_dir() == inner_dir

    def test_get_max_concurrent_always_returns_value(self) -> None:
        """get_max_concurrent always returns a value (not inherited since never None)."""
        outer = ExecutionSettings(max_concurrent=100)
        inner = ExecutionSettings(max_concurrent=10)

        with outer:
            current = get_execution_settings()
            assert current is not None
            assert current.get_max_concurrent() == 100

            with inner:
                current = get_execution_settings()
                assert current is not None
                assert current.get_max_concurrent() == 10

    def test_get_checkpoint_dir_returns_none_when_not_set(self) -> None:
        """get_checkpoint_dir returns None when not set in any context."""
        outer = ExecutionSettings(max_concurrent=100)
        inner = ExecutionSettings(max_concurrent=10)

        with outer:
            with inner:
                current = get_execution_settings()
                assert current is not None
                # Neither outer nor inner has checkpoint_dir set
                assert current.get_checkpoint_dir() is None

    def test_get_scheduler_returns_none_by_default(self) -> None:
        """get_scheduler returns None when not set."""
        with ExecutionSettings():
            current = get_execution_settings()
            assert current is not None
            assert current.get_scheduler() is None


class TestCheckpointManagerCreation:
    """Tests for CheckpointManager creation in context."""

    def test_checkpoint_manager_created_with_dir(self, tmp_path: Path) -> None:
        """CheckpointManager is created when checkpoint_dir is set."""
        settings = ExecutionSettings(checkpoint_dir=tmp_path)

        with settings:
            current = get_execution_settings()
            assert current is not None
            manager = current.get_checkpoint_manager()
            assert manager is not None
            assert manager.checkpoint_dir == tmp_path

    def test_no_checkpoint_manager_without_dir(self) -> None:
        """No CheckpointManager when checkpoint_dir is not set."""
        settings = ExecutionSettings()

        with settings:
            current = get_execution_settings()
            assert current is not None
            manager = current.get_checkpoint_manager()
            assert manager is None

    @pytest.mark.asyncio
    async def test_checkpoint_manager_created_async(self, tmp_path: Path) -> None:
        """CheckpointManager is created in async context."""
        settings = ExecutionSettings(checkpoint_dir=tmp_path)

        async with settings:
            current = get_execution_settings()
            assert current is not None
            manager = current.get_checkpoint_manager()
            assert manager is not None
            assert manager.checkpoint_dir == tmp_path

    def test_nested_inherits_checkpoint_manager(self, tmp_path: Path) -> None:
        """Inner context without checkpoint_dir inherits outer's manager."""
        outer = ExecutionSettings(checkpoint_dir=tmp_path)
        inner = ExecutionSettings(max_concurrent=10)

        with outer:
            current = get_execution_settings()
            assert current is not None
            outer_manager = current.get_checkpoint_manager()
            assert outer_manager is not None

            with inner:
                # Inner should inherit outer's manager
                current = get_execution_settings()
                assert current is not None
                inner_manager = current.get_checkpoint_manager()
                assert inner_manager is outer_manager

    def test_nested_with_own_checkpoint_manager(self, tmp_path: Path) -> None:
        """Inner context with checkpoint_dir gets its own manager."""
        outer_dir = tmp_path / "outer"
        inner_dir = tmp_path / "inner"
        outer_dir.mkdir()
        inner_dir.mkdir()

        outer = ExecutionSettings(checkpoint_dir=outer_dir)
        inner = ExecutionSettings(checkpoint_dir=inner_dir)

        with outer:
            current = get_execution_settings()
            assert current is not None
            outer_manager = current.get_checkpoint_manager()
            assert outer_manager is not None
            assert outer_manager.checkpoint_dir == outer_dir

            with inner:
                current = get_execution_settings()
                assert current is not None
                inner_manager = current.get_checkpoint_manager()
                assert inner_manager is not None
                assert inner_manager.checkpoint_dir == inner_dir
                assert inner_manager is not outer_manager

            # Back to outer's manager
            current = get_execution_settings()
            assert current is not None
            assert current.get_checkpoint_manager() is outer_manager


class TestCallbacks:
    """Tests for callback configuration."""

    def test_on_task_complete_callback(self) -> None:
        """on_task_complete callback is stored."""
        calls: list[str] = []

        def on_complete(node_id: str, result: object) -> None:
            calls.append(node_id)

        settings = ExecutionSettings(on_task_complete=on_complete)

        with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.on_task_complete is on_complete
            # Verify it's callable
            assert current.on_task_complete is not None
            current.on_task_complete("test_node", None)
            assert calls == ["test_node"]

    def test_on_task_failed_callback(self) -> None:
        """on_task_failed callback is stored."""
        calls: list[str] = []

        def on_failed(node_id: str, error: Exception) -> None:
            calls.append(node_id)

        settings = ExecutionSettings(on_task_failed=on_failed)

        with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.on_task_failed is on_failed
            # Verify it's callable
            assert current.on_task_failed is not None
            current.on_task_failed("test_node", ValueError("test"))
            assert calls == ["test_node"]


class TestProfileConfiguration:
    """Tests for profiling configuration (reserved for PR-067)."""

    def test_profile_defaults(self) -> None:
        """Profiling is disabled by default."""
        settings = ExecutionSettings()
        assert settings.profile is False
        assert settings.profile_path is None
        assert settings.profile_counters is True
        assert settings.profile_include_args is True

    def test_profile_enabled(self) -> None:
        """Profiling can be enabled."""
        settings = ExecutionSettings(
            profile=True,
            profile_path="/traces/run.json",
            profile_counters=False,
            profile_include_args=False,
        )
        assert settings.profile is True
        assert settings.profile_path == "/traces/run.json"
        assert settings.profile_counters is False
        assert settings.profile_include_args is False


class TestIntegration:
    """Integration tests for ExecutionSettings."""

    def test_context_available_in_simulated_module(self) -> None:
        """Simulates how a module would access execution settings."""
        settings = ExecutionSettings(max_concurrent=42)

        def simulated_module_check() -> int | None:
            ctx = get_execution_settings()
            if ctx is not None:
                return ctx.max_concurrent
            return None

        # Outside context
        assert simulated_module_check() is None

        # Inside context
        with settings:
            assert simulated_module_check() == 42

        # After context
        assert simulated_module_check() is None

    @pytest.mark.asyncio
    async def test_async_context_with_checkpoint_lifecycle(
        self, tmp_path: Path
    ) -> None:
        """Full lifecycle test with checkpoint manager in async context."""
        checkpoint_dir = tmp_path / "checkpoints"

        settings = ExecutionSettings(
            checkpoint_dir=checkpoint_dir,
            max_concurrent=50,
        )

        async with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.max_concurrent == 50

            manager = current.get_checkpoint_manager()
            assert manager is not None
            assert manager.checkpoint_dir == checkpoint_dir
            assert checkpoint_dir.exists()  # Manager creates the directory

        # After exit, context is cleared
        assert get_execution_settings() is None

    def test_deeply_nested_contexts(self) -> None:
        """Deeply nested contexts all restore properly."""
        contexts = [ExecutionSettings(max_concurrent=i * 10) for i in range(1, 6)]

        assert get_execution_settings() is None

        def assert_max_concurrent(expected: int) -> None:
            current = get_execution_settings()
            assert current is not None
            assert current.max_concurrent == expected

        with contexts[0]:
            assert_max_concurrent(10)
            with contexts[1]:
                assert_max_concurrent(20)
                with contexts[2]:
                    assert_max_concurrent(30)
                    with contexts[3]:
                        assert_max_concurrent(40)
                        with contexts[4]:
                            assert_max_concurrent(50)
                        assert_max_concurrent(40)
                    assert_max_concurrent(30)
                assert_max_concurrent(20)
            assert_max_concurrent(10)

        assert get_execution_settings() is None


class TestRepr:
    """Tests for string representation."""

    def test_repr_excludes_internal_fields(self) -> None:
        """repr() excludes internal state fields."""
        settings = ExecutionSettings(max_concurrent=50)
        repr_str = repr(settings)

        # Should include public fields
        assert "max_concurrent=50" in repr_str

        # Should exclude internal fields (they have repr=False)
        assert "_token" not in repr_str
        assert "_checkpoint_manager" not in repr_str
        assert "_parent" not in repr_str
