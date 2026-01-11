"""Unit tests for the ExecutionSettings context manager."""

from pathlib import Path

import pytest

from plait.execution.context import ExecutionSettings, get_execution_settings


@pytest.fixture(autouse=True)
def clean_context() -> None:
    """Ensure execution settings context is clean before each test.

    This fixture runs automatically before each test to prevent
    context leakage between tests.
    """
    from plait.execution.context import _execution_settings

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
        assert settings.task_timeout is None
        assert settings.max_task_retries == 0
        assert settings.task_retry_delay == 1.0
        assert settings.streaming is False
        assert settings.preserve_order is False
        assert settings.on_progress is None
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

        def on_progress(done: int, total: int) -> None:
            pass

        settings = ExecutionSettings(
            resources=None,  # Would normally be ResourceConfig/ResourceManager
            checkpoint_dir="/data/checkpoints",
            max_concurrent=50,
            scheduler=None,  # Would normally be a Scheduler
            on_task_complete=on_complete,
            on_task_failed=on_failed,
            task_timeout=120.0,
            max_task_retries=3,
            task_retry_delay=2.0,
            streaming=True,
            preserve_order=True,
            on_progress=on_progress,
            profile=True,
            profile_path="/traces/trace.json",
            profile_counters=False,
            profile_include_args=False,
        )

        assert settings.checkpoint_dir == "/data/checkpoints"
        assert settings.max_concurrent == 50
        assert settings.on_task_complete is on_complete
        assert settings.on_task_failed is on_failed
        assert settings.task_timeout == 120.0
        assert settings.max_task_retries == 3
        assert settings.task_retry_delay == 2.0
        assert settings.streaming is True
        assert settings.preserve_order is True
        assert settings.on_progress is on_progress
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
    """Tests for profiling configuration."""

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

    def test_profiler_not_created_when_disabled(self) -> None:
        """Profiler is not created when profile=False."""
        settings = ExecutionSettings(profile=False)

        with settings:
            assert settings.get_profiler() is None
            assert settings.profiler is None

    def test_profiler_created_when_enabled(self) -> None:
        """Profiler is created when profile=True."""
        settings = ExecutionSettings(profile=True)

        with settings:
            profiler = settings.get_profiler()
            assert profiler is not None
            # Verify it's the right type
            from plait.profiling import TraceProfiler

            assert isinstance(profiler, TraceProfiler)

    def test_profiler_property_shorthand(self) -> None:
        """profiler property is shorthand for get_profiler()."""
        settings = ExecutionSettings(profile=True)

        with settings:
            assert settings.profiler is settings.get_profiler()
            assert settings.profiler is not None

    def test_profiler_inherits_settings(self) -> None:
        """Profiler is created with correct settings."""
        settings = ExecutionSettings(
            profile=True,
            profile_counters=False,
            profile_include_args=False,
        )

        with settings:
            profiler = settings.get_profiler()
            assert profiler is not None
            assert profiler.include_counters is False
            assert profiler.include_args is False

    def test_profiler_default_settings(self) -> None:
        """Profiler uses default settings when not specified."""
        settings = ExecutionSettings(profile=True)

        with settings:
            profiler = settings.get_profiler()
            assert profiler is not None
            assert profiler.include_counters is True
            assert profiler.include_args is True

    @pytest.mark.asyncio
    async def test_profiler_created_async_context(self) -> None:
        """Profiler is created in async context."""
        settings = ExecutionSettings(profile=True)

        async with settings:
            assert settings.profiler is not None

    def test_nested_context_inherits_profiler(self) -> None:
        """Inner context without profile setting inherits outer's profiler."""
        outer = ExecutionSettings(profile=True)
        inner = ExecutionSettings(max_concurrent=10)

        with outer:
            outer_profiler = outer.get_profiler()
            assert outer_profiler is not None

            with inner:
                # Inner should inherit outer's profiler
                inner_profiler = inner.get_profiler()
                assert inner_profiler is outer_profiler

    def test_nested_context_with_own_profiler(self) -> None:
        """Inner context with profile=True gets its own profiler."""
        outer = ExecutionSettings(profile=True)
        inner = ExecutionSettings(profile=True)

        with outer:
            outer_profiler = outer.get_profiler()
            assert outer_profiler is not None

            with inner:
                # Inner has its own profiler
                inner_profiler = inner.get_profiler()
                assert inner_profiler is not None
                assert inner_profiler is not outer_profiler

    def test_profiler_export_on_sync_exit(self, tmp_path: Path) -> None:
        """Profiler trace is exported on sync context exit."""
        output_path = tmp_path / "trace.json"
        settings = ExecutionSettings(profile=True, profile_path=output_path)

        with settings:
            profiler = settings.profiler
            assert profiler is not None
            # Record some events
            profiler.add_instant_event("test")

        # Trace should be exported after exit
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_profiler_export_on_async_exit(self, tmp_path: Path) -> None:
        """Profiler trace is exported on async context exit."""
        output_path = tmp_path / "trace.json"
        settings = ExecutionSettings(profile=True, profile_path=output_path)

        async with settings:
            profiler = settings.profiler
            assert profiler is not None
            # Record some events
            profiler.add_instant_event("test")

        # Trace should be exported after exit
        assert output_path.exists()

    def test_profiler_export_auto_path(self, tmp_path: Path) -> None:
        """Profiler generates timestamped path when none specified."""
        import os

        # Change to tmp_path so the auto-generated traces dir is there
        orig_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            settings = ExecutionSettings(profile=True)

            with settings:
                profiler = settings.profiler
                assert profiler is not None
                profiler.add_instant_event("test")

            # Trace should be exported to traces directory
            traces_dir = tmp_path / "traces"
            assert traces_dir.exists()
            trace_files = list(traces_dir.glob("trace_*.json"))
            assert len(trace_files) == 1
        finally:
            os.chdir(orig_dir)

    def test_profiler_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Profiler export creates parent directories."""
        output_path = tmp_path / "nested" / "dirs" / "trace.json"
        settings = ExecutionSettings(profile=True, profile_path=output_path)

        with settings:
            pass  # Just enter and exit

        assert output_path.exists()

    def test_profiler_cleared_after_exit(self, tmp_path: Path) -> None:
        """Profiler reference is cleared after context exit."""
        output_path = tmp_path / "trace.json"
        settings = ExecutionSettings(profile=True, profile_path=output_path)

        with settings:
            assert settings._profiler is not None

        # Internal profiler should be cleared
        assert settings._profiler is None

    @pytest.mark.asyncio
    async def test_profiler_cleared_after_async_exit(self, tmp_path: Path) -> None:
        """Profiler reference is cleared after async context exit."""
        output_path = tmp_path / "trace.json"
        settings = ExecutionSettings(profile=True, profile_path=output_path)

        async with settings:
            assert settings._profiler is not None

        # Internal profiler should be cleared
        assert settings._profiler is None


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


class TestTimeoutRetryConfiguration:
    """Tests for timeout and retry configuration."""

    def test_task_timeout_default_none(self) -> None:
        """task_timeout defaults to None."""
        settings = ExecutionSettings()
        assert settings.task_timeout is None

    def test_task_timeout_can_be_set(self) -> None:
        """task_timeout can be set to a float."""
        settings = ExecutionSettings(task_timeout=60.0)
        assert settings.task_timeout == 60.0

    def test_max_task_retries_default_zero(self) -> None:
        """max_task_retries defaults to 0."""
        settings = ExecutionSettings()
        assert settings.max_task_retries == 0

    def test_max_task_retries_can_be_set(self) -> None:
        """max_task_retries can be set."""
        settings = ExecutionSettings(max_task_retries=3)
        assert settings.max_task_retries == 3

    def test_task_retry_delay_default_one(self) -> None:
        """task_retry_delay defaults to 1.0."""
        settings = ExecutionSettings()
        assert settings.task_retry_delay == 1.0

    def test_task_retry_delay_can_be_set(self) -> None:
        """task_retry_delay can be set."""
        settings = ExecutionSettings(task_retry_delay=2.5)
        assert settings.task_retry_delay == 2.5

    def test_all_timeout_retry_settings(self) -> None:
        """All timeout/retry settings can be set together."""
        settings = ExecutionSettings(
            task_timeout=120.0,
            max_task_retries=5,
            task_retry_delay=0.5,
        )
        assert settings.task_timeout == 120.0
        assert settings.max_task_retries == 5
        assert settings.task_retry_delay == 0.5

    def test_get_task_timeout(self) -> None:
        """get_task_timeout returns the timeout value."""
        settings = ExecutionSettings(task_timeout=30.0)
        with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.get_task_timeout() == 30.0

    def test_get_task_timeout_none(self) -> None:
        """get_task_timeout returns None when not set."""
        settings = ExecutionSettings()
        with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.get_task_timeout() is None

    def test_get_task_timeout_inherits_from_parent(self) -> None:
        """get_task_timeout inherits from parent context."""
        outer = ExecutionSettings(task_timeout=60.0)
        inner = ExecutionSettings(max_task_retries=3)  # No timeout

        with outer:
            with inner:
                current = get_execution_settings()
                assert current is not None
                # Inner inherits outer's timeout
                assert current.get_task_timeout() == 60.0

    def test_get_task_timeout_override(self) -> None:
        """Inner task_timeout overrides outer."""
        outer = ExecutionSettings(task_timeout=60.0)
        inner = ExecutionSettings(task_timeout=30.0)

        with outer:
            current = get_execution_settings()
            assert current is not None
            assert current.get_task_timeout() == 60.0

            with inner:
                current = get_execution_settings()
                assert current is not None
                assert current.get_task_timeout() == 30.0

            current = get_execution_settings()
            assert current is not None
            assert current.get_task_timeout() == 60.0

    def test_get_max_task_retries(self) -> None:
        """get_max_task_retries returns the retries value."""
        settings = ExecutionSettings(max_task_retries=5)
        with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.get_max_task_retries() == 5

    def test_get_task_retry_delay(self) -> None:
        """get_task_retry_delay returns the delay value."""
        settings = ExecutionSettings(task_retry_delay=2.0)
        with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.get_task_retry_delay() == 2.0


class TestStreamingConfiguration:
    """Tests for streaming execution configuration."""

    def test_streaming_default_false(self) -> None:
        """streaming defaults to False."""
        settings = ExecutionSettings()
        assert settings.streaming is False

    def test_streaming_can_be_enabled(self) -> None:
        """streaming can be set to True."""
        settings = ExecutionSettings(streaming=True)
        assert settings.streaming is True

    def test_preserve_order_default_false(self) -> None:
        """preserve_order defaults to False."""
        settings = ExecutionSettings()
        assert settings.preserve_order is False

    def test_preserve_order_can_be_enabled(self) -> None:
        """preserve_order can be set to True."""
        settings = ExecutionSettings(preserve_order=True)
        assert settings.preserve_order is True

    def test_on_progress_default_none(self) -> None:
        """on_progress defaults to None."""
        settings = ExecutionSettings()
        assert settings.on_progress is None

    def test_on_progress_callback(self) -> None:
        """on_progress can be set to a callback."""
        calls: list[tuple[int, int]] = []

        def on_progress(done: int, total: int) -> None:
            calls.append((done, total))

        settings = ExecutionSettings(on_progress=on_progress)
        assert settings.on_progress is on_progress

        # Verify callback is callable
        settings.on_progress(5, 10)
        assert calls == [(5, 10)]

    def test_get_streaming(self) -> None:
        """get_streaming returns the streaming value."""
        settings = ExecutionSettings(streaming=True)
        with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.get_streaming() is True

    def test_get_preserve_order(self) -> None:
        """get_preserve_order returns the preserve_order value."""
        settings = ExecutionSettings(preserve_order=True)
        with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.get_preserve_order() is True

    def test_get_on_progress(self) -> None:
        """get_on_progress returns the callback."""

        def callback(done: int, total: int) -> None:
            pass

        settings = ExecutionSettings(on_progress=callback)
        with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.get_on_progress() is callback

    def test_get_on_progress_inherits_from_parent(self) -> None:
        """get_on_progress inherits from parent context."""

        def callback(done: int, total: int) -> None:
            pass

        outer = ExecutionSettings(on_progress=callback)
        inner = ExecutionSettings(streaming=True)  # No on_progress

        with outer:
            with inner:
                current = get_execution_settings()
                assert current is not None
                # Inner inherits outer's on_progress
                assert current.get_on_progress() is callback

    def test_streaming_and_preserve_order_combination(self) -> None:
        """streaming and preserve_order can be used together."""
        settings = ExecutionSettings(streaming=True, preserve_order=True)

        with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.get_streaming() is True
            assert current.get_preserve_order() is True

    def test_nested_context_streaming_override(self) -> None:
        """Inner context can have different streaming settings."""
        outer = ExecutionSettings(streaming=True)
        inner = ExecutionSettings(streaming=False)

        with outer:
            current = get_execution_settings()
            assert current is not None
            assert current.get_streaming() is True

            with inner:
                current = get_execution_settings()
                assert current is not None
                # Inner overrides to False
                assert current.get_streaming() is False

            # Back to outer
            current = get_execution_settings()
            assert current is not None
            assert current.get_streaming() is True

    @pytest.mark.asyncio
    async def test_async_context_with_streaming(self) -> None:
        """Streaming settings work in async context."""
        settings = ExecutionSettings(streaming=True, preserve_order=True)

        async with settings:
            current = get_execution_settings()
            assert current is not None
            assert current.get_streaming() is True
            assert current.get_preserve_order() is True
