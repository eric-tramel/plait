"""Unit tests for the Scheduler class."""

import asyncio

import pytest

from inf_engine.execution.scheduler import Scheduler

# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Initialization Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerInit:
    """Tests for Scheduler initialization."""

    def test_init_default_max_concurrent(self) -> None:
        """Scheduler uses default max_concurrent of 100."""
        scheduler = Scheduler()

        assert scheduler.max_concurrent == 100

    def test_init_custom_max_concurrent(self) -> None:
        """Scheduler accepts custom max_concurrent value."""
        scheduler = Scheduler(max_concurrent=50)

        assert scheduler.max_concurrent == 50

    def test_init_max_concurrent_of_one(self) -> None:
        """Scheduler accepts max_concurrent of 1 (serial execution)."""
        scheduler = Scheduler(max_concurrent=1)

        assert scheduler.max_concurrent == 1

    def test_init_large_max_concurrent(self) -> None:
        """Scheduler accepts large max_concurrent values."""
        scheduler = Scheduler(max_concurrent=10000)

        assert scheduler.max_concurrent == 10000

    def test_init_zero_max_concurrent_raises(self) -> None:
        """Scheduler raises ValueError for max_concurrent of 0."""
        with pytest.raises(ValueError, match="max_concurrent must be at least 1"):
            Scheduler(max_concurrent=0)

    def test_init_negative_max_concurrent_raises(self) -> None:
        """Scheduler raises ValueError for negative max_concurrent."""
        with pytest.raises(ValueError, match="max_concurrent must be at least 1"):
            Scheduler(max_concurrent=-1)

    def test_init_creates_semaphore(self) -> None:
        """Scheduler creates internal semaphore."""
        scheduler = Scheduler(max_concurrent=10)

        assert hasattr(scheduler, "_semaphore")
        assert isinstance(scheduler._semaphore, asyncio.Semaphore)

    def test_init_active_count_is_zero(self) -> None:
        """Scheduler starts with zero active tasks."""
        scheduler = Scheduler()

        assert scheduler.active_count == 0

    def test_init_available_slots_equals_max_concurrent(self) -> None:
        """Scheduler starts with all slots available."""
        scheduler = Scheduler(max_concurrent=25)

        assert scheduler.available_slots == 25


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Acquire/Release Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerAcquireRelease:
    """Tests for Scheduler acquire/release methods."""

    @pytest.mark.asyncio
    async def test_acquire_increments_active_count(self) -> None:
        """acquire() increments active_count."""
        scheduler = Scheduler(max_concurrent=10)

        await scheduler.acquire()

        assert scheduler.active_count == 1
        scheduler.release()  # cleanup

    @pytest.mark.asyncio
    async def test_release_decrements_active_count(self) -> None:
        """release() decrements active_count."""
        scheduler = Scheduler(max_concurrent=10)
        await scheduler.acquire()
        assert scheduler.active_count == 1

        scheduler.release()

        assert scheduler.active_count == 0

    @pytest.mark.asyncio
    async def test_multiple_acquires(self) -> None:
        """Multiple acquires increment active_count correctly."""
        scheduler = Scheduler(max_concurrent=10)

        await scheduler.acquire()
        await scheduler.acquire()
        await scheduler.acquire()

        assert scheduler.active_count == 3
        assert scheduler.available_slots == 7

        # cleanup
        scheduler.release()
        scheduler.release()
        scheduler.release()

    @pytest.mark.asyncio
    async def test_release_without_acquire_raises(self) -> None:
        """release() without acquire raises ValueError."""
        scheduler = Scheduler(max_concurrent=10)

        with pytest.raises(ValueError, match="Cannot release: no active tasks"):
            scheduler.release()

    @pytest.mark.asyncio
    async def test_acquire_respects_concurrency_limit(self) -> None:
        """acquire() blocks when at max concurrent tasks."""
        scheduler = Scheduler(max_concurrent=2)

        # Acquire both slots
        await scheduler.acquire()
        await scheduler.acquire()
        assert scheduler.active_count == 2
        assert scheduler.available_slots == 0

        # Third acquire should block - test with timeout
        acquired = False

        async def try_acquire() -> None:
            nonlocal acquired
            await scheduler.acquire()
            acquired = True

        # Start the blocking acquire
        task = asyncio.create_task(try_acquire())

        # Give it a moment - it should NOT complete
        await asyncio.sleep(0.01)
        assert not acquired, "acquire() should block when at capacity"

        # Release a slot - now it should complete
        scheduler.release()
        await asyncio.sleep(0.01)
        assert acquired, "acquire() should complete after release"

        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        scheduler.release()
        scheduler.release()

    @pytest.mark.asyncio
    async def test_acquire_release_cycle(self) -> None:
        """Multiple acquire/release cycles work correctly."""
        scheduler = Scheduler(max_concurrent=5)

        for _ in range(10):
            await scheduler.acquire()
            assert scheduler.active_count == 1
            scheduler.release()
            assert scheduler.active_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Context Manager Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerContextManager:
    """Tests for Scheduler async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_acquires_on_enter(self) -> None:
        """async with scheduler acquires a slot on entry."""
        scheduler = Scheduler(max_concurrent=10)

        async with scheduler:
            assert scheduler.active_count == 1

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exit(self) -> None:
        """async with scheduler releases slot on exit."""
        scheduler = Scheduler(max_concurrent=10)

        async with scheduler:
            assert scheduler.active_count == 1

        assert scheduler.active_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exception(self) -> None:
        """async with scheduler releases slot even on exception."""
        scheduler = Scheduler(max_concurrent=10)

        with pytest.raises(ValueError):
            async with scheduler:
                assert scheduler.active_count == 1
                raise ValueError("Test error")

        assert scheduler.active_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_returns_scheduler(self) -> None:
        """async with scheduler as s returns the scheduler."""
        scheduler = Scheduler(max_concurrent=10)

        async with scheduler as s:
            assert s is scheduler

    @pytest.mark.asyncio
    async def test_nested_context_managers(self) -> None:
        """Multiple nested context managers work correctly."""
        scheduler = Scheduler(max_concurrent=10)

        async with scheduler:
            assert scheduler.active_count == 1
            async with scheduler:
                assert scheduler.active_count == 2
                async with scheduler:
                    assert scheduler.active_count == 3
                assert scheduler.active_count == 2
            assert scheduler.active_count == 1
        assert scheduler.active_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Semaphore Behavior Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerSemaphoreBehavior:
    """Tests for Scheduler semaphore concurrency limiting behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_tasks_respect_limit(self) -> None:
        """Concurrent tasks are limited by max_concurrent."""
        scheduler = Scheduler(max_concurrent=3)
        max_observed = 0
        completed = 0

        async def task() -> None:
            nonlocal max_observed, completed
            async with scheduler:
                max_observed = max(max_observed, scheduler.active_count)
                await asyncio.sleep(0.01)  # Simulate work
                completed += 1

        # Run 10 concurrent tasks with limit of 3
        tasks = [asyncio.create_task(task()) for _ in range(10)]
        await asyncio.gather(*tasks)

        assert completed == 10
        assert max_observed <= 3

    @pytest.mark.asyncio
    async def test_serial_execution_with_max_one(self) -> None:
        """max_concurrent=1 forces serial execution."""
        scheduler = Scheduler(max_concurrent=1)
        execution_order: list[int] = []
        in_critical_section = False

        async def task(task_id: int) -> None:
            nonlocal in_critical_section
            async with scheduler:
                # Check no other task is in critical section
                assert not in_critical_section, "Tasks should not overlap"
                in_critical_section = True
                execution_order.append(task_id)
                await asyncio.sleep(0.001)  # Simulate work
                in_critical_section = False

        tasks = [asyncio.create_task(task(i)) for i in range(5)]
        await asyncio.gather(*tasks)

        assert len(execution_order) == 5

    @pytest.mark.asyncio
    async def test_all_tasks_eventually_complete(self) -> None:
        """All tasks complete even when exceeding concurrency limit."""
        scheduler = Scheduler(max_concurrent=2)
        completed_tasks: list[int] = []

        async def task(task_id: int) -> None:
            async with scheduler:
                await asyncio.sleep(0.001)
                completed_tasks.append(task_id)

        # Run more tasks than the concurrency limit
        tasks = [asyncio.create_task(task(i)) for i in range(20)]
        await asyncio.gather(*tasks)

        assert len(completed_tasks) == 20
        assert set(completed_tasks) == set(range(20))

    @pytest.mark.asyncio
    async def test_available_slots_updates_correctly(self) -> None:
        """available_slots updates as tasks acquire and release."""
        scheduler = Scheduler(max_concurrent=5)

        assert scheduler.available_slots == 5

        await scheduler.acquire()
        assert scheduler.available_slots == 4

        await scheduler.acquire()
        assert scheduler.available_slots == 3

        scheduler.release()
        assert scheduler.available_slots == 4

        scheduler.release()
        assert scheduler.available_slots == 5

    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self) -> None:
        """Scheduler handles high concurrency correctly."""
        scheduler = Scheduler(max_concurrent=50)
        counter = 0
        max_concurrent_observed = 0

        async def increment() -> None:
            nonlocal counter, max_concurrent_observed
            async with scheduler:
                max_concurrent_observed = max(
                    max_concurrent_observed, scheduler.active_count
                )
                counter += 1

        # Run many tasks concurrently
        tasks = [asyncio.create_task(increment()) for _ in range(200)]
        await asyncio.gather(*tasks)

        assert counter == 200
        assert max_concurrent_observed <= 50


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Property Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerProperties:
    """Tests for Scheduler properties."""

    def test_max_concurrent_is_readonly_attribute(self) -> None:
        """max_concurrent is set at init and accessible."""
        scheduler = Scheduler(max_concurrent=42)

        assert scheduler.max_concurrent == 42

    @pytest.mark.asyncio
    async def test_active_count_reflects_current_state(self) -> None:
        """active_count accurately reflects current active tasks."""
        scheduler = Scheduler(max_concurrent=10)

        assert scheduler.active_count == 0

        await scheduler.acquire()
        assert scheduler.active_count == 1

        await scheduler.acquire()
        assert scheduler.active_count == 2

        scheduler.release()
        assert scheduler.active_count == 1

        scheduler.release()
        assert scheduler.active_count == 0

    @pytest.mark.asyncio
    async def test_available_slots_is_computed_correctly(self) -> None:
        """available_slots = max_concurrent - active_count."""
        scheduler = Scheduler(max_concurrent=10)

        for i in range(10):
            assert scheduler.available_slots == 10 - i
            await scheduler.acquire()

        assert scheduler.available_slots == 0

        for i in range(10):
            scheduler.release()
            assert scheduler.available_slots == i + 1


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Import Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerImports:
    """Tests for Scheduler module imports."""

    def test_import_from_execution_package(self) -> None:
        """Scheduler can be imported from execution package."""
        from inf_engine.execution import Scheduler as SchedulerFromPackage

        assert SchedulerFromPackage is Scheduler

    def test_import_from_scheduler_module(self) -> None:
        """Scheduler can be imported from scheduler module."""
        from inf_engine.execution.scheduler import Scheduler as SchedulerFromModule

        assert SchedulerFromModule is Scheduler
