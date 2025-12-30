"""Profiling infrastructure for inf-engine.

This module provides execution profiling capabilities using Chrome Trace Event
Format for visualization in Perfetto or Chrome DevTools. Enable profiling via
ExecutionSettings to capture detailed timing and execution data.

Example:
    >>> from inf_engine.execution.context import ExecutionSettings
    >>> from inf_engine.profiling import TraceProfiler
    >>>
    >>> # Enable profiling through ExecutionSettings
    >>> async with ExecutionSettings(
    ...     resources=config,
    ...     profile=True,
    ...     profile_path="./traces/run.json",
    ... ) as settings:
    ...     result = await pipeline(input_data)
    >>>
    >>> # Trace file is automatically exported on context exit
    >>> # Open with: https://ui.perfetto.dev

Example with direct profiler access:
    >>> profiler = TraceProfiler(include_counters=True)
    >>> profiler.task_start("task_1", "gpt-4", "MyModule")
    >>> # ... task executes ...
    >>> profiler.task_end("task_1", "gpt-4", duration_ms=150.0)
    >>> profiler.export("./trace.json")
"""

from inf_engine.profiling.profiler import (
    EndpointStats,
    ProfilerStatistics,
    TraceEvent,
    TraceProfiler,
)

__all__ = [
    "TraceEvent",
    "TraceProfiler",
    "ProfilerStatistics",
    "EndpointStats",
]
