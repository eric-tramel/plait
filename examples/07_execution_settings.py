#!/usr/bin/env python3
"""Example 07: ExecutionSettings and Module Binding.

This example demonstrates modern patterns for executing inference modules:

1. **bind() method**: Bind resources to a module for direct `await module(input)`
2. **ExecutionSettings context**: Share resources across multiple modules
3. **Configuration priority**: Understanding context < bound < kwargs precedence
4. **Batch execution**: Processing multiple inputs with `await module([a, b, c])`

These patterns provide a clean, PyTorch-like API for module execution.

Run with: python examples/07_execution_settings.py
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import MagicMock

from plait.execution.context import ExecutionSettings
from plait.execution.executor import run
from plait.module import Module

# ─────────────────────────────────────────────────────────────────────────────
# Mock Modules for Examples
# ─────────────────────────────────────────────────────────────────────────────


class TextProcessor(Module):
    """Mock module that processes text with a transformation."""

    def __init__(self, name: str = "processor") -> None:
        super().__init__()
        self.name = name

    def forward(self, text: str) -> str:
        return f"[{self.name}] {text}"


class AsyncProcessor(Module):
    """Mock async module that simulates processing delay."""

    def __init__(self, name: str, delay_ms: float = 50) -> None:
        super().__init__()
        self.name = name
        self.delay_ms = delay_ms

    async def forward(self, text: str) -> str:
        await asyncio.sleep(self.delay_ms / 1000)
        return f"[{self.name}] {text}"


class Pipeline(Module):
    """A simple multi-step pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.step1 = TextProcessor("step1")
        self.step2 = TextProcessor("step2")
        self.step3 = TextProcessor("step3")

    def forward(self, text: str) -> str:
        r1 = self.step1(text)
        r2 = self.step2(r1)
        r3 = self.step3(r2)
        return r3


class ParallelPipeline(Module):
    """Pipeline with parallel branches."""

    def __init__(self) -> None:
        super().__init__()
        self.branch_a = AsyncProcessor("A", delay_ms=30)
        self.branch_b = AsyncProcessor("B", delay_ms=30)
        self.branch_c = AsyncProcessor("C", delay_ms=30)

    def forward(self, text: str) -> dict[str, str]:
        return {
            "a": self.branch_a(text),
            "b": self.branch_b(text),
            "c": self.branch_c(text),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Basic bind() Usage
# ─────────────────────────────────────────────────────────────────────────────


async def example_basic_bind() -> None:
    """Demonstrate binding resources to a module for direct execution.

    The bind() method attaches resources to a module, enabling the
    intuitive `await module(input)` pattern.
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic bind() Usage")
    print("=" * 70)

    # Create a mock resources object (in real usage, this would be ResourceConfig)
    mock_resources = MagicMock(name="resources")

    # Create and bind the module
    pipeline = Pipeline().bind(resources=mock_resources)

    print("\nPipeline bound with resources.")
    print(f"  _bound_resources: {pipeline._bound_resources}")
    print(f"  _bound_config: {pipeline._bound_config}")

    # Now you can call the module directly with await
    print("\nCalling: await pipeline('Hello')")
    result = await pipeline("Hello")
    print(f"Result: {result}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: bind() with Configuration
# ─────────────────────────────────────────────────────────────────────────────


async def example_bind_with_config() -> None:
    """Demonstrate bind() with additional configuration options.

    bind() accepts max_concurrent and any other execution options.
    These are stored and passed to run() automatically.
    """
    print("\n" + "=" * 70)
    print("Example 2: bind() with Configuration")
    print("=" * 70)

    mock_resources = MagicMock(name="resources")
    checkpoint_dir = Path(mkdtemp(prefix="plait_bind_"))

    try:
        # Bind with additional configuration
        pipeline = Pipeline().bind(
            resources=mock_resources,
            max_concurrent=50,  # Limit concurrency
            checkpoint_dir=checkpoint_dir,  # Enable checkpointing
        )

        print("\nBound configuration:")
        print(f"  max_concurrent: {pipeline._bound_config.get('max_concurrent')}")
        print(f"  checkpoint_dir: {pipeline._bound_config.get('checkpoint_dir')}")

        # Execute
        result = await pipeline("test input")
        print(f"\nResult: {result}")

    finally:
        import shutil

        shutil.rmtree(checkpoint_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: ExecutionSettings Context Manager
# ─────────────────────────────────────────────────────────────────────────────


async def example_execution_settings() -> None:
    """Demonstrate ExecutionSettings for shared resources.

    ExecutionSettings provides a context manager for sharing resources
    and configuration across multiple module executions.
    """
    print("\n" + "=" * 70)
    print("Example 3: ExecutionSettings Context Manager")
    print("=" * 70)

    mock_resources = MagicMock(name="shared_resources")

    # Create multiple modules (not individually bound)
    pipeline1 = Pipeline()
    pipeline2 = ParallelPipeline()

    print("\nModules created without individual binding.")
    print(f"  pipeline1._bound_resources: {pipeline1._bound_resources}")
    print(f"  pipeline2._bound_resources: {pipeline2._bound_resources}")

    # Use ExecutionSettings to share resources
    print("\nUsing ExecutionSettings context:")
    async with ExecutionSettings(resources=mock_resources, max_concurrent=25):
        # Both modules can now be called with await
        print("  Calling pipeline1...")
        result1 = await pipeline1("input A")
        print(f"    Result: {result1}")

        print("  Calling pipeline2...")
        result2 = await pipeline2("input B")
        print(f"    Result: {result2}")

    print("\nBoth modules shared the same resources and configuration!")


# ─────────────────────────────────────────────────────────────────────────────
# Example 4: Configuration Priority Order
# ─────────────────────────────────────────────────────────────────────────────


async def example_configuration_priority() -> None:
    """Demonstrate configuration priority: kwargs > bound > context.

    When multiple sources provide the same setting, the highest
    priority source wins:
    1. Call-time kwargs (highest)
    2. Bound settings (from bind())
    3. Context settings (from ExecutionSettings)
    4. Defaults (lowest)
    """
    print("\n" + "=" * 70)
    print("Example 4: Configuration Priority Order")
    print("=" * 70)

    context_resources = MagicMock(name="context_resources")
    bound_resources = MagicMock(name="bound_resources")

    pipeline = Pipeline()

    # Bind with specific settings
    pipeline.bind(
        resources=bound_resources,
        max_concurrent=50,  # Bound setting
    )

    print("\nConfiguration sources:")
    print("  Context: resources=context_resources, max_concurrent=100")
    print("  Bound:   resources=bound_resources, max_concurrent=50")
    print("  Kwargs:  max_concurrent=25")

    async with ExecutionSettings(resources=context_resources, max_concurrent=100):
        # Override max_concurrent at call time
        result = await pipeline("test", max_concurrent=25)

        print("\nResulting configuration (priority order):")
        print("  resources: bound_resources (bound > context)")
        print("  max_concurrent: 25 (kwargs > bound > context)")

    print(f"\nResult: {result}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 5: Batch Execution
# ─────────────────────────────────────────────────────────────────────────────


async def example_batch_execution() -> None:
    """Demonstrate batch execution with list inputs.

    When you pass a list as the first argument, the module
    processes each item and returns a list of results.
    """
    print("\n" + "=" * 70)
    print("Example 5: Batch Execution")
    print("=" * 70)

    mock_resources = MagicMock(name="resources")
    pipeline = Pipeline().bind(resources=mock_resources)

    # Prepare batch inputs
    inputs = ["apple", "banana", "cherry", "date"]

    print(f"\nBatch inputs: {inputs}")
    print("Calling: await pipeline([...])")

    start = time.time()
    results = await pipeline(inputs)
    elapsed = (time.time() - start) * 1000

    print(f"\nBatch results ({elapsed:.0f}ms):")
    for inp, result in zip(inputs, results, strict=True):
        print(f"  '{inp}' -> '{result}'")

    print(f"\nProcessed {len(inputs)} inputs in a single call!")


# ─────────────────────────────────────────────────────────────────────────────
# Example 6: Nested Contexts
# ─────────────────────────────────────────────────────────────────────────────


async def example_nested_contexts() -> None:
    """Demonstrate nested ExecutionSettings contexts.

    Inner contexts override outer contexts. When the inner
    context exits, the outer context is restored.
    """
    print("\n" + "=" * 70)
    print("Example 6: Nested Contexts")
    print("=" * 70)

    outer_resources = MagicMock(name="outer_resources")
    inner_resources = MagicMock(name="inner_resources")

    pipeline = Pipeline()

    print("\nNested context demonstration:")

    async with ExecutionSettings(resources=outer_resources, max_concurrent=100):
        print("  Outer context: max_concurrent=100")
        result1 = await pipeline("outer call")

        async with ExecutionSettings(resources=inner_resources, max_concurrent=10):
            print("  Inner context: max_concurrent=10 (overrides outer)")
            result2 = await pipeline("inner call")

        print("  Back to outer context: max_concurrent=100")
        result3 = await pipeline("outer call again")

    print("\nResults:")
    print(f"  Outer: {result1}")
    print(f"  Inner: {result2}")
    print(f"  Outer again: {result3}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 7: Mixing run() and bind()
# ─────────────────────────────────────────────────────────────────────────────


async def example_mixing_patterns() -> None:
    """Demonstrate that run() and bind() can coexist.

    You can use run() directly for fine-grained control,
    or bind() for a cleaner API. Both work together.
    """
    print("\n" + "=" * 70)
    print("Example 7: Mixing run() and bind() Patterns")
    print("=" * 70)

    mock_resources = MagicMock(name="resources")

    pipeline = Pipeline()

    print("\nUsing run() directly:")
    result1 = await run(pipeline, "via run()", resources=mock_resources)
    print(f"  Result: {result1}")

    print("\nUsing bind() and await:")
    pipeline.bind(resources=mock_resources)
    result2 = await pipeline("via bind()")
    print(f"  Result: {result2}")

    print("\nUsing ExecutionSettings:")
    pipeline2 = Pipeline()  # Fresh, unbound module
    async with ExecutionSettings(resources=mock_resources):
        result3 = await pipeline2("via context")
        print(f"  Result: {result3}")

    print("\nAll three patterns work seamlessly!")


# ─────────────────────────────────────────────────────────────────────────────
# Example 8: Shared Checkpointing with ExecutionSettings
# ─────────────────────────────────────────────────────────────────────────────


async def example_shared_checkpointing() -> None:
    """Demonstrate shared checkpointing across multiple pipelines.

    ExecutionSettings can set up a shared CheckpointManager that
    tracks progress across all module executions in the context.
    """
    print("\n" + "=" * 70)
    print("Example 8: Shared Checkpointing with ExecutionSettings")
    print("=" * 70)

    checkpoint_dir = Path(mkdtemp(prefix="plait_shared_"))
    mock_resources = MagicMock(name="resources")

    try:
        pipeline1 = Pipeline()
        pipeline2 = ParallelPipeline()

        print(f"\nCheckpoint directory: {checkpoint_dir}")
        print("Running multiple pipelines with shared checkpointing...")

        async with ExecutionSettings(
            resources=mock_resources,
            checkpoint_dir=checkpoint_dir,
        ):
            # Both pipelines share the checkpoint directory
            await pipeline1("input 1")
            await pipeline2("input 2")

        # List checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("*.json"))
        print(f"\nCheckpoint files created: {len(checkpoint_files)}")
        for f in checkpoint_files:
            print(f"  - {f.name}")

    finally:
        import shutil

        shutil.rmtree(checkpoint_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Example 9: Method Chaining with bind()
# ─────────────────────────────────────────────────────────────────────────────


async def example_method_chaining() -> None:
    """Demonstrate method chaining with bind().

    bind() returns self, enabling fluent API patterns.
    """
    print("\n" + "=" * 70)
    print("Example 9: Method Chaining with bind()")
    print("=" * 70)

    mock_resources = MagicMock(name="resources")

    # Create, bind, and call in one expression
    result = await Pipeline().bind(resources=mock_resources)("Hello, chained!")

    print("\nOne-liner: Pipeline().bind(resources=...).(...)")
    print(f"Result: {result}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


async def main() -> None:
    """Run all ExecutionSettings examples."""
    print("=" * 70)
    print("INF-ENGINE: ExecutionSettings and Module Binding Examples")
    print("=" * 70)
    print("""
These examples demonstrate modern patterns for executing inference modules
with clean, PyTorch-like APIs: bind(), ExecutionSettings, and batch execution.
""")

    await example_basic_bind()
    await example_bind_with_config()
    await example_execution_settings()
    await example_configuration_priority()
    await example_batch_execution()
    await example_nested_contexts()
    await example_mixing_patterns()
    await example_shared_checkpointing()
    await example_method_chaining()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("""
1. bind(resources): Attach resources for `await module(input)` pattern
2. ExecutionSettings: Share resources across multiple modules
3. Priority: call-time kwargs > bound settings > context settings
4. Batch: `await module([a, b, c])` returns `[result_a, result_b, result_c]`
5. Nested contexts properly restore outer settings on exit
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
