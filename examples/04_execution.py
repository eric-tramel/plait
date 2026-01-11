#!/usr/bin/env python3
"""Execution: Running pipelines with run(), bind(), and ExecutionSettings.

Demonstrates:
- run() for explicit execution
- bind() for the await module(input) pattern
- ExecutionSettings context for shared resources
- Batch execution with list inputs
- Concurrency control
- Error handling
- Container modules: Sequential, ModuleList, ModuleDict

Note: Uses mock modules to show patterns without API keys.

Run: python examples/04_execution.py
"""

import asyncio
import time
from collections import OrderedDict
from unittest.mock import MagicMock

from plait import ModuleDict, ModuleList, Sequential
from plait.execution.context import ExecutionSettings
from plait.execution.executor import run
from plait.execution.scheduler import Scheduler
from plait.execution.state import ExecutionState
from plait.module import Module
from plait.tracing.tracer import Tracer

# --- Mock Modules (simulate LLM behavior) ---


class TextProcessor(Module):
    """Synchronous text processor."""

    def __init__(self, prefix: str = "", suffix: str = "") -> None:
        super().__init__()
        self.prefix = prefix
        self.suffix = suffix

    def forward(self, text: str) -> str:
        return f"{self.prefix}{text}{self.suffix}"


class AsyncProcessor(Module):
    """Async processor that simulates network latency."""

    def __init__(self, name: str, delay_ms: float = 30) -> None:
        super().__init__()
        self.name = name
        self.delay_ms = delay_ms

    async def forward(self, text: str) -> str:
        await asyncio.sleep(self.delay_ms / 1000)
        return f"[{self.name}:{text}]"


class TextCombiner(Module):
    """Combine multiple text inputs."""

    def __init__(self, sep: str = " + ") -> None:
        super().__init__()
        self.sep = sep

    def forward(self, *args: str) -> str:
        return self.sep.join(str(a) for a in args)


# --- Pipelines ---


class LinearPipeline(Module):
    """Sequential: step1 -> step2 -> step3."""

    def __init__(self) -> None:
        super().__init__()
        self.step1 = TextProcessor(prefix="[1:", suffix="]")
        self.step2 = TextProcessor(prefix="[2:", suffix="]")
        self.step3 = TextProcessor(prefix="[3:", suffix="]")

    def forward(self, text: str) -> str:
        r1 = self.step1(text)
        r2 = self.step2(r1)
        return self.step3(r2)


class ParallelPipeline(Module):
    """Fan-out: three parallel branches."""

    def __init__(self) -> None:
        super().__init__()
        self.a = AsyncProcessor("A")
        self.b = AsyncProcessor("B")
        self.c = AsyncProcessor("C")

    def forward(self, text: str) -> dict[str, str]:
        return {"a": self.a(text), "b": self.b(text), "c": self.c(text)}


class DiamondPipeline(Module):
    """Fan-out + fan-in: branches -> combiner."""

    def __init__(self) -> None:
        super().__init__()
        self.left = TextProcessor(prefix="L:")
        self.right = TextProcessor(prefix="R:")
        self.combine = TextCombiner()

    def forward(self, text: str) -> str:
        left_result = self.left(text)
        right_result = self.right(text)
        return self.combine(left_result, right_result)


# --- Container-Based Pipelines ---


def create_linear_pipeline() -> Sequential:
    """Create a sequential pipeline using the Sequential container.

    This is equivalent to LinearPipeline but using the container.
    """
    return Sequential(
        OrderedDict(
            [
                ("step1", TextProcessor(prefix="[1:", suffix="]")),
                ("step2", TextProcessor(prefix="[2:", suffix="]")),
                ("step3", TextProcessor(prefix="[3:", suffix="]")),
            ]
        )
    )


class ParallelWithDict(Module):
    """Fan-out using ModuleDict for named parallel branches."""

    def __init__(self) -> None:
        super().__init__()
        self.branches = ModuleDict(
            {
                "a": AsyncProcessor("A"),
                "b": AsyncProcessor("B"),
                "c": AsyncProcessor("C"),
            }
        )

    def forward(self, text: str) -> dict[str, str]:
        return {key: self.branches[key](text) for key in self.branches}


class DynamicLayers(Module):
    """Dynamic layers using ModuleList."""

    def __init__(self, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = ModuleList(
            [AsyncProcessor(f"layer{i}", delay_ms=20) for i in range(num_layers)]
        )

    def forward(self, text: str) -> list[str]:
        # Process through all layers in parallel
        return [layer(text) for layer in self.layers]


async def demo_run() -> None:
    """Demonstrate run() for explicit execution."""
    print("\n1. run() - Explicit Execution")
    print("-" * 40)

    pipeline = LinearPipeline()
    result = await run(pipeline, "hello")
    print("   Input: 'hello'")
    print(f"   Output: '{result}'")


async def demo_parallel() -> None:
    """Demonstrate parallel execution."""
    print("\n2. Parallel Execution")
    print("-" * 40)

    pipeline = ParallelPipeline()
    start = time.time()
    results = await run(pipeline, "input")
    elapsed = (time.time() - start) * 1000

    print("   3 parallel branches, each ~30ms")
    print(f"   Elapsed: {elapsed:.0f}ms (parallel < 90ms)")
    print(f"   Results: {results}")


async def demo_bind() -> None:
    """Demonstrate bind() for direct await pattern."""
    print("\n3. bind() - Direct Await Pattern")
    print("-" * 40)

    mock_resources = MagicMock(name="resources")
    pipeline = LinearPipeline().bind(resources=mock_resources)

    # Now call directly with await
    result = await pipeline("world")
    print("   pipeline = LinearPipeline().bind(resources=...)")
    print("   result = await pipeline('world')")
    print(f"   Output: '{result}'")


async def demo_execution_settings() -> None:
    """Demonstrate ExecutionSettings for shared resources."""
    print("\n4. ExecutionSettings - Shared Resources")
    print("-" * 40)

    mock_resources = MagicMock(name="shared")
    pipeline1 = LinearPipeline()
    pipeline2 = DiamondPipeline()

    async with ExecutionSettings(resources=mock_resources):
        result1 = await pipeline1("one")
        result2 = await pipeline2("two")

    print("   async with ExecutionSettings(resources=...):")
    print(f"      pipeline1: '{result1}'")
    print(f"      pipeline2: '{result2}'")


async def demo_batch() -> None:
    """Demonstrate batch execution."""
    print("\n5. Batch Execution")
    print("-" * 40)

    mock_resources = MagicMock(name="resources")
    pipeline = LinearPipeline().bind(resources=mock_resources)

    inputs = ["alpha", "beta", "gamma"]
    results = await pipeline(inputs)

    print(f"   inputs = {inputs}")
    print("   results = await pipeline(inputs)")
    for inp, out in zip(inputs, results, strict=True):
        print(f"      '{inp}' -> '{out}'")


async def demo_concurrency() -> None:
    """Demonstrate concurrency control."""
    print("\n6. Concurrency Control")
    print("-" * 40)

    # Create pipeline with many parallel tasks
    class ManyTasks(Module):
        def __init__(self) -> None:
            super().__init__()
            self.tasks = [AsyncProcessor(f"t{i}", delay_ms=20) for i in range(10)]

        def forward(self, text: str) -> list[str]:
            return [t(text) for t in self.tasks]

    pipeline = ManyTasks()

    # High concurrency
    start = time.time()
    await run(pipeline, "x", max_concurrent=100)
    high = (time.time() - start) * 1000

    # Low concurrency
    start = time.time()
    await run(pipeline, "x", max_concurrent=2)
    low = (time.time() - start) * 1000

    print("   10 tasks x 20ms each")
    print(f"   max_concurrent=100: {high:.0f}ms")
    print(f"   max_concurrent=2:   {low:.0f}ms")


async def demo_state_inspection() -> None:
    """Demonstrate execution state inspection."""
    print("\n7. Execution State Inspection")
    print("-" * 40)

    pipeline = DiamondPipeline()
    tracer = Tracer()
    graph = tracer.trace(pipeline, "test")

    state = ExecutionState(graph)
    print("   Before execution:")
    for node_id, status in state.status.items():
        print(f"      {node_id}: {status.name}")

    scheduler = Scheduler()
    await scheduler.execute(state)

    print("   After execution:")
    for node_id, status in state.status.items():
        print(f"      {node_id}: {status.name}")


async def demo_config_priority() -> None:
    """Demonstrate configuration priority."""
    print("\n8. Configuration Priority")
    print("-" * 40)
    print("   Priority: kwargs > bound > context > defaults")

    mock_res = MagicMock(name="resources")
    pipeline = LinearPipeline().bind(resources=mock_res, max_concurrent=50)

    async with ExecutionSettings(resources=mock_res, max_concurrent=100):
        # kwargs override bound and context
        await pipeline("test", max_concurrent=10)

    print("   context: max_concurrent=100")
    print("   bound:   max_concurrent=50")
    print("   kwargs:  max_concurrent=10")
    print("   -> Uses 10 (kwargs wins)")


async def demo_containers() -> None:
    """Demonstrate container-based pipelines."""
    print("\n9. Container-Based Pipelines")
    print("-" * 40)

    # Sequential container
    print("\n   Sequential container:")
    seq_pipeline = create_linear_pipeline()
    result = await run(seq_pipeline, "hello")
    print(f"   Input: 'hello' -> Output: '{result}'")
    print(
        f"   Access by name: seq_pipeline.step1 = {type(seq_pipeline.step1).__name__}"
    )
    print(f"   Access by index: seq_pipeline[0] = {type(seq_pipeline[0]).__name__}")

    # ModuleDict container
    print("\n   ModuleDict container:")
    parallel = ParallelWithDict()
    start = time.time()
    results = await run(parallel, "input")
    elapsed = (time.time() - start) * 1000
    print(f"   Branches: {list(parallel.branches.keys())}")
    print(f"   Results: {results}")
    print(f"   Elapsed: {elapsed:.0f}ms (parallel)")

    # ModuleList container
    print("\n   ModuleList container:")
    dynamic = DynamicLayers(num_layers=4)
    results = await run(dynamic, "test")
    print(f"   Layers: {len(dynamic.layers)}")
    print(f"   Results: {results}")

    # Add a layer dynamically
    dynamic.layers.append(AsyncProcessor("layer4", delay_ms=20))
    print(f"   After append: {len(dynamic.layers)} layers")


async def main() -> None:
    """Run all execution demos."""
    print("=" * 60)
    print("plait: Execution Examples")
    print("=" * 60)

    await demo_run()
    await demo_parallel()
    await demo_bind()
    await demo_execution_settings()
    await demo_batch()
    await demo_concurrency()
    await demo_state_inspection()
    await demo_config_priority()
    await demo_containers()

    print("\n" + "=" * 60)
    print("Execution Patterns Summary:")
    print("  run(module, input)      - Explicit execution")
    print("  module.bind(resources)  - Enable await module(input)")
    print("  ExecutionSettings(...)  - Share resources across modules")
    print("  await module([a,b,c])   - Batch execution")
    print("  Sequential(...)         - Chain modules together")
    print("  ModuleList([...])       - Dynamic list of modules")
    print("  ModuleDict({...})       - Named module access")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
