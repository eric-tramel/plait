#!/usr/bin/env python3
"""Execution examples demonstrating the run() function and bound execution.

This example demonstrates how to execute inference modules using:

1. **run() function**: Explicit tracing and execution with full control
2. **bind() method**: Attach resources for `await module(input)` pattern
3. **ExecutionSettings**: Share resources across multiple modules

Note: Since we don't have actual LLM endpoints configured, these examples
use mock modules that simulate computation. The API is identical - once
resources are configured, the same patterns work with real LLM calls.

Bound execution pattern (preferred for production):

    # Bind resources once, call directly
    pipeline = MyPipeline().bind(resources=config)
    result = await pipeline("input")

    # Batch execution
    results = await pipeline(["input_a", "input_b", "input_c"])

    # Or use ExecutionSettings for shared resources
    async with ExecutionSettings(resources=config):
        result1 = await pipeline1("input")
        result2 = await pipeline2("input")

This mirrors PyTorch's intuitive model(x) -> y pattern.

Run with: python examples/05_execution.py
"""

import asyncio
import time
from unittest.mock import MagicMock

from inf_engine.execution.context import ExecutionSettings
from inf_engine.execution.executor import run
from inf_engine.execution.scheduler import Scheduler
from inf_engine.execution.state import ExecutionState, TaskResult
from inf_engine.module import InferenceModule
from inf_engine.tracing.tracer import Tracer

# ─────────────────────────────────────────────────────────────────────────────
# Mock Modules for Examples
# These simulate LLM-like operations without needing actual API calls
# ─────────────────────────────────────────────────────────────────────────────


class TextProcessor(InferenceModule):
    """Mock module that processes text with a transformation."""

    def __init__(self, prefix: str = "", suffix: str = "") -> None:
        super().__init__()
        self.prefix = prefix
        self.suffix = suffix

    def forward(self, text: str) -> str:
        return f"{self.prefix}{text}{self.suffix}"


class TextCombiner(InferenceModule):
    """Mock module that combines multiple text inputs."""

    def __init__(self, separator: str = " | ") -> None:
        super().__init__()
        self.separator = separator

    def forward(self, *args: str) -> str:
        return self.separator.join(str(arg) for arg in args)


class AsyncTextProcessor(InferenceModule):
    """Mock async module that simulates network latency."""

    def __init__(self, operation: str, delay_ms: float = 50) -> None:
        super().__init__()
        self.operation = operation
        self.delay_ms = delay_ms

    async def forward(self, text: str) -> str:
        await asyncio.sleep(self.delay_ms / 1000)
        return f"[{self.operation}({text})]"


class FailingProcessor(InferenceModule):
    """Mock module that always fails (for error handling demos)."""

    def __init__(self, error_message: str = "Processing failed") -> None:
        super().__init__()
        self.error_message = error_message

    def forward(self, text: str) -> str:
        raise RuntimeError(self.error_message)


# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Simple Execution
# ─────────────────────────────────────────────────────────────────────────────


class SimplePipeline(InferenceModule):
    """A simple two-step pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.step1 = TextProcessor(prefix="[STEP1: ", suffix="]")
        self.step2 = TextProcessor(prefix="[STEP2: ", suffix="]")

    def forward(self, text: str) -> str:
        result1 = self.step1(text)
        result2 = self.step2(result1)
        return result2


async def demo_simple_execution() -> None:
    """Demonstrate basic execution with run()."""
    print("1. Simple Execution")
    print("-" * 40)

    pipeline = SimplePipeline()

    # Execute the pipeline
    result = await run(pipeline, "Hello, World!")

    print("   Input: 'Hello, World!'")
    print(f"   Output: '{result}'")
    print("   (Data flowed through two processing steps)")


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Linear Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class LinearChain(InferenceModule):
    """A linear chain of processing steps."""

    def __init__(self) -> None:
        super().__init__()
        self.extract = TextProcessor(prefix="EXTRACTED(", suffix=")")
        self.analyze = TextProcessor(prefix="ANALYZED(", suffix=")")
        self.summarize = TextProcessor(prefix="SUMMARY(", suffix=")")

    def forward(self, document: str) -> str:
        # Each step depends on the previous
        extracted = self.extract(document)
        analyzed = self.analyze(extracted)
        summary = self.summarize(analyzed)
        return summary


async def demo_linear_execution() -> None:
    """Demonstrate linear pipeline execution."""
    print("\n2. Linear Pipeline Execution")
    print("-" * 40)

    pipeline = LinearChain()
    result = await run(pipeline, "Important document content")

    print("   Input: 'Important document content'")
    print(f"   Output: '{result}'")
    print("   (extract -> analyze -> summarize)")


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Parallel Execution (Fan-out)
# ─────────────────────────────────────────────────────────────────────────────


class ParallelAnalysis(InferenceModule):
    """Multiple analyzers processing input in parallel."""

    def __init__(self) -> None:
        super().__init__()
        self.technical = AsyncTextProcessor("TECHNICAL", delay_ms=50)
        self.business = AsyncTextProcessor("BUSINESS", delay_ms=50)
        self.user_exp = AsyncTextProcessor("UX", delay_ms=50)

    def forward(self, text: str) -> dict[str, str]:
        # All three run from the same input - can execute concurrently
        return {
            "technical": self.technical(text),
            "business": self.business(text),
            "user_exp": self.user_exp(text),
        }


async def demo_parallel_execution() -> None:
    """Demonstrate parallel (fan-out) execution."""
    print("\n3. Parallel Execution (Fan-out)")
    print("-" * 40)

    pipeline = ParallelAnalysis()

    start_time = time.time()
    results = await run(pipeline, "Product requirements")
    elapsed_ms = (time.time() - start_time) * 1000

    print("   Input: 'Product requirements'")
    print("   Outputs (3 parallel branches):")
    for node_id, value in results.items():
        print(f"      {node_id}: {value}")
    print(f"\n   Elapsed time: {elapsed_ms:.1f}ms")
    print("   (If sequential: ~150ms, parallel: ~50-80ms)")


# ─────────────────────────────────────────────────────────────────────────────
# Example 4: Diamond Pattern (Fan-out + Fan-in)
# ─────────────────────────────────────────────────────────────────────────────


class DiamondPipeline(InferenceModule):
    """Fan-out to parallel processors, then fan-in to combiner."""

    def __init__(self) -> None:
        super().__init__()
        self.branch_a = TextProcessor(prefix="[A: ", suffix="]")
        self.branch_b = TextProcessor(prefix="[B: ", suffix="]")
        self.combiner = TextCombiner(separator=" + ")

    def forward(self, text: str) -> str:
        # Fan-out: both branches process the same input
        result_a = self.branch_a(text)
        result_b = self.branch_b(text)
        # Fan-in: combiner waits for both branches
        return self.combiner(result_a, result_b)


async def demo_diamond_execution() -> None:
    """Demonstrate diamond pattern execution."""
    print("\n4. Diamond Pattern Execution")
    print("-" * 40)

    pipeline = DiamondPipeline()
    result = await run(pipeline, "input")

    print("   Input: 'input'")
    print(f"   Output: '{result}'")
    print("\n   Execution flow:")
    print("      input")
    print("        |")
    print("    +---+---+")
    print("    |       |")
    print("   [A]     [B]   <- parallel")
    print("    |       |")
    print("    +---+---+")
    print("        |")
    print("    combiner     <- waits for both")


# ─────────────────────────────────────────────────────────────────────────────
# Example 5: Concurrency Control
# ─────────────────────────────────────────────────────────────────────────────


class ManyParallelTasks(InferenceModule):
    """Many tasks that can run in parallel."""

    def __init__(self, n_tasks: int = 10) -> None:
        super().__init__()
        self.processors = [
            AsyncTextProcessor(f"task_{i}", delay_ms=30) for i in range(n_tasks)
        ]

    def forward(self, text: str) -> list[str]:
        return [proc(text) for proc in self.processors]


async def demo_concurrency_control() -> None:
    """Demonstrate concurrency limits with max_concurrent."""
    print("\n5. Concurrency Control")
    print("-" * 40)

    pipeline = ManyParallelTasks(n_tasks=10)

    # Test with different concurrency limits
    print("   10 tasks, each taking ~30ms:")

    # High concurrency
    start = time.time()
    await run(pipeline, "test", max_concurrent=100)
    high_elapsed = (time.time() - start) * 1000
    print(f"      max_concurrent=100: {high_elapsed:.0f}ms")

    # Limited concurrency
    start = time.time()
    await run(pipeline, "test", max_concurrent=3)
    low_elapsed = (time.time() - start) * 1000
    print(f"      max_concurrent=3:   {low_elapsed:.0f}ms")

    print("\n   Lower concurrency = longer time but less resource pressure")


# ─────────────────────────────────────────────────────────────────────────────
# Example 6: Execution State Inspection
# ─────────────────────────────────────────────────────────────────────────────


async def demo_execution_state() -> None:
    """Demonstrate execution state inspection."""
    print("\n6. Execution State Inspection")
    print("-" * 40)

    # Create a pipeline and trace it
    pipeline = DiamondPipeline()
    tracer = Tracer()
    graph = tracer.trace(pipeline, "test input")

    # Create execution state
    state = ExecutionState(graph)

    print("   Before execution:")
    for node_id, status in state.status.items():
        print(f"      {node_id}: {status.name}")

    # Execute with the scheduler
    scheduler = Scheduler(max_concurrent=10)

    completed_order: list[str] = []

    def on_complete(node_id: str, result: TaskResult) -> None:
        completed_order.append(node_id)

    await scheduler.execute(state, on_complete=on_complete)

    print("\n   After execution:")
    for node_id, status in state.status.items():
        print(f"      {node_id}: {status.name}")

    print(f"\n   Completion order: {completed_order}")

    # Show results
    outputs = state.get_outputs()
    print(f"\n   Final output: '{list(outputs.values())[0]}'")


# ─────────────────────────────────────────────────────────────────────────────
# Example 7: Error Handling
# ─────────────────────────────────────────────────────────────────────────────


class PipelineWithFailure(InferenceModule):
    """Pipeline where a middle step fails."""

    def __init__(self) -> None:
        super().__init__()
        self.step1 = TextProcessor(prefix="[OK1: ", suffix="]")
        self.step2 = FailingProcessor("Simulated API error")
        self.step3 = TextProcessor(prefix="[OK3: ", suffix="]")  # Never reached

    def forward(self, text: str) -> str:
        r1 = self.step1(text)
        r2 = self.step2(r1)
        r3 = self.step3(r2)
        return r3


async def demo_error_handling() -> None:
    """Demonstrate error handling in execution."""
    print("\n7. Error Handling")
    print("-" * 40)

    pipeline = PipelineWithFailure()

    # Trace and create state manually to inspect errors
    tracer = Tracer()
    graph = tracer.trace(pipeline, "input")
    state = ExecutionState(graph)
    scheduler = Scheduler()

    await scheduler.execute(state)

    print("   Pipeline: step1 -> step2 (fails) -> step3")
    print("\n   Task statuses:")
    for node_id, status in state.status.items():
        print(f"      {node_id}: {status.name}")

    print("\n   Errors:")
    for node_id, error in state.errors.items():
        print(f"      {node_id}: {error}")

    print("\n   Note: step3 was CANCELLED because step2 failed")


# ─────────────────────────────────────────────────────────────────────────────
# Example 8: Multiple Inputs
# ─────────────────────────────────────────────────────────────────────────────


class MultiInputPipeline(InferenceModule):
    """Pipeline that takes multiple inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.process_query = TextProcessor(prefix="Q: ")
        self.process_context = TextProcessor(prefix="C: ")
        self.combine = TextCombiner(separator=" | ")

    def forward(self, query: str, context: str) -> str:
        processed_query = self.process_query(query)
        processed_context = self.process_context(context)
        return self.combine(processed_query, processed_context)


async def demo_multiple_inputs() -> None:
    """Demonstrate execution with multiple inputs."""
    print("\n8. Multiple Inputs")
    print("-" * 40)

    pipeline = MultiInputPipeline()

    # Pass multiple arguments to run()
    result = await run(pipeline, "What is X?", "X is a concept")

    print("   Inputs:")
    print("      query: 'What is X?'")
    print("      context: 'X is a concept'")
    print(f"   Output: '{result}'")

    # Can also use keyword arguments
    result_kwargs = await run(
        pipeline, query="How does Y work?", context="Y is a mechanism"
    )
    print("\n   With kwargs:")
    print(f"      '{result_kwargs}'")


# ─────────────────────────────────────────────────────────────────────────────
# Example 9: Complex Multi-Stage Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class ComplexPipeline(InferenceModule):
    """Complex pipeline: preprocess -> [analyze_a, analyze_b] -> synthesize.

    This demonstrates combining linear, parallel, and fan-in patterns.
    """

    def __init__(self) -> None:
        super().__init__()
        self.preprocess = TextProcessor(prefix="CLEANED(", suffix=")")
        self.analyze_a = AsyncTextProcessor("ASPECT_A", delay_ms=40)
        self.analyze_b = AsyncTextProcessor("ASPECT_B", delay_ms=40)
        self.synthesize = TextCombiner(separator=" + ")

    def forward(self, document: str) -> str:
        # Stage 1: Preprocess (linear)
        cleaned = self.preprocess(document)
        # Stage 2: Parallel analysis (fan-out from cleaned)
        result_a = self.analyze_a(cleaned)
        result_b = self.analyze_b(cleaned)
        # Stage 3: Synthesize (fan-in)
        return self.synthesize(result_a, result_b)


async def demo_complex_pipeline() -> None:
    """Demonstrate complex multi-stage pipeline execution."""
    print("\n9. Complex Multi-Stage Pipeline")
    print("-" * 40)

    pipeline = ComplexPipeline()

    start = time.time()
    result = await run(pipeline, "Raw document text")
    elapsed = (time.time() - start) * 1000

    print("   Input: 'Raw document text'")
    print(f"   Output: '{result}'")
    print(f"   Elapsed: {elapsed:.0f}ms")
    print("\n   Structure:")
    print("      input -> preprocess -> [analyze_a, analyze_b] -> synthesize")
    print("      (analyze_a and analyze_b run in parallel)")


# ─────────────────────────────────────────────────────────────────────────────
# Example 10: Bound Execution with bind()
# ─────────────────────────────────────────────────────────────────────────────


async def demo_bound_execution() -> None:
    """Demonstrate bound execution pattern with bind().

    The bind() method attaches resources to a module, enabling the
    clean `await module(input)` pattern instead of `await run(module, input)`.
    """
    print("\n10. Bound Execution with bind()")
    print("-" * 40)

    # Mock resources (in production, this would be ResourceConfig)
    mock_resources = MagicMock(name="resources")

    # Bind resources to the pipeline
    pipeline = SimplePipeline().bind(resources=mock_resources)

    print("   Pattern: pipeline.bind(resources=...) then await pipeline(input)")

    # Now call directly with await
    result = await pipeline("Hello, bound world!")

    print("   Input: 'Hello, bound world!'")
    print(f"   Output: '{result}'")
    print("\n   This is the preferred pattern for production code!")


# ─────────────────────────────────────────────────────────────────────────────
# Example 11: ExecutionSettings Context
# ─────────────────────────────────────────────────────────────────────────────


async def demo_execution_settings() -> None:
    """Demonstrate ExecutionSettings for shared resources.

    ExecutionSettings provides a context manager for sharing resources
    across multiple module executions without binding each one.
    """
    print("\n11. ExecutionSettings Context")
    print("-" * 40)

    mock_resources = MagicMock(name="shared_resources")

    # Create multiple unbound modules
    pipeline1 = SimplePipeline()
    pipeline2 = LinearChain()

    print("   Using ExecutionSettings to share resources across modules:")

    async with ExecutionSettings(resources=mock_resources, max_concurrent=50):
        # Both modules can be called with await
        result1 = await pipeline1("Input for pipeline 1")
        result2 = await pipeline2("Input for pipeline 2")

        print(f"   Pipeline 1: '{result1}'")
        print(f"   Pipeline 2: '{result2}'")

    print("\n   Both modules shared the same resources and settings!")


# ─────────────────────────────────────────────────────────────────────────────
# Example 12: Batch Execution
# ─────────────────────────────────────────────────────────────────────────────


async def demo_batch_execution() -> None:
    """Demonstrate batch execution with list inputs.

    When you pass a list as the first argument to a bound module,
    it processes each item and returns a list of results.
    """
    print("\n12. Batch Execution")
    print("-" * 40)

    mock_resources = MagicMock(name="resources")
    pipeline = SimplePipeline().bind(resources=mock_resources)

    # Batch inputs
    inputs = ["alpha", "beta", "gamma"]

    print(f"   Batch inputs: {inputs}")

    start = time.time()
    results = await pipeline(inputs)
    elapsed = (time.time() - start) * 1000

    print(f"   Batch results ({elapsed:.0f}ms):")
    for inp, res in zip(inputs, results, strict=True):
        print(f"      '{inp}' -> '{res}'")

    print("\n   Batch processing: one call, multiple results!")


# ─────────────────────────────────────────────────────────────────────────────
# Run all demos
# ─────────────────────────────────────────────────────────────────────────────


async def main() -> None:
    """Run all execution demos."""
    print("=" * 60)
    print("inf-engine: Execution Examples")
    print("=" * 60)

    # Basic run() examples
    await demo_simple_execution()
    await demo_linear_execution()
    await demo_parallel_execution()
    await demo_diamond_execution()
    await demo_concurrency_control()
    await demo_execution_state()
    await demo_error_handling()
    await demo_multiple_inputs()
    await demo_complex_pipeline()

    # Modern bound execution patterns
    await demo_bound_execution()
    await demo_execution_settings()
    await demo_batch_execution()

    print("\n" + "=" * 60)
    print("Execution Patterns Summary:")
    print("=" * 60)
    print("""
  1. run(module, input)     - Explicit execution with full control
  2. module.bind(resources) - Attach resources, then `await module(input)`
  3. ExecutionSettings()    - Share resources across multiple modules
  4. await module([...])    - Batch execution with list inputs

The bind() and ExecutionSettings patterns are preferred for production!
""")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
