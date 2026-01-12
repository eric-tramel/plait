#!/usr/bin/env python3
"""Compare plait vs DSPy performance and output.

This script implements the same extract-and-compare pipeline in both
plait and DSPy, then compares execution time, memory usage, and outputs.

The workflow demonstrates parallel execution:
1. Takes TWO documents as input
2. Extracts main facts from BOTH documents in parallel (fan-out)
3. Runs a compare-and-contrast analysis on the extracted facts

This highlights plait's automatic parallel execution for independent operations.

Run from repository root:

    uv run --with dspy docs/comparison/compare_dspy.py

Environment variables required:
    OPENAI_API_KEY: Your OpenAI API key
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# Sample input documents for comparison
SAMPLE_DOC_1 = """
Electric vehicles (EVs) are revolutionizing the automotive industry. Battery
technology has improved dramatically, with modern lithium-ion batteries offering
ranges of 300+ miles on a single charge. The cost of EVs has decreased
significantly, making them accessible to more consumers. Charging infrastructure
is expanding rapidly, with fast-charging stations appearing along major highways.
Major automakers have committed to transitioning their fleets to electric,
with some planning to phase out internal combustion engines entirely by 2035.
"""

SAMPLE_DOC_2 = """
Hydrogen fuel cell vehicles represent an alternative approach to sustainable
transportation. These vehicles generate electricity through a chemical reaction
between hydrogen and oxygen, producing only water as a byproduct. Refueling
takes just minutes, similar to traditional gasoline vehicles. However, hydrogen
infrastructure remains limited, with fewer than 100 public stations in the US.
Production of green hydrogen is still expensive and energy-intensive. Several
major automakers are investing in fuel cell technology for heavy-duty vehicles
where battery weight would be prohibitive.
"""


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    output: str
    execution_time_ms: float
    peak_memory_mb: float
    error: str | None = None


def measure_execution(
    name: str,
    func: Callable[[], str],
) -> BenchmarkResult:
    """Measure execution time and memory for a synchronous function."""
    gc.collect()
    tracemalloc.start()

    start_time = time.perf_counter()
    error = None
    output = ""

    try:
        output = func()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    end_time = time.perf_counter()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name=name,
        output=output,
        execution_time_ms=(end_time - start_time) * 1000,
        peak_memory_mb=peak_memory / (1024 * 1024),
        error=error,
    )


async def measure_execution_async(
    name: str,
    func: Callable[[], Awaitable[Any]],
) -> BenchmarkResult:
    """Measure execution time and memory for an async function."""
    gc.collect()
    tracemalloc.start()

    start_time = time.perf_counter()
    error = None
    output = ""

    try:
        output = await func()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    end_time = time.perf_counter()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name=name,
        output=str(output),
        execution_time_ms=(end_time - start_time) * 1000,
        peak_memory_mb=peak_memory / (1024 * 1024),
        error=error,
    )


# =============================================================================
# DSPy Implementation
# =============================================================================


def run_dspy(doc1: str, doc2: str) -> str:
    """Run the extract-and-compare pipeline using DSPy.

    Note: DSPy is synchronous by default, so the two extractions run
    sequentially. This demonstrates the performance difference when
    compared to plait's automatic parallel execution.
    """
    import dspy  # type: ignore[import-not-found]

    class ExtractFacts(dspy.Signature):
        """Extract the main facts from the document as a bulleted list."""

        document: str = dspy.InputField()
        facts: str = dspy.OutputField(desc="Bulleted list of main facts")

    class CompareAndContrast(dspy.Signature):
        """Compare and contrast the facts from two documents."""

        facts_doc1: str = dspy.InputField(desc="Facts from first document")
        facts_doc2: str = dspy.InputField(desc="Facts from second document")
        comparison: str = dspy.OutputField(
            desc="Compare and contrast analysis highlighting similarities and differences"
        )

    class ExtractAndCompare(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fast_lm = dspy.LM("openai/gpt-4o-mini")
            self.smart_lm = dspy.LM("openai/gpt-4o")
            self.extract = dspy.Predict(ExtractFacts)
            self.compare = dspy.Predict(CompareAndContrast)

        def forward(self, doc1: str, doc2: str) -> str:
            # Extract facts from both documents
            # Note: These run SEQUENTIALLY in DSPy (no automatic parallelism)
            with dspy.context(lm=self.fast_lm):
                facts1 = self.extract(document=doc1).facts
                facts2 = self.extract(document=doc2).facts

            # Compare and contrast
            with dspy.context(lm=self.smart_lm):
                comparison = self.compare(
                    facts_doc1=facts1, facts_doc2=facts2
                ).comparison
            return comparison

    pipeline = ExtractAndCompare()
    return pipeline(doc1=doc1, doc2=doc2)


# =============================================================================
# plait Implementation
# =============================================================================


async def run_plait(doc1: str, doc2: str) -> str:
    """Run the extract-and-compare pipeline using plait.

    The two fact extractions are independent operations that plait
    automatically executes in parallel, reducing total execution time.
    """
    from plait import LLMInference, Module, Parameter
    from plait.resources import OpenAIEndpointConfig, ResourceConfig

    class FactsCombiner(Module):
        """Combine two facts into a comparison prompt.

        This module formats the facts from two documents into a single
        prompt for comparison. Using a module ensures proper tracing
        during the forward pass (Proxy objects are resolved correctly).
        """

        def forward(self, facts1: str, facts2: str) -> str:
            return (
                f"Compare and contrast these facts:\n\n"
                f"Document 1 Facts:\n{facts1}\n\n"
                f"Document 2 Facts:\n{facts2}"
            )

    class ExtractAndCompare(Module):
        def __init__(self) -> None:
            super().__init__()
            self.comparison_style = Parameter(
                value="Highlight key similarities and differences. Be thorough but concise.",
                description="Controls the style of comparison output.",
            )
            self.extractor = LLMInference(
                alias="fast",
                system_prompt="Extract the main facts from the document as a bulleted list.",
            )
            self.combiner = FactsCombiner()
            self.comparer = LLMInference(
                alias="smart",
                system_prompt=self.comparison_style,
            )

        def forward(self, doc1: str, doc2: str) -> str:
            # These two calls are INDEPENDENT - plait runs them in PARALLEL
            facts1 = self.extractor(doc1)
            facts2 = self.extractor(doc2)

            # Combine facts using the combiner module (resolves Proxy objects)
            combined = self.combiner(facts1, facts2)

            # This depends on both facts, so it waits for both to complete
            return self.comparer(combined)

    resources = ResourceConfig(
        endpoints={
            "fast": OpenAIEndpointConfig(
                model="gpt-4o-mini",
                max_concurrent=20,
            ),
            "smart": OpenAIEndpointConfig(
                model="gpt-4o",
                max_concurrent=5,
            ),
        }
    )

    pipeline = ExtractAndCompare().bind(resources=resources)
    result = await pipeline(doc1, doc2)
    # Extract payload from Value object
    return str(result.payload if hasattr(result, "payload") else result)


# =============================================================================
# Comparison Report
# =============================================================================


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print a formatted comparison of benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON: plait vs DSPy")
    print("=" * 70)

    # Performance table
    print("\n## Performance Metrics\n")
    print(f"{'Framework':<20} {'Time (ms)':<15} {'Peak Memory (MB)':<20}")
    print("-" * 55)

    for result in results:
        if result.error:
            print(f"{result.name:<20} {'ERROR':<15} {'N/A':<20}")
        else:
            print(
                f"{result.name:<20} "
                f"{result.execution_time_ms:<15.2f} "
                f"{result.peak_memory_mb:<20.2f}"
            )

    # Calculate differences if both succeeded
    successful = [r for r in results if not r.error]
    if len(successful) == 2:
        time_diff = successful[1].execution_time_ms - successful[0].execution_time_ms
        time_pct = (time_diff / successful[0].execution_time_ms) * 100

        print(f"\nTime difference: {time_diff:+.2f} ms ({time_pct:+.1f}%)")
        print(
            "\nNote: plait runs the two fact extractions in PARALLEL,"
            "\nwhile DSPy runs them SEQUENTIALLY. This accounts for"
            "\nthe performance difference in this fan-out workflow."
        )

    # Outputs
    print("\n## Outputs\n")
    for result in results:
        print(f"### {result.name}\n")
        if result.error:
            print(f"ERROR: {result.error}\n")
        else:
            print(f"{result.output}\n")

    # Note about differences
    print("\n## Notes\n")
    print("- This workflow demonstrates PARALLEL EXECUTION:")
    print("  - plait: Automatically runs independent operations in parallel")
    print("  - DSPy: Runs operations sequentially (sync-first design)")
    print("- plait uses gpt-4o-mini for extraction and gpt-4o for comparison")
    print("- DSPy uses dspy.context() to switch models per call")

    print("\n" + "=" * 70)


# =============================================================================
# Main
# =============================================================================


async def main() -> int:
    """Run the comparison benchmark."""
    parser = argparse.ArgumentParser(description="Compare plait vs DSPy performance")
    parser.add_argument(
        "--doc1",
        type=str,
        help="Path to first document (uses sample if not provided)",
    )
    parser.add_argument(
        "--doc2",
        type=str,
        help="Path to second document (uses sample if not provided)",
    )
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        return 1

    # Get input documents
    if args.doc1:
        with open(args.doc1) as f:
            doc1 = f.read()
        print(f"Document 1 from: {args.doc1}")
    else:
        doc1 = SAMPLE_DOC_1
        print("Document 1: Sample (Electric Vehicles)")

    if args.doc2:
        with open(args.doc2) as f:
            doc2 = f.read()
        print(f"Document 2 from: {args.doc2}")
    else:
        doc2 = SAMPLE_DOC_2
        print("Document 2: Sample (Hydrogen Fuel Cells)")

    print(f"Document 1 length: {len(doc1)} characters")
    print(f"Document 2 length: {len(doc2)} characters")

    results: list[BenchmarkResult] = []

    # Run DSPy
    print("\nRunning DSPy (sequential extraction)...")
    result = measure_execution("DSPy", lambda: run_dspy(doc1, doc2))
    results.append(result)

    # Run plait
    print("Running plait (parallel extraction)...")
    result = await measure_execution_async("plait", lambda: run_plait(doc1, doc2))
    results.append(result)

    # Print comparison
    print_comparison(results)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
