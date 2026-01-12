#!/usr/bin/env python3
"""Compare plait vs Pydantic AI performance and output.

This script implements the same extract-and-compare pipeline in both
plait and Pydantic AI, then compares execution time, memory usage, and outputs.

The workflow demonstrates parallel execution:
1. Takes TWO documents as input
2. Extracts main facts from BOTH documents in parallel (fan-out)
3. Runs a compare-and-contrast analysis on the extracted facts

This highlights plait's automatic parallel execution vs Pydantic AI's
manual asyncio.gather() approach.

Run from repository root:

    uv run --with pydantic-ai docs/comparison/compare_pydantic_ai.py

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
# Pydantic AI Implementation
# =============================================================================


async def run_pydantic_ai(doc1: str, doc2: str) -> str:
    """Run the extract-and-compare pipeline using Pydantic AI.

    Pydantic AI requires manual asyncio.gather() to achieve parallel execution.
    The user must explicitly manage concurrency.
    """
    from pydantic_ai import Agent  # type: ignore[import-not-found]

    extractor = Agent(
        "openai:gpt-4o-mini",
        system_prompt="Extract the main facts from the document as a bulleted list.",
    )

    comparer = Agent(
        "openai:gpt-4o",
        system_prompt="Highlight key similarities and differences. Be thorough but concise.",
    )

    # Must use asyncio.gather() explicitly for parallel execution
    facts1_result, facts2_result = await asyncio.gather(
        extractor.run(doc1),
        extractor.run(doc2),
    )

    comparison_result = await comparer.run(
        f"Compare and contrast these facts:\n\n"
        f"Document 1 Facts:\n{facts1_result.output}\n\n"
        f"Document 2 Facts:\n{facts2_result.output}"
    )
    return str(comparison_result.output)


# =============================================================================
# plait Implementation
# =============================================================================


async def run_plait(doc1: str, doc2: str) -> str:
    """Run the extract-and-compare pipeline using plait.

    The two fact extractions are independent operations that plait
    automatically executes in parallel, reducing total execution time.
    No explicit asyncio.gather() needed.
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
            # No asyncio.gather() needed!
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
    print("BENCHMARK COMPARISON: plait vs Pydantic AI")
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
            "\nNote: Both frameworks can run extractions in parallel,"
            "\nbut plait does this AUTOMATICALLY from data dependencies,"
            "\nwhile Pydantic AI requires MANUAL asyncio.gather()."
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
    print("- This workflow demonstrates PARALLEL EXECUTION approaches:")
    print("  - plait: Automatic parallelism from data dependencies (no boilerplate)")
    print("  - Pydantic AI: Manual asyncio.gather() required for concurrency")
    print("- plait uses gpt-4o-mini for extraction and gpt-4o for comparison")
    print("- Both frameworks support async execution natively")

    print("\n" + "=" * 70)


# =============================================================================
# Main
# =============================================================================


async def main() -> int:
    """Run the comparison benchmark."""
    parser = argparse.ArgumentParser(
        description="Compare plait vs Pydantic AI performance"
    )
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

    # Run Pydantic AI
    print("\nRunning Pydantic AI (manual asyncio.gather())...")
    result = await measure_execution_async(
        "Pydantic AI", lambda: run_pydantic_ai(doc1, doc2)
    )
    results.append(result)

    # Run plait
    print("Running plait (automatic parallel extraction)...")
    result = await measure_execution_async("plait", lambda: run_plait(doc1, doc2))
    results.append(result)

    # Print comparison
    print_comparison(results)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
