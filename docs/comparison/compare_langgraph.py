#!/usr/bin/env python3
"""Compare plait vs LangGraph performance and output.

This script implements the same summarize-and-analyze pipeline in both
plait and LangGraph, then compares execution time, memory usage, and outputs.

Run from repository root:

    uv run --with langgraph --with langchain-openai docs/comparison/compare_langgraph.py

Or with a specific input file:

    uv run --with langgraph --with langchain-openai docs/comparison/compare_langgraph.py --input myfile.txt

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

# Sample input text for comparison
SAMPLE_TEXT = """
Artificial intelligence has transformed numerous industries over the past decade.
Machine learning algorithms now power recommendation systems, autonomous vehicles,
and medical diagnosis tools. The emergence of large language models has further
accelerated this transformation, enabling new applications in content generation,
code assistance, and conversational interfaces.

However, these advances come with significant challenges. Concerns about bias in
AI systems, the environmental impact of training large models, and the potential
for job displacement have sparked important debates. Researchers and policymakers
are working to address these issues while continuing to push the boundaries of
what AI can achieve.

The future of AI likely involves more efficient training methods, better
interpretability of model decisions, and stronger safeguards against misuse.
As these technologies mature, their integration into daily life will only deepen,
making it essential to develop frameworks for responsible AI development and
deployment.
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
# LangGraph Implementation
# =============================================================================


async def run_langgraph(text: str) -> str:
    """Run the summarize-and-analyze pipeline using LangGraph."""
    from typing import TypedDict

    from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
    from langgraph.graph import END, StateGraph  # type: ignore[import-not-found]

    class State(TypedDict):
        text: str
        summary: str
        analysis: str

    INSTRUCTIONS = "Be concise and highlight key insights."

    async def summarize(state: State) -> dict[str, str]:
        llm = ChatOpenAI(model="gpt-4o-mini")
        result = await llm.ainvoke(
            f"Summarize the input text concisely.\n\n{state['text']}"
        )
        return {"summary": str(result.content)}

    async def analyze(state: State) -> dict[str, str]:
        llm = ChatOpenAI(model="gpt-4o")
        result = await llm.ainvoke(
            f"{INSTRUCTIONS}\n\nAnalyze this summary:\n{state['summary']}"
        )
        return {"analysis": str(result.content)}

    # Build graph
    graph = StateGraph(State)
    graph.add_node("summarize", summarize)
    graph.add_node("analyze", analyze)
    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "analyze")
    graph.add_edge("analyze", END)

    app = graph.compile()
    result = await app.ainvoke({"text": text, "summary": "", "analysis": ""})
    return str(result["analysis"])


# =============================================================================
# plait Implementation
# =============================================================================


async def run_plait(text: str) -> str:
    """Run the summarize-and-analyze pipeline using plait."""
    from plait import LLMInference, Module, Parameter
    from plait.resources import OpenAIEndpointConfig, ResourceConfig

    class SummarizeAndAnalyze(Module):
        def __init__(self) -> None:
            super().__init__()
            self.instructions = Parameter(
                value="Be concise and highlight key insights.",
                description="Controls the style of analysis output.",
            )
            self.summarizer = LLMInference(
                alias="fast",
                system_prompt="Summarize the input text concisely.",
            )
            self.analyzer = LLMInference(
                alias="smart",
                system_prompt=self.instructions,
            )

        def forward(self, text: str) -> str:
            summary = self.summarizer(text)
            return self.analyzer(f"Analyze this summary:\n{summary}")

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

    pipeline = SummarizeAndAnalyze().bind(resources=resources)
    result = await pipeline(text)
    return str(result)


# =============================================================================
# Comparison Report
# =============================================================================


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print a formatted comparison of benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON: plait vs LangGraph")
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

    # Outputs
    print("\n## Outputs\n")
    for result in results:
        print(f"### {result.name}\n")
        if result.error:
            print(f"ERROR: {result.error}\n")
        else:
            print(f"{result.output}\n")

    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


async def main() -> int:
    """Run the comparison benchmark."""
    parser = argparse.ArgumentParser(
        description="Compare plait vs LangGraph performance"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to input text file (uses sample text if not provided)",
    )
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        return 1

    # Get input text
    if args.input:
        with open(args.input) as f:
            text = f.read()
        print(f"Using input from: {args.input}")
    else:
        text = SAMPLE_TEXT
        print("Using sample text")

    print(f"Input length: {len(text)} characters")

    results: list[BenchmarkResult] = []

    # Run LangGraph
    print("\nRunning LangGraph...")
    result = await measure_execution_async("LangGraph", lambda: run_langgraph(text))
    results.append(result)

    # Run plait
    print("Running plait...")
    result = await measure_execution_async("plait", lambda: run_plait(text))
    results.append(result)

    # Print comparison
    print_comparison(results)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
