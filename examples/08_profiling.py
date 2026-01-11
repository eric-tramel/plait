#!/usr/bin/env python3
"""Example 08: Execution Profiling with Real LLM Calls.

This example demonstrates profiling infrastructure with a comprehensive DAG
workflow that showcases:

- Batch processing with parallel execution
- Sequential pipeline stages with dependencies
- Multiple LLM endpoints (fast and smart models)
- Rate limiting behavior under load
- Chrome Trace Format output for Perfetto visualization

The pipeline analyzes a batch of tech articles through multiple stages:
1. Summarization (fast model)
2. Theme extraction (fast model)
3. Deep analysis combining summary + themes (smart model)

Prerequisites:
- Set OPENAI_API_KEY environment variable
- pip install openai (included in plait dependencies)

Run with: python examples/08_profiling.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from plait.execution.context import ExecutionSettings
from plait.module import LLMInference, Module
from plait.resources.config import EndpointConfig, ResourceConfig

# Single trace output location
TRACE_PATH = Path("traces/profiling_example.json")
TRACE_PATH.parent.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Resource Configuration
# ─────────────────────────────────────────────────────────────────────────────


def create_resources() -> ResourceConfig:
    """Create resource configuration with gpt-4o and gpt-4o-mini endpoints."""
    return ResourceConfig(
        endpoints={
            "fast": EndpointConfig(
                provider_api="openai",
                model="gpt-4o-mini",
                rate_limit=500,  # 500 RPM
                max_concurrent=64,
            ),
            "smart": EndpointConfig(
                provider_api="openai",
                model="gpt-4o",
                rate_limit=500,  # 500 RPM
                max_concurrent=64,
            ),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Stage Document Analysis Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TextCombiner(Module):
    """Combine summary and themes into a formatted prompt.

    This is needed because string formatting during tracing would convert
    proxy objects to "Proxy(...)" strings. By using a module, the actual
    string formatting happens at execution time.
    """

    def forward(self, summary: str, themes: str) -> str:
        return f"Summary: {summary}\n\nThemes: {themes}"


class DocumentAnalyzer(Module):
    """Analyze a single document through multiple stages.

    This pipeline demonstrates a complex DAG structure for a single document:
    - Stage 1 (Summarize): Condense the document (fast model)
    - Stage 2 (Themes): Extract key themes from summary (fast model)
    - Stage 3 (Combine): Format summary + themes for analysis
    - Stage 4 (Analysis): Deep analysis of combined content (smart model)

    When called with a batch of documents, each document is processed
    independently, enabling parallel execution across the batch.
    """

    def __init__(self) -> None:
        super().__init__()
        # Stage 1: Summarize
        self.summarizer = LLMInference(
            alias="fast",
            system_prompt=(
                "Summarize the following text in 2-3 sentences. "
                "Focus on the main point and key details."
            ),
            max_tokens=150,
        )
        # Stage 2: Extract themes
        self.theme_extractor = LLMInference(
            alias="fast",
            system_prompt=(
                "Extract 3-5 key themes from this summary as a comma-separated list. "
                "Be concise - just the theme names, no explanations."
            ),
            max_tokens=100,
        )
        # Stage 3: Combine summary and themes
        self.combiner = TextCombiner()
        # Stage 4: Deep analysis (depends on combined summary + themes)
        self.analyzer = LLMInference(
            alias="smart",
            system_prompt=(
                "You are given a summary and extracted themes from a document. "
                "Provide insights about:\n"
                "1. The significance of this topic\n"
                "2. Potential implications\n"
                "3. Related considerations\n"
                "Be concise but insightful (3-4 sentences)."
            ),
            max_tokens=200,
        )

    def forward(self, document: str) -> dict[str, str]:
        # Stage 1: Summarize the document
        summary = self.summarizer(document)

        # Stage 2: Extract themes (depends on summary)
        themes = self.theme_extractor(summary)

        # Stage 3: Combine for analysis (happens at execution time, not tracing)
        combined = self.combiner(summary, themes)

        # Stage 4: Deep analysis (depends on combined)
        analysis = self.analyzer(combined)

        return {
            "summary": summary,
            "themes": themes,
            "analysis": analysis,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sample Data
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_ARTICLES = [
    """
    Artificial intelligence is transforming how we work and live. Machine learning
    models can now generate text, images, and code with remarkable quality. This has
    profound implications for education, employment, and creativity. Companies are
    racing to deploy AI solutions while researchers debate safety and alignment.
    The technology continues to advance rapidly, with new capabilities emerging
    every few months.
    """,
    """
    Cloud computing has revolutionized infrastructure management. Organizations can
    now scale resources on-demand, paying only for what they use. This shift from
    capital expenditure to operational expenditure changes how businesses plan and
    budget for technology. Major providers continue to expand their offerings,
    adding AI services, edge computing, and specialized hardware options.
    """,
    """
    Cybersecurity threats are evolving faster than ever. Ransomware attacks have
    become more sophisticated, targeting critical infrastructure and supply chains.
    Organizations must adopt zero-trust architectures and invest in employee
    training. The rise of AI-powered attacks means defenders need equally advanced
    tools to detect and respond to threats in real-time.
    """,
    """
    The semiconductor industry faces unprecedented challenges and opportunities.
    Global chip shortages have highlighted the fragility of supply chains.
    Governments are investing billions in domestic manufacturing capacity.
    Meanwhile, the push for more powerful AI chips drives innovation in chip
    architecture, with new designs optimized for machine learning workloads.
    """,
    """
    Remote work has permanently changed corporate culture. Companies are rethinking
    office space, collaboration tools, and management practices. Hybrid models are
    emerging as the new norm, balancing flexibility with in-person collaboration.
    This shift has implications for real estate, urban planning, and work-life
    balance across industries.
    """,
    """
    Quantum computing is moving from research labs to practical applications.
    While full-scale quantum advantage remains years away, hybrid classical-quantum
    systems are showing promise in optimization and simulation problems. Major tech
    companies and startups are racing to achieve quantum supremacy, with significant
    implications for cryptography, drug discovery, and materials science.
    """,
    """
    The creator economy continues to expand, with platforms enabling individuals
    to monetize content and build audiences. New tools for video editing, music
    production, and writing are democratizing creative work. However, concerns
    about algorithm-driven content and platform dependency raise questions about
    sustainability and creator rights in this evolving landscape.
    """,
    """
    Electric vehicles are reaching mainstream adoption as battery technology
    improves and charging infrastructure expands. Traditional automakers are
    pivoting their entire product lines toward electrification. This transition
    has ripple effects across the energy sector, mining industry, and urban
    planning, while raising questions about grid capacity and rare earth supply.
    """,
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


async def main() -> None:
    """Run the profiling example with real LLM calls."""
    print("=" * 70)
    print("INF-ENGINE: Execution Profiling Example")
    print("=" * 70)
    print(f"""
This example runs a multi-stage document analysis pipeline on {len(SAMPLE_ARTICLES)}
tech articles, demonstrating profiling with real OpenAI API calls.

Each document goes through 3 LLM stages:
  1. Summarization   (fast model)
  2. Theme extraction (fast model, depends on summary)
  3. Deep analysis    (smart model, depends on summary + themes)

Total LLM calls: {len(SAMPLE_ARTICLES)} docs x 3 stages = {len(SAMPLE_ARTICLES) * 3} calls

Endpoints:
  - fast: gpt-4o-mini (500 RPM, 64 concurrent)
  - smart: gpt-4o (500 RPM, 64 concurrent)

Trace output: {TRACE_PATH}
""")

    resources = create_resources()
    pipeline = DocumentAnalyzer()

    print("Running pipeline with profiling enabled...")
    print("-" * 70)

    async with ExecutionSettings(
        resources=resources,
        profile=True,
        profile_path=TRACE_PATH,
        max_concurrent=64,
    ) as settings:
        # Process all documents through the pipeline using batch execution
        # Each document is processed independently, enabling parallelism
        documents = [doc.strip() for doc in SAMPLE_ARTICLES]
        results = await pipeline(documents)

        # Display profiler statistics
        profiler = settings.get_profiler()
        if profiler:
            stats = profiler.get_statistics()
            print("\n" + "=" * 70)
            print("Execution Statistics")
            print("=" * 70)
            print(f"  Total tasks executed: {stats.total_tasks}")
            print(f"  Completed successfully: {stats.completed_tasks}")
            print(f"  Failed tasks: {stats.failed_tasks}")
            print(f"  Total duration: {stats.total_duration_ms:.0f}ms")
            print(f"  Average task duration: {stats.avg_duration_ms:.0f}ms")

            print("\nPer-Endpoint Statistics:")
            for alias, ep_stats in stats.endpoints.items():
                print(f"\n  {alias}:")
                print(f"    Tasks: {ep_stats.task_count}")
                print(f"    Avg duration: {ep_stats.avg_duration_ms:.0f}ms")
                print(f"    Min duration: {ep_stats.min_duration_ms:.0f}ms")
                print(f"    Max duration: {ep_stats.max_duration_ms:.0f}ms")

    # Display results summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for i, result in enumerate(results):
        print(f"\n[Article {i + 1}]")
        print(f"  Summary: {result['summary'][:80]}...")
        print(f"  Themes: {result['themes'][:60]}...")
        print(f"  Analysis: {result['analysis'][:80]}...")

    print("\n" + "=" * 70)
    print("Profiling Complete!")
    print("=" * 70)
    print(f"""
Trace file saved to: {TRACE_PATH.absolute()}

To visualize the trace:
  1. Open https://ui.perfetto.dev in your browser
  2. Drag and drop the trace file onto the page
  3. Explore the execution timeline

What to look for in the trace:
  - Process tracks for each endpoint (fast, smart)
  - Parallel task execution across documents
  - Sequential dependencies within each document (summary -> themes -> analysis)
  - Task duration variations across different LLM calls
  - Rate limiting effects if requests exceed limits
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
