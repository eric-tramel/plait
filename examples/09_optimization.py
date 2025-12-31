#!/usr/bin/env python3
"""Optimization example: Multi-step research synthesis pipeline.

This example demonstrates optimization of a multi-step workflow where
multiple LLM calls with learnable parameters work together. Feedback
from the final output propagates backward through the entire graph.

Pipeline Architecture:
    User Query
        |
        v
    Query Analyzer  <- Learnable prompt
        |
        v
    Researcher      <- Learnable prompt
        |
        v
    Synthesizer     <- Learnable prompt
        |
        v
    Final Answer
        |
        v
    Loss Function   <- Evaluates quality
        |
        v
    Backward Pass   <- Updates all 3 prompts

Key features demonstrated:
- Multi-step pipeline with 3 learnable parameters
- LLMRubricLoss for structured evaluation with Likert scale
- Multiple rubrics: quality, human-alignment, and engagement
- CompositeLoss combining rubric evaluation + programmatic checks
- Feedback propagation through computation graph
- Progressive improvement across training epochs
- Profiling with ExecutionSettings for performance analysis

Requirements:
    export OPENAI_API_KEY=your-api-key

Run with:
    python examples/09_optimization.py
"""

import asyncio
import os
import sys
from pathlib import Path

from inf_engine.execution.context import ExecutionSettings
from inf_engine.execution.executor import run
from inf_engine.module import InferenceModule, LLMInference
from inf_engine.optimization import (
    CompositeLoss,
    LLMRubricLoss,
    RubricLevel,
    SFAOptimizer,
    VerifierLoss,
)
from inf_engine.parameter import Parameter
from inf_engine.resources.config import OpenAIEndpointConfig, ResourceConfig

# =============================================================================
# Multi-Step Research Pipeline
# =============================================================================


class ResearchPipeline(InferenceModule):
    """A 3-step research pipeline with learnable prompts at each stage.

    Pipeline stages:
    1. Query Analyzer: Breaks down complex queries into sub-questions
    2. Researcher: Gathers information for each question
    3. Synthesizer: Combines research into a coherent answer

    All three prompts are learnable and will be jointly optimized.
    """

    def __init__(self) -> None:
        super().__init__()

        # Stage 1: Query Analysis (intentionally weak initial prompt)
        self.analyzer_prompt = Parameter(
            "Break down the user's query into 2-3 specific research questions.",
            description=(
                "System prompt for query analysis. Should instruct the LLM to "
                "identify the core question, break it into 2-3 specific sub-questions, "
                "ensure questions are concrete and answerable, and output as numbered list."
            ),
        )
        self.analyzer = LLMInference(
            alias="worker",
            system_prompt=self.analyzer_prompt,
            temperature=0.3,
        )

        # Stage 2: Research
        self.researcher_prompt = Parameter(
            "Answer the given research questions with relevant information.",
            description=(
                "System prompt for research. Should instruct the LLM to address "
                "each question with factual information, provide specific details, "
                "note caveats, and keep answers focused and concise."
            ),
        )
        self.researcher = LLMInference(
            alias="worker",
            system_prompt=self.researcher_prompt,
            temperature=0.5,
        )

        # Stage 3: Synthesis
        self.synthesizer_prompt = Parameter(
            "Combine the research notes into a well-structured helpful answer.",
            description=(
                "System prompt for synthesis. Should instruct the LLM to directly "
                "address the original query, integrate all research notes, present "
                "a coherent response, and use clear formatting when helpful."
            ),
        )
        self.synthesizer = LLMInference(
            alias="worker",
            system_prompt=self.synthesizer_prompt,
            temperature=0.7,
        )

    def forward(self, query: str) -> str:
        """Execute the 3-step pipeline."""
        # Note: We pass proxies directly to avoid f-string formatting issues
        # during tracing. Each module's system prompt provides context.

        # Step 1: Analyze query into sub-questions
        questions = self.analyzer(query)

        # Step 2: Research each question
        research = self.researcher(questions)

        # Step 3: Synthesize into final answer
        answer = self.synthesizer(research)
        return answer


# =============================================================================
# Evaluation Setup
# =============================================================================


def check_format(output: str) -> tuple[bool, str]:
    """Verify response has appropriate length and avoids markdown."""
    word_count = len(output.split())

    issues = []
    if word_count < 100:
        issues.append(f"Too brief ({word_count} words, need 100+)")
    elif word_count > 600:
        issues.append(f"Too verbose ({word_count} words, max 600)")

    # Penalize heavy markdown formatting (we want clean prose)
    markdown_markers = ["##", "###", "**", "```", "- ", "* ", "1. "]
    markdown_count = sum(output.count(m) for m in markdown_markers)
    if markdown_count > 5:
        issues.append(f"Too much markdown ({markdown_count} markers)")

    if issues:
        return False, "; ".join(issues)
    return True, f"Good: {word_count} words, minimal formatting"


# Rubric definitions for multi-axis evaluation
QUALITY_RUBRIC = [
    RubricLevel(1, "Poor", "Incorrect or irrelevant information, misses the point"),
    RubricLevel(2, "Weak", "Partially correct but lacks depth or has errors"),
    RubricLevel(3, "Adequate", "Correct and reasonably complete, but generic"),
    RubricLevel(4, "Good", "Accurate, well-reasoned, addresses the question directly"),
    RubricLevel(5, "Excellent", "Insightful, precise, and perfectly addresses intent"),
]

HUMAN_ALIGNMENT_RUBRIC = [
    RubricLevel(
        1,
        "AI Slop",
        "Obvious AI output: 'I'd be happy to help', hedging, filler phrases",
    ),
    RubricLevel(
        2, "Robotic", "Stilted language, excessive caveats, assistant-like phrasing"
    ),
    RubricLevel(3, "Neutral", "Passable but still reads somewhat artificial"),
    RubricLevel(
        4, "Natural", "Reads like thoughtful human writing, minimal AI markers"
    ),
    RubricLevel(
        5,
        "Expert Human",
        "Indistinguishable from skilled human expert, authentic voice",
    ),
]

ENGAGEMENT_RUBRIC = [
    RubricLevel(1, "Boring", "Dry, lifeless, would stop reading immediately"),
    RubricLevel(2, "Dull", "Informative but tedious, hard to stay engaged"),
    RubricLevel(3, "Acceptable", "Gets the job done, neither engaging nor boring"),
    RubricLevel(4, "Interesting", "Holds attention, some memorable points"),
    RubricLevel(5, "Compelling", "Genuinely interesting, would want to read more"),
]


# Training and evaluation data
TRAINING_DATA = [
    {
        "query": (
            "What are the main differences between microservices and monolithic "
            "architecture? When should a startup choose one over the other?"
        ),
    },
    {
        "query": (
            "How has remote work affected software team productivity? "
            "What practices help address the challenges?"
        ),
    },
    {
        "query": (
            "Explain how database indexing works and when to add indexes "
            "vs when they might hurt performance."
        ),
    },
    {
        "query": (
            "Compare REST, GraphQL, and gRPC for APIs. "
            "What are the tradeoffs and when would you choose each?"
        ),
    },
]

EVAL_DATA = [
    {
        "query": (
            "What strategies can engineering teams use to reduce technical debt "
            "while still meeting feature delivery deadlines?"
        ),
    },
]


# =============================================================================
# Training Loop
# =============================================================================


async def train_research_pipeline() -> None:
    """Train the multi-step research pipeline."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    print("=" * 70)
    print("inf-engine: Multi-Step Pipeline Optimization")
    print("=" * 70)
    print("\nOptimizing a 3-step research pipeline:")
    print("  Query Analyzer -> Researcher -> Synthesizer")
    print("\nFeedback propagates to ALL THREE prompts via backward pass.\n")

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    resources = ResourceConfig(
        endpoints={
            # Worker model for pipeline steps
            "worker": OpenAIEndpointConfig(model="gpt-4o-mini", max_concurrent=5),
            # Judge model for rubric evaluation
            "judge": OpenAIEndpointConfig(model="gpt-4o", max_concurrent=3),
            # Optimizer models
            "optimizer/aggregator": OpenAIEndpointConfig(model="gpt-4o"),
            "optimizer/updater": OpenAIEndpointConfig(model="gpt-4o"),
        }
    )

    # Create pipeline and optimizer
    pipeline = ResearchPipeline()
    optimizer = SFAOptimizer(pipeline.parameters(), conservatism=0.3)
    optimizer.bind(resources)

    # Create loss function with multiple rubric-based evaluations
    # Quality rubric: correctness, information density, intent alignment
    quality_loss = LLMRubricLoss(
        criteria=(
            "Evaluate correctness, information density (conciseness without "
            "sacrificing clarity), and alignment with the user's actual intent. "
            "Penalize vague, generic, or off-topic responses."
        ),
        rubric=QUALITY_RUBRIC,
        alias="judge",
    )

    # Human alignment rubric: no AI slop, natural writing
    alignment_loss = LLMRubricLoss(
        criteria=(
            "Evaluate how human-like the writing is. Penalize AI assistant "
            "phrases ('I'd be happy to', 'Great question!'), excessive hedging, "
            "unnecessary caveats, and robotic phrasing. Reward authentic voice."
        ),
        rubric=HUMAN_ALIGNMENT_RUBRIC,
        alias="judge",
    )

    # Engagement rubric: how compelling is the text
    engagement_loss = LLMRubricLoss(
        criteria=(
            "Evaluate how engaging and interesting the text is to read. "
            "Consider: Does it hold attention? Is it memorable? Would you "
            "want to keep reading? Penalize dry, lifeless prose."
        ),
        rubric=ENGAGEMENT_RUBRIC,
        alias="judge",
    )

    # Format verification: length and markdown checks
    format_loss = VerifierLoss(check_format)

    # Combine all losses with weights
    loss_fn = CompositeLoss(
        [
            (quality_loss, 0.35),  # Correctness and relevance
            (alignment_loss, 0.30),  # Human-like writing
            (engagement_loss, 0.20),  # Engaging prose
            (format_loss, 0.15),  # Length and format checks
        ]
    )
    loss_fn.bind(resources)

    # Show initial state
    print("Initial prompts (intentionally weak):")
    for param in pipeline.parameters():
        print(f"  {param._name}: {param.value!r}")
    print("\nLoss function (4 components):")
    print("  35% Quality (correctness, density, intent)")
    print("  30% Human alignment (no AI slop)")
    print("  20% Engagement (compelling prose)")
    print("  15% Format verification (length, no markdown)")
    print("\nOptimizer: SFAOptimizer (conservatism=0.3)\n")

    # -------------------------------------------------------------------------
    # Training (with profiling)
    # -------------------------------------------------------------------------
    trace_path = Path("./traces/optimization_run.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    num_epochs = 3
    all_scores: list[list[float]] = []

    async with ExecutionSettings(
        resources=resources,
        profile=True,
        profile_path=trace_path,
    ) as settings:
        for epoch in range(num_epochs):
            print(f"{'=' * 70}")
            print(f"EPOCH {epoch + 1}/{num_epochs}")
            print(f"{'=' * 70}")

            optimizer.zero_feedback()

            # Run all forward passes in parallel
            print(f"\nRunning {len(TRAINING_DATA)} forward passes in parallel...")
            forward_tasks = [
                run(pipeline, item["query"], resources=resources, record=True)
                for item in TRAINING_DATA
            ]
            forward_results = await asyncio.gather(*forward_tasks)

            # Run all loss computations in parallel
            print("Computing losses in parallel...")
            loss_tasks = [
                loss_fn(output, record=record, context={"query": item["query"]})
                for (output, record), item in zip(
                    forward_results, TRAINING_DATA, strict=True
                )
            ]
            feedbacks = await asyncio.gather(*loss_tasks)

            # Backward passes (accumulate feedback)
            epoch_scores: list[float] = []
            for i, feedback in enumerate(feedbacks):
                score = feedback.score if feedback.score is not None else 0.5
                epoch_scores.append(score)
                await feedback.backward(optimizer=optimizer)
                print(f"  [{i + 1}] Score: {score:.2f}")

            # Epoch summary and parameter update
            avg_score = sum(epoch_scores) / len(epoch_scores)
            all_scores.append(epoch_scores)
            print(f"\nEpoch {epoch + 1} average: {avg_score:.2f}")

            print("Running optimizer.step()...")
            updates = await optimizer.step()
            if updates:
                print(f"Updated {len(updates)} parameters")
            else:
                print("No updates this epoch")

        # ---------------------------------------------------------------------
        # Evaluation on held-out data
        # ---------------------------------------------------------------------
        print(f"\n{'=' * 70}")
        print("EVALUATION (Held-Out Query)")
        print(f"{'=' * 70}")

        for item in EVAL_DATA:
            query = item["query"]
            print(f"\nQuery: {query}")

            output = await run(pipeline, query, resources=resources)
            feedback = await loss_fn(output, context={"query": query})
            score = feedback.score if feedback.score is not None else 0.5

            print(f"\nResponse:\n{str(output)[:500]}...")
            print(f"\nScore: {score:.2f}")

        # Get profiler statistics before exiting context
        profiler = settings.get_profiler()

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("TRAINING PROGRESS")
    print(f"{'=' * 70}")

    print(f"{'Epoch':<10} {'Avg':<10} {'Min':<10} {'Max':<10}")
    print("-" * 40)
    for i, scores in enumerate(all_scores):
        avg = sum(scores) / len(scores)
        print(f"{i + 1:<10} {avg:<10.2f} {min(scores):<10.2f} {max(scores):<10.2f}")

    if len(all_scores) >= 2:
        first_avg = sum(all_scores[0]) / len(all_scores[0])
        last_avg = sum(all_scores[-1]) / len(all_scores[-1])
        improvement = last_avg - first_avg
        sign = "+" if improvement >= 0 else ""
        print(
            f"\nImprovement: {sign}{improvement:.2f} ({sign}{improvement * 100:.0f}%)"
        )

    print(f"\n{'=' * 70}")
    print("FINAL PROMPTS")
    print(f"{'=' * 70}")
    for param in pipeline.parameters():
        print(f"\n{param._name}:")
        print("-" * 40)
        print(param.value)

    # -------------------------------------------------------------------------
    # Profiling Statistics
    # -------------------------------------------------------------------------
    if profiler:
        stats = profiler.get_statistics()
        print(f"\n{'=' * 70}")
        print("PROFILING STATISTICS")
        print(f"{'=' * 70}")
        print(f"Total tasks:      {stats.total_tasks}")
        print(f"Completed:        {stats.completed_tasks}")
        print(f"Failed:           {stats.failed_tasks}")
        print(f"Total duration:   {stats.total_duration_ms:.1f}ms")
        print(f"Avg task time:    {stats.avg_duration_ms:.1f}ms")
        print(f"Min task time:    {stats.min_duration_ms:.1f}ms")
        print(f"Max task time:    {stats.max_duration_ms:.1f}ms")

        if stats.endpoints:
            print(f"\n{'Endpoint':<25} {'Tasks':<8} {'Avg(ms)':<10} {'Max(ms)':<10}")
            print("-" * 53)
            for endpoint, ep_stats in stats.endpoints.items():
                print(
                    f"{endpoint:<25} {ep_stats.task_count:<8} "
                    f"{ep_stats.avg_duration_ms:<10.1f} {ep_stats.max_duration_ms:<10.1f}"
                )

        print(f"\nTrace exported to: {trace_path}")
        print("Open with https://ui.perfetto.dev for visualization")

    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(train_research_pipeline())
