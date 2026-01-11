#!/usr/bin/env python3
"""Optimization: Training pipelines with backward passes.

Demonstrates:
- train()/eval() modes for implicit record management
- TracedOutput: forward passes that carry ForwardRecord
- Loss functions: VerifierLoss, LLMRubricLoss, CompositeLoss
- Backward pass: feedback propagation through computation graph
- SFAOptimizer: LLM-based parameter updates

Requirements:
    export OPENAI_API_KEY=your-api-key

Run: python examples/05_optimization.py
"""

import asyncio
import os
import sys

from plait.execution.context import ExecutionSettings
from plait.module import LLMInference, Module
from plait.optimization import (
    CompositeLoss,
    LLMRubricLoss,
    RubricLevel,
    SFAOptimizer,
    VerifierLoss,
)
from plait.parameter import Parameter
from plait.resources.config import OpenAIEndpointConfig, ResourceConfig

# --- Learnable Pipeline ---


class ResearchAssistant(Module):
    """Two-stage assistant with learnable prompts.

    Stage 1: Break down query into sub-questions
    Stage 2: Research and synthesize answer
    """

    def __init__(self) -> None:
        super().__init__()
        # Learnable: can be optimized via backward passes
        self.breakdown_prompt = Parameter(
            "Break the query into 2-3 specific questions.",
            description="Instructs how to decompose complex queries",
        )
        self.synthesis_prompt = Parameter(
            "Provide a helpful answer based on the breakdown.",
            description="Instructs how to synthesize the final response",
        )

        self.breakdown = LLMInference(
            alias="worker",
            system_prompt=self.breakdown_prompt,
            temperature=0.3,
        )
        self.synthesize = LLMInference(
            alias="worker",
            system_prompt=self.synthesis_prompt,
            temperature=0.7,
        )

    def forward(self, query: str) -> str:
        questions = self.breakdown(query)
        return self.synthesize(questions)


# --- Loss Functions ---


def check_length(output: str) -> tuple[bool, str]:
    """Verify response has appropriate length."""
    words = len(output.split())
    if words < 50:
        return False, f"Too brief ({words} words, need 50+)"
    if words > 400:
        return False, f"Too verbose ({words} words, max 400)"
    return True, f"Good length ({words} words)"


QUALITY_RUBRIC = [
    RubricLevel(1, "Poor", "Incorrect, irrelevant, or missing the point"),
    RubricLevel(2, "Weak", "Partially correct but lacks depth"),
    RubricLevel(3, "Adequate", "Correct but generic"),
    RubricLevel(4, "Good", "Accurate and directly addresses the question"),
    RubricLevel(5, "Excellent", "Insightful, precise, comprehensive"),
]


# --- Training Loop ---


async def train() -> None:
    """Train the research assistant."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    print("=" * 60)
    print("plait: Optimization Example")
    print("=" * 60)
    print("\nTraining a 2-stage research assistant:")
    print("  Query -> Breakdown -> Synthesis -> Answer")
    print("  Both prompts are learnable and will be optimized.\n")

    # Setup resources
    resources = ResourceConfig(
        endpoints={
            "worker": OpenAIEndpointConfig(model="gpt-4o-mini", max_concurrent=3),
            "judge": OpenAIEndpointConfig(model="gpt-4o", max_concurrent=2),
            "optimizer/aggregator": OpenAIEndpointConfig(model="gpt-4o"),
            "optimizer/updater": OpenAIEndpointConfig(model="gpt-4o"),
        }
    )

    # Create pipeline and optimizer
    pipeline = ResearchAssistant()
    optimizer = SFAOptimizer(pipeline.parameters(), conservatism=0.3)
    optimizer.bind(resources)

    # Create loss function
    quality_loss = LLMRubricLoss(
        criteria="Evaluate correctness, clarity, and usefulness.",
        rubric=QUALITY_RUBRIC,
        alias="judge",
    )
    length_loss = VerifierLoss(check_length)
    loss_fn = CompositeLoss([(quality_loss, 0.7), (length_loss, 0.3)])
    loss_fn.bind(resources)

    # Training data
    queries = [
        "What are the pros and cons of microservices vs monoliths?",
        "How does database indexing work and when should I use it?",
    ]

    print("Initial prompts:")
    for param in pipeline.parameters():
        print(f"  {param._name}: '{param.value}'")
    print()

    # Training loop
    async with ExecutionSettings(resources=resources):
        pipeline.train()  # Enable training mode

        for epoch in range(2):
            print(f"{'=' * 60}")
            print(f"EPOCH {epoch + 1}")
            print(f"{'=' * 60}")

            optimizer.zero_feedback()

            # Forward passes (return TracedOutput in train mode)
            outputs = await asyncio.gather(*[pipeline(q) for q in queries])

            # Compute loss (batch)
            feedback = await loss_fn(outputs)
            print(f"Average score: {feedback.score:.2f}")

            # Backward pass (propagates to all parameters)
            await feedback.backward(optimizer=optimizer)

            # Update parameters
            updates = await optimizer.step()
            print(f"Updated {len(updates)} parameters")

        pipeline.eval()  # Back to eval mode

    print(f"\n{'=' * 60}")
    print("OPTIMIZED PROMPTS")
    print(f"{'=' * 60}")
    for param in pipeline.parameters():
        print(f"\n{param._name}:")
        print(f"  {param.value}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(train())
