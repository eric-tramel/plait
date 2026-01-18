#!/usr/bin/env python3
"""Optimization: Training pipelines with backward passes.

Demonstrates:
- train()/eval() modes for implicit record management
- Value outputs with tape ids for backward propagation
- Loss functions: VerifierLoss, LLMRubricLoss, CompositeLoss
- Backward pass: Value-based propagation through computation graph
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
from plait.values import Value, ValueKind, collect_records

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


def aggregate_losses(values: list[Value]) -> Value:
    """Aggregate per-sample loss Values into a single Value."""
    payload: list[list[str]] = []
    scores: list[float] = []
    for value in values:
        payload.extend(value.payload)
        score = value.meta.get("score")
        if score is not None:
            scores.append(float(score))
    meta: dict[str, float] = {}
    if scores:
        meta["score"] = sum(scores) / len(scores)
    return Value(kind=ValueKind.STRUCTURED, payload=payload, meta=meta)


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
    bind_loss = getattr(loss_fn, "bind", None)
    if callable(bind_loss):
        bind_loss(resources)

    # Training data
    queries = [
        "What are the pros and cons of microservices vs monoliths?",
        "How does database indexing work and when should I use it?",
    ]

    print("Initial prompts:")
    params = list(pipeline.parameters())
    for param in params:
        print(f"  {param._name}: '{param.value}'")
    unique_params = {param._id: param for param in params}
    print(f"\nTotal parameters (including references): {len(params)}")
    print(f"Unique parameters: {len(unique_params)}")
    print("Unique parameter ids:")
    for param_id, param in unique_params.items():
        name = param._name or "<unnamed>"
        print(f"  {name}: {param_id}")
    print()

    # Training loop
    async with ExecutionSettings(resources=resources):
        pipeline.train()  # Enable training mode

        for epoch in range(2):
            print(f"{'=' * 60}")
            print(f"EPOCH {epoch + 1}")
            print(f"{'=' * 60}")

            optimizer.zero_feedback()

            # Forward passes (return Value with tape ids in train mode)
            outputs = await asyncio.gather(*[pipeline(q) for q in queries])

            # Compute loss (per-sample) and aggregate
            loss_values = await asyncio.gather(*[loss_fn(out) for out in outputs])
            loss_value = aggregate_losses(loss_values)
            avg_score = loss_value.meta.get("score")
            if avg_score is not None:
                print(f"Average score: {avg_score:.2f}")
            else:
                print("Average score: n/a")

            def _snippet(text: str, limit: int = 200) -> str:
                cleaned = " ".join(str(text).split())
                return cleaned if len(cleaned) <= limit else cleaned[:limit] + "..."

            async def _debug_backward_trace(
                outputs=outputs,
                feedback=loss_value,
                queries=queries,
            ) -> None:
                from plait.optimization.backward import (
                    BackwardContext,
                    _combine_feedback,
                    _resolve_input_node,
                )

                print("\nBackward trace (per record):")
                for idx, output in enumerate(outputs, start=1):
                    record = collect_records(output)[0]
                    query = queries[idx - 1]
                    print(f"\nRecord {idx}: {query}")
                    graph = record.graph
                    feedback_map = {
                        output_id: [feedback] for output_id in graph.output_ids
                    }

                    for node_id in reversed(graph.topological_order()):
                        if node_id in graph.input_ids:
                            continue

                        downstream = feedback_map.get(node_id, [])
                        if not downstream:
                            continue

                        combined = _combine_feedback(downstream)
                        ctx = BackwardContext(
                            node_id=node_id,
                            inputs=record.node_inputs.get(node_id, {}),
                            output=record.node_outputs.get(node_id),
                            graph=graph,
                            all_results=record.node_outputs,
                            downstream_feedback=downstream,
                            reasoning_llm=None,
                        )

                        module = record.module_map[node_id]
                        result = await module.backward(combined, ctx)
                        module_name = (
                            getattr(module, "_name", None) or module.__class__.__name__
                        )
                        params = record.node_parameters.get(node_id, [])
                        if params:
                            param_labels = ", ".join(
                                f"{(p._name or 'param')}#{p._id[:8]}" for p in params
                            )
                        else:
                            param_labels = "none"

                        print(
                            f"  Node {node_id} [{module_name}] params: {param_labels}"
                        )
                        print(f"    combined_feedback: {_snippet(combined.payload)}")

                        for param_name, param_fb in result.parameter_feedback.items():
                            if isinstance(param_fb, Value):
                                snippet = _snippet(param_fb.payload)
                            else:
                                snippet = _snippet(param_fb)
                            print(f"    param_feedback[{param_name}]: {snippet}")

                        for input_name, input_fb in result.input_feedback.items():
                            input_node_id = _resolve_input_node(
                                node_id, input_name, record
                            )
                            target = input_node_id or "unknown"
                            print(
                                f"    input_feedback[{input_name}] -> {target}: "
                                f"{_snippet(input_fb.payload)}"
                            )

                        for input_name, input_fb in result.input_feedback.items():
                            input_node_id = _resolve_input_node(
                                node_id, input_name, record
                            )
                            if input_node_id:
                                feedback_map.setdefault(input_node_id, []).append(
                                    input_fb
                                )

            await _debug_backward_trace()

            # Backward pass (propagates to all parameters)
            await Value.backward(outputs, grad=loss_value, optimizer=optimizer)

            print("\nParameter feedback buffers:")
            for param in pipeline.parameters():
                if param._feedback_buffer:
                    print(
                        f"  {param._name or 'param'}#{param._id[:8]}: "
                        f"{_snippet(param._feedback_buffer[-1])}"
                    )

            # Update parameters
            updates = await optimizer.step()
            print(f"Updated {len(updates)} parameters")
            if updates:
                print("Applied updates:")
                for key, value in updates.items():
                    print(f"  {key}: {_snippet(value, 160)}")

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
