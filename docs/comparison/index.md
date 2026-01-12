# Framework Comparison

Each comparison page shows how the same example—a summarize-and-analyze
pipeline—is implemented in plait and the compared framework. This "Rosetta
Stone" approach makes it easy to see the key differences at a glance.

## The Reference Example

All comparisons use this plait pipeline from the [Getting Started](../getting-started.md) guide:

```python
from plait import Module, LLMInference, Parameter
from plait.resources import OpenAIEndpointConfig, ResourceConfig


class SummarizeAndAnalyze(Module):
    def __init__(self):
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
        "fast": OpenAIEndpointConfig(model="gpt-4o-mini", max_concurrent=20),
        "smart": OpenAIEndpointConfig(model="gpt-4o", max_concurrent=5),
    }
)

pipeline = SummarizeAndAnalyze().bind(resources=resources)
result = await pipeline("Your input text...")
```

This example demonstrates:

- **Two-stage pipeline**: summarize then analyze
- **Multi-model**: fast model (gpt-4o-mini) and smart model (gpt-4o)
- **Learnable parameter**: `instructions` can be optimized via backward pass
- **Resource configuration**: aliases separate module logic from deployment

## Quick Comparison

| Feature | plait | Pydantic AI | LangGraph | DSPy |
|---------|-------|-------------|-----------|------|
| **Graph definition** | Implicit (tracing) | Explicit (pydantic-graph) | Explicit (add_node/edge) | Implicit (composition) |
| **Multi-model** | Alias-based | Per-agent | Per-node | Global config |
| **Learnable params** | `Parameter` class | No | No | Compile-time |
| **Optimization** | Runtime backward pass | No | No | Compile-time |
| **Execution** | Async-first | Async | Async | Sync-first |

## Detailed Comparisons

- [plait vs Pydantic AI](pydantic-ai.md) — Agent-based workflows and Pydantic integration
- [plait vs LangGraph](langgraph.md) — State graphs and checkpoint-based workflows
- [plait vs DSPy](dspy.md) — Compile-time vs runtime optimization

## When to Choose plait

Choose plait when you need:

- **Runtime optimization**: Improve prompts based on feedback during execution
- **PyTorch-like patterns**: Familiar Module/forward/backward API
- **Automatic DAG capture**: Write normal Python, get optimized execution
- **Multi-model pipelines**: Different models for different steps
- **Centralized resources**: Separate module logic from deployment configuration
