# plait vs Pydantic AI

This comparison uses the same example—a summarize-and-analyze pipeline—to show
how each framework approaches the same problem.

## The Example: Summarize and Analyze

A two-stage pipeline that:

1. Summarizes input text using a fast model (gpt-4o-mini)
2. Analyzes the summary using a smarter model (gpt-4o)
3. Has a configurable instruction parameter

## plait Implementation

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
result = await pipeline("Your input text...")
```

## Pydantic AI Implementation

```python
from pydantic_ai import Agent

# Pydantic AI uses separate Agent instances
summarizer = Agent(
    'openai:gpt-4o-mini',
    system_prompt="Summarize the input text concisely.",
)

analyzer = Agent(
    'openai:gpt-4o',
    system_prompt="Be concise and highlight key insights.",
)


async def summarize_and_analyze(text: str) -> str:
    summary_result = await summarizer.run(text)
    analysis_result = await analyzer.run(
        f"Analyze this summary:\n{summary_result.data}"
    )
    return analysis_result.data
```

For a more structured approach, Pydantic AI offers `pydantic-graph`:

```python
from pydantic_graph import Graph, Node, End
from pydantic_ai import Agent

summarizer = Agent('openai:gpt-4o-mini', system_prompt="Summarize concisely.")
analyzer = Agent('openai:gpt-4o', system_prompt="Be concise, highlight insights.")


class SummarizeNode(Node):
    async def run(self, ctx) -> str:
        result = await summarizer.run(ctx.state['text'])
        ctx.state['summary'] = result.data
        return 'analyze'


class AnalyzeNode(Node):
    async def run(self, ctx) -> End:
        result = await analyzer.run(
            f"Analyze this summary:\n{ctx.state['summary']}"
        )
        ctx.state['analysis'] = result.data
        return End()


graph = Graph(nodes=[SummarizeNode(), AnalyzeNode()])
result = await graph.run({'text': 'Your input text...'})
print(result.state['analysis'])
```

## Key Differences

| Aspect | plait | Pydantic AI |
|--------|-------|-------------|
| **Structure** | Single `Module` class with `forward()` | Separate `Agent` instances or `pydantic-graph` nodes |
| **Graph definition** | Implicit from code flow | Explicit nodes and edges (pydantic-graph) |
| **Model binding** | Aliases resolved via `ResourceConfig` | Model specified per Agent |
| **Learnable params** | `Parameter` class for optimizable values | Not supported |
| **Concurrency config** | Centralized in `ResourceConfig` | Per-agent or manual |

### Graph Definition

**plait**: The DAG is captured automatically by tracing `forward()`. Write normal
Python and the framework builds the execution graph.

**Pydantic AI**: With `pydantic-graph`, you explicitly define nodes and edges.
Each node is a class that returns the next node name.

### Learnable Parameters

**plait**: The `Parameter` class holds values that can be optimized through
backward passes. The `instructions` parameter above can improve over time based
on feedback.

**Pydantic AI**: No built-in support for learnable parameters. Prompts are static
once defined.

### Resource Configuration

**plait**: Models are bound via aliases (`"fast"`, `"smart"`) that map to
`EndpointConfig` objects. This separates module logic from deployment
configuration.

**Pydantic AI**: Model is specified directly on each Agent (`'openai:gpt-4o'`).
Configuration is coupled to agent definition.

## When to Choose Each

### Choose plait when:

- You want to **optimize prompts through feedback** over time
- You prefer **PyTorch-like patterns** (Module, forward, backward)
- You need **automatic DAG capture** from Python code
- You want **centralized resource configuration** separate from module logic

### Choose Pydantic AI when:

- You need **agent-based workflows** with tools and function calling
- Your codebase **already uses Pydantic** extensively
- You want **dependency injection** for clean testing patterns
- **Streaming responses** are critical for your UX
