# plait vs Pydantic AI

This comparison uses the same example—an extract-and-compare pipeline—to show
how each framework approaches the same problem, with a focus on parallel execution.

## The Example: Extract and Compare

A three-stage pipeline that:

1. Takes two documents as input
2. Extracts main facts from both documents (can run in parallel)
3. Compares and contrasts the extracted facts

This workflow highlights **plait's automatic parallel execution** vs Pydantic AI's manual `asyncio.gather()` approach.

## plait Implementation

```python
from plait import Module, LLMInference, Parameter
from plait.resources import OpenAIEndpointConfig, ResourceConfig


class ExtractAndCompare(Module):
    def __init__(self):
        super().__init__()
        self.comparison_style = Parameter(
            value="Highlight key similarities and differences.",
            description="Controls the style of comparison output.",
        )
        self.extractor = LLMInference(
            alias="fast",
            system_prompt="Extract the main facts as a bulleted list.",
        )
        self.comparer = LLMInference(
            alias="smart",
            system_prompt=self.comparison_style,
        )

    def forward(self, doc1: str, doc2: str) -> str:
        # These two calls are INDEPENDENT - plait runs them in PARALLEL
        facts1 = self.extractor(doc1)
        facts2 = self.extractor(doc2)

        # This depends on both facts, waits for both to complete
        return self.comparer(
            f"Compare and contrast:\n\n"
            f"Document 1:\n{facts1}\n\n"
            f"Document 2:\n{facts2}"
        )


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
```

## Pydantic AI Implementation

```python
import asyncio
from pydantic_ai import Agent


extractor = Agent(
    'openai:gpt-4o-mini',
    system_prompt="Extract the main facts as a bulleted list.",
)

comparer = Agent(
    'openai:gpt-4o',
    system_prompt="Highlight key similarities and differences.",
)


async def extract_and_compare(doc1: str, doc2: str) -> str:
    # Must use asyncio.gather() explicitly for parallel execution
    facts1_result, facts2_result = await asyncio.gather(
        extractor.run(doc1),
        extractor.run(doc2),
    )

    comparison_result = await comparer.run(
        f"Compare and contrast:\n\n"
        f"Document 1:\n{facts1_result.output}\n\n"
        f"Document 2:\n{facts2_result.output}"
    )
    return comparison_result.output
```

For a more structured approach, Pydantic AI offers `pydantic-graph`:

```python
import asyncio
from pydantic_graph import Graph, Node, End
from pydantic_ai import Agent

extractor = Agent('openai:gpt-4o-mini', system_prompt="Extract facts as bullets.")
comparer = Agent('openai:gpt-4o', system_prompt="Highlight similarities/differences.")


class ExtractNode(Node):
    async def run(self, ctx) -> str:
        # Must manually implement parallel extraction
        facts1, facts2 = await asyncio.gather(
            extractor.run(ctx.state['doc1']),
            extractor.run(ctx.state['doc2']),
        )
        ctx.state['facts1'] = facts1.output
        ctx.state['facts2'] = facts2.output
        return 'compare'


class CompareNode(Node):
    async def run(self, ctx) -> End:
        result = await comparer.run(
            f"Compare:\n{ctx.state['facts1']}\n\n{ctx.state['facts2']}"
        )
        ctx.state['comparison'] = result.output
        return End()


graph = Graph(nodes=[ExtractNode(), CompareNode()])
result = await graph.run({'doc1': doc1, 'doc2': doc2})
print(result.state['comparison'])
```

## Key Differences

| Aspect | plait | Pydantic AI |
|--------|-------|-------------|
| **Structure** | Single `Module` class with `forward()` | Separate `Agent` instances or `pydantic-graph` nodes |
| **Parallel execution** | Automatic from data flow | Manual `asyncio.gather()` |
| **Graph definition** | Implicit from code flow | Explicit nodes and edges (pydantic-graph) |
| **Model binding** | Aliases resolved via `ResourceConfig` | Model specified per Agent |
| **Learnable params** | `Parameter` class for optimizable values | Not supported |
| **Concurrency config** | Centralized in `ResourceConfig` | Per-agent or manual |

### Parallel Execution

**plait**: Automatically detects that the two extraction calls are independent and
runs them concurrently. No special syntax needed.

```python
def forward(self, doc1: str, doc2: str) -> str:
    facts1 = self.extractor(doc1)  # These run
    facts2 = self.extractor(doc2)  # in parallel!
    return self.comparer(...)      # Waits for both
```

**Pydantic AI**: Requires explicit `asyncio.gather()` to run operations concurrently.

```python
async def extract_and_compare(doc1: str, doc2: str) -> str:
    # Must explicitly group concurrent operations
    facts1_result, facts2_result = await asyncio.gather(
        extractor.run(doc1),
        extractor.run(doc2),
    )
    comparison_result = await comparer.run(...)
    return comparison_result.output
```

### Graph Definition

**plait**: The DAG is captured automatically by tracing `forward()`. Write normal
Python and the framework builds the execution graph.

```python
def forward(self, doc1: str, doc2: str) -> str:
    facts1 = self.extractor(doc1)      # Node A
    facts2 = self.extractor(doc2)      # Node B (parallel with A)
    return self.comparer(f"{facts1}\n{facts2}")  # Node C depends on A and B
```

**Pydantic AI**: With `pydantic-graph`, you explicitly define nodes and edges.
Each node is a class that returns the next node name.

```python
class ExtractNode(Node):
    async def run(self, ctx) -> str:
        # Parallel extraction requires manual asyncio.gather()
        facts1, facts2 = await asyncio.gather(...)
        return 'compare'  # Next node name

class CompareNode(Node):
    async def run(self, ctx) -> End:
        return End()
```

### Learnable Parameters

**plait**: The `Parameter` class holds values that can be optimized through
backward passes. The `comparison_style` parameter above can improve over time based
on feedback.

```python
self.comparison_style = Parameter(
    value="Highlight key similarities and differences.",
    description="Controls the style of comparison output.",
)
```

**Pydantic AI**: No built-in support for learnable parameters. Prompts are static
once defined.

### Resource Configuration

**plait**: Models are bound via aliases (`"fast"`, `"smart"`) that map to
`EndpointConfig` objects. This separates module logic from deployment
configuration.

```python
resources = ResourceConfig(
    endpoints={
        "fast": OpenAIEndpointConfig(model="gpt-4o-mini", max_concurrent=20),
        "smart": OpenAIEndpointConfig(model="gpt-4o", max_concurrent=5),
    }
)
pipeline = ExtractAndCompare().bind(resources=resources)
```

**Pydantic AI**: Model is specified directly on each Agent (`'openai:gpt-4o'`).
Configuration is coupled to agent definition.

```python
extractor = Agent('openai:gpt-4o-mini', system_prompt="...")
comparer = Agent('openai:gpt-4o', system_prompt="...")
```

## When to Choose Each

### Choose plait when:

- You want **automatic parallel execution** without explicit `asyncio.gather()`
- You want to **optimize prompts through feedback** over time
- You prefer **PyTorch-like patterns** (Module, forward, backward)
- You need **automatic DAG capture** from Python code
- You want **centralized resource configuration** separate from module logic

### Choose Pydantic AI when:

- You need **agent-based workflows** with tools and function calling
- Your codebase **already uses Pydantic** extensively
- You want **dependency injection** for clean testing patterns
- **Streaming responses** are critical for your UX
