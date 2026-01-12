# Framework Comparison

Each comparison page shows how the same example—an extract-and-compare
pipeline—is implemented in plait and the compared framework. This "Rosetta
Stone" approach makes it easy to see the key differences at a glance.

## The Reference Example

All comparisons use this plait pipeline that demonstrates **automatic parallel execution**:

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
        return self.comparer(f"Compare:\n{facts1}\n\nvs:\n{facts2}")


resources = ResourceConfig(
    endpoints={
        "fast": OpenAIEndpointConfig(model="gpt-4o-mini", max_concurrent=20),
        "smart": OpenAIEndpointConfig(model="gpt-4o", max_concurrent=5),
    }
)

pipeline = ExtractAndCompare().bind(resources=resources)
result = await pipeline(doc1, doc2)
```

This example demonstrates:

- **Fan-out parallelism**: Two independent extractions run concurrently
- **Automatic dependency tracking**: The comparison waits for both extractions
- **Multi-model**: Fast model (gpt-4o-mini) for extraction, smart model (gpt-4o) for comparison
- **Learnable parameter**: `comparison_style` can be optimized via backward pass
- **Resource configuration**: Aliases separate module logic from deployment

## Quick Comparison

| Feature | plait | Pydantic AI | LangGraph | DSPy |
|---------|-------|-------------|-----------|------|
| **Graph definition** | Implicit (tracing) | Explicit (pydantic-graph) | Explicit (add_node/edge) | Implicit (composition) |
| **Parallel execution** | Automatic | Manual (asyncio.gather) | Explicit (Send) | Sequential |
| **Multi-model** | Alias-based | Per-agent | Per-node | Global config |
| **Learnable params** | `Parameter` class | No | No | Compile-time |
| **Optimization** | Runtime backward pass | No | No | Compile-time |
| **Execution** | Async-first | Async | Async | Sync-first |

## Parallel Execution Comparison

The extract-and-compare workflow highlights how each framework handles parallelism:

| Framework | Approach | Boilerplate Required |
|-----------|----------|---------------------|
| **plait** | Automatic from data dependencies | None - just write sequential code |
| **Pydantic AI** | Manual `asyncio.gather()` | Must explicitly group concurrent calls |
| **LangGraph** | Explicit `Send()` + reducers | Must define fan-out functions and state reducers |
| **DSPy** | Sequential by default | No built-in parallelism (sync-first) |

### Code Comparison

**plait** - Automatic parallelism:
```python
def forward(self, doc1: str, doc2: str) -> str:
    facts1 = self.extractor(doc1)  # These run
    facts2 = self.extractor(doc2)  # in parallel!
    return self.comparer(f"{facts1}\n\n{facts2}")
```

**Pydantic AI** - Manual gather:
```python
facts1, facts2 = await asyncio.gather(
    extractor.run(doc1),
    extractor.run(doc2),
)
```

**LangGraph** - Explicit Send:
```python
def fan_out(state):
    return [
        Send("extract", {"doc": state["doc1"]}),
        Send("extract", {"doc": state["doc2"]}),
    ]
graph.set_conditional_entry_point(fan_out)
```

**DSPy** - Sequential:
```python
facts1 = self.extract(document=doc1).facts  # Runs first
facts2 = self.extract(document=doc2).facts  # Runs second
```

## Benchmark Results

Real-world performance comparison using the extract-and-compare workflow with two sample documents (Electric Vehicles vs Hydrogen Fuel Cells). Each benchmark makes 3 LLM calls: 2 extractions (gpt-4o-mini) + 1 comparison (gpt-4o).

| Framework | Time (ms) | Memory (MB) | vs plait Time | vs plait Memory |
|-----------|-----------|-------------|---------------|-----------------|
| **plait** | **6,944** | **0.4** | — | — |
| LangGraph | 10,071 | 26.2 | +45% slower | 65x more |
| Pydantic AI | 8,672 | 17.6 | +25% slower | 44x more |
| DSPy | 13,447 | 76.0 | +94% slower | 190x more |

**Key findings:**

- **plait is fastest** due to automatic parallel execution of independent operations
- **plait uses minimal memory** (~0.5 MB) compared to other frameworks (17-76 MB)
- **DSPy is slowest** because it runs extractions sequentially (no built-in parallelism)
- **LangGraph** requires explicit `Send()` configuration but achieves parallelism
- **Pydantic AI** requires manual `asyncio.gather()` for concurrent execution

Run the benchmarks yourself:
```bash
make doctest
```

## Detailed Comparisons

- [plait vs Pydantic AI](pydantic-ai.md) — Agent-based workflows and Pydantic integration
- [plait vs LangGraph](langgraph.md) — State graphs and checkpoint-based workflows
- [plait vs DSPy](dspy.md) — Compile-time vs runtime optimization

## When to Choose plait

Choose plait when you need:

- **Automatic parallelism**: Independent operations run concurrently without boilerplate
- **Runtime optimization**: Improve prompts based on feedback during execution
- **PyTorch-like patterns**: Familiar Module/forward/backward API
- **Automatic DAG capture**: Write normal Python, get optimized execution
- **Multi-model pipelines**: Different models for different steps
- **Centralized resources**: Separate module logic from deployment configuration
