# plait vs DSPy

> **Run the comparison:** `uv run --with dspy --with rich docs/comparison/compare_dspy.py`
>
> [View full source](compare_dspy.py)

This comparison uses the same example--an extract-and-compare pipeline--to show
how each framework approaches the same problem, with a focus on parallel execution.

## The Example: Extract and Compare

A three-stage pipeline that:

1. Takes two documents as input
2. Extracts main facts from both documents (can run in parallel)
3. Compares and contrasts the extracted facts

This workflow highlights **plait's automatic parallel execution** for independent operations.

## plait Implementation

```python
from plait import Module, LLMInference, Parameter
from plait.resources import OpenAIEndpointConfig, ResourceConfig


class FactsCombiner(Module):
    """Combine two facts into a comparison prompt."""

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
        facts1 = self.extractor(doc1)
        facts2 = self.extractor(doc2)

        # Combine facts using the combiner module (resolves Proxy objects)
        combined = self.combiner(facts1, facts2)

        # This depends on both facts, waits for both to complete
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
```

## DSPy Implementation

```python
import dspy


class ExtractFacts(dspy.Signature):
    """Extract the main facts from the document as a bulleted list."""

    document: str = dspy.InputField()
    facts: str = dspy.OutputField(desc="Bulleted list of main facts")


class CompareAndContrast(dspy.Signature):
    """Compare and contrast the facts from two documents."""

    facts_doc1: str = dspy.InputField(desc="Facts from first document")
    facts_doc2: str = dspy.InputField(desc="Facts from second document")
    comparison: str = dspy.OutputField(
        desc="Compare and contrast analysis highlighting similarities and differences"
    )


class ExtractAndCompare(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        # Disable caching to ensure fair comparison with plait
        self.fast_lm = dspy.LM("openai/gpt-4o-mini", cache=False)
        self.smart_lm = dspy.LM("openai/gpt-4o", cache=False)
        self.extract = dspy.Predict(ExtractFacts)
        self.compare = dspy.Predict(CompareAndContrast)

    def forward(self, doc1: str, doc2: str) -> str:
        # These run SEQUENTIALLY - no automatic parallelism in DSPy
        with dspy.context(lm=self.fast_lm):
            facts1 = self.extract(document=doc1).facts
            facts2 = self.extract(document=doc2).facts

        with dspy.context(lm=self.smart_lm):
            comparison = self.compare(
                facts_doc1=facts1, facts_doc2=facts2
            ).comparison
        return comparison


pipeline = ExtractAndCompare()
result = pipeline(doc1=doc1, doc2=doc2)
```

## Key Differences

| Aspect | plait | DSPy |
|--------|-------|------|
| **Structure** | `Module` with `LLMInference` | `dspy.Module` with `Signature` |
| **Parallel execution** | Automatic from data flow | Sequential (sync-first) |
| **Prompts** | Explicit system prompts | Signature docstrings |
| **Multi-model** | Aliases map to different endpoints | `dspy.context(lm=...)` per call |
| **Optimization** | Runtime backward pass | Compile-time teleprompters |
| **Execution** | Async-first | Sync-first |

### Parallel Execution

**plait**: Automatically detects that the two extraction calls are independent and
runs them concurrently. No special syntax needed.

```python
def forward(self, doc1: str, doc2: str) -> str:
    facts1 = self.extractor(doc1)  # These run
    facts2 = self.extractor(doc2)  # in parallel!
    return self.comparer(...)      # Waits for both
```

**DSPy**: Executes synchronously by default. Each call completes before the next begins.

```python
def forward(self, doc1: str, doc2: str) -> str:
    facts1 = self.extract(document=doc1).facts  # Runs first
    facts2 = self.extract(document=doc2).facts  # Runs second
    return self.compare(...).comparison          # Runs third
```

### Prompt Definition

**plait**: Prompts are explicit strings passed to `LLMInference`. The `Parameter`
class makes prompts learnable.

```python
self.extractor = LLMInference(
    alias="fast",
    system_prompt="Extract the main facts from the document as a bulleted list.",
)
```

**DSPy**: Prompts are derived from `Signature` docstrings and field names. The
framework generates the actual prompt.

```python
class ExtractFacts(dspy.Signature):
    """Extract the main facts from the document as a bulleted list."""
    document: str = dspy.InputField()
    facts: str = dspy.OutputField(desc="Bulleted list of main facts")
```

### Multi-Model Configuration

**plait**: Different aliases can map to different models via `ResourceConfig`.

```python
resources = ResourceConfig(
    endpoints={
        "fast": OpenAIEndpointConfig(model="gpt-4o-mini"),
        "smart": OpenAIEndpointConfig(model="gpt-4o"),
    }
)
```

**DSPy**: Use `dspy.context(lm=...)` to switch models per call, or store LM
instances as module attributes. Use `cache=False` to disable caching for benchmarking.

```python
self.fast_lm = dspy.LM("openai/gpt-4o-mini", cache=False)
self.smart_lm = dspy.LM("openai/gpt-4o", cache=False)

# In forward():
with dspy.context(lm=self.fast_lm):
    facts1 = self.extract(document=doc1).facts
with dspy.context(lm=self.smart_lm):
    comparison = self.compare(...).comparison
```

### Optimization Philosophy

**plait**: Runtime optimization through backward passes. Feedback flows through
the graph and an LLM synthesizes parameter updates.

```python
# Training loop
module.train()
optimizer = SFAOptimizer(module.parameters())

for example in training_data:
    output = await module(example["input"])
    feedback = await loss_fn(output, target=example["target"])
    await feedback.backward()

await optimizer.step()  # LLM reasons about improvements
```

**DSPy**: Compile-time optimization using teleprompters. The framework finds
good few-shot examples before deployment.

```python
from dspy.teleprompt import BootstrapFewShot

def metric(example, pred, trace=None):
    return example.answer == pred.comparison

teleprompter = BootstrapFewShot(metric=metric)
compiled = teleprompter.compile(pipeline, trainset=examples)
compiled.save("optimized.json")
```

### Execution Model

**plait**: Async-first with automatic parallelism from data flow.

```python
result = await pipeline(doc1, doc2)
# Independent operations run in parallel automatically
```

**DSPy**: Synchronous by default.

```python
result = pipeline(doc1=doc1, doc2=doc2)
# Operations run sequentially
```

## When to Choose Each

### Choose plait when:

- You want **automatic parallel execution** without explicit concurrency management
- You want **runtime optimization** based on feedback during execution
- You need **multi-model pipelines** with different models per step
- **Async execution** and high throughput are important
- You prefer **explicit prompts** over generated ones

### Choose DSPy when:

- You want **compile-time optimization** with few-shot examples
- You prefer **declarative signatures** over explicit prompts
- Built-in **reasoning patterns** (ChainOfThought, ReAct) fit your use case
- **Metric-driven optimization** matches your evaluation approach
