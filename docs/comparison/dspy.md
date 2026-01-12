# plait vs DSPy

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

## DSPy Implementation

```python
import dspy


class Summarize(dspy.Signature):
    """Summarize the input text concisely."""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()


class Analyze(dspy.Signature):
    """Be concise and highlight key insights."""
    summary: str = dspy.InputField()
    analysis: str = dspy.OutputField()


class SummarizeAndAnalyze(dspy.Module):
    def __init__(self):
        super().__init__()
        # Create separate LM instances for each step
        self.fast_lm = dspy.LM('openai/gpt-4o-mini')
        self.smart_lm = dspy.LM('openai/gpt-4o')
        self.summarize = dspy.Predict(Summarize)
        self.analyze = dspy.Predict(Analyze)

    def forward(self, text: str) -> str:
        # Use fast model for summarization
        with dspy.context(lm=self.fast_lm):
            summary = self.summarize(text=text).summary
        # Use smart model for analysis
        with dspy.context(lm=self.smart_lm):
            analysis = self.analyze(summary=summary).analysis
        return analysis


pipeline = SummarizeAndAnalyze()
result = pipeline(text="Your input text...")
```

## Key Differences

| Aspect | plait | DSPy |
|--------|-------|------|
| **Structure** | `Module` with `LLMInference` | `dspy.Module` with `Signature` |
| **Prompts** | Explicit system prompts | Signature docstrings |
| **Multi-model** | Aliases map to different endpoints | `dspy.context(lm=...)` per call |
| **Optimization** | Runtime backward pass | Compile-time teleprompters |
| **Execution** | Async-first | Sync-first |

### Prompt Definition

**plait**: Prompts are explicit strings passed to `LLMInference`. The `Parameter`
class makes prompts learnable.

```python
self.summarizer = LLMInference(
    alias="fast",
    system_prompt="Summarize the input text concisely.",
)
```

**DSPy**: Prompts are derived from `Signature` docstrings and field names. The
framework generates the actual prompt.

```python
class Summarize(dspy.Signature):
    """Summarize the input text concisely."""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()
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
instances as module attributes.

```python
self.fast_lm = dspy.LM('openai/gpt-4o-mini')
self.smart_lm = dspy.LM('openai/gpt-4o')

# In forward():
with dspy.context(lm=self.fast_lm):
    summary = self.summarize(text=text).summary
with dspy.context(lm=self.smart_lm):
    analysis = self.analyze(summary=summary).analysis
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
    return example.answer == pred.analysis

teleprompter = BootstrapFewShot(metric=metric)
compiled = teleprompter.compile(pipeline, trainset=examples)
compiled.save("optimized.json")
```

### Execution Model

**plait**: Async-first with automatic parallelism from data flow.

```python
result = await pipeline("input")
results = await pipeline(["input1", "input2", "input3"])  # Parallel
```

**DSPy**: Synchronous by default.

```python
result = pipeline(text="input")
results = [pipeline(text=t) for t in texts]  # Sequential
```

## When to Choose Each

### Choose plait when:

- You want **runtime optimization** based on feedback during execution
- You need **multi-model pipelines** with different models per step
- **Async execution** and high throughput are important
- You prefer **explicit prompts** over generated ones

### Choose DSPy when:

- You want **compile-time optimization** with few-shot examples
- You prefer **declarative signatures** over explicit prompts
- Built-in **reasoning patterns** (ChainOfThought, ReAct) fit your use case
- **Metric-driven optimization** matches your evaluation approach
