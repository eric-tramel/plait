# plait Examples

Focused examples demonstrating the plait API. Each example covers one concept.

## Examples

| Example | Description | Requires API Key |
|---------|-------------|------------------|
| [01_module.py](01_module.py) | Module, Parameter, composition | No |
| [02_llm_pipeline.py](02_llm_pipeline.py) | LLMInference, sequential/parallel patterns | No |
| [03_tracing.py](03_tracing.py) | DAG capture and visualization | No |
| [04_execution.py](04_execution.py) | run(), bind(), ExecutionSettings, batch | No |
| [05_optimization.py](05_optimization.py) | Backward pass, prompt optimization | Yes |

## Running Examples

```bash
# Run a specific example
python examples/01_module.py

# Run all examples (except optimization which needs API key)
for f in examples/0[1-4]*.py; do python "$f"; done

# For optimization example
export OPENAI_API_KEY=your-key
python examples/05_optimization.py
```

## Quick Reference

### Module Basics (01_module.py)

```python
from plait.module import Module
from plait.parameter import Parameter

class MyPipeline(Module):
    def __init__(self):
        super().__init__()
        self.prompt = Parameter("Be helpful.", requires_grad=True)
        self.child = AnotherModule()

    def forward(self, text: str) -> str:
        return self.child(text)
```

### LLM Pipelines (02_llm_pipeline.py)

```python
from plait.module import LLMInference, Module

class Summarizer(Module):
    def __init__(self):
        super().__init__()
        self.llm = LLMInference(
            alias="fast",  # Bound to endpoint at runtime
            system_prompt="Summarize concisely.",
            temperature=0.3,
        )

    def forward(self, text: str) -> str:
        return self.llm(text)
```

### Execution (04_execution.py)

```python
from plait.execution.executor import run
from plait.execution.context import ExecutionSettings

# Option 1: Explicit run()
result = await run(pipeline, "input", resources=config)

# Option 2: bind() for direct await
pipeline = MyPipeline().bind(resources=config)
result = await pipeline("input")
results = await pipeline(["a", "b", "c"])  # Batch

# Option 3: ExecutionSettings context
async with ExecutionSettings(resources=config):
    result1 = await pipeline1("input")
    result2 = await pipeline2("input")
```

### Optimization (05_optimization.py)

```python
from plait.optimization import SFAOptimizer, LLMRubricLoss

optimizer = SFAOptimizer(pipeline.parameters())
loss_fn = LLMRubricLoss(criteria="...", rubric=[...], alias="judge")

pipeline.train()  # Enable training mode
output = await pipeline(query)  # Returns Value with tape ids
loss_val = await loss_fn(output)
await output.backward(grad=loss_val, optimizer=optimizer)
await optimizer.step()
pipeline.eval()  # Back to inference mode
```

## Resource Configuration

```python
from plait.resources.config import ResourceConfig, EndpointConfig

resources = ResourceConfig(
    endpoints={
        "fast": EndpointConfig(
            provider_api="openai",
            model="gpt-4o-mini",
            max_concurrent=20,
        ),
        "smart": EndpointConfig(
            provider_api="openai",
            model="gpt-4o",
            max_concurrent=5,
        ),
    }
)
```
