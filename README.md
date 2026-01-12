# plait

**A PyTorch-inspired framework for building, executing, and optimizing LLM inference pipelines.**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/eric-tramel/plait/actions/workflows/ci.yml/badge.svg)](https://github.com/eric-tramel/plait/actions/workflows/ci.yml)
![coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Feric-tramel%2F7a82abe8c7223509141ec19827a441cf%2Fraw%2Fcoverage.json)

---

plait brings the familiar PyTorch programming model to compound AI systems. Define your LLM pipelines as modules with `forward()` methods, trace them into execution DAGs, and run them with automatic concurrency, backpressure, and resource management.

## Why plait?

Most LLM applications are **systems**, not single calls: they chain multiple LLM invocations, pass structured data between steps, run verifiers, and need consistent handling for retries and rate limits. In plain Python, this complexity leaks everywhere.

plait moves that complexity into a shared runtime:

1. **Write normal module composition** in `forward()` - no async boilerplate
2. **Trace it into a DAG** - dependencies discovered automatically
3. **Execute with a scheduler** - concurrent I/O, rate limiting, retries handled for you
4. **Optimize with feedback** - backward passes propagate feedback to improve prompts

### Framework Comparison

| Feature | plait | DSPy | LangGraph | Pydantic AI |
|---------|:-----:|:----:|:---------:|:-----------:|
| **Automatic parallelism** | ✅ | ❌ | ❌ | ❌ |
| **Implicit graph definition** | ✅ | ✅ | ❌ | ❌ |
| **Runtime optimization** | ✅ | ❌ | ❌ | ❌ |
| **Multi-model pipelines** | ✅ | ✅ | ✅ | ✅ |
| **Async-first execution** | ✅ | ❌ | ✅ | ✅ |
| **PyTorch-like API** | ✅ | ❌ | ❌ | ❌ |
| **Learnable parameters** | ✅ | ✅ | ❌ | ❌ |

### Benchmark: Extract-and-Compare Pipeline

Real-world performance on a fan-out workflow (2 parallel extractions + 1 comparison):

| Framework | Time | Memory | Notes |
|-----------|------|--------|-------|
| **plait** | **6.9s** | **0.4 MB** | Automatic parallel execution |
| Pydantic AI | 8.7s | 17.6 MB | Requires manual `asyncio.gather()` |
| LangGraph | 10.1s | 26.2 MB | Requires explicit `Send()` config |
| DSPy | 13.4s | 76.0 MB | Sequential execution only |

plait is **up to 2x faster** and uses **up to 99% less memory** than alternatives. [See detailed comparisons →](docs/comparison/)

## Features

- **PyTorch-like API**: `Module` with `forward()` and `backward()` methods
- **Automatic DAG capture**: Trace-based graph construction from eager-mode code
- **Async execution**: Maximum throughput with adaptive backpressure and rate limiting
- **Resource management**: Decouple module definitions from endpoint configuration
- **LLM-based optimization**: Backward passes that propagate feedback to update prompts
- **Execution profiling**: Chrome Trace Format export for performance visualization

## Installation

```bash
# Install with uv (recommended)
uv add plait

# Or with pip
pip install plait
```

**Requirements**: Python 3.13+

## Quick Start

Define a pipeline as a module composition:

```python
from plait import Module, LLMInference, Parameter
from plait.resources import OpenAIEndpointConfig, ResourceConfig


class SummarizeAndAnalyze(Module):
    """A two-stage pipeline: summarize, then analyze."""

    def __init__(self):
        super().__init__()
        # Learnable instruction that can be optimized via backward passes
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


# Configure OpenAI endpoints separately from module definition
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

# Bind resources to the pipeline, then execute
pipeline = SummarizeAndAnalyze().bind(resources=resources)
result = await pipeline("Your input text...")
```

The pipeline is traced into a DAG, and the scheduler runs nodes concurrently where dependencies allow. Independent branches execute in parallel without manual `asyncio.gather()` calls.

## Core Concepts

| PyTorch | plait | Purpose |
|---------|-------|---------|
| `nn.Module` | `Module` | Base class for operations |
| `nn.Parameter` | `Parameter` | Learnable values (prompts, instructions) |
| `forward()` | `forward()` | Define computation |
| `backward()` | `backward()` | Propagate feedback |
| `torch.fx.Tracer` | `Tracer` | Capture computation graph |
| `torch.optim.*` | `Optimizer` | Update parameters |

### Module

The base class for all operations. Compose modules by assigning them as attributes:

```python
class DocumentProcessor(Module):
    def __init__(self):
        super().__init__()
        self.extractor = LLMInference(alias="fast", system_prompt="Extract key facts.")
        self.analyzer = MultiPerspectiveAnalysis()  # Another Module
        self.reporter = LLMInference(alias="smart", system_prompt="Write a report.")

    def forward(self, document: str) -> str:
        facts = self.extractor(document)
        analyses = self.analyzer(facts)
        return self.reporter(str(analyses))
```

### LLMInference

The atomic unit for LLM API calls. Uses aliases that are bound to endpoints at execution time:

```python
llm = LLMInference(
    alias="reasoning",           # Bound to endpoint config at runtime
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=500,
)
```

### Parameter

Learnable values (typically prompts or instructions) that can be optimized:

```python
instructions = Parameter(
    value="Be concise and accurate.",
    description="System instructions for the assistant.",
    requires_grad=True,  # Enable optimization
)
```

### ResourceConfig

Decouple module definitions from infrastructure. The same pipeline can run against different endpoints:

```python
from plait.resources import (
    OpenAIEndpointConfig,
    AnthropicEndpointConfig,
    EndpointConfig,
    ResourceConfig,
)

# Development: use cheaper models
dev_resources = ResourceConfig(
    endpoints={
        "fast": OpenAIEndpointConfig(model="gpt-4o-mini", max_concurrent=5),
        "smart": OpenAIEndpointConfig(model="gpt-4o-mini", max_concurrent=5),
    }
)

# Production: use appropriate models with rate limiting
prod_resources = ResourceConfig(
    endpoints={
        "fast": OpenAIEndpointConfig(
            model="gpt-4o-mini",
            max_concurrent=50,
            rate_limit=1000.0,  # requests per minute
        ),
        "smart": OpenAIEndpointConfig(
            model="gpt-4o",
            max_concurrent=20,
            rate_limit=500.0,
        ),
    }
)

# Self-hosted models with OpenAI-compatible API (vLLM, TGI, etc.)
local_resources = ResourceConfig(
    endpoints={
        "fast": EndpointConfig(
            provider_api="vllm",
            model="mistral-7b",
            base_url="http://vllm.internal:8000/v1",
            max_concurrent=50,
        ),
    }
)

# Bind resources to a pipeline
pipeline = MyPipeline().bind(resources=dev_resources)
result = await pipeline("input text")

# Or use ExecutionSettings for shared resources across multiple pipelines
async with ExecutionSettings(resources=prod_resources):
    result1 = await pipeline1("input")
    result2 = await pipeline2("input")
```

## Examples

The `examples/` directory contains focused, runnable examples:

| Example | Description |
|---------|-------------|
| `01_module.py` | Module, Parameter, and composition |
| `02_llm_pipeline.py` | LLMInference and pipeline patterns |
| `03_tracing.py` | DAG capture and visualization |
| `04_execution.py` | run(), bind(), ExecutionSettings, batch |
| `05_optimization.py` | Backward pass and prompt optimization |

Run an example:

```bash
python examples/01_module.py
```

## Documentation

For detailed architecture and design documentation, see the [`design_docs/`](design_docs/) directory:

- [Architecture Overview](design_docs/architecture.md) - System design and component interactions
- [Module](design_docs/inference_module.md) - Core module system
- [Tracing](design_docs/tracing.md) - How DAGs are captured from code
- [Execution](design_docs/execution.md) - Scheduler, state, and error handling
- [Resources](design_docs/resources.md) - Endpoint configuration and rate limiting
- [Optimization](design_docs/optimization.md) - Feedback propagation and learning

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/eric-tramel/plait.git
cd plait

# Install dependencies with uv
uv sync

# Run all checks
make ci
```

### Commands

```bash
make ci           # Run all checks (lint, types, test)
make lint         # Format and lint with ruff
make types        # Type check with ty
make test         # Run all pytest tests
make test-unit    # Run unit tests only
```

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

## License

Apache-2.0 License - see [LICENSE](LICENSE) for details.
