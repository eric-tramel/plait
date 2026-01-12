# plait vs Pydantic AI

This document provides a comprehensive comparison between plait and
[Pydantic AI](https://ai.pydantic.dev/), helping you understand when each
framework is the better choice for your use case.

## Framework Overview

### Pydantic AI

Pydantic AI is an agent framework built by the creators of Pydantic. It focuses
on building production-ready AI agents with:

- **Agent-centric design**: First-class support for tools, function calling, and
  agentic workflows
- **Pydantic integration**: Native use of Pydantic models for structured output
  and validation
- **Dependency injection**: Clean separation of concerns for testing and
  configuration
- **Streaming support**: First-class streaming for real-time responses
- **Multi-model support**: Works with OpenAI, Anthropic, Gemini, Groq, Mistral,
  and Ollama

### plait

plait is a PyTorch-inspired framework for LLM inference pipelines, focusing on:

- **Module composition**: Build complex pipelines from reusable components
- **Automatic DAG capture**: Write normal Python code, get optimized execution
- **LLM-based optimization**: Improve pipelines through backward pass feedback
- **Async-first execution**: Maximum throughput with adaptive backpressure

## Workflow Implementation Comparison

### Defining an LLM Task

**Pydantic AI approach** - Agent with tools:

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class SupportResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

support_agent = Agent(
    'openai:gpt-4o',
    result_type=SupportResponse,
    system_prompt="You are a helpful customer support agent.",
)

@support_agent.tool
def lookup_order(order_id: str) -> dict:
    """Look up order details from the database."""
    return {"status": "shipped", "tracking": "1Z999..."}

# Usage
result = await support_agent.run("Where is my order #12345?")
print(result.data.answer)  # Structured SupportResponse
```

**plait approach** - Module with forward():

```python
from plait import Module, LLMInference, Parameter
from dataclasses import dataclass

@dataclass
class SupportResponse:
    answer: str
    confidence: float
    sources: list[str]

class SupportAgent(Module):
    def __init__(self):
        super().__init__()
        self.instructions = Parameter(
            value="You are a helpful customer support agent.",
            description="Agent persona and behavior guidelines.",
        )
        self.llm = LLMInference(
            alias="support",
            system_prompt=self.instructions,
            response_format=SupportResponse,
        )

    def forward(self, query: str) -> SupportResponse:
        return self.llm(query)

# Usage
agent = SupportAgent().bind(resources=config)
result = await agent("Where is my order #12345?")
print(result.answer)  # Structured SupportResponse
```

### Building Complex Workflows

**Pydantic AI approach** - pydantic-graph for explicit workflow:

```python
from pydantic_graph import Graph, Node, End
from pydantic_ai import Agent

summarize_agent = Agent('openai:gpt-4o-mini', system_prompt="Summarize text.")
analyze_agent = Agent('openai:gpt-4o', system_prompt="Analyze themes.")

class SummarizeNode(Node):
    async def run(self, ctx) -> str:
        result = await summarize_agent.run(ctx.state['text'])
        ctx.state['summary'] = result.data
        return 'analyze'

class AnalyzeNode(Node):
    async def run(self, ctx) -> End:
        result = await analyze_agent.run(ctx.state['summary'])
        ctx.state['analysis'] = result.data
        return End()

graph = Graph(nodes=[SummarizeNode(), AnalyzeNode()])
result = await graph.run({'text': 'Long document...'})
```

**plait approach** - implicit DAG from forward():

```python
from plait import Module, LLMInference

class SummarizeAndAnalyze(Module):
    def __init__(self):
        super().__init__()
        self.summarizer = LLMInference(
            alias="fast",
            system_prompt="Summarize the input text concisely.",
        )
        self.analyzer = LLMInference(
            alias="smart",
            system_prompt="Analyze the key themes and implications.",
        )

    def forward(self, text: str) -> str:
        summary = self.summarizer(text)
        analysis = self.analyzer(summary)
        return analysis

# Usage - DAG is captured automatically from forward()
pipeline = SummarizeAndAnalyze().bind(resources=config)
result = await pipeline("Long document...")
```

### Parallel Execution

**Pydantic AI approach** - explicit parallel nodes in pydantic-graph:

```python
from pydantic_graph import Graph, Node, ParallelNode

class ParallelAnalysis(ParallelNode):
    async def run(self, ctx):
        # Define parallel branches
        return [
            ('technical', TechnicalNode()),
            ('business', BusinessNode()),
            ('user', UserNode()),
        ]
```

**plait approach** - automatic parallelism from data flow:

```python
class MultiPerspective(Module):
    def __init__(self):
        super().__init__()
        self.technical = LLMInference(alias="llm", system_prompt="Technical view.")
        self.business = LLMInference(alias="llm", system_prompt="Business view.")
        self.user = LLMInference(alias="llm", system_prompt="User view.")

    def forward(self, text: str) -> dict[str, str]:
        # These run in parallel automatically - no shared dependencies
        return {
            "technical": self.technical(text),
            "business": self.business(text),
            "user": self.user(text),
        }
```

## Key Differentiators

### Structured Output

| Aspect | Pydantic AI | plait |
|--------|-------------|-------|
| Schema definition | Pydantic BaseModel | Python dataclass |
| Validation | Pydantic validators | Type checking |
| Nested models | Full Pydantic support | Nested dataclasses |
| Dynamic schemas | Runtime model creation | Static dataclasses |

**Pydantic AI** excels at complex validation with Pydantic's full validator
ecosystem:

```python
from pydantic import BaseModel, field_validator

class Analysis(BaseModel):
    sentiment: float
    topics: list[str]

    @field_validator('sentiment')
    def validate_sentiment(cls, v):
        if not -1 <= v <= 1:
            raise ValueError('Sentiment must be between -1 and 1')
        return v
```

**plait** uses simpler dataclasses with type hints:

```python
from dataclasses import dataclass

@dataclass
class Analysis:
    sentiment: float
    topics: list[str]
```

### Dependency Injection

**Pydantic AI** has first-class dependency injection for testing:

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

@dataclass
class Dependencies:
    db: Database
    api_key: str

agent = Agent('openai:gpt-4o', deps_type=Dependencies)

@agent.tool
def lookup_user(ctx: RunContext[Dependencies], user_id: str) -> dict:
    return ctx.deps.db.get_user(user_id)

# Testing with mock dependencies
mock_deps = Dependencies(db=MockDatabase(), api_key="test")
result = await agent.run("Find user 123", deps=mock_deps)
```

**plait** uses constructor injection and module composition:

```python
class UserLookup(Module):
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.llm = LLMInference(alias="llm")

    def forward(self, user_id: str) -> dict:
        user_data = self.db.get_user(user_id)
        return self.llm(f"Summarize user: {user_data}")

# Testing
mock_db = MockDatabase()
lookup = UserLookup(db=mock_db).bind(resources=config)
```

### Graph Definition Philosophy

**Pydantic AI (pydantic-graph)**: Explicit graph definition with nodes and edges.
You define the workflow structure upfront.

```python
class ReviewNode(Node):
    async def run(self, ctx) -> str:
        if ctx.state['score'] > 0.8:
            return 'approve'
        return 'escalate'
```

**plait**: Implicit graph from code execution. The DAG is captured by tracing
your forward() method.

```python
def forward(self, text: str) -> str:
    summary = self.summarizer(text)  # Node A
    if len(summary) > 100:           # Conditional logic
        return self.detailed(summary) # Node B
    return self.brief(summary)        # Node C
```

### Learning and Optimization

This is plait's most distinctive feature. **Pydantic AI does not have built-in
optimization** - prompts are static once defined.

**plait** enables continuous improvement through backward passes:

```python
class CustomerSupport(Module):
    def __init__(self):
        super().__init__()
        self.persona = Parameter(
            value="You are a helpful customer support agent.",
            description="Agent persona - can be optimized based on feedback.",
        )
        self.llm = LLMInference(alias="support", system_prompt=self.persona)

    def forward(self, query: str) -> str:
        return self.llm(query)

# Training loop
module.train()
optimizer = SFAOptimizer(module.parameters())

for example in training_data:
    output = await module(example["input"])
    feedback = await loss_fn(output, target=example["target"])
    await feedback.backward()

await optimizer.step()  # Updates persona based on accumulated feedback
```

## Unique plait Strengths

### 1. PyTorch-like API

Developers familiar with PyTorch will immediately understand plait:

```python
# PyTorch pattern
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)

    def forward(self, x):
        return self.layer1(x)

# plait pattern (identical structure)
class Pipeline(Module):
    def __init__(self):
        super().__init__()
        self.summarizer = LLMInference(alias="llm")

    def forward(self, text):
        return self.summarizer(text)
```

### 2. Automatic DAG Capture via Tracing

No need to manually define nodes and edges:

```python
class ComplexPipeline(Module):
    def forward(self, text: str) -> dict:
        # The framework captures this as a DAG automatically
        summary = self.summarizer(text)           # A
        keywords = self.extractor(summary)        # B (depends on A)
        sentiment = self.analyzer(text)           # C (parallel with A)
        return {
            "summary": summary,
            "keywords": keywords,
            "sentiment": sentiment,
        }
        # Captured DAG: Input -> A -> B
        #                    -> C
```

### 3. LLM-based Backward Pass

Optimize prompts using natural language feedback:

```python
# Feedback flows backward through the graph
await feedback.backward()

# Parameters accumulate feedback from multiple samples
# Optimizer synthesizes improvements using LLM reasoning
await optimizer.step()
```

### 4. Async-first with Backpressure

Built for high-throughput production use:

```python
# Adaptive rate limiting when hitting API limits
# Re-queues tasks with adjusted timing
# No dropped work, graceful degradation
pipeline = Pipeline().bind(resources=config, max_concurrent=50)
results = await pipeline(batch_of_1000_inputs)
```

## When to Choose Each

### Choose Pydantic AI when:

- You need **agent-based workflows** with tools and function calling
- Your codebase **already uses Pydantic** extensively
- You want **dependency injection** for clean testing patterns
- **Streaming responses** are critical for your UX
- You need **multi-model orchestration** (different LLM providers)

### Choose plait when:

- You want to **optimize prompts through feedback** over time
- You prefer **PyTorch-like patterns** for building pipelines
- You need **automatic parallel execution** from data flow
- You're building **complex DAG-structured** inference pipelines
- You want **training-loop style** optimization for LLM outputs

## Migration Considerations

### From Pydantic AI to plait

1. **Agents become Modules**: Convert `Agent` classes to `Module` subclasses
2. **Tools become helper methods**: Move tool logic to methods called in forward()
3. **Dependencies become constructor args**: Pass dependencies at construction time
4. **result_type becomes response_format**: Use dataclasses instead of Pydantic models

### From plait to Pydantic AI

1. **Modules become Agents**: Convert `Module` subclasses to `Agent` instances
2. **forward() becomes agent.run()**: Move inference logic to agent configuration
3. **Parameters are lost**: Pydantic AI doesn't support learnable parameters
4. **Implicit DAGs need explicit graphs**: Use pydantic-graph for complex workflows

## Summary

| Criterion | Pydantic AI | plait |
|-----------|-------------|-------|
| Best for | Agents with tools | Pipeline optimization |
| Learning curve | Low (if you know Pydantic) | Low (if you know PyTorch) |
| Optimization | Manual prompt tuning | Automated via backward() |
| Parallelism | Explicit parallel nodes | Automatic from data flow |
| Testing | Dependency injection | Constructor injection |
| Ecosystem | Pydantic, FastAPI | PyTorch-like patterns |

Both frameworks are excellent choices - the right one depends on whether you're
building agentic workflows (Pydantic AI) or optimizable inference pipelines
(plait).
