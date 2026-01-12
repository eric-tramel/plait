# plait vs DSPY

This document provides a comprehensive comparison between plait and
[DSPY](https://dspy.ai/), helping you understand when each framework is the
better choice for your use case.

## Framework Overview

### DSPY

DSPY (Declarative Self-improving Python) is a framework for programming with
foundation models. It focuses on:

- **Signatures**: Declarative input/output specifications for LLM tasks
- **Modules**: Composable building blocks like ChainOfThought, ReAct, ProgramOfThought
- **Optimizers**: Automatic prompt optimization using training examples
- **Compilation**: Pre-deployment optimization for production performance
- **Assertions**: Runtime constraints and validation for LLM outputs

### plait

plait is a PyTorch-inspired framework for LLM inference pipelines, focusing on:

- **Module composition**: Build complex pipelines from reusable components
- **Automatic DAG capture**: Write normal Python code, get optimized execution
- **LLM-based optimization**: Improve pipelines through backward pass feedback
- **Async-first execution**: Maximum throughput with adaptive backpressure

Both frameworks share a focus on optimization, but take fundamentally different
approaches.

## Workflow Implementation Comparison

### Defining an LLM Task

**DSPY approach** - Signatures and Modules:

```python
import dspy

class Summarize(dspy.Signature):
    """Summarize the input text concisely."""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()

class SummarizeAndAnalyze(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought(Summarize)
        self.analyze = dspy.ChainOfThought("summary -> analysis")

    def forward(self, text: str) -> str:
        summary = self.summarize(text=text).summary
        analysis = self.analyze(summary=summary).analysis
        return analysis

# Configure LLM
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

# Usage
pipeline = SummarizeAndAnalyze()
result = pipeline(text="Long document...")
```

**plait approach** - Modules with LLMInference:

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

# Usage
pipeline = SummarizeAndAnalyze().bind(resources=config)
result = await pipeline("Long document...")
```

### Chain of Thought Reasoning

**DSPY approach** - Built-in ChainOfThought module:

```python
class QuestionAnswer(dspy.Signature):
    """Answer the question with step-by-step reasoning."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# ChainOfThought adds reasoning automatically
cot = dspy.ChainOfThought(QuestionAnswer)
result = cot(question="What is 2 + 2?")
# result.reasoning contains the thought process
# result.answer contains the final answer
```

**plait approach** - Explicit reasoning in prompt or module:

```python
from dataclasses import dataclass

@dataclass
class ReasonedAnswer:
    reasoning: str
    answer: str

class QuestionAnswer(Module):
    def __init__(self):
        super().__init__()
        self.llm = LLMInference(
            alias="smart",
            system_prompt="""Answer the question with step-by-step reasoning.

First, explain your thought process.
Then, provide the final answer.""",
            response_format=ReasonedAnswer,
        )

    def forward(self, question: str) -> ReasonedAnswer:
        return self.llm(question)
```

### Retrieval-Augmented Generation (RAG)

**DSPY approach** - Retrieve module:

```python
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM

class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str) -> str:
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question).answer

# Configure retriever
retriever = ChromadbRM(collection_name="docs", persist_directory="./chroma")
dspy.configure(rm=retriever)
```

**plait approach** - Compose with retrieval logic:

```python
class RAG(Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever  # External retriever
        self.generator = LLMInference(
            alias="smart",
            system_prompt="Answer based on the provided context.",
        )

    def forward(self, question: str) -> str:
        # Retrieval happens outside the traced graph
        passages = self.retriever.search(question, k=3)
        context = "\n".join(passages)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        return self.generator(prompt)
```

## Optimization Philosophy

This is the most significant difference between the two frameworks.

### DSPY: Compile-time Optimization

DSPY optimizes prompts **before deployment** using training examples:

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

class QAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str) -> str:
        return self.qa(question=question).answer

# Training data
trainset = [
    dspy.Example(question="What is 2+2?", answer="4"),
    dspy.Example(question="Capital of France?", answer="Paris"),
    # ... more examples
]

# Define a metric
def exact_match(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# Compile (optimize prompts)
teleprompter = BootstrapFewShot(metric=exact_match)
compiled_qa = teleprompter.compile(QAModule(), trainset=trainset)

# Save optimized prompts
compiled_qa.save("optimized_qa.json")
```

The compiler:

1. Runs examples through the module
2. Identifies successful patterns
3. Generates optimized few-shot examples
4. Produces a static, optimized module

### plait: Runtime Optimization

plait optimizes parameters **during execution** using backward passes:

```python
from plait import Module, LLMInference, Parameter
from plait.optimization import SFAOptimizer, LLMRubricLoss

class QAModule(Module):
    def __init__(self):
        super().__init__()
        self.instructions = Parameter(
            value="Answer the question accurately.",
            description="Instructions for answering questions.",
        )
        self.llm = LLMInference(alias="llm", system_prompt=self.instructions)

    def forward(self, question: str) -> str:
        return self.llm(question)

# Training data
trainset = [
    {"input": "What is 2+2?", "target": "4"},
    {"input": "Capital of France?", "target": "Paris"},
]

# Configure loss and optimizer
loss_fn = LLMRubricLoss(criteria="accuracy", rubric=[...])
optimizer = SFAOptimizer(module.parameters())

# Training loop (runs at any time, not just deployment)
module.train()
for example in trainset:
    output = await module(example["input"])
    feedback = await loss_fn(output, target=example["target"])
    await feedback.backward()

await optimizer.step()  # Update parameters using LLM reasoning
```

The optimizer:

1. Accumulates natural language feedback
2. Uses an LLM to reason about improvements
3. Generates updated parameter values
4. Can run continuously, not just at compile time

## Key Differentiators

### Optimization Approach

| Aspect | DSPY | plait |
|--------|------|-------|
| When | Compile-time (before deployment) | Runtime (anytime) |
| Method | Few-shot example selection | LLM-based parameter updates |
| Input | Training examples | Natural language feedback |
| Output | Optimized prompts with examples | Updated parameter strings |
| Continuous | Requires recompilation | Continuous improvement |

### Module Patterns

**DSPY** provides built-in reasoning patterns:

```python
# Chain of Thought
dspy.ChainOfThought(signature)

# ReAct (Reasoning + Acting)
dspy.ReAct(signature, tools=[...])

# Program of Thought
dspy.ProgramOfThought(signature)

# Multi-Chain Comparison
dspy.MultiChainComparison(signature, M=3)
```

**plait** uses composable modules:

```python
# Sequential
class Pipeline(Module):
    def forward(self, x):
        return self.step2(self.step1(x))

# Parallel fan-out
class Parallel(Module):
    def forward(self, x):
        return {"a": self.a(x), "b": self.b(x)}

# Conditional
class Conditional(Module):
    def forward(self, x):
        if self.classifier(x):
            return self.branch_a(x)
        return self.branch_b(x)
```

### Assertions and Constraints

**DSPY** has built-in assertion support:

```python
import dspy
from dspy.primitives.assertions import assert_transform_module

class ValidatedQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str) -> str:
        result = self.qa(question=question)
        # Assertion - will retry if fails
        dspy.Assert(
            len(result.answer) > 10,
            "Answer must be detailed (>10 characters)"
        )
        return result.answer

# Wrap with assertion handling
validated = assert_transform_module(ValidatedQA())
```

**plait** uses loss functions for validation:

```python
from plait.optimization import VerifierLoss, CompositeLoss

# Programmatic verification
length_check = VerifierLoss(
    verifier=lambda out: (
        len(out) > 10,
        "Answer must be detailed (>10 characters)"
    )
)

# Combine with quality assessment
quality_loss = LLMRubricLoss(criteria="helpfulness", rubric=[...])

# Composite loss
loss = CompositeLoss([
    (length_check, 0.3),
    (quality_loss, 0.7),
])
```

### Execution Model

**DSPY** is primarily synchronous:

```python
# Synchronous execution
result = module(question="What is AI?")

# Batch processing
results = [module(question=q) for q in questions]
```

**plait** is async-first:

```python
# Async execution
result = await module("What is AI?")

# Concurrent batch processing
results = await module(["Q1", "Q2", "Q3"])  # Runs in parallel

# Streaming results
async for result in module(large_batch):
    process(result)
```

## Unique plait Strengths

### 1. PyTorch-like API

Familiar patterns for ML practitioners:

```python
# PyTorch
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 20)

    def forward(self, x):
        return self.layer(x)

# plait (same pattern)
class Pipeline(Module):
    def __init__(self):
        super().__init__()
        self.llm = LLMInference(alias="llm")

    def forward(self, text):
        return self.llm(text)

# Same training loop pattern too
optimizer.zero_feedback()
output = await module(input)
feedback = await loss_fn(output)
await feedback.backward()
await optimizer.step()
```

### 2. Automatic DAG Capture

Write natural Python, get optimized execution:

```python
def forward(self, text: str) -> dict:
    # Framework traces this automatically
    summary = self.summarizer(text)
    keywords = self.extractor(summary)
    sentiment = self.analyzer(text)  # Parallel with summarizer
    return {"summary": summary, "keywords": keywords, "sentiment": sentiment}
```

### 3. Natural Language Feedback

Optimization uses human-readable feedback:

```python
# Feedback is natural language, not just metrics
feedback = Feedback(
    content="The response was too formal. Use a friendlier tone.",
    score=0.6,
)
await feedback.backward()  # Propagates through graph

# Parameters updated based on feedback
# "Be helpful" -> "Be helpful and friendly"
```

### 4. Continuous Runtime Optimization

Improve during production, not just at compile time:

```python
# Can optimize based on real user feedback
async def handle_request(query: str, user_feedback: str | None):
    output = await module(query)

    if user_feedback:
        feedback = Feedback(content=user_feedback)
        await feedback.backward()
        # Accumulates for next optimizer.step()

    return output

# Periodic optimization
async def optimize_loop():
    while True:
        await asyncio.sleep(3600)  # Every hour
        await optimizer.step()
```

### 5. Async-first with Backpressure

Built for production throughput:

```python
# Handles rate limits gracefully
pipeline = Pipeline().bind(resources=config, max_concurrent=100)
results = await pipeline(batch_of_10000)  # Adaptive backpressure
```

## Unique DSPY Strengths

### 1. Built-in Reasoning Patterns

Pre-built modules for common patterns:

```python
# Chain of Thought - adds reasoning
dspy.ChainOfThought(signature)

# ReAct - interleaved reasoning and tool use
dspy.ReAct(signature, tools=[search, calculate])

# Program of Thought - generates and executes code
dspy.ProgramOfThought(signature)
```

### 2. Compile-time Optimization

Optimize once, deploy everywhere:

```python
# Compile with examples
compiled = teleprompter.compile(module, trainset=examples)

# Save optimized version
compiled.save("production_module.json")

# Load for deployment (no runtime optimization needed)
production = Module.load("production_module.json")
```

### 3. Metric-driven Optimization

Optimize for specific metrics:

```python
def my_metric(example, prediction, trace=None):
    # Custom metric function
    return score_between_0_and_1

teleprompter = BootstrapFewShot(metric=my_metric)
```

### 4. Assertion System

Runtime validation with retry:

```python
dspy.Assert(condition, message)  # Hard constraint
dspy.Suggest(condition, message)  # Soft constraint
```

## When to Choose Each

### Choose DSPY when:

- You want **pre-deployment optimization** with static prompts
- You need **built-in reasoning patterns** (CoT, ReAct, PoT)
- You have **well-defined training examples** for optimization
- You prefer **synchronous execution** patterns
- You want **metric-driven** rather than feedback-driven optimization
- **Compilation** fits your deployment workflow

### Choose plait when:

- You want **continuous runtime optimization** based on feedback
- You prefer **PyTorch-like patterns** for building pipelines
- You need **async-first execution** with high throughput
- You want **natural language feedback** to drive improvements
- **Automatic parallelism** from data flow is important
- You're building **production inference pipelines** with backpressure

## Migration Considerations

### From DSPY to plait

1. **Signatures become response_format**: Convert Signature fields to dataclass
2. **ChainOfThought becomes explicit prompting**: Add reasoning instructions to prompts
3. **Compile becomes train loop**: Replace compilation with runtime optimization
4. **Examples become training data**: Use examples in training loop instead

```python
# DSPY
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

cot = dspy.ChainOfThought(QA)

# plait equivalent
@dataclass
class QAResponse:
    reasoning: str
    answer: str

class QA(Module):
    def __init__(self):
        super().__init__()
        self.llm = LLMInference(
            alias="llm",
            system_prompt="Think step by step, then answer.",
            response_format=QAResponse,
        )

    def forward(self, question: str) -> QAResponse:
        return self.llm(question)
```

### From plait to DSPY

1. **Modules become DSPY Modules**: Similar class structure
2. **LLMInference becomes Signatures**: Define I/O with fields
3. **Training loop becomes compilation**: Use teleprompters
4. **Feedback becomes metrics**: Convert feedback to metric functions

## Comparison Summary

| Criterion | DSPY | plait |
|-----------|------|-------|
| Best for | Pre-deployment optimization | Runtime optimization |
| Optimization timing | Compile-time | Runtime |
| Optimization input | Examples + metrics | Natural language feedback |
| Reasoning patterns | Built-in (CoT, ReAct) | Explicit in prompts |
| Execution model | Sync-first | Async-first |
| Parallelism | Manual | Automatic from DAG |
| Continuous learning | Requires recompilation | Built-in |
| API style | Signature-based | PyTorch-like |

Both frameworks excel at optimization - DSPY for pre-deployment prompt
engineering, plait for continuous runtime improvement. They can even be
complementary: use DSPY to find good initial prompts, then use plait for
ongoing optimization in production.
