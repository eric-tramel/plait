# plait vs LangGraph

This document provides a comprehensive comparison between plait and
[LangGraph](https://www.langchain.com/langgraph), helping you understand when
each framework is the better choice for your use case.

## Framework Overview

### LangGraph

LangGraph is a library for building stateful, multi-actor applications with LLMs.
It extends the LangChain ecosystem with:

- **Explicit state graphs**: Define workflows as nodes and edges with conditional routing
- **Stateful execution**: Built-in state management with TypedDict schemas
- **Persistence**: Checkpointing for resumable and human-in-the-loop workflows
- **Streaming**: Multiple streaming modes for real-time output
- **LangChain integration**: Seamless use of LangChain tools, chains, and retrievers

### plait

plait is a PyTorch-inspired framework for LLM inference pipelines, focusing on:

- **Module composition**: Build complex pipelines from reusable components
- **Automatic DAG capture**: Write normal Python code, get optimized execution
- **LLM-based optimization**: Improve pipelines through backward pass feedback
- **Async-first execution**: Maximum throughput with adaptive backpressure

## Workflow Implementation Comparison

### Defining a Basic Workflow

**LangGraph approach** - Explicit StateGraph:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

class State(TypedDict):
    text: str
    summary: str
    analysis: str

def summarize(state: State) -> State:
    llm = ChatOpenAI(model="gpt-4o-mini")
    summary = llm.invoke(f"Summarize: {state['text']}")
    return {"summary": summary.content}

def analyze(state: State) -> State:
    llm = ChatOpenAI(model="gpt-4o")
    analysis = llm.invoke(f"Analyze: {state['summary']}")
    return {"analysis": analysis.content}

# Build explicit graph
graph = StateGraph(State)
graph.add_node("summarize", summarize)
graph.add_node("analyze", analyze)
graph.add_edge("summarize", "analyze")
graph.add_edge("analyze", END)
graph.set_entry_point("summarize")

app = graph.compile()
result = await app.ainvoke({"text": "Long document..."})
```

**plait approach** - Implicit DAG from forward():

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

# DAG is captured automatically from forward()
pipeline = SummarizeAndAnalyze().bind(resources=config)
result = await pipeline("Long document...")
```

### Conditional Routing

**LangGraph approach** - Explicit conditional edges:

```python
from langgraph.graph import StateGraph, END

class State(TypedDict):
    query: str
    category: str
    response: str

def categorize(state: State) -> State:
    llm = ChatOpenAI(model="gpt-4o-mini")
    category = llm.invoke(f"Categorize: {state['query']}")
    return {"category": category.content}

def route_query(state: State) -> str:
    """Conditional routing based on category."""
    if "technical" in state["category"].lower():
        return "technical_support"
    elif "billing" in state["category"].lower():
        return "billing_support"
    return "general_support"

def technical_support(state: State) -> State:
    # Handle technical queries
    return {"response": "Technical response..."}

def billing_support(state: State) -> State:
    # Handle billing queries
    return {"response": "Billing response..."}

def general_support(state: State) -> State:
    # Handle general queries
    return {"response": "General response..."}

graph = StateGraph(State)
graph.add_node("categorize", categorize)
graph.add_node("technical_support", technical_support)
graph.add_node("billing_support", billing_support)
graph.add_node("general_support", general_support)

graph.add_conditional_edges(
    "categorize",
    route_query,
    {
        "technical_support": "technical_support",
        "billing_support": "billing_support",
        "general_support": "general_support",
    }
)
graph.add_edge("technical_support", END)
graph.add_edge("billing_support", END)
graph.add_edge("general_support", END)
graph.set_entry_point("categorize")
```

**plait approach** - Python conditionals in forward():

```python
class SupportRouter(Module):
    def __init__(self):
        super().__init__()
        self.categorizer = LLMInference(
            alias="fast",
            system_prompt="Categorize as: technical, billing, or general",
        )
        self.technical = LLMInference(
            alias="smart",
            system_prompt="You are a technical support specialist.",
        )
        self.billing = LLMInference(
            alias="smart",
            system_prompt="You are a billing support specialist.",
        )
        self.general = LLMInference(
            alias="smart",
            system_prompt="You are a general support agent.",
        )

    def forward(self, query: str) -> str:
        category = self.categorizer(query)

        # Python conditionals work naturally
        if "technical" in str(category).lower():
            return self.technical(query)
        elif "billing" in str(category).lower():
            return self.billing(query)
        else:
            return self.general(query)
```

### Parallel Execution

**LangGraph approach** - Parallel node branches:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    text: str
    technical: str
    business: str
    user: str
    synthesis: str

def technical_analysis(state: State) -> State:
    llm = ChatOpenAI()
    result = llm.invoke(f"Technical perspective: {state['text']}")
    return {"technical": result.content}

def business_analysis(state: State) -> State:
    llm = ChatOpenAI()
    result = llm.invoke(f"Business perspective: {state['text']}")
    return {"business": result.content}

def user_analysis(state: State) -> State:
    llm = ChatOpenAI()
    result = llm.invoke(f"User perspective: {state['text']}")
    return {"user": result.content}

def synthesize(state: State) -> State:
    llm = ChatOpenAI()
    combined = f"{state['technical']}\n{state['business']}\n{state['user']}"
    result = llm.invoke(f"Synthesize: {combined}")
    return {"synthesis": result.content}

graph = StateGraph(State)
graph.add_node("technical", technical_analysis)
graph.add_node("business", business_analysis)
graph.add_node("user", user_analysis)
graph.add_node("synthesize", synthesize)

# Fan-out from start to parallel nodes
graph.set_entry_point("technical")
graph.add_edge("technical", "business")  # Sequential for this example
# For true parallel, use branches or map-reduce patterns

# Fan-in to synthesis
graph.add_edge("user", "synthesize")
graph.add_edge("synthesize", END)
```

**plait approach** - Automatic parallelism from data flow:

```python
class MultiPerspective(Module):
    def __init__(self):
        super().__init__()
        self.technical = LLMInference(alias="llm", system_prompt="Technical view.")
        self.business = LLMInference(alias="llm", system_prompt="Business view.")
        self.user = LLMInference(alias="llm", system_prompt="User view.")
        self.synthesizer = LLMInference(alias="smart", system_prompt="Synthesize.")

    def forward(self, text: str) -> str:
        # These three run in parallel automatically - no shared dependencies
        technical = self.technical(text)
        business = self.business(text)
        user = self.user(text)

        # Fan-in: synthesizer depends on all three
        combined = f"Technical: {technical}\nBusiness: {business}\nUser: {user}"
        return self.synthesizer(combined)
```

## Key Differentiators

### State Management

| Aspect | LangGraph | plait |
|--------|-----------|-------|
| State definition | TypedDict schema | Function arguments/returns |
| State passing | Explicit in graph | Implicit via data flow |
| State mutation | Node returns updates | Immutable values |
| Global state | Built-in | Via module attributes |

**LangGraph** manages state explicitly through TypedDict:

```python
class State(TypedDict):
    messages: list[str]
    current_step: str
    user_data: dict

def my_node(state: State) -> State:
    # Access and update state
    state["messages"].append("New message")
    return {"current_step": "next"}
```

**plait** uses values flowing through the graph:

```python
def forward(self, query: str) -> str:
    # Data flows through as function arguments
    result = self.step1(query)
    return self.step2(result)
```

### Checkpointing and Persistence

**LangGraph** has built-in persistence for resumable workflows:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Enable persistence
memory = SqliteSaver.from_conn_string(":memory:")
app = graph.compile(checkpointer=memory)

# Resume from checkpoint
config = {"configurable": {"thread_id": "user-123"}}
result = await app.ainvoke(state, config=config)

# Get state at any point
snapshot = app.get_state(config)
```

**plait** focuses on execution optimization rather than persistence:

```python
# Checkpointing is for execution recovery, not workflow state
pipeline = Pipeline().bind(
    resources=config,
    checkpoint_dir=Path("./checkpoints"),
)
# Checkpoints help resume failed batch processing
results = await pipeline(large_batch)
```

### Graph Definition Philosophy

**LangGraph**: Explicit graph definition. You declare nodes and edges upfront,
which provides clear visualization but requires more boilerplate.

```python
# Every connection is explicit
graph.add_node("step1", step1_fn)
graph.add_node("step2", step2_fn)
graph.add_edge("step1", "step2")
graph.add_conditional_edges("step2", route_fn, {"a": "step3", "b": "step4"})
```

**plait**: Implicit graph from tracing. Write natural Python code and the
framework captures the DAG.

```python
def forward(self, x):
    a = self.step1(x)
    b = self.step2(a)
    if some_condition(b):
        return self.step3(b)
    return self.step4(b)
# Graph captured automatically by tracing forward()
```

### Streaming

**LangGraph** offers multiple streaming modes:

```python
# Stream node outputs
async for event in app.astream(state):
    print(event)

# Stream tokens from LLM
async for chunk in app.astream_events(state):
    if chunk["event"] == "on_llm_stream":
        print(chunk["data"]["chunk"])
```

**plait** supports streaming for batch results:

```python
async with ExecutionSettings(resources=config, streaming=True):
    async for result in pipeline(batch_inputs):
        if result.ok:
            print(f"Completed {result.index}: {result.output}")
```

### Learning and Optimization

This is plait's key differentiator. **LangGraph does not have built-in
optimization** - graph structure and prompts are static.

**plait** enables continuous improvement through backward passes:

```python
class OptimizablePipeline(Module):
    def __init__(self):
        super().__init__()
        self.instructions = Parameter(
            value="Process the input carefully.",
            description="Processing instructions that can be optimized.",
        )
        self.processor = LLMInference(alias="llm", system_prompt=self.instructions)

    def forward(self, text: str) -> str:
        return self.processor(text)

# Training loop
module.train()
optimizer = SFAOptimizer(module.parameters())

for example in training_data:
    output = await module(example["input"])
    feedback = await loss_fn(output, target=example["target"])
    await feedback.backward()

await optimizer.step()  # Parameters updated based on feedback
```

## Unique plait Strengths

### 1. PyTorch-like API

The familiar pattern reduces learning curve:

```python
# PyTorch neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)

    def forward(self, x):
        return self.fc1(x)

# plait LLM pipeline (same pattern)
class Pipeline(Module):
    def __init__(self):
        super().__init__()
        self.llm = LLMInference(alias="llm")

    def forward(self, text):
        return self.llm(text)
```

### 2. Automatic DAG Capture

No manual graph construction:

```python
class ComplexPipeline(Module):
    def forward(self, text: str) -> dict:
        # Framework traces this and builds the DAG automatically
        summary = self.summarizer(text)           # Node A
        keywords = self.extractor(summary)        # Node B (depends on A)
        sentiment = self.analyzer(text)           # Node C (parallel with A)
        return {"summary": summary, "keywords": keywords, "sentiment": sentiment}

# Resulting DAG:
#   Input -> Summarizer (A) -> Extractor (B)
#        \-> Analyzer (C)
```

### 3. LLM-based Backward Pass

Optimize using natural language feedback:

```python
# Define what "good" looks like
loss_fn = LLMRubricLoss(
    criteria="helpfulness and clarity",
    rubric=[
        RubricLevel(1, "Poor", "Unhelpful or confusing"),
        RubricLevel(5, "Excellent", "Clear and actionable"),
    ],
)

# Feedback flows backward through the graph
feedback = await loss_fn(output)
await feedback.backward()

# Optimizer updates parameters based on accumulated feedback
await optimizer.step()
```

### 4. Async-first with Adaptive Backpressure

Built for production throughput:

```python
# When hitting rate limits:
# 1. Re-queue the task
# 2. Adjust rate limiter
# 3. No dropped work

pipeline = Pipeline().bind(resources=config, max_concurrent=100)
results = await pipeline(batch_of_10000_inputs)  # Handles backpressure
```

## When to Choose Each

### Choose LangGraph when:

- You need **complex state machines** with many conditional branches
- **Human-in-the-loop** workflows require checkpointing and resumption
- You're in the **LangChain ecosystem** with existing tools and chains
- You want **visual workflow definition** with explicit graph structure
- **Token-level streaming** from LLM responses is required
- You need **durable execution** across process restarts

### Choose plait when:

- You want to **optimize prompts through feedback** over time
- You prefer **PyTorch-like patterns** for building pipelines
- You need **automatic parallelism** inferred from data flow
- You want **less boilerplate** - just write Python code
- You're building **high-throughput batch processing** pipelines
- **Training-loop style** optimization fits your workflow

## Migration Considerations

### From LangGraph to plait

1. **Nodes become Modules**: Convert node functions to Module subclasses
2. **State becomes function args**: Replace TypedDict state with parameters
3. **Edges become data flow**: Let the DAG capture dependencies automatically
4. **Conditional edges become Python if/else**: Use normal conditionals in forward()
5. **Checkpointing scope changes**: plait checkpoints for batch recovery, not state

### From plait to LangGraph

1. **Modules become nodes**: Convert Module classes to node functions
2. **Define State schema**: Create TypedDict for all intermediate values
3. **Add explicit edges**: Declare all connections with add_edge()
4. **Add conditional edges**: Replace Python conditionals with route functions
5. **Lose optimization**: LangGraph doesn't have backward pass support

## Comparison Summary

| Criterion | LangGraph | plait |
|-----------|-----------|-------|
| Best for | Stateful workflows | Pipeline optimization |
| Graph definition | Explicit (nodes + edges) | Implicit (tracing) |
| State management | TypedDict schemas | Function arguments |
| Parallelism | Explicit branches | Automatic from data flow |
| Optimization | None built-in | LLM backward pass |
| Persistence | Full checkpointing | Batch recovery |
| Streaming | Multiple modes | Batch result streaming |
| Ecosystem | LangChain tools | PyTorch-like patterns |

Both frameworks are excellent - LangGraph excels at complex stateful
orchestration, while plait excels at building optimizable inference pipelines.
