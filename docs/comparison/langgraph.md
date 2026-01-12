# plait vs LangGraph

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

## LangGraph Implementation

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


class State(TypedDict):
    text: str
    summary: str
    analysis: str


# Instructions stored as module-level constant (not learnable)
INSTRUCTIONS = "Be concise and highlight key insights."


def summarize(state: State) -> State:
    llm = ChatOpenAI(model="gpt-4o-mini")
    result = llm.invoke(f"Summarize the input text concisely.\n\n{state['text']}")
    return {"summary": result.content}


def analyze(state: State) -> State:
    llm = ChatOpenAI(model="gpt-4o")
    result = llm.invoke(
        f"{INSTRUCTIONS}\n\nAnalyze this summary:\n{state['summary']}"
    )
    return {"analysis": result.content}


# Build explicit graph
graph = StateGraph(State)
graph.add_node("summarize", summarize)
graph.add_node("analyze", analyze)
graph.set_entry_point("summarize")
graph.add_edge("summarize", "analyze")
graph.add_edge("analyze", END)

app = graph.compile()
result = await app.ainvoke({"text": "Your input text..."})
print(result["analysis"])
```

## Key Differences

| Aspect | plait | LangGraph |
|--------|-------|-----------|
| **Structure** | Single `Module` class with `forward()` | `StateGraph` with node functions |
| **Graph definition** | Implicit from code flow | Explicit `add_node()` and `add_edge()` |
| **State passing** | Function arguments and returns | `TypedDict` state object |
| **Model binding** | Aliases in `ResourceConfig` | Direct instantiation per node |
| **Learnable params** | `Parameter` class | Not supported |

### Graph Definition

**plait**: The DAG is captured automatically by tracing `forward()`. Dependencies
are inferred from how values flow through the code.

```python
def forward(self, text: str) -> str:
    summary = self.summarizer(text)      # Node A
    return self.analyzer(summary)         # Node B depends on A
    # Graph: text -> A -> B -> output
```

**LangGraph**: You explicitly declare nodes and edges. The graph structure is
separate from the node logic.

```python
graph.add_node("summarize", summarize)
graph.add_node("analyze", analyze)
graph.add_edge("summarize", "analyze")
graph.add_edge("analyze", END)
```

### State Management

**plait**: Values flow through function arguments and returns. No explicit state
schema required.

**LangGraph**: State is a `TypedDict` that nodes read from and write to. Each
node returns a partial state update.

```python
class State(TypedDict):
    text: str
    summary: str
    analysis: str

def summarize(state: State) -> State:
    # Read from state, return updates
    return {"summary": result.content}
```

### Learnable Parameters

**plait**: The `Parameter` class holds values that can be optimized through
backward passes.

```python
self.instructions = Parameter(
    value="Be concise and highlight key insights.",
    description="Controls the style of analysis output.",
)
```

**LangGraph**: No built-in support for learnable parameters. Configuration values
are static constants or environment variables.

### Conditional Routing

**plait**: Use normal Python conditionals in `forward()`:

```python
def forward(self, text: str) -> str:
    category = self.classifier(text)
    if "technical" in category:
        return self.technical_handler(text)
    return self.general_handler(text)
```

**LangGraph**: Use `add_conditional_edges()` with a routing function:

```python
def route(state: State) -> str:
    if "technical" in state["category"]:
        return "technical_handler"
    return "general_handler"

graph.add_conditional_edges("classifier", route, {
    "technical_handler": "technical_handler",
    "general_handler": "general_handler",
})
```

## When to Choose Each

### Choose plait when:

- You want to **optimize prompts through feedback** over time
- You prefer **implicit graph construction** from Python code
- You need **automatic parallelism** inferred from data dependencies
- You want **centralized resource configuration** separate from module logic

### Choose LangGraph when:

- You need **complex state machines** with many conditional branches
- **Human-in-the-loop** workflows require checkpointing and resumption
- You're in the **LangChain ecosystem** with existing tools and chains
- You want **explicit visual control** over graph structure
- **Durable execution** across process restarts is required
