# plait vs LangGraph

This comparison uses the same example—an extract-and-compare pipeline—to show
how each framework approaches the same problem, with a focus on parallel execution.

## The Example: Extract and Compare

A three-stage pipeline that:

1. Takes two documents as input
2. Extracts main facts from both documents (can run in parallel)
3. Compares and contrasts the extracted facts

This workflow highlights **plait's automatic parallel execution** vs LangGraph's explicit fan-out configuration.

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

## LangGraph Implementation

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from langchain_openai import ChatOpenAI


# Reducer function to collect facts from parallel branches
def add_facts(existing: list[str], new: list[str]) -> list[str]:
    return existing + new


class State(TypedDict):
    doc1: str
    doc2: str
    facts: Annotated[list[str], add_facts]
    comparison: str


COMPARISON_STYLE = "Highlight key similarities and differences."


async def extract_facts(state: dict[str, str]) -> dict[str, list[str]]:
    """Extract facts from a single document."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    result = await llm.ainvoke(
        f"Extract the main facts as a bulleted list:\n\n{state['document']}"
    )
    return {"facts": [str(result.content)]}


async def compare_facts(state: State) -> dict[str, str]:
    """Compare and contrast facts from both documents."""
    llm = ChatOpenAI(model="gpt-4o")
    result = await llm.ainvoke(
        f"{COMPARISON_STYLE}\n\n"
        f"Compare and contrast:\n\n"
        f"Document 1:\n{state['facts'][0]}\n\n"
        f"Document 2:\n{state['facts'][1]}"
    )
    return {"comparison": str(result.content)}


def fan_out_to_extractors(state: State) -> list[Send]:
    """Fan out to parallel extraction nodes using Send()."""
    return [
        Send("extract_facts", {"document": state["doc1"]}),
        Send("extract_facts", {"document": state["doc2"]}),
    ]


# Build graph with explicit fan-out
graph = StateGraph(State)
graph.add_node("extract_facts", extract_facts)
graph.add_node("compare_facts", compare_facts)

# Set up fan-out: entry -> parallel extractions -> comparison
graph.set_conditional_entry_point(fan_out_to_extractors)
graph.add_edge("extract_facts", "compare_facts")
graph.add_edge("compare_facts", END)

app = graph.compile()
result = await app.ainvoke({
    "doc1": doc1,
    "doc2": doc2,
    "facts": [],
    "comparison": "",
})
print(result["comparison"])
```

## Key Differences

| Aspect | plait | LangGraph |
|--------|-------|-----------|
| **Structure** | Single `Module` class with `forward()` | `StateGraph` with node functions |
| **Parallel execution** | Automatic from data flow | Explicit `Send()` + reducers |
| **Graph definition** | Implicit from code flow | Explicit `add_node()` and `add_edge()` |
| **State passing** | Function arguments and returns | `TypedDict` state object |
| **Model binding** | Aliases in `ResourceConfig` | Direct instantiation per node |
| **Learnable params** | `Parameter` class | Not supported |

### Parallel Execution

**plait**: Automatically detects that the two extraction calls are independent and
runs them concurrently. No special syntax or configuration needed.

```python
def forward(self, doc1: str, doc2: str) -> str:
    facts1 = self.extractor(doc1)  # These run
    facts2 = self.extractor(doc2)  # in parallel!
    return self.comparer(...)      # Waits for both
```

**LangGraph**: Requires explicit `Send()` for fan-out, plus a reducer function to
collect results from parallel branches.

```python
# Define a reducer to collect parallel results
def add_facts(existing: list[str], new: list[str]) -> list[str]:
    return existing + new

class State(TypedDict):
    facts: Annotated[list[str], add_facts]  # Reducer annotation

# Define fan-out function
def fan_out_to_extractors(state: State) -> list[Send]:
    return [
        Send("extract_facts", {"document": state["doc1"]}),
        Send("extract_facts", {"document": state["doc2"]}),
    ]

# Configure conditional entry point
graph.set_conditional_entry_point(fan_out_to_extractors)
```

### Graph Definition

**plait**: The DAG is captured automatically by tracing `forward()`. Dependencies
are inferred from how values flow through the code.

```python
def forward(self, doc1: str, doc2: str) -> str:
    facts1 = self.extractor(doc1)      # Node A
    facts2 = self.extractor(doc2)      # Node B (parallel with A)
    return self.comparer(f"{facts1}\n{facts2}")  # Node C depends on A and B
    # Graph: doc1 -> A ─┐
    #        doc2 -> B ─┼─> C -> output
```

**LangGraph**: You explicitly declare nodes and edges. The graph structure is
separate from the node logic.

```python
graph.add_node("extract_facts", extract_facts)
graph.add_node("compare_facts", compare_facts)
graph.set_conditional_entry_point(fan_out_to_extractors)
graph.add_edge("extract_facts", "compare_facts")
graph.add_edge("compare_facts", END)
```

### State Management

**plait**: Values flow through function arguments and returns. No explicit state
schema required.

**LangGraph**: State is a `TypedDict` that nodes read from and write to. Each
node returns a partial state update. Parallel execution requires reducer annotations.

```python
class State(TypedDict):
    doc1: str
    doc2: str
    facts: Annotated[list[str], add_facts]  # Reducer for parallel results
    comparison: str

def extract_facts(state: dict[str, str]) -> dict[str, list[str]]:
    # Read from state, return updates
    return {"facts": [result.content]}
```

### Learnable Parameters

**plait**: The `Parameter` class holds values that can be optimized through
backward passes.

```python
self.comparison_style = Parameter(
    value="Highlight key similarities and differences.",
    description="Controls the style of comparison output.",
)
```

**LangGraph**: No built-in support for learnable parameters. Configuration values
are static constants or environment variables.

### Conditional Routing

**plait**: Use normal Python conditionals in `forward()`:

```python
def forward(self, doc1: str, doc2: str) -> str:
    category = self.classifier(doc1)
    if "technical" in category:
        return self.technical_comparer(doc1, doc2)
    return self.general_comparer(doc1, doc2)
```

**LangGraph**: Use `add_conditional_edges()` with a routing function:

```python
def route(state: State) -> str:
    if "technical" in state["category"]:
        return "technical_comparer"
    return "general_comparer"

graph.add_conditional_edges("classifier", route, {
    "technical_comparer": "technical_comparer",
    "general_comparer": "general_comparer",
})
```

## When to Choose Each

### Choose plait when:

- You want **automatic parallel execution** without explicit fan-out configuration
- You want to **optimize prompts through feedback** over time
- You prefer **implicit graph construction** from Python code
- You want **centralized resource configuration** separate from module logic

### Choose LangGraph when:

- You need **complex state machines** with many conditional branches
- **Human-in-the-loop** workflows require checkpointing and resumption
- You're in the **LangChain ecosystem** with existing tools and chains
- You want **explicit visual control** over graph structure
- **Durable execution** across process restarts is required
