# Framework Comparison

This section compares plait with other prominent LLM frameworks to help you
choose the right tool for your use case. Each framework has different strengths,
and understanding these differences will help you make an informed decision.

## Overview

plait is a PyTorch-inspired framework for building, executing, and optimizing
LLM inference pipelines. Its key differentiators are:

- **PyTorch-like API**: Familiar `Module`, `forward()`, and `backward()` patterns
- **Automatic DAG capture**: Trace-based graph construction from eager Python code
- **LLM-based optimization**: Backward passes propagate natural language feedback
- **Async-first execution**: Maximum throughput with adaptive backpressure

## Feature Comparison Matrix

| Feature | plait | Pydantic AI | LangGraph | DSPY |
|---------|-------|-------------|-----------|------|
| **Core Paradigm** | PyTorch-like modules | Agent-based | Explicit state graph | Signatures/Modules |
| **DAG Definition** | Implicit (tracing) | Explicit (pydantic-graph) | Explicit (add_node/edge) | Implicit (composition) |
| **Optimization** | LLM backward pass | N/A | N/A | Compile-time prompting |
| **Async Execution** | Native async-first | Native async | Async supported | Sync-first |
| **Type Safety** | Type hints + validation | Pydantic models | TypedDict state | Signature fields |
| **Structured Output** | Dataclass response_format | Native Pydantic | Manual parsing | Field types |
| **Learning/Feedback** | Parameter + backward() | N/A | N/A | Optimizers |
| **Primary Use Case** | Inference optimization | Agent workflows | Stateful workflows | Prompt optimization |

## When to Use Each Framework

### Choose plait when you need:

- **Optimization through feedback**: Your pipeline should learn and improve over time
- **PyTorch familiarity**: You want the same mental model as neural network training
- **Complex DAG execution**: Multiple parallel and sequential LLM calls with dependencies
- **Automatic graph construction**: Write normal Python code, get optimized execution

### Choose Pydantic AI when you need:

- **Agent-based workflows**: Tools, function calling, and agentic patterns
- **Strong Pydantic integration**: Your codebase already uses Pydantic extensively
- **Dependency injection**: Clean testing patterns with injectable dependencies
- **Streaming responses**: First-class support for streaming LLM output

### Choose LangGraph when you need:

- **Complex state machines**: Workflows with conditional branching and loops
- **Human-in-the-loop**: Checkpointing and resumable workflows
- **LangChain ecosystem**: Integration with existing LangChain tools and chains
- **Visual workflow design**: Explicit graph definition for complex flows

### Choose DSPY when you need:

- **Automated prompt engineering**: Let the framework optimize prompts for you
- **Modular prompt patterns**: Chain of thought, ReAct, and other reasoning patterns
- **Dataset-driven optimization**: Improve prompts using evaluation datasets
- **Compilation**: Pre-optimize prompts before deployment

## Complementary Usage

These frameworks are not mutually exclusive. Common combinations include:

1. **DSPY + plait**: Use DSPY to find good initial prompts, then use plait's
   runtime optimization for continued improvement
2. **LangGraph + plait**: Use LangGraph for complex stateful orchestration,
   with plait modules handling individual inference steps

## Detailed Comparisons

For in-depth analysis of each framework, see:

- [plait vs Pydantic AI](pydantic-ai.md) - Agent definition and dependency injection
- [plait vs LangGraph](langgraph.md) - Graph definition and state management
- [plait vs DSPY](dspy.md) - Optimization philosophy and prompt engineering
