#!/usr/bin/env python3
"""Tracing: Capturing execution DAGs from module definitions.

Demonstrates:
- Using Tracer to capture computation graphs
- Inspecting nodes and dependencies
- Parallel (fan-out) and diamond (fan-in) patterns
- Graph traversal (topological order, ancestors, descendants)
- Generating DOT format for visualization

Run: python examples/03_tracing.py
"""

from typing import Any

from plait.graph import visualize_graph
from plait.module import LLMInference, Module
from plait.tracing.proxy import Proxy
from plait.tracing.tracer import Tracer

# --- Sequential Pipeline ---


class Sequential(Module):
    """Linear two-step pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.step1 = LLMInference(alias="fast", system_prompt="Summarize.")
        self.step2 = LLMInference(alias="smart", system_prompt="Analyze.")

    def forward(self, text: str) -> Proxy:
        summary = self.step1(text)
        return self.step2(summary)


# --- Parallel Fan-out ---


class Parallel(Module):
    """Multiple analyzers processing the same input."""

    def __init__(self) -> None:
        super().__init__()
        self.a = LLMInference(alias="llm", system_prompt="Perspective A")
        self.b = LLMInference(alias="llm", system_prompt="Perspective B")
        self.c = LLMInference(alias="llm", system_prompt="Perspective C")

    def forward(self, text: str) -> dict[str, Proxy]:
        # All three depend only on input - can run in parallel
        return {"a": self.a(text), "b": self.b(text), "c": self.c(text)}


# --- Diamond Pattern (Fan-out + Fan-in) ---


class Diamond(Module):
    """Fan-out to branches, fan-in to synthesizer."""

    def __init__(self) -> None:
        super().__init__()
        self.branch_a = LLMInference(alias="fast", system_prompt="View A")
        self.branch_b = LLMInference(alias="fast", system_prompt="View B")
        self.synth = LLMInference(alias="smart", system_prompt="Synthesize")

    def forward(self, text: str) -> Proxy:
        a = self.branch_a(text)
        b = self.branch_b(text)
        return self.synth(a, b)  # Waits for both branches


# --- Multi-Stage Pipeline ---


class MultiStage(Module):
    """Complex pipeline: preprocess -> [a, b] -> combine."""

    def __init__(self) -> None:
        super().__init__()
        self.preprocess = LLMInference(alias="fast", system_prompt="Clean")
        self.analyze_a = LLMInference(alias="llm", system_prompt="Aspect A")
        self.analyze_b = LLMInference(alias="llm", system_prompt="Aspect B")
        self.combine = LLMInference(alias="smart", system_prompt="Combine")

    def forward(self, text: str) -> Proxy:
        cleaned = self.preprocess(text)
        a = self.analyze_a(cleaned)  # Fan-out from cleaned
        b = self.analyze_b(cleaned)
        return self.combine(a, b)  # Fan-in


def print_graph_info(name: str, graph: Any) -> None:
    """Print basic graph information."""
    print(f"\n{name}")
    print("-" * 40)
    print(f"   Nodes: {len(graph.nodes)}")
    print(f"   Inputs: {graph.input_ids}")
    print(f"   Outputs: {graph.output_ids}")


if __name__ == "__main__":
    print("=" * 60)
    print("plait: Tracing and DAG Capture")
    print("=" * 60)

    tracer = Tracer()

    # Sequential
    print("\n1. Sequential Pipeline")
    print("-" * 40)
    seq_graph = tracer.trace(Sequential(), "input text")
    print(f"   Nodes: {len(seq_graph.nodes)}")
    print("   Execution order:")
    for i, node_id in enumerate(seq_graph.topological_order(), 1):
        node = seq_graph.nodes[node_id]
        deps = node.dependencies if node.dependencies else "(none)"
        print(f"      {i}. {node_id} <- {deps}")

    # Parallel
    print("\n2. Parallel Fan-out")
    print("-" * 40)
    par_graph = tracer.trace(Parallel(), "input text")
    print(f"   Nodes: {len(par_graph.nodes)}")
    print(f"   Outputs (independent): {len(par_graph.output_ids)}")
    input_id = par_graph.input_ids[0]
    print("   Dependencies:")
    for node_id, node in par_graph.nodes.items():
        if node_id != input_id:
            print(f"      {node_id} <- {node.dependencies}")

    # Diamond
    print("\n3. Diamond Pattern (Fan-out + Fan-in)")
    print("-" * 40)
    diamond_graph = tracer.trace(Diamond(), "input text")
    print(f"   Nodes: {len(diamond_graph.nodes)}")
    print("   Structure:")
    print("      input")
    print("       / \\")
    print("      A   B   <- parallel")
    print("       \\ /")
    print("      synth   <- waits for both")
    synth_node = diamond_graph.nodes[diamond_graph.output_ids[0]]
    print(f"   Synthesizer depends on: {synth_node.dependencies}")

    # Multi-stage
    print("\n4. Multi-Stage Pipeline")
    print("-" * 40)
    multi_graph = tracer.trace(MultiStage(), "input text")
    print(f"   Nodes: {len(multi_graph.nodes)}")
    print("   Topological order:")
    for i, node_id in enumerate(multi_graph.topological_order(), 1):
        node = multi_graph.nodes[node_id]
        deps_count = len(node.dependencies)
        print(f"      {i}. {node_id} ({deps_count} deps)")

    # Graph traversal
    print("\n5. Graph Traversal")
    print("-" * 40)
    # Find ancestors
    output_id = multi_graph.output_ids[0]
    ancestors = multi_graph.ancestors(output_id)
    print(f"   Ancestors of output: {sorted(ancestors)}")

    # Find what depends on preprocessing
    for node_id, _node in multi_graph.nodes.items():
        if "preprocess" in node_id.lower():
            descendants = multi_graph.descendants(node_id)
            print(f"   Descendants of preprocess: {sorted(descendants)}")
            break

    # DOT visualization
    print("\n6. DOT Visualization")
    print("-" * 40)
    dot = visualize_graph(diamond_graph)
    print("   DOT output (for Graphviz):")
    for line in dot.split("\n"):
        print(f"      {line}")

    print("\n   To render: save as graph.dot, run 'dot -Tpng graph.dot -o graph.png'")
    print("   Or paste into https://viz-js.com")

    print("\n" + "=" * 60)
