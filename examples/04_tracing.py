#!/usr/bin/env python3
"""Tracing and DAG capture examples.

This example demonstrates how to use the Tracer to capture execution graphs
from module definitions. Tracing converts eager-mode Python code into a
directed acyclic graph (DAG) that can be analyzed and executed efficiently.

Run with: python examples/04_tracing.py
"""

from plait.module import InferenceModule, LLMInference
from plait.parameter import Parameter
from plait.tracing.proxy import Proxy
from plait.tracing.tracer import Tracer

# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Basic Tracing
# ─────────────────────────────────────────────────────────────────────────────


class SimplePipeline(InferenceModule):
    """A simple two-step pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.step1 = LLMInference(alias="fast", system_prompt="Summarize briefly.")
        self.step2 = LLMInference(alias="smart", system_prompt="Analyze deeply.")

    def forward(self, text: str) -> Proxy:
        summary = self.step1(text)
        analysis = self.step2(summary)
        return analysis


def demo_basic_tracing() -> None:
    """Demonstrate basic tracing of a module."""
    print("1. Basic Tracing")
    print("-" * 40)

    # Create the module and tracer
    pipeline = SimplePipeline()
    tracer = Tracer()

    # Trace the module with sample input
    graph = tracer.trace(pipeline, "Some long document text...")

    # Inspect the captured graph
    print(f"   Nodes captured: {len(graph.nodes)}")
    print(f"   Input nodes: {graph.input_ids}")
    print(f"   Output nodes: {graph.output_ids}")

    # Show each node and its dependencies
    print("\n   Graph structure:")
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        deps = node.dependencies if node.dependencies else "(none)"
        print(f"      {node_id}")
        print(f"         module: {node.module_name}")
        print(f"         depends on: {deps}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Parallel Fan-out Pattern
# ─────────────────────────────────────────────────────────────────────────────


class ParallelAnalysis(InferenceModule):
    """Multiple analyzers processing the same input in parallel."""

    def __init__(self) -> None:
        super().__init__()
        self.technical = LLMInference(alias="llm", system_prompt="Technical analysis.")
        self.business = LLMInference(alias="llm", system_prompt="Business analysis.")
        self.user = LLMInference(alias="llm", system_prompt="User perspective.")

    def forward(self, text: str) -> dict[str, Proxy]:
        return {
            "technical": self.technical(text),
            "business": self.business(text),
            "user": self.user(text),
        }


def demo_parallel_tracing() -> None:
    """Demonstrate tracing of parallel (fan-out) patterns."""
    print("\n2. Parallel Fan-out Pattern")
    print("-" * 40)

    module = ParallelAnalysis()
    tracer = Tracer()
    graph = tracer.trace(module, "Product requirements document...")

    print(f"   Nodes: {len(graph.nodes)}")
    print(f"   Outputs: {len(graph.output_ids)} (all independent)")

    # Show that all analyzers depend only on the input (can run in parallel)
    print("\n   Dependency analysis:")
    input_id = graph.input_ids[0]
    for node_id, node in graph.nodes.items():
        if node_id != input_id:
            print(f"      {node_id} depends on: {node.dependencies}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Diamond Pattern (Fan-out + Fan-in)
# ─────────────────────────────────────────────────────────────────────────────


class DiamondPipeline(InferenceModule):
    """Fan-out to parallel analyzers, then fan-in to synthesizer."""

    def __init__(self) -> None:
        super().__init__()
        self.branch_a = LLMInference(alias="fast", system_prompt="Perspective A")
        self.branch_b = LLMInference(alias="fast", system_prompt="Perspective B")
        self.synthesizer = LLMInference(alias="smart", system_prompt="Synthesize")

    def forward(self, text: str) -> Proxy:
        # Fan-out: both branches process the same input
        a_result = self.branch_a(text)
        b_result = self.branch_b(text)
        # Fan-in: synthesizer combines both results
        return self.synthesizer(a_result, b_result)


def demo_diamond_tracing() -> None:
    """Demonstrate tracing of diamond dependency pattern."""
    print("\n3. Diamond Pattern (Fan-out + Fan-in)")
    print("-" * 40)

    module = DiamondPipeline()
    tracer = Tracer()
    graph = tracer.trace(module, "Input document...")

    print(f"   Nodes: {len(graph.nodes)}")
    print(f"   Input: {graph.input_ids}")
    print(f"   Output: {graph.output_ids}")

    # Visualize the diamond structure
    print("\n   Diamond structure:")
    print("      input:input_0")
    print("         |")
    print("    +----+----+")
    print("    |         |")
    print("    v         v")
    print("  branch_a  branch_b")
    print("    |         |")
    print("    +----+----+")
    print("         |")
    print("         v")
    print("    synthesizer")

    # Show the synthesizer's dependencies
    synth_node = graph.nodes[graph.output_ids[0]]
    print(f"\n   Synthesizer depends on: {synth_node.dependencies}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 4: Graph Traversal
# ─────────────────────────────────────────────────────────────────────────────


class LinearChain(InferenceModule):
    """A linear chain of four processing steps."""

    def __init__(self) -> None:
        super().__init__()
        self.step1 = LLMInference(alias="a", system_prompt="Step 1")
        self.step2 = LLMInference(alias="b", system_prompt="Step 2")
        self.step3 = LLMInference(alias="c", system_prompt="Step 3")
        self.step4 = LLMInference(alias="d", system_prompt="Step 4")

    def forward(self, text: str) -> Proxy:
        r1 = self.step1(text)
        r2 = self.step2(r1)
        r3 = self.step3(r2)
        r4 = self.step4(r3)
        return r4


def demo_graph_traversal() -> None:
    """Demonstrate graph traversal methods."""
    print("\n4. Graph Traversal")
    print("-" * 40)

    module = LinearChain()
    tracer = Tracer()
    graph = tracer.trace(module, "Input...")

    # Topological order - valid execution sequence
    print("   Topological order (valid execution sequence):")
    for i, node_id in enumerate(graph.topological_order(), 1):
        print(f"      {i}. {node_id}")

    # Ancestors - what must complete before a node can run
    print("\n   Ancestors of LLMInference_3 (must complete first):")
    ancestors = graph.ancestors("LLMInference_3")
    for node_id in sorted(ancestors):
        print(f"      - {node_id}")

    # Descendants - what depends on a node (for failure cascading)
    print("\n   Descendants of LLMInference_2 (affected if it fails):")
    descendants = graph.descendants("LLMInference_2")
    for node_id in sorted(descendants):
        print(f"      - {node_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 5: Inspecting Node Details
# ─────────────────────────────────────────────────────────────────────────────


class DetailedModule(InferenceModule):
    """Module with parameters for detailed inspection."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt = Parameter(
            "Custom instruction",
            description="Custom instruction parameter",
            requires_grad=True,
        )
        self.llm = LLMInference(
            alias="assistant",
            system_prompt=self.prompt,
            temperature=0.7,
            max_tokens=500,
        )

    def forward(self, query: str, context: str) -> Proxy:
        # Multiple inputs are captured as separate input nodes
        combined = f"Context: {context}\nQuery: {query}"
        return self.llm(combined)


def demo_node_inspection() -> None:
    """Demonstrate detailed node inspection."""
    print("\n5. Node Inspection")
    print("-" * 40)

    module = DetailedModule()
    tracer = Tracer()
    graph = tracer.trace(module, "What is X?", context="X is a concept...")

    print("   Input nodes:")
    for input_id in graph.input_ids:
        node = graph.nodes[input_id]
        # Access the stored input value
        from plait.tracing.tracer import InputNode

        if isinstance(node.module, InputNode):
            value = node.module.value
            print(f"      {input_id}: '{value[:30]}...'")

    print("\n   Parameters captured:")
    for name, param in graph.parameters.items():
        grad = "learnable" if param.requires_grad else "fixed"
        print(f"      {name} ({grad}): '{param.value[:30]}...'")

    # Inspect the LLM node
    print("\n   LLM node details:")
    for node_id, node in graph.nodes.items():
        if isinstance(node.module, LLMInference):
            print(f"      ID: {node_id}")
            print(f"      Alias: {node.module.alias}")
            print(f"      Temperature: {node.module.temperature}")
            print(f"      Max tokens: {node.module.max_tokens}")
            print(f"      Dependencies: {node.dependencies}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 6: Multiple Independent Traces
# ─────────────────────────────────────────────────────────────────────────────


def demo_multiple_traces() -> None:
    """Demonstrate that traces are independent."""
    print("\n6. Multiple Independent Traces")
    print("-" * 40)

    module = SimplePipeline()
    tracer = Tracer()

    # Trace with different inputs
    graph1 = tracer.trace(module, "First document")
    graph2 = tracer.trace(module, "Second document")

    print("   Each trace captures the full graph independently:")
    print(f"      Graph 1: {len(graph1.nodes)} nodes")
    print(f"      Graph 2: {len(graph2.nodes)} nodes")

    # Show that input values differ
    from plait.tracing.tracer import InputNode

    input1 = graph1.nodes["input:input_0"].module
    input2 = graph2.nodes["input:input_0"].module
    if isinstance(input1, InputNode) and isinstance(input2, InputNode):
        print(f"\n   Graph 1 input: '{input1.value}'")
        print(f"   Graph 2 input: '{input2.value}'")

    # Graphs are independent objects
    print(f"\n   Graphs are independent: {graph1 is not graph2}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 7: Complex Multi-Stage Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class ComplexPipeline(InferenceModule):
    """A multi-stage pipeline with fan-out and fan-in.

    Structure:
        input -> preprocess -> [analyze_a, analyze_b] -> synthesize
    """

    def __init__(self) -> None:
        super().__init__()
        self.preprocess = LLMInference(alias="fast", system_prompt="Clean input")
        self.analyze_a = LLMInference(alias="llm", system_prompt="Perspective A")
        self.analyze_b = LLMInference(alias="llm", system_prompt="Perspective B")
        self.synthesize = LLMInference(alias="smart", system_prompt="Synthesize")

    def forward(self, text: str) -> Proxy:
        # Stage 1: Preprocess
        cleaned = self.preprocess(text)
        # Stage 2: Parallel analysis (fan-out from cleaned)
        result_a = self.analyze_a(cleaned)
        result_b = self.analyze_b(cleaned)
        # Stage 3: Synthesize (fan-in)
        return self.synthesize(result_a, result_b)


def demo_complex_pipeline() -> None:
    """Demonstrate tracing of a complex multi-stage pipeline."""
    print("\n7. Complex Multi-Stage Pipeline")
    print("-" * 40)

    module = ComplexPipeline()
    tracer = Tracer()
    graph = tracer.trace(module, "Complex input document...")

    print(f"   Total nodes: {len(graph.nodes)}")
    print(f"   Total parameters: {len(graph.parameters)}")

    print("\n   Execution order:")
    for i, node_id in enumerate(graph.topological_order(), 1):
        node = graph.nodes[node_id]
        deps = len(node.dependencies)
        print(f"      {i}. {node_id} ({deps} dependencies)")

    # Show the final node's full ancestry
    final_node_id = graph.output_ids[0]
    all_ancestors = graph.ancestors(final_node_id)
    print(f"\n   Final node depends on {len(all_ancestors)} other nodes")


# ─────────────────────────────────────────────────────────────────────────────
# Example 8: Graph Visualization
# ─────────────────────────────────────────────────────────────────────────────


def demo_graph_visualization() -> None:
    """Demonstrate graph visualization with DOT format output."""
    print("\n8. Graph Visualization (DOT Format)")
    print("-" * 40)

    from plait.graph import visualize_graph

    # Create a diamond pattern pipeline for visualization
    module = DiamondPipeline()
    tracer = Tracer()
    graph = tracer.trace(module, "Input document...")

    # Generate DOT format output
    dot = visualize_graph(graph)

    print("   DOT format output (can be rendered with Graphviz):\n")
    for line in dot.split("\n"):
        print(f"      {line}")

    print("\n   To render this graph:")
    print("      1. Save the DOT output to a file: graph.dot")
    print("      2. Run: dot -Tpng graph.dot -o graph.png")
    print("      3. Or use an online viewer like viz-js.com")

    print("\n   Node shapes in the visualization:")
    print("      • box = input nodes")
    print("      • doubleoctagon = output nodes")
    print("      • ellipse = intermediate nodes")


# ─────────────────────────────────────────────────────────────────────────────
# Run all demos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("plait: Tracing and DAG Capture Examples")
    print("=" * 60)

    demo_basic_tracing()
    demo_parallel_tracing()
    demo_diamond_tracing()
    demo_graph_traversal()
    demo_node_inspection()
    demo_multiple_traces()
    demo_complex_pipeline()
    demo_graph_visualization()

    print("\n" + "=" * 60)
    print("Tracing captures the execution graph for async execution!")
    print("=" * 60)
