#!/usr/bin/env python3
"""LLM pipeline definitions.

This example shows how to define LLM-based inference pipelines using
LLMInference modules. These pipelines define the structure of LLM calls
and can be traced to capture their execution graph.

Note: Actual execution of these pipelines requires resource configuration
(Phase 4) to bind LLM aliases to real endpoints. See 05_execution.py for
examples using mock modules that demonstrate execution patterns.

Run with: python examples/03_llm_pipelines.py
"""

from inf_engine.module import InferenceModule, LLMInference
from inf_engine.parameter import Parameter

# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Simple LLM Module
# ─────────────────────────────────────────────────────────────────────────────


class Summarizer(InferenceModule):
    """A simple summarization module."""

    def __init__(self) -> None:
        super().__init__()
        self.llm = LLMInference(
            alias="fast_llm",
            system_prompt="You are a concise summarizer. Summarize the given text in 2-3 sentences.",
            temperature=0.3,
            max_tokens=150,
        )

    def forward(self, text: str) -> str:
        # In production, this would be traced and executed by run()
        return self.llm(text)


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Sequential Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class SummarizeAndAnalyze(InferenceModule):
    """A two-stage pipeline: summarize, then analyze."""

    def __init__(self) -> None:
        super().__init__()
        self.summarizer = LLMInference(
            alias="fast_llm",
            system_prompt="Summarize the text concisely.",
            temperature=0.3,
        )
        self.analyzer = LLMInference(
            alias="smart_llm",
            system_prompt="Analyze the key themes and implications of the summary.",
            temperature=0.7,
        )

    def forward(self, text: str) -> str:
        summary = self.summarizer(text)
        analysis = self.analyzer(summary)
        return analysis


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Parallel Analysis (Fan-out)
# ─────────────────────────────────────────────────────────────────────────────


class MultiPerspectiveAnalysis(InferenceModule):
    """Analyze text from multiple perspectives in parallel."""

    def __init__(self) -> None:
        super().__init__()
        self.technical = LLMInference(
            alias="llm",
            system_prompt="Analyze from a technical/engineering perspective.",
        )
        self.business = LLMInference(
            alias="llm",
            system_prompt="Analyze from a business/strategic perspective.",
        )
        self.user = LLMInference(
            alias="llm",
            system_prompt="Analyze from an end-user experience perspective.",
        )

    def forward(self, text: str) -> dict[str, str]:
        # These three calls can execute in parallel (same input, no dependencies)
        return {
            "technical": self.technical(text),
            "business": self.business(text),
            "user": self.user(text),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Example 4: Fan-in Synthesis
# ─────────────────────────────────────────────────────────────────────────────


class ComprehensiveAnalyzer(InferenceModule):
    """Analyze from multiple perspectives, then synthesize."""

    def __init__(self) -> None:
        super().__init__()
        self.analyzer = MultiPerspectiveAnalysis()
        self.synthesizer = LLMInference(
            alias="smart_llm",
            system_prompt="Synthesize the following perspectives into a cohesive executive summary.",
            temperature=0.5,
            max_tokens=500,
        )

    def forward(self, text: str) -> str:
        perspectives = self.analyzer(text)

        # Format perspectives for synthesis
        combined = "\n\n".join(
            f"## {name.title()} Perspective\n{analysis}"
            for name, analysis in perspectives.items()
        )

        return self.synthesizer(combined)


# ─────────────────────────────────────────────────────────────────────────────
# Example 5: Learnable System Prompts
# ─────────────────────────────────────────────────────────────────────────────


class AdaptiveAssistant(InferenceModule):
    """An assistant with learnable system prompts that can be optimized."""

    def __init__(self) -> None:
        super().__init__()
        # Learnable instruction parameter
        self.instructions = Parameter(
            "You are a helpful assistant. Be concise and accurate.",
            requires_grad=True,
        )
        self.llm = LLMInference(
            alias="assistant_llm",
            system_prompt=self.instructions,  # Pass Parameter directly
            temperature=0.7,
        )

    def forward(self, user_query: str) -> str:
        return self.llm(user_query)


# ─────────────────────────────────────────────────────────────────────────────
# Example 6: Complex Nested Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class DocumentProcessor(InferenceModule):
    """A complex document processing pipeline."""

    def __init__(self) -> None:
        super().__init__()
        # Stage 1: Extract key information
        self.extractor = LLMInference(
            alias="fast_llm",
            system_prompt="Extract the key facts and figures from this document.",
            temperature=0.2,
        )

        # Stage 2: Analyze from multiple angles
        self.multi_analysis = MultiPerspectiveAnalysis()

        # Stage 3: Generate final report
        self.reporter = LLMInference(
            alias="smart_llm",
            system_prompt="Generate a comprehensive report based on the analysis.",
            temperature=0.5,
            max_tokens=1000,
        )

    def forward(self, document: str) -> str:
        # Extract → Multi-analyze → Report
        key_info = self.extractor(document)
        analyses = self.multi_analysis(key_info)

        # Combine analyses for final report
        analysis_text = "\n".join(f"{k}: {v}" for k, v in analyses.items())
        return self.reporter(analysis_text)


# ─────────────────────────────────────────────────────────────────────────────
# Demo: Inspect Pipeline Structure
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("inf-engine: LLM Pipelines Example")
    print("=" * 60)
    print("\nNote: This demo shows pipeline structure and parameters.")
    print("Actual LLM execution requires resource configuration (Phase 4).")
    print("See 05_execution.py for execution patterns with mock modules.\n")

    # Example 1: Simple module
    print("1. Simple LLM Module (Summarizer)")
    print("-" * 40)
    summarizer = Summarizer()
    print(f"   Modules: {len(list(summarizer.modules()))}")
    print(f"   Parameters: {len(list(summarizer.parameters()))}")
    for name, param in summarizer.named_parameters():
        print(f"      {name}: '{param.value[:50]}...'")

    # Example 2: Sequential pipeline
    print("\n2. Sequential Pipeline (SummarizeAndAnalyze)")
    print("-" * 40)
    pipeline = SummarizeAndAnalyze()
    print(f"   Modules: {len(list(pipeline.modules()))}")
    print("   Module tree:")
    for name, module in pipeline.named_modules():
        indent = "   " if name == "" else "      "
        display_name = name if name else "(root)"
        print(f"{indent}{display_name}: {type(module).__name__}")

    # Example 3: Parallel analysis
    print("\n3. Parallel Analysis (MultiPerspectiveAnalysis)")
    print("-" * 40)
    multi = MultiPerspectiveAnalysis()
    print(f"   LLM modules (can run in parallel): {len(list(multi.children()))}")
    print("   System prompts:")
    for name, param in multi.named_parameters():
        print(f"      {name}: '{param.value[:40]}...'")

    # Example 4: Fan-in synthesis
    print("\n4. Fan-in Synthesis (ComprehensiveAnalyzer)")
    print("-" * 40)
    analyzer = ComprehensiveAnalyzer()
    print(f"   Total modules: {len(list(analyzer.modules()))}")
    print(f"   Total parameters: {len(list(analyzer.parameters()))}")
    print("   Module tree:")
    for name, module in analyzer.named_modules():
        indent = "      " * name.count(".") + "   "
        display_name = name if name else "(root)"
        print(f"{indent}{display_name}: {type(module).__name__}")

    # Example 5: Learnable prompts
    print("\n5. Learnable System Prompts (AdaptiveAssistant)")
    print("-" * 40)
    assistant = AdaptiveAssistant()
    print("   Parameters:")
    for name, param in assistant.named_parameters():
        grad_status = "learnable" if param.requires_grad else "fixed"
        print(f"      {name} ({grad_status}): '{param.value[:40]}...'")

    # Example 6: Complex pipeline
    print("\n6. Complex Nested Pipeline (DocumentProcessor)")
    print("-" * 40)
    processor = DocumentProcessor()
    print(f"   Total modules: {len(list(processor.modules()))}")
    print(
        f"   LLMInference modules: {len([m for m in processor.modules() if isinstance(m, LLMInference)])}"
    )
    print(
        f"   Learnable parameters: {len([p for p in processor.parameters() if p.requires_grad])}"
    )
    print("   Module tree:")
    for name, module in processor.named_modules():
        if name == "":
            print(f"   (root): {type(module).__name__}")
        else:
            depth = name.count(".")
            indent = "      " * (depth + 1)
            print(f"{indent}{name.split('.')[-1]}: {type(module).__name__}")

    print("\n" + "=" * 60)
    print("These pipelines can be traced and executed with resource config!")
    print("=" * 60)
