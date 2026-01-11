#!/usr/bin/env python3
"""LLM Pipelines: Defining inference workflows with LLMInference.

Demonstrates:
- LLMInference module with aliases for endpoint binding
- Sequential pipelines (summarize -> analyze)
- Parallel fan-out (multiple perspectives)
- Fan-in synthesis (combine parallel results)
- Learnable system prompts via Parameter

Note: These pipelines define structure only. To execute with real LLMs,
bind resources and use await. See 04_execution.py.

Run: python examples/02_llm_pipeline.py
"""

from plait.module import LLMInference, Module
from plait.parameter import Parameter

# --- Simple LLM Module ---


class Summarizer(Module):
    """Summarize text concisely."""

    def __init__(self) -> None:
        super().__init__()
        self.llm = LLMInference(
            alias="fast",  # Bound to endpoint at runtime
            system_prompt="Summarize the given text in 2-3 sentences.",
            temperature=0.3,
            max_tokens=150,
        )

    def forward(self, text: str) -> str:
        return self.llm(text)


# --- Sequential Pipeline ---


class SummarizeAndAnalyze(Module):
    """Two-stage pipeline: summarize, then analyze."""

    def __init__(self) -> None:
        super().__init__()
        self.summarizer = LLMInference(
            alias="fast",
            system_prompt="Summarize concisely.",
        )
        self.analyzer = LLMInference(
            alias="smart",  # Use a more capable model
            system_prompt="Analyze key themes and implications.",
            temperature=0.7,
        )

    def forward(self, text: str) -> str:
        summary = self.summarizer(text)
        return self.analyzer(summary)


# --- Parallel Fan-out ---


class MultiPerspective(Module):
    """Analyze from multiple perspectives in parallel."""

    def __init__(self) -> None:
        super().__init__()
        self.technical = LLMInference(
            alias="fast", system_prompt="Analyze from a technical perspective."
        )
        self.business = LLMInference(
            alias="fast", system_prompt="Analyze from a business perspective."
        )
        self.user = LLMInference(
            alias="fast", system_prompt="Analyze from a user experience perspective."
        )

    def forward(self, text: str) -> dict[str, str]:
        # All three process the same input - can run in parallel
        return {
            "technical": self.technical(text),
            "business": self.business(text),
            "user": self.user(text),
        }


# --- Fan-in Synthesis ---


class ComprehensiveAnalyzer(Module):
    """Fan-out to multiple perspectives, fan-in to synthesize."""

    def __init__(self) -> None:
        super().__init__()
        self.perspectives = MultiPerspective()
        self.synthesizer = LLMInference(
            alias="smart",
            system_prompt="Synthesize these perspectives into an executive summary.",
            max_tokens=500,
        )

    def forward(self, text: str) -> str:
        analyses = self.perspectives(text)
        # Use explicit key access instead of .items() iteration
        # (Proxy objects don't support iteration during tracing)
        combined = (
            f"## Technical\n{analyses['technical']}\n\n"
            f"## Business\n{analyses['business']}\n\n"
            f"## User\n{analyses['user']}"
        )
        return self.synthesizer(combined)


# --- Learnable System Prompts ---


class AdaptiveAssistant(Module):
    """Assistant with learnable instructions that can be optimized."""

    def __init__(self) -> None:
        super().__init__()
        # Learnable parameter - can be improved via backward passes
        self.instructions = Parameter(
            "Be helpful and concise.",
            description="System instructions for the assistant",
            requires_grad=True,
        )
        self.llm = LLMInference(
            alias="smart",
            system_prompt=self.instructions,  # Pass Parameter directly
            temperature=0.7,
        )

    def forward(self, query: str) -> str:
        return self.llm(query)


# --- Complex Pipeline ---


class DocumentProcessor(Module):
    """Multi-stage document processing pipeline.

    Structure: extract -> [technical, business, user] -> report
    """

    def __init__(self) -> None:
        super().__init__()
        self.extractor = LLMInference(
            alias="fast",
            system_prompt="Extract key facts and figures.",
            temperature=0.2,
        )
        self.analyzer = MultiPerspective()
        self.reporter = LLMInference(
            alias="smart",
            system_prompt="Generate a comprehensive report.",
            max_tokens=1000,
        )

    def forward(self, document: str) -> str:
        facts = self.extractor(document)
        analyses = self.analyzer(facts)
        # Use explicit key access instead of .items() iteration
        combined = (
            f"Technical: {analyses['technical']}\n"
            f"Business: {analyses['business']}\n"
            f"User: {analyses['user']}"
        )
        return self.reporter(combined)


if __name__ == "__main__":
    print("=" * 60)
    print("plait: LLM Pipeline Examples")
    print("=" * 60)
    print("\nThese examples show pipeline structure and parameters.")
    print("To execute with real LLMs, see 04_execution.py.\n")

    # Simple module
    print("1. Simple LLM Module (Summarizer)")
    summarizer = Summarizer()
    print(f"   Alias: {summarizer.llm.alias}")
    print(f"   Temperature: {summarizer.llm.temperature}")

    # Sequential
    print("\n2. Sequential Pipeline (SummarizeAndAnalyze)")
    seq = SummarizeAndAnalyze()
    print("   Module tree:")
    for name, mod in seq.named_modules():
        if name:
            print(f"      {name}: {type(mod).__name__}")

    # Parallel
    print("\n3. Parallel Fan-out (MultiPerspective)")
    multi = MultiPerspective()
    print(f"   Branches: {len(list(multi.children()))}")
    for name, child in multi.named_children():
        if isinstance(child, LLMInference):
            print(f"      {name}: alias={child.alias}")

    # Fan-in
    print("\n4. Fan-in Synthesis (ComprehensiveAnalyzer)")
    analyzer = ComprehensiveAnalyzer()
    print(f"   Total modules: {len(list(analyzer.modules()))}")

    # Learnable prompts
    print("\n5. Learnable System Prompts (AdaptiveAssistant)")
    assistant = AdaptiveAssistant()
    for name, param in assistant.named_parameters():
        grad = "learnable" if param.requires_grad else "fixed"
        print(f"   {name} ({grad}): '{param.value[:40]}...'")

    # Complex pipeline
    print("\n6. Complex Pipeline (DocumentProcessor)")
    processor = DocumentProcessor()
    print(f"   Total modules: {len(list(processor.modules()))}")
    llm_count = sum(1 for m in processor.modules() if isinstance(m, LLMInference))
    print(f"   LLMInference count: {llm_count}")
    print("   Module tree:")
    for name, mod in processor.named_modules():
        if name == "":
            print(f"      (root): {type(mod).__name__}")
        else:
            depth = name.count(".")
            indent = "      " + "  " * depth
            short_name = name.split(".")[-1]
            print(f"{indent}{short_name}: {type(mod).__name__}")

    print("\n" + "=" * 60)
    print("Use bind() or ExecutionSettings to connect to real endpoints.")
    print("=" * 60)
