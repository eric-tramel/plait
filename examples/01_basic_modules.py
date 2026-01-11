#!/usr/bin/env python3
"""Basic module creation and usage.

This example demonstrates how to create custom InferenceModule subclasses
with forward() methods and use them in a PyTorch-like manner.

Run with: python examples/01_basic_modules.py
"""

from plait.module import InferenceModule

# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Simple Module
# ─────────────────────────────────────────────────────────────────────────────


class TextCleaner(InferenceModule):
    """A simple module that cleans text by stripping whitespace and lowercasing."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, text: str) -> str:
        return text.strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Module with Configuration
# ─────────────────────────────────────────────────────────────────────────────


class TextFormatter(InferenceModule):
    """A module that formats text with configurable prefix and suffix."""

    def __init__(self, prefix: str = "", suffix: str = "") -> None:
        super().__init__()
        self.prefix = prefix
        self.suffix = suffix

    def forward(self, text: str) -> str:
        return f"{self.prefix}{text}{self.suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Module Composition
# ─────────────────────────────────────────────────────────────────────────────


class TextPipeline(InferenceModule):
    """A pipeline that cleans and formats text."""

    def __init__(self) -> None:
        super().__init__()
        self.cleaner = TextCleaner()
        self.formatter = TextFormatter(prefix="[", suffix="]")

    def forward(self, text: str) -> str:
        cleaned = self.cleaner(text)
        formatted = self.formatter(cleaned)
        return formatted


# ─────────────────────────────────────────────────────────────────────────────
# Example 4: Module with Multiple Outputs
# ─────────────────────────────────────────────────────────────────────────────


class TextAnalyzer(InferenceModule):
    """A module that analyzes text and returns multiple metrics."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, text: str) -> dict[str, int | str]:
        return {
            "original": text,
            "length": len(text),
            "word_count": len(text.split()),
            "uppercase": text.upper(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("plait: Basic Modules Example")
    print("=" * 60)

    # Example 1: Simple module
    print("\n1. Simple Module (TextCleaner)")
    print("-" * 40)
    cleaner = TextCleaner()
    result = cleaner("  Hello World!  ")
    print("   Input:  '  Hello World!  '")
    print(f"   Output: '{result}'")

    # Example 2: Configured module
    print("\n2. Configured Module (TextFormatter)")
    print("-" * 40)
    formatter = TextFormatter(prefix=">>> ", suffix=" <<<")
    result = formatter("important message")
    print("   Input:  'important message'")
    print(f"   Output: '{result}'")

    # Example 3: Composed modules
    print("\n3. Composed Modules (TextPipeline)")
    print("-" * 40)
    pipeline = TextPipeline()
    result = pipeline("  MESSY Input Text  ")
    print("   Input:  '  MESSY Input Text  '")
    print(f"   Output: '{result}'")

    # Show module structure
    print("\n   Module structure:")
    for name, module in pipeline.named_modules():
        indent = "   " if name == "" else "      "
        display_name = name if name else "(root)"
        print(f"{indent}{display_name}: {type(module).__name__}")

    # Example 4: Multiple outputs
    print("\n4. Multiple Outputs (TextAnalyzer)")
    print("-" * 40)
    analyzer = TextAnalyzer()
    result = analyzer("Hello World")
    print("   Input:  'Hello World'")
    print(f"   Output: {result}")

    print("\n" + "=" * 60)
    print("Done!")
