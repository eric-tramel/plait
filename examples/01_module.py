#!/usr/bin/env python3
"""Module: The building block for inference pipelines.

Demonstrates:
- Creating custom Module subclasses with forward()
- Composing modules into hierarchies
- Using Parameter for learnable values
- Module introspection (children, parameters, state_dict)

Run: python examples/01_module.py
"""

from plait.module import Module
from plait.parameter import Parameter

# --- Basic Module ---


class TextCleaner(Module):
    """Clean text by normalizing whitespace and case."""

    def forward(self, text: str) -> str:
        return text.strip().lower()


class TextFormatter(Module):
    """Format text with configurable prefix/suffix."""

    def __init__(self, prefix: str = "", suffix: str = "") -> None:
        super().__init__()
        self.prefix = prefix
        self.suffix = suffix

    def forward(self, text: str) -> str:
        return f"{self.prefix}{text}{self.suffix}"


# --- Module Composition ---


class TextPipeline(Module):
    """A pipeline composing cleaner and formatter modules."""

    def __init__(self) -> None:
        super().__init__()
        self.cleaner = TextCleaner()
        self.formatter = TextFormatter(prefix="[", suffix="]")

    def forward(self, text: str) -> str:
        cleaned = self.cleaner(text)
        return self.formatter(cleaned)


# --- Parameters (Learnable Values) ---


class PromptTemplate(Module):
    """A module with a learnable prompt template.

    Parameters can be optimized via backward passes to improve outputs.
    """

    def __init__(self, template: str) -> None:
        super().__init__()
        self.template = Parameter(
            template,
            description="Prompt template that formats user input",
            requires_grad=True,  # Learnable
        )

    def forward(self, context: str) -> str:
        return self.template.value.format(context=context)


class Assistant(Module):
    """Assistant with both fixed and learnable parameters."""

    def __init__(self, name: str, instructions: str) -> None:
        super().__init__()
        # Fixed parameter (not optimized)
        self.name = Parameter(name, description="Assistant name", requires_grad=False)
        # Learnable parameter (optimized during training)
        self.instructions = Parameter(
            instructions,
            description="Behavior instructions",
            requires_grad=True,
        )

    def forward(self, query: str) -> str:
        return f"[{self.name.value}] {self.instructions.value}\nUser: {query}"


# --- Nested Modules with Parameters ---


class Tagger(Module):
    """Add a tag to text."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = Parameter(tag, description="Tag to add", requires_grad=True)

    def forward(self, text: str) -> str:
        return f"[{self.tag.value}] {text}"


class MultiTagger(Module):
    """Apply multiple tags via nested modules."""

    def __init__(self) -> None:
        super().__init__()
        self.priority = Tagger("HIGH")
        self.category = Tagger("TASK")

    def forward(self, text: str) -> str:
        tagged = self.priority(text)
        return self.category(tagged)


if __name__ == "__main__":
    print("=" * 60)
    print("plait: Module Examples")
    print("=" * 60)

    # Basic module
    print("\n1. Basic Module")
    cleaner = TextCleaner()
    print(f"   TextCleaner('  Hello World  ') -> '{cleaner('  Hello World  ')}'")

    # Configured module
    print("\n2. Configured Module")
    formatter = TextFormatter(prefix=">>> ", suffix=" <<<")
    print(f"   TextFormatter('hello') -> '{formatter('hello')}'")

    # Composed modules
    print("\n3. Module Composition")
    pipeline = TextPipeline()
    print(f"   TextPipeline('  MESSY Input  ') -> '{pipeline('  MESSY Input  ')}'")

    # Module tree
    print("\n   Module tree:")
    for name, mod in pipeline.named_modules():
        display = name if name else "(root)"
        print(f"      {display}: {type(mod).__name__}")

    # Parameters
    print("\n4. Learnable Parameters")
    template = PromptTemplate("Analyze: {context}")
    print(f"   Template: '{template.template.value}'")
    print(f"   Output: '{template('sales data')}'")
    print(f"   Learnable: {template.template.requires_grad}")

    # Mixed parameters
    print("\n5. Fixed vs Learnable Parameters")
    assistant = Assistant("Bot", "Be helpful.")
    for name, param in assistant.named_parameters():
        grad = "learnable" if param.requires_grad else "fixed"
        print(f"   {name} ({grad}): '{param.value}'")

    # Nested parameters
    print("\n6. Nested Parameters")
    tagger = MultiTagger()
    print(f"   Output: '{tagger('Task description')}'")
    print("   All parameters:")
    for name, param in tagger.named_parameters():
        print(f"      {name}: '{param.value}'")

    # State dict (save/load)
    print("\n7. State Serialization")
    state = tagger.state_dict()
    print(f"   state_dict(): {state}")

    new_tagger = MultiTagger()
    new_tagger.load_state_dict({"priority.tag": "URGENT", "category.tag": "BUG"})
    print(f"   After load: {new_tagger.state_dict()}")

    print("\n" + "=" * 60)
