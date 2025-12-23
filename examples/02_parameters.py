#!/usr/bin/env python3
"""Working with learnable parameters.

This example demonstrates how to use Parameter objects for values
that can be optimized during training (backward passes).

Run with: python examples/02_parameters.py
"""

from inf_engine.module import InferenceModule
from inf_engine.parameter import Parameter

# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Module with a Learnable Parameter
# ─────────────────────────────────────────────────────────────────────────────


class PromptTemplate(InferenceModule):
    """A module with a learnable prompt template.

    The template parameter can be optimized through backward passes
    to improve the quality of generated outputs.
    """

    def __init__(self, template: str) -> None:
        super().__init__()
        # This parameter can be updated during optimization
        self.template = Parameter(template, requires_grad=True)

    def forward(self, context: str) -> str:
        return self.template.value.format(context=context)


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Module with Fixed and Learnable Parameters
# ─────────────────────────────────────────────────────────────────────────────


class Assistant(InferenceModule):
    """An assistant with both fixed and learnable parameters."""

    def __init__(self, name: str, instructions: str) -> None:
        super().__init__()
        # Fixed parameter - won't be optimized
        self.name = Parameter(name, requires_grad=False)
        # Learnable parameter - will be optimized
        self.instructions = Parameter(instructions, requires_grad=True)

    def forward(self, user_input: str) -> str:
        return (
            f"[{self.name.value}]\n"
            f"Instructions: {self.instructions.value}\n"
            f"User: {user_input}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Nested Modules with Parameters
# ─────────────────────────────────────────────────────────────────────────────


class Tagger(InferenceModule):
    """A module that adds a tag to text."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = Parameter(tag, requires_grad=True)

    def forward(self, text: str) -> str:
        return f"[{self.tag.value}] {text}"


class MultiTagger(InferenceModule):
    """A module that applies multiple tags."""

    def __init__(self) -> None:
        super().__init__()
        self.priority_tagger = Tagger("PRIORITY")
        self.category_tagger = Tagger("GENERAL")
        self.status_tagger = Tagger("NEW")

    def forward(self, text: str) -> str:
        result = self.priority_tagger(text)
        result = self.category_tagger(result)
        result = self.status_tagger(result)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Example 4: Simulating Parameter Updates (Preview of Optimization)
# ─────────────────────────────────────────────────────────────────────────────


class LearnableGreeter(InferenceModule):
    """A greeter with a learnable greeting style."""

    def __init__(self) -> None:
        super().__init__()
        self.greeting = Parameter("Hello", requires_grad=True)
        self.punctuation = Parameter("!", requires_grad=True)

    def forward(self, name: str) -> str:
        return f"{self.greeting.value}, {name}{self.punctuation.value}"


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("inf-engine: Parameters Example")
    print("=" * 60)

    # Example 1: Learnable parameter
    print("\n1. Learnable Parameter (PromptTemplate)")
    print("-" * 40)
    template = PromptTemplate("Please analyze the following: {context}")
    result = template("quarterly sales data")
    print(f"   Template: '{template.template.value}'")
    print(f"   Output:   '{result}'")
    print(f"   Learnable: {template.template.requires_grad}")

    # Example 2: Mixed parameters
    print("\n2. Fixed and Learnable Parameters (Assistant)")
    print("-" * 40)
    assistant = Assistant(
        name="Helper Bot",
        instructions="Be concise and helpful.",
    )
    result = assistant("What is Python?")
    print(f"   Name (fixed):        '{assistant.name.value}'")
    print(f"   Instructions (learn): '{assistant.instructions.value}'")
    print(f"   Output:\n{result}")

    # Example 3: Nested parameters
    print("\n3. Nested Modules with Parameters (MultiTagger)")
    print("-" * 40)
    tagger = MultiTagger()
    result = tagger("Important message")
    print(f"   Output: '{result}'")

    print("\n   All parameters in module tree:")
    for name, param in tagger.named_parameters():
        print(f"      {name}: '{param.value}' (requires_grad={param.requires_grad})")

    # Example 4: Simulating parameter updates
    print("\n4. Simulating Parameter Updates (LearnableGreeter)")
    print("-" * 40)
    greeter = LearnableGreeter()

    print("   Before optimization:")
    print(f"      {greeter('World')}")

    # Simulate what the optimizer would do
    print("\n   Simulating feedback and update...")
    greeter.greeting.accumulate_feedback("Consider using 'Hey' for informal tone")
    greeter.punctuation.accumulate_feedback("Try '!!' for more enthusiasm")

    print(f"   Feedback buffer for 'greeting': {greeter.greeting._feedback_buffer}")
    print(
        f"   Feedback buffer for 'punctuation': {greeter.punctuation._feedback_buffer}"
    )

    # Apply updates (normally done by optimizer)
    greeter.greeting.apply_update("Hey")
    greeter.punctuation.apply_update("!!")

    print("\n   After optimization:")
    print(f"      {greeter('World')}")

    # Show that feedback buffer is cleared after apply_update
    print(f"   Feedback buffer cleared: {greeter.greeting._feedback_buffer}")

    print("\n" + "=" * 60)
    print("Done!")
