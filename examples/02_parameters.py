#!/usr/bin/env python3
"""Working with learnable parameters.

This example demonstrates how to use Parameter objects for values
that can be optimized during training (backward passes).

Run with: python examples/02_parameters.py
"""

from plait.module import Module
from plait.parameter import Parameter

# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Module with a Learnable Parameter
# ─────────────────────────────────────────────────────────────────────────────


class PromptTemplate(Module):
    """A module with a learnable prompt template.

    The template parameter can be optimized through backward passes
    to improve the quality of generated outputs.
    """

    def __init__(self, template: str) -> None:
        super().__init__()
        # This parameter can be updated during optimization
        self.template = Parameter(
            template,
            description="Prompt template that formats user context",
            requires_grad=True,
        )

    def forward(self, context: str) -> str:
        return self.template.value.format(context=context)


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Module with Fixed and Learnable Parameters
# ─────────────────────────────────────────────────────────────────────────────


class Assistant(Module):
    """An assistant with both fixed and learnable parameters."""

    def __init__(self, name: str, instructions: str) -> None:
        super().__init__()
        # Fixed parameter - won't be optimized
        self.name = Parameter(name, description="Assistant name", requires_grad=False)
        # Learnable parameter - will be optimized
        self.instructions = Parameter(
            instructions,
            description="Assistant behavior instructions",
            requires_grad=True,
        )

    def forward(self, user_input: str) -> str:
        return (
            f"[{self.name.value}]\n"
            f"Instructions: {self.instructions.value}\n"
            f"User: {user_input}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Nested Modules with Parameters
# ─────────────────────────────────────────────────────────────────────────────


class Tagger(Module):
    """A module that adds a tag to text."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = Parameter(tag, description="Tag to add to text", requires_grad=True)

    def forward(self, text: str) -> str:
        return f"[{self.tag.value}] {text}"


class MultiTagger(Module):
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


class LearnableGreeter(Module):
    """A greeter with a learnable greeting style."""

    def __init__(self) -> None:
        super().__init__()
        self.greeting = Parameter(
            "Hello", description="Greeting word", requires_grad=True
        )
        self.punctuation = Parameter(
            "!", description="Punctuation to use", requires_grad=True
        )

    def forward(self, name: str) -> str:
        return f"{self.greeting.value}, {name}{self.punctuation.value}"


# ─────────────────────────────────────────────────────────────────────────────
# Example 5: Saving and Loading Parameters (state_dict)
# ─────────────────────────────────────────────────────────────────────────────


class OptimizableAssistant(Module):
    """An assistant with multiple learnable prompts that can be saved/loaded."""

    def __init__(self) -> None:
        super().__init__()
        self.system = Parameter(
            "You are a helpful assistant.",
            description="System prompt for the assistant",
            requires_grad=True,
        )
        self.greeting = Parameter(
            "Hello!", description="Greeting message", requires_grad=True
        )

    def forward(self, user_input: str) -> str:
        return f"[System: {self.system.value}]\n{self.greeting.value} {user_input}"


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("plait: Parameters Example")
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

    # Example 5: Saving and loading parameters
    print("\n5. Saving and Loading Parameters (state_dict)")
    print("-" * 40)

    # Create and customize a module
    assistant = OptimizableAssistant()
    print("   Original parameters:")
    print(f"      system: '{assistant.system.value}'")
    print(f"      greeting: '{assistant.greeting.value}'")

    # Simulate optimization - update the parameters
    assistant.system.value = "You are an expert Python programmer."
    assistant.greeting.value = "Greetings, developer!"
    print("\n   After 'optimization':")
    print(f"      system: '{assistant.system.value}'")
    print(f"      greeting: '{assistant.greeting.value}'")

    # Save the state
    state = assistant.state_dict()
    print(f"\n   Saved state_dict: {state}")

    # Create a fresh instance and load the state
    new_assistant = OptimizableAssistant()
    print("\n   Fresh instance parameters:")
    print(f"      system: '{new_assistant.system.value}'")
    print(f"      greeting: '{new_assistant.greeting.value}'")

    new_assistant.load_state_dict(state)
    print("\n   After load_state_dict:")
    print(f"      system: '{new_assistant.system.value}'")
    print(f"      greeting: '{new_assistant.greeting.value}'")

    # JSON serialization example
    import json

    json_str = json.dumps(state, indent=2)
    print(f"\n   JSON serialized:\n{json_str}")

    print("\n" + "=" * 60)
    print("Done!")
