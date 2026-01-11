#!/usr/bin/env python3
"""Containers: PyTorch-style module and parameter containers.

Demonstrates:
- Sequential: Chaining modules together (output -> input)
- ModuleList: List-like container for dynamic module management
- ModuleDict: Dict-like container for named module access
- ParameterList: List-like container for parameters
- ParameterDict: Dict-like container for parameters
- Parameter collection through nested containers
- Combining containers for complex architectures

Run: python examples/06_containers.py
"""

from collections import OrderedDict

from plait import (
    Module,
    ModuleDict,
    ModuleList,
    Parameter,
    ParameterDict,
    ParameterList,
    Sequential,
)

# --- Basic Modules for Demonstration ---


class TextProcessor(Module):
    """Simple text processor with configurable prefix/suffix."""

    def __init__(self, name: str, prefix: str = "", suffix: str = "") -> None:
        super().__init__()
        self.processor_name = name
        self.prefix = prefix
        self.suffix = suffix

    def forward(self, text: str) -> str:
        return f"{self.prefix}{text}{self.suffix}"


class LearnableProcessor(Module):
    """Text processor with a learnable parameter."""

    def __init__(self, name: str, template: str) -> None:
        super().__init__()
        self.processor_name = name
        self.template = Parameter(
            template,
            description=f"Template for {name}",
            requires_grad=True,
        )

    def forward(self, text: str) -> str:
        return self.template.value.format(input=text)


# --- Sequential Container ---


def demo_sequential() -> None:
    """Demonstrate Sequential container features."""
    print("\n" + "=" * 60)
    print("1. SEQUENTIAL CONTAINER")
    print("=" * 60)

    # Basic Sequential with positional args
    print("\n1.1 Basic Sequential (positional args)")
    print("-" * 40)
    pipeline = Sequential(
        TextProcessor("step1", prefix="[1:", suffix="]"),
        TextProcessor("step2", prefix="[2:", suffix="]"),
        TextProcessor("step3", prefix="[3:", suffix="]"),
    )
    result = pipeline("hello")
    print("   Input: 'hello'")
    print(f"   Output: '{result}'")
    print(f"   Length: {len(pipeline)}")

    # Sequential with OrderedDict for named access
    print("\n1.2 Named Sequential (OrderedDict)")
    print("-" * 40)
    named_pipeline = Sequential(
        OrderedDict(
            [
                ("preprocess", TextProcessor("pre", prefix="<", suffix=">")),
                ("transform", TextProcessor("trans", prefix="[", suffix="]")),
                ("postprocess", TextProcessor("post", prefix="(", suffix=")")),
            ]
        )
    )
    result = named_pipeline("data")
    print("   Input: 'data'")
    print(f"   Output: '{result}'")
    print("   Named access:")
    print(f"      pipeline.preprocess: {type(named_pipeline.preprocess).__name__}")
    print(f"      pipeline.transform: {type(named_pipeline.transform).__name__}")

    # Indexing and slicing
    print("\n1.3 Indexing and Slicing")
    print("-" * 40)
    print(f"   pipeline[0]: {type(pipeline[0]).__name__}")
    print(f"   pipeline[-1]: {type(pipeline[-1]).__name__}")
    sliced = pipeline[0:2]
    print(f"   pipeline[0:2]: Sequential with {len(sliced)} modules")

    # Iteration
    print("\n1.4 Iteration")
    print("-" * 40)
    for i, module in enumerate(pipeline):
        print(f"   Module {i}: {type(module).__name__}")

    # Append
    print("\n1.5 Dynamic Append")
    print("-" * 40)
    print(f"   Before append: {len(pipeline)} modules")
    pipeline.append(TextProcessor("step4", prefix="[4:", suffix="]"))
    print(f"   After append: {len(pipeline)} modules")
    result = pipeline("test")
    print(f"   New output: '{result}'")


# --- ModuleList Container ---


def demo_module_list() -> None:
    """Demonstrate ModuleList container features."""
    print("\n" + "=" * 60)
    print("2. MODULELIST CONTAINER")
    print("=" * 60)

    # Basic ModuleList
    print("\n2.1 Basic ModuleList")
    print("-" * 40)
    layers = ModuleList(
        [TextProcessor(f"layer{i}", prefix=f"L{i}:", suffix="") for i in range(3)]
    )
    print(f"   Created ModuleList with {len(layers)} modules")

    # Manual iteration in forward
    class LayeredProcessor(Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = ModuleList(
                [
                    TextProcessor(f"layer{i}", prefix=f"[{i}:", suffix="]")
                    for i in range(3)
                ]
            )

        def forward(self, text: str) -> str:
            for layer in self.layers:
                text = layer(text)
            return text

    processor = LayeredProcessor()
    result = processor("input")
    print(f"   LayeredProcessor('input'): '{result}'")

    # Indexing
    print("\n2.2 Indexing and Membership")
    print("-" * 40)
    print(f"   layers[0]: {type(layers[0]).__name__}")
    print(f"   layers[-1]: {type(layers[-1]).__name__}")
    print(f"   layers[0] in layers: {layers[0] in layers}")

    # Mutations
    print("\n2.3 List Mutations")
    print("-" * 40)
    print(f"   Initial length: {len(layers)}")

    # Append
    layers.append(TextProcessor("new", prefix="NEW:"))
    print(f"   After append: {len(layers)}")

    # Extend
    layers.extend([TextProcessor("ext1"), TextProcessor("ext2")])
    print(f"   After extend: {len(layers)}")

    # Insert
    layers.insert(0, TextProcessor("first", prefix="FIRST:"))
    print(f"   After insert at 0: {len(layers)}")

    # Pop
    popped = layers.pop()
    print(f"   Popped last: {type(popped).__name__}")
    print(f"   After pop: {len(layers)}")

    popped = layers.pop(0)
    print(f"   Popped first: {type(popped).__name__}")
    print(f"   After pop(0): {len(layers)}")

    # Slicing
    print("\n2.4 Slicing")
    print("-" * 40)
    slice_result = layers[1:3]
    print(f"   layers[1:3]: ModuleList with {len(slice_result)} modules")


# --- ModuleDict Container ---


def demo_module_dict() -> None:
    """Demonstrate ModuleDict container features."""
    print("\n" + "=" * 60)
    print("3. MODULEDICT CONTAINER")
    print("=" * 60)

    # Basic ModuleDict
    print("\n3.1 Basic ModuleDict")
    print("-" * 40)
    modules = ModuleDict(
        {
            "encoder": TextProcessor("enc", prefix="ENC[", suffix="]"),
            "decoder": TextProcessor("dec", prefix="DEC[", suffix="]"),
            "classifier": TextProcessor("cls", prefix="CLS[", suffix="]"),
        }
    )
    print(f"   Created ModuleDict with {len(modules)} modules")
    print(f"   Keys: {list(modules.keys())}")

    # Access patterns
    print("\n3.2 Access Patterns")
    print("-" * 40)
    print(f"   modules['encoder']: {type(modules['encoder']).__name__}")
    print(f"   modules.encoder: {type(modules.encoder).__name__}")
    print(f"   'encoder' in modules: {'encoder' in modules}")

    # Iteration
    print("\n3.3 Iteration")
    print("-" * 40)
    print("   Keys:")
    for key in modules:
        print(f"      {key}")
    print("   Items:")
    for key, mod in modules.items():
        print(f"      {key}: {type(mod).__name__}")

    # Mutations
    print("\n3.4 Dict Mutations")
    print("-" * 40)

    # Add new module
    modules["attention"] = TextProcessor("attn", prefix="ATTN[", suffix="]")
    print(f"   After adding 'attention': {list(modules.keys())}")

    # Update
    modules.update(
        {
            "norm": TextProcessor("norm", prefix="NORM[", suffix="]"),
            "output": TextProcessor("out", prefix="OUT[", suffix="]"),
        }
    )
    print(f"   After update: {list(modules.keys())}")

    # Pop
    popped = modules.pop("output")
    print(f"   Popped 'output': {type(popped).__name__}")
    print(f"   After pop: {list(modules.keys())}")

    # Delete
    del modules["norm"]
    print(f"   After del 'norm': {list(modules.keys())}")

    # Using in forward
    print("\n3.5 Using ModuleDict in Forward")
    print("-" * 40)

    class MultiHeadProcessor(Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = ModuleDict(
                {
                    "head_a": TextProcessor("a", prefix="A:"),
                    "head_b": TextProcessor("b", prefix="B:"),
                    "head_c": TextProcessor("c", prefix="C:"),
                }
            )

        def forward(self, text: str) -> dict[str, str]:
            return {key: self.heads[key](text) for key in self.heads}

    multi = MultiHeadProcessor()
    results = multi("input")
    print("   MultiHeadProcessor('input'):")
    for key, value in results.items():
        print(f"      {key}: '{value}'")


# --- Parameter Containers ---


def demo_parameter_containers() -> None:
    """Demonstrate ParameterList and ParameterDict containers."""
    print("\n" + "=" * 60)
    print("4. PARAMETER CONTAINERS")
    print("=" * 60)

    # ParameterList
    print("\n4.1 ParameterList")
    print("-" * 40)

    class MultiPromptModule(Module):
        """Module with multiple prompts stored in a ParameterList."""

        def __init__(self) -> None:
            super().__init__()
            self.prompts = ParameterList(
                [
                    Parameter("Be concise", description="Style prompt"),
                    Parameter("Be helpful", description="Tone prompt"),
                    Parameter("Be accurate", description="Quality prompt"),
                ]
            )

        def forward(self, text: str) -> str:
            # Combine all prompts
            combined = " | ".join(p.value for p in self.prompts)
            return f"[{combined}] {text}"

    multi_prompt = MultiPromptModule()
    print(f"   ParameterList length: {len(multi_prompt.prompts)}")
    print("   Accessing by index:")
    for i, param in enumerate(multi_prompt.prompts):
        print(f"      prompts[{i}]: '{param.value}'")

    # Mutations
    print("\n   Mutations:")
    multi_prompt.prompts.append(Parameter("Be creative", description="Creativity"))
    print(f"   After append: {len(multi_prompt.prompts)} prompts")
    multi_prompt.prompts.insert(0, Parameter("Be clear", description="Clarity"))
    print(f"   After insert at 0: {len(multi_prompt.prompts)} prompts")

    # Named parameters
    print("\n   Named parameters:")
    for name, param in multi_prompt.named_parameters():
        print(f"      {name}: '{param.value}'")

    # ParameterDict
    print("\n4.2 ParameterDict")
    print("-" * 40)

    class TaskRouter(Module):
        """Module with task-specific prompts stored in a ParameterDict."""

        def __init__(self) -> None:
            super().__init__()
            self.task_prompts = ParameterDict(
                {
                    "summarize": Parameter(
                        "Summarize the following text concisely",
                        description="Summary task prompt",
                    ),
                    "translate": Parameter(
                        "Translate the following text to French",
                        description="Translation task prompt",
                    ),
                    "analyze": Parameter(
                        "Analyze the sentiment of the following text",
                        description="Analysis task prompt",
                    ),
                }
            )

        def forward(self, task: str, text: str) -> str:
            prompt = self.task_prompts[task].value
            return f"{prompt}: {text}"

    router = TaskRouter()
    print(f"   ParameterDict keys: {list(router.task_prompts.keys())}")
    print("   Accessing by key:")
    for key in router.task_prompts:
        print(
            f"      task_prompts['{key}']: '{router.task_prompts[key].value[:30]}...'"
        )

    # Add new task
    print("\n   Mutations:")
    router.task_prompts["classify"] = Parameter(
        "Classify the following text into categories",
        description="Classification task prompt",
    )
    print(f"   After adding 'classify': {list(router.task_prompts.keys())}")

    # Named parameters
    print("\n   Named parameters:")
    for name, param in router.named_parameters():
        print(f"      {name}: '{param.value[:25]}...'")

    # Combined example
    print("\n4.3 Mixing Parameter Containers")
    print("-" * 40)

    class ConfigurableAgent(Module):
        """Module combining direct params, ParameterList, and ParameterDict."""

        def __init__(self) -> None:
            super().__init__()
            # Direct parameter
            self.system_prompt = Parameter(
                "You are a helpful assistant",
                description="Main system prompt",
            )
            # List of style modifiers
            self.style_hints = ParameterList(
                [
                    Parameter("formal", description="Formality level"),
                    Parameter("technical", description="Technical level"),
                ]
            )
            # Dict of task-specific overrides
            self.overrides = ParameterDict(
                {
                    "code": Parameter("Use code blocks", description="Code override"),
                    "math": Parameter("Use LaTeX", description="Math override"),
                }
            )

        def forward(self, text: str) -> str:
            return text

    agent = ConfigurableAgent()
    print("   All parameters collected:")
    for name, param in agent.named_parameters():
        print(f"      {name}: '{param.value}'")


# --- Parameter Collection ---


def demo_parameter_collection() -> None:
    """Demonstrate parameter collection through containers."""
    print("\n" + "=" * 60)
    print("5. PARAMETER COLLECTION (Module Containers)")
    print("=" * 60)

    # Create nested structure with learnable parameters
    print("\n5.1 Nested Container with Parameters")
    print("-" * 40)

    class NestedPipeline(Module):
        def __init__(self) -> None:
            super().__init__()
            self.stages = Sequential(
                LearnableProcessor("stage1", "First: {input}"),
                LearnableProcessor("stage2", "Second: {input}"),
            )
            self.branches = ModuleDict(
                {
                    "path_a": LearnableProcessor("branch_a", "PathA: {input}"),
                    "path_b": LearnableProcessor("branch_b", "PathB: {input}"),
                }
            )

        def forward(self, text: str) -> dict[str, str]:
            staged = self.stages(text)
            return {key: self.branches[key](staged) for key in self.branches}

    pipeline = NestedPipeline()

    # Collect all parameters
    print("   All parameters:")
    for name, param in pipeline.named_parameters():
        print(f"      {name}: '{param.value}'")

    # State dict
    print("\n5.2 State Dict (Save/Load)")
    print("-" * 40)
    state = pipeline.state_dict()
    print("   state_dict():")
    for key, value in state.items():
        print(f"      {key}: '{value}'")

    # Load modified state
    new_state = {
        "stages.0.template": "Modified First: {input}",
        "stages.1.template": "Modified Second: {input}",
        "branches.path_a.template": "Modified PathA: {input}",
        "branches.path_b.template": "Modified PathB: {input}",
    }
    pipeline.load_state_dict(new_state)
    print("\n   After load_state_dict():")
    for name, param in pipeline.named_parameters():
        print(f"      {name}: '{param.value}'")


# --- Complex Architectures ---


def demo_complex_architecture() -> None:
    """Demonstrate combining containers for complex architectures."""
    print("\n" + "=" * 60)
    print("6. COMPLEX ARCHITECTURES")
    print("=" * 60)

    # Encoder-Decoder with multiple layers
    print("\n6.1 Encoder-Decoder Architecture")
    print("-" * 40)

    class EncoderDecoder(Module):
        def __init__(self, num_layers: int = 2) -> None:
            super().__init__()
            self.encoder = Sequential(
                OrderedDict(
                    [
                        (f"enc_layer{i}", TextProcessor(f"enc{i}", prefix=f"E{i}:"))
                        for i in range(num_layers)
                    ]
                )
            )
            self.decoder = Sequential(
                OrderedDict(
                    [
                        (f"dec_layer{i}", TextProcessor(f"dec{i}", prefix=f"D{i}:"))
                        for i in range(num_layers)
                    ]
                )
            )

        def forward(self, text: str) -> str:
            encoded = self.encoder(text)
            return self.decoder(encoded)

    enc_dec = EncoderDecoder(num_layers=3)
    result = enc_dec("data")
    print("   Input: 'data'")
    print(f"   Output: '{result}'")
    print(f"   Encoder layers: {len(enc_dec.encoder)}")
    print(f"   Decoder layers: {len(enc_dec.decoder)}")

    # Multi-branch parallel processing
    print("\n6.2 Multi-Branch Processing")
    print("-" * 40)

    class MultiBranch(Module):
        def __init__(self) -> None:
            super().__init__()
            self.preprocessor = TextProcessor("pre", prefix="PRE:")
            self.branches = ModuleDict(
                {
                    "fast": Sequential(
                        TextProcessor("fast1", prefix="F1:"),
                        TextProcessor("fast2", prefix="F2:"),
                    ),
                    "accurate": Sequential(
                        TextProcessor("acc1", prefix="A1:"),
                        TextProcessor("acc2", prefix="A2:"),
                        TextProcessor("acc3", prefix="A3:"),
                    ),
                }
            )
            self.combiner = TextProcessor("combine", prefix="COMBINED:")

        def forward(self, text: str) -> str:
            preprocessed = self.preprocessor(text)
            results = [self.branches[key](preprocessed) for key in self.branches]
            return self.combiner(" | ".join(results))

    multi_branch = MultiBranch()
    result = multi_branch("input")
    print("   Input: 'input'")
    print(f"   Output: '{result}'")

    # Module tree
    print("\n6.3 Module Tree")
    print("-" * 40)
    print("   MultiBranch module tree:")
    for name, mod in multi_branch.named_modules():
        if name == "":
            print(f"      (root): {type(mod).__name__}")
        else:
            depth = name.count(".")
            indent = "      " + "  " * depth
            short_name = name.split(".")[-1]
            print(f"{indent}{short_name}: {type(mod).__name__}")


if __name__ == "__main__":
    print("=" * 60)
    print("plait: Container Modules and Parameters")
    print("=" * 60)
    print("\nContainers provide PyTorch-style composition:")
    print("  Module Containers:")
    print("    - Sequential: Chain modules (output -> input)")
    print("    - ModuleList: Dynamic list of modules")
    print("    - ModuleDict: Named module access")
    print("  Parameter Containers:")
    print("    - ParameterList: Dynamic list of parameters")
    print("    - ParameterDict: Named parameter access")

    demo_sequential()
    demo_module_list()
    demo_module_dict()
    demo_parameter_containers()
    demo_parameter_collection()
    demo_complex_architecture()

    print("\n" + "=" * 60)
    print("Container Summary:")
    print("  Module Containers:")
    print("    Sequential(mod1, mod2, ...)     - Chain modules")
    print("    Sequential(OrderedDict([...]))  - Named chain")
    print("    ModuleList([mod1, mod2, ...])   - List of modules")
    print("    ModuleDict({'key': mod, ...})   - Dict of modules")
    print("  Parameter Containers:")
    print("    ParameterList([p1, p2, ...])    - List of parameters")
    print("    ParameterDict({'key': p, ...})  - Dict of parameters")
    print("=" * 60)
