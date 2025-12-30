"""Integration tests for module composition patterns.

This file contains tests for PR-007: Module composition integration tests.
Tests verify that modules can be composed together in various patterns
(sequential, parallel, nested) and that introspection works correctly
across the module hierarchy.
"""

from inf_engine.module import InferenceModule, LLMInference
from inf_engine.parameter import Parameter

# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures: Composable Modules
# ─────────────────────────────────────────────────────────────────────────────


class Doubler(InferenceModule):
    """Simple module that doubles an integer."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: int) -> int:
        return x * 2


class Adder(InferenceModule):
    """Module that adds a fixed amount to an integer."""

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, x: int) -> int:
        return x + self.amount


class Prefixer(InferenceModule):
    """Module that prefixes a string with a learnable prefix."""

    def __init__(self, prefix: str, requires_grad: bool = True) -> None:
        super().__init__()
        self.prefix = Parameter(prefix, description="test", requires_grad=requires_grad)

    def forward(self, text: str) -> str:
        return f"{self.prefix.value}: {text}"


class Uppercaser(InferenceModule):
    """Module that converts text to uppercase."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, text: str) -> str:
        return text.upper()


class Reverser(InferenceModule):
    """Module that reverses a string."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, text: str) -> str:
        return text[::-1]


# ─────────────────────────────────────────────────────────────────────────────
# Simple Composition Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSimpleComposition:
    """Tests for simple module composition (one level of nesting)."""

    def test_module_containing_another_module(self) -> None:
        """A module can contain another module as a child."""

        class Container(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Doubler()

            def forward(self, x: int) -> int:
                return self.inner(x) + 1

        container = Container()

        # Child is registered
        assert "inner" in container._children
        assert container._children["inner"] is container.inner

        # Execution works
        result = container(5)
        assert result == 11  # (5 * 2) + 1

    def test_module_with_parameter_child(self) -> None:
        """A module can contain a child with parameters."""

        class Wrapper(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.prefixer = Prefixer("INFO")

            def forward(self, text: str) -> str:
                return self.prefixer(text)

        wrapper = Wrapper()

        # Child is registered
        assert "prefixer" in wrapper._children

        # Parameter is accessible through hierarchy
        params = list(wrapper.parameters())
        assert len(params) == 1
        assert params[0].value == "INFO"

        # Execution works
        result = wrapper("hello")
        assert result == "INFO: hello"

    def test_module_with_llm_inference_child(self) -> None:
        """A module can contain LLMInference as a child."""

        class Assistant(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.llm = LLMInference(
                    alias="assistant",
                    system_prompt="You are helpful.",
                    temperature=0.7,
                )

            def forward(self, prompt: str) -> str:
                # In real usage, this would be traced, not executed directly
                return f"Would call LLM with: {prompt}"

        assistant = Assistant()

        # LLMInference is registered as child
        assert "llm" in assistant._children
        assert isinstance(assistant.llm, LLMInference)

        # System prompt parameter is accessible
        params = list(assistant.parameters())
        assert len(params) == 1
        assert params[0].value == "You are helpful."


# ─────────────────────────────────────────────────────────────────────────────
# Nested Composition Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNestedComposition:
    """Tests for nested module composition (multiple levels)."""

    def test_two_levels_of_nesting(self) -> None:
        """Modules can be nested two levels deep."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.doubler = Doubler()

            def forward(self, x: int) -> int:
                return self.doubler(x)

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

            def forward(self, x: int) -> int:
                return self.inner(x) + 10

        outer = Outer()

        # All modules are found
        all_modules = list(outer.modules())
        assert len(all_modules) == 3  # outer, inner, doubler

        # Execution works through nesting
        result = outer(5)
        assert result == 20  # (5 * 2) + 10

    def test_three_levels_of_nesting(self) -> None:
        """Modules can be nested three levels deep."""

        class Level3(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.param = Parameter("deep", description="test")

            def forward(self, x: str) -> str:
                return f"{self.param.value}:{x}"

        class Level2(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level3 = Level3()

            def forward(self, x: str) -> str:
                return self.level3(x.upper())

        class Level1(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.level2 = Level2()

            def forward(self, x: str) -> str:
                return f"[{self.level2(x)}]"

        root = Level1()

        # All modules are found
        all_modules = list(root.modules())
        assert len(all_modules) == 3

        # Named modules have correct hierarchical names
        named = dict(root.named_modules())
        assert "" in named  # root
        assert "level2" in named
        assert "level2.level3" in named

        # Parameters are found through hierarchy
        params = list(root.parameters())
        assert len(params) == 1
        assert params[0].value == "deep"

        # Named parameters have correct hierarchical names
        named_params = dict(root.named_parameters())
        assert "level2.level3.param" in named_params

        # Execution works through all levels
        result = root("hello")
        assert result == "[deep:HELLO]"

    def test_deeply_nested_llm_inference(self) -> None:
        """LLMInference can be deeply nested and still be discoverable."""

        class Stage1(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.summarizer = LLMInference(
                    alias="fast", system_prompt="Summarize briefly."
                )

        class Stage2(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.stage1 = Stage1()
                self.analyzer = LLMInference(
                    alias="smart", system_prompt="Analyze thoroughly."
                )

        class Pipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.stage2 = Stage2()
                self.finalizer = LLMInference(alias="fast", system_prompt="Finalize.")

        pipeline = Pipeline()

        # All modules are found (6 total: pipeline, stage2, stage1, summarizer, analyzer, finalizer)
        all_modules = list(pipeline.modules())
        assert len(all_modules) == 6

        # All LLMInference instances are found
        llm_modules = [m for m in pipeline.modules() if isinstance(m, LLMInference)]
        assert len(llm_modules) == 3

        # All system prompts are found
        params = list(pipeline.parameters())
        assert len(params) == 3
        param_values = {p.value for p in params}
        assert param_values == {
            "Summarize briefly.",
            "Analyze thoroughly.",
            "Finalize.",
        }

        # Named parameters have correct hierarchical names
        named_params = dict(pipeline.named_parameters())
        assert "stage2.stage1.summarizer.system_prompt" in named_params
        assert "stage2.analyzer.system_prompt" in named_params
        assert "finalizer.system_prompt" in named_params


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Children Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleChildren:
    """Tests for modules with multiple children."""

    def test_module_with_multiple_children(self) -> None:
        """A module can have multiple child modules."""

        class MultiChild(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.doubler = Doubler()
                self.adder = Adder(10)
                self.adder2 = Adder(100)

            def forward(self, x: int) -> dict[str, int]:
                return {
                    "doubled": self.doubler(x),
                    "plus_10": self.adder(x),
                    "plus_100": self.adder2(x),
                }

        multi = MultiChild()

        # All children are registered
        assert len(multi._children) == 3
        assert "doubler" in multi._children
        assert "adder" in multi._children
        assert "adder2" in multi._children

        # Execution works
        result = multi(5)
        assert result == {"doubled": 10, "plus_10": 15, "plus_100": 105}

    def test_module_with_mixed_children(self) -> None:
        """A module can have children of different types."""

        class MixedChildren(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.custom = Prefixer("TAG")
                self.llm = LLMInference(alias="default", system_prompt="Be helpful.")

        mixed = MixedChildren()

        # Both children are registered
        assert len(mixed._children) == 2
        assert isinstance(mixed.custom, Prefixer)
        assert isinstance(mixed.llm, LLMInference)

        # Both parameters are found
        params = list(mixed.parameters())
        assert len(params) == 2

    def test_parallel_llm_inference_modules(self) -> None:
        """Multiple LLMInference modules for parallel analysis."""

        class MultiPerspective(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.technical = LLMInference(
                    alias="llm", system_prompt="Technical analysis."
                )
                self.business = LLMInference(
                    alias="llm", system_prompt="Business analysis."
                )
                self.user = LLMInference(alias="llm", system_prompt="User perspective.")

        multi = MultiPerspective()

        # All children are registered
        assert len(multi._children) == 3

        # All system prompts are found
        params = list(multi.parameters())
        assert len(params) == 3

        # Named parameters are correct
        named = dict(multi.named_parameters())
        assert "technical.system_prompt" in named
        assert "business.system_prompt" in named
        assert "user.system_prompt" in named


# ─────────────────────────────────────────────────────────────────────────────
# Sequential Composition Pattern Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSequentialComposition:
    """Tests for sequential composition pattern (A → B → C)."""

    def test_simple_sequential_pipeline(self) -> None:
        """Sequential pipeline passes output of one module to the next."""

        class Pipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.step1 = Uppercaser()
                self.step2 = Reverser()

            def forward(self, text: str) -> str:
                result = self.step1(text)
                result = self.step2(result)
                return result

        pipeline = Pipeline()

        # "hello" → "HELLO" → "OLLEH"
        result = pipeline("hello")
        assert result == "OLLEH"

    def test_long_sequential_pipeline(self) -> None:
        """Long sequential pipeline with many steps."""

        class LongPipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.add_1 = Adder(1)
                self.double = Doubler()
                self.add_10 = Adder(10)
                self.double_again = Doubler()
                self.add_100 = Adder(100)

            def forward(self, x: int) -> int:
                x = self.add_1(x)
                x = self.double(x)
                x = self.add_10(x)
                x = self.double_again(x)
                x = self.add_100(x)
                return x

        pipeline = LongPipeline()

        # 5 → 6 → 12 → 22 → 44 → 144
        result = pipeline(5)
        assert result == 144

        # All children registered
        assert len(pipeline._children) == 5

    def test_sequential_with_parameters(self) -> None:
        """Sequential pipeline where each step has parameters."""

        class ParameterizedPipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.step1 = Prefixer("STEP1")
                self.step2 = Prefixer("STEP2")
                self.step3 = Prefixer("STEP3")

            def forward(self, text: str) -> str:
                result = self.step1(text)
                result = self.step2(result)
                result = self.step3(result)
                return result

        pipeline = ParameterizedPipeline()

        # All parameters are found
        params = list(pipeline.parameters())
        assert len(params) == 3

        # Named parameters have correct names
        named = dict(pipeline.named_parameters())
        assert "step1.prefix" in named
        assert "step2.prefix" in named
        assert "step3.prefix" in named

        # Execution works
        result = pipeline("hello")
        assert result == "STEP3: STEP2: STEP1: hello"


# ─────────────────────────────────────────────────────────────────────────────
# Parallel Composition Pattern Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParallelComposition:
    """Tests for parallel composition pattern (fan-out)."""

    def test_parallel_fanout(self) -> None:
        """Parallel fan-out passes same input to multiple modules."""

        class FanOut(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.double = Doubler()
                self.add_10 = Adder(10)
                self.add_100 = Adder(100)

            def forward(self, x: int) -> dict[str, int]:
                return {
                    "doubled": self.double(x),
                    "plus_10": self.add_10(x),
                    "plus_100": self.add_100(x),
                }

        fanout = FanOut()

        result = fanout(5)
        assert result == {
            "doubled": 10,
            "plus_10": 15,
            "plus_100": 105,
        }

    def test_parallel_with_different_types(self) -> None:
        """Parallel modules can return different types."""

        class MixedOutput(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.upper = Uppercaser()
                self.reverse = Reverser()
                self.prefix = Prefixer("TAG")

            def forward(self, text: str) -> dict[str, str]:
                return {
                    "upper": self.upper(text),
                    "reverse": self.reverse(text),
                    "prefixed": self.prefix(text),
                }

        mixed = MixedOutput()

        result = mixed("hello")
        assert result == {
            "upper": "HELLO",
            "reverse": "olleh",
            "prefixed": "TAG: hello",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fan-in Composition Pattern Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFanInComposition:
    """Tests for fan-in composition pattern."""

    def test_simple_fanin(self) -> None:
        """Fan-in combines outputs from parallel modules."""

        class Analyzer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.view1 = Prefixer("View1")
                self.view2 = Prefixer("View2")

            def forward(self, text: str) -> dict[str, str]:
                return {
                    "v1": self.view1(text),
                    "v2": self.view2(text),
                }

        class Synthesizer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.analyzer = Analyzer()

            def forward(self, text: str) -> str:
                perspectives = self.analyzer(text)
                return " | ".join(perspectives.values())

        synth = Synthesizer()

        result = synth("data")
        assert result == "View1: data | View2: data"

        # All modules are found
        all_modules = list(synth.modules())
        assert len(all_modules) == 4  # synth, analyzer, view1, view2


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Discovery Through Hierarchy Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParameterDiscovery:
    """Tests for discovering parameters through module hierarchy."""

    def test_parameters_from_direct_children(self) -> None:
        """Parameters in direct children are discovered."""

        class Parent(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.child = Prefixer("child_prefix")

        parent = Parent()
        params = list(parent.parameters())

        assert len(params) == 1
        assert params[0].value == "child_prefix"

    def test_parameters_from_nested_children(self) -> None:
        """Parameters in nested children are discovered."""

        class Deep(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.param = Parameter("deep_value", description="test")

        class Middle(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.deep = Deep()
                self.param = Parameter("middle_value", description="test")

        class Top(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.middle = Middle()
                self.param = Parameter("top_value", description="test")

        top = Top()
        params = list(top.parameters())

        assert len(params) == 3
        values = {p.value for p in params}
        assert values == {"top_value", "middle_value", "deep_value"}

    def test_named_parameters_hierarchical_names(self) -> None:
        """Named parameters have correct hierarchical dot-separated names."""

        class Inner(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.weight = Parameter("w", description="test")
                self.bias = Parameter("b", description="test")

        class Outer(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.inner1 = Inner()
                self.inner2 = Inner()
                self.own_param = Parameter("own", description="test")

        outer = Outer()
        named = dict(outer.named_parameters())

        expected_names = {
            "own_param",
            "inner1.weight",
            "inner1.bias",
            "inner2.weight",
            "inner2.bias",
        }
        assert set(named.keys()) == expected_names

    def test_llm_inference_system_prompts_discovered(self) -> None:
        """System prompts from LLMInference are discovered as parameters."""

        class ChatBot(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.greeter = LLMInference(alias="a", system_prompt="Greet users.")
                self.helper = LLMInference(alias="b", system_prompt="Help users.")
                self.closer = LLMInference(alias="c", system_prompt="Say goodbye.")

        chatbot = ChatBot()
        params = list(chatbot.parameters())

        assert len(params) == 3
        prompts = {p.value for p in params}
        assert prompts == {"Greet users.", "Help users.", "Say goodbye."}


# ─────────────────────────────────────────────────────────────────────────────
# Complex Composition Patterns
# ─────────────────────────────────────────────────────────────────────────────


class TestComplexPatterns:
    """Tests for complex composition patterns combining multiple techniques."""

    def test_diamond_pattern(self) -> None:
        """Diamond pattern: A → (B, C) → D."""

        class DiamondModule(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.start = Uppercaser()
                self.left = Prefixer("LEFT")
                self.right = Prefixer("RIGHT")

            def forward(self, text: str) -> str:
                upper = self.start(text)
                left_result = self.left(upper)
                right_result = self.right(upper)
                return f"{left_result} | {right_result}"

        diamond = DiamondModule()

        result = diamond("hello")
        assert result == "LEFT: HELLO | RIGHT: HELLO"

    def test_complex_nested_with_shared_modules(self) -> None:
        """Complex nesting with modules at different levels."""

        class Processor(InferenceModule):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.prefix = Parameter(name, description="test")

            def forward(self, x: str) -> str:
                return f"[{self.prefix.value}:{x}]"

        class SubPipeline(InferenceModule):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.proc1 = Processor(f"{name}_1")
                self.proc2 = Processor(f"{name}_2")

            def forward(self, x: str) -> str:
                return self.proc2(self.proc1(x))

        class MainPipeline(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.sub_a = SubPipeline("A")
                self.sub_b = SubPipeline("B")
                self.final = Processor("FINAL")

            def forward(self, x: str) -> str:
                a_result = self.sub_a(x)
                b_result = self.sub_b(x)
                combined = f"{a_result}+{b_result}"
                return self.final(combined)

        main = MainPipeline()

        # Count all modules
        all_modules = list(main.modules())
        assert len(all_modules) == 8  # main, sub_a, sub_b, 4 processors, final

        # Count all parameters
        params = list(main.parameters())
        assert len(params) == 5

        # Verify parameter names
        named = dict(main.named_parameters())
        expected_names = {
            "sub_a.proc1.prefix",
            "sub_a.proc2.prefix",
            "sub_b.proc1.prefix",
            "sub_b.proc2.prefix",
            "final.prefix",
        }
        assert set(named.keys()) == expected_names

        # Execution works
        result = main("x")
        expected = "[FINAL:[A_2:[A_1:x]]+[B_2:[B_1:x]]]"
        assert result == expected

    def test_module_reuse_different_names(self) -> None:
        """Same module class used multiple times with different names."""

        class MultiAdder(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.add_1 = Adder(1)
                self.add_2 = Adder(2)
                self.add_3 = Adder(3)

            def forward(self, x: int) -> int:
                return self.add_1(x) + self.add_2(x) + self.add_3(x)

        multi = MultiAdder()

        # Each has its own registration
        assert len(multi._children) == 3
        assert multi.add_1.amount == 1
        assert multi.add_2.amount == 2
        assert multi.add_3.amount == 3

        # Execution: (5+1) + (5+2) + (5+3) = 21
        result = multi(5)
        assert result == 21


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestCompositionEdgeCases:
    """Edge case tests for module composition."""

    def test_empty_module_has_no_children(self) -> None:
        """Module with no children has empty children list."""

        class Empty(InferenceModule):
            def __init__(self) -> None:
                super().__init__()

            def forward(self) -> str:
                return "empty"

        empty = Empty()

        assert list(empty.children()) == []
        assert list(empty.parameters()) == []
        assert list(empty.modules()) == [empty]

    def test_module_only_has_parameters(self) -> None:
        """Module with only parameters (no child modules)."""

        class ParamsOnly(InferenceModule):
            def __init__(self) -> None:
                super().__init__()
                self.p1 = Parameter("v1", description="test")
                self.p2 = Parameter("v2", description="test")

            def forward(self) -> str:
                return f"{self.p1.value},{self.p2.value}"

        params_only = ParamsOnly()

        assert list(params_only.children()) == []
        assert len(list(params_only.parameters())) == 2
        assert list(params_only.modules()) == [params_only]

    def test_deep_nesting_performance(self) -> None:
        """Deep nesting doesn't cause performance issues."""

        # Create 50-level deep nesting
        class DeepModule(InferenceModule):
            def __init__(self, depth: int) -> None:
                super().__init__()
                self.param = Parameter(f"depth_{depth}", description="test")
                if depth > 0:
                    self.child = DeepModule(depth - 1)

            def forward(self, x: int) -> int:
                if hasattr(self, "child"):
                    return self.child(x) + 1
                return x

        deep = DeepModule(50)

        # Should complete quickly
        all_modules = list(deep.modules())
        assert len(all_modules) == 51

        all_params = list(deep.parameters())
        assert len(all_params) == 51

        # Execution works
        result = deep(0)
        assert result == 50

    def test_wide_module_with_many_children(self) -> None:
        """Module with many children at same level."""

        class WideModule(InferenceModule):
            def __init__(self, num_children: int) -> None:
                super().__init__()
                for i in range(num_children):
                    setattr(self, f"child_{i}", Adder(i))

            def forward(self, x: int) -> int:
                total = 0
                for i in range(len(self._children)):
                    child = getattr(self, f"child_{i}")
                    total += child(x)
                return total

        wide = WideModule(100)

        assert len(list(wide.children())) == 100
        assert len(list(wide.modules())) == 101  # wide + 100 children

        # Sum of (x + 0) + (x + 1) + ... + (x + 99) = 100x + sum(0..99)
        # = 100*1 + (99*100/2) = 100 + 4950 = 5050
        result = wide(1)
        assert result == 5050
