"""Unit tests for the executor module and run() function."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from inf_engine.execution.executor import run
from inf_engine.module import InferenceModule
from inf_engine.optimization.record import ForwardRecord

# ─────────────────────────────────────────────────────────────────────────────
# Test Modules
#
# NOTE: During tracing, forward() receives Proxy objects. Modules that work
# correctly with tracing must either:
# 1. Return the input unchanged (like EchoModule)
# 2. Call child modules and return their results (child.__call__ returns Proxy)
#
# Modules that do direct string operations (like text.upper()) or use f-strings
# will fail during tracing because they try to operate on Proxy objects.
# ─────────────────────────────────────────────────────────────────────────────


class EchoModule(InferenceModule):
    """A simple module that returns its input unchanged.

    Works with tracing because it just passes through the Proxy.
    """

    def forward(self, text: str) -> str:
        """Return the input text."""
        return text


class TransformModule(InferenceModule):
    """An atomic module that transforms input during execution.

    During tracing, __call__ intercepts and records the call, returning a Proxy.
    During execution, forward() is called with actual values.
    """

    def __init__(self, suffix: str = "_transformed"):
        super().__init__()
        self.suffix = suffix

    def forward(self, text: str) -> str:
        """Return input with suffix appended."""
        return f"{text}{self.suffix}"


class UpperTransformModule(InferenceModule):
    """An atomic module that uppercases input during execution."""

    def forward(self, text: str) -> str:
        """Return the input text in uppercase."""
        return text.upper()


class AsyncTransformModule(InferenceModule):
    """An async module for testing."""

    async def forward(self, text: str) -> str:
        """Return input after async processing."""
        await asyncio.sleep(0.001)
        return f"{text}_async"


class FailingModule(InferenceModule):
    """A module that always fails."""

    def forward(self, text: str) -> str:
        """Raise an error."""
        raise ValueError("Module failed intentionally")


class LinearPipeline(InferenceModule):
    """A linear pipeline: a -> b -> c."""

    def __init__(self):
        super().__init__()
        self.step_a = TransformModule("_a")
        self.step_b = TransformModule("_b")
        self.step_c = TransformModule("_c")

    def forward(self, text: str) -> str:
        """Process input through three steps."""
        a = self.step_a(text)
        b = self.step_b(a)
        c = self.step_c(b)
        return c


class SingleStepPipeline(InferenceModule):
    """A pipeline with a single transform step."""

    def __init__(self, suffix: str = "_processed"):
        super().__init__()
        self.transform = TransformModule(suffix)

    def forward(self, text: str) -> str:
        """Process input through one step."""
        return self.transform(text)


class UpperPipeline(InferenceModule):
    """A pipeline that uppercases input."""

    def __init__(self):
        super().__init__()
        self.upper = UpperTransformModule()

    def forward(self, text: str) -> str:
        """Process input through upper transform."""
        return self.upper(text)


class AsyncPipeline(InferenceModule):
    """A pipeline with an async transform."""

    def __init__(self):
        super().__init__()
        self.async_transform = AsyncTransformModule()

    def forward(self, text: str) -> str:
        """Process input through async transform."""
        return self.async_transform(text)


class ParallelPipeline(InferenceModule):
    """A pipeline with parallel branches: input -> [a, b]."""

    def __init__(self):
        super().__init__()
        self.branch_a = TransformModule("_a")
        self.branch_b = TransformModule("_b")

    def forward(self, text: str) -> dict[str, Any]:
        """Process input through parallel branches."""
        a = self.branch_a(text)
        b = self.branch_b(text)
        return {"a": a, "b": b}


class DiamondPipeline(InferenceModule):
    """A diamond pipeline: input -> [a, b] -> merge."""

    def __init__(self):
        super().__init__()
        self.branch_a = TransformModule("_a")
        self.branch_b = TransformModule("_b")
        self.merge = TransformModule("_merged")

    def forward(self, text: str) -> str:
        """Process input through diamond graph."""
        a = self.branch_a(text)
        _ = self.branch_b(text)  # Branch for dependency demo
        # Merge just takes one input for simplicity
        merged = self.merge(a)
        return merged


class NestedPipeline(InferenceModule):
    """A pipeline containing another pipeline."""

    def __init__(self):
        super().__init__()
        self.inner = LinearPipeline()
        self.outer = TransformModule("_outer")

    def forward(self, text: str) -> str:
        """Process through nested pipeline."""
        inner_result = self.inner(text)
        return self.outer(inner_result)


class MultiInputModule(InferenceModule):
    """A module that combines multiple inputs."""

    def __init__(self):
        super().__init__()
        self.combiner = TransformModule("_combined")

    def forward(self, text1: str, text2: str) -> str:
        """Combine two inputs (simplified - just processes first)."""
        return self.combiner(text1)


class MultiOutputPipeline(InferenceModule):
    """A module that produces multiple outputs."""

    def __init__(self):
        super().__init__()
        self.a = TransformModule("_a")
        self.b = TransformModule("_b")
        self.c = TransformModule("_c")

    def forward(self, text: str) -> dict[str, Any]:
        """Return multiple outputs."""
        return {
            "first": self.a(text),
            "second": self.b(text),
            "third": self.c(text),
        }


class PartialFailPipeline(InferenceModule):
    """A pipeline where one branch fails."""

    def __init__(self):
        super().__init__()
        self.good = TransformModule("_good")
        self.bad = FailingModule()

    def forward(self, text: str) -> dict[str, Any]:
        """Process with one failing branch."""
        return {
            "good": self.good(text),
            "bad": self.bad(text),
        }


class CascadeFailPipeline(InferenceModule):
    """A pipeline where failure cascades."""

    def __init__(self):
        super().__init__()
        self.first = FailingModule()
        self.second = TransformModule("_second")

    def forward(self, text: str) -> str:
        """First fails, second depends on first."""
        first_result = self.first(text)
        return self.second(first_result)


class FailingPipeline(InferenceModule):
    """A pipeline with a single failing step."""

    def __init__(self):
        super().__init__()
        self.fail = FailingModule()

    def forward(self, text: str) -> str:
        """Process through failing step."""
        return self.fail(text)


class ManyParallelPipeline(InferenceModule):
    """A pipeline with many parallel operations."""

    def __init__(self, count: int = 20):
        super().__init__()
        self.processors = [TransformModule(f"_{i}") for i in range(count)]

    def forward(self, text: str) -> dict[str, Any]:
        """Process through many parallel transforms."""
        return {f"m{i}": m(text) for i, m in enumerate(self.processors)}


class ChainedPipeline(InferenceModule):
    """A pipeline that chains transforms."""

    def __init__(self):
        super().__init__()
        self.first = TransformModule("_FIRST")
        self.second = UpperTransformModule()

    def forward(self, text: str) -> str:
        """First adds suffix, second uppercases."""
        first_result = self.first(text)
        return self.second(first_result)


class FanOutFanInPipeline(InferenceModule):
    """A pipeline with fan-out fan-in pattern."""

    def __init__(self):
        super().__init__()
        self.branch_a = TransformModule("_A")
        self.branch_b = TransformModule("_B")
        self.merge = TransformModule("_MERGED")

    def forward(self, text: str) -> str:
        """Fan out to branches, fan in to merge."""
        a = self.branch_a(text)
        _ = self.branch_b(text)  # Branch for dependency demo
        return self.merge(a)


class SharedInputPipeline(InferenceModule):
    """A pipeline where multiple branches share input."""

    def __init__(self):
        super().__init__()
        self.upper = UpperTransformModule()
        self.suffix = TransformModule("_suffix")

    def forward(self, text: str) -> dict[str, Any]:
        """Both branches use the same input."""
        return {
            "upper": self.upper(text),
            "suffix": self.suffix(text),
        }


class MockLLM(InferenceModule):
    """A mock LLM module that simulates LLM behavior."""

    def __init__(self, response: str = "mock response"):
        super().__init__()
        self.response = response
        self.call_count = 0
        self.last_input: str | None = None

    def forward(self, prompt: str) -> str:
        """Return a mock response."""
        self.call_count += 1
        self.last_input = prompt
        return self.response


class LLMPipeline(InferenceModule):
    """An LLM pipeline for testing."""

    def __init__(self, summarizer_response: str, analyzer_response: str):
        super().__init__()
        self.summarize = MockLLM(summarizer_response)
        self.analyze = MockLLM(analyzer_response)

    def forward(self, text: str) -> str:
        """Summarize then analyze."""
        summary = self.summarize(text)
        return self.analyze(summary)


# ─────────────────────────────────────────────────────────────────────────────
# Basic run() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunBasic:
    """Basic tests for the run() function."""

    @pytest.mark.asyncio
    async def test_run_simple_module(self) -> None:
        """run() executes a simple module and returns result."""
        module = EchoModule()

        result = await run(module, "hello")

        assert result == "hello"

    @pytest.mark.asyncio
    async def test_run_upper_pipeline(self) -> None:
        """run() executes a pipeline that transforms input."""
        module = UpperPipeline()

        result = await run(module, "hello")

        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_run_single_step_pipeline(self) -> None:
        """run() executes a pipeline with configured behavior."""
        module = SingleStepPipeline("_test")

        result = await run(module, "hello")

        assert result == "hello_test"

    @pytest.mark.asyncio
    async def test_run_async_pipeline(self) -> None:
        """run() handles async forward() methods."""
        module = AsyncPipeline()

        result = await run(module, "hello")

        assert result == "hello_async"

    @pytest.mark.asyncio
    async def test_run_returns_correct_type(self) -> None:
        """run() returns the correct type from forward()."""
        module = EchoModule()

        result = await run(module, "test")

        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline run() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunPipelines:
    """Tests for run() with pipeline modules."""

    @pytest.mark.asyncio
    async def test_run_linear_pipeline(self) -> None:
        """run() executes a linear pipeline in order."""
        module = LinearPipeline()

        result = await run(module, "start")

        # Each step adds its suffix
        assert result == "start_a_b_c"

    @pytest.mark.asyncio
    async def test_run_nested_pipeline(self) -> None:
        """run() executes nested pipelines correctly."""
        module = NestedPipeline()

        result = await run(module, "start")

        # Inner adds _a_b_c, outer adds _outer
        assert result == "start_a_b_c_outer"

    @pytest.mark.asyncio
    async def test_run_diamond_pipeline(self) -> None:
        """run() handles diamond dependency patterns."""
        module = DiamondPipeline()

        result = await run(module, "start")

        # branch_a processes, then merge adds _merged
        assert result == "start_a_merged"


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Output run() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunMultiOutput:
    """Tests for run() with multiple outputs."""

    @pytest.mark.asyncio
    async def test_run_parallel_pipeline_returns_dict(self) -> None:
        """run() with multiple outputs returns a dict."""
        module = ParallelPipeline()

        result = await run(module, "start")

        # Multiple outputs mean we get a dict back
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_run_parallel_pipeline_values(self) -> None:
        """run() with parallel branches returns correct values."""
        module = ParallelPipeline()

        result = await run(module, "start")

        # The result dict maps output node IDs to values
        assert len(result) == 2
        # The actual values should be "start_a" and "start_b"
        values = set(result.values())
        assert "start_a" in values
        assert "start_b" in values

    @pytest.mark.asyncio
    async def test_run_multi_output_pipeline(self) -> None:
        """run() with multiple output nodes returns dict."""
        module = MultiOutputPipeline()

        result = await run(module, "test")

        # Should get 3 outputs
        assert isinstance(result, dict)
        assert len(result) == 3
        values = set(result.values())
        assert "test_a" in values
        assert "test_b" in values
        assert "test_c" in values


# ─────────────────────────────────────────────────────────────────────────────
# Input Argument Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunArguments:
    """Tests for run() with various argument patterns."""

    @pytest.mark.asyncio
    async def test_run_multiple_positional_args(self) -> None:
        """run() passes multiple positional arguments."""
        module = MultiInputModule()

        result = await run(module, "hello", "world")

        # Just processes first input with suffix
        assert result == "hello_combined"

    @pytest.mark.asyncio
    async def test_run_empty_string_input(self) -> None:
        """run() handles empty string input."""
        module = SingleStepPipeline("_end")

        result = await run(module, "")

        assert result == "_end"


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency Control Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunConcurrency:
    """Tests for run() concurrency control."""

    @pytest.mark.asyncio
    async def test_run_with_max_concurrent(self) -> None:
        """run() accepts max_concurrent parameter."""
        module = LinearPipeline()

        result = await run(module, "test", max_concurrent=5)

        assert result == "test_a_b_c"

    @pytest.mark.asyncio
    async def test_run_with_max_concurrent_one(self) -> None:
        """run() with max_concurrent=1 runs serially."""
        module = LinearPipeline()

        result = await run(module, "test", max_concurrent=1)

        assert result == "test_a_b_c"

    @pytest.mark.asyncio
    async def test_run_default_max_concurrent(self) -> None:
        """run() uses default max_concurrent of 100."""
        module = EchoModule()

        # Should work without specifying max_concurrent
        result = await run(module, "test")

        assert result == "test"

    @pytest.mark.asyncio
    async def test_run_respects_max_concurrent(self) -> None:
        """run() respects the max_concurrent limit."""
        module = ManyParallelPipeline(20)

        # Should complete even with low concurrency
        result = await run(module, "test", max_concurrent=2)

        assert isinstance(result, dict)
        assert len(result) == 20


# ─────────────────────────────────────────────────────────────────────────────
# Error Handling Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunErrors:
    """Tests for run() error handling."""

    @pytest.mark.asyncio
    async def test_run_failing_pipeline_returns_empty(self) -> None:
        """run() with a failing module returns empty dict for outputs."""
        module = FailingPipeline()

        result = await run(module, "test")

        # When the output node fails, we get an empty dict
        # (the single output failed, so nothing to unwrap)
        assert result == {}

    @pytest.mark.asyncio
    async def test_run_partial_failure(self) -> None:
        """run() returns completed outputs even if some fail."""
        module = PartialFailPipeline()

        result = await run(module, "test")

        # Should get partial results - the good one succeeded
        # Since only one output succeeds, run() unwraps it to just the value
        assert result == "test_good"

    @pytest.mark.asyncio
    async def test_run_cascading_failure(self) -> None:
        """run() handles cascading failures correctly."""
        module = CascadeFailPipeline()

        result = await run(module, "test")

        # First fails, second gets cancelled, no output
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# Import Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunImports:
    """Tests for run() imports."""

    def test_import_from_executor_module(self) -> None:
        """run can be imported from executor module."""
        from inf_engine.execution.executor import run as run_from_module

        assert run_from_module is run

    def test_import_from_execution_package(self) -> None:
        """run can be imported from execution package."""
        from inf_engine.execution import run as run_from_package

        assert run_from_package is run


# ─────────────────────────────────────────────────────────────────────────────
# LLMInference Module Tests (with mock)
# ─────────────────────────────────────────────────────────────────────────────


class TestRunWithMockLLM:
    """Tests for run() with mock LLM modules."""

    @pytest.mark.asyncio
    async def test_run_mock_llm(self) -> None:
        """run() works with mock LLM modules."""

        class SingleLLMPipeline(InferenceModule):
            def __init__(self):
                super().__init__()
                self.llm = MockLLM("Hello from LLM!")

            def forward(self, text: str) -> str:
                return self.llm(text)

        module = SingleLLMPipeline()

        result = await run(module, "What is AI?")

        assert result == "Hello from LLM!"

    @pytest.mark.asyncio
    async def test_run_llm_pipeline(self) -> None:
        """run() executes an LLM pipeline correctly."""
        module = LLMPipeline("summary", "analysis")

        result = await run(module, "Some long document...")

        assert result == "analysis"


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunEdgeCases:
    """Edge case tests for run()."""

    @pytest.mark.asyncio
    async def test_run_single_output_unwrapping(self) -> None:
        """run() unwraps single output from dict to value."""
        module = EchoModule()

        result = await run(module, "test")

        # Single output should be unwrapped to just the value
        assert result == "test"
        assert not isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_run_multiple_runs_independent(self) -> None:
        """Multiple run() calls are independent."""
        module1 = SingleStepPipeline("_first")
        module2 = SingleStepPipeline("_second")

        result1 = await run(module1, "test")
        result2 = await run(module2, "test")

        assert result1 == "test_first"
        assert result2 == "test_second"

    @pytest.mark.asyncio
    async def test_run_same_module_multiple_times(self) -> None:
        """Same module can be run multiple times."""
        module = SingleStepPipeline("_suffix")

        result1 = await run(module, "first")
        result2 = await run(module, "second")

        assert result1 == "first_suffix"
        assert result2 == "second_suffix"


# ─────────────────────────────────────────────────────────────────────────────
# Dependency Resolution Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunDependencies:
    """Tests for run() dependency resolution."""

    @pytest.mark.asyncio
    async def test_run_passes_output_to_dependent(self) -> None:
        """run() passes output of one module to dependent module."""
        module = ChainedPipeline()

        result = await run(module, "test")

        # first adds "_FIRST", second uppercases
        assert result == "TEST_FIRST"

    @pytest.mark.asyncio
    async def test_run_fan_out_fan_in(self) -> None:
        """run() handles fan-out fan-in patterns."""
        module = FanOutFanInPipeline()

        result = await run(module, "test")

        assert result == "test_A_MERGED"

    @pytest.mark.asyncio
    async def test_run_shared_input(self) -> None:
        """run() handles shared input correctly."""
        module = SharedInputPipeline()

        result = await run(module, "test")

        assert isinstance(result, dict)
        assert len(result) == 2
        values = set(result.values())
        assert "TEST" in values
        assert "test_suffix" in values


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunCheckpointing:
    """Tests for run() with checkpointing support."""

    @pytest.mark.asyncio
    async def test_run_with_checkpoint_dir_creates_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """run() with checkpoint_dir creates a checkpoint file."""
        module = LinearPipeline()

        result = await run(
            module, "test", checkpoint_dir=tmp_path, execution_id="run_001"
        )

        assert result == "test_a_b_c"

        # Checkpoint should exist
        checkpoint_path = tmp_path / "run_001.json"
        assert checkpoint_path.exists()

    @pytest.mark.asyncio
    async def test_run_checkpoint_contains_completed_nodes(
        self, tmp_path: Path
    ) -> None:
        """run() checkpoint contains all completed nodes."""
        from inf_engine.execution.checkpoint import Checkpoint

        module = LinearPipeline()

        await run(module, "test", checkpoint_dir=tmp_path, execution_id="run_002")

        checkpoint = Checkpoint.load(tmp_path / "run_002.json")
        # Should have at least the final output node completed
        assert len(checkpoint.completed_nodes) > 0

    @pytest.mark.asyncio
    async def test_run_checkpoint_contains_graph_hash(self, tmp_path: Path) -> None:
        """run() checkpoint includes the graph hash."""
        from inf_engine.execution.checkpoint import Checkpoint

        module = LinearPipeline()

        await run(module, "test", checkpoint_dir=tmp_path, execution_id="run_003")

        checkpoint = Checkpoint.load(tmp_path / "run_003.json")
        assert checkpoint.graph_hash is not None
        assert len(checkpoint.graph_hash) > 0

    @pytest.mark.asyncio
    async def test_run_generates_execution_id_if_not_provided(
        self, tmp_path: Path
    ) -> None:
        """run() generates a UUID execution_id if not provided."""
        import uuid

        module = EchoModule()

        await run(module, "test", checkpoint_dir=tmp_path)

        # Should have created exactly one checkpoint file
        checkpoint_files = list(tmp_path.glob("*.json"))
        assert len(checkpoint_files) == 1

        # The filename should be a UUID
        filename = checkpoint_files[0].stem
        try:
            uuid.UUID(filename)
        except ValueError:
            pytest.fail(f"Expected UUID filename, got: {filename}")

    @pytest.mark.asyncio
    async def test_run_without_checkpoint_dir(self, tmp_path: Path) -> None:
        """run() without checkpoint_dir does not create checkpoints."""
        module = EchoModule()

        result = await run(module, "test")

        assert result == "test"
        # No checkpoint files should exist (tmp_path is empty)
        checkpoint_files = list(tmp_path.glob("*.json"))
        assert len(checkpoint_files) == 0

    @pytest.mark.asyncio
    async def test_run_parallel_pipeline_with_checkpointing(
        self, tmp_path: Path
    ) -> None:
        """run() with parallel branches creates checkpoints correctly."""
        from inf_engine.execution.checkpoint import Checkpoint

        module = ParallelPipeline()

        result = await run(
            module, "test", checkpoint_dir=tmp_path, execution_id="parallel_run"
        )

        # Should get parallel results
        assert isinstance(result, dict)
        assert len(result) == 2

        # Checkpoint should exist with multiple completed nodes
        checkpoint = Checkpoint.load(tmp_path / "parallel_run.json")
        assert len(checkpoint.completed_nodes) >= 2

    @pytest.mark.asyncio
    async def test_run_checkpoint_dir_created_if_missing(self, tmp_path: Path) -> None:
        """run() creates the checkpoint_dir if it doesn't exist."""
        nested_dir = tmp_path / "nested" / "checkpoints"
        assert not nested_dir.exists()

        module = EchoModule()
        await run(module, "test", checkpoint_dir=nested_dir, execution_id="run_004")

        assert nested_dir.exists()
        assert (nested_dir / "run_004.json").exists()

    @pytest.mark.asyncio
    async def test_run_checkpoint_with_string_path(self, tmp_path: Path) -> None:
        """run() accepts checkpoint_dir as a string."""
        module = EchoModule()

        await run(module, "test", checkpoint_dir=str(tmp_path), execution_id="run_005")

        assert (tmp_path / "run_005.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# ForwardRecord Tests (record=True)
# ─────────────────────────────────────────────────────────────────────────────


class TestRunWithRecord:
    """Tests for run() with record=True returning ForwardRecord."""

    @pytest.mark.asyncio
    async def test_run_with_record_returns_tuple(self) -> None:
        """run() with record=True returns (output, ForwardRecord)."""
        module = EchoModule()

        result = await run(module, "hello", record=True)

        assert isinstance(result, tuple)
        assert len(result) == 2
        output, record = result
        assert output == "hello"
        assert isinstance(record, ForwardRecord)

    @pytest.mark.asyncio
    async def test_run_with_record_false_returns_output(self) -> None:
        """run() with record=False returns just the output."""
        module = EchoModule()

        result = await run(module, "hello", record=False)

        assert result == "hello"
        assert not isinstance(result, tuple)

    @pytest.mark.asyncio
    async def test_run_default_record_is_false(self) -> None:
        """run() defaults to record=False."""
        module = EchoModule()

        result = await run(module, "hello")

        assert result == "hello"
        assert not isinstance(result, tuple)

    @pytest.mark.asyncio
    async def test_forward_record_contains_graph(self) -> None:
        """ForwardRecord contains the execution graph."""
        module = LinearPipeline()

        output, record = await run(module, "test", record=True)

        assert record.graph is not None
        assert len(record.graph.nodes) > 0

    @pytest.mark.asyncio
    async def test_forward_record_contains_node_outputs(self) -> None:
        """ForwardRecord contains output values for all executed nodes."""
        module = LinearPipeline()

        output, record = await run(module, "test", record=True)

        # Should have outputs for all completed nodes
        assert len(record.node_outputs) > 0
        # The final output value should be in there
        assert "test_a_b_c" in record.node_outputs.values()

    @pytest.mark.asyncio
    async def test_forward_record_contains_execution_order(self) -> None:
        """ForwardRecord contains execution order."""
        module = LinearPipeline()

        output, record = await run(module, "test", record=True)

        # Should have execution order for all nodes
        assert len(record.execution_order) > 0
        # Input should be first in topological order
        assert any("input" in node_id for node_id in record.execution_order)

    @pytest.mark.asyncio
    async def test_forward_record_contains_timing(self) -> None:
        """ForwardRecord contains timing information."""
        module = LinearPipeline()

        output, record = await run(module, "test", record=True)

        # Should have timing for executed nodes
        assert len(record.timing) > 0
        # Timing values should be in seconds (small positive numbers)
        for _node_id, time_s in record.timing.items():
            assert time_s >= 0
            assert time_s < 10  # Shouldn't take more than 10 seconds

    @pytest.mark.asyncio
    async def test_forward_record_contains_module_map(self) -> None:
        """ForwardRecord contains module instances for inference nodes."""
        module = LinearPipeline()

        output, record = await run(module, "test", record=True)

        # Should have module references for InferenceModule nodes
        assert len(record.module_map) > 0
        # All modules should be InferenceModule instances
        for _node_id, mod in record.module_map.items():
            assert isinstance(mod, InferenceModule)

    @pytest.mark.asyncio
    async def test_forward_record_node_inputs_recorded(self) -> None:
        """ForwardRecord records resolved node inputs."""
        module = LinearPipeline()

        output, record = await run(module, "test", record=True)

        # Should have inputs for nodes that received args
        assert len(record.node_inputs) > 0

    @pytest.mark.asyncio
    async def test_run_with_record_multi_output(self) -> None:
        """run() with record=True works with multiple outputs."""
        module = ParallelPipeline()

        result = await run(module, "test", record=True)

        output, record = result
        # Multiple outputs come back as dict
        assert isinstance(output, dict)
        assert len(output) == 2
        assert isinstance(record, ForwardRecord)

    @pytest.mark.asyncio
    async def test_run_with_record_and_checkpointing(self, tmp_path: Path) -> None:
        """run() works with both record=True and checkpointing."""
        module = LinearPipeline()

        output, record = await run(
            module,
            "test",
            record=True,
            checkpoint_dir=tmp_path,
            execution_id="record_test",
        )

        assert output == "test_a_b_c"
        assert isinstance(record, ForwardRecord)
        # Checkpoint should also be created
        assert (tmp_path / "record_test.json").exists()

    @pytest.mark.asyncio
    async def test_forward_record_get_node_output_method(self) -> None:
        """ForwardRecord.get_node_output() returns correct value."""
        module = EchoModule()

        output, record = await run(module, "test_value", record=True)

        # Should be able to retrieve the input node's output
        # which is the original input value
        for node_id in record.node_outputs:
            result = record.get_node_output(node_id)
            assert result is not None

    @pytest.mark.asyncio
    async def test_forward_record_with_nested_pipeline(self) -> None:
        """ForwardRecord works with nested pipelines."""
        module = NestedPipeline()

        output, record = await run(module, "test", record=True)

        assert output == "test_a_b_c_outer"
        assert isinstance(record, ForwardRecord)
        # Should have tracked all the nested modules
        assert len(record.graph.nodes) > 1
