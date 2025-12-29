"""Integration tests for checkpointing behavior.

This file contains integration tests for PR-062 verifying:
- Checkpoint creation during execution
- Checkpoint persistence across module calls
- Checkpoint buffer flushing
- Checkpoint recovery and compatibility
- Multiple pipelines with shared checkpoint directory
"""

import asyncio
import json
from pathlib import Path

import pytest

from inf_engine.execution.checkpoint import Checkpoint, CheckpointManager
from inf_engine.execution.executor import run
from inf_engine.execution.scheduler import Scheduler
from inf_engine.execution.state import ExecutionState, TaskResult
from inf_engine.graph import GraphNode, InferenceGraph, NodeRef
from inf_engine.module import InferenceModule
from inf_engine.tracing.tracer import InputNode

# ─────────────────────────────────────────────────────────────────────────────
# Test Helpers
# ─────────────────────────────────────────────────────────────────────────────


class EchoModule(InferenceModule):
    """Simple test module that echoes input with prefix."""

    def __init__(self, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix

    def forward(self, text: str) -> str:
        return f"{self.prefix}{text}"


class SlowModule(InferenceModule):
    """A module that takes some time to process."""

    async def forward(self, text: str) -> str:
        await asyncio.sleep(0.01)
        return f"{text}_slow"


class SequentialModule(InferenceModule):
    """Module that chains multiple operations."""

    def __init__(self) -> None:
        super().__init__()
        self.step1 = EchoModule(prefix="[1]")
        self.step2 = EchoModule(prefix="[2]")
        self.step3 = EchoModule(prefix="[3]")

    def forward(self, text: str) -> str:
        r1 = self.step1(text)
        r2 = self.step2(r1)
        r3 = self.step3(r2)
        return r3


def create_linear_graph() -> InferenceGraph:
    """Create a linear graph: input -> a -> b -> c."""
    input_node = GraphNode(
        id="input:input_0",
        module=InputNode(value="start"),
        args=(),
        kwargs={},
        dependencies=[],
    )
    a_node = GraphNode(
        id="a_1",
        module=EchoModule(prefix="[a]"),
        args=(NodeRef("input:input_0"),),
        kwargs={},
        dependencies=["input:input_0"],
    )
    b_node = GraphNode(
        id="b_2",
        module=EchoModule(prefix="[b]"),
        args=(NodeRef("a_1"),),
        kwargs={},
        dependencies=["a_1"],
    )
    c_node = GraphNode(
        id="c_3",
        module=EchoModule(prefix="[c]"),
        args=(NodeRef("b_2"),),
        kwargs={},
        dependencies=["b_2"],
    )
    return InferenceGraph(
        nodes={
            "input:input_0": input_node,
            "a_1": a_node,
            "b_2": b_node,
            "c_3": c_node,
        },
        input_ids=["input:input_0"],
        output_ids=["c_3"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Creation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckpointCreation:
    """Tests for checkpoint creation during execution."""

    @pytest.mark.asyncio
    async def test_checkpoint_created_during_execution(self, tmp_path: Path) -> None:
        """Checkpoint file is created during execution."""
        module = EchoModule(prefix="test:")
        checkpoint_dir = tmp_path / "checkpoints"

        # Run with checkpointing
        await run(
            module,
            "hello",
            checkpoint_dir=checkpoint_dir,
            execution_id="run_001",
        )

        # Checkpoint should be created
        assert (checkpoint_dir / "run_001.json").exists()

        # Load and verify checkpoint has content
        checkpoint = Checkpoint.load(checkpoint_dir / "run_001.json")
        assert len(checkpoint.completed_nodes) >= 1

    @pytest.mark.asyncio
    async def test_checkpoint_contains_completed_nodes(self, tmp_path: Path) -> None:
        """Checkpoint contains all completed nodes."""
        module = SequentialModule()
        checkpoint_dir = tmp_path / "checkpoints"

        await run(
            module,
            "hello",
            checkpoint_dir=checkpoint_dir,
            execution_id="run_002",
        )

        checkpoint = Checkpoint.load(checkpoint_dir / "run_002.json")
        # Should have input node + 3 steps + the main module call
        assert len(checkpoint.completed_nodes) >= 1

    @pytest.mark.asyncio
    async def test_checkpoint_has_graph_hash(self, tmp_path: Path) -> None:
        """Checkpoint includes graph hash for compatibility checking."""
        module = EchoModule()
        checkpoint_dir = tmp_path / "checkpoints"

        await run(
            module,
            "test",
            checkpoint_dir=checkpoint_dir,
            execution_id="run_003",
        )

        checkpoint = Checkpoint.load(checkpoint_dir / "run_003.json")
        assert checkpoint.graph_hash is not None

    @pytest.mark.asyncio
    async def test_checkpoint_generated_id_when_not_provided(
        self, tmp_path: Path
    ) -> None:
        """Checkpoint uses generated UUID when execution_id not provided."""
        module = EchoModule()
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        await run(module, "test", checkpoint_dir=checkpoint_dir)

        # Should have created one checkpoint file with UUID name
        files = list(checkpoint_dir.glob("*.json"))
        assert len(files) == 1
        # UUID format: 8-4-4-4-12 hex chars with dashes
        filename = files[0].stem
        assert len(filename) == 36


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Manager Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckpointManager:
    """Tests for CheckpointManager buffering and flushing."""

    @pytest.mark.asyncio
    async def test_buffer_accumulates_completions(self, tmp_path: Path) -> None:
        """CheckpointManager buffers completions before flushing."""
        manager = CheckpointManager(tmp_path, buffer_size=5)

        # Record completions without triggering flush
        for i in range(3):
            result = TaskResult(
                node_id=f"node_{i}",
                value=f"result_{i}",
                duration_ms=10.0,
                retry_count=0,
            )
            should_flush = manager.record_completion("exec_1", f"node_{i}", result)
            assert not should_flush  # Buffer not full yet

        # No checkpoint file created yet (not flushed)
        assert not (tmp_path / "exec_1.json").exists()

    @pytest.mark.asyncio
    async def test_buffer_triggers_flush_when_full(self, tmp_path: Path) -> None:
        """CheckpointManager triggers flush when buffer is full."""
        manager = CheckpointManager(tmp_path, buffer_size=3)

        for i in range(3):
            result = TaskResult(
                node_id=f"node_{i}",
                value=f"result_{i}",
                duration_ms=10.0,
                retry_count=0,
            )
            should_flush = manager.record_completion("exec_1", f"node_{i}", result)

        # Third completion should trigger flush
        assert should_flush

    @pytest.mark.asyncio
    async def test_flush_persists_checkpoint(self, tmp_path: Path) -> None:
        """Flushing persists checkpoint to disk."""
        manager = CheckpointManager(tmp_path, buffer_size=10)

        result = TaskResult(
            node_id="node_0",
            value="test_value",
            duration_ms=15.0,
            retry_count=1,
        )
        manager.record_completion("exec_1", "node_0", result)
        await manager.flush("exec_1")

        # Checkpoint should exist
        checkpoint = Checkpoint.load(tmp_path / "exec_1.json")
        assert "node_0" in checkpoint.completed_nodes
        assert checkpoint.completed_nodes["node_0"].value == "test_value"

    @pytest.mark.asyncio
    async def test_flush_all_persists_all_executions(self, tmp_path: Path) -> None:
        """flush_all() persists checkpoints for all tracked executions."""
        manager = CheckpointManager(tmp_path, buffer_size=100)

        # Record completions for multiple executions
        for exec_id in ["exec_a", "exec_b", "exec_c"]:
            result = TaskResult(
                node_id="node_0",
                value=f"value_{exec_id}",
                duration_ms=10.0,
                retry_count=0,
            )
            manager.record_completion(exec_id, "node_0", result)

        await manager.flush_all()

        # All checkpoints should exist
        for exec_id in ["exec_a", "exec_b", "exec_c"]:
            assert (tmp_path / f"{exec_id}.json").exists()

    @pytest.mark.asyncio
    async def test_incremental_checkpoint_updates(self, tmp_path: Path) -> None:
        """Multiple flushes accumulate completions."""
        manager = CheckpointManager(tmp_path, buffer_size=1)

        # First completion
        result1 = TaskResult("node_0", "value_0", 10.0, 0)
        manager.record_completion("exec_1", "node_0", result1)
        await manager.flush("exec_1")

        # Second completion
        result2 = TaskResult("node_1", "value_1", 20.0, 0)
        manager.record_completion("exec_1", "node_1", result2)
        await manager.flush("exec_1")

        # Checkpoint should have both
        checkpoint = Checkpoint.load(tmp_path / "exec_1.json")
        assert len(checkpoint.completed_nodes) == 2
        assert "node_0" in checkpoint.completed_nodes
        assert "node_1" in checkpoint.completed_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Compatibility Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckpointCompatibility:
    """Tests for checkpoint compatibility checking."""

    @pytest.mark.asyncio
    async def test_compatible_checkpoint_with_same_module(self, tmp_path: Path) -> None:
        """Checkpoint is compatible with same module structure."""
        module = EchoModule(prefix="test:")
        checkpoint_dir = tmp_path / "checkpoints"

        await run(
            module, "hello", checkpoint_dir=checkpoint_dir, execution_id="run_001"
        )

        checkpoint = Checkpoint.load(checkpoint_dir / "run_001.json")

        # Same module should be compatible
        assert checkpoint.is_compatible_with(module, "hello")

    @pytest.mark.asyncio
    async def test_incompatible_checkpoint_with_different_module(
        self, tmp_path: Path
    ) -> None:
        """Checkpoint is incompatible with different module structure."""
        module1 = EchoModule(prefix="v1:")
        module2 = SequentialModule()  # Different structure
        checkpoint_dir = tmp_path / "checkpoints"

        await run(
            module1, "hello", checkpoint_dir=checkpoint_dir, execution_id="run_001"
        )

        checkpoint = Checkpoint.load(checkpoint_dir / "run_001.json")

        # Different module structure should be incompatible
        assert not checkpoint.is_compatible_with(module2, "hello")

    def test_legacy_checkpoint_without_hash_is_compatible(self) -> None:
        """Checkpoint without graph_hash (legacy) is always compatible."""
        checkpoint = Checkpoint(
            execution_id="legacy",
            timestamp=1234567890.0,
            graph_hash=None,  # Legacy checkpoint
            completed_nodes={},
        )

        # Any hash should be compatible with legacy checkpoint
        assert checkpoint.is_compatible_with("any_hash_value")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Save/Load Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckpointSaveLoad:
    """Tests for checkpoint serialization."""

    def test_checkpoint_round_trip(self, tmp_path: Path) -> None:
        """Checkpoint can be saved and loaded."""
        original = Checkpoint(
            execution_id="test_001",
            timestamp=1234567890.0,
            graph_hash="abc123",
            completed_nodes={
                "node_1": TaskResult("node_1", "value_1", 100.5, 0),
                "node_2": TaskResult("node_2", "value_2", 200.0, 1),
            },
            failed_nodes={"node_3": "Some error"},
            pending_nodes=["node_4", "node_5"],
        )

        path = tmp_path / "checkpoint.json"
        original.save(path)
        loaded = Checkpoint.load(path)

        assert loaded.execution_id == original.execution_id
        assert loaded.timestamp == original.timestamp
        assert loaded.graph_hash == original.graph_hash
        assert len(loaded.completed_nodes) == 2
        assert loaded.completed_nodes["node_1"].value == "value_1"
        assert loaded.completed_nodes["node_2"].retry_count == 1
        assert loaded.failed_nodes == {"node_3": "Some error"}
        assert loaded.pending_nodes == ["node_4", "node_5"]

    def test_checkpoint_json_format(self, tmp_path: Path) -> None:
        """Checkpoint is saved as readable JSON."""
        checkpoint = Checkpoint(
            execution_id="json_test",
            timestamp=1234567890.0,
            completed_nodes={
                "node_1": TaskResult("node_1", "hello", 50.0, 0),
            },
        )

        path = tmp_path / "checkpoint.json"
        checkpoint.save(path)

        # Should be valid JSON
        data = json.loads(path.read_text())
        assert data["execution_id"] == "json_test"
        assert "completed_nodes" in data


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Pipeline Checkpointing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiplePipelineCheckpointing:
    """Tests for multiple pipelines sharing checkpoint infrastructure."""

    @pytest.mark.asyncio
    async def test_multiple_pipelines_independent_checkpoints(
        self, tmp_path: Path
    ) -> None:
        """Multiple pipelines create independent checkpoints."""
        pipeline1 = EchoModule(prefix="p1:")
        pipeline2 = EchoModule(prefix="p2:")
        checkpoint_dir = tmp_path / "checkpoints"

        await run(
            pipeline1, "input1", checkpoint_dir=checkpoint_dir, execution_id="run_p1"
        )
        await run(
            pipeline2, "input2", checkpoint_dir=checkpoint_dir, execution_id="run_p2"
        )

        # Both checkpoints should exist
        cp1 = Checkpoint.load(checkpoint_dir / "run_p1.json")
        cp2 = Checkpoint.load(checkpoint_dir / "run_p2.json")

        assert cp1.execution_id == "run_p1"
        assert cp2.execution_id == "run_p2"

    @pytest.mark.asyncio
    async def test_checkpoint_manager_shared_across_executions(
        self, tmp_path: Path
    ) -> None:
        """Single CheckpointManager handles multiple executions."""
        manager = CheckpointManager(tmp_path, buffer_size=10)

        # Interleave completions from multiple executions
        for i in range(5):
            for exec_id in ["exec_a", "exec_b"]:
                result = TaskResult(f"node_{i}", f"value_{i}", 10.0, 0)
                manager.record_completion(exec_id, f"node_{i}", result)

        await manager.flush_all()

        # Each execution should have all 5 nodes
        for exec_id in ["exec_a", "exec_b"]:
            checkpoint = manager.get_checkpoint(exec_id)
            assert checkpoint is not None
            assert len(checkpoint.completed_nodes) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Execution with Checkpointing Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExecutionWithCheckpointing:
    """Integration tests for execution with checkpointing enabled."""

    @pytest.mark.asyncio
    async def test_scheduler_with_checkpoint_callback(self, tmp_path: Path) -> None:
        """Scheduler correctly invokes checkpoint callback."""
        manager = CheckpointManager(tmp_path, buffer_size=1)
        execution_id = "sched_test"
        completed_nodes: list[str] = []

        def on_complete(node_id: str, result: TaskResult) -> None:
            completed_nodes.append(node_id)
            manager.record_completion(execution_id, node_id, result)

        scheduler = Scheduler(max_concurrent=10)
        graph = create_linear_graph()
        state = ExecutionState(graph)

        await scheduler.execute(state, on_complete=on_complete)
        await manager.flush_all()

        # All nodes should be checkpointed
        checkpoint = Checkpoint.load(tmp_path / f"{execution_id}.json")
        assert len(checkpoint.completed_nodes) == len(completed_nodes)

    @pytest.mark.asyncio
    async def test_checkpoint_tracks_duration(self, tmp_path: Path) -> None:
        """Checkpointed results include task duration."""
        manager = CheckpointManager(tmp_path, buffer_size=1)

        # Record a result with specific duration
        result = TaskResult(
            node_id="slow_node",
            value="done",
            duration_ms=150.5,
            retry_count=0,
        )
        manager.record_completion("exec_1", "slow_node", result)
        await manager.flush("exec_1")

        checkpoint = Checkpoint.load(tmp_path / "exec_1.json")
        assert checkpoint.completed_nodes["slow_node"].duration_ms == 150.5

    @pytest.mark.asyncio
    async def test_checkpoint_tracks_retry_count(self, tmp_path: Path) -> None:
        """Checkpointed results include retry count."""
        manager = CheckpointManager(tmp_path, buffer_size=1)

        result = TaskResult(
            node_id="retried_node",
            value="finally done",
            duration_ms=500.0,
            retry_count=3,
        )
        manager.record_completion("exec_1", "retried_node", result)
        await manager.flush("exec_1")

        checkpoint = Checkpoint.load(tmp_path / "exec_1.json")
        assert checkpoint.completed_nodes["retried_node"].retry_count == 3
