"""Tests for Checkpoint and CheckpointManager.

Tests the save/load functionality and round-trip serialization
for execution state persistence, as well as the CheckpointManager
for buffered checkpoint writes.
"""

import asyncio
from pathlib import Path

import pytest

from plait.execution.checkpoint import Checkpoint, CheckpointManager
from plait.execution.state import TaskResult


class TestCheckpointCreation:
    """Tests for Checkpoint instantiation."""

    def test_creation_basic(self) -> None:
        """Checkpoint can be created with required fields."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
        )
        assert checkpoint.execution_id == "run_001"
        assert checkpoint.timestamp == 1703520000.0
        assert checkpoint.completed_nodes == {}
        assert checkpoint.failed_nodes == {}
        assert checkpoint.pending_nodes == []

    def test_creation_with_completed_nodes(self) -> None:
        """Checkpoint can be created with completed nodes."""
        result = TaskResult(
            node_id="LLMInference_1",
            value="Generated text",
            duration_ms=250.0,
            retry_count=0,
        )
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            completed_nodes={"LLMInference_1": result},
        )
        assert len(checkpoint.completed_nodes) == 1
        assert checkpoint.completed_nodes["LLMInference_1"].value == "Generated text"

    def test_creation_with_failed_nodes(self) -> None:
        """Checkpoint can be created with failed nodes."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            failed_nodes={
                "LLMInference_2": "API timeout after 3 retries",
                "LLMInference_3": "Invalid response format",
            },
        )
        assert len(checkpoint.failed_nodes) == 2
        assert (
            checkpoint.failed_nodes["LLMInference_2"] == "API timeout after 3 retries"
        )

    def test_creation_with_pending_nodes(self) -> None:
        """Checkpoint can be created with pending nodes."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            pending_nodes=["LLMInference_4", "LLMInference_5", "LLMInference_6"],
        )
        assert len(checkpoint.pending_nodes) == 3
        assert "LLMInference_4" in checkpoint.pending_nodes

    def test_creation_with_all_fields(self) -> None:
        """Checkpoint can be created with all fields populated."""
        completed = {
            "node_1": TaskResult("node_1", "result_1", 100.0, 0),
            "node_2": TaskResult("node_2", "result_2", 150.0, 1),
        }
        failed = {"node_3": "Error message"}
        pending = ["node_4", "node_5"]

        checkpoint = Checkpoint(
            execution_id="full_run",
            timestamp=1703520000.0,
            completed_nodes=completed,
            failed_nodes=failed,
            pending_nodes=pending,
        )

        assert checkpoint.execution_id == "full_run"
        assert len(checkpoint.completed_nodes) == 2
        assert len(checkpoint.failed_nodes) == 1
        assert len(checkpoint.pending_nodes) == 2


class TestCheckpointSave:
    """Tests for Checkpoint.save() method."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """save() creates a JSON file at the specified path."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
        )
        file_path = tmp_path / "checkpoint.json"
        checkpoint.save(file_path)
        assert file_path.exists()

    def test_save_writes_json(self, tmp_path: Path) -> None:
        """save() writes valid JSON content."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
        )
        file_path = tmp_path / "checkpoint.json"
        checkpoint.save(file_path)

        import json

        data = json.loads(file_path.read_text())
        assert data["execution_id"] == "run_001"
        assert data["timestamp"] == 1703520000.0

    def test_save_includes_completed_nodes(self, tmp_path: Path) -> None:
        """save() serializes completed_nodes correctly."""
        result = TaskResult(
            node_id="node_1",
            value="test result",
            duration_ms=123.45,
            retry_count=2,
        )
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            completed_nodes={"node_1": result},
        )
        file_path = tmp_path / "checkpoint.json"
        checkpoint.save(file_path)

        import json

        data = json.loads(file_path.read_text())
        assert "node_1" in data["completed_nodes"]
        assert data["completed_nodes"]["node_1"]["value"] == "test result"
        assert data["completed_nodes"]["node_1"]["duration_ms"] == 123.45
        assert data["completed_nodes"]["node_1"]["retry_count"] == 2

    def test_save_includes_failed_nodes(self, tmp_path: Path) -> None:
        """save() serializes failed_nodes correctly."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            failed_nodes={"node_1": "Error message"},
        )
        file_path = tmp_path / "checkpoint.json"
        checkpoint.save(file_path)

        import json

        data = json.loads(file_path.read_text())
        assert data["failed_nodes"]["node_1"] == "Error message"

    def test_save_includes_pending_nodes(self, tmp_path: Path) -> None:
        """save() serializes pending_nodes correctly."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            pending_nodes=["node_a", "node_b"],
        )
        file_path = tmp_path / "checkpoint.json"
        checkpoint.save(file_path)

        import json

        data = json.loads(file_path.read_text())
        assert data["pending_nodes"] == ["node_a", "node_b"]

    def test_save_overwrites_existing_file(self, tmp_path: Path) -> None:
        """save() overwrites an existing file."""
        file_path = tmp_path / "checkpoint.json"
        file_path.write_text('{"old": "data"}')

        checkpoint = Checkpoint(
            execution_id="new_run",
            timestamp=1703520000.0,
        )
        checkpoint.save(file_path)

        import json

        data = json.loads(file_path.read_text())
        assert data["execution_id"] == "new_run"
        assert "old" not in data

    def test_save_handles_complex_value(self, tmp_path: Path) -> None:
        """save() handles complex values that are JSON serializable."""
        result = TaskResult(
            node_id="node_1",
            value={"nested": {"data": [1, 2, 3], "text": "hello"}},
            duration_ms=100.0,
            retry_count=0,
        )
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            completed_nodes={"node_1": result},
        )
        file_path = tmp_path / "checkpoint.json"
        checkpoint.save(file_path)

        import json

        data = json.loads(file_path.read_text())
        assert data["completed_nodes"]["node_1"]["value"]["nested"]["data"] == [1, 2, 3]


class TestCheckpointLoad:
    """Tests for Checkpoint.load() classmethod."""

    def test_load_reads_file(self, tmp_path: Path) -> None:
        """load() reads a JSON file and returns a Checkpoint."""
        file_path = tmp_path / "checkpoint.json"
        file_path.write_text(
            '{"execution_id": "run_001", "timestamp": 1703520000.0, '
            '"completed_nodes": {}, "failed_nodes": {}, "pending_nodes": []}'
        )

        checkpoint = Checkpoint.load(file_path)
        assert checkpoint.execution_id == "run_001"
        assert checkpoint.timestamp == 1703520000.0

    def test_load_restores_completed_nodes(self, tmp_path: Path) -> None:
        """load() restores completed_nodes with TaskResult objects."""
        file_path = tmp_path / "checkpoint.json"
        file_path.write_text(
            """{
            "execution_id": "run_001",
            "timestamp": 1703520000.0,
            "completed_nodes": {
                "node_1": {
                    "node_id": "node_1",
                    "value": "test result",
                    "duration_ms": 123.45,
                    "retry_count": 2
                }
            },
            "failed_nodes": {},
            "pending_nodes": []
        }"""
        )

        checkpoint = Checkpoint.load(file_path)
        assert "node_1" in checkpoint.completed_nodes
        result = checkpoint.completed_nodes["node_1"]
        assert isinstance(result, TaskResult)
        assert result.node_id == "node_1"
        assert result.value == "test result"
        assert result.duration_ms == 123.45
        assert result.retry_count == 2

    def test_load_restores_failed_nodes(self, tmp_path: Path) -> None:
        """load() restores failed_nodes as a dict."""
        file_path = tmp_path / "checkpoint.json"
        file_path.write_text(
            """{
            "execution_id": "run_001",
            "timestamp": 1703520000.0,
            "completed_nodes": {},
            "failed_nodes": {"node_1": "Error message"},
            "pending_nodes": []
        }"""
        )

        checkpoint = Checkpoint.load(file_path)
        assert checkpoint.failed_nodes["node_1"] == "Error message"

    def test_load_restores_pending_nodes(self, tmp_path: Path) -> None:
        """load() restores pending_nodes as a list."""
        file_path = tmp_path / "checkpoint.json"
        file_path.write_text(
            """{
            "execution_id": "run_001",
            "timestamp": 1703520000.0,
            "completed_nodes": {},
            "failed_nodes": {},
            "pending_nodes": ["node_a", "node_b"]
        }"""
        )

        checkpoint = Checkpoint.load(file_path)
        assert checkpoint.pending_nodes == ["node_a", "node_b"]

    def test_load_handles_missing_optional_fields(self, tmp_path: Path) -> None:
        """load() handles missing optional fields with defaults."""
        file_path = tmp_path / "checkpoint.json"
        file_path.write_text(
            '{"execution_id": "run_001", "timestamp": 1703520000.0, "completed_nodes": {}}'
        )

        checkpoint = Checkpoint.load(file_path)
        assert checkpoint.failed_nodes == {}
        assert checkpoint.pending_nodes == []

    def test_load_handles_missing_retry_count(self, tmp_path: Path) -> None:
        """load() handles missing retry_count in older checkpoints."""
        file_path = tmp_path / "checkpoint.json"
        file_path.write_text(
            """{
            "execution_id": "run_001",
            "timestamp": 1703520000.0,
            "completed_nodes": {
                "node_1": {
                    "node_id": "node_1",
                    "value": "result",
                    "duration_ms": 100.0
                }
            },
            "failed_nodes": {},
            "pending_nodes": []
        }"""
        )

        checkpoint = Checkpoint.load(file_path)
        assert checkpoint.completed_nodes["node_1"].retry_count == 0

    def test_load_raises_for_nonexistent_file(self, tmp_path: Path) -> None:
        """load() raises FileNotFoundError for missing files."""
        file_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            Checkpoint.load(file_path)

    def test_load_raises_for_invalid_json(self, tmp_path: Path) -> None:
        """load() raises JSONDecodeError for invalid JSON."""
        import json

        file_path = tmp_path / "invalid.json"
        file_path.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            Checkpoint.load(file_path)


class TestCheckpointRoundTrip:
    """Tests for save/load round-trip behavior."""

    def test_round_trip_basic(self, tmp_path: Path) -> None:
        """Checkpoint survives save/load round-trip."""
        original = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
        )
        file_path = tmp_path / "checkpoint.json"
        original.save(file_path)
        loaded = Checkpoint.load(file_path)

        assert loaded.execution_id == original.execution_id
        assert loaded.timestamp == original.timestamp
        assert loaded.completed_nodes == original.completed_nodes
        assert loaded.failed_nodes == original.failed_nodes
        assert loaded.pending_nodes == original.pending_nodes

    def test_round_trip_with_completed_nodes(self, tmp_path: Path) -> None:
        """Checkpoint with completed_nodes survives round-trip."""
        original = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            completed_nodes={
                "node_1": TaskResult("node_1", "result_1", 100.0, 0),
                "node_2": TaskResult("node_2", "result_2", 200.0, 1),
            },
        )
        file_path = tmp_path / "checkpoint.json"
        original.save(file_path)
        loaded = Checkpoint.load(file_path)

        assert len(loaded.completed_nodes) == 2
        assert loaded.completed_nodes["node_1"].value == "result_1"
        assert loaded.completed_nodes["node_1"].duration_ms == 100.0
        assert loaded.completed_nodes["node_1"].retry_count == 0
        assert loaded.completed_nodes["node_2"].value == "result_2"
        assert loaded.completed_nodes["node_2"].retry_count == 1

    def test_round_trip_with_failed_nodes(self, tmp_path: Path) -> None:
        """Checkpoint with failed_nodes survives round-trip."""
        original = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            failed_nodes={
                "node_1": "Error 1",
                "node_2": "Error 2",
            },
        )
        file_path = tmp_path / "checkpoint.json"
        original.save(file_path)
        loaded = Checkpoint.load(file_path)

        assert loaded.failed_nodes == original.failed_nodes

    def test_round_trip_with_pending_nodes(self, tmp_path: Path) -> None:
        """Checkpoint with pending_nodes survives round-trip."""
        original = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            pending_nodes=["node_a", "node_b", "node_c"],
        )
        file_path = tmp_path / "checkpoint.json"
        original.save(file_path)
        loaded = Checkpoint.load(file_path)

        assert loaded.pending_nodes == original.pending_nodes

    def test_round_trip_with_all_fields(self, tmp_path: Path) -> None:
        """Checkpoint with all fields survives round-trip."""
        original = Checkpoint(
            execution_id="comprehensive_run",
            timestamp=1703520123.456,
            completed_nodes={
                "input:0": TaskResult("input:0", "input text", 0.1, 0),
                "LLMInference_1": TaskResult("LLMInference_1", "generated", 500.0, 0),
                "LLMInference_2": TaskResult("LLMInference_2", "more text", 450.0, 2),
            },
            failed_nodes={
                "LLMInference_3": "API rate limit exceeded",
                "LLMInference_4": "Connection timeout",
            },
            pending_nodes=["LLMInference_5", "LLMInference_6"],
        )
        file_path = tmp_path / "checkpoint.json"
        original.save(file_path)
        loaded = Checkpoint.load(file_path)

        assert loaded.execution_id == original.execution_id
        assert loaded.timestamp == original.timestamp
        assert len(loaded.completed_nodes) == 3
        assert loaded.completed_nodes["LLMInference_2"].retry_count == 2
        assert len(loaded.failed_nodes) == 2
        assert loaded.failed_nodes["LLMInference_3"] == "API rate limit exceeded"
        assert len(loaded.pending_nodes) == 2
        assert "LLMInference_5" in loaded.pending_nodes

    def test_round_trip_with_complex_value(self, tmp_path: Path) -> None:
        """Checkpoint with complex JSON-serializable values survives round-trip."""
        complex_value = {
            "response": "Generated text",
            "metadata": {
                "tokens_used": 150,
                "model": "gpt-4",
                "finish_reason": "stop",
            },
            "embeddings": [0.1, 0.2, 0.3],
        }
        original = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            completed_nodes={
                "node_1": TaskResult("node_1", complex_value, 100.0, 0),
            },
        )
        file_path = tmp_path / "checkpoint.json"
        original.save(file_path)
        loaded = Checkpoint.load(file_path)

        loaded_value = loaded.completed_nodes["node_1"].value
        assert loaded_value["response"] == "Generated text"
        assert loaded_value["metadata"]["tokens_used"] == 150
        assert loaded_value["embeddings"] == [0.1, 0.2, 0.3]

    def test_round_trip_preserves_node_id_consistency(self, tmp_path: Path) -> None:
        """Round-trip preserves node_id in both dict key and TaskResult."""
        original = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            completed_nodes={
                "my_node": TaskResult("my_node", "value", 100.0, 0),
            },
        )
        file_path = tmp_path / "checkpoint.json"
        original.save(file_path)
        loaded = Checkpoint.load(file_path)

        assert "my_node" in loaded.completed_nodes
        assert loaded.completed_nodes["my_node"].node_id == "my_node"


class TestCheckpointToDictFromDict:
    """Tests for internal serialization methods."""

    def test_to_dict_returns_dict(self) -> None:
        """_to_dict() returns a dictionary."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
        )
        data = checkpoint._to_dict()
        assert isinstance(data, dict)
        assert data["execution_id"] == "run_001"

    def test_from_dict_returns_checkpoint(self) -> None:
        """_from_dict() returns a Checkpoint instance."""
        data = {
            "execution_id": "run_001",
            "timestamp": 1703520000.0,
            "completed_nodes": {},
            "failed_nodes": {},
            "pending_nodes": [],
        }
        checkpoint = Checkpoint._from_dict(data)
        assert isinstance(checkpoint, Checkpoint)
        assert checkpoint.execution_id == "run_001"

    def test_to_dict_from_dict_round_trip(self) -> None:
        """_to_dict() and _from_dict() are inverses."""
        original = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            completed_nodes={
                "node_1": TaskResult("node_1", "value", 100.0, 1),
            },
            failed_nodes={"node_2": "error"},
            pending_nodes=["node_3"],
        )
        data = original._to_dict()
        loaded = Checkpoint._from_dict(data)

        assert loaded.execution_id == original.execution_id
        assert loaded.timestamp == original.timestamp
        assert loaded.completed_nodes["node_1"].value == "value"
        assert loaded.failed_nodes == original.failed_nodes
        assert loaded.pending_nodes == original.pending_nodes


class TestCheckpointGraphHash:
    """Tests for graph_hash field and compatibility checking."""

    def test_creation_without_graph_hash(self) -> None:
        """Checkpoint can be created without graph_hash."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
        )
        assert checkpoint.graph_hash is None

    def test_creation_with_graph_hash(self) -> None:
        """Checkpoint can be created with graph_hash."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            graph_hash="abc123def456",
        )
        assert checkpoint.graph_hash == "abc123def456"

    def test_save_includes_graph_hash(self, tmp_path: Path) -> None:
        """save() serializes graph_hash when present."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            graph_hash="deadbeef1234",
        )
        file_path = tmp_path / "checkpoint.json"
        checkpoint.save(file_path)

        import json

        data = json.loads(file_path.read_text())
        assert data["graph_hash"] == "deadbeef1234"

    def test_save_omits_graph_hash_when_none(self, tmp_path: Path) -> None:
        """save() omits graph_hash when None."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
        )
        file_path = tmp_path / "checkpoint.json"
        checkpoint.save(file_path)

        import json

        data = json.loads(file_path.read_text())
        assert "graph_hash" not in data

    def test_load_restores_graph_hash(self, tmp_path: Path) -> None:
        """load() restores graph_hash from JSON."""
        file_path = tmp_path / "checkpoint.json"
        file_path.write_text(
            """{
            "execution_id": "run_001",
            "timestamp": 1703520000.0,
            "graph_hash": "sha256hashvalue",
            "completed_nodes": {},
            "failed_nodes": {},
            "pending_nodes": []
        }"""
        )

        checkpoint = Checkpoint.load(file_path)
        assert checkpoint.graph_hash == "sha256hashvalue"

    def test_load_handles_missing_graph_hash(self, tmp_path: Path) -> None:
        """load() handles legacy checkpoints without graph_hash."""
        file_path = tmp_path / "checkpoint.json"
        file_path.write_text(
            """{
            "execution_id": "run_001",
            "timestamp": 1703520000.0,
            "completed_nodes": {}
        }"""
        )

        checkpoint = Checkpoint.load(file_path)
        assert checkpoint.graph_hash is None

    def test_round_trip_with_graph_hash(self, tmp_path: Path) -> None:
        """Checkpoint with graph_hash survives round-trip."""
        original = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            graph_hash="0123456789abcdef",
        )
        file_path = tmp_path / "checkpoint.json"
        original.save(file_path)
        loaded = Checkpoint.load(file_path)

        assert loaded.graph_hash == original.graph_hash


class TestCheckpointIsCompatibleWith:
    """Tests for is_compatible_with() method."""

    def test_compatible_when_hashes_match(self) -> None:
        """is_compatible_with() returns True when hashes match."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            graph_hash="matching_hash_123",
        )
        assert checkpoint.is_compatible_with("matching_hash_123") is True

    def test_incompatible_when_hashes_differ(self) -> None:
        """is_compatible_with() returns False when hashes differ."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            graph_hash="original_hash",
        )
        assert checkpoint.is_compatible_with("different_hash") is False

    def test_compatible_when_checkpoint_has_no_hash(self) -> None:
        """is_compatible_with() returns True for legacy checkpoints."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            graph_hash=None,
        )
        # Legacy checkpoints are assumed compatible
        assert checkpoint.is_compatible_with("any_hash") is True

    def test_compatible_with_empty_string_hash(self) -> None:
        """is_compatible_with() handles empty string hashes correctly."""
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            graph_hash="",
        )
        assert checkpoint.is_compatible_with("") is True
        assert checkpoint.is_compatible_with("non_empty") is False

    def test_compatible_with_module_same_structure(self) -> None:
        """is_compatible_with() accepts a module and traces it internally."""
        from plait.module import InferenceModule
        from plait.tracing.tracer import Tracer

        class SimpleModule(InferenceModule):
            def forward(self, x: str) -> str:
                return x

        # First, get the hash for this module
        module = SimpleModule()
        tracer = Tracer()
        graph = tracer.trace(module, "test")
        expected_hash = graph.compute_hash()

        # Create checkpoint with that hash
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            graph_hash=expected_hash,
        )

        # Create a new instance and check compatibility via module
        module2 = SimpleModule()
        assert checkpoint.is_compatible_with(module2, "test") is True

    def test_compatible_with_module_different_structure(self) -> None:
        """is_compatible_with() detects different module structures."""
        from plait.module import InferenceModule
        from plait.tracing.tracer import Tracer

        class ModuleA(InferenceModule):
            def __init__(self):
                super().__init__()
                self.child = ModuleB()

            def forward(self, x: str) -> str:
                return self.child(x)

        class ModuleB(InferenceModule):
            def forward(self, x: str) -> str:
                return x

        class ModuleC(InferenceModule):
            def forward(self, x: str) -> str:
                return x

        # Get hash for ModuleA
        module_a = ModuleA()
        tracer = Tracer()
        graph = tracer.trace(module_a, "test")
        hash_a = graph.compute_hash()

        # Create checkpoint with ModuleA's hash
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            graph_hash=hash_a,
        )

        # ModuleC has different structure
        module_c = ModuleC()
        assert checkpoint.is_compatible_with(module_c, "test") is False

    def test_compatible_with_module_legacy_checkpoint(self) -> None:
        """is_compatible_with() returns True for legacy checkpoints with modules."""
        from plait.module import InferenceModule

        class SimpleModule(InferenceModule):
            def forward(self, x: str) -> str:
                return x

        # Legacy checkpoint without hash
        checkpoint = Checkpoint(
            execution_id="run_001",
            timestamp=1703520000.0,
            graph_hash=None,
        )

        module = SimpleModule()
        assert checkpoint.is_compatible_with(module, "test") is True


class TestCheckpointManagerCreation:
    """Tests for CheckpointManager initialization."""

    def test_creation_with_path(self, tmp_path: Path) -> None:
        """CheckpointManager can be created with a Path."""
        manager = CheckpointManager(tmp_path)
        assert manager.checkpoint_dir == tmp_path
        assert manager.buffer_size == 10  # default
        assert manager.flush_interval == 60.0  # default

    def test_creation_with_string_path(self, tmp_path: Path) -> None:
        """CheckpointManager can be created with a string path."""
        manager = CheckpointManager(str(tmp_path))
        assert manager.checkpoint_dir == tmp_path

    def test_creation_with_custom_buffer_size(self, tmp_path: Path) -> None:
        """CheckpointManager accepts custom buffer_size."""
        manager = CheckpointManager(tmp_path, buffer_size=20)
        assert manager.buffer_size == 20

    def test_creation_with_custom_flush_interval(self, tmp_path: Path) -> None:
        """CheckpointManager accepts custom flush_interval."""
        manager = CheckpointManager(tmp_path, flush_interval=30.0)
        assert manager.flush_interval == 30.0

    def test_creation_creates_directory(self, tmp_path: Path) -> None:
        """CheckpointManager creates the checkpoint directory if needed."""
        new_dir = tmp_path / "subdir" / "checkpoints"
        assert not new_dir.exists()
        CheckpointManager(new_dir)
        assert new_dir.exists()

    def test_creation_raises_for_invalid_buffer_size(self, tmp_path: Path) -> None:
        """CheckpointManager raises ValueError for buffer_size < 1."""
        with pytest.raises(ValueError, match="buffer_size must be at least 1"):
            CheckpointManager(tmp_path, buffer_size=0)

    def test_creation_raises_for_negative_flush_interval(self, tmp_path: Path) -> None:
        """CheckpointManager raises ValueError for negative flush_interval."""
        with pytest.raises(ValueError, match="flush_interval must be non-negative"):
            CheckpointManager(tmp_path, flush_interval=-1.0)


class TestCheckpointManagerRecordCompletion:
    """Tests for CheckpointManager.record_completion() method."""

    def test_record_completion_buffers_result(self, tmp_path: Path) -> None:
        """record_completion() adds result to buffer."""
        manager = CheckpointManager(tmp_path, buffer_size=10)
        result = TaskResult("node_1", "value", 100.0, 0)

        manager.record_completion("run_001", "node_1", result)

        assert "run_001" in manager._buffers
        assert len(manager._buffers["run_001"]) == 1
        assert manager._buffers["run_001"][0] == ("node_1", result)

    def test_record_completion_returns_false_when_buffer_not_full(
        self, tmp_path: Path
    ) -> None:
        """record_completion() returns False when buffer is not full."""
        manager = CheckpointManager(tmp_path, buffer_size=10)
        result = TaskResult("node_1", "value", 100.0, 0)

        should_flush = manager.record_completion("run_001", "node_1", result)

        assert should_flush is False

    def test_record_completion_returns_true_when_buffer_full(
        self, tmp_path: Path
    ) -> None:
        """record_completion() returns True when buffer reaches buffer_size."""
        manager = CheckpointManager(tmp_path, buffer_size=3)

        # Add 2 items - should not trigger flush
        for i in range(2):
            result = TaskResult(f"node_{i}", f"value_{i}", 100.0, 0)
            should_flush = manager.record_completion("run_001", f"node_{i}", result)
            assert should_flush is False

        # Add 3rd item - should trigger flush
        result = TaskResult("node_2", "value_2", 100.0, 0)
        should_flush = manager.record_completion("run_001", "node_2", result)
        assert should_flush is True

    def test_record_completion_tracks_multiple_executions(self, tmp_path: Path) -> None:
        """record_completion() maintains separate buffers per execution."""
        manager = CheckpointManager(tmp_path, buffer_size=10)
        result1 = TaskResult("node_1", "value_1", 100.0, 0)
        result2 = TaskResult("node_2", "value_2", 100.0, 0)

        manager.record_completion("run_001", "node_1", result1)
        manager.record_completion("run_002", "node_2", result2)

        assert len(manager._buffers["run_001"]) == 1
        assert len(manager._buffers["run_002"]) == 1


class TestCheckpointManagerFlush:
    """Tests for CheckpointManager.flush() method."""

    @pytest.mark.asyncio
    async def test_flush_creates_checkpoint_file(self, tmp_path: Path) -> None:
        """flush() creates a checkpoint file on disk."""
        manager = CheckpointManager(tmp_path)
        result = TaskResult("node_1", "value", 100.0, 0)
        manager.record_completion("run_001", "node_1", result)

        await manager.flush("run_001")

        checkpoint_path = tmp_path / "run_001.json"
        assert checkpoint_path.exists()

    @pytest.mark.asyncio
    async def test_flush_writes_buffered_completions(self, tmp_path: Path) -> None:
        """flush() writes all buffered completions to the checkpoint."""
        manager = CheckpointManager(tmp_path, buffer_size=10)

        # Add multiple completions
        for i in range(3):
            result = TaskResult(f"node_{i}", f"value_{i}", 100.0 * i, 0)
            manager.record_completion("run_001", f"node_{i}", result)

        await manager.flush("run_001")

        # Load and verify
        checkpoint = Checkpoint.load(tmp_path / "run_001.json")
        assert len(checkpoint.completed_nodes) == 3
        assert checkpoint.completed_nodes["node_0"].value == "value_0"
        assert checkpoint.completed_nodes["node_1"].value == "value_1"
        assert checkpoint.completed_nodes["node_2"].value == "value_2"

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(self, tmp_path: Path) -> None:
        """flush() clears the buffer after writing."""
        manager = CheckpointManager(tmp_path)
        result = TaskResult("node_1", "value", 100.0, 0)
        manager.record_completion("run_001", "node_1", result)

        await manager.flush("run_001")

        assert manager._buffers["run_001"] == []

    @pytest.mark.asyncio
    async def test_flush_merges_with_existing_checkpoint(self, tmp_path: Path) -> None:
        """flush() merges new completions with existing checkpoint."""
        manager = CheckpointManager(tmp_path)

        # First flush
        result1 = TaskResult("node_1", "value_1", 100.0, 0)
        manager.record_completion("run_001", "node_1", result1)
        await manager.flush("run_001")

        # Second flush
        result2 = TaskResult("node_2", "value_2", 200.0, 0)
        manager.record_completion("run_001", "node_2", result2)
        await manager.flush("run_001")

        # Load and verify both are present
        checkpoint = Checkpoint.load(tmp_path / "run_001.json")
        assert len(checkpoint.completed_nodes) == 2
        assert "node_1" in checkpoint.completed_nodes
        assert "node_2" in checkpoint.completed_nodes

    @pytest.mark.asyncio
    async def test_flush_does_nothing_for_empty_buffer(self, tmp_path: Path) -> None:
        """flush() does nothing if buffer is empty."""
        manager = CheckpointManager(tmp_path)

        # Flush without any completions
        await manager.flush("run_001")

        # No file should be created
        checkpoint_path = tmp_path / "run_001.json"
        assert not checkpoint_path.exists()

    @pytest.mark.asyncio
    async def test_flush_includes_graph_hash(self, tmp_path: Path) -> None:
        """flush() includes graph_hash in the checkpoint."""
        manager = CheckpointManager(tmp_path)
        manager.set_graph_hash("run_001", "abc123")

        result = TaskResult("node_1", "value", 100.0, 0)
        manager.record_completion("run_001", "node_1", result)
        await manager.flush("run_001")

        checkpoint = Checkpoint.load(tmp_path / "run_001.json")
        assert checkpoint.graph_hash == "abc123"


class TestCheckpointManagerFlushAll:
    """Tests for CheckpointManager.flush_all() method."""

    @pytest.mark.asyncio
    async def test_flush_all_flushes_all_executions(self, tmp_path: Path) -> None:
        """flush_all() flushes buffers for all tracked executions."""
        manager = CheckpointManager(tmp_path)

        # Add completions to multiple executions
        manager.record_completion(
            "run_001", "node_1", TaskResult("node_1", "v1", 100.0, 0)
        )
        manager.record_completion(
            "run_002", "node_2", TaskResult("node_2", "v2", 200.0, 0)
        )
        manager.record_completion(
            "run_003", "node_3", TaskResult("node_3", "v3", 300.0, 0)
        )

        await manager.flush_all()

        # All checkpoint files should exist
        assert (tmp_path / "run_001.json").exists()
        assert (tmp_path / "run_002.json").exists()
        assert (tmp_path / "run_003.json").exists()

    @pytest.mark.asyncio
    async def test_flush_all_clears_all_buffers(self, tmp_path: Path) -> None:
        """flush_all() clears all buffers after flushing."""
        manager = CheckpointManager(tmp_path)

        manager.record_completion(
            "run_001", "node_1", TaskResult("node_1", "v1", 100.0, 0)
        )
        manager.record_completion(
            "run_002", "node_2", TaskResult("node_2", "v2", 200.0, 0)
        )

        await manager.flush_all()

        assert manager._buffers["run_001"] == []
        assert manager._buffers["run_002"] == []


class TestCheckpointManagerGetCheckpoint:
    """Tests for CheckpointManager.get_checkpoint() method."""

    @pytest.mark.asyncio
    async def test_get_checkpoint_returns_existing(self, tmp_path: Path) -> None:
        """get_checkpoint() returns an existing checkpoint."""
        manager = CheckpointManager(tmp_path)
        result = TaskResult("node_1", "value", 100.0, 0)
        manager.record_completion("run_001", "node_1", result)
        await manager.flush("run_001")

        checkpoint = manager.get_checkpoint("run_001")

        assert checkpoint is not None
        assert checkpoint.execution_id == "run_001"
        assert "node_1" in checkpoint.completed_nodes

    def test_get_checkpoint_returns_none_for_nonexistent(self, tmp_path: Path) -> None:
        """get_checkpoint() returns None for nonexistent checkpoint."""
        manager = CheckpointManager(tmp_path)

        checkpoint = manager.get_checkpoint("nonexistent")

        assert checkpoint is None


class TestCheckpointManagerSetGraphHash:
    """Tests for CheckpointManager.set_graph_hash() method."""

    def test_set_graph_hash_stores_hash(self, tmp_path: Path) -> None:
        """set_graph_hash() stores the hash for an execution."""
        manager = CheckpointManager(tmp_path)

        manager.set_graph_hash("run_001", "abc123")

        assert manager._graph_hashes["run_001"] == "abc123"

    def test_set_graph_hash_allows_none(self, tmp_path: Path) -> None:
        """set_graph_hash() allows None as a value."""
        manager = CheckpointManager(tmp_path)

        manager.set_graph_hash("run_001", None)

        assert manager._graph_hashes["run_001"] is None


class TestCheckpointManagerConcurrency:
    """Tests for CheckpointManager thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_flushes_are_safe(self, tmp_path: Path) -> None:
        """Concurrent flush operations don't corrupt data."""
        manager = CheckpointManager(tmp_path, buffer_size=100)

        # Add many completions
        for i in range(50):
            result = TaskResult(f"node_{i}", f"value_{i}", float(i), 0)
            manager.record_completion("run_001", f"node_{i}", result)

        # Flush concurrently multiple times
        await asyncio.gather(
            manager.flush("run_001"),
            manager.flush("run_001"),
            manager.flush("run_001"),
        )

        # Should have all completions
        checkpoint = Checkpoint.load(tmp_path / "run_001.json")
        assert len(checkpoint.completed_nodes) == 50


class TestCheckpointManagerFlushInterval:
    """Tests for CheckpointManager flush interval behavior."""

    def test_record_completion_respects_flush_interval(self, tmp_path: Path) -> None:
        """record_completion() returns True when flush_interval has elapsed."""
        manager = CheckpointManager(tmp_path, buffer_size=100, flush_interval=0.0)

        result = TaskResult("node_1", "value", 100.0, 0)

        # First completion initializes last_flush
        should_flush = manager.record_completion("run_001", "node_1", result)

        # With flush_interval=0, should always suggest flush (after first)
        result2 = TaskResult("node_2", "value", 100.0, 0)
        should_flush = manager.record_completion("run_001", "node_2", result2)
        assert should_flush is True
