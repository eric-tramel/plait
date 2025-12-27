"""Tests for Checkpoint dataclass.

Tests the save/load functionality and round-trip serialization
for execution state persistence.
"""

from pathlib import Path

import pytest

from inf_engine.execution.checkpoint import Checkpoint
from inf_engine.execution.state import TaskResult


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
