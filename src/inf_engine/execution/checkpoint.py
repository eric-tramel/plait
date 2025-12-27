"""Checkpoint types for execution state persistence.

This module provides the Checkpoint dataclass for saving and loading
execution state to disk, enabling progress recovery for long-running pipelines.

Example:
    >>> from pathlib import Path
    >>> from inf_engine.execution.checkpoint import Checkpoint
    >>> from inf_engine.execution.state import TaskResult
    >>>
    >>> # Create a checkpoint with completed work
    >>> checkpoint = Checkpoint(
    ...     execution_id="run_001",
    ...     timestamp=1703520000.0,
    ...     completed_nodes={
    ...         "node_1": TaskResult(
    ...             node_id="node_1",
    ...             value="result text",
    ...             duration_ms=150.5,
    ...             retry_count=0,
    ...         ),
    ...     },
    ...     failed_nodes={},
    ...     pending_nodes=["node_2", "node_3"],
    ... )
    >>>
    >>> # Save to disk
    >>> checkpoint.save(Path("/tmp/checkpoint.json"))
    >>>
    >>> # Load from disk
    >>> loaded = Checkpoint.load(Path("/tmp/checkpoint.json"))
    >>> loaded.execution_id
    'run_001'
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inf_engine.execution.state import TaskResult


@dataclass
class Checkpoint:
    """A saved execution checkpoint for progress persistence and recovery.

    Captures the state of a graph execution at a point in time, including
    which nodes have completed, failed, or are still pending. Checkpoints
    can be saved to disk and loaded later to resume execution or analyze
    execution history.

    Attributes:
        execution_id: Unique identifier for the execution run.
        timestamp: Unix timestamp when the checkpoint was created.
        graph_hash: Deterministic hash of the graph structure. Used to detect
            incompatibility when loading a checkpoint for a modified pipeline.
            None for legacy checkpoints created before this field was added.
        completed_nodes: Dictionary mapping node IDs to their TaskResult.
        failed_nodes: Dictionary mapping failed node IDs to error messages.
        pending_nodes: List of node IDs that were pending when checkpointed.

    Example:
        >>> checkpoint = Checkpoint(
        ...     execution_id="run_001",
        ...     timestamp=1703520000.0,
        ...     completed_nodes={
        ...         "LLMInference_1": TaskResult(
        ...             node_id="LLMInference_1",
        ...             value="Generated text",
        ...             duration_ms=250.0,
        ...             retry_count=0,
        ...         ),
        ...     },
        ...     failed_nodes={
        ...         "LLMInference_2": "API timeout after 3 retries",
        ...     },
        ...     pending_nodes=["LLMInference_3", "LLMInference_4"],
        ... )

    Note:
        The checkpoint captures a snapshot of execution state. For live
        executions, use CheckpointManager which handles buffered writes
        and periodic flushing.
    """

    execution_id: str
    timestamp: float
    graph_hash: str | None = None
    completed_nodes: dict[str, TaskResult] = field(default_factory=dict)
    failed_nodes: dict[str, str] = field(default_factory=dict)
    pending_nodes: list[str] = field(default_factory=list)

    def is_compatible_with(self, graph_hash: str) -> bool:
        """Check if this checkpoint is compatible with a graph.

        Compares the stored graph_hash with the provided hash to determine
        if the checkpoint can be used with the given graph structure.

        Args:
            graph_hash: The hash from InferenceGraph.compute_hash().

        Returns:
            True if the checkpoint is compatible (same graph structure),
            False if the graph has changed. Also returns True if this
            checkpoint has no graph_hash (legacy checkpoint).

        Example:
            >>> checkpoint = Checkpoint.load(Path("checkpoint.json"))
            >>> graph = tracer.trace(pipeline, "input")
            >>> if checkpoint.is_compatible_with(graph.compute_hash()):
            ...     # Safe to resume from checkpoint
            ...     pass
            ... else:
            ...     # Pipeline has changed, start fresh
            ...     pass
        """
        if self.graph_hash is None:
            # Legacy checkpoint without hash - assume compatible
            return True
        return self.graph_hash == graph_hash

    def save(self, path: Path) -> None:
        """Save the checkpoint to disk as JSON.

        Serializes the checkpoint to a JSON file that can be loaded later.
        The file is written atomically (content is fully written before
        the file is considered complete).

        Args:
            path: The file path to save the checkpoint to. Parent directories
                must exist.

        Raises:
            OSError: If the file cannot be written.
            TypeError: If any values are not JSON serializable.

        Example:
            >>> checkpoint.save(Path("./checkpoints/run_001.json"))

        Note:
            The JSON format uses indentation for human readability.
            TaskResult values must be JSON serializable for this to work.
            Complex objects in TaskResult.value may fail serialization.
        """
        data = self._to_dict()
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """Load a checkpoint from disk.

        Reads a JSON checkpoint file and reconstructs the Checkpoint object,
        including all TaskResult objects.

        Args:
            path: The file path to load the checkpoint from.

        Returns:
            A Checkpoint instance with all fields restored.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If required fields are missing from the JSON.

        Example:
            >>> checkpoint = Checkpoint.load(Path("./checkpoints/run_001.json"))
            >>> checkpoint.execution_id
            'run_001'
            >>> len(checkpoint.completed_nodes)
            15
        """
        data = json.loads(path.read_text())
        return cls._from_dict(data)

    def _to_dict(self) -> dict[str, Any]:
        """Convert the checkpoint to a JSON-serializable dictionary.

        Returns:
            A dictionary containing all checkpoint data in a format
            suitable for JSON serialization.
        """
        data: dict[str, Any] = {
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "completed_nodes": {
                node_id: {
                    "node_id": result.node_id,
                    "value": result.value,
                    "duration_ms": result.duration_ms,
                    "retry_count": result.retry_count,
                }
                for node_id, result in self.completed_nodes.items()
            },
            "failed_nodes": self.failed_nodes,
            "pending_nodes": self.pending_nodes,
        }
        if self.graph_hash is not None:
            data["graph_hash"] = self.graph_hash
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Create a Checkpoint from a dictionary.

        Args:
            data: Dictionary containing checkpoint data, typically from
                JSON deserialization.

        Returns:
            A Checkpoint instance with fields populated from the dictionary.
        """
        return cls(
            execution_id=data["execution_id"],
            timestamp=data["timestamp"],
            graph_hash=data.get("graph_hash"),
            completed_nodes={
                node_id: TaskResult(
                    node_id=node_id,
                    value=result["value"],
                    duration_ms=result["duration_ms"],
                    retry_count=result.get("retry_count", 0),
                )
                for node_id, result in data["completed_nodes"].items()
            },
            failed_nodes=data.get("failed_nodes", {}),
            pending_nodes=data.get("pending_nodes", []),
        )
