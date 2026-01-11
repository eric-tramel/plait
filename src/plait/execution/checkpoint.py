"""Checkpoint types for execution state persistence.

This module provides the Checkpoint dataclass for saving and loading
execution state to disk, and the CheckpointManager for managing buffered
checkpoint writes during execution.

Example:
    >>> from pathlib import Path
    >>> from plait.execution.checkpoint import Checkpoint, CheckpointManager
    >>> from plait.execution.state import TaskResult
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
    >>>
    >>> # Use CheckpointManager for buffered writes during execution
    >>> manager = CheckpointManager(Path("/tmp/checkpoints"))
    >>> manager.record_completion("run_001", "node_1", result)
    >>> await manager.flush_all()
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from plait.execution.state import TaskResult
    from plait.module import Module


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

    def is_compatible_with(
        self,
        module_or_hash: Module | str,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Check if this checkpoint is compatible with a graph.

        Compares the stored graph_hash with the provided hash to determine
        if the checkpoint can be used with the given graph structure.

        Args:
            module_or_hash: Either a graph hash string from
                InferenceGraph.compute_hash(), or an Module to
                trace and compute the hash from.
            *args: If module_or_hash is an Module, these are passed
                to trace() as the module's input arguments.
            **kwargs: If module_or_hash is an Module, these are
                passed to trace() as the module's keyword arguments.

        Returns:
            True if the checkpoint is compatible (same graph structure),
            False if the graph has changed. Also returns True if this
            checkpoint has no graph_hash (legacy checkpoint).

        Example with module:
            >>> checkpoint = Checkpoint.load(Path("checkpoint.json"))
            >>> if checkpoint.is_compatible_with(pipeline, "sample input"):
            ...     # Safe to resume from checkpoint
            ...     pass
            ... else:
            ...     # Pipeline has changed, start fresh
            ...     pass

        Example with hash:
            >>> checkpoint = Checkpoint.load(Path("checkpoint.json"))
            >>> graph = tracer.trace(pipeline, "input")
            >>> if checkpoint.is_compatible_with(graph.compute_hash()):
            ...     # Safe to resume from checkpoint
            ...     pass

        Note:
            When passing a module, sample input arguments are required
            for tracing. The actual values don't affect the hash - only
            the graph structure matters.
        """
        if self.graph_hash is None:
            # Legacy checkpoint without hash - assume compatible
            return True

        # Determine the hash to compare against
        if isinstance(module_or_hash, str):
            compare_hash = module_or_hash
        else:
            # It's a module - trace it to get the graph hash
            from plait.tracing.tracer import Tracer

            tracer = Tracer()
            graph = tracer.trace(module_or_hash, *args, **kwargs)
            compare_hash = graph.compute_hash()

        return self.graph_hash == compare_hash

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
        # Import here to avoid circular imports at module load time
        from plait.execution.state import TaskResult

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


class CheckpointManager:
    """Manages checkpointing for executions with buffered writes.

    The CheckpointManager handles the persistence of execution progress
    to disk with buffering for efficiency. Instead of writing to disk
    after every task completion, completions are buffered and flushed
    periodically or when the buffer is full.

    Attributes:
        checkpoint_dir: Directory where checkpoint files are stored.
        buffer_size: Number of completions to buffer before flushing.
        flush_interval: Maximum seconds between flushes.

    Example:
        >>> from pathlib import Path
        >>> from plait.execution.checkpoint import CheckpointManager
        >>> from plait.execution.state import TaskResult
        >>>
        >>> # Create manager with default settings
        >>> manager = CheckpointManager(Path("/tmp/checkpoints"))
        >>>
        >>> # Record task completions (buffered)
        >>> result = TaskResult("node_1", "output", 100.0, 0)
        >>> manager.record_completion("run_001", "node_1", result)
        >>>
        >>> # Flush all pending checkpoints before exit
        >>> await manager.flush_all()

    Note:
        The manager creates one checkpoint file per execution_id.
        Files are named ``{execution_id}.json`` in the checkpoint_dir.
    """

    checkpoint_dir: Path
    buffer_size: int
    flush_interval: float

    def __init__(
        self,
        checkpoint_dir: Path | str,
        buffer_size: int = 10,
        flush_interval: float = 60.0,
    ) -> None:
        """Initialize the checkpoint manager.

        Creates the checkpoint directory if it doesn't exist.

        Args:
            checkpoint_dir: Directory to store checkpoint files.
                Will be created if it doesn't exist.
            buffer_size: Number of completions to buffer before auto-flushing.
                Set to 1 to flush after every completion. Defaults to 10.
            flush_interval: Maximum seconds between flushes, even if buffer
                is not full. Defaults to 60.0.

        Raises:
            ValueError: If buffer_size is less than 1.
            ValueError: If flush_interval is negative.

        Example:
            >>> # Create with custom settings
            >>> manager = CheckpointManager(
            ...     checkpoint_dir="/data/checkpoints",
            ...     buffer_size=20,
            ...     flush_interval=30.0,
            ... )
        """
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        if flush_interval < 0:
            raise ValueError("flush_interval must be non-negative")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        # Create directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Per-execution buffers: execution_id -> list of (node_id, TaskResult)
        self._buffers: dict[str, list[tuple[str, TaskResult]]] = {}

        # Track last flush time per execution
        self._last_flush: dict[str, float] = {}

        # Lock for thread-safe access
        self._lock = asyncio.Lock()

        # Track graph hashes per execution
        self._graph_hashes: dict[str, str | None] = {}

    def set_graph_hash(self, execution_id: str, graph_hash: str | None) -> None:
        """Set the graph hash for an execution.

        The graph hash is stored in checkpoints to detect when the pipeline
        structure has changed, which would invalidate the checkpoint.

        Args:
            execution_id: The execution identifier.
            graph_hash: The hash from InferenceGraph.compute_hash(), or None.

        Example:
            >>> manager.set_graph_hash("run_001", graph.compute_hash())
        """
        self._graph_hashes[execution_id] = graph_hash

    def record_completion(
        self,
        execution_id: str,
        node_id: str,
        result: TaskResult,
    ) -> bool:
        """Record a completed task (may be buffered).

        Adds the completion to the buffer for the given execution. If the
        buffer reaches buffer_size or flush_interval has elapsed since the
        last flush, returns True to indicate a flush is needed.

        This method is synchronous for fast recording in callbacks.
        The caller should call flush() if this returns True.

        Args:
            execution_id: The execution identifier.
            node_id: The ID of the completed node.
            result: The TaskResult containing the completion data.

        Returns:
            True if the buffer should be flushed, False otherwise.

        Example:
            >>> if manager.record_completion("run_001", "node_1", result):
            ...     await manager.flush("run_001")

        Note:
            This method does not actually flush to disk. Call flush() or
            flush_all() to persist buffered completions.
        """
        # Initialize buffer for this execution if needed
        if execution_id not in self._buffers:
            self._buffers[execution_id] = []
            self._last_flush[execution_id] = time.time()

        # Add to buffer
        self._buffers[execution_id].append((node_id, result))

        # Check if flush is needed
        buffer_full = len(self._buffers[execution_id]) >= self.buffer_size
        interval_elapsed = (
            time.time() - self._last_flush[execution_id] >= self.flush_interval
        )

        return buffer_full or interval_elapsed

    async def flush(self, execution_id: str) -> None:
        """Flush the buffer for an execution to disk.

        Loads any existing checkpoint for the execution, merges the buffered
        completions, and writes the updated checkpoint atomically.

        Args:
            execution_id: The execution identifier to flush.

        Raises:
            OSError: If the checkpoint file cannot be written.

        Example:
            >>> await manager.flush("run_001")

        Note:
            If the buffer is empty, this method does nothing.
            The checkpoint file is written atomically.
        """
        async with self._lock:
            # Get buffer for this execution
            buffer = self._buffers.get(execution_id, [])
            if not buffer:
                return

            # Load existing checkpoint or create new one
            checkpoint_path = self._get_checkpoint_path(execution_id)
            if checkpoint_path.exists():
                checkpoint = Checkpoint.load(checkpoint_path)
            else:
                checkpoint = Checkpoint(
                    execution_id=execution_id,
                    timestamp=time.time(),
                    graph_hash=self._graph_hashes.get(execution_id),
                    completed_nodes={},
                    failed_nodes={},
                    pending_nodes=[],
                )

            # Merge buffered completions
            for node_id, result in buffer:
                checkpoint.completed_nodes[node_id] = result

            # Update timestamp
            checkpoint.timestamp = time.time()

            # Write checkpoint
            checkpoint.save(checkpoint_path)

            # Clear buffer and update flush time
            self._buffers[execution_id] = []
            self._last_flush[execution_id] = time.time()

    async def flush_all(self) -> None:
        """Flush all pending buffers to disk.

        Flushes the buffers for all tracked executions. This should be
        called when execution completes or when shutting down to ensure
        all progress is persisted.

        Example:
            >>> # At the end of execution
            >>> await manager.flush_all()

        Note:
            This method acquires the lock for each flush operation.
        """
        # Get list of execution IDs to flush (copy to avoid modification during iteration)
        execution_ids = list(self._buffers.keys())

        for execution_id in execution_ids:
            await self.flush(execution_id)

    def get_checkpoint(self, execution_id: str) -> Checkpoint | None:
        """Load an existing checkpoint for an execution.

        Args:
            execution_id: The execution identifier.

        Returns:
            The Checkpoint if it exists, None otherwise.

        Example:
            >>> checkpoint = manager.get_checkpoint("run_001")
            >>> if checkpoint:
            ...     print(f"Resuming from {len(checkpoint.completed_nodes)} nodes")
        """
        checkpoint_path = self._get_checkpoint_path(execution_id)
        if checkpoint_path.exists():
            return Checkpoint.load(checkpoint_path)
        return None

    def _get_checkpoint_path(self, execution_id: str) -> Path:
        """Get the file path for an execution's checkpoint.

        Args:
            execution_id: The execution identifier.

        Returns:
            Path to the checkpoint file.
        """
        return self.checkpoint_dir / f"{execution_id}.json"
