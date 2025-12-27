#!/usr/bin/env python3
"""Example 06: Checkpointing for Long-Running Pipelines.

This example demonstrates how to use checkpointing to track progress
during execution of inference pipelines. Checkpoints persist task
completions to disk, enabling:

1. Progress monitoring during long-running batch jobs
2. Failure analysis after execution
3. Future resumption capabilities (when implemented)

Key Concepts:
- CheckpointManager: Manages buffered checkpoint writes
- checkpoint_dir: Directory for storing checkpoint files
- execution_id: Unique identifier for each execution run
- Graph hashing: Detects pipeline structure changes

Run with: python examples/06_checkpointing.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from tempfile import mkdtemp

from inf_engine.execution.checkpoint import Checkpoint, CheckpointManager
from inf_engine.execution.executor import run
from inf_engine.execution.state import TaskResult
from inf_engine.module import InferenceModule

# ─────────────────────────────────────────────────────────────────────────────
# Example Modules
# ─────────────────────────────────────────────────────────────────────────────


class SlowProcessor(InferenceModule):
    """A module that simulates slow processing."""

    def __init__(self, name: str, delay: float = 0.1):
        super().__init__()
        self.name = name
        self.delay = delay

    def forward(self, text: str) -> str:
        """Process text with a small delay."""
        time.sleep(self.delay)
        return f"[{self.name}] {text}"


class DataPipeline(InferenceModule):
    """A pipeline that processes data through multiple stages."""

    def __init__(self):
        super().__init__()
        self.clean = SlowProcessor("clean", 0.05)
        self.enrich = SlowProcessor("enrich", 0.05)
        self.validate = SlowProcessor("validate", 0.05)
        self.format = SlowProcessor("format", 0.05)

    def forward(self, data: str) -> str:
        """Process data through cleaning, enrichment, validation, and formatting."""
        cleaned = self.clean(data)
        enriched = self.enrich(cleaned)
        validated = self.validate(enriched)
        formatted = self.format(validated)
        return formatted


class BatchAnalyzer(InferenceModule):
    """A pipeline that analyzes data in parallel branches."""

    def __init__(self):
        super().__init__()
        self.summarize = SlowProcessor("summarize", 0.05)
        self.extract = SlowProcessor("extract", 0.05)
        self.classify = SlowProcessor("classify", 0.05)

    def forward(self, text: str) -> dict[str, str]:
        """Analyze text through parallel processing branches."""
        return {
            "summary": self.summarize(text),
            "entities": self.extract(text),
            "category": self.classify(text),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Basic Checkpointing with run()
# ─────────────────────────────────────────────────────────────────────────────


async def example_basic_checkpointing() -> None:
    """Demonstrate basic checkpointing with the run() function.

    The run() function accepts a checkpoint_dir parameter that enables
    automatic checkpointing. Each task completion is recorded and
    periodically flushed to disk.
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic Checkpointing with run()")
    print("=" * 70)

    # Create a temporary directory for checkpoints
    checkpoint_dir = Path(mkdtemp(prefix="inf_engine_checkpoints_"))

    try:
        pipeline = DataPipeline()

        print(f"\nCheckpoint directory: {checkpoint_dir}")
        print("Running pipeline with checkpointing enabled...")

        # Run with checkpointing
        result = await run(
            pipeline,
            "raw input data",
            checkpoint_dir=checkpoint_dir,
            execution_id="data_pipeline_001",
        )

        print(f"\nResult: {result}")

        # Load and inspect the checkpoint
        checkpoint_path = checkpoint_dir / "data_pipeline_001.json"
        print(f"\nCheckpoint saved to: {checkpoint_path}")

        checkpoint = Checkpoint.load(checkpoint_path)
        print("\nCheckpoint contents:")
        print(f"  - Execution ID: {checkpoint.execution_id}")
        print(f"  - Timestamp: {checkpoint.timestamp}")
        hash_display = checkpoint.graph_hash[:16] if checkpoint.graph_hash else "N/A"
        print(f"  - Graph hash: {hash_display}...")
        print(f"  - Completed nodes: {len(checkpoint.completed_nodes)}")

        print("\nCompleted tasks:")
        for node_id, task_result in checkpoint.completed_nodes.items():
            print(f"  - {node_id}: {task_result.duration_ms:.2f}ms")

    finally:
        # Clean up
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Using CheckpointManager Directly
# ─────────────────────────────────────────────────────────────────────────────


async def example_checkpoint_manager() -> None:
    """Demonstrate direct use of CheckpointManager.

    For more control over checkpointing, you can use CheckpointManager
    directly to record completions and manage buffered writes.
    """
    print("\n" + "=" * 70)
    print("Example 2: Using CheckpointManager Directly")
    print("=" * 70)

    checkpoint_dir = Path(mkdtemp(prefix="inf_engine_manager_"))

    try:
        # Create a checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            buffer_size=3,  # Flush after every 3 completions
            flush_interval=60.0,  # Or after 60 seconds
        )

        print(f"\nCheckpoint directory: {checkpoint_dir}")
        print(f"Buffer size: {manager.buffer_size}")
        print(f"Flush interval: {manager.flush_interval}s")

        # Set graph hash for this execution
        manager.set_graph_hash("manual_run", "example_hash_12345")

        # Simulate recording task completions
        print("\nRecording task completions...")
        tasks = [
            ("task_1", "result_1", 50.0),
            ("task_2", "result_2", 75.0),
            ("task_3", "result_3", 100.0),  # Buffer full after this
            ("task_4", "result_4", 125.0),
            ("task_5", "result_5", 150.0),
        ]

        for node_id, value, duration in tasks:
            result = TaskResult(
                node_id=node_id,
                value=value,
                duration_ms=duration,
                retry_count=0,
            )
            should_flush = manager.record_completion("manual_run", node_id, result)
            print(f"  Recorded {node_id} - should_flush: {should_flush}")

            if should_flush:
                await manager.flush("manual_run")
                print("  Flushed to disk!")

        # Ensure all remaining are flushed
        await manager.flush_all()
        print("\nFinal flush complete.")

        # Load and inspect
        checkpoint = manager.get_checkpoint("manual_run")
        if checkpoint:
            print(f"\nCheckpoint has {len(checkpoint.completed_nodes)} completed nodes")

    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Checkpoint Compatibility Checking
# ─────────────────────────────────────────────────────────────────────────────


async def example_checkpoint_compatibility() -> None:
    """Demonstrate checkpoint compatibility checking with graph hashes.

    Checkpoints store a hash of the graph structure. This allows
    detection of when a pipeline has changed, which would make
    a checkpoint invalid for resumption.

    The is_compatible_with() method can accept either:
    - A module (with sample inputs) - traces internally to compute hash
    - A pre-computed hash string
    """
    print("\n" + "=" * 70)
    print("Example 3: Checkpoint Compatibility Checking")
    print("=" * 70)

    checkpoint_dir = Path(mkdtemp(prefix="inf_engine_compat_"))

    try:
        # Run the original pipeline
        pipeline_v1 = DataPipeline()

        print("\nRunning original pipeline...")
        await run(
            pipeline_v1,
            "test data",
            checkpoint_dir=checkpoint_dir,
            execution_id="versioned_run",
        )

        # Load the checkpoint
        checkpoint = Checkpoint.load(checkpoint_dir / "versioned_run.json")
        hash_display = checkpoint.graph_hash[:16] if checkpoint.graph_hash else "N/A"
        print(f"Checkpoint graph hash: {hash_display}...")

        # Check compatibility by passing a module directly
        # The method traces the module internally to compute the hash
        pipeline_v2 = DataPipeline()
        is_compatible = checkpoint.is_compatible_with(pipeline_v2, "test data")
        print(f"\nSame pipeline type compatible: {is_compatible}")

        if is_compatible:
            print("  -> Same structure, checkpoint could be used for resumption")
        else:
            print("  -> Structure changed, checkpoint is invalid")

        # Demonstrate with a different pipeline type
        different_pipeline = BatchAnalyzer()
        is_compatible_different = checkpoint.is_compatible_with(
            different_pipeline, "test data"
        )
        print(f"\nDifferent pipeline compatible: {is_compatible_different}")

        if not is_compatible_different:
            print("  -> Different structure detected, checkpoint is invalid")

    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Example 4: Inspecting Checkpoint Files
# ─────────────────────────────────────────────────────────────────────────────


async def example_inspect_checkpoint() -> None:
    """Demonstrate inspecting checkpoint files.

    Checkpoints are stored as JSON files, making them easy to
    inspect manually or with external tools.
    """
    print("\n" + "=" * 70)
    print("Example 4: Inspecting Checkpoint Files")
    print("=" * 70)

    checkpoint_dir = Path(mkdtemp(prefix="inf_engine_inspect_"))

    try:
        # Run a pipeline with parallel branches
        pipeline = BatchAnalyzer()

        await run(
            pipeline,
            "Important document text",
            checkpoint_dir=checkpoint_dir,
            execution_id="batch_analysis",
        )

        # Read the raw JSON
        checkpoint_path = checkpoint_dir / "batch_analysis.json"
        print(f"\nCheckpoint file: {checkpoint_path}")
        print("\nRaw JSON content:")
        print("-" * 40)

        with open(checkpoint_path) as f:
            data = json.load(f)
            print(json.dumps(data, indent=2))

        print("-" * 40)

        # Load as Checkpoint object
        checkpoint = Checkpoint.load(checkpoint_path)

        print("\nParsed checkpoint analysis:")
        print(f"  Total completed: {len(checkpoint.completed_nodes)}")
        print(f"  Total failed: {len(checkpoint.failed_nodes)}")
        print(f"  Total pending: {len(checkpoint.pending_nodes)}")

        # Calculate total execution time
        total_time = sum(r.duration_ms for r in checkpoint.completed_nodes.values())
        print(f"  Total execution time: {total_time:.2f}ms")

    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Example 5: Auto-Generated Execution IDs
# ─────────────────────────────────────────────────────────────────────────────


async def example_auto_execution_id() -> None:
    """Demonstrate auto-generated execution IDs.

    When execution_id is not provided, run() generates a UUID
    automatically. This is useful for batch processing where
    each run needs a unique identifier.
    """
    print("\n" + "=" * 70)
    print("Example 5: Auto-Generated Execution IDs")
    print("=" * 70)

    checkpoint_dir = Path(mkdtemp(prefix="inf_engine_auto_id_"))

    try:
        pipeline = DataPipeline()

        print(f"\nCheckpoint directory: {checkpoint_dir}")
        print("Running pipeline without explicit execution_id...")

        # Run without execution_id
        await run(
            pipeline,
            "test input",
            checkpoint_dir=checkpoint_dir,
        )

        # Find the generated checkpoint file
        checkpoint_files = list(checkpoint_dir.glob("*.json"))
        print("\nGenerated checkpoint file(s):")
        for f in checkpoint_files:
            print(f"  - {f.name}")

        # Load and show the auto-generated ID
        if checkpoint_files:
            checkpoint = Checkpoint.load(checkpoint_files[0])
            print(f"\nAuto-generated execution_id: {checkpoint.execution_id}")

    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


async def main() -> None:
    """Run all checkpointing examples."""
    print("=" * 70)
    print("INF-ENGINE: Checkpointing Examples")
    print("=" * 70)
    print("""
These examples demonstrate checkpointing for tracking execution
progress in long-running inference pipelines.
""")

    await example_basic_checkpointing()
    await example_checkpoint_manager()
    await example_checkpoint_compatibility()
    await example_inspect_checkpoint()
    await example_auto_execution_id()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
