"""Graph data structures for representing traced inference pipelines.

This module provides the core data structures for representing execution graphs
captured during tracing. The graph structure enables automatic parallelization,
dependency tracking, and optimization of inference pipelines.

Example:
    >>> from inf_engine.graph import GraphNode, InferenceGraph
    >>> from inf_engine.module import LLMInference
    >>>
    >>> # Create nodes representing operations
    >>> input_node = GraphNode(
    ...     id="input:text",
    ...     module=None,  # Input nodes have no module
    ...     args=(),
    ...     kwargs={},
    ...     dependencies=[],
    ...     module_name="Input(text)",
    ... )
    >>> llm_node = GraphNode(
    ...     id="LLMInference_1",
    ...     module=LLMInference(alias="fast"),
    ...     args=("input:text",),
    ...     kwargs={},
    ...     dependencies=["input:text"],
    ... )
    >>>
    >>> # Create the graph
    >>> graph = InferenceGraph(
    ...     nodes={"input:text": input_node, "LLMInference_1": llm_node},
    ...     input_ids=["input:text"],
    ...     output_ids=["LLMInference_1"],
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inf_engine.module import InferenceModule
    from inf_engine.parameter import Parameter


@dataclass
class GraphNode:
    """A single operation in the execution graph.

    Represents one module invocation captured during tracing. Contains
    information about the operation, its dependencies, and metadata
    for scheduling and debugging.

    Attributes:
        id: Unique identifier for this node within the graph.
        module: The InferenceModule instance to execute. May be None for
            special nodes like inputs.
        args: Positional arguments as a tuple of node IDs (for Proxy args)
            or literal values.
        kwargs: Keyword arguments as a dict of node IDs (for Proxy kwargs)
            or literal values.
        dependencies: List of node IDs this node depends on. The node
            cannot execute until all dependencies have completed.
        priority: Execution priority for scheduling. Higher values indicate
            higher priority. Defaults to 0.
        branch_condition: Node ID of the condition proxy for conditional
            execution. None if this node is unconditional.
        branch_value: The branch value (True/False) this node belongs to.
            Only meaningful when branch_condition is set.
        module_name: Human-readable name for the module, typically the
            class name. Auto-populated from module if empty.
        module_path: Full hierarchical path in the module tree, using
            dot notation (e.g., "encoder.layer1.llm").

    Example:
        >>> from inf_engine.module import LLMInference
        >>> node = GraphNode(
        ...     id="LLMInference_1",
        ...     module=LLMInference(alias="gpt4"),
        ...     args=("input:prompt",),
        ...     kwargs={"temperature": 0.7},
        ...     dependencies=["input:prompt"],
        ... )
        >>> node.module_name
        'LLMInference'
        >>> node.dependencies
        ['input:prompt']
    """

    id: str
    module: InferenceModule | None
    args: tuple[str | Any, ...]
    kwargs: dict[str, str | Any]
    dependencies: list[str]
    priority: int = 0
    branch_condition: str | None = None
    branch_value: bool | None = None
    module_name: str = ""
    module_path: str = ""

    def __post_init__(self) -> None:
        """Auto-populate module_name from module if not provided.

        If module_name is empty and a module is present, sets module_name
        to the module's class name.
        """
        if not self.module_name and self.module is not None:
            self.module_name = self.module.__class__.__name__


@dataclass
class InferenceGraph:
    """Complete execution graph captured from tracing.

    Represents the full dependency graph of an inference pipeline,
    including all operations and their relationships. The graph is
    directed and acyclic (DAG), where edges represent data dependencies.

    Attributes:
        nodes: Dictionary mapping node IDs to GraphNode instances.
        input_ids: List of node IDs that are entry points (no dependencies).
        output_ids: List of node IDs that are exit points (graph outputs).
        parameters: Dictionary mapping parameter names to Parameter instances
            collected from the traced module tree.

    Example:
        >>> # A simple linear graph: input -> llm1 -> llm2 -> output
        >>> graph = InferenceGraph(
        ...     nodes={
        ...         "input:text": input_node,
        ...         "LLM_1": llm1_node,
        ...         "LLM_2": llm2_node,
        ...     },
        ...     input_ids=["input:text"],
        ...     output_ids=["LLM_2"],
        ... )
        >>> len(graph.nodes)
        3

    Note:
        Graph traversal methods (topological_order, ancestors, descendants)
        will be added in subsequent PRs.
    """

    nodes: dict[str, GraphNode]
    input_ids: list[str]
    output_ids: list[str]
    parameters: dict[str, Parameter] = field(default_factory=dict)
