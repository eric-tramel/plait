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

    """

    nodes: dict[str, GraphNode]
    input_ids: list[str]
    output_ids: list[str]
    parameters: dict[str, Parameter] = field(default_factory=dict)

    def topological_order(self) -> list[str]:
        """Return node IDs in valid execution order.

        Performs a depth-first traversal starting from output nodes,
        visiting dependencies before each node. This ensures nodes are
        ordered such that all dependencies of a node appear before it.

        Returns:
            A list of node IDs in topological order. Nodes with no
            dependencies appear first, followed by nodes whose dependencies
            have been satisfied.

        Note:
            This method assumes the graph is acyclic (DAG). Cyclic graphs
            will result in infinite recursion. Only nodes reachable from
            output_ids are included in the result.

        Example:
            >>> # Linear graph: input -> llm1 -> llm2
            >>> graph.topological_order()
            ['input:text', 'LLM_1', 'LLM_2']

            >>> # Diamond graph: input -> [a, b] -> merge
            >>> graph.topological_order()
            ['input:text', 'a', 'b', 'merge']  # a, b order may vary
        """
        visited: set[str] = set()
        order: list[str] = []

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            for dep_id in self.nodes[node_id].dependencies:
                visit(dep_id)
            order.append(node_id)

        for output_id in self.output_ids:
            visit(output_id)

        return order

    def ancestors(self, node_id: str) -> set[str]:
        """Get all nodes this node depends on, directly or indirectly.

        Traverses the dependency graph backwards from the given node,
        collecting all nodes that must complete before this node can execute.

        Args:
            node_id: The ID of the node to find ancestors for.

        Returns:
            A set of node IDs representing all ancestors. Does not include
            the node itself. Returns an empty set if the node has no
            dependencies.

        Raises:
            KeyError: If node_id is not in the graph.

        Example:
            >>> # Graph: input -> a -> b -> c
            >>> graph.ancestors("c")
            {'input', 'a', 'b'}

            >>> # Input nodes have no ancestors
            >>> graph.ancestors("input")
            set()
        """
        result: set[str] = set()
        queue = list(self.nodes[node_id].dependencies)

        while queue:
            current = queue.pop()
            if current not in result:
                result.add(current)
                queue.extend(self.nodes[current].dependencies)

        return result

    def descendants(self, node_id: str) -> set[str]:
        """Get all nodes that depend on this node, directly or indirectly.

        Traverses the dependency graph forwards from the given node,
        collecting all nodes that require this node's output. Used for
        failure cascading when a node fails and its descendants must
        be cancelled.

        Args:
            node_id: The ID of the node to find descendants for.

        Returns:
            A set of node IDs representing all descendants. Does not include
            the node itself. Returns an empty set if no other nodes depend
            on this node.

        Raises:
            KeyError: If node_id is not in the graph.

        Example:
            >>> # Graph: input -> a -> b -> c
            >>> graph.descendants("input")
            {'a', 'b', 'c'}

            >>> # Output nodes have no descendants
            >>> graph.descendants("c")
            set()
        """
        result: set[str] = set()
        queue = [node_id]

        while queue:
            current = queue.pop()
            for nid, node in self.nodes.items():
                if current in node.dependencies and nid not in result:
                    result.add(nid)
                    queue.append(nid)

        return result
