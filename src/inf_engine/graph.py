"""Graph data structures for representing traced inference pipelines.

This module provides the core data structures for representing execution graphs
captured during tracing. The graph structure enables automatic parallelization,
dependency tracking, and optimization of inference pipelines.

Example:
    >>> from inf_engine.graph import GraphNode, InferenceGraph
    >>> from inf_engine.module import LLMInference
    >>> from inf_engine.tracing.tracer import InputNode
    >>>
    >>> # Create nodes representing operations
    >>> input_node = GraphNode(
    ...     id="input:text",
    ...     module=InputNode(value="sample text"),
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
    from inf_engine.tracing.tracer import GetItemOp, InputNode, IterOp, MethodOp


@dataclass(frozen=True)
class NodeRef:
    """A typed reference to a node in the execution graph.

    NodeRef wraps a node ID string to distinguish it from literal string
    values in args and kwargs. This prevents collision when a literal string
    argument happens to match a node ID.

    Attributes:
        node_id: The ID of the referenced node.

    Example:
        >>> ref = NodeRef("LLMInference_1")
        >>> ref.node_id
        'LLMInference_1'
        >>> str(ref)
        'NodeRef(LLMInference_1)'

        >>> # Used in GraphNode args to reference another node's output
        >>> node = GraphNode(
        ...     id="LLMInference_2",
        ...     module=module,
        ...     args=(NodeRef("LLMInference_1"),),  # Reference, not literal
        ...     kwargs={"literal_key": "literal_value"},  # Literal string
        ...     dependencies=["LLMInference_1"],
        ... )

    Note:
        NodeRef is frozen (immutable) and can be used as a dict key or
        in sets. Two NodeRefs with the same node_id are considered equal.
    """

    node_id: str

    def __repr__(self) -> str:
        """Return a string representation of the NodeRef."""
        return f"NodeRef({self.node_id})"


@dataclass
class GraphNode:
    """A single operation in the execution graph.

    Represents one module invocation captured during tracing. Contains
    information about the operation, its dependencies, and metadata
    for scheduling and debugging.

    Attributes:
        id: Unique identifier for this node within the graph.
        module: The operation to execute. For inference nodes, an
            InferenceModule instance. For input nodes, an InputNode
            containing the input value. For data access operations,
            a GetItemOp, IterOp, or MethodOp. May be None for special cases.
        args: Positional arguments as a tuple of NodeRef (for references
            to other nodes) or literal values.
        kwargs: Keyword arguments as a dict of NodeRef (for references)
            or literal values.
        dependencies: List of node IDs this node depends on. The node
            cannot execute until all dependencies have completed.
        priority: Execution priority for scheduling. Lower values indicate
            higher precedence (0 runs before 1). Defaults to 0.
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
        >>> from inf_engine.graph import NodeRef
        >>> node = GraphNode(
        ...     id="LLMInference_1",
        ...     module=LLMInference(alias="gpt4"),
        ...     args=(NodeRef("input:prompt"),),
        ...     kwargs={"temperature": 0.7},
        ...     dependencies=["input:prompt"],
        ... )
        >>> node.module_name
        'LLMInference'
        >>> node.dependencies
        ['input:prompt']
    """

    id: str
    module: InferenceModule | InputNode | GetItemOp | IterOp | MethodOp | None
    args: tuple[NodeRef | Any, ...]
    kwargs: dict[str, NodeRef | Any]
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
        output_structure: The original structure of the forward() return value,
            with node IDs in place of Proxy objects. Used to reconstruct
            results with user-defined keys. Can be a string (single node ID),
            dict (mapping user keys to node IDs), list, or None.
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
    output_structure: str | dict[str, Any] | list[Any] | None = None
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

        Raises:
            ValueError: If the graph contains a cycle. The error message
                includes the cycle path for debugging.

        Note:
            Only nodes reachable from output_ids are included in the result.

        Example:
            >>> # Linear graph: input -> llm1 -> llm2
            >>> graph.topological_order()
            ['input:text', 'LLM_1', 'LLM_2']

            >>> # Diamond graph: input -> [a, b] -> merge
            >>> graph.topological_order()
            ['input:text', 'a', 'b', 'merge']  # a, b order may vary

            >>> # Cyclic graph raises ValueError
            >>> graph.topological_order()
            ValueError: Cycle detected in graph: a -> b -> c -> a
        """
        visited: set[str] = set()
        visiting: set[str] = set()  # Track nodes in current DFS path
        order: list[str] = []

        def visit(node_id: str, path: list[str]) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                # Found a cycle - construct the cycle path
                cycle_start = path.index(node_id)
                cycle_path = path[cycle_start:] + [node_id]
                cycle_str = " -> ".join(cycle_path)
                raise ValueError(f"Cycle detected in graph: {cycle_str}")

            visiting.add(node_id)
            path.append(node_id)

            for dep_id in self.nodes[node_id].dependencies:
                visit(dep_id, path)

            path.pop()
            visiting.remove(node_id)
            visited.add(node_id)
            order.append(node_id)

        for output_id in self.output_ids:
            visit(output_id, [])

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
