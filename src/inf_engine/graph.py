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

import hashlib
import json
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
            Non-node dependencies (e.g., parameter refs like "param:...")
            are skipped as they represent static values, not graph nodes.

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
            # Skip non-node dependencies (e.g., param refs)
            if node_id not in self.nodes:
                return
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
            A set of node IDs representing all ancestors (only actual graph
            nodes, not parameter refs). Does not include the node itself.
            Returns an empty set if the node has no dependencies.

        Raises:
            KeyError: If node_id is not in the graph.

        Note:
            Non-node dependencies (e.g., parameter refs like "param:...")
            are excluded from the result as they represent static values.

        Example:
            >>> # Graph: input -> a -> b -> c
            >>> graph.ancestors("c")
            {'input', 'a', 'b'}

            >>> # Input nodes have no ancestors
            >>> graph.ancestors("input")
            set()
        """
        result: set[str] = set()
        queue = [d for d in self.nodes[node_id].dependencies if d in self.nodes]

        while queue:
            current = queue.pop()
            if current not in result:
                result.add(current)
                # Only queue dependencies that are actual nodes
                queue.extend(
                    d for d in self.nodes[current].dependencies if d in self.nodes
                )

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

    def compute_hash(self) -> str:
        """Compute a deterministic hash of the graph structure.

        Creates a content-addressed hash based on the logical structure
        of the graph, independent of node IDs. The hash is computed using
        a Merkle-tree approach where each node's hash includes its
        dependencies' hashes.

        The hash is based on:
        - Module type (class name) for each node
        - Module configuration (alias, system_prompt, temperature, etc.)
        - Dependency structure (captured via parent hashes)
        - Input/output structure

        Returns:
            A hex string representing the SHA-256 hash of the graph.
            The same logical graph structure will always produce the
            same hash, even across different Python sessions.

        Example:
            >>> graph = tracer.trace(pipeline, "input")
            >>> hash1 = graph.compute_hash()
            >>> # Same pipeline traced again produces same hash
            >>> graph2 = tracer.trace(pipeline, "different input")
            >>> hash2 = graph2.compute_hash()
            >>> hash1 == hash2
            True

        Note:
            Input values are NOT included in the hash, only the structure.
            This allows checkpoints to be reused across different inputs
            as long as the pipeline structure is the same.
        """
        # Get topological order to ensure deterministic traversal
        topo_order = self.topological_order()

        # Compute hash for each node, incorporating parent hashes
        node_hashes: dict[str, str] = {}

        for node_id in topo_order:
            node = self.nodes[node_id]
            node_hashes[node_id] = self._compute_node_hash(node, node_hashes)

        # Combine all output node hashes for final graph hash
        output_hashes = [node_hashes[oid] for oid in sorted(self.output_ids)]
        graph_data = {
            "output_hashes": output_hashes,
            "num_nodes": len(self.nodes),
            "num_inputs": len(self.input_ids),
            "num_outputs": len(self.output_ids),
        }
        graph_json = json.dumps(graph_data, sort_keys=True)
        return hashlib.sha256(graph_json.encode()).hexdigest()

    def _compute_node_hash(self, node: GraphNode, parent_hashes: dict[str, str]) -> str:
        """Compute a deterministic hash for a single node.

        Args:
            node: The GraphNode to hash.
            parent_hashes: Dictionary of already-computed hashes for
                dependency nodes.

        Returns:
            A hex string representing the SHA-256 hash of this node.

        Note:
            Non-node dependencies (e.g., parameter refs) are included in the
            hash by their ref string directly, since they represent static
            values rather than graph nodes.
        """
        # Get dependency hashes in sorted order for determinism
        # For node dependencies, use their computed hash
        # For non-node dependencies (e.g., param refs), use the ref string directly
        dep_hashes: list[str] = []
        for dep in sorted(node.dependencies):
            if dep in parent_hashes:
                dep_hashes.append(parent_hashes[dep])
            else:
                # Non-node dependency (e.g., param ref) - hash the ref string
                dep_hashes.append(hashlib.sha256(dep.encode()).hexdigest())

        # Extract module configuration
        module_config = self._extract_module_config(node.module)

        # Build canonical representation
        node_data = {
            "module_type": node.module.__class__.__name__ if node.module else "None",
            "module_config": module_config,
            "dependency_hashes": dep_hashes,
            "priority": node.priority,
            "branch_condition": node.branch_condition,
            "branch_value": node.branch_value,
        }

        node_json = json.dumps(node_data, sort_keys=True)
        return hashlib.sha256(node_json.encode()).hexdigest()

    def _extract_module_config(
        self, module: InferenceModule | InputNode | GetItemOp | IterOp | MethodOp | None
    ) -> dict[str, Any]:
        """Extract configuration from a module for hashing.

        Args:
            module: The module to extract configuration from.

        Returns:
            A dictionary of configuration values that define the module's
            behavior. Does not include runtime state or input values.
        """
        if module is None:
            return {}

        # Import here to avoid circular imports
        from inf_engine.module import LLMInference
        from inf_engine.tracing.tracer import GetItemOp, InputNode, IterOp, MethodOp

        config: dict[str, Any] = {"class": module.__class__.__name__}

        if isinstance(module, LLMInference):
            config["alias"] = module.alias
            # Include system_prompt value if it exists
            if module.system_prompt is not None:
                config["system_prompt"] = module.system_prompt.value
            config["temperature"] = module.temperature
            config["max_tokens"] = module.max_tokens
            config["response_format"] = (
                module.response_format.__name__
                if module.response_format is not None
                else None
            )
        elif isinstance(module, InputNode):
            # Don't include the value - only that it's an input
            config["input_name"] = getattr(module, "name", None)
        elif isinstance(module, GetItemOp):
            config["key"] = module.key
        elif isinstance(module, IterOp):
            pass  # No additional config
        elif isinstance(module, MethodOp):
            config["method_name"] = module.method

        return config


def visualize_graph(graph: InferenceGraph) -> str:
    """Generate a DOT representation of an inference graph.

    Creates a Graphviz DOT format string suitable for rendering with
    graphviz or online tools like viz.js. Useful for debugging and
    understanding pipeline structure.

    Args:
        graph: The InferenceGraph to visualize.

    Returns:
        A string in DOT format representing the graph. Can be rendered
        using Graphviz: `echo "output" | dot -Tpng -o graph.png`

    Example:
        >>> from inf_engine.graph import GraphNode, InferenceGraph, visualize_graph
        >>> from inf_engine.tracing.tracer import InputNode
        >>> from inf_engine.module import LLMInference
        >>>
        >>> # Create a simple graph: input -> llm
        >>> input_node = GraphNode(
        ...     id="input:text",
        ...     module=InputNode(value="hello"),
        ...     args=(),
        ...     kwargs={},
        ...     dependencies=[],
        ...     module_name="Input(text)",
        ... )
        >>> llm_node = GraphNode(
        ...     id="LLMInference_1",
        ...     module=LLMInference(alias="fast"),
        ...     args=(),
        ...     kwargs={},
        ...     dependencies=["input:text"],
        ... )
        >>> graph = InferenceGraph(
        ...     nodes={"input:text": input_node, "LLMInference_1": llm_node},
        ...     input_ids=["input:text"],
        ...     output_ids=["LLMInference_1"],
        ... )
        >>> dot = visualize_graph(graph)
        >>> "digraph InferenceGraph" in dot
        True
        >>> '"input:text" -> "LLMInference_1"' in dot
        True

    Note:
        Input nodes are rendered as boxes, output nodes as double octagons,
        and other nodes as ellipses. Branch conditions (if present) are
        shown in the node label.
    """
    lines = ["digraph InferenceGraph {"]
    lines.append("  rankdir=TB;")

    # Add nodes with appropriate shapes
    for node_id, node in graph.nodes.items():
        label = node.module_name if node.module_name else node_id

        # Add branch information to label if present
        if node.branch_condition is not None:
            label += f"\\n[{node.branch_value}]"

        # Determine shape based on node role
        shape = "ellipse"
        if node_id in graph.input_ids:
            shape = "box"
        elif node_id in graph.output_ids:
            shape = "doubleoctagon"

        lines.append(f'  "{node_id}" [label="{label}", shape={shape}];')

    # Add edges for dependencies
    for node_id, node in graph.nodes.items():
        for dep_id in node.dependencies:
            lines.append(f'  "{dep_id}" -> "{node_id}";')

    lines.append("}")
    return "\n".join(lines)
