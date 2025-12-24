"""Tracer for capturing execution graphs from eager-mode code.

The Tracer records an InferenceGraph by instrumenting forward() execution.
Similar to torch.fx.Tracer, it captures the dependency graph without
executing actual computations.

Example:
    >>> from inf_engine.tracing.tracer import Tracer
    >>> from inf_engine.module import InferenceModule, LLMInference
    >>>
    >>> class Pipeline(InferenceModule):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.llm = LLMInference(alias="fast")
    ...
    ...     def forward(self, text: str) -> str:
    ...         return self.llm(text)
    >>>
    >>> tracer = Tracer()
    >>> # graph = tracer.trace(Pipeline(), "input text")  # Coming in PR-016
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inf_engine.graph import GraphNode

if TYPE_CHECKING:
    from inf_engine.module import InferenceModule
    from inf_engine.tracing.proxy import Proxy


class Tracer:
    """Records an InferenceGraph by tracing forward() execution.

    Similar to torch.fx.Tracer, but designed for inference modules.
    The tracer captures module calls and their dependencies to build
    an execution graph that can be executed asynchronously.

    Attributes:
        nodes: Dictionary mapping node IDs to GraphNode instances.
        input_ids: List of node IDs that represent inputs to the graph.
        output_ids: List of node IDs that represent outputs of the graph.

    Example:
        >>> tracer = Tracer()
        >>> tracer.nodes
        {}
        >>> tracer.input_ids
        []
        >>> tracer.output_ids
        []

    Note:
        The tracer maintains internal state for generating unique node IDs
        and tracking the current position in the module hierarchy. This
        state is reset when a new trace begins.
    """

    def __init__(self) -> None:
        """Initialize a new Tracer instance.

        Creates empty storage for nodes, inputs, and outputs, and
        initializes internal counters and stacks for tracing.

        Example:
            >>> tracer = Tracer()
            >>> len(tracer.nodes)
            0
        """
        # Node storage
        self.nodes: dict[str, GraphNode] = {}
        self.input_ids: list[str] = []
        self.output_ids: list[str] = []

        # Internal state for ID generation
        self._node_counter: int = 0

        # Stack for tracking module hierarchy during nested calls
        self._module_stack: list[str] = []

        # Stack for tracking branch context during conditional tracing
        # Each entry is (condition_node_id, branch_value)
        self._branch_stack: list[tuple[str, bool]] = []

    def _generate_id(self, module: InferenceModule) -> str:
        """Generate a unique node ID for a module invocation.

        Creates an ID by combining the module's class name with an
        incrementing counter. This ensures each node has a unique
        identifier within the graph.

        Args:
            module: The module being invoked.

        Returns:
            A unique string identifier in the format "ClassName_N"
            where N is an incrementing counter.

        Example:
            >>> from inf_engine.module import LLMInference
            >>> tracer = Tracer()
            >>> module = LLMInference(alias="test")
            >>> tracer._generate_id(module)
            'LLMInference_1'
            >>> tracer._generate_id(module)
            'LLMInference_2'
        """
        self._node_counter += 1
        name = module.__class__.__name__
        return f"{name}_{self._node_counter}"

    def reset(self) -> None:
        """Reset the tracer to its initial state.

        Clears all recorded nodes, inputs, outputs, and resets
        internal counters and stacks. Call this before starting
        a new trace to ensure clean state.

        Example:
            >>> tracer = Tracer()
            >>> # After some tracing operations...
            >>> tracer.reset()
            >>> tracer.nodes
            {}
            >>> tracer._node_counter
            0
        """
        self.nodes.clear()
        self.input_ids.clear()
        self.output_ids.clear()
        self._node_counter = 0
        self._module_stack.clear()
        self._branch_stack.clear()

    def record_getitem(self, proxy: Proxy, key: Any) -> Proxy:
        """Record a getitem operation on a proxy.

        Creates a new graph node representing the indexing operation
        and returns a proxy for that node. This enables tracing of
        dictionary/list access patterns.

        Args:
            proxy: The proxy being indexed.
            key: The key or index being accessed.

        Returns:
            A new Proxy representing the result of the indexing operation.

        Raises:
            NotImplementedError: This stub will be implemented in a future PR.

        Note:
            This is a stub method. Full implementation coming in PR-015.
        """
        raise NotImplementedError("record_getitem will be implemented in PR-015")
