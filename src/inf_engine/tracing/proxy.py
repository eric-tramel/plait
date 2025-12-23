"""Proxy objects for symbolic tracing.

During tracing, module calls return Proxy objects instead of actual values.
Proxies are symbolic placeholders that track data flow through the graph.

Example:
    During tracing, when a module is called:

    >>> # Inside a trace context
    >>> output = some_module(input_proxy)
    >>> # output is a Proxy, not an actual value
    >>> print(output)
    Proxy(SomeModule_1)

    When a Proxy is passed to another module, it creates a dependency edge:

    >>> # This creates a dependency: SomeModule_1 -> AnotherModule_2
    >>> result = another_module(output)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Type alias for Tracer - will be properly typed when Tracer is implemented (PR-013+).
# Using Any for now since Tracer doesn't exist yet.
Tracer = Any


@dataclass
class Proxy:
    """Symbolic placeholder returned during tracing.

    When a Proxy is passed to another module, it creates a dependency edge
    in the inference graph. Proxies track the data flow through the module
    graph without executing actual computations.

    Attributes:
        node_id: Unique identifier for the graph node this proxy represents.
        tracer: Reference to the Tracer that created this proxy.
        output_index: Index for multi-output nodes (default 0 for single output).
        _metadata: Additional metadata for profiling and debugging.

    Example:
        >>> # During tracing, module calls return proxies
        >>> proxy = Proxy(node_id="LLMInference_1", tracer=tracer)
        >>> print(proxy)
        Proxy(LLMInference_1)

        >>> # Proxies can be indexed to create new dependency nodes
        >>> item = proxy["key"]  # Creates a getitem node
    """

    node_id: str
    tracer: Tracer
    output_index: int = 0
    _metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return string representation of the proxy.

        Returns:
            A string in the format 'Proxy(node_id)'.
        """
        return f"Proxy({self.node_id})"

    def __getitem__(self, key: Any) -> Proxy:
        """Handle dictionary/list indexing.

        Creates a new graph node representing the indexing operation
        and returns a proxy for that node.

        Args:
            key: The key or index to access.

        Returns:
            A new Proxy representing the result of the indexing operation.

        Raises:
            AttributeError: If tracer doesn't have record_getitem method.

        Example:
            >>> # Access a dictionary key
            >>> value = proxy["result"]

            >>> # Access a list index
            >>> first = proxy[0]
        """
        return self.tracer.record_getitem(self, key)
