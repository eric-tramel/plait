"""ForwardRecord for capturing forward pass state.

This module provides the ForwardRecord dataclass which captures the state
of a forward pass execution for use in backward propagation. Similar to
PyTorch's autograd tape, it stores the computation graph and all intermediate
values needed to propagate feedback during backward().

Example:
    >>> from inf_engine import run
    >>> from inf_engine.optimization.record import ForwardRecord
    >>>
    >>> # Execute with recording enabled
    >>> output, record = await run(module, input_text, record=True)
    >>>
    >>> # Record contains graph and execution details
    >>> isinstance(record, ForwardRecord)
    True
    >>> len(record.node_outputs) > 0
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inf_engine.graph import InferenceGraph
    from inf_engine.module import InferenceModule


@dataclass
class ForwardRecord:
    """Captures forward pass state for backward propagation.

    Stores the computation graph and all intermediate values needed
    to propagate feedback during backward(). Created when running
    with record=True, analogous to PyTorch's autograd tape.

    This class enables the optimization system to trace back through
    the computation graph during backward passes, allowing feedback
    to be propagated to the appropriate Parameters.

    Attributes:
        graph: The InferenceGraph representing the computation structure.
        node_inputs: Dictionary mapping node_id to the resolved input values
            that were passed to each node during execution.
        node_outputs: Dictionary mapping node_id to the output value
            produced by each node during execution.
        module_map: Dictionary mapping node_id to the InferenceModule instance
            that was executed for that node. Used to call module.backward().
        execution_order: List of node IDs in the order they were executed.
            Useful for debugging and understanding execution flow.
        timing: Dictionary mapping node_id to execution time in seconds.
            Useful for profiling and identifying bottlenecks.

    Example:
        >>> # ForwardRecord is returned from run() when record=True
        >>> output, record = await run(module, "Hello", record=True)
        >>>
        >>> # Access the computation graph
        >>> len(record.graph.nodes)
        3
        >>>
        >>> # Access intermediate outputs
        >>> record.node_outputs["LLMInference_1"]
        'Response from LLM...'
        >>>
        >>> # Access the module that produced a node
        >>> record.module_map["LLMInference_1"]
        <LLMInference alias='assistant'>

    Note:
        ForwardRecord instances should be passed to loss functions via
        the `record=record` parameter to enable `feedback.backward()`.
    """

    graph: InferenceGraph
    node_inputs: dict[str, dict[str, Any]]
    node_outputs: dict[str, Any]
    module_map: dict[str, InferenceModule]
    execution_order: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)

    def get_node_input(self, node_id: str) -> dict[str, Any]:
        """Get the resolved input values for a node.

        Args:
            node_id: The ID of the node to get inputs for.

        Returns:
            A dictionary of input name to value mappings.

        Raises:
            KeyError: If the node_id is not in the record.
        """
        return self.node_inputs[node_id]

    def get_node_output(self, node_id: str) -> Any:
        """Get the output value produced by a node.

        Args:
            node_id: The ID of the node to get output for.

        Returns:
            The output value produced by the node.

        Raises:
            KeyError: If the node_id is not in the record.
        """
        return self.node_outputs[node_id]

    def get_module(self, node_id: str) -> InferenceModule:
        """Get the module instance for a node.

        Args:
            node_id: The ID of the node to get the module for.

        Returns:
            The InferenceModule instance that executed for this node.

        Raises:
            KeyError: If the node_id is not in the record.
        """
        return self.module_map[node_id]
