"""Backward pass infrastructure for feedback propagation.

This module provides the infrastructure for propagating feedback
backward through traced computation graphs. The backward pass enables
LLM-based optimization by distributing feedback from loss functions
to Parameters throughout the module tree.

The core components are:
- AncestorContext: Ancestor and sibling context for parameter updates
- BackwardContext: Information available to modules during backward pass
- BackwardResult: Result of a module's backward pass
- _propagate_backward: Function to traverse graph in reverse topological order

Example:
    >>> from plait.optimization.backward import BackwardContext, BackwardResult
    >>>
    >>> # BackwardContext provides execution context during backward
    >>> ctx = BackwardContext(
    ...     node_id="LLMInference_1",
    ...     inputs={"prompt": "Hello"},
    ...     output="Hi there!",
    ...     graph=graph,
    ...     all_results=node_outputs,
    ...     downstream_feedback=[feedback],
    ... )
    >>>
    >>> # BackwardResult collects feedback for inputs and parameters
    >>> result = BackwardResult()
    >>> result.input_feedback["prompt"] = feedback
    >>> result.parameter_feedback["system_prompt"] = "Be more concise"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from plait.graph import InferenceGraph
    from plait.module import LLMInference
    from plait.optimization.feedback import Feedback
    from plait.optimization.optimizer import Optimizer
    from plait.optimization.record import ForwardRecord
    from plait.parameter import Parameter


@dataclass
class AncestorContext:
    """Context about ancestor and sibling values/parameters for optimization.

    When updating parameters during optimizer.step(), the optimizer needs
    visibility into the full context of values and parameters that influenced
    the current computation. This enables coherent, context-aware updates.

    Attributes:
        ancestor_values: Dictionary mapping node IDs to their output values
            for all ancestor nodes in the computation graph.
        ancestor_params: Dictionary mapping parameter names to Parameter
            objects for all parameters in ancestor nodes.
        sibling_values: Dictionary mapping node IDs to their output values
            for sibling nodes (nodes at the same level sharing ancestors).
        sibling_params: Dictionary mapping parameter names to Parameter
            objects for parameters in sibling nodes.

    Example:
        >>> # For a node in a diamond graph: input -> [A, B] -> merge
        >>> # When updating merge's parameters:
        >>> ctx = AncestorContext(
        ...     ancestor_values={"input": "input text", "A": "A output", "B": "B output"},
        ...     ancestor_params={"A.system_prompt": param_a, "B.system_prompt": param_b},
        ...     sibling_values={},  # merge has no siblings
        ...     sibling_params={},
        ... )

    Note:
        - Ancestors are nodes upstream in the computation graph that
          contributed to producing the current node's input.
        - Siblings are nodes at the same topological level that share
          common ancestors but don't directly depend on each other.
        - This context enables the optimizer to make consistent updates
          that consider how parameters interact with each other.
    """

    ancestor_values: dict[str, Any] = field(default_factory=dict)
    ancestor_params: dict[str, Parameter] = field(default_factory=dict)
    sibling_values: dict[str, Any] = field(default_factory=dict)
    sibling_params: dict[str, Parameter] = field(default_factory=dict)

    def has_ancestors(self) -> bool:
        """Check if this context has any ancestor information.

        Returns:
            True if there are any ancestor values or parameters.
        """
        return bool(self.ancestor_values or self.ancestor_params)

    def has_siblings(self) -> bool:
        """Check if this context has any sibling information.

        Returns:
            True if there are any sibling values or parameters.
        """
        return bool(self.sibling_values or self.sibling_params)


@dataclass
class BackwardContext:
    """Context available to modules during async backward pass.

    Provides access to inputs, outputs, graph structure, and
    optionally an LLM for reasoning about feedback. This context
    enables modules to make informed decisions about how to
    distribute feedback to their inputs and parameters.

    Attributes:
        node_id: The ID of the current node in the graph.
        inputs: Dictionary of resolved input values from the forward pass,
            keyed by input name or position.
        output: The value this module produced during forward pass.
        graph: The full InferenceGraph for structural information.
        all_results: Dictionary of all node outputs from the forward pass,
            useful for understanding the broader execution context.
        downstream_feedback: List of feedback items from nodes that
            depend on this node's output.
        reasoning_llm: Optional LLM provided by the optimizer for
            generating sophisticated feedback during backward pass.

    Example:
        >>> # Context is created by _propagate_backward and passed to backward()
        >>> async def backward(self, feedback: Feedback, ctx: BackwardContext):
        ...     # Access what this module received as input
        ...     input_text = ctx.inputs.get("prompt", "")
        ...
        ...     # Access what this module produced
        ...     output_text = ctx.output
        ...
        ...     # Use reasoning LLM if available
        ...     if ctx.reasoning_llm:
        ...         analysis = await ctx.reason("How should we improve?")
    """

    # This module's execution
    node_id: str
    inputs: dict[str, Any]
    output: Any

    # Graph context
    graph: InferenceGraph
    all_results: dict[str, Any]

    # Feedback from downstream nodes
    downstream_feedback: list[Feedback]

    # Optimizer-provided LLM for backward reasoning (optional)
    reasoning_llm: LLMInference | None = None

    async def reason(self, prompt: str) -> str:
        """Use the optimizer's LLM for backward-pass reasoning.

        This enables custom backward() implementations to use LLM
        reasoning to generate better parameter feedback. The reasoning
        LLM is provided by the optimizer and is separate from the
        modules being optimized.

        Args:
            prompt: The reasoning prompt to send to the LLM.

        Returns:
            The LLM's response as a string.

        Raises:
            RuntimeError: If no reasoning LLM is available. This happens
                when backward() is called without an optimizer, or when
                the optimizer doesn't have a reasoning_llm configured.

        Example:
            >>> async def backward(self, feedback, ctx):
            ...     if ctx.reasoning_llm:
            ...         analysis = await ctx.reason(
            ...             f"Given feedback: {feedback.content}\\n"
            ...             f"How should we improve the system prompt?"
            ...         )
            ...         # Use analysis to generate parameter feedback
        """
        if self.reasoning_llm is None:
            raise RuntimeError(
                "No reasoning LLM available. Pass optimizer to backward() "
                "or configure optimizer with reasoning_model."
            )
        # Use normal module call - not traced during backward pass
        # The reasoning_llm should be bound to execute properly
        return await self.reasoning_llm(prompt)


@dataclass
class BackwardResult:
    """Result of a module's backward pass.

    Contains feedback to propagate to inputs and to accumulate
    in parameters. Each module's backward() method returns a
    BackwardResult that specifies how feedback should flow.

    Attributes:
        input_feedback: Dictionary mapping input names to Feedback
            objects that should be propagated to upstream nodes.
            Keys should match the input names from the forward pass.
        parameter_feedback: Dictionary mapping parameter names to
            feedback strings that should be accumulated into those
            parameters. The strings describe how the parameter
            should be improved.

    Example:
        >>> async def backward(self, feedback, ctx) -> BackwardResult:
        ...     result = BackwardResult()
        ...
        ...     # Propagate feedback to the input node
        ...     result.input_feedback["prompt"] = Feedback(
        ...         content=f"Downstream received: {feedback.content}",
        ...         score=feedback.score,
        ...     )
        ...
        ...     # Accumulate feedback for a learnable parameter
        ...     result.parameter_feedback["system_prompt"] = (
        ...         f"The output received this feedback: {feedback.content}. "
        ...         f"Consider adjusting the system prompt to address this."
        ...     )
        ...
        ...     return result
    """

    # Feedback for each input (keyed by input name or position)
    input_feedback: dict[str, Feedback] = field(default_factory=dict)

    # Feedback for each parameter (keyed by parameter name)
    parameter_feedback: dict[str, str] = field(default_factory=dict)


def build_ancestor_context(
    node_id: str,
    record: ForwardRecord,
) -> AncestorContext:
    """Build ancestor context for a node from a ForwardRecord.

    Collects all ancestor and sibling values and parameters for a given
    node in the computation graph. This context is used during optimizer
    step() to provide full visibility into upstream computations.

    Args:
        node_id: The ID of the node to build context for.
        record: The ForwardRecord containing graph, outputs, and module map.

    Returns:
        An AncestorContext with ancestor and sibling values/parameters.

    Example:
        >>> ctx = build_ancestor_context("merge_node", record)
        >>> ctx.ancestor_values
        {"input": "text", "a": "A output", "b": "B output"}
    """
    graph = record.graph

    # Collect ancestor context
    ancestor_values, ancestor_params = graph.collect_ancestor_context(
        node_id,
        record.node_outputs,
        record.module_map,
    )

    # Collect sibling context
    sibling_values, sibling_params = graph.collect_sibling_context(
        node_id,
        record.node_outputs,
        record.module_map,
    )

    return AncestorContext(
        ancestor_values=ancestor_values,
        ancestor_params=ancestor_params,
        sibling_values=sibling_values,
        sibling_params=sibling_params,
    )


def _combine_feedback(feedbacks: list[Feedback]) -> Feedback:
    """Combine multiple feedback items into one.

    When a node has multiple downstream dependents, their feedback
    needs to be combined before being passed to the node's backward().
    This function concatenates feedback content and averages scores.

    Args:
        feedbacks: List of Feedback objects to combine.

    Returns:
        A single Feedback object with combined content and averaged score.

    Example:
        >>> fb1 = Feedback(content="Too verbose", score=0.6)
        >>> fb2 = Feedback(content="Good structure", score=0.8)
        >>> combined = _combine_feedback([fb1, fb2])
        >>> combined.score
        0.7
    """
    from plait.optimization.feedback import Feedback, FeedbackType

    if len(feedbacks) == 1:
        return feedbacks[0]

    # Simple concatenation with score averaging
    combined_content = "\n\n".join(
        f"[Downstream {i + 1}] {fb.content}" for i, fb in enumerate(feedbacks)
    )
    scores = [fb.score for fb in feedbacks if fb.score is not None]
    avg_score = sum(scores) / len(scores) if scores else None

    return Feedback(
        content=combined_content,
        score=avg_score,
        feedback_type=FeedbackType.COMPOSITE,
    )


def _resolve_input_node(
    node_id: str,
    input_name: str,
    record: ForwardRecord,
) -> str | None:
    """Resolve which node provided a given input to a node.

    Maps from an input name (like "prompt" or "0") to the node_id
    that provided that input during the forward pass. Handles both
    NodeRef (from classic tracing) and ValueRef (from Value-driven tracing).

    Args:
        node_id: The node receiving the input.
        input_name: The name or index of the input.
        record: The ForwardRecord containing execution information.

    Returns:
        The node_id of the input provider, or None if not found.
    """
    from plait.graph import NodeRef
    from plait.values import ValueRef

    def _get_ref_id(arg: Any) -> str | None:
        """Extract node ID from NodeRef or ValueRef."""
        if isinstance(arg, NodeRef):
            return arg.node_id
        elif isinstance(arg, ValueRef):
            return arg.ref
        return None

    node = record.graph.nodes[node_id]

    # Check positional args
    if input_name.isdigit():
        idx = int(input_name)
        if idx < len(node.args):
            arg = node.args[idx]
            ref_id = _get_ref_id(arg)
            if ref_id is not None:
                return ref_id
    else:
        # Check kwargs
        if input_name in node.kwargs:
            arg = node.kwargs[input_name]
            ref_id = _get_ref_id(arg)
            if ref_id is not None:
                return ref_id

        # Check args by iterating (for cases where input_name is a parameter name)
        for arg in node.args:
            ref_id = _get_ref_id(arg)
            if ref_id is not None:
                # If we only have one arg, assume it matches
                if len(node.args) == 1:
                    return ref_id

    return None


def _build_dependents_map(graph: InferenceGraph) -> dict[str, list[str]]:
    """Build a mapping from node_id to its dependents (reverse of dependencies).

    Args:
        graph: The inference graph.

    Returns:
        Dictionary mapping each node_id to a list of node IDs that depend on it.
    """
    dependents: dict[str, list[str]] = {node_id: [] for node_id in graph.nodes}

    for node_id, node in graph.nodes.items():
        for dep_id in node.dependencies:
            if dep_id in dependents:
                dependents[dep_id].append(node_id)

    return dependents


async def _propagate_backward(
    feedback: Feedback,
    record: ForwardRecord,
    reasoning_llm: LLMInference | None = None,
    optimizer: Optimizer | None = None,
) -> None:
    """Propagate feedback backward through a traced graph.

    Traverses nodes in reverse topological order, calling each module's
    backward() method and accumulating feedback into Parameters. This
    is the core algorithm for LLM-based optimization.

    The algorithm:
    1. Capture the record for topological ordering (if optimizer provided)
    2. Initialize output nodes with the loss feedback
    3. For each node in reverse topological order:
       a. Gather feedback from all downstream nodes
       b. Combine feedback if multiple sources
       c. Create BackwardContext with forward pass information
       d. Call module.backward() to get BackwardResult
       e. Distribute input feedback to upstream nodes
       f. Accumulate parameter feedback into Parameters

    Args:
        feedback: The feedback to propagate (from loss function).
        record: ForwardRecord from the forward pass containing graph,
            node inputs/outputs, and module map.
        reasoning_llm: Optional LLM for backward reasoning, typically
            provided by the optimizer.
        optimizer: Optional optimizer to capture the record for ordered
            parameter updates in step().

    Example:
        >>> # Called internally by Feedback.backward()
        >>> await feedback.backward()
        >>>
        >>> # Or with explicit optimizer
        >>> await feedback.backward(optimizer=my_optimizer)
    """
    # Capture record for topological ordering in step()
    if optimizer is not None:
        optimizer.capture_record(record)
    graph = record.graph
    feedback_map: dict[str, Feedback] = {}

    # Build reverse dependency map (node -> nodes that depend on it)
    dependents_map = _build_dependents_map(graph)

    # Initialize with output feedback
    for output_id in graph.output_ids:
        feedback_map[output_id] = feedback

    # Process in reverse topological order
    for node_id in reversed(graph.topological_order()):
        # Skip input nodes - they don't have backward()
        if node_id in graph.input_ids:
            continue

        # Gather feedback from downstream nodes (nodes that depend on this node)
        downstream_feedback: list[Feedback] = []
        for dependent_id in dependents_map.get(node_id, []):
            if dependent_id in feedback_map:
                downstream_feedback.append(feedback_map[dependent_id])

        # Also check if this is an output node (no dependents but has feedback)
        if not downstream_feedback and node_id in feedback_map:
            downstream_feedback.append(feedback_map[node_id])

        if not downstream_feedback:
            continue

        # Combine downstream feedback if multiple
        combined = _combine_feedback(downstream_feedback)

        # Create context with forward pass information
        ctx = BackwardContext(
            node_id=node_id,
            inputs=record.node_inputs.get(node_id, {}),
            output=record.node_outputs.get(node_id),
            graph=graph,
            all_results=record.node_outputs,
            downstream_feedback=downstream_feedback,
            reasoning_llm=reasoning_llm,
        )

        # Call module's backward (async)
        module = record.module_map[node_id]
        result = await module.backward(combined, ctx)

        # Distribute input feedback to upstream nodes
        for input_name, input_fb in result.input_feedback.items():
            input_node_id = _resolve_input_node(node_id, input_name, record)
            if input_node_id:
                feedback_map[input_node_id] = input_fb

        # Accumulate parameter feedback
        for param_name, param_fb in result.parameter_feedback.items():
            # Find the parameter in the module's tree
            for name, param in module.named_parameters():
                if name == param_name or name.endswith(f".{param_name}"):
                    param.accumulate_feedback(param_fb)
                    break
