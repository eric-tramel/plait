"""Tracer for capturing execution graphs from eager-mode code.

The Tracer records an InferenceGraph by instrumenting forward() execution.
Similar to torch.fx.Tracer, it captures the dependency graph without
executing actual computations.

Example:
    >>> from plait.tracing.tracer import Tracer
    >>> from plait.module import Module, LLMInference
    >>> from plait.values import Value
    >>>
    >>> class Pipeline(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.llm = LLMInference(alias="fast")
    ...
    ...     def forward(self, text: Value) -> Value:
    ...         return self.llm(text)
    >>>
    >>> tracer = Tracer()
    >>> graph = tracer.trace_values(Pipeline(), "input text")
    >>> graph.input_ids
    ['input:input_0']
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from plait.graph import GraphNode, InferenceGraph
from plait.tracing.context import trace_context
from plait.values import (
    Value,
    ValueKind,
    ValueRef,
    collect_refs,
    replace_values_with_refs,
    valueify,
)

if TYPE_CHECKING:
    from plait.module import Module


@dataclass
class InputNode:
    """Placeholder node representing an input to the traced graph.

    InputNode wraps the actual input value provided during tracing.
    During execution, the scheduler retrieves the value from the InputNode
    to feed into downstream operations.

    Attributes:
        value: The actual input value captured during tracing.

    Example:
        >>> node = InputNode(value="Hello, world!")
        >>> node.value
        'Hello, world!'

        >>> # During tracing, input nodes are created automatically
        >>> tracer = Tracer()
        >>> graph = tracer.trace_values(LLMInference(alias="test"), "input text")
        >>> input_node = graph.nodes[graph.input_ids[0]].module
        >>> input_node.value
        'input text'
    """

    value: Any


@dataclass
class GetItemOp:
    """Operation node representing dictionary/list indexing.

    Created when a Value is indexed during tracing. During execution,
    this operation retrieves the value at the specified key from the
    source node's output.

    Attributes:
        key: The key or index used for the access.

    Example:
        >>> op = GetItemOp(key="result")
        >>> op.key
        'result'
    """

    key: Any


@dataclass
class IterOp:
    """Operation node representing iteration over a Value.

    Created when a Value is iterated during tracing. During execution, this operation
    yields elements from the source node's output.

    Example:
        >>> op = IterOp()
    """

    pass


@dataclass
class MethodOp:
    """Operation node representing a method call on a Value.

    Created when methods like keys(), values(), or items() are called
    on a Value during tracing. During execution, this operation calls the specified
    method on the source node's output.

    Attributes:
        method: The name of the method being called.

    Example:
        >>> op = MethodOp(method="keys")
        >>> op.method
        'keys'
    """

    method: str


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

    def _generate_id(self, module: Module) -> str:
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
            >>> from plait.module import LLMInference
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

    def bind_inputs(self, inputs: Any, prefix: str = "input") -> Any:
        """Bind input values with refs and create input nodes.

        Recursively traverses the input structure and assigns refs to Values.
        For each Value encountered, an input node is created in the graph
        and the Value's ref is set to point to that node.

        Args:
            inputs: Input values (Value, list, tuple, dict, or literal).
                Values will have their refs set to point to input nodes.
                Literals are converted to Values first.
            prefix: Prefix for naming input nodes (default "input").

        Returns:
            The input structure with all Values having refs assigned.
            The structure is preserved (lists remain lists, dicts remain dicts).

        Example:
            >>> tracer = Tracer()
            >>> v = Value(ValueKind.TEXT, "hello")
            >>> bound = tracer.bind_inputs(v, prefix="input")
            >>> bound.ref
            'input:input_0'

            >>> tracer = Tracer()
            >>> inputs = [Value(ValueKind.TEXT, "a"), Value(ValueKind.TEXT, "b")]
            >>> bound = tracer.bind_inputs(inputs, prefix="input")
            >>> [v.ref for v in bound]
            ['input:input_0', 'input:input_1']

        Note:
            This method creates input nodes in the graph for each Value.
            The refs follow the format 'input:{prefix}_{index}' for positional
            inputs or 'input:{prefix}_{key}' for dict keys.
        """
        counter = [0]  # Mutable counter for nested calls

        def _bind(obj: Any, key_prefix: str, use_index: bool) -> Any:
            if isinstance(obj, Value):
                # Create input node and assign ref
                idx = counter[0]
                counter[0] += 1
                name = f"{key_prefix}_{idx}" if use_index else key_prefix
                node_id = f"input:{name}"

                self.input_ids.append(node_id)
                node = GraphNode(
                    id=node_id,
                    module=InputNode(obj.payload),
                    args=(),
                    kwargs={},
                    dependencies=[],
                    module_name=f"Input({name})",
                )
                self.nodes[node_id] = node

                # Return new Value with ref set
                return Value(
                    kind=obj.kind,
                    payload=obj.payload,
                    ref=node_id,
                    meta=obj.meta.copy(),
                )
            elif isinstance(obj, dict):
                return {
                    k: _bind(v, f"{key_prefix}_{k}", use_index=use_index)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [_bind(item, key_prefix, use_index=True) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(_bind(item, key_prefix, use_index=True) for item in obj)
            else:
                # Literal value - convert to Value first, then bind
                val = valueify(obj)
                return _bind(val, key_prefix, use_index=use_index)

        use_index = not isinstance(inputs, dict)
        return _bind(inputs, prefix, use_index=use_index)

    def record_call(
        self,
        module: Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Value:
        """Record a module invocation during tracing.

        Called by Module.__call__ when tracing is active. Creates
        a graph node representing the module call and tracks dependencies
        based on Value objects in the arguments.

        Dependencies are collected from Value objects via their .ref
        attributes (using collect_refs).

        Args are stored with Value objects replaced by ValueRef(ref).

        Args:
            module: The module being called.
            args: Positional arguments passed to the module.
            kwargs: Keyword arguments passed to the module.

        Returns:
            A Value representing the eventual output of this call,
            with ref set to the generated node_id.

        Example:
            >>> tracer = Tracer()
            >>> module = LLMInference(alias="test")
            >>> input_value = Value(ValueKind.TEXT, "hello", ref="input:0")
            >>> output = tracer.record_call(module, (input_value,), {})
            >>> output.ref
            'LLMInference_1'
            >>> tracer.nodes['LLMInference_1'].dependencies
            ['input:0']

        Note:
            This method mutates the tracer's internal node registry by adding
            a new GraphNode for the module invocation.
        """
        node_id = self._generate_id(module)

        # Collect dependencies from Value objects (via collect_refs)
        dependencies = collect_refs(*args, **kwargs)

        # Replace Value objects with ValueRef placeholders in args/kwargs
        processed_args = list(replace_values_with_refs(args))
        processed_kwargs = replace_values_with_refs(kwargs)

        # Get branch context if we're inside a conditional
        branch_condition: str | None = None
        branch_value: bool | None = None
        if self._branch_stack:
            branch_condition, branch_value = self._branch_stack[-1]

        # Create the graph node
        node = GraphNode(
            id=node_id,
            module=module,
            args=tuple(processed_args),
            kwargs=processed_kwargs,
            dependencies=dependencies,
            branch_condition=branch_condition,
            branch_value=branch_value,
            module_path=".".join(self._module_stack) if self._module_stack else "",
        )
        self.nodes[node_id] = node

        # Return Value with ref pointing to this node
        return Value(kind=ValueKind.RESPONSE, payload=None, ref=node_id)

    def record_getitem(self, value: Value, key: Any) -> Value:
        """Record a getitem operation on a Value during tracing."""
        if value.ref is None:
            return value

        self._node_counter += 1
        node_id = f"getitem_{self._node_counter}"

        # Get branch context if we're inside a conditional
        branch_condition: str | None = None
        branch_value: bool | None = None
        if self._branch_stack:
            branch_condition, branch_value = self._branch_stack[-1]

        node = GraphNode(
            id=node_id,
            module=GetItemOp(key=key),
            args=(ValueRef(value.ref),),
            kwargs={},
            dependencies=[value.ref],
            branch_condition=branch_condition,
            branch_value=branch_value,
            module_name=f"getitem[{key!r}]",
        )
        self.nodes[node_id] = node

        payload = None
        kind = ValueKind.STRUCTURED
        try:
            payload = value.payload[key]
            from plait.values import _infer_kind

            kind = (
                ValueKind.STRUCTURED
                if isinstance(payload, (dict, list, tuple))
                else _infer_kind(payload)
            )
        except Exception:
            payload = None

        return Value(kind=kind, payload=payload, ref=node_id)

    def record_iter(self, value: Value) -> Value:
        """Record an iteration operation on a Value during tracing."""
        if value.ref is None:
            return value

        self._node_counter += 1
        node_id = f"iter_{self._node_counter}"

        # Get branch context if we're inside a conditional
        branch_condition: str | None = None
        branch_value: bool | None = None
        if self._branch_stack:
            branch_condition, branch_value = self._branch_stack[-1]

        node = GraphNode(
            id=node_id,
            module=IterOp(),
            args=(ValueRef(value.ref),),
            kwargs={},
            dependencies=[value.ref],
            branch_condition=branch_condition,
            branch_value=branch_value,
            module_name="iter",
        )
        self.nodes[node_id] = node

        return Value(kind=ValueKind.STRUCTURED, payload=None, ref=node_id)

    def record_method(self, value: Value, method: str) -> Value:
        """Record a method call on a Value during tracing."""
        if value.ref is None:
            return value

        self._node_counter += 1
        node_id = f"method_{self._node_counter}"

        # Get branch context if we're inside a conditional
        branch_condition: str | None = None
        branch_value: bool | None = None
        if self._branch_stack:
            branch_condition, branch_value = self._branch_stack[-1]

        node = GraphNode(
            id=node_id,
            module=MethodOp(method=method),
            args=(ValueRef(value.ref),),
            kwargs={},
            dependencies=[value.ref],
            branch_condition=branch_condition,
            branch_value=branch_value,
            module_name=f".{method}()",
        )
        self.nodes[node_id] = node

        return Value(kind=ValueKind.STRUCTURED, payload=None, ref=node_id)

    def _collect_output_ids(self, output: Any) -> list[str]:
        """Extract node IDs from the output structure.

        Recursively traverses the output to find all Value objects
        and collect their node IDs/refs. Supports nested structures including
        dictionaries, lists, and tuples.

        Args:
            output: The output from forward(), which may be a Value,
                a collection containing Values, or a literal value.

        Returns:
            A list of node IDs representing the outputs of the traced graph.
            Returns an empty list if the output contains no Values.
        """
        if isinstance(output, Value):
            # Use collect_refs for Values - handles nested structures too
            refs = collect_refs(output)
            return refs
        elif isinstance(output, dict):
            ids: list[str] = []
            for value in output.values():
                ids.extend(self._collect_output_ids(value))
            return ids
        elif isinstance(output, (list, tuple)):
            ids = []
            for item in output:
                ids.extend(self._collect_output_ids(item))
            return ids
        else:
            return []

    def _capture_output_structure(
        self, output: Any
    ) -> str | dict[str, Any] | list[Any] | None:
        """Capture the original structure of forward() output with node IDs.

        Preserves the structure of the output (dict keys, list order) while
        replacing Value objects with their node IDs/refs. This allows
        reconstruction of results with user-defined keys after execution.

        Args:
            output: The output from forward(), which may be a Value,
                a collection containing Values, or a literal value.

        Returns:
            The output structure with Value objects replaced by node IDs.
            - For a single Value: returns the ref string
            - For a dict: returns dict with same keys, node_ids/refs as values
            - For a list/tuple: returns list with node_ids/refs
            - For non-Value values: returns None

        """
        if isinstance(output, Value):
            return output.ref
        elif isinstance(output, dict):
            result: dict[str, Any] = {}
            for key, value in output.items():
                captured = self._capture_output_structure(value)
                if captured is not None:
                    result[key] = captured
            return result if result else None
        elif isinstance(output, (list, tuple)):
            result_list: list[Any] = []
            for item in output:
                captured = self._capture_output_structure(item)
                if captured is not None:
                    result_list.append(captured)
            return result_list if result_list else None
        else:
            return None

    def trace(
        self,
        module: Module,
        *args: Any,
        **kwargs: Any,
    ) -> InferenceGraph:
        """Trace a module's forward() using Value-driven capture.

        This is an alias for trace_values() to preserve API compatibility.
        """
        return self.trace_values(module, *args, **kwargs)

    def trace_values(
        self,
        module: Module,
        *args: Any,
        **kwargs: Any,
    ) -> InferenceGraph:
        """Trace a module's forward() using Value-driven capture.

        This method uses the Value-based tracing approach where:
        - Inputs are wrapped as Values with refs pointing to input nodes
        - Dependencies are discovered via Value.ref (using collect_refs)
        - Module calls return Values with refs pointing to graph nodes

        Args:
            module: The Module to trace.
            *args: Positional arguments to pass to forward(). Each argument
                is converted to a Value and bound with an input ref.
            **kwargs: Keyword arguments to pass to forward(). Each kwarg
                is converted to a Value and bound with an input ref.

        Returns:
            An InferenceGraph containing all traced nodes, input IDs,
            output IDs, and parameters from the module tree.

        Note:
            This method resets the tracer state before tracing, so each
            call produces an independent graph. The trace context is
            automatically cleaned up when tracing completes.

        Example:
            >>> from plait.module import Module, LLMInference
            >>>
            >>> class SimplePipeline(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.llm = LLMInference(alias="fast")
            ...
            ...     def forward(self, text: Value) -> Value:
            ...         return self.llm(text)
            >>>
            >>> tracer = Tracer()
            >>> graph = tracer.trace_values(SimplePipeline(), "input text")
            >>> len(graph.nodes)  # 1 input + 1 LLM call = 2
            2
            >>> graph.input_ids
            ['input:input_0']
        """
        # Reset to ensure clean state
        self.reset()

        with trace_context(self):
            # Convert args and kwargs to Values
            input_values = valueify(args)
            kwarg_values = valueify(kwargs)

            # Bind inputs to create input nodes and assign refs
            bound_args = self.bind_inputs(input_values, prefix="input")
            bound_kwargs = self.bind_inputs(kwarg_values, prefix="input")

            # Unpack bound args if it's a tuple/list
            if isinstance(bound_args, (list, tuple)):
                forward_args = tuple(bound_args)
            else:
                forward_args = (bound_args,)

            # Unpack bound kwargs if it's a dict
            if isinstance(bound_kwargs, dict):
                forward_kwargs = bound_kwargs
            else:
                forward_kwargs = {}

            # Execute forward with bound Values
            output = module.forward(*forward_args, **forward_kwargs)

            # Collect output node IDs from Values
            self.output_ids = self._collect_output_ids(output)
            output_structure = self._capture_output_structure(output)

        return InferenceGraph(
            nodes=dict(self.nodes),
            input_ids=list(self.input_ids),
            output_ids=list(self.output_ids),
            output_structure=output_structure,
            parameters=dict(module.named_parameters()),
        )
