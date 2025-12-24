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
    >>> graph = tracer.trace(Pipeline(), "input text")
    >>> graph.input_ids
    ['input:input_0']
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from inf_engine.graph import GraphNode, InferenceGraph
from inf_engine.tracing.context import trace_context
from inf_engine.tracing.proxy import Proxy

if TYPE_CHECKING:
    from inf_engine.module import InferenceModule


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
        >>> proxy = tracer._create_input_node("text", "input text")
        >>> input_node = tracer.nodes[proxy.node_id].module
        >>> input_node.value
        'input text'
    """

    value: Any


@dataclass
class GetItemOp:
    """Operation node representing dictionary/list indexing.

    Created when a Proxy is indexed with proxy[key]. During execution,
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
    """Operation node representing iteration over a proxy.

    Created when a Proxy is iterated. During execution, this operation
    yields elements from the source node's output.

    Example:
        >>> op = IterOp()
    """

    pass


@dataclass
class MethodOp:
    """Operation node representing a method call on a proxy.

    Created when methods like keys(), values(), or items() are called
    on a Proxy. During execution, this operation calls the specified
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

    def _create_input_node(self, name: str, value: Any) -> Proxy:
        """Create a node representing an input to the traced graph.

        Creates an InputNode that wraps the given value and registers it
        in the tracer's node storage. The node ID is added to input_ids
        to mark it as a graph entry point.

        Args:
            name: A descriptive name for this input (e.g., "input_0", "text").
            value: The actual input value to capture.

        Returns:
            A Proxy representing this input node. The proxy can be passed
            to other modules to create dependency edges.

        Example:
            >>> tracer = Tracer()
            >>> proxy = tracer._create_input_node("text", "Hello, world!")
            >>> proxy.node_id
            'input:text'
            >>> tracer.input_ids
            ['input:text']
            >>> tracer.nodes['input:text'].module.value
            'Hello, world!'
        """
        node_id = f"input:{name}"
        self.input_ids.append(node_id)

        node = GraphNode(
            id=node_id,
            module=InputNode(value),
            args=(),
            kwargs={},
            dependencies=[],
            module_name=f"Input({name})",
        )
        self.nodes[node_id] = node

        return Proxy(node_id=node_id, tracer=self)

    def record_call(
        self,
        module: InferenceModule,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Proxy:
        """Record a module invocation during tracing.

        Called by InferenceModule.__call__ when tracing is active. Creates
        a graph node representing the module call and tracks dependencies
        based on any Proxy objects in the arguments.

        Args:
            module: The module being called.
            args: Positional arguments passed to the module.
            kwargs: Keyword arguments passed to the module.

        Returns:
            A Proxy representing the eventual output of this call.

        Example:
            >>> tracer = Tracer()
            >>> module = LLMInference(alias="test")
            >>> input_proxy = tracer._create_input_node("text", "hello")
            >>> output_proxy = tracer.record_call(module, (input_proxy,), {})
            >>> output_proxy.node_id
            'LLMInference_1'
            >>> tracer.nodes['LLMInference_1'].dependencies
            ['input:text']

        Note:
            This method mutates the tracer's internal node registry by adding
            a new GraphNode for the module invocation.
        """
        node_id = self._generate_id(module)

        # Extract dependencies from proxy arguments
        dependencies: list[str] = []
        processed_args: list[str | Any] = []
        for arg in args:
            if isinstance(arg, Proxy):
                dependencies.append(arg.node_id)
                processed_args.append(arg.node_id)
            else:
                processed_args.append(arg)

        processed_kwargs: dict[str, str | Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, Proxy):
                dependencies.append(value.node_id)
                processed_kwargs[key] = value.node_id
            else:
                processed_kwargs[key] = value

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

        return Proxy(node_id=node_id, tracer=self)

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

        Example:
            >>> tracer = Tracer()
            >>> input_proxy = tracer._create_input_node("data", {"key": "value"})
            >>> result = tracer.record_getitem(input_proxy, "key")
            >>> result.node_id
            'getitem_1'
            >>> tracer.nodes['getitem_1'].dependencies
            ['input:data']
        """
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
            args=(proxy.node_id,),
            kwargs={},
            dependencies=[proxy.node_id],
            branch_condition=branch_condition,
            branch_value=branch_value,
            module_name=f"getitem[{key!r}]",
        )
        self.nodes[node_id] = node

        return Proxy(node_id=node_id, tracer=self)

    def record_iter(self, proxy: Proxy) -> Proxy:
        """Record an iteration operation on a proxy.

        Creates a new graph node representing iteration over the proxy's
        value. This enables tracing of iteration patterns like for loops.

        Args:
            proxy: The proxy being iterated.

        Returns:
            A new Proxy representing the iterator.

        Example:
            >>> tracer = Tracer()
            >>> input_proxy = tracer._create_input_node("data", [1, 2, 3])
            >>> result = tracer.record_iter(input_proxy)
            >>> result.node_id
            'iter_1'
        """
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
            args=(proxy.node_id,),
            kwargs={},
            dependencies=[proxy.node_id],
            branch_condition=branch_condition,
            branch_value=branch_value,
            module_name="iter",
        )
        self.nodes[node_id] = node

        return Proxy(node_id=node_id, tracer=self)

    def record_method(self, proxy: Proxy, method: str) -> Proxy:
        """Record a method call on a proxy.

        Creates a new graph node representing a method call like
        keys(), values(), or items() on the proxy's value.

        Args:
            proxy: The proxy on which the method is called.
            method: The name of the method being called.

        Returns:
            A new Proxy representing the result of the method call.

        Example:
            >>> tracer = Tracer()
            >>> input_proxy = tracer._create_input_node("data", {"a": 1})
            >>> result = tracer.record_method(input_proxy, "keys")
            >>> result.node_id
            'method_1'
            >>> tracer.nodes['method_1'].module.method
            'keys'
        """
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
            args=(proxy.node_id,),
            kwargs={},
            dependencies=[proxy.node_id],
            branch_condition=branch_condition,
            branch_value=branch_value,
            module_name=f".{method}()",
        )
        self.nodes[node_id] = node

        return Proxy(node_id=node_id, tracer=self)

    def _collect_output_ids(self, output: Any) -> list[str]:
        """Extract node IDs from the output structure.

        Recursively traverses the output to find all Proxy objects and
        collect their node IDs. Supports nested structures including
        dictionaries, lists, and tuples.

        Args:
            output: The output from forward(), which may be a Proxy,
                a collection containing Proxies, or a literal value.

        Returns:
            A list of node IDs representing the outputs of the traced graph.
            Returns an empty list if the output contains no Proxy objects.

        Example:
            >>> tracer = Tracer()
            >>> proxy1 = tracer._create_input_node("a", "val1")
            >>> proxy2 = tracer._create_input_node("b", "val2")
            >>>
            >>> # Single proxy output
            >>> tracer._collect_output_ids(proxy1)
            ['input:a']
            >>>
            >>> # Dict output
            >>> tracer._collect_output_ids({"x": proxy1, "y": proxy2})
            ['input:a', 'input:b']
            >>>
            >>> # List output
            >>> tracer._collect_output_ids([proxy1, proxy2])
            ['input:a', 'input:b']
            >>>
            >>> # Literal output (no proxies)
            >>> tracer._collect_output_ids("literal string")
            []
        """
        if isinstance(output, Proxy):
            return [output.node_id]
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
        replacing Proxy objects with their node IDs. This allows reconstruction
        of results with user-defined keys after execution.

        Args:
            output: The output from forward(), which may be a Proxy,
                a collection containing Proxies, or a literal value.

        Returns:
            The output structure with Proxy objects replaced by node IDs.
            - For a single Proxy: returns the node_id string
            - For a dict: returns dict with same keys, node_ids as values
            - For a list/tuple: returns list with node_ids
            - For non-Proxy values: returns None

        Example:
            >>> tracer = Tracer()
            >>> proxy1 = tracer._create_input_node("a", "val1")
            >>> proxy2 = tracer._create_input_node("b", "val2")
            >>>
            >>> # Single proxy output
            >>> tracer._capture_output_structure(proxy1)
            'input:a'
            >>>
            >>> # Dict output preserves keys
            >>> tracer._capture_output_structure({"summary": proxy1, "analysis": proxy2})
            {'summary': 'input:a', 'analysis': 'input:b'}
            >>>
            >>> # List output preserves order
            >>> tracer._capture_output_structure([proxy1, proxy2])
            ['input:a', 'input:b']
        """
        if isinstance(output, Proxy):
            return output.node_id
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
        module: InferenceModule,
        *args: Any,
        **kwargs: Any,
    ) -> InferenceGraph:
        """Trace a module's forward() and return the captured graph.

        Executes the module's forward() method with Proxy objects representing
        the inputs, capturing all module invocations into an InferenceGraph.
        The trace context is set so that nested module calls are recorded.

        Args:
            module: The InferenceModule to trace.
            *args: Positional arguments to pass to forward(). Each argument
                becomes an input node in the graph.
            **kwargs: Keyword arguments to pass to forward(). Each kwarg
                becomes an input node in the graph.

        Returns:
            An InferenceGraph containing all traced nodes, input IDs,
            output IDs, and parameters from the module tree.

        Note:
            This method resets the tracer state before tracing, so each
            call produces an independent graph. The trace context is
            automatically cleaned up when tracing completes.

        Example:
            >>> from inf_engine.module import InferenceModule, LLMInference
            >>>
            >>> class SimplePipeline(InferenceModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.llm = LLMInference(alias="fast")
            ...
            ...     def forward(self, text: str) -> str:
            ...         return self.llm(text)
            >>>
            >>> tracer = Tracer()
            >>> graph = tracer.trace(SimplePipeline(), "input text")
            >>> len(graph.nodes)  # 1 input + 1 LLM call = 2
            2
            >>> graph.input_ids
            ['input:input_0']
            >>> "LLMInference_1" in graph.output_ids
            True
        """
        # Reset to ensure clean state
        self.reset()

        with trace_context(self):
            # Create input proxies for positional arguments
            input_proxies: list[Proxy] = []
            for i, arg in enumerate(args):
                proxy = self._create_input_node(f"input_{i}", arg)
                input_proxies.append(proxy)

            # Create input proxies for keyword arguments
            kwarg_proxies: dict[str, Proxy] = {}
            for key, value in kwargs.items():
                proxy = self._create_input_node(f"input_{key}", value)
                kwarg_proxies[key] = proxy

            # Execute forward with proxies
            output = module.forward(*input_proxies, **kwarg_proxies)

            # Collect output node IDs and preserve structure
            self.output_ids = self._collect_output_ids(output)
            output_structure = self._capture_output_structure(output)

        return InferenceGraph(
            nodes=dict(self.nodes),
            input_ids=list(self.input_ids),
            output_ids=list(self.output_ids),
            output_structure=output_structure,
            parameters=dict(module.named_parameters()),
        )
