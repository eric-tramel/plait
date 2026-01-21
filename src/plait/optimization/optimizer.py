"""Optimizer classes for parameter updates via LLM.

This module provides the optimizer infrastructure for aggregating feedback
and updating parameters, following the PyTorch optimizer pattern.

The core workflow mirrors torch.optim:
    1. Initialize optimizer with parameters: `optimizer = SFAOptimizer(module.parameters())`
    2. Bind to resources: `optimizer.bind(resources)`
    3. Clear feedback: `optimizer.zero_feedback()`
    4. Accumulate feedback via backward passes
    5. Update parameters: `await optimizer.step()`

Example:
    >>> from plait.optimization import SFAOptimizer
    >>>
    >>> # Create optimizer with module parameters
    >>> optimizer = SFAOptimizer(
    ...     module.parameters(),
    ...     conservatism=0.7,
    ... )
    >>> optimizer.bind(resources)
    >>>
    >>> # Training loop
    >>> optimizer.zero_feedback()
    >>> step = TrainingStep(module, loss_fn)
    >>> step.train()
    >>> for example in batch:
    ...     loss_val = await step(example["input"], target=example["target"])
    ...     await loss_val.backward()
    >>> updates = await optimizer.step()
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Self

from plait.errors import OptimizationError

if TYPE_CHECKING:
    from plait.graph import InferenceGraph
    from plait.module import Module
    from plait.optimization.record import ForwardRecord
    from plait.parameter import Parameter
    from plait.resources.config import ResourceConfig
    from plait.resources.manager import ResourceManager

logger = logging.getLogger(__name__)

_active_optimizer: ContextVar[Optimizer | None] = ContextVar(
    "plait_active_optimizer",
    default=None,
)
_default_optimizer: Optimizer | None = None


def get_active_optimizer() -> Optimizer | None:
    """Return the active optimizer for backward passes, if set."""
    active = _active_optimizer.get()
    return active if active is not None else _default_optimizer


@contextmanager
def use_optimizer(optimizer: Optimizer) -> Any:
    """Temporarily set the active optimizer for backward passes."""
    token = _active_optimizer.set(optimizer)
    try:
        yield optimizer
    finally:
        _active_optimizer.reset(token)


class _OptimizerLLMWrapper:
    """Wrapper to make optimizer's LLMs callable as bound modules.

    LLMInference modules cannot be traced directly because they are atomic
    (no child modules). This wrapper creates a minimal composite module
    that can be traced and executed.
    """

    def __init__(self, alias: str, system_prompt: str) -> None:
        """Initialize the wrapper with LLM configuration.

        Args:
            alias: Resource alias for the LLM endpoint.
            system_prompt: System prompt for the LLM.
        """
        from plait.module import LLMInference, Module

        # Create a wrapper module class dynamically
        class _Wrapper(Module):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.llm = LLMInference(alias=alias, system_prompt=system_prompt)

            def forward(inner_self, prompt: str) -> str:
                return inner_self.llm(prompt)

        self._module = _Wrapper()
        self._bound = False

    def bind(self, resources: ResourceConfig | ResourceManager) -> None:
        """Bind the wrapper module to resources."""
        self._module.bind(resources)
        self._bound = True

    async def __call__(self, prompt: str) -> str:
        """Execute the LLM with the given prompt."""
        if not self._bound:
            raise RuntimeError("LLM wrapper not bound. Call bind() first.")
        return await self._module(prompt)


class Optimizer(ABC):
    """Base optimizer for parameter updates via LLM.

    Follows torch.optim pattern: initialized with parameters,
    accumulates feedback across backward() calls, updates on step().

    Optimizers use internal LLMInference modules for aggregation and
    update generation. These use fixed aliases that must be configured
    in ResourceConfig:
    - "optimizer/aggregator": Synthesizes multiple feedback items
    - "optimizer/updater": Generates improved parameter values
    - "optimizer/reasoning": Optional LLM for backward-pass reasoning

    Attributes:
        AGGREGATOR_ALIAS: Fixed alias for the feedback aggregator LLM.
        UPDATER_ALIAS: Fixed alias for the parameter updater LLM.
        REASONING_ALIAS: Fixed alias for the optional reasoning LLM.
        params: List of Parameters being optimized.
        aggregator: Internal LLM for aggregating feedback.
        updater: Internal LLM for generating parameter updates.
        reasoning_llm: Optional LLM for backward-pass reasoning.

    Example:
        >>> class MyOptimizer(Optimizer):
        ...     async def step(self) -> dict[str, str]:
        ...         # Custom update logic
        ...         updates = {}
        ...         for param in self.params:
        ...             if param._feedback_buffer:
        ...                 new_value = await self._compute_update(param)
        ...                 param.apply_update(new_value)
        ...                 updates[param._name or str(id(param))] = new_value
        ...         return updates

    Note:
        The ResourceConfig must include the optimizer aliases. Example:
        ```python
        resources = ResourceConfig(endpoints={
            "optimizer/aggregator": EndpointConfig(...),
            "optimizer/updater": EndpointConfig(...),
        })
        ```
    """

    # Fixed aliases for optimizer's internal LLMs
    AGGREGATOR_ALIAS = "optimizer/aggregator"
    UPDATER_ALIAS = "optimizer/updater"
    REASONING_ALIAS = "optimizer/reasoning"

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        reasoning_model: str | None = None,
    ) -> None:
        """Initialize the optimizer with parameters to optimize.

        Args:
            params: Parameters to optimize (e.g., module.parameters()).
                These are stored as a list for repeated iteration.
            reasoning_model: Optional model identifier for backward-pass
                reasoning. If provided, optimizer.reasoning_llm is available
                for custom backward implementations.

        Example:
            >>> optimizer = MyOptimizer(
            ...     module.parameters(),
            ...     reasoning_model="gpt-4o",
            ... )
        """
        self.params = list(params)
        self._param_key_map: dict[str, str] = {}
        used_keys: set[str] = set()
        for param in self.params:
            base = param._get_hierarchical_name() or param._id
            key = base
            if key in used_keys:
                key = f"{base}:{param._id}"
            used_keys.add(key)
            self._param_key_map[param._id] = key
        self._step_count = 0
        self._records: list[ForwardRecord] = []

        # Internal LLMs with fixed aliases (wrapped for execution)
        self.aggregator = _OptimizerLLMWrapper(
            alias=self.AGGREGATOR_ALIAS,
            system_prompt=self._aggregator_system_prompt(),
        )
        self.updater = _OptimizerLLMWrapper(
            alias=self.UPDATER_ALIAS,
            system_prompt=self._updater_system_prompt(),
        )
        self.reasoning_llm: _OptimizerLLMWrapper | None = None
        if reasoning_model:
            self.reasoning_llm = _OptimizerLLMWrapper(
                alias=self.REASONING_ALIAS,
                system_prompt=self._reasoning_system_prompt(),
            )

        self._bound = False
        global _default_optimizer
        _default_optimizer = self
        _active_optimizer.set(self)

    def bind(self, resources: ResourceConfig | ResourceManager) -> Self:
        """Bind optimizer's internal LLMs to resources.

        The ResourceConfig must include the optimizer aliases:
        - "optimizer/aggregator": Required for aggregating feedback
        - "optimizer/updater": Required for generating updates
        - "optimizer/reasoning": Required only if reasoning_model was specified

        Args:
            resources: ResourceConfig or ResourceManager containing the
                optimizer endpoint configurations.

        Returns:
            Self for method chaining.

        Raises:
            KeyError: If required optimizer aliases are not in resources.

        Example:
            >>> resources = ResourceConfig(endpoints={
            ...     "optimizer/aggregator": EndpointConfig(model="gpt-4o"),
            ...     "optimizer/updater": EndpointConfig(model="gpt-4o"),
            ... })
            >>> optimizer = SFAOptimizer(module.parameters()).bind(resources)
        """
        self.aggregator.bind(resources)
        self.updater.bind(resources)
        if self.reasoning_llm:
            self.reasoning_llm.bind(resources)
        self._bound = True
        global _default_optimizer
        _default_optimizer = self
        _active_optimizer.set(self)
        return self

    def activate(self) -> Any:
        """Return a context manager that sets this optimizer as active."""
        return use_optimizer(self)

    def zero_feedback(self) -> None:
        """Clear all parameter feedback buffers and accumulated records.

        Like torch.optim.Optimizer.zero_grad(), this clears accumulated
        feedback and ForwardRecords to prepare for a new mini-batch. Should
        be called at the beginning of each mini-batch iteration.

        This resets all state for a new training batch:
        - Clears feedback buffers on all parameters
        - Clears accumulated ForwardRecords captured during backward passes

        Example:
            >>> for batch in batches:
            ...     optimizer.zero_feedback()
            ...     for example in batch:
            ...         # Forward, loss, backward
            ...         await loss_val.backward()
            ...     await optimizer.step()
        """
        global _default_optimizer
        _default_optimizer = self
        for param in self.params:
            param.zero_feedback()
        self._records.clear()

    def capture_record(self, record: ForwardRecord) -> None:
        """Capture a ForwardRecord during backward pass.

        Called by _propagate_backward_value() to provide graph context for
        ordered parameter updates in step(). The captured records are
        used to determine topological ordering and upstream dependencies.

        Args:
            record: The ForwardRecord from the forward pass.

        Example:
            >>> # Called internally by backward propagation
            >>> optimizer.capture_record(record)
        """
        self._records.append(record)

    @abstractmethod
    async def step(self) -> dict[str, str]:
        """Aggregate accumulated feedback and update parameters.

        Should be called after accumulating feedback from a mini-batch
        of examples via Value.backward(). This method processes all
        accumulated feedback and generates updated parameter values.

        Parameters are updated in forward topological order based on the
        captured ForwardRecords. This ensures upstream parameters are updated
        before downstream ones, allowing downstream updates to maintain
        consistency with upstream changes.

        Returns:
            Dictionary mapping parameter names to their new values.
            Only includes parameters that were actually updated.

        Raises:
            RuntimeError: If optimizer is not bound to resources.
            RuntimeError: If no ForwardRecords have been captured.

        Example:
            >>> updates = await optimizer.step()
            >>> for name, new_value in updates.items():
            ...     print(f"{name}: {new_value[:50]}...")
        """
        pass

    def _param_key(self, param: Parameter) -> str:
        """Get a consistent key for a parameter.

        Args:
            param: The parameter to get a key for.

        Returns:
            A string key identifying this parameter.
        """
        key = self._param_key_map.get(param._id)
        if key:
            return key
        return param._id

    def _aggregator_system_prompt(self) -> str:
        """System prompt for the feedback aggregator LLM.

        Returns:
            System prompt string for aggregating multiple feedback items.
        """
        return (
            "You synthesize multiple feedback items into a coherent summary. "
            "Identify common themes, prioritize impactful suggestions, and "
            "resolve any conflicting feedback."
        )

    def _updater_system_prompt(self) -> str:
        """System prompt for the parameter updater LLM.

        Returns:
            System prompt string for generating improved parameter values.
        """
        return (
            "You improve text parameters based on feedback. "
            "Make targeted changes that address the feedback while "
            "preserving aspects that work well."
        )

    def _reasoning_system_prompt(self) -> str:
        """System prompt for the reasoning LLM.

        Returns:
            System prompt string for backward-pass reasoning.
        """
        return (
            "You analyze why outputs received certain feedback and "
            "suggest specific parameter improvements."
        )


class SFAOptimizer(Optimizer):
    """Stochastic Feedback Ascent optimizer.

    Makes small, targeted changes based on accumulated feedback rather than
    aggressive rewrites. Good for fine-tuning working prompts.

    The conservatism parameter controls how aggressive updates are:
    - 0.0: Aggressive, may significantly rewrite parameters
    - 1.0: Very conservative, minimal changes only

    The algorithm:
    1. Capture ForwardRecords during backward passes
    2. Compute topological levels for parameters
    3. Process levels sequentially (params within a level in parallel)
    4. Each update receives context about upstream parameter changes

    Attributes:
        conservatism: How conservative updates should be (0-1).
        max_retries: Maximum retry attempts for failed update generation.

    Example:
        >>> optimizer = SFAOptimizer(
        ...     module.parameters(),
        ...     conservatism=0.7,
        ... )
        >>> optimizer.bind(resources)
        >>>
        >>> # Training loop
        >>> for batch in batches:
        ...     optimizer.zero_feedback()
        ...     for example in batch:
        ...         output, record = await run(module, example["input"], record=True)
        ...         loss_val = await loss_fn(output, example["target"])
    ...         await loss_val.backward()
        ...     updates = await optimizer.step()
        ...     print(f"Updated {len(updates)} parameters")

    Note:
        Higher conservatism values result in smaller, more targeted changes.
        Start with conservatism=0.7 and adjust based on results.
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        conservatism: float = 0.7,
        max_retries: int = 3,
        reasoning_model: str | None = None,
    ) -> None:
        """Initialize the SFAOptimizer.

        Args:
            params: Parameters to optimize (e.g., module.parameters()).
            conservatism: How conservative updates should be (0-1).
                Higher values result in smaller changes. Defaults to 0.7.
            max_retries: Maximum number of retry attempts when the updater
                LLM returns an empty or invalid response. Defaults to 3.
            reasoning_model: Optional model identifier for backward-pass
                reasoning.

        Raises:
            ValueError: If conservatism is not in [0, 1] range.
            ValueError: If max_retries is negative.

        Example:
            >>> optimizer = SFAOptimizer(
            ...     module.parameters(),
            ...     conservatism=0.5,  # Moderate changes
            ...     max_retries=3,
            ... )
        """
        if not 0.0 <= conservatism <= 1.0:
            raise ValueError(f"conservatism must be in [0, 1], got {conservatism}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")

        super().__init__(params, reasoning_model=reasoning_model)
        self.conservatism = conservatism
        self.max_retries = max_retries

    async def step(self) -> dict[str, str]:
        """Update parameters in topological order with upstream visibility.

        The algorithm:
        1. Build parameter-to-node mapping from captured records
        2. Compute update levels (topological partitioning)
        3. Process levels sequentially, parameters within level in parallel
        4. Each update receives context of upstream parameter changes

        Returns:
            Dictionary mapping parameter names to new values.

        Raises:
            RuntimeError: If optimizer is not bound to resources.
            RuntimeError: If no ForwardRecords have been captured.
        """
        if not self._bound:
            raise RuntimeError("Optimizer not bound. Call bind(resources) first.")

        if not self._records:
            raise RuntimeError(
                "No ForwardRecords captured. Ensure module.train() is called "
                "and backward() is invoked before step()."
            )

        # Snapshot current values before any updates
        previous_values = {self._param_key(param): param.value for param in self.params}

        # Build parameter dependency graph across all records (tapes)
        param_nodes, edges, reverse_edges, param_topo_index = self._build_param_dag(
            self._records
        )

        # Restrict to parameters managed by this optimizer
        allowed_keys = {self._param_key(param) for param in self.params}
        param_nodes = {k: v for k, v in param_nodes.items() if k in allowed_keys}
        edges = {
            k: {dst for dst in dests if dst in allowed_keys}
            for k, dests in edges.items()
            if k in allowed_keys
        }
        reverse_edges = {
            k: {src for src in srcs if src in allowed_keys}
            for k, srcs in reverse_edges.items()
            if k in allowed_keys
        }
        param_topo_index = {
            k: v for k, v in param_topo_index.items() if k in allowed_keys
        }

        # Ensure all optimizer params are present in the graph (even if isolated)
        for param in self.params:
            key = self._param_key(param)
            param_nodes.setdefault(key, param)
            edges.setdefault(key, set())
            reverse_edges.setdefault(key, set())
            param_topo_index.setdefault(key, 0)

        # Compute update levels from parameter dependency graph
        param_order = {self._param_key(p): i for i, p in enumerate(self.params)}
        levels = self._compute_param_levels(
            param_nodes,
            edges,
            param_topo_index,
            param_order,
        )

        # Track applied updates for visibility
        applied_updates: dict[str, str] = {}

        # Process levels sequentially
        for level in levels:
            # Filter to params that need updating
            params_to_update = [
                p for p in level if p.requires_grad and p._feedback_buffer
            ]

            if not params_to_update:
                continue

            # Update all params in this level in parallel
            level_results = await asyncio.gather(
                *[
                    self._update_parameter_with_context(
                        param=param,
                        previous_values=previous_values,
                        applied_updates=applied_updates,
                        upstream_params=self._get_upstream_params_from_graph(
                            param, reverse_edges, param_nodes
                        ),
                    )
                    for param in params_to_update
                ]
            )

            # Apply updates and record them
            for param, new_value in level_results:
                param.apply_update(new_value)
                key = self._param_key(param)
                applied_updates[key] = new_value

        self._step_count += 1
        return applied_updates

    def _build_param_to_node_mapping(
        self,
        module_map: dict[str, Module],
    ) -> dict[str, set[str]]:
        """Map parameter keys to the nodes containing them.

        Args:
            module_map: Mapping from node_id to module instance.

        Returns:
            Dict mapping param key to set of node IDs where it appears.
        """
        mapping: dict[str, set[str]] = {}

        for node_id, module in module_map.items():
            for _name, param in module.named_parameters():
                key = self._param_key(param)
                mapping.setdefault(key, set()).add(node_id)

        return mapping

    def _build_node_to_param_mapping(
        self,
        module_map: dict[str, Module],
    ) -> dict[str, list[Parameter]]:
        """Map node IDs to parameters in that node's module.

        Args:
            module_map: Mapping from node_id to module instance.

        Returns:
            Dict mapping node_id to list of Parameters in that module.
        """
        mapping: dict[str, list[Parameter]] = {}

        for node_id, module in module_map.items():
            params = list(module.parameters())
            if params:
                mapping[node_id] = params

        return mapping

    def _compute_update_levels(
        self,
        graph: InferenceGraph,
        param_to_nodes: dict[str, set[str]],
    ) -> list[list[Parameter]]:
        """Partition parameters into topological levels.

        Level 0 contains parameters with no upstream parameters.
        Level N contains parameters whose upstream params are all in levels < N.

        Args:
            graph: The inference graph.
            param_to_nodes: Mapping from param key to node IDs containing it.

        Returns:
            List of levels, where each level is a list of Parameters
            that can be updated in parallel.
        """
        # Get topological order of nodes
        topo_order = graph.topological_order()
        node_to_topo_index = {nid: i for i, nid in enumerate(topo_order)}

        # Compute "earliest" topological index for each parameter
        # (a param might appear in multiple nodes; use the earliest)
        param_topo_index: dict[str, int] = {}
        for param in self.params:
            key = self._param_key(param)
            nodes = param_to_nodes.get(key, set())
            if nodes:
                param_topo_index[key] = min(node_to_topo_index[n] for n in nodes)
            else:
                # Parameter not in graph (e.g., not used in this forward pass)
                param_topo_index[key] = -1

        # Group parameters by their topological index
        index_to_params: dict[int, list[Parameter]] = {}
        for param in self.params:
            key = self._param_key(param)
            idx = param_topo_index[key]
            if idx >= 0:  # Skip params not in graph
                index_to_params.setdefault(idx, []).append(param)

        # Convert to ordered list of levels
        levels = [index_to_params[idx] for idx in sorted(index_to_params.keys())]

        return levels

    def _record_node_parameters(
        self,
        record: ForwardRecord,
    ) -> dict[str, list[Parameter]]:
        """Return direct parameters used by each node in a record."""
        if record.node_parameters:
            return record.node_parameters

        node_parameters: dict[str, list[Parameter]] = {}
        for node_id, module in record.module_map.items():
            direct = getattr(module, "direct_parameters", None)
            if callable(direct):
                params = list(direct())
            else:
                params = list(module.parameters())
            if params:
                node_parameters[node_id] = params
        return node_parameters

    def _build_param_dag(
        self,
        records: list[ForwardRecord],
    ) -> tuple[
        dict[str, Parameter],
        dict[str, set[str]],
        dict[str, set[str]],
        dict[str, int],
    ]:
        """Build a parameter dependency graph across all records."""
        param_nodes: dict[str, Parameter] = {}
        edges: dict[str, set[str]] = {}
        reverse_edges: dict[str, set[str]] = {}
        param_topo_index: dict[str, int] = {}

        for record in records:
            node_parameters = self._record_node_parameters(record)
            if not node_parameters:
                continue

            topo_order = record.graph.topological_order()
            node_to_topo_index = {nid: i for i, nid in enumerate(topo_order)}

            for node_id, params in node_parameters.items():
                idx = node_to_topo_index.get(node_id)
                if idx is None:
                    continue
                for param in params:
                    key = self._param_key(param)
                    param_nodes[key] = param
                    if key not in param_topo_index or idx < param_topo_index[key]:
                        param_topo_index[key] = idx

            for node_id, params in node_parameters.items():
                if not params:
                    continue
                ancestor_nodes = record.graph.ancestors(node_id)
                if not ancestor_nodes:
                    continue
                for anc_id in ancestor_nodes:
                    for anc_param in node_parameters.get(anc_id, []):
                        anc_key = self._param_key(anc_param)
                        for param in params:
                            key = self._param_key(param)
                            if anc_key == key:
                                continue
                            edges.setdefault(anc_key, set()).add(key)
                            reverse_edges.setdefault(key, set()).add(anc_key)

        for key in param_nodes:
            edges.setdefault(key, set())
            reverse_edges.setdefault(key, set())
            param_topo_index.setdefault(key, 0)

        return param_nodes, edges, reverse_edges, param_topo_index

    def _compute_param_levels(
        self,
        param_nodes: dict[str, Parameter],
        edges: dict[str, set[str]],
        param_topo_index: dict[str, int],
        param_order: dict[str, int],
    ) -> list[list[Parameter]]:
        """Compute topological levels from a parameter dependency DAG."""
        indegree: dict[str, int] = dict.fromkeys(param_nodes, 0)
        for src, dests in edges.items():
            indegree.setdefault(src, 0)
            for dst in dests:
                indegree[dst] = indegree.get(dst, 0) + 1

        remaining = set(indegree.keys())
        levels: list[list[Parameter]] = []

        while remaining:
            zero_indegree = [k for k in remaining if indegree.get(k, 0) == 0]
            if not zero_indegree:
                # Cycle detected; fall back to stable ordering by topo index
                ordered = sorted(
                    remaining,
                    key=lambda k: (
                        param_topo_index.get(k, 0),
                        param_order.get(k, 0),
                    ),
                )
                levels.extend([[param_nodes[k]] for k in ordered])
                break

            zero_indegree = sorted(
                zero_indegree,
                key=lambda k: (
                    param_topo_index.get(k, 0),
                    param_order.get(k, 0),
                ),
            )
            levels.append([param_nodes[k] for k in zero_indegree])

            for key in zero_indegree:
                remaining.remove(key)
                for dst in edges.get(key, set()):
                    indegree[dst] -= 1

        return levels

    def _get_upstream_params_from_graph(
        self,
        param: Parameter,
        reverse_edges: dict[str, set[str]],
        param_nodes: dict[str, Parameter],
    ) -> list[Parameter]:
        """Get upstream parameters using the parameter dependency graph."""
        key = self._param_key(param)
        upstream_keys: set[str] = set()
        stack = list(reverse_edges.get(key, set()))

        while stack:
            current = stack.pop()
            if current in upstream_keys:
                continue
            upstream_keys.add(current)
            stack.extend(reverse_edges.get(current, set()))

        return [param_nodes[k] for k in upstream_keys if k in param_nodes]

    def _get_upstream_params(
        self,
        param: Parameter,
        param_to_nodes: dict[str, set[str]],
        node_to_params: dict[str, list[Parameter]],
        graph: InferenceGraph,
    ) -> list[Parameter]:
        """Get all parameters upstream of this parameter.

        Args:
            param: The parameter to find upstream params for.
            param_to_nodes: Mapping from param key to node IDs.
            node_to_params: Mapping from node ID to parameters.
            graph: The inference graph.

        Returns:
            List of Parameters that are upstream (ancestors) of this param.
        """
        key = self._param_key(param)
        nodes = param_to_nodes.get(key, set())

        if not nodes:
            return []

        # Get all ancestor nodes
        ancestor_nodes: set[str] = set()
        for node_id in nodes:
            ancestor_nodes.update(graph.ancestors(node_id))

        # Collect parameters from ancestor nodes
        upstream_params: list[Parameter] = []
        seen: set[str] = set()

        for node_id in ancestor_nodes:
            for p in node_to_params.get(node_id, []):
                p_key = self._param_key(p)
                if p_key not in seen and p_key != key:
                    seen.add(p_key)
                    upstream_params.append(p)

        return upstream_params

    def _validate_update(self, value: str) -> bool:
        """Validate that an update value is non-empty and meaningful.

        Args:
            value: The update value to validate.

        Returns:
            True if the value is valid, False otherwise.
        """
        if not value:
            return False
        if not value.strip():
            return False
        return True

    async def _update_parameter_with_context(
        self,
        param: Parameter,
        previous_values: dict[str, str],
        applied_updates: dict[str, str],
        upstream_params: list[Parameter],
    ) -> tuple[Parameter, str]:
        """Generate an update for a parameter with upstream visibility.

        Uses retry logic to handle empty or invalid LLM responses. If the
        updater returns an empty response, it will retry up to max_retries
        times before raising an OptimizationError.

        Args:
            param: The parameter to update.
            previous_values: Snapshot of all param values before step().
            applied_updates: Updates already applied in earlier levels.
            upstream_params: Parameters upstream of this one.

        Returns:
            Tuple of (parameter, new_value).

        Raises:
            OptimizationError: If no valid update could be generated after
                exhausting all retry attempts.
        """
        # Aggregate feedback for this parameter
        aggregated = await self._aggregate_feedback(param)

        # Build upstream context showing what changed
        upstream_context = self._build_upstream_context(
            upstream_params,
            previous_values,
            applied_updates,
        )

        # Generate update with retry logic
        param_name = self._param_key(param)
        attempts = 0
        max_attempts = self.max_retries + 1  # +1 for the initial attempt

        while attempts < max_attempts:
            attempts += 1

            new_value = await self._generate_update_with_context(
                param,
                aggregated,
                upstream_context,
            )

            if self._validate_update(new_value):
                if attempts > 1:
                    logger.info(
                        "Successfully generated update for parameter '%s' "
                        "on attempt %d",
                        param_name,
                        attempts,
                    )
                return (param, new_value)

            # Invalid response, log and potentially retry
            if attempts < max_attempts:
                logger.warning(
                    "Empty or invalid update for parameter '%s' "
                    "(attempt %d/%d), retrying...",
                    param_name,
                    attempts,
                    max_attempts,
                )
            else:
                logger.error(
                    "Failed to generate valid update for parameter '%s' "
                    "after %d attempts",
                    param_name,
                    attempts,
                )

        # Exhausted all retries
        raise OptimizationError(
            f"Failed to generate valid update for parameter '{param_name}' "
            f"after {attempts} attempts. The updater LLM returned empty or "
            "invalid responses.",
            parameter_name=param_name,
            attempts=attempts,
        )

    async def _aggregate_feedback(self, param: Parameter) -> str:
        """Combine all feedback items into one coherent summary.

        When multiple feedback items have accumulated (from fan-out
        within a graph and/or multiple training examples), this method
        synthesizes them into a single actionable summary.

        Args:
            param: The parameter whose feedback should be aggregated.

        Returns:
            Aggregated feedback summary as a string.
        """
        feedbacks = param._feedback_buffer

        if len(feedbacks) == 1:
            return feedbacks[0]

        feedback_items = "\n".join(
            f'<feedback index="{i + 1}">{fb}</feedback>'
            for i, fb in enumerate(feedbacks)
        )

        prompt = f"""<task>Aggregate feedback for a parameter</task>

<parameter name="{param._name}">
<description>{param.description}</description>
<current-value>{param.value}</current-value>
</parameter>

<feedback-items count="{len(feedbacks)}">
{feedback_items}
</feedback-items>

<instructions>
Synthesize the feedback items into a single coherent summary that:
- Identifies common themes across feedback items
- Prioritizes the most impactful suggestions
- Notes and resolves any conflicting feedback
- Provides specific, actionable recommendations
</instructions>

<output-format>
Provide a single summary paragraph. No bullet points, no numbered lists.
</output-format>"""

        return await self.aggregator(prompt)

    def _build_upstream_context(
        self,
        upstream_params: list[Parameter],
        previous_values: dict[str, str],
        applied_updates: dict[str, str],
    ) -> str:
        """Build context string showing upstream parameter changes.

        Args:
            upstream_params: Parameters that are upstream in the graph.
            previous_values: Values before any updates.
            applied_updates: Updates already applied.

        Returns:
            Formatted string describing upstream changes, or empty if none.
        """
        changed_params = []

        for param in upstream_params:
            key = self._param_key(param)
            if key in applied_updates:
                changed_params.append(
                    {
                        "name": param._name or key,
                        "description": param.description,
                        "previous": previous_values.get(key, ""),
                        "updated": applied_updates[key],
                    }
                )

        if not changed_params:
            return ""

        param_blocks = []
        for p in changed_params:
            param_blocks.append(
                f"""<upstream-parameter name="{p["name"]}">
<description>{p["description"]}</description>
<previous-value>{p["previous"]}</previous-value>
<updated-value>{p["updated"]}</updated-value>
</upstream-parameter>"""
            )

        return f"""<upstream-updates>
<note>
The following parameters appear earlier in the pipeline and have already
been updated in this optimization step. Your update should maintain
consistency with these changes.
</note>
{chr(10).join(param_blocks)}
</upstream-updates>"""

    async def _generate_update_with_context(
        self,
        param: Parameter,
        aggregated: str,
        upstream_context: str,
    ) -> str:
        """Generate improved parameter value with upstream context.

        Args:
            param: The parameter to update.
            aggregated: Aggregated feedback for this parameter.
            upstream_context: Context about upstream parameter changes.

        Returns:
            The new parameter value.
        """
        prompt = f"""<task>Update a parameter based on feedback</task>

<parameter name="{param._name}">
<description>{param.description}</description>
<current-value>{param.value}</current-value>
</parameter>

<aggregated-feedback>
{aggregated}
</aggregated-feedback>

{upstream_context}

<update-guidelines>
<conservatism>{self.conservatism:.1f}</conservatism>
<conservatism-scale>0 = aggressive rewrites, 1 = minimal changes</conservatism-scale>
<instructions>
- Address the key points in the feedback
- Preserve aspects that are working well
- Make changes proportional to the conservatism level
- Maintain consistency with any upstream parameter changes
</instructions>
</update-guidelines>

<output-format>
Return ONLY the new parameter value. No explanations, no markdown, no quotes.
</output-format>"""

        return await self.updater(prompt)
