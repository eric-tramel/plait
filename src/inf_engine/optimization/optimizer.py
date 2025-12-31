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
    >>> from inf_engine.optimization import SFAOptimizer
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
    >>> for example in batch:
    ...     output, record = await run(module, example["input"], record=True)
    ...     feedback = await loss_fn(output, example["target"], record=record)
    ...     await feedback.backward(optimizer=optimizer)
    >>> updates = await optimizer.step()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from inf_engine.parameter import Parameter
    from inf_engine.resources.config import ResourceConfig
    from inf_engine.resources.manager import ResourceManager


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
        from inf_engine.module import InferenceModule, LLMInference

        # Create a wrapper module class dynamically
        class _Wrapper(InferenceModule):
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
        self._step_count = 0

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
        return self

    def zero_feedback(self) -> None:
        """Clear all parameter feedback buffers.

        Like torch.optim.Optimizer.zero_grad(), this clears accumulated
        feedback to prepare for a new mini-batch. Should be called at
        the beginning of each mini-batch iteration.

        Example:
            >>> for batch in batches:
            ...     optimizer.zero_feedback()
            ...     for example in batch:
            ...         # Forward, loss, backward
            ...         await feedback.backward(optimizer=optimizer)
            ...     await optimizer.step()
        """
        for param in self.params:
            param.zero_feedback()

    @abstractmethod
    async def step(self) -> dict[str, str]:
        """Aggregate accumulated feedback and update parameters.

        Should be called after accumulating feedback from a mini-batch
        of examples via feedback.backward(). This method processes all
        accumulated feedback and generates updated parameter values.

        Returns:
            Dictionary mapping parameter names to their new values.
            Only includes parameters that were actually updated.

        Raises:
            RuntimeError: If optimizer is not bound to resources.

        Example:
            >>> updates = await optimizer.step()
            >>> for name, new_value in updates.items():
            ...     print(f"{name}: {new_value[:50]}...")
        """
        pass

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
    1. For each parameter with accumulated feedback:
       a. Aggregate all feedback items into a coherent summary
       b. Generate an improved value based on the summary
       c. Apply the update and clear the feedback buffer

    Attributes:
        conservatism: How conservative updates should be (0-1).

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
        ...         feedback = await loss_fn(output, example["target"], record=record)
        ...         await feedback.backward(optimizer=optimizer)
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
        reasoning_model: str | None = None,
    ) -> None:
        """Initialize the SFAOptimizer.

        Args:
            params: Parameters to optimize (e.g., module.parameters()).
            conservatism: How conservative updates should be (0-1).
                Higher values result in smaller changes. Defaults to 0.7.
            reasoning_model: Optional model identifier for backward-pass
                reasoning.

        Raises:
            ValueError: If conservatism is not in [0, 1] range.

        Example:
            >>> optimizer = SFAOptimizer(
            ...     module.parameters(),
            ...     conservatism=0.5,  # Moderate changes
            ... )
        """
        if not 0.0 <= conservatism <= 1.0:
            raise ValueError(f"conservatism must be in [0, 1], got {conservatism}")

        super().__init__(params, reasoning_model=reasoning_model)
        self.conservatism = conservatism

    async def step(self) -> dict[str, str]:
        """Aggregate accumulated feedback and update parameters.

        For each parameter with accumulated feedback:
        1. Aggregates all feedback items into a coherent summary
        2. Generates an improved value based on conservatism level
        3. Applies the update and clears the feedback buffer

        Returns:
            Dictionary mapping parameter names to their new values.

        Raises:
            RuntimeError: If optimizer is not bound to resources.
        """
        if not self._bound:
            raise RuntimeError("Optimizer not bound. Call bind(resources) first.")

        updates: dict[str, str] = {}

        for param in self.params:
            if not param.requires_grad:
                continue
            if not param._feedback_buffer:
                continue

            # Aggregate all feedback (from fan-out + mini-batch)
            aggregated = await self._aggregate_feedback(param)

            # Generate conservative update
            new_value = await self._generate_update(param, aggregated)

            param.apply_update(new_value)
            updates[param._name or str(id(param))] = new_value

        self._step_count += 1
        return updates

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

        prompt = f"""
Aggregate the following {len(feedbacks)} feedback items about a parameter.

Parameter: {param._name}
Description: {param.description}

Current value:
{param.value}

Feedback items:
{chr(10).join(f"{i + 1}. {fb}" for i, fb in enumerate(feedbacks))}

Synthesize into a single summary that:
1. Identifies common themes across feedback items
2. Prioritizes the most impactful suggestions
3. Notes and resolves any conflicting feedback
4. Provides specific, actionable recommendations

Summary:
""".strip()

        return await self.aggregator(prompt)

    async def _generate_update(self, param: Parameter, aggregated: str) -> str:
        """Generate improved parameter value.

        Uses the aggregated feedback to generate a new parameter value
        that addresses the feedback while respecting the conservatism level.

        Args:
            param: The parameter to update.
            aggregated: Aggregated feedback summary.

        Returns:
            The new parameter value as a string.
        """
        prompt = f"""
Update the following parameter based on aggregated feedback.

Parameter: {param._name}
Description: {param.description}

Current value:
{param.value}

Aggregated feedback:
{aggregated}

Conservatism level: {self.conservatism:.1f} (0=aggressive, 1=minimal changes)

Generate an improved version that:
1. Addresses the key points in the feedback
2. Preserves aspects that are working well
3. Makes changes proportional to conservatism level

Return ONLY the new parameter value, nothing else:
""".strip()

        return await self.updater(prompt)
