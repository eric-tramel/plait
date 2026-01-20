# Optimization System

The optimization system enables LLM-based "learning" through backward passes. Instead of numerical gradients, we propagate natural language feedback through the computation graph to improve Parameters (prompts, instructions, etc.).

## Conceptual Model

Traditional neural network:
```
Forward:  input → layers → output
Backward: loss.backward() → gradients accumulate → optimizer.step()
```

plait:
```
Forward:  input → Modules → output
Backward: output.backward(grad=loss_value) → feedback accumulates → optimizer.step()
```

The API mirrors PyTorch exactly:

```python
# PyTorch
model.train()
optimizer.zero_grad()
for x, y in batch:
    output = model(x)
    loss = criterion(output, y)
    loss.backward()           # Gradients accumulate in .grad
optimizer.step()              # Single update from all gradients

# plait
module.train()
optimizer.zero_feedback()
for example in batch:
    output = await module(example.input)   # Value (record implicit)
    feedback = await loss_fn(output)       # Extracts record automatically
    await output.backward(grad=feedback)              # Value accumulates in Parameters
await optimizer.step()                     # Single update from all feedback
```

## Core Components

### Parameter

Parameters are learnable string values with required self-documentation:

```python
@dataclass
class Parameter:
    """
    A learnable string value that can be optimized via backward passes.

    The description field is required when requires_grad=True and should
    explain what this parameter represents, enabling the optimizer to
    understand how to improve it.
    """

    value: str
    description: str | None = None  # Required when requires_grad=True
    requires_grad: bool = True

    # Internal state (managed by framework)
    _feedback_buffer: list[str] = field(default_factory=list, repr=False)
    _name: str | None = field(default=None, repr=False)  # Set by parent module

    def accumulate_feedback(self, feedback: str) -> None:
        """
        Add feedback to buffer. Called by backward().

        Value accumulates from two sources:
        1. Fan-out within a single graph (multiple downstream nodes)
        2. Multiple training examples in a mini-batch
        """
        if self.requires_grad:
            self._feedback_buffer.append(feedback)

    def zero_feedback(self) -> None:
        """Clear feedback buffer. Called by optimizer.zero_feedback()."""
        self._feedback_buffer.clear()

    def apply_update(self, new_value: str) -> None:
        """Apply new value and clear buffer. Called by optimizer.step()."""
        self.value = new_value
        self._feedback_buffer.clear()

    def __str__(self) -> str:
        return self.value
```

The `description` enables generic optimization - the optimizer LLM knows *what* it's improving:

See `parameters.md` for the full Parameter specification and lifecycle.

```python
class CustomerSupport(Module):
    def __init__(self):
        super().__init__()

        self.persona = Parameter(
            value="You are a helpful customer support agent.",
            description=(
                "Defines the agent's identity and baseline behavior. "
                "Should establish tone (friendly, professional) and "
                "primary goal (helping customers resolve issues)."
            ),
        )

        self.response_format = Parameter(
            value="1. Acknowledge\n2. Explain\n3. Resolve",
            description=(
                "Template structure for response formatting. "
                "Each step should be a brief phrase guiding the response flow."
            ),
        )
```

### Value

`plait.values.Value` is the single canonical container for both forward
outputs and backward feedback. There is no optimizer-specific Value type.

Key points:
- `kind`: `ValueKind` discriminator (TEXT, STRUCTURED, etc.)
- `payload`: underlying data
- `ref`: optional graph node id
- `meta`: metadata; tape ids live in `meta["_tape_ids"]`

Feedback/loss Values use `ValueKind.STRUCTURED` with canonical payload
shape `list[list[str]]`. Optional numeric scores live in `meta["score"]`.

```python
from plait.values import Value, ValueKind

loss = Value(ValueKind.STRUCTURED, [["Too verbose"]], meta={"score": 0.2})
await loss.backward()
```

Note: Some legacy snippets below may use the shorthand
`Value(content=..., score=..., feedback_type=...)`. That is equivalent to
`Value(kind=ValueKind.STRUCTURED, payload=normalize_feedback_payload(content), meta={"score": score})`
and still refers to the single `plait.values.Value` implementation.

Backward discovers the active optimizer automatically (set when an optimizer
is instantiated or bound). If multiple optimizers exist, use
`with optimizer.activate():` to select one for a backward pass.

### ForwardRecord

Captures forward pass state for backward propagation (like PyTorch's autograd tape):

```python
@dataclass
class ForwardRecord:
    """
    Captures forward pass state for backward propagation.

    Stores the computation graph and all intermediate values needed
    to propagate feedback during backward(). Created automatically
    when module is in training mode and wrapped in Value.
    """

    graph: InferenceGraph
    node_inputs: dict[str, dict[str, Any]]   # node_id -> resolved input values
    node_outputs: dict[str, Any]              # node_id -> output value
    module_map: dict[str, Module]    # node_id -> module instance
    node_parameters: dict[str, list[Parameter]]  # node_id -> direct parameters

    # Optional metadata
    execution_order: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
```

### Value (Training Mode)

When a module is in training mode, forward passes return `Value` containers
that carry tape ids in `meta["_tape_ids"]`. This eliminates manual record
management while keeping the output shape intact (including nested
structures).

```python
# Training mode: returns Value with tape ids attached
module.train()
output = await module(input)
output.payload
output.meta["_tape_ids"]

# Eval mode: returns raw value, no recording overhead
module.eval()
output = await module(input)
```

Loss functions consume outputs directly and return `Value` feedback:

```python
feedback = await loss_fn(output)  # Works with Value or raw value
await feedback.backward()
```

This design mirrors PyTorch where tensors with `requires_grad=True` automatically
participate in autograd, without explicit flags on each operation.

### BackwardContext

Information available to modules during backward pass:

```python
@dataclass
class BackwardContext:
    """
    Context available to modules during async backward pass.

    Provides access to inputs, outputs, graph structure, and
    optionally an LLM for reasoning about feedback.
    """

    # This module's execution
    node_id: str
    inputs: dict[str, Any]       # Resolved input values from forward
    output: Any                   # What this module produced

    # Graph context
    graph: InferenceGraph
    all_results: dict[str, Any]  # All node outputs from forward

    # Value from downstream nodes
    downstream_feedback: list[Value]

    # Optimizer-provided LLM for backward reasoning (optional)
    reasoning_llm: LLMInference | None = None

    async def reason(self, prompt: str) -> str:
        """
        Use the optimizer's LLM for backward-pass reasoning.

        This enables custom backward() implementations to use LLM
        reasoning to generate better parameter feedback.

        Args:
            prompt: The reasoning prompt.

        Returns:
            The LLM's response.

        Raises:
            RuntimeError: If no reasoning LLM is available.
        """
        if self.reasoning_llm is None:
            raise RuntimeError(
                "No reasoning LLM available. Pass optimizer to backward() "
                "or configure optimizer with reasoning_model."
            )
        # Use normal module call - not traced during backward pass
        return await self.reasoning_llm(prompt)


@dataclass
class BackwardResult:
    """
    Result of a module's backward pass.

    Contains feedback to propagate to inputs and to accumulate
    in parameters.
    """

    # Value for each input (keyed by input name or position)
    input_feedback: dict[str, Value] = field(default_factory=dict)

    # Value for each parameter (keyed by parameter name)
    parameter_feedback: dict[str, str] = field(default_factory=dict)
```

## Loss Functions

Loss functions evaluate outputs and produce Value. We provide a comprehensive taxonomy covering different evaluation strategies.

### Loss Taxonomy

```
Loss (ABC)
├── Single-Sample Losses (evaluate one output at a time)
│   ├── VerifierLoss          - Programmatic checks (fast, deterministic)
│   ├── LLMFeedbackLoss       - Freeform LLM critique
│   ├── HumanFeedbackLoss     - Freeform human critique (stdout)
│   ├── LLMRubricLoss         - LLM evaluation against Likert scale
│   └── HumanRubricLoss       - Human evaluation against Likert scale (stdout)
│
├── Contrastive Losses (compare multiple outputs)
│   ├── LLMPreferenceLoss     - LLM picks winner from pair
│   ├── HumanPreferenceLoss   - Human picks winner from pair (stdout)
│   ├── LLMRankingLoss        - LLM ranks n outputs
│   └── HumanRankingLoss      - Human ranks n outputs (stdout)
│
└── CompositeLoss             - Weighted combination of losses
```

### Structured Output Schemas

LLM-based losses use structured output for reliable parsing. We define response schemas as dataclasses:

```python
from dataclasses import dataclass
from typing import Literal


@dataclass
class RubricResponse:
    """Structured response for rubric-based evaluation."""
    score: int
    justification: str
    feedback: str


@dataclass
class PreferenceResponse:
    """Structured response for pairwise preference."""
    winner: Literal["A", "B"]
    reason: str
    a_strengths: str
    a_weaknesses: str
    b_strengths: str
    b_weaknesses: str


@dataclass
class RankingResponse:
    """Structured response for n-way ranking."""
    ranking: list[int]  # Indices in order from best to worst
    best_qualities: str
    worst_issues: str
    comparison: str
```

These are passed to `LLMInference(response_format=...)` to enable structured output mode:

```python
# The judge returns a RubricResponse object, not a string
judge = LLMInference(
    alias="judge",
    system_prompt="Evaluate outputs on the given rubric.",
    response_format=RubricResponse,
)

# When called, returns structured data
response: RubricResponse = await judge(prompt)
print(response.score)      # int
print(response.feedback)   # str
```

### Base Class

```python
class Loss(ABC):
    """
    Abstract base class for loss functions.

    Loss functions evaluate outputs and produce Value that can be
    propagated backward through the graph via output.backward(grad=loss_value).

    LLM-based losses use internal LLMInference modules with structured
    output (response_format) for reliable parsing. These modules are
    called through the normal __call__ interface, not special methods.
    """

    @abstractmethod
    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Value:
        """
        Compute feedback for an output.

        Args:
            output: The module output to evaluate.
            target: Optional target/expected output for comparison.
            record: ForwardRecord from run(..., record=True). Required
                    for output.backward(grad=loss_value) to work.
            context: Optional additional context for evaluation.

        Returns:
            Value object. If record is provided, output.backward(grad=loss_value)
            can be called to propagate through the graph.
        """
        pass

    def _attach_record(self, feedback: Value, record: ForwardRecord | None) -> Value:
        """Attach ForwardRecord to feedback for backward()."""
        if record is not None:
            feedback._record = record
        return feedback

---

## Single-Sample Losses

These losses evaluate one output at a time.

### VerifierLoss

Programmatic verification using code:

```python
class VerifierLoss(Loss):
    """
    Loss from programmatic verification.

    Uses code to evaluate outputs (e.g., format checks, keyword presence,
    JSON validity). Fast and deterministic.
    """

    def __init__(
        self,
        verifier: Callable[[Any], tuple[bool, str]],
        success_feedback: str = "Output passed verification.",
    ):
        """
        Args:
            verifier: Function taking output and returning (passed, message).
            success_feedback: Value message when verification passes.
        """
        self.verifier = verifier
        self.success_feedback = success_feedback

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Value:
        passed, message = self.verifier(output)

        feedback = Value(
            content=message if not passed else self.success_feedback,
            score=1.0 if passed else 0.0,
            feedback_type=ValueKind.STRUCTURED,
        )
        return self._attach_record(feedback, record)
```

### LLMFeedbackLoss

Freeform LLM critique without structured scoring:

```python
class LLMFeedbackLoss(Loss):
    """
    Freeform LLM feedback without structured scoring.

    The LLM provides critical feedback on the output without being
    constrained to a specific rubric or scale. Useful for open-ended
    improvement suggestions.
    """

    def __init__(
        self,
        alias: str = "judge",
        criteria: str | None = None,
    ):
        """
        Args:
            alias: Resource alias for the judge LLM.
            criteria: Optional focus areas for feedback.
        """
        self.criteria = criteria
        # Internal LLM module - no structured output needed for freeform
        self.judge = LLMInference(
            alias=alias,
            system_prompt=(
                "You are a critical reviewer. Provide specific, actionable "
                "feedback on how the output could be improved. Be constructive "
                "but thorough in identifying weaknesses."
            ),
        )

    def bind(self, resources: ResourceConfig) -> Self:
        """Bind the internal judge module to resources."""
        self.judge.bind(resources)
        return self

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Value:
        prompt = f"""
Output to critique:
{output}

{f"Context: {context}" if context else ""}
{f"Expected behavior: {target}" if target else ""}
{f"Focus areas: {self.criteria}" if self.criteria else ""}

Provide detailed, actionable feedback:
"""
        # Use normal module call - returns string for freeform feedback
        response = await self.judge(prompt)

        feedback = Value(
            content=response,
            score=None,  # No structured score
            feedback_type=ValueKind.STRUCTURED,
        )
        return self._attach_record(feedback, record)
```

### HumanFeedbackLoss

Freeform human feedback via stdout:

```python
class HumanFeedbackLoss(Loss):
    """
    Freeform human feedback collected via stdout/stdin.

    Prompts the user to provide critical feedback on each output.
    Useful for RLHF-style training with human-in-the-loop.
    """

    def __init__(
        self,
        prompt_template: str | None = None,
        show_context: bool = True,
    ):
        """
        Args:
            prompt_template: Custom prompt shown to user. Use {output},
                            {target}, {context} placeholders.
            show_context: Whether to display context/target to user.
        """
        self.prompt_template = prompt_template
        self.show_context = show_context

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Value:
        # Display output to user
        print("\n" + "=" * 60)
        print("OUTPUT TO EVALUATE:")
        print("-" * 60)
        print(output)

        if self.show_context:
            if target:
                print("-" * 60)
                print(f"Expected: {target}")
            if context:
                print(f"Context: {context}")

        print("=" * 60)

        # Collect feedback
        if self.prompt_template:
            prompt = self.prompt_template.format(
                output=output, target=target, context=context
            )
            print(prompt)
        else:
            print("Please provide feedback on this output.")
            print("What could be improved? (Enter empty line to finish)")

        lines = []
        while True:
            line = input("> ")
            if not line:
                break
            lines.append(line)

        content = "\n".join(lines) if lines else "No feedback provided."

        feedback = Value(
            content=content,
            score=None,
            feedback_type=ValueKind.STRUCTURED,
        )
        return self._attach_record(feedback, record)
```

### LLMRubricLoss

LLM evaluation against a Likert scale rubric:

```python
@dataclass
class RubricLevel:
    """A single level in a Likert scale rubric."""
    score: int
    label: str
    description: str


class LLMRubricLoss(Loss):
    """
    LLM evaluation against a structured Likert scale rubric.

    The LLM evaluates the output against specific criteria using
    a defined scale (e.g., 1-5 or 1-7), providing both a score
    and justification. Uses structured output for reliable parsing.
    """

    def __init__(
        self,
        criteria: str,
        rubric: list[RubricLevel],
        alias: str = "judge",
    ):
        """
        Args:
            criteria: What aspect to evaluate.
            rubric: List of RubricLevel defining the scale.
            alias: Resource alias for the judge LLM.

        Example rubric:
            [
                RubricLevel(1, "Poor", "Fails to address the query"),
                RubricLevel(2, "Below Average", "Partially addresses query"),
                RubricLevel(3, "Average", "Adequately addresses query"),
                RubricLevel(4, "Good", "Thoroughly addresses query"),
                RubricLevel(5, "Excellent", "Exceptionally addresses query"),
            ]
        """
        self.criteria = criteria
        self.rubric = sorted(rubric, key=lambda r: r.score)
        self._max_score = max(r.score for r in rubric)
        self._min_score = min(r.score for r in rubric)

        # Internal LLM module with structured output
        self.judge = LLMInference(
            alias=alias,
            system_prompt=self._build_system_prompt(),
            response_format=RubricResponse,  # Structured output
        )

    def _build_system_prompt(self) -> str:
        rubric_text = "\n".join(
            f"  {level.score} - {level.label}: {level.description}"
            for level in self.rubric
        )
        return f"""You evaluate outputs against a rubric.

Criteria: {self.criteria}

Rating Scale:
{rubric_text}

Always provide a score, justification, and actionable feedback."""

    def bind(self, resources: ResourceConfig) -> Self:
        """Bind the internal judge module to resources."""
        self.judge.bind(resources)
        return self

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Value:
        prompt = f"""
Output to evaluate:
{output}

{f"Expected/Target: {target}" if target else ""}
"""
        # Structured output returns RubricResponse directly
        response: RubricResponse = await self.judge(prompt)

        # Normalize score to 0-1 range
        normalized_score = (response.score - self._min_score) / (self._max_score - self._min_score)

        content = f"Justification: {response.justification}\n\nFeedback: {response.feedback}"

        feedback = Value(
            content=content,
            score=normalized_score,
            feedback_type=ValueKind.STRUCTURED,
            metadata={"raw_score": response.score, "criteria": self.criteria},
        )
        return self._attach_record(feedback, record)
```

### HumanRubricLoss

Human evaluation against a Likert scale via stdout:

```python
class HumanRubricLoss(Loss):
    """
    Human evaluation against a structured Likert scale rubric.

    Displays the output and rubric to the user via stdout, then
    collects their score and optional feedback.
    """

    def __init__(
        self,
        criteria: str,
        rubric: list[RubricLevel],
        require_feedback: bool = True,
    ):
        """
        Args:
            criteria: What aspect to evaluate.
            rubric: List of RubricLevel defining the scale.
            require_feedback: Whether to require written feedback.
        """
        self.criteria = criteria
        self.rubric = sorted(rubric, key=lambda r: r.score)
        self.require_feedback = require_feedback
        self._max_score = max(r.score for r in rubric)
        self._min_score = min(r.score for r in rubric)

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Value:
        # Display output and rubric
        print("\n" + "=" * 60)
        print(f"EVALUATE: {self.criteria}")
        print("=" * 60)
        print("\nOutput:")
        print("-" * 40)
        print(output)
        print("-" * 40)

        if target:
            print(f"\nExpected: {target}")

        print("\nRating Scale:")
        for level in self.rubric:
            print(f"  [{level.score}] {level.label}: {level.description}")

        # Collect score
        while True:
            try:
                score_input = input(f"\nYour score ({self._min_score}-{self._max_score}): ")
                score = int(score_input)
                if self._min_score <= score <= self._max_score:
                    break
                print(f"Please enter a number between {self._min_score} and {self._max_score}")
            except ValueError:
                print("Please enter a valid number")

        # Collect feedback
        feedback_text = ""
        if self.require_feedback:
            print("\nProvide feedback (enter empty line to finish):")
            lines = []
            while True:
                line = input("> ")
                if not line:
                    break
                lines.append(line)
            feedback_text = "\n".join(lines)

        # Normalize score to 0-1
        normalized_score = (score - self._min_score) / (self._max_score - self._min_score)

        feedback = Value(
            content=feedback_text or f"Score: {score}/{self._max_score}",
            score=normalized_score,
            feedback_type=ValueKind.STRUCTURED,
            metadata={"raw_score": score},
        )
        return self._attach_record(feedback, record)
```

---

## Contrastive Losses

Contrastive losses compare multiple outputs to generate feedback based on relative quality. They require running the module multiple times (or with different parameters) and comparing results.

### ContrastiveLoss Base

```python
class ContrastiveLoss(Loss):
    """
    Base class for contrastive losses that compare multiple outputs.

    Contrastive losses generate feedback by comparing outputs rather
    than evaluating them in isolation. This often produces more
    actionable feedback about what makes one output better than another.
    """

    def _generate_contrastive_feedback(
        self,
        winner: Any,
        loser: Any,
        reason: str,
    ) -> str:
        """Generate feedback explaining why winner is better than loser."""
        return f"""
The preferred output was better because: {reason}

To improve, the output should:
- Emulate qualities of the preferred response
- Avoid weaknesses identified in the rejected response

Preferred output characteristics:
{self._summarize_output(winner)}

Rejected output weaknesses:
{self._summarize_output(loser)}
"""

    def _summarize_output(self, output: Any) -> str:
        """Truncate output for feedback if too long."""
        text = str(output)
        if len(text) > 200:
            return text[:200] + "..."
        return text
```

### LLMPreferenceLoss

LLM selects between two outputs:

```python
class LLMPreferenceLoss(ContrastiveLoss):
    """
    LLM pairwise preference comparison.

    Given two outputs (e.g., from current vs previous parameters),
    the LLM selects which is better and explains why. Value is
    generated from the contrast. Uses structured output for reliable parsing.
    """

    def __init__(
        self,
        criteria: str,
        alias: str = "judge",
    ):
        """
        Args:
            criteria: What aspect to compare on.
            alias: Resource alias for the judge LLM.
        """
        self.criteria = criteria

        # Internal LLM module with structured output
        self.judge = LLMInference(
            alias=alias,
            system_prompt=f"You compare two outputs on: {criteria}. "
                          "Determine which is better and explain why.",
            response_format=PreferenceResponse,  # Structured output
        )

    def bind(self, resources: ResourceConfig) -> Self:
        """Bind the internal judge module to resources."""
        self.judge.bind(resources)
        return self

    async def compare(
        self,
        output_a: Any,
        output_b: Any,
        *,
        record_a: ForwardRecord | None = None,
        record_b: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[Value, Value]:
        """
        Compare two outputs and return feedback for each.

        Returns:
            Tuple of (feedback_for_a, feedback_for_b). The preferred
            output gets positive feedback, the rejected gets improvement
            suggestions based on the contrast.
        """
        prompt = f"""
{f"Context: {context}" if context else ""}

Output A:
{output_a}

Output B:
{output_b}

Which output is better?
"""
        # Structured output returns PreferenceResponse directly
        response: PreferenceResponse = await self.judge(prompt)

        # Generate contrastive feedback
        if response.winner == "A":
            fb_a = Value(
                content=f"Preferred. Strengths: {response.a_strengths}",
                score=1.0,
                feedback_type=ValueKind.STRUCTURED,
            )
            fb_b = Value(
                content=self._generate_contrastive_feedback(
                    output_a, output_b,
                    f"{response.reason}\n\nWeaknesses: {response.b_weaknesses}"
                ),
                score=0.0,
                feedback_type=ValueKind.STRUCTURED,
            )
        else:
            fb_a = Value(
                content=self._generate_contrastive_feedback(
                    output_b, output_a,
                    f"{response.reason}\n\nWeaknesses: {response.a_weaknesses}"
                ),
                score=0.0,
                feedback_type=ValueKind.STRUCTURED,
            )
            fb_b = Value(
                content=f"Preferred. Strengths: {response.b_strengths}",
                score=1.0,
                feedback_type=ValueKind.STRUCTURED,
            )

        # Attach records
        if record_a:
            fb_a._record = record_a
        if record_b:
            fb_b._record = record_b

        return fb_a, fb_b

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Value:
        """Single-output interface (requires target as comparison)."""
        if target is None:
            raise ValueError("LLMPreferenceLoss requires target for comparison")
        fb_output, _ = await self.compare(output, target, record_a=record, context=context)
        return fb_output
```

### HumanPreferenceLoss

Human selects between two outputs via stdout:

```python
class HumanPreferenceLoss(ContrastiveLoss):
    """
    Human pairwise preference comparison via stdout.

    Displays two outputs side-by-side and asks the user to select
    which is better and why.
    """

    def __init__(
        self,
        criteria: str,
        require_reason: bool = True,
    ):
        """
        Args:
            criteria: What aspect to compare on.
            require_reason: Whether to require explanation.
        """
        self.criteria = criteria
        self.require_reason = require_reason

    async def compare(
        self,
        output_a: Any,
        output_b: Any,
        *,
        record_a: ForwardRecord | None = None,
        record_b: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[Value, Value]:
        """Compare two outputs and return feedback for each."""
        print("\n" + "=" * 60)
        print(f"COMPARE: {self.criteria}")
        print("=" * 60)

        if context:
            print(f"\nContext: {context}")

        print("\n[A] Output A:")
        print("-" * 40)
        print(output_a)

        print("\n[B] Output B:")
        print("-" * 40)
        print(output_b)

        print("=" * 60)

        # Get preference
        while True:
            choice = input("\nWhich is better? (A/B): ").strip().upper()
            if choice in ("A", "B"):
                break
            print("Please enter A or B")

        # Get reason
        reason = ""
        if self.require_reason:
            print("\nWhy is it better? (enter empty line to finish):")
            lines = []
            while True:
                line = input("> ")
                if not line:
                    break
                lines.append(line)
            reason = "\n".join(lines)

        # Generate feedback
        winner, loser = (output_a, output_b) if choice == "A" else (output_b, output_a)

        if choice == "A":
            fb_a = Value(
                content=f"Preferred by human. Reason: {reason}",
                score=1.0,
                feedback_type=ValueKind.STRUCTURED,
            )
            fb_b = Value(
                content=self._generate_contrastive_feedback(winner, loser, reason),
                score=0.0,
                feedback_type=ValueKind.STRUCTURED,
            )
        else:
            fb_a = Value(
                content=self._generate_contrastive_feedback(winner, loser, reason),
                score=0.0,
                feedback_type=ValueKind.STRUCTURED,
            )
            fb_b = Value(
                content=f"Preferred by human. Reason: {reason}",
                score=1.0,
                feedback_type=ValueKind.STRUCTURED,
            )

        if record_a:
            fb_a._record = record_a
        if record_b:
            fb_b._record = record_b

        return fb_a, fb_b

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Value:
        """Single-output interface (requires target as comparison)."""
        if target is None:
            raise ValueError("HumanPreferenceLoss requires target for comparison")
        fb_output, _ = await self.compare(output, target, record_a=record, context=context)
        return fb_output
```

### LLMRankingLoss

LLM ranks n outputs:

```python
class LLMRankingLoss(ContrastiveLoss):
    """
    LLM ranking of multiple outputs.

    Given n outputs, the LLM ranks them from best to worst and
    explains the ranking. Value is generated based on relative
    position and comparison to better-ranked outputs. Uses structured
    output for reliable parsing.
    """

    def __init__(
        self,
        criteria: str,
        n: int = 4,
        alias: str = "judge",
    ):
        """
        Args:
            criteria: What aspect to rank on.
            n: Expected number of outputs to compare (default 4).
            alias: Resource alias for the judge LLM.
        """
        self.criteria = criteria
        self.n = n

        # Internal LLM module with structured output
        self.judge = LLMInference(
            alias=alias,
            system_prompt=f"You rank outputs from best to worst on: {criteria}. "
                          "Provide the ranking as a list of indices and explain your reasoning.",
            response_format=RankingResponse,  # Structured output
        )

    def bind(self, resources: ResourceConfig) -> Self:
        """Bind the internal judge module to resources."""
        self.judge.bind(resources)
        return self

    async def rank(
        self,
        outputs: list[Any],
        *,
        records: list[ForwardRecord | None] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Value]:
        """
        Rank multiple outputs and return feedback for each.

        Args:
            outputs: List of outputs to rank.
            records: Optional list of ForwardRecords (same length as outputs).
            context: Optional context for ranking.

        Returns:
            List of Value objects in same order as inputs.
            Scores are normalized by rank (best=1.0, worst=0.0).
        """
        if len(outputs) < 2:
            raise ValueError("Need at least 2 outputs to rank")

        # Build prompt
        output_strs = []
        for i, out in enumerate(outputs):
            output_strs.append(f"[{i+1}] {out}")

        prompt = f"""
{f"Context: {context}" if context else ""}

Outputs to rank:
{chr(10).join(output_strs)}

Rank these {len(outputs)} outputs from BEST to WORST.
"""
        # Structured output returns RankingResponse directly
        response: RankingResponse = await self.judge(prompt)

        # Convert 1-indexed ranking to 0-indexed
        ranking = [r - 1 for r in response.ranking]

        # Generate feedback based on rank
        feedbacks = []
        n = len(outputs)
        for i, output in enumerate(outputs):
            rank = ranking.index(i) + 1  # 1-indexed rank for display
            score = (n - rank) / (n - 1) if n > 1 else 1.0  # Normalize to 0-1

            if rank == 1:
                content = f"Ranked #1 (best). {response.best_qualities}"
            elif rank == n:
                content = f"Ranked #{rank} (worst). {response.worst_issues}\n\n"
                content += f"To improve, emulate the #1 output's qualities: {response.best_qualities}"
            else:
                content = f"Ranked #{rank}/{n}. {response.comparison}\n\n"
                content += f"To improve, move toward qualities of higher-ranked outputs."

            fb = Value(
                content=content,
                score=score,
                feedback_type=ValueKind.STRUCTURED,
                metadata={"rank": rank, "total": n},
            )

            if records and i < len(records) and records[i]:
                fb._record = records[i]

            feedbacks.append(fb)

        return feedbacks

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Value:
        """Single-output interface - ranks against target."""
        if target is None:
            raise ValueError("LLMRankingLoss requires target for comparison")
        targets = target if isinstance(target, list) else [target]
        outputs = [output] + targets
        records = [record] + [None] * len(targets)
        feedbacks = await self.rank(outputs, records=records, context=context)
        return feedbacks[0]
```

### HumanRankingLoss

Human ranks n outputs via stdout:

```python
class HumanRankingLoss(ContrastiveLoss):
    """
    Human ranking of multiple outputs via stdout.

    Displays n outputs and asks the user to rank them from best to
    worst, with optional feedback.
    """

    def __init__(
        self,
        criteria: str,
        n: int = 4,
        require_feedback: bool = True,
    ):
        """
        Args:
            criteria: What aspect to rank on.
            n: Expected number of outputs to compare.
            require_feedback: Whether to require written feedback.
        """
        self.criteria = criteria
        self.n = n
        self.require_feedback = require_feedback

    async def rank(
        self,
        outputs: list[Any],
        *,
        records: list[ForwardRecord | None] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Value]:
        """Rank multiple outputs and return feedback for each."""
        if len(outputs) < 2:
            raise ValueError("Need at least 2 outputs to rank")

        # Display outputs
        print("\n" + "=" * 60)
        print(f"RANK THESE OUTPUTS: {self.criteria}")
        print("=" * 60)

        if context:
            print(f"\nContext: {context}")

        for i, out in enumerate(outputs):
            print(f"\n[{i+1}] Output {i+1}:")
            print("-" * 40)
            print(out)

        print("=" * 60)

        # Get ranking
        while True:
            try:
                ranking_input = input(
                    f"\nRank from best to worst (e.g., '3,1,2' if 3 is best): "
                )
                ranking = [int(x.strip()) - 1 for x in ranking_input.split(",")]
                if len(ranking) == len(outputs) and set(ranking) == set(range(len(outputs))):
                    break
                print(f"Please rank all {len(outputs)} outputs exactly once")
            except ValueError:
                print("Please enter comma-separated numbers")

        # Get feedback
        feedback_text = ""
        if self.require_feedback:
            print("\nExplain your ranking (enter empty line to finish):")
            lines = []
            while True:
                line = input("> ")
                if not line:
                    break
                lines.append(line)
            feedback_text = "\n".join(lines)

        # Generate feedback based on rank
        feedbacks = []
        n = len(outputs)
        for i, output in enumerate(outputs):
            rank = ranking.index(i) + 1
            score = (n - rank) / (n - 1) if n > 1 else 1.0

            if rank == 1:
                content = f"Ranked #1 (best) by human.\n\n{feedback_text}"
            elif rank == n:
                content = f"Ranked #{rank} (worst) by human.\n\n{feedback_text}"
            else:
                content = f"Ranked #{rank}/{n} by human.\n\n{feedback_text}"

            fb = Value(
                content=content,
                score=score,
                feedback_type=ValueKind.STRUCTURED,
                metadata={"rank": rank, "total": n},
            )

            if records and i < len(records) and records[i]:
                fb._record = records[i]

            feedbacks.append(fb)

        return feedbacks

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Value:
        """Single-output interface."""
        if target is None:
            raise ValueError("HumanRankingLoss requires target for comparison")
        targets = target if isinstance(target, list) else [target]
        outputs = [output] + targets
        records = [record] + [None] * len(targets)
        feedbacks = await self.rank(outputs, records=records, context=context)
        return feedbacks[0]
```

---

## Composite Loss

### CompositeLoss

Combine multiple loss functions:

```python
class CompositeLoss(Loss):
    """
    Combine multiple loss functions with weights.

    Useful for multi-objective optimization (e.g., helpfulness + safety).
    """

    def __init__(
        self,
        losses: list[tuple[Loss, float]],  # (loss, weight) pairs
        aggregator: LLMInference | None = None,
    ):
        """
        Args:
            losses: List of (loss_function, weight) pairs.
            aggregator: Optional LLM to synthesize feedback. If None,
                       feedback is concatenated with weights.
        """
        self.losses = losses
        self.aggregator = aggregator

    async def __call__(
        self,
        output: Any,
        target: Any | None = None,
        *,
        record: ForwardRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> Value:
        # Gather all feedback
        feedbacks = []
        weighted_score = 0.0
        total_weight = 0.0

        for loss, weight in self.losses:
            # Don't pass record to sub-losses (we attach to composite)
            fb = await loss(output, target, context=context)
            feedbacks.append((fb, weight))
            if fb.score is not None:
                weighted_score += fb.score * weight
                total_weight += weight

        # Aggregate feedback text
        if self.aggregator:
            combined = await self._llm_aggregate(feedbacks)
        else:
            combined = self._simple_aggregate(feedbacks)

        feedback = Value(
            content=combined,
            score=weighted_score / total_weight if total_weight > 0 else None,
            feedback_type=ValueKind.STRUCTURED,
        )
        return self._attach_record(feedback, record)

    def _simple_aggregate(self, feedbacks: list[tuple[Value, float]]) -> str:
        parts = []
        for fb, weight in feedbacks:
            parts.append(f"[Weight: {weight}] {fb.content}")
        return "\n\n".join(parts)

    async def _llm_aggregate(self, feedbacks: list[tuple[Value, float]]) -> str:
        prompt = "Synthesize the following feedback into a coherent summary:\n\n"
        for fb, weight in feedbacks:
            prompt += f"--- Value (weight: {weight}) ---\n{fb.content}\n\n"
        prompt += "Provide a unified summary of the key points and suggestions."
        return await self.aggregator(prompt)
```

---

## Loss Usage Examples

```python
# Programmatic verification
verifier = VerifierLoss(
    verifier=lambda out: (
        "error" not in out.lower(),
        "Output contains error message" if "error" in out.lower() else "OK"
    )
)

# Freeform LLM feedback
llm_feedback = LLMFeedbackLoss(
    alias="judge",
    criteria="clarity and completeness",
)

# Rubric-based evaluation
rubric = [
    RubricLevel(1, "Poor", "Fails to address the query"),
    RubricLevel(2, "Below Average", "Partially addresses query with errors"),
    RubricLevel(3, "Average", "Adequately addresses query"),
    RubricLevel(4, "Good", "Thoroughly addresses query"),
    RubricLevel(5, "Excellent", "Exceptionally clear and complete"),
]

llm_rubric = LLMRubricLoss(
    criteria="helpfulness",
    rubric=rubric,
    alias="judge",
)

human_rubric = HumanRubricLoss(
    criteria="helpfulness",
    rubric=rubric,
)

# Preference comparison
llm_pref = LLMPreferenceLoss(
    criteria="overall quality",
    alias="judge",
)

# Ranking multiple outputs
llm_ranking = LLMRankingLoss(
    criteria="response quality",
    n=4,
    alias="judge",
)

# Composite: combine multiple signals
composite = CompositeLoss([
    (verifier, 0.3),           # 30% weight on format
    (llm_rubric, 0.5),         # 50% weight on helpfulness
    (LLMFeedbackLoss(criteria="style"), 0.2),  # 20% weight on open feedback
])
```

## Module Backward Implementation

### Default Behavior

```python
class Module:
    async def backward(
        self,
        feedback: Value,
        ctx: BackwardContext,
    ) -> BackwardResult:
        """
        Propagate feedback backward through this module.

        Default implementation passes feedback unchanged to all inputs.
        Override for custom backward logic.

        Args:
            feedback: Combined feedback from downstream nodes.
            ctx: Context with inputs, output, graph, and optional reasoning LLM.

        Returns:
            BackwardResult with input_feedback and parameter_feedback.
        """
        result = BackwardResult()

        # Pass feedback to all inputs unchanged
        for input_name in ctx.inputs:
            result.input_feedback[input_name] = feedback

        return result
```

### LLMInference Backward

```python
class LLMInference(Module):
    async def backward(
        self,
        feedback: Value,
        ctx: BackwardContext,
    ) -> BackwardResult:
        """
        Backward pass for LLM inference.

        Generates feedback for both the input prompt and any
        learnable parameters (like system_prompt).
        """
        result = BackwardResult()

        # Value for the input prompt
        result.input_feedback["prompt"] = Value(
            content=f"The LLM output received this feedback: {feedback.payload}",
            score=feedback.meta.get("score"),
            feedback_type=feedback.feedback_type,
        )

        # Value for learnable parameters
        if isinstance(self.system_prompt, Parameter) and self.system_prompt.requires_grad:
            result.parameter_feedback["system_prompt"] = f"""
The LLM module with system prompt:
"{self.system_prompt.value}"

Parameter description: {self.system_prompt.description}

Received input: {ctx.inputs.get('prompt', '')[:500]}...

Produced output: {str(ctx.output)[:500]}...

Received this feedback: {feedback.payload}
{f"Score: {feedback.meta.get("score")}" if feedback.meta.get("score") is not None else ""}

Suggest specific improvements to the system prompt that would address this feedback.
"""

        return result
```

### Custom Backward with Reasoning

```python
class SmartResponder(Module):
    """Example module with custom backward that uses optimizer's reasoning LLM."""

    def __init__(self):
        super().__init__()
        self.instructions = Parameter(
            value="Be helpful and concise.",
            description="Instructions for generating responses. Should guide tone and style.",
        )
        self.llm = LLMInference(alias="assistant")

    def forward(self, query: str) -> str:
        return self.llm(f"{self.instructions}\n\nQuery: {query}")

    async def backward(
        self,
        feedback: Value,
        ctx: BackwardContext,
    ) -> BackwardResult:
        result = BackwardResult()

        # Pass feedback to input
        result.input_feedback["query"] = Value(
            content=f"Response feedback: {feedback.payload}",
            score=feedback.meta.get("score"),
        )

        # Use reasoning LLM for sophisticated parameter feedback
        if self.instructions.requires_grad and ctx.reasoning_llm:
            analysis = await ctx.reason(f"""
Analyze why this response received negative feedback and suggest parameter improvements.

Parameter: instructions
Description: {self.instructions.description}
Current value: "{self.instructions.value}"

Input query: {ctx.inputs.get('query', '')}
Generated response: {ctx.output}

Value received: {feedback.payload}
Score: {feedback.meta.get("score")}

Provide specific, actionable suggestions for improving the instructions parameter.
""")
            result.parameter_feedback["instructions"] = analysis
        elif self.instructions.requires_grad:
            # Fallback without reasoning LLM
            result.parameter_feedback["instructions"] = (
                f"Output received feedback: {feedback.payload}. "
                f"Consider adjusting instructions to address this."
            )

        return result
```

## Optimizer

Optimizers aggregate feedback and update parameters, following the PyTorch pattern:

```python
class Optimizer(ABC):
    """
    Base optimizer for parameter updates via LLM.

    Follows torch.optim pattern: initialized with parameters,
    accumulates feedback across backward() calls, updates on step().

    Optimizers use internal LLMInference modules for aggregation and
    update generation. These use fixed aliases that must be configured
    in ResourceConfig.
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
    ):
        """
        Args:
            params: Parameters to optimize (e.g., module.parameters()).
            reasoning_model: Optional model for backward-pass reasoning.
                            If provided, optimizer.reasoning_llm is available.
        """
        self.params = list(params)
        self._step_count = 0

        # Internal LLMs with fixed aliases
        self.aggregator = LLMInference(
            alias=self.AGGREGATOR_ALIAS,
            system_prompt=self._aggregator_system_prompt(),
        )
        self.updater = LLMInference(
            alias=self.UPDATER_ALIAS,
            system_prompt=self._updater_system_prompt(),
        )
        self.reasoning_llm: LLMInference | None = None
        if reasoning_model:
            self.reasoning_llm = LLMInference(
                alias=self.REASONING_ALIAS,
                system_prompt=self._reasoning_system_prompt(),
            )

        self._bound = False

    def bind(self, resources: ResourceConfig) -> Self:
        """
        Bind optimizer's internal LLMs to resources.

        The ResourceConfig must include the optimizer aliases:
        - "optimizer/aggregator"
        - "optimizer/updater"
        - "optimizer/reasoning" (if reasoning_model was specified)
        """
        # Bind internal modules
        self.aggregator.bind(resources)
        self.updater.bind(resources)
        if self.reasoning_llm:
            self.reasoning_llm.bind(resources)
        self._bound = True
        return self

    def zero_feedback(self) -> None:
        """Clear all parameter feedback buffers (like zero_grad)."""
        for param in self.params:
            param.zero_feedback()

    @abstractmethod
    async def step(self) -> dict[str, str]:
        """
        Aggregate accumulated feedback and update parameters.

        Should be called after accumulating feedback from a mini-batch
        of examples via output.backward(grad=loss_value).

        Returns:
            Dict mapping parameter names to their new values.
        """
        pass

    def _aggregator_system_prompt(self) -> str:
        return (
            "You synthesize multiple feedback items into a coherent summary. "
            "Identify common themes, prioritize impactful suggestions, and "
            "resolve any conflicting feedback."
        )

    def _updater_system_prompt(self) -> str:
        return (
            "You improve text parameters based on feedback. "
            "Make targeted changes that address the feedback while "
            "preserving aspects that work well."
        )

    def _reasoning_system_prompt(self) -> str:
        return (
            "You analyze why outputs received certain feedback and "
            "suggest specific parameter improvements."
        )
```

### SFAOptimizer

Stochastic Value Ascent - conservative, incremental updates:

```python
class SFAOptimizer(Optimizer):
    """
    Stochastic Value Ascent optimizer.

    Makes small, targeted changes based on accumulated feedback rather than
    aggressive rewrites. Good for fine-tuning working prompts.

    The conservatism parameter controls how aggressive updates are:
    - 0.0: Aggressive, may significantly rewrite parameters
    - 1.0: Very conservative, minimal changes only
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        conservatism: float = 0.7,
        reasoning_model: str | None = None,
    ):
        """
        Args:
            params: Parameters to optimize.
            conservatism: How conservative updates should be (0-1).
            reasoning_model: Optional model for backward reasoning.
        """
        super().__init__(params, reasoning_model=reasoning_model)
        self.conservatism = conservatism

    async def step(self) -> dict[str, str]:
        if not self._bound:
            raise RuntimeError("Optimizer not bound. Call bind(resources) first.")

        updates = {}

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
        """Combine all feedback items into one coherent summary."""
        feedbacks = param._feedback_buffer

        if len(feedbacks) == 1:
            return feedbacks[0]

        prompt = f"""
Aggregate the following {len(feedbacks)} feedback items about a parameter.

Parameter: {param._name}
Description: {param.description}

Current value:
{param.value}

Value items:
{chr(10).join(f"{i+1}. {fb}" for i, fb in enumerate(feedbacks))}

Synthesize into a single summary that:
1. Identifies common themes across feedback items
2. Prioritizes the most impactful suggestions
3. Notes and resolves any conflicting feedback
4. Provides specific, actionable recommendations

Summary:
"""
        return await self.aggregator(prompt)

    async def _generate_update(self, param: Parameter, aggregated: str) -> str:
        """Generate improved parameter value."""
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
"""
        return await self.updater(prompt)
```

### MomentumOptimizer

Tracks feedback history for smoother updates:

```python
class MomentumOptimizer(Optimizer):
    """
    Optimizer that considers feedback history.

    Maintains a running history of feedback, giving more weight to
    consistent feedback over time. Helps avoid oscillation from
    noisy individual samples.
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        momentum: float = 0.9,
        history_size: int = 10,
        reasoning_model: str | None = None,
    ):
        """
        Args:
            params: Parameters to optimize.
            momentum: Weight given to historical feedback (0-1).
            history_size: Number of past aggregations to remember.
            reasoning_model: Optional model for backward reasoning.
        """
        super().__init__(params, reasoning_model=reasoning_model)
        self.momentum = momentum
        self.history_size = history_size
        self._history: dict[str, list[str]] = defaultdict(list)

    async def step(self) -> dict[str, str]:
        # Implementation considers both current feedback and history
        ...
```

## Backward Propagation

The `_propagate_backward_value` function handles graph traversal:

```python
async def _propagate_backward_value(
    feedback: Value,
    record: ForwardRecord,
    reasoning_llm: LLMInference | None = None,
) -> None:
    """
    Propagate feedback backward through a traced graph.

    Traverses nodes in reverse topological order, calling each module's
    backward() method and accumulating feedback into Parameters.

    Args:
        feedback: The feedback to propagate (from loss function).
        record: ForwardRecord from the forward pass.
        reasoning_llm: Optional LLM for backward reasoning.
    """
    graph = record.graph
    feedback_map: dict[str, Value] = {}

    # Initialize with output feedback
    for output_id in graph.output_ids:
        feedback_map[output_id] = feedback

    # Process in reverse topological order
    for node_id in reversed(graph.topological_order()):
        node = graph.nodes[node_id]

        # Skip input nodes
        if node_id in graph.input_ids:
            continue

        # Gather feedback from downstream nodes
        downstream_feedback = []
        for dependent_id in node.dependents:
            if dependent_id in feedback_map:
                downstream_feedback.append(feedback_map[dependent_id])

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
            input_node_id = _resolve_input_node(node, input_name, graph)
            if input_node_id:
                feedback_map[input_node_id] = input_fb

        # Accumulate parameter feedback
        for param_name, param_fb in result.parameter_feedback.items():
            for name, param in module.named_parameters():
                if name.endswith(param_name):
                    param.accumulate_feedback(param_fb)


def _combine_feedback(feedbacks: list[Value]) -> Value:
    """Combine multiple feedback items into one."""
    if len(feedbacks) == 1:
        return normalize_feedback_value(feedbacks[0])

    normalized = [normalize_feedback_value(fb) for fb in feedbacks]
    combined_payload: list[list[str]] = []
    for fb in normalized:
        combined_payload.extend(fb.payload)

    meta: dict[str, Any] = {}
    scores = [fb.meta.get("score") for fb in normalized if "score" in fb.meta]
    if scores:
        meta["scores"] = scores

    return Value(kind=ValueKind.STRUCTURED, payload=combined_payload, meta=meta)
```

## Batch Training API

Training requires three batch-native operations: forward passes, loss computation, and backward passes. Each operation has a clean batch interface for concurrent execution. Training mode (`module.train()`) enables automatic record capture via `Value`.

### Batch Forward (Training Mode)

In training mode, forward passes return `Value` containers with tape ids
attached in `meta["_tape_ids"]`:

```python
module.train()

# Single input - returns Value
output = await module(input)
output.payload
output.meta["_tape_ids"]

# Batch inputs - returns list[Value]
outputs = await module(batch_inputs)
```

For inference (no recording overhead):

```python
module.eval()
output = await module(input)  # raw value
```

### Batch Loss Computation

Losses are async callables that return `Value` (typically
`ValueKind.STRUCTURED` with `list[list[str]]` payloads). For batches, you can
compute per-sample losses and call backward on each sample, or aggregate and
propagate once using the outputs' tape ids.

```python
module.train()
outputs = await module(batch_inputs)
losses = [await loss_fn(out) for out in outputs]

# Option 1: backward per sample
for loss in losses:
    await loss.backward()

# Option 2: aggregate and propagate once using outputs' tapes
combined_payload = [item for loss in losses for item in loss.payload]
grad = Value(ValueKind.STRUCTURED, combined_payload)
await Value.backward(outputs, grad=grad)
```

This keeps the API PyTorch-like while relying on a single canonical `Value`
implementation.

---

## Mini-Batch Training

Mini-batch training is the primary pattern, accumulating feedback across multiple samples before updating. The aggregated loss API makes this clean and PyTorch-like:

```python
async def train(
    module: Module,
    dataset: list[dict[str, Any]],
    loss_fn: Loss,
    optimizer: Optimizer,
    *,
    epochs: int = 1,
    batch_size: int = 8,
    shuffle: bool = True,
) -> TrainingHistory:
    """
    Train an inference module using mini-batch optimization.

    Args:
        module: Module to train (must be bound to resources).
        dataset: List of {"input": ..., "target": ...} examples.
        loss_fn: Loss function to evaluate outputs.
        optimizer: Optimizer for parameter updates (must be bound).
        epochs: Number of passes through dataset.
        batch_size: Examples per optimizer step.
        shuffle: Whether to shuffle dataset each epoch.

    Returns:
        TrainingHistory with losses and parameter snapshots.
    """
    history = TrainingHistory()

    # Enable training mode (returns Value with records)
    module.train()

    for epoch in range(epochs):
        epoch_scores: list[float] = []

        indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(indices)

        # Process in mini-batches
        for batch_start in range(0, len(dataset), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch = [dataset[idx] for idx in batch_indices]

            optimizer.zero_feedback()

            # ─────────────────────────────────────────────────────────
            # Batch forward (concurrent, returns Value)
            # ─────────────────────────────────────────────────────────
            batch_inputs = [example["input"] for example in batch]
            outputs = await module(batch_inputs)  # list[Value]

            # ─────────────────────────────────────────────────────────
            # Batch loss → single aggregated Value (PyTorch-like)
            # ─────────────────────────────────────────────────────────
            feedback = await loss_fn(
                outputs,  # List auto-detected as batch
                target=[example.get("target") for example in batch],
                context=[{"input": example["input"]} for example in batch],
            )
            if feedback.meta.get("score") is not None:
                epoch_scores.append(feedback.meta.get("score"))

            # ─────────────────────────────────────────────────────────
            # Single backward call → propagates through ALL samples
            # ─────────────────────────────────────────────────────────
            await output.backward(grad=feedback)

            # Single optimizer step per batch
            updates = await optimizer.step()

            history.record_step(
                epoch=epoch,
                batch_start=batch_start,
                score=feedback.meta.get("score"),
                updates=updates,
            )

        # Epoch summary
        avg_score = sum(epoch_scores) / len(epoch_scores) if epoch_scores else None
        history.record_epoch(epoch, avg_score)
        print(f"Epoch {epoch + 1}/{epochs}: avg_score = {avg_score:.3f}")

    # Switch back to eval mode
    module.eval()

    return history
```

### Sequential Alternative

For cases requiring per-sample control (custom logging, early stopping per sample), you can use the sequential loop:

```python
module.train()
for example in batch:
    output = await module(example["input"])  # Value
    feedback = await loss_fn(output, context=example)  # Single sample
    await output.backward(grad=feedback)
```

## Value Accumulation

Value accumulates from two sources:

### 1. Fan-out Within a Single Graph

When a node has multiple downstream dependents:

```
         ┌─────────────────┐
         │ SharedProcessor │
         │   param: P1     │
         └────────┬────────┘
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
 ┌────────┐  ┌────────┐  ┌────────┐
 │ Task A │  │ Task B │  │ Task C │
 └────┬───┘  └────┬───┘  └────┬───┘
      │           │           │
      ▼           ▼           ▼
 Value A  Value B  Value C

One backward() call → P1._feedback_buffer gets 3 items
```

### 2. Mini-batch Across Samples

Multiple training examples each contribute feedback:

```
Sample 1: forward → backward → P1 gets feedback_1
Sample 2: forward → backward → P1 gets feedback_2
Sample 3: forward → backward → P1 gets feedback_3
Sample 4: forward → backward → P1 gets feedback_4

Four backward() calls → P1._feedback_buffer has 4 items
```

### Combined

With batch_size=4 and fan-out=3, a parameter could have up to 12 feedback items before `optimizer.step()` aggregates them all.

## Complete Example

```python
from plait import (
    Module, LLMInference, Parameter,
    run, train,
    SFAOptimizer,
    ResourceConfig,
)

# ═══════════════════════════════════════════════════════════════════
#  1. DEFINE MODULE WITH SELF-DOCUMENTING PARAMETERS
# ═══════════════════════════════════════════════════════════════════

class CustomerSupport(Module):
    def __init__(self):
        super().__init__()

        self.persona = Parameter(
            value="You are a helpful customer support agent.",
            description=(
                "Defines the agent's identity and baseline behavior. "
                "Should establish tone (friendly, professional) and "
                "primary goal (helping customers resolve issues)."
            ),
        )

        self.response_guidelines = Parameter(
            value="Be concise and helpful.",
            description=(
                "Specific instructions for response style and structure. "
                "Can include formatting rules, required elements, and tone guidance."
            ),
        )

        self.llm = LLMInference(alias="assistant")

    def forward(self, query: str) -> str:
        prompt = f"""{self.persona}

Guidelines: {self.response_guidelines}

Customer query: {query}

Response:"""
        return self.llm(prompt)


# ═══════════════════════════════════════════════════════════════════
#  2. SETUP RESOURCES (including optimizer aliases)
# ═══════════════════════════════════════════════════════════════════

resources = ResourceConfig({
    # Module endpoints
    "assistant": {
        "model": "gpt-4o-mini",
        "max_concurrent": 10,
    },
    # Loss function endpoint
    "judge": {
        "model": "gpt-4o",
        "max_concurrent": 5,
    },
    # Optimizer endpoints (fixed aliases)
    "optimizer/aggregator": {
        "model": "gpt-4o",
        "max_concurrent": 2,
    },
    "optimizer/updater": {
        "model": "gpt-4o",
        "max_concurrent": 2,
    },
})


# ═══════════════════════════════════════════════════════════════════
#  3. CREATE AND BIND COMPONENTS
# ═══════════════════════════════════════════════════════════════════

module = CustomerSupport().bind(resources)

loss_fn = LLMJudge(
    judge=LLMInference(alias="judge").bind(resources),
    criteria=(
        "Evaluate the response on: "
        "1) Empathy - acknowledges customer feelings "
        "2) Clarity - easy to understand "
        "3) Helpfulness - actually solves the problem "
        "4) Professionalism - appropriate tone"
    ),
)

optimizer = SFAOptimizer(
    module.parameters(),  # [persona, response_guidelines]
    conservatism=0.6,
).bind(resources)


# ═══════════════════════════════════════════════════════════════════
#  4. TRAINING DATA
# ═══════════════════════════════════════════════════════════════════

dataset = [
    {"input": "My order hasn't arrived and it's been 2 weeks!",
     "target": "Empathetic + tracking lookup + resolution options"},
    {"input": "I want a refund for this broken product",
     "target": "Apology + clear refund process + timeline"},
    {"input": "Your website crashed and I lost my cart",
     "target": "Sympathy + help recovering + compensation"},
    {"input": "I was charged twice for the same order",
     "target": "Immediate concern + verification + refund"},
    {"input": "The product doesn't match the description",
     "target": "Apology + options (return/exchange/refund)"},
    {"input": "I need to change my shipping address",
     "target": "Can-do attitude + clear process + confirmation"},
    {"input": "When will this item be back in stock?",
     "target": "Helpful info + waitlist option + alternatives"},
    {"input": "I'm having trouble applying a coupon code",
     "target": "Patience + troubleshooting + manual application"},
]


# ═══════════════════════════════════════════════════════════════════
#  5. TRAIN WITH MINI-BATCHES
# ═══════════════════════════════════════════════════════════════════

history = await train(
    module=module,
    dataset=dataset,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=3,
    batch_size=4,  # 4 samples before each optimizer.step()
)

# After training, parameters have been updated:
print("Updated persona:")
print(module.persona.value)

print("\nUpdated guidelines:")
print(module.response_guidelines.value)


# ═══════════════════════════════════════════════════════════════════
#  6. MANUAL TRAINING LOOP (equivalent to above)
# ═══════════════════════════════════════════════════════════════════

# For more control, use the explicit batch loop:
module.train()  # Enable recording via Value

for epoch in range(3):
    for batch_start in range(0, len(dataset), 4):
        batch = dataset[batch_start:batch_start + 4]

        optimizer.zero_feedback()

        # Batch forward (returns Value with records)
        inputs = [example["input"] for example in batch]
        outputs = await module(inputs)  # list[Value]

        # Batch loss computation (extracts records automatically)
        feedbacks = await loss_fn.batch(
            outputs,
            targets=[example["target"] for example in batch],
        )

        # Batch backward
        await Value.backward(outputs, grad=loss_value)

        # Step - like optimizer.step() in PyTorch
        await optimizer.step()

module.eval()  # Disable recording for inference
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Parameter.description | Required when requires_grad=True | Enables generic optimization; forces documentation |
| output.backward(grad=loss_value) | On Value object | Mirrors PyTorch loss.backward() |
| ForwardRecord | In-memory | Like autograd tape; simple and fast |
| Mini-batch | Primary pattern | More robust updates; matches PyTorch |
| Optimizer aliases | Fixed (`optimizer/*`) | Simple, predictable resource config |
| backward() | Async | Consistent with framework; allows LLM reasoning |
| Value accumulation | List in Parameter | Handles both fan-out and mini-batch naturally |
| Value | Output carries record | Implicit flow; mirrors PyTorch autograd |
| Training mode | `module.train()` enables recording | PyTorch-like; clean separation of train/eval |
| Loss auto-extraction | Detects Value | No manual record passing needed |
| Loss.batch() | Default concurrent impl | Single-sample losses auto-parallelize |
| Value.backward() | Static method | Accumulation is append-only, safe to parallelize |
| Aggregation | LLM-based | Natural language requires LLM to synthesize |
