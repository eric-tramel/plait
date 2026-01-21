"""Loss modules for evaluating module outputs.

Losses are Module instances that return Value feedback (structured payloads)
and can participate in tracing so loss outputs can call backward() directly.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from plait.module import Module
from plait.values import (
    Value,
    ValueKind,
    collect_tape_ids,
    normalize_feedback_payload,
    unwrap,
)

if False:  # pragma: no cover - type checking imports only
    pass


# =============================================================================
# Helpers
# =============================================================================


def _normalize_actions(actions: Any) -> list[str]:
    normalized = normalize_feedback_payload(actions)
    flattened = [item.strip() for inner in normalized for item in inner]
    return [item for item in flattened if item]


def _flatten_payload(payload: Any) -> list[str]:
    normalized = normalize_feedback_payload(payload)
    return [item for inner in normalized for item in inner]


def _value_from_actions(
    actions: list[str],
    *,
    score: float | None = None,
    meta: dict[str, Any] | None = None,
) -> Value:
    payload = normalize_feedback_payload(actions)
    meta = {} if meta is None else dict(meta)
    if score is not None:
        meta["score"] = score
    return Value(kind=ValueKind.STRUCTURED, payload=payload, meta=meta)


def _attach_attrs(fn: Any, **attrs: Any) -> Any:
    for key, value in attrs.items():
        setattr(fn, key, value)
    return fn


def _attach_tape_ids(value: Any, tape_ids: list[str]) -> None:
    if not tape_ids:
        return

    def _attach(obj: Any) -> None:
        if isinstance(obj, Value):
            ids = obj.meta.setdefault("_tape_ids", [])
            for tape_id in tape_ids:
                if tape_id not in ids:
                    ids.append(tape_id)
            _attach(obj.payload)
        elif isinstance(obj, dict):
            for item in obj.values():
                _attach(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _attach(item)

    _attach(value)


class LossModule(Module):
    """Module wrapper around a loss callable.

    The wrapped loss callable is responsible for producing a Value with
    structured feedback payloads. This wrapper:
    - Allows loss callables to participate as graph nodes
    - Propagates tape ids from input Values when invoked directly
    - Passes feedback back to the primary output input during backward()
    """

    _loss_fn: Callable[..., Any]
    _feedback_input: str | None

    def __init__(
        self,
        loss_fn: Callable[..., Any],
        *,
        feedback_input: str | None = "output",
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_loss_fn", loss_fn)
        object.__setattr__(self, "_feedback_input", feedback_input)

    async def forward(self, *args: Any, **kwargs: Any) -> Value:
        tape_ids = collect_tape_ids({"args": args, "kwargs": kwargs})
        result = self._loss_fn(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if tape_ids:
            _attach_tape_ids(result, tape_ids)
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the loss directly, avoiding nested runs during execution."""
        from plait.tracing.context import get_trace_context

        tracer = get_trace_context()
        if tracer is not None:
            return tracer.record_call(self, args, kwargs)
        return self.forward(*args, **kwargs)

    async def backward(self, feedback: Any, ctx: Any) -> Any:
        from plait.optimization.backward import BackwardResult
        from plait.values import normalize_feedback_value

        result = BackwardResult()
        normalized = normalize_feedback_value(feedback)

        target_key = None
        if self._feedback_input and self._feedback_input in ctx.inputs:
            target_key = self._feedback_input
        elif "arg_0" in ctx.inputs:
            target_key = "0"
        else:
            for name in ctx.inputs:
                if name.startswith("arg_") and name[4:].isdigit():
                    target_key = name[4:]
                    break
            if target_key is None and ctx.inputs:
                target_key = next(iter(ctx.inputs))

        if target_key is not None:
            result.input_feedback[target_key] = normalized

        return result

    def bind(
        self, resources: Any, max_concurrent: int = 100, **kwargs: Any
    ) -> LossModule:
        super().bind(resources, max_concurrent=max_concurrent, **kwargs)
        bind_method = getattr(self._loss_fn, "bind", None)
        if callable(bind_method):
            bind_method(resources)
        return self

    def __getattr__(self, name: str) -> Any:
        loss_fn = object.__getattribute__(self, "_loss_fn")
        return getattr(loss_fn, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_loss_fn", "_feedback_input"}:
            object.__setattr__(self, name, value)
            return
        loss_fn = self.__dict__.get("_loss_fn")
        if loss_fn is not None and hasattr(loss_fn, name):
            setattr(loss_fn, name, value)
        super().__setattr__(name, value)


def _parse_json_list(value: Any) -> list[str] | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            import json

            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, list):
                return _normalize_actions(parsed)
        return None
    if isinstance(value, list):
        return _normalize_actions(value)
    return None


# =============================================================================
# LLM Wrapper for Loss Functions
# =============================================================================


class _LossLLMWrapper:
    """Wrapper to make LLMInference callable as a bound module."""

    def __init__(
        self,
        alias: str,
        system_prompt: str,
        temperature: float = 0.0,
        response_format: type | None = None,
    ) -> None:
        from plait.module import LLMInference, Module

        self._alias = alias

        class _Wrapper(Module):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.llm = LLMInference(
                    alias=alias,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    response_format=response_format,
                )

            def forward(inner_self, prompt: str) -> Any:
                return inner_self.llm(prompt)

        self._module = _Wrapper()
        self._bound = False

    @property
    def alias(self) -> str:
        return self._alias

    @property
    def system_prompt(self) -> Any:
        return self._module.llm.system_prompt

    @property
    def _bound_resources(self) -> Any:
        return self._module.llm._bound_resources

    def bind(self, resources: Any) -> None:
        self._module.bind(resources)
        self._module.llm.bind(resources)
        self._bound = True

    async def __call__(self, prompt: str) -> Any:
        if not self._bound:
            raise RuntimeError("LLM wrapper not bound. Call bind() first.")
        return await self._module(prompt)


# =============================================================================
# Structured Output Schemas for LLM-based Losses
# =============================================================================


@dataclass
class RubricLevel:
    score: int
    label: str
    description: str


@dataclass
class RubricResponse:
    score: int
    justification: str
    actionable_improvements: list[str]


@dataclass
class PreferenceResponse:
    winner: Literal["A", "B"]
    reason: str
    a_strengths: str
    a_weaknesses: str
    b_strengths: str
    b_weaknesses: str


@dataclass
class RankingResponse:
    ranking: list[int]
    best_qualities: str
    worst_issues: str
    comparison: str


# =============================================================================
# Loss Factories
# =============================================================================


def VerifierLoss(
    verifier: Callable[[Any], tuple[bool, str]],
    success_feedback: str = "Output passed verification.",
) -> LossModule:
    async def loss_fn(
        output: Any,
        target: Any | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> Value:
        _ = target, context
        actual_output = unwrap(output)
        passed, message = verifier(actual_output)
        actions = [] if passed else _normalize_actions(message)
        meta = {"verifier": True}
        return _value_from_actions(actions, score=1.0 if passed else 0.0, meta=meta)

    loss_fn = _attach_attrs(
        loss_fn, verifier=verifier, success_feedback=success_feedback
    )
    return LossModule(loss_fn)


def LLMJudge(
    alias: str = "judge",
    criteria: str | None = None,
) -> LossModule:
    wrapper = _LossLLMWrapper(
        alias=alias,
        system_prompt=(
            "You are a critical reviewer. Provide ONLY actionable "
            "improvements (no general reviews or praise). Respond with a "
            "JSON array of strings. If no improvement is needed, respond "
            "with []"
        ),
    )

    async def loss_fn(
        output: Any,
        target: Any | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> Value:
        actual_output = unwrap(output)
        actual_target = unwrap(target) if target is not None else None

        prompt_parts = [f"Output to critique:\n{actual_output}"]
        if context:
            prompt_parts.append(f"Context: {context}")
        if actual_target:
            prompt_parts.append(f"Expected behavior: {actual_target}")
        if criteria:
            prompt_parts.append(f"Focus areas: {criteria}")
        prompt_parts.append(
            "\nProvide ONLY actionable improvements as a JSON array of strings. "
            "If no improvement is needed, return []."
        )
        prompt = "\n\n".join(prompt_parts)

        loss_any = cast(Any, loss_fn)
        judge = loss_any.judge
        response = await judge(prompt)
        actions = _parse_json_list(response)
        if actions is None:
            actions = _normalize_actions(response)

        return _value_from_actions(actions)

    def bind(resources: Any) -> Any:
        loss_any = cast(Any, loss_fn)
        judge = loss_any.judge
        judge.bind(resources)
        return loss_fn

    loss_fn = _attach_attrs(loss_fn, bind=bind, judge=wrapper, criteria=criteria)
    return LossModule(loss_fn)


def CompositeLoss(
    losses: list[tuple[Callable[..., Any], float]],
    aggregator: Any | None = None,
) -> LossModule:
    async def loss_fn(
        output: Any,
        target: Any | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> Value:
        total_weight = 0.0
        weighted_score = 0.0
        collected: list[tuple[Value, float]] = []

        for loss, weight in losses:
            value = await loss(output, target, context=context)
            collected.append((value, weight))
            score = value.meta.get("score") if isinstance(value, Value) else None
            if score is not None:
                weighted_score += float(score) * weight
                total_weight += weight

        if aggregator is not None:
            prompt = (
                "Synthesize the following items into a concise list of actionable "
                "improvements:\n\n"
            )
            for fb, weight in collected:
                prompt += f"--- Feedback (weight: {weight}) ---\n{fb.payload}\n\n"
            prompt += (
                "Provide ONLY actionable improvements (no general reviews). If no "
                "improvements are needed, return []."
            )
            response = await aggregator(prompt)
            actions = _parse_json_list(response)
            if actions is None:
                actions = _normalize_actions(response)
            combined = _value_from_actions(actions)
        else:
            merged_actions: list[str] = []
            for fb, weight in collected:
                actions = _flatten_payload(fb.payload)
                if actions:
                    merged_actions.append(f"[Weight: {weight}] " + " | ".join(actions))
            combined = _value_from_actions(merged_actions)

        if total_weight > 0:
            combined.meta["score"] = weighted_score / total_weight
        return combined

    def bind(resources: Any) -> Any:
        for loss, _ in losses:
            bind_method = getattr(loss, "bind", None)
            if callable(bind_method):
                bind_method(resources)
        if aggregator is not None:
            aggregator.bind(resources)
        return loss_fn

    loss_fn = _attach_attrs(loss_fn, bind=bind, losses=losses, aggregator=aggregator)
    return LossModule(loss_fn)


def HumanFeedbackLoss(
    prompt_template: str | None = None,
    show_context: bool = True,
) -> LossModule:
    async def loss_fn(
        output: Any,
        target: Any | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> Value:
        actual_output = unwrap(output)
        actual_target = unwrap(target) if target is not None else None

        print("\n" + "=" * 60)
        print("OUTPUT TO EVALUATE:")
        print("-" * 60)
        print(actual_output)

        if show_context:
            if actual_target:
                print("-" * 60)
                print(f"Expected: {actual_target}")
            if context:
                print(f"Context: {context}")

        print("=" * 60)

        if prompt_template:
            prompt = prompt_template.format(
                output=actual_output, target=actual_target, context=context
            )
            print(prompt)
        else:
            print("Please provide actionable improvements only.")
            print(
                "Enter one improvement per line. If no improvement is needed, "
                "press Enter on an empty line."
            )

        lines = []
        while True:
            line = input("> ")
            if not line:
                break
            lines.append(line)

        actions = _normalize_actions(lines)
        return _value_from_actions(actions)

    loss_fn = _attach_attrs(
        loss_fn, prompt_template=prompt_template, show_context=show_context
    )
    return LossModule(loss_fn)


def LLMRubricLoss(
    criteria: str,
    rubric: list[RubricLevel],
    alias: str = "judge",
) -> LossModule:
    rubric = sorted(rubric, key=lambda r: r.score)
    max_score = max(r.score for r in rubric)
    min_score = min(r.score for r in rubric)

    rubric_text = "\n".join(
        f"  {level.score} - {level.label}: {level.description}" for level in rubric
    )
    system_prompt = f"""You evaluate outputs against a rubric and respond in JSON format.

Criteria: {criteria}

Rating Scale:
{rubric_text}

You must respond with a JSON object containing exactly these fields:
- \"score\": integer from the rating scale above
- \"justification\": string explaining why you assigned this score
- \"actionable_improvements\": array of strings with ONLY actionable improvements.
  If no improvement is needed, return an empty array []"""

    judge = _LossLLMWrapper(
        alias=alias,
        system_prompt=system_prompt,
        response_format=RubricResponse,
    )
    fallback_judge = _LossLLMWrapper(alias=alias, system_prompt=system_prompt)

    def _normalize_actionable_improvements(value: Any) -> list[str]:
        parsed = _parse_json_list(value)
        if parsed is not None:
            return parsed
        return _normalize_actions(value)

    def _parse_response(payload: Any) -> tuple[float, str, list[str]]:
        if isinstance(payload, str):
            import json

            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "LLMRubricLoss expected JSON but received non-JSON response."
                ) from exc

        if isinstance(payload, dict):
            normalized = {
                str(key).strip().lower(): value for key, value in payload.items()
            }
            raw_score = normalized.get("score")
            if raw_score is None:
                for key in ("rating", "grade", "score_value", "overall_score"):
                    if key in normalized:
                        raw_score = normalized[key]
                        break
            if raw_score is None:
                raise ValueError(
                    "LLMRubricLoss expected a 'score' field in the judge response."
                )
            if isinstance(raw_score, str):
                try:
                    raw_score = int(raw_score)
                except ValueError:
                    raw_score = float(raw_score)

            justification = normalized.get("justification", "No justification provided")
            improvements_raw = normalized.get("actionable_improvements")
            if improvements_raw is None:
                for key in (
                    "actionableimprovements",
                    "actionable_improvement",
                    "actionableimprovement",
                    "improvements",
                    "suggestions",
                    "recommendations",
                    "feedback",
                ):
                    if key in normalized:
                        improvements_raw = normalized[key]
                        break
            actionable_improvements = _normalize_actionable_improvements(
                improvements_raw
            )
            return float(raw_score), str(justification), actionable_improvements

        improvements_value = getattr(payload, "actionable_improvements", None)
        if improvements_value is None and hasattr(payload, "feedback"):
            improvements_value = payload.feedback
        return (
            float(payload.score),
            str(payload.justification),
            _normalize_actionable_improvements(improvements_value),
        )

    async def loss_fn(
        output: Any,
        target: Any | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> Value:
        actual_output = unwrap(output)
        actual_target = unwrap(target) if target is not None else None

        prompt_parts = [f"Output to evaluate:\n{actual_output}"]
        if actual_target:
            prompt_parts.append(f"Expected/Target: {actual_target}")
        prompt = "\n\n".join(prompt_parts)

        loss_any = cast(Any, loss_fn)
        judge = loss_any.judge
        response = await judge(prompt)
        try:
            raw_score, justification, actions = _parse_response(response)
        except ValueError:
            retry_prompt = (
                f"{prompt}\n\nReturn a JSON object with keys: "
                "score, justification, actionable_improvements. "
                "Do not include any other text."
            )
            loss_any = cast(Any, loss_fn)
            fallback_judge = loss_any._fallback_judge
            fallback_response = await fallback_judge(retry_prompt)
            raw_score, justification, actions = _parse_response(fallback_response)

        normalized_score = (raw_score - min_score) / (max_score - min_score)
        meta = {
            "score": normalized_score,
            "raw_score": raw_score,
            "criteria": criteria,
            "justification": justification,
        }
        return _value_from_actions(actions, meta=meta)

    def bind(resources: Any) -> Any:
        loss_any = cast(Any, loss_fn)
        judge = loss_any.judge
        fallback_judge = loss_any._fallback_judge
        judge.bind(resources)
        fallback_judge.bind(resources)
        return loss_fn

    loss_fn = _attach_attrs(
        loss_fn,
        bind=bind,
        criteria=criteria,
        rubric=rubric,
        judge=judge,
        _fallback_judge=fallback_judge,
        _max_score=max_score,
        _min_score=min_score,
    )
    return LossModule(loss_fn)


def HumanRubricLoss(
    criteria: str,
    rubric: list[RubricLevel],
    require_feedback: bool = True,
) -> LossModule:
    rubric = sorted(rubric, key=lambda r: r.score)
    max_score = max(r.score for r in rubric)
    min_score = min(r.score for r in rubric)

    async def loss_fn(
        output: Any,
        target: Any | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> Value:
        actual_output = unwrap(output)
        actual_target = unwrap(target) if target is not None else None

        print("\n" + "=" * 60)
        print(f"EVALUATE: {criteria}")
        print("=" * 60)
        print("\nOutput:")
        print("-" * 40)
        print(actual_output)
        print("-" * 40)

        if actual_target:
            print(f"\nExpected: {actual_target}")

        print("\nRating Scale:")
        for level in rubric:
            print(f"  [{level.score}] {level.label}: {level.description}")

        while True:
            try:
                score_input = input(f"\nYour score ({min_score}-{max_score}): ")
                score = int(score_input)
                if min_score <= score <= max_score:
                    break
                print(f"Please enter a number between {min_score} and {max_score}")
            except ValueError:
                print("Please enter a valid number")

        actions: list[str] = []
        if require_feedback:
            print("\nProvide actionable improvements only (empty line to finish):")
            lines = []
            while True:
                line = input("> ")
                if not line:
                    break
                lines.append(line)
            actions = _normalize_actions(lines)

        normalized_score = (score - min_score) / (max_score - min_score)
        return _value_from_actions(
            actions, meta={"score": normalized_score, "raw_score": score}
        )

    loss_fn = _attach_attrs(
        loss_fn,
        criteria=criteria,
        rubric=rubric,
        require_feedback=require_feedback,
        _max_score=max_score,
        _min_score=min_score,
    )
    return LossModule(loss_fn)


# =============================================================================
# Contrastive helpers and losses
# =============================================================================


def _summarize_output(output: Any) -> str:
    text = str(output)
    if len(text) > 200:
        return text[:200] + "..."
    return text


def _generate_contrastive_feedback(winner: Any, loser: Any, reason: str) -> str:
    return (
        "The preferred output was better because: "
        f"{reason}\n\n"
        "To improve, the output should:\n"
        "- Emulate qualities of the preferred response\n"
        "- Avoid weaknesses identified in the rejected response\n\n"
        "Preferred output characteristics:\n"
        f"{_summarize_output(winner)}\n\n"
        "Rejected output weaknesses:\n"
        f"{_summarize_output(loser)}"
    )


def LLMPreferenceLoss(
    criteria: str,
    alias: str = "judge",
) -> LossModule:
    judge = _LossLLMWrapper(
        alias=alias,
        system_prompt=(
            f"You compare two outputs on: {criteria}. "
            "Determine which is better and respond in JSON format with these fields:\n"
            '- "winner": either "A" or "B"\n'
            '- "reason": string explaining why the winner is better\n'
            '- "a_strengths": string listing strengths of output A\n'
            '- "a_weaknesses": string listing weaknesses of output A\n'
            '- "b_strengths": string listing strengths of output B\n'
            '- "b_weaknesses": string listing weaknesses of output B'
        ),
        response_format=PreferenceResponse,
    )

    async def compare(
        output_a: Any,
        output_b: Any,
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[Value, Value]:
        prompt_parts = []
        if context:
            prompt_parts.append(f"Context: {context}")
        prompt_parts.append(f"Output A:\n{unwrap(output_a)}")
        prompt_parts.append(f"Output B:\n{unwrap(output_b)}")
        prompt_parts.append("Which output is better?")
        prompt = "\n\n".join(prompt_parts)

        loss_any = cast(Any, loss_fn)
        judge = loss_any.judge
        response = await judge(prompt)
        if isinstance(response, str):
            import json

            response = json.loads(response)

        if isinstance(response, dict):
            winner = response["winner"]
            reason = response["reason"]
            a_strengths = response["a_strengths"]
            a_weaknesses = response["a_weaknesses"]
            b_strengths = response["b_strengths"]
            b_weaknesses = response["b_weaknesses"]
        else:
            winner = response.winner
            reason = response.reason
            a_strengths = response.a_strengths
            a_weaknesses = response.a_weaknesses
            b_strengths = response.b_strengths
            b_weaknesses = response.b_weaknesses

        if winner == "A":
            fb_a = _value_from_actions(
                _normalize_actions(f"Preferred. Strengths: {a_strengths}"),
                score=1.0,
            )
            fb_b = _value_from_actions(
                _normalize_actions(
                    _generate_contrastive_feedback(
                        output_a,
                        output_b,
                        f"{reason}\n\nWeaknesses: {b_weaknesses}",
                    )
                ),
                score=0.0,
            )
        else:
            fb_a = _value_from_actions(
                _normalize_actions(
                    _generate_contrastive_feedback(
                        output_b,
                        output_a,
                        f"{reason}\n\nWeaknesses: {a_weaknesses}",
                    )
                ),
                score=0.0,
            )
            fb_b = _value_from_actions(
                _normalize_actions(f"Preferred. Strengths: {b_strengths}"),
                score=1.0,
            )
        return fb_a, fb_b

    async def loss_fn(
        output: Any,
        target: Any | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> Value:
        if target is None:
            raise ValueError("LLMPreferenceLoss requires target for comparison")
        fb_output, _ = await compare(output, target, context=context)
        return fb_output

    def bind(resources: Any) -> Any:
        loss_any = cast(Any, loss_fn)
        judge = loss_any.judge
        judge.bind(resources)
        return loss_fn

    loss_fn = _attach_attrs(
        loss_fn,
        bind=bind,
        compare=compare,
        criteria=criteria,
        judge=judge,
    )
    return LossModule(loss_fn)


def HumanPreferenceLoss(
    criteria: str,
    require_reason: bool = True,
) -> LossModule:
    async def compare(
        output_a: Any,
        output_b: Any,
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[Value, Value]:
        print("\n" + "=" * 60)
        print(f"COMPARE: {criteria}")
        print("=" * 60)

        if context:
            print(f"\nContext: {context}")

        print("\n[A] Output A:")
        print("-" * 40)
        print(unwrap(output_a))

        print("\n[B] Output B:")
        print("-" * 40)
        print(unwrap(output_b))

        print("=" * 60)

        while True:
            choice = input("\nWhich is better? (A/B): ").strip().upper()
            if choice in ("A", "B"):
                break
            print("Please enter A or B")

        reason = ""
        if require_reason:
            print("\nWhy is it better? (enter empty line to finish):")
            lines = []
            while True:
                line = input("> ")
                if not line:
                    break
                lines.append(line)
            reason = "\n".join(lines)

        winner, loser = (output_a, output_b) if choice == "A" else (output_b, output_a)

        if choice == "A":
            fb_a = _value_from_actions(
                _normalize_actions(f"Preferred by human. Reason: {reason}"),
                score=1.0,
            )
            fb_b = _value_from_actions(
                _normalize_actions(
                    _generate_contrastive_feedback(winner, loser, reason)
                ),
                score=0.0,
            )
        else:
            fb_a = _value_from_actions(
                _normalize_actions(
                    _generate_contrastive_feedback(winner, loser, reason)
                ),
                score=0.0,
            )
            fb_b = _value_from_actions(
                _normalize_actions(f"Preferred by human. Reason: {reason}"),
                score=1.0,
            )
        return fb_a, fb_b

    async def loss_fn(
        output: Any,
        target: Any | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> Value:
        if target is None:
            raise ValueError("HumanPreferenceLoss requires target for comparison")
        fb_output, _ = await compare(output, target, context=context)
        return fb_output

    loss_fn = _attach_attrs(
        loss_fn,
        compare=compare,
        criteria=criteria,
        require_reason=require_reason,
    )
    return LossModule(loss_fn)


def LLMRankingLoss(
    criteria: str,
    n: int = 4,
    alias: str = "judge",
) -> LossModule:
    judge = _LossLLMWrapper(
        alias=alias,
        system_prompt=(
            f"You rank outputs from best to worst on: {criteria}. "
            "Respond in JSON format with these fields:\n"
            '- "ranking": array of indices in order from best to worst (e.g., [2, 0, 1, 3])\n'
            '- "reasoning": string explaining the ranking decisions\n'
            '- "per_output_actionable_improvements": array of arrays of strings. '
            "Each inner array contains ONLY actionable improvements for that output "
            "(use [] when no improvements are needed)."
        ),
        response_format=RankingResponse,
    )

    async def rank(
        outputs: list[Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[Value]:
        if len(outputs) < 2:
            raise ValueError("Need at least 2 outputs to rank")

        output_strs = [f"[{i + 1}] {unwrap(out)}" for i, out in enumerate(outputs)]

        prompt_parts = []
        if context:
            prompt_parts.append(f"Context: {context}")
        prompt_parts.append("Outputs to rank:")
        prompt_parts.append("\n".join(output_strs))
        prompt_parts.append(f"Rank these {len(outputs)} outputs from BEST to WORST.")
        prompt = "\n\n".join(prompt_parts)

        loss_any = cast(Any, loss_fn)
        judge = loss_any.judge
        response = await judge(prompt)
        if isinstance(response, str):
            import json

            response = json.loads(response)

        if isinstance(response, dict):
            raw_ranking = response["ranking"]
            best_qualities = response.get("best_qualities") or response.get("reasoning")
            worst_issues = response.get("worst_issues") or response.get("reasoning")
            comparison = response.get("comparison") or response.get("reasoning")
            per_output = response.get("per_output_actionable_improvements")
        else:
            raw_ranking = response.ranking
            best_qualities = response.best_qualities
            worst_issues = response.worst_issues
            comparison = response.comparison
            per_output = getattr(response, "per_output_actionable_improvements", None)

        ranking = [r - 1 for r in raw_ranking]
        feedbacks: list[Value] = []
        total = len(outputs)

        for i in range(total):
            rank_position = ranking.index(i) + 1
            score = (total - rank_position) / (total - 1) if total > 1 else 1.0
            if per_output and i < len(per_output):
                actions = _normalize_actions(per_output[i])
            else:
                if rank_position == 1:
                    actions = _normalize_actions(f"Ranked #1 (best). {best_qualities}")
                elif rank_position == total:
                    actions = _normalize_actions(
                        f"Ranked #{rank_position} (worst). {worst_issues}"
                    )
                else:
                    actions = _normalize_actions(
                        f"Ranked #{rank_position}/{total}. {comparison}"
                    )
            fb = _value_from_actions(
                actions, meta={"score": score, "rank": rank_position, "total": total}
            )
            feedbacks.append(fb)

        return feedbacks

    async def loss_fn(
        output: Any,
        target: Any | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> Value:
        if target is None:
            raise ValueError("LLMRankingLoss requires target for comparison")
        targets = target if isinstance(target, list) else [target]
        outputs = [output] + targets
        feedbacks = await rank(outputs, context=context)
        return feedbacks[0]

    def bind(resources: Any) -> Any:
        loss_any = cast(Any, loss_fn)
        judge = loss_any.judge
        judge.bind(resources)
        return loss_fn

    loss_fn = _attach_attrs(
        loss_fn,
        bind=bind,
        rank=rank,
        criteria=criteria,
        n=n,
        judge=judge,
    )
    return LossModule(loss_fn)


def HumanRankingLoss(
    criteria: str,
    n: int = 4,
    require_feedback: bool = True,
) -> LossModule:
    async def rank(
        outputs: list[Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[Value]:
        if len(outputs) < 2:
            raise ValueError("Need at least 2 outputs to rank")

        print("\n" + "=" * 60)
        print(f"RANK: {criteria}")
        print("=" * 60)
        if context:
            print(f"\nContext: {context}")

        for idx, out in enumerate(outputs, start=1):
            print(f"\n[{idx}] Output {idx}:")
            print("-" * 40)
            print(unwrap(out))

        print("=" * 60)

        indices = list(range(1, len(outputs) + 1))
        while True:
            ranking_input = input("\nEnter ranking (e.g., 2,1,3): ").strip()
            try:
                ranking = [
                    int(x.strip()) for x in ranking_input.split(",") if x.strip()
                ]
            except ValueError:
                ranking = []
            if sorted(ranking) == indices:
                break
            print("Please enter a valid ranking with each output listed once.")

        feedback_text = ""
        if require_feedback:
            print("\nProvide any overall feedback (empty line to finish):")
            lines = []
            while True:
                line = input("> ")
                if not line:
                    break
                lines.append(line)
            feedback_text = "\n".join(lines)

        total = len(outputs)
        feedbacks: list[Value] = []
        for i in range(total):
            rank_position = ranking.index(i + 1) + 1
            score = (total - rank_position) / (total - 1) if total > 1 else 1.0
            if feedback_text:
                actions = _normalize_actions(feedback_text)
            else:
                if rank_position == 1:
                    actions = _normalize_actions("Ranked #1 (best).")
                elif rank_position == total:
                    actions = _normalize_actions(f"Ranked #{rank_position} (worst).")
                else:
                    actions = _normalize_actions(f"Ranked #{rank_position}/{total}.")
            fb = _value_from_actions(
                actions, meta={"score": score, "rank": rank_position, "total": total}
            )
            feedbacks.append(fb)
        return feedbacks

    async def loss_fn(
        output: Any,
        target: Any | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> Value:
        if target is None:
            raise ValueError("HumanRankingLoss requires target for comparison")
        targets = target if isinstance(target, list) else [target]
        outputs = [output] + targets
        feedbacks = await rank(outputs, context=context)
        return feedbacks[0]

    loss_fn = _attach_attrs(
        loss_fn,
        rank=rank,
        criteria=criteria,
        n=n,
        require_feedback=require_feedback,
    )
    return LossModule(loss_fn)


__all__ = [
    "LossModule",
    "RubricLevel",
    "RubricResponse",
    "PreferenceResponse",
    "RankingResponse",
    "VerifierLoss",
    "LLMJudge",
    "LLMRubricLoss",
    "HumanRubricLoss",
    "HumanFeedbackLoss",
    "CompositeLoss",
    "LLMPreferenceLoss",
    "HumanPreferenceLoss",
    "LLMRankingLoss",
    "HumanRankingLoss",
]
