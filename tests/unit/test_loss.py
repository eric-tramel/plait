"""Tests for loss module API returning Value objects."""

from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from plait.optimization.loss import (
    CompositeLoss,
    HumanFeedbackLoss,
    HumanPreferenceLoss,
    HumanRankingLoss,
    HumanRubricLoss,
    LLMJudge,
    LLMPreferenceLoss,
    LLMRankingLoss,
    LLMRubricLoss,
    RubricLevel,
    VerifierLoss,
)
from plait.values import Value, ValueKind


@pytest.mark.asyncio
async def test_verifier_loss_pass_and_fail() -> None:
    def verifier(x: str) -> tuple[bool, str]:
        return (x == "ok", "bad")

    loss_fn = VerifierLoss(verifier)

    ok_val = await loss_fn("ok")
    assert ok_val.meta["score"] == 1.0
    assert ok_val.payload == []

    bad_val = await loss_fn("no")
    assert bad_val.meta["score"] == 0.0
    assert bad_val.payload == [["bad"]]


@pytest.mark.asyncio
async def test_loss_module_propagates_tape_ids() -> None:
    loss_fn = VerifierLoss(lambda x: (False, "bad"))
    value = Value(ValueKind.TEXT, "ok", meta={"_tape_ids": ["tape123"]})

    loss_val = await loss_fn(value)
    assert "tape123" in loss_val.meta.get("_tape_ids", [])


@pytest.mark.asyncio
async def test_llm_judge_loss_parses_json_list() -> None:
    loss_fn = cast(Any, LLMJudge(criteria="clarity"))
    loss_fn.judge = AsyncMock(return_value='["Improve tone", "Add examples"]')

    value = await loss_fn("output")
    assert value.payload == [["Improve tone", "Add examples"]]


@pytest.mark.asyncio
async def test_composite_loss_weighted_score() -> None:
    loss_a = VerifierLoss(lambda x: (False, "A"))
    loss_b = VerifierLoss(lambda x: (True, "B"))
    composite = CompositeLoss([(loss_a, 0.25), (loss_b, 0.75)])

    value = await composite("out")
    assert value.meta["score"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_llm_rubric_loss_parses_response() -> None:
    rubric = [
        RubricLevel(1, "Poor", "bad"),
        RubricLevel(5, "Great", "good"),
    ]
    loss_fn = cast(Any, LLMRubricLoss(criteria="quality", rubric=rubric))
    loss_fn.judge = AsyncMock(
        return_value={
            "score": 5,
            "justification": "Great",
            "actionable_improvements": ["None"],
        }
    )

    value = await loss_fn("output")
    assert value.meta["score"] == 1.0
    assert value.payload == [["None"]]


@pytest.mark.asyncio
async def test_llm_preference_loss_compare() -> None:
    loss_fn = cast(Any, LLMPreferenceLoss(criteria="quality"))
    loss_fn.judge = AsyncMock(
        return_value={
            "winner": "A",
            "reason": "More accurate",
            "a_strengths": "Accurate",
            "a_weaknesses": "",
            "b_strengths": "",
            "b_weaknesses": "Inaccurate",
        }
    )

    fb_a, fb_b = await loss_fn.compare("A", "B")
    assert fb_a.meta["score"] == 1.0
    assert fb_b.meta["score"] == 0.0


@pytest.mark.asyncio
async def test_llm_ranking_loss_rank() -> None:
    loss_fn = cast(Any, LLMRankingLoss(criteria="quality"))
    loss_fn.judge = AsyncMock(
        return_value={
            "ranking": [2, 1, 3],
            "best_qualities": "Best",
            "worst_issues": "Worst",
            "comparison": "Mixed",
        }
    )

    feedbacks = await loss_fn.rank(["A", "B", "C"])
    assert feedbacks[1].meta["score"] == 1.0
    assert feedbacks[0].meta["score"] == 0.5
    assert feedbacks[2].meta["score"] == 0.0


@pytest.mark.asyncio
async def test_human_feedback_loss_collects_lines() -> None:
    loss_fn = cast(Any, HumanFeedbackLoss())
    with patch("builtins.input", side_effect=["Line 1", "Line 2", ""]):
        with patch("builtins.print"):
            value = await loss_fn("output")

    assert value.payload == [["Line 1", "Line 2"]]


@pytest.mark.asyncio
async def test_human_rubric_loss_collects_score_and_feedback() -> None:
    rubric = [RubricLevel(1, "Low", "bad"), RubricLevel(2, "High", "good")]
    loss_fn = cast(
        Any, HumanRubricLoss(criteria="quality", rubric=rubric, require_feedback=True)
    )

    with patch("builtins.input", side_effect=["2", "Improve", ""]):
        with patch("builtins.print"):
            value = await loss_fn("output")

    assert value.meta["score"] == 1.0
    assert value.payload == [["Improve"]]


@pytest.mark.asyncio
async def test_human_preference_loss_compare() -> None:
    loss_fn = cast(Any, HumanPreferenceLoss(criteria="quality", require_reason=False))
    with patch("builtins.input", side_effect=["A"]):
        with patch("builtins.print"):
            fb_a, fb_b = await loss_fn.compare("A", "B")

    assert fb_a.meta["score"] == 1.0
    assert fb_b.meta["score"] == 0.0


@pytest.mark.asyncio
async def test_human_ranking_loss_rank() -> None:
    loss_fn = cast(Any, HumanRankingLoss(criteria="quality", require_feedback=False))
    with patch("builtins.input", side_effect=["2,1"]):
        with patch("builtins.print"):
            feedbacks = await loss_fn.rank(["A", "B"])

    assert feedbacks[1].meta["score"] == 1.0
    assert feedbacks[0].meta["score"] == 0.0
