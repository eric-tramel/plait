"""Tests for loss helper utilities."""

from unittest.mock import AsyncMock

import pytest

from plait.optimization.loss import (
    _flatten_payload,
    _LossLLMWrapper,
    _normalize_actions,
    _parse_json_list,
    _value_from_actions,
)
from plait.resources.config import ResourceConfig
from plait.values import ValueKind


def test_normalize_actions_variants() -> None:
    assert _normalize_actions(None) == []
    assert _normalize_actions("   ") == []
    assert _normalize_actions(" ok ") == ["ok"]
    assert _normalize_actions(["a", " ", 1]) == ["a", "1"]
    assert _normalize_actions(0) == ["0"]


def test_flatten_payload_and_value_from_actions() -> None:
    assert _flatten_payload([["a", ""], ["b"]]) == ["a", "b"]

    value = _value_from_actions(["a", "b"], score=0.5, meta={"source": "test"})
    assert value.kind == ValueKind.STRUCTURED
    assert value.payload == [["a", "b"]]
    assert value.meta["score"] == 0.5
    assert value.meta["source"] == "test"


def test_parse_json_list_variants() -> None:
    assert _parse_json_list('["a", "b"]') == ["a", "b"]
    assert _parse_json_list(["a", " "]) == ["a"]
    assert _parse_json_list("not json") is None
    assert _parse_json_list('{"a": 1}') is None
    assert _parse_json_list("[invalid") is None
    assert _parse_json_list(123) is None


@pytest.mark.asyncio
async def test_loss_llm_wrapper_properties_and_bind() -> None:
    wrapper = _LossLLMWrapper(alias="judge", system_prompt="test")
    assert wrapper.alias == "judge"
    assert wrapper.system_prompt is not None
    assert wrapper._bound_resources is None

    with pytest.raises(RuntimeError):
        await wrapper("hello")

    wrapper.bind(ResourceConfig(endpoints={}))
    assert wrapper._bound_resources is not None


@pytest.mark.asyncio
async def test_loss_llm_wrapper_forward_and_call() -> None:
    wrapper = _LossLLMWrapper(alias="judge", system_prompt="test")
    wrapper._module.llm = lambda prompt: f"echo:{prompt}"
    assert wrapper._module.forward("hi") == "echo:hi"

    wrapper._bound = True
    wrapper._module = AsyncMock(return_value="ok")
    result = await wrapper("hello")
    assert result == "ok"
