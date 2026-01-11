"""LLM client implementations for various providers.

This module provides abstract and concrete client implementations for
communicating with LLM endpoints. The abstract `LLMClient` class defines
the interface that all provider-specific clients must implement.

Supported providers:
- OpenAI (via OpenAIClient)
- OpenAI-compatible (via OpenAICompatibleClient) for vLLM, TGI, etc.
- Anthropic (via AnthropicClient) for Claude models
"""

from plait.clients.anthropic import AnthropicClient
from plait.clients.base import LLMClient
from plait.clients.openai import (
    OpenAIClient,
    OpenAICompatibleClient,
    RateLimitError,
)

__all__ = [
    "AnthropicClient",
    "LLMClient",
    "OpenAIClient",
    "OpenAICompatibleClient",
    "RateLimitError",
]
