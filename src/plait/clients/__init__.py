"""LLM client implementations for various providers.

This module provides abstract and concrete client implementations for
communicating with LLM endpoints. The abstract `LLMClient` class defines
the interface that all provider-specific clients must implement.

Supported providers:
- OpenAI (via OpenAIClient)
- OpenAI-compatible (via OpenAICompatibleClient) for vLLM, TGI, etc.
- (More to come: Anthropic, Ollama, etc.)
"""

from plait.clients.base import LLMClient
from plait.clients.openai import OpenAIClient, OpenAICompatibleClient
from plait.errors import RateLimitError

__all__ = ["LLMClient", "OpenAIClient", "OpenAICompatibleClient", "RateLimitError"]
