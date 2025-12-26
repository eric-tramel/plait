"""LLM client implementations for various providers.

This module provides abstract and concrete client implementations for
communicating with LLM endpoints. The abstract `LLMClient` class defines
the interface that all provider-specific clients must implement.

Supported providers:
- OpenAI (via OpenAIClient)
- OpenAI-compatible (via OpenAICompatibleClient) for vLLM, TGI, etc.
- (More to come: Anthropic, Ollama, etc.)
"""

from inf_engine.clients.base import LLMClient
from inf_engine.clients.openai import OpenAIClient, RateLimitError

__all__ = ["LLMClient", "OpenAIClient", "RateLimitError"]
