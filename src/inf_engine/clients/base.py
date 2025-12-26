"""Abstract base class for LLM clients.

This module defines the `LLMClient` abstract base class that all LLM provider
clients must implement. It provides a unified async interface for making
completion requests across different providers.
"""

from abc import ABC, abstractmethod

from inf_engine.types import LLMRequest, LLMResponse


class LLMClient(ABC):
    """Abstract base class for LLM clients.

    Defines the interface that all LLM provider clients must implement.
    Each provider (OpenAI, Anthropic, vLLM, etc.) has its own concrete
    implementation that translates the unified request format to the
    provider's specific API.

    The primary method is `complete()`, which takes an `LLMRequest` and
    returns an `LLMResponse`. All implementations must be async to support
    concurrent execution of multiple LLM calls.

    Subclasses should:
    - Implement `complete()` with provider-specific API calls
    - Handle provider-specific errors and translate them to common exceptions
    - Extract timing metrics if available from the provider

    Example:
        >>> class MockClient(LLMClient):
        ...     async def complete(self, request: LLMRequest) -> LLMResponse:
        ...         return LLMResponse(
        ...             content=f"Mock response to: {request.prompt}",
        ...             input_tokens=len(request.prompt.split()),
        ...             output_tokens=5,
        ...             finish_reason="stop",
        ...             model="mock-model",
        ...         )
        ...
        >>> import asyncio
        >>> client = MockClient()
        >>> request = LLMRequest(prompt="Hello, world!")
        >>> response = asyncio.run(client.complete(request))
        >>> response.content
        'Mock response to: Hello, world!'
    """

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a completion request and return the response.

        Takes a provider-agnostic `LLMRequest` and returns a provider-agnostic
        `LLMResponse`. Implementations handle the translation to and from
        the specific provider's API format.

        Args:
            request: The completion request containing the prompt, optional
                system message, temperature, max tokens, and other parameters.

        Returns:
            An `LLMResponse` containing the generated content, token counts,
            finish reason, model identifier, and optional timing metrics.

        Raises:
            RateLimitError: If the provider returns a rate limit error (429).
                The error should include `retry_after` if available.
            Exception: For other API errors (network issues, auth failures,
                invalid requests, etc.).

        Note:
            This method is async and should be awaited. Implementations should
            use async HTTP clients for non-blocking I/O.
        """
        pass
