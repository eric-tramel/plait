"""OpenAI API client implementations.

This module provides client classes for OpenAI and OpenAI-compatible APIs:

- `OpenAIClient`: For the official OpenAI API
- `OpenAICompatibleClient`: For self-hosted models using OpenAI-compatible APIs
  (vLLM, TGI, Ollama, etc.)

Both clients implement the `LLMClient` interface for unified access.
"""

import os
from typing import Any

import openai
import openai.types.chat

from plait.clients.base import LLMClient
from plait.types import LLMRequest, LLMResponse


class RateLimitError(Exception):
    """Raised when the API returns a rate limit error (429).

    This exception wraps rate limit errors from providers and includes
    the retry-after hint when available. The scheduler can use this
    information to delay and requeue the task.

    Args:
        retry_after: Optional number of seconds to wait before retrying.
            May be None if the provider did not include this header.
        message: Optional error message. Defaults to a generic message.

    Example:
        >>> try:
        ...     await client.complete(request)
        ... except RateLimitError as e:
        ...     if e.retry_after:
        ...         await asyncio.sleep(e.retry_after)
    """

    def __init__(
        self,
        retry_after: float | None = None,
        message: str = "Rate limit exceeded",
    ):
        super().__init__(message)
        self.retry_after = retry_after


class OpenAIClient(LLMClient):
    """Client for the OpenAI API.

    Implements the `LLMClient` interface for making async completion requests
    to OpenAI's chat completions endpoint. Supports all standard parameters
    including tools, response format, and custom base URLs.

    Args:
        model: The model identifier to use (e.g., "gpt-4o", "gpt-4o-mini").
        base_url: Optional custom base URL for the API. If None, uses the
            default OpenAI endpoint. Useful for Azure OpenAI or proxies.
        api_key: Optional API key. If None, reads from OPENAI_API_KEY
            environment variable.
        timeout: Request timeout in seconds. Defaults to 300.0 (5 minutes).

    Example:
        >>> client = OpenAIClient(model="gpt-4o-mini")
        >>> request = LLMRequest(prompt="Hello, world!")
        >>> response = await client.complete(request)
        >>> print(response.content)
        'Hello! How can I help you today?'

        >>> # With custom endpoint
        >>> client = OpenAIClient(
        ...     model="gpt-4o",
        ...     base_url="https://my-proxy.example.com/v1",
        ...     api_key="sk-...",
        ... )
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
    ):
        self.model = model
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a completion request against the OpenAI API.

        Translates the provider-agnostic `LLMRequest` to OpenAI's chat
        completions format and returns a provider-agnostic `LLMResponse`.

        Args:
            request: The completion request containing prompt and parameters.

        Returns:
            An `LLMResponse` with the generated content and metadata.

        Raises:
            RateLimitError: If OpenAI returns a 429 rate limit error.
                Includes `retry_after` if the header was provided.
            openai.APIError: For other API errors (auth, network, etc.).

        Note:
            This method builds the messages list from the request, including
            the system prompt if provided. Tool calls are extracted from the
            response when the model requests them.
        """
        messages = self._build_messages(request)
        kwargs = self._build_request_kwargs(request)

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )

            return self._parse_response(response)

        except openai.RateLimitError as e:
            retry_after = self._extract_retry_after(e)
            raise RateLimitError(retry_after=retry_after) from e

    def _build_messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Build the messages list from the request.

        Args:
            request: The completion request.

        Returns:
            A list of message dicts in OpenAI's format.
        """
        messages: list[dict[str, Any]] = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.prompt})

        return messages

    def _build_request_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        """Build optional kwargs for the API call.

        Args:
            request: The completion request.

        Returns:
            A dict of optional parameters to pass to the API.
        """
        kwargs: dict[str, Any] = {
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        if request.stop is not None:
            kwargs["stop"] = request.stop

        if request.tools is not None:
            kwargs["tools"] = [
                {"type": "function", "function": tool} for tool in request.tools
            ]

        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice

        if request.response_format is not None:
            kwargs["response_format"] = {"type": "json_object"}

        if request.extra_body is not None:
            kwargs["extra_body"] = request.extra_body

        return kwargs

    def _parse_response(
        self, response: openai.types.chat.ChatCompletion
    ) -> LLMResponse:
        """Parse the OpenAI response into an LLMResponse.

        Args:
            response: The raw OpenAI chat completion response.

        Returns:
            A provider-agnostic LLMResponse.
        """
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls if present
        tool_calls: list[dict[str, Any]] | None = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                # Standard tool calls have a function attribute
                func = getattr(tc, "function", None)
                if func is not None:
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "name": func.name,
                            "arguments": func.arguments,
                        }
                    )

        # Determine finish reason
        finish_reason = choice.finish_reason or "unknown"

        # Get usage info
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            content=message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            model=response.model,
            tool_calls=tool_calls,
        )

    def _extract_retry_after(self, error: openai.RateLimitError) -> float | None:
        """Extract the retry-after value from a rate limit error.

        Args:
            error: The OpenAI rate limit error.

        Returns:
            The retry-after value in seconds, or None if not available.
        """
        if hasattr(error, "response") and error.response is not None:
            retry_header = error.response.headers.get("retry-after")
            if retry_header:
                try:
                    return float(retry_header)
                except ValueError:
                    pass
        return None


class OpenAICompatibleClient(OpenAIClient):
    """Client for OpenAI-compatible APIs (vLLM, TGI, Ollama, etc.).

    This client is designed for self-hosted models that expose an
    OpenAI-compatible API. It inherits all functionality from `OpenAIClient`
    but simplifies configuration for local/internal deployments:

    - `base_url` is required (no default OpenAI endpoint)
    - `api_key` defaults to "not-needed" (most self-hosted servers don't require auth)

    Common use cases:
    - vLLM servers: High-throughput serving of open-source models
    - TGI (Text Generation Inference): HuggingFace's inference server
    - Ollama: Local model runner for development
    - LiteLLM: Unified API gateway for multiple providers

    Args:
        model: The model identifier as configured on the server.
        base_url: The base URL of the OpenAI-compatible API endpoint.
            Must include the path (e.g., "http://localhost:8000/v1").
        api_key: Optional API key. Defaults to "not-needed" since most
            self-hosted servers don't require authentication.
        timeout: Request timeout in seconds. Defaults to 300.0 (5 minutes).

    Example:
        >>> # Connect to a local vLLM server
        >>> client = OpenAICompatibleClient(
        ...     model="mistral-7b",
        ...     base_url="http://localhost:8000/v1",
        ... )
        >>> request = LLMRequest(prompt="Hello!")
        >>> response = await client.complete(request)

        >>> # Connect to an internal TGI server
        >>> client = OpenAICompatibleClient(
        ...     model="llama-70b",
        ...     base_url="http://tgi-server.internal:8080/v1",
        ...     timeout=600.0,  # Longer timeout for large models
        ... )

        >>> # With authentication (if required)
        >>> client = OpenAICompatibleClient(
        ...     model="gpt-j",
        ...     base_url="http://secure-endpoint.internal/v1",
        ...     api_key="internal-api-key",
        ... )

    Note:
        This client reuses all parsing and error handling from `OpenAIClient`.
        If the self-hosted server has API differences, you may need to
        subclass and override specific methods.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "not-needed",
        timeout: float = 300.0,
    ):
        """Initialize the OpenAI-compatible client.

        Args:
            model: The model identifier as configured on the server.
            base_url: The base URL of the API endpoint (required).
            api_key: API key for authentication. Defaults to "not-needed".
            timeout: Request timeout in seconds. Defaults to 300.0.
        """
        self.model = model
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
