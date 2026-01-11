"""Anthropic API client implementation.

This module provides the `AnthropicClient` class for making async completion
requests to Anthropic's Claude models. It implements the `LLMClient` interface
for unified access across providers.
"""

import os
from typing import Any, cast

import anthropic
import anthropic.types

from plait.clients.base import LLMClient
from plait.clients.openai import RateLimitError
from plait.types import LLMRequest, LLMResponse


class AnthropicClient(LLMClient):
    """Client for the Anthropic API.

    Implements the `LLMClient` interface for making async completion requests
    to Anthropic's messages endpoint. Supports all standard parameters
    including tools, system prompts, and extended thinking.

    Args:
        model: The model identifier to use (e.g., "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022").
        base_url: Optional custom base URL for the API. If None, uses the
            default Anthropic endpoint. Useful for proxies.
        api_key: Optional API key. If None, reads from ANTHROPIC_API_KEY
            environment variable.
        timeout: Request timeout in seconds. Defaults to 300.0 (5 minutes).
        max_tokens: Default maximum tokens for responses. Anthropic requires
            this parameter; defaults to 4096 if not specified in requests.

    Example:
        >>> client = AnthropicClient(model="claude-sonnet-4-20250514")
        >>> request = LLMRequest(prompt="Hello, world!")
        >>> response = await client.complete(request)
        >>> print(response.content)
        'Hello! How can I help you today?'

        >>> # With custom endpoint
        >>> client = AnthropicClient(
        ...     model="claude-sonnet-4-20250514",
        ...     base_url="https://my-proxy.example.com",
        ...     api_key="sk-ant-...",
        ... )
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.default_max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(
            base_url=base_url,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=timeout,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a completion request against the Anthropic API.

        Translates the provider-agnostic `LLMRequest` to Anthropic's messages
        format and returns a provider-agnostic `LLMResponse`.

        Args:
            request: The completion request containing prompt and parameters.

        Returns:
            An `LLMResponse` with the generated content and metadata.

        Raises:
            RateLimitError: If Anthropic returns a 429 rate limit error.
                Includes `retry_after` if the header was provided.
            anthropic.APIError: For other API errors (auth, network, etc.).

        Note:
            This method builds the messages list from the request, including
            the system prompt if provided. Tool calls are extracted from the
            response when the model requests them.
        """
        messages = self._build_messages(request)
        kwargs = self._build_request_kwargs(request)

        try:
            response = await self._client.messages.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )

            return self._parse_response(response)

        except anthropic.RateLimitError as e:
            retry_after = self._extract_retry_after(e)
            raise RateLimitError(retry_after=retry_after) from e

    def _build_messages(
        self, request: LLMRequest
    ) -> list[anthropic.types.MessageParam]:
        """Build the messages list from the request.

        Args:
            request: The completion request.

        Returns:
            A list of message dicts in Anthropic's format.
        """
        messages: list[anthropic.types.MessageParam] = []
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
            "max_tokens": request.max_tokens or self.default_max_tokens,
        }

        # Only set temperature if not using extended thinking
        # (extended thinking requires temperature=1)
        if request.extra_body and request.extra_body.get("thinking"):
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = request.temperature

        if request.system_prompt is not None:
            kwargs["system"] = request.system_prompt

        if request.stop is not None:
            kwargs["stop_sequences"] = request.stop

        if request.tools is not None:
            kwargs["tools"] = self._convert_tools(request.tools)

        if request.tool_choice is not None:
            kwargs["tool_choice"] = self._convert_tool_choice(request.tool_choice)

        # Handle extended thinking (beta feature)
        if request.extra_body is not None:
            thinking = request.extra_body.get("thinking")
            if thinking:
                kwargs["thinking"] = thinking

        return kwargs

    def _convert_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[anthropic.types.ToolParam]:
        """Convert OpenAI-style tools to Anthropic format.

        Args:
            tools: List of tool definitions in OpenAI format.

        Returns:
            List of tools in Anthropic format.
        """
        anthropic_tools: list[anthropic.types.ToolParam] = []
        for tool in tools:
            anthropic_tool: anthropic.types.ToolParam = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {"type": "object"}),
            }
            anthropic_tools.append(anthropic_tool)
        return anthropic_tools

    def _convert_tool_choice(self, tool_choice: str | dict[str, Any]) -> dict[str, Any]:
        """Convert tool_choice to Anthropic format.

        Args:
            tool_choice: The tool choice setting.

        Returns:
            Tool choice in Anthropic format as a dict.
        """
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"type": "auto"}
            elif tool_choice == "none":
                return {"type": "auto", "disable_parallel_tool_use": True}
            elif tool_choice == "required":
                return {"type": "any"}
        elif isinstance(tool_choice, dict):
            # Specific tool choice
            if "function" in tool_choice:
                return {"type": "tool", "name": tool_choice["function"]["name"]}
            elif "name" in tool_choice:
                return {"type": "tool", "name": tool_choice["name"]}

        return {"type": "auto"}

    def _parse_response(self, response: anthropic.types.Message) -> LLMResponse:
        """Parse the Anthropic response into an LLMResponse.

        Args:
            response: The raw Anthropic message response.

        Returns:
            A provider-agnostic LLMResponse.
        """
        # Extract content from content blocks
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                text_block = cast(anthropic.types.TextBlock, block)
                content_parts.append(text_block.text)
            elif block.type == "thinking":
                # Extended thinking content
                thinking_block = cast(anthropic.types.ThinkingBlock, block)
                reasoning_parts.append(thinking_block.thinking)
            elif block.type == "tool_use":
                tool_block = cast(anthropic.types.ToolUseBlock, block)
                tool_calls.append(
                    {
                        "id": tool_block.id,
                        "name": tool_block.name,
                        "arguments": self._serialize_tool_input(tool_block.input),
                    }
                )

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts) if reasoning_parts else None

        # Determine finish reason
        stop_reason = response.stop_reason or "unknown"
        finish_reason = self._map_stop_reason(stop_reason)

        # Get usage info
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            model=response.model,
            reasoning=reasoning,
            tool_calls=tool_calls if tool_calls else None,
        )

    def _serialize_tool_input(self, tool_input: object) -> str:
        """Serialize tool input to JSON string.

        Args:
            tool_input: The tool input object from Anthropic.

        Returns:
            JSON string representation.
        """
        import json

        if isinstance(tool_input, str):
            return tool_input
        return json.dumps(tool_input)

    def _map_stop_reason(self, stop_reason: str) -> str:
        """Map Anthropic stop reasons to standard finish reasons.

        Args:
            stop_reason: The Anthropic stop reason.

        Returns:
            Standardized finish reason string.
        """
        mapping = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }
        return mapping.get(stop_reason, stop_reason)

    def _extract_retry_after(self, error: anthropic.RateLimitError) -> float | None:
        """Extract the retry-after value from a rate limit error.

        Args:
            error: The Anthropic rate limit error.

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
