"""Request and response types for LLM API calls.

This module defines the data structures used to communicate with LLM clients,
providing a unified interface across different providers (OpenAI, Anthropic, etc.).
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMRequest:
    """A request to an LLM endpoint.

    Encapsulates all parameters needed for a completion request, providing
    a provider-agnostic interface that LLM clients translate to their
    specific API formats.

    Args:
        prompt: The user message or prompt to send to the model.
        system_prompt: Optional system message that sets the model's behavior.
            If None, the model uses its default behavior.
        temperature: Sampling temperature controlling randomness.
            Higher values (e.g., 1.0) make output more random,
            lower values (e.g., 0.0) make it more deterministic.
        max_tokens: Maximum number of tokens to generate in the response.
            If None, uses the model's default or maximum limit.
        response_format: Optional type hint for structured output parsing.
            When set, the client may use JSON mode or structured output
            features to ensure the response matches the expected format.
        stop: Optional list of sequences where the model should stop generating.
            When any of these sequences is encountered, generation stops.
        tools: Optional list of tool definitions the model can call.
            Each tool is a dict matching the provider's tool format.
            Typically includes 'name', 'description', and 'parameters' schema.
        tool_choice: Controls how the model selects tools.
            Can be "auto", "none", "required", or a dict specifying a tool.
        extra_body: Provider-specific parameters not covered by standard fields.
            Passed directly to the API request body. Useful for beta features,
            reasoning effort controls, or provider extensions.

    Example:
        >>> request = LLMRequest(
        ...     prompt="What is the capital of France?",
        ...     system_prompt="You are a helpful geography assistant.",
        ...     temperature=0.7,
        ...     max_tokens=100,
        ... )
        >>> request.prompt
        'What is the capital of France?'

        >>> # Request with tools
        >>> request_with_tools = LLMRequest(
        ...     prompt="What's the weather in Paris?",
        ...     tools=[{
        ...         "name": "get_weather",
        ...         "description": "Get current weather for a location",
        ...         "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
        ...     }],
        ...     tool_choice="auto",
        ... )

        >>> # Request with provider-specific params
        >>> request_extra = LLMRequest(
        ...     prompt="Think step by step...",
        ...     extra_body={"reasoning_effort": "high"},
        ... )
    """

    prompt: str
    system_prompt: str | None = None
    temperature: float = 1.0
    max_tokens: int | None = None
    response_format: type | None = None
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the request to a dictionary representation.

        Creates a dictionary containing all request parameters, useful for
        serialization, logging, or debugging. Only includes non-None optional
        fields.

        Returns:
            A dictionary with the request parameters.

        Example:
            >>> request = LLMRequest(
            ...     prompt="Hello",
            ...     temperature=0.5,
            ... )
            >>> d = request.to_dict()
            >>> d["prompt"]
            'Hello'
            >>> d["temperature"]
            0.5
            >>> "system_prompt" in d
            False
        """
        result: dict[str, Any] = {
            "prompt": self.prompt,
            "temperature": self.temperature,
        }
        if self.system_prompt is not None:
            result["system_prompt"] = self.system_prompt
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.response_format is not None:
            result["response_format"] = self.response_format.__name__
        if self.stop is not None:
            result["stop"] = self.stop
        if self.tools is not None:
            result["tools"] = self.tools
        if self.tool_choice is not None:
            result["tool_choice"] = self.tool_choice
        if self.extra_body is not None:
            result["extra_body"] = self.extra_body
        return result


@dataclass
class LLMResponse:
    """A response from an LLM endpoint.

    Encapsulates the completion result along with metadata about token usage,
    timing, and generation details. Provides a unified interface across different
    LLM providers.

    Args:
        content: The generated text content from the model.
            May be empty if the response consists only of tool calls.
        input_tokens: Number of tokens in the input prompt.
            Used for cost tracking and usage monitoring.
        output_tokens: Number of tokens in the generated response.
            Used for cost tracking and usage monitoring.
        finish_reason: The reason the model stopped generating.
            Common values include:
            - "stop": Natural completion or stop sequence reached
            - "length": Hit max_tokens limit
            - "content_filter": Blocked by content filtering
            - "tool_calls": Model requested tool invocation
        model: The model identifier that generated this response.
            May differ from the requested model if the provider uses
            model aliases or versioned endpoints.
        reasoning: Optional reasoning or thinking content from the model.
            Contains the model's chain-of-thought when extended thinking
            is enabled (e.g., Claude's thinking blocks).
        tool_calls: Optional list of tool calls requested by the model.
            Each tool call is a dict with 'id', 'name', and 'arguments'.
            Present when finish_reason is "tool_calls".
        time_to_first_token_ms: Time in milliseconds from request to first token.
            Measures initial latency before streaming begins.
        completion_time_ms: Total time in milliseconds to complete the response.
            Includes all processing from request to final token.
        queue_time_ms: Time in milliseconds spent waiting in provider queue.
            Available when the provider reports queue/wait time separately.

    Example:
        >>> response = LLMResponse(
        ...     content="The capital of France is Paris.",
        ...     input_tokens=15,
        ...     output_tokens=8,
        ...     finish_reason="stop",
        ...     model="gpt-4o-mini",
        ... )
        >>> response.content
        'The capital of France is Paris.'
        >>> response.total_tokens
        23

        >>> # Response with tool calls
        >>> tool_response = LLMResponse(
        ...     content="",
        ...     input_tokens=20,
        ...     output_tokens=15,
        ...     finish_reason="tool_calls",
        ...     model="gpt-4o",
        ...     tool_calls=[{
        ...         "id": "call_123",
        ...         "name": "get_weather",
        ...         "arguments": '{"location": "Paris"}'
        ...     }],
        ... )
        >>> tool_response.has_tool_calls
        True

        >>> # Response with timing metrics
        >>> timed_response = LLMResponse(
        ...     content="Hello!",
        ...     input_tokens=5,
        ...     output_tokens=2,
        ...     finish_reason="stop",
        ...     model="gpt-4o",
        ...     time_to_first_token_ms=150.5,
        ...     completion_time_ms=320.0,
        ... )
        >>> timed_response.has_timing
        True
    """

    content: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    model: str
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    time_to_first_token_ms: float | None = None
    completion_time_ms: float | None = None
    queue_time_ms: float | None = None

    @property
    def total_tokens(self) -> int:
        """Calculate the total number of tokens used.

        Returns:
            Sum of input and output tokens.

        Example:
            >>> response = LLMResponse(
            ...     content="Hello",
            ...     input_tokens=10,
            ...     output_tokens=5,
            ...     finish_reason="stop",
            ...     model="gpt-4o",
            ... )
            >>> response.total_tokens
            15
        """
        return self.input_tokens + self.output_tokens

    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls.

        Returns:
            True if tool_calls is not None and not empty.

        Example:
            >>> response = LLMResponse(
            ...     content="",
            ...     input_tokens=10,
            ...     output_tokens=5,
            ...     finish_reason="tool_calls",
            ...     model="gpt-4o",
            ...     tool_calls=[{"id": "1", "name": "test", "arguments": "{}"}],
            ... )
            >>> response.has_tool_calls
            True
        """
        return self.tool_calls is not None and len(self.tool_calls) > 0

    @property
    def has_reasoning(self) -> bool:
        """Check if the response contains reasoning content.

        Returns:
            True if reasoning is not None and not empty.

        Example:
            >>> response = LLMResponse(
            ...     content="42",
            ...     input_tokens=10,
            ...     output_tokens=5,
            ...     finish_reason="stop",
            ...     model="claude-3-opus",
            ...     reasoning="Let me think...",
            ... )
            >>> response.has_reasoning
            True
        """
        return self.reasoning is not None and len(self.reasoning) > 0

    @property
    def has_timing(self) -> bool:
        """Check if any timing metrics are available.

        Returns:
            True if at least one timing metric is present.

        Example:
            >>> response = LLMResponse(
            ...     content="Hello",
            ...     input_tokens=5,
            ...     output_tokens=1,
            ...     finish_reason="stop",
            ...     model="gpt-4o",
            ...     completion_time_ms=250.0,
            ... )
            >>> response.has_timing
            True
        """
        return (
            self.time_to_first_token_ms is not None
            or self.completion_time_ms is not None
            or self.queue_time_ms is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the response to a dictionary representation.

        Creates a dictionary containing all response fields, useful for
        serialization, logging, or debugging. Only includes non-None optional
        fields.

        Returns:
            A dictionary with all response fields.

        Example:
            >>> response = LLMResponse(
            ...     content="Paris",
            ...     input_tokens=10,
            ...     output_tokens=1,
            ...     finish_reason="stop",
            ...     model="gpt-4o",
            ... )
            >>> d = response.to_dict()
            >>> d["content"]
            'Paris'
            >>> d["total_tokens"]
            11
        """
        result: dict[str, Any] = {
            "content": self.content,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "model": self.model,
        }
        if self.reasoning is not None:
            result["reasoning"] = self.reasoning
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.time_to_first_token_ms is not None:
            result["time_to_first_token_ms"] = self.time_to_first_token_ms
        if self.completion_time_ms is not None:
            result["completion_time_ms"] = self.completion_time_ms
        if self.queue_time_ms is not None:
            result["queue_time_ms"] = self.queue_time_ms
        return result
