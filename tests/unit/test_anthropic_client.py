"""Unit tests for AnthropicClient implementation.

Tests validate the Anthropic client behavior with mocked API calls, including:
- Basic completion requests
- System prompt handling
- Tool calls
- Extended thinking
- Rate limit error translation
- Configuration options
"""

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from plait.clients import AnthropicClient, RateLimitError
from plait.clients.anthropic import AnthropicClient as AnthropicClientDirect
from plait.types import LLMRequest, LLMResponse


class TestAnthropicClientInit:
    """Tests for AnthropicClient initialization."""

    @patch("plait.clients.anthropic.anthropic.AsyncAnthropic")
    def test_init_with_model(self, mock_client_class: MagicMock) -> None:
        """Client initializes with model name."""
        client = AnthropicClient(model="claude-sonnet-4-20250514")
        assert client.model == "claude-sonnet-4-20250514"
        mock_client_class.assert_called_once()

    @patch("plait.clients.anthropic.anthropic.AsyncAnthropic")
    def test_init_with_custom_base_url(self, mock_client_class: MagicMock) -> None:
        """Client passes custom base_url to AsyncAnthropic."""
        AnthropicClient(model="claude-sonnet-4-20250514", base_url="https://custom.api")
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["base_url"] == "https://custom.api"

    @patch("plait.clients.anthropic.anthropic.AsyncAnthropic")
    def test_init_with_api_key(self, mock_client_class: MagicMock) -> None:
        """Client passes api_key to AsyncAnthropic."""
        AnthropicClient(model="claude-sonnet-4-20250514", api_key="sk-ant-test-key")
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-ant-test-key"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-env-key"})
    @patch("plait.clients.anthropic.anthropic.AsyncAnthropic")
    def test_init_with_api_key_from_env(self, mock_client_class: MagicMock) -> None:
        """Client uses ANTHROPIC_API_KEY from environment if not provided."""
        AnthropicClient(model="claude-sonnet-4-20250514")
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-ant-env-key"

    @patch("plait.clients.anthropic.anthropic.AsyncAnthropic")
    def test_init_with_custom_timeout(self, mock_client_class: MagicMock) -> None:
        """Client passes custom timeout to AsyncAnthropic."""
        AnthropicClient(model="claude-sonnet-4-20250514", timeout=60.0)
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["timeout"] == 60.0

    @patch("plait.clients.anthropic.anthropic.AsyncAnthropic")
    def test_init_with_default_timeout(self, mock_client_class: MagicMock) -> None:
        """Client uses default 300s timeout."""
        AnthropicClient(model="claude-sonnet-4-20250514")
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["timeout"] == 300.0

    @patch("plait.clients.anthropic.anthropic.AsyncAnthropic")
    def test_init_with_custom_max_tokens(self, mock_client_class: MagicMock) -> None:
        """Client stores custom default max_tokens."""
        client = AnthropicClient(model="claude-sonnet-4-20250514", max_tokens=8192)
        assert client.default_max_tokens == 8192

    @patch("plait.clients.anthropic.anthropic.AsyncAnthropic")
    def test_init_with_default_max_tokens(self, mock_client_class: MagicMock) -> None:
        """Client uses default 4096 max_tokens."""
        client = AnthropicClient(model="claude-sonnet-4-20250514")
        assert client.default_max_tokens == 4096


def create_mock_response(
    content: str = "Hello!",
    input_tokens: int = 10,
    output_tokens: int = 5,
    stop_reason: str = "end_turn",
    model: str = "claude-sonnet-4-20250514",
    thinking: str | None = None,
    tool_calls: list | None = None,
) -> MagicMock:
    """Create a mock Anthropic Message response."""
    response = MagicMock()
    response.model = model
    response.stop_reason = stop_reason

    # Create content blocks
    content_blocks = []

    if thinking:
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = thinking
        content_blocks.append(thinking_block)

    if content:
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = content
        content_blocks.append(text_block)

    if tool_calls:
        for tc in tool_calls:
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.id = tc["id"]
            tool_block.name = tc["name"]
            tool_block.input = tc["input"]
            content_blocks.append(tool_block)

    response.content = content_blocks

    # Create usage
    response.usage = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens

    return response


class TestAnthropicClientComplete:
    """Tests for AnthropicClient.complete() method."""

    @pytest.fixture
    def mock_anthropic(self):
        """Create a mock Anthropic client."""
        with patch("plait.clients.anthropic.anthropic.AsyncAnthropic") as mock_class:
            mock_instance = MagicMock()
            mock_instance.messages = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_basic_completion(self, mock_anthropic: MagicMock) -> None:
        """Client completes a basic request."""
        mock_response = create_mock_response(content="Hello, how can I help?")
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(prompt="Hello!")
        response = await client.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello, how can I help?"
        assert response.finish_reason == "stop"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.input_tokens == 10
        assert response.output_tokens == 5

    @pytest.mark.asyncio
    async def test_completion_with_system_prompt(
        self, mock_anthropic: MagicMock
    ) -> None:
        """Client includes system prompt as a parameter."""
        mock_response = create_mock_response()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(
            prompt="What is Python?",
            system_prompt="You are a helpful assistant.",
        )
        await client.complete(request)

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are a helpful assistant."
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "What is Python?"}

    @pytest.mark.asyncio
    async def test_completion_without_system_prompt(
        self, mock_anthropic: MagicMock
    ) -> None:
        """Client sends only user message when no system prompt."""
        mock_response = create_mock_response()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(prompt="Hello!")
        await client.complete(request)

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert "system" not in call_kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello!"}

    @pytest.mark.asyncio
    async def test_completion_with_temperature(self, mock_anthropic: MagicMock) -> None:
        """Client passes temperature parameter."""
        mock_response = create_mock_response()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(prompt="Hello!", temperature=0.7)
        await client.complete(request)

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_completion_with_max_tokens(self, mock_anthropic: MagicMock) -> None:
        """Client passes max_tokens parameter."""
        mock_response = create_mock_response()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(prompt="Hello!", max_tokens=100)
        await client.complete(request)

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_completion_uses_default_max_tokens(
        self, mock_anthropic: MagicMock
    ) -> None:
        """Client uses default max_tokens when not specified."""
        mock_response = create_mock_response()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514", max_tokens=8192)
        request = LLMRequest(prompt="Hello!")
        await client.complete(request)

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_completion_with_stop_sequences(
        self, mock_anthropic: MagicMock
    ) -> None:
        """Client passes stop sequences."""
        mock_response = create_mock_response()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(prompt="Hello!", stop=["END", "STOP"])
        await client.complete(request)

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["stop_sequences"] == ["END", "STOP"]

    @pytest.mark.asyncio
    async def test_completion_with_tools(self, mock_anthropic: MagicMock) -> None:
        """Client formats tools for Anthropic API."""
        mock_response = create_mock_response()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(
            prompt="What's the weather?",
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            tool_choice="auto",
        )
        await client.complete(request)

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    @pytest.mark.asyncio
    async def test_completion_with_tool_calls_response(
        self, mock_anthropic: MagicMock
    ) -> None:
        """Client parses tool calls from response."""
        mock_response = create_mock_response(
            content="",
            stop_reason="tool_use",
            tool_calls=[
                {
                    "id": "call_123",
                    "name": "get_weather",
                    "input": {"location": "Paris"},
                }
            ],
        )
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(prompt="What's the weather?")
        response = await client.complete(request)

        assert response.finish_reason == "tool_calls"
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["id"] == "call_123"
        assert response.tool_calls[0]["name"] == "get_weather"
        assert response.tool_calls[0]["arguments"] == '{"location": "Paris"}'

    @pytest.mark.asyncio
    async def test_completion_with_thinking(self, mock_anthropic: MagicMock) -> None:
        """Client parses thinking content from response."""
        mock_response = create_mock_response(
            content="The answer is 42.",
            thinking="Let me think about this step by step...",
        )
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(
            prompt="What's the meaning of life?",
            extra_body={"thinking": {"type": "enabled", "budget_tokens": 1000}},
        )
        response = await client.complete(request)

        assert response.content == "The answer is 42."
        assert response.reasoning == "Let me think about this step by step..."
        assert response.has_reasoning

    @pytest.mark.asyncio
    async def test_completion_thinking_forces_temperature_1(
        self, mock_anthropic: MagicMock
    ) -> None:
        """Client sets temperature to 1.0 when thinking is enabled."""
        mock_response = create_mock_response()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(
            prompt="Hello!",
            temperature=0.5,  # This should be overridden
            extra_body={"thinking": {"type": "enabled", "budget_tokens": 1000}},
        )
        await client.complete(request)

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 1.0

    @pytest.mark.asyncio
    async def test_completion_empty_content(self, mock_anthropic: MagicMock) -> None:
        """Client handles empty content in response."""
        mock_response = create_mock_response(content="")
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(prompt="Hello!")
        response = await client.complete(request)

        assert response.content == ""


class TestAnthropicClientStopReasonMapping:
    """Tests for stop reason mapping."""

    @pytest.fixture
    def mock_anthropic(self):
        """Create a mock Anthropic client."""
        with patch("plait.clients.anthropic.anthropic.AsyncAnthropic") as mock_class:
            mock_instance = MagicMock()
            mock_instance.messages = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stop_reason,expected_finish_reason",
        [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
            ("unknown_reason", "unknown_reason"),
        ],
    )
    async def test_stop_reason_mapping(
        self,
        mock_anthropic: MagicMock,
        stop_reason: str,
        expected_finish_reason: str,
    ) -> None:
        """Client maps Anthropic stop reasons to standard finish reasons."""
        mock_response = create_mock_response(stop_reason=stop_reason)
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(prompt="Hello!")
        response = await client.complete(request)

        assert response.finish_reason == expected_finish_reason


class TestAnthropicClientToolChoice:
    """Tests for tool choice conversion."""

    @pytest.fixture
    def mock_anthropic(self):
        """Create a mock Anthropic client."""
        with patch("plait.clients.anthropic.anthropic.AsyncAnthropic") as mock_class:
            mock_instance = MagicMock()
            mock_instance.messages = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_choice,expected",
        [
            ("auto", {"type": "auto"}),
            ("none", {"type": "auto", "disable_parallel_tool_use": True}),
            ("required", {"type": "any"}),
            (
                {"function": {"name": "get_weather"}},
                {"type": "tool", "name": "get_weather"},
            ),
            ({"name": "get_weather"}, {"type": "tool", "name": "get_weather"}),
        ],
    )
    async def test_tool_choice_conversion(
        self,
        mock_anthropic: MagicMock,
        tool_choice: str | dict,
        expected: dict,
    ) -> None:
        """Client converts tool_choice to Anthropic format."""
        mock_response = create_mock_response()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(
            prompt="Hello!",
            tools=[{"name": "get_weather", "description": "Get weather"}],
            tool_choice=tool_choice,
        )
        await client.complete(request)

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == expected


class TestAnthropicClientRateLimit:
    """Tests for Anthropic rate limit handling."""

    @pytest.fixture
    def mock_anthropic(self):
        """Create a mock Anthropic client."""
        with patch("plait.clients.anthropic.anthropic.AsyncAnthropic") as mock_class:
            mock_instance = MagicMock()
            mock_instance.messages = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_rate_limit_error_raised(self, mock_anthropic: MagicMock) -> None:
        """Client raises RateLimitError on Anthropic rate limit."""
        anthropic_error = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_anthropic.messages.create = AsyncMock(side_effect=anthropic_error)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(prompt="Hello!")

        with pytest.raises(RateLimitError):
            await client.complete(request)

    @pytest.mark.asyncio
    async def test_rate_limit_with_retry_after(self, mock_anthropic: MagicMock) -> None:
        """Client extracts retry-after header from rate limit error."""
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "30"}
        mock_response.status_code = 429

        anthropic_error = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )
        mock_anthropic.messages.create = AsyncMock(side_effect=anthropic_error)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(prompt="Hello!")

        with pytest.raises(RateLimitError) as exc_info:
            await client.complete(request)

        assert exc_info.value.retry_after == 30.0

    @pytest.mark.asyncio
    async def test_rate_limit_without_retry_after(
        self, mock_anthropic: MagicMock
    ) -> None:
        """Client handles rate limit without retry-after header."""
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status_code = 429

        anthropic_error = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )
        mock_anthropic.messages.create = AsyncMock(side_effect=anthropic_error)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(prompt="Hello!")

        with pytest.raises(RateLimitError) as exc_info:
            await client.complete(request)

        assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_rate_limit_invalid_retry_after(
        self, mock_anthropic: MagicMock
    ) -> None:
        """Client handles invalid retry-after header value."""
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "invalid"}
        mock_response.status_code = 429

        anthropic_error = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )
        mock_anthropic.messages.create = AsyncMock(side_effect=anthropic_error)

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        request = LLMRequest(prompt="Hello!")

        with pytest.raises(RateLimitError) as exc_info:
            await client.complete(request)

        assert exc_info.value.retry_after is None


class TestAnthropicClientImports:
    """Tests for module imports."""

    def test_import_from_clients_package(self) -> None:
        """AnthropicClient can be imported from plait.clients."""
        from plait.clients import AnthropicClient as ImportedClient

        assert ImportedClient is AnthropicClientDirect

    def test_import_from_anthropic_module(self) -> None:
        """AnthropicClient can be imported from plait.clients.anthropic."""
        from plait.clients.anthropic import AnthropicClient as ImportedClient

        assert ImportedClient is AnthropicClientDirect

    def test_clients_module_exports_anthropic_client(self) -> None:
        """clients module __all__ includes AnthropicClient."""
        import plait.clients as clients_module

        assert "AnthropicClient" in clients_module.__all__


class TestAnthropicClientInheritance:
    """Tests for AnthropicClient class hierarchy."""

    def test_is_llm_client(self) -> None:
        """AnthropicClient is an LLMClient."""
        from plait.clients import LLMClient

        assert issubclass(AnthropicClient, LLMClient)

    @patch("plait.clients.anthropic.anthropic.AsyncAnthropic")
    def test_instance_is_llm_client(self, mock_client_class: MagicMock) -> None:
        """AnthropicClient instance is an LLMClient."""
        from plait.clients import LLMClient

        client = AnthropicClient(model="claude-sonnet-4-20250514")
        assert isinstance(client, LLMClient)
