"""Unit tests for OpenAIClient implementation.

Tests validate the OpenAI client behavior with mocked API calls, including:
- Basic completion requests
- System prompt handling
- Tool calls
- Rate limit error translation
- Configuration options
"""

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from inf_engine.clients import OpenAIClient, RateLimitError
from inf_engine.clients.openai import OpenAIClient as OpenAIClientDirect
from inf_engine.clients.openai import RateLimitError as RateLimitErrorDirect
from inf_engine.resources.types import LLMRequest, LLMResponse


class TestRateLimitError:
    """Tests for the RateLimitError exception."""

    def test_creation_with_retry_after(self) -> None:
        """RateLimitError stores retry_after value."""
        error = RateLimitError(retry_after=30.0)
        assert error.retry_after == 30.0
        assert str(error) == "Rate limit exceeded"

    def test_creation_without_retry_after(self) -> None:
        """RateLimitError works without retry_after."""
        error = RateLimitError()
        assert error.retry_after is None

    def test_creation_with_custom_message(self) -> None:
        """RateLimitError accepts custom message."""
        error = RateLimitError(message="Too many requests")
        assert str(error) == "Too many requests"

    def test_is_exception(self) -> None:
        """RateLimitError is an Exception subclass."""
        error = RateLimitError()
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """RateLimitError can be raised and caught."""
        with pytest.raises(RateLimitError) as exc_info:
            raise RateLimitError(retry_after=60.0)
        assert exc_info.value.retry_after == 60.0


class TestOpenAIClientInit:
    """Tests for OpenAIClient initialization."""

    @patch("inf_engine.clients.openai.openai.AsyncOpenAI")
    def test_init_with_model(self, mock_client_class: MagicMock) -> None:
        """Client initializes with model name."""
        client = OpenAIClient(model="gpt-4o")
        assert client.model == "gpt-4o"
        mock_client_class.assert_called_once()

    @patch("inf_engine.clients.openai.openai.AsyncOpenAI")
    def test_init_with_custom_base_url(self, mock_client_class: MagicMock) -> None:
        """Client passes custom base_url to AsyncOpenAI."""
        OpenAIClient(model="gpt-4o", base_url="https://custom.api/v1")
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["base_url"] == "https://custom.api/v1"

    @patch("inf_engine.clients.openai.openai.AsyncOpenAI")
    def test_init_with_api_key(self, mock_client_class: MagicMock) -> None:
        """Client passes api_key to AsyncOpenAI."""
        OpenAIClient(model="gpt-4o", api_key="sk-test-key")
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-test-key"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-key"})
    @patch("inf_engine.clients.openai.openai.AsyncOpenAI")
    def test_init_with_api_key_from_env(self, mock_client_class: MagicMock) -> None:
        """Client uses OPENAI_API_KEY from environment if not provided."""
        OpenAIClient(model="gpt-4o")
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-env-key"

    @patch("inf_engine.clients.openai.openai.AsyncOpenAI")
    def test_init_with_custom_timeout(self, mock_client_class: MagicMock) -> None:
        """Client passes custom timeout to AsyncOpenAI."""
        OpenAIClient(model="gpt-4o", timeout=60.0)
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["timeout"] == 60.0

    @patch("inf_engine.clients.openai.openai.AsyncOpenAI")
    def test_init_with_default_timeout(self, mock_client_class: MagicMock) -> None:
        """Client uses default 300s timeout."""
        OpenAIClient(model="gpt-4o")
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["timeout"] == 300.0


def create_mock_response(
    content: str = "Hello!",
    input_tokens: int = 10,
    output_tokens: int = 5,
    finish_reason: str = "stop",
    model: str = "gpt-4o-mini",
    tool_calls: list | None = None,
) -> MagicMock:
    """Create a mock OpenAI ChatCompletion response."""
    response = MagicMock()
    response.model = model

    # Create choice with message
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    response.choices = [choice]

    # Create usage
    response.usage = MagicMock()
    response.usage.prompt_tokens = input_tokens
    response.usage.completion_tokens = output_tokens

    return response


class TestOpenAIClientComplete:
    """Tests for OpenAIClient.complete() method."""

    @pytest.fixture
    def mock_openai(self):
        """Create a mock OpenAI client."""
        with patch("inf_engine.clients.openai.openai.AsyncOpenAI") as mock_class:
            mock_instance = MagicMock()
            mock_instance.chat = MagicMock()
            mock_instance.chat.completions = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_basic_completion(self, mock_openai: MagicMock) -> None:
        """Client completes a basic request."""
        mock_response = create_mock_response(content="Hello, how can I help?")
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o-mini")
        request = LLMRequest(prompt="Hello!")
        response = await client.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello, how can I help?"
        assert response.finish_reason == "stop"
        assert response.model == "gpt-4o-mini"
        assert response.input_tokens == 10
        assert response.output_tokens == 5

    @pytest.mark.asyncio
    async def test_completion_with_system_prompt(self, mock_openai: MagicMock) -> None:
        """Client includes system prompt in messages."""
        mock_response = create_mock_response()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(
            prompt="What is Python?",
            system_prompt="You are a helpful assistant.",
        )
        await client.complete(request)

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0] == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        assert messages[1] == {"role": "user", "content": "What is Python?"}

    @pytest.mark.asyncio
    async def test_completion_without_system_prompt(
        self, mock_openai: MagicMock
    ) -> None:
        """Client sends only user message when no system prompt."""
        mock_response = create_mock_response()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Hello!")
        await client.complete(request)

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello!"}

    @pytest.mark.asyncio
    async def test_completion_with_temperature(self, mock_openai: MagicMock) -> None:
        """Client passes temperature parameter."""
        mock_response = create_mock_response()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Hello!", temperature=0.7)
        await client.complete(request)

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_completion_with_max_tokens(self, mock_openai: MagicMock) -> None:
        """Client passes max_tokens parameter."""
        mock_response = create_mock_response()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Hello!", max_tokens=100)
        await client.complete(request)

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_completion_without_max_tokens(self, mock_openai: MagicMock) -> None:
        """Client omits max_tokens when not specified."""
        mock_response = create_mock_response()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Hello!")
        await client.complete(request)

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in call_kwargs

    @pytest.mark.asyncio
    async def test_completion_with_stop_sequences(self, mock_openai: MagicMock) -> None:
        """Client passes stop sequences."""
        mock_response = create_mock_response()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Hello!", stop=["END", "STOP"])
        await client.complete(request)

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["stop"] == ["END", "STOP"]

    @pytest.mark.asyncio
    async def test_completion_with_tools(self, mock_openai: MagicMock) -> None:
        """Client formats tools for OpenAI API."""
        mock_response = create_mock_response()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o")
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

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_completion_with_tool_calls_response(
        self, mock_openai: MagicMock
    ) -> None:
        """Client parses tool calls from response."""
        # Create mock tool call
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"location": "Paris"}'

        mock_response = create_mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="What's the weather?")
        response = await client.complete(request)

        assert response.finish_reason == "tool_calls"
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["id"] == "call_123"
        assert response.tool_calls[0]["name"] == "get_weather"
        assert response.tool_calls[0]["arguments"] == '{"location": "Paris"}'

    @pytest.mark.asyncio
    async def test_completion_with_response_format(
        self, mock_openai: MagicMock
    ) -> None:
        """Client enables JSON mode when response_format is set."""
        mock_response = create_mock_response(content='{"answer": "yes"}')
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        class MySchema:
            pass

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Is this JSON?", response_format=MySchema)
        await client.complete(request)

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_completion_with_extra_body(self, mock_openai: MagicMock) -> None:
        """Client passes extra_body parameters."""
        mock_response = create_mock_response()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(
            prompt="Think step by step.",
            extra_body={"reasoning_effort": "high"},
        )
        await client.complete(request)

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {"reasoning_effort": "high"}

    @pytest.mark.asyncio
    async def test_completion_empty_content(self, mock_openai: MagicMock) -> None:
        """Client handles empty/None content in response."""
        mock_response = create_mock_response()
        mock_response.choices[0].message.content = None
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Hello!")
        response = await client.complete(request)

        assert response.content == ""

    @pytest.mark.asyncio
    async def test_completion_no_usage(self, mock_openai: MagicMock) -> None:
        """Client handles missing usage info."""
        mock_response = create_mock_response()
        mock_response.usage = None
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Hello!")
        response = await client.complete(request)

        assert response.input_tokens == 0
        assert response.output_tokens == 0

    @pytest.mark.asyncio
    async def test_completion_no_finish_reason(self, mock_openai: MagicMock) -> None:
        """Client handles missing finish reason."""
        mock_response = create_mock_response()
        mock_response.choices[0].finish_reason = None
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Hello!")
        response = await client.complete(request)

        assert response.finish_reason == "unknown"


class TestOpenAIClientRateLimit:
    """Tests for OpenAI rate limit handling."""

    @pytest.fixture
    def mock_openai(self):
        """Create a mock OpenAI client."""
        with patch("inf_engine.clients.openai.openai.AsyncOpenAI") as mock_class:
            mock_instance = MagicMock()
            mock_instance.chat = MagicMock()
            mock_instance.chat.completions = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_rate_limit_error_raised(self, mock_openai: MagicMock) -> None:
        """Client raises RateLimitError on OpenAI rate limit."""
        openai_error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_openai.chat.completions.create = AsyncMock(side_effect=openai_error)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Hello!")

        with pytest.raises(RateLimitError):
            await client.complete(request)

    @pytest.mark.asyncio
    async def test_rate_limit_with_retry_after(self, mock_openai: MagicMock) -> None:
        """Client extracts retry-after header from rate limit error."""
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "30"}
        mock_response.status_code = 429

        openai_error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )
        mock_openai.chat.completions.create = AsyncMock(side_effect=openai_error)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Hello!")

        with pytest.raises(RateLimitError) as exc_info:
            await client.complete(request)

        assert exc_info.value.retry_after == 30.0

    @pytest.mark.asyncio
    async def test_rate_limit_without_retry_after(self, mock_openai: MagicMock) -> None:
        """Client handles rate limit without retry-after header."""
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status_code = 429

        openai_error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )
        mock_openai.chat.completions.create = AsyncMock(side_effect=openai_error)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Hello!")

        with pytest.raises(RateLimitError) as exc_info:
            await client.complete(request)

        assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_rate_limit_invalid_retry_after(self, mock_openai: MagicMock) -> None:
        """Client handles invalid retry-after header value."""
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "invalid"}
        mock_response.status_code = 429

        openai_error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )
        mock_openai.chat.completions.create = AsyncMock(side_effect=openai_error)

        client = OpenAIClient(model="gpt-4o")
        request = LLMRequest(prompt="Hello!")

        with pytest.raises(RateLimitError) as exc_info:
            await client.complete(request)

        assert exc_info.value.retry_after is None


class TestOpenAIClientImports:
    """Tests for module imports."""

    def test_import_from_clients_package(self) -> None:
        """OpenAIClient can be imported from inf_engine.clients."""
        from inf_engine.clients import OpenAIClient as ImportedClient

        assert ImportedClient is OpenAIClientDirect

    def test_import_rate_limit_error_from_clients(self) -> None:
        """RateLimitError can be imported from inf_engine.clients."""
        from inf_engine.clients import RateLimitError as ImportedError

        assert ImportedError is RateLimitErrorDirect

    def test_import_from_openai_module(self) -> None:
        """OpenAIClient can be imported from inf_engine.clients.openai."""
        from inf_engine.clients.openai import OpenAIClient as ImportedClient

        assert ImportedClient is OpenAIClientDirect

    def test_clients_module_exports_openai_client(self) -> None:
        """clients module __all__ includes OpenAIClient."""
        import inf_engine.clients as clients_module

        assert "OpenAIClient" in clients_module.__all__

    def test_clients_module_exports_rate_limit_error(self) -> None:
        """clients module __all__ includes RateLimitError."""
        import inf_engine.clients as clients_module

        assert "RateLimitError" in clients_module.__all__
