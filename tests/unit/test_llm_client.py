"""Unit tests for LLMClient abstract base class.

Tests validate the interface contract that all LLM client implementations
must follow, including:
- Abstract base class behavior (cannot instantiate directly)
- Subclass implementation requirements
- Complete method interface validation
"""

import pytest

from plait.clients import LLMClient
from plait.clients.base import LLMClient as LLMClientBase
from plait.types import LLMRequest, LLMResponse


class MockLLMClient(LLMClient):
    """A mock LLM client for testing that captures requests."""

    def __init__(self) -> None:
        self.last_request: LLMRequest | None = None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a mock response and capture the request."""
        self.last_request = request
        return LLMResponse(
            content=f"Response to: {request.prompt[:20]}",
            input_tokens=len(request.prompt.split()),
            output_tokens=5,
            finish_reason="stop",
            model="mock-model",
        )


class TestLLMClientAbstract:
    """Tests for LLMClient abstract base class behavior."""

    def test_cannot_instantiate_directly(self) -> None:
        """LLMClient is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LLMClient()  # type: ignore[abstract]

    def test_subclass_without_complete_raises(self) -> None:
        """Subclass without complete() implementation raises TypeError."""

        class IncompleteClient(LLMClient):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteClient()  # type: ignore[abstract]

    def test_subclass_with_complete_instantiates(self) -> None:
        """Subclass with complete() implementation can be instantiated."""

        class ValidClient(LLMClient):
            async def complete(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(
                    content="test",
                    input_tokens=1,
                    output_tokens=1,
                    finish_reason="stop",
                    model="test-model",
                )

        client = ValidClient()
        assert isinstance(client, LLMClient)


class TestLLMClientInterface:
    """Tests for LLMClient interface usage."""

    @pytest.fixture
    def mock_client(self) -> MockLLMClient:
        """Create a mock LLM client for testing."""
        return MockLLMClient()

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(
        self, mock_client: MockLLMClient
    ) -> None:
        """complete() returns an LLMResponse object."""
        request = LLMRequest(prompt="Hello, world!")
        response = await mock_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.content == "Response to: Hello, world!"
        assert response.finish_reason == "stop"
        assert response.model == "mock-model"

    @pytest.mark.asyncio
    async def test_complete_receives_request(self, mock_client: MockLLMClient) -> None:
        """complete() receives the LLMRequest object."""
        request = LLMRequest(
            prompt="Test prompt",
            system_prompt="You are helpful.",
            temperature=0.5,
            max_tokens=100,
        )
        await mock_client.complete(request)

        captured = mock_client.last_request
        assert captured is not None
        assert captured.prompt == "Test prompt"
        assert captured.system_prompt == "You are helpful."
        assert captured.temperature == 0.5
        assert captured.max_tokens == 100

    @pytest.mark.asyncio
    async def test_complete_with_tools(self, mock_client: MockLLMClient) -> None:
        """complete() handles requests with tools."""
        request = LLMRequest(
            prompt="What's the weather?",
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
            tool_choice="auto",
        )
        response = await mock_client.complete(request)

        assert isinstance(response, LLMResponse)
        captured = mock_client.last_request
        assert captured is not None
        assert captured.tools is not None
        assert len(captured.tools) == 1
        assert captured.tool_choice == "auto"

    @pytest.mark.asyncio
    async def test_complete_with_extra_body(self, mock_client: MockLLMClient) -> None:
        """complete() handles requests with extra_body parameters."""
        request = LLMRequest(
            prompt="Think carefully...",
            extra_body={"reasoning_effort": "high"},
        )
        response = await mock_client.complete(request)

        assert isinstance(response, LLMResponse)
        captured = mock_client.last_request
        assert captured is not None
        assert captured.extra_body == {"reasoning_effort": "high"}


class TestLLMClientImport:
    """Tests for LLMClient module imports."""

    def test_import_from_clients_package(self) -> None:
        """LLMClient can be imported from plait.clients."""
        from plait.clients import LLMClient as ImportedClient

        assert ImportedClient is LLMClientBase

    def test_import_from_base_module(self) -> None:
        """LLMClient can be imported from plait.clients.base."""
        from plait.clients.base import LLMClient as ImportedClient

        assert ImportedClient is LLMClientBase

    def test_clients_module_exports(self) -> None:
        """clients module __all__ includes LLMClient."""
        import plait.clients as clients_module

        assert "LLMClient" in clients_module.__all__
