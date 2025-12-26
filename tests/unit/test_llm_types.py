"""Unit tests for LLMRequest and LLMResponse types."""

from inf_engine.types import LLMRequest, LLMResponse


class TestLLMRequest:
    """Tests for LLMRequest dataclass."""

    def test_creation_minimal(self) -> None:
        """Test creating a request with only required field."""
        request = LLMRequest(prompt="Hello, world!")

        assert request.prompt == "Hello, world!"
        assert request.system_prompt is None
        assert request.temperature == 1.0
        assert request.max_tokens is None
        assert request.response_format is None
        assert request.stop is None
        assert request.tools is None
        assert request.tool_choice is None
        assert request.extra_body is None

    def test_creation_with_all_fields(self) -> None:
        """Test creating a request with all fields specified."""
        tools = [{"name": "test_tool", "description": "A test tool"}]
        extra = {"reasoning_effort": "high"}

        request = LLMRequest(
            prompt="What is the capital of France?",
            system_prompt="You are a geography expert.",
            temperature=0.7,
            max_tokens=100,
            response_format=dict,
            stop=[".", "!"],
            tools=tools,
            tool_choice="auto",
            extra_body=extra,
        )

        assert request.prompt == "What is the capital of France?"
        assert request.system_prompt == "You are a geography expert."
        assert request.temperature == 0.7
        assert request.max_tokens == 100
        assert request.response_format is dict
        assert request.stop == [".", "!"]
        assert request.tools == tools
        assert request.tool_choice == "auto"
        assert request.extra_body == extra

    def test_temperature_range(self) -> None:
        """Test that temperature accepts various valid values."""
        # Low temperature (deterministic)
        request_low = LLMRequest(prompt="test", temperature=0.0)
        assert request_low.temperature == 0.0

        # High temperature (creative)
        request_high = LLMRequest(prompt="test", temperature=2.0)
        assert request_high.temperature == 2.0

    def test_tools_list(self) -> None:
        """Test request with tool definitions."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "get_time",
                "description": "Get the current time",
                "parameters": {"type": "object", "properties": {}},
            },
        ]
        request = LLMRequest(prompt="What's the weather?", tools=tools)

        assert request.tools is not None
        assert len(request.tools) == 2
        assert request.tools[0]["name"] == "get_weather"

    def test_tool_choice_string(self) -> None:
        """Test tool_choice with string values."""
        request_auto = LLMRequest(prompt="test", tool_choice="auto")
        assert request_auto.tool_choice == "auto"

        request_none = LLMRequest(prompt="test", tool_choice="none")
        assert request_none.tool_choice == "none"

        request_required = LLMRequest(prompt="test", tool_choice="required")
        assert request_required.tool_choice == "required"

    def test_tool_choice_dict(self) -> None:
        """Test tool_choice with dict to specify a tool."""
        inner_spec: dict[str, str] = {"name": "get_weather"}
        tool_spec: dict[str, str | dict[str, str]] = {
            "type": "function",
            "function": inner_spec,
        }
        request = LLMRequest(prompt="test", tool_choice=tool_spec)

        assert request.tool_choice == tool_spec
        assert inner_spec["name"] == "get_weather"

    def test_extra_body(self) -> None:
        """Test extra_body for provider-specific params."""
        extra = {
            "reasoning_effort": "high",
            "beta_feature": True,
            "custom_param": {"nested": "value"},
        }
        request = LLMRequest(prompt="test", extra_body=extra)

        assert request.extra_body == extra
        # Use the local variable to avoid type narrowing issues
        assert extra["reasoning_effort"] == "high"

    def test_to_dict_minimal(self) -> None:
        """Test to_dict with minimal request (only required fields)."""
        request = LLMRequest(prompt="Hello")
        result = request.to_dict()

        assert result == {
            "prompt": "Hello",
            "temperature": 1.0,
        }
        # Optional fields should not be present
        assert "system_prompt" not in result
        assert "max_tokens" not in result
        assert "response_format" not in result
        assert "stop" not in result
        assert "tools" not in result
        assert "tool_choice" not in result
        assert "extra_body" not in result

    def test_to_dict_with_all_fields(self) -> None:
        """Test to_dict with all fields specified."""
        tools = [{"name": "test"}]
        extra = {"key": "value"}

        request = LLMRequest(
            prompt="Hello",
            system_prompt="Be helpful",
            temperature=0.5,
            max_tokens=50,
            response_format=dict,
            stop=["\n"],
            tools=tools,
            tool_choice="auto",
            extra_body=extra,
        )
        result = request.to_dict()

        assert result["prompt"] == "Hello"
        assert result["system_prompt"] == "Be helpful"
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 50
        assert result["response_format"] == "dict"
        assert result["stop"] == ["\n"]
        assert result["tools"] == tools
        assert result["tool_choice"] == "auto"
        assert result["extra_body"] == extra

    def test_to_dict_with_some_optional_fields(self) -> None:
        """Test to_dict with only some optional fields."""
        request = LLMRequest(
            prompt="Hello",
            max_tokens=100,
            tools=[{"name": "test"}],
        )
        result = request.to_dict()

        assert result["prompt"] == "Hello"
        assert result["temperature"] == 1.0
        assert result["max_tokens"] == 100
        assert result["tools"] == [{"name": "test"}]
        assert "system_prompt" not in result
        assert "response_format" not in result
        assert "stop" not in result
        assert "tool_choice" not in result
        assert "extra_body" not in result

    def test_equality(self) -> None:
        """Test that dataclass equality works correctly."""
        request1 = LLMRequest(prompt="Hello", temperature=0.5)
        request2 = LLMRequest(prompt="Hello", temperature=0.5)
        request3 = LLMRequest(prompt="Hello", temperature=0.7)

        assert request1 == request2
        assert request1 != request3


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_creation(self) -> None:
        """Test creating a response with all required fields."""
        response = LLMResponse(
            content="The capital of France is Paris.",
            input_tokens=15,
            output_tokens=8,
            finish_reason="stop",
            model="gpt-4o-mini",
        )

        assert response.content == "The capital of France is Paris."
        assert response.input_tokens == 15
        assert response.output_tokens == 8
        assert response.finish_reason == "stop"
        assert response.model == "gpt-4o-mini"
        assert response.reasoning is None
        assert response.tool_calls is None

    def test_creation_with_all_fields(self) -> None:
        """Test creating a response with all fields including optional."""
        tool_calls = [{"id": "call_1", "name": "test", "arguments": "{}"}]

        response = LLMResponse(
            content="Result",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
            reasoning="Let me think...",
            tool_calls=tool_calls,
        )

        assert response.content == "Result"
        assert response.reasoning == "Let me think..."
        assert response.tool_calls == tool_calls

    def test_total_tokens(self) -> None:
        """Test that total_tokens property calculates correctly."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
        )

        assert response.total_tokens == 15

    def test_total_tokens_zero(self) -> None:
        """Test total_tokens with zero values."""
        response = LLMResponse(
            content="",
            input_tokens=0,
            output_tokens=0,
            finish_reason="stop",
            model="gpt-4o",
        )

        assert response.total_tokens == 0

    def test_has_tool_calls_true(self) -> None:
        """Test has_tool_calls returns True when tool_calls present."""
        response = LLMResponse(
            content="",
            input_tokens=10,
            output_tokens=5,
            finish_reason="tool_calls",
            model="gpt-4o",
            tool_calls=[{"id": "1", "name": "test", "arguments": "{}"}],
        )

        assert response.has_tool_calls is True

    def test_has_tool_calls_false_none(self) -> None:
        """Test has_tool_calls returns False when tool_calls is None."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
        )

        assert response.has_tool_calls is False

    def test_has_tool_calls_false_empty(self) -> None:
        """Test has_tool_calls returns False when tool_calls is empty list."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
            tool_calls=[],
        )

        assert response.has_tool_calls is False

    def test_has_reasoning_true(self) -> None:
        """Test has_reasoning returns True when reasoning present."""
        response = LLMResponse(
            content="42",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="claude-3-opus",
            reasoning="Let me think step by step...",
        )

        assert response.has_reasoning is True

    def test_has_reasoning_false_none(self) -> None:
        """Test has_reasoning returns False when reasoning is None."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
        )

        assert response.has_reasoning is False

    def test_has_reasoning_false_empty(self) -> None:
        """Test has_reasoning returns False when reasoning is empty string."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
            reasoning="",
        )

        assert response.has_reasoning is False

    def test_finish_reason_stop(self) -> None:
        """Test response with stop finish reason."""
        response = LLMResponse(
            content="Done",
            input_tokens=5,
            output_tokens=1,
            finish_reason="stop",
            model="gpt-4o",
        )

        assert response.finish_reason == "stop"

    def test_finish_reason_length(self) -> None:
        """Test response with length finish reason (hit max_tokens)."""
        response = LLMResponse(
            content="This response was truncated due to...",
            input_tokens=10,
            output_tokens=100,
            finish_reason="length",
            model="gpt-4o",
        )

        assert response.finish_reason == "length"

    def test_finish_reason_content_filter(self) -> None:
        """Test response with content_filter finish reason."""
        response = LLMResponse(
            content="",
            input_tokens=10,
            output_tokens=0,
            finish_reason="content_filter",
            model="gpt-4o",
        )

        assert response.finish_reason == "content_filter"

    def test_finish_reason_tool_calls(self) -> None:
        """Test response with tool_calls finish reason."""
        response = LLMResponse(
            content="",
            input_tokens=10,
            output_tokens=20,
            finish_reason="tool_calls",
            model="gpt-4o",
            tool_calls=[
                {
                    "id": "call_abc123",
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                }
            ],
        )

        assert response.finish_reason == "tool_calls"
        assert response.has_tool_calls is True

    def test_to_dict_minimal(self) -> None:
        """Test to_dict with only required fields."""
        response = LLMResponse(
            content="Paris",
            input_tokens=10,
            output_tokens=1,
            finish_reason="stop",
            model="gpt-4o",
        )
        result = response.to_dict()

        assert result == {
            "content": "Paris",
            "input_tokens": 10,
            "output_tokens": 1,
            "total_tokens": 11,
            "finish_reason": "stop",
            "model": "gpt-4o",
        }
        # Optional fields should not be present
        assert "reasoning" not in result
        assert "tool_calls" not in result

    def test_to_dict_with_reasoning(self) -> None:
        """Test to_dict includes reasoning when present."""
        response = LLMResponse(
            content="42",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="claude-3-opus",
            reasoning="The answer to life, the universe, and everything...",
        )
        result = response.to_dict()

        assert (
            result["reasoning"] == "The answer to life, the universe, and everything..."
        )
        assert result["content"] == "42"

    def test_to_dict_with_tool_calls(self) -> None:
        """Test to_dict includes tool_calls when present."""
        tool_calls = [
            {
                "id": "call_1",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            }
        ]
        response = LLMResponse(
            content="",
            input_tokens=10,
            output_tokens=15,
            finish_reason="tool_calls",
            model="gpt-4o",
            tool_calls=tool_calls,
        )
        result = response.to_dict()

        assert result["tool_calls"] == tool_calls

    def test_to_dict_with_all_optional_fields(self) -> None:
        """Test to_dict with all optional fields."""
        tool_calls = [{"id": "1", "name": "test", "arguments": "{}"}]
        response = LLMResponse(
            content="Result",
            input_tokens=10,
            output_tokens=20,
            finish_reason="stop",
            model="claude-3-opus",
            reasoning="Thinking...",
            tool_calls=tool_calls,
        )
        result = response.to_dict()

        assert result["reasoning"] == "Thinking..."
        assert result["tool_calls"] == tool_calls

    def test_equality(self) -> None:
        """Test that dataclass equality works correctly."""
        response1 = LLMResponse(
            content="Hello",
            input_tokens=5,
            output_tokens=1,
            finish_reason="stop",
            model="gpt-4o",
        )
        response2 = LLMResponse(
            content="Hello",
            input_tokens=5,
            output_tokens=1,
            finish_reason="stop",
            model="gpt-4o",
        )
        response3 = LLMResponse(
            content="Hello",
            input_tokens=5,
            output_tokens=1,
            finish_reason="length",  # Different finish reason
            model="gpt-4o",
        )

        assert response1 == response2
        assert response1 != response3

    def test_empty_content(self) -> None:
        """Test response with empty content (e.g., content filter)."""
        response = LLMResponse(
            content="",
            input_tokens=10,
            output_tokens=0,
            finish_reason="content_filter",
            model="gpt-4o",
        )

        assert response.content == ""
        assert response.total_tokens == 10

    def test_tool_calls_structure(self) -> None:
        """Test typical tool call structure matches expected format."""
        function_info: dict[str, str] = {
            "name": "get_weather",
            "arguments": '{"location": "Paris", "unit": "celsius"}',
        }
        tool_call: dict[str, str | dict[str, str]] = {
            "id": "call_abc123",
            "type": "function",
            "function": function_info,
        }
        tool_calls = [tool_call]

        response = LLMResponse(
            content="",
            input_tokens=20,
            output_tokens=30,
            finish_reason="tool_calls",
            model="gpt-4o",
            tool_calls=tool_calls,
        )

        # Use the local variable to avoid type narrowing issues
        assert response.tool_calls == tool_calls
        assert tool_call["id"] == "call_abc123"
        assert function_info["name"] == "get_weather"

    def test_timing_metrics_all(self) -> None:
        """Test response with all timing metrics."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
            time_to_first_token_ms=150.5,
            completion_time_ms=320.0,
            queue_time_ms=50.0,
        )

        assert response.time_to_first_token_ms == 150.5
        assert response.completion_time_ms == 320.0
        assert response.queue_time_ms == 50.0
        assert response.has_timing is True

    def test_timing_metrics_partial(self) -> None:
        """Test response with only some timing metrics."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
            completion_time_ms=250.0,
        )

        assert response.time_to_first_token_ms is None
        assert response.completion_time_ms == 250.0
        assert response.queue_time_ms is None
        assert response.has_timing is True

    def test_has_timing_false(self) -> None:
        """Test has_timing returns False when no timing metrics."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
        )

        assert response.has_timing is False

    def test_has_timing_with_ttft_only(self) -> None:
        """Test has_timing with only time_to_first_token_ms."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
            time_to_first_token_ms=100.0,
        )

        assert response.has_timing is True

    def test_has_timing_with_queue_time_only(self) -> None:
        """Test has_timing with only queue_time_ms."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
            queue_time_ms=25.0,
        )

        assert response.has_timing is True

    def test_to_dict_with_timing(self) -> None:
        """Test to_dict includes timing metrics when present."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
            time_to_first_token_ms=100.0,
            completion_time_ms=200.0,
            queue_time_ms=30.0,
        )
        result = response.to_dict()

        assert result["time_to_first_token_ms"] == 100.0
        assert result["completion_time_ms"] == 200.0
        assert result["queue_time_ms"] == 30.0

    def test_to_dict_without_timing(self) -> None:
        """Test to_dict excludes timing metrics when not present."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o",
        )
        result = response.to_dict()

        assert "time_to_first_token_ms" not in result
        assert "completion_time_ms" not in result
        assert "queue_time_ms" not in result


class TestLLMTypesIntegration:
    """Integration tests for LLMRequest and LLMResponse together."""

    def test_request_response_round_trip(self) -> None:
        """Test a typical request-response flow."""
        # Create a request
        request = LLMRequest(
            prompt="What is 2 + 2?",
            system_prompt="You are a math tutor.",
            temperature=0.0,
            max_tokens=10,
        )

        # Create a corresponding response
        response = LLMResponse(
            content="4",
            input_tokens=len(request.prompt.split()),
            output_tokens=1,
            finish_reason="stop",
            model="gpt-4o-mini",
        )

        # Both should serialize to dicts
        request_dict = request.to_dict()
        response_dict = response.to_dict()

        assert "prompt" in request_dict
        assert "content" in response_dict
        assert response.content == "4"

    def test_tool_use_flow(self) -> None:
        """Test a request with tools and response with tool calls."""
        tool_calls = [
            {
                "id": "call_123",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            }
        ]

        # Request with tool definitions
        request = LLMRequest(
            prompt="What's the weather in Paris?",
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

        # Response with tool call
        response = LLMResponse(
            content="",
            input_tokens=30,
            output_tokens=20,
            finish_reason="tool_calls",
            model="gpt-4o",
            tool_calls=tool_calls,
        )

        assert request.tools is not None
        assert response.has_tool_calls is True
        assert tool_calls[0]["name"] == "get_weather"

    def test_reasoning_flow(self) -> None:
        """Test a request and response with reasoning/thinking."""
        reasoning_text = (
            "Let me break this down:\n17 * 23 = 17 * 20 + 17 * 3\n= 340 + 51\n= 391"
        )

        # Request that might trigger extended thinking
        request = LLMRequest(
            prompt="Solve this step by step: What is 17 * 23?",
            extra_body={"reasoning_effort": "high"},
        )

        # Response with reasoning
        response = LLMResponse(
            content="391",
            input_tokens=15,
            output_tokens=100,
            finish_reason="stop",
            model="claude-3-opus",
            reasoning=reasoning_text,
        )

        assert request.extra_body is not None
        assert response.has_reasoning is True
        assert "391" in reasoning_text
