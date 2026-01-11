"""Integration tests for resource management.

This file contains integration tests for:
- PR-048: Resource management integration with mocked LLM endpoints

Tests validate the full resource management system including:
- Multiple endpoint aliases
- Concurrent request handling with semaphores
- Rate limiting behavior
- Integration with ExecutionSettings
- Batch execution across multiple endpoints
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plait.execution.context import ExecutionSettings
from plait.execution.executor import run
from plait.execution.scheduler import Scheduler
from plait.execution.state import ExecutionState
from plait.module import InferenceModule, LLMInference
from plait.resources.config import EndpointConfig, ResourceConfig
from plait.tracing.tracer import Tracer
from plait.types import LLMResponse


@pytest.fixture(autouse=True)
def clean_context() -> None:
    """Ensure execution settings context is clean before each test."""
    from plait.execution.context import _execution_settings, get_execution_settings

    current = get_execution_settings()
    while current is not None:
        if current._token is not None:
            _execution_settings.reset(current._token)
            current._token = None
        current = get_execution_settings()


def create_mock_response(content: str = "Mock response") -> LLMResponse:
    """Create a mock LLM response for testing."""
    return LLMResponse(
        content=content,
        input_tokens=10,
        output_tokens=20,
        finish_reason="stop",
        model="mock-model",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test Modules for Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class MultiEndpointPipeline(InferenceModule):
    """Pipeline that uses multiple LLM endpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.fast = LLMInference(alias="fast")
        self.smart = LLMInference(alias="smart")

    def forward(self, text: str) -> dict[str, str]:
        return {
            "fast_response": self.fast(text),
            "smart_response": self.smart(text),
        }


class SequentialPipeline(InferenceModule):
    """Pipeline that chains LLM calls sequentially."""

    def __init__(self) -> None:
        super().__init__()
        self.step1 = LLMInference(alias="fast", system_prompt="Extract key points.")
        self.step2 = LLMInference(alias="smart", system_prompt="Analyze the points.")

    def forward(self, text: str) -> str:
        key_points = self.step1(text)
        analysis = self.step2(key_points)
        return analysis


class ParallelAnalysisPipeline(InferenceModule):
    """Pipeline that runs multiple analyses in parallel."""

    def __init__(self) -> None:
        super().__init__()
        self.analysis_a = LLMInference(alias="fast")
        self.analysis_b = LLMInference(alias="fast")
        self.analysis_c = LLMInference(alias="fast")

    def forward(self, text: str) -> dict[str, str]:
        return {
            "a": self.analysis_a(text),
            "b": self.analysis_b(text),
            "c": self.analysis_c(text),
        }


class SimpleLLMModule(InferenceModule):
    """Simple module with a single LLM call."""

    def __init__(self, alias: str = "fast") -> None:
        super().__init__()
        self.llm = LLMInference(alias=alias)

    def forward(self, text: str) -> str:
        return self.llm(text)


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Alias Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleAliases:
    """Integration tests for using multiple endpoint aliases."""

    @pytest.mark.asyncio
    async def test_module_uses_different_endpoints(self) -> None:
        """Different modules use their configured endpoint aliases."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=10,
                ),
                "smart": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o",
                    max_concurrent=5,
                ),
            }
        )

        # Track which aliases are called
        alias_calls: list[str] = []

        async def mock_complete(request: Any) -> LLMResponse:
            return create_mock_response(f"Response for: {request.prompt}")

        with patch("plait.resources.manager.OpenAIClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.complete = mock_complete
            MockClient.return_value = mock_client

            from plait.resources.manager import ResourceManager

            manager = ResourceManager(config)

            # Manually track calls by wrapping get_client
            original_get_client = manager.get_client

            def tracking_get_client(alias: str) -> Any:
                alias_calls.append(alias)
                return original_get_client(alias)

            # Type: ignore needed for monkey-patching in tests
            manager.get_client = tracking_get_client  # type: ignore[method-assign]

            pipeline = MultiEndpointPipeline()
            tracer = Tracer()
            graph = tracer.trace(pipeline, "Test input")
            state = ExecutionState(graph)

            scheduler = Scheduler(resource_manager=manager)
            await scheduler.execute(state)

        # Both aliases should have been called
        assert "fast" in alias_calls
        assert "smart" in alias_calls

    @pytest.mark.asyncio
    async def test_endpoints_return_different_results(self) -> None:
        """Different endpoints can return different results."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                ),
                "smart": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o",
                ),
            }
        )

        # Use different responses based on model
        response_map = {
            "gpt-4o-mini": "Fast response!",
            "gpt-4o": "Smart and detailed response!",
        }

        with patch("plait.resources.manager.OpenAIClient") as MockClient:
            # Create separate mock clients per endpoint
            fast_client = AsyncMock()
            fast_client.complete = AsyncMock(
                return_value=create_mock_response(response_map["gpt-4o-mini"])
            )

            smart_client = AsyncMock()
            smart_client.complete = AsyncMock(
                return_value=create_mock_response(response_map["gpt-4o"])
            )

            # Return different clients based on call order
            MockClient.side_effect = [fast_client, smart_client]

            from plait.resources.manager import ResourceManager

            manager = ResourceManager(config)

            pipeline = MultiEndpointPipeline()
            tracer = Tracer()
            graph = tracer.trace(pipeline, "Test input")
            state = ExecutionState(graph)

            scheduler = Scheduler(resource_manager=manager)
            await scheduler.execute(state)

            outputs = state.get_outputs()

        # Verify different responses from different endpoints
        assert "Fast response!" in str(outputs)
        assert "Smart and detailed response!" in str(outputs)


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent Request Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConcurrentRequests:
    """Integration tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_requests(self) -> None:
        """Endpoint semaphore limits concurrent requests."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=2,  # Only 2 concurrent requests
                ),
            }
        )

        # Track concurrent request count
        current_concurrent = 0
        max_concurrent_seen = 0
        lock = asyncio.Lock()

        async def slow_complete(request: Any) -> LLMResponse:
            nonlocal current_concurrent, max_concurrent_seen
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent_seen:
                    max_concurrent_seen = current_concurrent
            await asyncio.sleep(0.05)  # Simulate network delay
            async with lock:
                current_concurrent -= 1
            return create_mock_response("Done")

        with patch("plait.resources.manager.OpenAIClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.complete = slow_complete
            MockClient.return_value = mock_client

            from plait.resources.manager import ResourceManager

            manager = ResourceManager(config)

            pipeline = ParallelAnalysisPipeline()
            tracer = Tracer()
            graph = tracer.trace(pipeline, "Test")
            state = ExecutionState(graph)

            scheduler = Scheduler(resource_manager=manager, max_concurrent=10)
            await scheduler.execute(state)

        # Max concurrent should not exceed semaphore limit
        assert max_concurrent_seen <= 2, (
            f"Expected max 2 concurrent, but saw {max_concurrent_seen}"
        )

    @pytest.mark.asyncio
    async def test_parallel_requests_complete_faster(self) -> None:
        """Parallel requests complete faster than sequential."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=10,  # Allow high concurrency
                ),
            }
        )

        delay_per_request = 0.03

        async def delayed_complete(request: Any) -> LLMResponse:
            await asyncio.sleep(delay_per_request)
            return create_mock_response("Done")

        with patch("plait.resources.manager.OpenAIClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.complete = delayed_complete
            MockClient.return_value = mock_client

            from plait.resources.manager import ResourceManager

            manager = ResourceManager(config)

            pipeline = ParallelAnalysisPipeline()
            tracer = Tracer()
            graph = tracer.trace(pipeline, "Test")
            state = ExecutionState(graph)

            scheduler = Scheduler(resource_manager=manager, max_concurrent=10)

            start = time.monotonic()
            await scheduler.execute(state)
            elapsed = time.monotonic() - start

        # 3 parallel requests should take ~30ms, not ~90ms
        sequential_time = 3 * delay_per_request
        assert elapsed < sequential_time * 0.8, (
            f"Elapsed {elapsed:.3f}s should be less than sequential {sequential_time:.3f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ResourceManager with Scheduler Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResourceManagerSchedulerIntegration:
    """Integration tests for ResourceManager with Scheduler."""

    @pytest.mark.asyncio
    async def test_scheduler_executes_llm_through_resource_manager(self) -> None:
        """Scheduler executes LLMInference modules through ResourceManager."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                ),
            }
        )

        requests_received: list[Any] = []

        async def tracking_complete(request: Any) -> LLMResponse:
            requests_received.append(request)
            return create_mock_response(f"Processed: {request.prompt}")

        with patch("plait.resources.manager.OpenAIClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.complete = tracking_complete
            MockClient.return_value = mock_client

            from plait.resources.manager import ResourceManager

            manager = ResourceManager(config)

            module = SimpleLLMModule(alias="fast")
            tracer = Tracer()
            graph = tracer.trace(module, "Hello, world!")
            state = ExecutionState(graph)

            scheduler = Scheduler(resource_manager=manager)
            await scheduler.execute(state)

            outputs = state.get_outputs()

        # Request should have been made
        assert len(requests_received) == 1
        assert "Hello, world!" in requests_received[0].prompt

        # Output should contain response
        assert "Processed:" in str(outputs)

    @pytest.mark.asyncio
    async def test_scheduler_builds_correct_llm_request(self) -> None:
        """Scheduler builds LLMRequest with correct parameters."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                ),
            }
        )

        captured_request: Any = None

        async def capture_complete(request: Any) -> LLMResponse:
            nonlocal captured_request
            captured_request = request
            return create_mock_response("Done")

        with patch("plait.resources.manager.OpenAIClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.complete = capture_complete
            MockClient.return_value = mock_client

            from plait.resources.manager import ResourceManager

            manager = ResourceManager(config)

            # Create module with custom settings
            llm = LLMInference(
                alias="fast",
                system_prompt="You are helpful.",
                temperature=0.7,
                max_tokens=100,
            )

            class CustomModule(InferenceModule):
                def __init__(self) -> None:
                    super().__init__()
                    self.llm = llm

                def forward(self, text: str) -> str:
                    return self.llm(text)

            module = CustomModule()
            tracer = Tracer()
            graph = tracer.trace(module, "Test prompt")
            state = ExecutionState(graph)

            scheduler = Scheduler(resource_manager=manager)
            await scheduler.execute(state)

        # Verify request parameters
        assert captured_request is not None
        assert captured_request.prompt == "Test prompt"
        assert captured_request.system_prompt == "You are helpful."
        assert captured_request.temperature == 0.7
        assert captured_request.max_tokens == 100


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionSettings Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExecutionSettingsResourceIntegration:
    """Integration tests for ExecutionSettings with resources."""

    @pytest.mark.asyncio
    async def test_execution_settings_provides_resources(self) -> None:
        """ExecutionSettings context provides resources for module execution."""
        mock_resources = MagicMock()

        module = SimpleLLMModule(alias="fast")

        async def mock_run_func(*args: Any, **kwargs: Any) -> str:
            # Verify resources are passed
            assert kwargs.get("resources") is mock_resources
            return "Executed with resources"

        with patch("plait.execution.executor.run", side_effect=mock_run_func):
            async with ExecutionSettings(resources=mock_resources):
                result = await module("test input")

        assert result == "Executed with resources"

    @pytest.mark.asyncio
    async def test_bound_resources_override_context(self) -> None:
        """Bound resources take precedence over ExecutionSettings context."""
        context_resources = MagicMock(name="context")
        bound_resources = MagicMock(name="bound")

        module = SimpleLLMModule(alias="fast")
        module.bind(resources=bound_resources)

        received_resources: list[Any] = []

        async def mock_run_func(*args: Any, **kwargs: Any) -> str:
            received_resources.append(kwargs.get("resources"))
            return "Done"

        with patch("plait.execution.executor.run", side_effect=mock_run_func):
            async with ExecutionSettings(resources=context_resources):
                await module("test")

        # Bound resources should be used, not context resources
        assert received_resources[0] is bound_resources

    @pytest.mark.asyncio
    async def test_multiple_modules_share_context_resources(self) -> None:
        """Multiple unbound modules share ExecutionSettings resources."""
        mock_resources = MagicMock()

        module1 = SimpleLLMModule(alias="fast")
        module2 = SimpleLLMModule(alias="smart")

        received_resources: list[Any] = []

        async def mock_run_func(*args: Any, **kwargs: Any) -> str:
            received_resources.append(kwargs.get("resources"))
            return "Done"

        with patch("plait.execution.executor.run", side_effect=mock_run_func):
            async with ExecutionSettings(resources=mock_resources):
                await module1("input 1")
                await module2("input 2")

        # Both should use same context resources
        assert len(received_resources) == 2
        assert all(r is mock_resources for r in received_resources)


# ─────────────────────────────────────────────────────────────────────────────
# Batch Execution Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchExecutionWithResources:
    """Integration tests for batch execution with resource management."""

    @pytest.mark.asyncio
    async def test_batch_execution_uses_resources(self) -> None:
        """Batch execution properly uses resources from context."""
        mock_resources = MagicMock()

        module = SimpleLLMModule(alias="fast")

        call_count = 0

        async def mock_run_func(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            assert kwargs.get("resources") is mock_resources
            return f"Result_{call_count}"

        with patch("plait.execution.executor.run", side_effect=mock_run_func):
            async with ExecutionSettings(resources=mock_resources):
                results = await module(["a", "b", "c"])

        assert len(results) == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_batch_execution_runs_concurrently(self) -> None:
        """Batch items execute concurrently, not sequentially."""
        mock_resources = MagicMock()

        module = SimpleLLMModule(alias="fast")

        call_times: list[float] = []
        delay = 0.03

        async def slow_run(*args: Any, **kwargs: Any) -> str:
            call_times.append(time.monotonic())
            await asyncio.sleep(delay)
            return "Done"

        with patch("plait.execution.executor.run", side_effect=slow_run):
            async with ExecutionSettings(resources=mock_resources):
                start = time.monotonic()
                await module(["a", "b", "c", "d", "e"])
                total_time = time.monotonic() - start

        # Should be much faster than sequential (5 * 30ms = 150ms)
        sequential_time = 5 * delay
        assert total_time < sequential_time * 0.7, (
            f"Total time {total_time:.3f}s should be < {sequential_time * 0.7:.3f}s"
        )

        # All calls should start within a small window
        if len(call_times) >= 2:
            spread = max(call_times) - min(call_times)
            assert spread < 0.02, f"Call spread {spread:.3f}s should be < 0.02s"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndPipelines:
    """End-to-end tests for complete pipeline execution with resources."""

    @pytest.mark.asyncio
    async def test_sequential_pipeline_executes(self) -> None:
        """Sequential pipeline executes steps in order."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini"),
                "smart": EndpointConfig(provider_api="openai", model="gpt-4o"),
            }
        )

        call_order: list[str] = []

        async def tracking_complete(request: Any) -> LLMResponse:
            # Extract which step by system prompt
            if "Extract" in (request.system_prompt or ""):
                call_order.append("step1")
                return create_mock_response("Key points: A, B, C")
            else:
                call_order.append("step2")
                return create_mock_response("Analysis complete")

        with patch("plait.resources.manager.OpenAIClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.complete = tracking_complete
            MockClient.return_value = mock_client

            from plait.resources.manager import ResourceManager

            manager = ResourceManager(config)

            pipeline = SequentialPipeline()
            tracer = Tracer()
            graph = tracer.trace(pipeline, "Document text")
            state = ExecutionState(graph)

            scheduler = Scheduler(resource_manager=manager)
            await scheduler.execute(state)

        # Step 1 should complete before step 2 (sequential dependency)
        assert call_order.index("step1") < call_order.index("step2")

    @pytest.mark.asyncio
    async def test_run_function_with_resources(self) -> None:
        """run() function accepts resources parameter."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini"),
            }
        )

        async def mock_complete(request: Any) -> LLMResponse:
            return create_mock_response(f"Response: {request.prompt}")

        with patch("plait.resources.manager.OpenAIClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.complete = mock_complete
            MockClient.return_value = mock_client

            from plait.resources.manager import ResourceManager

            manager = ResourceManager(config)

            module = SimpleLLMModule(alias="fast")

            # Call run() with resources - verifies the parameter is accepted
            result = await run(module, "Hello", resources=manager)

        # Should get a result back
        assert "Response:" in str(result)


# ─────────────────────────────────────────────────────────────────────────────
# Error Handling Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResourceErrorHandling:
    """Integration tests for error handling with resources."""

    @pytest.mark.asyncio
    async def test_missing_alias_raises_error(self) -> None:
        """Using an undefined alias raises KeyError."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini"),
            }
        )

        with patch("plait.resources.manager.OpenAIClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            from plait.resources.manager import ResourceManager

            manager = ResourceManager(config)

            # Module uses "unknown" alias which isn't configured
            module = SimpleLLMModule(alias="unknown")
            tracer = Tracer()
            graph = tracer.trace(module, "Test")
            state = ExecutionState(graph)

            scheduler = Scheduler(resource_manager=manager)
            await scheduler.execute(state)

            # Check that the task failed
            errors = state.errors
            assert len(errors) > 0
            # The error should be about missing alias/client
            error_str = str(list(errors.values())[0])
            assert "unknown" in error_str.lower() or "key" in error_str.lower()

    @pytest.mark.asyncio
    async def test_llm_error_propagates(self) -> None:
        """LLM client errors propagate correctly."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(provider_api="openai", model="gpt-4o-mini"),
            }
        )

        async def failing_complete(request: Any) -> LLMResponse:
            raise RuntimeError("API error: Connection refused")

        with patch("plait.resources.manager.OpenAIClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.complete = failing_complete
            MockClient.return_value = mock_client

            from plait.resources.manager import ResourceManager

            manager = ResourceManager(config)

            module = SimpleLLMModule(alias="fast")
            tracer = Tracer()
            graph = tracer.trace(module, "Test")
            state = ExecutionState(graph)

            scheduler = Scheduler(resource_manager=manager)
            await scheduler.execute(state)

            # Task should have failed with the error
            errors = state.errors
            assert len(errors) > 0
            error_str = str(list(errors.values())[0])
            assert "API error" in error_str or "Connection" in error_str


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Examples Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigurationExamples:
    """Tests that validate configuration examples from design docs."""

    def test_development_configuration(self) -> None:
        """Development configuration from design docs works correctly."""
        dev_resources = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=5,
                ),
                "smart": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o",
                    max_concurrent=2,
                ),
            },
        )

        assert "fast" in dev_resources
        assert "smart" in dev_resources
        assert dev_resources["fast"].model == "gpt-4o-mini"
        assert dev_resources["smart"].model == "gpt-4o"
        assert dev_resources["fast"].max_concurrent == 5
        assert dev_resources["smart"].max_concurrent == 2

    def test_production_vllm_configuration(self) -> None:
        """Production vLLM configuration from design docs works correctly."""
        prod_resources = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="vllm",
                    model="mistral-7b",
                    base_url="http://vllm-fast.internal:8000",
                    max_concurrent=50,
                    rate_limit=100.0,
                ),
                "smart": EndpointConfig(
                    provider_api="vllm",
                    model="llama-70b",
                    base_url="http://vllm-smart.internal:8000",
                    max_concurrent=20,
                    rate_limit=30.0,
                ),
            },
        )

        assert prod_resources["fast"].provider_api == "vllm"
        assert prod_resources["fast"].base_url == "http://vllm-fast.internal:8000"
        assert prod_resources["fast"].rate_limit == 100.0
        assert prod_resources["smart"].rate_limit == 30.0

    @pytest.mark.asyncio
    async def test_resource_config_with_resource_manager(self) -> None:
        """ResourceConfig integrates correctly with ResourceManager."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=10,
                    rate_limit=600.0,
                ),
            }
        )

        with patch("plait.resources.manager.OpenAIClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            from plait.resources.manager import ResourceManager

            manager = ResourceManager(config)

            # Verify all resources are set up
            assert "fast" in manager.clients
            assert "fast" in manager.semaphores
            assert "fast" in manager.rate_limiters

            # Verify semaphore has correct value
            assert manager.semaphores["fast"]._value == 10

            # Verify rate limiter has correct RPM
            assert manager.rate_limiters["fast"].rpm == 600.0
