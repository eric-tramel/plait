"""Unit tests for ResourceMetrics.

Tests validate EndpointMetrics dataclass properties and ResourceMetrics
thread-safe metrics collection and aggregation.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from plait.resources import EndpointMetrics, ResourceMetrics
from plait.resources.config import EndpointConfig, ResourceConfig
from plait.resources.metrics import EndpointMetrics as EndpointMetricsDirect
from plait.resources.metrics import ResourceMetrics as ResourceMetricsDirect
from plait.types import LLMResponse


class TestEndpointMetricsCreation:
    """Tests for EndpointMetrics dataclass creation."""

    def test_default_values(self) -> None:
        """EndpointMetrics initializes with default zero values."""
        metrics = EndpointMetrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.rate_limited_requests == 0
        assert metrics.total_input_tokens == 0
        assert metrics.total_output_tokens == 0
        assert metrics.total_latency_ms == 0.0
        assert metrics.min_latency_ms == float("inf")
        assert metrics.max_latency_ms == 0.0

    def test_custom_values(self) -> None:
        """EndpointMetrics can be created with custom values."""
        metrics = EndpointMetrics(
            total_requests=100,
            successful_requests=95,
            failed_requests=3,
            rate_limited_requests=2,
            total_input_tokens=10000,
            total_output_tokens=5000,
            total_latency_ms=50000.0,
            min_latency_ms=50.0,
            max_latency_ms=1000.0,
        )

        assert metrics.total_requests == 100
        assert metrics.successful_requests == 95
        assert metrics.failed_requests == 3
        assert metrics.rate_limited_requests == 2
        assert metrics.total_input_tokens == 10000
        assert metrics.total_output_tokens == 5000
        assert metrics.total_latency_ms == 50000.0
        assert metrics.min_latency_ms == 50.0
        assert metrics.max_latency_ms == 1000.0


class TestEndpointMetricsProperties:
    """Tests for EndpointMetrics computed properties."""

    def test_avg_latency_ms_zero_requests(self) -> None:
        """avg_latency_ms returns 0.0 when no successful requests."""
        metrics = EndpointMetrics()
        assert metrics.avg_latency_ms == 0.0

    def test_avg_latency_ms_with_requests(self) -> None:
        """avg_latency_ms computes average correctly."""
        metrics = EndpointMetrics(
            successful_requests=4,
            total_latency_ms=400.0,
        )
        assert metrics.avg_latency_ms == 100.0

    def test_success_rate_zero_requests(self) -> None:
        """success_rate returns 0.0 when no requests made."""
        metrics = EndpointMetrics()
        assert metrics.success_rate == 0.0

    def test_success_rate_all_successful(self) -> None:
        """success_rate returns 1.0 when all requests successful."""
        metrics = EndpointMetrics(
            total_requests=10,
            successful_requests=10,
        )
        assert metrics.success_rate == 1.0

    def test_success_rate_partial(self) -> None:
        """success_rate computes fraction correctly."""
        metrics = EndpointMetrics(
            total_requests=10,
            successful_requests=7,
        )
        assert metrics.success_rate == 0.7

    def test_total_tokens(self) -> None:
        """total_tokens returns sum of input and output tokens."""
        metrics = EndpointMetrics(
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        assert metrics.total_tokens == 1500


class TestResourceMetricsCreation:
    """Tests for ResourceMetrics initialization."""

    def test_empty_init(self) -> None:
        """ResourceMetrics initializes with no metrics."""
        metrics = ResourceMetrics()
        assert metrics.get_all_stats() == {}

    def test_has_lock(self) -> None:
        """ResourceMetrics creates a threading lock."""
        metrics = ResourceMetrics()
        assert hasattr(metrics, "_lock")
        assert isinstance(metrics._lock, type(threading.Lock()))


class TestResourceMetricsRecordSuccess:
    """Tests for ResourceMetrics.record_success method."""

    def test_record_success_updates_counts(self) -> None:
        """record_success increments total and successful requests."""
        metrics = ResourceMetrics()
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o-mini",
        )

        metrics.record_success("fast", 0.150, response)

        stats = metrics.get_alias_stats("fast")
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["failed_requests"] == 0

    def test_record_success_updates_tokens(self) -> None:
        """record_success tracks input and output tokens."""
        metrics = ResourceMetrics()
        response = LLMResponse(
            content="Paris",
            input_tokens=100,
            output_tokens=25,
            finish_reason="stop",
            model="gpt-4o",
        )

        metrics.record_success("smart", 0.250, response)

        stats = metrics.get_alias_stats("smart")
        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 25
        assert stats["total_tokens"] == 125

    def test_record_success_updates_latency(self) -> None:
        """record_success tracks latency statistics."""
        metrics = ResourceMetrics()
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o-mini",
        )

        metrics.record_success("fast", 0.150, response)  # 150ms

        stats = metrics.get_alias_stats("fast")
        assert stats["avg_latency_ms"] == 150.0
        assert stats["min_latency_ms"] == 150.0
        assert stats["max_latency_ms"] == 150.0

    def test_record_success_multiple_requests(self) -> None:
        """record_success accumulates metrics across requests."""
        metrics = ResourceMetrics()

        for i in range(3):
            response = LLMResponse(
                content=f"Response {i}",
                input_tokens=10 * (i + 1),
                output_tokens=5 * (i + 1),
                finish_reason="stop",
                model="gpt-4o-mini",
            )
            metrics.record_success("fast", 0.100 * (i + 1), response)

        stats = metrics.get_alias_stats("fast")
        assert stats["total_requests"] == 3
        assert stats["successful_requests"] == 3
        assert stats["total_input_tokens"] == 10 + 20 + 30  # 60
        assert stats["total_output_tokens"] == 5 + 10 + 15  # 30
        assert stats["min_latency_ms"] == pytest.approx(100.0)
        assert stats["max_latency_ms"] == pytest.approx(300.0)


class TestResourceMetricsRecordError:
    """Tests for ResourceMetrics.record_error method."""

    def test_record_error_updates_counts(self) -> None:
        """record_error increments total and failed requests."""
        metrics = ResourceMetrics()

        metrics.record_error("fast", 0.050, ValueError("API error"))

        stats = metrics.get_alias_stats("fast")
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 1

    def test_record_error_multiple(self) -> None:
        """record_error accumulates failure counts."""
        metrics = ResourceMetrics()

        for _ in range(3):
            metrics.record_error("fast", 0.050, RuntimeError("error"))

        stats = metrics.get_alias_stats("fast")
        assert stats["total_requests"] == 3
        assert stats["failed_requests"] == 3

    def test_record_error_does_not_affect_latency_stats(self) -> None:
        """record_error does not update latency statistics."""
        metrics = ResourceMetrics()

        # First a successful request
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o-mini",
        )
        metrics.record_success("fast", 0.200, response)

        # Then a failed request
        metrics.record_error("fast", 0.500, ValueError("error"))

        stats = metrics.get_alias_stats("fast")
        # Latency stats should only reflect the successful request
        assert stats["avg_latency_ms"] == 200.0
        assert stats["min_latency_ms"] == 200.0
        assert stats["max_latency_ms"] == 200.0


class TestResourceMetricsRecordRateLimit:
    """Tests for ResourceMetrics.record_rate_limit method."""

    def test_record_rate_limit_updates_counts(self) -> None:
        """record_rate_limit increments total and rate_limited requests."""
        metrics = ResourceMetrics()

        metrics.record_rate_limit("fast")

        stats = metrics.get_alias_stats("fast")
        assert stats["total_requests"] == 1
        assert stats["rate_limited_requests"] == 1
        assert stats["failed_requests"] == 0

    def test_record_rate_limit_multiple(self) -> None:
        """record_rate_limit accumulates rate limit counts."""
        metrics = ResourceMetrics()

        for _ in range(5):
            metrics.record_rate_limit("fast")

        stats = metrics.get_alias_stats("fast")
        assert stats["total_requests"] == 5
        assert stats["rate_limited_requests"] == 5


class TestResourceMetricsGetAliasStats:
    """Tests for ResourceMetrics.get_alias_stats method."""

    def test_get_alias_stats_unknown_alias(self) -> None:
        """get_alias_stats returns default stats for unknown alias."""
        metrics = ResourceMetrics()

        stats = metrics.get_alias_stats("unknown")

        assert stats["total_requests"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["min_latency_ms"] is None  # inf converted to None

    def test_get_alias_stats_returns_all_fields(self) -> None:
        """get_alias_stats returns all expected fields."""
        metrics = ResourceMetrics()
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o-mini",
        )
        metrics.record_success("fast", 0.150, response)

        stats = metrics.get_alias_stats("fast")

        expected_keys = {
            "total_requests",
            "successful_requests",
            "failed_requests",
            "rate_limited_requests",
            "success_rate",
            "avg_latency_ms",
            "min_latency_ms",
            "max_latency_ms",
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
        }
        assert set(stats.keys()) == expected_keys


class TestResourceMetricsGetAllStats:
    """Tests for ResourceMetrics.get_all_stats method."""

    def test_get_all_stats_empty(self) -> None:
        """get_all_stats returns empty dict when no metrics recorded."""
        metrics = ResourceMetrics()
        assert metrics.get_all_stats() == {}

    def test_get_all_stats_multiple_aliases(self) -> None:
        """get_all_stats returns stats for all recorded aliases."""
        metrics = ResourceMetrics()
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o-mini",
        )

        metrics.record_success("fast", 0.100, response)
        metrics.record_success("smart", 0.200, response)
        metrics.record_rate_limit("slow")

        all_stats = metrics.get_all_stats()

        assert "fast" in all_stats
        assert "smart" in all_stats
        assert "slow" in all_stats
        assert len(all_stats) == 3


class TestResourceMetricsEstimateCost:
    """Tests for ResourceMetrics.estimate_cost method."""

    def test_estimate_cost_no_requests(self) -> None:
        """estimate_cost returns 0.0 when no requests made."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    input_cost_per_1m=0.15,
                    output_cost_per_1m=0.60,
                ),
            }
        )
        metrics = ResourceMetrics()

        assert metrics.estimate_cost(config) == 0.0

    def test_estimate_cost_calculates_correctly(self) -> None:
        """estimate_cost uses per-1M token pricing."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    input_cost_per_1m=1.0,  # $1 per 1M input tokens
                    output_cost_per_1m=2.0,  # $2 per 1M output tokens
                ),
            }
        )
        metrics = ResourceMetrics()
        response = LLMResponse(
            content="Hello",
            input_tokens=500_000,  # 0.5M tokens = $0.50
            output_tokens=250_000,  # 0.25M tokens = $0.50
            finish_reason="stop",
            model="gpt-4o-mini",
        )
        metrics.record_success("fast", 0.100, response)

        cost = metrics.estimate_cost(config)

        assert cost == pytest.approx(1.0)  # $0.50 + $0.50

    def test_estimate_cost_unknown_alias(self) -> None:
        """estimate_cost ignores aliases not in config."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    input_cost_per_1m=1.0,
                    output_cost_per_1m=2.0,
                ),
            }
        )
        metrics = ResourceMetrics()
        metrics.record_rate_limit("unknown")

        cost = metrics.estimate_cost(config)

        assert cost == 0.0


class TestResourceMetricsReset:
    """Tests for ResourceMetrics.reset method."""

    def test_reset_clears_all_metrics(self) -> None:
        """reset clears all collected metrics."""
        metrics = ResourceMetrics()
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o-mini",
        )
        metrics.record_success("fast", 0.100, response)
        metrics.record_error("smart", 0.050, ValueError("error"))

        metrics.reset()

        assert metrics.get_all_stats() == {}


class TestResourceMetricsGetEndpointMetrics:
    """Tests for ResourceMetrics.get_endpoint_metrics method."""

    def test_get_endpoint_metrics_unknown_alias(self) -> None:
        """get_endpoint_metrics returns None for unknown alias."""
        metrics = ResourceMetrics()
        assert metrics.get_endpoint_metrics("unknown") is None

    def test_get_endpoint_metrics_returns_copy(self) -> None:
        """get_endpoint_metrics returns a copy of EndpointMetrics."""
        metrics = ResourceMetrics()
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o-mini",
        )
        metrics.record_success("fast", 0.100, response)

        endpoint = metrics.get_endpoint_metrics("fast")
        assert endpoint is not None
        assert endpoint.total_requests == 1
        assert endpoint.successful_requests == 1

        # Modifying the copy should not affect the original
        endpoint.total_requests = 999
        assert metrics.get_alias_stats("fast")["total_requests"] == 1


class TestResourceMetricsThreadSafety:
    """Tests for ResourceMetrics thread safety."""

    def test_concurrent_record_success(self) -> None:
        """record_success is thread-safe under concurrent access."""
        metrics = ResourceMetrics()
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o-mini",
        )

        def record_requests(alias: str, count: int) -> None:
            for _ in range(count):
                metrics.record_success(alias, 0.100, response)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(record_requests, "fast", 100),
                executor.submit(record_requests, "fast", 100),
                executor.submit(record_requests, "smart", 50),
                executor.submit(record_requests, "smart", 50),
            ]
            for f in futures:
                f.result()

        assert metrics.get_alias_stats("fast")["total_requests"] == 200
        assert metrics.get_alias_stats("smart")["total_requests"] == 100

    def test_concurrent_mixed_operations(self) -> None:
        """Mixed record operations are thread-safe."""
        metrics = ResourceMetrics()
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            model="gpt-4o-mini",
        )

        def success_worker() -> None:
            for _ in range(50):
                metrics.record_success("fast", 0.100, response)

        def error_worker() -> None:
            for _ in range(30):
                metrics.record_error("fast", 0.050, ValueError("error"))

        def rate_limit_worker() -> None:
            for _ in range(20):
                metrics.record_rate_limit("fast")

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(success_worker),
                executor.submit(error_worker),
                executor.submit(rate_limit_worker),
            ]
            for f in futures:
                f.result()

        stats = metrics.get_alias_stats("fast")
        assert stats["total_requests"] == 100
        assert stats["successful_requests"] == 50
        assert stats["failed_requests"] == 30
        assert stats["rate_limited_requests"] == 20


class TestResourceMetricsImports:
    """Tests for ResourceMetrics imports."""

    def test_import_from_resources_package(self) -> None:
        """ResourceMetrics can be imported from plait.resources."""
        from plait.resources import ResourceMetrics as ImportedMetrics

        assert ImportedMetrics is ResourceMetricsDirect

    def test_import_endpoint_metrics_from_resources_package(self) -> None:
        """EndpointMetrics can be imported from plait.resources."""
        from plait.resources import EndpointMetrics as ImportedEndpointMetrics

        assert ImportedEndpointMetrics is EndpointMetricsDirect

    def test_resources_module_exports(self) -> None:
        """resources module __all__ includes metrics types."""
        import plait.resources as resources_module

        assert "ResourceMetrics" in resources_module.__all__
        assert "EndpointMetrics" in resources_module.__all__
