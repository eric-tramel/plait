"""Resource metrics for endpoint observability.

This module provides metrics collection and aggregation for LLM endpoints,
enabling monitoring of request rates, latency, token usage, and error rates.
"""

import threading
from dataclasses import dataclass, field
from typing import Any

from inf_engine.resources.config import ResourceConfig
from inf_engine.types import LLMResponse


@dataclass
class EndpointMetrics:
    """Metrics for a single LLM endpoint.

    Tracks request counts, token usage, latency statistics, and error rates
    for an individual endpoint. All values are cumulative from the start of
    metrics collection.

    Attributes:
        total_requests: Total number of requests made to this endpoint.
        successful_requests: Number of requests that completed successfully.
        failed_requests: Number of requests that failed with an error.
        rate_limited_requests: Number of requests that hit rate limits.
        total_input_tokens: Cumulative input tokens across all requests.
        total_output_tokens: Cumulative output tokens across all requests.
        total_latency_ms: Sum of latencies in milliseconds for successful requests.
        min_latency_ms: Minimum observed latency in milliseconds.
        max_latency_ms: Maximum observed latency in milliseconds.

    Example:
        >>> metrics = EndpointMetrics()
        >>> metrics.total_requests
        0
        >>> metrics.success_rate
        0.0

        >>> # After recording some requests
        >>> metrics.total_requests = 100
        >>> metrics.successful_requests = 95
        >>> metrics.failed_requests = 5
        >>> metrics.success_rate
        0.95
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    total_latency_ms: float = 0.0
    min_latency_ms: float = field(default=float("inf"))
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds.

        Returns:
            Average latency across successful requests, or 0.0 if no
            successful requests have been recorded.

        Example:
            >>> metrics = EndpointMetrics(
            ...     successful_requests=3,
            ...     total_latency_ms=300.0,
            ... )
            >>> metrics.avg_latency_ms
            100.0
        """
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a fraction.

        Returns:
            Fraction of successful requests (0.0 to 1.0), or 0.0 if
            no requests have been made.

        Example:
            >>> metrics = EndpointMetrics(
            ...     total_requests=10,
            ...     successful_requests=9,
            ... )
            >>> metrics.success_rate
            0.9
        """
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used (input + output).

        Returns:
            Sum of input and output tokens.

        Example:
            >>> metrics = EndpointMetrics(
            ...     total_input_tokens=1000,
            ...     total_output_tokens=500,
            ... )
            >>> metrics.total_tokens
            1500
        """
        return self.total_input_tokens + self.total_output_tokens


class ResourceMetrics:
    """Aggregated metrics across all LLM endpoints.

    Provides thread-safe collection and retrieval of endpoint metrics.
    Each endpoint is tracked independently by its alias, and metrics
    can be queried individually or aggregated.

    The class uses a threading lock to ensure safe concurrent access
    from multiple async tasks that may complete simultaneously.

    Example:
        >>> from inf_engine.types import LLMResponse
        >>> metrics = ResourceMetrics()
        >>> response = LLMResponse(
        ...     content="Hello",
        ...     input_tokens=10,
        ...     output_tokens=5,
        ...     finish_reason="stop",
        ...     model="gpt-4o-mini",
        ... )
        >>> metrics.record_success("fast", 0.150, response)
        >>> stats = metrics.get_alias_stats("fast")
        >>> stats["total_requests"]
        1
        >>> stats["success_rate"]
        1.0

    Note:
        All metric recording methods acquire a lock to ensure thread safety.
        For high-throughput scenarios, consider batching metric updates.
    """

    def __init__(self) -> None:
        """Initialize an empty metrics collection.

        Creates an empty dictionary for endpoint metrics and a lock
        for thread-safe access.
        """
        self._metrics: dict[str, EndpointMetrics] = {}
        self._lock = threading.Lock()

    def _get_or_create_metrics(self, alias: str) -> EndpointMetrics:
        """Get or create EndpointMetrics for an alias.

        Internal helper that must be called while holding the lock.

        Args:
            alias: The endpoint alias.

        Returns:
            The EndpointMetrics instance for the alias.
        """
        if alias not in self._metrics:
            self._metrics[alias] = EndpointMetrics()
        return self._metrics[alias]

    def record_success(
        self,
        alias: str,
        duration: float,
        response: LLMResponse,
    ) -> None:
        """Record a successful request.

        Updates metrics for the endpoint with request count, token usage,
        and latency statistics.

        Args:
            alias: The endpoint alias that handled the request.
            duration: Request duration in seconds.
            response: The LLMResponse from the successful call.

        Example:
            >>> response = LLMResponse(
            ...     content="Paris",
            ...     input_tokens=15,
            ...     output_tokens=3,
            ...     finish_reason="stop",
            ...     model="gpt-4o",
            ... )
            >>> metrics = ResourceMetrics()
            >>> metrics.record_success("smart", 0.250, response)
        """
        with self._lock:
            m = self._get_or_create_metrics(alias)
            m.total_requests += 1
            m.successful_requests += 1
            m.total_input_tokens += response.input_tokens
            m.total_output_tokens += response.output_tokens

            latency_ms = duration * 1000
            m.total_latency_ms += latency_ms
            m.min_latency_ms = min(m.min_latency_ms, latency_ms)
            m.max_latency_ms = max(m.max_latency_ms, latency_ms)

    def record_error(
        self,
        alias: str,
        duration: float,
        error: Exception,
    ) -> None:
        """Record a failed request.

        Updates metrics for the endpoint with the failure count.
        Duration is captured but not included in latency statistics
        since the request did not complete successfully.

        Args:
            alias: The endpoint alias where the error occurred.
            duration: Request duration in seconds before failure.
            error: The exception that caused the failure.

        Example:
            >>> metrics = ResourceMetrics()
            >>> try:
            ...     raise ValueError("API error")
            ... except ValueError as e:
            ...     metrics.record_error("fast", 0.050, e)
        """
        with self._lock:
            m = self._get_or_create_metrics(alias)
            m.total_requests += 1
            m.failed_requests += 1

    def record_rate_limit(self, alias: str) -> None:
        """Record a rate-limited request.

        Updates metrics for the endpoint with the rate limit count.
        Rate-limited requests are counted separately from errors since
        they may be retried successfully.

        Args:
            alias: The endpoint alias that was rate-limited.

        Example:
            >>> metrics = ResourceMetrics()
            >>> metrics.record_rate_limit("fast")
            >>> metrics.get_alias_stats("fast")["rate_limited_requests"]
            1
        """
        with self._lock:
            m = self._get_or_create_metrics(alias)
            m.total_requests += 1
            m.rate_limited_requests += 1

    def get_alias_stats(self, alias: str) -> dict[str, Any]:
        """Get statistics for a specific endpoint.

        Returns a dictionary with key metrics for the endpoint,
        suitable for logging, monitoring dashboards, or debugging.

        Args:
            alias: The endpoint alias to query.

        Returns:
            Dictionary containing:
            - total_requests: Total request count
            - successful_requests: Successful request count
            - failed_requests: Failed request count
            - rate_limited_requests: Rate-limited request count
            - success_rate: Fraction of successful requests (0.0-1.0)
            - avg_latency_ms: Average latency in milliseconds
            - min_latency_ms: Minimum latency (or None if no requests)
            - max_latency_ms: Maximum latency in milliseconds
            - total_input_tokens: Total input tokens used
            - total_output_tokens: Total output tokens used
            - total_tokens: Combined input + output tokens

        Example:
            >>> metrics = ResourceMetrics()
            >>> stats = metrics.get_alias_stats("unknown")
            >>> stats["total_requests"]
            0
        """
        with self._lock:
            m = self._get_or_create_metrics(alias)
            return {
                "total_requests": m.total_requests,
                "successful_requests": m.successful_requests,
                "failed_requests": m.failed_requests,
                "rate_limited_requests": m.rate_limited_requests,
                "success_rate": m.success_rate,
                "avg_latency_ms": m.avg_latency_ms,
                "min_latency_ms": m.min_latency_ms
                if m.min_latency_ms != float("inf")
                else None,
                "max_latency_ms": m.max_latency_ms,
                "total_input_tokens": m.total_input_tokens,
                "total_output_tokens": m.total_output_tokens,
                "total_tokens": m.total_tokens,
            }

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all endpoints.

        Returns a dictionary mapping each endpoint alias to its statistics.

        Returns:
            Dictionary mapping aliases to their stats dictionaries.

        Example:
            >>> metrics = ResourceMetrics()
            >>> metrics.record_rate_limit("fast")
            >>> all_stats = metrics.get_all_stats()
            >>> "fast" in all_stats
            True
        """
        with self._lock:
            aliases = list(self._metrics.keys())

        # Release lock before calling get_alias_stats to avoid nested locking
        return {alias: self.get_alias_stats(alias) for alias in aliases}

    def estimate_cost(self, config: ResourceConfig) -> float:
        """Estimate total cost based on token usage and endpoint pricing.

        Uses the `input_cost_per_1m` and `output_cost_per_1m` fields from
        each endpoint's configuration to calculate the total estimated cost.

        Args:
            config: The ResourceConfig containing endpoint pricing information.

        Returns:
            Estimated total cost in dollars across all endpoints.

        Example:
            >>> from inf_engine.resources.config import EndpointConfig, ResourceConfig
            >>> config = ResourceConfig(
            ...     endpoints={
            ...         "fast": EndpointConfig(
            ...             provider_api="openai",
            ...             model="gpt-4o-mini",
            ...             input_cost_per_1m=0.15,
            ...             output_cost_per_1m=0.60,
            ...         ),
            ...     }
            ... )
            >>> metrics = ResourceMetrics()
            >>> # After recording some requests with tokens...
            >>> cost = metrics.estimate_cost(config)
        """
        total = 0.0
        with self._lock:
            for alias, m in self._metrics.items():
                endpoint = config.get(alias)
                if endpoint is not None:
                    # Convert from per-1M to per-token cost
                    input_cost = (
                        m.total_input_tokens / 1_000_000
                    ) * endpoint.input_cost_per_1m
                    output_cost = (
                        m.total_output_tokens / 1_000_000
                    ) * endpoint.output_cost_per_1m
                    total += input_cost + output_cost
        return total

    def reset(self) -> None:
        """Reset all metrics to zero.

        Clears all collected metrics. Useful for testing or when starting
        a new measurement period.

        Example:
            >>> metrics = ResourceMetrics()
            >>> metrics.record_rate_limit("fast")
            >>> metrics.reset()
            >>> metrics.get_all_stats()
            {}
        """
        with self._lock:
            self._metrics.clear()

    def get_endpoint_metrics(self, alias: str) -> EndpointMetrics | None:
        """Get the raw EndpointMetrics object for an alias.

        Returns a copy of the EndpointMetrics object for direct access
        to all fields. Returns None if no metrics exist for the alias.

        Args:
            alias: The endpoint alias to query.

        Returns:
            A copy of the EndpointMetrics, or None if not found.

        Example:
            >>> metrics = ResourceMetrics()
            >>> metrics.record_rate_limit("fast")
            >>> endpoint = metrics.get_endpoint_metrics("fast")
            >>> endpoint.rate_limited_requests
            1
        """
        with self._lock:
            if alias not in self._metrics:
                return None
            m = self._metrics[alias]
            # Return a copy to avoid external mutation
            return EndpointMetrics(
                total_requests=m.total_requests,
                successful_requests=m.successful_requests,
                failed_requests=m.failed_requests,
                rate_limited_requests=m.rate_limited_requests,
                total_input_tokens=m.total_input_tokens,
                total_output_tokens=m.total_output_tokens,
                total_latency_ms=m.total_latency_ms,
                min_latency_ms=m.min_latency_ms,
                max_latency_ms=m.max_latency_ms,
            )
