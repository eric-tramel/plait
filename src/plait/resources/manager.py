"""Resource manager for LLM endpoints.

This module provides the `ResourceManager` class that handles runtime
resource coordination, including client creation, concurrency control,
rate limiting, and metrics collection.
"""

import asyncio
from typing import Any

from plait.clients.anthropic import AnthropicClient
from plait.clients.base import LLMClient
from plait.clients.openai import OpenAIClient, OpenAICompatibleClient
from plait.resources.config import EndpointConfig, ResourceConfig
from plait.resources.metrics import ResourceMetrics
from plait.resources.rate_limit import RateLimiter


class ResourceManager:
    """Manages LLM endpoints and coordinates resource access.

    ResourceManager is responsible for:
    - Creating and managing LLM clients for each configured endpoint
    - Controlling concurrency with per-endpoint semaphores
    - Managing rate limiters for endpoints with RPM limits
    - Collecting metrics for endpoint observability

    The manager creates clients based on the `provider_api` field in each
    endpoint configuration, semaphores for concurrency control, rate
    limiters for endpoints with rate limits configured, and a shared
    metrics collector for monitoring request counts, latencies, and tokens.

    Args:
        config: The ResourceConfig containing endpoint definitions.

    Attributes:
        config: The resource configuration.
        clients: Dict mapping aliases to LLMClient instances.
        semaphores: Dict mapping aliases to asyncio.Semaphore instances.
            Only created for endpoints with `max_concurrent` set.
        rate_limiters: Dict mapping aliases to RateLimiter instances.
            Only created for endpoints with `rate_limit` set.
        metrics: ResourceMetrics instance for tracking endpoint usage.

    Example:
        >>> config = ResourceConfig(
        ...     endpoints={
        ...         "fast": EndpointConfig(
        ...             provider_api="openai",
        ...             model="gpt-4o-mini",
        ...             max_concurrent=10,
        ...             rate_limit=600.0,  # 600 RPM
        ...         ),
        ...         "smart": EndpointConfig(
        ...             provider_api="openai",
        ...             model="gpt-4o",
        ...             max_concurrent=5,
        ...         ),
        ...     }
        ... )
        >>> manager = ResourceManager(config)
        >>> "fast" in manager.clients
        True
        >>> "fast" in manager.semaphores
        True
        >>> "fast" in manager.rate_limiters
        True
        >>> manager.metrics is not None
        True

    Note:
        The ResourceManager creates clients during initialization. If a
        provider is not supported, initialization will raise a ValueError.
    """

    def __init__(self, config: ResourceConfig):
        """Initialize the resource manager with endpoint configurations.

        Creates LLM clients, semaphores, rate limiters, and a metrics
        collector for each configured endpoint. Clients are created based
        on the `provider_api` field, semaphores are created for endpoints
        with `max_concurrent` set, and rate limiters are created for
        endpoints with `rate_limit` set.

        Args:
            config: The ResourceConfig containing endpoint definitions.

        Raises:
            ValueError: If an endpoint has an unsupported provider_api.
        """
        self.config = config

        # Per-endpoint resources
        self.clients: dict[str, LLMClient] = {}
        self.semaphores: dict[str, asyncio.Semaphore] = {}
        self.rate_limiters: dict[str, RateLimiter] = {}

        # Metrics collection
        self.metrics = ResourceMetrics()

        # Initialize clients, semaphores, and rate limiters
        self._initialize()

    def _initialize(self) -> None:
        """Initialize clients, semaphores, and rate limiters for each endpoint.

        Iterates through all configured endpoints and creates:
        - An LLMClient instance based on the provider_api
        - An asyncio.Semaphore if max_concurrent is set
        - A RateLimiter if rate_limit is set

        Raises:
            ValueError: If an endpoint has an unsupported provider_api.
        """
        for alias, endpoint in self.config.endpoints.items():
            # Create client
            self.clients[alias] = self._create_client(endpoint)

            # Create semaphore if max_concurrent is set
            if endpoint.max_concurrent is not None:
                self.semaphores[alias] = asyncio.Semaphore(endpoint.max_concurrent)

            # Create rate limiter if rate_limit is set
            if endpoint.rate_limit is not None:
                self.rate_limiters[alias] = RateLimiter(rpm=endpoint.rate_limit)

    def _create_client(self, endpoint: EndpointConfig) -> LLMClient:
        """Create the appropriate client for an endpoint.

        Dispatches to the correct client implementation based on the
        endpoint's provider_api field.

        Args:
            endpoint: The endpoint configuration.

        Returns:
            An LLMClient instance configured for the endpoint.

        Raises:
            ValueError: If the provider_api is not supported.

        Note:
            Currently supported providers:
            - "openai": Uses OpenAIClient
            - "vllm": Uses OpenAICompatibleClient
            - "anthropic": Uses AnthropicClient
        """
        match endpoint.provider_api:
            case "openai":
                return OpenAIClient(
                    model=endpoint.model,
                    base_url=endpoint.base_url,
                    api_key=endpoint.get_api_key(),
                    timeout=endpoint.timeout,
                )
            case "vllm":
                if endpoint.base_url is None:
                    raise ValueError(
                        "vllm endpoint requires base_url, but none was provided"
                    )
                return OpenAICompatibleClient(
                    model=endpoint.model,
                    base_url=endpoint.base_url,
                    api_key=endpoint.get_api_key() or "not-needed",
                    timeout=endpoint.timeout,
                )
            case "anthropic":
                return AnthropicClient(
                    model=endpoint.model,
                    base_url=endpoint.base_url,
                    api_key=endpoint.get_api_key(),
                    timeout=endpoint.timeout,
                )
            case _:
                raise ValueError(f"Unknown provider: {endpoint.provider_api}")

    def get_client(self, alias: str) -> LLMClient:
        """Get the LLM client for an alias.

        Args:
            alias: The endpoint alias.

        Returns:
            The LLMClient for the given alias.

        Raises:
            KeyError: If the alias is not found.

        Example:
            >>> manager = ResourceManager(config)
            >>> client = manager.get_client("fast")
        """
        return self.clients[alias]

    def get_semaphore(self, alias: str) -> asyncio.Semaphore | None:
        """Get the semaphore for an alias, if one exists.

        Args:
            alias: The endpoint alias.

        Returns:
            The asyncio.Semaphore for the alias, or None if no
            max_concurrent was configured for that endpoint.

        Example:
            >>> manager = ResourceManager(config)
            >>> semaphore = manager.get_semaphore("fast")
            >>> if semaphore:
            ...     async with semaphore:
            ...         # limited concurrency
            ...         pass
        """
        return self.semaphores.get(alias)

    def get_rate_limiter(self, alias: str) -> RateLimiter | None:
        """Get the rate limiter for an alias, if one exists.

        Args:
            alias: The endpoint alias.

        Returns:
            The RateLimiter for the alias, or None if no
            rate_limit was configured for that endpoint.

        Example:
            >>> manager = ResourceManager(config)
            >>> limiter = manager.get_rate_limiter("fast")
            >>> if limiter:
            ...     await limiter.acquire()
            ...     # make request
            ...     limiter.recover()
        """
        return self.rate_limiters.get(alias)

    def __contains__(self, alias: object) -> bool:
        """Check if an alias is managed by this ResourceManager.

        Args:
            alias: The alias to check.

        Returns:
            True if the alias exists in clients, False otherwise.

        Example:
            >>> "fast" in manager
            True
        """
        try:
            return alias in self.clients
        except TypeError:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get resource utilization statistics for all endpoints.

        Returns a dictionary with availability and metrics information
        for each configured endpoint. This is useful for monitoring
        dashboards and debugging.

        Returns:
            Dictionary mapping aliases to their stats, including:
            - available: Current semaphore availability (if configured)
            - max: Maximum concurrent requests (if configured)
            - metrics: Endpoint metrics from ResourceMetrics

        Example:
            >>> config = ResourceConfig(
            ...     endpoints={
            ...         "fast": EndpointConfig(
            ...             provider_api="openai",
            ...             model="gpt-4o-mini",
            ...             max_concurrent=10,
            ...         ),
            ...     }
            ... )
            >>> manager = ResourceManager(config)
            >>> stats = manager.get_stats()
            >>> stats["fast"]["max"]
            10
            >>> stats["fast"]["metrics"]["total_requests"]
            0
        """
        result: dict[str, Any] = {}
        for alias in self.clients:
            endpoint = self.config.endpoints[alias]
            alias_stats: dict[str, Any] = {
                "metrics": self.metrics.get_alias_stats(alias),
            }

            # Include semaphore stats if available
            if alias in self.semaphores:
                alias_stats["available"] = self.semaphores[alias]._value
                alias_stats["max"] = endpoint.max_concurrent

            result[alias] = alias_stats

        return result
