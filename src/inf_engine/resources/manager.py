"""Resource manager for LLM endpoints.

This module provides the `ResourceManager` class that handles runtime
resource coordination, including client creation and concurrency control.
"""

import asyncio

from inf_engine.clients.base import LLMClient
from inf_engine.clients.openai import OpenAIClient, OpenAICompatibleClient
from inf_engine.resources.config import EndpointConfig, ResourceConfig


class ResourceManager:
    """Manages LLM endpoints and coordinates resource access.

    ResourceManager is responsible for:
    - Creating and managing LLM clients for each configured endpoint
    - Controlling concurrency with per-endpoint semaphores
    - Providing a unified interface for executing LLM calls

    The manager creates clients based on the `provider_api` field in each
    endpoint configuration and enforces concurrency limits via semaphores.

    Args:
        config: The ResourceConfig containing endpoint definitions.

    Attributes:
        config: The resource configuration.
        clients: Dict mapping aliases to LLMClient instances.
        semaphores: Dict mapping aliases to asyncio.Semaphore instances.
            Only created for endpoints with `max_concurrent` set.

    Example:
        >>> config = ResourceConfig(
        ...     endpoints={
        ...         "fast": EndpointConfig(
        ...             provider_api="openai",
        ...             model="gpt-4o-mini",
        ...             max_concurrent=10,
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

    Note:
        The ResourceManager creates clients during initialization. If a
        provider is not supported, initialization will raise a ValueError.
    """

    def __init__(self, config: ResourceConfig):
        """Initialize the resource manager with endpoint configurations.

        Creates LLM clients and semaphores for each configured endpoint.
        Clients are created based on the `provider_api` field, and semaphores
        are only created for endpoints with `max_concurrent` set.

        Args:
            config: The ResourceConfig containing endpoint definitions.

        Raises:
            ValueError: If an endpoint has an unsupported provider_api.
        """
        self.config = config

        # Per-endpoint resources
        self.clients: dict[str, LLMClient] = {}
        self.semaphores: dict[str, asyncio.Semaphore] = {}

        # Initialize clients and semaphores
        self._initialize()

    def _initialize(self) -> None:
        """Initialize clients and semaphores for each endpoint.

        Iterates through all configured endpoints and creates:
        - An LLMClient instance based on the provider_api
        - An asyncio.Semaphore if max_concurrent is set

        Raises:
            ValueError: If an endpoint has an unsupported provider_api.
        """
        for alias, endpoint in self.config.endpoints.items():
            # Create client
            self.clients[alias] = self._create_client(endpoint)

            # Create semaphore if max_concurrent is set
            if endpoint.max_concurrent is not None:
                self.semaphores[alias] = asyncio.Semaphore(endpoint.max_concurrent)

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
            - "anthropic": Not yet implemented (raises ValueError)
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
                raise ValueError(
                    "Provider 'anthropic' is not yet supported. "
                    "AnthropicClient will be added in a future release."
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
