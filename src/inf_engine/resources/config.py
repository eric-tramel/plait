"""Configuration dataclasses for LLM endpoints.

This module defines the configuration structure for LLM endpoints,
separating module definitions from infrastructure configuration.
"""

import hashlib
import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class EndpointConfig:
    """Configuration for a single LLM endpoint.

    Defines all settings for connecting to and using an LLM endpoint,
    including provider details, concurrency limits, retry behavior,
    and cost tracking.

    Args:
        provider_api: The LLM provider API type. Determines which client
            implementation to use for API calls.
        model: The model identifier to use for completions.
        base_url: Custom endpoint URL for self-hosted or proxy deployments.
            If None, uses the provider's default URL.
        api_key: API key or environment variable name for authentication.
            If the value matches an environment variable name, that variable's
            value is used. Otherwise, the value is treated as a literal API key.
            If None, the client will attempt to read from provider-specific
            environment variables (e.g., OPENAI_API_KEY).
        max_concurrent: Maximum number of parallel requests to this endpoint.
        rate_limit: Maximum requests per second. If None, no rate limiting
            is applied beyond concurrency limits.
        max_retries: Number of retry attempts for failed requests.
        retry_delay: Base delay in seconds between retry attempts.
        timeout: Request timeout in seconds.
        input_cost_per_1m: Cost in dollars per 1 million input tokens.
            Used for cost estimation and tracking.
        output_cost_per_1m: Cost in dollars per 1 million output tokens.
            Used for cost estimation and tracking.

    Example:
        >>> config = EndpointConfig(
        ...     provider_api="openai",
        ...     model="gpt-4o-mini",
        ...     max_concurrent=20,
        ...     rate_limit=10.0,
        ... )

        >>> # Self-hosted vLLM endpoint
        >>> vllm_config = EndpointConfig(
        ...     provider_api="vllm",
        ...     model="mistral-7b",
        ...     base_url="http://vllm.internal:8000",
        ...     max_concurrent=50,
        ... )

        >>> # Using environment variable name for API key
        >>> config = EndpointConfig(
        ...     provider_api="openai",
        ...     model="gpt-4o",
        ...     api_key="MY_OPENAI_KEY",  # Reads from $MY_OPENAI_KEY
        ... )
    """

    provider_api: Literal["openai", "anthropic", "vllm"]
    model: str
    base_url: str | None = None
    api_key: str | None = None

    # Concurrency limits
    max_concurrent: int = 10
    rate_limit: float | None = None

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0

    # Timeout
    timeout: float = 300.0

    # Cost tracking
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0

    def get_api_key(self) -> str | None:
        """Resolve the API key from environment or literal value.

        If api_key is None, returns None.
        If api_key matches an environment variable name, returns that variable's value.
        Otherwise, returns api_key as a literal value.

        Returns:
            The resolved API key, or None if not configured.

        Example:
            >>> import os
            >>> os.environ["MY_API_KEY"] = "secret-123"
            >>> config = EndpointConfig(
            ...     provider_api="openai",
            ...     model="gpt-4o",
            ...     api_key="MY_API_KEY",
            ... )
            >>> config.get_api_key()
            'secret-123'

            >>> # Literal key (not found in environment)
            >>> config = EndpointConfig(
            ...     provider_api="openai",
            ...     model="gpt-4o",
            ...     api_key="sk-literal-key",
            ... )
            >>> config.get_api_key()
            'sk-literal-key'
        """
        if self.api_key is None:
            return None

        # Check if api_key is an environment variable name
        env_value = os.environ.get(self.api_key)
        if env_value is not None:
            return env_value

        # Otherwise treat as literal key
        return self.api_key

    @property
    def hash_key(self) -> str:
        """Generate a hash key for indexing this endpoint.

        Combines the base_url and resolved API key into a stable hash
        that can be used to identify unique endpoint configurations.

        Returns:
            A hex string hash of the base_url and resolved API key.

        Example:
            >>> config1 = EndpointConfig(
            ...     provider_api="openai",
            ...     model="gpt-4o",
            ...     api_key="sk-test-key",
            ... )
            >>> config2 = EndpointConfig(
            ...     provider_api="openai",
            ...     model="gpt-4o-mini",  # Different model
            ...     api_key="sk-test-key",  # Same key
            ... )
            >>> config1.hash_key == config2.hash_key  # Same endpoint
            True
        """
        base_url_str = self.base_url or ""
        api_key_str = self.get_api_key() or ""
        combined = f"{base_url_str}:{api_key_str}"
        return hashlib.sha256(combined.encode()).hexdigest()


class OpenAIEndpointConfig(EndpointConfig):
    """EndpointConfig preset for OpenAI API.

    Provides sensible defaults for OpenAI endpoints:
    - provider_api: "openai"
    - api_key: "OPENAI_API_KEY" (reads from environment)

    Args:
        model: The OpenAI model identifier (e.g., "gpt-4o", "gpt-4o-mini").
        api_key: API key or env var name. Defaults to "OPENAI_API_KEY".
        **kwargs: Additional EndpointConfig options.

    Example:
        >>> config = OpenAIEndpointConfig(model="gpt-4o-mini")
        >>> config.provider_api
        'openai'
        >>> config.api_key
        'OPENAI_API_KEY'
    """

    def __init__(
        self,
        model: str,
        api_key: str = "OPENAI_API_KEY",
        **kwargs,
    ):
        super().__init__(
            provider_api="openai",
            model=model,
            api_key=api_key,
            **kwargs,
        )


class AnthropicEndpointConfig(EndpointConfig):
    """EndpointConfig preset for Anthropic API.

    Provides sensible defaults for Anthropic endpoints:
    - provider_api: "anthropic"
    - api_key: "ANTHROPIC_API_KEY" (reads from environment)

    Args:
        model: The Anthropic model identifier (e.g., "claude-sonnet-4-20250514").
        api_key: API key or env var name. Defaults to "ANTHROPIC_API_KEY".
        **kwargs: Additional EndpointConfig options.

    Example:
        >>> config = AnthropicEndpointConfig(model="claude-sonnet-4-20250514")
        >>> config.provider_api
        'anthropic'
        >>> config.api_key
        'ANTHROPIC_API_KEY'
    """

    def __init__(
        self,
        model: str,
        api_key: str = "ANTHROPIC_API_KEY",
        **kwargs,
    ):
        super().__init__(
            provider_api="anthropic",
            model=model,
            api_key=api_key,
            **kwargs,
        )


class NvidiaBuildEndpointConfig(EndpointConfig):
    """EndpointConfig preset for NVIDIA Build API.

    Provides sensible defaults for NVIDIA Build endpoints:
    - provider_api: "openai" (NVIDIA uses OpenAI-compatible API)
    - base_url: "https://integrate.api.nvidia.com/v1"
    - api_key: "NVIDIA_API_KEY" (reads from environment)

    Args:
        model: The NVIDIA model identifier (e.g., "meta/llama-3.1-405b-instruct").
        api_key: API key or env var name. Defaults to "NVIDIA_API_KEY".
        **kwargs: Additional EndpointConfig options.

    Example:
        >>> config = NvidiaBuildEndpointConfig(model="meta/llama-3.1-405b-instruct")
        >>> config.provider_api
        'openai'
        >>> config.base_url
        'https://integrate.api.nvidia.com/v1'
        >>> config.api_key
        'NVIDIA_API_KEY'
    """

    def __init__(
        self,
        model: str,
        api_key: str = "NVIDIA_API_KEY",
        **kwargs,
    ):
        super().__init__(
            provider_api="openai",
            model=model,
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            **kwargs,
        )
