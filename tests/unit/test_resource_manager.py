"""Unit tests for ResourceManager.

Tests validate ResourceManager initialization, client creation,
semaphore management, and rate limiter management.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from inf_engine.clients.base import LLMClient
from inf_engine.resources import ResourceManager
from inf_engine.resources.config import EndpointConfig, ResourceConfig
from inf_engine.resources.manager import ResourceManager as ResourceManagerDirect
from inf_engine.resources.rate_limit import RateLimiter


class TestResourceManagerInit:
    """Tests for ResourceManager initialization."""

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_with_empty_config(self, mock_client_class: MagicMock) -> None:
        """ResourceManager initializes with empty config."""
        config = ResourceConfig(endpoints={})
        manager = ResourceManager(config)

        assert manager.config is config
        assert manager.clients == {}
        assert manager.semaphores == {}

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_stores_config(self, mock_client_class: MagicMock) -> None:
        """ResourceManager stores the provided config."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                )
            }
        )
        manager = ResourceManager(config)

        assert manager.config is config

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_creates_clients(self, mock_client_class: MagicMock) -> None:
        """ResourceManager creates clients for each endpoint."""
        mock_client = MagicMock(spec=LLMClient)
        mock_client_class.return_value = mock_client

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
        manager = ResourceManager(config)

        assert "fast" in manager.clients
        assert "smart" in manager.clients
        assert len(manager.clients) == 2

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_creates_semaphores_when_max_concurrent_set(
        self, mock_client_class: MagicMock
    ) -> None:
        """ResourceManager creates semaphores for endpoints with max_concurrent."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=10,
                ),
            }
        )
        manager = ResourceManager(config)

        assert "fast" in manager.semaphores
        assert isinstance(manager.semaphores["fast"], asyncio.Semaphore)

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_no_semaphore_when_max_concurrent_none(
        self, mock_client_class: MagicMock
    ) -> None:
        """ResourceManager skips semaphore when max_concurrent is None."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=None,
                ),
            }
        )
        manager = ResourceManager(config)

        assert "fast" not in manager.semaphores

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_semaphore_has_correct_value(
        self, mock_client_class: MagicMock
    ) -> None:
        """Semaphore is initialized with max_concurrent value."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=5,
                ),
            }
        )
        manager = ResourceManager(config)

        # Check initial semaphore value
        semaphore = manager.semaphores["fast"]
        assert semaphore._value == 5


class TestResourceManagerCreateClient:
    """Tests for ResourceManager._create_client method."""

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_create_openai_client(self, mock_client_class: MagicMock) -> None:
        """Creates OpenAIClient for openai provider."""
        mock_client = MagicMock(spec=LLMClient)
        mock_client_class.return_value = mock_client

        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    api_key="sk-test-key",
                    timeout=60.0,
                ),
            }
        )
        manager = ResourceManager(config)

        mock_client_class.assert_called_once_with(
            model="gpt-4o-mini",
            base_url=None,
            api_key="sk-test-key",
            timeout=60.0,
        )
        assert manager.clients["fast"] is mock_client

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_create_openai_client_with_base_url(
        self, mock_client_class: MagicMock
    ) -> None:
        """Creates OpenAIClient with custom base_url."""
        config = ResourceConfig(
            endpoints={
                "proxy": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o",
                    base_url="https://my-proxy.example.com/v1",
                    api_key="sk-proxy-key",
                ),
            }
        )
        ResourceManager(config)

        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["base_url"] == "https://my-proxy.example.com/v1"

    @patch("inf_engine.resources.manager.OpenAICompatibleClient")
    def test_create_vllm_client(self, mock_client_class: MagicMock) -> None:
        """Creates OpenAICompatibleClient for vllm provider."""
        mock_client = MagicMock(spec=LLMClient)
        mock_client_class.return_value = mock_client

        config = ResourceConfig(
            endpoints={
                "local": EndpointConfig(
                    provider_api="vllm",
                    model="mistral-7b",
                    base_url="http://localhost:8000/v1",
                    timeout=120.0,
                ),
            }
        )
        manager = ResourceManager(config)

        mock_client_class.assert_called_once_with(
            model="mistral-7b",
            base_url="http://localhost:8000/v1",
            api_key="not-needed",
            timeout=120.0,
        )
        assert manager.clients["local"] is mock_client

    @patch("inf_engine.resources.manager.OpenAICompatibleClient")
    def test_create_vllm_client_with_api_key(
        self, mock_client_class: MagicMock
    ) -> None:
        """Creates OpenAICompatibleClient with custom api_key if provided."""
        config = ResourceConfig(
            endpoints={
                "secure": EndpointConfig(
                    provider_api="vllm",
                    model="llama-70b",
                    base_url="http://secure.internal:8000/v1",
                    api_key="internal-key",
                ),
            }
        )
        ResourceManager(config)

        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["api_key"] == "internal-key"

    def test_create_vllm_client_requires_base_url(self) -> None:
        """Raises ValueError if vllm endpoint has no base_url."""
        config = ResourceConfig(
            endpoints={
                "invalid": EndpointConfig(
                    provider_api="vllm",
                    model="mistral-7b",
                    base_url=None,
                ),
            }
        )

        with pytest.raises(ValueError, match="vllm endpoint requires base_url"):
            ResourceManager(config)

    def test_create_anthropic_client_not_supported(self) -> None:
        """Raises ValueError for anthropic provider (not yet implemented)."""
        config = ResourceConfig(
            endpoints={
                "claude": EndpointConfig(
                    provider_api="anthropic",
                    model="claude-sonnet-4-20250514",
                ),
            }
        )

        with pytest.raises(ValueError, match="anthropic.*not yet supported"):
            ResourceManager(config)

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_create_unknown_provider_raises(self, mock_client_class: MagicMock) -> None:
        """Raises ValueError for unknown provider."""
        config = ResourceConfig(
            endpoints={
                "unknown": EndpointConfig(
                    provider_api="unknown_provider",  # type: ignore[arg-type]
                    model="some-model",
                ),
            }
        )

        with pytest.raises(ValueError, match="Unknown provider"):
            ResourceManager(config)


class TestResourceManagerGetClient:
    """Tests for ResourceManager.get_client method."""

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_get_client_returns_client(self, mock_client_class: MagicMock) -> None:
        """get_client returns the client for an alias."""
        mock_client = MagicMock(spec=LLMClient)
        mock_client_class.return_value = mock_client

        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                ),
            }
        )
        manager = ResourceManager(config)

        assert manager.get_client("fast") is mock_client

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_get_client_raises_for_unknown_alias(
        self, mock_client_class: MagicMock
    ) -> None:
        """get_client raises KeyError for unknown alias."""
        config = ResourceConfig(endpoints={})
        manager = ResourceManager(config)

        with pytest.raises(KeyError):
            manager.get_client("unknown")


class TestResourceManagerGetSemaphore:
    """Tests for ResourceManager.get_semaphore method."""

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_get_semaphore_returns_semaphore(
        self, mock_client_class: MagicMock
    ) -> None:
        """get_semaphore returns semaphore when max_concurrent is set."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=10,
                ),
            }
        )
        manager = ResourceManager(config)

        semaphore = manager.get_semaphore("fast")
        assert isinstance(semaphore, asyncio.Semaphore)

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_get_semaphore_returns_none_when_not_set(
        self, mock_client_class: MagicMock
    ) -> None:
        """get_semaphore returns None when max_concurrent is not set."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=None,
                ),
            }
        )
        manager = ResourceManager(config)

        assert manager.get_semaphore("fast") is None

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_get_semaphore_returns_none_for_unknown_alias(
        self, mock_client_class: MagicMock
    ) -> None:
        """get_semaphore returns None for unknown alias."""
        config = ResourceConfig(endpoints={})
        manager = ResourceManager(config)

        assert manager.get_semaphore("unknown") is None


class TestResourceManagerContains:
    """Tests for ResourceManager.__contains__ method."""

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_contains_returns_true_for_existing_alias(
        self, mock_client_class: MagicMock
    ) -> None:
        """__contains__ returns True for existing alias."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                ),
            }
        )
        manager = ResourceManager(config)

        assert "fast" in manager

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_contains_returns_false_for_missing_alias(
        self, mock_client_class: MagicMock
    ) -> None:
        """__contains__ returns False for missing alias."""
        config = ResourceConfig(endpoints={})
        manager = ResourceManager(config)

        assert "unknown" not in manager

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_contains_handles_unhashable_types(
        self, mock_client_class: MagicMock
    ) -> None:
        """__contains__ returns False for unhashable types."""
        config = ResourceConfig(endpoints={})
        manager = ResourceManager(config)

        assert [1, 2, 3] not in manager  # type: ignore[operator]


class TestResourceManagerImports:
    """Tests for ResourceManager imports."""

    def test_import_from_resources_package(self) -> None:
        """ResourceManager can be imported from inf_engine.resources."""
        from inf_engine.resources import ResourceManager as ImportedManager

        assert ImportedManager is ResourceManagerDirect

    def test_import_from_manager_module(self) -> None:
        """ResourceManager can be imported from inf_engine.resources.manager."""
        from inf_engine.resources.manager import ResourceManager as ImportedManager

        assert ImportedManager is ResourceManagerDirect

    def test_resources_module_exports_manager(self) -> None:
        """resources module __all__ includes ResourceManager."""
        import inf_engine.resources as resources_module

        assert "ResourceManager" in resources_module.__all__


class TestResourceManagerApiKeyResolution:
    """Tests for API key resolution via EndpointConfig.get_api_key()."""

    @patch("inf_engine.resources.manager.OpenAIClient")
    @patch.dict("os.environ", {"MY_API_KEY": "resolved-key"})
    def test_resolves_api_key_from_environment(
        self, mock_client_class: MagicMock
    ) -> None:
        """ResourceManager uses EndpointConfig.get_api_key() for resolution."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    api_key="MY_API_KEY",
                ),
            }
        )
        ResourceManager(config)

        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["api_key"] == "resolved-key"

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_uses_literal_api_key_when_not_env_var(
        self, mock_client_class: MagicMock
    ) -> None:
        """Uses literal api_key when not found in environment."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    api_key="sk-literal-key",
                ),
            }
        )
        ResourceManager(config)

        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-literal-key"

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_passes_none_when_api_key_not_set(
        self, mock_client_class: MagicMock
    ) -> None:
        """Passes None when api_key is not configured."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    api_key=None,
                ),
            }
        )
        ResourceManager(config)

        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["api_key"] is None


class TestResourceManagerRateLimiterInit:
    """Tests for ResourceManager rate limiter initialization."""

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_creates_rate_limiters_when_rate_limit_set(
        self, mock_client_class: MagicMock
    ) -> None:
        """ResourceManager creates rate limiters for endpoints with rate_limit."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    rate_limit=600.0,  # 600 RPM
                ),
            }
        )
        manager = ResourceManager(config)

        assert "fast" in manager.rate_limiters
        assert isinstance(manager.rate_limiters["fast"], RateLimiter)

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_no_rate_limiter_when_rate_limit_none(
        self, mock_client_class: MagicMock
    ) -> None:
        """ResourceManager skips rate limiter when rate_limit is None."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    rate_limit=None,
                ),
            }
        )
        manager = ResourceManager(config)

        assert "fast" not in manager.rate_limiters

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_rate_limiter_has_correct_rpm(
        self, mock_client_class: MagicMock
    ) -> None:
        """Rate limiter is initialized with correct RPM from rate_limit."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    rate_limit=1200.0,  # 1200 RPM
                ),
            }
        )
        manager = ResourceManager(config)

        limiter = manager.rate_limiters["fast"]
        assert limiter.rpm == 1200.0
        assert limiter.max_rpm == 1200.0

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_creates_rate_limiters_for_multiple_endpoints(
        self, mock_client_class: MagicMock
    ) -> None:
        """ResourceManager creates rate limiters for multiple endpoints."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    rate_limit=600.0,
                ),
                "smart": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o",
                    rate_limit=300.0,
                ),
                "no_limit": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o",
                    rate_limit=None,
                ),
            }
        )
        manager = ResourceManager(config)

        assert "fast" in manager.rate_limiters
        assert "smart" in manager.rate_limiters
        assert "no_limit" not in manager.rate_limiters
        assert manager.rate_limiters["fast"].rpm == 600.0
        assert manager.rate_limiters["smart"].rpm == 300.0

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_empty_rate_limiters_for_empty_config(
        self, mock_client_class: MagicMock
    ) -> None:
        """ResourceManager has empty rate_limiters for empty config."""
        config = ResourceConfig(endpoints={})
        manager = ResourceManager(config)

        assert manager.rate_limiters == {}


class TestResourceManagerGetRateLimiter:
    """Tests for ResourceManager.get_rate_limiter method."""

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_get_rate_limiter_returns_limiter(
        self, mock_client_class: MagicMock
    ) -> None:
        """get_rate_limiter returns rate limiter when rate_limit is set."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    rate_limit=600.0,
                ),
            }
        )
        manager = ResourceManager(config)

        limiter = manager.get_rate_limiter("fast")
        assert isinstance(limiter, RateLimiter)
        assert limiter.rpm == 600.0

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_get_rate_limiter_returns_none_when_not_set(
        self, mock_client_class: MagicMock
    ) -> None:
        """get_rate_limiter returns None when rate_limit is not set."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    rate_limit=None,
                ),
            }
        )
        manager = ResourceManager(config)

        assert manager.get_rate_limiter("fast") is None

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_get_rate_limiter_returns_none_for_unknown_alias(
        self, mock_client_class: MagicMock
    ) -> None:
        """get_rate_limiter returns None for unknown alias."""
        config = ResourceConfig(endpoints={})
        manager = ResourceManager(config)

        assert manager.get_rate_limiter("unknown") is None


class TestResourceManagerCombinedResources:
    """Tests for ResourceManager with semaphores and rate limiters together."""

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_creates_both_semaphore_and_rate_limiter(
        self, mock_client_class: MagicMock
    ) -> None:
        """ResourceManager creates both semaphore and rate limiter when both set."""
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
        manager = ResourceManager(config)

        assert "fast" in manager.semaphores
        assert "fast" in manager.rate_limiters
        assert isinstance(manager.semaphores["fast"], asyncio.Semaphore)
        assert isinstance(manager.rate_limiters["fast"], RateLimiter)

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_only_semaphore_no_rate_limiter(
        self, mock_client_class: MagicMock
    ) -> None:
        """ResourceManager creates only semaphore when rate_limit not set."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=10,
                    rate_limit=None,
                ),
            }
        )
        manager = ResourceManager(config)

        assert "fast" in manager.semaphores
        assert "fast" not in manager.rate_limiters

    @patch("inf_engine.resources.manager.OpenAIClient")
    def test_init_only_rate_limiter_no_semaphore(
        self, mock_client_class: MagicMock
    ) -> None:
        """ResourceManager creates only rate limiter when max_concurrent not set."""
        config = ResourceConfig(
            endpoints={
                "fast": EndpointConfig(
                    provider_api="openai",
                    model="gpt-4o-mini",
                    max_concurrent=None,
                    rate_limit=600.0,
                ),
            }
        )
        manager = ResourceManager(config)

        assert "fast" not in manager.semaphores
        assert "fast" in manager.rate_limiters
